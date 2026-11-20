"""Browser steps for the one-load Kontext trial, executed by tools/browser.mjs."""
from __future__ import annotations

import json
from typing import Any

from tools.kontext_trial_fixtures import INSTRUCTION

Step = dict[str, Any]


def expect(expression: str) -> Step:
    return {"expect": f"(async () => ({expression}))()", "fatal": True}


def until(predicate: str, timeout: int = 20_000) -> Step:
    return expect(f"trialUntil({predicate}, {timeout})")


def press(selector: str, action: str = "click") -> list[Step]:
    key = json.dumps(selector)
    return [
        until(f"() => document.querySelector({key})?.getBoundingClientRect().height > 0"),
        {"eval": f"document.querySelector({key}).scrollIntoView({{block:'center'}})"},
        until(f"() => trialHit({key})"), {action: selector},
    ]


def begin(url: str, pid: int) -> list[Step]:
    return [
        {"navigate": url}, {"inject": "kontext_trial_page.js"},
        until("() => lastDefaults && modelRuntimeSettings && !galleryPending"),
        expect(f"(await trialGet('/api/kontext-trial')).pid === {pid}"),
        {"eval": "if (!els.allModels.checked) els.allModels.click()"},
        until("() => !galleryPending"),
    ]


def power(load: bool) -> list[Step]:
    state, target, button = ("unloaded", "ready", "load") if load else ("ready", "unloaded", "unload")
    return [expect(f"(await trialGet('/api/status')).state === '{state}'"),
            *press(f"#btn-{button}"),
            until(f"async () => (await trialStatus()) === '{target}'", 250_000)]


def gallery_card(image_id: int) -> str:
    return f'#gallery .card[data-id="{image_id}"] > img'


def edit_dialog(source: int, recipe: str) -> list[Step]:
    return [
        *press(gallery_card(source), "rightclick"),
        until("() => trialMarkEditItem()"),
        *press('[data-trial-edit-item]'),
        until(f"() => els.editDialog.open && editSource?.id === {source} && editPreview"),
        {"eval": f"trialSet('#edit-recipe', {json.dumps(recipe)}, 'change')"},
    ]


def edit(source: int, recipe: str, width: int, height: int, seed: int, *, cancel: bool = False) -> list[Step]:
    steps = [
        *edit_dialog(source, recipe),
        until(f"() => !editPreviewStale && editPreview?.available && editPreview.edit.recipe === {json.dumps(recipe)}"
              f" && editPreview.width === {width} && editPreview.height === {height}"),
        {"eval": f"replaceBoxText(els.editInstruction, {json.dumps(INSTRUCTION)}); trialSet('#edit-seed', '{seed}')"},
        until("() => els.editSteps.value === '28' && els.editGuidance.value === '2.5'"
              " && els.editRun.getAttribute('aria-disabled') === 'false'"),
        {"eval": "trialRememberJobs()"}, *press("#edit-run"),
        until("async () => !!(await trialNewJob())"),
    ]
    if cancel:
        steps.extend([
            until("() => busy && currentJob === trialJob && !els.cancel.hidden"),
            {"eval": "els.cancel.scrollIntoView({block:'center'})"},
            until("() => trialHit('#btn-cancel')"),
            until("trialCancelOnThree", 240_000),
            {"log": "({cancel_at_step: trialCancelAt})"},
        ])
    terminal = "cancelled" if cancel else "done"
    return steps + [until(f"async () => trialFinished('{terminal}')", 270_000)]


def more_seeds(image_id: int) -> list[Step]:
    return [
        *press(gallery_card(image_id)), until("() => els.lightbox.open"),
        *press("#lb-more-seeds"), until("() => els.moreSeedsDialog.open && msAnswer"),
        {"eval": "trialSet('#more-seeds-count', '2')"},
        until("() => msAnswer?.queued === 2 && msAnswer.results[0].first === 12346"
              " && msAnswer.results[0].last === 12347 && els.moreSeedsRun.getAttribute('aria-disabled') === 'false'"),
        {"eval": "trialRememberJobs()"}, *press("#more-seeds-run"),
        until("async () => !!(await trialNewJob())"),
        until("async () => await trialFinished('done')", 270_000),
    ]


def refusals(square: int, wide: int, result: int, run: str) -> list[Step]:
    finish = json.dumps(f"/api/runs/{run}/finish")
    return [
        {"eval": "trialRememberJobs()"},
        *edit_dialog(wide, "kontext-1024-v1"),
        until("() => !editPreviewStale && editProblem?.includes('square')"
              " && els.editRun.getAttribute('aria-disabled') === 'true'"),
        *press("#edit-run"),
        until("() => !els.editError.hidden && els.editError.textContent.includes('square')"),
        {"key": "Escape"}, until("() => !els.editDialog.open"),
        expect(f"await trialRefused('/api/instruction-edit', {{source:{wide},recipe:'kontext-1024-v1',"
               "prompt:'x',width:1024,height:1024}, /square/i)"),
        expect(f"await trialRefused('/api/instruction-edit', {{source:{square},recipe:'kontext-v1',"
               "prompt:'x',width:1024,height:1024}, /size|dimension|512|geometry/i)"),
        *press(gallery_card(result)), until("() => els.lightbox.open"),
        *press("#lb-reuse"),
        until(f"() => els.editDialog.open && editFrom?.id === {result} && editSource?.id === {square}"
              " && els.editSeed.value === '12345' && editPreview?.edit.recipe === 'kontext-1024-v1'"
              " && document.querySelector('#edit-recipe').value === 'kontext-1024-v1'"),
        {"key": "Escape"}, until("() => !els.editDialog.open && els.lightbox.open"), *press("#lb-close"),
        expect("els.generate.getAttribute('aria-disabled') === 'true'"), *press("#btn-generate"),
        until("() => els.error.textContent.includes('does not generate from text')"),
        {"eval": f"toggleSelected(galleryRecords.find(r => r.id === {result}))"}, *press("#btn-rerender"),
        until("() => els.galleryError.textContent.includes('An instruction edit renders at its source')"),
        expect(f"(await trialPost({finish} + '/preview', {{}})).data.available === false"),
        expect(f"await trialRefused({finish}, {{}}, /Edit by instruction/)"),
        expect("await trialNoNewJobs()"), {"log": "trialRefusals"},
    ]
