"""The CARD#10 step packs (`tools/card/`) keep to their own rules.

docs/plan-card-harness.md, "CARD#10 step packs": one `tools/browser.mjs`
step file per fix commit `CARD#10` lists, plus the idle Ctrl-C. Each is read
here as text only - no browser, no server, no git (CI's checkouts are
shallow): every step a kind `browser.mjs` runs, no fake-only knob, nothing
written round the page's own input path, each commit pack named by a sha
verified.md's Behaviour section lists, and the adapter listing waited for
before the first press that could load. Packs are not smokes: nothing here
runs them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "tools" / "card"
BROWSER = (ROOT / "tools" / "browser.mjs").read_text(encoding="utf-8")

# The commits CARD#10 asks to see on the card (TODO.md's row; verified.md,
# Behaviour), and the idle Ctrl-C, which has no fix commit.
COMMITS = ("9377de1", "7733f81", "6a7570d", "ad01fdc", "2a4d2cc", "2b89ed5", "1744dc3")
PACKS = sorted([f"card10-{sha}.json" for sha in COMMITS] + ["card10-ctrl-c-idle.json"])

# What a step may carry beside its one action, as browser.mjs reads it.
MODIFIERS = {"expect": {"fatal"}, "key": {"native", "repeat"}, "rightclick": {"shift"}}
# The presses that can start a load: Load, Generate, Run grid, a pass's
# Refine, Switch & load, or Ctrl+Enter.
PRESS = re.compile(r"#btn-(?:load|generate|grid|model-switch)\b|#refine-run\b")
# The page's own resource timing for its first adapter listing.
LISTING = "performance.getEntriesByName(location.origin + '/api/loras')"
# browser.mjs gives up on a call the browser has not answered in this long.
ANSWER_MS = 300_000


def _kinds() -> set[str]:
    """The action kinds browser.mjs's step loop dispatches on."""
    loop = BROWSER[BROWSER.index("for (const [n, step] of steps.entries())"):]
    return set(re.findall(r"(?:if|else if) \(step\.(\w+)\)", loop))


def _steps(name: str) -> list[dict]:
    return json.loads((CARD / name).read_text(encoding="utf-8"))


def _code(step: dict) -> str:
    """Every string a step holds, as one text."""
    return json.dumps(step, ensure_ascii=False)


def _behaviour() -> str:
    text = (ROOT / "docs" / "verified.md").read_text(encoding="utf-8")
    start = text.index("\n## Behaviour\n")
    return text[start : text.index("\n## ", start + 1)]


def test_the_packs_are_the_eight_and_none_is_a_smoke() -> None:
    assert sorted(p.name for p in CARD.iterdir()) == PACKS
    # run_smoke.py and CI's browser job read AGENTS.md's port table, which
    # names smokes alone.
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "card10" not in agents
    assert all(not name.startswith("smoke") for name in PACKS)


def test_browser_mjs_still_runs_the_kinds_this_test_knows() -> None:
    kinds = _kinds()
    assert {"navigate", "wait", "eval", "log", "expect", "key", "keycombo", "type", "wheel",
            "hover", "click", "viewport", "shot"} <= kinds
    for kind, extra in MODIFIERS.items():
        assert kind in kinds
        for modifier in extra:
            assert f"step.{modifier}" in BROWSER


@pytest.mark.parametrize("name", PACKS)
def test_every_step_is_one_kind_browser_mjs_runs(name: str) -> None:
    kinds = _kinds()
    steps = _steps(name)
    assert isinstance(steps, list) and steps
    for n, step in enumerate(steps, 1):
        assert isinstance(step, dict), (name, n)
        actions = [key for key in step if key in kinds]
        assert len(actions) == 1, (name, n, step)
        extra = set(step) - {actions[0]}
        assert extra <= MODIFIERS.get(actions[0], set()), (name, n, step)
        if "fatal" in step:
            assert step["fatal"] is True and isinstance(step["expect"], str) and step["expect"].strip()
        if "keycombo" in step:
            combo = step["keycombo"]
            assert len(combo) >= 2 and set(combo[:-1]) <= {"Alt", "Control", "Meta", "Shift"}, (name, n)
        if "viewport" in step:
            assert len(step["viewport"]) in (2, 3), (name, n)


@pytest.mark.parametrize("name", PACKS)
def test_no_pack_reaches_round_the_page(name: str) -> None:
    for n, step in enumerate(_steps(name), 1):
        code = _code(step)
        # No fake-only knob and no state only a fake has.
        assert "/api/fake/" not in code, (name, n)
        # The prompt and negative go in by typing; the action under check is
        # a press, never the handler behind it.
        assert not re.search(r"els\.(?:prompt|negative)\.value\s*=(?!=)", code), (name, n)
        assert "startJob(" not in code, (name, n)
        # Evals observe: no request of the pack's own (the adapter listing is
        # read from the page's resource timing, never asked again).
        assert not re.search(r"(?<![\w.$])(?:fetch|api)\(", code), (name, n)
        # Nor any press, scroll, focus or field write from an eval: the page
        # is moved by `wheel` and filled by `type`.
        assert not re.search(
            r"\.(?:click|focus|scrollTo|scrollBy|scrollIntoView|dispatchEvent)\(|"
            r"(?<![\w$])(?:scrollTo|scrollBy)\(|\.(?:value|scrollTop|checked|open)\s*=(?!=)",
            code,
        ), (name, n)


@pytest.mark.parametrize("name", PACKS)
def test_no_awaited_expression_outlasts_browser_mjs(name: str) -> None:
    assert f"const ANSWER_MS = {ANSWER_MS};" in BROWSER
    for n, step in enumerate(_steps(name), 1):
        if "wait" in step:
            assert step["wait"] < ANSWER_MS, (name, n)
        for value in re.findall(r"__until\(.*?,\s*(\d+)\)", _code(step)):
            assert int(value) <= 240_000, (name, n)


@pytest.mark.parametrize("sha", COMMITS)
def test_each_commit_pack_names_a_sha_verified_md_lists(sha: str) -> None:
    behaviour = _behaviour()
    if sha == "1744dc3":
        # R50's row, the typed Ctrl-C's check.
        where = next(line for line in behaviour.splitlines() if line.startswith("| Ctrl-C while a job waits on a real first load, R50"))
    else:
        # The paragraph after the table listing what was fixed and checked
        # only on the fake.
        where = next(p for p in behaviour.split("\n\n") if p.startswith("The rest of what these rows found"))
    assert sha in where
    first = _steps(f"card10-{sha}.json")[0]
    assert set(first) == {"log"} and first["log"].startswith(f'"{sha} '), first


def test_the_idle_pack_says_what_it_checks_first() -> None:
    first = _steps("card10-ctrl-c-idle.json")[0]
    assert set(first) == {"log"} and "idle server with a model loaded" in first["log"]
    # It has no commit, and names none of CARD#10's.
    assert not any(sha in first["log"] for sha in COMMITS)


@pytest.mark.parametrize("name", PACKS)
def test_each_pack_waits_for_the_adapter_listing_before_it_can_load(name: str) -> None:
    steps = _steps(name)
    press = next(
        n for n, step in enumerate(steps)
        if any(isinstance(step.get(k), str) and PRESS.search(step[k]) for k in ("click", "dblclick", "key"))
        or (step.get("keycombo") or [None])[-1] == "Enter"
    )
    helpers = {
        m.group(1)
        for step in steps[:press] if "eval" in step
        for m in re.finditer(r"window\.(__\w+)\s*=\s*\(\)\s*=>\s*" + re.escape(LISTING), step["eval"])
    }
    waits = [
        step for step in steps[:press]
        if step.get("fatal") is True
        and (LISTING in step["expect"] or any(f"{h}()" in step["expect"] for h in helpers))
    ]
    assert waits, name


def _clicks(steps: list[dict], target: str) -> list[int]:
    return [n for n, step in enumerate(steps) if step.get("click") == target]


# Empty bin deletes every image in the recycle bin from disk, so a pack that
# presses it proves first, from the page's own state, that the bin holds its
# own source alone, and before its first press that could load, that the
# page opened on a scratch gallery.
EMPTIES = [name for name in PACKS if _clicks(_steps(name), "#btn-empty-bin")]
BIN_ONLY_SOURCE = ("binMode", "binCount === 1", "galleryRecords.length === 1", "galleryRecords[0].id === __src",
                   "querySelectorAll('#gallery .card[data-id]').length === 1")
SCRATCH = ("total === 0", "!galleryRecords.length", "binCount === 0", "galleryReadHead === null",
           "!els.allModels.checked")


def test_the_packs_that_empty_the_bin_are_the_two_refine_packs() -> None:
    assert EMPTIES == ["card10-2b89ed5.json", "card10-ad01fdc.json"]


@pytest.mark.parametrize("name", EMPTIES)
def test_a_pack_empties_the_bin_only_when_it_holds_its_source_alone(name: str) -> None:
    steps = _steps(name)
    for press in _clicks(steps, "#btn-empty-bin"):
        opened = max(n for n in _clicks(steps, "#btn-bin") if n < press)
        checks = [
            step for step in steps[opened + 1 : press]
            if step.get("fatal") is True and all(term in step["expect"] for term in BIN_ONLY_SOURCE)
        ]
        assert checks, (name, press)


@pytest.mark.parametrize("name", EMPTIES)
def test_a_pack_that_empties_the_bin_starts_on_a_scratch_gallery(name: str) -> None:
    steps = _steps(name)
    press = next(
        n for n, step in enumerate(steps)
        if any(isinstance(step.get(k), str) and PRESS.search(step[k]) for k in ("click", "dblclick", "key"))
        or (step.get("keycombo") or [None])[-1] == "Enter"
    )
    checks = [
        step for step in steps[:press]
        if step.get("fatal") is True and all(term in step["expect"] for term in SCRATCH)
    ]
    assert checks, name


@pytest.mark.parametrize("sha", ["1744dc3", "ctrl-c-idle"])
def test_the_ctrl_c_packs_leave_the_signal_to_a_person(sha: str) -> None:
    steps = _steps(f"card10-{sha}.json")
    asks = [n for n, step in enumerate(steps) if ">>> NOW: type Ctrl-C" in step.get("log", "")]
    assert len(asks) == 1
    # Nothing sends it: no keycombo of the pack's is a Ctrl-C, and the waits
    # after the ask are evals, so a person slower than one of them is no
    # failure until the last check.
    assert not any(step.get("keycombo") == ["Control", "c"] for step in steps)
    after = steps[asks[0] + 1 :]
    assert sum(1 for step in after if "eval" in step and "__until(" in step["eval"]) >= 2
