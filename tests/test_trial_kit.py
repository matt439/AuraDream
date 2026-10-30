"""The DEV#21 trial kit's driver stays aimed at the current page, cold.

docs/design/trials/README.md, "Refresh for the current app (DEV#21)", stage 1:
`steps_for()` builds from a synthetic version 2 manifest; every `#id` it
clicks or fills exists in `web/index.html` (or is one a page script makes);
a version 1 manifest is refused; the size fields are reached through their
disclosure, the queue row by its named cancel button. No server, no browser.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tools import trial_fixture, trial_walkthrough

ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
SCRIPTS = "\n".join(path.read_text(encoding="utf-8") for path in sorted((ROOT / "web").glob("*.js")))


def _manifest(variant: str = "A", version: int = 2) -> dict:
    """What `trial_fixture.prepare` writes, with made-up ids (never read as real)."""
    setup = {"id": "a" * 32, "name": "Lighthouse setup A", "prompt": "an amber paper lighthouse, saved setup study",
             "width": 512, "height": 768, "seed": 4600}
    return {
        "fixture_version": version, "variant": variant, "port": 8312, "model": trial_fixture.MODEL,
        "data": str(ROOT / "scratch-never-created"),
        "roles": {"keeper": 3, "complete_last": 62, "source": 65, "child": 66, "binned_reject": 67,
                  "live_reject": 68, "foreign": [2, 1], "pick_in_order": [3, 66],
                  "then_refine_base": 69, "then_refine_pass": 70, "setup": setup,
                  "saved_search": {"id": "b" * 32, "name": "Lighthouse grid A", "query": "run:0123456789ab"},
                  "snippet": {"id": "c" * 32, "name": "Harbor light A", "text": "soft harbor light, long shadows"}},
        "jobs": {"complete": {"run_id": "0123456789ab"}},
        "counts": {"live": 69, "binned": 1, "all_rows": 70},
    }


def _pressed(steps: list[dict]) -> list[str]:
    """Every selector a step presses (a click; a fill starts with one)."""
    return [step["click"] for step in steps if isinstance(step.get("click"), str)]


def _ids(selector: str) -> list[str]:
    return re.findall(r"#([A-Za-z][\w-]*)", selector)


@pytest.mark.parametrize("variant", ["A", "B"])
def test_every_pressed_id_is_on_the_page(variant):
    steps = trial_walkthrough.steps_for(_manifest(variant))
    missing = []
    for selector in _pressed(steps):
        for element_id in _ids(selector):
            # trial-* ids are tagged at run time onto a runtime-numbered target.
            if element_id.startswith("trial-"):
                continue
            if f'id="{element_id}"' not in INDEX and f"'{element_id}'" not in SCRIPTS:
                missing.append(element_id)
    assert not missing, f"the driver presses ids the page no longer has: {sorted(set(missing))}"


def test_trial_tags_are_set_before_they_are_pressed():
    steps = trial_walkthrough.steps_for(_manifest())
    for index, step in enumerate(steps):
        for element_id in _ids(step.get("click", "") if isinstance(step.get("click"), str) else ""):
            if element_id.startswith("trial-"):
                assert any(element_id in str(earlier.get("eval", "")) for earlier in steps[:index]), element_id


@pytest.mark.parametrize("version", [1, None, 3])
def test_other_manifest_versions_are_refused(version, tmp_path, monkeypatch, capsys):
    manifest = _manifest(version=version)
    with pytest.raises(ValueError, match="fixture version"):
        trial_walkthrough.steps_for(manifest)
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "steps.json"
    monkeypatch.setattr("sys.argv", ["trial_walkthrough.py", "--manifest", str(path), "--output", str(output)])
    with pytest.raises(SystemExit) as refused:
        trial_walkthrough.main()
    assert refused.value.code == 2 and not output.exists()
    assert "fixture version" in capsys.readouterr().err


def test_fixture_and_driver_agree_on_the_version():
    assert trial_fixture.FIXTURE_VERSION == trial_walkthrough.FIXTURE_VERSION == 2


def test_size_fields_are_reached_through_their_disclosure():
    """W/H sit in the closed Custom dimensions disclosure (UI#19): the first
    press on either comes after a press on its summary, as a participant's."""
    pressed = _pressed(trial_walkthrough.steps_for(_manifest()))
    assert re.search(r'<details[^>]*id="custom-size"[^>]*>(?:(?!</details>).)*id="width"', INDEX, re.S)
    summary = pressed.index("#custom-size-summary")
    assert summary < pressed.index("#width") and summary < pressed.index("#height")


def test_every_fill_checks_its_press_landed():
    steps = trial_walkthrough.steps_for(_manifest())
    checked = 0
    for index, step in enumerate(steps):
        if step.get("keycombo") == ["Control", "a"] and isinstance(steps[index - 2].get("click"), str):
            selector = steps[index - 2]["click"]
            assert steps[index - 1] == {"expect": f"document.activeElement===document.querySelector({json.dumps(selector)})"}
            checked += 1
    assert checked >= 20


def test_the_waiting_row_is_cancelled_by_its_named_button():
    steps = trial_walkthrough.steps_for(_manifest())
    tag = next(step["eval"] for step in steps if "trial-cancel-waiting" in str(step.get("eval", "")))
    assert "[data-queue-action='cancel']" in tag
    assert 'queueAction' in SCRIPTS and 'data-queue-action="${action}"' in SCRIPTS


def test_keyboard_routes_are_used():
    keys = [step["key"] for step in trial_walkthrough.steps_for(_manifest()) if "key" in step]
    # Focus search, contextual help and More seeds by their keys (UI#17).
    assert {"/", "?", "m"} <= set(keys)


def test_the_driver_reads_only_roles_the_fixture_writes():
    walkthrough = (ROOT / "tools" / "trial_walkthrough.py").read_text(encoding="utf-8")
    fixture = (ROOT / "tools" / "trial_fixture.py").read_text(encoding="utf-8")
    roles_block = fixture[fixture.index('manifest["roles"] = {'):fixture.index('manifest["counts"]')]
    read = set(re.findall(r"roles\[['\"](\w+)['\"]\]", walkthrough))
    assert read and all(f'"{name}":' in roles_block for name in read)
    for name in ("setup", "saved_search", "snippet", "then_refine_base", "then_refine_pass"):
        assert f'"{name}":' in roles_block


def test_the_more_seeds_refusal_is_recorded():
    fixture = (ROOT / "tools" / "trial_fixture.py").read_text(encoding="utf-8")
    assert '("foreign_more_seeds", "/api/more-seeds"' in fixture
