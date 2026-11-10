"""The smoke driver hears what the page throws; a scripted socket, no browser.

`tools/browser.mjs` runs whole against a fixture that answers the DevTools
calls and delivers `Runtime.exceptionThrown` / `Runtime.exceptionRevoked` as
the browser would. What Chrome actually sends for a throwing handler or an
unhandled rejection is the browser's to say (docs/rules.md, Tests).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "browser_page_errors_fixture.mjs"

THROWN = {"method": "Runtime.exceptionThrown", "params": {"timestamp": 1, "exceptionDetails": {
    "exceptionId": 4, "text": "Uncaught", "lineNumber": 11, "columnNumber": 2,
    "url": "http://127.0.0.1:7861/static/app.js",
    "exception": {"type": "object", "description": "TypeError: press is not a function\n    at HTMLButtonElement.onclick (app.js:12:3)"},
}}}
REVOKED = {"method": "Runtime.exceptionRevoked", "params": {"reason": "Handler added to rejected promise", "exceptionId": 4}}


def _run(tmp_path: Path, steps: list[dict], script: dict) -> subprocess.CompletedProcess[str]:
    node = shutil.which("node")
    assert node is not None, "Node is required for the browser harness"
    path = tmp_path / "steps.json"
    path.write_text(json.dumps(steps), encoding="utf-8")
    request = dict(harness=str(ROOT / "tools/browser.mjs"), steps=str(path),
                   url=(ROOT / "tools/browser.mjs").as_uri(), script=script)
    environment = {**os.environ, "AURADREAM_BROWSER": "owned-synthetic-browser",
                   "TEMP": str(tmp_path), "TMP": str(tmp_path), "TMPDIR": str(tmp_path)}
    for name in ("AURADREAM_BROWSER_DIAGNOSTICS", "AURADREAM_SCREENSHOT_DIR", "AURADREAM_PORT_OFFSET"):
        environment.pop(name, None)
    done = subprocess.run([node, str(FIXTURE), json.dumps(request)],
                          cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert "unexpected" not in done.stderr, done.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["steps.json"]
    return done


def test_a_page_error_fails_a_run_whose_expects_all_pass(tmp_path):
    done = _run(tmp_path, [{"eval": "press()"}, {"expect": "after"}], {"press()": {"events": [THROWN]}})
    assert done.returncode == 1, done.stdout + done.stderr
    assert "  ! page error: Uncaught TypeError: press is not a function" in done.stdout
    assert "at HTMLButtonElement.onclick (app.js:12:3) (http://127.0.0.1:7861/static/app.js:12)" in done.stdout
    assert 'during step 1: {"eval":"press()"}' in done.stdout
    # The steps after it still run, and no expectation is counted as failed.
    assert "  ok   after" in done.stdout
    assert "expectation(s) failed" not in done.stdout
    assert "1 page error(s)" in done.stdout


def test_a_rejection_handled_later_is_taken_back(tmp_path):
    done = _run(tmp_path, [{"eval": "reject()"}, {"eval": "handle()"}, {"expect": "after"}],
                {"reject()": {"events": [THROWN]}, "handle()": {"events": [REVOKED]}})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "  ! page error: Uncaught TypeError" in done.stdout
    assert "page error 4 taken back: Handler added to rejected promise" in done.stdout
    assert "page error(s)" not in done.stdout


def test_each_page_error_is_counted_beside_failed_expects(tmp_path):
    second = json.loads(json.dumps(THROWN))
    second["params"]["exceptionDetails"].update(exceptionId=5, text="Uncaught (in promise)", url="",
                                                exception={"type": "string", "value": "plain reason"})
    done = _run(tmp_path, [{"eval": "press()"}, {"expect": "falsy"}],
                {"press()": {"events": [THROWN, second]}, "falsy": {"reply": {"result": {"value": False}}}})
    assert done.returncode == 1
    assert "  ! page error: Uncaught (in promise) plain reason\n" in done.stdout
    assert "1 expectation(s) failed" in done.stdout and "2 page error(s)" in done.stdout


def test_a_steps_own_exception_and_a_quiet_run_are_no_page_error(tmp_path):
    thrown_by_step = {"result": {"type": "object"}, "exceptionDetails": {
        "exceptionId": 9, "text": "Uncaught", "exception": {"description": "TypeError: step's own"}}}
    done = _run(tmp_path, [{"eval": "null.x"}, {"expect": "after"}], {"null.x": {"reply": thrown_by_step}})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "  ! TypeError: step's own" in done.stdout
    assert "page error" not in done.stdout
