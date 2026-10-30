"""Opt-in owned-browser diagnostics are bounded; no browser or network runs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "browser_diagnostics_fixture.mjs"


def test_failed_owned_browser_reports_bounded_stderr_tail(tmp_path):
    node = shutil.which("node")
    assert node is not None, "Node is required for the browser harness"
    steps = tmp_path / "steps.json"
    steps.write_text('[{"expect":"true"}]', encoding="utf-8")
    request = dict(harness=str(ROOT / "tools/browser.mjs"), steps=str(steps),
                   url=(ROOT / "tools/browser.mjs").as_uri())
    environment = {**os.environ, "AURADREAM_BROWSER": "owned-synthetic-browser",
                   "AURADREAM_BROWSER_DIAGNOSTICS": "1",
                   "TEMP": str(tmp_path), "TMP": str(tmp_path), "TMPDIR": str(tmp_path)}
    done = subprocess.run([node, str(FIXTURE), json.dumps(request)],
                          cwd=ROOT, env=environment, capture_output=True, text=True, timeout=15)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "the browser exited before DevTools was ready" in done.stderr
    assert "browser diagnostics: exit=7 signal=none" in done.stderr
    tail = done.stderr.split("browser stderr tail:\n", 1)[1].split("\nbrowser stderr tail [end]", 1)[0]
    assert len(tail) == 16384 and tail.endswith("final-owned-stderr")
    assert "clipped-fixture-prefix" not in done.stderr
    assert "unexpected network" not in done.stderr and "unexpected socket" not in done.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == ["steps.json"]
