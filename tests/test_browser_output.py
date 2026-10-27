"""Portable screenshot names and output roots, without starting a browser."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")


def _node(body: str, value) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["node", "--input-type=module", "-e", (
            "import { screenshotNames, screenshotPath, screenshotRoot, writeScreenshot } "
            f"from {json.dumps((ROOT / 'tools' / 'browser-output.mjs').as_uri())};"
            "let input = ''; for await (const part of process.stdin) input += part;"
            "const value = JSON.parse(input);" + body
        )], input=json.dumps(value), capture_output=True, text=True, timeout=15,
    )


def _check_repository_smokes(smokes: list[dict]) -> subprocess.CompletedProcess:
    # Only screenshot fields are paths: C:\\gallery\\outputs in fixture text
    # is deliberately legitimate data rather than an output destination.
    return _node(
        "for (const {name, steps} of value) {"
        "try { for (const file of screenshotNames(steps)) screenshotPath('.', file); }"
        "catch (error) { throw new Error(name + ': ' + error.message); } }", smokes,
    )


def test_repository_smoke_screenshot_destinations_are_portable():
    smokes = [
        {"name": path.name, "steps": json.loads(path.read_text(encoding="utf-8"))}
        for path in sorted((ROOT / "tools").glob("smoke*.json"))
    ]
    done = _check_repository_smokes(smokes)
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize("kind", ["shot", "fullshot", "elshot"])
@pytest.mark.parametrize("name", [
    "/tmp/a.png", "C:\\Temp\\a.png", "C:a.png", "../a.png", "a/../../b.png",
    "CON.png", "NUL", "folder/Lpt1.png", "trailing./a.png", "trailing /a.png", "a.png.", "a.png ",
])
def test_repository_smoke_check_rejects_a_bad_destination(kind, name):
    step = {kind: ["body", name] if kind == "elshot" else name}
    done = _check_repository_smokes([{"name": "bad-smoke.json", "steps": [step]}])
    assert done.returncode != 0
    assert "bad-smoke.json: screenshot destination" in done.stderr


def test_every_screenshot_form_creates_its_output_under_one_root(tmp_path):
    root = tmp_path / "new" / "screenshots"
    done = _node(
        "const steps = [{shot: 'view/a.png'}, {fullshot: 'full/b.png'}, {elshot: ['body', 'element/c.png']}];"
        "const root = screenshotRoot(steps, value);"
        "for (const name of screenshotNames(steps)) writeScreenshot(root, name, 'cGl4ZWxz');"
        "console.log(root);", str(root),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert Path(done.stdout.strip()) == root
    assert sorted(path.relative_to(root).as_posix() for path in root.rglob("*.png")) == [
        "element/c.png", "full/b.png", "view/a.png",
    ]
    assert all(path.read_bytes() == b"pixels" for path in root.rglob("*.png"))


def test_no_screenshots_need_no_output_directory(tmp_path):
    root = tmp_path / "unused"
    done = _node("console.log(screenshotRoot([{expect: 'true'}], value));", str(root))
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip() == "null"
    assert not root.exists()
