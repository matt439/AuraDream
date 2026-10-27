"""`tools/browser.mjs` leaves no browser profile behind, however its run ends.

Each run gives the browser a profile of its own under the system's temp
directory. Removing it after a bare `proc.kill()` failed quietly whenever
Edge's other processes still held it, and a run killed outright never got
that far: the profiles reached 49 GiB in Windows' Temp. These run the real
script against a real browser, with the temp directory pointed at
a directory of each test's own, and skip where Node or a browser is missing.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "local-diffusion-browser-"
# browser.mjs's own list, after LOCAL_DIFFUSION_BROWSER.
_CANDIDATES = [
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/microsoft-edge",
]


def _browser() -> str | None:
    named = os.environ.get("LOCAL_DIFFUSION_BROWSER")
    return next((p for p in [named, *_CANDIDATES] if p and Path(p).exists()), None)


pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or _browser() is None, reason="needs Node and Edge or Chrome",
)


def _env(temp: Path) -> dict[str, str]:
    # os.tmpdir() reads TMPDIR on Linux and macOS, TEMP then TMP on Windows.
    return {**os.environ, "TMPDIR": str(temp), "TEMP": str(temp), "TMP": str(temp)}


@pytest.fixture
def temp():
    """The run's temp directory, removed afterwards whatever it holds.

    Short, under the system's own temp rather than `tmp_path`: Chromium on
    Linux puts its singleton socket in the temp directory, and under pytest's
    long per-test path that passed the 108 bytes a socket's path may take,
    so the browser never came up.
    """
    path = Path(tempfile.mkdtemp(prefix="ld-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _command(temp: Path, steps: list[dict]) -> list[str]:
    path = temp / "steps.json"
    path.write_text(json.dumps(steps), encoding="utf-8")
    return ["node", str(ROOT / "tools" / "browser.mjs"), str(path)]


def _run(temp: Path, steps: list[dict]) -> subprocess.CompletedProcess:
    return subprocess.run(
        _command(temp, steps), cwd=ROOT, env=_env(temp), capture_output=True, text=True, timeout=120,
    )


def _profiles(temp: Path) -> list[str]:
    return sorted(p.name for p in temp.iterdir() if p.name.startswith(PREFIX))


def test_a_passing_run_removes_its_profile(temp):
    done = _run(temp, [{"navigate": "about:blank"}, {"expect": "1 + 1 === 2"}])
    assert done.returncode == 0, done.stdout + done.stderr
    assert _profiles(temp) == []


def test_a_failed_expect_removes_its_profile(temp):
    done = _run(temp, [{"expect": "true"}, {"expect": "false"}, {"expect": "null.x"}])
    assert done.returncode == 1, done.stdout + done.stderr
    assert "ok   true" in done.stdout, done.stdout + done.stderr
    assert _profiles(temp) == []


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no catchable SIGTERM")
def test_a_run_stopped_midway_removes_its_profile(temp):
    run = subprocess.Popen(
        _command(temp, [{"wait": 60000}]), cwd=ROOT, env=_env(temp),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        # Stopped once the browser is up and holding its profile.
        deadline = time.monotonic() + 30
        while not any((temp / name / "DevToolsActivePort").exists() for name in _profiles(temp)):
            assert run.poll() is None and time.monotonic() < deadline, "the browser did not come up"
            time.sleep(0.1)
        time.sleep(0.5)
        run.send_signal(signal.SIGTERM)
        run.wait(timeout=30)
    finally:
        if run.poll() is None:
            run.kill()
    assert _profiles(temp) == []


def test_a_stale_profile_is_swept_and_a_live_one_kept(temp):
    stale, live = temp / f"{PREFIX}crashed", temp / f"{PREFIX}running"
    for profile in (stale, live):
        (profile / "Default").mkdir(parents=True)
        (profile / "Default" / "Cache").write_bytes(b"x" * 1024)
    hours_ago = time.time() - 7 * 3600
    os.utime(stale, (hours_ago, hours_ago))
    done = _run(temp, [{"expect": "true"}])
    assert done.returncode == 0, done.stdout + done.stderr
    assert _profiles(temp) == [live.name]


def test_all_screenshot_forms_use_the_selected_root_and_survive_failure(temp):
    output = temp / "retained" / "screenshots"
    done = subprocess.run(
        _command(temp, [
            {"navigate": "data:text/html,<body style='width:200px;height:200px'>screenshots</body>"},
            {"shot": "viewport.png"}, {"fullshot": "nested/full.png"},
            {"elshot": ["body", "nested/element.png"]}, {"expect": "false"},
        ]), cwd=ROOT, env={**_env(temp), "LOCAL_DIFFUSION_SCREENSHOT_DIR": str(output)},
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 1, done.stdout + done.stderr
    assert sorted(path.relative_to(output).as_posix() for path in output.rglob("*.png")) == [
        "nested/element.png", "nested/full.png", "viewport.png",
    ]
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in output.rglob("*.png"))
    assert _profiles(temp) == []
