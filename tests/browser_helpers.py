"""The cold runner the browser-harness preflight tests share.

Each test file drives the real `tools/browser.mjs` through its own `.mjs`
fixture, which captures the modules and poisons every later effect; the
fixtures stay separate (each guards different reads). What they share is
this Python side: the steps file, the owned temp and output directories,
the synthetic browser name, and the checks that nothing was written, the
steps file was left as it was and no private fixture value leaked.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "tools" / "browser.mjs"
SYNTHETIC_BROWSER = "owned-synthetic-browser"


def run_harness_cold(tmp_path: Path, fixture: Path, steps_name: str, text: str, *,
                     request: dict | None = None, environment: dict[str, str | None] | None = None,
                     secrets: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    """Run `fixture` over the harness with `text` as its steps file.

    `request` adds the fixture's own keys; `environment` sets variables over
    the shared ones, None removing one. Every `secrets` value must stay out
    of both streams.
    """
    script = tmp_path / steps_name
    script.write_text(text, encoding="utf-8")
    output = tmp_path / "must-not-create-output"
    temporary = tmp_path / "owned-temp"
    temporary.mkdir()
    body = dict(harness=str(HARNESS), script=str(script), url=HARNESS.as_uri(), output=str(output),
                temporary=str(temporary), **(request or {}))
    env = {**os.environ, "AURADREAM_BROWSER": SYNTHETIC_BROWSER,
           "AURADREAM_SCREENSHOT_DIR": str(output),
           "TMPDIR": str(temporary), "TEMP": str(temporary), "TMP": str(temporary)}
    for name, value in (environment or {}).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    done = subprocess.run(["node", str(fixture), json.dumps(body)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)
    assert script.read_text(encoding="utf-8") == text
    assert not output.exists() and list(temporary.iterdir()) == []
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(["owned-temp", steps_name])
    for secret in secrets:
        assert secret not in done.stdout + done.stderr
    return done
