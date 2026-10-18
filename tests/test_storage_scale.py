"""A real disposable process exercising multi-MB bundles and a 200-image bin."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("encrypted", [False, True], ids=["plain", "password"])
def test_storage_scale_in_a_disposable_process(tmp_path, encrypted, record_property):
    command = [sys.executable, str(Path(__file__).resolve().parent.parent / "tools" / "storage_scale.py"),
               "--data", str(tmp_path)]
    if encrypted:
        command.append("--encrypted")
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(tmp_path.iterdir())  # the worker removes only the scratch it created
    report = json.loads(result.stdout)
    assert (report["rows"], report["binned"], report["live_after_empty"], report["masks_after_empty"]) == (208, 200, 11, 2)
    assert report["encrypted"] is encrypted
    for key, value in report.items():
        record_property(key, json.dumps(value) if isinstance(value, dict) else value)
    print(json.dumps(report, sort_keys=True))
