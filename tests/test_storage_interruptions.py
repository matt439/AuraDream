"""Real process-death recovery at bundle write/install/publication boundaries."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("boundary", ["export-write", "export-verified", "export-removed",
                                     "import-installed", "import-before-publish", "import-published"])
def test_storage_worker_death_recovers_on_restart(tmp_path, boundary, record_property):
    command = [sys.executable, str(Path(__file__).resolve().parent.parent / "tools" / "storage_interrupt.py"),
               "--data", str(tmp_path), "--boundary", boundary]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(tmp_path.iterdir())
    report = json.loads(result.stdout)
    assert report["boundary"] == boundary
    assert report["reservation_reacquired"] is True
    assert report["reservation_held_before_kill"] is True
    assert report["killed_pid"] != report["restart_pid"]
    for name, value in report.items():
        record_property(name, json.dumps(value) if isinstance(value, dict) else value)
    print(json.dumps(report, sort_keys=True))
