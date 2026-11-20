"""The rehearsal can run cold. Real receipts bind the device, software and local pin."""

import json
import subprocess
import sys

from tools import kontext_trial_environment as environment


def test_rehearsal_never_probes_the_card_cache_or_frameworks():
    code = """
import json, sys
from tools.kontext_trial_environment import capture_environment
value = capture_environment("fake")
print(json.dumps([value, sorted(set(sys.modules) & {
    "torch", "diffusers", "backend.config", "backend.storage", "huggingface_hub.constants"
})]))
"""
    process = subprocess.run([sys.executable, "-c", code], cwd=environment.ROOT,
                             capture_output=True, text=True, check=True)
    value, imported = json.loads(process.stdout)
    assert imported == []
    assert value["holds"] == []
    assert "gpu" not in value and "cached_config_sha256" not in value
    assert len(value["controller_sha256"]["tools/card_guard.py"]) == 64


def test_real_identity_holds_the_wrong_device_and_an_absent_snapshot(tmp_path, monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(environment, "_gpu_zero", lambda: {"name": "other", "index": "0", "uuid": "unused"})
    value = environment.capture_environment("real")
    assert "physical GPU 0 must be the RTX 5080" in value["holds"]
    assert any("config pin" in reason for reason in value["holds"])
    assert list(tmp_path.iterdir()) == []
