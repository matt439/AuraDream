"""What the model switch and catalogue tests share: the checkpoint ids and
body they use, and the autouse reset of the selection and switch state.

`test_api_model_switch.py` and `test_api_model_catalogue.py` import these by
name, the autouse fixture included.
"""
from __future__ import annotations

import pytest

from backend import main, model_selection


ANIMAGINE = "cagliostrolab/animagine-xl-4.0"
TURBO = "krea/Krea-2-Turbo"
PAYLOAD = {"prompt": "a lighthouse", "width": 256, "height": 256}


@pytest.fixture(autouse=True)
def fresh_selection(monkeypatch):
    monkeypatch.setattr(model_selection, "_selection", (None, 0))
    monkeypatch.setattr(main, "_switching", False)
    monkeypatch.setattr(main, "_model_requests", 0)
    monkeypatch.setattr(main, "_model_request_paths", {})
    monkeypatch.setattr(main, "_switch_draining", False)
    # A refusal behind a held request comes after the drain; keep it short.
    monkeypatch.setattr(main, "SWITCH_DRAIN_SECONDS", 0.3)
