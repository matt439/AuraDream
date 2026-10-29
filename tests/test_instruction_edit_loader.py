"""Custom editor aliases refuse on their index before any synthetic weight load."""
from __future__ import annotations

from dataclasses import replace
import json

import pytest

from backend import instruction_edit
from backend.backends import diffusers_backend as module, loading
from backend.launch import CHECKPOINTS, FORMATS
from tests.test_pipeline_ownership import synthetic_loader


@pytest.mark.parametrize("class_name", sorted(loading.INSTRUCTION_EDITOR_CLASSES))
def test_custom_editor_class_refuses_before_any_pipeline_weight_preparation(synthetic_loader, monkeypatch, tmp_path, class_name):
    backend, _, events = synthetic_loader
    selected = replace(module.settings, model_id="owned/custom-editor-alias", device="cpu", offload="none", quantization="none")
    monkeypatch.setattr(module, "current_settings", lambda startup: selected)
    path = tmp_path / "model_index.json"
    path.write_text(json.dumps({"_class_name": class_name}), encoding="utf-8")
    reads = []

    def supplied(model_id, filename):
        reads.append((model_id, filename))
        assert model_id == selected.model_id and filename == "model_index.json"
        return str(path)

    monkeypatch.setattr(loading, "hub_file", supplied)
    # The existing optional optimization is deliberately inconclusive; a
    # separate strict decision must refuse before reaching this fallback.
    monkeypatch.setattr(loading, "unused_components", lambda model_id: (None, {}))
    with pytest.raises(ValueError, match="Instruction editing is not available"):
        backend.load()
    assert reads == [(selected.model_id, "model_index.json")]
    assert not backend.is_ready() and backend._call_params == set()
    assert not {"overrides", "pretrained", "place"} & set(events)


@pytest.mark.parametrize("broken", ["unreadable", "json", "list", "missing", "invalid-class"])
def test_uninspectable_custom_alias_cannot_fall_through_to_from_pretrained(synthetic_loader, monkeypatch, tmp_path, broken):
    backend, _, events = synthetic_loader
    selected = replace(module.settings, model_id="owned/custom-alias", device="cpu", offload="none", quantization="none")
    monkeypatch.setattr(module, "current_settings", lambda startup: selected)
    path = tmp_path / "model_index.json"
    value = {"list": [], "missing": {}, "invalid-class": {"_class_name": ["custom", "Pipeline"]}}.get(broken)
    path.write_text("{" if broken == "json" else json.dumps(value), encoding="utf-8")

    def supplied(model_id, filename):
        assert filename == "model_index.json"
        if broken == "unreadable":
            raise TimeoutError("synthetic index timeout")
        return str(path)

    monkeypatch.setattr(loading, "hub_file", supplied)
    with pytest.raises(ValueError, match="custom checkpoint"):
        backend.load()
    assert not backend.is_ready() and not {"overrides", "pretrained", "place"} & set(events)


def test_custom_noneditor_class_is_a_checked_identity_not_an_execution_grant(monkeypatch, tmp_path):
    path = tmp_path / "model_index.json"
    path.write_text('{"_class_name":"FluxPipeline"}', encoding="utf-8")
    monkeypatch.setattr(loading, "hub_file", lambda model_id, filename: str(path))
    assert loading.require_custom_pipeline_class("owned/ordinary-custom-id") == "FluxPipeline"


def test_catalogue_and_enumerated_nunchaku_ids_keep_their_existing_index_behavior(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("known catalogue identity reached custom-id index inspection")

    monkeypatch.setattr(loading, "hub_file", forbidden)
    ids = {checkpoint.repo for checkpoint in CHECKPOINTS}
    ids.update(repo for checkpoint in CHECKPOINTS for fmt, _ in FORMATS
               if (repo := checkpoint.lite_repo(fmt)) is not None)
    assert all(loading.require_custom_pipeline_class(model_id) is None for model_id in ids)


@pytest.mark.parametrize("model_id", [instruction_edit.KONTEXT_V1.model_id.upper(), "Qwen/Qwen-Image-Edit", "QWEN/QWEN-IMAGE-EDIT"])
def test_named_edit_only_checkpoints_refuse_without_any_index_or_framework_access(model_id):
    with pytest.raises(ValueError, match="not available"):
        instruction_edit.require_model_available(model_id)


def test_every_require_model_available_caller_passes_qualified_and_selection():
    # Every load entry asks the instance's gate and the selection's loader
    # facts (slice 3); a call that left either out would treat the gate as
    # closed, or an open gate as outside any envelope, without saying why.
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sources = [*sorted((root / "backend").rglob("*.py")), *sorted((root / "tools").rglob("*.py")),
               root / "tests" / "fakes.py"]
    calls = []
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            if name != "require_model_available":
                continue
            keywords = {keyword.arg for keyword in node.keywords}
            calls.append((path.relative_to(root).as_posix(), node.lineno, {"qualified", "loader"} <= keywords))
    callers = {path for path, _, _ in calls}
    # The load entries the plan names, and the stand-in's own copies.
    assert {
        "backend/backends/diffusers_backend.py", "backend/main.py", "backend/model_selection.py", "tests/fakes.py",
    } <= callers, calls
    assert [call for call in calls if not call[2]] == []
