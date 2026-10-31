"""The one instruction-edit gate: a backend instance's literal answer.

Since the release only the real diffusers backend's body is the literal
True; every other gate is the literal False, and no environment variable
moves either. The seam opens only for the literal True, and the measured
envelope still decides. The config pin compares bytes and never grants
execution. No weights, downloads or GPU.
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType

import pytest

from backend import instruction_edit as edit
from backend import instruction_edit_preflight as preflight
from backend import main
from backend.backends.base import GenerationBackend
from tests.test_instruction_edit_preflight import _check, _configs


BACKEND = Path(__file__).resolve().parents[1] / "backend"
HEX64 = re.compile(r"[0-9a-f]{64}")


class _Bare(GenerationBackend):
    """Only the abstract methods: the base class's own gate answers."""

    def load(self): ...
    def unload(self): ...
    def is_ready(self): return False
    def generate(self, request, seed, on_progress=None, should_cancel=None, on_preview=None): return []
    def describe(self): return {"model_id": "x", "quantization": "none", "offload": "none"}


def _real_backend():
    from backend.backends.diffusers_backend import DiffusersBackend

    return DiffusersBackend()


def test_the_base_answers_false_and_the_real_backend_the_literal_true():
    assert _Bare().instruction_edit_qualified() is False
    assert _real_backend().instruction_edit_qualified() is True


def test_no_environment_variable_opens_the_real_backend(monkeypatch):
    names = set()
    for path in BACKEND.rglob("*.py"):
        names.update(re.findall(r"AURADREAM_[A-Z0-9_]+", path.read_text(encoding="utf-8")))
    # And the names a later change might be tempted to invent for it.
    names.update({
        "AURADREAM_INSTRUCTION_EDIT", "AURADREAM_INSTRUCTION_EDIT_QUALIFIED", "AURADREAM_EDIT",
        "AURADREAM_EDIT_QUALIFIED", "AURADREAM_KONTEXT", "AURADREAM_QUALIFIED", "AURADREAM_UNSAFE",
    })
    for name in names:
        monkeypatch.setenv(name, "true")
    # Nothing opens the base, and the real backend's answer is its literal,
    # neither widened nor closed by a setting: the envelope decides there.
    assert _real_backend().instruction_edit_qualified() is True
    assert _Bare().instruction_edit_qualified() is False
    for name in names:
        monkeypatch.setenv(name, "false")
    assert _real_backend().instruction_edit_qualified() is True
    assert _Bare().instruction_edit_qualified() is False


def test_only_the_diffusers_backend_gate_returns_the_literal_true():
    found = []
    for path in BACKEND.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        owners = {
            id(item): node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef) for item in node.body
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "instruction_edit_qualified":
                body = [
                    statement for statement in node.body
                    if not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
                            and isinstance(statement.value.value, str))
                ]
                # One literal return: no setting, environment or call is read.
                assert len(body) == 1 and isinstance(body[0], ast.Return), path
                value = body[0].value
                owner = (path.name, owners.get(id(node)))
                released = owner == ("diffusers_backend.py", "DiffusersBackend")
                assert isinstance(value, ast.Constant) and value.value is released, path
                assert not node.decorator_list, path
                found.append(owner)
            # Nothing may rebind it: no assignment, lambda or setattr by name.
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
                    assert name != "instruction_edit_qualified", path
            if isinstance(node, ast.Constant) and node.value == "instruction_edit_qualified":
                pytest.fail(f"{path} names the gate as a string (setattr/getattr rebinding)")
    assert ("base.py", "GenerationBackend") in found
    assert ("diffusers_backend.py", "DiffusersBackend") in found


def test_execution_opens_only_for_the_literal_true():
    spec = edit.InstructionEditSpec(source=1, recipe="kontext-v1")
    for closed in (False, 1, "true", object(), None):
        with pytest.raises(ValueError, match="not available"):
            edit.require_execution_available(spec, qualified=closed)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not available"):
        edit.require_execution_available(spec)
    assert edit.require_execution_available(spec, qualified=True) is None
    # Structure is still checked before the gate, open or closed.
    for qualified in (False, True):
        with pytest.raises(ValueError):
            edit.require_execution_available({"source": 0, "recipe": "kontext-v1"}, qualified=qualified)


def test_model_gate_always_refuses_qwen_and_opens_kontext_only_when_qualified():
    # Qwen has no recipe: its own words, never Kontext's measurement hold.
    for model_id in ("Qwen/Qwen-Image-Edit", "QWEN/QWEN-IMAGE-EDIT"):
        for qualified in (False, True):
            with pytest.raises(ValueError) as caught:
                edit.require_model_available(model_id, qualified=qualified)
            assert str(caught.value) == edit.INSTRUCTION_EDITOR_UNSUPPORTED
    for closed in (False, 1, "yes"):
        for model_id in (edit.KONTEXT_V1.model_id, edit.KONTEXT_V1.model_id.upper()):
            with pytest.raises(ValueError) as caught:
                edit.require_model_available(model_id, qualified=closed)  # type: ignore[arg-type]
            assert str(caught.value) == edit.INSTRUCTION_EDIT_UNAVAILABLE
    # Open, only the recipe's exact id loads, as `bind_request` compares it,
    # and only inside the measured loader envelope.
    measured = edit.LoaderChoice(
        quantization="fp8", offload="model", device="cuda", platform="win32",
        loader_execution=None, diffusers_version="0.40.0",
    )
    assert edit.require_model_available(edit.KONTEXT_V1.model_id, qualified=True, loader=measured) is None
    with pytest.raises(ValueError) as caught:
        edit.require_model_available(edit.KONTEXT_V1.model_id, qualified=True)
    assert str(caught.value) == edit.INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="an unknown loader")
    with pytest.raises(ValueError) as caught:
        edit.require_model_available(
            edit.KONTEXT_V1.model_id, qualified=True, loader=dataclasses.replace(measured, offload="sequential"),
        )
    assert str(caught.value) == edit.INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="sequential offload")
    for loader in (None, measured):
        with pytest.raises(ValueError) as caught:
            edit.require_model_available(edit.KONTEXT_V1.model_id.lower(), qualified=True, loader=loader)
        assert str(caught.value) == edit.INSTRUCTION_EDIT_EXACT_ID
    assert edit.require_model_available("black-forest-labs/FLUX.1-Krea-dev") is None


def test_editor_classes_are_refused_in_their_own_words():
    # Its callers are the released backend: Kontext's class elsewhere is the id's fault.
    assert edit.editor_class_refusal("FluxKontextPipeline") == edit.INSTRUCTION_EDIT_EXACT_ID
    for class_name in ("QwenImageEditPipeline", "QwenImageEditPlusPipeline"):
        refusal = edit.editor_class_refusal(class_name)
        assert refusal == edit.INSTRUCTION_EDITOR_UNSUPPORTED and "measured" not in refusal


_ENVELOPE_LOADER = edit.LoaderChoice(
    quantization="fp8", offload="model", device="cuda", platform="win32",
    loader_execution=None, diffusers_version="0.40.0",
)


def test_text_to_image_refusal_names_the_gate_unless_an_edit_can_queue():
    kontext = edit.KONTEXT_V1.model_id
    assert edit.text_to_image_refusal(kontext, qualified=True, loader=_ENVELOPE_LOADER) == (
        edit.INSTRUCTION_EDIT_TEXT_TO_IMAGE
    )
    for qualified in (False, 1):
        held = edit.text_to_image_refusal(kontext, qualified=qualified, loader=_ENVELOPE_LOADER)  # type: ignore[arg-type]
        assert held.endswith(edit.INSTRUCTION_EDIT_UNAVAILABLE) and "Use Edit by instruction" not in held
    # Open but outside the envelope: the loader's words, never the pointer.
    int8 = dataclasses.replace(_ENVELOPE_LOADER, quantization="int8")
    held = edit.text_to_image_refusal(kontext, qualified=True, loader=int8)
    assert "int8 quantization is not measured" in held and "Use Edit by instruction" not in held


def test_unavailable_reason_gives_its_five_answers():
    kontext = edit.KONTEXT_V1.model_id
    krea = "black-forest-labs/FLUX.1-Krea-dev"
    # 1. The gate closed: UNAVAILABLE, whatever is selected or loaded.
    for loader in (None, _ENVELOPE_LOADER):
        assert edit.edit_unavailable_reason(kontext, qualified=False, loader=loader) == edit.INSTRUCTION_EDIT_UNAVAILABLE
        assert edit.edit_unavailable_reason(kontext, qualified=1, loader=loader) == (  # type: ignore[arg-type]
            edit.INSTRUCTION_EDIT_UNAVAILABLE
        )
        assert edit.edit_unavailable_reason(krea, qualified=False, loader=loader) == edit.INSTRUCTION_EDIT_UNAVAILABLE
    # 2. Open, another model: NEEDS_MODEL, before the loader is asked.
    assert edit.edit_unavailable_reason(krea, qualified=True, loader=_ENVELOPE_LOADER) == (
        edit.INSTRUCTION_EDIT_NEEDS_MODEL
    )
    # Exact, as `bind_request` compares: a case variant needs the editor selected.
    assert edit.edit_unavailable_reason(kontext.upper(), qualified=True, loader=_ENVELOPE_LOADER) == (
        edit.INSTRUCTION_EDIT_NEEDS_MODEL
    )
    # 3. Open, Kontext, outside the loader envelope: LOADER_UNQUALIFIED naming it.
    prefix = edit.INSTRUCTION_EDIT_LOADER_UNQUALIFIED.split("{choice}")[0]
    for change, words in (
        ({"quantization": "int8"}, "int8 quantization"), ({"offload": "sequential"}, "sequential offload"),
        ({"device": "cpu"}, "the device 'cpu'"), ({"platform": "linux"}, "Linux (WSL)"),
        ({"diffusers_version": "0.41.0"}, "diffusers 0.41.0"),
    ):
        reason = edit.edit_unavailable_reason(
            kontext, qualified=True, loader=dataclasses.replace(_ENVELOPE_LOADER, **change),
        )
        assert reason == edit.INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice=words)
    # Without its loader facts an open gate is never inside the envelope.
    assert edit.edit_unavailable_reason(kontext, qualified=True).startswith(prefix)  # type: ignore[union-attr]
    # 4 and 5 are the source's: the server answers None inside the envelope,
    # and only a 1:1 source's geometry passes (`geometry_refusal`).
    assert edit.edit_unavailable_reason(kontext, qualified=True, loader=_ENVELOPE_LOADER) is None
    assert edit.geometry_refusal(edit.geometry(768, 512)) == edit.INSTRUCTION_EDIT_ASPECT_UNQUALIFIED.format(
        a=3, b=2, w=576, h=384,
    )
    assert edit.geometry_refusal(edit.geometry(256, 256)) is None


def test_replay_reason_points_at_the_dialog_only_where_it_could_queue():
    assert edit.replay_held_reason(None) == edit.INSTRUCTION_EDIT_REPLAY_HELD
    assert edit.replay_held_reason(edit.INSTRUCTION_EDIT_NEEDS_MODEL) == edit.INSTRUCTION_EDIT_REPLAY_HELD
    assert edit.replay_held_reason(edit.INSTRUCTION_EDIT_UNAVAILABLE) == edit.INSTRUCTION_EDIT_UNAVAILABLE
    loader = edit.INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="int8 quantization")
    assert edit.replay_held_reason(loader) == loader


def test_a_backend_reports_its_host_and_builds_its_loader_from_the_selection():
    import sys

    from backend.config import settings
    from backend.model_selection import current_settings

    selected = current_settings(settings)
    assert _Bare().instruction_edit_host() == (sys.platform, edit.installed_diffusers_version())
    loader = _Bare().instruction_edit_loader(selected)
    assert (loader.quantization, loader.offload, loader.device) == (
        selected.quantization, selected.offload, selected.device,
    )
    assert loader.loader_execution is selected.loader_execution
    # The closed gate answers first, whatever the host.
    assert _Bare().instruction_edit_refusal(selected) == edit.INSTRUCTION_EDIT_UNAVAILABLE


def test_wording_keeps_the_phrases_its_callers_match():
    assert "not available on this backend" in edit.INSTRUCTION_EDIT_UNAVAILABLE
    assert "Kontext" in edit.INSTRUCTION_EDIT_UNAVAILABLE and "card" in edit.INSTRUCTION_EDIT_UNAVAILABLE
    assert "awaiting" not in edit.INSTRUCTION_EDIT_UNAVAILABLE and "build" not in edit.INSTRUCTION_EDIT_UNAVAILABLE
    assert "not available" in edit.INSTRUCTION_EDITOR_UNSUPPORTED and "no recipe" in edit.INSTRUCTION_EDITOR_UNSUPPORTED
    for text in (edit.INSTRUCTION_EDIT_NEEDS_MODEL, edit.INSTRUCTION_EDIT_TEXT_TO_IMAGE, edit.INSTRUCTION_EDIT_REPLAY_HELD):
        assert "Kontext" in text or "Edit by instruction" in text


def test_status_defaults_take_the_backend_answer_over_the_profile():
    kontext = edit.KONTEXT_V1.model_id
    held = main._merged_defaults({"model_id": kontext})
    assert held["instruction_edit_supported"] is False
    assert held["instruction_edit_reason"] == edit.INSTRUCTION_EDIT_UNAVAILABLE
    opened = main._merged_defaults({
        "model_id": kontext, "instruction_edit_supported": True, "instruction_edit_reason": None,
    })
    assert opened["instruction_edit_supported"] is True and opened["instruction_edit_reason"] is None
    needs = main._merged_defaults({
        "model_id": "black-forest-labs/FLUX.1-Krea-dev", "instruction_edit_supported": False,
        "instruction_edit_reason": edit.INSTRUCTION_EDIT_NEEDS_MODEL,
    })
    assert needs["instruction_edit_reason"] == edit.INSTRUCTION_EDIT_NEEDS_MODEL
    # Generate's hold on the editor follows the backend's answer too.
    assert held["text_to_image_reason"] == edit.text_to_image_refusal(kontext, qualified=False)
    answered = main._merged_defaults({
        "model_id": kontext, "instruction_edit_supported": True, "instruction_edit_reason": None,
        "text_to_image_reason": edit.INSTRUCTION_EDIT_TEXT_TO_IMAGE,
    })
    assert answered["text_to_image_reason"] == edit.INSTRUCTION_EDIT_TEXT_TO_IMAGE
    plain = main._merged_defaults({"model_id": "black-forest-labs/FLUX.1-Krea-dev", "text_to_image_reason": "x"})
    assert plain["text_to_image_reason"] is None


def test_status_serves_the_held_reason_as_a_default(client):
    defaults = client.get("/api/status").json()["defaults"]
    assert defaults["instruction_edit_supported"] is False
    assert defaults["instruction_edit_reason"] == edit.INSTRUCTION_EDIT_UNAVAILABLE


# Config pin


def _hashes(configs):
    return {
        path: hashlib.sha256(json.dumps(
            configs[path], sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        for path in preflight.CONFIG_PATHS
    }


def test_pin_covers_every_config_path_with_lowercase_sha256():
    assert isinstance(preflight.QUALIFIED_CONFIG_HASHES, MappingProxyType)
    assert tuple(preflight.QUALIFIED_CONFIG_HASHES) == preflight.CONFIG_PATHS
    assert all(HEX64.fullmatch(value) for value in preflight.QUALIFIED_CONFIG_HASHES.values())
    assert len(set(preflight.QUALIFIED_CONFIG_HASHES.values())) == len(preflight.CONFIG_PATHS)


def test_synthetic_configs_validate_but_do_not_match_the_pin():
    result = _check()
    assert result.matches_pin is False and result.execution_available is False
    with pytest.raises(ValueError, match="preflight refused") as caught:
        preflight.require_pinned(result)
    assert not HEX64.search(str(caught.value))


def test_pin_passes_matching_bytes_and_refuses_one_changed_config(monkeypatch):
    configs = _configs()
    monkeypatch.setattr(preflight, "QUALIFIED_CONFIG_HASHES", MappingProxyType(_hashes(configs)))
    result = _check(configs)
    assert result.matches_pin is True
    assert preflight.require_pinned(result) is result
    assert result.execution_available is False
    for path in preflight.CONFIG_PATHS:
        changed = copy.deepcopy(configs)
        if path == "model_index.json":
            changed[path]["_diffusers_version"] = "synthetiC"
        else:
            changed[path]["_diffusers_version"] = "s"
        check = _check(changed)
        assert check.matches_pin is False
        with pytest.raises(ValueError, match=re.escape(path)) as caught:
            preflight.require_pinned(check)
        assert not HEX64.search(str(caught.value))


def test_pin_refuses_a_forged_preflight_identity(monkeypatch):
    configs = _configs()
    monkeypatch.setattr(preflight, "QUALIFIED_CONFIG_HASHES", MappingProxyType(_hashes(configs)))
    result = _check(configs)
    for change in ({"revision": "main"}, {"model_id": "black-forest-labs/FLUX.1-dev"},
                   {"config_hashes": result.config_hashes[:-1]},
                   {"config_hashes": tuple(reversed(result.config_hashes))}):
        forged = dataclasses.replace(result, **change)
        assert forged.matches_pin is False
        with pytest.raises(ValueError, match="preflight refused"):
            preflight.require_pinned(forged)
    with pytest.raises(ValueError, match="preflight refused"):
        preflight.require_pinned({"config_hashes": result.config_hashes})  # type: ignore[arg-type]
