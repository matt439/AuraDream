"""Supplied-config refusal and installed image-to-CPU-tensor audit, no weights."""
from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
import subprocess
import sys

import pytest

from backend import instruction_edit as edit
from backend import instruction_edit_preflight as preflight


def _configs():
    # These are synthetic boundary fixtures, never copied/qualified Hub configs.
    return {
        "model_index.json": {
            "_class_name": "FluxKontextPipeline", "_diffusers_version": "synthetic",
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "transformer": ["diffusers", "FluxTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "text_encoder_2": ["transformers", "T5EncoderModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "tokenizer_2": ["transformers", "T5TokenizerFast"],
            "image_encoder": [None, None], "feature_extractor": [None, None],
        },
        "scheduler/scheduler_config.json": {"_class_name": "FlowMatchEulerDiscreteScheduler"},
        "transformer/config.json": {
            "_class_name": "FluxTransformer2DModel", "in_channels": 64,
            "joint_attention_dim": 4096, "pooled_projection_dim": 768, "guidance_embeds": True,
        },
        "vae/config.json": {
            "_class_name": "AutoencoderKL", "block_out_channels": [128, 256, 512, 512],
            "latent_channels": 16, "in_channels": 3, "out_channels": 3,
            "scaling_factor": 0.3611, "shift_factor": 0.1159,
        },
        "text_encoder/config.json": {
            "architectures": ["CLIPTextModel"], "model_type": "clip_text_model",
            "hidden_size": 768, "max_position_embeddings": 77,
        },
        "text_encoder_2/config.json": {
            "architectures": ["T5EncoderModel"], "model_type": "t5", "d_model": 4096,
        },
    }


def _check(configs=None, **changes):
    return preflight.validate_configs(
        changes.get("model_id", edit.KONTEXT_V1.model_id),
        changes.get("revision", edit.KONTEXT_V1.revision),
        _configs() if configs is None else configs,
    )


def test_synthetic_preflight_records_pinned_config_identity_but_grants_no_execution():
    configs = _configs()
    before = copy.deepcopy(configs)
    result = _check(configs)
    assert configs == before
    assert result.revision == edit.KONTEXT_V1.revision
    assert result.model_id == edit.KONTEXT_V1.model_id
    assert (result.vae_scale_factor, result.latent_channels, result.max_sequence_length) == (8, 16, 512)
    assert result.execution_available is False
    assert tuple(path for path, _ in result.config_hashes) == preflight.CONFIG_PATHS
    assert all(len(digest) == 64 for _, digest in result.config_hashes)
    assert _check(json.loads(json.dumps(configs, sort_keys=True))) == result
    configs["scheduler/scheduler_config.json"]["shift"] = 2.0
    assert _check(configs).config_hashes != result.config_hashes
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.execution_available = True
    with pytest.raises(ValueError, match="not available"):
        edit.require_execution_available({"source": 1, "recipe": "kontext-v1"})


@pytest.mark.parametrize("change", [
    {"model_id": "black-forest-labs/FLUX.1-dev"}, {"revision": "main"}, {"revision": ""},
])
def test_preflight_requires_frozen_public_checkpoint_and_complete_pin(change):
    with pytest.raises(ValueError, match="recipe identity"):
        _check(**change)


@pytest.mark.parametrize("path", preflight.CONFIG_PATHS)
def test_preflight_requires_every_fixed_config_before_weights(path):
    configs = _configs()
    del configs[path]
    with pytest.raises(ValueError, match="config inventory"):
        _check(configs)


@pytest.mark.parametrize("path,key,value", [
    ("model_index.json", "_class_name", "FluxPipeline"),
    ("model_index.json", "transformer", ["custom.library", "FluxTransformer2DModel"]),
    ("model_index.json", "tokenizer_2", ["transformers", "T5Tokenizer"]),
    ("model_index.json", "image_encoder", ["transformers", "CLIPVisionModelWithProjection"]),
    ("model_index.json", "unexpected_component", ["diffusers", "AutoencoderKL"]),
    ("scheduler/scheduler_config.json", "_class_name", "EulerDiscreteScheduler"),
    ("transformer/config.json", "in_channels", 32),
    ("transformer/config.json", "joint_attention_dim", 2048),
    ("transformer/config.json", "pooled_projection_dim", 1024),
    ("transformer/config.json", "guidance_embeds", 1),
    ("vae/config.json", "block_out_channels", [128, 256, 512]),
    ("vae/config.json", "block_out_channels", [128, True, 512, 512]),
    ("vae/config.json", "latent_channels", 4),
    ("vae/config.json", "in_channels", True),
    ("vae/config.json", "out_channels", 4),
    ("vae/config.json", "scaling_factor", 0),
    ("vae/config.json", "shift_factor", None),
    ("vae/config.json", "shift_factor", float("inf")),
    ("text_encoder/config.json", "architectures", ["CLIPModel"]),
    ("text_encoder/config.json", "model_type", "clip"),
    ("text_encoder/config.json", "hidden_size", "768"),
    ("text_encoder/config.json", "max_position_embeddings", 512),
    ("text_encoder_2/config.json", "architectures", ["T5ForConditionalGeneration"]),
    ("text_encoder_2/config.json", "model_type", "umt5"),
    ("text_encoder_2/config.json", "d_model", 2048),
])
def test_preflight_refuses_component_and_geometry_mismatches(path, key, value):
    configs = _configs()
    configs[path][key] = value
    with pytest.raises(ValueError, match="preflight refused"):
        _check(configs)


@pytest.mark.parametrize("value", [None, [], "{not parsed}", {1: "not JSON"}, {"value": {"set"}}, {"value": float("nan")}])
def test_supplied_configs_must_be_bounded_json_objects(value):
    configs = _configs()
    configs["vae/config.json"] = value
    with pytest.raises(ValueError, match="preflight refused"):
        _check(configs)


def test_preflight_refuses_remote_code_and_oversized_configs_without_echoing_content():
    for changes in ({"auto_map": {"AutoModel": "secret.value"}}, {"trust_remote_code": True},
                    {"unused": "secret.value" * preflight.MAX_CONFIG_BYTES}):
        configs = _configs()
        configs["text_encoder/config.json"].update(changes)
        with pytest.raises(ValueError, match="text_encoder/config.json") as caught:
            _check(configs)
        assert "secret.value" not in str(caught.value)
    configs = _configs()
    configs["../private.json"] = {}
    with pytest.raises(ValueError, match="config inventory"):
        _check(configs)



@pytest.mark.parametrize("source_size", [(512, 512), (600, 400), (400, 600), (640, 480), (480, 640), (640, 360), (360, 640)])
def test_complete_installed_pil_processor_path_preserves_frozen_geometry_and_rgb_tensor(source_size):
    # A fresh interpreter isolates the real installed CPU libraries from other
    # tests' framework stubs, CUDA state and global tensor defaults. The child
    # forbids model/application/pytest imports and poisons CUDA initialization.
    script = Path(__file__).with_name("fixtures") / "instruction_edit_processor_cpu.py"
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(script), *(str(value) for value in source_size)],
        capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    expected = edit.geometry(*source_size)
    assert receipt["source_size"] == list(source_size)
    assert receipt["tensor_shape"] == [1, 3, expected.height, expected.width]
    assert receipt["cuda_initialized"] is False and receipt["cuda_initialization_attempts"] == []



def test_config_preflight_is_import_cold_and_has_no_file_or_network_access():
    root = str(Path(__file__).resolve().parents[1])
    script = """
import importlib.abc
import sys
sys.path.insert(0, ROOT)
class RefuseModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'diffusers', 'huggingface_hub'} or fullname in {
            'backend.config', 'backend.storage', 'backend.main',
        }:
            raise AssertionError('unexpected warm import: ' + fullname)
sys.meta_path.insert(0, RefuseModels())
from backend.instruction_edit_preflight import validate_configs
from backend.instruction_edit import KONTEXT_V1
import builtins
def forbidden(*args, **kwargs):
    raise AssertionError('preflight must consume supplied configs only')
builtins.open = forbidden
result = validate_configs(KONTEXT_V1.model_id, KONTEXT_V1.revision, CONFIGS)
assert not result.execution_available
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", "ROOT = " + repr(root) + "\nCONFIGS = " + repr(_configs()) + "\n" + script],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
