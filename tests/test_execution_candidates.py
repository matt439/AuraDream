"""Cold attention/cache experiments never grant nondefault execution."""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend import execution


FACTS = execution.CompilerMetadata(
    platform="win32", python_version=(3, 13), torch="2.13.0+cu130",
    diffusers="0.40.0", torchao="0.18.0", triton=None, triton_windows=None,
)
TARGET = dict(model_id="black-forest-labs/FLUX.1-Krea-dev", quantization="fp8", offload="model", device="cuda")
CANDIDATES = [
    (execution.attention_candidate_reason, execution.ExecutionSpec(attention_backend="cudnn-v1")),
    (execution.cache_candidate_reason, execution.ExecutionSpec(step_cache="fbc-005-v1")),
]


@pytest.mark.parametrize("candidate, spec", CANDIDATES, ids=["cudnn", "fbc"])
@pytest.mark.parametrize("platform", ["win32", "linux"])
@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:12"])
@pytest.mark.parametrize("external", [None, "native"])
def test_supported_cold_candidate_needs_no_triton_and_stays_unavailable(candidate, spec, platform, device, external):
    facts = dataclasses.replace(FACTS, platform=platform)
    before = facts, spec.model_dump(), dict(TARGET)
    assert candidate(spec, facts, **{**TARGET, "device": device}, external_attention_backend=external) is None
    assert (facts, spec.model_dump(), TARGET) == before
    assert execution.unavailable_reason(spec) == execution.UNAVAILABLE
    with pytest.raises(ValueError, match="not available"):
        execution.require_default_execution(spec)


@pytest.mark.parametrize("candidate, spec", CANDIDATES, ids=["cudnn", "fbc"])
@pytest.mark.parametrize("changes", [
    {"model_id": "black-forest-labs/FLUX.1-dev"},
    {"model_id": "local/FLUX.1-Krea-dev"},
    {"quantization": "bf16"}, {"quantization": "int8"}, {"quantization": "nunchaku"},
    {"offload": "sequential"}, {"offload": "none"},
    {"device": "cpu"}, {"device": "cuda:"}, {"device": "cuda:-1"},
    {"device": "cuda:1:0"}, {"device": "cuda: 0"}, {"device": "cuda:٠"},
    {"device": "CUDA"}, {"num_images": 2}, {"num_images": 0},
    {"lora": "weight-zero.safetensors"}, {"lora": ""},
    {"refine": 1}, {"refine": 0}, {"mask": "a" * 64}, {"mask": ""},
    {"outpaint": True}, {"then_refine": True},
])
def test_unsupported_checkpoint_loader_batch_adapter_and_pass_are_refused(candidate, spec, changes):
    assert candidate(spec, FACTS, **{**TARGET, **changes}) is not None


@pytest.mark.parametrize("candidate, spec", CANDIDATES, ids=["cudnn", "fbc"])
@pytest.mark.parametrize("changes", [
    {"platform": "darwin"}, {"python_version": (3, 14)},
    {"torch": None}, {"torch": "2.13.0+cpu"}, {"torch": "2.13.0+cu128"}, {"torch": "2.14.0+cu130"},
    {"diffusers": None}, {"diffusers": "0.39.0"}, {"torchao": None}, {"torchao": "0.19.0"},
])
def test_candidate_requires_the_source_inspected_dependency_environment(candidate, spec, changes):
    assert candidate(spec, dataclasses.replace(FACTS, **changes), **TARGET) is not None


@pytest.mark.parametrize("candidate, spec", CANDIDATES, ids=["cudnn", "fbc"])
@pytest.mark.parametrize("external", ["", "NATIVE", " native ", "_native_cudnn", "flash", "secret\n" * 1000])
def test_external_override_is_conservatively_refused_without_echo(candidate, spec, external):
    reason = candidate(spec, FACTS, **TARGET, external_attention_backend=external)
    assert reason == "A nonnative external attention override needs a separate execution policy."
    assert len(reason) < 100


@pytest.mark.parametrize("candidate, field, recipe, conflicting", [
    (execution.attention_candidate_reason, "attention_backend", "cudnn-v1", "compilation"),
    (execution.attention_candidate_reason, "attention_backend", "cudnn-v1", "step_cache"),
    (execution.cache_candidate_reason, "step_cache", "fbc-005-v1", "compilation"),
    (execution.cache_candidate_reason, "step_cache", "fbc-005-v1", "attention_backend"),
])
def test_explicit_recipe_is_required_and_combinations_are_refused(candidate, field, recipe, conflicting):
    assert candidate(execution.ExecutionSpec(), FACTS, **TARGET) is not None
    value = {field: recipe, conflicting: {"compilation": "regional-v1", "step_cache": "fbc-005-v1",
                                        "attention_backend": "cudnn-v1"}[conflicting]}
    assert candidate(execution.ExecutionSpec(**value), FACTS, **TARGET) is not None


@pytest.mark.parametrize("candidate, spec", CANDIDATES, ids=["cudnn", "fbc"])
@pytest.mark.parametrize("change", [{"unknown": None}, {"compilation": "future"}])
def test_internal_spec_copies_cannot_hide_unknown_data(candidate, spec, change):
    with pytest.raises(ValidationError):
        candidate(spec.model_copy(update=change), FACTS, **TARGET)


@pytest.mark.parametrize("true_cfg", [None, 1.0])
def test_cache_accepts_only_absent_or_off_true_cfg(true_cfg):
    assert execution.cache_candidate_reason(CANDIDATES[1][1], FACTS, **TARGET, true_cfg=true_cfg) is None


@pytest.mark.parametrize("true_cfg", [1.000001, 2.0, 20.0, 0.5, float("nan"), float("inf")])
def test_cache_refuses_unconditioned_or_invalid_true_cfg(true_cfg):
    assert execution.cache_candidate_reason(CANDIDATES[1][1], FACTS, **TARGET, true_cfg=true_cfg) is not None


def test_cache_refuses_an_existing_cache_without_inspecting_a_model():
    assert execution.cache_candidate_reason(CANDIDATES[1][1], FACTS, **TARGET, cache_present=True) == (
        "The step-cache candidate cannot share an already enabled cache."
    )


def test_pure_candidates_import_cold_use_only_supplied_facts_and_leave_environment_unchanged(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
                       DIFFUSERS_ATTN_BACKEND="an-unrelated-hostile-external-value")
    child = r'''
import importlib.abc
import json
import os
import sys

blocked = {"torch", "torchao", "diffusers", "huggingface_hub", "backend.config"}
class ColdOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked or fullname.split(".", 1)[0] in blocked:
            raise AssertionError("candidate imported a runtime dependency: " + fullname)
        return None
sys.meta_path.insert(0, ColdOnly())
before = dict(os.environ)
from backend import execution
def forbidden(*args, **kwargs):
    raise AssertionError("candidate read metadata or environment instead of supplied facts")
execution.version = forbidden
execution.compiler_metadata = forbidden
os.getenv = forbidden
facts = execution.CompilerMetadata("win32", (3, 13), "2.13.0+cu130", "0.40.0", "0.18.0", None, None)
target = dict(model_id="black-forest-labs/FLUX.1-Krea-dev", quantization="fp8", offload="model", device="cuda")
for candidate, spec in (
    (execution.attention_candidate_reason, execution.ExecutionSpec(attention_backend="cudnn-v1")),
    (execution.cache_candidate_reason, execution.ExecutionSpec(step_cache="fbc-005-v1")),
):
    assert candidate(spec, facts, **target) is None
    assert candidate(spec, facts, **target, external_attention_backend="native") is None
    assert candidate(spec, facts, **target, external_attention_backend="") is not None
    try:
        execution.require_default_execution(spec)
    except ValueError as error:
        assert str(error) == execution.UNAVAILABLE
    else:
        raise AssertionError("candidate opened runtime admission")
assert before == dict(os.environ)
assert not any(name in blocked or name.split(".", 1)[0] in blocked for name in sys.modules)
print(json.dumps({"cold": True, "environment_unchanged": True, "runtime_refused": True}))
'''
    result = subprocess.run([sys.executable, "-B", "-c", child], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    (tmp_path / "candidate.stdout").write_text(result.stdout, encoding="utf-8")
    (tmp_path / "candidate.stderr").write_text(result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert json.loads(result.stdout) == {"cold": True, "environment_unchanged": True, "runtime_refused": True}
