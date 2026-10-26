"""Installed FBC arithmetic on tiny CPU blocks in one isolated, guarded worker.

No Flux constructor/checkpoint, CUDA initialization, compiler backend, network
or application activation is exercised. Defining model/block bindings and
registry bootstrap are synthetic; installed hooks and their math are real.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


_WORKER = r'''
import json
import socket
import sys
import traceback
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

root = Path(sys.argv[1])
receipt = {"success": False, "phase": "start", "cases": {}, "guards": {}}
calls = {"native_cuda_init": 0, "lazy_cuda_init": 0, "cuda_compute": 0, "compile": 0, "network": 0}

def forbidden(kind):
    def fail(*args, **kwargs):
        calls[kind] += 1
        raise AssertionError("CPU FBC contract reached forbidden " + kind)
    return fail

socket.socket.connect = forbidden("network")
socket.socket.connect_ex = forbidden("network")
socket.create_connection = forbidden("network")

try:
    receipt["phase"] = "torch import and native guards"
    import torch

    assert not torch.cuda.is_initialized(), "torch import initialized CUDA"
    native_guard = forbidden("native_cuda_init")
    torch._C._cuda_init = native_guard
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    torch.cuda._lazy_init = forbidden("lazy_cuda_init")
    torch.cuda.init = forbidden("lazy_cuda_init")
    torch.Tensor.cuda = forbidden("cuda_compute")
    for name in ("synchronize", "current_device", "get_device_capability", "get_device_properties"):
        setattr(torch.cuda, name, forbidden("cuda_compute"))
    torch.compile = forbidden("compile")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # config must prepare the environment before any Diffusers import.
    from backend import config
    receipt["phase"] = "installed FBC/registry imports"
    from diffusers.hooks._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS
    from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
    from diffusers.hooks.first_block_cache import (
        _FBC_BLOCK_HOOK, _FBC_LEADER_BLOCK_HOOK, FBCBlockHook, FBCHeadBlockHook, FirstBlockCacheConfig,
    )
    from diffusers.hooks.hooks import HookRegistry, StateManager
    from diffusers.models.cache_utils import CacheMixin
    from backend.backends import Cancelled
    from backend.backends import fbc_cache

    class JointBlock(torch.nn.Module):
        def __init__(self, hidden_delta, encoder_delta):
            super().__init__()
            self.hidden_delta, self.encoder_delta, self.calls = hidden_delta, encoder_delta, 0

        def forward(self, hidden_states, encoder_hidden_states):
            self.calls += 1
            return encoder_hidden_states + self.encoder_delta, hidden_states + self.hidden_delta

    class SingleBlock(JointBlock):
        pass

    class TinyDenoiser(CacheMixin, torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList([JointBlock(1.0, 2.0), JointBlock(10.0, 20.0)])
            self.single_transformer_blocks = torch.nn.ModuleList([SingleBlock(100.0, 200.0)])
            self.config = SimpleNamespace(num_layers=2, num_single_layers=1)

        def forward(self, hidden_states, encoder_hidden_states):
            for block in [*self.transformer_blocks, *self.single_transformer_blocks]:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                )
            return encoder_hidden_states, hidden_states

    # Avoid auto-registering/importing every real model family. The public
    # register/get methods and actual metadata object remain in use.
    TransformerBlockRegistry._registry = {}
    TransformerBlockRegistry._is_registered = True
    for cls in (JointBlock, SingleBlock):
        TransformerBlockRegistry.register(cls, TransformerBlockMetadata(
            return_hidden_states_index=1, return_encoder_hidden_states_index=0,
        ))
    fbc_cache._api = lambda: fbc_cache._Api(
        TinyDenoiser, (JointBlock, SingleBlock), torch.nn.ModuleList, HookRegistry, StateManager,
        FBCHeadBlockHook, FBCBlockHook, FirstBlockCacheConfig, TransformerBlockRegistry.get,
        _ALL_TRANSFORMER_BLOCK_IDENTIFIERS, (_FBC_LEADER_BLOCK_HOOK, _FBC_BLOCK_HOOK),
    )

    hidden = torch.ones((1, 2, 2), dtype=torch.float32, device="cpu")
    encoder = torch.full((1, 2, 2), 3.0, dtype=torch.float32, device="cpu")

    def blocks(model):
        return [*model.transformer_blocks, *model.single_transformer_blocks]

    def counts(model):
        return [block.calls for block in blocks(model)]

    def forward(model, context="cond"):
        with torch.no_grad(), model.cache_context(context):
            output = model(hidden_states=hidden, encoder_hidden_states=encoder)
        assert all(value.device.type == "cpu" for value in output)
        return output

    def equal(actual, expected):
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))

    def manager(model):
        hook = model.transformer_blocks[0]._diffusers_hook.get_hook(_FBC_LEADER_BLOCK_HOOK)
        assert type(hook) is FBCHeadBlockHook and hook.threshold == 0.05
        return hook.state_manager

    def restored(model, owner):
        owner.assert_off()
        assert not owner.invalidated and not owner.active and not model.is_cache_enabled
        for block in blocks(model):
            assert "forward" not in vars(block)
            assert block.forward.__func__ in (JointBlock.forward, SingleBlock.forward)
            assert not hasattr(block, "_diffusers_hook")

    def owned():
        model = TinyDenoiser()
        discarded = []
        owner = fbc_cache.FirstBlockCacheOwner(model, discard_shared=lambda: discarded.append(True))
        return model, owner, discarded

    receipt["phase"] = "actual reuse and threshold recomputation"
    model, owner, discarded = owned()
    baseline = forward(model)
    for block in blocks(model):
        block.calls = 0
    with owner.cached_call():
        state = manager(model)
        equal(forward(model), baseline)
        first = counts(model)
        equal(forward(model), baseline)
        reused = counts(model)
        assert first == [1, 1, 1] and reused == [2, 1, 1]
        model.transformer_blocks[0].hidden_delta = 2.0
        changed = forward(model)
        assert counts(model) == [3, 2, 2]
        equal(changed, (baseline[0], baseline[1] + 1.0))
    assert state._state_cache == {} and state._current_context is None
    restored(model, owner)
    equal(forward(model), changed)
    assert discarded == []
    owner.close()
    receipt["cases"]["reuse"] = {"first": first, "reused": reused, "changed": [3, 2, 2]}

    receipt["phase"] = "actual cond/uncond separation"
    model, owner, discarded = owned()
    with owner.cached_call():
        state = manager(model)
        outputs = [forward(model, context) for context in ("cond", "uncond", "cond", "uncond")]
        for output in outputs:
            equal(output, outputs[0])
        assert counts(model) == [4, 2, 2]
        assert set(state._state_cache) == {"cond", "uncond"}
        assert state._state_cache["cond"] is not state._state_cache["uncond"]
    assert state._state_cache == {} and state._current_context is None
    restored(model, owner)
    owner.close()
    assert discarded == []
    receipt["cases"]["contexts"] = {"calls": [4, 2, 2], "separate": True, "reset": True}

    receipt["phase"] = "actual cached/off/cached freshness"
    model, owner, discarded = owned()
    with owner.cached_call():
        old_state = manager(model)
        expected = forward(model)
        equal(forward(model), expected)
    restored(model, owner)
    equal(forward(model), expected)
    assert counts(model) == [3, 2, 2]
    with owner.cached_call():
        new_state = manager(model)
        assert new_state is not old_state and new_state._state_cache == {}
        equal(forward(model), expected)
        assert counts(model) == [4, 3, 3]
        equal(forward(model), expected)
        assert counts(model) == [5, 3, 3]
    assert old_state._state_cache == new_state._state_cache == {}
    restored(model, owner)
    owner.close()
    assert discarded == []
    receipt["cases"]["fresh_calls"] = {"after_off": [3, 2, 2], "new_first": [4, 3, 3], "new_reuse": [5, 3, 3]}

    receipt["phase"] = "actual cancellation and forward restoration"
    model, owner, discarded = owned()
    cancelled = Cancelled()
    try:
        with owner.cached_call():
            state = manager(model)
            with torch.no_grad(), model.cache_context("cond"):
                expected = model(hidden_states=hidden, encoder_hidden_states=encoder)
                raise cancelled
    except Cancelled as caught:
        assert caught is cancelled
    else:
        raise AssertionError("cancellation did not propagate")
    assert state._current_context is None and state._state_cache == {}
    restored(model, owner)
    equal(forward(model), expected)
    assert counts(model) == [2, 2, 2]
    with owner.cached_call():
        assert manager(model) is not state
        equal(forward(model), expected)
        assert counts(model) == [3, 3, 3]
    restored(model, owner)
    owner.close()
    assert discarded == []
    receipt["cases"]["cancel"] = {"propagated": True, "reset": True, "off_calls": [2, 2, 2], "new_first": [3, 3, 3]}

    assert torch._C._cuda_init is native_guard
    assert not torch.cuda.is_initialized()
    assert all(count == 0 for count in calls.values()), calls
    artifacts = {name: [str(path.relative_to(root)) for path in (root / name).rglob("*") if path.is_file()]
                 for name in ("inductor", "triton", "debug")}
    assert not any(artifacts.values()), artifacts
    receipt.update(success=True, phase="complete", guards=calls, cuda_initialized=False, native_guard_retained=True,
                   torch=version("torch"), diffusers=version("diffusers"), threads=torch.get_num_threads(),
                   interop_threads=torch.get_num_interop_threads(), tensor_shape=list(hidden.shape),
                   synthetic_registry=True, compiler_artifacts=artifacts)
except BaseException as exc:
    receipt.update(error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}, guards=calls)
    if "torch" in globals():
        receipt["cuda_initialized"] = torch.cuda.is_initialized()
    raise
finally:
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
'''


@pytest.fixture(scope="module")
def runtime_receipt(tmp_path_factory):
    root = tmp_path_factory.mktemp("fbc-runtime")
    paths = {name: root / name for name in ("tmp", "hf", "torch", "inductor", "triton", "debug")}
    for path in paths.values():
        path.mkdir()
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(("TORCHDYNAMO_", "TORCH_COMPILE_", "TORCHINDUCTOR_", "TRITON_", "LOCAL_DIFFUSION_")) or name in {
            "TORCH_TRACE", "TORCH_LOGS", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
        }:
            env.pop(name)
    env.update({
        "TMP": str(paths["tmp"]), "TEMP": str(paths["tmp"]), "TMPDIR": str(paths["tmp"]),
        "HF_HOME": str(paths["hf"]), "HF_HUB_CACHE": str(paths["hf"] / "hub"),
        "TORCH_HOME": str(paths["torch"]), "TORCHINDUCTOR_CACHE_DIR": str(paths["inductor"]),
        "TRITON_CACHE_DIR": str(paths["triton"]), "TORCH_COMPILE_DEBUG_DIR": str(paths["debug"]),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1",
        "LOCAL_DIFFUSION_DEVICE": "cpu", "LOCAL_DIFFUSION_OFFLOAD": "none",
    })
    done = subprocess.run([sys.executable, "-c", _WORKER, str(root)], env=env,
                          cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=90)
    (root / "stdout.txt").write_text(done.stdout, encoding="utf-8")
    (root / "stderr.txt").write_text(done.stderr, encoding="utf-8")
    path = root / "receipt.json"
    assert done.returncode == 0, f"worker exit {done.returncode}; receipt: {path}\n{done.stdout}\n{done.stderr}"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["success"] and receipt["phase"] == "complete"
    return receipt


def test_actual_fbc_computes_then_reuses_and_recomputes_changed_residual(runtime_receipt):
    assert runtime_receipt["cases"]["reuse"] == {"first": [1, 1, 1], "reused": [2, 1, 1], "changed": [3, 2, 2]}


def test_actual_fbc_separates_contexts_and_starts_every_call_fresh(runtime_receipt):
    assert runtime_receipt["cases"]["contexts"] == {"calls": [4, 2, 2], "separate": True, "reset": True}
    assert runtime_receipt["cases"]["fresh_calls"]["new_first"] == [4, 3, 3]


def test_actual_fbc_cancellation_restores_normal_forward_and_next_first_call(runtime_receipt):
    assert runtime_receipt["cases"]["cancel"] == {
        "propagated": True, "reset": True, "off_calls": [2, 2, 2], "new_first": [3, 3, 3],
    }


def test_actual_fbc_worker_keeps_native_guards_and_cpu_resource_limits(runtime_receipt):
    assert runtime_receipt["cuda_initialized"] is False and runtime_receipt["native_guard_retained"] is True
    assert set(runtime_receipt["guards"]) == {"native_cuda_init", "lazy_cuda_init", "cuda_compute", "compile", "network"}
    assert not any(runtime_receipt["guards"].values())
    assert runtime_receipt["threads"] == runtime_receipt["interop_threads"] == 1
    assert runtime_receipt["tensor_shape"] == [1, 2, 2]
    assert not any(runtime_receipt["compiler_artifacts"].values())
