"""Exact installed FLUX/FBC integration on tiny CPU modules and stand-in SDPA.

The owner, registry bootstrap, metadata, cache hooks and block forwards are
real. No checkpoint, CUDA, network or compiler work is permitted.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


_WORKER = r'''
import hashlib
import inspect
import json
import socket
import sys
import traceback
from importlib.metadata import version
from pathlib import Path

root = Path(sys.argv[1])
receipt = {"success": False, "phase": "start"}
calls = {"native_cuda_init": 0, "lazy_cuda_init": 0, "cuda_compute": 0, "compile": 0, "network": 0}

def forbidden(kind):
    def fail(*args, **kwargs):
        calls[kind] += 1
        raise AssertionError("CPU FLUX/FBC contract reached forbidden " + kind)
    return fail

socket.socket.connect = forbidden("network")
socket.socket.connect_ex = forbidden("network")
socket.create_connection = forbidden("network")
try:
    import torch
    assert not torch.cuda.is_initialized()
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
    # Exclude only the unrelated optional backend which compiles on import.
    sys.modules["bitsandbytes"] = None

    from backend import config
    from backend.backends import Cancelled, fbc_cache
    from diffusers.hooks import _helpers, first_block_cache, hooks
    from diffusers.models import cache_utils
    from diffusers.models.transformers import transformer_flux as flux

    receipt["phase"] = "production API and installed registry bootstrap"
    torch.manual_seed(23)
    model = flux.FluxTransformer2DModel(
        in_channels=4, out_channels=4, num_layers=1, num_single_layers=1,
        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=8,
        pooled_projection_dim=8, axes_dims_rope=(2, 2, 4),
    )
    assert type(model) is flux.FluxTransformer2DModel
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count < 50000
    # Do not replace _api, _inspect, block types or registry metadata.
    layout = fbc_cache.inspect_fbc_layout(model)
    assert layout.block_paths == ("transformer_blocks.0", "single_transformer_blocks.0")
    assert layout.recipe == "fbc-005-v1" and layout.threshold == .05
    assert _helpers.TransformerBlockRegistry._is_registered
    for block_type in (flux.FluxTransformerBlock, flux.FluxSingleTransformerBlock):
        metadata = _helpers.TransformerBlockRegistry.get(block_type)
        assert metadata._cls is block_type
        assert metadata.return_hidden_states_index == 1 and metadata.return_encoder_hidden_states_index == 0
    fbc_cache.require_native_attention(model)
    blocks = [model.transformer_blocks[0], model.single_transformer_blocks[0]]
    original_forwards = [block.forward.__func__ for block in blocks]
    assert all("forward" not in vars(block) for block in blocks)
    # A real pre-existing root registry's child cache must survive 0.42's
    # enable/disable invalidation, including an exceptional cache context.
    registry = hooks.HookRegistry.check_if_exists_or_initialize(model)
    prior_child_cache = []
    registry._child_registries_cache = prior_child_cache
    discarded = []
    owner = fbc_cache.FirstBlockCacheOwner(model, discard_shared=lambda: discarded.append(True))

    sdpa = []
    def recording_sdpa(*, query, key, value, **kwargs):
        assert all(tensor.device.type == "cpu" for tensor in (query, key, value))
        sdpa.append(list(query.shape))
        return query.clone()

    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    torch.nn.functional.scaled_dot_product_attention = recording_sdpa
    hidden = torch.randn((1, 2, 16), device="cpu")
    encoder = torch.randn((1, 1, 16), device="cpu")
    temb = torch.randn((1, 16), device="cpu")
    def forward(context="cond"):
        with torch.no_grad(), model.cache_context(context):
            h, e = hidden.clone(), encoder.clone()
            for block in blocks:
                e, h = block(hidden_states=h, encoder_hidden_states=e, temb=temb)
            return e, h

    def equal(actual, expected):
        assert all(torch.allclose(a, b, rtol=0, atol=1e-6) for a, b in zip(actual, expected))

    def manager():
        hook = blocks[0]._diffusers_hook.get_hook(first_block_cache._FBC_LEADER_BLOCK_HOOK)
        assert type(hook) is first_block_cache.FBCHeadBlockHook and hook.threshold == .05
        return hook.state_manager

    def restored():
        owner.assert_off()
        assert not model.is_cache_enabled and not owner.active and not owner.invalidated
        assert model._diffusers_hook is registry and registry._child_registries_cache is prior_child_cache
        for block, original in zip(blocks, original_forwards):
            assert "forward" not in vars(block) and block.forward.__func__ is original
            assert not hasattr(block, "_diffusers_hook")
        fbc_cache.require_native_attention(model)

    try:
        receipt["phase"] = "exact blocks, real cache hooks and separate contexts"
        baseline = forward()
        sdpa.clear()
        with owner.cached_call():
            state = manager()
            equal(forward(), baseline)
            first = len(sdpa)
            equal(forward(), baseline)
            repeated = len(sdpa)
            equal(forward("uncond"), baseline)
            separate = len(sdpa)
            equal(forward("uncond"), baseline)
            separate_repeated = len(sdpa)
            assert [first, repeated, separate, separate_repeated] == [2, 3, 5, 6]
            assert set(state._state_cache) == {"cond", "uncond"}
        assert state._state_cache == {} and state._context is None
        restored()
        sdpa.clear()
        equal(forward(), baseline)
        assert len(sdpa) == 2
        with owner.cached_call():
            fresh = manager()
            assert fresh is not state and fresh._state_cache == {}
            equal(forward(), baseline)
            assert len(sdpa) == 4
        restored()

        receipt["phase"] = "exceptional context cleanup and return to ordinary forward"
        marker = Cancelled()
        try:
            with owner.cached_call():
                cancelled_state = manager()
                with torch.no_grad(), model.cache_context("cond"):
                    # Deliberately fail while the installed context remains set.
                    blocks[0](hidden_states=hidden, encoder_hidden_states=encoder, temb=temb)
                    raise marker
        except Cancelled as error:
            assert error is marker
        else:
            raise AssertionError("cancellation was swallowed")
        assert cancelled_state._context is None and cancelled_state._state_cache == {}
        restored()
        equal(forward(), baseline)
        assert discarded == []
        owner.close()
        assert owner.closed
    finally:
        torch.nn.functional.scaled_dot_product_attention = original_sdpa
        if not owner.closed:
            owner.close()

    assert torch._C._cuda_init is native_guard and not torch.cuda.is_initialized()
    assert not any(calls.values()), calls
    receipt.update({
        "success": True, "phase": "complete", "diffusers": version("diffusers"),
        "torch": version("torch"), "parameter_count": parameter_count,
        "block_paths": list(layout.block_paths), "registry_bootstrap": "installed-unmodified",
        "sdpa_stand_in_counts": [first, repeated, separate, separate_repeated],
        "fresh_call": True, "cancel_reset": True, "ordinary_forward_restored": True,
        "child_cache_identity_restored": True, "owner_closed": owner.closed,
        "guards": calls, "cuda_initialized": torch.cuda.is_initialized(),
        "native_init_guard_intact": torch._C._cuda_init is native_guard,
        "source_sha256": {name: hashlib.sha256(Path(inspect.getsourcefile(module)).read_bytes()).hexdigest()
                          for name, module in (("transformer_flux", flux), ("fbc_cache", fbc_cache),
                                               ("first_block_cache", first_block_cache), ("hooks", hooks),
                                               ("cache_utils", cache_utils), ("metadata", _helpers))},
    })
except BaseException as error:
    receipt.update(error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}, guards=calls)
    raise
finally:
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
'''


def test_installed_exact_flux_fbc_owner_integrates_and_restores_on_cpu(tmp_path):
    paths = {name: tmp_path / name for name in ("tmp", "hf", "torch", "inductor", "triton", "debug")}
    for path in paths.values():
        path.mkdir()
    env = {
        name: value for name, value in os.environ.items()
        if not name.startswith(("AURADREAM_", "TORCHINDUCTOR_", "TRITON_", "TORCHDYNAMO_", "TORCH_COMPILE_"))
        and name not in {"DIFFUSERS_ATTN_BACKEND", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "TORCH_TRACE", "TORCH_LOGS"}
    }
    env.update({
        "TMP": str(paths["tmp"]), "TEMP": str(paths["tmp"]), "TMPDIR": str(paths["tmp"]),
        "HF_HOME": str(paths["hf"]), "HF_HUB_CACHE": str(paths["hf"] / "hub"),
        "TORCH_HOME": str(paths["torch"]), "TORCHINDUCTOR_CACHE_DIR": str(paths["inductor"]),
        "TRITON_CACHE_DIR": str(paths["triton"]), "TORCH_COMPILE_DEBUG_DIR": str(paths["debug"]),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "AURADREAM_DEVICE": "cpu", "AURADREAM_OFFLOAD": "none",
    })
    result = subprocess.run([sys.executable, "-B", "-c", _WORKER, str(tmp_path)], env=env,
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=90)
    (tmp_path / "worker.stdout").write_text(result.stdout, encoding="utf-8")
    (tmp_path / "worker.stderr").write_text(result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["success"] and receipt["phase"] == "complete" and receipt["diffusers"] == "0.42.1"
    assert receipt["parameter_count"] < 50000 and receipt["registry_bootstrap"] == "installed-unmodified"
    assert receipt["block_paths"] == ["transformer_blocks.0", "single_transformer_blocks.0"]
    assert receipt["sdpa_stand_in_counts"] == [2, 3, 5, 6]
    assert all(receipt[name] for name in ("fresh_call", "cancel_reset", "ordinary_forward_restored",
                                         "child_cache_identity_restored", "owner_closed"))
    assert not any(receipt["guards"].values()) and receipt["native_init_guard_intact"] and not receipt["cuda_initialized"]
    assert set(receipt["source_sha256"]) == {"transformer_flux", "fbc_cache", "first_block_cache", "hooks", "cache_utils", "metadata"}
    assert not any(path.is_file() for name in ("inductor", "triton", "debug") for path in paths[name].rglob("*"))
