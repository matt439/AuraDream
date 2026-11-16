"""Installed FLUX processor/dispatcher ownership with tiny CPU-only tensors.

SDPA itself is replaced by a recording stand-in: no attention kernel, model
checkpoint, compiler, network or CUDA initialization runs in this probe.
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
from importlib.metadata import version
from pathlib import Path

root = Path(sys.argv[1])
receipt = {"success": False, "phase": "start"}
calls = {"native_cuda_init": 0, "lazy_cuda_init": 0, "cuda_compute": 0, "compile": 0, "network": 0}

def forbidden(kind):
    def fail(*args, **kwargs):
        calls[kind] += 1
        raise AssertionError("CPU attention contract reached forbidden " + kind)
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
    # Linux's unrelated optional bitsandbytes backend compiles at import.
    # Exclude it, as in the separate installed-FBC CPU probe.
    sys.modules["bitsandbytes"] = None

    from backend import config
    from backend.backends import attention_selection as selection
    from diffusers.models import attention_dispatch as dispatch
    from diffusers.models import modeling_utils
    from diffusers.models.transformers import transformer_flux as flux

    receipt["phase"] = "tiny exact FLUX layout"
    torch.manual_seed(17)
    model = flux.FluxTransformer2DModel(
        in_channels=4, out_channels=4, num_layers=1, num_single_layers=1,
        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=8,
        pooled_projection_dim=8, axes_dims_rope=(2, 2, 4),
    )
    assert type(model) is flux.FluxTransformer2DModel
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count < 50000
    processors = dict(model.attn_processors)
    assert set(processors) == {
        "transformer_blocks.0.attn.processor", "single_transformer_blocks.0.attn.processor",
    }
    assert all(type(processor) is flux.FluxAttnProcessor for processor in processors.values())
    before = [(processor._attention_backend, "_attention_backend" in vars(processor))
              for processor in processors.values()]
    registry_before = dispatch._AttentionBackendRegistry.get_active_backend()
    assert registry_before == (dispatch.AttentionBackendName.NATIVE, dispatch._native_attention)
    selection.require_native_attention(model)

    def flags():
        return {name: getattr(torch.backends.cuda, name + "_sdp_enabled")()
                for name in ("flash", "mem_efficient", "math", "cudnn")}

    saved_flags = flags()
    observed = []
    fail_sdpa = False
    def recording_sdpa(*, query, key, value, **kwargs):
        assert all(tensor.device.type == "cpu" for tensor in (query, key, value))
        observed.append({"flags": flags(), "dtype": str(query.dtype),
                         "shape": list(query.shape), "contiguous": all(t.is_contiguous() for t in (query, key, value))})
        if fail_sdpa:
            raise RuntimeError("synthetic SDPA failure")
        return query.clone()

    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    torch.nn.functional.scaled_dot_product_attention = recording_sdpa
    try:
        receipt["phase"] = "real processor to installed forced-cuDNN boundary"
        handle = selection.configure_attention(model, "cudnn-v1")
        assert handle is not None
        handle.require(model)
        assert dispatch._AttentionBackendRegistry.get_active_backend() == registry_before
        assert all(p._attention_backend is dispatch.AttentionBackendName._NATIVE_CUDNN for p in processors.values())
        hidden = torch.ones((1, 2, 16), dtype=torch.float32, device="cpu")
        encoder = torch.ones((1, 1, 16), dtype=torch.float32, device="cpu")
        with torch.no_grad():
            joint = model.transformer_blocks[0].attn(hidden, encoder_hidden_states=encoder)
            single = model.single_transformer_blocks[0].attn(hidden)
        assert [list(t.shape) for t in joint] == [[1, 2, 16], [1, 1, 16]]
        assert list(single.shape) == [1, 2, 16]
        assert len(observed) == 2
        assert all(row["flags"] == {"flash": False, "mem_efficient": False, "math": False, "cudnn": True}
                   and row["contiguous"] for row in observed)
        assert flags() == saved_flags
        fail_sdpa = True
        try:
            with torch.no_grad():
                model.single_transformer_blocks[0].attn(hidden)
        except RuntimeError as error:
            assert str(error) == "synthetic SDPA failure"
        else:
            raise AssertionError("forced SDPA failure was hidden or retried")
        assert len(observed) == 3 and flags() == saved_flags
        handle.require(model)
        handle.restore()
        assert handle.closed
        assert before == [(p._attention_backend, "_attention_backend" in vars(p)) for p in processors.values()]
        selection.require_native_attention(model)
        fail_sdpa = False
        with torch.no_grad():
            model.single_transformer_blocks[0].attn(hidden)
        assert len(observed) == 4 and observed[-1]["flags"] == saved_flags
        assert dispatch._AttentionBackendRegistry.get_active_backend() == registry_before
    finally:
        torch.nn.functional.scaled_dot_product_attention = original_sdpa
        if "handle" in locals() and handle is not None and not handle.closed:
            handle.restore()

    receipt.update({
        "success": True, "phase": "complete", "diffusers": version("diffusers"),
        "torch": version("torch"), "parameter_count": parameter_count,
        "processor_paths": sorted(processors), "sdpa_stand_in": observed,
        "cuda_initialized": torch.cuda.is_initialized(), "guards": calls,
        "native_init_guard_intact": torch._C._cuda_init is native_guard,
        "source_sha256": {name: hashlib.sha256(Path(inspect.getsourcefile(module)).read_bytes()).hexdigest()
                          for name, module in (("attention_dispatch", dispatch), ("transformer_flux", flux),
                                               ("modeling_utils", modeling_utils))},
    })
    assert not any(calls.values()) and not receipt["cuda_initialized"]
finally:
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
'''


def test_installed_flux_processors_own_forced_cudnn_and_restore_native_on_cpu(tmp_path):
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
    assert receipt["success"] and receipt["phase"] == "complete"
    assert receipt["diffusers"] == "0.42.1" and receipt["parameter_count"] < 50000
    assert len(receipt["processor_paths"]) == 2 and len(receipt["sdpa_stand_in"]) == 4
    assert not any(receipt["guards"].values())
    assert receipt["native_init_guard_intact"] and not receipt["cuda_initialized"]
    assert set(receipt["source_sha256"]) == {"attention_dispatch", "transformer_flux", "modeling_utils"}
    assert not any(path.is_file() for name in ("inductor", "triton", "debug") for path in paths[name].rglob("*"))
