"""Installed FLUX regional registration only; no compiler backend or device work."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import socket
import sys

from backend import config as _app_config  # noqa: F401 - before framework imports

calls = {"network": 0, "cuda_init": 0, "cuda_compute": 0, "compiler_backend": 0}


def forbidden(kind):
    def fail(*args, **kwargs):
        calls[kind] += 1
        raise AssertionError("CPU registration probe reached forbidden " + kind)
    return fail


socket.socket.connect = forbidden("network")
socket.socket.connect_ex = forbidden("network")
socket.create_connection = forbidden("network")

import torch  # noqa: E402

assert not torch.cuda.is_initialized()
assert version("diffusers") == "0.42.1", "recheck the private regional candidate on a dependency bump"
native_guard = forbidden("cuda_init")
torch._C._cuda_init = native_guard
torch.cuda.init = native_guard
torch.cuda._lazy_init = native_guard
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.Tensor.cuda = forbidden("cuda_compute")
for name in ("synchronize", "current_device", "get_device_capability", "get_device_properties"):
    setattr(torch.cuda, name, forbidden("cuda_compute"))
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
original_compile = torch.compile
torch.compile = forbidden("compiler_backend")
# An unrelated optional backend can compile at import on Linux.
sys.modules["bitsandbytes"] = None

from torch._dynamo import config as dynamo_config  # noqa: E402
from diffusers.models import modeling_utils  # noqa: E402
from diffusers.models.transformers import transformer_flux as flux  # noqa: E402
from backend.backends import regional_compile as regional  # noqa: E402


def tiny_flux():
    model = flux.FluxTransformer2DModel(
        in_channels=4, out_channels=4, num_layers=1, num_single_layers=1,
        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=8,
        pooled_projection_dim=8, axes_dims_rope=(2, 2, 4),
    )
    assert type(model) is flux.FluxTransformer2DModel
    assert sum(parameter.numel() for parameter in model.parameters()) < 50000
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    return model


def flags():
    return dynamo_config.suppress_errors, dynamo_config.fail_on_recompile_limit_hit


initial_flags = flags()
observations = []
expected = {"backend": "inductor", "fullgraph": True, "dynamic": False, "mode": "default"}
failure = RuntimeError("synthetic second-region registration failure")
failure_root = None


def record_compile(function=None, **kwargs):
    # Execute actual ModelMixin.compile_repeated_blocks and nn.Module.compile,
    # replacing only their compiler boundary with this recording stand-in.
    owner = getattr(function, "__self__", None)
    assert type(owner) in (flux.FluxTransformerBlock, flux.FluxSingleTransformerBlock)
    assert function.__func__ is torch.nn.Module._call_impl
    assert kwargs == expected
    assert flags() == (False, True)
    assert all(parameter.device.type == "cpu" for parameter in owner.parameters())
    observations.append({"class": type(owner).__name__, "arguments": kwargs, "flags": list(flags())})
    if failure_root is owner:
        raise failure
    return function


torch.compile = record_compile
normal_discard = []
second_discard = []
failure_discard = []
try:
    with dynamo_config.patch({"suppress_errors": True, "fail_on_recompile_limit_hit": False}):
        caller_flags = flags()
        model = tiny_flux()
        regions = [model.transformer_blocks[0], model.single_transformer_blocks[0]]
        facts = regional.inspect_regions(model)
        assert facts.block_counts == (("FluxTransformerBlock", 1), ("FluxSingleTransformerBlock", 1))
        assert all(region._compiled_call_impl is None for region in regions)
        registration = regional.register_regions(model, discard_on_failure=lambda: normal_discard.append(True))
        assert not normal_discard and flags() == caller_flags
        assert registration.require(model) == facts
        assert model._compiled_call_impl is None
        assert all(region._compiled_call_impl.__self__ is region for region in regions)
        assert len(observations) == 2

        try:
            regional.register_regions(model, discard_on_failure=lambda: second_discard.append(True))
        except ValueError as error:
            assert "already compiled" in str(error)
        else:
            raise AssertionError("a registered region was accepted as a fresh candidate")
        assert second_discard == [True] and len(observations) == 2
        registration.close()
        assert registration.closed and registration.facts is None
        # Close drops owned references; it must not unwrap modules or reset
        # compiler state behind the pipeline's owner.
        assert all(region._compiled_call_impl.__self__ is region for region in regions)
        try:
            registration.require(model)
        except RuntimeError:
            pass
        else:
            raise AssertionError("a closed registration retained authority")

        failed = tiny_flux()
        failure_root = failed.single_transformer_blocks[0]
        try:
            regional.register_regions(failed, discard_on_failure=lambda: failure_discard.append(True))
        except RuntimeError as error:
            assert error is failure
        else:
            raise AssertionError("a partially registered candidate was published")
        assert failure_discard == [True] and flags() == caller_flags
        assert failed._compiled_call_impl is None
        assert failed.transformer_blocks[0]._compiled_call_impl is not None
        assert failed.single_transformer_blocks[0]._compiled_call_impl is None
finally:
    torch.compile = original_compile

assert flags() == initial_flags
assert not any(calls.values()) and not torch.cuda.is_initialized()
assert torch._C._cuda_init is native_guard
receipt = {
    "ok": True, "mode": "registration-with-recording-compiler-stand-in",
    "torch_version": torch.__version__, "diffusers_version": version("diffusers"),
    "observations": observations, "counts": {"normal": len(normal_discard), "precompiled": len(second_discard),
                                             "failed": len(failure_discard)},
    "flags_restored": True, "root_compiled": False, "cuda_initialized": False,
    "guards": calls, "native_init_guard_intact": True,
    "source_sha256": {
        name: hashlib.sha256(Path(inspect.getsourcefile(module)).read_bytes()).hexdigest()
        for name, module in (("transformer_flux", flux), ("modeling_utils", modeling_utils), ("torch_module", torch.nn.Module))
    },
}
print(json.dumps(receipt))
