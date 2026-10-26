"""Owned-process CPU policy check; no regional kernels or model qualification.

The parent configures every temporary/cache/debug directory before imports.
fullgraph=False is diagnostic: fullgraph=True has an independent hard failure
at the recompile limit and would not isolate fail_on_recompile_limit_hit.
"""
from __future__ import annotations

import json
import sys

from backend import config as _app_config  # noqa: F401 - prepare environment before torch

import torch

assert not torch.cuda.is_initialized()
initialization_attempts: list[str] = []
availability_checks = 0


def no_cuda_initialization(*_args, **_kwargs):
    initialization_attempts.append("cuda initialization")
    raise AssertionError("this CPU policy check must not initialize CUDA")


torch.cuda.init = no_cuda_initialization
torch.cuda._lazy_init = no_cuda_initialization
# cuda.random captures _lazy_init by value; guarding the native entry also
# covers that original alias. CUDA-wheel Dynamo preserves CUDA RNG even for
# CPU inputs when availability is true, so this child explicitly models an
# unavailable accelerator rather than querying or initializing the real card.
torch._C._cuda_init = no_cuda_initialization


def cpu_only_availability():
    global availability_checks
    availability_checks += 1
    return False


torch.cuda.is_available = cpu_only_availability
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from torch._dynamo import config as dynamo_config  # noqa: E402
from torch._dynamo.exc import BackendCompilerFailed, FailOnRecompileLimitHit  # noqa: E402

from backend.backends.regional_compile import strict_compiler_policy  # noqa: E402


class Arithmetic(torch.nn.Module):
    def forward(self, value):
        return value * 2 + 1


class BackendFailure(torch.nn.Module):
    def forward(self, value):
        return value + 3


def flags():
    return dynamo_config.suppress_errors, dynamo_config.fail_on_recompile_limit_hit, dynamo_config.recompile_limit


original = flags()
counts = {"graphs": 0, "calls": 0, "failed_graphs": 0}


def cpu_backend(graph, example_inputs):
    assert all(not isinstance(value, torch.Tensor) or value.device.type == "cpu" for value in example_inputs)
    counts["graphs"] += 1

    def run(*args):
        assert all(not isinstance(value, torch.Tensor) or value.device.type == "cpu" for value in args)
        counts["calls"] += 1
        return graph.forward(*args)

    return run


failure = RuntimeError("intentional CPU graph backend failure")


def broken_backend(_graph, example_inputs):
    assert all(not isinstance(value, torch.Tensor) or value.device.type == "cpu" for value in example_inputs)
    counts["failed_graphs"] += 1
    raise failure


value = torch.ones(2, device="cpu", dtype=torch.float32)
other_shape = torch.ones(3, device="cpu", dtype=torch.float32)

with dynamo_config.patch({"suppress_errors": True, "fail_on_recompile_limit_hit": False, "recompile_limit": 1}):
    caller = flags()
    assert caller == (True, False, 1)
    with strict_compiler_policy():
        assert flags() == (False, True, 1)
        with dynamo_config.patch({"suppress_errors": True, "fail_on_recompile_limit_hit": False}):
            with strict_compiler_policy():
                assert flags() == (False, True, 1)
            assert flags() == caller
        assert flags() == (False, True, 1)
        compiled = torch.compile(Arithmetic(), backend=cpu_backend, dynamic=False, fullgraph=False,
                                 isolate_recompiles=True)
        with torch.no_grad():
            first = compiled(value)
            repeated = compiled(value)
        assert first.device.type == repeated.device.type == "cpu"
        assert torch.equal(first, value * 2 + 1) and torch.equal(repeated, first)
        assert counts == {"graphs": 1, "calls": 2, "failed_graphs": 0}
        try:
            with torch.no_grad():
                compiled(other_shape)
        except FailOnRecompileLimitHit as error:
            assert "fail_on_recompile_limit_hit" in str(error)
            limit_exception = type(error).__name__
        else:
            raise AssertionError("a static shape limit must fail instead of returning an eager result")
        assert counts == {"graphs": 1, "calls": 2, "failed_graphs": 0}
    assert flags() == caller
    try:
        with strict_compiler_policy():
            assert flags() == (False, True, 1)
            broken = torch.compile(BackendFailure(), backend=broken_backend, dynamic=False, fullgraph=False,
                                   isolate_recompiles=True)
            with torch.no_grad():
                broken(value)
    except BackendCompilerFailed as error:
        assert error.inner_exception is failure
        backend_exception = type(error).__name__
    else:
        raise AssertionError("a backend failure must propagate instead of being suppressed")
    assert flags() == caller

assert flags() == original
assert counts == {"graphs": 1, "calls": 2, "failed_graphs": 1}
assert initialization_attempts == [] and not torch.cuda.is_initialized()
assert torch._C._cuda_init is no_cuda_initialization
assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
assert not any(name.split(".", 1)[0] in {"diffusers", "transformers", "huggingface_hub"} for name in sys.modules)
print(json.dumps({
    "ok": True, "torch": torch.__version__, "device": "cpu", "threads": 1,
    "diagnostic_fullgraph": False, "counts": counts, "limit_error": limit_exception,
    "backend_error": backend_exception, "flags_restored": flags() == original,
    "cuda_initialized": torch.cuda.is_initialized(), "cuda_initialization_attempts": initialization_attempts,
    "cuda_availability": "forced_false_for_cpu_isolation", "availability_checks": availability_checks,
    "native_init_guard_intact": torch._C._cuda_init is no_cuda_initialization,
}))
