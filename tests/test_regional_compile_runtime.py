"""Real scoped Dynamo semantics in one isolated, explicitly CPU subprocess."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_real_strict_policy_propagates_backend_and_limit_errors_on_cpu(tmp_path):
    root = Path(__file__).resolve().parents[1]
    worker = root / "tests" / "fixtures" / "regional_compile_cpu.py"
    environment = {
        name: value for name, value in os.environ.items()
        if not name.startswith(("AURADREAM_", "TORCHINDUCTOR_", "TORCHDYNAMO_", "TORCH_COMPILE_"))
        and name not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "TORCH_TRACE", "TORCH_LOGS",
                         "TORCH_COMPILE_DEBUG", "TORCHDYNAMO_VERBOSE"}
    }
    paths = {
        "TEMP": "temporary", "TMP": "temporary", "TMPDIR": "temporary",
        "TORCHINDUCTOR_CACHE_DIR": "inductor-cache", "TRITON_CACHE_DIR": "triton-cache",
        "TORCH_COMPILE_DEBUG_DIR": "compile-debug", "TORCH_HOME": "torch-home",
        "XDG_CACHE_HOME": "cache", "HF_HOME": "huggingface",
    }
    for name, child in paths.items():
        target = tmp_path / child
        target.mkdir(exist_ok=True)
        environment[name] = str(target)
    environment.update({
        "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1",
        "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    })
    result = subprocess.run([sys.executable, "-B", str(worker)], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, encoding="utf-8", timeout=90)
    (tmp_path / "worker.stdout").write_text(result.stdout, encoding="utf-8")
    (tmp_path / "worker.stderr").write_text(result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt["ok"] and receipt["flags_restored"]
    assert receipt["device"] == "cpu" and receipt["threads"] == 1
    assert receipt["diagnostic_fullgraph"] is False
    assert receipt["counts"] == {"graphs": 1, "calls": 2, "failed_graphs": 1}
    assert receipt["limit_error"] == "FailOnRecompileLimitHit"
    assert receipt["backend_error"] == "BackendCompilerFailed"
    assert receipt["cuda_initialized"] is False and receipt["cuda_initialization_attempts"] == []
    assert receipt["cuda_availability"] == "forced_false_for_cpu_isolation"
    assert receipt["native_init_guard_intact"] is True
