"""Run WSL setup cold with inert driver, uv, torch and doctor stand-ins.

The setup script is copied beside its environment file into a temporary
checkout. No installed uv, driver, torch, model, gallery or service is used.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="WSL bash script, cold Linux fixture")


def _executable(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\nset -e\n" + body + "\n", encoding="ascii")
    path.chmod(0o755)


@pytest.fixture
def setup(tmp_path):
    project = tmp_path / "project"
    shutil.copytree(ROOT / "wsl", project / "wsl")
    backend = project / "backend"
    backend.mkdir()
    (backend / "__init__.py").touch()
    (backend / "doctor.py").write_text("import os\nprint('Cold doctor ran')\nraise SystemExit(int(os.environ.get('STUB_DOCTOR_EXIT', '0')))\n")
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    _executable(stubs / "nvidia-smi", 'echo "CUDA Version: ${STUB_DRIVER_CUDA:-13.2}"')
    _executable(stubs / "uv", 'echo "uv $*" >>"$STUB_LOG"\n'
                'if [ "$1" = run ]; then shift 2; exec "$STUB_PYTHON" "$@"; fi')
    home = tmp_path / "home"
    home.mkdir()
    python_stubs = tmp_path / "python-stubs"
    python_stubs.mkdir()
    (python_stubs / "torch.py").write_text(
        "import os\n__version__ = 'stub'\n"
        "class cuda:\n"
        "    @staticmethod\n"
        "    def is_available(): return os.environ.get('STUB_CUDA', '1') == '1'\n"
        "    @staticmethod\n"
        "    def get_device_capability(): return (int(os.environ.get('STUB_CAP', '8')), 6)\n"
        "    @staticmethod\n"
        "    def get_device_name(index): return 'Cold card'\n"
    )
    for package in ("kernels", "bitsandbytes"):
        metadata = python_stubs / f"{package}-0.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(f"Name: {package}\nVersion: 0.0\n")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("AURADREAM_", "HF_", "UV_", "PYTHON", "TORCH"))}
    env.update(HOME=str(home), PATH=f"{stubs}:/usr/bin:/bin", STUB_PYTHON=sys.executable,
               STUB_LOG=str(tmp_path / "calls.log"), PYTHONPATH=str(python_stubs))
    return project, env


def _run(setup):
    project, env = setup
    return subprocess.run(["/bin/bash", str(project / "wsl" / "setup.sh")], env=env,
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30)


def test_no_cuda_stops_before_doctor_and_ready(setup):
    _, env = setup
    env["STUB_CUDA"] = "0"
    result = _run(setup)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "cuda MISSING" in result.stdout
    assert "Update the Windows NVIDIA driver" in result.stderr
    assert "Ready." not in result.stdout and "Cold doctor ran" not in result.stdout
    assert "backend.doctor" not in Path(env["STUB_LOG"]).read_text()


@pytest.mark.parametrize("driver", ["12.9", "13.2"])
def test_driver_floor_is_checked_before_uv_and_success_runs_the_doctor(setup, driver):
    _, env = setup
    env["STUB_DRIVER_CUDA"] = driver
    result = _run(setup)
    if driver == "12.9":
        assert result.returncode == 1
        assert "driver supports CUDA 12.9" in result.stderr
        assert not Path(env["STUB_LOG"]).exists()
        assert "Ready." not in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Windows NVIDIA driver: CUDA 13.2" in result.stdout
        assert "Cold doctor ran" in result.stdout and "Ready." in result.stdout
        assert "export AURADREAM_NUNCHAKU_FORMAT=int4" in result.stdout
        assert Path(env["STUB_LOG"]).read_text().splitlines() == [
            "uv python install 3.13", "uv sync", "uv run python -", "uv run python -m backend.doctor",
        ]


def test_a_failed_doctor_stops_before_ready(setup):
    _, env = setup
    env["STUB_DOCTOR_EXIT"] = "1"
    result = _run(setup)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Cold doctor ran" in result.stdout and "Ready." not in result.stdout


def test_all_setups_pin_the_same_uv_installer():
    version = "0.12.7"
    assert f"$UvVersion = '{version}'" in (ROOT / "install.ps1").read_text()
    for script in ("linux/setup.sh", "wsl/setup.sh"):
        text = (ROOT / script).read_text()
        assert f"UV_VERSION={version}" in text
        assert 'https://astral.sh/uv/$UV_VERSION/install.sh' in text
