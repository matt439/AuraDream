"""`python -m backend.doctor`: read-only checks in plain words, with fakes for
torch, nvidia-smi, RAM, disks and the port (no GPU, no Hub, no network)."""
from __future__ import annotations

import socket
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import doctor
from backend.doctor import FAIL, PASS, WARN, Check

Usage = namedtuple("Usage", "total used free")


def _torch(available=True, capability=(12, 0), arch=("sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"),
           cuda="13.0", name="NVIDIA GeForce RTX 5080"):
    return SimpleNamespace(
        __version__="2.13.0+cu130",
        version=SimpleNamespace(cuda=cuda),
        cuda=SimpleNamespace(
            is_available=lambda: available,
            get_device_name=lambda index=0: name,
            get_device_capability=lambda index=0: capability,
            get_arch_list=lambda: list(arch),
        ),
    )


@pytest.mark.parametrize(("capability", "arch", "ok"), [
    ((12, 0), ["sm_75", "sm_120"], True),
    ((8, 9), ["sm_75", "sm_80", "sm_86"], True),     # later minor of a built major
    ((7, 5), ["sm_75", "sm_80"], True),
    ((7, 0), ["sm_75", "sm_80"], False),             # V100: sm_75 does not run on 7.0
    ((6, 1), ["sm_75", "sm_120"], False),            # GTX 10 series
    ((12, 1), ["sm_100"], False),                    # another major
    ((6, 1), ["sm_75", "compute_60"], True),         # PTX JIT-compiles forward
    ((12, 0), ["sm_90a", "sm_120"], True),
])
def test_arch_supported(capability, arch, ok):
    assert doctor.arch_supported(capability, arch) is ok


def test_a_supported_card_passes_and_names_the_range():
    [check] = doctor.check_gpu(_torch())
    assert check.status == PASS
    assert "RTX 5080" in check.detail and "12.0" in check.detail and "sm_75 to sm_120" in check.detail


def test_a_gtx_10_card_fails_with_the_generation_it_needs():
    [check] = doctor.check_gpu(_torch(capability=(6, 1), name="NVIDIA GeForce GTX 1080"))
    assert check.status == FAIL
    assert "6.1" in check.detail and "RTX 20" in check.fix


def test_no_cuda_names_the_driver_not_the_toolkit():
    [check] = doctor.check_gpu(_torch(available=False))
    assert check.status == FAIL
    assert "driver" in check.fix and "CUDA 13" in check.fix and "Do not install the CUDA Toolkit" in check.fix


def test_a_cpu_only_torch_says_uv_sync():
    [check] = doctor.check_gpu(_torch(cuda=None))
    assert check.status == FAIL and "uv sync" in check.fix


def _smi(stdout="", returncode=0):
    def run(args, **kwargs):
        assert kwargs.get("timeout")
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr="")
    return run


@pytest.mark.parametrize(("header", "version"), [
    ("| NVIDIA-SMI 580.10   Driver Version: 580.10   CUDA Version: 13.0     |", (13, 0)),
    ("| NVIDIA-SMI 616.92   KMD Version: 616.92   CUDA UMD Version: 13.4     |", (13, 4)),
    ("| NVIDIA-SMI 551.86   Driver Version: 551.86   CUDA Version: 12.4     |", (12, 4)),
    ("no version here", None),
])
def test_parse_driver_cuda(header, version):
    assert doctor.parse_driver_cuda(header) == version


def test_a_current_driver_passes():
    [check] = doctor.check_driver(lambda name: "nvidia-smi", _smi("CUDA Version: 13.0"))
    assert check.status == PASS and "13.0" in check.detail


def test_an_old_driver_fails_with_the_nvidia_link():
    [check] = doctor.check_driver(lambda name: "nvidia-smi", _smi("CUDA Version: 12.4"))
    assert check.status == FAIL
    assert "12.4" in check.detail and doctor.DRIVER_URL in check.fix


def test_an_unreadable_version_only_warns():
    [check] = doctor.check_driver(lambda name: "nvidia-smi", _smi("a format nobody has seen"))
    assert check.status == WARN


def test_a_missing_nvidia_smi_warns_and_a_broken_one_fails():
    assert doctor.check_driver(lambda name: None, _smi())[0].status == WARN
    assert doctor.check_driver(lambda name: "nvidia-smi", _smi(returncode=9))[0].status == FAIL

    def raising(args, **kwargs):
        raise OSError("cannot run")
    assert doctor.check_driver(lambda name: "nvidia-smi", raising)[0].status == FAIL


def test_ram_against_the_measured_figures():
    gib = doctor.GIB
    [big] = doctor.check_ram(lambda: (64 * gib, 40 * gib))
    assert big.status == PASS and "27.9 GiB" in big.detail and "64.0 GiB total" in big.detail
    [small] = doctor.check_ram(lambda: (32 * gib, 20 * gib))
    assert small.status == WARN and "SDXL" in small.fix and "not been measured" in small.detail
    assert doctor.check_ram(lambda: None)[0].status == WARN


def test_parse_meminfo():
    text = "MemTotal:       65536000 kB\nMemFree:  1000 kB\nMemAvailable:   32768000 kB\n"
    assert doctor.parse_meminfo(text) == (65536000 * 1024, 32768000 * 1024)
    assert doctor.parse_meminfo("nothing") is None


def test_memory_status_reads_this_machine():
    status = doctor.memory_status()
    if status is not None:
        total, available = status
        assert total > 0 and 0 <= available <= total


def test_disk_groups_by_drive_and_warns_low_download_space(tmp_path):
    hub = tmp_path / "hf" / "hub"
    converted = tmp_path / "hf" / "auradream" / "converted"   # need not exist yet
    outputs = tmp_path / "outputs"
    seen = []

    def usage(path):
        seen.append(Path(path))
        return Usage(100e9, 70e9, 30e9)

    checks = doctor.check_disk([("model downloads (Hugging Face cache)", hub), ("converted copies", converted),
                                ("images (outputs)", outputs)], usage)
    assert [c.status for c in checks] == [WARN]   # one drive, under the plan
    assert str(hub) in checks[0].detail and str(converted) in checks[0].detail and "HF_HOME" in checks[0].fix
    assert all(path.exists() for path in seen)    # asked of the nearest existing parent


def test_disk_fails_when_nearly_full_and_passes_with_room(tmp_path):
    full = doctor.check_disk([("images (outputs)", tmp_path)], lambda path: Usage(100e9, 99e9, 1e9))
    assert full[0].status == FAIL
    roomy = doctor.check_disk([("model downloads (Hugging Face cache)", tmp_path)],
                              lambda path: Usage(1e12, 1e11, 9e11))
    assert roomy[0].status == PASS
    images_only = doctor.check_disk([("images (outputs)", tmp_path)], lambda path: Usage(100e9, 70e9, 30e9))
    assert images_only[0].status == PASS          # the 50 GB plan is the downloads'


def test_disk_places_follow_hf_home(monkeypatch, tmp_path):
    from huggingface_hub import constants
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(constants, "HF_HOME", str(tmp_path))
    places = dict(doctor.disk_places())
    assert places["model downloads (Hugging Face cache)"] == tmp_path / "hub"
    assert places["converted copies"] == tmp_path / "auradream" / "converted"


def test_port_free_ours_or_another_program():
    assert doctor.check_port("127.0.0.1", 7860, busy=lambda h, p: False, ours=lambda u: False)[0].status == PASS
    [ours] = doctor.check_port("127.0.0.1", 7860, busy=lambda h, p: True, ours=lambda u: True)
    assert ours.status == PASS and "already running at http://127.0.0.1:7860/" in ours.detail
    [other] = doctor.check_port("127.0.0.1", 7860, busy=lambda h, p: True, ours=lambda u: False)
    assert other.status == FAIL and "AURADREAM_PORT" in other.fix and "7861" in other.fix


def test_port_check_on_a_real_held_port():
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        port = held.getsockname()[1]
        assert doctor.check_port("127.0.0.1", port, ours=lambda u: False)[0].status == FAIL


def test_hf_login_is_the_token_file_only(tmp_path):
    token = tmp_path / "token"
    [missing] = doctor.check_hf_login(token, environ={})
    assert missing.status == WARN and "hf auth login" in missing.fix
    token.write_text("hf_secretvalue")
    [present] = doctor.check_hf_login(token, environ={})
    assert present.status == PASS and "hf_secretvalue" not in present.detail
    [env] = doctor.check_hf_login(tmp_path / "absent", environ={"HF_TOKEN": "hf_other"})
    assert env.status == PASS and "hf_other" not in env.detail


def test_report_exit_status_is_one_only_for_a_fail():
    lines, status = doctor.report([Check(PASS, "A", "fine."), Check(WARN, "B", "hmm.", "do this")])
    assert status == 0 and lines == ["PASS  A: fine.", "WARN  B: hmm.", "      What to do: do this"]
    assert doctor.report([Check(FAIL, "C", "broken.", "fix")])[1] == 1


def test_a_check_that_raises_becomes_a_warning():
    def check_boom():
        raise RuntimeError("bad")
    [check] = doctor.run_checks([check_boom])
    assert check.status == WARN and check.title == "boom" and "bad" in check.detail


def test_main_prints_every_check_and_returns_the_status(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "CHECKS", ())
    monkeypatch.setattr(doctor, "run_checks", lambda checks=(): [Check(FAIL, "GPU", "none.", "update")])
    assert doctor.main([]) == 1
    out = capsys.readouterr().out
    assert "FAIL  GPU: none." in out and "What to do: update" in out


def test_importing_the_doctor_never_imports_diffusers():
    code = ("import sys, backend.doctor as d; d.check_port(busy=lambda h, p: False); "
            "print(any(m.split('.')[0] in ('diffusers', 'transformers') for m in sys.modules))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                            cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
