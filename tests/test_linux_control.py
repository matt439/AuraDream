"""linux/control.sh, run for real by bash in the stub sandbox
(`tests/linux_script_helpers.py`): status, stop, start, restart, logs and
doctor against the stub systemd and a loopback AuraDream (Linux only)."""
from __future__ import annotations

import pytest

from tests.linux_script_helpers import (  # noqa: F401  (fixtures by name)
    on_linux,
    _calls,
    _changes,
    _env_file,
    _run,
    _service_active,
    _stub,
    _venv,
    auradream_listening,
    sandbox,
)


# --- control.sh ---------------------------------------------------------------------

@on_linux
def test_control_status_of_an_active_answering_service(sandbox, auradream_listening):
    (sandbox["root"] / ".git").mkdir()
    _venv(sandbox)
    _service_active(sandbox)
    auradream_listening["track"]["service"] = True
    auradream_listening["busy"]["status"].update({
        "model_id": "cagliostrolab/animagine-xl-4.0", "model_label": "Animagine XL 4.0",
        "quantization": "fp8", "offload": "model", "state": "ready",
        "vram_used_mb": 4289, "vram_total_mb": 7842, "device_name": "NVIDIA GeForce RTX 3070",
        "disk_free_mb": 819200, "commit": "abc1234",
        "keep_awake": {"kind": "systemd-inhibit", "holding": True, "wanted": True,
                       "reasons": ["in use (a request in the last 20 minutes)"], "detail": None},
    })
    sandbox["env"]["STUB_AHEAD_BEHIND"] = "0\t3"
    result = _run(sandbox, "control.sh", "status")
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "active (running) since Tue 2026-11-03 10:00:00 AEDT" in out
    assert "Enabled" in out and "Lingering is on" in out
    assert "Branch main at abc1234 Keep awake (2 days ago)" in out
    assert "3 commit(s) behind origin/main" in out
    assert "It answers." in out
    assert "Commit:       abc1234\n" in out              # the process's own, the checkout's too
    assert "Animagine XL 4.0 (cagliostrolab/animagine-xl-4.0)" in out and "fp8, model offload" in out
    assert "Jobs:         none" in out and "Card:         free" in out
    assert "systemd-inhibit, holding: in use (a request in the last 20 minutes)" in out
    assert "4289 of 7842 MiB in use (NVIDIA GeForce RTX 3070)" in out and "800.0 GiB" in out
    assert "AuraDream 1000 tester 4321 systemd-inhibit" in out and "UPower" not in out
    assert "The service is running and AuraDream answers." in out
    assert any(c.endswith("fetch --quiet") for c in _calls(sandbox))
    assert _changes(sandbox) == []


@on_linux
@pytest.mark.parametrize(("awake", "said"), [
    # Idle is not off: the first status on the live server called a released
    # hold "off" because only `wanted` was read.
    ({"kind": "systemd-inhibit", "holding": False, "wanted": False, "reasons": []},
     "systemd-inhibit, not holding: idle, so the PC may sleep"),
    ({"kind": "off", "holding": False, "wanted": False, "reasons": []},
     "off (AURADREAM_KEEP_AWAKE=false)"),
])
def test_control_status_tells_idle_from_switched_off(sandbox, auradream_listening, awake, said):
    (sandbox["root"] / ".git").mkdir()
    _venv(sandbox)
    _service_active(sandbox)
    auradream_listening["track"]["service"] = True
    auradream_listening["busy"]["status"]["keep_awake"] = {**awake, "detail": None, "idle_minutes": 20}
    result = _run(sandbox, "control.sh", "status")
    assert f"Keep awake:   {said}" in result.stdout, result.stdout + result.stderr


@on_linux
@pytest.mark.parametrize(("running", "said"), [
    # A pull without a restart: the checkout moved, the process did not.
    ("def5678", "Commit:       def5678, and the checkout is at abc1234: restart it to run the checkout's code"),
    ("abc1234e9", "Commit:       abc1234e9\n"),          # the same commit, abbreviated longer
    (None, None),                                         # a server that could not tell: no line
])
def test_control_status_names_the_running_commit(sandbox, auradream_listening, running, said):
    (sandbox["root"] / ".git").mkdir()
    _venv(sandbox)
    _service_active(sandbox)
    auradream_listening["track"]["service"] = True
    auradream_listening["busy"]["status"]["commit"] = running
    result = _run(sandbox, "control.sh", "status")
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    if said is None:
        assert "Commit:" not in out
    else:
        assert said in out, out


@on_linux
def test_control_status_of_a_stopped_service_still_reports(sandbox, auradream_listening):
    (sandbox["root"] / ".git").mkdir()
    _venv(sandbox)
    auradream_listening["track"]["service"] = True        # stopped: nothing answers
    sandbox["env"]["STUB_FETCH_EXIT"] = "128"             # and no network: noted, not fatal
    result = _run(sandbox, "control.sh", "status")
    out = result.stdout
    assert result.returncode == 1, out + result.stderr
    assert "inactive (dead)" in out
    assert "git fetch failed, so that is as of the last fetch: fatal: unable to access the remote" in out
    assert "It does not answer" in out
    assert "== Sleep holds" in out                         # everything it can is still printed
    assert "The service is not running, or AuraDream does not answer" in out
    assert _changes(sandbox) == []


@on_linux
def test_control_stop_refuses_a_busy_server_and_force_stops(sandbox, auradream_listening):
    _venv(sandbox)
    _service_active(sandbox)
    auradream_listening["busy"]["status"]["active_jobs"] = ["job-1"]
    result = _run(sandbox, "control.sh", "stop")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "The AuraDream service is busy: 1 job(s) queued or running." in result.stdout
    assert "--force" in result.stdout
    assert _changes(sandbox) == []
    assert (sandbox["state"] / "active").exists()

    forced = _run(sandbox, "control.sh", "stop", "--force")
    assert forced.returncode == 0, forced.stdout + forced.stderr
    assert "stopping it although it is busy: 1 job(s) queued or running" in forced.stdout
    assert _changes(sandbox) == ["systemctl --user stop auradream.service"]
    assert not (sandbox["state"] / "active").exists()


@on_linux
def test_control_stop_of_an_idle_server(sandbox, auradream_listening):
    _venv(sandbox)
    _service_active(sandbox)
    result = _run(sandbox, "control.sh", "stop")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "It is idle." in result.stdout and "is stopped" in result.stdout
    assert _changes(sandbox) == ["systemctl --user stop auradream.service"]


@on_linux
def test_control_restart_waits_for_the_page(sandbox, auradream_listening):
    _venv(sandbox)
    _service_active(sandbox)
    auradream_listening["track"]["service"] = True
    sandbox["env"]["STUB_START_DELAY"] = "2"               # the page answers 2 s after the restart
    result = _run(sandbox, "control.sh", "restart")
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "It is idle." in out and "Waiting for the page" in out
    assert f"AuraDream answers at http://127.0.0.1:{auradream_listening['port']}/" in out
    assert _changes(sandbox) == ["systemctl --user restart auradream.service"]


@on_linux
def test_control_restart_refuses_a_busy_server(sandbox, auradream_listening):
    _venv(sandbox)
    _service_active(sandbox)
    auradream_listening["busy"]["bundles"]["state"] = "running"
    result = _run(sandbox, "control.sh", "restart")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "a gallery bundle operation is running" in result.stdout
    assert _changes(sandbox) == []


@on_linux
def test_control_start_says_when_the_page_never_answers(sandbox):
    _venv(sandbox)
    sandbox["env"]["AURADREAM_INSTALL_TEST_WAIT"] = "2"    # nothing listens on port 1
    result = _run(sandbox, "control.sh", "start")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Started." in result.stdout and "did not answer within 2 s" in result.stdout
    assert _changes(sandbox) == ["systemctl --user start auradream.service"]


@on_linux
def test_control_start_stops_waiting_for_a_failed_unit(sandbox):
    _venv(sandbox)
    (sandbox["state"] / "failed").touch()
    result = _run(sandbox, "control.sh", "start")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "The AuraDream service failed while starting." in result.stdout


@on_linux
def test_control_start_stops_waiting_for_an_auto_restarting_unit(sandbox):
    """Restart=on-failure: a crash while starting is auto-restart, not failed."""
    _venv(sandbox)
    (sandbox["state"] / "auto-restart").touch()
    result = _run(sandbox, "control.sh", "start")       # the default 60 s wait
    assert result.returncode == 1, result.stdout + result.stderr
    assert "it is crashing, not slow" in result.stdout
    assert "did not answer within" not in result.stdout


@on_linux
def test_control_restart_stops_waiting_when_the_unit_restarts(sandbox):
    """NRestarts growing since the start is a crash too, whatever SubState says."""
    _venv(sandbox)
    (sandbox["state"] / "active").touch()
    sandbox["env"]["STUB_CRASH_AFTER"] = "1"
    result = _run(sandbox, "control.sh", "restart", "--force")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "it is crashing, not slow" in result.stdout
    assert "did not answer within" not in result.stdout


@on_linux
@pytest.mark.parametrize(("args", "expected"), [
    ((), "journalctl --user -u auradream.service -n 100 --no-pager"),
    (("-n", "20"), "journalctl --user -u auradream.service -n 20 --no-pager"),
    (("--lines", "5", "--follow"), "journalctl --user -u auradream.service -n 5 -f"),
    (("-f",), "journalctl --user -u auradream.service -n 100 -f"),
])
def test_control_logs_passes_lines_and_follow_through(sandbox, args, expected):
    result = _run(sandbox, "control.sh", "logs", *args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Uvicorn running" in result.stdout
    assert [c for c in _calls(sandbox) if c.startswith("journalctl")] == [expected]


@on_linux
def test_control_doctor_runs_with_the_service_settings(sandbox):
    env_file = _env_file(sandbox)
    env_file.parent.mkdir(parents=True)
    env_file.write_text("AURADREAM_HOST=0.0.0.0\nAURADREAM_PORT=7999\n")
    _stub(sandbox["stubs"], "uv", 'printf "doctor sees %s:%s in %s\\n" "$AURADREAM_HOST" "$AURADREAM_PORT" "$PWD"')
    result = _run(sandbox, "control.sh", "doctor")
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"doctor sees 0.0.0.0:7999 in {sandbox['root']}" in result.stdout
    assert [c for c in _calls(sandbox) if c.startswith("uv")] == ["uv run --no-sync python -m backend.doctor"]


@on_linux
@pytest.mark.parametrize("args", [("bogus",), ("logs", "--lines", "many"), ("stop", "--now")])
def test_control_refuses_an_unknown_command_or_option(sandbox, args):
    result = _run(sandbox, "control.sh", *args)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "Unknown" in result.stderr
    assert _changes(sandbox) == []


@on_linux
def test_control_help_and_root(sandbox):
    result = _run(sandbox, "control.sh")
    assert result.returncode == 0 and "bash linux/control.sh status" in result.stdout
    sandbox["env"]["STUB_UID"] = "0"
    result = _run(sandbox, "control.sh", "stop", "--force")
    assert result.returncode == 1 and "as root" in result.stdout
    assert _calls(sandbox) == []
