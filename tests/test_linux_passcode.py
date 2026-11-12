"""The scripts' passcode, run for real by bash in the stub sandbox
(`tests/linux_script_helpers.py`): read from the service's environment file,
sent as a header, refused in its own words, never printed (Linux only)."""
from __future__ import annotations

import os
import shutil

import pytest

from tests.linux_script_helpers import (  # noqa: F401  (fixtures by name)
    SECRET,
    on_linux,
    _asking,
    _calls,
    _changes,
    _checkout_with_gallery,
    _env_file,
    _no_git_or_uv,
    _passcode_file,
    _run,
    _service_active,
    _snapshot,
    _stub,
    _venv,
    auradream_listening,
    sandbox,
)


# --- The passcode -------------------------------------------------------------------
#
# A server that asks (docs/design-notes.md, "A passcode beyond loopback"): the
# scripts read AURADREAM_PASSCODE from the service's environment file and send
# it as a header; a 401 and a 429 are each said in their own words, never as
# a server that is down or idle; and a passcode bash would read differently
# from systemd is refused before the file is sourced. The value is never
# printed.


@on_linux
def test_install_service_names_the_passcode_in_a_commented_line(sandbox):
    (sandbox["root"] / ".venv").mkdir()
    assert _run(sandbox, "install-service.sh").returncode == 0
    text = _env_file(sandbox).read_text()
    assert "\n#AURADREAM_PASSCODE=\n" in text
    assert "letters, digits and . _ - + @ % : , /" in text and "8 characters or more" in text
    assert "never reuse a password" in text
    # Commented: nothing is set, and the file as written passes its own check.
    assert not [line for line in text.splitlines() if line.startswith("AURADREAM_PASSCODE")]
    again = _run(sandbox, "install-service.sh", "--no-start")
    assert again.returncode == 0 and "[stop]" not in again.stdout, again.stdout + again.stderr


@on_linux
@pytest.mark.parametrize("passcode", [None, "", SECRET])
def test_install_service_warns_when_its_file_has_no_active_passcode(sandbox, passcode):
    (sandbox["root"] / ".venv").mkdir()
    if passcode is not None:
        _passcode_file(sandbox, passcode)
    result = _run(sandbox, "install-service.sh", "--no-start")
    assert result.returncode == 0, result.stdout + result.stderr
    warning = "[warn] No passcode is set: anyone who can reach this server can use it."
    assert (warning in result.stdout) is not bool(passcode)
    if not passcode:
        assert str(_env_file(sandbox)) in result.stdout
        assert "systemctl --user restart auradream" in result.stdout
        assert "Plain http leaves the passcode and pictures readable" in result.stdout
    assert SECRET not in result.stdout + result.stderr


@on_linux
def test_control_status_reads_through_the_passcode(sandbox, auradream_listening):
    (sandbox["root"] / ".git").mkdir()
    _asking(sandbox, auradream_listening)
    auradream_listening["busy"]["status"].update({"model_id": "m/checkpoint", "commit": "abc1234"})
    result = _run(sandbox, "control.sh", "status")
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "It answers." in out and "Checkpoint:   m/checkpoint" in out and "Jobs:         none" in out
    assert "could not be read" not in out
    assert auradream_listening["gate"]["sent"] == 1
    assert SECRET not in out + result.stderr


@on_linux
def test_stop_and_update_ask_whether_it_is_busy_through_the_passcode(sandbox, auradream_listening):
    _checkout_with_gallery(sandbox)
    _asking(sandbox, auradream_listening)
    auradream_listening["busy"]["status"]["active_jobs"] = ["job-1"]
    busy = _run(sandbox, "control.sh", "stop")
    assert busy.returncode == 1, busy.stdout + busy.stderr
    assert "The AuraDream service is busy: 1 job(s) queued or running." in busy.stdout
    updating = _run(sandbox, "update.sh", "--restart")
    assert updating.returncode == 1 and "The AuraDream service is busy: 1 job(s)" in updating.stdout
    assert _changes(sandbox) == [] and _no_git_or_uv(sandbox)

    auradream_listening["busy"]["status"]["active_jobs"] = []
    idle = _run(sandbox, "control.sh", "stop")
    assert idle.returncode == 0, idle.stdout + idle.stderr
    assert "It is idle." in idle.stdout
    assert _changes(sandbox) == ["systemctl --user stop auradream.service"]
    for result in (busy, updating, idle):
        assert SECRET not in result.stdout + result.stderr


@on_linux
@pytest.mark.parametrize(("file", "locked", "words"), [
    # A 401: no passcode in the file, or one the server does not accept.
    (None, False, "it asks for a passcode, and AURADREAM_PASSCODE is missing from the service's environment file"),
    ("another-passcode", False, "it did not accept the AURADREAM_PASSCODE these scripts read"),
    # A 429: the lock on wrong passcodes, the right one refused too.
    (SECRET, True, "it is refusing every passcode for now, after too many wrong ones just now: wait 7 s"),
])
def test_a_refused_passcode_is_said_in_its_own_words_and_never_taken_for_idle(
        sandbox, auradream_listening, file, locked, words):
    _checkout_with_gallery(sandbox)
    _asking(sandbox, auradream_listening, file=file)
    auradream_listening["gate"]["locked"] = locked
    before = _snapshot(sandbox["root"])

    stop = _run(sandbox, "control.sh", "stop")
    assert stop.returncode == 1, stop.stdout + stop.stderr
    assert "Whether the AuraDream service is busy could not be asked" in stop.stdout and words in stop.stdout
    assert "--force" in stop.stdout
    restart = _run(sandbox, "control.sh", "restart")
    assert restart.returncode == 1 and words in restart.stdout
    update = _run(sandbox, "update.sh", "--restart")
    assert update.returncode == 1, update.stdout + update.stderr
    assert "whether it is busy could not be asked" in update.stdout and words in update.stdout
    # Nothing stopped, nothing pulled, no database copy.
    assert _changes(sandbox) == [] and _no_git_or_uv(sandbox)
    assert (sandbox["state"] / "active").exists() and _snapshot(sandbox["root"]) == before

    # status: the server is up; only its status is closed.
    status = _run(sandbox, "control.sh", "status")
    assert status.returncode == 0, status.stdout + status.stderr
    assert "It answers." in status.stdout and f"Its status could not be read: {words}" in status.stdout
    assert "It does not answer" not in status.stdout
    assert "The service is running and AuraDream answers." in status.stdout

    for result in (stop, restart, update, status):
        said = result.stdout + result.stderr
        assert SECRET not in said and "another-passcode" not in said


@on_linux
def test_a_forced_restart_without_the_passcode_still_sees_the_page_come_up(sandbox, auradream_listening):
    _asking(sandbox, auradream_listening, file=None)
    auradream_listening["track"]["service"] = True
    sandbox["env"]["AURADREAM_INSTALL_TEST_WAIT"] = "20"
    result = _run(sandbox, "control.sh", "restart", "--force")
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "whether it is busy could not be asked" in out and "restarting it anyway" in out
    assert f"AuraDream answers at http://127.0.0.1:{auradream_listening['port']}/" in out
    assert "These scripts cannot ask it anything more: it asks for a passcode" in out
    assert "did not answer within" not in out
    assert _changes(sandbox) == ["systemctl --user restart auradream.service"]


@on_linux
@pytest.mark.parametrize("probe", ["venv-python", "curl"])
def test_the_title_probe_reads_the_sign_in_page_as_auradream(sandbox, auradream_listening, probe):
    # No passcode anywhere here: the probe needs none.
    _checkout_with_gallery(sandbox)
    if probe == "venv-python":
        _venv(sandbox)
    elif not shutil.which("curl", path="/usr/bin:/bin"):
        pytest.skip("no curl here")
    auradream_listening["gate"]["passcode"] = SECRET
    result = _run(sandbox, "update.sh")
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"AuraDream is running at http://127.0.0.1:{auradream_listening['port']}/" in result.stdout
    assert auradream_listening["gate"]["sent"] == 0 and _no_git_or_uv(sandbox)


# What bash, sourcing the file, would read differently from systemd. The
# first would set the passcode to "correct" and run a program named horse.
# Each is (the value, a part of it that must never be printed).
_UNSAFE = [
    ("correct horse", "correct"), ("pa$$word-12", "word-12"), ("$(touch {marker})-12345", "12345"),
    ("`touch {marker}`-12345", "12345"), ('say"quoted"12', "quoted"), ("it's-a-passcode", "a-passcode"),
    ("semi;colon-12; touch {marker}", "colon-12"), ("hash #tag-1234", "tag-1234"), ("back\\slash-12", "slash-12"),
    ("amp&ersand-12", "ersand"), ("pipe|touch {marker}", "pipe|"), ("star*glob-123", "glob-123"),
    ("tilde~home-12", "tilde"), ("bang!bang-123", "bang-123"), ("less<more>-12", "more"),
    ("(parens)-1234", "parens"), ("Windows-line-end\r", "Windows-line-end"),
    # Outside ASCII: "letters" are ASCII's, whatever the shell's locale.
    ("pässcode-12345", "sscode-12345"),
]


@on_linux
@pytest.mark.parametrize(("value", "part"), _UNSAFE)
@pytest.mark.parametrize(("script", "args"), [
    ("control.sh", ("status",)), ("control.sh", ("stop", "--force")), ("control.sh", ("doctor",)),
    ("update.sh", ("--dry-run",)), ("update.sh", ("--restart",)), ("setup.sh", ("--dry-run",)),
    ("install-service.sh", ()), ("install-service.sh", ("--dry-run",)),
])
def test_a_passcode_the_file_cannot_hold_is_refused_before_the_file_is_sourced(sandbox, script, args, value, part):
    marker = sandbox["tmp"] / "sourced"
    value = value.replace("{marker}", str(marker))
    _checkout_with_gallery(sandbox)
    (sandbox["root"] / ".venv").mkdir()
    _service_active(sandbox)
    # Programs of those names are on PATH: sourcing the file would run one.
    for name in ("horse", "touch"):
        _stub(sandbox["stubs"], name, f': >"{marker}"')
    env_file = _passcode_file(sandbox, value, more="AURADREAM_PORT=1\n")
    before = _snapshot(sandbox["root"], sandbox["home"])
    result = _run(sandbox, script, *args)
    said = result.stdout + result.stderr
    assert result.returncode == 1, said
    assert f"[stop] The AURADREAM_PASSCODE line in {env_file} has a character" in result.stdout
    assert "letters, digits and . _ - + @ % : , /" in result.stdout
    assert not marker.exists()
    assert not [c for c in _calls(sandbox) if c.startswith(("horse", "touch"))]
    # The value is never printed, whole or in part.
    assert value.strip() not in said and part not in said
    assert _snapshot(sandbox["root"], sandbox["home"]) == before
    assert _changes(sandbox) == [] and _no_git_or_uv(sandbox)


@on_linux
@pytest.mark.parametrize(("line", "sent"), [
    (f"AURADREAM_PASSCODE={SECRET}", 2), (f"  AURADREAM_PASSCODE={SECRET}  ", 2), ("AURADREAM_PASSCODE=", 0),
    # A comment is no setting, whatever it holds.
    ("#AURADREAM_PASSCODE=correct horse", 0), ("# AURADREAM_PASSCODE=pa$$word", 0),
    ("OTHER_AURADREAM_PASSCODE=1", 0),
])
def test_a_plain_passcode_or_none_is_read(sandbox, auradream_listening, line, sent):
    _venv(sandbox)
    _service_active(sandbox)
    env_file = _env_file(sandbox)
    env_file.parent.mkdir(parents=True)
    env_file.write_text(f"AURADREAM_HOST=0.0.0.0\n{line}\n")
    result = _run(sandbox, "control.sh", "stop")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[stop]" not in result.stdout and "It is idle." in result.stdout
    # Sent (with both of the busy check's reads) only where the file sets one.
    assert auradream_listening["gate"]["sent"] == sent


@on_linux
def test_the_passcode_travels_in_the_environment_never_as_an_argument(sandbox, auradream_listening):
    # The python the scripts ask with is replaced by one that records its
    # arguments and then runs the real one.
    real = _venv(sandbox)
    target = os.readlink(real)
    seen = sandbox["tmp"] / "argv"
    real.unlink()
    real.write_text(f'#!/bin/bash\nprintf "%s\\n" "$*" >>"{seen}"\nexec "{target}" "$@"\n')
    real.chmod(0o755)
    _service_active(sandbox)
    auradream_listening["gate"]["passcode"] = SECRET
    _passcode_file(sandbox, SECRET)
    result = _run(sandbox, "control.sh", "stop")
    assert result.returncode == 0, result.stdout + result.stderr
    assert auradream_listening["gate"]["sent"] == 2 and seen.exists()
    assert SECRET not in seen.read_text()


@on_linux
def test_a_redirect_never_carries_the_scripts_passcode_elsewhere(sandbox, auradream_listening):
    from tests.passcode_stub import AskingServer

    elsewhere = AskingServer(None, {"/api/status": {"state": "unloaded", "active_jobs": []},
                                    "/api/gallery/bundles/status": {"state": "idle"}})
    try:
        _asking(sandbox, auradream_listening)
        auradream_listening["gate"]["redirect"] = elsewhere.url.rstrip("/")
        stop = _run(sandbox, "control.sh", "stop")
        status = _run(sandbox, "control.sh", "status")
        # Not followed: the redirect is an answer these scripts cannot use.
        assert elsewhere.requests == []
        assert stop.returncode == 1, stop.stdout + stop.stderr
        assert "could not be asked" in stop.stdout and "307" in stop.stdout
        assert _changes(sandbox) == [] and "Checkpoint:" not in status.stdout
        assert SECRET not in stop.stdout + stop.stderr + status.stdout + status.stderr
    finally:
        elsewhere.close()


@on_linux
def test_a_shells_passcode_no_request_can_carry_is_never_an_answer(sandbox, auradream_listening):
    # No request is made with it, so nothing has answered: never "It
    # answers.", and never an idle server.
    _venv(sandbox)
    _service_active(sandbox)
    sandbox["env"]["AURADREAM_PASSCODE"] = "wxyz\x01qrst-horse"
    status = _run(sandbox, "control.sh", "status")
    assert "It answers." not in status.stdout, status.stdout + status.stderr
    assert "AURADREAM_PASSCODE holds a control character" in status.stdout
    assert status.returncode == 1
    stop = _run(sandbox, "control.sh", "stop")
    assert stop.returncode == 1 and "could not be asked" in stop.stdout
    assert "AURADREAM_PASSCODE holds a control character" in stop.stdout
    assert _changes(sandbox) == [] and auradream_listening["gate"]["sent"] == 0
    for result in (status, stop):
        said = result.stdout + result.stderr
        assert "wxyz" not in said and "qrst" not in said
