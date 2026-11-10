"""What the callers that are not a browser share about the passcode
(`backend.server_probe`): the header, the gate's refusal read from an
answer, and the running commit asked through it. Against a loopback
stand-in (tests/passcode_stub.py) over real HTTP; the header is also read
back by the real gate. The commit's own tests are tests/test_build_info.py.
"""
from __future__ import annotations

import pytest

from backend import passcode, server_probe
from tests.passcode_stub import RETRY_AFTER, AskingServer

PASSCODE = "pässcode-狐-horse"            # outside ASCII, and outside Latin-1
STATUS = {"state": "unloaded", "commit": "abc1234"}


@pytest.fixture
def serve():
    servers: list[AskingServer] = []

    def start(asked: str | None, routes: dict | None = None) -> AskingServer:
        servers.append(AskingServer(asked, {"/api/status": STATUS} if routes is None else routes))
        return servers[-1]

    yield start
    for server in servers:
        server.close()


def test_the_names_are_the_gates_own():
    assert server_probe.PASSCODE_VARIABLE == passcode.VARIABLE
    assert server_probe.PASSCODE_REQUIRED_CODE == passcode.REQUIRED_CODE


def test_the_environments_passcode_is_read_as_the_server_reads_it():
    assert server_probe.environment_passcode({}) is None
    assert server_probe.environment_passcode({"AURADREAM_PASSCODE": "   "}) is None
    assert server_probe.environment_passcode({"AURADREAM_PASSCODE": " correct horse \n"}) == "correct horse"
    assert passcode.normalise(" correct horse \n") == "correct horse"


def test_no_passcode_is_no_header():
    assert server_probe.passcode_headers(None) == {}
    assert server_probe.passcode_headers("") == {}


@pytest.mark.parametrize("value", ["correct horse", PASSCODE, "tab\tinside", "ÿ-latin-1-ÿ"])
def test_the_header_is_one_the_real_gate_reads_as_that_passcode(value):
    [(name, sent)] = server_probe.passcode_headers(value).items()
    assert name == "Authorization"
    # What urllib puts on the wire for the str, as the layer finds it.
    scope = {"headers": [(b"authorization", sent.encode("latin-1"))]}
    assert sent.encode("latin-1") == b"Bearer " + value.encode("utf-8")
    gate = passcode.Gate(value, port=7860)
    assert gate.matches(passcode._bearer(scope))
    assert not passcode.Gate(value + "x", port=7860).matches(passcode._bearer(scope))


def test_the_header_opens_the_real_app_behind_its_gate(client, monkeypatch):
    from backend import main

    monkeypatch.setattr(main, "_passcode_gate", passcode.Gate(PASSCODE, port=main.settings.port))
    assert client.get("/api/status").status_code == 401
    [(name, sent)] = server_probe.passcode_headers(PASSCODE).items()
    answer = client.get("/api/status", headers={name: sent.encode("latin-1")})
    assert answer.status_code == 200 and answer.json()["passcode"] is True
    # The coded 401 is what `refusal` keys on.
    assert client.get("/api/status").json()["code"] == server_probe.PASSCODE_REQUIRED_CODE


@pytest.mark.parametrize("value", ["wxyz\nqrst-12", "wxyz\rqrst", "wxyz\x00qrst", "wxyz\x7fqrst"])
def test_a_control_character_is_refused_without_quoting_the_passcode(value):
    with pytest.raises(ValueError) as refused:
        server_probe.passcode_headers(value)
    said = str(refused.value)
    assert "AURADREAM_PASSCODE holds a control character" in said
    assert value not in said and "wxyz" not in said and "qrst" not in said


def test_a_server_that_asks_for_nothing_is_never_sent_the_passcode(serve):
    server = serve(None)
    assert server_probe.running_commit(server.url, PASSCODE) == "abc1234"
    assert server.requests == [("GET", "/api/status", False)]


def test_the_commit_is_read_through_the_passcode(serve):
    server = serve(PASSCODE)
    assert server_probe.running_commit(server.url, PASSCODE) == "abc1234"
    # Asked first without it; only the gate's own 401 is answered with it.
    assert server.requests == [("GET", "/api/status", False), ("GET", "/api/status", True)]


def test_a_gate_that_refuses_is_said_and_is_never_none(serve):
    server = serve(PASSCODE)
    with pytest.raises(server_probe.PasscodeRefused) as missing:
        server_probe.running_commit(server.url)
    assert (missing.value.status, missing.value.sent, missing.value.locked) == (401, False, False)
    assert "asks for a passcode, and AURADREAM_PASSCODE is not set here" in missing.value.reason
    assert server.requests == [("GET", "/api/status", False)]

    with pytest.raises(server_probe.PasscodeRefused) as wrong:
        server_probe.running_commit(server.url, "not-the-passcode")
    assert (wrong.value.status, wrong.value.sent) == (401, True)
    assert "did not accept the passcode in AURADREAM_PASSCODE" in wrong.value.reason

    server.locked = True
    with pytest.raises(server_probe.PasscodeRefused) as locked:
        server_probe.running_commit(server.url, PASSCODE)       # the right one, too
    assert (locked.value.status, locked.value.locked, locked.value.retry_after) == (429, True, RETRY_AFTER)
    assert f"wait {RETRY_AFTER} s" in locked.value.reason

    for refused in (missing.value, wrong.value, locked.value):
        for said in (str(refused), refused.reason, refused.fix, repr(refused)):
            assert PASSCODE not in said and "not-the-passcode" not in said
        # A caller that has queued work is never told to run it again.
        assert "again" not in refused.fix


def test_a_401_that_is_not_the_gates_is_another_program_and_gets_no_passcode(serve):
    server = serve("anything-else")
    del server.required["code"]
    assert server_probe.running_commit(server.url, PASSCODE) is None
    assert server.requests == [("GET", "/api/status", False)]


def test_a_redirect_never_carries_the_passcode_elsewhere(serve):
    elsewhere = serve(None)
    # Signed out it asks; given the passcode it points somewhere else.
    server = serve(PASSCODE, {"/api/status": (307, {"Location": elsewhere.url + "api/status"})})
    assert server_probe.running_commit(server.url, PASSCODE) is None
    assert server.requests == [("GET", "/api/status", False), ("GET", "/api/status", True)]
    assert elsewhere.requests == []


def test_the_title_probe_reads_the_sign_in_page_as_auradream(serve):
    server = serve(PASSCODE)
    assert server_probe.auradream_at(server.url) is True
    assert server.requests == [("GET", "/", False)]


def test_the_status_is_read_through_the_passcode_with_the_callers_timeout(serve):
    server = serve(PASSCODE)
    assert server_probe.status_json(server.url + "api/status", PASSCODE, timeout=5) == STATUS
