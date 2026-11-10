"""The passcode's parts, with no app: the setting, when it is asked, the
token, the comparison and the limiter (`backend/passcode.py`).

The HTTP side is tests/test_api_passcode.py; the walk of every route is
tests/test_api_passcode_routes.py.
"""
from __future__ import annotations

import dataclasses
import hashlib
import hmac
import os
import subprocess
import sys
from pathlib import Path

import pytest

from backend import passcode
from backend.config import settings
from backend.passcode import Gate, Limiter

ROOT = Path(__file__).resolve().parent.parent
PASSCODE = "nine-lamps-by-the-quay"


class Clock:
    def __init__(self, at: float = 1000.0) -> None:
        self.at = at

    def __call__(self) -> float:
        return self.at


@pytest.fixture(scope="module")
def key() -> bytes:
    return Gate(PASSCODE, port=7860)._key


def gate_at(wall: Clock, clock: Clock | None = None) -> Gate:
    return Gate(PASSCODE, port=7860, now=wall, clock=clock or Clock())


# ---------------------------------------------------------------- the setting


@pytest.mark.parametrize(("raw", "kept"), [
    (None, None), ("", None), ("   ", None), ("\t\n", None),
    ("  open sesame 9 \n", "open sesame 9"),
    ("pa$$ w#rd; \\ 'q'", "pa$$ w#rd; \\ 'q'"),
])
def test_the_setting_is_stripped_and_empty_is_unset(raw, kept):
    assert passcode.normalise(raw) == kept


def test_the_floor_is_eight_characters_and_its_words_never_hold_the_value():
    assert passcode.setting_problem(None) is None
    assert passcode.setting_problem("12345678") is None
    assert passcode.setting_problem("x" * passcode.MAX_LENGTH) is None
    short = passcode.setting_problem("zq7-zq7")
    assert short is not None and "AURADREAM_PASSCODE" in short and "8 characters" in short
    assert "zq7" not in short
    long = passcode.setting_problem("zq7" * 100)
    assert long is not None and "256" in long and "zq7" not in long


def test_the_setting_comes_from_the_environment_and_stays_out_of_the_reprs():
    # A fresh interpreter: `settings` reads the environment once, at import.
    # The value goes in through the environment, never as an argument.
    code = ("from backend.config import settings; from backend import passcode; "
            "print(len(settings.passcode), settings.passcode == passcode.normalise('  spaced out 77  ')); "
            "print(repr(settings))")
    env = {**os.environ, "AURADREAM_PASSCODE": "  spaced out 77  "}
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[0] == "13 True"
    assert "spaced out" not in done.stdout and "spaced out" not in done.stderr


def test_the_settings_repr_never_holds_the_passcode():
    assert settings.passcode is None        # the tests' environment sets none
    assert PASSCODE not in repr(dataclasses.replace(settings, passcode=PASSCODE))


def test_the_gate_keeps_a_digest_and_a_key_never_the_passcode():
    gate = Gate(PASSCODE, port=7860)
    kept = repr(vars(gate)) + repr(vars(gate.limiter))
    assert PASSCODE not in kept and PASSCODE.encode() not in vars(gate).values()
    assert gate._key == hashlib.scrypt(
        PASSCODE.encode(), salt=b"auradream.passcode.session.v1", n=2 ** 14, r=8, p=1, dklen=32,
    )
    with pytest.raises(ValueError):
        Gate("   ", port=7860)


# ------------------------------------------------------------ when it is asked


@pytest.mark.parametrize(("host", "allowed", "on"), [
    ("127.0.0.1", ("127.0.0.1", "localhost"), False),
    ("localhost", ("127.0.0.1", "localhost"), False),
    ("::1", ("127.0.0.1", "localhost", "::1"), False),
    ("127.0.0.1", (), False),
    ("0.0.0.0", ("127.0.0.1", "localhost"), True),
    ("::", ("127.0.0.1", "localhost"), True),
    ("192.168.1.20", ("192.168.1.20",), True),
    # A proxy in front of a loopback bind: on the network though its bind is not.
    ("127.0.0.1", ("127.0.0.1", "localhost", "my-box"), True),
    ("127.0.0.1", ("192.168.1.20",), True),
    ("127.0.0.1", ("aura.example",), True),
])
def test_it_is_asked_when_the_server_is_reachable_beyond_this_machine(host, allowed, on):
    assert passcode.asked(host, allowed) is on
    built = passcode.gate_for(PASSCODE, host, allowed, 7860)
    assert (built is not None) is on
    # Unset, it is never asked, wherever the server listens.
    assert passcode.gate_for(None, host, allowed, 7860) is None


def test_the_cookie_is_named_for_its_port():
    # Cookies ignore ports: two servers on one host would share a name.
    assert Gate(PASSCODE, port=7861).cookie_name == "auradream_session_7861"


# ------------------------------------------------------------------ the token


def test_a_minted_token_is_valid_until_it_expires():
    wall = Clock(1_800_000_000.5)
    gate = gate_at(wall)
    token = gate.mint()
    version, expires, mac = token.split(".")
    assert version == "v1" and int(expires) == 1_800_000_000 + 30 * 24 * 3600
    assert mac == hmac.new(gate._key, f"v1.{expires}".encode(), hashlib.sha256).hexdigest()
    assert gate.valid(token)
    wall.at = int(expires) - 1
    assert gate.valid(token)
    wall.at = int(expires)
    assert not gate.valid(token)
    wall.at = int(expires) + 10 ** 6
    assert not gate.valid(token)


def test_a_tampered_token_is_refused(key):
    now = 1_800_000_000
    token = passcode.mint_token(key, now + 60)
    assert passcode.token_expiry(key, token, now) == now + 60
    version, expires, mac = token.split(".")
    later = f"{version}.{int(expires) + 3600}.{mac}"
    flipped = f"{version}.{expires}.{'0' if mac[0] != '0' else '1'}{mac[1:]}"
    for bad in [later, flipped, f"{version}.{expires}.{mac.upper()}", f"{version}.{expires}.",
                f"{version}.{expires}", f"{version}.{expires}.{mac}.x", f"{expires}.{mac}", "", None,
                f"{version}.0{expires}.{mac}", f"{version}.+{expires}.{mac}", f"{version}. {expires}.{mac}",
                f"{version}.{expires}.{mac}é", "x" * 5000]:
        assert passcode.token_expiry(key, bad, now) is None, bad


def test_another_version_is_refused_even_with_a_mac_of_its_own(key):
    now = 1_800_000_000
    expires = now + 60
    mac = hmac.new(key, f"v2.{expires}".encode(), hashlib.sha256).hexdigest()
    assert passcode.token_expiry(key, f"v2.{expires}.{mac}", now) is None
    # And v1's MAC does not carry to another version's name.
    v1_mac = passcode.mint_token(key, expires).split(".")[2]
    assert passcode.token_expiry(key, f"v2.{expires}.{v1_mac}", now) is None


def test_another_passcodes_token_is_refused(key):
    # Changing the passcode changes the key: the only revocation there is.
    now = 1_800_000_000
    other = Gate(PASSCODE + "!", port=7860)._key
    assert other != key
    assert passcode.token_expiry(key, passcode.mint_token(other, now + 60), now) is None


def test_a_token_is_renewable_once_it_is_a_day_old():
    wall = Clock(1_800_000_000)
    gate = gate_at(wall)
    token = gate.mint()
    assert not gate.renewable(token)
    wall.at += 24 * 3600 - 5
    assert not gate.renewable(token)
    wall.at += 10
    assert gate.renewable(token)
    fresh = gate.mint()
    assert fresh != token and gate.valid(fresh) and not gate.renewable(fresh)
    wall.at += 31 * 24 * 3600
    assert not gate.renewable(token) and not gate.valid(token)
    assert not gate.renewable("v1.nonsense") and not gate.renewable(None)


def test_the_cookies_attributes():
    gate = Gate(PASSCODE, port=7860)
    plain = gate.cookie("v1.1.aa", secure=False)
    assert plain == "auradream_session_7860=v1.1.aa; HttpOnly; Max-Age=2592000; Path=/; SameSite=Lax"
    assert gate.cookie("v1.1.aa", secure=True) == plain + "; Secure"
    assert gate.cookie("", secure=False) == "auradream_session_7860=; HttpOnly; Max-Age=0; Path=/; SameSite=Lax"


# ------------------------------------------------------------- the comparison


def test_the_comparison_strips_as_the_setting_does():
    gate = Gate("  " + PASSCODE + "\n", port=7860)
    assert gate.matches(PASSCODE)
    assert gate.matches(" " + PASSCODE + " ")       # a phone's keyboard adds a space
    for wrong in ["", " ", PASSCODE[:-1], PASSCODE + "x", PASSCODE.upper(), "nine lamps by the quay"]:
        assert not gate.matches(wrong), wrong


def test_the_comparison_takes_any_characters_and_refuses_the_over_long():
    spoken = "pässwörd ☂ with spaces $ # ; \\"
    gate = Gate(spoken, port=7860)
    assert gate.matches(spoken) and not gate.matches("passwörd ☂ with spaces $ # ; \\")
    longest = "k" * passcode.MAX_LENGTH
    assert Gate(longest, port=7860).matches(longest)
    # Past 256 characters it is refused unread, even were it the setting.
    over = "k" * (passcode.MAX_LENGTH + 1)
    assert not Gate(over, port=7860).matches(over)


# ---------------------------------------------------------------- the limiter


def test_five_wrong_are_free_and_then_each_waits_longer_up_to_a_minute():
    clock = Clock()
    limiter = Limiter(clock)
    for _ in range(4):
        limiter.wrong()
        assert limiter.locked_for() == 0
    waits = []
    for _ in range(10):
        limiter.wrong()
        waits.append(limiter.locked_for())
        clock.at += limiter.locked_for()
        assert limiter.locked_for() == 0
    assert waits == [1, 2, 4, 8, 16, 32, 60, 60, 60, 60]
    assert limiter.wrong_in_a_row == 14


def test_a_lock_runs_down_on_the_clock_and_a_right_one_resets():
    clock = Clock()
    limiter = Limiter(clock)
    for _ in range(8):
        limiter.wrong()
    assert limiter.locked_for() == 8
    clock.at += 3.5
    assert limiter.locked_for() == 4.5
    clock.at += 100
    assert limiter.locked_for() == 0
    limiter.right()
    assert limiter.wrong_in_a_row == 0 and limiter.locked_for() == 0
    for _ in range(4):
        limiter.wrong()
    assert limiter.locked_for() == 0


def test_a_day_of_wrong_guesses_never_overflows_the_wait():
    clock = Clock()
    limiter = Limiter(clock)
    for _ in range(5000):
        limiter.wrong()
    assert limiter.locked_for() == 60


def test_an_attempt_during_a_lock_is_neither_compared_nor_counted():
    clock = Clock()
    gate = gate_at(Clock(1_800_000_000), clock)
    for _ in range(5):
        assert gate.attempt("wrong guess") == passcode.Attempt(False)
    assert gate.limiter.wrong_in_a_row == 5
    # Locked for a second: a wrong one is not counted, and the right one is
    # refused too, or the answer would say which guess was right.
    assert gate.attempt("another guess") == passcode.Attempt(False, 1)
    assert gate.attempt(PASSCODE) == passcode.Attempt(False, 1)
    assert gate.limiter.wrong_in_a_row == 5
    clock.at += 0.25
    assert gate.attempt(PASSCODE).retry_after == 1      # whole seconds, rounded up
    clock.at += 0.75
    assert gate.attempt(PASSCODE) == passcode.Attempt(True)
    assert gate.limiter.wrong_in_a_row == 0
    # The count starts again from none.
    for _ in range(4):
        assert gate.attempt("wrong guess") == passcode.Attempt(False)
    assert gate.attempt(PASSCODE) == passcode.Attempt(True)


def test_the_lock_grows_while_the_guessing_goes_on():
    clock = Clock()
    gate = gate_at(Clock(1_800_000_000), clock)
    for _ in range(5):
        gate.attempt("wrong guess")
    seen = []
    for _ in range(8):
        clock.at += gate.limiter.locked_for()
        gate.attempt("wrong guess")
        seen.append(gate.attempt(PASSCODE).retry_after)
    assert seen == [2, 4, 8, 16, 32, 60, 60, 60]


def test_matches_alone_never_counts():
    gate = Gate(PASSCODE, port=7860)
    for _ in range(20):
        assert not gate.matches("wrong guess")
    assert gate.limiter.wrong_in_a_row == 0


def test_the_open_list_is_exact_paths_for_reading_only():
    gate = Gate(PASSCODE, port=7860)
    for path in passcode.OPEN_PATHS:
        assert gate.is_open("GET", path) and gate.is_open("HEAD", path)
        assert not gate.is_open("POST", path) and not gate.is_open("DELETE", path)
        assert not gate.is_open("GET", path + "/") and not gate.is_open("GET", path + "x")
    for closed in ["/", "/static/", "/static/icons/", "/static/app.js", "/static/offline.html",
                   "/static/icons/../app.js", "/sw.js", "/api/status", "/api/fake/passcode"]:
        assert not gate.is_open("GET", closed), closed
    # The fake server's knobs stay open on the fake alone.
    fake = Gate(PASSCODE, port=7860, open_prefixes=("/api/fake/",))
    assert fake.is_open("POST", "/api/fake/passcode") and not fake.is_open("POST", "/api/status")
