"""The installed Hub CLI's login menu and expiry notes match our instructions.

Only its menu/persistence boundary runs, with every input and token save
stubbed. Nothing signs in, reads credentials or asks the network.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub import _login

from backend import hf_sign_in


@pytest.mark.parametrize("choice, method", [(0, "browser"), (1, "token")])
def test_instructions_match_the_installed_hub_login_menu(monkeypatch, choice, method):
    captured = []

    def choose(prompt, options):
        captured.append((prompt, options))
        return choice

    monkeypatch.setattr(_login.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(_login, "select_choice", choose)
    assert _login._prompt_login_method() == method
    assert captured == [
        ("How would you like to log in?", ["Log in with your browser", "Paste an access token"]),
    ]
    instructions = hf_sign_in.login_instructions()
    for option in captured[0][1]:
        assert option in instructions
    assert str(Path(hf_sign_in.__file__).resolve().parents[1]) in instructions
    assert "pasted input hidden" in instructions
    assert "double-click install.cmd" in instructions


def test_replacing_a_refused_sign_in_forces_the_installed_cli_menu(monkeypatch):
    from huggingface_hub.cli import auth

    logins = []
    monkeypatch.setattr(auth, "out", SimpleNamespace(mode=auth.OutputFormat.human))
    monkeypatch.setattr(auth, "login", lambda **kwargs: logins.append(kwargs))
    auth.auth_login(force=True)
    assert logins == [{"token": None, "add_to_git_credential": False, "skip_if_logged_in": False}]
    instructions = hf_sign_in.login_instructions(replace=True)
    assert "`uv run hf auth login --force`" in instructions
    # The installer sees the existing token and skips its sign-in offer.
    assert "double-click install.cmd" not in instructions


@pytest.mark.parametrize("refresh", [None, "not-a-refresh-token"])
def test_browser_login_retains_expiry_and_reports_renewal(monkeypatch, refresh):
    saved = []

    def save(*args, **kwargs):
        saved.append((args, kwargs))
        return "stub", "stub-user"

    monkeypatch.setattr(_login, "_validate_and_save_token", save)
    monkeypatch.setattr(_login.time, "time", lambda: 1000)
    response = {"access_token": "not-an-access-token", "expires_in": 2 * 86400}
    if refresh is not None:
        response["refresh_token"] = refresh
    _login._save_oauth_token(response)
    assert saved == [(("not-an-access-token",), {
        "add_to_git_credential": False,
        "refresh_token": refresh,
        "expires_at": 1000 + 2 * 86400,
    })]
    expected = (
        "This token will be refreshed automatically when it expires."
        if refresh else "This token expires in 2 days. Log in again to renew it."
    )
    assert _login._expiration_note(response) == expected
