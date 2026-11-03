"""GEN#41: a fresh visit with no Hugging Face sign-in is offered an open checkpoint.

The sign-in is the token file's existence or HF_TOKEN, never read or sent;
the suggestion is a status field the page preselects from, and nothing the
server runs or saves changes with it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend import doctor, hf_sign_in, main
from backend.doctor import PASS, WARN
from backend.launch import CHECKPOINTS
from backend.model_defaults import DEFAULT_MODEL_ID, OPEN_DEFAULT_MODEL_ID
from tests.model_switch_helpers import ANIMAGINE

# Before conftest's fixture replaces it for every other test.
REAL_TOKEN_PRESENT = main._hf_token_present


def test_token_source_is_presence_only(tmp_path):
    token = tmp_path / "token"
    assert hf_sign_in.token_source(token, {}) is None
    assert not hf_sign_in.token_present(token, {})
    token.write_text("hf_secretvalue")
    assert hf_sign_in.token_source(token, {}) == "file"
    assert hf_sign_in.token_source(tmp_path / "absent", {"HF_TOKEN": "hf_other"}) == "env"
    # An empty variable is no sign-in; a directory where the file goes is not one either.
    assert hf_sign_in.token_source(tmp_path / "absent", {"HF_TOKEN": ""}) is None
    assert hf_sign_in.token_source(tmp_path, {}) is None


def test_the_legacy_variable_is_a_sign_in_as_huggingface_hub_reads_it(tmp_path):
    absent = tmp_path / "absent"
    assert hf_sign_in.token_source(absent, {"HUGGING_FACE_HUB_TOKEN": "hf_legacy"}) == "env"
    # Blank once trimmed is none, as huggingface_hub's _clean_token has it.
    assert hf_sign_in.token_source(absent, {"HUGGING_FACE_HUB_TOKEN": " \r\n"}) is None
    assert hf_sign_in.token_source(absent, {"HF_TOKEN": "  "}) is None
    # HF_TOKEN set, even blank, is the one huggingface_hub takes, as here.
    assert hf_sign_in.token_source(absent, {"HF_TOKEN": " ", "HUGGING_FACE_HUB_TOKEN": "hf_legacy"}) is None
    assert hf_sign_in.token_source(absent, {"HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": "hf_legacy"}) == "env"


def test_the_rule_matches_huggingface_hubs_own(monkeypatch):
    from huggingface_hub.utils import _auth

    cases = [
        {}, {"HF_TOKEN": "hf_a"}, {"HUGGING_FACE_HUB_TOKEN": "hf_b"}, {"HF_TOKEN": " \n"},
        {"HF_TOKEN": " ", "HUGGING_FACE_HUB_TOKEN": "hf_b"}, {"HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": "hf_b"},
        {"HUGGING_FACE_HUB_TOKEN": "\r\n"},
    ]
    for case in cases:
        environ = dict(case)
        monkeypatch.setattr(_auth.os, "environ", environ)
        assert hf_sign_in._environment_token(environ) is (_auth._get_token_from_environment() is not None), case


def test_token_file_is_never_opened(tmp_path, monkeypatch):
    token = tmp_path / "token"
    token.write_text("hf_secretvalue")

    def refuse(*args, **kwargs):
        raise AssertionError("the token file must not be read")

    monkeypatch.setattr(Path, "open", refuse)
    monkeypatch.setattr(Path, "read_text", refuse)
    monkeypatch.setattr(Path, "read_bytes", refuse)
    assert hf_sign_in.token_present(token, {})


def test_default_path_is_huggingface_hubs_own():
    from huggingface_hub import constants

    assert hf_sign_in.default_token_path() == Path(constants.HF_TOKEN_PATH)


def test_doctor_uses_the_shared_predicate(tmp_path, monkeypatch):
    token = tmp_path / "token"
    monkeypatch.setattr(hf_sign_in, "default_token_path", lambda: token)
    [missing] = doctor.check_hf_login(environ={})
    assert missing.status == WARN
    token.write_text("hf_secretvalue")
    [present] = doctor.check_hf_login(environ={})
    assert present.status == PASS and str(tmp_path) in present.detail


def test_open_default_is_an_open_catalogue_checkpoint():
    entry = next(c for c in CHECKPOINTS if c.repo == OPEN_DEFAULT_MODEL_ID)
    assert entry.gate == "open" and OPEN_DEFAULT_MODEL_ID != DEFAULT_MODEL_ID
    gated = next(c for c in CHECKPOINTS if c.repo == DEFAULT_MODEL_ID)
    assert "gate" in gated.gate


@pytest.fixture
def fresh(client, monkeypatch, tmp_path):
    """A plain start on the built-in default with the real presence check
    pointed at an empty temporary token path and HF_TOKEN unset."""
    assert client.get("/api/status").json()["model_id"] == DEFAULT_MODEL_ID
    token = tmp_path / "token"
    monkeypatch.setattr(hf_sign_in, "default_token_path", lambda: token)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(main, "_hf_token_present", REAL_TOKEN_PRESENT)
    monkeypatch.setattr(main, "_model_unsourced", True)
    return token


def test_status_reports_the_token_and_names_the_open_checkpoint(client, fresh, monkeypatch):
    body = client.get("/api/status").json()
    assert body["hf_token_present"] is False
    assert body["open_model_suggestion"] == OPEN_DEFAULT_MODEL_ID
    # A suggestion only: the server's selection is the default still.
    assert body["model_id"] == DEFAULT_MODEL_ID
    fresh.write_text("hf_secretvalue")
    body = client.get("/api/status").json()
    assert body["hf_token_present"] is True and body["open_model_suggestion"] is None
    fresh.unlink()
    monkeypatch.setenv("HF_TOKEN", "hf_other")
    body = client.get("/api/status").json()
    assert body["hf_token_present"] is True and body["open_model_suggestion"] is None
    assert "hf_other" not in str(body)
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacy")
    body = client.get("/api/status").json()
    assert body["hf_token_present"] is True and body["open_model_suggestion"] is None
    assert "hf_legacy" not in str(body)


def test_an_explicit_or_saved_choice_is_never_suggested_over(client, fresh, monkeypatch, fake):
    assert client.get("/api/status").json()["open_model_suggestion"] == OPEN_DEFAULT_MODEL_ID
    # A saved choice or AURADREAM_MODEL_ID filled the variable at this start.
    monkeypatch.setattr(main, "_model_unsourced", False)
    assert client.get("/api/status").json()["open_model_suggestion"] is None
    monkeypatch.setattr(main, "_model_unsourced", True)
    # --model, --defaults or --menu named the checkpoint.
    for source in ("--model", "--defaults", "--menu"):
        monkeypatch.setattr(main, "_startup_overrides", {"model_id": source})
        assert client.get("/api/status").json()["open_model_suggestion"] is None
    monkeypatch.setattr(main, "_startup_overrides", {})
    # A save of the same default, from any tab, is a choice made.
    reply = client.post("/api/model/switch", json={"model_id": DEFAULT_MODEL_ID, "load": False})
    assert reply.status_code == 200, reply.text
    assert reply.json()["open_model_suggestion"] is None
    body = client.get("/api/status").json()
    assert body["open_model_suggestion"] is None and body["hf_token_present"] is False
    assert body["model_id"] == DEFAULT_MODEL_ID and fake.loads == 0


def test_another_selected_checkpoint_is_not_suggested_over(client, fresh, monkeypatch):
    monkeypatch.setattr(main.model_selection, "_selection", (ANIMAGINE, 0))
    body = client.get("/api/status").json()
    assert body["model_id"] == ANIMAGINE and body["open_model_suggestion"] is None


def test_the_open_checkpoint_is_served_under_the_default_loader(client, fresh):
    status = client.get("/api/status").json()
    listed = client.get(
        f"/api/models?quantization={status['quantization']}&offload={status['offload']}",
    ).json()["models"]
    assert OPEN_DEFAULT_MODEL_ID in {entry["model_id"] for entry in listed}


# --- The real link: a saved choice or AURADREAM_MODEL_ID is a sourced start ---

LAUNCH_VARIABLES = (
    "AURADREAM_MODEL_ID", "AURADREAM_QUANT", "AURADREAM_OFFLOAD", "AURADREAM_PRELOAD",
    "AURADREAM_COMPILATION", "AURADREAM_ATTENTION",
)


@pytest.fixture
def launch_env(monkeypatch, tmp_path):
    """A start's environment as `python -m backend.main` sees it before
    configure: no model variables, a scratch database beside any settings."""
    from backend import startup_preferences

    for name in (*LAUNCH_VARIABLES, startup_preferences.OVERRIDES_VARIABLE,
                 startup_preferences.LOADER_EXECUTION_ERROR_VARIABLE):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("AURADREAM_DB_PATH", str(tmp_path / "gallery.db"))
    return tmp_path / "gallery.db"


def _configured_unsourced(argv):
    import os

    from backend import launch

    launch.configure(argv, ask=lambda _prompt: pytest.fail("asked a terminal question"))
    return main._compute_model_unsourced(os.environ)


def test_a_plain_start_with_nothing_saved_is_unsourced(launch_env):
    assert _configured_unsourced([]) is True


def test_a_saved_default_choice_is_a_sourced_start(launch_env):
    from backend import startup_preferences

    # Apply & save of the very default the suggestion would replace.
    startup_preferences.save(
        launch_env, startup_preferences.StartupPreferences(DEFAULT_MODEL_ID, "fp8", "model", False),
    )
    assert _configured_unsourced([]) is False


def test_the_environment_or_model_flag_is_a_sourced_start(launch_env, monkeypatch):
    assert _configured_unsourced(["--model", DEFAULT_MODEL_ID]) is False
    monkeypatch.delenv("AURADREAM_MODEL_ID")
    monkeypatch.setenv("AURADREAM_MODEL_ID", DEFAULT_MODEL_ID)
    assert _configured_unsourced([]) is False


def test_main_computes_the_flag_after_its_launcher_runs():
    """`python -m backend.main` runs configure before the flag is computed;
    the other way round, every saved choice would read as unsourced."""
    import inspect

    source = inspect.getsource(main)
    launcher = source.index("    configure()\n")
    computed = source.index("_model_unsourced = _compute_model_unsourced(os.environ)")
    assert launcher < computed
