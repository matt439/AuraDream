"""Server startup is silent; --menu keeps the optional terminal picker."""
from __future__ import annotations

import io

import pytest

from backend import launch

TURBO = "krea/Krea-2-Turbo"
TURBO_LITE = "lite-infer/krea-2-turbo-nunchaku-lite-nvfp4_r32-bnb4-text-encoder"
FLUX = "black-forest-labs/FLUX.1-Krea-dev"


def answers(*lines: str):
    """An `ask` that hands out the given answers in turn and refuses to be asked more."""
    queue = list(lines)

    def ask(_prompt: str) -> str:
        assert queue, "the menu asked one question more than the test expected"
        return queue.pop(0)

    ask.remaining = queue  # type: ignore[attr-defined]
    return ask


def never_ask(_prompt: str) -> str:
    raise AssertionError("the menu must not run")


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    # Save and restore explicitly. `monkeypatch.delenv` records nothing for a
    # variable that was absent - the normal case - so a configure() that then
    # wrote LOCAL_DIFFUSION_MODEL_ID leaked it into every test collected afterwards.
    import os

    names = ("LOCAL_DIFFUSION_MODEL_ID", "LOCAL_DIFFUSION_QUANT", "LOCAL_DIFFUSION_OFFLOAD", "LOCAL_DIFFUSION_PRELOAD")
    saved = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ.pop(name, None)
    monkeypatch.setenv("LOCAL_DIFFUSION_DB_PATH", str(tmp_path / "local-diffusion.db"))
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def test_enter_three_times_keeps_the_shipped_defaults(clean_env):
    choice = launch.prompt(ask=answers("", "", ""), out=io.StringIO(), env={}, linux=False)
    assert choice == launch.Choice(FLUX, "fp8", "model")


def test_the_environment_is_the_default(clean_env):
    env = {"LOCAL_DIFFUSION_MODEL_ID": TURBO, "LOCAL_DIFFUSION_QUANT": "int8", "LOCAL_DIFFUSION_OFFLOAD": "sequential"}
    choice = launch.prompt(ask=answers("", "", ""), out=io.StringIO(), env=env, linux=False)
    assert choice == launch.Choice(TURBO, "int8", "sequential")


def test_numbers_and_names_both_select(clean_env):
    out = io.StringIO()
    choice = launch.prompt(ask=answers("int8", "2", "none"), out=out, env={}, linux=False)
    assert choice == launch.Choice(TURBO, "int8", "none")


def test_rouwei_selects_the_official_epsilon_checkpoint(clean_env):
    number = str(next(i for i, c in enumerate(launch.CHECKPOINTS, 1) if c.name == "RouWei 0.8 epsilon"))
    out = io.StringIO()
    choice = launch.prompt(ask=answers("fp8", number, "model"), out=out, env={}, linux=False)
    assert choice == launch.Choice("Minthy/RouWei-0.8", "fp8", "model")
    assert "official epsilon single file" in out.getvalue()


def test_a_bad_answer_is_asked_again(clean_env):
    out = io.StringIO()
    ask = answers("9", "fp9", "", "", "")
    choice = launch.prompt(ask=ask, out=out, env={}, linux=False)
    assert choice.quantization == "fp8"
    assert "pick 1-" in out.getvalue()


def test_noobai_selects_the_v_prediction_release(clean_env):
    number = str(next(i for i, c in enumerate(launch.CHECKPOINTS, 1)
                      if c.name == "NoobAI-XL V-pred 1.0"))
    out = io.StringIO()
    choice = launch.prompt(ask=answers("fp8", number, "model"), out=out, env={}, linux=False)
    assert choice == launch.Choice("Laxhar/noobai-XL-Vpred-1.0", "fp8", "model")
    assert "V-prediction Euler" in out.getvalue()


def test_nunchaku_lists_the_lite_repos_and_asks_the_format(clean_env):
    out = io.StringIO()
    # quant nunchaku, format nvfp4 (default), checkpoint Turbo, offload default
    choice = launch.prompt(ask=answers("nunchaku", "", "2", ""), out=out, env={}, linux=True)
    assert choice == launch.Choice(TURBO_LITE, "nunchaku", "model")
    listing = out.getvalue()
    assert "Krea 2 Raw" not in listing, "no Nunchaku Lite build exists for Raw"
    assert "base 4B" not in listing
    assert "RouWei" not in listing, "no Nunchaku Lite build is mapped for RouWei"
    assert "NoobAI" not in listing, "no Nunchaku Lite build is mapped for NoobAI"


def test_int4_names_the_older_cards_repo(clean_env):
    choice = launch.prompt(ask=answers("nunchaku", "int4", "1", ""), out=io.StringIO(), env={}, linux=True)
    assert choice.model_id == "lite-infer/flux.1-krea-dev-nunchaku-lite-int4_r32-bnb4-text-encoder"


def test_a_lite_repo_in_the_environment_marks_its_family_and_format(clean_env):
    env = {"LOCAL_DIFFUSION_MODEL_ID": TURBO_LITE.replace("nvfp4", "int4"), "LOCAL_DIFFUSION_QUANT": "nunchaku"}
    out = io.StringIO()
    choice = launch.prompt(ask=answers("", "", "", ""), out=out, env=env, linux=True)
    assert choice.model_id == env["LOCAL_DIFFUSION_MODEL_ID"]
    assert choice.quantization == "nunchaku"


def test_nunchaku_is_not_offered_off_linux(clean_env):
    out = io.StringIO()
    choice = launch.prompt(
        ask=answers("", "", ""), out=out, env={"LOCAL_DIFFUSION_QUANT": "nunchaku"}, linux=False
    )
    assert "nunchaku" not in out.getvalue().split("Checkpoint")[0]
    assert choice.quantization == "fp8", "an unofferable default falls back to fp8"


def test_another_repo_id_can_be_typed_or_pasted(clean_env):
    # "another repo id" is the option after the last catalogue entry, whatever
    # the catalogue's length is this year.
    another = str(len(launch.CHECKPOINTS) + 1)
    typed = launch.prompt(
        ask=answers("", another, "nobody/whatever", ""), out=io.StringIO(), env={}, linux=False
    )
    assert typed.model_id == "nobody/whatever"
    pasted = launch.prompt(ask=answers("", "some/repo", ""), out=io.StringIO(), env={}, linux=False)
    assert pasted.model_id == "some/repo"


def test_an_unrecognised_id_in_the_environment_survives_bare_enters(clean_env):
    # An id the catalogue does not know - typed at "another repo id" last time,
    # or a local directory - used to leave the marker on FLUX, so the three
    # Enters the header invites started a 32 GB download of the wrong model.
    env = {"LOCAL_DIFFUSION_MODEL_ID": "nobody/whatever"}
    out = io.StringIO()
    choice = launch.prompt(ask=answers("", "", ""), out=out, env=env, linux=False)
    assert choice.model_id == "nobody/whatever"
    assert "nobody/whatever" in out.getvalue()


def test_eof_at_the_repo_id_prompt_takes_the_default(clean_env):
    # Stdin lost after picking "another repo id" used to loop forever at 100%
    # CPU with no server started.
    calls = []

    def ask(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return ""  # quant
        if len(calls) == 2:
            return str(len(launch.CHECKPOINTS) + 1)  # another repo id
        raise EOFError

    choice = launch.prompt(ask=ask, out=io.StringIO(), env={}, linux=False)
    assert choice.model_id == FLUX
    assert len(calls) < 10


def test_eof_takes_the_default(clean_env):
    def ask(_prompt: str) -> str:
        raise EOFError

    choice = launch.prompt(ask=ask, out=io.StringIO(), env={}, linux=False)
    assert choice == launch.Choice(FLUX, "fp8", "model")


def test_ctrl_c_exits_130(clean_env):
    def ask(_prompt: str) -> str:
        raise KeyboardInterrupt

    with pytest.raises(SystemExit) as exc:
        launch.prompt(ask=ask, out=io.StringIO(), env={}, linux=False)
    assert exc.value.code == 130


def test_the_menu_prints_the_equivalent_command(clean_env):
    out = io.StringIO()
    launch.prompt(ask=answers("", "2", ""), out=out, env={}, linux=False)
    assert f"--model {TURBO} --quant fp8 --offload model" in out.getvalue()


# --- configure(): when the menu runs at all ----------------------------------


def test_menu_flag_asks_and_sets_the_environment(clean_env):
    choice = launch.configure(["--menu"], ask=answers("", "2", ""), out=io.StringIO(), interactive=True)
    assert choice is not None
    import os

    assert os.environ["LOCAL_DIFFUSION_MODEL_ID"] == TURBO
    assert os.environ["LOCAL_DIFFUSION_QUANT"] == "fp8"
    assert os.environ["LOCAL_DIFFUSION_OFFLOAD"] == "model"


def test_no_arguments_on_a_terminal_never_asks(clean_env):
    assert launch.configure([], ask=never_ask, interactive=True) is None


def test_any_argument_skips_the_menu(clean_env):
    import os

    assert launch.configure(["--model", TURBO], ask=never_ask, interactive=True) is None
    assert os.environ["LOCAL_DIFFUSION_MODEL_ID"] == TURBO
    assert "LOCAL_DIFFUSION_QUANT" not in os.environ, "what the flags do not name is left to config.py"


def test_defaults_flag_skips_the_menu_and_touches_nothing(clean_env):
    import os

    assert launch.configure(["--defaults"], ask=never_ask, interactive=True) is None
    assert "LOCAL_DIFFUSION_MODEL_ID" not in os.environ


def test_without_a_terminal_there_is_no_menu(clean_env):
    assert launch.configure([], ask=never_ask, interactive=False) is None


def test_a_bad_quant_flag_is_rejected_by_argparse(clean_env):
    with pytest.raises(SystemExit) as exc:
        launch.configure(["--quant", "fp4"], ask=never_ask, interactive=True)
    assert exc.value.code == 2


def test_the_cli_shares_the_flags(clean_env):
    from backend import generate

    parser = generate._build_parser()
    args = parser.parse_args(["hello", "--model", TURBO, "--quant", "int8", "--offload", "none"])
    launch.apply_env(args)
    import os

    assert (os.environ["LOCAL_DIFFUSION_MODEL_ID"], os.environ["LOCAL_DIFFUSION_QUANT"], os.environ["LOCAL_DIFFUSION_OFFLOAD"]) == (
        TURBO,
        "int8",
        "none",
    )


def test_a_lite_id_spelled_by_hand_reaches_the_menu(clean_env):
    # ERNIE's re-upload has no `_r32`, so its entry cannot use `_LITE`; the id
    # the menu hands out has to be the one lite-infer actually publishes, in
    # either format.
    lite = [c for c in launch.CHECKPOINTS if c.lite]
    number = str(next(i for i, c in enumerate(lite, 1) if c.name == "ERNIE-Image Turbo"))
    for fmt in ("nvfp4", "int4"):
        choice = launch.prompt(ask=answers("nunchaku", fmt, number, ""), out=io.StringIO(), env={}, linux=True)
        assert choice.model_id == f"lite-infer/ERNIE-Image-Turbo-nunchaku-lite-{fmt}-bnb4-text-encoder"
