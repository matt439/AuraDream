"""A throwaway gallery - its own database file and outputs directory - and
the app on the fake backend (`fake`, `client`) for the HTTP tests."""
from __future__ import annotations

import asyncio
import dataclasses
import os
import shutil
import sys

import pytest


from fastapi.testclient import TestClient

# The suite runs on config.py's defaults, whatever the shell exports. `settings`
# reads the environment once, when `backend.config` is first imported - by the
# imports below - so this has to happen here, not in a fixture: `wsl/env.sh`
# exports Nunchaku and a lite-infer checkpoint, and tests written against the
# default checkpoint and fp8 failed under it. Where the gallery and the adapters
# live stays the shell's: the tests redirect those themselves, and one that
# forgot should reach this environment's gallery, not the one under /mnt/c.
# The browser `tools/browser.mjs` drives is no app setting, and
# `test_browser_profile.py` finds it there.
assert "backend.config" not in sys.modules, "backend.config was imported before the tests' environment was set"
_KEPT = {
    "AURADREAM_OUTPUTS_DIR", "AURADREAM_DB_PATH", "AURADREAM_LORAS_DIR",
    "AURADREAM_BROWSER",
}
for _name in [name for name in os.environ if name.startswith("AURADREAM_") and name not in _KEPT]:
    del os.environ[_name]
# The names from before the rename to AuraDream are read by nothing, but
# importing `main` warns about any still set; the warning has its own tests.
for _name in [name for name in os.environ if name.startswith("LOCAL_DIFFUSION_")]:
    del os.environ[_name]

from backend import bundle_crypto, main, prompt_encoding, storage  # noqa: E402
from backend.jobs import JobManager  # noqa: E402
from tests.api_helpers import ORIGIN  # noqa: E402
from tests.fakes import FakeBackend, FakeCritic, FakeTranslator, fake_tokenizer  # noqa: E402
from tests import shard  # noqa: E402

# The CUDA build of torch on Windows with no NVIDIA driver (the CI runner) dies
# with an access violation when transformers' masks ask whether a CUDA stream
# is capturing - native, so its try/except cannot catch it. With no device
# none is: answer that without the call. A machine with a GPU is untouched.
if sys.platform == "win32":
    import torch

    if not torch.cuda.is_available():
        torch.cuda.is_current_stream_capturing = lambda: False


_SHIPPED_SCRYPT = (bundle_crypto._LOG2_N, bundle_crypto._R, bundle_crypto._P)


@pytest.fixture
def node():
    """The Node binary the browser-script tests drive.

    Node is required, as it is for the classic-script guard: a missing one
    fails these tests instead of skipping them, so a run without it cannot
    pass by checking less.
    """
    path = shutil.which("node")
    if path is None:
        pytest.fail("Node is required for the browser-script tests; install it and put it on PATH")
    return path


@pytest.fixture(autouse=True)
def no_real_card(monkeypatch):
    """Every test runs on an unknown card, whatever this machine has, so a
    card-keyed default (the Nunchaku format) is the one CI sees; a test that
    wants a card patches `card_facts.card` itself."""
    from backend import card_facts

    monkeypatch.setattr(card_facts, "card", lambda device="cuda": None)


@pytest.fixture(autouse=True)
def no_real_credential_vault(monkeypatch):
    """All credential qualification is stub-only; never call the user's vault."""
    from backend import backup_credentials

    def forbidden():
        pytest.fail("A test attempted a real Windows credential API; inject an explicit stub instead")

    monkeypatch.setattr(backup_credentials, "_credential_api", forbidden)


@pytest.fixture(autouse=True)
def no_real_keep_awake(monkeypatch):
    """No test takes a real systemd-inhibit hold (a Linux runner has one):
    the server's lifespan and the CLI get an inhibitor that holds nothing.
    Tests of the hold build theirs with a fake popen."""
    from backend import keep_awake

    monkeypatch.setattr(keep_awake, "platform_inhibitor", lambda *args, **kwargs: keep_awake.NoInhibitor())
    yield
    keep_awake.install(None)


@pytest.fixture(autouse=True)
def cheap_scrypt(monkeypatch):
    # New bundles get the smallest cost readers accept (16 MiB, not 128 MiB).
    monkeypatch.setattr(bundle_crypto, "_LOG2_N", 14)


@pytest.fixture
def shipped_scrypt():
    """The scrypt (log2 n, r, p) the module ships, before `cheap_scrypt`."""
    return _SHIPPED_SCRYPT


@pytest.fixture
def store(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    # `settings` is frozen, so point the module at a copy rather than mutating it.
    monkeypatch.setattr(
        storage,
        "settings",
        dataclasses.replace(storage.settings, db_path=tmp_path / "auradream.db", outputs_dir=outputs),
    )
    storage.init_db()
    return storage


@pytest.fixture(autouse=True)
def no_hub(monkeypatch):
    """Nothing here downloads. `load()` reads a checkpoint's model_index.json
    ahead of the weights, so a test that stubs `from_pretrained` must not be
    let through to the Hub for that one file either; a test that needs an
    index replaces this stub with its own."""
    from backend.backends import loading

    def refuse(model_id, filename):
        raise RuntimeError(f"the tests do not reach the Hub ({model_id}/{filename})")

    monkeypatch.setattr(loading, "hub_file", refuse)
    # Single-file conversion has its own downloads. Tests exercising that
    # path replace these too; a missed stub must never pull real weights.
    import huggingface_hub

    def refuse_download(*args, **kwargs):
        raise RuntimeError("the tests do not download model files")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", refuse_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", refuse_download)

    # diffusers 0.42 resolves a revision against the Hub before every
    # `download`/`from_pretrained` (`hub_utils._resolve_revision`), swallowing
    # only Hub and transport errors, so this RuntimeError surfaces. Its
    # binding is taken at diffusers' import: patch the source, and the
    # imported copy when diffusers is already loaded. Never imports diffusers.
    def refuse_resolve(*args, **kwargs):
        raise RuntimeError("the tests do not resolve revisions on the Hub")

    monkeypatch.setattr(huggingface_hub, "resolve_revision", refuse_resolve)
    hub_utils = sys.modules.get("diffusers.utils.hub_utils")
    if hub_utils is not None:
        monkeypatch.setattr(hub_utils, "resolve_revision", refuse_resolve)
    # A switch asks the Hub whether this sign-in may download its checkpoint
    # (`hub_access.check`). Unanswered, as here, the switch goes ahead;
    # tests of the check replace this with the answer they need.
    from backend import hub_access

    def unanswered(source):
        raise RuntimeError(f"the tests do not reach the Hub ({source.repo_id})")

    monkeypatch.setattr(hub_access, "_ask", unanswered)


@pytest.fixture
def fake(store, monkeypatch):
    monkeypatch.setattr(main, "settings", dataclasses.replace(
        main.settings, db_path=store.settings.db_path,
    ))
    monkeypatch.setattr(main, "_startup_preload", None)
    monkeypatch.setattr(main, "_switch_error", None)
    monkeypatch.setattr(main, "_startup_overrides", {})
    # A fresh visit's facts, never this machine's: signed in, the checkpoint
    # named by the environment, nothing saved (tests of the suggestion set them).
    monkeypatch.setattr(main, "_hf_token_present", lambda: True)
    monkeypatch.setattr(main, "_model_unsourced", False)
    monkeypatch.setattr(main, "_model_choice_saved", False)
    monkeypatch.setattr(main.model_selection, "_selection", (None, 0))
    backend = FakeBackend()
    translator = FakeTranslator()
    backend.translator = translator
    critic = FakeCritic()
    backend.critic = critic
    monkeypatch.setattr(main, "backend", backend)
    monkeypatch.setattr(main, "prompt_translator", translator)
    monkeypatch.setattr(main, "prompt_critic", critic)
    monkeypatch.setattr(main, "_load_state", {"state": "unloaded", "detail": None})
    monkeypatch.setattr(main, "_load_lock", asyncio.Lock())
    # A fresh manager per test: shutting the app down drains its one worker
    # thread for good. Wired the way main.py wires it, so the single load path
    # the whole app relies on is the one under test.
    monkeypatch.setattr(main, "jobs", JobManager(backend, main.ensure_loaded))
    monkeypatch.setattr(main, "bundle_manager", main.api_bundles.BundleManager(
        lambda: main.jobs,
        lambda: main._switching,
    ))
    # The analyser's tokenizers without the Hub, and a cache that starts and
    # ends empty: it is module state, and another test file reads it too.
    monkeypatch.setattr(prompt_encoding, "_load_tokenizer", fake_tokenizer)
    prompt_encoding.unload()
    yield backend
    prompt_encoding.unload()


@pytest.fixture
def client(fake):
    # base_url has to be a host the app answers to - which is the point of the
    # middleware, and why TestClient's default "testserver" would 400.
    with TestClient(main.app, base_url=ORIGIN) as c:
        yield c


def pytest_addoption(parser):
    parser.addoption("--shard", metavar="K/N", default=None,
                     help="run only the K-th of N parts of the collected tests (tests/shard.py)")


def pytest_collection_modifyitems(config, items):
    value = config.getoption("--shard")
    if not value:
        return
    try:
        index, count = shard.parse(value)
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc
    kept = [item for item in items if shard.part(item.nodeid, count) == index]
    dropped = [item for item in items if shard.part(item.nodeid, count) != index]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = kept
