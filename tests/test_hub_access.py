"""A switch asks the Hub before it releases the working checkpoint
(`hub_access`, `main._refuse_hub_access`). Nothing here reaches the network:
the Hub's answers are raised by hand, as huggingface_hub raises them."""
from __future__ import annotations

import time

import httpx
import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    OfflineModeIsEnabled,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from backend import hub_access, main, model_info, model_selection, startup_preferences
from backend.hub_access import _ask as REAL_ASK  # before conftest's stub replaces it

KREA_DEV = "black-forest-labs/FLUX.1-Krea-dev"
TURBO = "krea/Krea-2-Turbo"
ANIMAGINE = "cagliostrolab/animagine-xl-4.0"
NETAYUME = "duongve/NetaYume-Lumina-Image-2.0"
Z_ANIME = "SeeSee21/Z-Anime"


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("HEAD", "https://huggingface.co/x"))


def _error(cls, status: int):
    return cls(f"{status} Client Error.", response=_response(status))


@pytest.fixture(autouse=True)
def fresh_selection(monkeypatch):
    monkeypatch.setattr(model_selection, "_selection", (None, 0))
    monkeypatch.setattr(main, "_switching", False)
    monkeypatch.setattr(main, "_model_requests", 0)
    monkeypatch.setattr(main, "_switch_draining", False)


@pytest.fixture
def uncached(monkeypatch):
    """Nothing on disk, so every check asks the Hub."""
    monkeypatch.setattr(hub_access, "download_facts", lambda model_id: {"cache_state": "absent"})


@pytest.fixture
def hub(monkeypatch, uncached):
    """The Hub's answers by repository (an exception, or None for allowed);
    `asked` records each source asked."""
    answers: dict[str, BaseException | None] = {}
    asked: list[hub_access.Source] = []

    def ask(source):
        asked.append(source)
        answer = answers.get(source.repo_id)
        if answer is not None:
            raise answer

    monkeypatch.setattr(hub_access, "_ask", ask)
    return answers, asked


def test_sources_follow_the_download_recipe_with_its_revisions_and_files():
    [krea] = hub_access.sources(KREA_DEV)
    assert (krea.repo_id, krea.revision) == (KREA_DEV, "main")
    assert krea.filename.endswith(".json")
    # A single-file checkpoint is checked at its pinned commit, on the file
    # its conversion reads, beside the pipeline repo it borrows the rest from.
    references = model_info.download_references()
    pinned = [references["sources"][item["source"]] for item in references["models"][NETAYUME]["sources"]]
    found = hub_access.sources(NETAYUME)
    assert [(s.repo_id, s.revision) for s in found] == [(p["repo_id"], p["revision"]) for p in pinned]
    assert found[0].filename in pinned[0]["files"]
    assert all(len(s.revision) == 40 for s in found)
    # A recipe without a file list is asked through the repository's auth check.
    [dev] = hub_access.sources("black-forest-labs/FLUX.1-dev")
    assert dev.filename is None
    # Off the recipe: the id itself, unless it is a local path.
    assert hub_access.sources("someone/custom") == [hub_access.Source("someone/custom", None, None)]


def test_a_local_path_is_never_asked(tmp_path):
    assert hub_access.sources(str(tmp_path)) == []


def test_ask_heads_the_file_at_the_recipe_revision_with_a_timeout(monkeypatch):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata",
                        lambda url, timeout=None: calls.append((url, timeout)))
    monkeypatch.setattr(huggingface_hub.HfApi, "auth_check",
                        lambda self, repo_id: calls.append(("auth", repo_id)))
    REAL_ASK(hub_access.Source(NETAYUME, "a" * 40, "file.safetensors"))
    REAL_ASK(hub_access.Source("someone/custom", None, None))
    assert calls == [
        (f"https://huggingface.co/{NETAYUME}/resolve/{'a' * 40}/file.safetensors", hub_access.REQUEST_SECONDS),
        ("auth", "someone/custom"),
    ]


def test_cached_files_need_no_answer(monkeypatch):
    monkeypatch.setattr(hub_access, "download_facts", lambda model_id: {"cache_state": "complete"})

    def never(source):
        raise AssertionError("a cached checkpoint asked the Hub")

    monkeypatch.setattr(hub_access, "_ask", never)
    assert hub_access.check(TURBO) is None


def test_a_gated_repo_is_refused_with_the_catalogues_steps(hub):
    answers, _ = hub
    answers[TURBO] = _error(GatedRepoError, 403)
    refusal = hub_access.check(TURBO)
    assert refusal.kind == "gated" and refusal.url == f"https://huggingface.co/{TURBO}"
    assert refusal.detail.startswith("Krea 2 Turbo is gated on Hugging Face")
    assert "wait for its authors to approve it" in refusal.detail
    assert "uv run hf auth login" in refusal.detail
    answers[KREA_DEV] = _error(GatedRepoError, 401)
    assert "accept its licence" in hub_access.check(KREA_DEV).detail


def test_a_refused_second_source_names_its_own_repo(hub):
    answers, asked = hub
    second = hub_access.sources(Z_ANIME)[1].repo_id
    answers[second] = _error(GatedRepoError, 403)
    refusal = hub_access.check(Z_ANIME)
    assert refusal.kind == "gated" and refusal.url == f"https://huggingface.co/{second}"
    assert f"needs {second}, which is gated" in refusal.detail
    assert [s.repo_id for s in asked] == [Z_ANIME, second]


@pytest.mark.parametrize("error, kind", [
    (_error(RepositoryNotFoundError, 404), "not_found"),
    (_error(RevisionNotFoundError, 404), "not_found"),
    (_error(HfHubHTTPError, 401), "gated"),
    (_error(HfHubHTTPError, 403), "gated"),
])
def test_other_refusals(hub, error, kind):
    answers, _ = hub
    answers[ANIMAGINE] = error
    refusal = hub_access.check(ANIMAGINE)
    assert refusal.kind == kind and ANIMAGINE in refusal.detail


@pytest.mark.parametrize("error", [
    OfflineModeIsEnabled("offline"),
    httpx.ConnectError("no route"),
    httpx.ReadTimeout("slow"),
    _error(HfHubHTTPError, 500),
    _error(RemoteEntryNotFoundError, 404),
    RuntimeError("anything else"),
])
def test_an_answer_that_settles_nothing_lets_the_switch_go_on(hub, error):
    answers, _ = hub
    answers[ANIMAGINE] = error
    assert hub_access.check(ANIMAGINE) is None


def test_an_unsettled_source_does_not_hide_a_refused_one(hub):
    answers, _ = hub
    first, second = (s.repo_id for s in hub_access.sources(Z_ANIME))
    answers[first] = httpx.ConnectError("no route")
    answers[second] = _error(GatedRepoError, 403)
    assert hub_access.check(Z_ANIME).kind == "gated"


# The route.

@pytest.mark.parametrize("load", [True, False])
def test_a_refused_switch_keeps_the_working_checkpoint(client, fake, hub, load):
    answers, _ = hub
    answers[TURBO] = _error(GatedRepoError, 403)
    initial = client.get("/api/status").json()
    assert fake.loaded and initial["model_id"] != TURBO
    reply = client.post("/api/model/switch", json={"model_id": TURBO, "load": load})
    assert reply.status_code == 403, reply.text
    detail = reply.json()["detail"]
    assert detail.startswith("Krea 2 Turbo is gated on Hugging Face")
    assert detail.endswith(f"{main._model_name(initial['model_id'])} is still selected.")
    after = client.get("/api/status").json()
    assert (after["model_id"], after["model_revision"]) == (initial["model_id"], initial["model_revision"])
    assert after["state"] == initial["state"] and not after["switching"]
    assert after["switch_error"] is None
    # Nothing saved, nothing released.
    assert startup_preferences.load(main.settings.db_path) is None
    assert fake.loaded and fake.loads == 0


def test_a_missing_repo_is_a_404(client, fake, hub):
    answers, _ = hub
    answers[ANIMAGINE] = _error(RepositoryNotFoundError, 404)
    reply = client.post("/api/model/switch", json={"model_id": ANIMAGINE})
    assert reply.status_code == 404 and "is not on Hugging Face" in reply.json()["detail"]
    assert fake.loaded and fake.loads == 0


def test_an_allowed_switch_goes_on(client, fake, hub):
    _, asked = hub
    reply = client.post("/api/model/switch", json={"model_id": ANIMAGINE})
    assert reply.status_code == 200, reply.text
    assert [s.repo_id for s in asked] == [ANIMAGINE]


def test_a_check_that_outlasts_its_wait_lets_the_switch_go_on(client, fake, monkeypatch):
    monkeypatch.setattr(hub_access, "CHECK_SECONDS", 0.1)
    monkeypatch.setattr(hub_access, "check", lambda model_id: time.sleep(1) or None)
    started = time.monotonic()
    reply = client.post("/api/model/switch", json={"model_id": ANIMAGINE})
    assert reply.status_code == 200, reply.text
    assert time.monotonic() - started < 1


def test_loading_options_alone_ask_nothing(client, fake, monkeypatch):
    def never(model_id):
        raise AssertionError("the same checkpoint was checked")

    monkeypatch.setattr(hub_access, "check", never)
    reply = client.post("/api/model/switch", json={"model_id": client.get("/api/status").json()["model_id"],
                                                   "quantization": "int8"})
    assert reply.status_code == 200, reply.text


def test_a_busy_server_refuses_before_asking(client, fake, monkeypatch):
    def never(model_id):
        raise AssertionError("a busy switch asked the Hub")

    monkeypatch.setattr(hub_access, "check", never)
    monkeypatch.setattr(main, "_switching", True)
    assert client.post("/api/model/switch", json={"model_id": ANIMAGINE}).status_code == 409


def test_a_switch_that_lands_during_the_check_is_refused_after_it(client, fake, monkeypatch):
    def meanwhile(model_id):
        main._switching = True  # another tab's switch, reserved while this one asked

    monkeypatch.setattr(hub_access, "check", meanwhile)
    reply = client.post("/api/model/switch", json={"model_id": ANIMAGINE})
    assert reply.status_code == 409 and "switching" in reply.json()["detail"]
    assert fake.loaded and fake.loads == 0
