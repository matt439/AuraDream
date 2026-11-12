"""lite-infer's renamed repositories are read-only aliases (GEN#47).

The Hub answers every lite id the catalogue writes with a 307 to a new name
(checked 2026-11-12). The app keeps writing the old ids - settings, PNG
metadata, `CellKey` and run ids already hold them - and reads a new name as
the same entry and download recipe. The 307s themselves are answered here
by an httpx `MockTransport` under huggingface_hub's own client: nothing
reaches the network."""
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub import constants, file_download
from huggingface_hub.utils import _http

from backend import api_runs, hub_access, main, model_benchmarks, model_info, model_selection, startup_preferences
from backend.hub_access import _ask as REAL_ASK  # before conftest's stub replaces it
from backend.launch import CHECKPOINTS, FORMATS, _match, catalogue_entry, written_id
from tests.model_switch_helpers import fresh_selection  # noqa: F401  (autouse, by name)

LITE = [(entry, fmt) for entry in CHECKPOINTS if entry.lite for fmt, _ in FORMATS]
# Seen on the Hub's API (a 307 Location), 2026-11-12.
OBSERVED = {
    "lite-infer/krea-2-turbo-nunchaku-lite-int4_r32-bnb4-text-encoder": "lite-infer/krea-2-turbo-nunchaku-lite-int4-r32",
    "lite-infer/Qwen-Image-nunchaku-lite-nvfp4_r32-bnb4-text-encoder": "lite-infer/Qwen-Image-nunchaku-lite-nvfp4-r32",
    "lite-infer/ERNIE-Image-Turbo-nunchaku-lite-int4-bnb4-text-encoder": "lite-infer/ERNIE-Image-Turbo-nunchaku-lite-int4",
}
KREA_INT4 = "lite-infer/krea-2-turbo-nunchaku-lite-int4_r32-bnb4-text-encoder"
KREA_INT4_NEW = OBSERVED[KREA_INT4]


def test_each_lite_id_has_its_observed_new_name():
    aliases = {entry.lite_repo(fmt): entry.lite_alias(fmt) for entry, fmt in LITE}
    assert OBSERVED.items() <= aliases.items()
    assert len(set(aliases.values())) == len(aliases)
    for old, new in aliases.items():
        assert new != old and "bnb4" not in new and new.startswith("lite-infer/")


@pytest.mark.parametrize(("entry", "fmt"), LITE, ids=[f"{e.name} {f}" for e, f in LITE])
def test_a_new_name_is_the_same_entry_format_and_written_id(entry, fmt):
    old, new = entry.lite_repo(fmt), entry.lite_alias(fmt)
    assert _match(new) == _match(old) == (entry, fmt)
    assert catalogue_entry(new) is entry and entry.names(new)
    assert written_id(new) == old and written_id(old) == old


def test_written_ids_are_unchanged():
    # Every id the app has ever written reads as itself: keys and run ids
    # computed from a stored request cannot move.
    for entry in CHECKPOINTS:
        assert written_id(entry.repo) == entry.repo
    for entry, fmt in LITE:
        assert written_id(entry.lite_repo(fmt)) == entry.lite_repo(fmt)
    for other in ("someone/custom", "lite-infer/flux-3-image-klein-nunchaku-lite-nvfp4-r32", "C:/models/local"):
        assert written_id(other) == other


def test_every_exact_id_site_reads_a_new_name_as_its_entry(monkeypatch):
    new = KREA_INT4_NEW
    entry, _ = _match(KREA_INT4)
    assert api_runs._model_name(new) == api_runs._model_name(KREA_INT4) == entry.name
    assert model_benchmarks.speed_for(new) == model_benchmarks.speed_for(KREA_INT4)
    assert model_info.download_reference_bytes(new) == model_info.download_reference_bytes(KREA_INT4) is not None
    assert model_info._catalogue_entry(new) is entry
    assert hub_access.sources(new) == hub_access.sources(KREA_INT4)
    assert hub_access.sources(new)[0].repo_id == KREA_INT4  # asked by the id the app downloads


def test_the_environment_variable_starts_under_the_written_id():
    # `settings` reads the environment at import: a fresh interpreter.
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("AURADREAM_", "DIFFUSERS_"))}
    result = subprocess.run(
        [sys.executable, "-c", "from backend.config import settings; print(settings.model_id)"],
        env={**clean, "AURADREAM_MODEL_ID": KREA_INT4_NEW}, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == KREA_INT4


@pytest.fixture
def linux_nunchaku(monkeypatch):
    monkeypatch.setattr(model_selection, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setenv("DIFFUSERS_TRUST_REMOTE_KERNELS", "true")
    monkeypatch.setitem(model_selection.imported_modules, "diffusers.utils.constants",
                        SimpleNamespace(DIFFUSERS_TRUST_REMOTE_KERNELS=True))


def test_a_switch_to_a_new_name_saves_the_written_id(client, fake, monkeypatch, linux_nunchaku):
    flux = next(entry for entry in CHECKPOINTS if entry.lite)
    monkeypatch.setattr(main, "settings", dataclasses.replace(
        main.settings, model_id=flux.lite_repo("int4"), quantization="nunchaku"))
    reply = client.post("/api/model/switch", json={"model_id": KREA_INT4_NEW, "load": False})
    assert reply.status_code == 200, reply.text
    assert reply.json()["model_id"] == KREA_INT4
    saved = startup_preferences.load(main.settings.db_path)
    assert saved is not None and saved.model_id == KREA_INT4
    assert model_selection.validate(main.settings, KREA_INT4_NEW, load=False).model_id == KREA_INT4


# --- through a 307, as the Hub answers ---------------------------------------

COMMIT = "c" * 40
BODY = b'{"_class_name": "Krea2Pipeline"}'


@pytest.fixture
def hub(monkeypatch):
    """huggingface_hub's own client over a MockTransport: every old id answers
    a 307 to its new name, as the Hub's API does; `gated` repos answer 403."""
    requests: list[tuple[str, str]] = []
    gated: set[str] = set()

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        requests.append((request.method, path))
        for old, new in OBSERVED.items():
            if f"/{old}/" in path or path.endswith(f"/{old}"):
                return httpx.Response(307, headers={"Location": path.replace(old, new)})
        if any(f"/{repo}/" in path or path.endswith(f"/{repo}") for repo in gated):
            return httpx.Response(403, headers={"X-Error-Code": "GatedRepo", "X-Error-Message": "gated"})
        if path.endswith("/auth-check"):
            return httpx.Response(200, json={})
        if "/resolve/" in path:
            headers = {"X-Repo-Commit": COMMIT, "ETag": '"etag-1"', "Content-Length": str(len(BODY))}
            return httpx.Response(200, headers=headers, content=b"" if request.method == "HEAD" else BODY)
        return httpx.Response(404)

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    _http.set_client_factory(lambda: httpx.Client(transport=httpx.MockTransport(handle), follow_redirects=True))
    yield requests, gated
    _http.set_client_factory(_http.default_client_factory)


def test_the_access_check_follows_a_307_to_the_renamed_repository(hub, monkeypatch):
    requests, _ = hub
    monkeypatch.setattr(hub_access, "_ask", REAL_ASK)
    monkeypatch.setattr(hub_access, "download_facts", lambda model_id: {"cache_state": "absent"})
    assert hub_access.check(KREA_INT4) is None
    assert hub_access.check(KREA_INT4_NEW) is None
    heads = [path for method, path in requests if method == "HEAD"]
    assert any(KREA_INT4 in path for path in heads) and any(KREA_INT4_NEW in path for path in heads)
    # The repository's auth check (a recipe without files) follows it too.
    REAL_ASK(hub_access.Source(KREA_INT4, None, None))
    assert ("GET", f"/api/models/{KREA_INT4_NEW}/auth-check") in requests


def test_a_refusal_behind_the_307_is_still_a_refusal(hub, monkeypatch):
    _, gated = hub
    gated.add(KREA_INT4_NEW)
    monkeypatch.setattr(hub_access, "_ask", REAL_ASK)
    monkeypatch.setattr(hub_access, "download_facts", lambda model_id: {"cache_state": "absent"})
    refusal = hub_access.check(KREA_INT4)
    assert refusal is not None and refusal.kind == "gated"
    assert refusal.url == f"https://huggingface.co/{KREA_INT4}"


def test_a_download_through_the_307_lands_where_the_cache_check_reads(hub, tmp_path):
    requests, _ = hub
    path = file_download.hf_hub_download(KREA_INT4, "model_index.json", revision="main", cache_dir=str(tmp_path))
    with open(path, "rb") as stream:
        assert stream.read() == BODY
    # Cached under the id asked for, at the commit the renamed repo answered.
    folder = tmp_path / ("models--" + KREA_INT4.replace("/", "--"))
    assert (folder / "refs" / "main").read_text(encoding="utf-8") == COMMIT
    assert (folder / "snapshots" / COMMIT / "model_index.json").is_file()
    assert any(KREA_INT4_NEW in p for _, p in requests)
    source = {"repo_id": KREA_INT4, "revision": "main", "commit": COMMIT,
              "files": {"model_index.json": len(BODY)}, "indexes": {}}
    assert model_info._hub_source_state(source, ["model_index.json"], str(tmp_path)) == "complete"
