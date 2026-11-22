"""Offline tag lookups and snapshot replacement through the app's routes."""
from __future__ import annotations

from backend import tag_lookup, tag_snapshot, tags
from tests.api_analysis_helpers import ANIMAGINE, analyse, claim

SNAPSHOT = "/api/tags/snapshot"


def _row(name, *, year=2020, aliases=()):
    return {
        "name": name,
        "category": 0,
        "post_count": 500,
        "created_at": f"{year}-01-01T00:00:00Z",
        "is_deprecated": False,
        "consequent_aliases": [
            {"antecedent_name": alias, "status": "active"} for alias in aliases
        ],
    }


def _download(monkeypatch, *pages):
    monkeypatch.setattr(tag_snapshot, "PAGE_SIZE", 1)
    monkeypatch.setattr(tag_snapshot, "PAGES", len(pages))
    monkeypatch.setattr(tag_snapshot, "PAGE_INTERVAL", 0)

    def fetch(path, params):
        assert path == "/tags.json"
        item = pages[int(params["page"]) - 1]
        if isinstance(item, Exception):
            raise item
        return [item]

    monkeypatch.setattr(tags, "request_json", fetch)


def test_installed_snapshot_answers_after_cold_start_with_network_forbidden(client, fake, monkeypatch):
    claim(monkeypatch, fake, ANIMAGINE)
    _download(monkeypatch, _row("twintails", aliases=["twin_tails"]), _row("future_tag", year=2030))
    assert client.get(SNAPSHOT).json()["snapshot_date"] is None
    response = client.post(SNAPSHOT, json={"action": "download"})
    assert response.status_code == 200
    status = response.json()
    assert status["enabled"] is True and status["tag_count"] == 2
    captured = status["snapshot_date"]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("An installed offline tag index reached the network")

    monkeypatch.setattr(tags, "request_json", forbidden)
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", tags.DanbooruTagVocabulary(forbidden))
    monkeypatch.setattr(tag_snapshot, "_cache", None)
    result = client.get("/api/tags/suggest", params={"q": "twin tails"}).json()
    assert result["snapshot_date"] == captured
    assert [item["canonical"] for item in result["suggestions"]] == ["twintails"]
    result = client.post("/api/tags/validate", json={
        "tags": ["1girl", "twin tails", "some_absent_tag", "future tag"],
        "prompt": "1girl, twin tails",
    }).json()
    assert result["snapshot_date"] == captured
    guide, alias, absent, future = result["results"]
    assert guide["valid"] and guide["source"] == "model"
    assert (alias["code"], alias["canonical"], alias["category"]) == ("alias", "twintails", "general")
    assert absent["reason"] == f"absent from the Danbooru snapshot dated {captured}"
    assert future["code"] == "after_cutoff"
    assert result["slots"] == [{"tag": "1girl", "slot": "Person count"}, {"tag": "twin tails", "slot": "Details"}]
    check = analyse(client, prompt="1girl, twin tails")
    assert check["tag_snapshot_date"] == captured


def test_interrupted_refresh_preserves_previous_index_and_can_switch_source(client, fake, monkeypatch, store):
    claim(monkeypatch, fake, ANIMAGINE)
    _download(monkeypatch, _row("initial_tag"))
    installed = client.post(SNAPSHOT, json={"action": "download"}).json()
    index_file = tag_snapshot.path_for(store.settings.db_path)
    before = index_file.read_bytes()

    _download(monkeypatch, _row("replacement_tag"), tags.TagLookupUnavailable("interrupted"))
    refused = client.post(SNAPSHOT, json={"action": "download"})
    assert refused.status_code == 503
    assert index_file.read_bytes() == before
    monkeypatch.setattr(tag_snapshot, "_cache", None)
    assert client.get(SNAPSHOT).json() == installed
    suggestion = client.get("/api/tags/suggest", params={"q": "initial tag"}).json()
    assert [item["canonical"] for item in suggestion["suggestions"]] == ["initial_tag"]

    monkeypatch.setattr(tag_lookup, "tag_vocabulary", tags.DanbooruTagVocabulary(lambda *_: []))
    live = client.post(SNAPSHOT, json={"action": "live"}).json()
    assert live == {**installed, "enabled": False}
    result = client.get("/api/tags/suggest", params={"q": "initial tag"}).json()
    assert result["snapshot_date"] is None and result["suggestions"] == []
    assert client.post(SNAPSHOT, json={"action": "offline"}).json() == installed
