"""RouWei's own prompt conventions through its profile, rules and HTTP helper."""
from __future__ import annotations

from backend import main, prompt_encoding, prompt_rules, prompt_sources, tags
from backend.model_defaults import _FALLBACK, _profile_for, defaults_for, guide_for
from tests.api_analysis_helpers import Vocabulary, analyse, claim, rules
from tests.prompt_rules_helpers import only, run


ROUWEI = "Minthy/RouWei-0.8"


def test_rouwei_epsilon_profile_does_not_cover_other_releases():
    profile = defaults_for(ROUWEI)
    assert (profile["steps"], profile["guidance_scale"], profile["refine_strength"]) == (28, 6, 0.4)
    assert profile["refine_supported"] and profile["negative_prompt_supported"]
    assert not profile["mu_supported"] and not profile["true_cfg_supported"]
    assert _profile_for(ROUWEI.upper()) is _profile_for(ROUWEI)
    for other in ("Minthy/RouWei-0.8-Vpred", "someone/RouWei-0.8", "Minthy/RouWei-0.7"):
        assert _profile_for(other) is _FALLBACK
        assert prompt_encoding.encoders_for(other) is None
    guide = guide_for(ROUWEI)
    assert guide["pixels"] == (1.0, 1.5) and guide["pixels_word"] == "recommended"
    assert guide["aspects"] is None and guide["resolutions"] is None
    helper = guide["tag_helper"]
    assert helper["vocabulary_cutoff"] == "2025-04-30" and helper["hybrid_prompt"]
    assert helper["artist_prefix"] == "by "
    assert helper["negative_prompt"] == "worst quality, low quality, watermark"
    assert helper["groups"][0] == {"label": "Quality", "tags": ["masterpiece", "best quality"]}
    assert not helper.get("recommended_suffix") and not helper["order"]
    assert not any(group["label"] == "Rating" for group in helper["groups"])


def test_rouwei_optional_quality_and_hybrid_prose_do_not_inherit_stricter_rules():
    for prompt in ("1girl, solo, smile", "A girl watches the evening sky."):
        findings = run(ROUWEI, prompt, negative_prompt="low quality")
        assert not any(f.rule in {"guide.style", "guide.quality", "guide.order"} for f in findings)
    assert prompt_rules.documented_order(ROUWEI) is None
    assert prompt_rules.reorder_tags(ROUWEI, "masterpiece, 1girl, smile", None) is None
    sources = prompt_sources.sources_for(ROUWEI)
    assert [e.id for e in sources.excerpts if e.order] == ["rouwei.artist_order"]
    assert "never recommend BREAK" in sources.caveat


def test_rouwei_artist_prefix_is_by_and_the_critic_reads_its_fix():
    prompt = "1girl, wlop, smile"
    (finding,) = only(run(ROUWEI, prompt, {"wlop": "artist"}), "guide.artist_prefix")
    assert finding.suggestion == "1girl, by wlop, smile"
    assert "puts by before" in finding.message
    assert not only(run(ROUWEI, finding.suggestion, {"wlop": "artist"}), "guide.artist_prefix")
    assert "by wlop" not in prompt_rules.lookup_items(ROUWEI, finding.suggestion)
    assert prompt_rules.artists_read(ROUWEI, finding.suggestion, ["1girl", "wlop", "smile"], ["wlop"])
    assert not prompt_rules.artists_read(
        ROUWEI, "A girl painted by wlop smiles.", ["wlop"], ["wlop"]
    )
    assert not prompt_rules.artists_read(
        ROUWEI, "A girl painted by @wlop smiles.", ["wlop"], ["wlop"]
    )
    # Lettering is never rewritten as an artist name.
    assert not only(run(ROUWEI, 'sign says "wlop"', {"wlop": "artist"}), "guide.artist_prefix")


def test_rouwei_prefixed_artist_chips_query_only_the_name(client, fake, monkeypatch):
    claim(monkeypatch, fake, ROUWEI)
    seen = []

    def fetch(path, params):
        seen.append(params)
        return [
            {"name": "wlop", "category": 1, "post_count": 100, "created_at": "2020-01-01"},
            {"name": "smile", "category": 0, "post_count": 100, "created_at": "2020-01-01"},
            {"name": "future", "category": 1, "post_count": 100, "created_at": "2025-05-01"},
        ]

    monkeypatch.setattr(main, "tag_vocabulary", tags.DanbooruTagVocabulary(fetch))
    response = client.post("/api/tags/validate", json={"tags": ["by wlop", "by smile", "by future", "best quality"]})
    assert response.status_code == 200, response.text
    artist, nonartist, too_new, quality = response.json()["results"]
    assert artist["tag"] == "by wlop" and artist["valid"] and artist["category"] == "artist"
    assert nonartist["valid"] is False and "not a Danbooru artist" in nonartist["reason"]
    assert too_new["valid"] is False and too_new["code"] == "after_cutoff"
    assert quality["valid"] and quality["source"] == "model"
    assert set(seen[0]["search[name_normalize]"].split(",")) == {"wlop", "smile", "future"}
    # A failure remains unchecked for the original, prefixed chip.
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary(offline=True))
    offline = client.post("/api/tags/validate", json={"tags": ["by wlop"]}).json()
    assert not offline["available"] and offline["results"][0]["valid"] is None
    assert offline["results"][0]["tag"] == "by wlop"


def test_rouwei_artist_suggestions_add_prefix_and_respect_a_typed_one(client, fake, monkeypatch):
    claim(monkeypatch, fake, ROUWEI)
    seen = []

    class Suggestions:
        def suggest(self, query, cutoff):
            seen.append((query, cutoff))
            return [
                {"text": "wlop", "canonical": "wlop", "category": "artist", "post_count": 10},
                {"text": "wlop style", "canonical": "wlop_style", "category": "general", "post_count": 5},
            ]

    monkeypatch.setattr(main, "tag_vocabulary", Suggestions())
    for query, expected in (("wlo", ["by wlop", "wlop style"]), ("by wlo", ["by wlop"])):
        response = client.get("/api/tags/suggest", params={"q": query})
        assert response.status_code == 200, response.text
        assert [item["text"] for item in response.json()["suggestions"]] == expected
    assert seen == [("wlo", "2025-04-30"), ("wlo", "2025-04-30")]


def test_rouwei_artist_prefix_is_not_part_of_the_tag_name_bound(client, fake, monkeypatch):
    claim(monkeypatch, fake, ROUWEI)
    name = "a" * tags.TAG_NAME_MAX
    vocabulary = Vocabulary({name: "artist"})
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
    results = main._lookup_tags(guide_for(ROUWEI)["tag_helper"], ["by " + name, "by " + name + "x"])
    assert results[0]["valid"] and results[0]["tag"] == "by " + name
    assert results[1]["valid"] is False and results[1]["reason"] == "longer than any Danbooru tag"
    assert vocabulary.calls[0][0] == [name]
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary(offline=True))
    offline = client.post("/api/tags/validate", json={"tags": ["by " + name, "by " + name + "x", "by masterpiece"]}).json()
    assert not offline["available"]
    assert [result["valid"] for result in offline["results"]] == [None, False, None]


def test_rouwei_analysis_keeps_prefix_and_reports_literal_break(client, fake, monkeypatch):
    claim(monkeypatch, fake, ROUWEI)
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary({"wlop": "artist"}))
    body = analyse(client, prompt="1girl, by wlop, BREAK, smile")
    assert not rules(body["findings"], "guide.artist_prefix")
    assert any("BREAK starts no new chunk" in f["message"] for f in body["findings"])
    assert [encoder["name"] for encoder in body["encoders"]] == ["CLIP-L", "CLIP-G"]
