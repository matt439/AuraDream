"""Guide order, artists, and XML rewrites through prompt-analysis routes."""
from __future__ import annotations

import json

from backend import main, prompt_rules, tag_lookup
from backend.model_defaults import guide_for

from tests.api_analysis_helpers import (
    ANIMA,
    ANIMAGINE,
    FIREFLY,
    FLUX,
    TAG_PROMPT,
    Vocabulary,
    analyse,
    claim,
    rules,
)
from tests.fakes import FakeCritic


def test_the_order_says_whether_it_was_checked_and_what_the_authors_give(
    client, fake, monkeypatch
):
    # FLUX.1's order is documented but no rule reads a sentence's order: the
    # page must not let "nothing to flag" pass for a checked one.
    claim(monkeypatch, fake, FLUX)
    body = analyse(client, prompt="Golden hour light over misty mountains, a lone hiker on a ridge")
    assert body["order"] is not None and body["order_checked"] is False
    assert body["author_orders"] == []
    assert body["categories_detail"] is None
    # A tag checkpoint's comma items are placed - wholly with the tag index's
    # categories, partly without them, which is said beside the flag.
    claim(monkeypatch, fake, ANIMAGINE)
    monkeypatch.setattr(
        tag_lookup, "tag_vocabulary", Vocabulary({"green hair": "general", FIREFLY: "character"})
    )
    body = analyse(client, prompt=TAG_PROMPT)
    assert body["order_checked"] is True and body["author_orders"] == []
    assert body["categories_detail"] is None
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", Vocabulary(offline=True))
    body = analyse(client, prompt=TAG_PROMPT)
    assert body["order_checked"] is True and body["categories_detail"] is not None
    # Anima's guide states no order for the app to check, but its card gives
    # one: shown as the authors', with its source.
    claim(monkeypatch, fake, ANIMA)
    body = analyse(client, prompt="1girl, smile")
    assert body["order"] is None and body["order_checked"] is False
    (card,) = body["author_orders"]
    assert card["text"].startswith("[quality/meta/year/safety tags] [1girl/1boy/1other etc]")
    assert card["url"] == "https://huggingface.co/circlestone-labs/Anima"
    assert card["note"] == "the card's tag order"
    claim(monkeypatch, fake, "duongve/NetaYume-Lumina-Image-2.0")
    body = analyse(client, prompt="1girl, smile")
    assert [o["note"].split("; ")[-1] for o in body["author_orders"]] == [
        "its standard sequence for the tags", "its order for the natural-language part",
    ]


def test_the_style_reported_is_the_one_the_rules_applied(client, fake, monkeypatch):
    claim(monkeypatch, fake, ANIMAGINE)
    prompt = "1girl, solo, the legend of zelda, princess zelda, smile"
    vocabulary = Vocabulary({"the legend of zelda": "copyright", "princess zelda": "character"})
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", vocabulary)
    body = analyse(client, prompt=prompt)
    # Its shape alone reads as partly prose; with the index's categories the
    # rules read it as the tag list it is, and so does the report.
    assert body["style"] == "tags" and rules(body["findings"], "guide.style") == []


def test_anima_asks_the_tag_index_which_items_are_artists(client, fake, monkeypatch):
    # Anima's guide has no tag helper, but "put @ before an artist name" needs
    # an artist's category: the index is asked with no cutoff of its own.
    claim(monkeypatch, fake, ANIMA)
    vocabulary = Vocabulary({"wlop": "artist", "smile": "general"})
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", vocabulary)
    prompt = "1girl, wlop, smile, @big chungus"
    body = analyse(client, prompt=prompt)
    ((asked, special, cutoff, year),) = vocabulary.calls
    assert asked == ["1girl", "wlop", "smile"] and (special, cutoff, year) == (set(), None, False)
    (artist,) = rules(body["findings"], "guide.artist_prefix")
    assert artist["suggestion"] == "1girl, @wlop, smile, @big chungus"
    assert prompt[artist["start"]:artist["end"]] == "wlop"
    assert body["categories_detail"] is None
    assert body["categories_unread"] is False
    # Offline, the rule stands down, and the answer says why - there is no
    # order here to say it beside - and that no item was checked at all.
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", Vocabulary(offline=True))
    offline = analyse(client, prompt="1girl, wlop, smile")
    assert rules(offline["findings"], "guide.artist_prefix") == []
    assert offline["order"] is None and offline["categories_detail"].startswith(
        "the tag index was unreachable"
    )
    assert offline["categories_unread"] is True
    # Prose: only its tag-shaped items are asked about - an artist's name is
    # a whole comma item, and a clause is none - never its clauses.
    quiet = Vocabulary()
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", quiet)
    prose = "A girl painted by wlop stands in the rain, smiling at the viewer, soft light"
    assert prompt_rules.detect_style(prose) == "prose"
    analyse(client, prompt=prose)
    assert [asked for asked, *_ in quiet.calls] == [["soft light"]]


def test_anima_checks_the_artists_of_tags_ahead_of_sentences(client, fake, monkeypatch):
    # The card's own shape - its positive prefix and tags, then sentences
    # ("Use tags, natural language, or both") - reads as prose. The artist
    # check once skipped it without a word: "nothing to flag" over an artist
    # written without its @, the index never asked.
    claim(monkeypatch, fake, ANIMA)
    vocabulary = Vocabulary({"wlop": "artist", "1girl": "general"})
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", vocabulary)
    prompt = (
        "masterpiece, best quality, score_7, safe, 1girl, wlop, A girl stands in the rain "
        "under a red umbrella. She looks up at the grey sky."
    )
    assert prompt_rules.detect_style(prompt) == "prose"
    body = analyse(client, prompt=prompt)
    ((asked, *_),) = vocabulary.calls
    assert asked == ["masterpiece", "best quality", "score_7", "safe", "1girl", "wlop"]
    (artist,) = rules(body["findings"], "guide.artist_prefix")
    assert prompt[artist["start"]:artist["end"]] == "wlop"
    assert artist["suggestion"] == prompt.replace(", wlop,", ", @wlop,")
    assert artist["message"].startswith("wlop is an artist in Danbooru's index")
    applied = analyse(client, prompt=artist["suggestion"])
    assert rules(applied["findings"], "guide.artist_prefix") == []
    # Offline, the panel is told no item was checked, not handed a clean bill.
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", Vocabulary(offline=True))
    offline = analyse(client, prompt=prompt)
    assert rules(offline["findings"], "guide.artist_prefix") == []
    assert offline["categories_detail"].startswith("the tag index was unreachable")
    assert offline["categories_unread"] is True
    # A prose prompt with no tag-shaped item asks nothing and says nothing.
    quiet = Vocabulary()
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", quiet)
    sentences = "A girl stands in the rain under a red umbrella. She looks up at the grey sky."
    body = analyse(client, prompt=sentences)
    assert quiet.calls == [] and body["categories_detail"] is None


def test_several_artists_are_named_together(client, fake, monkeypatch):
    claim(monkeypatch, fake, ANIMA)
    vocabulary = Vocabulary({"wlop": "artist", "kantoku": "artist", "smile": "general"})
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", vocabulary)
    body = analyse(client, prompt="1girl, wlop, kantoku, smile")
    (artist,) = rules(body["findings"], "guide.artist_prefix")
    assert artist["message"] == (
        "wlop and kantoku are artists in Danbooru's index; the guide puts @ before an "
        "artist name."
    )
    assert artist["suggestion"] == "1girl, @wlop, @kantoku, smile"
    # One artist written twice is named once, as first written.
    (artist,) = rules(analyse(client, prompt="1girl, wlop, kantoku, WLOP, smile")["findings"],
                      "guide.artist_prefix")
    assert artist["message"].startswith("wlop and kantoku are artists in Danbooru's index;")
    assert artist["suggestion"] == "1girl, @wlop, @kantoku, @WLOP, smile"


def _rewriting(monkeypatch, rewrite: str) -> None:
    """The fake critic answering with `rewrite` as its rewrite."""
    answer = FakeCritic.answer

    def rewritten(text, ordered):
        raw = json.loads(answer(text, ordered))
        raw["rewrite"] = rewrite
        return json.dumps(raw)

    monkeypatch.setattr(FakeCritic, "answer", staticmethod(rewritten))


def test_a_rewrite_is_checked_for_artists_with_the_analysis_own_answer(
    client, fake, monkeypatch
):
    # The rewrite's check asks the tag index nothing of its own: it reads the
    # artists the critique's analysis was told about, and says whether those
    # answer for every item of the rewrite's that a check would ask about.
    claim(monkeypatch, fake, ANIMA)
    vocabulary = Vocabulary(
        {"wlop": "artist", "kantoku": "artist", "1girl": "general", "smile": "general"}
    )
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", vocabulary)

    def critique(rewrite):
        _rewriting(monkeypatch, rewrite)
        calls = len(vocabulary.calls)
        response = client.post("/api/prompt/critique", json={"prompt": "1girl, wlop, smile"})
        assert response.status_code == 200, response.text
        # One lookup, the analysis's; none for the rewrite.
        assert len(vocabulary.calls) == calls + 1
        return response.json()

    # The @ the guide asks for, added: fixed, and the page may say so.
    body = critique("1girl, @wlop, smile")
    assert rules(body["rewrite_findings"], "guide.artist_prefix") == []
    assert body["rewrite_artists_checked"] is True
    # Left out: flagged on the rewrite too, from the same answer.
    body = critique("smile, wlop, 1girl")
    (artist,) = rules(body["rewrite_findings"], "guide.artist_prefix")
    assert artist["suggestion"] == "smile, @wlop, 1girl"
    assert body["rewrite_artists_checked"] is True
    # An artist the analysis never asked about is not called checked.
    body = critique("1girl, @wlop, kantoku, smile")
    assert rules(body["rewrite_findings"], "guide.artist_prefix") == []
    assert body["rewrite_artists_checked"] is False
    # Nor is the artist kept with no @ where the rule reads no name - in a
    # clause, or a sentence - which the page would otherwise call fixed.
    for kept in (
        "1girl, smile, a painting in the style of wlop with soft light",
        "A girl painted by wlop smiles in the rain.",
    ):
        body = critique(kept)
        assert rules(body["rewrite_findings"], "guide.artist_prefix") == []
        assert body["rewrite_artists_checked"] is False, kept
    # Nor is anything where the analysis's lookup gave no answer.
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", Vocabulary(offline=True))
    _rewriting(monkeypatch, "1girl, @wlop, smile")
    body = client.post("/api/prompt/critique", json={"prompt": "1girl, wlop, smile"}).json()
    assert body["rewrite_artists_checked"] is False
    # And never on a guide whose rules read no artist.
    claim(monkeypatch, fake, FLUX)
    _rewriting(monkeypatch, "A lighthouse in thick fog")
    body = client.post("/api/prompt/critique", json={"prompt": "A lighthouse in fog"}).json()
    assert body["rewrite_artists_checked"] is False


def test_newbies_blocks_keep_their_documented_order_through_the_critic(
    client, fake, monkeypatch
):
    # The fake critic's reorder puts the last comma item first: here the
    # scene block ahead of the character, which the guide's order reverses.
    newbie = "NewBie-AI/NewBie-image-Exp0.1"
    claim(monkeypatch, fake, newbie)
    prompt = (
        "<character_1><n>miku</n><gender>1girl</gender></character_1>, "
        "<general_tags><style>anime</style></general_tags>"
    )
    body = client.post("/api/prompt/critique", json={"prompt": prompt}).json()
    assert body["reordered"] is None and body["reordered_kind"] is None
    assert body["set_aside"] == [
        "The critic's reordered version was left out: it puts <general_tags> ahead of a "
        "character block, against the documented order."
    ]
    # The authors' own example, whose comma items the fake rotates across
    # its blocks: refused for what it did, which changed no word.
    example = guide_for(newbie)["examples"][0]
    response = client.post("/api/prompt/critique", json={"prompt": example})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["reordered"] is None
    assert (
        "The critic's reordered version was left out: it moved words between XML blocks, where "
        "only whole blocks move, so no attribute changes owner; a reordering of your own can go "
        "under Compare."
    ) in body["set_aside"]
