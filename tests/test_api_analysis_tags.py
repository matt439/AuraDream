"""Tag lookup and chip order slots through prompt-analysis HTTP routes."""
from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError
import re

from backend import main, prompt_rules, tags
from backend.critic import build_context
from backend.model_defaults import guide_for
from backend.schemas import (
    GenerateRequest,
    PromptAnalysisRequest,
    PromptMeasureRequest,
    TAG_SLOT_PROMPT_MAX,
)

from tests.api_analysis_helpers import (
    ANIMA,
    ANIMAGINE,
    FIREFLY,
    FLUX,
    IN_ORDER,
    TAG_PROMPT,
    Vocabulary,
    analyse,
    claim,
    rules,
)
from tests.fakes import CutReply


KOHAKU = "KBlueLeaf/Kohaku-XL-Zeta"


def test_a_tag_checkpoint_orders_by_the_categories_the_chips_see(client, fake, monkeypatch):
    claim(monkeypatch, fake, ANIMAGINE)
    vocabulary = Vocabulary({"green hair": "general", FIREFLY: "character"})
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
    body = analyse(client, prompt=TAG_PROMPT)
    # Asked the way POST /api/tags/validate asks: the guide's cutoff, its
    # special tags and year form - and only about what a rule would read.
    helper = guide_for(ANIMAGINE)["tag_helper"]
    ((tags, special, cutoff, year),) = vocabulary.calls
    assert tags == ["green hair", FIREFLY]
    assert special == main._special_tags(helper) and "absurdres" in special
    assert (cutoff, year) == (helper["vocabulary_cutoff"], True)

    assert body["style"] == "tags"
    assert body["order"]["steps"] == helper["order"]
    assert body["reordered"] == IN_ORDER
    (order,) = rules(body["findings"], "guide.order")
    assert order["severity"] == "info" and order["suggestion"] == IN_ORDER
    assert TAG_PROMPT[order["start"]:order["end"]] == "green hair"


def test_an_unreachable_tag_index_costs_the_order_check_not_the_answer(client, fake, monkeypatch):
    claim(monkeypatch, fake, ANIMAGINE)
    vocabulary = Vocabulary(offline=True)
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
    body = analyse(client, prompt=TAG_PROMPT)
    assert len(vocabulary.calls) == 1
    assert rules(body["findings"], "guide.order") == [] and body["reordered"] is None
    assert body["order"] is not None and [e["name"] for e in body["encoders"]] == ["CLIP-L", "CLIP-G"]
    # ... and says what it cost. Placed from the guide's own groups alone, a
    # character after a general tag goes unflagged: without the detail, this
    # answer was the one an in-order prompt gets with the index reachable.
    assert body["order_checked"] is True
    assert body["categories_detail"] == (
        "the tag index was unreachable, so no tag has its Danbooru category"
    )
    monkeypatch.setattr(
        main, "tag_vocabulary", Vocabulary({"green hair": "general", FIREFLY: "character"})
    )
    in_order = analyse(client, prompt=IN_ORDER)
    assert in_order["order_checked"] is True and in_order["categories_detail"] is None
    assert rules(in_order["findings"], "guide.order") == [] and in_order["reordered"] is None

    def verdict(answer):
        return (
            answer["order_checked"], answer["categories_detail"], answer["reordered"],
            sorted(finding["rule"] for finding in answer["findings"]),
        )

    assert verdict(body) != verdict(in_order)


def test_a_broken_tag_index_reply_costs_the_order_check_not_the_answer(client, fake, monkeypatch):
    claim(monkeypatch, fake, ANIMAGINE)
    # A chunked reply cut off mid-body is http.client's error, not an OSError.
    monkeypatch.setattr(main, "tag_vocabulary", tags.DanbooruTagVocabulary())
    monkeypatch.setattr(tags, "urlopen", lambda request, timeout: CutReply())
    body = analyse(client, prompt=TAG_PROMPT)
    assert rules(body["findings"], "guide.order") == [] and body["reordered"] is None
    # An answer came, and was no use: not "unreachable".
    unusable = "the tag index gave no usable reply, so no tag has its Danbooru category"
    assert body["categories_detail"] == unusable
    # The chips read it the same way: unchecked, never an error.
    chips = client.post("/api/tags/validate", json={"tags": ["green hair"]})
    assert chips.status_code == 200 and chips.json()["available"] is False
    # A reply that parses into rows the lookup cannot read costs no more.
    monkeypatch.setattr(
        main, "tag_vocabulary", tags.DanbooruTagVocabulary(lambda path, params: ["not a row"])
    )
    body = analyse(client, prompt=TAG_PROMPT)
    assert rules(body["findings"], "guide.order") == [] and body["reordered"] is None
    assert body["categories_detail"] == unusable

    # Only a connection that never answered is the index unreachable; an
    # error status is an answer.
    def failing(error):
        def urlopen(request, timeout):
            raise error
        return urlopen

    for error, detail in [
        (HTTPError("https://danbooru.donmai.us", 503, "Service Unavailable", None, None), unusable),
        (URLError(ConnectionRefusedError(10061, "refused")), "the tag index was unreachable, "
         "so no tag has its Danbooru category"),
        (TimeoutError("timed out"), "the tag index was unreachable, "
         "so no tag has its Danbooru category"),
    ]:
        monkeypatch.setattr(main, "tag_vocabulary", tags.DanbooruTagVocabulary())
        monkeypatch.setattr(tags, "urlopen", failing(error))
        assert analyse(client, prompt=TAG_PROMPT)["categories_detail"] == detail, error


def test_the_tag_lookup_is_capped_at_the_chips_limit_and_skipped_for_prose(
    client, fake, monkeypatch
):
    claim(monkeypatch, fake, ANIMAGINE)
    vocabulary = Vocabulary()
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
    body = analyse(client, prompt=", ".join(f"tag{i}" for i in range(80)))
    ((tags, *_),) = vocabulary.calls
    assert len(tags) == main._MAX_TAG_ITEMS == 64
    # The 16 past the cap are placed by the guide's groups alone, and the
    # answer says so rather than passing them.
    assert body["order_checked"] is True and body["categories_detail"] == (
        "the tag index was asked about the first 64 of 80 tags only, so the rest have "
        "no Danbooru category"
    )
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary())
    at_the_cap = analyse(client, prompt=", ".join(f"tag{i}" for i in range(64)))
    assert at_the_cap["categories_detail"] is None
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
    # Prose is not a tag list to categorise, even in comma pieces the order
    # rule would ask about - the user's sentences are not sent to Danbooru -
    # and FLUX.1 has no tag index.
    prose = "A girl with green hair stands in the rain, looking at the viewer, soft light"
    assert prompt_rules.detect_style(prose) == "prose"
    assert prompt_rules.category_items(ANIMAGINE, prose)
    analyse(client, prompt=prose)
    claim(monkeypatch, fake, FLUX)
    analyse(client, prompt="1girl, green hair")
    assert len(vocabulary.calls) == 1


def chips(client, prompt, *, send_prompt=True):
    """POST /api/tags/validate as the tray sends it: the first 64 comma items,
    and the prompt they were split from."""
    tags = [part.strip() for part in prompt.split(",") if part.strip()][:64]
    body = {"tags": tags, **({"prompt": prompt} if send_prompt else {})}
    response = client.post("/api/tags/validate", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_the_chips_slots_are_checks_order(client, fake, monkeypatch):
    # Each chip's step of the documented order is the order rule's placement
    # with the categories a Check of the same prompt places with - never one
    # of the page's: the reorder Check offers puts the slots in order.
    claim(monkeypatch, fake, ANIMAGINE)
    vocabulary = Vocabulary({"green hair": "general", FIREFLY: "character"})
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
    body = chips(client, TAG_PROMPT)
    assert body["available"] is True
    assert body["order_steps"] == guide_for(ANIMAGINE)["tag_helper"]["order"]
    assert [(s["tag"], s["slot"]) for s in body["slots"]] == [
        ("1girl", "Person count"), ("green hair", "Details"), (FIREFLY, "Character"),
        ("masterpiece", "Quality"), ("high score", "Quality"), ("great score", "Quality"),
        ("absurdres", "Quality"),
    ]
    # Asked as the chips ask, then as Check asks - only what a rule reads.
    assert [call[0] for call in vocabulary.calls] == [
        [s["tag"] for s in body["slots"]], ["green hair", FIREFLY],
    ]
    check = analyse(client, prompt=TAG_PROMPT)
    assert check["reordered"] == IN_ORDER
    steps = body["order_steps"]
    ordered = [steps.index(s["slot"]) for s in chips(client, IN_ORDER)["slots"]]
    assert ordered == sorted(ordered)
    # Without the prompt - the Refine dialog's unknown-tag line - the
    # verdicts alone, and nothing asked beyond the chips.
    vocabulary.calls.clear()
    bare = chips(client, TAG_PROMPT, send_prompt=False)
    assert bare["slots"] is None and bare["order_steps"] is None
    assert len(vocabulary.calls) == 1


def test_a_chip_the_rules_cannot_place_has_no_slot(client, fake, monkeypatch):
    claim(monkeypatch, fake, ANIMAGINE)
    vocabulary = Vocabulary({"green hair": "general"})
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
    # An unknown tag, an @ name and lettering, each beside placed tags.
    body = chips(client, '1girl, zzkx oc, @wlop, "hello, world", green hair, masterpiece')
    assert [s["slot"] for s in body["slots"]] == [
        "Person count", None, None, None, None, "Details", "Quality",
    ]
    # Two items are an order, with the one lookup Check makes for what a
    # rule reads beside the chips' own.
    vocabulary.calls.clear()
    assert chips(client, "1girl, smile")["slots"] is not None
    assert [call[0] for call in vocabulary.calls] == [["1girl", "smile"], ["smile"]]
    # One item, or a list the rules stand down from: no order, no slots -
    # and no category lookup beside the chips' own for an order nothing reads.
    for prompt in (
        "masterpiece", "1girl\ngreen hair\nmasterpiece", "<x>, 1girl, masterpiece, smile",
    ):
        vocabulary.calls.clear()
        body = chips(client, prompt)
        assert body["slots"] is None and body["order_steps"] is None, prompt
        assert body["results"], prompt
        assert len(vocabulary.calls) == 1, prompt
    # Past the 64 the chips look up, a guide tag still has its slot: the
    # slots cover every comma item, not only those asked about.
    long = ", ".join(["1girl", *(f"tag{i}" for i in range(70)), "masterpiece"])
    body = chips(client, long)
    assert len(body["results"]) == 64 and len(body["slots"]) == 72
    assert body["slots"][-1] == {"tag": "masterpiece", "slot": "Quality"}
    # A prompt longer than a Check reads is no prompt here: verdicts, no
    # slots, never a refusal.
    over = "1girl, " + "x" * TAG_SLOT_PROMPT_MAX
    body = client.post("/api/tags/validate", json={"tags": ["1girl"], "prompt": over}).json()
    assert body["results"] and body["slots"] is None


# A hybrid prompt's sentence, longer than any tag: one comma item of 213
# characters refused the whole request (a 1-200 bound on every item), and the
# tray said Danbooru was unreachable, its guide tags unchecked and its order
# captions gone.
SENTENCE = (
    "a girl with long green hair standing alone in the rain under a red umbrella while neon "
    "lights from the shops reflect on the wet street and the crowd hurries past her in the "
    "cold blue evening dusk light of the city"
)


def test_an_item_longer_than_any_tag_is_unknown_without_asking(client, fake, monkeypatch):
    claim(monkeypatch, fake, KOHAKU)
    vocabulary = Vocabulary({"1girl": "general"})
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
    assert len(SENTENCE) > 200
    body = chips(client, f"1girl, {SENTENCE}, masterpiece, safe")
    assert body["available"] is True
    verdicts = {r["tag"]: r for r in body["results"]}
    assert verdicts[SENTENCE]["valid"] is False and verdicts[SENTENCE]["code"] == "not_found"
    assert verdicts[SENTENCE]["reason"] == "longer than any Danbooru tag"
    assert [r["tag"] for r in body["results"]] == ["1girl", SENTENCE, "masterpiece", "safe"]
    # Never asked of the index; the slots ride beside the verdicts.
    assert all(SENTENCE not in call[0] for call in vocabulary.calls)
    assert body["slots"] is not None
    # Unreachable, the guide's own tags are still the model's, and the long
    # item is still unknown: no index is needed to say so.
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary(offline=True))
    offline = chips(client, f"1girl, {SENTENCE}, masterpiece, safe")
    verdicts = {r["tag"]: r for r in offline["results"]}
    assert offline["available"] is False
    assert verdicts[SENTENCE]["valid"] is False and verdicts[SENTENCE]["code"] == "not_found"
    assert verdicts["masterpiece"]["valid"] is True
    # Only a blank item, or one longer than a prompt can be, is refused.
    for tag in ("  ", "x" * 4001):
        assert client.post("/api/tags/validate", json={"tags": [tag]}).status_code == 422


def test_the_pages_measured_length_is_every_requests(client):
    # tag-tokens.js measures the first PROMPT_MAX_CHARS code points: a prompt past
    # it was refused whole, and the line said "cut unknown".
    source = (Path(__file__).resolve().parents[1] / "web" / "tag-tokens.js").read_text(encoding="utf-8")
    match = re.search(r"^const PROMPT_MAX_CHARS = (\d+);$", source, re.M)
    assert match
    for model in (PromptMeasureRequest, GenerateRequest):
        field = model.model_fields["prompt"]
        assert [m.max_length for m in field.metadata if hasattr(m, "max_length")] == [int(match.group(1))]
    measured = client.post("/api/prompt/measure", json={"prompt": "a" * int(match.group(1))})
    assert measured.status_code == 200


def test_an_unreachable_index_costs_the_slots_one_wait(client, fake, monkeypatch):
    # Offline, Check's categories are None too: the guide's own groups are
    # placed, the rest have no slot - and the index is not asked twice.
    claim(monkeypatch, fake, ANIMAGINE)
    vocabulary = Vocabulary(offline=True)
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
    body = chips(client, TAG_PROMPT)
    assert body["available"] is False and len(vocabulary.calls) == 1
    assert [s["slot"] for s in body["slots"]] == [
        "Person count", None, None, "Quality", "Quality", "Quality", "Quality",
    ]


def test_a_failure_in_the_slots_never_costs_the_verdicts(client, fake, monkeypatch):
    # The slots are advisory, as the lookup is: a fault placing them leaves
    # the chips' verdicts answered, and the page's chips checked.
    claim(monkeypatch, fake, ANIMAGINE)
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary({"green hair": "general"}))

    def broken(*args, **kwargs):
        raise ValueError("placement fault")

    monkeypatch.setattr(prompt_rules, "order_slots", broken)
    body = chips(client, TAG_PROMPT)
    assert body["available"] is True
    assert [row["tag"] for row in body["results"]] == [
        part.strip() for part in TAG_PROMPT.split(",")
    ]
    assert body["slots"] is None and body["order_steps"] is None


def test_no_slots_where_the_guide_states_no_order(client, fake, monkeypatch):
    # Every Danbooru-vocabulary guide today documents an order; one that did
    # not would get verdicts and no slots. Anima reads Danbooru-style tags
    # and states none.
    claim(monkeypatch, fake, ANIMA)
    assert prompt_rules.documented_order(ANIMA) is None
    monkeypatch.setattr(main, "_tag_helper_config", lambda: main._BARE_DANBOORU)
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary({"green hair": "general"}))
    body = chips(client, "1girl, green hair, masterpiece")
    assert body["results"] and body["slots"] is None and body["order_steps"] is None


def test_the_order_is_partly_checked_exactly_where_an_item_went_unplaced(
    client, fake, monkeypatch
):
    # One rule for "partly checked": some item the order rule reads kept
    # every slot a category could give it, where that decides the order
    # (`prompt_rules.places_every_item` false, placing with the categories
    # the rules had) - the predicate the critic holds a reorder of its own
    # to. `order_detail` says so, with the items and why each has no
    # category. A tag the index answered for and did not know, and a prose
    # prompt's clauses, which are never looked up, left `categories_detail`
    # empty, and the order read as checked while the critic's block said
    # Check could not place every tag.
    claim(monkeypatch, fake, ANIMAGINE)
    vocabulary = Vocabulary({"green hair": "general", FIREFLY: "character"})
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
    unknown = (
        'the tag index has no usable Danbooru category for 1 tag, so it is not placed: "zzkx oc"'
    )
    # An OC's name, a trigger word, a typo after a general tag: out of order
    # were it a character, in it were it a general tag - nothing is flagged,
    # and that is no pass.
    body = analyse(client, prompt="green hair, zzkx oc")
    assert body["order_checked"] is True and body["categories_detail"] is None
    assert rules(body["findings"], "guide.order") == [] and body["reordered"] is None
    assert body["order_detail"] == unknown
    # Every tag placed: checked, not partly. So too where an unplaced item's
    # slot decides nothing - ahead of a general tag, between 1girl and
    # masterpiece, it is in order whatever it is; ahead of 1girl, out of it
    # whatever it is - and for one item alone, which no order can be wrong
    # about: the lane first said "Partly checked" of all of these. Nor does a
    # tag written twice make it so: its copies take one slot, whatever it is.
    for prompt in (
        TAG_PROMPT, IN_ORDER, "zzkx oc, green hair", "1girl, zzkx oc, masterpiece",
        "zzkx oc, masterpiece", "zzkx oc", "a girl with green hair standing in the rain",
        '1girl, "OPEN", masterpiece', "1girl, @wlop, masterpiece",
        "1girl, zzkx oc, zzkx oc, masterpiece", "zzkx oc, zzkx oc",
    ):
        body = analyse(client, prompt=prompt)
        assert body["order_checked"] is True and body["order_detail"] is None, prompt
    # Beside another item it still decides the order, and is named once.
    body = analyse(client, prompt="green hair, zzkx oc, zzkx oc")
    assert body["order_detail"] == unknown
    body = analyse(client, prompt="zzkx oc, 1girl")
    assert rules(body["findings"], "guide.order") != [] and body["reordered"] == "1girl, zzkx oc"
    assert body["order_checked"] is True and body["order_detail"] is None
    # Which slots an unplaced item keeps is the guide's: Kohaku's meta tags
    # come after quality, so there it may sit out of order before masterpiece.
    claim(monkeypatch, fake, KOHAKU)
    body = analyse(client, prompt="1girl, zzkx oc, masterpiece")
    assert body["order_checked"] is True and body["order_detail"] == unknown
    claim(monkeypatch, fake, ANIMAGINE)
    # The index knows a tag and withholds its category - deprecated, or
    # created past the checkpoint's cutoff: not "unknown", which the chips,
    # reading the same rows, would contradict.
    rows = {
        "green_hair": {"name": "green_hair", "category": 0, "post_count": 5,
                       "created_at": "2010-01-01"},
        "old_tag": {"name": "old_tag", "category": 0, "post_count": 5,
                    "created_at": "2010-01-01", "is_deprecated": True},
        "new_hero_(game)": {"name": "new_hero_(game)", "category": 4, "post_count": 5,
                            "created_at": "2099-01-01"},
    }
    monkeypatch.setattr(main, "tag_vocabulary", tags.DanbooruTagVocabulary(
        fetch_json=lambda path, params: [
            rows[name] for name in params["search[name_normalize]"].split(",") if name in rows
        ]
    ))
    withheld = analyse(client, prompt="green hair, old tag, new hero \\(game\\)")
    assert withheld["categories_detail"] is None
    assert withheld["order_detail"] == (
        "the tag index has no usable Danbooru category for 2 tags, so they are not placed: "
        '"old tag", "new hero \\(game\\)"'
    )
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
    # Prose is never looked up, and a clause that is not a whole sentence is
    # placed by nothing; whole sentences sit in the general slot.
    vocabulary.calls.clear()
    prose = "A girl with green hair stands in the rain, looking at the viewer, soft light"
    body = analyse(client, prompt=prose)
    assert vocabulary.calls == [] and body["style"] == "prose"
    assert body["order_checked"] is True and body["categories_detail"] is None
    assert body["order_detail"] == (
        "the tag index is not asked about a prose prompt, so 3 clauses that are not whole "
        'sentences are not placed: "A girl with green hair stands in the rain", '
        '"looking at the viewer", "soft light"'
    )
    sentences = analyse(client, prompt="A girl stands in the rain. She smiles softly.")
    assert sentences["order_checked"] is True and sentences["order_detail"] is None
    # An item never asked about - lettering, an @ name - beside a tag the
    # index did not know: each under its own reason.
    body = analyse(client, prompt='1girl, "OPEN", zzkx oc, @wlop, masterpiece')
    assert body["order_detail"] == (
        f"{unknown}; the tag index is not asked about 2 items, so they are not placed: "
        '""OPEN"", "@wlop"'
    )
    # The lookup's own reasons lead, with the items they cost.
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary(offline=True))
    body = analyse(client, prompt=TAG_PROMPT)
    assert body["order_detail"] == (
        "the tag index was unreachable, so no tag has its Danbooru category, and 2 tags are "
        f'not placed: "green hair", "{FIREFLY}"'
    )
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary())
    body = analyse(client, prompt=", ".join(f"tag{i}" for i in range(70)))
    assert body["order_detail"] == (
        "the tag index was asked about the first 64 of 70 tags only, so the rest have no "
        'Danbooru category, and 6 tags are not placed: "tag64", "tag65", "tag66" and 3 more; '
        "the tag index has no usable Danbooru category for 64 tags, so they are not placed: "
        '"tag0", "tag1", "tag2" and 61 more'
    )
    # The lookup's detail alone is no partial order: past the cap sits only a
    # phrase the rules place as prose, and every tag before it is known.
    known = [f"tag{i}" for i in range(64)]
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary(dict.fromkeys(known, "general")))
    body = analyse(client, prompt=", ".join([*known, "looking at the viewer"]))
    assert body["categories_detail"].startswith("the tag index was asked about the first 64 of 65")
    assert body["order_checked"] is True and body["order_detail"] is None
    # Where the guide states no order, there is none to have checked partly:
    # the artist check's note is `categories_detail` alone.
    claim(monkeypatch, fake, ANIMA)
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary(offline=True))
    body = analyse(client, prompt="1girl, wlop, smile")
    assert body["order"] is None and body["order_detail"] is None
    assert body["categories_detail"].startswith("the tag index was unreachable")
    claim(monkeypatch, fake, FLUX)
    body = analyse(client, prompt="1girl, zzkx oc, green hair")
    assert body["order_checked"] is False and body["order_detail"] is None


def test_the_page_and_the_critic_read_partial_placement_alike(client, fake, monkeypatch):
    # The critic decides whether a reorder of its own may stand from the same
    # placement (`CriticContext.placed_partly`): wherever the analysis says
    # the order was partly checked, the critic treats it as partly placed, and
    # nowhere else.
    claim(monkeypatch, fake, ANIMAGINE)
    prompts = [
        TAG_PROMPT, IN_ORDER, "zzkx oc, green hair", "green hair, masterpiece, 1girl",
        "A girl with green hair stands in the rain, looking at the viewer, soft light",
        "A girl stands in the rain. She smiles softly.", '1girl, "OPEN", masterpiece',
        ", ".join(f"tag{i}" for i in range(70)), "1girl\ngreen hair", "",
    ]
    # A lone item and ones whose slot decides nothing, beside ones it does.
    prompts += [
        "zzkx oc", "green hair, zzkx oc", "1girl, zzkx oc, masterpiece", "zzkx oc, 1girl",
        "a girl with green hair standing in the rain", "zzkx oc, yyqv oc",
        "1girl, zzkx oc, zzkx oc, masterpiece", "green hair, zzkx oc, zzkx oc",
    ]
    for model_id in (ANIMAGINE, KOHAKU):
        claim(monkeypatch, fake, model_id)
        for vocabulary in (
            Vocabulary({"green hair": "general", FIREFLY: "character"}), Vocabulary(offline=True),
        ):
            monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
            for prompt in prompts:
                request = PromptAnalysisRequest(prompt=prompt or "a lighthouse")
                analysis = main._analyse(request)
                context = build_context(model_id, request, request, analysis)
                assert context.placed_partly == (analysis.order_detail is not None), prompt
                assert analysis.order_detail is None or analysis.order_checked, prompt
