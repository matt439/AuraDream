"""Critic grounding cites each guide and its authors accurately."""
from __future__ import annotations

import dataclasses
import re

import pytest

from backend import guides, prompt_encoding, prompt_sources
from backend.model_defaults import _profile_for

from tests.critic_helpers import (
    KLEIN,
    _catalogue_repos,
    _context,
)


# --- The grounding -----------------------------------------------------------


def test_every_guide_constant_is_grounded():
    assert set(prompt_sources.GROUNDING) == set(prompt_sources.guide_constants())


def test_aesthetic_critic_gets_its_quality_advice_instead_of_the_family_score_prefix():
    from backend.backends.anima_loading import AESTHETIC_MODEL_ID

    sources = prompt_sources.sources_for(AESTHETIC_MODEL_ID)
    assert sources.guide is guides.ANIMA_AESTHETIC_GUIDE
    excerpts = {excerpt.id: excerpt for excerpt in sources.excerpts}
    assert "anima.prefix" not in excerpts
    assert "anima.aesthetic_quality" in excerpts and "anima.aesthetic_scores" in excerpts
    assert "anima.order" in excerpts and "anima.artist" in excerpts
    assert "score_7" not in " ".join(excerpt.text for excerpt in sources.excerpts)
    assert all(excerpt.url == sources.url for excerpt in sources.excerpts)
    # The exception must reach the actual critic context, with Turbo's
    # existing family prefix remaining available for Turbo.
    context = _context(AESTHETIC_MODEL_ID, ordered=False)
    cited = " ".join(citation.text for citation in context.citations.values())
    assert "not using score_* tags in both the positive and negative prompt" in cited
    assert "does not need positive quality tags" in cited
    assert "score_7" not in cited and "distillation" not in cited
    assert "anima.prefix" in {
        excerpt.id for excerpt in prompt_sources.sources_for("circlestone-labs/Anima").excerpts
    }


@pytest.mark.parametrize("model_id", _catalogue_repos())
def test_every_catalogue_checkpoint_resolves_to_its_sources(model_id):
    sources = prompt_sources.sources_for(model_id)
    assert sources.guide is _profile_for(model_id)["guide"]
    assert sources.guide_name != "FALLBACK_GUIDE"
    assert sources.sentences
    # The guide's page; each sentence's own link is held by
    # test_a_sentence_links_the_page_its_rule_findings_do.
    assert sources.url == sources.guide["analysis"]["url"]
    assert sources.url.startswith("https://")
    assert all(sentence.url.startswith("https://") for sentence in sources.sentences)
    assert sources.excerpts


def _every_guide() -> dict[str, str]:
    """A checkpoint for each guide constant: the catalogue's, the two
    undistilled ones outside it, and an unknown id for the fallback."""
    ids = [*_catalogue_repos(), "Tongyi-MAI/Z-Image", "baidu/ERNIE-Image", "nobody/whatever"]
    found = {prompt_sources.sources_for(model_id).guide_name: model_id for model_id in ids}
    assert set(found) == set(prompt_sources.GROUNDING)
    return found


def _rule_entries(guide) -> list[dict]:
    """The guide's `analysis` entries that cite its prose."""
    return [
        entry
        for value in (guide.get("analysis") or {}).values()
        for entry in (value if isinstance(value, list) else [value])
        if isinstance(entry, dict) and entry.get("source") != "author"
    ]


def _rule_pages(sources) -> dict[str, set[str]]:
    """Per G sentence, the url of each rule entry citing its words - the url
    that entry's finding links."""
    analysis = sources.guide.get("analysis") or {}
    pages: dict[str, set[str]] = {}
    for sentence in sources.sentences:
        for entry in _rule_entries(sources.guide):
            cite = entry["cite"]
            if cite in sentence.text or sentence.text in cite:
                pages.setdefault(sentence.text, set()).add(entry.get("url") or analysis.get("url"))
    return pages


def test_a_sentence_links_the_page_its_rule_findings_do():
    # A rule finding and a critic cite of the same words open the same page:
    # on FLUX.1 the negation sentence is BFL's technical page in both. So no
    # sentence carries the words of two pages - FLUX.1's opening once folded
    # BFL's basics and building pages into one, and a critic's order issue
    # citing it opened the page the order rule finding did not.
    for name, model_id in _every_guide().items():
        sources = prompt_sources.sources_for(model_id)
        own = dict(prompt_sources.GROUNDING[name].sentence_urls)
        pages = _rule_pages(sources)
        for sentence in sources.sentences:
            cited = pages.get(sentence.text)
            if not cited:
                assert sentence.url == own.get(sentence.text, sources.url), (name, sentence.text)
                continue
            # A page named by hand would point the two layers apart.
            assert sentence.text not in own, (name, sentence.text)
            assert cited == {sentence.url}, (name, sentence.text, cited)
    flux = prompt_sources.sources_for("black-forest-labs/FLUX.1-dev")
    negation = next(s for s in flux.sentences if "rather than what you do not" in s.text)
    assert negation.url == "https://docs.bfl.ai/guides/prompting_unified_technical"


_BASICS = "https://docs.bfl.ai/guides/prompting_unified_basics"


_BUILDING = "https://docs.bfl.ai/guides/prompting_unified_building"


_TECHNICAL = "https://docs.bfl.ai/guides/prompting_unified_technical"


_KREA2_GUIDE = "https://github.com/krea-ai/krea-2/blob/main/docs/prompting.md"


_KREA2_EXPANSION = "https://github.com/krea-ai/krea-2/blob/main/docs/expansion.txt"


_KREA2_README = "https://github.com/krea-ai/krea-2"


_KREA2_REPORT = "https://www.krea.ai/blog/krea-2-technical-report"


_ILLUSTRIOUS_REPORT = "https://arxiv.org/html/2409.19946v1"


_ILLUSTRIOUS_CARD = "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0"


_ILLUSTRIOUS_RELEASE = "https://civarchive.com/models/1369089?modelVersionId=1546777"


_ERNIE_CARD = "https://huggingface.co/baidu/ERNIE-Image-Turbo"


_ERNIE_REPORT = "https://arxiv.org/abs/2605.25347"


_TOKENIZER_CONFIG = "https://huggingface.co/baidu/{}/blob/main/tokenizer/tokenizer_config.json"


_NEWBIE_CARD = (
    "https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1/tree/"
    "4f7825a1a7e1ed5c1882a52e3c51d1119a43e298"
)


_NEWBIE_CODE = (
    "https://github.com/NewBieAI-Lab/NewBie-image-Exp0.1/blob/"
    "709e9a482fb2b861ad6aa50dcafb0198cad66c1d/finetune.py"
)


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        (
            "black-forest-labs/FLUX.1-dev",
            [
                ("Plain descriptive sentences, n", _BASICS),
                ("Subject first, then setting, s", _BUILDING),
                ("30-80 words suits most images;", _BUILDING),
                ("Put any text to render in quot", _BASICS),
                ("There is no negative prompt un", _TECHNICAL),
            ],
        ),
        (
            "black-forest-labs/FLUX.1-Krea-dev",
            [
                ("FLUX.1 prompting - descriptive", _BASICS),
                ("Put the subject first, in 30-8", _BUILDING),
                ("There is no negative prompt wi", _TECHNICAL),
                ("This checkpoint is tuned for p", "https://bfl.ai/blog/flux-1-krea-dev"),
                ("Camera settings - lens, framin", _BUILDING),
                ("Krea recommends 28-32 steps an", "https://github.com/krea-ai/flux-krea"),
            ],
        ),
        (
            "black-forest-labs/FLUX.1-schnell",
            [
                ("FLUX.1 prompting - descriptive", _BASICS),
                ("Put the subject first.", _BUILDING),
                ("schnell's T5 encoder reads up ", _BASICS),
                ("Guidance is off on schnell, so", _TECHNICAL),
            ],
        ),
        (
            "krea/Krea-2-Turbo",
            [
                ("Natural language.", _KREA2_GUIDE),
                ("One cohesive paragraph.", _KREA2_EXPANSION),
                ("Krea's guide: long, detailed p", _KREA2_GUIDE),
                ("Krea's report says the trainin", _KREA2_REPORT),
                ("Put words to render in quotati", _KREA2_GUIDE),
                ("Text rendering was a reward in", _KREA2_REPORT),
                ("The Qwen3-VL encoder reads 512", _KREA2_REPORT),
                ("Turbo runs guidance-free at CF", _KREA2_README),
            ],
        ),
        (
            "krea/Krea-2-Raw",
            [
                ("Natural language.", _KREA2_GUIDE),
                ("One cohesive paragraph.", _KREA2_EXPANSION),
                ("Krea's guide: long, detailed p", _KREA2_GUIDE),
                ("Krea's report says the trainin", _KREA2_REPORT),
                ("Put words to render in quotati", _KREA2_GUIDE),
                ("Text rendering was a reward in", _KREA2_REPORT),
                ("The Qwen3-VL encoder reads 512", _KREA2_REPORT),
                ("Raw is the undistilled checkpo", _KREA2_README),
                ("It wants about 52 steps at CFG", _KREA2_README),
            ],
        ),
        (
            "OnomaAIResearch/Illustrious-XL-v2.0",
            [
                ("Illustrious XL v2.0 is an untu", _ILLUSTRIOUS_RELEASE),
                ("Add one composition tag (upper", _ILLUSTRIOUS_CARD),
                ("Tag lists and natural-language", _ILLUSTRIOUS_RELEASE),
                ("The training order is person c", _ILLUSTRIOUS_REPORT),
                ("The two CLIP encoders each sto", _ILLUSTRIOUS_REPORT),
                ("The rating slot takes Danbooru", _ILLUSTRIOUS_REPORT),
                ("End a tag prompt with one qual", _ILLUSTRIOUS_REPORT),
                ("Every example the authors publ", _ILLUSTRIOUS_CARD),
                ("Their v0.1 card runs both of i", _ILLUSTRIOUS_CARD),
                ("Their v2.0 release example use", _ILLUSTRIOUS_RELEASE),
                ("A negative prompt applies at C", _ILLUSTRIOUS_REPORT),
            ],
        ),
        (
            "baidu/ERNIE-Image-Turbo",
            [
                ("A content-rich, detailed visua", _ERNIE_CARD),
                ("Dense and layout-sensitive tex", _ERNIE_CARD),
                ("Baidu's report names Ministral", _ERNIE_REPORT),
                ("The tokenizer config caps a pr", _TOKENIZER_CONFIG.format("ERNIE-Image-Turbo")),
                ("Turbo runs at guidance 1, wher", _ERNIE_CARD),
            ],
        ),
        (
            "baidu/ERNIE-Image",
            [
                ("A content-rich, detailed visua", _ERNIE_CARD),
                ("Dense and layout-sensitive tex", _ERNIE_CARD),
                ("Baidu's report names Ministral", _ERNIE_REPORT),
                ("The tokenizer config caps a pr", _TOKENIZER_CONFIG.format("ERNIE-Image")),
                ("The undistilled ERNIE-Image sa", "https://huggingface.co/baidu/ERNIE-Image"),
            ],
        ),
        (
            "NewBie-AI/NewBie-image-Exp0.1",
            [
                ("NewBie image Exp0.1 is an expe", _NEWBIE_CARD),
                ("It accepts natural language, t", _NEWBIE_CARD),
                ("The authors recommend XML for ", _NEWBIE_CARD),
                ("Use a separate <character_1>, ", _NEWBIE_CARD),
                ("Gemma reads up to 512 tokens a", _NEWBIE_CODE),
            ],
        ),
    ],
)
def test_each_sentence_links_the_page_that_says_it(model_id, expected):
    # Where a guide's prose draws on more than the rules' page - BFL's three
    # pages, Krea's guide, expansion prompt, README and report, Illustrious's
    # report, v0.1 card and v2.0 release post (through its archive), ERNIE's
    # card, report and tokenizer config, NewBie's training code - each
    # sentence opens its own. What no page says is `unsourced` and not here:
    # Illustrious's "Give it a style", schnell's 256 tokens, ERNIE's Expand.
    sources = prompt_sources.sources_for(model_id)
    assert [
        (sentence.text[:30], sentence.url) for sentence in sources.sentences
    ] == expected


def test_an_unknown_checkpoint_has_only_the_fallback():
    sources = prompt_sources.sources_for("nobody/whatever")
    assert sources.guide_name == "FALLBACK_GUIDE" and sources.guide is guides.FALLBACK_GUIDE
    assert sources.excerpts == () and sources.url is None
    assert "no author sources" in sources.caveat
    context = _context("nobody/whatever", ordered=False)
    assert context.citations
    assert all(citation.url is None for citation in context.citations.values())
    # Its sentences are the app's general advice: nothing may call them the
    # authors', and its opening label is not a source at all.
    assert "authors" not in context.system.split("\n", 1)[0]
    assert "Unrecognised checkpoint." not in {c.text for c in context.citations.values()}
    assert "from the checkpoint's authors" in _context(KLEIN).system


def test_unsourced_sentences_are_not_citable():
    listed = 0
    for name, model_id in _every_guide().items():
        grounding = prompt_sources.GROUNDING[name]
        sentences = prompt_sources.guide_sentences(prompt_sources.guide_constants()[name]["prompt"])
        # Verbatim sentences of the guide, or a guide edit has left them behind.
        for sentence in grounding.unsourced:
            assert sentence in sentences, (name, sentence)
        for sentence, url in grounding.sentence_urls:
            assert sentence in sentences and url.startswith("https://"), (name, sentence)
        if not grounding.unsourced:
            continue
        # Listed, the ? badge keeps them and the critic never sees them.
        context = _context(model_id, ordered=False)
        cited = [sentence.text for sentence in prompt_sources.sources_for(model_id).sentences]
        guide = prompt_sources.guide_constants()[name]
        prose = guide["prompt"]
        cites = [entry["cite"] for entry in _rule_entries(guide)]
        spans = [(prose.index(cite), prose.index(cite) + len(cite)) for cite in cites]
        for sentence in grounding.unsourced:
            listed += 1
            assert sentence not in cited, (name, sentence)
            assert sentence not in {c.text for c in context.citations.values()}, (name, sentence)
            assert sentence not in context.system, (name, sentence)
            # Nor may a rule finding show its words as the guide's.
            start = prose.index(sentence)
            end = start + len(sentence)
            assert not any(a < end and b > start for a, b in spans), (name, sentence)
    assert listed
    krea = prompt_sources.sources_for("black-forest-labs/FLUX.1-Krea-dev")
    texts = [sentence.text for sentence in krea.sentences]
    assert not any("photo language" in text or "give it room" in text for text in texts)
    # Its prompting links to BFL's family guide; only the settings to Krea.
    assert krea.sentences[0].url == krea.url and krea.url.startswith("https://docs.bfl.ai/")
    assert krea.sentences[-1].url == "https://github.com/krea-ai/flux-krea"
    context = _context("black-forest-labs/FLUX.1-Krea-dev")
    assert "photo language" not in context.system and "give it room" not in context.system


def test_an_unsourced_sentence_is_never_cited(monkeypatch):
    # The filter itself, on a guide that lists nothing: klein's camera advice,
    # listed, leaves the sources, the citations and the system message, and
    # the ids after it close up.
    before = prompt_sources.sources_for(KLEIN)
    target = "Name a camera, lens or film stock for photorealism."
    texts = [sentence.text for sentence in before.sentences]
    assert target in texts and target in _context(KLEIN).system
    grounding = prompt_sources.GROUNDING[before.guide_name]
    assert not grounding.unsourced
    monkeypatch.setitem(
        prompt_sources.GROUNDING,
        before.guide_name,
        dataclasses.replace(grounding, unsourced=(target,)),
    )
    after = prompt_sources.sources_for(KLEIN)
    assert [sentence.text for sentence in after.sentences] == [t for t in texts if t != target]
    context = _context(KLEIN)
    guide_ids = [cite_id for cite_id in context.citations if cite_id.startswith("G")]
    assert guide_ids == [f"G{i}" for i in range(1, len(texts))]
    assert target not in {citation.text for citation in context.citations.values()}
    assert target not in context.system
    # Its old id now names the sentence after it.
    old = f"G{texts.index(target) + 1}"
    assert context.citations[old].text == texts[texts.index(target) + 1]


# The app talking about itself - its menus, autocomplete and chips, the Expand
# button, the box, the sampler it runs, a starting point it calls unofficial.
# Not "this app" alone: "Gemma reads up to 512 tokens ... in this app" is what
# the pipeline reads, and the token test below holds it to prompt_encoding.
_APP_VOICE = re.compile(
    r"Tags menu|Examples|[Aa]utocomplete|the chips|\bExpand\b|this box|not an official"
    r"|This app uses|not implemented by this"
)


def test_no_citable_sentence_is_the_app_talking():
    for name, model_id in _every_guide().items():
        for sentence in prompt_sources.sources_for(model_id).sentences:
            assert not _APP_VOICE.search(sentence.text), (name, sentence.text)
    # And no author is made to say what the card does not: Z-Image's card
    # calls a negative prompt optional.
    z_image = prompt_sources.sources_for("Tongyi-MAI/Z-Image")
    assert not any("strongly" in sentence.text for sentence in z_image.sentences)
    assert "strongly" not in guides.Z_IMAGE_GUIDE["prompt"]


# A token count a sentence states: "77 tokens", "a 256-token total sequence",
# "stops reading at 512", "up to 2048" - not the card's "300-token training
# context", which Kohaku's sentence sets against the pipeline's 77.
_TOKEN_COUNT = re.compile(
    r"\b(\d+)[- ]tokens?\b(?! training)|\b(?:at|to) (\d{2,})\b(?! ?(?:MP|px|steps|words))"
)


def _encoder_lengths(model_id: str) -> set[int]:
    """What each of the checkpoint's encoders reads in this app, from
    prompt_encoding's table: Qwen-Image's kept window, Krea 2's
    max_sequence_length (its cut counts the system prefix it drops), else the
    pipeline's cut, else the tokenizer's own - CLIP's 77, Ministral's 2048."""
    from tests.fakes import fake_tokenizer

    lengths = set()
    for spec in prompt_encoding.encoders_for(model_id) or []:
        if spec.window:
            lengths.add(spec.window[1])
        elif spec.file.endswith("pipeline_krea2.py"):
            lengths.add(prompt_encoding.KREA2_SEQUENCE)
        elif spec.max_length is None:
            lengths.add(fake_tokenizer(spec.source).model_max_length)
        else:
            lengths.add(spec.max_length)
    return lengths


def test_no_citable_sentence_misstates_a_token_limit():
    # schnell's "stops at 256 tokens, about 180 words" was BFL's reference;
    # this app's FluxPipeline reads 512, and a critic citing it asked for 180
    # words beside an encoder row that showed no cut.
    stated = 0
    for name, model_id in _every_guide().items():
        lengths = _encoder_lengths(model_id)
        for sentence in prompt_sources.sources_for(model_id).sentences:
            counts = {int(a or b) for a, b in _TOKEN_COUNT.findall(sentence.text)}
            stated += len(counts)
            assert counts <= lengths, (name, sentence.text, lengths)
            assert "180 words" not in sentence.text, (name, sentence.text)
            # BFL's current basics page: other languages "with the same
            # level of quality".
            assert "most precise language" not in sentence.text, (name, sentence.text)
    assert stated
    assert "most precise language" not in guides.FLUX1_GUIDE["prompt"]


def test_no_guide_sentence_runs_into_the_next():
    # WAI's negative recipe ends in a comma, and the app's own sentence after
    # it read as the card's until the recipe was quoted and closed.
    opener = re.compile(
        r"[,;]\s+(?:The|A|An|This|That|These|Those|It|Its|They|Their|There|Every|Each|Both)\b"
    )
    for name, guide in prompt_sources.guide_constants().items():
        for sentence in prompt_sources.guide_sentences(guide["prompt"]):
            assert not opener.search(sentence), (name, sentence)


def test_the_sentences_are_the_guide_prose():
    for name, guide in prompt_sources.guide_constants().items():
        sentences = prompt_sources.guide_sentences(guide["prompt"])
        assert " ".join(sentences) == " ".join(guide["prompt"].split()), name
        # The splitter takes ". " for a sentence end; an abbreviation would
        # cut a sentence in two.
        assert not re.search(r"\b(e\.g|i\.e|etc|vs)\. ", guide["prompt"]), name


def test_excerpts_are_short_sourced_and_labelled():
    for name, grounding in prompt_sources.GROUNDING.items():
        ids = [excerpt.id for excerpt in grounding.excerpts]
        assert len(ids) == len(set(ids)), name
        for excerpt in grounding.excerpts:
            assert excerpt.url.startswith("https://"), excerpt.id
            assert 0 < len(excerpt.text) <= 600, excerpt.id
            assert "{" not in excerpt.text and "civitai.com" not in excerpt.url, excerpt.id
            if excerpt.original is not None:
                # A translation says so, and keeps the words it translates.
                assert "translation" in (excerpt.note or ""), excerpt.id
                assert re.search(r"[\u4e00-\u9fff]", excerpt.original), excerpt.id
            else:
                assert not re.search(r"[\u4e00-\u9fff]", excerpt.text), excerpt.id


def test_author_orders_are_marked_only_where_the_guide_states_none():
    # Shown under Order as the authors' - never "no order is documented" -
    # and only where the app checks none of its own.
    marked = set()
    for name, grounding in prompt_sources.GROUNDING.items():
        ordered = [excerpt.id for excerpt in grounding.excerpts if excerpt.order]
        if ordered:
            assert "order" not in prompt_sources.guide_constants()[name].get("analysis", {}), name
        marked.update(ordered)
    assert marked == {
        "anima.order", "neta.sequence", "neta.prose_order", "ernie.sections", "qwen.portrait_flow",
        "rouwei.artist_order", "noobai.order",
    }


def test_borrowed_sources_say_whose_they_are():
    grounding = prompt_sources.GROUNDING
    assert "Krea published no prompting rules" in grounding["KREA_DEV_GUIDE"].caveat
    assert all("Krea published no" in e.note for e in grounding["KREA_DEV_GUIDE"].excerpts
               if e.url.startswith("https://docs.bfl.ai/guides/prompting_unified"))
    assert all("inherited" in e.note for e in grounding["NETAYUME_GUIDE"].excerpts)
    assert "system prefix" in grounding["NETAYUME_GUIDE"].caveat
    flux2 = [e for e in grounding["KLEIN_GUIDE"].excerpts if e.url.endswith("prompting_guide_flux2")]
    assert flux2 and all("klein by family" in e.note for e in flux2)
    assert "4-step" in grounding["Z_ANIME_GUIDE"].caveat
