"""NoobAI V-pred keeps its own runtime limits and author vocabulary."""
from __future__ import annotations

from backend import guides, model_defaults, prompt_encoding, prompt_rules, prompt_sources
from backend.backends import loading
from backend.schemas import GenerateRequest, PromptGuide
from tests.fakes import fake_tokenizer
from tests.prompt_rules_helpers import run


MODEL = "Laxhar/noobai-XL-Vpred-1.0"


def test_noobai_profile_is_specific_to_the_vpred_release():
    defaults = model_defaults.defaults_for(MODEL)
    assert (defaults["steps"], defaults["guidance_scale"]) == (28, 5.0)
    assert defaults["negative_prompt_supported"] and defaults["cfg_threshold"] == 1
    assert not defaults["mu_supported"] and not defaults["true_cfg_supported"]
    assert not defaults["reference_supported"]
    assert defaults["refine_supported"] and defaults["refine_strength"] == 0.4
    assert defaults["refine_tight_pixels"] is None  # no borrowed GPU measurement
    for model in (MODEL, MODEL.upper(), MODEL.replace("-", "_")):
        assert model_defaults._profile_for(model)["guide"] is guides.NOOBAI_GUIDE
    assert model_defaults._profile_for("Laxhar/noobai-XL_v1.0") is model_defaults._FALLBACK
    assert model_defaults._profile_for("Laxhar/noobai-XL-Vpred-0.75") is model_defaults._FALLBACK
    request = GenerateRequest(prompt="a greenhouse", negative_prompt="low quality", mu=1.15)
    applied = request.applied(
        negative_prompt_supported=True, cfg_threshold=1, mu_supported=False,
        true_cfg_supported=False, reference_supported=False,
    )
    assert applied.mu is None and applied.negative_prompt == "low quality"


def test_noobai_guide_preserves_its_vocabulary_and_exact_recommended_shapes():
    guide = PromptGuide(**model_defaults.guide_for(MODEL))
    helper = guide.tag_helper
    assert helper is not None and helper.hybrid_prompt
    groups = {group.label: group.tags for group in helper.groups}
    assert groups["Quality"] == [
        "masterpiece", "best quality", "good quality", "normal quality", "worst quality",
    ]
    assert groups["Aesthetic"] == ["very awa", "worst aesthetic"]
    assert groups["Date"] == ["old", "early", "mid", "recent", "newest"]
    assert groups["Rating"] == ["safe"]
    assert helper.year_tag and helper.vocabulary_cutoff == "2024-10-23"
    assert helper.artist_prefix == "artist:"
    assert helper.negative_prompt.startswith("nsfw, worst quality, old, early, low quality")
    assert len(guide.examples) == 1
    assert guide.examples[0].startswith("masterpiece, best quality,artist:john_kafka")
    assert "arlecchino" in guide.examples[0] and "graphite" in guide.examples[0]
    assert guide.pixels_word == "recommended"
    assert guide.resolutions == [
        (768, 1344), (832, 1216), (896, 1152), (1024, 1024),
        (1152, 896), (1216, 832), (1344, 768),
    ]
    assert (848, 1264) not in guide.resolutions


def test_noobai_prefix_and_caption_order_are_distinct_author_advice():
    sources = prompt_sources.sources_for(MODEL)
    assert sources.guide is guides.NOOBAI_GUIDE
    [order] = [excerpt for excerpt in sources.excerpts if excerpt.order]
    assert order.id == "noobai.order" and "<artists>" in order.text
    assert prompt_rules.documented_order(MODEL) is None
    prompt = guides._NOOBAI_PREFIX + " artist:john_kafka, 1girl, greenhouse"
    # The author's two-quality prefix and underscored artist example must not
    # inherit another checkpoint's exclusive tiers, spelling or total order.
    findings = run(MODEL, prompt)
    assert not any(finding.severity == "warn" for finding in findings)
    assert not any(finding.rule in {
        "guide.order", "guide.quality", "guide.underscores", "guide.artist_prefix",
    } for finding in findings)
    [negative] = [finding for finding in findings if finding.rule == "guide.negative_prompt"]
    assert negative.suggestion == guides.NOOBAI_GUIDE["tag_helper"]["negative_prompt"]
    assert negative.url == sources.url
    assert not any("Autocomplete" in sentence.text or "Examples" in sentence.text
                   for sentence in sources.sentences)


def test_noobai_encoders_follow_the_same_pin_as_the_weights():
    for model in (MODEL, MODEL.upper()):
        specs = prompt_encoding.encoders_for(model)
        assert specs is not None
        assert [(spec.name, spec.source.repo, spec.source.subfolder, spec.source.revision)
                for spec in specs] == [
            ("CLIP-L", MODEL, "tokenizer", loading.NOOB_REVISION),
            ("CLIP-G", MODEL, "tokenizer_2", loading.NOOB_REVISION),
        ]
        assert all(spec.max_length is None and not spec.pooled_only for spec in specs)
    assert prompt_encoding.encoders_for("Laxhar/noobai-XL_v1.0") is None


def test_noobai_both_prompt_fields_obey_both_clip_budgets(monkeypatch):
    prompt_encoding.unload()
    monkeypatch.setattr(prompt_encoding, "_load_tokenizer", fake_tokenizer)
    try:
        text = " ".join(f"word{index}" for index in range(80))
        reports, findings, detail = prompt_encoding.measure(MODEL, text, text)
        assert detail is None
        assert {(report.name, report.field) for report in reports} == {
            (name, field) for name in ("CLIP-L", "CLIP-G")
            for field in ("prompt", "negative_prompt")
        }
        assert all(report.budget == 75 and report.cut is not None for report in reports)
        cuts = [finding for finding in findings if finding.rule == "encoder.cut"]
        # Identical cuts are combined per field, naming both affected encoders.
        assert {finding.field for finding in cuts} == {"prompt", "negative_prompt"}
        assert len(cuts) == 2 and all(finding.severity == "warn" for finding in cuts)
        assert all("CLIP-L and CLIP-G" in finding.message for finding in cuts)
    finally:
        prompt_encoding.unload()
