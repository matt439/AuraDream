"""The facts the UI and CLI hang off, checked against the real classes.

`cfg_threshold()` and `negative_prompt_supported` are derived from the loaded
pipeline's `__call__` signature, and `model_defaults` hard-codes the same two
answers per checkpoint for the unloaded case - so the profile's guess and the
pipeline's answer have to agree, and until now only a human kept them in step.
The classes import without weights or a GPU; the id -> class mapping is the
one assumption this cannot verify offline (the loader reads it from the repo's
model_index.json).
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import diffusers
import pytest

from backend import generate, launch, sizes
from backend.backends import diffusers_backend as module
from backend.backends.diffusers_backend import DiffusersBackend
from backend.model_defaults import (
    _FALLBACK,
    _PROFILES,
    _profile_for,
    defaults_for,
    guide_for,
    hints_for,
)

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    ("black-forest-labs/FLUX.1-Krea-dev", "FluxPipeline"),
    ("krea/Krea-2-Turbo", "Krea2Pipeline"),
    ("krea/Krea-2-Raw", "Krea2Pipeline"),
    ("black-forest-labs/FLUX.2-klein-4B", "Flux2KleinPipeline"),
    ("black-forest-labs/FLUX.2-klein-base-4B", "Flux2KleinPipeline"),
    ("black-forest-labs/FLUX.1-dev", "FluxPipeline"),
    ("black-forest-labs/FLUX.1-schnell", "FluxPipeline"),
    ("Tongyi-MAI/Z-Image-Turbo", "ZImagePipeline"),
    ("SeeSee21/Z-Anime", "ZImagePipeline"),
    ("Qwen/Qwen-Image", "QwenImagePipeline"),
    ("baidu/ERNIE-Image-Turbo", "ErnieImagePipeline"),
    ("cagliostrolab/animagine-xl-4.0", "StableDiffusionXLPipeline"),
    ("KBlueLeaf/Kohaku-XL-Zeta", "StableDiffusionXLPipeline"),
    ("OnomaAIResearch/Illustrious-XL-v2.0", "StableDiffusionXLPipeline"),
    ("LyliaEngine/waiIllustriousSDXL_v170", "StableDiffusionXLPipeline"),
    ("duongve/NetaYume-Lumina-Image-2.0", "Lumina2Pipeline"),
]

SCHNELL_LITE = "lite-infer/flux.1-schnell-nunchaku-lite-nvfp4_r32-bnb4-text-encoder"
Z_IMAGE_LITE = "lite-infer/z-image-turbo-nunchaku-lite-nvfp4_r32-bnb4-text-encoder"
ERNIE_LITE = "lite-infer/ERNIE-Image-Turbo-nunchaku-lite-nvfp4-bnb4-text-encoder"


def _loaded_like(class_name: str) -> DiffusersBackend:
    cls = getattr(diffusers, class_name)
    backend = DiffusersBackend()
    # The real signature under a stand-in of the same *name*: the threshold
    # rule keys on the class name, and constructing a pipeline needs weights.
    backend._pipe = type(class_name, (), {})()
    backend._call_params = set(inspect.signature(cls.__call__).parameters)
    return backend


@pytest.mark.parametrize("model_id, class_name", CASES)
def test_the_profile_guess_matches_the_real_pipeline(model_id, class_name):
    backend = _loaded_like(class_name)
    profile = defaults_for(model_id)
    assert backend.cfg_threshold() == profile["cfg_threshold"]
    assert ("negative_prompt" in backend._call_params) is profile["negative_prompt_supported"]
    # And every family has what load() insists on.
    assert module._REQUIRED_PARAMS <= backend._call_params


def test_the_distilled_variants_beat_their_family_token():
    # The table is first-match and the ids carry the family's token too:
    # schnell at the flux profile's 28 steps, or Z-Image Turbo at the
    # ZImagePipeline's own 50, is the mistake the table exists to prevent -
    # on the official id and on the lite-infer re-upload alike.
    for repo in ("black-forest-labs/FLUX.1-schnell", SCHNELL_LITE):
        profile = defaults_for(repo)
        assert (profile["steps"], profile["guidance_scale"]) == (4, 0.0), repo
        assert profile["true_cfg_supported"], "the FluxPipeline knob is still there"
    assert defaults_for("black-forest-labs/FLUX.1-dev")["steps"] == 28
    for repo in ("Tongyi-MAI/Z-Image-Turbo", Z_IMAGE_LITE):
        profile = defaults_for(repo)
        assert (profile["steps"], profile["guidance_scale"]) == (8, 0.0), repo
        assert profile["cfg_threshold"] == 0.0, "ZImagePipeline guides from zero"
    # Qwen-Image's uncond pass is `true_cfg_scale`, as on FLUX.1, and the
    # controls have to know it is a separate knob before the weights arrive.
    qwen = defaults_for("lite-infer/Qwen-Image-nunchaku-lite-nvfp4_r32-bnb4-text-encoder")
    assert qwen["true_cfg_supported"] and qwen["steps"] == 50
    assert "true_cfg_scale" in inspect.signature(diffusers.QwenImagePipeline.__call__).parameters
    # The undistilled family entry is the pipeline's own signature defaults.
    params = inspect.signature(diffusers.ZImagePipeline.__call__).parameters
    base = defaults_for("Tongyi-MAI/Z-Image")
    assert base["steps"] == params["num_inference_steps"].default
    assert base["guidance_scale"] == params["guidance_scale"].default
    # ERNIE-Image Turbo the same way: 8 steps at guidance 1.0 under the classic
    # `> 1` gate, a negative prompt and no true CFG knob, on both ids - and the
    # family entry at the ErnieImagePipeline's own 50 / 4.0.
    for repo in ("baidu/ERNIE-Image-Turbo", ERNIE_LITE):
        profile = defaults_for(repo)
        assert (profile["steps"], profile["guidance_scale"]) == (8, 1.0), repo
        assert profile["cfg_threshold"] == 1.0 and not profile["true_cfg_supported"]
    params = inspect.signature(diffusers.ErnieImagePipeline.__call__).parameters
    assert "true_cfg_scale" not in params and "sigmas" not in params
    base = defaults_for("baidu/ERNIE-Image")
    assert base["steps"] == params["num_inference_steps"].default
    assert base["guidance_scale"] == params["guidance_scale"].default


def test_animagine_uses_the_model_cards_controls_and_guide():
    from backend.schemas import PromptGuide

    profile = defaults_for("cagliostrolab/animagine-xl-4.0")
    assert (profile["steps"], profile["guidance_scale"], profile["mu"]) == (28, 5.0, None)
    assert profile["cfg_threshold"] == 1.0
    assert profile["negative_prompt_supported"] and not profile["true_cfg_supported"]
    assert not profile["reference_supported"] and not profile["mu_supported"]
    guide = guide_for("cagliostrolab/animagine-xl-4.0")
    assert guide["pixels"] == (0.98, 1.05)
    assert guide["pixels_word"] == "recommended"
    assert guide["aspects"] == [
        "1:1", "9:7", "3:2", "7:4", "12:5", "7:9", "2:3", "4:7", "5:12"
    ]
    assert guide["resolutions"] == [
        (1024, 1024), (1152, 896), (1216, 832), (1344, 768), (1536, 640),
        (896, 1152), (832, 1216), (768, 1344), (640, 1536),
    ]
    assert any("masterpiece, high score" in prompt for prompt in guide["examples"])
    helper = guide["tag_helper"]
    assert (helper["vocabulary"], helper["vocabulary_cutoff"]) == (
        "danbooru", "2025-01-07"
    )
    assert helper["order"] == [
        "Person count", "Character", "Series", "Rating", "Details", "Quality"
    ]
    assert helper["recommended_suffix"] == [
        "masterpiece", "high score", "great score", "absurdres"
    ]
    assert helper["negative_prompt"].startswith("lowres, bad anatomy")
    assert {group["label"] for group in helper["groups"]} == {
        "Person count", "Rating", "Quality", "Score"
    }
    assert PromptGuide(**guide).tag_helper.negative_prompt == helper["negative_prompt"]
    assert guide_for("black-forest-labs/FLUX.1-dev")["tag_helper"] is None
    assert guide_for("black-forest-labs/FLUX.1-dev")["aspects"] is None
    assert guide_for("black-forest-labs/FLUX.1-dev")["resolutions"] is None
    guide["aspects"].append("16:9")
    assert "16:9" not in guide_for("cagliostrolab/animagine-xl-4.0")["aspects"]
    helper["groups"][0]["tags"].append("changed by a caller")
    assert "changed by a caller" not in guide_for("cagliostrolab/animagine-xl-4.0")[
        "tag_helper"
    ]["groups"][0]["tags"]


def test_illustrious_has_hybrid_prompting_and_a_native_not_maximum_size_range():
    from backend.schemas import PromptGuide

    model = "OnomaAIResearch/Illustrious-XL-v2.0"
    profile = defaults_for(model)
    assert (profile["steps"], profile["guidance_scale"], profile["mu"]) == (28, 5, None)
    assert not profile["mu_supported"] and not profile["true_cfg_supported"]
    assert profile["negative_prompt_supported"] and not profile["reference_supported"]
    guide = PromptGuide(**guide_for(model))
    assert guide.pixels == (1.0, 2.25) and guide.pixels_word == "documented"
    prompt_hint = hints_for(model)["prompt"]
    assert "natural-language" in prompt_hint
    assert "August 2024 cutoff" in prompt_hint
    assert "masterpiece, newest" in prompt_hint
    assert "not an official fixed suffix" in prompt_hint
    assert "general, sensitive, questionable or explicit" in prompt_hint
    # The base is untuned and styleless by the authors' own account; the hint
    # says so first, since a bare prompt is why the base looks worse than an
    # aesthetic-tuned checkpoint with its recommended tags.
    assert "untuned base" in prompt_hint and "no default style" in prompt_hint
    assert "artist tag" in prompt_hint
    assert "upper body, cowboy shot, portrait or full body" in prompt_hint
    # The negative prompt is the authors': the v0.1 card's, verbatim, and the
    # v2.0 release example's short form. The old "start empty" and the app's
    # invented baseline are gone.
    assert "worst quality, comic, multiple views" in prompt_hint
    assert "worst quality, bad quality." in prompt_hint
    assert "Start empty" not in prompt_hint and "not an author recipe" not in prompt_hint
    assert "4 MP" in hints_for(model)["size"]
    assert "2048x2048 v1.0" in hints_for(model)["size"]
    assert "1824x1248 at 28 steps and CFG 6.5" in hints_for(model)["size"]
    assert "6.5 in their v2.0 release example" in hints_for(model)["guidance_scale"]
    assert len(guide.examples) == 3
    assert guide.examples[0] == "1boy, doctor, masterpiece, looking at viewer"
    assert all(example.count(",") < 12 for example in guide.examples)
    helper = guide.tag_helper
    assert helper is not None and helper.vocabulary == "danbooru"
    assert helper.vocabulary_cutoff == "2024-08-31" and helper.hybrid_prompt
    assert helper.recommended_suffix == []
    assert helper.negative_prompt.startswith("worst quality, comic, multiple views")
    assert helper.negative_prompt.endswith("extra digits, fewer digits")
    assert helper.negative_prompt in prompt_hint
    assert helper.order == [
        "Person count", "Character", "Rating", "General", "Artist", "Quality", "Year"
    ]
    groups = {group.label: group for group in helper.groups}
    assert groups["Person count"].tags == ["1girl", "1boy"]
    # Danbooru's rating names are a post field, not index tags, so the remote
    # lookup can never accept them; the local group is the only path in.
    assert groups["Rating"].tags == ["general", "sensitive", "questionable", "explicit"]
    assert groups["Rating"].exclusive
    assert groups["Quality"].tags == [
        "worst quality", "bad quality", "average quality", "good quality",
        "best quality", "masterpiece",
    ]
    assert groups["Year"].tags == ["oldest", "old", "modern", "recent", "newest"]


def test_wai_v17_has_its_own_profile_and_author_recipe():
    from backend.schemas import PromptGuide
    from backend.tags import unchecked_results

    model = "LyliaEngine/waiIllustriousSDXL_v170"
    profile = defaults_for(model)
    assert (profile["steps"], profile["guidance_scale"], profile["mu"]) == (28, 6, None)
    assert profile["cfg_threshold"] == 1
    assert profile["negative_prompt_supported"]
    assert not profile["mu_supported"] and not profile["true_cfg_supported"]
    assert not profile["reference_supported"]
    assert _profile_for(model) is _profile_for(model.upper())
    assert _profile_for("LyliaEngine/waiIllustriousSDXL_v180") is _FALLBACK
    checkpoint = next(item for item in launch.CHECKPOINTS if item.repo == model)
    assert checkpoint.name == "WAI Illustrious v17" and checkpoint.lite is None

    guide = PromptGuide(**guide_for(model))
    assert guide.pixels == (1024**2 / 1e6, 1024 * 1344 / 1e6)
    # The author's reference sizes, which the tip says are no maximum: the
    # size line words them as examples, never as what WAI is "for".
    assert guide.pixels_word == "example"
    assert guide.examples == [",masterpiece,best quality,amazing quality,"]
    helper = guide.tag_helper
    assert helper is not None and helper.hybrid_prompt
    assert helper.vocabulary == "danbooru" and helper.vocabulary_cutoff is None
    assert not helper.year_tag
    assert helper.recommended_suffix == ["masterpiece", "best quality", "amazing quality"]
    assert helper.negative_prompt == "bad quality,worst quality,worst detail,sketch,censor,"
    groups = {group.label: group for group in helper.groups}
    assert groups["Rating"].tags == ["general", "sensitive", "nsfw", "explicit"]
    assert groups["Rating"].exclusive and not groups["Quality"].exclusive
    assert groups["Quality"].tags == helper.recommended_suffix
    special = {tag for group in helper.groups for tag in group.tags}
    assert all(
        row["valid"] and row["source"] == "model"
        for row in unchecked_results(sorted(special), special, allow_year_tag=False)
    )
    hints = hints_for(model)
    assert "15-30" in hints["steps"] and "CFG 5-7" in hints["guidance_scale"]
    assert "Euler Ancestral" in hints["mu"]
    assert "untuned base" not in hints["prompt"]
    assert "no vocabulary cutoff" in hints["prompt"]
    assert "no published maximum" in hints["size"]
    assert "documented reference sizes" in hints["size"]
    assert "1.5x high-resolution" in hints["size"]


def test_anima_turbo_has_its_own_defaults_and_documented_guide():
    from backend.schemas import PromptGuide

    model = "circlestone-labs/Anima"
    profile = defaults_for(model)
    assert (profile["steps"], profile["guidance_scale"], profile["mu"]) == (8, 1.0, None)
    assert profile["cfg_threshold"] == 1.0
    assert profile["negative_prompt_supported"]
    assert not profile["mu_supported"] and not profile["true_cfg_supported"]
    assert not profile["reference_supported"]
    checkpoint = next(c for c in launch.CHECKPOINTS if c.repo == model)
    assert checkpoint.name == "Anima Turbo v1.1" and checkpoint.lite is None
    assert checkpoint.summary() == "8 steps, CFG 1"
    guide = PromptGuide(**guide_for(model))
    assert guide.pixels == (512**2 / 1e6, 1536**2 / 1e6)
    assert guide.pixels_word == "documented" and guide.tag_helper is None
    assert guide.examples[0].startswith("Digital painting of a fiery dragon")
    # The caption whole: the card asks for two sentences, and gives two.
    assert guide.examples[0].endswith(
        "molten rock formation. The background is a gradient of dark purple to orange."
    )
    # Every tier the strip marks in range is named in range: 1.5 MP too.
    low, high = guide.pixels
    marked = {
        tier for tier in sizes.TIERS
        if low * 0.9 <= sizes.SIZES["1:1"][tier][0] ** 2 / 1e6 <= high * 1.1
    }
    assert marked == {1, 1.5, 2}
    assert "The 1, 1.5 and 2 MP presets are inside this range" in hints_for(model)["size"]
    hint = hints_for(model)
    assert "8-12" in hint["steps"]
    assert "natural language" in hint["prompt"] and "@" in hint["prompt"]
    assert "spaces instead of underscores" in hint["prompt"]
    assert "two sentences" in hint["prompt"]
    assert "no effect" in hint["prompt"] and "Unavailable" in hint["mu"]


def test_anima_turbo_profile_only_matches_the_official_resolved_id():
    model = "circlestone-labs/Anima"
    assert _profile_for(model.upper()) is _profile_for(model)
    for unrelated in (
        "someone/anima", "circlestone-labs/Anima-Base", "circlestone-labs/Anima-Aesthetic",
    ):
        assert _profile_for(unrelated) is _FALLBACK
    assert defaults_for("cagliostrolab/animagine-xl-4.0")["steps"] == 28


def test_netayume_uses_the_pinned_releases_profile_and_qualified_family_guide():
    from backend.schemas import PromptGuide

    model = "duongve/NetaYume-Lumina-Image-2.0"
    profile = defaults_for(model)
    assert (profile["steps"], profile["guidance_scale"], profile["mu"]) == (40, 5.0, None)
    assert profile["mu_supported"] and profile["negative_prompt_supported"]
    assert profile["cfg_threshold"] == 1.0
    assert not profile["true_cfg_supported"] and not profile["reference_supported"]
    checkpoint = next(c for c in launch.CHECKPOINTS if c.repo == model)
    assert checkpoint.name == "NetaYume Lumina v4" and checkpoint.lite is None
    assert checkpoint.summary() == "40 steps, CFG 5"
    assert _profile_for(model.upper()) is _profile_for(model)
    assert _profile_for("somebody/NetaYume-Lumina-Image-2.0") is _FALLBACK
    assert _profile_for("duongve/NetaYume-Lumina-Image-2.0-v2") is _FALLBACK

    guide = PromptGuide(**guide_for(model))
    assert guide.pixels == (1024**2 / 1e6, 1024 * 1536 / 1e6)
    assert guide.pixels_word == "example" and guide.tag_helper is None
    assert guide.examples[0].startswith("Gawr Gura, 1girl, solo, anime style")
    assert "<Prompt Start>" not in guide.examples[0]
    hints = hints_for(model)
    assert "v2 family guidance" in hints["steps"] and "40-50" in hints["steps"]
    assert "4-7" in hints["guidance_scale"]
    assert "fixed shift of 6" in hints["mu"]
    assert "Danbooru tags" in hints["prompt"]
    assert "256-token" in hints["prompt"] and "applied automatically" in hints["prompt"]
    assert "v4 resolution range" in hints["size"] and "not a supported maximum" in hints["size"]


def test_newbie_has_release_defaults_and_an_unparsed_xml_example():
    from xml.etree import ElementTree

    from backend.schemas import PromptGuide

    model = "NewBie-AI/NewBie-image-Exp0.1"
    profile = defaults_for(model)
    assert (profile["steps"], profile["guidance_scale"], profile["mu"]) == (28, 4.0, None)
    assert profile["mu_supported"] and profile["negative_prompt_supported"]
    assert profile["cfg_threshold"] == 1.0
    assert not profile["true_cfg_supported"] and not profile["reference_supported"]
    checkpoint = next(c for c in launch.CHECKPOINTS if c.repo == model)
    assert checkpoint.name == "NewBie image Exp0.1" and checkpoint.lite is None
    assert checkpoint.summary() == "28 steps, CFG 4"
    assert _profile_for(model.upper()) is _profile_for(model)
    assert _profile_for("somebody/NewBie-image-Exp0.1") is _FALLBACK
    assert _profile_for("NewBie-AI/NewBie-image-Exp0.2") is _FALLBACK

    guide = PromptGuide(**guide_for(model))
    assert guide.pixels == (1024**2 / 1e6, 1024**2 / 1e6)
    assert guide.pixels_word == "example" and guide.tag_helper is None
    # XML is a structured prompt, not comma chips. Preserve two complete
    # character blocks, their distinct attributes and the shared scene.
    xml = ElementTree.fromstring("<prompt>" + guide.examples[0] + "</prompt>")
    assert [node.tag for node in xml] == ["character_1", "character_2", "general_tags"]
    assert xml.findtext("character_1/n") == "$character_1$"
    assert xml.findtext("character_2/n") == "$character_2$"
    assert "blue_hair" in xml.findtext("character_1/appearance")
    assert "pink_hair" in xml.findtext("character_2/appearance")
    assert xml.findtext("character_1/position") == "center_left"
    assert xml.findtext("character_2/position") == "center_right"
    assert xml.findtext("general_tags/objects") == "briefcase"
    assert guide.examples[1] == "1girl"
    hints = hints_for(model)
    assert "quickstart" in hints["steps"]
    assert "author's sampling script" in hints["guidance_scale"]
    assert "starting point" in hints["guidance_scale"]
    assert "FlowMatch Euler" in hints["mu"]
    assert "multiple characters" in hints["prompt"] and "XML" in hints["prompt"]
    assert "no minimum or maximum" in hints["size"]
    assert "not a supported maximum" in hints["size"]


def test_every_catalogue_entry_has_a_profile():
    # The catalogue and profile table are joined by id or family token; an
    # entry that missed every token would print the fallback's "28 steps" in
    # the menu and run a distilled model at 28 steps. By identity, since the
    # flux profile is value-identical to the fallback in every number.
    repos = [c.repo for c in launch.CHECKPOINTS] + [
        c.lite_repo(fmt) for c in launch.CHECKPOINTS for fmt, _ in launch.FORMATS if c.lite
    ]
    for repo in repos:
        profile = _profile_for(repo)
        assert profile is not _FALLBACK, repo
        assert isinstance(profile["mu_supported"], bool), repo


def _catalogue_repos() -> list[str]:
    return [c.repo for c in launch.CHECKPOINTS] + [
        c.lite_repo(fmt) for c in launch.CHECKPOINTS for fmt, _ in launch.FORMATS if c.lite
    ]


def test_every_catalogue_entry_has_a_prompting_guide():
    # The guide is the model card's, not the fallback's: an entry that reached
    # the fallback would show "Unrecognised checkpoint" under a ? badge beside
    # a checkpoint the menu offers by name, with an empty Examples menu.
    for repo in _catalogue_repos():
        guide = guide_for(repo)
        assert guide["examples"], repo
        assert guide["pixels"] is not None, repo
        low, high = guide["pixels"]
        assert 0 < low <= high, repo
        assert guide["pixels_word"] in {"documented", "recommended", "trained", "example"}, repo
        hints = hints_for(repo)
        assert "Unrecognised" not in hints["prompt"] and "Unrecognised" not in hints["size"], repo
    unknown = guide_for("nobody/whatever")
    assert unknown["examples"] == [] and unknown["pixels"] is None
    assert unknown["aspects"] is None
    assert unknown["tag_helper"] is None


def test_ernies_guide_is_baidus_table_of_seven_sizes():
    # Baidu's card recommends seven 1 MP sizes. Without them as data every
    # 1 MP preset - 12:5's 2.4:1 included - was in the guide, the strip
    # carried no marks, and Baidu's own 768x1376 typed in was not recognised.
    table = [(1024, 1024), (848, 1264), (1264, 848), (768, 1376), (1376, 768), (896, 1200), (1200, 896)]
    for repo in ("baidu/ERNIE-Image-Turbo", ERNIE_LITE, "baidu/ERNIE-Image"):
        guide = guide_for(repo)
        assert guide["resolutions"] == table, repo
        assert guide["aspects"] == ["1:1", "3:4", "4:3", "2:3", "3:2", "9:16", "16:9"], repo
    # Five framings are Baidu's sizes exactly at 1 MP; 9:16 and 16:9 stand
    # for 768x1376 and 1376x768, 16 px shorter - which the size tip says.
    exact = {sizes.SIZES[aspect][1] for aspect in ("1:1", "3:4", "4:3", "2:3", "3:2")}
    assert exact <= set(table)
    assert sizes.SIZES["9:16"][1] == (768, 1376 - 16) and sizes.SIZES["16:9"][1] == (1376 - 16, 768)
    size = hints_for("baidu/ERNIE-Image-Turbo")["size"]
    assert "9:16 and 16:9 stand for 768x1376 and 1376x768" in size
    assert all(f"{w}x{h}" in size for w, h in table)


def test_a_guides_aspects_are_presets_and_its_sizes_their_framings():
    # The strip marks a preset by its aspect's name, so a name the table has
    # not got would mark nothing; and a size of the card's table should be
    # one of the framings marked, or the page would call it unrecommended
    # once it is picked from the strip rather than typed.
    for repo in _catalogue_repos():
        guide = guide_for(repo)
        if guide["aspects"] is None:
            assert guide["resolutions"] is None, repo
            continue
        assert set(guide["aspects"]) <= set(sizes.SIZES), repo
        ratios = [int(a) / int(b) for a, b in (x.split(":") for x in guide["aspects"])]
        for width, height in guide["resolutions"]:
            assert any(abs(width / height / r - 1) < 0.03 for r in ratios), (repo, width, height)


def test_the_example_word_is_exactly_where_the_prose_says_no_bound():
    # "example" is the span of the authors' example sizes where they publish
    # no range; the size line words it as the examples' size, never as what
    # the checkpoint is "for". Each such guide's tip says it is no maximum,
    # and no guide whose tip says so claims a stronger word.
    disclaims = ("not a supported maximum", "no published maximum")
    for repo in _catalogue_repos():
        word = guide_for(repo)["pixels_word"]
        size = hints_for(repo)["size"]
        assert (word == "example") == any(phrase in size for phrase in disclaims), (repo, word)
    assert {
        repo for repo in _catalogue_repos() if guide_for(repo)["pixels_word"] == "example"
    } == {
        "duongve/NetaYume-Lumina-Image-2.0", "NewBie-AI/NewBie-image-Exp0.1",
        "LyliaEngine/waiIllustriousSDXL_v170",
    }


def test_a_fixed_shift_is_the_profiles_mu_static():
    # Where the scheduler's shift is a fixed number, a blank Mu is that
    # number, and the page says it ("auto (1.79)") and sorts a sheet's Mu
    # none at it. NetaYume's and NewBie's are the loaders' pinned scheduler
    # shift; Z-Anime's and Z-Image's are the shipped scheduler configs' 6 and
    # 3 (design notes). Everywhere else the shift is derived, or Mu is off.
    import math

    from backend.backends import netayume_loading, newbie_loading

    fixed = {repo: defaults_for(repo)["mu_static"] for repo in _catalogue_repos()}
    assert fixed["duongve/NetaYume-Lumina-Image-2.0"] == math.log(netayume_loading.RECIPE["shift"])
    assert fixed["NewBie-AI/NewBie-image-Exp0.1"] == math.log(newbie_loading.RECIPE["shift"])
    assert fixed["SeeSee21/Z-Anime"] == math.log(6)
    assert fixed["Tongyi-MAI/Z-Image-Turbo"] == defaults_for("Tongyi-MAI/Z-Image")["mu_static"] == math.log(3)
    named = {repo for repo, value in fixed.items() if value is not None}
    assert {repo for repo in named if not repo.startswith("lite-infer/")} == {
        "duongve/NetaYume-Lumina-Image-2.0", "NewBie-AI/NewBie-image-Exp0.1", "SeeSee21/Z-Anime",
        "Tongyi-MAI/Z-Image-Turbo",
    }
    for repo in named:
        profile = defaults_for(repo)
        # A number only where Mu applies and the profile sends none itself.
        assert profile["mu_supported"] and profile["mu"] is None, repo
        # And the Mu tip quotes the same number the placeholder does.
        assert f"mu {profile['mu_static']:.2f}" in hints_for(repo)["mu"].replace("mu of ", "mu "), repo
    # FLUX derives its shift from the size: no number to give.
    assert defaults_for("black-forest-labs/FLUX.1-Krea-dev")["mu_static"] is None
    assert defaults_for("nobody/whatever")["mu_static"] is None


def test_every_guide_admits_the_1_mp_presets():
    # 1 MP is the size every one of these was trained around, and the page opens
    # on a 1 MP preset; the size line warns a tenth outside the range, so every
    # 1 MP preset (1360x768 is 1.04 MP, 896x1200 is 1.08) has to sit inside
    # that tolerance or a fresh page would open with a warning on it.
    for repo in _catalogue_repos():
        low, high = guide_for(repo)["pixels"]
        for aspect, tiers in sizes.SIZES.items():
            width, height = tiers[1]
            pixels = width * height / 1e6
            assert low * 0.9 <= pixels <= high * 1.1, (repo, aspect)


def test_krea_dev_is_flux_to_the_controls_and_its_own_to_the_guide():
    # The krea-dev entry exists for the guide alone: Krea's README recommends
    # 1024-1280 px, which FLUX.1 [dev]'s card does not. To every control it is
    # the flux profile, so nothing about a request changes.
    krea, flux = "black-forest-labs/FLUX.1-Krea-dev", "black-forest-labs/FLUX.1-dev"
    assert defaults_for(krea) == defaults_for(flux)
    assert {k: v for k, v in hints_for(krea).items() if k not in ("prompt", "size")} == {
        k: v for k, v in hints_for(flux).items() if k not in ("prompt", "size")
    }
    assert hints_for(krea)["prompt"] != hints_for(flux)["prompt"]
    assert guide_for(krea)["pixels_word"] == "recommended"
    assert guide_for(flux)["pixels"] == (0.1, 2.0)


def test_flux1_names_its_cfg_box_distilled_and_no_other_profile_does():
    # FLUX.1 [dev]'s box labelled CFG is, by its own tip, the distilled
    # guidance value the model takes as an input; True CFG beside it is the
    # real one. The authors call it CFG, so the name keeps the word and is
    # qualified: "CFG (distilled)" on the panel's and the grid's labels. Every
    # profile sharing FLUX.1 [dev]'s numbers - Krea dev and the lite repos -
    # says so, and no other: schnell's transformer takes no guidance at all,
    # klein's is baked in, and the rest run real CFG in that box.
    distilled = {id(dict(_PROFILES)["flux"]), id(dict(_PROFILES)["krea-dev"])}
    named = set()
    for repo in _catalogue_repos():
        label = hints_for(repo).get("cfg_label")
        if id(_profile_for(repo)) in distilled:
            assert label == "CFG (distilled)", repo
            named.add(repo)
        else:
            assert label is None, repo
    assert {"black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-Krea-dev"} <= named
    assert "cfg_label" not in hints_for("nobody/whatever")
    # A loaded pipeline that answers the negative prompt otherwise gets the
    # generic tip, which calls the box classifier-free guidance: the name
    # goes with the tip it would contradict. One that agrees keeps both.
    flux = "black-forest-labs/FLUX.1-Krea-dev"
    assert hints_for(flux, negative_prompt_supported=True)["cfg_label"] == "CFG (distilled)"
    assert "cfg_label" not in hints_for(flux, negative_prompt_supported=False)


def test_the_page_writes_the_cfg_label_on_both_cfg_boxes():
    # applyHints (status.js) writes the hint, or plain CFG where a status
    # has none - a server restarted under another checkpoint must not keep
    # FLUX.1's word - on the panel's box and the grid's axis, whose labels
    # carry the ids it writes through and start as the plain word.
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert '<label for="cfg" id="cfg-label">CFG</label>' in html
    assert '<label for="grid-cfg" id="grid-cfg-label">CFG</label>' in html
    status = (ROOT / "web" / "status.js").read_text(encoding="utf-8")
    apply = status[status.index("function applyHints"):status.index("function applyGuide")]
    assert "hints.cfg_label ?? 'CFG'" in apply
    assert "els.cfgLabel" in apply and "els.gridCfgLabel" in apply


def test_the_size_table_is_the_same_in_the_browser_and_the_cli():
    # completed_cells keys on width and height, so a preset that differs
    # between the two makes --resume unable to continue a browser grid.
    src = (ROOT / "web" / "state.js").read_text(encoding="utf-8")
    block = re.search(r"const SIZES = \{(.*?)\n\};", src, re.S).group(1)
    found = {}
    for aspect, body in re.findall(r"'([\d:]+)':\s*\{([^}]*)\}", block):
        found[aspect] = {
            float(tier): tuple(int(n) for n in re.findall(r"\d+", dims))
            for tier, dims in re.findall(r"([\d.]+): \[([^\]]*)\]", body)
        }
    assert found == sizes.SIZES
    assert list(found) == list(sizes.SIZES), "the preset strip is Object.keys of this copy"
    assert generate.SIZES is sizes.SIZES


def test_the_readme_size_table_matches_the_code():
    # The fourth copy. Rows read `| 16:9 | 1360×768 | 1920×1088 | 2736×1536 |`.
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    table = re.search(r"## Sizes.*?\n\n(\|.*?)\n\n", text, re.S).group(1)
    found = {}
    for line in table.splitlines()[2:]:
        aspect, *cells = [cell.strip() for cell in line.strip("|").split("|")]
        found[aspect] = {
            tier: tuple(int(n) for n in re.findall(r"\d+", cell))
            for tier, cell in zip(sizes.TIERS, cells)
        }
    assert found == sizes.SIZES
    assert list(found) == list(sizes.SIZES)


def test_the_aspects_are_in_strip_order():
    # Square first, then each portrait framing followed by its landscape
    # inverse, pairs from nearest-square to widest. The strip draws two adjacent
    # inverses as one segmented box, so the pairing is this order and nothing
    # else; --aspect's choices take it too.
    def ratio(aspect):
        wide, high = map(int, aspect.split(":"))
        return wide / high

    order = list(sizes.SIZES)
    assert order[0] == "1:1"
    pairs = list(zip(order[1::2], order[2::2]))
    assert len(pairs) * 2 + 1 == len(order)
    for portrait, landscape in pairs:
        assert portrait == ":".join(reversed(landscape.split(":")))
        assert ratio(portrait) < 1 < ratio(landscape)
    widths = [ratio(landscape) for _, landscape in pairs]
    assert widths == sorted(widths)


def test_the_1_5_mp_row_is_the_model_cards():
    # The row exists because Qwen-Image's README shapes and FLUX.1 Krea's
    # recommended ceiling both fell between 1 and 2 MP; a change to it that
    # loses either has lost the reason for it.
    qwen = {(1664, 928), (928, 1664), (1472, 1104), (1104, 1472), (1584, 1056), (1056, 1584)}
    row = {sizes.SIZES[aspect][1.5] for aspect in sizes.SIZES}
    assert qwen <= row
    # Krea's "between 1024 and 1280 pixels" is encoded by the guide as 1-1.64
    # MP; the row has to land inside the size line's tolerance on it, or a
    # preset the row was made for would open with a warning under it. Qwen's
    # 3:2 pair, at 1.67 MP, is the one entry past the range itself.
    low, high = guide_for("black-forest-labs/FLUX.1-Krea-dev")["pixels"]
    assert all(low * 0.9 <= w * h / 1e6 <= high * 1.1 for w, h in row)
    assert all(1.5 <= w * h / 1e6 <= 1.7 for w, h in row)
    # 3:2 and 2:3 exist for ERNIE-Image's card shapes at 1 MP.
    assert (sizes.SIZES["3:2"][1], sizes.SIZES["2:3"][1]) == ((1264, 848), (848, 1264))
    for aspect, ratio in (("3:2", 1.5), ("2:3", 2 / 3)):
        for w, h in sizes.SIZES[aspect].values():
            assert abs(w / h / ratio - 1) < 0.011, (aspect, w, h)
    assert all(w % 16 == 0 and h % 16 == 0 for w, h in sizes.every_size())
    # And --mp 1.5 reaches it from the CLI.
    args = generate._build_parser().parse_args(["x", "--aspect", "16:9", "--mp", "1.5"])
    assert generate._resolve_size(args, generate._build_parser()) == (1664, 928)


def test_animagines_additional_aspects_span_every_tier():
    # The card's six previously missing recommended dimensions are exact at
    # 1 MP. Higher tiers keep each framing within 1% and the 4 MP row's pixel
    # count remains effectively constant, so an aspect switch is not a VRAM bet.
    animagine = {
        "9:7": (1152, 896), "7:9": (896, 1152),
        "7:4": (1344, 768), "4:7": (768, 1344),
        "12:5": (1536, 640), "5:12": (640, 1536),
    }
    assert {aspect: sizes.SIZES[aspect][1] for aspect in animagine} == animagine
    for aspect in animagine:
        expected = int(aspect.split(":")[0]) / int(aspect.split(":")[1])
        for width, height in sizes.SIZES[aspect].values():
            assert abs(width / height / expected - 1) < 0.01, (aspect, width, height)
    assert all(
        abs(width * height / 2048**2 - 1) < 0.004
        for width, height in (tiers[4] for tiers in sizes.SIZES.values())
    )


def test_every_non_square_aspect_has_an_inverse_at_every_tier():
    # The browser's orientation button derives the opposite name rather than
    # carrying a second mapping, so the shared table must stay pair-complete.
    for aspect, tiers in sizes.SIZES.items():
        wide, high = aspect.split(":")
        if wide == high:
            continue
        opposite = sizes.SIZES[f"{high}:{wide}"]
        assert {tier: (height, width) for tier, (width, height) in tiers.items()} == opposite


def test_the_size_guard_rails_are_the_largest_preset():
    # config.py used to derive these in prose and then hard-code rounder
    # numbers; the HTML's `max` on the two boxes is the same bound by hand.
    from backend.config import settings

    assert settings.max_side == max(max(size) for size in sizes.every_size()) == 3168
    assert settings.max_pixels == max(w * h for w, h in sizes.every_size()) == 3168 * 1328
    # The batch guard keeps the known two-at-4-MP case and admits every aspect
    # at the UI's Batch maximum in the nominal 1 MP row. The latter is the
    # larger and is why the old 8,411,136-pixel default was raised; it grew
    # again with the cap (8 -> 12).
    from backend.seeds import MAX_BATCH

    assert settings.max_batch_pixels == max(
        2 * sizes.largest_pixels(), MAX_BATCH * sizes.largest_tier_pixels(1)
    ) == 12_902_400
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    bounds = re.findall(r'id="(?:width|height)"[^>]*max="(\d+)"', html)
    assert bounds == [str(settings.max_side)] * 2
    # The Batch box's `max` is the same cap by hand.
    assert re.findall(r'id="batch"[^>]*max="(\d+)"', html) == [str(MAX_BATCH)]
    # Every preset passes the budget it defines.
    from backend.schemas import GenerateRequest

    for width, height in sizes.every_size():
        GenerateRequest(prompt="x", width=width, height=height).check_budget()
    for tiers in sizes.SIZES.values():
        width, height = tiers[1]
        GenerateRequest(
            prompt="x", width=width, height=height, num_images=MAX_BATCH
        ).check_budget()


def test_a_refine_pass_reaches_every_2_mp_preset_at_2x():
    # A pass's own guard: past the 4 MP presets, since it takes its
    # composition from the source. The page's Refine dialog derives the same
    # two numbers from its copy of the table (web/refine.js).
    from backend.config import settings
    from backend.schemas import GenerateRequest
    from backend.seeds import MAX_BATCH

    assert settings.refine_max_side == 2 * max(max(t[2]) for t in sizes.SIZES.values()) == 4448
    assert settings.refine_max_pixels == 4 * sizes.largest_tier_pixels(2) == 4 * 1776 * 1184
    for tiers in sizes.SIZES.values():
        for tier in (1, 1.5, 2):
            width, height = (2 * side for side in tiers[tier])
            GenerateRequest(
                prompt="x", width=width, height=height, refine=1, num_images=MAX_BATCH
            ).check_budget()
    src = (ROOT / "web" / "refine.js").read_text(encoding="utf-8")
    assert "const REFINE_MAX_SIDE = 2 * Math.max(...ASPECTS.flatMap((a) => SIZES[a][2]));" in src
    assert (
        "const REFINE_MAX_PIXELS = 4 * Math.max(...ASPECTS.map((a) => SIZES[a][2][0] * SIZES[a][2][1]));"
        in src
    )


def test_a_refine_passs_tight_line_is_a_row_of_verified_md():
    # The Refine dialog warns past `refine_tight_pixels` and does not open
    # past it. A VRAM figure, so it is quoted, never modelled: each profile's
    # `refine_measured` is the largest pass verified.md's "Past 4 MP" table ran
    # without spilling, and every checkpoint the table does not measure is
    # None - the panel's generic line - rather than a sibling's number:
    # Animagine runs WAI's pipeline, but was not measured. FLUX.1 dev and the
    # lite repos are Krea dev's profile in every number, so they quote its
    # run - and the size line names that run, never "this checkpoint".
    from backend.config import settings

    measured = {
        "black-forest-labs/FLUX.1-Krea-dev": ("FLUX.1 Krea dev", 1536, 1536),
        "black-forest-labs/FLUX.1-dev": ("FLUX.1 Krea dev", 1536, 1536),
        "SeeSee21/Z-Anime": ("Z-Anime", 1536, 1536),
        "LyliaEngine/waiIllustriousSDXL_v170": ("WAI v17", 2560, 3360),
    }
    text = (ROOT / "docs" / "verified.md").read_text(encoding="utf-8")
    section = text.split("### Past 4 MP", 1)[1].split("\n## ", 1)[0]
    intro = section.split("\n| ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| ")]
    repos = [c.repo for c in launch.CHECKPOINTS] + [
        c.lite_repo(fmt) for c in launch.CHECKPOINTS for fmt, _ in launch.FORMATS if c.lite
    ]
    for repo in repos:
        tight = defaults_for(repo)["refine_tight_pixels"]
        words = hints_for(repo).get("refine_tight")
        base = next((m for m in measured if _profile_for(repo) is _profile_for(m)), None)
        if base is None:
            assert tight is None and words is None, repo
            continue
        label, width, height = measured[base]
        assert _profile_for(repo)["refine_measured"] == (label, "fp8", width, height), repo
        assert tight == width * height, repo
        row = next(line for line in rows if line.startswith(f"| {label},") and f"{width}x{height}" in line)
        assert "spilling" not in row, row
        assert words == f"past the {width}×{height} ({width * height / 1e6:.2f} MP) measured on {label} at fp8"
    # Every run in the table is the quantization the profiles name.
    assert "fp8 and model offload" in " ".join(intro.split())
    assert "measured on FLUX.1 Krea dev" in hints_for("black-forest-labs/FLUX.1-dev")["refine_tight"]
    # WAI ran past the pass's own guard: nothing the dialog offers is tight.
    assert defaults_for("LyliaEngine/waiIllustriousSDXL_v170")["refine_tight_pixels"] > settings.refine_max_pixels
    # The size line keeps its words where the number is the generic one.
    assert defaults_for("cagliostrolab/animagine-xl-4.0")["refine_tight_pixels"] is None
    assert "refine_tight" not in hints_for("black-forest-labs/FLUX.1-Krea-dev", refine_supported=False)


def test_the_strength_sweep_is_the_dialogs_compare_3():
    # The tip's example and the dialog's Compare 3 are the same three
    # strengths: half the checkpoint's own, it, and half as much again, on
    # the 0.05 grid, less any that leaves no step. One arithmetic, written
    # twice in the same order so the floats round alike.
    from backend.model_defaults import strength_sweep

    assert strength_sweep(0.1, 28) == [0.05, 0.1, 0.15]
    assert strength_sweep(0.4, 28) == [0.2, 0.4, 0.6]
    assert strength_sweep(0.5, 4) == [0.25, 0.5, 0.75]
    # 0.05 of 10 steps leaves none, and 1.2 is past 1.
    assert strength_sweep(0.1, 10) == [0.1, 0.15]
    assert strength_sweep(0.8, 28) == [0.4, 0.8]
    assert "such as 0.05,0.1,0.15," in hints_for("black-forest-labs/FLUX.1-Krea-dev")["refine"]
    src = (ROOT / "web" / "refine.js").read_text(encoding="utf-8")
    assert "Math.floor(factor * start / 0.05 + 0.5) * 0.05" in src
    # The FLUX.1 note under Strength rides with the pass, and only there.
    assert "sigma 0.75" in hints_for("black-forest-labs/FLUX.1-Krea-dev")["refine_large"]
    assert "refine_large" not in hints_for("black-forest-labs/FLUX.1-Krea-dev", refine_supported=False)
    assert "refine_large" not in hints_for("cagliostrolab/animagine-xl-4.0")
