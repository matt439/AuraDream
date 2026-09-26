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
from backend.model_defaults import _FALLBACK, _profile_for, defaults_for, guide_for, hints_for

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
    assert guide.pixels_word == "documented" and guide.tag_helper is None
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
    assert guide.pixels_word == "documented" and guide.tag_helper is None
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
        assert guide["pixels_word"] in {"documented", "recommended", "trained"}, repo
        hints = hints_for(repo)
        assert "Unrecognised" not in hints["prompt"] and "Unrecognised" not in hints["size"], repo
    unknown = guide_for("nobody/whatever")
    assert unknown["examples"] == [] and unknown["pixels"] is None
    assert unknown["aspects"] is None
    assert unknown["tag_helper"] is None


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
    # 1024-1280 px and photo language, which FLUX.1 [dev]'s card does not. To
    # every control it is the flux profile, so nothing about a request changes.
    krea, flux = "black-forest-labs/FLUX.1-Krea-dev", "black-forest-labs/FLUX.1-dev"
    assert defaults_for(krea) == defaults_for(flux)
    assert {k: v for k, v in hints_for(krea).items() if k not in ("prompt", "size")} == {
        k: v for k, v in hints_for(flux).items() if k not in ("prompt", "size")
    }
    assert hints_for(krea)["prompt"] != hints_for(flux)["prompt"]
    assert guide_for(krea)["pixels_word"] == "recommended"
    assert guide_for(flux)["pixels"] == (0.1, 2.0)


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
