"""The encoder table and its contracts.

Every catalogue checkpoint resolves to its encoders, and pinned, lite and
nested sources match the loaders' constants. The templates, lengths and
enable_thinking switches copied into backend/prompt_encoding.py are held to
the installed diffusers source and the repo's own pipelines. Real tokenizers
already in the Hub cache are checked against the pipelines' own truncation
(skipped otherwise), and the fakes' budgets against theirs. The budget
arithmetic on fakes is in test_prompt_encoding_budgets.py, the weighting
syntax in test_prompt_encoding_syntax.py and the load, cache and lock
lifecycle in test_prompt_encoding_loading.py.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import re
from pathlib import Path

import pytest

from backend import launch, model_defaults
from backend import prompt_encoding as pe
from backend.backends import anima_loading, loading, netayume_loading, newbie_loading, single_file
from tests import fakes
# fresh_cache is autouse: imported by name so pe._tokenizers never leaks
# between the real and fake budget tests, which read it through _clean.
from tests.prompt_encoding_helpers import fresh_cache, rules  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


def names(model_id):
    return [spec.name for spec in pe.encoders_for(model_id)]


# --- the table ------------------------------------------------------------------


def _catalogue_ids():
    for checkpoint in launch.CHECKPOINTS:
        yield checkpoint.repo
        for fmt, _ in launch.FORMATS:
            if checkpoint.lite_repo(fmt):
                yield checkpoint.lite_repo(fmt)


@pytest.mark.parametrize("model_id", list(_catalogue_ids()))
def test_every_catalogue_checkpoint_resolves_to_its_encoders(model_id):
    specs = pe.encoders_for(model_id)
    assert specs, f"{model_id} has no encoder entry"
    for spec in specs:
        assert spec.name and spec.source.repo and spec.source.subfolder
        assert spec.cite.endswith(f"({spec.file})")
    assert len({spec.name for spec in specs}) == len(specs)


def test_ordinary_repos_read_their_own_tokenizers_and_pinned_releases_their_pins():
    # A lite-infer re-upload carries its own tokenizer folders.
    lite = "lite-infer/flux.1-dev-nunchaku-lite-nvfp4_r32-bnb4-text-encoder"
    assert [(s.name, s.source.repo, s.source.subfolder) for s in pe.encoders_for(lite)] == [
        ("CLIP-L", lite, "tokenizer"), ("T5-XXL", lite, "tokenizer_2"),
    ]
    sdxl = single_file.SDXL_CONFIG
    for model_id in (single_file.ILLUSTRIOUS_ID, single_file.WAI_ID, "Minthy/RouWei-0.8"):
        sources = [spec.source for spec in pe.encoders_for(model_id)]
        assert [(s.repo, s.subfolder, s.revision, s.tree) for s in sources] == [
            (sdxl["repo_id"], "tokenizer", sdxl["revision"], model_id),
            (sdxl["repo_id"], "tokenizer_2", sdxl["revision"], model_id),
        ]
    (z_anime,) = pe.encoders_for("SeeSee21/Z-Anime")
    revision, folder = loading.NESTED_MODELS["SeeSee21/Z-Anime"]
    assert (z_anime.source.subfolder, z_anime.source.revision) == (f"{folder}/tokenizer", revision)
    (neta,) = pe.encoders_for(netayume_loading.MODEL_ID)
    shared = netayume_loading.SHARED
    assert (neta.source.repo, neta.source.revision) == (shared["repo_id"], shared["revision"])
    assert neta.prefix == netayume_loading.SYSTEM_PROMPT + " <Prompt Start> "
    gemma_spec, jina_spec = pe.encoders_for(newbie_loading.MODEL_ID)
    assert gemma_spec.source.subfolder == "text_encoder"
    assert jina_spec.source.subfolder == "clip_model"
    assert gemma_spec.source.revision == newbie_loading.SOURCE["revision"]
    assert (gemma_spec.max_length, jina_spec.max_length) == (
        newbie_loading.RECIPE["gemma_max_length"], newbie_loading.RECIPE["jina_max_length"]
    )
    qwen_spec, t5_spec = pe.encoders_for(anima_loading.MODEL_ID)
    assert qwen_spec.source.repo == anima_loading.SHARED["repo_id"]
    assert t5_spec.source.subfolder == "t5_tokenizer"


def test_anima_is_matched_exactly_and_never_by_animagines_token():
    assert names("cagliostrolab/animagine-xl-4.0") == ["CLIP-L", "CLIP-G"]
    assert names("circlestone-labs/Anima") == ["Qwen3-0.6B", "T5 ids"]
    assert names(anima_loading.AESTHETIC_MODEL_ID) == ["Qwen3-0.6B", "T5 ids"]
    assert names(anima_loading.AESTHETIC_MODEL_ID.upper()) == ["Qwen3-0.6B", "T5 ids"]
    assert pe.encoders_for("circlestone-labs/Anima@aesthetic-v1.0") is None
    assert pe.encoders_for("someone/anima-remix") is None
    assert pe.encoders_for("someone/unknown-model") is None
    # klein's encoder follows its size; FLUX.1's CLIP and T5 never reach it.
    assert names("black-forest-labs/FLUX.2-klein-9B") == ["Qwen3-8B"]
    assert names("black-forest-labs/FLUX.2-klein-base-4B") == ["Qwen3-4B"]


# --- the pipelines' own templates -------------------------------------------------


def _diffusers(relative: str) -> ast.Module:
    locations = importlib.util.find_spec("diffusers").submodule_search_locations
    return ast.parse((Path(next(iter(locations))) / relative).read_text(encoding="utf-8"))


def _repo(relative: str) -> ast.Module:
    return ast.parse((ROOT / relative).read_text(encoding="utf-8"))


def _self_values(tree: ast.Module) -> dict[str, object]:
    values = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and getattr(target.value, "id", None) == "self":
                try:
                    values[target.attr] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return values


def _default(tree: ast.Module, cls: str, function: str, arg: str) -> object:
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == cls):
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == function:
                args = item.args
                positional = [a.arg for a in args.posonlyargs + args.args]
                named = positional[len(positional) - len(args.defaults):]
                defaults = dict(zip(named, args.defaults))
                defaults.update((a.arg, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d)
                return ast.literal_eval(defaults[arg])
    raise AssertionError(f"{cls}.{function} not found")


def _keywords(tree: ast.Module, callee: str) -> list[dict[str, str]]:
    """The keywords of every call to `callee(...)` or `<anything>.callee(...)`, unparsed."""
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if callee in (getattr(node.func, "attr", None), getattr(node.func, "id", None)):
            calls.append({kw.arg: ast.unparse(kw.value) for kw in node.keywords if kw.arg})
    return calls


def _call_default(tree: ast.Module, cls: str) -> object:
    return _default(tree, cls, "__call__", "max_sequence_length")


def test_krea2_and_qwen_image_templates_match_the_installed_pipelines():
    krea = _diffusers("pipelines/krea2/pipeline_krea2.py")
    values = _self_values(krea)
    assert values["prompt_template_encode_prefix"] == pe.QWEN_SYSTEM_PREFIX
    assert values["prompt_template_encode_suffix"] == pe.QWEN_ASSISTANT_SUFFIX
    assert values["prompt_template_encode_start_idx"] == pe.KREA2_PREFIX_TOKENS
    assert values["prompt_template_encode_num_suffix_tokens"] == pe.KREA2_SUFFIX_TOKENS
    assert _call_default(krea, "Krea2Pipeline") == pe.KREA2_SEQUENCE
    assert {
        "max_length": (
            "max_sequence_length + prefix_idx - self.prompt_template_encode_num_suffix_tokens"
        ),
        "truncation": "True", "padding": "'max_length'", "return_tensors": "'pt'",
    } in _keywords(krea, "tokenizer")

    qwen_image = _diffusers("pipelines/qwenimage/pipeline_qwenimage.py")
    values = _self_values(qwen_image)
    template = pe.QWEN_SYSTEM_PREFIX + "{}" + pe.QWEN_ASSISTANT_SUFFIX
    assert values["prompt_template_encode"] == template
    assert values["prompt_template_encode_start_idx"] == pe.QWEN_IMAGE_PREFIX_TOKENS
    assert values["tokenizer_max_length"] == pe.QWEN_IMAGE_TOKENIZER_MAX
    assert _call_default(qwen_image, "QwenImagePipeline") == pe.QWEN_IMAGE_SEQUENCE
    assert "self.tokenizer_max_length + drop_idx" in [
        kw.get("max_length") for kw in _keywords(qwen_image, "tokenizer")
    ]


def test_chat_template_switches_and_lengths_match_the_installed_pipelines():
    z_image = _diffusers("pipelines/z_image/pipeline_z_image.py")
    assert [kw["enable_thinking"] for kw in _keywords(z_image, "apply_chat_template")] == ["True"]
    assert _call_default(z_image, "ZImagePipeline") == pe.Z_IMAGE_SEQUENCE
    klein = _diffusers("pipelines/flux2/pipeline_flux2_klein.py")
    assert [kw["enable_thinking"] for kw in _keywords(klein, "apply_chat_template")] == ["False"]
    assert _call_default(klein, "Flux2KleinPipeline") == pe.KLEIN_SEQUENCE
    assert pe.encoders_for("Tongyi-MAI/Z-Image-Turbo")[0].chat_thinking is True
    assert pe.encoders_for("black-forest-labs/FLUX.2-klein-4B")[0].chat_thinking is False


def test_lumina_flux_ernie_and_sdxl_lengths_match_the_installed_pipelines():
    lumina = _diffusers("pipelines/lumina2/pipeline_lumina2.py")
    constants = {node.value for node in ast.walk(lumina) if isinstance(node, ast.Constant)}
    assert pe.LUMINA_SEPARATOR in constants
    assert _call_default(lumina, "Lumina2Pipeline") == pe.LUMINA_SEQUENCE
    flux = _diffusers("pipelines/flux/pipeline_flux.py")
    assert _call_default(flux, "FluxPipeline") == pe.FLUX_T5_SEQUENCE
    lengths = [kw.get("max_length") for kw in _keywords(flux, "tokenizer")]
    assert "self.tokenizer_max_length" in lengths
    # ERNIE truncates with no max_length: the tokenizer's model_max_length.
    ernie = _diffusers("pipelines/ernie_image/pipeline_ernie_image.py")
    (call,) = [kw for kw in _keywords(ernie, "tokenizer") if "truncation" in kw]
    assert call["truncation"] == "True" and "max_length" not in call
    assert pe.encoders_for("baidu/ERNIE-Image-Turbo")[0].max_length is None
    sdxl = _diffusers("pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py")
    lengths = [kw.get("max_length") for kw in _keywords(sdxl, "tokenizer")]
    assert "tokenizer.model_max_length" in lengths


@pytest.mark.parametrize("pipeline, module", [
    ("ZImagePipeline", "pipelines/z_image/pipeline_z_image_inpaint.py"),
    ("StableDiffusionXLPipeline", "pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_inpaint.py"),
])
def test_the_inpaint_classes_build_the_prompt_their_pipelines_do(pipeline, module):
    # An inpaint runs the inpaint class, while Check measures with the
    # text-to-image class's strings (`encoder_table`): the two have to
    # build the same prompt, or Check's counts are wrong for an inpaint.
    import diffusers

    from backend import refine

    text = getattr(diffusers, pipeline)
    inpaint = refine.sibling(text, "Inpaint")
    assert inpaint is not None and inspect.getsourcefile(inpaint).replace("\\", "/").endswith(module)
    for name in ("encode_prompt", "_encode_prompt", "_get_clip_prompt_embeds"):
        if hasattr(text, name):
            assert inspect.getsource(getattr(inpaint, name)) == inspect.getsource(getattr(text, name)), name
    tree = _diffusers(module)
    if pipeline == "ZImagePipeline":
        assert [kw["enable_thinking"] for kw in _keywords(tree, "apply_chat_template")] == ["True"]
        assert _call_default(tree, inpaint.__name__) == pe.Z_IMAGE_SEQUENCE
    else:
        assert "tokenizer.model_max_length" in [kw.get("max_length") for kw in _keywords(tree, "tokenizer")]


def test_the_repos_own_pipelines_lengths_match():
    anima = _repo("backend/backends/anima_pipeline.py")
    lengths = [
        ast.literal_eval(value)
        for node in ast.walk(anima) if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "max_sequence_length"
    ]
    assert lengths == [pe.ANIMA_SEQUENCE]
    newbie = _repo("backend/backends/newbie_pipeline.py")
    calls = _keywords(newbie, "tokenizer") + _keywords(newbie, "tokenizer_2")
    recipe = newbie_loading.RECIPE
    assert sorted(int(kw["max_length"]) for kw in calls if "max_length" in kw) == sorted(
        [recipe["gemma_max_length"], recipe["jina_max_length"]]
    )


# --- the real tokenizers, where this machine has them ----------------------------


def _cached(source: pe.TokenizerSource) -> Path | None:
    from huggingface_hub.constants import HF_HUB_CACHE

    snapshots = Path(HF_HUB_CACHE) / f"models--{source.repo.replace('/', '--')}" / "snapshots"
    candidates = [snapshots / source.revision] if source.revision else sorted(snapshots.glob("*"))
    for snapshot in candidates:
        folder = snapshot / source.subfolder
        if (folder / "tokenizer_config.json").is_file():
            return folder
    return None


# Measured budgets for a one-line prompt. NetaYume's instruction costs 18
# tokens and the space after "<Prompt Start>" joins the first word.
REAL = [
    ("black-forest-labs/FLUX.1-Krea-dev", {"CLIP-L": 75, "T5-XXL": 511}),
    ("krea/Krea-2-Turbo", {"Qwen3-VL-4B": 507}),
    ("black-forest-labs/FLUX.2-klein-4B", {"Qwen3-4B": 509}),
    ("SeeSee21/Z-Anime", {"Qwen3-4B": 509}),
    ("KBlueLeaf/Kohaku-XL-Zeta", {"CLIP-L": 75, "CLIP-G": 75}),
    ("duongve/NetaYume-Lumina-Image-2.0", {"Gemma 2 2B": 237}),
    ("NewBie-AI/NewBie-image-Exp0.1", {"Gemma 3 4B": 511, "Jina CLIP v2": 2046}),
    ("circlestone-labs/Anima", {"Qwen3-0.6B": 512, "T5 ids": 511}),
]
# Where the chat template's closing turn starts being lost, measured the same
# way: klein's empty thinking block costs it four tokens more than Z-Anime's.
REAL_CLEAN = {"black-forest-labs/FLUX.2-klein-4B": 500, "SeeSee21/Z-Anime": 504}
FOX = "Cinematic photo of a red fox in fresh snow, golden hour"


def _clean(model_id):
    (spec,) = pe.encoders_for(model_id)
    return pe._read(spec, pe._tokenizers[spec.source], "prompt", FOX).clean


@pytest.mark.parametrize("model_id, budgets", REAL)
def test_real_tokenizers_agree_with_the_pipelines_own_truncation(monkeypatch, model_id, budgets):
    folders = {}
    for spec in pe.encoders_for(model_id):
        folder = _cached(spec.source)
        if folder is None:
            pytest.skip(f"{spec.source.label} is not in the Hub cache")
        folders[spec.source] = folder
    monkeypatch.setattr(pe, "_load_tokenizer", lambda source: pe._from_pretrained(folders[source]))
    reports, findings, detail = pe.measure(model_id, FOX, None)
    assert detail is None and findings == []
    assert {report.name: report.budget for report in reports} == budgets
    if model_id in REAL_CLEAN:
        assert _clean(model_id) == REAL_CLEAN[model_id]

    # Past every budget, the text before the cut is exactly what the
    # pipeline's own truncating tokenizer call keeps after its template.
    long = " ".join(f"forest{i % 7}" for i in range(700))
    for spec in pe.encoders_for(model_id):
        loaded = pe._tokenizers[spec.source]
        tokenizer = loaded.tokenizer
        reading = pe._read(spec, loaded, "prompt", long)
        prefix, suffix = pe._frame(spec, tokenizer, long, negative=False)
        limit = spec.max_length or tokenizer.model_max_length
        kept = tokenizer(prefix + long + suffix, truncation=True, max_length=limit)["input_ids"]
        content = len(kept) - tokenizer.num_special_tokens_to_add()
        assert reading.cut is not None
        read = tokenizer(prefix + long[: reading.cut], add_special_tokens=False)["input_ids"]
        assert len(read) == content

    # The authors' own examples fit their encoders. A template's slot that
    # stands as a whole comma item ("[lighting]") reads as A1111's brackets
    # here; the assembled check reports that span once, as the author's slot
    # (`prompt_check.checked`), so it is no warning of the encoder layer's.
    slots = model_defaults.guide_for(model_id)["slots"]
    for example in model_defaults._profile_for(model_id)["guide"]["examples"]:
        _, findings, _ = pe.measure(model_id, example, None)
        spans = {m.span() for pattern in slots for m in re.finditer(pattern, example, re.I)}
        assert not [
            finding for finding in findings
            if finding.severity == "warn"
            and not (finding.rule == "encoder.weighting" and (finding.start, finding.end) in spans)
        ], example[:60]


def test_real_clip_reads_its_control_tokens_in_any_case(monkeypatch):
    specs = pe.encoders_for("KBlueLeaf/Kohaku-XL-Zeta")
    folders = {spec.source: _cached(spec.source) for spec in specs}
    if None in folders.values():
        pytest.skip("Kohaku XL Zeta's tokenizers are not in the Hub cache")
    monkeypatch.setattr(pe, "_load_tokenizer", lambda source: pe._from_pretrained(folders[source]))
    prompt = "1girl, <|EndOfText|>, <|STARTOFTEXT|>, solo"
    _, findings, _ = pe.measure("KBlueLeaf/Kohaku-XL-Zeta", prompt, None)
    controls = rules(findings, "encoder.control_token")
    assert [prompt[f.start:f.end] for f in controls] == ["<|EndOfText|>", "<|STARTOFTEXT|>"]
    assert not rules(findings, "encoder.unknown_chars")


# Qwen-Image is not measured above, but its budget is the pipeline's own
# arithmetic: it drops its 34-token system turn and keeps 512, and the
# template's closing turn goes past 507.
FAKE_ONLY = [("Qwen/Qwen-Image", {"Qwen2.5-VL-7B": pe.QWEN_IMAGE_SEQUENCE})]
FAKE_CLEAN = {**REAL_CLEAN, "Qwen/Qwen-Image": 507}


@pytest.mark.parametrize("model_id, budgets", REAL + FAKE_ONLY)
def test_the_fake_tokenizers_budgets_are_the_real_ones(monkeypatch, model_id, budgets):
    # The API tests and the fake server take the fakes' budgets as the real.
    monkeypatch.setattr(pe, "_load_tokenizer", fakes.fake_tokenizer)
    reports, _, detail = pe.measure(model_id, FOX, None)
    assert detail is None
    assert {report.name: report.budget for report in reports} == budgets
    if model_id in FAKE_CLEAN:
        assert _clean(model_id) == FAKE_CLEAN[model_id]
