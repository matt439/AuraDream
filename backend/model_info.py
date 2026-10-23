"""Cold, compact catalogue facts from the app's sourced guides and profiles.

The size labels show upper recommended or documented areas, not starting sizes
or a measured quality ranking. Native, documented, trained and example bounds
retain their own labels. Notes distinguish author guidance from local advice.
Sources are backend/guides.py,
docs/anime-model-guide.md and (for Z-Anime's iteration advice) docs/verified.md;
Hugging Face access comes from the catalogue's gate word.
Download references and offline cache inventories are retained, with their
selected files/revisions and source URLs, in model-downloads.json.
"""
from functools import lru_cache
import json
from pathlib import Path, PurePosixPath
import time

from .launch import CHECKPOINTS, FORMATS
from .model_benchmarks import speed_for
from .model_defaults import defaults_for


_FLUX_SIZE = (
    "2 MP",
    "The FLUX.1 guide suggests rendering at 1–2 MP before Refine; the author "
    "documents 0.1–2 MP at any aspect ratio.",
)
_KLEIN_SIZE = (
    "2 MP",
    "Official klein examples use 1.05 MP. FLUX.2 guidance recommends up to "
    "2 MP and documents a 4 MP maximum; reference inputs are resized to about 1 MP.",
)
_ANIMA_SIZE = (
    "2.36 MP",
    "Anima's documented area span is 512-square to 1536-square, about 0.26–2.36 MP. "
    "Only a 1.05 MP square has been measured locally; equal area does not "
    "establish equal quality at every aspect ratio.",
)

# Exact catalogue selectors: no inherited Illustrious range on WAI or RouWei,
# no Turbo range on Krea Raw, and no bare 'anima' match catching Animagine.
_FACTS: dict[str, tuple[str, tuple[str, str]]] = {
    "black-forest-labs/FLUX.1-Krea-dev": ("Photorealism", (
        "1.64 MP",
        "Krea recommends 1024–1280 sides; 1280-square is 1.6384 MP. "
        "Shape still matters; the broader "
        "FLUX.1 family documents up to 2 MP.",
    )),
    "krea/Krea-2-Turbo": ("General", (
        "4 MP",
        "Krea documents 1024–2048 sides, with a 2048-square example (4.19 MP). "
        "Pretraining topped out at 1024 pixels; 1 MP is the most familiar size.",
    )),
    "krea/Krea-2-Raw": ("General", (
        "1 MP",
        "Raw was trained up to 1k, about 1 MP. Larger sizes are outside that "
        "training; Krea recommends Turbo for inference and Raw for training.",
    )),
    "cagliostrolab/animagine-xl-4.0": ("Anime", (
        "1 MP",
        "Author-recommended square, portrait and landscape shapes cover "
        "0.98–1.05 MP. Larger areas can degrade; use the highlighted aspect presets.",
    )),
    "KBlueLeaf/Kohaku-XL-Zeta": ("Anime", (
        "1 MP",
        "The author recommends 1024-square (1.05 MP) or a similar total pixel "
        "count at another aspect ratio. Larger areas are locally unmeasured.",
    )),
    "OnomaAIResearch/Illustrious-XL-v2.0": ("Anime", (
        "2.25 MP",
        "The authors describe about 1–2.25 MP native output, with varied aspect "
        "ratios; the larger training square is exactly 2.36 MP. About 4 MP is "
        "possible with quality loss. The 2.25 MP native bound is the authors' "
        "approximate wording, not a measured optimum.",
    )),
    "LyliaEngine/waiIllustriousSDXL_v170": ("Stylised anime", (
        "1.38 MP",
        "The author's 1.05–1.38 MP reference sizes include a "
        "portrait. These are examples, not a published operating range or maximum. "
        "The author's showcase uses a separate hires pass.",
    )),
    "Minthy/RouWei-0.8": ("Anime", (
        "1.5 MP",
        "The author recommends about 1–1.5 MP, any aspect ratio with sides "
        "divisible by 64. App presets use multiples of 16, so not every preset "
        "follows that divisibility recommendation.",
    )),
    "Laxhar/noobai-XL-Vpred-1.0": ("Anime", (
        "1 MP",
        "The author recommends about 1 MP: seven listed shapes cover "
        "1.01–1.05 MP, with 832×1216 emphasized. No larger native range is established.",
    )),
    "SeeSee21/Z-Anime": ("Anime", (
        "2 MP",
        "Local checks recommend up to 2 MP for practical iteration. The author "
        "documents 0.26–4.2 MP at any aspect ratio, a broader supported range. "
        "The coherent 4.19 MP sample took 8 minutes 39 seconds and nearly filled the card.",
    )),
    "circlestone-labs/Anima": ("Stylised anime", _ANIMA_SIZE),
    "circlestone-labs/Anima@aesthetic-v1.1": ("Stylised anime", _ANIMA_SIZE),
    "Gazingstars123/Anima-2.9B": ("Stylised anime", (
        "1.77 MP",
        "The preview author's recommended 812×1216 and 1152×1536 shapes span "
        "about 0.99–1.77 MP. The listed 1536-square size (2.36 MP) is marked "
        "'iffy'; it is not the recommended upper figure here. This variant "
        "has not been measured locally.",
    )),
    "duongve/NetaYume-Lumina-Image-2.0": ("Stylised anime", (
        "1.57 MP",
        "Inherited/v2 examples cover 1.05–1.57 MP, "
        "including square and portrait; they establish no v4 operating range "
        "or maximum. Larger areas are locally unmeasured.",
    )),
    "NewBie-AI/NewBie-image-Exp0.1": ("Anime", (
        "1.05 MP",
        "The author's quickstart and local check use a 1.05 MP square. "
        "This is an example, not a published minimum, maximum or square-only "
        "requirement; higher areas are locally unmeasured.",
    )),
    "black-forest-labs/FLUX.1-dev": ("General", _FLUX_SIZE),
    "black-forest-labs/FLUX.1-schnell": ("General", _FLUX_SIZE),
    "black-forest-labs/FLUX.2-klein-4B": ("General", _KLEIN_SIZE),
    "black-forest-labs/FLUX.2-klein-9B": ("General", _KLEIN_SIZE),
    "black-forest-labs/FLUX.2-klein-base-4B": ("General", _KLEIN_SIZE),
    "black-forest-labs/FLUX.2-klein-base-9B": ("General", _KLEIN_SIZE),
    "Tongyi-MAI/Z-Image-Turbo": ("General", (
        "2 MP",
        "The guide puts 1–2 MP within the training range. The author documents "
        "512-square to 2048-square total area at any aspect ratio; about 4 MP "
        "is the documented ceiling, beyond the training range.",
    )),
    "Qwen/Qwen-Image": ("Text rendering", (
        "1.76 MP",
        "The author's reference shapes cover about 1.5–1.76 MP; the largest "
        "is 1328-square, also the upper training size. Several 1.5 MP presets "
        "match the reference shapes exactly. 2 MP is just "
        "above those examples and 4 MP well above them.",
    )),
    "baidu/ERNIE-Image-Turbo": ("General", (
        "1 MP",
        "Baidu documents seven recommended shapes at about 1 MP. Larger "
        "local renders have completed, but those do not establish a quality "
        "recommendation beyond the author's sizes.",
    )),
}


# Size evidence weaker or different than a recommendation stays visible even
# when its tooltip is closed. Never turn an author's example into a maximum.
_SIZE_LABELS = {
    "krea/Krea-2-Turbo": "Documented max",
    "krea/Krea-2-Raw": "Trained max",
    "OnomaAIResearch/Illustrious-XL-v2.0": "Native max",
    "circlestone-labs/Anima": "Documented max",
    "circlestone-labs/Anima@aesthetic-v1.1": "Documented max",
    "LyliaEngine/waiIllustriousSDXL_v170": "Reference size",
    "Qwen/Qwen-Image": "Reference size",
    "duongve/NetaYume-Lumina-Image-2.0": "Largest example",
    "NewBie-AI/NewBie-image-Exp0.1": "Largest example",
}


# Checkpoints whose torchao builds are larger than this 16 GB card: the
# official repo's row says so, the lite-infer 4-bit row does not. From the
# catalogue's own record (launch.CHECKPOINTS' comment, docs/verified.md).
_FIT = {
    "Qwen/Qwen-Image": ("Over 16 GB", (
        "Larger than a 16 GB card at FP8, INT8 or BF16 (the 20B transformer "
        "alone is about 20 GB at one byte a weight). Only the 4-bit Nunchaku build (WSL) has run "
        "on this card, at about 12 GiB; sequential offload is unmeasured."
    )),
}


_LOGIN = "then sign in once with `uv run hf auth login` and a read token"
# Hugging Face access before the first download, from the catalogue's own gate
# word (launch.CHECKPOINTS, as the --menu prints it). lite-infer's re-uploads
# are not gated.
_ACCESS = {
    "click-through": ("Accept licence", (
        "Gated on Hugging Face: accept the licence on the model page (a click-through), "
        f"{_LOGIN}, before the first download."
    )),
    "manual": ("Needs approval", (
        "Gated on Hugging Face: request access on the model page. Its authors approve "
        f"each request by hand, which can take days; once approved, {_LOGIN}."
    )),
}
_OPEN = ("Open", "Not gated on Hugging Face: downloads without a sign-in or licence step.")
_LITE_OPEN = ("Open", (
    "This lite-infer 4-bit build is not gated on Hugging Face: downloads without a "
    "sign-in or licence step. The model-page link opens the official repo."
))


@lru_cache(maxsize=1)
def download_references() -> dict:
    """Retained metadata only; no Hub calls or model imports."""
    return json.loads(Path(__file__).with_name("model-downloads.json").read_text(encoding="utf-8"))


_cache_revision = 0


def cache_revision() -> int:
    return _cache_revision


def invalidate_cache_facts() -> None:
    """A failed load can have downloaded some files too."""
    global _cache_revision
    _cached_state.cache_clear()
    _cache_revision += 1


def _relative_file(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(path.parts) and not path.is_absolute() and all(part not in {".", ".."} for part in path.parts) and "\\" not in name and ":" not in name


def _file_set_state(files: dict[str, int], indexes: dict, find) -> str:
    """Check names, sizes and small JSON descriptors, never weight contents.

    The retained list includes every shard and tokenizer asset the selected
    loader downloads. An index or descriptor cannot stand in for its files.
    """
    present = False
    complete = True
    for name, expected_size in files.items():
        if not _relative_file(name) or type(expected_size) is not int or expected_size <= 0:
            return "unknown"
        path = find(name)
        if path is None:
            complete = False
            continue
        present = True
        if path.stat().st_size != expected_size:
            complete = False
            continue
        if name in indexes:
            # A cached descriptor from another recipe, or a damaged shard map,
            # cannot validate the expected files merely by having the same size.
            if expected_size > 2_000_000:
                return "unknown"
            try:
                if json.loads(path.read_text(encoding="utf-8")) != indexes[name]:
                    complete = False
            except (ValueError, UnicodeError):
                complete = False
    return "complete" if complete and files else "partial" if present else "absent"


def _hub_source_state(source: dict, names: list[str] | None, cache_dir: str) -> str:
    if names is None:
        return "unknown"
    from huggingface_hub import try_to_load_from_cache

    unexpected_revision = False

    def find(name):
        nonlocal unexpected_revision
        cached = try_to_load_from_cache(source["repo_id"], name, revision=source["revision"], cache_dir=cache_dir)
        if not isinstance(cached, str):
            return None
        path = Path(cached)
        if not path.is_file():
            return None
        # main is mutable. An inventory observed at one commit proves nothing
        # about a different cached revision, even when some files share sizes.
        if path.parents[len(PurePosixPath(name).parts) - 1].name != source["commit"]:
            unexpected_revision = True
            return None
        return path

    state = _file_set_state({name: source["files"][name] for name in names}, source["indexes"], find)
    return "unknown" if unexpected_revision else state


def _conversion_target(model_id: str):
    from .backends import anima_loading, netayume_loading, newbie_loading, single_file

    if model_id.lower() in single_file._CHECKPOINTS:
        return single_file._target(model_id)
    if anima_loading.matches(model_id):
        return anima_loading._target(model_id)
    if model_id.lower() == netayume_loading.MODEL_ID.lower():
        return netayume_loading._target()
    if model_id.lower() == newbie_loading.MODEL_ID.lower():
        return newbie_loading._target()
    return None


@lru_cache(maxsize=256)
def _cached_state(model_id: str, cache_dir: str, converted_home: str, minute: int, revision: int) -> tuple[str, str]:
    # converted_home and minute are cache keys: a changed HF_HOME and files
    # downloaded outside this process are rechecked without scanning the cache.
    references = download_references()
    recipe = references["models"].get(model_id)
    if recipe is None:
        return "unknown", "Could not establish that every required model file is cached."
    unverified_conversion = False
    try:
        if recipe["kind"] == "converted":
            target, marker, recipe_text = _conversion_target(model_id)
            converted = recipe["converted"]
            if json.loads(recipe_text) != converted["recipe"]:
                return "unknown", "The cached-file reference does not cover this conversion recipe."
            if target.exists() and not converted.get("files"):
                # A newly supported converter can have a complete source
                # inventory before its output has ever been serialized.
                # Its marker alone cannot establish complete converted files.
                unverified_conversion = True
            elif target.exists():
                marker_ok = marker.is_file() and marker.read_text(encoding="utf-8") == recipe_text
                state = _file_set_state(converted["files"], converted["indexes"],
                                        lambda name: target / name if (target / name).is_file() else None)
                if marker_ok and state == "complete":
                    return "complete", "The required converted model files are on disk. A first load still takes time."
                return "partial", "Some converted model files are missing or incomplete; loading may need repair."
        states = [_hub_source_state(references["sources"][item["source"]], item["files"], cache_dir)
                  for item in recipe["sources"]]
        if all(state == "complete" for state in states):
            if recipe["kind"] == "converted":
                return "complete", "The source files are cached; the first load still converts them locally."
            return "complete", "The required model files for the checked cached revision are on disk. A first load still takes time."
        if unverified_conversion:
            return "unknown", "The converted files have no complete size reference, and not every required source file is cached."
        if "unknown" in states:
            return "unknown", "Could not establish that every required model file is cached."
        if any(state != "absent" for state in states):
            return "partial", "Some required model files are missing or incomplete. The remaining download size is unknown."
        return "absent", "The required model files are not cached. A first load downloads them."
    except (OSError, ValueError, UnicodeError, KeyError, TypeError):
        return "unknown", "Could not establish that every required model file is cached."


def download_facts(model_id: str) -> dict[str, str]:
    references = download_references()
    recipe = references["models"].get(model_id)
    known = recipe is not None and all(item["files"] is not None for item in recipe["sources"])
    if known:
        size = sum(references["sources"][item["source"]]["files"][name]
                   for item in recipe["sources"] for name in item["files"])
        amount = f"{size / 1_000_000_000:.1f}".rstrip("0").rstrip(".")
        download = f"~{amount} GB" if size >= 100_000_000 else "<0.1 GB"
        observed_at = recipe.get("observed_at", references["observed_at"])
        note = (f"About {amount} GB for the full image-model download, from selected-file metadata checked "
                f"{observed_at}. This is not the amount still missing; cached files may reduce it. "
                "Future unpinned repository updates may change it.")
    else:
        download = "Unknown"
        note = "The full image-model download size is unknown; no complete selected-file size reference is available."
    from huggingface_hub.constants import HF_HOME, HF_HUB_CACHE

    state, cache_note = _cached_state(model_id, str(HF_HUB_CACHE), str(HF_HOME),
                                     int(time.monotonic() // 60), _cache_revision)
    return {"download": download, "download_note": note,
            "cache": {"complete": "On disk", "partial": "Part cached", "absent": "Not cached", "unknown": "Unknown"}[state],
            "cache_state": state, "cache_note": cache_note}


def details_for(
    model_id: str, quantization: str | None = None, offload: str = "model",
) -> dict[str, str]:
    checkpoint = next((
        entry for entry in CHECKPOINTS
        if model_id == entry.repo or any(model_id == entry.lite_repo(fmt) for fmt, _ in FORMATS)
    ), None)
    if checkpoint is None:
        return {
            "style": "Custom", "refine": "After load",
            "refine_note": "The loaded pipeline decides whether Refine is available.",
            "megapixels": "Unknown",
            "megapixels_label": "Max recommended",
            "megapixels_note": "No catalogue size guidance; check the checkpoint's model card.",
            "favourite_id": model_id, "url": "",
            "access": "Unknown",
            "access_note": "Not in the catalogue; its model page says whether it is gated.",
            "fit": "", "fit_note": "",
            **download_facts(model_id),
            **speed_for(model_id, quantization, offload),
        }
    style, (megapixels, note) = _FACTS[checkpoint.repo]
    lite = model_id != checkpoint.repo
    access, access_note = _LITE_OPEN if lite else next((
        value for prefix, value in _ACCESS.items() if checkpoint.gate.startswith(prefix)
    ), _OPEN)
    profile = defaults_for(model_id)
    fit, fit_note = ("", "") if lite else _FIT.get(checkpoint.repo, ("", ""))
    return {
        "style": style,
        "refine": "Yes" if profile["refine_supported"] else "No",
        "refine_note": (
            "This app has a compatible image-to-image Refine pass for this checkpoint."
            if profile["refine_supported"] else
            "This app has no compatible image-to-image Refine pass for this checkpoint."
        ),
        "megapixels": megapixels,
        "megapixels_label": _SIZE_LABELS.get(checkpoint.repo, "Max recommended"),
        "megapixels_note": note + " Direct generation, not Refine; no GPU memory guarantee.",
        # Lite builds share a favourite, but Anima's two releases stay distinct.
        "favourite_id": checkpoint.repo,
        "url": f"https://huggingface.co/{checkpoint.repo.split('@', 1)[0]}",
        "access": access,
        "access_note": access_note,
        "fit": fit,
        "fit_note": fit_note,
        **download_facts(model_id),
        **speed_for(model_id, quantization, offload),
    }
