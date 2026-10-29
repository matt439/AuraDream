"""Getting a checkpoint onto the card: what to quantize, what to leave out of
the download, and where the pipeline's components live.

Everything here runs ahead of `from_pretrained`, or right after it, and reads
the checkpoint's small config files rather than its weights - so the refusals
land before a multi-GB download rather than minutes into one. `settings` is a
parameter rather than an import: the backend's own `settings` is the one the
tests replace, and it hands that through.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from .. import loras
    from ..config import Settings

log = logging.getLogger(__name__)

# torchao weight-only modes the backend applies itself while loading.
QUANT_BACKENDS = ("fp8", "int8")
# A checkpoint quantized before upload. `transformer/config.json` carries this
# `quant_method`, and diffusers picks the quantizer up from there on its own.
NUNCHAKU = "nunchaku"
NUNCHAKU_METHOD = "nunchaku_lite"
NUNCHAKU_EXAMPLE = "lite-infer/krea-2-turbo-nunchaku-lite-nvfp4_r32-bnb4-text-encoder"

# Keys in model_index.json worth quantizing. The VAE stays in bf16: it is small
# and quantizing it shows up as visible decode artefacts.
QUANTIZABLE_PREFIXES = ("transformer", "unet", "text_encoder")

# Local subclasses still belong to their base library for quantization. Their
# serialized module paths let Diffusers import them without remote code.
COMPONENT_LIBRARY_OWNERS = {
    "backend.backends.newbie_transformer": "diffusers",
    "backend.backends.newbie_text_encoder": "transformers",
}

# Components image generation never runs, loaded as None initially -
# which diffusers allows for a class's `_optional_components`, and which keeps
# their folders out of the download too, *through the concrete class*: the
# base `DiffusionPipeline.from_pretrained` filters passed components against
# its own empty `_optional_components` (`_get_signature_keys` is a classmethod
# on `cls`, diffusers 0.40), takes `pe=None` for a pipeline kwarg, downloads
# the folder and loads the model - seen on the card, 7.7 GB pulled and 11 GB
# of RAM taken, before load() switched class. ErnieImagePipeline carries a 7.7 GB
# bf16 Ministral prompt rewriter (`pe`) and, unless `use_pe` is off, samples
# a longer prompt through it at temperature 0.6 before every image: the
# prompt the row records would not be the one the model saw. The explicit
# Expand action loads these two lazily, writes the result into the editor, and
# generate() still turns the hidden switch off.
UNUSED_COMPONENTS: dict[str, tuple[str, ...]] = {
    "ErnieImagePipeline": ("pe", "pe_tokenizer"),
}

INSTRUCTION_EDITOR_CLASSES = frozenset({"FluxKontextPipeline", "QwenImageEditPipeline", "QwenImageEditPlusPipeline"})

# The rewriter's weights, the bulk of what a first Expand downloads: the same
# file in every ERNIE-Image repository the catalogue names (7,663,346,520
# bytes in Baidu's Turbo and base repos, 8 fewer in lite-infer's re-upload).
# The page names the download before it is started (`ModelStatus.prompt_expander`).
PROMPT_EXPANDER_WEIGHTS = "pe/model.safetensors"
PROMPT_EXPANDER_BYTES = 7_663_346_520

# accelerate's `init_empty_weights()`, which builds the adapter fit's
# denoiser on the meta device (`denoiser`), works by replacing
# `torch.nn.Module.register_parameter` - for the whole process, every thread,
# while it lasts - and puts back whatever it found on entry. A load on another
# thread meanwhile had its parameters moved to meta as torchao registered
# them (`Float8Tensor.__new__() got multiple values for argument 'qdata'`,
# three loads out of three pressed within seconds of opening the page on
# WAI), and two of those builds overlapping can leave the patch in place for
# good: the one that leaves last restores the other's patch. So they never
# overlap a load or each other. A load waits for a build in progress, which
# takes tens of milliseconds; a build asked for while a load runs - minutes -
# does not wait but raises `LoadInProgress`, and the fit goes unchecked until
# the load, as for a config out of reach. The conversions the load runs
# (Anima, NetaYume, NewBie, Illustrious, WAI) patch the same way inside it,
# under the load's own hold. Every other weight load in the process - Expand's
# rewriter, the translator - runs with the pipeline loaded, when the fit reads
# the live denoiser and builds nothing.
_weights_guard = threading.Condition()
_loads_running = 0
_skeleton_building = False


class LoadInProgress(RuntimeError):
    """A meta-device build asked for while a load holds the weights guard."""


@contextmanager
def weights_loading() -> Iterator[None]:
    """Held across a load: no meta-device build starts until it ends, and one
    already under way is waited out first."""
    global _loads_running
    with _weights_guard:
        _loads_running += 1
        while _skeleton_building:
            _weights_guard.wait()
    try:
        yield
    finally:
        with _weights_guard:
            _loads_running -= 1
            _weights_guard.notify_all()


@contextmanager
def empty_weights() -> Iterator[None]:
    """accelerate's `init_empty_weights()`, one at a time and never beside a
    load (above); raises `LoadInProgress` rather than wait a load out."""
    global _skeleton_building
    from accelerate import init_empty_weights

    with _weights_guard:
        while True:
            if _loads_running:
                raise LoadInProgress("a model load is in progress")
            if not _skeleton_building:
                break
            _weights_guard.wait()
        _skeleton_building = True
    try:
        with init_empty_weights():
            yield
    finally:
        with _weights_guard:
            _skeleton_building = False
            _weights_guard.notify_all()


def prompt_expander_cached(model_id: str) -> bool:
    """Whether the rewriter's weights are on disk already: in a local tree, or
    in the Hub cache. File checks only - no network - so the status poll can
    ask it every few seconds, as it asks the critic's `ready()`."""
    import os

    if os.path.isdir(model_id):
        return os.path.isfile(os.path.join(model_id, *PROMPT_EXPANDER_WEIGHTS.split("/")))
    from huggingface_hub import try_to_load_from_cache

    try:
        found = try_to_load_from_cache(model_id, PROMPT_EXPANDER_WEIGHTS)
    except (OSError, ValueError):
        return False
    return isinstance(found, str) and os.path.isfile(found)

# Pipeline-level from_pretrained in Diffusers 0.40 does not resolve a
# subfolder. Materialize just this pinned tree in the Hub cache and hand the
# ordinary loader its local directory; keep the public id in runtime metadata.
NESTED_MODELS = {
    "SeeSee21/Z-Anime": ("0f5fb51464638a2f7328a1d74590281e63e1fde2", "diffusers"),
}
# The official root component tree avoids converting the duplicate single
# file. Its packaged scheduler is epsilon despite the author's v-pred recipe;
# component_overrides corrects it without writing into the Hub snapshot.
NOOB_MODEL_ID = "Laxhar/noobai-XL-Vpred-1.0"
NOOB_REVISION = "66aa55e3469c27c29a89813cd35dd95fb7485fa1"
# The nested Z-Anime encoder is scaled FP8 without a quantization_config:
# Transformers silently drops its *.scale tensors. The author identifies the
# default encoder as standard Z-Image Qwen3-4B. Load that parent's original
# BF16 component and apply our normal owned Transformers quantization config.
Z_ANIME_TEXT_ENCODER = {
    "pretrained_model_name_or_path": "Tongyi-MAI/Z-Image",
    "revision": "04cc4abb7c5069926f75c9bfde9ef43d49423021",
    "subfolder": "text_encoder",
}


class _NestedSnapshotOptions(TypedDict, total=False):
    ignore_patterns: list[str]


def resolve_model(model_id: str, *, weights: bool = False, quantization: str | None = None) -> str:
    """Resolve a pinned component tree, reading configs before any weights.

    The first pass supplies the manifest and component configs for quantizer
    checks. Only after those checks pass does load() request the whole tree.
    Pin both passes to one revision so configs cannot race a weight update.
    Other Hub ids and explicit local directories pass through unchanged.
    """
    from ..instruction_edit import KONTEXT_V1

    # Exact, as the load entry compares it: a case variant never reaches the
    # pinned editor tree (`require_model_available` refused it already).
    if model_id == KONTEXT_V1.model_id:
        return _resolve_kontext(weights=weights)
    if model_id.lower() == NOOB_MODEL_ID.lower():
        if quantization == NUNCHAKU:
            raise RuntimeError(
                "NoobAI-XL V-pred 1.0 is not a Nunchaku Lite checkpoint. "
                "Use AURADREAM_QUANT=fp8, int8 or none."
            )
        from huggingface_hub import snapshot_download

        patterns = ["model_index.json", "*/config.json", "scheduler/scheduler_config.json"]
        if weights:
            patterns += [
                "tokenizer/*", "tokenizer_2/*",
                "unet/diffusion_pytorch_model.safetensors",
                "vae/diffusion_pytorch_model.safetensors",
                "text_encoder/model.safetensors", "text_encoder_2/model.safetensors",
            ]
        log.info("resolving %s (%s)", NOOB_MODEL_ID, "weights" if weights else "configs")
        return snapshot_download(NOOB_MODEL_ID, revision=NOOB_REVISION, allow_patterns=patterns)
    source = NESTED_MODELS.get(model_id)
    if source is None:
        return model_id
    from huggingface_hub import snapshot_download

    revision, folder = source
    patterns = [f"{folder}/*"] if weights else [
        f"{folder}/model_index.json",
        f"{folder}/*/config.json",
        f"{folder}/scheduler/scheduler_config.json",
    ]
    log.info("resolving %s/%s (%s)", model_id, folder, "weights" if weights else "configs")
    options: _NestedSnapshotOptions = (
        {"ignore_patterns": [f"{folder}/text_encoder/*"]} if weights and model_id == "SeeSee21/Z-Anime" else {}
    )
    snapshot = snapshot_download(model_id, revision=revision, allow_patterns=patterns, **options)
    return str(Path(snapshot) / folder)


def _resolve_kontext(*, weights: bool) -> str:
    """FLUX.1 Kontext [dev] at its pinned revision: configs, then weights.

    The configs pass fetches only the six files the config pin covers and
    refuses any that differ (`require_pinned(validate_configs(...))`) before
    a weight is asked for. The weights pass is the call stage 3 measured,
    `FluxKontextPipeline.download` at the same revision - it leaves the
    repository's root single file on the Hub - and the pin is checked again
    over the tree it returned. Reaching here is no permission: the load entry
    has already asked the gate and the measured envelope.
    """
    from huggingface_hub import snapshot_download

    from ..instruction_edit import KONTEXT_V1
    from ..instruction_edit_preflight import CONFIG_PATHS

    if not weights:
        log.info("resolving %s (configs)", KONTEXT_V1.model_id)
        snapshot = snapshot_download(KONTEXT_V1.model_id, revision=KONTEXT_V1.revision, allow_patterns=list(CONFIG_PATHS))
        return require_kontext_pin(str(snapshot))
    import diffusers

    log.info("resolving %s (weights)", KONTEXT_V1.model_id)
    editor = getattr(diffusers, KONTEXT_V1.pipeline_class)
    snapshot = editor.download(KONTEXT_V1.model_id, revision=KONTEXT_V1.revision)
    return require_kontext_pin(str(snapshot))


def require_kontext_pin(snapshot: str) -> str:
    """Read the six pinned configs from `snapshot`, bounded, and require the pin.

    A file that cannot be read, is over the preflight's size bound or is not
    JSON is refused in the preflight's own words: the path only, never its
    content or a hash.
    """
    from ..instruction_edit import KONTEXT_V1
    from ..instruction_edit_preflight import CONFIG_PATHS, MAX_CONFIG_BYTES, require_pinned, validate_configs

    supplied: dict[str, object] = {}
    for path in CONFIG_PATHS:
        try:
            with open(Path(snapshot, *path.split("/")), "rb") as stream:
                raw = stream.read(MAX_CONFIG_BYTES + 1)
            if len(raw) > MAX_CONFIG_BYTES:
                raise ValueError("over the config bound")
            supplied[path] = json.loads(raw.decode("utf-8"))
        except (OSError, ValueError, RecursionError) as exc:
            raise ValueError(f"kontext-v1 config preflight refused {path}") from exc
    require_pinned(validate_configs(KONTEXT_V1.model_id, KONTEXT_V1.revision, supplied))
    return snapshot


def component_overrides(model_id: str, quant: Any, *, resolved_model_id: str | None = None) -> dict[str, Any]:
    """Pinned replacements for incorrectly packaged upstream components."""
    if model_id.lower() == NOOB_MODEL_ID.lower():
        from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler

        # The author prescribes Euler, v-prediction and zero terminal SNR.
        # Diffusers recommends trailing spacing for that zero-SNR endpoint;
        # the supplied leading spacing otherwise starts at 946 of 999 for
        # 28 steps. Keep this explicit integration choice with the recipe.
        source = resolved_model_id or resolve_model(model_id)
        return {"scheduler": EulerDiscreteScheduler.from_pretrained(
            source, subfolder="scheduler", local_files_only=True,
            prediction_type="v_prediction", rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
        )}
    if model_id != "SeeSee21/Z-Anime":
        return {}
    import torch
    from transformers import Qwen3ForCausalLM

    kwargs: dict[str, Any] = {"dtype": torch.bfloat16}
    if quant is not None:
        kwargs["quantization_config"] = quant.quant_mapping["text_encoder"]
    log.info("loading Z-Anime's standard Qwen3 encoder from its Z-Image parent")
    return {"text_encoder": Qwen3ForCausalLM.from_pretrained(**Z_ANIME_TEXT_ENCODER, **kwargs)}


def importable(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def lora_refusal(model_id: str) -> str | None:
    """Why this checkpoint takes no user LoRA whatever the installation, or None.

    A fact about the integration, answerable from the id before anything
    downloads: the Anima variants' and NewBie's pipelines are the app's own, with no
    adapter path. `DiffusersBackend.lora_unavailable_reason` asks this first,
    and the browser checks' fake server asks it of the id it claims, so the
    fake refuses an adapter exactly where the real server does. The ids are
    the loaders' own constants, so the two cannot drift apart.
    """
    from . import anima_loading, newbie_loading

    key = model_id.lower()
    if key == anima_loading.MODEL_ID.lower():
        return "User LoRAs are not supported by this Anima Turbo integration"
    if key == anima_loading.AESTHETIC_MODEL_ID.lower():
        return "User LoRAs are not supported by this Anima Aesthetic integration"
    if key == anima_loading.PREVIEW_MODEL_ID.lower():
        return "User LoRAs are not supported by this Anima 2.9B Preview integration"
    if key == newbie_loading.MODEL_ID.lower():
        return "User LoRAs are not supported by this NewBie integration"
    return None


def hub_file(model_id: str, filename: str) -> str:
    """Path to one file of a checkpoint, downloading only that file.

    A local directory is resolved locally: `from_pretrained` branches on
    `os.path.isdir` itself, but the callers here run *before* it - the
    fp8/int8 component scan and the Nunchaku preflight - and a directory id
    handed to the Hub client died with a validation error naming the model id
    and not the quantization.
    """
    import os

    if os.path.isdir(model_id):
        return os.path.join(model_id, filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model_id, filename)


def component_libraries(model_id: str) -> dict[str, str]:
    """Map component name -> owning library, from the repo's model_index.json.

    Entries look like `"text_encoder": ["transformers", "Qwen3VLModel"]`, which
    is exactly what decides whether a component needs the diffusers or the
    transformers flavour of TorchAoConfig.
    """
    with open(hub_file(model_id, "model_index.json"), encoding="utf-8") as fh:
        index = json.load(fh)

    libraries: dict[str, str] = {}
    for name, spec in index.items():
        # Only ["library", "ClassName"] pairs are components. Krea 2 also
        # carries plain config values, and `text_encoder_select_layers`
        # matches the prefix below without being a component at all.
        if name.startswith("_") or not isinstance(spec, (list, tuple)) or len(spec) != 2:
            continue
        library, component_class = spec
        if not isinstance(library, str) or not isinstance(component_class, str):
            continue
        if name.startswith(QUANTIZABLE_PREFIXES) and library:
            libraries[name] = COMPONENT_LIBRARY_OWNERS.get(library, library)
    return libraries


def unused_components(model_id: str) -> tuple[str | None, dict[str, None]]:
    """The pipeline class this checkpoint's index names, and the components
    of it to load as None.

    Read ahead of the load, since that is the only time diffusers takes the
    instruction. An index that cannot be read is not a failure here: the
    whole pipeline loads, which is what `from_pretrained` was about to do.
    """
    try:
        with open(hub_file(model_id, "model_index.json"), encoding="utf-8") as fh:
            index = json.load(fh)
    except Exception as exc:  # noqa: BLE001 - an optimisation, not a gate
        log.debug("model_index.json unavailable ahead of the load: %s", exc)
        return None, {}
    class_name = index.get("_class_name")
    names = UNUSED_COMPONENTS.get(class_name, ())
    return class_name, {name: None for name in names if name in index}


def require_custom_pipeline_class(model_id: str) -> str | None:
    """Inspect non-catalogue ids strictly before pipeline weights load.

    ``unused_components`` is a best-effort download optimization and cannot
    decide admission: its failure intentionally means load every component.
    A custom/local id must instead supply a readable index with a concrete
    pipeline class. Unknown/unreadable identity refuses, never retries through
    ``from_pretrained`` after a speculative download. Existing catalogue and
    its enumerated Nunchaku format ids retain their established loader path.
    """
    from ..instruction_edit import editor_class_refusal
    from ..launch import CHECKPOINTS, FORMATS

    known = {checkpoint.repo.casefold() for checkpoint in CHECKPOINTS}
    known.update(
        repo.casefold() for checkpoint in CHECKPOINTS for fmt, _ in FORMATS
        if (repo := checkpoint.lite_repo(fmt)) is not None
    )
    if model_id.casefold() in known:
        return None
    try:
        with open(hub_file(model_id, "model_index.json"), encoding="utf-8") as stream:
            index = json.load(stream)
    except Exception as exc:
        # Keep the original exception chain for the existing Hub/offline load
        # error mapping; do not echo arbitrary index contents or fetch errors.
        raise ValueError("Cannot inspect this custom checkpoint's pipeline class; no weights were loaded.") from exc
    class_name = index.get("_class_name") if isinstance(index, dict) else None
    if not isinstance(class_name, str) or not class_name.isidentifier():
        raise ValueError("A custom checkpoint needs a valid pipeline class in model_index.json; no weights were loaded.")
    if class_name in INSTRUCTION_EDITOR_CLASSES:
        raise ValueError(editor_class_refusal(class_name))
    return class_name


def quantization_config(settings: "Settings") -> Any:
    """Build a pipeline-level torchao config, or None for full bf16.

    diffusers and transformers each ship their own `TorchAoConfig` and each
    validates only its own class, so components need a config instance from
    their owning library, built from the same torchao `AOBaseConfig`. The
    single `quant_backend=` shorthand raises on any pipeline whose text
    encoder is a transformers model, because those signatures disagree.
    """
    if settings.quantization == NUNCHAKU:
        nunchaku_preflight(settings)
        log.info("%s is pre-quantized; loading with its own quantization config", settings.model_id)
        return None
    if settings.quantization not in QUANT_BACKENDS:
        return None

    from diffusers.quantizers.pipe_quant_config import PipelineQuantizationConfig
    from diffusers.quantizers.quantization_config import TorchAoConfig as DiffusersTorchAoConfig
    from torchao.quantization import Float8WeightOnlyConfig, Int8WeightOnlyConfig
    from transformers import TorchAoConfig as TransformersTorchAoConfig

    # Both are one byte per weight; int8 tends to be more accurate, fp8 keeps
    # more dynamic range for outlier channels.
    make = Float8WeightOnlyConfig if settings.quantization == "fp8" else Int8WeightOnlyConfig
    config_for = {
        "transformers": TransformersTorchAoConfig,
        "diffusers": DiffusersTorchAoConfig,
    }

    mapping = {}
    for component, library in component_libraries(settings.model_id).items():
        cls = config_for.get(library)
        if cls is None:
            log.warning("skipping %s: unknown library %r", component, library)
            continue
        refuse_prequantized(settings, component)
        mapping[component] = cls(quant_type=make())

    if not mapping:
        log.warning("no quantizable components found; loading in full bf16")
        return None
    log.info("quantizing %s to %s", sorted(mapping), settings.quantization)
    return PipelineQuantizationConfig(quant_mapping=mapping)


def refuse_prequantized(settings: "Settings", component: str) -> None:
    """The mirror image of `nunchaku_preflight`: torchao on top of a repo
    that is already quantized.

    `AURADREAM_QUANT=fp8` pointed at a Nunchaku Lite re-upload downloads ~9 GB and
    then dies deep inside transformers ("quantized with BitsAndBytesConfig
    but you are passing a TorchAoConfig"). One small config file says so up
    front. Checked per quantizable component, since the text encoder and the
    transformer name their quantizers separately.
    """
    with open(hub_file(settings.model_id, f"{component}/config.json"), encoding="utf-8") as fh:
        method = (json.load(fh).get("quantization_config") or {}).get("quant_method")
    if method:
        raise RuntimeError(
            f"{settings.model_id} is already quantized ({component}: quant_method="
            f"{method!r}); AURADREAM_QUANT={settings.quantization} would stack torchao on top "
            "and fail after the download. Use AURADREAM_QUANT=nunchaku (Linux) for a Nunchaku "
            "Lite repo, or point AURADREAM_MODEL_ID at an unquantized checkpoint."
        )


def nunchaku_preflight(settings: "Settings") -> None:
    """Refuse before the download, with the reason, when Nunchaku cannot run.

    Each way the mode can be broken otherwise surfaces minutes into a load,
    or not at all: the `kernels` package has no Windows builds and fails
    from inside the weight loader; bitsandbytes holds these checkpoints'
    NF4 text encoders; and a repo that is not pre-quantized loads without
    complaint in full bf16 and runs out of memory at the first image.
    """
    if sys.platform != "linux":
        raise RuntimeError(
            "AURADREAM_QUANT=nunchaku needs Linux: the Nunchaku Lite kernels are "
            "published for x86_64 Linux only. On Windows, run the server from "
            "WSL 2 (bash wsl/setup.sh, then bash wsl/serve.sh)."
        )
    missing = [name for name in ("kernels", "bitsandbytes") if not importable(name)]
    if missing:
        raise RuntimeError(
            f"AURADREAM_QUANT=nunchaku needs {' and '.join(missing)}; "
            "`uv sync` on Linux installs them."
        )
    with open(hub_file(settings.model_id, "transformer/config.json"), encoding="utf-8") as fh:
        method = (json.load(fh).get("quantization_config") or {}).get("quant_method")
    if method != NUNCHAKU_METHOD:
        raise RuntimeError(
            f"{settings.model_id} is not a Nunchaku Lite checkpoint (transformer "
            f"quant_method={method!r}); under AURADREAM_QUANT=nunchaku it would load in "
            f"full bf16. Point AURADREAM_MODEL_ID at a pre-quantized repo such as "
            f"{NUNCHAKU_EXAMPLE}."
        )


# A quantizing load used to keep the whole bf16 checkpoint mapped beside the
# quantized copy it made of it. safetensors hands each tensor out as a view of
# its shard's memory map, and the loaders keep every tensor they do not
# quantize - biases, norms, embeddings - as that view, so a few kilobytes per
# layer held every shard open for the life of the pipeline: on Windows that is
# working set and commit, about 46 GiB at FLUX.1's second transformer shard
# (verified.md, "RAM at the load", 2026-10-25). Copying those tensors out lets
# each shard go once diffusers is done with it, and a transformers component's
# files once it has loaded. Only while quantizing: a bf16 load keeps the
# mapping on purpose, paging weights in as offload first moves them.
_SHARD_LOADER = ("diffusers.models.model_loading_utils", "load_model_dict_into_meta")
_COMPONENT_LOADER = ("diffusers.pipelines.pipeline_utils", "load_sub_model")


def own_tensors(module: Any, names: "set[str] | None" = None) -> int:
    """Replace each plain CPU parameter and buffer of `module` - those in
    `names` when given - with a copy of its own, and answer how many.

    Quantized tensors (torchao subclasses) are new allocations already and
    are left alone; a weight tied to two modules stays one tensor.
    """
    import torch

    slots = [
        (store, key, value, f"{prefix}.{key}" if prefix else key)
        for prefix, sub in module.named_modules(remove_duplicate=False)
        for store in (sub._parameters, sub._buffers)
        for key, value in store.items()
        if value is not None
    ]
    # Chosen by name, replaced under every name: a tied alias the shard did
    # not list must follow, or the tie breaks and the old view stays alive.
    copies: dict[int, Any] = {}
    for _, _, value, name in slots:
        if id(value) in copies or (names is not None and name not in names):
            continue
        data = value.data if isinstance(value, torch.nn.Parameter) else value
        if type(data) is not torch.Tensor or data.device.type != "cpu":
            continue
        copy = data.clone()
        if isinstance(value, torch.nn.Parameter):
            copy = torch.nn.Parameter(copy, requires_grad=value.requires_grad)
        copies[id(value)] = copy
    for store, key, value, _ in slots:
        if id(value) in copies:
            store[key] = copies[id(value)]
    return len(copies)


@contextmanager
def checkpoint_files_released(active: bool) -> Iterator[None]:
    """While a quantizing `from_pretrained` runs, copy the tensors it leaves
    unquantized out of the files they were read from (above): diffusers
    models after each shard, transformers components once loaded.

    Patches two diffusers module globals and puts them back; `load()` holds
    it, under its own lock, and nothing else loads through either.
    """
    if not active:
        yield
        return
    import importlib

    import torch

    patched: list[tuple[Any, str, Any]] = []

    def patch(target: tuple[str, str], wrap: Any) -> None:
        module = importlib.import_module(target[0])
        original = getattr(module, target[1], None)
        if original is None:
            log.warning("%s.%s not found; quantized loads keep their files mapped", *target)
            return
        patched.append((module, target[1], original))
        setattr(module, target[1], wrap(original))

    def per_shard(original: Any) -> Any:
        def load_model_dict_into_meta(model: Any, state_dict: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(model, state_dict, *args, **kwargs)
            own_tensors(model, set(state_dict))
            return result
        return load_model_dict_into_meta

    def per_component(original: Any) -> Any:
        from diffusers.models.modeling_utils import ModelMixin

        def load_sub_model(*args: Any, **kwargs: Any) -> Any:
            loaded = original(*args, **kwargs)
            # A diffusers model let its shards go one at a time already.
            if isinstance(loaded, torch.nn.Module) and not isinstance(loaded, ModelMixin):
                own_tensors(loaded)
            return loaded
        return load_sub_model

    patch(_SHARD_LOADER, per_shard)
    patch(_COMPONENT_LOADER, per_component)
    try:
        yield
    finally:
        for module, name, original in reversed(patched):
            setattr(module, name, original)


def place_pipeline(pipe: Any, settings: "Settings") -> None:
    """Apply the configured device strategy, initially or after PE arrives."""
    import torch

    # The offload hooks resolve their own accelerator when given none,
    # which is always GPU 0: `AURADREAM_DEVICE=cuda:1` used to load onto cuda:0
    # while describe() reported that card's name and memory. Only a CUDA
    # device is passed through - offload cannot target the CPU.
    device = torch.device(settings.device)
    target = device if device.type == "cuda" else None
    if settings.offload == "sequential":
        pipe.enable_sequential_cpu_offload(device=target)
    elif settings.offload == "model":
        pipe.enable_model_cpu_offload(device=target)
    else:
        pipe.to(device)


def denoiser(model_id: str) -> "loras.Denoiser":
    """The checkpoint's denoiser as an adapter must match it, from configs alone.

    `model_index.json` names the pipeline class and the denoiser's; its
    `config.json` builds the module on the meta device - every name and every
    Linear's width, not one byte of weights, in tens of milliseconds - which
    is what lets the adapter listing and the submit check answer before the
    load. The live denoiser, once there is one, is the same map read off the
    real thing. Raises when a config is out of reach, or while a load runs
    (`LoadInProgress`, `empty_weights`); the caller decides what an
    unanswered question means.
    """
    from .. import loras

    with open(hub_file(model_id, "model_index.json"), encoding="utf-8") as fh:
        index = json.load(fh)
    component = next((name for name in ("transformer", "unet") if name in index), None)
    if component is None:
        raise ValueError(f"{model_id}'s index names neither a transformer nor a unet")
    with open(hub_file(model_id, f"{component}/config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    import diffusers

    pipeline = getattr(diffusers, index["_class_name"])
    model_cls = getattr(diffusers, index[component][1])
    with empty_weights():
        module = model_cls.from_config(config)
    return loras.Denoiser.of(pipeline, component, module)


def prompt_expander_components(model_id: str) -> dict[str, Any]:
    """Load ERNIE's optional PE model and tokenizer from their subfolders."""
    import torch
    import transformers

    try:
        with open(hub_file(model_id, "model_index.json"), encoding="utf-8") as fh:
            index = json.load(fh)
    except Exception as exc:  # noqa: BLE001 - turned into one actionable error
        raise RuntimeError(f"could not read the prompt expander manifest: {exc}") from exc

    classes: dict[str, Any] = {}
    for name in ("pe", "pe_tokenizer"):
        spec = index.get(name)
        if not (
            isinstance(spec, (list, tuple))
            and len(spec) == 2
            and spec[0] == "transformers"
            and isinstance(spec[1], str)
        ):
            raise RuntimeError(f"{model_id} does not publish an ERNIE prompt expander ({name})")
        cls = getattr(transformers, spec[1], None)
        if cls is None:
            raise RuntimeError(
                f"transformers {transformers.__version__} has no {spec[1]} for the prompt expander"
            )
        classes[name] = cls

    # Fail on the tiny tokenizer before beginning a multi-GB model load.
    components: dict[str, Any] = {}
    for name in ("pe_tokenizer", "pe"):
        kwargs: dict[str, Any] = {"subfolder": name}
        if name == "pe":
            kwargs["dtype"] = torch.bfloat16
        log.info("loading prompt expander component %s/%s", model_id, name)
        components[name] = classes[name].from_pretrained(model_id, **kwargs)
    return components
