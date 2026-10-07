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
from typing import TYPE_CHECKING, Any

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
# The nested Z-Anime encoder is scaled FP8 without a quantization_config:
# Transformers silently drops its *.scale tensors. The author identifies the
# default encoder as standard Z-Image Qwen3-4B. Load that parent's original
# BF16 component and apply our normal owned Transformers quantization config.
Z_ANIME_TEXT_ENCODER = {
    "pretrained_model_name_or_path": "Tongyi-MAI/Z-Image",
    "revision": "04cc4abb7c5069926f75c9bfde9ef43d49423021",
    "subfolder": "text_encoder",
}


def resolve_model(model_id: str, *, weights: bool = False) -> str:
    """Resolve a nested component tree, reading configs before any weights.

    The first pass supplies the manifest and component configs for quantizer
    checks. Only after those checks pass does load() request the whole tree.
    Pin both passes to one revision so configs cannot race a weight update.
    Ordinary Hub ids and explicit local directories pass through unchanged.
    """
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
    options = {"ignore_patterns": [f"{folder}/text_encoder/*"]} if weights and model_id == "SeeSee21/Z-Anime" else {}
    snapshot = snapshot_download(model_id, revision=revision, allow_patterns=patterns, **options)
    return str(Path(snapshot) / folder)


def component_overrides(model_id: str, quant: Any) -> dict[str, Any]:
    """Pinned replacements for incorrectly packaged upstream components."""
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
    downloads: both Anima variants' and NewBie's pipelines are the app's own, with no
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
        if not all(isinstance(part, str) for part in spec):
            continue
        if name.startswith(QUANTIZABLE_PREFIXES) and spec[0]:
            libraries[name] = COMPONENT_LIBRARY_OWNERS.get(spec[0], spec[0])
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

    from diffusers import PipelineQuantizationConfig
    from diffusers import TorchAoConfig as DiffusersTorchAoConfig
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

    `LOCAL_DIFFUSION_QUANT=fp8` pointed at a Nunchaku Lite re-upload downloads ~9 GB and
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
            f"{method!r}); LOCAL_DIFFUSION_QUANT={settings.quantization} would stack torchao on top "
            "and fail after the download. Use LOCAL_DIFFUSION_QUANT=nunchaku (Linux) for a Nunchaku "
            "Lite repo, or point LOCAL_DIFFUSION_MODEL_ID at an unquantized checkpoint."
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
            "LOCAL_DIFFUSION_QUANT=nunchaku needs Linux: the Nunchaku Lite kernels are "
            "published for x86_64 Linux only. On Windows, run the server from "
            "WSL 2 (bash wsl/setup.sh, then bash wsl/serve.sh)."
        )
    missing = [name for name in ("kernels", "bitsandbytes") if not importable(name)]
    if missing:
        raise RuntimeError(
            f"LOCAL_DIFFUSION_QUANT=nunchaku needs {' and '.join(missing)}; "
            "`uv sync` on Linux installs them."
        )
    with open(hub_file(settings.model_id, "transformer/config.json"), encoding="utf-8") as fh:
        method = (json.load(fh).get("quantization_config") or {}).get("quant_method")
    if method != NUNCHAKU_METHOD:
        raise RuntimeError(
            f"{settings.model_id} is not a Nunchaku Lite checkpoint (transformer "
            f"quant_method={method!r}); under LOCAL_DIFFUSION_QUANT=nunchaku it would load in "
            f"full bf16. Point LOCAL_DIFFUSION_MODEL_ID at a pre-quantized repo such as "
            f"{NUNCHAKU_EXAMPLE}."
        )


def place_pipeline(pipe: Any, settings: "Settings") -> None:
    """Apply the configured device strategy, initially or after PE arrives."""
    import torch

    # The offload hooks resolve their own accelerator when given none,
    # which is always GPU 0: `LOCAL_DIFFUSION_DEVICE=cuda:1` used to load onto cuda:0
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
