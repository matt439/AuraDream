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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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


def importable(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


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
            libraries[name] = spec[0]
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

    `KREA_QUANT=fp8` pointed at a Nunchaku Lite re-upload downloads ~9 GB and
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
            f"{method!r}); KREA_QUANT={settings.quantization} would stack torchao on top "
            "and fail after the download. Use KREA_QUANT=nunchaku (Linux) for a Nunchaku "
            "Lite repo, or point KREA_MODEL_ID at an unquantized checkpoint."
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
            "KREA_QUANT=nunchaku needs Linux: the Nunchaku Lite kernels are "
            "published for x86_64 Linux only. On Windows, run the server from "
            "WSL 2 (bash wsl/setup.sh, then bash wsl/serve.sh)."
        )
    missing = [name for name in ("kernels", "bitsandbytes") if not importable(name)]
    if missing:
        raise RuntimeError(
            f"KREA_QUANT=nunchaku needs {' and '.join(missing)}; "
            "`uv sync` on Linux installs them."
        )
    with open(hub_file(settings.model_id, "transformer/config.json"), encoding="utf-8") as fh:
        method = (json.load(fh).get("quantization_config") or {}).get("quant_method")
    if method != NUNCHAKU_METHOD:
        raise RuntimeError(
            f"{settings.model_id} is not a Nunchaku Lite checkpoint (transformer "
            f"quant_method={method!r}); under KREA_QUANT=nunchaku it would load in "
            f"full bf16. Point KREA_MODEL_ID at a pre-quantized repo such as "
            f"{NUNCHAKU_EXAMPLE}."
        )


def place_pipeline(pipe: Any, settings: "Settings") -> None:
    """Apply the configured device strategy, initially or after PE arrives."""
    import torch

    # The offload hooks resolve their own accelerator when given none,
    # which is always GPU 0: `KREA_DEVICE=cuda:1` used to load onto cuda:0
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
