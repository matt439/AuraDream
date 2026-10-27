"""Cold checks of supplied Kontext configs, never a download or load grant.

This validator checks a caller-supplied, pinned snapshot against the
integration's required component/geometry boundary, and ``require_pinned``
compares its canonical hashes with the six verified at the recipe's revision.
Matching the pin proves only the config bytes, never weights or a card
qualification; a future loader must obtain every config at the pin and pass
``require_pinned(validate_configs(...))`` before any weights.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType

from backend.instruction_edit import KONTEXT_V1, MAX_SEQUENCE_LENGTH


CONFIG_PATHS = (
    "model_index.json", "scheduler/scheduler_config.json", "transformer/config.json",
    "vae/config.json", "text_encoder/config.json", "text_encoder_2/config.json",
)
# Obtained and validated by `validate_configs` on 2026-10-27 at
# KONTEXT_V1.revision: canonical-JSON SHA-256 per path, in CONFIG_PATHS order.
# A config pin only - not a weights pin, and never an execution grant.
QUALIFIED_CONFIG_HASHES = MappingProxyType({
    "model_index.json": "e8d4c79ae755b47505713457f3465b6e9b30e6a4d88364f11caa2aa159b2446e",
    "scheduler/scheduler_config.json": "68bd9f2c97f180ecb643a14b46e76394ffc5cc5e7c7f13992766b975dd60fb1d",
    "transformer/config.json": "1727369bca1eaf7ea80fbce35f2eb539cf0d9c829b50ad30cdbab6e3499b37aa",
    "vae/config.json": "f42dc49aff7a6250b73b4a2270cc302a5a3f15ae8d690766bb16d0c4dce68f33",
    "text_encoder/config.json": "3b03552c28eb3bdd2ec0136b3e4d90266a56896274ff581b81b2fa1173afadc0",
    "text_encoder_2/config.json": "3e22acd66a0aee18b849b36e1afc133386fa8b9715553feed69038529c892164",
})
MAX_CONFIG_BYTES = 128 * 1024
_COMPONENTS = {
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "transformer": ["diffusers", "FluxTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderKL"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "text_encoder_2": ["transformers", "T5EncoderModel"],
    "tokenizer": ["transformers", "CLIPTokenizer"],
    "tokenizer_2": ["transformers", "T5TokenizerFast"],
}


@dataclass(frozen=True)
class ConfigPreflight:
    """Descriptive proof of supplied bytes, never runtime availability."""
    model_id: str
    revision: str
    config_hashes: tuple[tuple[str, str], ...]
    vae_scale_factor: int = 8
    latent_channels: int = 16
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
    execution_available: bool = False

    @property
    def matches_pin(self) -> bool:
        """Whether every supplied config hashes to the verified pin."""
        return (
            self.model_id == KONTEXT_V1.model_id and self.revision == KONTEXT_V1.revision
            and self.config_hashes == tuple(QUALIFIED_CONFIG_HASHES.items())
        )


def _refuse(path: str) -> ValueError:
    # Do not echo caller-controlled config content, paths or credential fields.
    return ValueError(f"kontext-v1 config preflight refused {path}")


def _json_value(value: object, depth: int = 0) -> bool:
    if depth > 32:
        return False
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_json_value(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return all(type(key) is str and _json_value(item, depth + 1) for key, item in value.items())
    return False


def _integer(config: dict, key: str, expected: int, path: str) -> None:
    if type(config.get(key)) is not int or config[key] != expected:
        raise _refuse(path)


def validate_configs(model_id: str, revision: str, supplied: Mapping[str, object]) -> ConfigPreflight:
    """Validate parsed JSON supplied by the caller; perform no file/Hub access.

    Required paths are fixed. Identity arguments must be the frozen public id
    and complete revision, not a branch/tag. Hashes describe canonical config
    JSON; they do not authenticate that a caller fetched the claimed revision.
    The future loader owns that binding and must keep execution unavailable
    until independent model/card qualification has completed.
    """
    if model_id != KONTEXT_V1.model_id or revision != KONTEXT_V1.revision:
        raise _refuse("recipe identity")
    if not isinstance(supplied, Mapping) or set(supplied) != set(CONFIG_PATHS):
        raise _refuse("config inventory")
    configs: dict[str, dict] = {}
    hashes = []
    for path in CONFIG_PATHS:
        value = supplied[path]
        if not isinstance(value, dict) or not _json_value(value):
            raise _refuse(path)
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(encoded) > MAX_CONFIG_BYTES or value.get("auto_map") or value.get("trust_remote_code"):
            raise _refuse(path)
        configs[path] = value
        hashes.append((path, hashlib.sha256(encoded).hexdigest()))
    index = configs["model_index.json"]
    if index.get("_class_name") != KONTEXT_V1.pipeline_class:
        raise _refuse("model_index.json")
    if any(index.get(component) != expected for component, expected in _COMPONENTS.items()):
        raise _refuse("model_index.json")
    for key, value in index.items():
        if key in _COMPONENTS or key in {"_class_name", "_diffusers_version", "_name_or_path"}:
            continue
        if key not in {"image_encoder", "feature_extractor"} or value != [None, None]:
            raise _refuse("model_index.json")
    for name in ("scheduler", "transformer", "vae"):
        path = "scheduler/scheduler_config.json" if name == "scheduler" else f"{name}/config.json"
        if configs[path].get("_class_name") != _COMPONENTS[name][1]:
            raise _refuse(path)
    vae = configs["vae/config.json"]
    blocks = vae.get("block_out_channels")
    if not isinstance(blocks, list) or len(blocks) != 4 or any(type(item) is not int or item <= 0 for item in blocks):
        raise _refuse("vae/config.json")
    _integer(vae, "latent_channels", 16, "vae/config.json")
    _integer(vae, "in_channels", 3, "vae/config.json")
    _integer(vae, "out_channels", 3, "vae/config.json")
    for key in ("shift_factor", "scaling_factor"):
        value = vae.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise _refuse("vae/config.json")
        if (
            (isinstance(value, float) and not math.isfinite(value))
            or (key == "scaling_factor" and value <= 0)
        ):
            raise _refuse("vae/config.json")
    transformer = configs["transformer/config.json"]
    for key, expected in (("in_channels", 64), ("joint_attention_dim", 4096), ("pooled_projection_dim", 768)):
        _integer(transformer, key, expected, "transformer/config.json")
    if transformer.get("guidance_embeds") is not True:
        raise _refuse("transformer/config.json")
    clip = configs["text_encoder/config.json"]
    t5 = configs["text_encoder_2/config.json"]
    if clip.get("architectures") != ["CLIPTextModel"] or clip.get("model_type") != "clip_text_model":
        raise _refuse("text_encoder/config.json")
    _integer(clip, "hidden_size", 768, "text_encoder/config.json")
    _integer(clip, "max_position_embeddings", 77, "text_encoder/config.json")
    if t5.get("architectures") != ["T5EncoderModel"] or t5.get("model_type") != "t5":
        raise _refuse("text_encoder_2/config.json")
    _integer(t5, "d_model", 4096, "text_encoder_2/config.json")
    return ConfigPreflight(model_id=model_id, revision=revision, config_hashes=tuple(hashes))


def require_pinned(preflight: ConfigPreflight) -> ConfigPreflight:
    """Refuse a validated snapshot whose bytes differ from the verified pin.

    Names the first differing path only; never echoes a supplied or pinned
    hash. Passing grants no execution: ``execution_available`` stays False.
    """
    if not isinstance(preflight, ConfigPreflight):
        raise _refuse("config pin")
    if preflight.model_id != KONTEXT_V1.model_id or preflight.revision != KONTEXT_V1.revision:
        raise _refuse("recipe identity")
    supplied = dict(preflight.config_hashes)
    if tuple(supplied) != tuple(QUALIFIED_CONFIG_HASHES):
        raise _refuse("config inventory")
    for path, expected in QUALIFIED_CONFIG_HASHES.items():
        if supplied[path] != expected:
            raise _refuse(path)
    if not preflight.matches_pin:
        raise _refuse("config pin")
    return preflight
