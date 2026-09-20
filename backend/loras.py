"""The LoRA adapter directory: what is in it, and the file behind a name.

One owner, because three callers need the same answer and each would otherwise
join a string onto a path of its own: the endpoint that lists the adapters for
the UI, the submit check that refuses a name this machine has not got, and the
backend that hands a file to `load_lora_weights`. A request carries a *name*
(`GenerateRequest._bare_name` refuses anything with a separator in it) and this
is the only place one becomes a path - resolved under the directory and then
checked to still be inside it, so a name that got past the schema cannot reach
another directory through a symlink or a Windows alias.

Also the fit: whether an adapter can attach to the running checkpoint at all.
A LoRA is a set of low-rank deltas addressed to modules by name - FLUX.1's
`single_transformer_blocks.0.attn.to_k`, Z-Image's `layers.0.attention.to_q` -
and PEFT injects it by looking those names up on the denoiser, so an adapter
trained for another family fails at the first cell with a traceback naming
neither model. The name and shape comparison here is that lookup, run ahead
of time: `targets()` reads what the file addresses, `mismatch()` compares it
with a `Denoiser`, and the listing hides what does not fit while the submit
check refuses it by name. A family table would have done the obvious cases
and got the subtle ones wrong: FLUX.1 and FLUX.2 [klein] share every block
name and differ in the count, FLUX.1 and Krea 2 share the names and differ
in the width, and the denoiser's own module list is the only thing that
knows either.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import settings

# What `load_lora_weights` reads. `.bin`/`.pt` LoRAs exist and are pickles:
# loading one executes whatever it contains, which is not a thing to do to a
# file that merely turned up in a directory.
SUFFIXES = (".safetensors",)


def available() -> list[dict]:
    """Every adapter in the directory, by name and size, in name order.

    A missing directory is the ordinary state - most installs have no adapters
    - and reads as an empty list rather than an error.
    """
    directory = settings.loras_dir
    if not directory.is_dir():
        return []
    found = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUFFIXES:
            try:
                size = path.stat().st_size
            except OSError:  # pragma: no cover - vanished between listing and stat
                continue
            found.append({"name": path.name, "bytes": size})
    return found


def path_for(name: str) -> Path:
    """The file for an adapter name, or a ValueError naming the directory."""
    directory = settings.loras_dir.resolve()
    path = (directory / name).resolve()
    if directory not in path.parents or path.suffix.lower() not in SUFFIXES or not path.is_file():
        raise ValueError(
            f"no LoRA named {name!r} in {settings.loras_dir}; "
            f"put a .safetensors adapter there (LOCAL_DIFFUSION_LORAS_DIR moves the directory)"
        )
    return path


# ------------------------------------------------------------------- the fit


@dataclass(frozen=True)
class Denoiser:
    """What an adapter is matched against.

    `pipeline` is the pipeline *class*, whose `lora_state_dict` reads the file
    the way the injection will; `component` the attribute it attaches to
    (`transformer` or `unet`); `class_name` that module's class, for the
    message; `layers` its Linear and Conv2d layers by module path as
    (in, out). Built from the live pipeline once there is one and from the
    checkpoint's configs on the meta device before that - names and shapes are
    all the comparison needs, and the configs carry both.
    """

    pipeline: Any
    component: str
    class_name: str
    layers: Mapping[str, tuple[int, int]]

    @classmethod
    def of(cls, pipeline: Any, component: str, module: Any) -> "Denoiser":
        import torch

        layers: dict[str, tuple[int, int]] = {}
        for name, child in module.named_modules():
            # An adapter already attached wraps its targets: `to_q` becomes a
            # PEFT layer holding the Linear as `to_q.base_layer`, beside its
            # own `to_q.lora_A.default`. The next adapter addresses `to_q`.
            parts = name.split(".")
            if any(part.startswith("lora_") for part in parts):
                continue
            name = name.removesuffix(".base_layer")
            if isinstance(child, torch.nn.Linear):
                layers[name] = (child.in_features, child.out_features)
            elif isinstance(child, torch.nn.Conv2d):
                layers[name] = (child.in_channels, child.out_channels)
        return cls(pipeline, component, type(module).__name__, layers)


# What each adapter file addresses, by the file's identity and the reader:
# the read is the whole file, and the listing and every submit ask again.
_targets: dict[tuple[str, int, int, str], dict[str, dict[str, tuple[int | None, int | None]]]] = {}


def targets(path: Path, pipeline: Any) -> dict[str, dict[str, tuple[int | None, int | None]]]:
    """The modules the adapter addresses, by component: path -> (in, out).

    Read through the pipeline's own `lora_state_dict`, so a kohya or a ComfyUI
    key layout is seen the way the injection will see it, then brought to the
    PEFT spelling exactly as `load_lora_adapter` brings it - SDXL's kohya
    conversion stops at `attn1.processor.to_q_lora.down.weight` and the loader
    finishes the job. `in` is `lora_A`'s second dimension and `out` `lora_B`'s
    first, for a Linear and a Conv2d alike; either is None when the file has
    only half the pair.
    """
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, pipeline.__name__)
    found = _targets.get(key)
    if found is not None:
        return found
    from diffusers.utils.state_dict_utils import convert_unet_state_dict_to_peft

    read = pipeline.lora_state_dict(str(path))
    state = read[0] if isinstance(read, tuple) else read
    by_component: dict[str, dict[str, Any]] = {}
    for name, tensor in state.items():
        component, _, rest = name.partition(".")
        by_component.setdefault(component, {})[rest] = tensor
    found = {}
    for component, tensors in by_component.items():
        if not any("lora_A" in name or "lora_B" in name for name in tensors):
            try:
                tensors = convert_unet_state_dict_to_peft(tensors)
            except Exception:  # noqa: BLE001 - not a layout it knows; the names decide
                pass
        modules: dict[str, list[int | None]] = {}
        for name, tensor in tensors.items():
            if name.endswith(".lora_A.weight"):
                modules.setdefault(name[: -len(".lora_A.weight")], [None, None])[0] = tensor.shape[1]
            elif name.endswith(".lora_B.weight"):
                modules.setdefault(name[: -len(".lora_B.weight")], [None, None])[1] = tensor.shape[0]
        if modules:
            found[component] = {name: (io[0], io[1]) for name, io in modules.items()}
    _targets[key] = found
    return found


def mismatch(name: str, denoiser: Denoiser) -> str | None:
    """Why the adapter cannot attach to this denoiser, or None when it can.

    The three ways an injection fails, in the order PEFT would hit them: the
    file addresses a component the pipeline has not got (a transformer LoRA on
    SDXL's unet), a module the denoiser has not got (FLUX.1's blocks on
    Z-Image), or the right module at the wrong width (FLUX.1's 3072 on Krea
    2's 6144). One reason, naming the adapter and the denoiser, because it is
    what the tooltip shows and the 422 carries.
    """
    reader = denoiser.pipeline.__name__
    try:
        found = targets(path_for(name), denoiser.pipeline)
    except Exception as exc:  # noqa: BLE001 - whatever the reader raised is the reason
        return f"{name} could not be read as a LoRA {reader} loads: {exc}"
    if not found:
        return f"{name} carries no LoRA weights {reader} can read"
    modules = found.get(denoiser.component)
    if modules is None:
        return (
            f"{name} attaches to {' and '.join(sorted(found))}; "
            f"{reader}'s denoiser is its {denoiser.component}"
        )
    missing = sorted(module for module in modules if module not in denoiser.layers)
    if missing:
        return (
            f"{name} attaches to {len(missing)} module{'s' if len(missing) > 1 else ''} "
            f"{denoiser.class_name} has not got, such as {missing[0]}"
        )
    for module, (a_in, b_out) in modules.items():
        f_in, f_out = denoiser.layers[module]
        if (a_in is not None and a_in != f_in) or (b_out is not None and b_out != f_out):
            return (
                f"{name}'s {module} is {a_in}->{b_out} and "
                f"{denoiser.class_name}'s is {f_in}->{f_out}"
            )
    return None
