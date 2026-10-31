"""Frozen instruction-edit contracts and the one gated execution seam.

This leaf imports neither application settings/storage nor a model framework.
The small Kontext geometry is an application recipe, not an author-recommended
model capability. Its first 512x512 card trial is recorded; other shapes remain
unqualified. Execution opens only where a caller passes
``qualified=True``, which comes from the running backend instance's
``instruction_edit_qualified()`` - never a setting. The real diffusers
backend answers the literal True since its release (stage 4), as the test and
fake stand-ins do; every other backend answers False.
``QUALIFIED_ENVELOPE`` records what stage 3 measured on the card and stage 4
released: pure data
and refusal words. Where the gate is open, the edit routes and the queue also
hold an unmeasured loader (``edit_unavailable_reason`` with a
``LoaderChoice``) or source aspect (``geometry_refusal``), and every load
entry refuses an unmeasured loader (``require_model_available``).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import metadata
from math import gcd
from typing import Annotated, Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .execution import canonical_loader_execution


RecipeName = Literal["kontext-v1"]
PositiveSource = Annotated[int, Field(strict=True, gt=0, le=2**63 - 1)]
PositiveDimension = Annotated[int, Field(strict=True, gt=0)]

MAX_SEQUENCE_LENGTH = 512
INSTRUCTION_EDIT_UNAVAILABLE = (
    "Instruction editing is not available on this backend: "
    "FLUX.1 Kontext [dev] runs only on the backend its card trial qualified."
)
INSTRUCTION_EDITOR_UNSUPPORTED = (
    "Instruction editing is not available for this checkpoint: it has no recipe "
    "in this build; only FLUX.1 Kontext [dev] is supported."
)
INSTRUCTION_EDIT_NEEDS_MODEL = "Select FLUX.1 Kontext [dev] in Models to run an instruction edit."
INSTRUCTION_EDIT_TEXT_TO_IMAGE = (
    "FLUX.1 Kontext [dev] edits a gallery image; it does not generate from text. "
    "Use Edit by instruction… on an image."
)
INSTRUCTION_EDIT_REPLAY_HELD = "Instruction edits are queued only from Edit by instruction… or More seeds."
INSTRUCTION_EDIT_COMMAND_LINE = (
    "FLUX.1 Kontext [dev] edits a gallery image; the command line generates from text "
    "and does not run instruction edits."
)
UNRESOLVED_SOURCE = "The recorded instruction-edit source has not been resolved to a gallery image."


@dataclass(frozen=True)
class Recipe:
    name: RecipeName
    model_id: str
    revision: str
    pipeline_class: str
    # Starting values from the author example and installed call, not measurements.
    default_steps: int
    default_guidance_scale: float
    negative_prompt: str = ""
    true_cfg: None = None
    mu: None = None


KONTEXT_V1 = Recipe(
    name="kontext-v1",
    model_id="black-forest-labs/FLUX.1-Kontext-dev",
    revision="24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d",
    pipeline_class="FluxKontextPipeline",
    default_steps=28,
    default_guidance_scale=2.5,
)
INSTRUCTION_EDIT_EXACT_ID = f"Instruction editing needs {KONTEXT_V1.model_id} exactly; select it in Models."


def recipe_for(name: str) -> Recipe:
    """A named source contract, never permission to load or render."""
    if name != KONTEXT_V1.name:
        raise ValueError("unsupported instruction-edit recipe")
    return KONTEXT_V1


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, revalidate_instances="always")

    @model_validator(mode="before")
    @classmethod
    def _validate_copied_instances(cls, value: object) -> object:
        # model_copy(update=...) deliberately skips validation. Preserve every
        # copied field, including unknown ones, when crossing a contract again.
        if isinstance(value, cls):
            return {**vars(value), **(value.__pydantic_extra__ or {})}
        return value


class InstructionEditSpec(_Contract):
    """A requested source id and explicit recipe, without inferred defaults."""
    source: PositiveSource
    recipe: RecipeName


class InstructionEditSource(_Contract):
    """Proof of original decoded RGB pixels, never authority to open a path.

    Unlike the gallery's 16-character display hash, ``hash`` is the full
    SHA-256 of ``image.convert('RGB').tobytes()`` before any resize. Dimensions
    and the original bare filename accompany it. No EXIF transpose is applied.
    """
    filename: Annotated[str, Field(min_length=5, max_length=255)]
    hash: Annotated[str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]
    width: PositiveDimension
    height: PositiveDimension

    @field_validator("filename")
    @classmethod
    def _bare_png(cls, value: str) -> str:
        if (
            not isinstance(value, str) or not 5 <= len(value) <= 255 or not value.lower().endswith(".png")
            or any(character in '<>:"/\\|?*' or ord(character) < 32 or ord(character) == 127 for character in value)
            or value.endswith((" ", "."))
        ):
            raise ValueError("instruction-edit source filename must be a bare PNG basename")
        # Windows device aliases stay aliases even with another extension.
        stem = value.split(".", 1)[0].rstrip(" .").upper()
        reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
        reserved.update(f"{prefix}{number}" for prefix in ("COM", "LPT") for number in (*range(1, 10), "¹", "²", "³"))
        if stem in reserved:
            raise ValueError("instruction-edit source filename must be a bare PNG basename")
        return value


class StoredInstructionEdit(_Contract):
    """Durable intent and proof survive an unresolved source-id remapping."""
    source: PositiveSource | None
    recipe: RecipeName
    edit_source: InstructionEditSource

    def request(self) -> InstructionEditSpec:
        """Convert the id only; later admission must prove pixels and existence.

        A positive id here is not evidence that the current gallery contains
        the recorded source. This method cannot resolve or validate a gallery.
        """
        stored = StoredInstructionEdit.model_validate(self)
        if stored.source is None:
            raise ValueError(UNRESOLVED_SOURCE)
        return InstructionEditSpec(source=stored.source, recipe=stored.recipe)


def stored_json(value: object) -> str:
    """Canonical complete JSON; an unresolved source remains explicit null."""
    stored = StoredInstructionEdit.model_validate(value)
    return json.dumps(stored.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stored_from_json(value: str) -> StoredInstructionEdit:
    """Strict portable parsing without current settings, a gallery or the Hub."""
    if not isinstance(value, str):
        raise ValueError("stored instruction edit must be JSON text")
    return StoredInstructionEdit.model_validate_json(value)


def bound_spec(value: object) -> StoredInstructionEdit:
    """Validate complete portable proof, including unresolved recovery intent.

    This checks spelling only. It does not prove that a local gallery id
    still names these pixels; admission must establish that separately.
    """
    return StoredInstructionEdit.model_validate(value)


def edit_key(value: object | None) -> str | None:
    """Canonical complete stored identity; raw unbound requests cannot key.

    Unresolved stored intent can still be grouped and exported. Converting
    it into a replay request requires a proven positive local source first.
    """
    return None if value is None else stored_json(bound_spec(value))


def descriptor_from_image(filename: str, image: Image.Image) -> InstructionEditSource:
    """Describe supplied pixels without opening a file or mutating the image."""
    # Refuse an invalid name before decoding/copying the supplied pixels.
    filename = InstructionEditSource._bare_png(filename)
    with image.convert("RGB") as rgb:
        return InstructionEditSource(
            filename=filename, hash=hashlib.sha256(rgb.tobytes()).hexdigest(),
            width=image.width, height=image.height,
        )


def _working_size(source_width: int, source_height: int) -> tuple[int, int]:
    if any(type(value) is not int or value <= 0 for value in (source_width, source_height)):
        raise ValueError("instruction-edit source dimensions must be positive integers")
    divisor = gcd(source_width, source_height)
    ratio = source_width // divisor, source_height // divisor
    # This named table is frozen. It does not snap approximate source ratios.
    sizes = {
        (1, 1): (512, 512),
        (3, 2): (576, 384),
        (2, 3): (384, 576),
        (4, 3): (512, 384),
        (3, 4): (384, 512),
        (16, 9): (512, 288),
        (9, 16): (288, 512),
    }
    if ratio not in sizes:
        raise ValueError(
            "kontext-v1 requires an exact source aspect of 1:1, 3:2, 2:3, 4:3, 3:4, 16:9 or 9:16; "
            "the source is not cropped or snapped to another aspect"
        )
    return sizes[ratio]


@dataclass(frozen=True)
class InstructionEditGeometry:
    source_width: int
    source_height: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if (
            type(self.width) is not int or type(self.height) is not int
            or (self.width, self.height) != _working_size(self.source_width, self.source_height)
        ):
            raise ValueError("instruction-edit geometry does not match the frozen source mapping")

    @property
    def source_size(self) -> tuple[int, int]:
        return self.source_width, self.source_height

    @property
    def working_size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def output_size(self) -> tuple[int, int]:
        return self.width, self.height

    def pipeline_kwargs(self) -> dict[str, int | bool]:
        """Explicit geometry only; these kwargs grant no execution support.

        Kontext otherwise expands output to its default max_area and resizes
        the source independently to a preferred resolution near one megapixel.
        """
        return {
            "width": self.width, "height": self.height, "max_area": self.width * self.height,
            "_auto_resize": False, "num_images_per_prompt": 1, "max_sequence_length": MAX_SEQUENCE_LENGTH,
        }


def geometry(source_width: int, source_height: int, *, recipe: str = "kontext-v1") -> InstructionEditGeometry:
    """Resolve the exact source aspect without cropping or loading a model."""
    recipe_for(recipe)
    width, height = _working_size(source_width, source_height)
    return InstructionEditGeometry(source_width, source_height, width, height)


def prepare_source(image: Image.Image, resolved: InstructionEditGeometry | None = None) -> Image.Image:
    """An owned RGB/Lanczos image; never transpose EXIF, crop or alter input.

    When geometry came from an earlier preview, require the actual original
    dimensions as well as the aspect. Provenance/existence checks belong to
    later source admission, not this supplied-pixel operation.
    """
    actual = geometry(image.width, image.height)
    if resolved is not None and (not isinstance(resolved, InstructionEditGeometry) or resolved != actual):
        raise ValueError("the instruction-edit source dimensions changed after geometry was resolved")
    with image.convert("RGB") as rgb:
        return rgb.resize(actual.working_size, resample=Image.Resampling.LANCZOS)


@dataclass(frozen=True)
class QualifiedEnvelope:
    """What stage 3 measured on this card, and nothing wider.

    Data only: nothing here opens the gate, and nothing here reaches
    ``applied()``, a ``CellKey``, a run id or the PNG chunk, so stored rows
    of an unqualified shape still bind and inspect. Batch 1 and no LoRA or
    references are already the edit request's own schema rules.
    """
    model_id: str
    revision: str
    recipe: RecipeName
    quantization: str
    offload: str
    devices: tuple[str, ...]
    platform: str
    diffusers_version: str
    # Reduced source aspects and the working sizes they render at.
    sizes: tuple[tuple[tuple[int, int], tuple[int, int]], ...]


QUALIFIED_ENVELOPE = QualifiedEnvelope(
    model_id=KONTEXT_V1.model_id,
    revision=KONTEXT_V1.revision,
    recipe=KONTEXT_V1.name,
    quantization="fp8",
    offload="model",
    devices=("cuda", "cuda:0"),
    platform="win32",
    diffusers_version="0.40.0",
    sizes=(((1, 1), (512, 512)),),
)

INSTRUCTION_EDIT_LOADER_UNQUALIFIED = (
    "FLUX.1 Kontext [dev] is qualified on this card only at fp8 with model offload "
    "on Windows, Eager; {choice} is not measured."
)
INSTRUCTION_EDIT_ASPECT_UNQUALIFIED = (
    "Only square sources are qualified for instruction editing (512x512); "
    "this {a}:{b} source would render at {w}x{h}, which is not measured."
)

_QUANTIZATION_WORDS = {"int8": "int8 quantization", "none": "BF16 (no quantization)", "nunchaku": "Nunchaku 4-bit"}
_OFFLOAD_WORDS = {"sequential": "sequential offload", "none": "no offload"}
_PLATFORM_WORDS = {"linux": "Linux (WSL)", "darwin": "macOS"}


def installed_diffusers_version() -> str | None:
    """The installed distribution's version, read without importing diffusers."""
    try:
        return metadata.version("diffusers")
    except metadata.PackageNotFoundError:
        return None


def _loader_execution_words(value: object) -> str | None:
    """The unmeasured part of a loader recipe, or None for plain Eager."""
    try:
        spec = canonical_loader_execution(value)
    except (ValueError, TypeError):
        return "an unreadable performance configuration"
    if spec is None:
        return None
    words = []
    if spec.compilation is not None:
        words.append(f"{spec.compilation} compilation")
    if spec.attention_backend is not None:
        words.append(f"{spec.attention_backend} attention")
    return " with ".join(words)


def loader_refusal(
    model_id: str, *, quantization: str, offload: str, device: str, platform: str,
    loader_execution: object, diffusers_version: str | None,
) -> str | None:
    """Why this loading combination is outside the measured envelope, or None.

    Pure: every fact is passed in (``installed_diffusers_version()`` and
    ``sys.platform`` are the callers' to read). The first unmeasured choice
    is named; this never answers whether the backend's gate is open.
    """
    envelope = QUALIFIED_ENVELOPE
    if model_id != envelope.model_id:
        choice = f"the checkpoint {model_id!r}"
    elif quantization != envelope.quantization:
        choice = _QUANTIZATION_WORDS.get(quantization, f"quantization {quantization!r}")
    elif offload != envelope.offload:
        choice = _OFFLOAD_WORDS.get(offload, f"offload {offload!r}")
    elif device not in envelope.devices:
        choice = f"the device {device!r}"
    elif platform != envelope.platform:
        choice = _PLATFORM_WORDS.get(platform, f"the platform {platform!r}")
    elif (execution := _loader_execution_words(loader_execution)) is not None:
        choice = execution
    elif diffusers_version != envelope.diffusers_version:
        choice = "a missing diffusers" if diffusers_version is None else f"diffusers {diffusers_version}"
    else:
        return None
    return INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice=choice)


@dataclass(frozen=True)
class LoaderChoice:
    """The loading facts ``loader_refusal`` compares, gathered in one place.

    A backend instance builds it (``GenerationBackend.instruction_edit_loader``)
    from the server's selection and the host it runs on; nothing here reads
    either.
    """
    quantization: str
    offload: str
    device: str
    platform: str
    loader_execution: object
    diffusers_version: str | None

    @classmethod
    def of(cls, selected: Any, *, platform: str, diffusers_version: str | None) -> LoaderChoice:
        """From a settings snapshot (``current_settings``), read by attribute.

        A snapshot without ``loader_execution`` is Eager, as the loader reads it.
        """
        return cls(
            quantization=selected.quantization, offload=selected.offload, device=selected.device,
            platform=platform, loader_execution=getattr(selected, "loader_execution", None),
            diffusers_version=diffusers_version,
        )

    def refusal(self, model_id: str) -> str | None:
        return loader_refusal(
            model_id, quantization=self.quantization, offload=self.offload, device=self.device,
            platform=self.platform, loader_execution=self.loader_execution,
            diffusers_version=self.diffusers_version,
        )


def option_refusals(
    model_id: str, *, quantizations: Iterable[str], offloads: Iterable[str],
) -> dict[str, dict[str, str]]:
    """The editor's quantization and offload choices the card never measured.

    Each named in ``loader_refusal``'s words for that choice alone, the rest
    of the loader taken as measured, for the Models picker to hold by its
    option; every other checkpoint, and a measured choice, has none. Pure,
    and no answer about the gate or this host: those stay the backend's.
    """
    envelope = QUALIFIED_ENVELOPE
    if model_id != envelope.model_id:
        return {}
    def words(quantization: str, offload: str) -> str | None:
        return loader_refusal(
            model_id, quantization=quantization, offload=offload, device=envelope.devices[0],
            platform=envelope.platform, loader_execution=None, diffusers_version=envelope.diffusers_version,
        )

    return {
        "quantization": {
            value: reason for value in quantizations
            if (reason := words(value, envelope.offload)) is not None
        },
        "offload": {
            value: reason for value in offloads
            if (reason := words(envelope.quantization, value)) is not None
        },
    }


def geometry_refusal(resolved: InstructionEditGeometry) -> str | None:
    """Why a frozen-table geometry is outside the measured sizes, or None.

    ``geometry()`` itself stays the whole frozen table: a stored 3:2 edit
    still binds and inspects, it is only not run.
    """
    if not isinstance(resolved, InstructionEditGeometry):
        raise TypeError("geometry_refusal needs a resolved InstructionEditGeometry")
    divisor = gcd(resolved.source_width, resolved.source_height)
    aspect = resolved.source_width // divisor, resolved.source_height // divisor
    if (aspect, resolved.working_size) in QUALIFIED_ENVELOPE.sizes:
        return None
    return INSTRUCTION_EDIT_ASPECT_UNQUALIFIED.format(
        a=aspect[0], b=aspect[1], w=resolved.width, h=resolved.height,
    )


def bound_geometry_refusal(value: object) -> str | None:
    """``geometry_refusal`` for a bound edit, from its recorded source size."""
    stored = StoredInstructionEdit.model_validate(value)
    source = stored.edit_source
    return geometry_refusal(geometry(source.width, source.height, recipe=stored.recipe))


def require_execution_available(spec: object, *, qualified: bool = False) -> None:
    """Validate the intent, then refuse unless the backend is qualified.

    ``qualified`` must be the literal True a backend instance answered;
    a truthy stand-in (1, a non-empty string) leaves the seam closed.
    """
    if isinstance(spec, StoredInstructionEdit):
        StoredInstructionEdit.model_validate(spec)
    else:
        InstructionEditSpec.model_validate(spec)
    if qualified is not True:
        raise ValueError(INSTRUCTION_EDIT_UNAVAILABLE)


def require_model_available(
    model_id: str, *, qualified: bool = False, loader: LoaderChoice | None = None,
) -> None:
    """Refuse the known edit-only checkpoints before any public load path.

    A hand-entered id must not bypass the closed release boundary merely
    because it is absent from the catalogue or accepts an image argument.
    Qwen-Image-Edit has no recipe and is always refused; Kontext only
    where the backend answered the literal True, by its exact id, and then
    only inside the measured loader envelope (``loader``, the backend's
    ``instruction_edit_loader`` for the selection being loaded). An open
    gate without its loader facts is never inside the envelope. Every
    caller passes both (``test_instruction_edit_loader``'s scan).
    """
    folded = model_id.casefold()
    if folded == "qwen/qwen-image-edit":
        raise ValueError(INSTRUCTION_EDITOR_UNSUPPORTED)
    if folded == KONTEXT_V1.model_id.casefold():
        if qualified is not True:
            raise ValueError(INSTRUCTION_EDIT_UNAVAILABLE)
        # The recipe's identity is exact (bind_request compares it so); a
        # case variant would load and then refuse every edit it was for.
        if model_id != KONTEXT_V1.model_id:
            raise ValueError(INSTRUCTION_EDIT_EXACT_ID)
        if not isinstance(loader, LoaderChoice):
            raise ValueError(INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="an unknown loader"))
        reason = loader.refusal(model_id)
        if reason is not None:
            raise ValueError(reason)


def is_edit_only(model_id: str) -> bool:
    """Whether ``model_id`` names a known edit-only checkpoint, in any case."""
    return model_id.casefold() in {"qwen/qwen-image-edit", KONTEXT_V1.model_id.casefold()}


def command_line_refusal(model_id: str) -> str | None:
    """The command line's words for an edit-only checkpoint, or None.

    ``generate.py`` renders from text and never runs an instruction edit,
    whatever any backend's gate answers, so this needs no instance.
    """
    if model_id.casefold() == "qwen/qwen-image-edit":
        return INSTRUCTION_EDITOR_UNSUPPORTED
    if is_edit_only(model_id):
        return INSTRUCTION_EDIT_COMMAND_LINE
    return None


def editor_class_refusal(class_name: str) -> str:
    """The words for a refused instruction-editor pipeline class.

    Every caller is the released diffusers backend, whose gate is open, so
    Kontext's class under another id (a mirror, a local copy) is refused for
    its id, not with a closed gate's words.
    """
    return INSTRUCTION_EDIT_EXACT_ID if class_name == KONTEXT_V1.pipeline_class else INSTRUCTION_EDITOR_UNSUPPORTED


def edit_unavailable_reason(model_id: str, *, qualified: bool, loader: LoaderChoice | None = None) -> str | None:
    """Why an instruction edit cannot queue on this server now, or None.

    In order: the gate (closed backends answer UNAVAILABLE whatever is
    selected), the selected checkpoint, then the measured loader envelope
    (``loader_refusal``). A qualified answer without its ``loader`` facts is
    never inside the envelope. The source's aspect is the caller's to add
    (``geometry_refusal``): this answers for the server, not one image.
    """
    if qualified is not True:
        return INSTRUCTION_EDIT_UNAVAILABLE
    # Exact, as bind_request compares: a case variant is not the recipe's id.
    if model_id != KONTEXT_V1.model_id:
        return INSTRUCTION_EDIT_NEEDS_MODEL
    if not isinstance(loader, LoaderChoice):
        return INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="an unknown loader")
    return loader.refusal(model_id)


def replay_held_reason(reason: str | None) -> str:
    """Generate, Swap tags and Finish's words for an edit.

    ``reason`` is ``edit_unavailable_reason``'s answer. They never queue an
    edit: where one could queue - or would, with the editor selected - they
    point at Edit by instruction… and More seeds (which runs one inside the
    envelope; Re-render refuses by id, ``submission.edit_rerender_detail``);
    a closed gate or an unmeasured loader is
    said as it is, so the pointer never leads to a server-wide refusal (a
    source's own aspect is still the dialog's to say).
    """
    if reason is None or reason == INSTRUCTION_EDIT_NEEDS_MODEL:
        return INSTRUCTION_EDIT_REPLAY_HELD
    return reason


def text_to_image_refusal(model_id: str, *, qualified: bool, loader: LoaderChoice | None = None) -> str:
    """Generate's words where the selected checkpoint only edits.

    Pointing at Edit by instruction… is right only where that can queue;
    otherwise the edit gate's own reason is said, never a dead end.
    """
    return text_to_image_words(edit_unavailable_reason(model_id, qualified=qualified, loader=loader))


def text_to_image_words(reason: str | None) -> str:
    """Generate's words on the editor, from ``edit_unavailable_reason``'s answer.

    For a caller that already holds a backend's answer (``defaults_for``),
    so the profile and the status never word the same hold two ways.
    """
    if reason is None:
        return INSTRUCTION_EDIT_TEXT_TO_IMAGE
    return (
        "FLUX.1 Kontext [dev] edits a gallery image; it does not generate from text, "
        f"and instruction editing is held: {reason}"
    )
