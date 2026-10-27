"""Frozen instruction-edit contracts and the one gated execution seam.

This leaf imports neither application settings/storage nor a model framework.
The small Kontext geometry is an application recipe for future qualification,
not a measured model capability. Execution opens only where a caller passes
``qualified=True``, which comes from the running backend instance's
``instruction_edit_qualified()`` - never a setting. The real diffusers
backend answers False until card qualification, so its load and execution
stay closed; only test and fake stand-ins can open the seam.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import gcd
from typing import Annotated, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


RecipeName = Literal["kontext-v1"]
PositiveSource = Annotated[int, Field(strict=True, gt=0, le=2**63 - 1)]
PositiveDimension = Annotated[int, Field(strict=True, gt=0)]

MAX_AREA = 512 * 512
MAX_SEQUENCE_LENGTH = 512
INSTRUCTION_EDIT_UNAVAILABLE = (
    "Instruction editing is not available in this build: "
    "FLUX.1 Kontext [dev] has not been loaded and measured on this card."
)
INSTRUCTION_EDITOR_UNSUPPORTED = (
    "Instruction editing is not available for this checkpoint: it has no recipe "
    "in this build; only FLUX.1 Kontext [dev] is planned."
)
INSTRUCTION_EDIT_NEEDS_MODEL = "Select FLUX.1 Kontext [dev] in Models to run an instruction edit."
INSTRUCTION_EDIT_TEXT_TO_IMAGE = (
    "FLUX.1 Kontext [dev] edits a gallery image; it does not generate from text. "
    "Use Edit by instruction… on an image."
)
INSTRUCTION_EDIT_REPLAY_HELD = (
    "Instruction edits are queued from Edit by instruction…; "
    "Generate, Re-render and More seeds do not run them yet."
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


def require_model_available(model_id: str, *, qualified: bool = False) -> None:
    """Refuse the known edit-only checkpoints before any public load path.

    A hand-entered id must not bypass the closed release boundary merely
    because it is absent from the catalogue or accepts an image argument.
    Qwen-Image-Edit has no recipe and is always refused; Kontext only
    where the backend answered the literal True.
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
            raise ValueError(f"Instruction editing needs {KONTEXT_V1.model_id} exactly; select it in Models.")


def editor_class_refusal(class_name: str) -> str:
    """The words for a refused instruction-editor pipeline class."""
    return INSTRUCTION_EDIT_UNAVAILABLE if class_name == KONTEXT_V1.pipeline_class else INSTRUCTION_EDITOR_UNSUPPORTED


def edit_unavailable_reason(model_id: str, *, qualified: bool) -> str | None:
    """Why an instruction edit cannot queue on this server now, or None.

    Closed backends answer UNAVAILABLE whatever is selected; a qualified one
    still needs the editor checkpoint to be the selected model.
    """
    if qualified is not True:
        return INSTRUCTION_EDIT_UNAVAILABLE
    # Exact, as bind_request compares: a case variant is not the recipe's id.
    if model_id != KONTEXT_V1.model_id:
        return INSTRUCTION_EDIT_NEEDS_MODEL
    return None


def text_to_image_refusal(model_id: str, *, qualified: bool) -> str:
    """Generate's words where the selected checkpoint only edits.

    Pointing at Edit by instruction… is right only where that can queue;
    otherwise the edit gate's own reason is said, never a dead end.
    """
    reason = edit_unavailable_reason(model_id, qualified=qualified)
    if reason is None:
        return INSTRUCTION_EDIT_TEXT_TO_IMAGE
    return (
        "FLUX.1 Kontext [dev] edits a gallery image; it does not generate from text, "
        f"and instruction editing is held: {reason}"
    )
