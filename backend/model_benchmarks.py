"""Cold speed summaries from the RTX 5080 samples in docs/verified.md.

These are coarse comparisons at the recorded settings, not generation ETAs.
Keep exact catalogue variants separate and never infer a timing from steps.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TypedDict

from .launch import CHECKPOINTS, FORMATS
from .model_defaults import defaults_for


@dataclass(frozen=True)
class Sample:
    seconds: float  # Upper end of the recorded sample/range, for the category.
    summary: str
    steps: int
    date: str
    timing: str
    size: str = "1024×1024 (1.05 MP)"
    guidance: str = ""
    caveat: str = ""
    quantization: str = "fp8"
    platform: str = "win32"
    format: str | None = None
    offload: str = "model"
    basis: str = "Measured"
    # A first image of a process on a busy PC is not the checkpoint's speed:
    # architectural twins read Medium and Fast from such samples, so it gets
    # no category.
    warm: bool = True

    @property
    def settings(self) -> str:
        quantization = "Nunchaku NVFP4" if self.quantization == "nunchaku" else "FP8"
        platform = "Windows" if self.platform == "win32" else "Linux/WSL"
        offload = "model offload" if self.offload == "model" else "no offload"
        return f"{quantization}, {platform}, {offload}"


_AFTER_CANCEL = "First completed image after cancellation and a repeat, respectively."
_BUSY_CPU = (
    "First image of a process; CPU about 75% busy with other work, so not a warm "
    "timing."
)
_BUSY = {"caveat": _BUSY_CPU, "basis": "First run, busy PC", "warm": False}
class _NunchakuSetup(TypedDict):
    quantization: str
    platform: str
    format: str
    size: str


_NVFP4: _NunchakuSetup = {"quantization": "nunchaku", "platform": "linux", "format": "nvfp4", "size": "1 MP"}

# The table's approximately 1 MP direct renders, at the model's normal step
# count. Keep WAI's actual 1.38 MP portrait visible. Krea dev's old 8-step
# samples must not price its 28-step default. Missing variants stay unmeasured.
_SAMPLES: dict[str, tuple[Sample, ...]] = {
    "black-forest-labs/FLUX.1-Krea-dev": (
        Sample(42.6, "~42 s · 1 MP", 28, "2026-09-27", "41.6–42.6 s", guidance="CFG 3.5",
               caveat="Warm samples; first image after loading took 60.8 s. Loading took 38.2 s separately."),
    ),
    "krea/Krea-2-Turbo": (
        Sample(15, "~15 s · 1 MP", 8, "2026-09-05", "15 s", **_NVFP4),
        Sample(26, "~26 s · 1 MP", 8, "2026-09-05", "26 s", offload="none",
               caveat="The recorded no-offload setup spilled into system RAM; 2 MP took 72 s.", **_NVFP4),
    ),
    "cagliostrolab/animagine-xl-4.0": (
        Sample(17.2, "~17 s · 1 MP", 28, "2026-10-05",
               "17.2 s; 0.22 s a step after the first", guidance="CFG 5", **_BUSY),
    ),
    "KBlueLeaf/Kohaku-XL-Zeta": (
        Sample(9.68, "~10 s · 1 MP", 28, "2026-09-26", "9.33 / 9.68 s", guidance="CFG 5",
               caveat=_AFTER_CANCEL),
    ),
    "OnomaAIResearch/Illustrious-XL-v2.0": (
        Sample(11.1, "~11 s · 1 MP", 28, "2026-10-05", "11.1 s; 0.22 s a step",
               guidance="CFG 5", **_BUSY),
    ),
    "LyliaEngine/waiIllustriousSDXL_v170": (
        Sample(9.64, "~10 s · 1.38 MP", 28, "2026-09-26", "9.25 / 9.64 s",
               size="1024×1344 (1.38 MP)", guidance="CFG 6", caveat=_AFTER_CANCEL),
    ),
    "Minthy/RouWei-0.8": (
        Sample(7.882, "~8 s · 1 MP", 28, "2026-10-07", "7.638 / 7.882 s", guidance="CFG 6",
               caveat=_AFTER_CANCEL),
    ),
    "Laxhar/noobai-XL-Vpred-1.0": (
        Sample(8.490, "~8 s · 1 MP", 28, "2026-10-07", "8.078 / 8.490 s", guidance="CFG 5",
               caveat=_AFTER_CANCEL + " Other CPU work and tests ran during qualification."),
    ),
    "SeeSee21/Z-Anime": (
        Sample(79.2, "~79 s · 1 MP", 40, "2026-09-19", "78.1 / 79.2 s", guidance="CFG 4",
               caveat=_AFTER_CANCEL + " The 4.19 MP sample took 519.1 s."),
    ),
    "circlestone-labs/Anima": (
        Sample(5.68, "~4–6 s · 1 MP", 8, "2026-09-26", "5.68 / 4.40 s", guidance="CFG 1",
               caveat=_AFTER_CANCEL),
    ),
    "circlestone-labs/Anima@aesthetic-v1.1": (
        Sample(23.229, "~23 s · 1 MP", 30, "2026-10-07", "23.229 / 22.582 s", guidance="CFG 4",
               caveat=_AFTER_CANCEL + " Includes previews; the CPU test suite ran during these samples."),
    ),
    "Gazingstars123/Anima-2.9B": (
        Sample(32.037072, "~32 s · 1 MP", 30, "2026-10-24", "32.037 / 31.203 s", guidance="CFG 4",
               caveat=_AFTER_CANCEL + " Includes previews; other project CPU work remained active. "
               "Runtime smoke samples, not a controlled benchmark."),
    ),
    "duongve/NetaYume-Lumina-Image-2.0": (
        Sample(38.74, "~39 s · 1 MP", 40, "2026-09-26", "38.46 / 38.74 s", guidance="CFG 5",
               caveat=_AFTER_CANCEL),
    ),
    "NewBie-AI/NewBie-image-Exp0.1": (
        Sample(39.39, "~39 s · 1 MP", 28, "2026-09-26", "39.39 / 38.74 s", guidance="CFG 4",
               caveat=_AFTER_CANCEL + " Author's XML prompt; includes previews."),
    ),
    "black-forest-labs/FLUX.1-dev": (
        Sample(30, "~30 s · 1 MP", 28, "2026-09-08", "30 s", **_NVFP4),
    ),
    "black-forest-labs/FLUX.1-schnell": (
        Sample(27.3, "~27 s · 1 MP", 4, "2026-10-05",
               "27.3 s, 19.5 s of it the first step; 1.4 s a step after", **_BUSY),
        Sample(23, "~16–23 s · 1 MP", 4, "2026-09-08", "16–23 s", **_NVFP4),
    ),
    "black-forest-labs/FLUX.2-klein-4B": (
        Sample(10.2, "~10 s · 1 MP", 4, "2026-09-08", "10.2 s", caveat="Text-to-image, without reference images."),
        Sample(4, "~4 s · 1 MP", 4, "2026-09-05", "4.0 s", **_NVFP4),
    ),
    "Tongyi-MAI/Z-Image-Turbo": (
        Sample(14.3, "~14 s · 1 MP", 8, "2026-10-05",
               "14.3 s; 0.9 s a step after the first", **_BUSY),
        Sample(10.6, "~8–11 s · 1 MP", 8, "2026-09-08", "8 s and 10.6 s",
               caveat="Two same-day samples. A dropped WSL page cache also produced 17–19 s before warming.", **_NVFP4),
    ),
    "Qwen/Qwen-Image": (
        Sample(78, "~78 s · 1 MP", 50, "2026-09-08", "78 s", guidance="True CFG 4 with a negative prompt",
               caveat="Warm sample; first image of a session took 210 s. Loading took 22 s separately.", **_NVFP4),
    ),
    "baidu/ERNIE-Image-Turbo": (
        Sample(10.2, "~10 s · 1 MP", 8, "2026-09-08", "9.6–10.2 s",
               caveat="Warm samples; first image of a process took 17–18 s. Loading took 22 s separately.", **_NVFP4),
    ),
    # The catalogue's last entry, which `speed_for` needs to reach it:
    # instruction edits of a 1:1 source, not text-to-image renders.
    "black-forest-labs/FLUX.1-Kontext-dev": (
        Sample(29.3, "~26–29 s · 512×512 edit", 28, "2026-10-27",
               "29.3 s for the first edit after loading; 25.8–26.8 s after",
               size="512×512 (0.26 MP)", guidance="guidance 2.5",
               caveat="Instruction edits of a square source, one image each, through the "
               "harness's stand-in gate. Loading took 38.1 s separately."),
    ),
}


def speed_for(
    model_id: str, quantization: str | None = None, offload: str = "model",
) -> dict[str, str]:
    """Return a measured sample or an explicitly labelled different-setup reference."""
    checkpoint = next((
        entry for entry in CHECKPOINTS
        if model_id == entry.repo or any(model_id == entry.lite_repo(fmt) for fmt, _ in FORMATS)
    ), None)
    fmt = next((
        name for name, _ in FORMATS
        if checkpoint is not None and checkpoint.lite_repo(name) == model_id
    ), None)
    quantization = quantization or ("nunchaku" if fmt else "fp8")
    samples = _SAMPLES.get(checkpoint.repo, ()) if checkpoint else ()
    if not samples:
        steps = defaults_for(model_id)["steps"]
        return {
            "speed_label": "Unmeasured",
            "speed_summary": f"{steps} default steps" if checkpoint else "No recorded timing",
            "speed_basis": "No timing",
            "speed_note": (
                "No RTX 5080 generation timing has been recorded for this exact checkpoint. "
                "Step count alone does not establish speed."
            ),
        }

    # Prefer the requested quantization/format, then its family (4-bit or
    # torchao), then platform and offload. A fallback is context only: its
    # category is never shown as this selection's, and BF16 or sequential
    # offload, far from any sample's setup, is Unmeasured.
    sample = max(samples, key=lambda value: (
        value.quantization == quantization and value.format == fmt,
        (value.quantization == "nunchaku") == (quantization == "nunchaku"),
        value.platform == sys.platform,
        value.offload == offload,
    ))
    matched = (
        sample.quantization == quantization and sample.format == fmt
        and sample.platform == sys.platform and sample.offload == offload
    )
    if matched and sample.warm:
        label = "Fast" if sample.seconds <= 15 else "Medium" if sample.seconds <= 45 else "Slow"
    elif matched:
        label = "Reference"
    else:
        label = "Unmeasured" if quantization == "none" or offload == "sequential" else "Reference"
    guidance = f", {sample.guidance}" if sample.guidance else ""
    context = (
        "" if matched and sample.warm
        else "Reference only: a first image on a busy PC, not a warm timing, so it "
        "gets no speed category. " if matched
        else "Reference from a different setup; this selection has no matching timing, "
        "so it gets no speed category. "
    )
    return {
        "speed_label": label,
        "speed_summary": sample.summary,
        "speed_basis": sample.basis if matched else _timed_with(sample, quantization, fmt, offload),
        "speed_note": (
            f"{context}RTX 5080 sample: {sample.timing} per image at {sample.size}, "
            f"{sample.steps} steps{guidance}; {sample.settings}, {sample.date}. "
            f"{sample.caveat + ' ' if sample.caveat else ''}"
            "Generation only; model loading is separate. "
            "Categories use the recorded sample's upper time: Fast ≤15 s, Medium >15–45 s, Slow >45 s. "
            "Single test runs, not a controlled comparison or an ETA; size, steps and other work change the time."
        ),
    }


def _timed_with(sample: Sample, quantization: str, fmt: str | None, offload: str) -> str:
    """Name, in plain words, what the reference sample ran with that differs."""
    if sample.quantization == "nunchaku":
        assert sample.format is not None, "a Nunchaku timing must name its format"
        quant = (
            "" if quantization == "nunchaku" and sample.format == fmt
            else sample.format.upper() if quantization == "nunchaku"
            else "on 4-bit Linux build"
        )
    else:
        quant = "" if sample.quantization == quantization else "FP8"
    where = ""
    if sample.platform != sys.platform and quant != "on 4-bit Linux build":
        where = "on Windows" if sample.platform == "win32" else "on Linux"
    placement = f"{'model' if sample.offload == 'model' else 'no'} offload" if sample.offload != offload else ""
    setup = ", ".join(filter(None, [" ".join(filter(None, [quant, where])), placement]))
    return f"timed {'' if setup.startswith('on ') else 'with '}{setup}"
