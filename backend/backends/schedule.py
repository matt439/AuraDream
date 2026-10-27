"""The flow-matching schedule, and how a requested mu becomes *the* shift.

Krea's CLI exposes --mu as the whole shift, which is what it has to mean here
too. A scheduler that shifts dynamically takes the value through
`set_timesteps(mu=...)`, which `forced_mu` intercepts; one that does not
takes a pre-shifted `sigmas` list from `shifted_sigmas`, less the static
shift it applies on top regardless (`static_shift`), so the total is still
e^mu - `requested_shift` says which, for the backend's call and for the
Refine preview alike. Assert on `scheduler.sigmas`, never on the list
computed here.

`start_sigma` is how much noise a refine pass starts at, for the Refine
dialog's preview (and the Inpaint and Outpaint dialogs'): the schedule the
image-to-image or inpaint class would set, on a copy of the loaded
scheduler, read at the pass's start index.
"""
from __future__ import annotations

import functools
import importlib
import inspect
import logging
import math
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, NamedTuple

log = logging.getLogger(__name__)


def shifted_sigmas(steps: int, mu: float) -> list[float]:
    """The schedule a shift of `mu` produces, computed here rather than by the scheduler.

        sigma' = e^mu / (e^mu + (1/sigma - 1))

    which is `FlowMatchEulerDiscreteScheduler._time_shift_exponential` over
    the default `linspace(1, 1/steps, steps)` ramp, to the byte. Only used
    for a scheduler that does *not* shift dynamically - see `shifts_itself`
    and `static_shift`, which decides how much of `mu` is left to apply.
    """
    out: list[float] = []
    for i in range(steps):
        sigma = 1.0 - i / steps
        out.append(math.exp(mu) / (math.exp(mu) + (1.0 / sigma - 1.0)))
    return out


def shifts_itself(pipe: Any) -> bool:
    """Does this pipeline's scheduler apply a shift of its own?

    `sigmas` is not the final schedule. Every flow-matching pipeline here
    derives a mu - 1.15 hard-coded for a distilled checkpoint, resolution
    dependent otherwise - and hands it to `set_timesteps`, which shifts
    whatever sigmas it was given whenever `use_dynamic_shifting` is set (it
    is, in both shipped `scheduler_config.json`s). The shift multiplies the
    odds sigma/(1-sigma) by e^mu, so two of them compose into a single shift
    of `mu_a + mu_b`: passing a pre-shifted schedule does not replace the
    pipeline's mu, it adds to it.
    """
    config = getattr(getattr(pipe, "scheduler", None), "config", None)
    return bool(getattr(config, "use_dynamic_shifting", False))


def static_shift(pipe: Any) -> float:
    """The shift a non-dynamic scheduler applies to *whatever* sigmas it gets.

    `use_dynamic_shifting: false` does not mean no shift. `set_timesteps` is
    an if/else, and the else branch runs `shift * s / (1 + (shift - 1) * s)`
    - which multiplies the odds s/(1-s) by `shift`, exactly as the dynamic
    branch multiplies them by `e^mu`. The two compose, so a pre-shifted
    schedule handed to a scheduler with `shift=3.0` (SD3, SD3.5; Lumina 2
    ships 6.0) landed at `mu + ln 3` while the row recorded `mu`. The
    instance attribute, not `config.shift`: `set_timesteps` reads
    `self.shift`, which `set_shift()` can change after construction.
    """
    scheduler = getattr(pipe, "scheduler", None)
    shift = getattr(scheduler, "shift", None)
    if shift is None:
        shift = getattr(getattr(scheduler, "config", None), "shift", 1.0)
    return float(shift)


def warn_about_post_processing(pipe: Any) -> None:
    """Say so once when the scheduler reshapes the schedule after the shift.

    `shift_terminal` stretches it and the karras/exponential/beta options
    replace it outright, in both branches - so a requested mu then means
    something other than "the shift that landed". None of the shipped
    checkpoints set any of them; an arbitrary repo might.
    """
    config = getattr(getattr(pipe, "scheduler", None), "config", None)
    active = [
        key
        for key in ("shift_terminal", "use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas")
        if getattr(config, key, None)
    ]
    if active:
        log.warning(
            "scheduler post-processes the schedule (%s); a requested mu is not the "
            "shift that lands",
            ", ".join(active),
        )


@contextmanager
def forced_mu(pipe: Any, mu: float | None):
    """Run the pipeline with the scheduler shifted by `mu` and nothing else.

    Intercepting the value on its way to the scheduler is the only way to
    say "mu is the whole shift" without re-deriving `calculate_shift` - and
    with it the latent sequence length, VAE scale factor and patch size -
    outside the pipeline that owns them.

    `functools.wraps` is load-bearing: `retrieve_timesteps` reads
    `inspect.signature(scheduler.set_timesteps)` to decide whether custom
    sigmas are supported at all, and only finds the real signature by
    following `__wrapped__`. Patching the instance is safe because
    generation is serialized through a single worker thread.
    """
    scheduler = getattr(pipe, "scheduler", None)
    original = getattr(scheduler, "set_timesteps", None)
    signature = inspect.signature(original) if original is not None else None
    if mu is None or signature is None or "mu" not in signature.parameters:
        if mu is not None:
            log.debug("scheduler takes no mu; ignoring mu=%s", mu)
        yield
        return

    assert scheduler is not None and original is not None  # A readable method supplied this signature.

    @functools.wraps(original)
    def set_timesteps(*args: Any, **kwargs: Any):
        bound = signature.bind(*args, **kwargs)
        bound.arguments["mu"] = mu
        return original(*bound.args, **bound.kwargs)

    scheduler.set_timesteps = set_timesteps
    try:
        yield
    finally:
        # Drop the instance attribute; the class's bound method resurfaces.
        scheduler.__dict__.pop("set_timesteps", None)


def mu_supported(pipe: Any, call_params: set[str]) -> bool:
    """Can a requested mu reach this pipeline's schedule at all?

    First this has to be a flow-matching scheduler. SDXL's Euler scheduler
    accepts a custom `sigmas` list too, but replacing its epsilon schedule
    with `shifted_sigmas` would not be a Mu adjustment - it would be a
    different, unrelated schedule. For flow matching there are two routes:
    a scheduler that shifts for itself takes the value through
    `set_timesteps(mu=...)`, which `forced_mu` intercepts; one that does not
    takes an explicit pre-shifted `sigmas` list, which needs the pipeline to
    accept `sigmas`. With neither, mu is dropped - and has to be recorded as
    dropped, or the row claims a shift the image never had.
    """
    scheduler = getattr(pipe, "scheduler", None)
    if not type(scheduler).__name__.startswith("FlowMatch"):
        return False
    if shifts_itself(pipe):
        original = getattr(scheduler, "set_timesteps", None)
        return original is not None and "mu" in inspect.signature(original).parameters
    return "sigmas" in call_params


def requested_shift(pipe: Any, steps: int, mu: float | None) -> tuple[list[float] | None, float | None]:
    """How a requested mu reaches this pipeline's schedule: `(sigmas, mu)`.

    Where the scheduler shifts for itself, the mu is for `forced_mu` and no
    schedule is handed over; where it does not, a pre-shifted schedule less
    its static shift, so the total is still e^mu, and no mu. `(None, None)`
    for no request: the pipeline's own shift. The backend's call and the
    Refine preview's `start_sigma` both ask here, so the preview's schedule is
    the one the pass would run.
    """
    if mu is not None and not shifts_itself(pipe):
        return shifted_sigmas(steps, mu - math.log(static_shift(pipe))), None
    return None, mu


class StartRecipe(NamedTuple):
    """What an image-to-image or inpaint class's `__call__` does to its
    scheduler before the pass starts, which `start_sigma` repeats on a copy:
    how it measures the latent sequence its shift is computed from, and the
    schedule it hands the scheduler when given none. `source` is the call's
    own statements, as written there - `tests/test_refine_schedules.py` holds
    each to the installed diffusers - and the callables do the same over (the
    pipeline's `vae_scale_factor`, height, width) and (the class's own module,
    steps).
    """

    source: tuple[str, ...]
    length: Callable[[int, int, int], int]
    ramp_source: str
    ramp: Callable[[Any, int], Any]


_PACKED = (
    "image_seq_len = (int(height) // self.vae_scale_factor // 2) * (int(width) // self.vae_scale_factor // 2)",
)
_LINSPACE = "sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas"

# FLUX.1's and Qwen-Image's calls: a packed 2x2 sequence and a linear ramp.
_PACKED_LINSPACE = StartRecipe(
    _PACKED,
    lambda scale, height, width: (int(height) // scale // 2) * (int(width) // scale // 2),
    _LINSPACE,
    lambda module, steps: module.np.linspace(1.0, 1 / steps, steps),
)
# Z-Image's: the latent size rounded to whole patches, and its own schedule.
_Z_IMAGE = StartRecipe(
    (
        "latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))",
        "latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))",
        "image_seq_len = (latent_height // 2) * (latent_width // 2)",
    ),
    lambda scale, height, width: (
        (2 * (int(height) // (scale * 2)) // 2) * (2 * (int(width) // (scale * 2)) // 2)
    ),
    "sigmas = get_default_z_image_sigmas(num_inference_steps)",
    lambda module, steps: module.get_default_z_image_sigmas(steps),
)

# The flow-matching image-to-image and inpaint classes diffusers ships beside
# this app's checkpoints. Each one's call takes mu from its module's
# `calculate_shift` over the sequence length, with the scheduler config's
# base and max shift (`SHIFT_CALL`), and hands both to its module's
# `retrieve_timesteps`. A class not listed here has no start sigma: its
# arithmetic would be a guess. FLUX.2 [klein]'s inpaint class computes its
# shift with `compute_empirical_mu` instead, so it has none.
START_RECIPES: dict[str, StartRecipe] = {
    "FluxImg2ImgPipeline": _PACKED_LINSPACE,
    "QwenImageImg2ImgPipeline": _PACKED_LINSPACE,
    "ZImageImg2ImgPipeline": _Z_IMAGE,
    # An inpaint's call makes the same statements as its family's
    # image-to-image one. Z-Image's and FLUX.1's are enabled; Qwen-Image's
    # family is deferred (`model_defaults.INPAINT_DEFERRED`), so
    # `start_sigma` is never asked for it until it is enabled.
    "ZImageInpaintPipeline": _Z_IMAGE,
    "FluxInpaintPipeline": _PACKED_LINSPACE,
    "QwenImageInpaintPipeline": _PACKED_LINSPACE,
}

# The shift each of them computes, as written in its call.
SHIFT_CALL = (
    "mu = calculate_shift(image_seq_len, "
    "self.scheduler.config.get('base_image_seq_len', 256), "
    "self.scheduler.config.get('max_image_seq_len', 4096), "
    "self.scheduler.config.get('base_shift', 0.5), "
    "self.scheduler.config.get('max_shift', 1.15))"
)


def start_sigma(pipe: Any, sibling: type, request: Any, start: int) -> float | None:
    """The sigma a refine pass (or an inpaint) starts at: the share of noise
    the source is mixed with before the pass denoises it, on a flow-matching
    schedule.

    Read, not modelled: a copy of the loaded scheduler, built from its config
    and given the loaded one's static shift, is handed what `sibling`'s call
    would hand it - the module's own `calculate_shift` over the sequence
    length its call measures (`START_RECIPES`), its default schedule, and a
    requested mu exactly as `requested_shift` and `forced_mu` apply it to
    the pass - through the module's own `retrieve_timesteps`, and the sigma
    at `start` (`refine.start_index`, times the scheduler's order, as the
    call's `get_timesteps` sets its begin index) is the answer. The loaded
    scheduler itself is never touched: a job may be running on it.

    None where that cannot be done without a guess: no scheduler, one that
    is not flow matching (SDXL's epsilon sigmas are no noise fraction), or a
    class whose call is not in `START_RECIPES`. `request` is the pass as
    `applied()` runs it: steps, mu, width and height.
    """
    scheduler = getattr(pipe, "scheduler", None)
    recipe = START_RECIPES.get(getattr(sibling, "__name__", ""))
    scale = getattr(pipe, "vae_scale_factor", None)
    if scheduler is None or recipe is None or scale is None or not type(scheduler).__name__.startswith("FlowMatch"):
        return None
    module = importlib.import_module(sibling.__module__)
    copy = type(scheduler).from_config(scheduler.config)
    if callable(getattr(copy, "set_shift", None)):
        copy.set_shift(static_shift(pipe))
    steps = request.steps
    sigmas, forced = requested_shift(pipe, steps, request.mu)
    if sigmas is None:
        sigmas = recipe.ramp(module, steps)
    config = copy.config
    mu = module.calculate_shift(
        recipe.length(scale, request.height, request.width),
        config.get("base_image_seq_len", 256),
        config.get("max_image_seq_len", 4096),
        config.get("base_shift", 0.5),
        config.get("max_shift", 1.15),
    )
    with forced_mu(SimpleNamespace(scheduler=copy), forced):
        module.retrieve_timesteps(copy, steps, "cpu", sigmas=sigmas, mu=mu)
    return float(copy.sigmas[start * copy.order])
