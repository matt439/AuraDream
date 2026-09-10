"""The flow-matching schedule, and how a requested mu becomes *the* shift.

Krea's CLI exposes --mu as the whole shift, which is what it has to mean here
too. A scheduler that shifts dynamically takes the value through
`set_timesteps(mu=...)`, which `forced_mu` intercepts; one that does not
takes a pre-shifted `sigmas` list from `shifted_sigmas`, less the static
shift it applies on top regardless (`static_shift`), so the total is still
e^mu. Assert on `scheduler.sigmas`, never on the list computed here.
"""
from __future__ import annotations

import functools
import inspect
import logging
import math
from contextlib import contextmanager
from typing import Any

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
