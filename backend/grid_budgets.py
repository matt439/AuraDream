"""Each resolved prompt's encoder budget, for the grid preview (PA#15).

Check measures the prompt box as typed, so on a grid it counts the `{name}`
template: one value can carry a resolved prompt past an encoder's budget,
where the pipeline cuts it in silence, while the template looks short. The
preview already lists the resolved prompts; this measures each distinct one,
exactly as `prompt_encoding.measure` measures Check's (the string the
pipeline builds, per encoder, the negative only as `applied()` runs it), and
says which ones lose text and which values every losing prompt shares.

Cold, as Check is: the CPU tokenizers on a daemon thread (the route's
`detached`), never the image model, never the gate. Bounded: the distinct
pairs are measured in render order up to `MAX_MEASURED`, and a first
tokenizer load is waited for `WAIT` seconds at most, after which `detail`
says the counts are unknown rather than that nothing is cut. A tokenizer
that cannot be read (gated, offline, still loading) stops the measuring at
the first prompt it misses: `prompt_encoding` caches only a load that
succeeded, so each further prompt would start another load and wait again.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from . import prompt_encoding
from .grid import VARIANT, Cell
from .schemas_encoder import GridPromptBudget, GridPromptBudgets
from .schemas_generate import GenerateRequest

# A preview can resolve to a thousand prompts; tokenizing each at every
# pause in typing is not free. Past this the answer says how many it read.
MAX_MEASURED = 128
# A cold tokenizer's first load is shared with the prompt box's own measure;
# the preview's totals never wait on it for long.
WAIT = 5.0


def _label(variables: dict[str, str] | None) -> str | None:
    if not variables:
        return None
    if VARIANT in variables:
        return variables[VARIANT]
    return ", ".join(f"{name}={value or 'none'}" for name, value in variables.items())


def _loses_text(budget: GridPromptBudget) -> bool:
    # A pooled-only encoder's cut is advice (FLUX.1's CLIP-L): another reads on.
    return any(report.cut is not None and not report.pooled_only for report in budget.encoders)


def _cutting_values(budgets: Sequence[GridPromptBudget]) -> dict[str, list[str]]:
    """Each variable's values whose every measured prompt loses text, where
    some other prompt does not: the value is then what tips them over. With
    every prompt cut it is the template, and no value is named."""
    if not any(b.cut for b in budgets) or all(b.cut for b in budgets):
        return {}
    seen: dict[str, dict[str, bool]] = {}
    for budget in budgets:
        for name, value in (budget.variables or {}).items():
            if name == VARIANT:
                continue
            every = seen.setdefault(name, {})
            every[value] = every.get(value, True) and budget.cut
    return {
        name: [value for value, cut in values.items() if cut]
        for name, values in seen.items()
        if any(values.values())
    }


def measure_cells(
    model_id: str,
    cells: Sequence[Cell],
    applied: Callable[[GenerateRequest], GenerateRequest],
    *,
    limit: int = MAX_MEASURED,
    wait: float = WAIT,
) -> GridPromptBudgets | None:
    """The grid's distinct resolved (prompt, negative) pairs, measured.

    None where nothing is substituted (no cell carries variables: Check's own
    count is the prompt's) or where the encoder table has no entry for the
    checkpoint (nothing would ever count it; Check says why). Blocking.
    """
    if not any(cell.variables for cell in cells):
        return None
    specs = prompt_encoding.encoders_for(model_id)
    if specs is None:
        return None
    expected = {spec.name for spec in specs}
    pairs: dict[tuple[str, str | None], dict[str, str] | None] = {}
    for cell in cells:
        run = applied(cell.request)
        pairs.setdefault((cell.request.prompt, run.negative_prompt), cell.variables)
    budgets: list[GridPromptBudget] = []
    detail = None
    # One wait for the whole preview, not one per prompt: a stalled load
    # would otherwise hold it `limit` times over.
    deadline = time.monotonic() + wait
    for (prompt, negative), variables in list(pairs.items())[:limit]:
        reports, _findings, unmeasured = prompt_encoding.measure(
            model_id, prompt, negative, wait=max(0.0, deadline - time.monotonic()),
        )
        detail = detail or unmeasured
        budget = GridPromptBudget(label=_label(variables), variables=variables, encoders=reports)
        budget.cut = _loses_text(budget)
        budgets.append(budget)
        if expected - {report.name for report in reports if report.field == "prompt"}:
            # An encoder went unread: its load failed or ran out of the
            # wait. One try per preview, as Check makes one per check.
            break
    return GridPromptBudgets(
        prompts=budgets,
        distinct=len(pairs),
        detail=detail,
        cutting_values=_cutting_values(budgets),
    )
