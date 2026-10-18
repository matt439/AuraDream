"""Read-only aggregate queries over the gallery: the preview's cost and size
estimates (and the formatting of that disk estimate), the starred-rate table
the stats CLI prints, and the Runs summary.

No writes and no file changes. Like `bundles`, everything storage owns -
`settings`, `_session`, `_LIVE` - is read through `storage` at call time, so
the tests' replacement of `storage.settings` reaches these queries too.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from collections.abc import Sequence
from typing import Any

from . import storage
# VARIANT is the variable a comparison's versions are recorded under, which
# `keep_rates` groups by.
from .grid import VARIANT
from .refine import start_index
from .schemas import RunSummary
from .search import TRUE_CFG_SQL, strict_scope

log = logging.getLogger(__name__)


def human_bytes(n: int | float) -> str:
    """Binary, and labelled so (KiB, MiB, GiB), as the page's humanBytes:
    the CLI's disk line and the preview's size are the same estimate."""
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def step_cost_ms(
    width: int, height: int, model_id: str | None = None, sample: int = 20
) -> float | None:
    """Median direct-render milliseconds per step at this size and checkpoint.

    Per *step*, not per image: a grid varies the step count across its cells, so
    an average image duration would over- or under-state most of them. Matched
    on exact dimensions - cost scales with pixels and offload behaviour, neither
    of which extrapolates honestly from a different size - and on the checkpoint,
    since klein's step is about half of Turbo's at the same size. None when the
    gallery has nothing to measure, which is the caller's cue to promise nothing.

    Refine rows are kept out: their fixed encode/decode costs dominate at low
    strengths and cannot price a direct render. See `refine_cost_ms`.
    """
    scope = "AND model_id = ?" if model_id else ""
    scope_params = (model_id,) if model_id else ()
    with storage._session() as conn:
        rows = conn.execute(
            f"""SELECT duration_ms, steps FROM images
               WHERE width = ? AND height = ? AND steps > 0 AND duration_ms > 0
                 AND refine IS NULL {scope}
               ORDER BY id DESC LIMIT ?""",
            (width, height, *scope_params, sample),
        ).fetchall()
    if not rows:
        return None
    costs = sorted(row["duration_ms"] / row["steps"] for row in rows)
    return costs[len(costs) // 2]


def refine_cost_ms(
    width: int, height: int, model_id: str, sample: int = 100, inpaint: bool = False,
) -> dict[int, float]:
    """Median full refine duration per image, grouped by denoised step count.

    Same size and checkpoint only, like the direct estimate. Matching the
    count keeps the fixed resize, text and VAE work in every image's price;
    one strength's timing is never extrapolated as a per-step rate. Unknown
    families provide no estimate until their schedule rounding is known.

    An inpaint is priced from inpaints alone (`inpaint`) and a refine pass
    from passes with no mask: the inpaint classes encode the source twice and
    composite, so neither history says what the other costs.
    """
    masked = "mask IS NOT NULL" if inpaint else "mask IS NULL"
    with storage._session() as conn:
        rows = conn.execute(
            f"""SELECT duration_ms, steps, strength FROM images
               WHERE width = ? AND height = ? AND model_id = ? AND steps > 0
                 AND duration_ms > 0 AND refine IS NOT NULL AND strength IS NOT NULL
                 AND {masked}
               ORDER BY id DESC LIMIT ?""",
            (width, height, model_id, sample),
        ).fetchall()
    durations: dict[int, list[float]] = {}
    for row in rows:
        start = start_index(row["steps"], row["strength"], model_id, inpaint=inpaint)
        if start is not None:
            durations.setdefault(row["steps"] - start, []).append(row["duration_ms"])
    return {steps: sorted(values)[len(values) // 2] for steps, values in durations.items()}


def image_bytes(width: int, height: int, model_id: str | None = None, sample: int = 20) -> int | None:
    """Median PNG size on disk at this size on this checkpoint, or None.

    The third of the preview's three estimates, and the weakest: PNG size
    varies with content far more than duration does. Same rule as the
    others - no files, no estimate - and the same scoping, since a klein
    render and a Turbo render of one size are not the same picture.
    """
    scope = "AND model_id = ?" if model_id else ""
    scope_params = (model_id,) if model_id else ()
    with storage._session() as conn:
        names = [
            row["filename"]
            for row in conn.execute(
                f"""SELECT filename FROM images WHERE width = ? AND height = ? {scope}
                    ORDER BY id DESC LIMIT ?""",
                (width, height, *scope_params, sample),
            )
        ]
    sizes = []
    for name in names:
        try:
            sizes.append((storage.settings.outputs_dir / name).stat().st_size)
        except OSError:
            continue
    if not sizes:
        return None
    sizes.sort()
    return sizes[len(sizes) // 2]


# The columns `keep_rates` can group by, as SQL. CFG and mu are rounded to the
# six decimals the axis parser produces, so a value that went through REAL and
# back does not split one setting into two buckets.
_GROUPS: dict[str, str] = {
    "steps": "steps",
    "cfg": "round(guidance_scale, 6)",
    "tcfg": f"round({TRUE_CFG_SQL}, 6)",
    "mu": "round(mu, 6)",
    "prompt": "prompt",
    "negative": "negative_prompt",
    "seed": "seed",
    "size": "width || 'x' || height",
    "model": "model_id",
    "quant": "quantization",
    # The adapter and the weight it ran at: "which settings produced my
    # favourites" is exactly the question a LoRA weight is chosen by.
    "lora": "lora",
    "lorascale": "round(lora_scale, 6)",
    # A refine pass's strength, and the image it started from: a strength
    # sweep is starred by how much of the source it kept.
    "strength": "round(strength, 6)",
    "refine": "refine",
    # An inpaint's mask, by the 16 digits the A1111 line prints: which region
    # of a source the kept passes repainted. NULL on everything else.
    "mask": "substr(mask, 1, 16)",
    # An outpaint by its spec (`fill:L,T,R,B@band`): which padding the kept
    # outpaints were made with. NULL on everything else.
    "outpaint": "outpaint",
    # A comparison's version by its label (`prompt_variants`): the starred
    # rate per version. `prompt` cannot say it - reorders share their opening
    # words, which is all a table cell has room for.
    VARIANT: f"json_extract(variables, '$.{VARIANT}')",
}


def keep_rates(
    search: str | None = None, by: Sequence[str] = ("steps", "cfg", "mu"), min_made: int = 1
) -> list[dict[str, Any]]:
    """Which settings the starred images cluster under: made, starred, and the rate.

    The naive count rewards whichever setting was used most; the rate is what
    is wanted, and every input is in the table. `search` is the gallery's own
    `key:value` grammar and scopes the rows - a run, where `expand()` crossed
    the axes against the same prompts and seeds, is the honest scope; all-time
    is confounded by whatever settings were in use while the prompts happened
    to be good. Groups with fewer than `min_made` images are dropped so a
    lucky 1-of-1 does not take first place. The column is *starred*, not
    kept: nothing records a rejection, so an untouched night reads 0% rather
    than unknown. Size is always in the grouping - a 4 MP re-render is not
    the same setting as its 1 MP source. Rows rank by rate, then by count;
    a tie keeps the order the settings were first made in, so an untouched
    comparison reads its versions as the grid wrote them.
    """
    unknown = [name for name in by if name not in _GROUPS]
    if unknown:
        raise ValueError(f"cannot group by {', '.join(unknown)}; one of {', '.join(_GROUPS)}")
    columns = list(dict.fromkeys([*by, "size"]))
    where, params = strict_scope(search)
    # The recycle bin is a rejection the table has no column for; a binned
    # image is neither made nor starred here.
    where.append(storage._LIVE)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    select = ", ".join(f"{_GROUPS[name]} AS {name}" for name in columns)
    group = ", ".join(_GROUPS[name] for name in columns)
    with storage._session() as conn:
        rows = conn.execute(
            f"""SELECT {select}, COUNT(*) AS made, SUM(favorite) AS starred
                FROM images {clause}
                GROUP BY {group}
                HAVING made >= ?
                ORDER BY CAST(starred AS REAL) / made DESC, made DESC, MIN(id)""",
            (*params, min_made),
        ).fetchall()
        if VARIANT in columns and "prompt" not in columns and rows:
            # A run can join comparisons with swapped original/yours labels.
            # Check complete assignments within each emitted stats group:
            # variant x colour is an ordinary grid and may aggregate colours,
            # and grouping by steps/seed can already separate the versions.
            emitted = {tuple(row[name] for name in columns) for row in rows}
            prompts = {}
            for row in conn.execute(
                f"SELECT {select}, variables, prompt FROM images {clause} "
                f"GROUP BY {group}, variables, prompt", params
            ):
                grouped = tuple(row[name] for name in columns)
                if grouped not in emitted or row[VARIANT] is None:
                    continue
                assignment = json.dumps(json.loads(row["variables"]), sort_keys=True)
                key = (grouped, assignment)
                if key in prompts and prompts[key] != row["prompt"]:
                    raise ValueError(
                        "a variant label refers to more than one prompt for the same "
                        "variable assignment in this scope; use --by variant,prompt "
                        "or narrow the scope with --job"
                    )
                prompts[key] = row["prompt"]
    return [
        {**{name: row[name] for name in columns}, "made": row["made"], "starred": row["starred"],
         "rate": row["starred"] / row["made"]}
        for row in rows
    ]


# The settings a run can vary, as the Runs picker names them: each one's
# count of distinct values over the run (a NULL counted as a value of its own
# where the column takes one), read back as RunSummary.axes.
_RUN_AXES = {
    "steps": "steps",
    "cfg": "guidance_scale",
    "mu": "COALESCE(mu, 'none')",
    "true_cfg": "COALESCE(true_cfg, 'none')",
    "lora_scale": "COALESCE(lora || '@' || lora_scale, 'none')",
    "strength": "COALESCE(strength, 'none')",
    "reference": "COALESCE(reference, 'none')",
    "seed": "seed",
    "variant": "CASE WHEN json_valid(variables) THEN json_extract(variables, '$.variant') END",
}
_AXIS_COUNTS = ", ".join(f"COUNT(DISTINCT {expr}) AS axis_{name}" for name, expr in _RUN_AXES.items())


def runs(limit: int = 20, model_id: str | None = None) -> list[RunSummary]:
    """The gallery grouped by run, the most recently written first.

    What `GET /api/runs` serves and the Runs picker lists: the night's output
    as one line per grid rather than 480 cards to scroll past. Rows written
    before runs had an id are not a run and are left out. Size, checkpoint and
    quantizer are constant within a run - `run_id` hashes them - so they are
    read off the oldest image along with its prompt and its seed, which is the
    base seed the grid was started from. `model_id` keeps the list to one
    checkpoint's runs, as the gallery it is a table of contents for is kept.
    Binned images are left out, as the gallery leaves them out, and a run
    entirely in the recycle bin is not listed.
    """
    scope = "AND model_id = ?" if model_id else ""
    scope_params = (model_id,) if model_id else ()
    with storage._session() as conn:
        groups = conn.execute(
            f"""SELECT run_id, COUNT(*) AS images, SUM(favorite) AS favorites,
                      COUNT(DISTINCT prompt) AS prompts, COUNT(DISTINCT job_id) AS jobs,
                      COUNT(content_hash) AS hashed,
                      COUNT(DISTINCT content_hash) AS distinct_hashes,
                      MIN(id) AS first_id, MAX(id) AS last_id,
                      MIN(created_at) AS started_at, MAX(created_at) AS finished_at,
                      {_AXIS_COUNTS},
                      COUNT(DISTINCT COALESCE(refine, -1)) AS n_sources, MIN(refine) AS refine,
                      MIN(strength) AS strength_min, MAX(strength) AS strength_max
               FROM images WHERE run_id IS NOT NULL AND binned_at IS NULL {scope}
               GROUP BY run_id ORDER BY last_id DESC LIMIT ?""",
            (*scope_params, limit),
        ).fetchall()
        firsts: dict[int, sqlite3.Row] = {}
        if groups:
            marks = ",".join("?" * len(groups))
            for row in conn.execute(
                f"SELECT * FROM images WHERE id IN ({marks})", [g["first_id"] for g in groups]
            ):
                firsts[row["id"]] = row
    summaries = []
    for group in groups:
        first = firsts[group["first_id"]]
        summaries.append(
            RunSummary(
                run_id=group["run_id"],
                images=group["images"],
                # Only when every image of the run is hashed: a count over
                # half a run is a number that looks like an answer.
                distinct=(
                    group["distinct_hashes"] if group["hashed"] == group["images"] else None
                ),
                favorites=group["favorites"],
                prompts=group["prompts"],
                jobs=group["jobs"],
                prompt=first["prompt"],
                seed=first["seed"],
                model_id=first["model_id"],
                quantization=first["quantization"],
                width=first["width"],
                height=first["height"],
                started_at=group["started_at"],
                finished_at=group["finished_at"],
                axes={name: group[f"axis_{name}"] for name in _RUN_AXES if group[f"axis_{name}"] > 1},
                refine=group["refine"] if group["n_sources"] == 1 else None,
                strengths=(
                    [group["strength_min"], group["strength_max"]]
                    if group["n_sources"] == 1 and group["refine"] is not None else None
                ),
            )
        )
    # Retained jobs can exist before their first image, or after every image
    # was permanently deleted. Keep those runs discoverable after restart.
    # Entirely binned runs stay in the bin, as before; this adds only runs
    # with no row at all. Read/decompress only these recovery candidates.
    with storage._session() as conn:
        recoverable = conn.execute(
            f"""SELECT run_id FROM run_jobs r
                WHERE NOT EXISTS (SELECT 1 FROM images i WHERE i.run_id = r.run_id)
                {'AND model_id = ?' if model_id else ''}
                GROUP BY run_id ORDER BY MAX(created_at) DESC LIMIT ?""",
            (*scope_params, limit),
        ).fetchall()
    for candidate in reversed(recoverable):
        try:
            saved = storage.retained_run_jobs(candidate["run_id"])
        except ValueError as exc:
            # One obsolete/corrupt recovery row must not take the whole
            # Runs picker down. Its direct action endpoint still says why
            # those cells cannot be read; never invent summary metadata.
            log.warning("omitting unreadable retained run %s: %s", candidate["run_id"], exc)
            continue
        if not saved or not saved[0]["cells"]:
            continue
        first, last = saved[0], saved[-1]
        cells = [cell for job in saved for cell in job["cells"]]
        request = first["cells"][0].request
        stamp = lambda value: datetime.fromtimestamp(value, timezone.utc).isoformat()
        recovered = RunSummary(
            run_id=first["run_id"], images=0, distinct=0, favorites=0,
            prompts=len({cell.request.prompt for cell in cells}), jobs=len(saved),
            prompt=first["metadata"].get("prompt_template") or request.prompt,
            seed=first["cells"][0].seed, model_id=first["model_id"],
            quantization=first["quantization"], width=first["width"], height=first["height"],
            started_at=stamp(first["created_at"]),
            finished_at=stamp((last.get("outcome") or {}).get("finished_at") or last["created_at"]),
        )
        # Preserve the existing live runs' id order, while inserting an empty
        # run near the time it began. A just-submitted zero-image run leads.
        position = next((index for index, summary in enumerate(summaries)
                         if summary.started_at <= recovered.started_at), len(summaries))
        summaries.insert(position, recovered)
    return summaries[:limit]
