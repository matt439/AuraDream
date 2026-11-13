"""Read-only aggregate queries over the gallery: the preview's cost and size
estimates (and the formatting of that disk estimate), and the pure arithmetic
that prices cells from them (`estimate_seconds`, `seeds_within_budget`), the
starred-rate table the stats CLI prints, and the Runs summary.

No writes and no file changes. Like `bundles`, everything storage owns -
`settings`, `_session`, `_LIVE` - is read through `storage` at call time, so
the tests' replacement of `storage.settings` reaches these queries too.
"""
from __future__ import annotations

import gzip
import json
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from collections.abc import Sequence
from typing import Any

from . import gallery_lock, storage
# VARIANT is the variable a comparison's versions are recorded under, which
# `keep_rates` groups by.
from .grid import VARIANT, Cell, denoised_steps
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
                 AND refine IS NULL AND edit IS NULL {scope}
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
    composite, so neither history says what the other costs. GAN preparation
    is excluded: it adds work that ordinary Refine history cannot price. So
    is an inpaint that ran only the masked area: it denoised its working
    size, not this one (`mask_crop_cost_ms` prices it).
    """
    masked = "mask IS NOT NULL AND mask_crop IS NULL" if inpaint else "mask IS NULL"
    with storage._session() as conn:
        rows = conn.execute(
            f"""SELECT duration_ms, steps, strength FROM images
               WHERE width = ? AND height = ? AND model_id = ? AND steps > 0
                 AND duration_ms > 0 AND refine IS NOT NULL AND edit IS NULL AND strength IS NOT NULL
                 AND {masked} AND upscaler IS NULL
               ORDER BY id DESC LIMIT ?""",
            (width, height, model_id, sample),
        ).fetchall()
    durations: dict[int, list[float]] = {}
    for row in rows:
        start = start_index(row["steps"], row["strength"], model_id, inpaint=inpaint)
        if start is not None:
            durations.setdefault(row["steps"] - start, []).append(row["duration_ms"])
    return {steps: sorted(values)[len(values) // 2] for steps, values in durations.items()}


def mask_crop_cost_ms(model_id: str, sample: int = 100) -> dict[tuple[int, int], float]:
    """Median full duration per image of the inpaints and outpaints that ran
    only the masked area, by working tier and denoised step count.

    Never by the source's size: a cropped pass denoises its working size,
    which is about its tier's pixels whatever the source, so its price is
    the tier's - read from cropped rows of the same checkpoint alone, the
    tier parsed from each row's key here, a malformed one skipped. No
    history, no price.
    """
    with storage._session() as conn:
        rows = conn.execute(
            """SELECT duration_ms, steps, strength, mask_crop FROM images
               WHERE model_id = ? AND steps > 0 AND duration_ms > 0 AND refine IS NOT NULL
                 AND edit IS NULL AND strength IS NOT NULL AND mask IS NOT NULL
                 AND mask_crop IS NOT NULL AND upscaler IS NULL
               ORDER BY id DESC LIMIT ?""",
            (model_id, sample),
        ).fetchall()
    durations: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        try:
            tier = storage.MaskCropSpec.parse_key(row["mask_crop"]).tier
        except ValueError:
            continue
        start = start_index(row["steps"], row["strength"], model_id, inpaint=True)
        if start is not None:
            durations.setdefault((tier, row["steps"] - start), []).append(row["duration_ms"])
    return {key: sorted(values)[len(values) // 2] for key, values in durations.items()}


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


# How long the picker's gallery timing waits for the gallery reservation. A
# save holds it for a moment, a bundle operation or a clear for as long as it
# runs: the catalogue says nothing of the gallery then rather than stalling
# behind it for `gallery_lock.SESSION_WAIT`.
TIMING_WAIT = 0.2


def gallery_timing(quantization: str, sample: int = 50) -> dict[str, dict[str, int]]:
    """Each checkpoint's typical direct render in this gallery, for the picker.

    One grouped query over each checkpoint's latest `sample` direct renders
    under `quantization`: no refine pass, no edit, Eager execution (a NULL
    column) and a recorded duration. Of those, the size used most, the step
    count used most at that size, and the median seconds per image there
    with how many images it is of. A tie goes to the more recent setting.

    History, not a measurement: rows record neither the offload mode nor
    whether an image was its process's first, so neither is filtered out -
    the median keeps one slow first image from moving the figure. It never
    feeds a speed category (`model_benchmarks` owns those). A checkpoint
    with no matching rows is absent, and a missing database answers nothing
    rather than creating one. A reserved gallery raises `GalleryBusy` after
    `TIMING_WAIT`, never the ordinary session's wait.
    """
    if not storage.settings.db_path.is_file():
        return {}
    with gallery_lock.hold(storage.settings.db_path, wait=TIMING_WAIT):
        conn = storage._connect()
        try:
            rows = conn.execute(
                """WITH latest AS (
                       SELECT id, model_id, width, height, steps, duration_ms,
                              ROW_NUMBER() OVER (PARTITION BY model_id ORDER BY id DESC) AS recent
                       FROM images
                       WHERE quantization = ? AND refine IS NULL AND edit IS NULL
                         AND execution IS NULL AND duration_ms > 0 AND steps > 0
                   )
                   SELECT model_id, width, height, steps, COUNT(*) AS made, MAX(id) AS newest,
                          group_concat(duration_ms) AS durations
                   FROM latest WHERE recent <= ?
                   GROUP BY model_id, width, height, steps""",
                (quantization, sample),
            ).fetchall()
        finally:
            conn.close()
    by_model: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_model.setdefault(row["model_id"], []).append(row)
    answers = {}
    for model_id, groups in by_model.items():
        sizes: dict[tuple[int, int], tuple[int, int]] = {}
        for row in groups:
            made, newest = sizes.get((row["width"], row["height"]), (0, 0))
            sizes[(row["width"], row["height"])] = (made + row["made"], max(newest, row["newest"]))
        size = max(sizes, key=lambda key: sizes[key])
        best = max(
            (row for row in groups if (row["width"], row["height"]) == size),
            key=lambda row: (row["made"], row["newest"]),
        )
        durations = sorted(int(value) for value in best["durations"].split(","))
        answers[model_id] = {
            "median_ms": durations[len(durations) // 2], "width": size[0], "height": size[1],
            "steps": best["steps"], "images": len(durations),
        }
    return answers


# A size's prices (`gallery_reports.execution_estimates`): the direct step
# cost, refine passes and whole-picture inpaints by denoised steps, and
# masked-area-only inpaints by (tier, denoised steps). A three-tuple is read
# as one with no cropped history.
SizePrices = tuple[float | None, dict[int, float], dict[int, float], dict[tuple[int, int], float]]


def estimate_seconds(
    cells: list[Cell], per_step_ms: float | None, *,
    refine_costs: dict[int, float] | None = None,
    inpaint_costs: dict[int, float] | None = None,
    crop_costs: dict[tuple[int, int], float] | None = None,
    by_size: dict[tuple[int, int], SizePrices] | None = None,
) -> float | None:
    """Generation time from direct step costs or matching refine durations.

    One helper for the preview and the CLI so the arithmetic is not written
    twice. Per *step* times the cell's steps times its batch: a grid's cells
    disagree about step count, so an average image duration would misstate
    most of them. Refine cells use full per-image durations at the same number
    of denoised steps: their fixed encode/decode work must not be multiplied
    away or priced from direct renders. An inpaint cell is priced from
    inpaints' durations (`inpaint_costs`) the same way, never from passes';
    one that runs only the masked area from cropped inpaints of its working
    tier (`crop_costs`), never by its source's size.
    No estimate if any cell has no matching history. The CLI pays a load per
    invocation on top.
    """
    priced = _cell_ms(cells, per_step_ms, refine_costs, inpaint_costs, by_size, crop_costs)
    return None if priced is None else round(sum(priced) / 1000, 1)


def _cell_ms(
    cells: list[Cell], per_step_ms: float | None, refine_costs: dict[int, float] | None,
    inpaint_costs: dict[int, float] | None = None,
    by_size: dict[tuple[int, int], SizePrices] | None = None,
    crop_costs: dict[tuple[int, int], float] | None = None,
) -> list[float] | None:
    """Each cell's price in milliseconds, as `estimate_seconds` sums them;
    None if any cell has no matching history."""
    priced = []
    for cell in cells:
        request = cell.request
        prices = (
            by_size.get((request.width, request.height), (None, {}, {}, {}))
            if by_size is not None else (per_step_ms, refine_costs, inpaint_costs, crop_costs)
        )
        direct, passes, inpaints = prices[:3]
        crops = prices[3] if len(prices) > 3 else None
        # The GAN adds work before VAE encoding. Ordinary Refine history
        # cannot price it; keep the estimate unknown until measured apart.
        if request.upscaler is not None:
            return None
        # Nor can a direct render's step price an instruction edit (More
        # seeds queues one): no edit history is qualified in this release.
        if request.edit is not None:
            return None
        if request.refine is None:
            if direct is None:
                return None
            image_ms = direct * denoised_steps(request)
        elif request.mask_crop is not None:
            image_ms = (crops or {}).get((request.mask_crop.tier, denoised_steps(request)))
            if image_ms is None:
                return None
        else:
            costs = inpaints if request.inpainting else passes
            image_ms = (costs or {}).get(denoised_steps(request))
            if image_ms is None:
                return None
        priced.append(image_ms * request.num_images)
    return priced


def seeds_within_budget(
    cells: list[Cell], seeds: int, per_step_ms: float | None, budget_seconds: float, *,
    refine_costs: dict[int, float] | None = None,
    inpaint_costs: dict[int, float] | None = None,
    crop_costs: dict[tuple[int, int], float] | None = None,
    by_size: dict[tuple[int, int], SizePrices] | None = None,
) -> int | None:
    """How many of a grid's seeds a time budget completes, by the job's own
    rule: a cell starts while the deadline has not passed and runs to its end
    (`JobManager._generate`), so a seed whose last cell starts in time
    completes, though it ends past the deadline. Dividing the budget by a
    seed's share of the estimate said 4 of 6 for a grid that made 5. Seeds are
    the outer loop, each running the same cells; priced as `estimate_seconds`
    prices, from the job's start - the load and any job ahead spend a budget
    too, which no estimate counts. None where it has no price."""
    priced = _cell_ms(cells, per_step_ms, refine_costs, inpaint_costs, by_size, crop_costs)
    if priced is None or seeds < 1 or len(cells) % seeds:
        return None
    per_seed = len(cells) // seeds
    elapsed_ms = 0.0
    for index, ms in enumerate(priced):
        if elapsed_ms > budget_seconds * 1000:
            return index // per_seed
        elapsed_ms += ms
    return seeds




def execution_estimates(
    cells: Sequence[Any], model_id: str, *, disk: bool = True,
) -> tuple[dict, int | None]:
    """Matching history per concrete canvas, and the sum of its PNG samples.

    Callers pass execution units in their real order, including Then refine's
    stand-ins. Missing history for one size never borrows another's sample.
    `disk=False` is time only: no PNG is stat()ed and the sum is None.
    """
    groups: dict[tuple[int, int], list[Any]] = {}
    for cell in cells:
        groups.setdefault((cell.request.width, cell.request.height), []).append(cell)
    prices = {}
    need: int | None = 0 if cells and disk else None
    for size, group in groups.items():
        if any(cell.request.edit is not None for cell in group):
            # There is no qualified edit history/estimator in this release.
            prices[size], need = (None, {}, {}, {}), None
            continue
        inpainting = [cell for cell in group if cell.request.inpainting]
        prices[size] = (
            step_cost_ms(*size, model_id),
            refine_cost_ms(*size, model_id),
            refine_cost_ms(*size, model_id, inpaint=True)
            if any(cell.request.mask_crop is None for cell in inpainting) else {},
            mask_crop_cost_ms(model_id)
            if any(cell.request.mask_crop is not None for cell in inpainting) else {},
        )
        if need is None:
            continue
        sample = image_bytes(*size, model_id)
        if sample is None:
            need = None
        else:
            need += sample * sum(cell.request.num_images for cell in group)
    return prices, need


def job_seconds(cells: Sequence[Any], model_id: str) -> float | None:
    """A queued job's time from matching history, or None: the preview's
    figure for the same execution units, without its disk sample, so the
    queue's price reads no PNG (`jobs.JobManager.submit_cells`)."""
    prices, _ = execution_estimates(cells, model_id, disk=False)
    return estimate_seconds(list(cells), None, by_size=prices)


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
    # Only the masked area by its spec (`bbox16:32@1`): which padding the
    # kept passes cropped with. NULL on every pass that ran the whole picture.
    "maskcrop": "mask_crop",
    "edit": "CASE WHEN edit IS NOT NULL THEN json_extract(edit, '$.recipe') END",
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
    "size": "width || 'x' || height",
    "crop_anchor": "COALESCE(crop_anchor, 'center')",
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


def _raw_sizes(blob: bytes) -> set[tuple[int, int]]:
    """One retained job's cell sizes from its raw requests, unvalidated: a
    list only needs the canvases, not a `GenerateRequest` per cell. Raises
    ValueError for a row that cannot be read, as `retained_run_jobs` would."""
    try:
        cells = json.loads(gzip.decompress(blob))
    except (TypeError, ValueError, OSError, EOFError) as exc:
        raise ValueError("unreadable retained cells") from exc
    if not isinstance(cells, list):
        raise ValueError("cells must be a list")
    sizes = set()
    for cell in cells:
        request = cell.get("request") if isinstance(cell, dict) else None
        size = (request.get("width"), request.get("height")) if isinstance(request, dict) else (None, None)
        if not all(type(side) is int and side > 0 for side in size):
            raise ValueError("invalid retained cell")
        sizes.add(size)
    return sizes


def _listed_run_sizes(conn: sqlite3.Connection, run_ids: list[str]) -> dict[str, list[tuple[int, int]]]:
    """Every listed run's sizes in the caller's one session (GAL#30): its
    retained requests define a run, its unfinished sizes included. The list
    is asked for after every finished job, and a session, gunzip and
    validation per run took 1.5 s for 200. A run with an unreadable retained
    row, or none, takes its images' sizes (binned rows too)."""
    if not run_ids:
        return {}
    marks = ",".join("?" * len(run_ids))
    retained: dict[str, set[tuple[int, int]]] = {}
    unreadable: set[str] = set()
    for row in conn.execute(f"SELECT run_id, cells FROM run_jobs WHERE run_id IN ({marks})", run_ids):
        if row["run_id"] in unreadable:
            continue
        try:
            retained.setdefault(row["run_id"], set()).update(_raw_sizes(row["cells"]))
        except ValueError:
            unreadable.add(row["run_id"])
            retained.pop(row["run_id"], None)
    sizes = {run_id: sorted(found) for run_id, found in retained.items() if found}
    rest = [run_id for run_id in run_ids if run_id not in sizes]
    if rest:
        images: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for row in conn.execute(
            f"SELECT DISTINCT run_id, width, height FROM images WHERE run_id IN ({','.join('?' * len(rest))})", rest,
        ):
            images[row["run_id"]].append((row["width"], row["height"]))
        sizes.update({run_id: sorted(images[run_id]) for run_id in rest})
    return sizes


def runs(limit: int = 20, model_id: str | None = None) -> list[RunSummary]:
    """The gallery grouped by run, the most recently written first.

    What `GET /api/runs` serves and the Runs picker lists: the night's output
    as one line per grid rather than 480 cards to scroll past. Rows written
    before runs had an id are not a run and are left out. Checkpoint and
    quantizer are constant within a run. Its distinct canvases come from
    retained requests, else its images; scalar dimensions remain representative.
    The oldest image supplies its prompt and its seed, which is the
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
        listed_sizes = _listed_run_sizes(conn, [group["run_id"] for group in groups])
    summaries = []
    for group in groups:
        first = firsts[group["first_id"]]
        sizes = listed_sizes[group["run_id"]]
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
                sizes=sizes, size_count=len(sizes),
                started_at=group["started_at"],
                finished_at=group["finished_at"],
                axes={name: group[f"axis_{name}"] for name in _RUN_AXES if group[f"axis_{name}"] > 1},
                refine=group["refine"] if group["n_sources"] == 1 else None,
                strengths=(
                    [group["strength_min"], group["strength_max"]]
                    if group["n_sources"] == 1 and group["refine"] is not None else None
                ),
                cover=group["first_id"],
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
            sizes=sorted({(cell.request.width, cell.request.height) for cell in cells}),
            size_count=len({(cell.request.width, cell.request.height) for cell in cells}),
            started_at=stamp(first["created_at"]),
            finished_at=stamp((last.get("outcome") or {}).get("finished_at") or last["created_at"]),
        )
        # Preserve the existing live runs' id order, while inserting an empty
        # run near the time it began. A just-submitted zero-image run leads.
        position = next((index for index, summary in enumerate(summaries)
                         if summary.started_at <= recovered.started_at), len(summaries))
        summaries.insert(position, recovered)
    return summaries[:limit]
