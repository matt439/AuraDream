"""The search box's grammar: `key:value` terms as SQL predicates, the rest as text.

Pure functions over strings, shared by the gallery's `list_images`, the
resume prune, `clear --where` and `stats` through `storage`. Nothing here
touches the database; `storage` owns the queries these clauses go into.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# `schemas_generate.effective_true_cfg` in SQL, for the filter and the
# grouping that run in the database: the scale the negative prompt ran at,
# which on a row from before the column was `guidance_scale` whenever there
# was one.
TRUE_CFG_SQL = (
    "COALESCE(true_cfg, CASE WHEN negative_prompt IS NOT NULL AND negative_prompt <> '' "
    "THEN guidance_scale END)"
)

# `key:value` in the search box. The column, and what a bare value means.
FILTERS: dict[str, tuple[str, Any]] = {
    # One image by the number the page names it by ("#12"), or a span of them:
    # "refine of #4" and the Refine dialog's title name ids the grid never
    # showed, and a pass's source could not be found from them.
    "id": ("id", int),
    "seed": ("seed", int),
    "steps": ("steps", int),
    "cfg": ("guidance_scale", float),
    # The scale the negative prompt ran at, through the same reading of older
    # rows the records get; `tcfg:none` is the images that ran without one.
    "tcfg": (TRUE_CFG_SQL, float),
    "mu": ("mu", float),
    "model": ("model_id", str),
    "quant": ("quantization", str),
    # The adapter by name - `lora:none` is everything rendered without one.
    "lora": ("lora", str),
    # A refine pass by the gallery id it started from - `refine:none` is the
    # direct renders - and by how far it was re-denoised.
    "refine": ("refine", int),
    "strength": ("strength", float),
    # An inpaint by its mask's hash - a substring, so the 16 digits the A1111
    # line prints find it - and `mask:none` everything that is no inpaint.
    "mask": ("mask", str),
    # Handled by inpaint_clause: `inpaint:yes` / `inpaint:no`, the passes
    # that had a mask and the ones that did not, without a hash to type.
    "inpaint": ("mask", None),
    # Handled by outpaint_clause: `outpaint:yes` / `outpaint:no`. An outpaint
    # has a mask too, so `inpaint:yes outpaint:no` is the painted ones.
    "outpaint": ("outpaint", None),
    "edit": ("edit", None),
    "job": ("job_id", str),
    "run": ("run_id", str),
    # Every image with these pixels - a substring, so a prefix of one is enough.
    "hash": ("content_hash", str),
    # Handled by date_clause: local calendar days, not a column with a cast.
    "date": ("created_at", None),
    # Handled by size_clause: `size:1024x1024`, the width and the height as
    # the facts list says them - the lightbox's search buttons write it.
    "size": ("width", None),
}
COMPARISONS = ("<=", ">=", "<", ">")
NULL_TOKENS = {"none", "null", "off"}
# Text keys whose `none` is NULL rather than a substring: `quant:none` is a
# stored mode (an unquantised load), so only `lora` reads it as "without".
NULLABLE_TEXT = {"lora", "mask"}
# SQLite's INTEGER: a number past it raises OverflowError at the query, not
# ValueError, and the routes answered 500.
SQLITE_INT = (-(2**63), 2**63 - 1)


def date_clause(raw: str) -> tuple[str, list]:
    """One local day or an inclusive `date:YYYY-MM-DD..YYYY-MM-DD` range.

    Single days also accept `today` and `yesterday`; ranges use two ordered
    absolute calendar dates. Every resulting SQL interval stays half-open.

    A *local* day: `created_at` is UTC, and an overnight run crosses midnight
    in every timezone, so the day the user means is the one on their clock,
    converted here. Both bounds are local calendar midnights: a clock-change
    day can be shorter or longer than 24 hours. A half-open `>= start AND < end`
    rather than a LIKE prefix
    - the prefix cannot use the index and would need the UTC date, which is
    the wrong one for anything made after about 4 pm in the Americas or before
    breakfast east of Greenwich. The bounds are formatted the way the rows are
    (`isoformat()` of an aware datetime), so the string comparison is the
    time comparison.
    """
    from datetime import date, timedelta

    if ".." in raw:
        first, _separator, last = raw.partition("..")
        try:
            first_day = date.fromisoformat(first)
            last_day = date.fromisoformat(last)
            if first_day.isoformat() != first or last_day.isoformat() != last or first_day > last_day:
                raise ValueError
        except ValueError:
            raise ValueError(f"date: {raw!r} is not an ordered YYYY-MM-DD..YYYY-MM-DD range") from None
    else:
        lowered = raw.lower()
        if lowered == "today":
            day = date.today()
        elif lowered == "yesterday":
            day = date.today() - timedelta(days=1)
        else:
            try:
                day = date.fromisoformat(raw)
            except ValueError:
                raise ValueError(f"date: {raw!r} is not YYYY-MM-DD, today or yesterday") from None
        first_day = last_day = day
    try:
        midnight = datetime.combine(first_day, datetime.min.time())
        next_midnight = datetime.combine(last_day + timedelta(days=1), datetime.min.time())
        start = midnight.astimezone(timezone.utc)
        end = next_midnight.astimezone(timezone.utc)
    except OverflowError:
        raise ValueError(f"date: {raw!r} is out of range") from None
    return "created_at >= ? AND created_at < ?", [start.isoformat(), end.isoformat()]


def size_clause(raw: str) -> tuple[str, list]:
    """Exact stored dimensions or portrait/landscape/square across resolutions."""
    orientation = {
        "portrait": "width < height",
        "landscape": "width > height",
        "square": "width = height",
    }.get(raw.lower())
    if orientation is not None:
        return orientation, []

    width, sep, height = raw.lower().replace("×", "x").partition("x")
    if not sep or not width.isdigit() or not height.isdigit():
        raise ValueError(f"size: {raw!r} is not WIDTHxHEIGHT")
    return "(width = ? AND height = ?)", [_in_range("size", raw, int(width)), _in_range("size", raw, int(height))]


def _yes_no_clause(key: str, column: str, raw: str) -> tuple[str, list]:
    lowered = raw.lower()
    if lowered in {"yes", "true", "1"}:
        return f"{column} IS NOT NULL", []
    if lowered in {"no", "false", "0"} or lowered in NULL_TOKENS:
        return f"{column} IS NULL", []
    raise ValueError(f"{key}: {raw!r} is not yes or no")


def outpaint_clause(raw: str) -> tuple[str, list]:
    """`outpaint:yes` and `outpaint:no`: an outpaint's spec on the row or
    none. Every outpaint is an inpaint too (`inpaint:yes`), mechanically."""
    return _yes_no_clause("outpaint", "outpaint", raw)


def inpaint_clause(raw: str) -> tuple[str, list]:
    """`inpaint:yes` (or `true`, `1`) and `inpaint:no` (`false`, `0`, `none`):
    a mask on the row or none. The mask is only ever set beside a source, so
    `inpaint:no refine:>0` is the plain refine passes."""
    lowered = raw.lower()
    if lowered in {"yes", "true", "1"}:
        return "mask IS NOT NULL", []
    if lowered in {"no", "false", "0"} or lowered in NULL_TOKENS:
        return "mask IS NULL", []
    raise ValueError(f"inpaint: {raw!r} is not yes or no")


def filter_clause(key: str, raw: str) -> tuple[str, list]:
    """One `key:value` term as SQL. Raises ValueError on a value that is not one.

    Exact (`cfg:0.5`), comparison (`steps:>=8`), range (`cfg:0-1`) or `none`,
    which is the only way to ask for the images that ran without a mu at all -
    `mu:0` is a different image and SQL will not match NULL with either. A
    text key matches a substring, or with a leading `=` the whole value;
    `lora:none` is the images rendered without an adapter (`lora:=none` is
    one named "none").
    """
    if key == "date":
        return date_clause(raw)
    if key == "size":
        return size_clause(raw)
    if key == "inpaint":
        return inpaint_clause(raw)
    if key == "outpaint":
        return outpaint_clause(raw)
    if key == "edit":
        return _yes_no_clause("edit", "edit", raw)
    column, cast = FILTERS[key]
    if cast is str:
        # `model:=<id>` is the whole id, exactly - the gallery's own checkpoint
        # scope in the box's words, where `model:circlestone-labs/Anima` is a
        # substring of Anima@aesthetic-v1.1 too. The Models picker's See all
        # writes it, so the grid it opens is the strip it came from.
        if raw.startswith("="):
            if raw == "=":
                raise ValueError(f"{key}: '=' names no value")
            return f"{column} = ?", [raw[1:]]
        if key in NULLABLE_TEXT and raw.lower() in NULL_TOKENS:
            return f"{column} IS NULL", []
        return rf"{column} LIKE ? ESCAPE '\'", [f"%{like_escape(raw)}%"]
    if raw.lower() in NULL_TOKENS:
        return f"{column} IS NULL", []
    for operator in COMPARISONS:
        if raw.startswith(operator):
            return f"{column} {operator} ?", [_cast(cast, key, raw[len(operator):])]
    # A range borrows the grid's own spelling, so `cfg:0-1` reads the same in
    # the search box as it does in the axis that produced those images.
    low, dash, high = raw.partition("-")
    if dash and low and high:
        return f"{column} BETWEEN ? AND ?", [_cast(cast, key, low), _cast(cast, key, high)]
    return f"{column} = ?", [_cast(cast, key, raw)]


def _cast(cast, key: str, raw: str):
    try:
        value = cast(raw)
    except ValueError:
        raise ValueError(f"{key}: {raw!r} is not a number") from None
    return _in_range(key, raw, value) if cast is int else value


def _in_range(key: str, raw: str, value: int) -> int:
    if not SQLITE_INT[0] <= value <= SQLITE_INT[1]:
        raise ValueError(f"{key}: {raw!r} is out of range")
    return value


def split_search(search: str) -> tuple[list[str], list, str]:
    """Split the box into SQL predicates and the words left over for FTS.

    Anything that is not a known `key:value` stays free text, so a prompt that
    happens to contain a colon still searches for itself.
    """
    clauses: list[str] = []
    params: list = []
    words: list[str] = []
    for token in search.split():
        key, sep, raw = token.partition(":")
        if sep and raw and key.lower() in FILTERS:
            clause, values = filter_clause(key.lower(), raw)
            clauses.append(clause)
            params.extend(values)
        else:
            words.append(token)
    return clauses, params, " ".join(words)


def like_escape(term: str) -> str:
    r"""Neutralise LIKE wildcards, for use with ESCAPE '\'."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def strict_scope(search: str | None) -> tuple[list[str], list]:
    """The search box's `key:value` terms as predicates, and nothing else.

    For the paths where a fuzzy match is a hazard - a delete, a statistic.
    `list_images`' LIKE fallback changes the predicate's semantics on any
    token with an odd quote, and a destructive filter resolved through a fuzzy
    index should not be confirmed blind: free text is refused, not matched.
    """
    if not search or not search.strip():
        return [], []
    filters, params, text = split_search(search)
    if text:
        raise ValueError(f"key:value terms only; {text!r} is free text")
    return filters, params
