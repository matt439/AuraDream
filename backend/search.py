"""The search box's grammar: `key:value` terms as SQL predicates, the rest as text.

Pure functions over strings, shared by the gallery's `list_images`, the
resume prune, `clear --where` and `stats` through `storage`. Nothing here
touches the database; `storage` owns the queries these clauses go into.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# `schemas.effective_true_cfg` in SQL, for the filter and the grouping that
# run in the database: the scale the negative prompt ran at, which on a row
# from before the column was `guidance_scale` whenever there was one.
TRUE_CFG_SQL = (
    "COALESCE(true_cfg, CASE WHEN negative_prompt IS NOT NULL AND negative_prompt <> '' "
    "THEN guidance_scale END)"
)

# `key:value` in the search box. The column, and what a bare value means.
FILTERS: dict[str, tuple[str, Any]] = {
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
    "job": ("job_id", str),
    "run": ("run_id", str),
    # Every image with these pixels - a substring, so a prefix of one is enough.
    "hash": ("content_hash", str),
    # Handled by date_clause: a day, not a column with a cast.
    "date": ("created_at", None),
}
COMPARISONS = ("<=", ">=", "<", ">")
NULL_TOKENS = {"none", "null", "off"}


def date_clause(raw: str) -> tuple[str, list]:
    """`date:2026-09-07`, `date:today`, `date:yesterday` as a half-open range.

    A *local* day: `created_at` is UTC, and an overnight run crosses midnight
    in every timezone, so the day the user means is the one on their clock,
    converted here. A half-open `>= start AND < end` rather than a LIKE prefix
    - the prefix cannot use the index and would need the UTC date, which is
    the wrong one for anything made after about 4 pm in the Americas or before
    breakfast east of Greenwich. The bounds are formatted the way the rows are
    (`isoformat()` of an aware datetime), so the string comparison is the
    time comparison.
    """
    from datetime import date, timedelta

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
    start = datetime.combine(day, datetime.min.time()).astimezone(timezone.utc)
    end = start + timedelta(days=1)
    return "created_at >= ? AND created_at < ?", [start.isoformat(), end.isoformat()]


def filter_clause(key: str, raw: str) -> tuple[str, list]:
    """One `key:value` term as SQL. Raises ValueError on a value that is not one.

    Exact (`cfg:0.5`), comparison (`steps:>=8`), range (`cfg:0-1`) or `none`,
    which is the only way to ask for the images that ran without a mu at all -
    `mu:0` is a different image and SQL will not match NULL with either.
    """
    if key == "date":
        return date_clause(raw)
    column, cast = FILTERS[key]
    if cast is str:
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
        return cast(raw)
    except ValueError:
        raise ValueError(f"{key}: {raw!r} is not a number") from None


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
