"""Which settings produced the favourites - as a rate, not a count.

    uv run python -m backend.stats                     # the newest run
    uv run python -m backend.stats --run ab12cd        # one grid, however many jobs
    uv run python -m backend.stats --job ab12cd        # one submission
    uv run python -m backend.stats --where "model:turbo steps:>=8"
    uv run python -m backend.stats --all --by steps,cfg --min 10

The favourite flag was queryable one value at a time and aggregated nowhere,
so "which CFG did I keep" was a scroll through the gallery. The naive count
would also mislead - it rewards whichever setting was used most - so this is
the *starred rate* per setting, over the rows a scope selects.

The default scope is the newest run: `expand()` crossed its axes against the
same prompts and seeds, so its rate is a fair comparison. All-time
(`--all`) is confounded - over months it rewards whatever settings were in
use while the prompts happened to be good - and is printed as indicative.
The column says *starred*, not kept: nothing records a rejection, so an
untouched night reads 0% rather than unknown.

Running the file directly (`python backend/stats.py`) works too - see the
sys.path bootstrap below.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from backend import storage
    from backend.config import settings
except ModuleNotFoundError as exc:  # pragma: no cover - startup guard
    sys.exit(
        f"Missing dependency: {exc.name}\n\n"
        f"Running under: {sys.executable}\n"
        "That interpreter is not the project environment. Use:\n\n"
        "    uv run python -m backend.stats\n"
    )


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    text = str(value)
    return text if len(text) <= 40 else text[:37] + "..."


def _table(rows: list[dict], columns: list[str]) -> str:
    headers = [*columns, "made", "starred", "rate"]
    body = [
        [*(_cell(row[c]) for c in columns), str(row["made"]), str(row["starred"]),
         f"{row['rate'] * 100:.0f}%"]
        for row in rows
    ]
    widths = [max(len(h), *(len(r[i]) for r in body)) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    out = [line, "  ".join("-" * w for w in widths)]
    out.extend("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in body)
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.stats",
        description="The starred rate per setting: which steps, CFG and mu produced the favourites.",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--run", metavar="ID", help="one grid, however many jobs it took")
    scope.add_argument("--job", metavar="ID", help="one submission")
    scope.add_argument(
        "--where", metavar="TERMS", help="key:value terms in the gallery's own grammar"
    )
    scope.add_argument("--all", action="store_true", help="the whole gallery (indicative)")
    parser.add_argument(
        "--by",
        default="steps,cfg,mu",
        help="comma-separated columns to group by: " + ", ".join(storage._GROUPS) + " (size is always included)",
    )
    parser.add_argument(
        "--min", type=int, default=5, metavar="N",
        help="drop settings with fewer images than this, so a lucky 1-of-1 does not lead (default 5)",
    )
    args = parser.parse_args(argv)

    if not settings.db_path.exists():
        print("No gallery yet.", file=sys.stderr)
        return 1

    if args.run:
        search, label = f"run:{args.run}", f"run {args.run}"
    elif args.job:
        search, label = f"job:{args.job}", f"job {args.job}"
    elif args.where:
        search, label = args.where, args.where
    elif args.all:
        search, label = None, "the whole gallery (indicative: settings and prompts changed together)"
    else:
        newest = storage.runs(limit=1)
        if not newest:
            print("No runs in the gallery; --all reads everything.", file=sys.stderr)
            return 1
        run = newest[0]
        search = f"run:{run.run_id}"
        label = f"the newest run, {run.run_id} ({run.images} images, {run.favorites} starred)"

    by = [part.strip() for part in args.by.split(",") if part.strip()]
    try:
        rows = storage.keep_rates(search, by=by, min_made=max(1, args.min))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"scope    : {label}")
    print(f"grouped  : {', '.join(dict.fromkeys([*by, 'size']))}; at least {max(1, args.min)} image(s) each")
    print()
    if not rows:
        print("Nothing to compare: no setting has that many images in this scope.")
        return 0
    print(_table(rows, list(dict.fromkeys([*by, "size"]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
