"""Where a smoke run's time went (`DEV#25`), read from its summary.json.

    uv run python tools/smoke_timings.py <run-root-or-summary.json> [--top N]

Given a summary.json, or a directory with exactly one summary.json below it
(a run root, or the `--artifacts` root above one), this prints the slowest N
smokes (port, model, result, seconds), each port-table row's total, and the
sum of smoke times beside the run's wall clock. Only the first pass's
`results` are counted; a rerun's receipts (`--rerun-failed-alone`) get one
line of their own.

Read-only and standard library: no server, no browser, nothing written.
A row is the `AGENTS.md` row that lists the smoke (two rows can share a
port); a smoke no longer in the table is its own row, by the receipt's port
and model. Exit 2 refuses a missing, malformed or ambiguous summary.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP = 25


class Refused(Exception):
    """A summary this tool will not read; exit 2 with the reason."""


def _runner():
    """`run_smoke.py`'s table reader, loaded by path so the tool runs as a script."""
    spec = importlib.util.spec_from_file_location("run_smoke", ROOT / "tools" / "run_smoke.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered first: a dataclass reads its own module from sys.modules.
    sys.modules.setdefault("run_smoke", module)
    spec.loader.exec_module(module)
    return module


def find_summary(path: Path) -> Path:
    """`path` itself, or the one summary.json below the directory `path`."""
    if path.is_file():
        return path
    if not path.is_dir():
        raise Refused(f"{path} does not exist")
    found = sorted(path.rglob("summary.json"))
    if not found:
        raise Refused(f"no summary.json below {path}")
    if len(found) > 1:
        names = ", ".join(str(item.relative_to(path)) for item in found)
        raise Refused(f"more than one summary.json below {path}: {names}; name one")
    return found[0]


def _receipts(summary: dict, key: str, path: Path) -> list[dict]:
    receipts = summary.get(key, [])
    if not isinstance(receipts, list):
        raise Refused(f"{path}: `{key}` is not a list")
    for receipt in receipts:
        if not (isinstance(receipt, dict) and isinstance(receipt.get("smoke"), str)
                and isinstance(receipt.get("duration_seconds"), (int, float))
                and not isinstance(receipt.get("duration_seconds"), bool)):
            raise Refused(f"{path}: a `{key}` receipt has no smoke name and duration_seconds")
    return receipts


def load(path: Path) -> dict:
    """The summary, its first-pass and rerun receipts checked for what is read."""
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise Refused(f"{path} is not a readable summary.json ({exc})") from exc
    if not isinstance(summary, dict) or "results" not in summary:
        raise Refused(f"{path} has no results list")
    wall = summary.get("duration_seconds")
    if not isinstance(wall, (int, float)) or isinstance(wall, bool):
        raise Refused(f"{path} has no duration_seconds")
    return {
        "wall": float(wall), "jobs": summary.get("jobs"), "offset": summary.get("offset"),
        "results": _receipts(summary, "results", path),
        "rerun": _receipts(summary, "rerun", path),
    }


def row_totals(results: list[dict], rows: list) -> list[dict]:
    """Each row's smoke count and seconds, slowest first; a smoke the table
    no longer lists is a row of its own by the receipt's port and model."""
    owner = {smoke: index for index, row in enumerate(rows) for smoke in row.smokes}
    totals: dict[object, dict] = {}
    for receipt in results:
        index = owner.get(receipt["smoke"])
        if index is None:
            key: object = ("receipt", receipt.get("port"), receipt.get("model"), receipt["smoke"])
            label = {"port": receipt.get("port"), "model": receipt.get("model"), "first": receipt["smoke"],
                     "listed": False}
        else:
            key = index
            row = rows[index]
            label = {"port": row.port, "model": row.model, "first": row.smokes[0], "listed": True}
        entry = totals.setdefault(key, {**label, "smokes": 0, "seconds": 0.0})
        entry["smokes"] += 1
        entry["seconds"] += float(receipt["duration_seconds"])
    return sorted(totals.values(), key=lambda entry: (-entry["seconds"], str(entry["port"]), entry["first"]))


def slowest(results: list[dict], top: int) -> list[dict]:
    """The `top` slowest receipts, ties by smoke name."""
    return sorted(results, key=lambda receipt: (-float(receipt["duration_seconds"]), receipt["smoke"]))[:top]


def report(path: Path, summary: dict, rows: list, top: int) -> str:
    results = summary["results"]
    smoke_time = sum(float(receipt["duration_seconds"]) for receipt in results)
    settings = ", ".join(f"{name} {summary[name]}" for name in ("jobs", "offset") if summary[name] is not None)
    lines = [
        f"summary: {path}",
        f"wall clock {summary['wall']:.1f} s; smoke time {smoke_time:.1f} s over {len(results)} smokes"
        + (f" ({settings})" if settings else ""),
        "",
        f"slowest {min(top, len(results))} of {len(results)} smokes:",
        f"{'seconds':>9}  {'result':<7} {'port':>5}  {'model':<40} smoke",
    ]
    for receipt in slowest(results, top):
        lines.append(
            f"{float(receipt['duration_seconds']):9.1f}  {str(receipt.get('result', '?')):<7} "
            f"{str(receipt.get('port', '?')):>5}  {str(receipt.get('model', '?')):<40} {receipt['smoke']}")
    lines += ["", "rows by total:", f"{'seconds':>9}  {'smokes':>6} {'port':>5}  {'model':<40} first smoke"]
    for entry in row_totals(results, rows):
        lines.append(
            f"{entry['seconds']:9.1f}  {entry['smokes']:6d} {str(entry['port']):>5}  {str(entry['model']):<40} "
            f"{entry['first']}" + ("" if entry["listed"] else " (not in the port table)"))
    if summary["rerun"]:
        rerun_time = sum(float(receipt["duration_seconds"]) for receipt in summary["rerun"])
        lines += ["", f"rerun alone: {len(summary['rerun'])} smokes, {rerun_time:.1f} s (not counted above)"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description="Where a smoke run's time went, from its summary.json.")
    parser.add_argument("path", type=Path, help="a summary.json, or a directory with exactly one below it")
    parser.add_argument("--top", type=int, default=TOP, help=f"how many of the slowest smokes to list (default {TOP})")
    args = parser.parse_args(argv)
    if args.top < 1:
        print("--top must be at least 1")
        return 2
    try:
        path = find_summary(args.path)
        summary = load(path)
    except Refused as exc:
        print(exc)
        return 2
    rows = _runner().parse_table((ROOT / "AGENTS.md").read_text(encoding="utf-8"))
    print(report(path, summary, rows, args.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
