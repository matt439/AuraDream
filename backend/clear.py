"""Delete generated images and their history.

    uv run python -m backend.clear                  # preview, then ask
    uv run python -m backend.clear --yes            # no prompt
    uv run python -m backend.clear --keep-favorites # spare starred images
    uv run python -m backend.clear --dry-run        # report only
    uv run python -m backend.clear --where "job:ab12cd"   # one run, not the gallery

Running the file directly (`python backend/clear.py`, or an editor's Run button)
works too - see the sys.path bootstrap below.

Safe to run while the server is up: rows are deleted rather than the database
file, so no open handle is invalidated.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Executed as a plain script there is no parent package, so `from . import ...`
# raises ImportError. Put the project root on sys.path and import absolutely,
# which works under both invocations.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from backend import storage
    from backend.config import settings
except ModuleNotFoundError as exc:  # pragma: no cover - startup guard
    # Almost always `py clear.py` or a bare `python clear.py`, which picks the
    # system interpreter rather than the project's .venv.
    sys.exit(
        f"Missing dependency: {exc.name}\n\n"
        f"Running under: {sys.executable}\n"
        "That interpreter is not the project environment. Use:\n\n"
        "    uv run python -m backend.clear\n"
    )


_human = storage.human_bytes


def _confirm(yes: bool) -> bool:
    if yes:
        return True
    # isatty() is not a reliable signal here - it can report a terminal that
    # has nothing to read - so let the read itself decide.
    try:
        answer = input("\nThis cannot be undone. Type 'clear' to continue: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNo confirmation given; nothing deleted. Pass --yes to skip this prompt.",
              file=sys.stderr)
        return False
    if answer != "clear":
        print("Aborted.")
        return False
    return True


def _clear_where(args: argparse.Namespace) -> int:
    """The filtered path: the rows a `key:value` scope selects, and their files.

    A 200-cell grid started with a typo in the prompt could only be removed by
    200 confirmations or by destroying the gallery - and leaving it in poisons
    `--resume` and the time estimate. The preview names what a substring term
    matched (`job:` and `model:` are LIKEs), free text is refused rather than
    resolved through the fuzzy fallback, and the orphan sweep stays off this
    path: a filtered clear that also swept every untracked PNG would be a new
    all-or-nothing hazard inside the feature meant to remove one.
    """
    if not settings.db_path.exists():
        print("No gallery yet.", file=sys.stderr)
        return 1
    try:
        found = storage.matching_preview(args.where)
    except ValueError as exc:
        print(f"--where: {exc}", file=sys.stderr)
        return 2
    print(f"database : {settings.db_path}")
    print(f"images   : {settings.outputs_dir}")
    print(f"filter   : {args.where}")
    print()
    print(f"  {found['records']:>6} records match ({found['favorites']} favorite)")
    if found["jobs"]:
        print(f"         jobs: {', '.join(found['jobs'])}")
    if len(found["models"]) > 1 or (found["models"] and "model:" in args.where):
        print(f"         checkpoints: {', '.join(found['models'])}")
    for prompt in found["prompts"][:3]:
        print(f"         - {prompt[:70]}")
    if len(found["prompts"]) > 3:
        print("         - ...")
    print()
    if found["records"] == 0:
        print("Nothing matches.")
        return 0
    kept = found["favorites"] if args.keep_favorites else 0
    print(
        f"Would delete {found['records'] - kept} record(s) and their files"
        + (f", keeping {kept} favorite(s)." if kept else ".")
    )
    if args.dry_run:
        print("\nDry run - nothing changed.")
        return 0
    if not _confirm(args.yes):
        return 1
    try:
        result = storage.delete_matching(args.where, keep_favorites=args.keep_favorites)
    except Exception as exc:  # noqa: BLE001 - report, with the post-deletion code
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(f"\nDeleted {result['records']} record(s), {result['files']} file(s).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.clear",
        description="Delete generated images and their gallery history.",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="skip the confirmation prompt")
    parser.add_argument(
        "--keep-favorites", action="store_true", help="keep images starred in the gallery"
    )
    parser.add_argument(
        "-n", "--dry-run", action="store_true", help="report what would go, change nothing"
    )
    parser.add_argument(
        "--where",
        metavar="TERMS",
        help="delete only the rows these key:value terms select - `job:ab12cd`, "
        "`run:ab12cd seed:42`, `date:yesterday` - and their files; no orphan sweep, "
        "and free text is refused",
    )
    args = parser.parse_args(argv)

    if args.where:
        return _clear_where(args)

    # A dry run reports; it does not create the database it is reporting on.
    if not args.dry_run:
        storage.init_db()
    before = storage.summarise()

    doomed = before["records"] - (before["favorites"] if args.keep_favorites else 0)
    print(f"database : {settings.db_path}")
    print(f"images   : {settings.outputs_dir}")
    print()
    print(f"  {before['records']:>6} records ({before['favorites']} favorite)")
    print(f"  {before['files_on_disk']:>6} PNG files, {_human(before['bytes_on_disk'])}")
    if before["orphan_files"]:
        print(f"  {before['orphan_files']:>6} of those have no database row")
    if before["partial_files"]:
        print(
            f"  {before['partial_files']:>6} stale .part file(s) from interrupted writes"
        )
    print()

    if before["records"] == 0 and before["files_on_disk"] == 0 and not before["partial_files"]:
        print("Nothing to clear.")
        return 0

    # Name the orphans here too. They are deleted either way - they have no row
    # left to carry a favorite flag - and the line above the prompt is the last
    # thing anyone reads before typing 'clear'.
    kept = before["favorites"] if args.keep_favorites else 0
    plan = f"Would delete {doomed} record(s) and their files"
    if before["orphan_files"]:
        plan += f", plus {before['orphan_files']} untracked PNG(s)"
    if before["partial_files"]:
        plan += f", plus {before['partial_files']} stale .part file(s)"
    print(plan + (f", keeping {kept} favorite(s)." if kept else "."))

    if args.dry_run:
        print("\nDry run - nothing changed.")
        return 0

    if not _confirm(args.yes):
        return 1

    try:
        result = storage.clear_history(keep_favorites=args.keep_favorites)
    except Exception as exc:  # noqa: BLE001 - report, and say it is not the abort code
        # Its own code: 1 is "aborted, nothing deleted", and this is the
        # opposite - the destructive step ran and then something went wrong.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    # The report first: what was deleted is the thing that must be printed
    # whatever happens next. VACUUM is an optimisation, and it raises "database
    # is locked" against a concurrent writer after the 15 s timeout.
    print(
        f"\nDeleted {result['records']} record(s), {result['files']} file(s)"
        + (f", {result['orphans']} orphan(s)" if result["orphans"] else "")
        + (f", {result['partials']} stale .part file(s)" if result["partials"] else "")
        + "."
    )
    try:
        storage.vacuum()
    except Exception as exc:  # noqa: BLE001 - best effort
        print(f"note: could not compact the database ({exc}); it will shrink later.", file=sys.stderr)
    after = storage.summarise()
    print(f"Remaining: {after['records']} record(s), {after['files_on_disk']} file(s), "
          f"{_human(after['bytes_on_disk'])}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
