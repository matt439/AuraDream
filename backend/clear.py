"""Delete generated images and their history.

    uv run python -m backend.clear                  # preview, then ask
    uv run python -m backend.clear --yes            # no prompt
    uv run python -m backend.clear --keep-favorites # spare starred images
    uv run python -m backend.clear --dry-run        # report only
    uv run python -m backend.clear --where "job:ab12cd"   # one run, not the gallery
    uv run python -m backend.clear --bin            # empty the recycle bin only
    uv run python -m backend.clear --bin --older-than 30  # only what was binned 30+ days ago

Running the file directly (`python backend/clear.py`, or an editor's Run button)
works too - see the sys.path bootstrap below.

Safe to run while the server is up: rows are deleted rather than the database
file, so no open handle is invalidated. Every path deletes for good, the
recycle bin's images included: the whole clear and `--where` reach them as
any other row (a binned star is not kept), and `--bin` deletes them alone.
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
    from backend import gallery_delete, gallery_reports, storage
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


_human = gallery_reports.human_bytes


def _keeping(favorites: int, sources: int) -> str:
    """The end of the "Would delete" line: what --keep-favorites spares.

    The starred images, and the images one of them was made from - a refine
    pass's source, a render's reference images - which storage keeps with them.
    """
    if not favorites and not sources:
        return "."
    spared = f", keeping {favorites} starred"
    if sources:
        spared += f" and {sources} image(s) a starred one was made from"
    return spared + "."


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
    # `--where ""` is the unfiltered path in disguise, and the browser's Delete
    # unstarred is the one caller storage lets ask about a blank scope.
    if not args.where.strip():
        print("--where: a filter is required; the whole gallery is `clear` without --where",
              file=sys.stderr)
        return 2
    try:
        found = gallery_delete.matching_preview(args.where)
    except ValueError as exc:
        print(f"--where: {exc}", file=sys.stderr)
        return 2
    print(f"database : {settings.db_path}")
    print(f"images   : {settings.outputs_dir}")
    print(f"filter   : {args.where}")
    print()
    print(
        f"  {found['records']:>6} records match ({found['favorites']} favorite)"
        + (f" ({found['binned']} in the recycle bin)" if found["binned"] else "")
    )
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
    # --keep-favorites keeps what a starred image was made from as well - a
    # pass's source, a render's references - and the count says so, since
    # `delete_matching` takes exactly the rest.
    kept = found["favorites"] if args.keep_favorites else 0
    sources = found["sources_kept"] if args.keep_favorites else 0
    doomed = found["records"] - kept - sources
    print(f"Would delete {doomed} record(s) and their files" + _keeping(kept, sources))
    if args.dry_run:
        print("\nDry run - nothing changed.")
        return 0
    if not _confirm(args.yes):
        return 1
    # Pinned to the count confirmed, as the browser's Delete unstarred and
    # `--bin` are: a matching image that landed while the prompt waited - the
    # server or a CLI grid still writing the run being cleared - refuses the
    # delete rather than going with the rest.
    try:
        result = gallery_delete.delete_matching(
            args.where, keep_favorites=args.keep_favorites, expected=doomed
        )
    except gallery_delete.SelectionChanged as exc:
        print(
            f"The gallery changed: {exc.found} record(s) match now, not the {exc.expected} "
            "shown. Nothing was deleted; run it again.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001 - report, with the post-deletion code
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(f"\nDeleted {result['records']} record(s), {result['files']} file(s)" + _masks(result) + ".")
    return 0


def _masks(result: dict) -> str:
    """The end of a "Deleted" line: the inpaint masks no image names any more."""
    return f", {result['masks']} unused inpaint mask(s)" if result.get("masks") else ""


def _clear_bin(args: argparse.Namespace) -> int:
    """Empty the recycle bin: its rows and their files, nothing else.

    The browser's Empty recycle bin, from the terminal, through the same
    storage call: the images previewed - their count and their ids' pin - are
    what the delete is pinned to, so an image binned in the browser meanwhile
    is refused rather than taken.
    With `--older-than DAYS`, only the images binned more than that many
    days ago: the cutoff is fixed when the preview is read and sent back
    unchanged, as the browser's age-scoped Empty does.
    No orphan sweep and no vacuum, as on the `--where` path.
    """
    if not settings.db_path.exists():
        print("No gallery yet.", file=sys.stderr)
        return 1
    days = args.older_than
    before = None if days is None else gallery_delete.bin_cutoff(days)
    found = gallery_delete.bin_summary(before)
    print(f"database : {settings.db_path}")
    print(f"images   : {settings.outputs_dir}")
    print()
    if before is None:
        print(f"  {found['records']:>6} images in the recycle bin, {_human(found['bytes'])}")
    else:
        print(
            f"  {found['records']:>6} of the {found['bin_records']} images in the recycle bin binned "
            f"more than {days} day(s) ago, {_human(found['bytes'])}"
        )
    if found["sources_in_use"]:
        print(
            f"  {len(found['sources_in_use']):>6} of them a live image was made from "
            "(a refine pass's source, a render's reference), which then cannot be made again"
        )
    print()
    if found["records"] == 0:
        print("The recycle bin is empty." if before is None or not found["bin_records"]
              else f"Nothing in the recycle bin was binned more than {days} day(s) ago.")
        return 0
    print(f"Would delete {found['records']} record(s) and their files.")
    if args.dry_run:
        print("\nDry run - nothing changed.")
        return 0
    if not _confirm(args.yes):
        return 1
    try:
        result = gallery_delete.empty_bin(expected=found["records"], pin=found["pin"], before=before)
    except gallery_delete.SelectionChanged as exc:
        now = (f"{exc.found} images are in it now, not the {exc.expected} shown"
               if exc.found != exc.expected else f"other images are in it now than the {exc.expected} shown")
        print(
            f"The recycle bin changed: {now}. Nothing was deleted; run it again.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001 - report, with the post-deletion code
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(f"\nDeleted {result['records']} record(s), {result['files']} file(s)" + _masks(result) + ".")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Redirected or piped, stdout on Windows encodes with the locale's code
    # page, where one CJK or emoji prompt in the preview was a
    # UnicodeEncodeError before anything was deleted. So does `> NUL`, which
    # answers isatty(); the Windows console is UTF-8 already. Elsewhere a
    # terminal keeps its own encoding. A StringIO has none to change, and a
    # scheduled `--yes` under pythonw has no stdout.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure) and (sys.platform == "win32" or not sys.stdout.isatty()):
        reconfigure(encoding="utf-8", errors="backslashreplace")
    from backend.launch import warn_legacy_env

    warn_legacy_env()  # a pre-rename variable left in a shell: named, not read
    parser = argparse.ArgumentParser(
        prog="python -m backend.clear",
        description="Delete generated images and their gallery history.",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="skip the confirmation prompt")
    parser.add_argument(
        "--keep-favorites", action="store_true",
        help="keep starred images, and the images they were made from (a refine pass's source, "
        "a render's references)",
    )
    parser.add_argument(
        "-n", "--dry-run", action="store_true", help="report what would go, change nothing"
    )
    parser.add_argument(
        "--where",
        metavar="TERMS",
        help="delete only the rows these key:value terms select - `job:ab12cd`, "
        "`run:ab12cd seed:42`, `date:yesterday`, `-seed:42` to leave one out (write "
        "`--where=-seed:42` when it comes first) - and their files; no orphan sweep, "
        "and free text is refused, `-word` included",
    )
    parser.add_argument(
        "--bin", action="store_true",
        help="empty the recycle bin only: the images the browser moved there, and their files",
    )
    parser.add_argument(
        "--older-than", type=int, metavar="DAYS",
        help="with --bin, only the images binned more than DAYS days ago",
    )
    args = parser.parse_args(argv)

    if args.older_than is not None and not args.bin:
        parser.error("--older-than scopes --bin; give --bin with it")
    if args.older_than is not None and args.older_than < 1:
        parser.error("--older-than takes a whole number of days, 1 or more")
    if args.bin:
        if args.where is not None or args.keep_favorites:
            parser.error("--bin empties the recycle bin alone; it takes neither --where nor --keep-favorites")
        return _clear_bin(args)
    if args.where:
        return _clear_where(args)

    # A dry run reports; it does not create the database it is reporting on.
    if not args.dry_run:
        storage.init_db()
    before = gallery_delete.summarise()

    kept = before["favorites"] if args.keep_favorites else 0
    sources = before["sources_kept"] if args.keep_favorites else 0
    doomed = before["records"] - kept - sources
    print(f"database : {settings.db_path}")
    print(f"images   : {settings.outputs_dir}")
    print()
    print(f"  {before['records']:>6} records ({before['favorites']} favorite)")
    if before["binned"]:
        print(f"  {before['binned']:>6} of those are in the recycle bin")
    print(f"  {before['files_on_disk']:>6} PNG files, {_human(before['bytes_on_disk'])}")
    if before["orphan_files"]:
        print(f"  {before['orphan_files']:>6} of those have no database row")
    if before["masks_on_disk"]:
        # Not gallery images, so in neither count above (their bytes are):
        # taken with the images whose inpaint they were.
        print(f"  {before['masks_on_disk']:>6} inpaint mask file(s)")
    if before["partial_files"]:
        print(
            f"  {before['partial_files']:>6} stale .part file(s) from interrupted writes"
        )
    if before["recoverable_partials"]:
        # Its row committed and the rename did not: a recorded image, finished
        # where its record is kept and deleted with it otherwise.
        print(
            f"  {before['recoverable_partials']:>6} recorded image(s) still named .part"
            " after an interrupted save"
        )
    print()

    if (before["records"] == 0 and before["files_on_disk"] == 0 and not before["partial_files"]
            and not before["masks_on_disk"]):
        print("Nothing to clear.")
        return 0

    # Name the orphans here too. They are deleted either way - they have no row
    # left to carry a favorite flag - and the line above the prompt is the last
    # thing anyone reads before typing 'clear'.
    plan = f"Would delete {doomed} record(s) and their files"
    if before["orphan_files"]:
        plan += f", plus {before['orphan_files']} untracked PNG(s)"
    if before["partial_files"]:
        plan += f", plus {before['partial_files']} stale .part file(s)"
    print(plan + _keeping(kept, sources))

    if args.dry_run:
        print("\nDry run - nothing changed.")
        return 0

    if not _confirm(args.yes):
        return 1

    try:
        result = gallery_delete.clear_history(keep_favorites=args.keep_favorites)
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
        + _masks(result)
        + "."
        # A kept image whose save was cut off between its row and its rename.
        + (f" Finished {result['recovered']} interrupted save(s)." if result["recovered"] else "")
    )
    try:
        gallery_delete.vacuum()
    except Exception as exc:  # noqa: BLE001 - best effort
        print(f"note: could not compact the database ({exc}); it will shrink later.", file=sys.stderr)
    after = gallery_delete.summarise()
    print(f"Remaining: {after['records']} record(s), {after['files_on_disk']} file(s), "
          f"{_human(after['bytes_on_disk'])}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
