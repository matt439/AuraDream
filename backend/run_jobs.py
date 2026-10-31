"""Retained run jobs: each admitted job's resolved cells and definition, kept
in the gallery database's `run_jobs` table for Finish run, and its terminal
outcome.

The table's schema and creation stay in `storage`. Like `gallery_delete`,
everything storage owns - `settings`, the one `_write_lock`, `_session` - is
read through `storage` at call time, so the tests' replacement of
`storage.settings` reaches these writes too. Callers reach these names as
`storage.<name>` (storage's `_MOVED`), which is also where tests patch them.
"""
from __future__ import annotations

import gzip
import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import gallery_lock, storage

if TYPE_CHECKING:
    from .grid import Cell


@gallery_lock.protected(lambda: storage.settings.db_path, waits=True)
def retain_run_job(definition: dict[str, Any], cells: Sequence[Cell]) -> None:
    """Keep a job's resolved, pre-outpaint-stamp cells before admission.

    The raw requests are intentional: applied settings are compared when a
    run is finished, and an outpaint's mask must be built again at submit.
    Compression happens before the write lock; every input is JSON-native.
    A failed write raises to the submitter, which releases any mask holds
    and never admits the job. This table is local recovery state, not PNG
    metadata or part of an exported bundle.
    """
    encoded = gzip.compress(json.dumps([
        {"request": cell.request.model_dump(mode="json"), "seed": cell.seed,
         "label": cell.label, "variables": cell.variables}
        for cell in cells
    ], ensure_ascii=True, separators=(",", ":")).encode("utf-8"), mtime=0)
    with storage._write_lock, storage._session() as conn:
        conn.execute(
            """INSERT INTO run_jobs
               (job_id, run_id, created_at, origin, model_id, quantization,
                width, height, batch, comparison, metadata, cells)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (definition["job_id"], definition["run_id"], definition["created_at"],
             definition["origin"], definition["model_id"], definition["quantization"],
             definition["width"], definition["height"], definition["batch"],
             int(definition["comparison"]), json.dumps(definition["metadata"], ensure_ascii=True), encoded),
        )


def retained_run_metadata(job_ids: Sequence[str]) -> dict[str, tuple[str, dict[str, Any]]]:
    """Each retained job's (run id, metadata), by job id, without decoding
    its cells: what names a Then refine pass run's base (its records' job is
    the base job, whose metadata carries the pass run id). A job not
    retained, or whose metadata is not an object, is left out."""
    ids = list(dict.fromkeys(job_ids))
    if not ids:
        return {}
    with storage._session() as conn:
        rows = conn.execute(
            f"SELECT job_id, run_id, metadata FROM run_jobs WHERE job_id IN ({','.join('?' * len(ids))})", ids,
        ).fetchall()
    found = {}
    for row in rows:
        try:
            metadata = json.loads(row["metadata"])
        except (TypeError, ValueError):
            continue
        if isinstance(metadata, dict):
            found[row["job_id"]] = (row["run_id"], metadata)
    return found


def retained_run_jobs(run_id: str | None = None) -> list[dict[str, Any]]:
    """Retained jobs in creation order, with raw cells and metadata decoded.

    A NULL outcome means the process ended before its terminal write, or
    that write failed. It never means the job is still running: the live
    JobManager alone answers that. A malformed/obsolete request is a worded
    ValueError for the recovery API rather than an unvalidated pipeline call.
    """
    from .grid import Cell
    from .schemas import GenerateRequest
    from .seeds import MAX_SEED

    where = " WHERE run_id = ?" if run_id is not None else ""
    with storage._session() as conn:
        rows = conn.execute(
            f"SELECT * FROM run_jobs{where} ORDER BY created_at, job_id",
            (run_id,) if run_id is not None else (),
        ).fetchall()
    jobs = []
    for row in rows:
        item = dict(row)
        try:
            raw = json.loads(gzip.decompress(item["cells"]))
            if not isinstance(raw, list):
                raise ValueError("cells must be a list")
            for cell in raw:
                if (not isinstance(cell, dict) or type(cell.get("seed")) is not int
                        or not 0 <= cell["seed"] <= MAX_SEED
                        or cell.get("label") is not None and not isinstance(cell["label"], str)
                        or cell.get("variables") is not None and (
                            not isinstance(cell["variables"], dict)
                            or not all(isinstance(key, str) and isinstance(value, str)
                                       for key, value in cell["variables"].items()))):
                    raise ValueError("invalid retained cell")
            item["cells"] = [Cell(
                request=GenerateRequest.model_validate(cell["request"]), seed=cell["seed"],
                label=cell["label"], variables=cell["variables"],
            ) for cell in raw]
            item["metadata"] = json.loads(item["metadata"])
            item["outcome"] = json.loads(item["outcome"]) if item["outcome"] is not None else None
            metadata, outcome = item["metadata"], item["outcome"]
            if not isinstance(metadata, dict) or outcome is not None and not isinstance(outcome, dict):
                raise ValueError("invalid retained metadata or outcome")
            budget = metadata.get("budget_minutes")
            if budget is not None and (type(budget) not in (int, float) or not 0 < budget <= 7 * 24 * 60):
                raise ValueError("invalid retained budget")
            if metadata.get("prompt_template") is not None and not isinstance(metadata["prompt_template"], str):
                raise ValueError("invalid retained prompt template")
            if outcome is not None:
                indices = outcome.get("completed_indices", [])
                if not isinstance(indices, list) or any(type(index) is not int for index in indices):
                    raise ValueError("invalid retained completion indices")
                if outcome.get("finished_at") is not None:
                    if type(outcome["finished_at"]) not in (int, float):
                        raise ValueError("invalid retained finish time")
                    datetime.fromtimestamp(outcome["finished_at"], timezone.utc)
        except (ValueError, TypeError, KeyError, OSError, EOFError, OverflowError) as exc:
            raise ValueError(f"the saved cells of job {row['job_id']} cannot be read by this build") from exc
        item["comparison"] = bool(item["comparison"])
        jobs.append(item)
    return jobs


def finish_run_job(job_id: str, outcome: dict[str, Any], *, database: Path | None = None) -> None:
    """Record one terminal outcome, without recreating a cleared job.

    Called on a daemon thread after `_finish` has made its terminal
    transition. Interruption or a disk error may leave outcome NULL; Finish
    still compares the cells with what the gallery holds.
    """
    # The caller captures the path before starting its daemon. A late thread
    # must not follow changed settings to another gallery (or recreate a
    # database that has gone). This writer needs no image schema migration.
    path = (database or storage.settings.db_path).resolve()
    with gallery_lock.hold(path, create=False, wait=gallery_lock.SESSION_WAIT), storage._write_lock:
        conn = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=15)
        try:
            with conn:
                if conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'run_jobs'"
                ).fetchone():
                    conn.execute(
                        "UPDATE run_jobs SET outcome = ? WHERE job_id = ? AND outcome IS NULL",
                        (json.dumps(outcome, ensure_ascii=True), job_id),
                    )
        finally:
            conn.close()
