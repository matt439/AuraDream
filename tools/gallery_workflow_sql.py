"""Profile the storage reads used by the 5,000-image workflow outside browser timings."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
import time
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from tools.gallery_workflow_benchmark import quantiles, validate_source
from tools.gallery_workflow_fixture import MODEL, copy_database


def inspect(base: Path) -> dict:
    validate_source(base)
    with tempfile.TemporaryDirectory(prefix="auradream-workflow-sql-") as scratch:
        database = Path(scratch) / "auradream.db"
        copy_database(base / "auradream.db", database)
        os.environ.update(AURADREAM_DB_PATH=str(database), AURADREAM_OUTPUTS_DIR=str(base / "outputs"))
        from backend import gallery_reports, storage

        original_connect = storage._connect
        queries: list[str] = []

        def observed_connect() -> sqlite3.Connection:
            connection = original_connect()
            connection.set_trace_callback(queries.append)
            return connection

        reads = {
            "prompt_page": lambda: storage.list_images_at_head(search="lighthouse", limit=48, model_id=MODEL),
            "starred_page": lambda: storage.list_images_at_head(search="lighthouse", limit=48, model_id=MODEL,
                                                                favorites_only=True),
            "run_sheet": lambda: storage.list_images_at_head(search="run:0000000000fa", limit=1024, model_id=MODEL,
                                                             favorites_only=True),
            "lineage": lambda: storage.lineage_images(["0000000000f9", "0000000000fa"], []),
            "held_seeds": lambda: storage.completed_cells(["A orchard in evening light, study 248",
                                                         "A lighthouse in evening light, study 249"],
                                                        256, 256, MODEL, "fp8"),
            "direct_cost": lambda: gallery_reports.step_cost_ms(256, 256, MODEL),
            "refine_cost": lambda: gallery_reports.refine_cost_ms(256, 256, MODEL),
        }
        report = {}
        with closing(sqlite3.connect(database)) as planner:
            for name, read in reads.items():
                queries.clear()
                with patch.object(storage, "_connect", observed_connect):
                    read()
                selects = [q for q in queries if q.lstrip().upper().startswith("SELECT")]
                plans = [row[-1] for sql in selects for row in planner.execute("EXPLAIN QUERY PLAN " + sql)]
                samples = []
                for _ in range(20):
                    started = time.perf_counter()
                    read()
                    samples.append((time.perf_counter() - started) * 1000)
                report[name] = {"n": len(samples), "ms": quantiles(samples),
                                "query_plans": list(dict.fromkeys(plans))}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    print(json.dumps(inspect(parser.parse_args().fixture.resolve()), indent=2))
