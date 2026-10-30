"""`tools/smoke_timings.py` reads a run's summary.json and writes nothing.

docs/plan-tooling.md, "Suite and smoke timings": synthetic summaries shaped
like `run_smoke.py`'s, checked for the slowest-first order, `--top`, each
row's total (two rows on one port kept apart) and the refusals of a
missing, malformed or ambiguous summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import smoke_timings


def _receipt(smoke: str, seconds: float, port: int = 7862, model: str = "owner/model", result: str = "passed"):
    return {"smoke": smoke, "model": model, "port": port, "phase": "browser", "result": result,
            "exit_code": 0 if result == "passed" else 1, "duration_seconds": seconds,
            "artifact_dir": f"{smoke}-x"}


def _summary(tmp_path: Path, results: list[dict], **extra) -> Path:
    root = tmp_path / "auradream-smoke-artifacts-x"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "summary.json"
    path.write_text(json.dumps({"duration_seconds": 100.0, "results": results, "jobs": 3, "offset": 0, **extra}),
                    encoding="utf-8")
    return path


def _row(port, model, smokes):
    return smoke_timings._runner().Row(port, model, None, smokes)


def _lines(capsys) -> list[str]:
    return capsys.readouterr().out.splitlines()


def _section(lines: list[str], heading: str) -> list[str]:
    start = next(index for index, line in enumerate(lines) if line.startswith(heading)) + 2
    end = next((index for index in range(start, len(lines)) if not lines[index]), len(lines))
    return lines[start:end]


def test_slowest_first_with_ties_by_name_and_top(tmp_path, capsys):
    path = _summary(tmp_path, [_receipt("smoke-b", 5.0), _receipt("smoke-a", 5.0),
                               _receipt("smoke-c", 30.25, result="failed"), _receipt("smoke-d", 1.0)])
    assert smoke_timings.main([str(path), "--top", "3"]) == 0
    lines = _lines(capsys)
    assert lines[1] == "wall clock 100.0 s; smoke time 41.2 s over 4 smokes (jobs 3, offset 0)"
    assert lines[3] == "slowest 3 of 4 smokes:"
    slow = _section(lines, "slowest")
    assert [line.split()[-1] for line in slow] == ["smoke-c", "smoke-a", "smoke-b"]
    assert slow[0].split()[:3] == ["30.2", "failed", "7862"]


def test_top_defaults_to_twenty_five_and_refuses_zero(tmp_path, capsys):
    path = _summary(tmp_path, [_receipt(f"smoke-{index:02d}", float(index)) for index in range(30)])
    assert smoke_timings.main([str(path)]) == 0
    slow = _section(_lines(capsys), "slowest")
    assert len(slow) == 25 and slow[0].split()[-1] == "smoke-29"
    assert smoke_timings.main([str(path), "--top", "0"]) == 2


def test_row_totals_follow_the_port_table_and_keep_a_shared_port_apart():
    rows = [_row(7896, "owner/model", ["smoke-switch", "smoke-hover"]),
            _row(7896, "owner/model", ["smoke-gallery"]),
            _row(7862, "owner/other", ["smoke-flux"])]
    results = [_receipt("smoke-switch", 10.0, port=7996), _receipt("smoke-hover", 15.0, port=7996),
               _receipt("smoke-gallery", 20.0, port=8096), _receipt("smoke-flux", 3.0),
               _receipt("smoke-gone", 1.5, port=7999, model="owner/old")]
    totals = smoke_timings.row_totals(results, rows)
    assert [(entry["port"], entry["first"], entry["smokes"], entry["seconds"]) for entry in totals] == [
        (7896, "smoke-switch", 2, 25.0), (7896, "smoke-gallery", 1, 20.0),
        (7862, "smoke-flux", 1, 3.0), (7999, "smoke-gone", 1, 1.5)]
    assert [entry["listed"] for entry in totals] == [True, True, True, False]


def test_the_real_table_groups_and_a_rerun_is_counted_apart(tmp_path, capsys):
    path = _summary(tmp_path, [_receipt("smoke-analysis", 4.0), _receipt("smoke-rewrite", 6.0),
                               _receipt("smoke-flux", 2.0)],
                    rerun=[_receipt("smoke-rewrite", 7.5)], verdicts={"smoke-rewrite": "passed-alone"})
    assert smoke_timings.main([str(tmp_path)]) == 0
    lines = _lines(capsys)
    rows = _section(lines, "rows by total")
    assert rows[0].split()[:3] == ["10.0", "2", "7874"] and rows[0].endswith("smoke-analysis")
    assert rows[1].split()[:3] == ["2.0", "1", "7862"]
    assert lines[-1] == "rerun alone: 1 smokes, 7.5 s (not counted above)"
    assert lines[0] == f"summary: {path}"


def test_reading_writes_nothing(tmp_path):
    path = _summary(tmp_path, [_receipt("smoke-flux", 2.0)])
    before = sorted(item.relative_to(tmp_path) for item in tmp_path.rglob("*"))
    assert smoke_timings.main([str(path)]) == 0
    assert sorted(item.relative_to(tmp_path) for item in tmp_path.rglob("*")) == before


def test_a_missing_summary_is_refused(tmp_path, capsys):
    assert smoke_timings.main([str(tmp_path / "nowhere")]) == 2
    assert "does not exist" in capsys.readouterr().out
    (tmp_path / "empty").mkdir()
    assert smoke_timings.main([str(tmp_path / "empty")]) == 2
    assert "no summary.json below" in capsys.readouterr().out


def test_more_than_one_summary_is_refused(tmp_path, capsys):
    for name in ("run-a", "run-b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text('{"duration_seconds": 1, "results": []}', encoding="utf-8")
    assert smoke_timings.main([str(tmp_path)]) == 2
    out = capsys.readouterr().out
    assert "more than one summary.json" in out and "run-a" in out and "run-b" in out
    assert smoke_timings.main([str(tmp_path / "run-a" / "summary.json")]) == 0


@pytest.mark.parametrize(("text", "message"), [
    ("not json", "is not a readable summary.json"),
    ("[]", "has no results list"),
    ('{"duration_seconds": 1}', "has no results list"),
    ('{"results": []}', "has no duration_seconds"),
    ('{"duration_seconds": true, "results": []}', "has no duration_seconds"),
    ('{"duration_seconds": 1, "results": {}}', "`results` is not a list"),
    ('{"duration_seconds": 1, "results": [{"smoke": "smoke-flux"}]}', "has no smoke name and duration_seconds"),
    ('{"duration_seconds": 1, "results": [{"duration_seconds": 2}]}', "has no smoke name and duration_seconds"),
    ('{"duration_seconds": 1, "results": [], "rerun": [{"smoke": "x", "duration_seconds": "2"}]}',
     "a `rerun` receipt"),
])
def test_a_malformed_summary_is_refused(tmp_path, capsys, text, message):
    path = tmp_path / "summary.json"
    path.write_text(text, encoding="utf-8")
    assert smoke_timings.main([str(path)]) == 2
    assert message in capsys.readouterr().out
