"""Offline session pack contracts, no model load or GPU measurement."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools import anima_refine_pack as pack


def receipt():
    """Deliberately fabricated evidence exercises validation, not a measured result."""
    doc = pack.plan()["receipt"]
    doc["session"] = {key: "fixture-only" for key in pack.SESSION_FIELDS}
    doc["load_count"] = 1
    for row in doc["cases"]:
        spec = row["request"]
        cancel = "cancel_after_completed_steps" in spec
        row.update(status="cancelled" if cancel else "completed", guard_ok=True,
                   metrics={key: 1.0 for key in pack.METRICS},
                   completed_steps=1 if cancel else {None: 8, .125: 1, .25: 2, .4: 4}[spec["strength"]],
                   start_sigma={None: None, .125: .3, .25: .5, .4: .75}[spec["strength"]],
                   source_sha256="a" * 64 if spec["strength"] else None,
                   visual_review="fixture, not visual evidence")
        if not cancel:
            row.update(pixel_sha256="a" * 64, png_sha256="b" * 64, metadata_sha256="c" * 64)
    return doc


def test_pack_is_bounded_and_fresh():
    p = pack.plan()
    assert p["mode"] == "offline-plan-not-a-run"
    assert len(p["cases"]) == 12
    assert len({c["id"] for c in p["cases"]}) == 12
    assert all(c["source"] == "baseline" for c in p["cases"] if c["strength"] is not None)
    assert [c["id"] for c in p["cases"] if c.get("optional")] == ["upscale-125"]
    p["receipt"]["cases"][0]["status"] = "completed"
    assert pack.plan()["receipt"]["cases"][0]["status"] == "not-run"
    assert pack.check(pack.plan()["receipt"])


def test_fabricated_receipt_can_be_complete_without_claiming_qualification():
    assert pack.check(receipt()) == []


@pytest.mark.parametrize("field", pack.SESSION_FIELDS)
def test_each_session_receipt_is_required(field):
    r = receipt()
    r["session"][field] = None
    assert any(field in error for error in pack.check(r))


@pytest.mark.parametrize("metric", pack.METRICS)
@pytest.mark.parametrize("bad", [None, True, -1, float("nan"), float("inf"), "unknown", 10 ** 500])
def test_missing_or_invalid_metrics_are_not_zero(metric, bad):
    r = receipt()
    r["cases"][0]["metrics"][metric] = bad
    assert any(metric in error for error in pack.check(r))


@pytest.mark.parametrize("change", ["repeat", "source", "guard", "output-on-cancel", "steps", "request", "load", "stop"])
def test_corrupted_evidence_refuses(change):
    r = receipt()
    if change == "repeat":
        r["cases"][4]["pixel_sha256"] = "d" * 64
    elif change == "source":
        r["cases"][1]["source_sha256"] = "d" * 64
    elif change == "guard":
        r["cases"][1]["guard_ok"] = False
    elif change == "output-on-cancel":
        r["cases"][8]["pixel_sha256"] = "d" * 64
    elif change == "steps":
        r["cases"][3]["completed_steps"] = 3
    elif change == "request":
        r["cases"][1]["request"]["seed"] += 1
    elif change == "load":
        r["load_count"] = 2
    else:
        r["stopped_reason"] = "guard tripped"
    assert pack.check(r)


def test_optional_enlargement_can_be_refused_but_not_a_required_case():
    r = receipt()
    r["cases"][-1].update(status="skipped", reason="Guard projection refused enlargement")
    assert pack.check(r) == []
    r["cases"][-1]["reason"] = ""
    assert pack.check(r)
    r = receipt()
    r["cases"][1].update(status="skipped", reason="Did not run")
    assert pack.check(r)


@pytest.mark.parametrize("bad", [None, [], "text", {"cases": [None]}, {"schema": "wrong"}])
def test_malformed_receipts_fail_without_crashing(bad):
    assert pack.check(bad)


def test_case_inventory_is_exact():
    r = receipt()
    r["cases"].append(copy.deepcopy(r["cases"][0]))
    assert pack.check(r)
    r = receipt()
    r["cases"].reverse()
    assert pack.check(r)


def test_cli_uses_only_stdlib_and_has_no_run_command(tmp_path):
    script = Path(pack.__file__).resolve()
    result = subprocess.run([sys.executable, "-I", "-S", str(script), "plan"], capture_output=True, text=True)
    assert result.returncode == 0
    doc = json.loads(result.stdout)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(doc["receipt"]), encoding="utf-8")
    result = subprocess.run([sys.executable, "-I", "-S", str(script), "check", str(path)], capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)["qualification"].startswith("not-decided")
    result = subprocess.run([sys.executable, "-I", "-S", str(script), "run"], capture_output=True, text=True)
    assert result.returncode != 0


def test_settings_and_scheduler_are_not_silently_changed():
    r = receipt()
    r["settings"]["cfg"] = 4
    assert pack.check(r)
    r = receipt()
    r["cases"][3]["start_sigma"] = .4
    assert pack.check(r)


def test_pack_tail_and_sigmas_match_private_cpu_wrapper():
    from backend import refine
    from backend.backends.anima_pipeline import AnimaTurboPipeline
    from backend.backends.anima_refine import AnimaTurboImg2ImgPipeline, steps_run
    from tests.test_anima_refine import _text_pipe

    runner = refine.build(_text_pipe(AnimaTurboPipeline), AnimaTurboImg2ImgPipeline)
    for strength, expected_steps, sigma in ((.125, 1, .3), (.25, 2, .5), (.4, 4, .75)):
        assert steps_run(8, strength) == expected_steps
        assert runner.start_sigma(8, strength) == pytest.approx(sigma)
