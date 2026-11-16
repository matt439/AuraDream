"""Offline protocol/receipt validation; all evidence here is fabricated."""
from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools import prompt_assistant_pack as pack


def completed_receipt():
    """A complete prompt-only fixture, never a measured or judged GPU result."""
    doc = pack.receipt()
    doc["session"] = {key: "fixture-only" for key in pack.SESSION_FIELDS}
    doc["session"].update(git_revision="a" * 40, gpu_total_mib=16303, system_ram_gib=64)
    doc["runtime"].update(selection="pinned", model_label=pack.BASELINE["model"],
                          model_sha256=pack.BASELINE["model_sha256"], server_build=pack.BASELINE["llama_cpp"],
                          server_sha256="b" * 64)
    doc["llm_preflight_available_ram_gib"] = 20
    doc["llm_preflight_card_mib"] = 2500
    doc["llm_preflight_busy_card_mib"] = 3500
    doc["integration"]["reason"] = "Deferred until prompt-only checks and separate image-load authorization."
    for row in doc["cases"]:
        if row["spec"].get("optional"):
            row["reason"] = "Runtime already cached; no fresh multi-GB download requested."
            continue
        row.update(status="completed", cache_state=row["spec"]["cache_state"],
                   request_receipt="fixture-request.json", response_receipt="fixture-response.json",
                   guard_ok=True, process_released=True,
                   metrics={key: 20.0 if "headroom" in key or "free_ram" in key else 1.0 for key in pack.METRICS})
        row["quality"].update(passed=True, notes="Fabricated fixture; no quality assessment occurred.")
        for criterion in row["quality"]["checks"]:
            criterion.update(passed=True, notes="Fixture-only verdict.")
        if row["spec"].get("history_from"):
            row["sent_history"] = [{"role": "user", "content": pack.CREATE},
                                   {"role": "assistant", "content": "Fabricated earlier answer."}]
    for item in [*doc["safety"].values(), doc["lookup_unavailable"]]:
        item.update(passed=True, evidence="Fixture-only evidence path; not a real observation.")
    return doc


def test_pack_is_fresh_bounded_and_explicitly_offline():
    plan = pack.plan()
    assert plan["mode"] == "offline-plan-not-a-run"
    assert plan["baseline"]["model"] == "Qwen3-4B Q8_0"
    assert len(plan["cases"]) == 6
    assert [row["id"] for row in plan["cases"] if row.get("optional")] == ["create-cold"]
    assert {row["request"]["mode"] for row in plan["cases"]} == {"create", "improve", "tags"}
    assert all(row["request"]["ticket"] for row in plan["cases"])
    assert "--busy-card-mib <MAX_IDLE_CARD_MIB>" in plan["guard_command_template"]
    assert pack.check(pack.receipt())
    plan["cases"][0]["request"]["prompt"] = "changed"
    assert pack.plan()["cases"][0]["request"]["prompt"] == ""
    assert pack.receipt()["cases"][0]["status"] == "not-run"


def test_prompt_only_fixture_can_be_complete_without_download_or_image_load():
    assert pack.check(completed_receipt()) == []


@pytest.mark.parametrize("field", pack.SESSION_FIELDS)
def test_session_evidence_is_required(field):
    doc = completed_receipt()
    doc["session"][field] = None
    assert any(field in error for error in pack.check(doc))


@pytest.mark.parametrize("metric", pack.METRICS)
@pytest.mark.parametrize("bad", [None, True, -1, float("nan"), float("inf"), "unknown", 10 ** 500])
def test_absent_or_invalid_measurements_are_not_zero(metric, bad):
    doc = completed_receipt()
    doc["cases"][1]["metrics"][metric] = bad
    assert any(metric in error for error in pack.check(doc))


@pytest.mark.parametrize("case", ["guard", "orphan", "quality", "quality-criterion", "cache", "request", "history", "revision", "limits", "preflight", "stop"])
def test_corrupted_or_incomplete_recorded_evidence_fails(case):
    doc = completed_receipt()
    row = doc["cases"][1]
    if case == "guard":
        row["guard_ok"] = False
    elif case == "orphan":
        row["process_released"] = False
    elif case == "quality":
        row["quality"]["passed"] = None
    elif case == "quality-criterion":
        row["quality"]["checks"][0]["notes"] = ""
    elif case == "cache":
        row["cache_state"] = "missing"
    elif case == "request":
        row["spec"]["request"]["instruction"] = "Different evaluation task"
    elif case == "history":
        doc["cases"][2]["sent_history"] = []
    elif case == "revision":
        doc["session"]["git_revision"] = "main"
    elif case == "limits":
        doc["limits"]["card_mib"] = 16303
    elif case == "preflight":
        doc["llm_preflight_available_ram_gib"] = 11
    else:
        doc["stopped_reason"] = "Guard tripped; retain the partial receipt."
    assert pack.check(doc)


@pytest.mark.parametrize("metric,value", [("whole_card_peak_mib", 15501), ("adapter_shared_rise_mib", 1025),
                                         ("process_shared_rise_mib", 513), ("minimum_free_ram_gib", 3.9),
                                         ("minimum_commit_headroom_gib", 3.9), ("elapsed_seconds", 601)])
def test_claiming_guard_pass_cannot_hide_out_of_bounds_metrics(metric, value):
    doc = completed_receipt()
    doc["cases"][1]["metrics"][metric] = value
    assert any(metric in error for error in pack.check(doc))


def test_optional_download_needs_a_reason_and_cannot_claim_measurements():
    doc = completed_receipt()
    doc["cases"][0]["reason"] = ""
    assert pack.check(doc)
    doc = completed_receipt()
    doc["cases"][0]["guard_ok"] = True
    assert pack.check(doc)
    doc = completed_receipt()
    doc["cases"][1].update(status="skipped", reason="Required cached baseline was omitted")
    assert pack.check(doc)


def test_later_image_stage_requires_budget_checkpoint_and_its_own_evidence():
    doc = completed_receipt()
    doc["integration"].update(status="completed", checkpoint=pack.ANIMAGINE, ram_budget_gib=16)
    assert pack.check(doc)
    for item in doc["integration"]["checks"].values():
        item.update(passed=True, evidence="Fixture-only integration evidence.")
    assert pack.check(doc) == []
    doc["integration"]["ram_budget_gib"] = None
    assert pack.check(doc)
    doc = completed_receipt()
    doc["integration"]["checks"]["release_before_generation"]["passed"] = True
    assert pack.check(doc)


def test_custom_model_and_server_paths_are_recorded_separately_from_the_baseline():
    doc = completed_receipt()
    doc["runtime"].update(selection="custom", model_label="test-model", model_sha256="c" * 64,
                          model_override="C:/fixtures/test-model.gguf")
    assert pack.check(doc) == []
    doc["runtime"]["model_override"] = "relative.gguf"
    assert pack.check(doc)
    doc["runtime"]["model_override"] = None
    assert pack.check(doc)
    doc = completed_receipt()
    doc["runtime"]["model_sha256"] = "c" * 64
    assert pack.check(doc)


@pytest.mark.parametrize("bad", [None, [], "text", {}, {"cases": [None]}, {"schema": "wrong"}])
def test_malformed_receipts_fail_without_crashing(bad):
    assert pack.check(bad)


def test_inventory_is_exact_and_smaller_cards_cannot_use_the_16gb_protocol():
    doc = completed_receipt()
    doc["cases"].append(copy.deepcopy(doc["cases"][0]))
    assert pack.check(doc)
    doc = completed_receipt()
    doc["cases"].reverse()
    assert pack.check(doc)
    doc = completed_receipt()
    doc["session"]["gpu_total_mib"] = 8192
    assert pack.check(doc)


@pytest.mark.parametrize("reading,ceiling", [(None, 3500), (2500, None), (4000, 3500), (2500, 15500),
                                           (True, 3500), (2500, float("inf"))])
def test_card_busy_preflight_is_explicit_and_admits_only_the_recorded_idle_baseline(reading, ceiling):
    doc = completed_receipt()
    doc["llm_preflight_card_mib"] = reading
    doc["llm_preflight_busy_card_mib"] = ceiling
    assert any("card-busy preflight" in error for error in pack.check(doc))


def test_pins_and_guard_snapshot_match_the_current_source_without_importing_backend():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "backend" / "critic_runtime.py").read_text(encoding="utf-8"))
    pins = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"RELEASE", "MODEL_SHA256", "MODEL_LABEL"}}
    assert pack.BASELINE == {"model": pins["MODEL_LABEL"], "llama_cpp": pins["RELEASE"],
                             "model_sha256": pins["MODEL_SHA256"]}
    from tools.card_guard import LIMITS
    assert pack.LIMITS == {key: getattr(LIMITS, key) for key in pack.LIMITS}


def test_cli_is_stdlib_only_refuses_overwrites_and_cannot_run_a_session(tmp_path):
    script = str(Path(pack.__file__).resolve())
    base = [sys.executable, "-I", "-S", script]
    result = subprocess.run([*base, "plan"], capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)["mode"] == "offline-plan-not-a-run"
    path = tmp_path / "blank.json"
    result = subprocess.run([*base, "receipt", "--output", str(path)], capture_output=True, text=True)
    assert result.returncode == 0 and json.loads(path.read_text(encoding="utf-8")) == pack.receipt()
    before = path.read_bytes()
    result = subprocess.run([*base, "receipt", "--output", str(path)], capture_output=True, text=True)
    assert result.returncode == 1 and path.read_bytes() == before
    result = subprocess.run([*base, "check", str(path)], capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)["qualification"].startswith("not-decided")
    path.write_text(json.dumps(completed_receipt()), encoding="utf-8")
    result = subprocess.run([*base, "check", str(path)], capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert result.returncode == 0 and report["receipt_complete"] is True
    assert report["recorded_stages"] == {"missing_cache_download": "not-run", "image_integration": "not-run"}
    assert report["qualification"].startswith("not-decided")
    result = subprocess.run([*base, "run"], capture_output=True, text=True)
    assert result.returncode != 0
