"""Print an offline prompt-assistant GPU protocol and audit a recorded receipt.

Only stdlib JSON/file operations: no backend, model, network, process or counter
access. These commands never start a session. Receipt completeness is not a GPU
measurement, a quality verdict or permission to run the protocol.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, TypeGuard

SCHEMA = "prompt-assistant-receipt-v1"
# Frozen baseline from critic_runtime.py, not a recommendation for a new model.
BASELINE = {"model": "Qwen3-4B Q8_0", "llama_cpp": "b11206",
            "model_sha256": "8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300"}
FLUX = "black-forest-labs/FLUX.1-Krea-dev"
ANIMAGINE = "cagliostrolab/animagine-xl-4.0"
CREATE = ('Offer two distinct prompts for one red fox wearing a blue scarf outside a small bookshop '
          'at dawn. Its sign must read exactly "OPEN LATE". No people. Keep those details in both options.')
PROSE = ('A woman with green hair and twintails holds a red umbrella beside a lighthouse. '
         'A sign reads "OPEN LATE". No other people.')
METRICS = ("elapsed_seconds", "whole_card_peak_mib", "adapter_shared_rise_mib",
           "process_shared_rise_mib", "process_ram_peak_gib", "minimum_free_ram_gib",
           "minimum_commit_headroom_gib", "whole_card_after_mib", "process_ram_after_gib")
SESSION_FIELDS = ("git_revision", "utc_started", "gpu_name", "gpu_total_mib", "system_ram_gib",
                  "platform", "driver", "guard_log", "preflight_receipt", "runtime_receipt",
                  "final_residency_receipt")
SAFETY = {
    "cold_without_image_weights": "All drafting cases run without loading image weights or calling ensure_loaded.",
    "stop_before_start": "Stopping a ticket before its POST arrives prevents that ticket starting inference.",
    "stop_active": "Stop during an answer ends its process tree before releasing the GPU gate.",
    "recover_after_stop": "A fresh request after Stop completes; a late reply cannot replace the new result.",
    "no_orphans": "No llama-server process or listening socket remains after every answer and cancellation.",
    "stale_apply_refused": "Editing the source or changing checkpoint prevents an old draft overwriting it.",
    "shutdown_releases_process": "Stopping the app during an answer leaves no llama-server process or socket.",
}
INTEGRATION = {
    "job_refused_before_prepare": "An image job queued/running refuses assistant work before preparation.",
    "load_refused_before_prepare": "An image-model load refuses assistant work before preparation.",
    "load_during_prepare_refused": "A load begun during first-use preparation prevents LLM inference afterwards.",
    "release_before_generation": "An image job submitted during an answer starts only after llama-server exits.",
    "generation_unchanged": "The same fixed-seed image before/after assistant use has identical decoded pixels.",
}
# Snapshot of the existing card_guard defaults. Never loosen them in a pack.
LIMITS = {"card_mib": 15500, "adapter_shared_rise_mib": 1024,
          "process_shared_rise_mib": 512, "ram_floor_gib": 4, "commit_floor_gib": 4}


def cases() -> list[dict[str, Any]]:
    """Fixed writing tasks, judged for retained intent rather than exact prose."""
    def task(name: str, mode: str, instruction: str, *, checkpoint: str = FLUX,
             prompt: str = "", negative: str | None = None, **extra: Any) -> dict[str, Any]:
        return {"id": name, "checkpoint": checkpoint,
                "request": {"ticket": f"assistant-pack-{name}", "mode": mode,
                            "instruction": instruction, "prompt": prompt,
                            "negative_prompt": negative, "history": []}, **extra}

    return [
        task("create-cold", "create", CREATE, cache_state="missing", optional=True,
             quality_checks=["Two useful, distinct drafts", "One red fox; blue scarf; bookshop; dawn",
                             'Lettering exactly OPEN LATE; no invented lettering', "No people"]),
        task("create-warm", "create", CREATE, cache_state="cached",
             quality_checks=["Two useful, distinct drafts", "Same essential details as create-cold",
                             "Record quality independently; sampling need not repeat identical wording"]),
        task("conversation", "create", "Keep the fox, scarf, bookshop and exact sign, but change dawn to dusk. "
             "Make the framing closer. Give two alternatives.", cache_state="cached", history_from="create-warm",
             quality_checks=["Uses the previous turn", "Dusk replaces dawn; framing is closer",
                             'Fox, blue scarf, bookshop and exact OPEN LATE lettering remain; no people']),
        task("improve-prose", "improve", "Improve clarity and composition while keeping every requested detail. "
             "Do not invent more subjects, lettering or a different mood.", prompt=PROSE,
             negative="crowd, extra people", cache_state="cached",
             quality_checks=["One woman; green hair; twintails; red umbrella; lighthouse",
                             'Lettering exactly OPEN LATE; no other people',
                             "No requested detail is silently removed or changed", "An improvement is useful"]),
        task("improve-tags", "improve", "Organize these tags for this checkpoint, keeping the subject, "
             "hair colour, hairstyle, umbrella and setting.", checkpoint=ANIMAGINE,
             prompt="1girl, green hair, twintails, red umbrella, lighthouse",
             negative="crowd, extra people", cache_state="cached",
             quality_checks=["Keeps 1girl, green hair, twintails, red umbrella and lighthouse",
                             "Uses this checkpoint's tag guidance without inventing author requirements"]),
        task("find-tags", "tags", "Find the Danbooru tag for a single stray lock of hair sticking up from "
             "the head. Also check zzkx_invalid_tag_20261116; do not treat an unverified tag as confirmed.",
             checkpoint=ANIMAGINE, cache_state="cached",
             quality_checks=["Finds ahoge and explains the match", "Validation comes from the tag index",
                             "Unknown tag is withheld or clearly unverified", "No invented dictionary definition"]),
    ]


def plan() -> dict[str, Any]:
    """Describe future presses/requests; this function cannot perform them."""
    return {
        "schema": "prompt-assistant-plan-v1", "mode": "offline-plan-not-a-run", "baseline": dict(BASELINE),
        "endpoints": {"ask": "POST /api/prompt/assist", "stop": "POST /api/prompt/assist/{ticket}/stop"},
        "constraints": [
            "Run only after Matt explicitly authorizes the GPU session; this pack grants no permission.",
            "Use a scratch gallery and a quiet RTX 5080 16 GB window. Start with image weights unloaded.",
            "State the system-RAM budget first. Reserve at least 12 GiB available for the LLM-only stage; "
            "this is a conservative admission budget, not a measured RAM claim. Keep the 4 GiB RAM/commit floors.",
            "The missing-cache create-cold case is optional. Use the already cached baseline first; record the "
            "download stage not-run with a reason. Only if separately authorized, use a separate empty cache for "
            "this app process, never clear the shared cache, and record elapsed time including download/preparation.",
            "Use card_guard and retain its raw log, limits, preflight, achieved sample period and final residency. "
            "The guard watches the app's process tree, including each transient llama-server.",
            "Measure the quiet desktop/app whole-card baseline before inference; choose and record an explicit "
            "MAX_IDLE_CARD_MIB busy-card ceiling above that baseline and below the card limit. Supply it to "
            "card_guard before asking the LLM. Never raise it to accommodate unrelated GPU work.",
            "Stop the session at the first spill, OOM, guard trip, missing counter coverage or timeout. "
            "Retain partial/failed evidence; never retry it into a passing receipt.",
            "Select each case's checkpoint without loading weights. Capture the actual request and full response.",
            "For conversation, retain create-warm's user/assistant turns in history; record the exact history sent. "
            "Do not replace the previous assistant answer with the expected result.",
            "Judge quality by eye against every listed check. A longer or more fluent prompt is not automatically better.",
            "Tag validation uses Danbooru; candidate names reach that service. Repeat with its lookup unavailable "
            "and verify the UI says unchecked instead of claiming verification.",
            "Image integration is a separate, optional later stage: leave it not-run during initial prompt-only "
            "GPU checks. After prompt-only checks pass and that stage is authorized, choose a separately measured "
            "image checkpoint and record its RAM preflight budget. No concurrent LLM/image inference is allowed.",
            "Image comparison uses one fixed prompt/seed/size/settings before and after assistant use; retain settings "
            "and decoded pixel hashes. Use tools/image_compare.py only on the saved images.",
            "Repeat this whole receipt for any different model/server override, platform or quantization. "
            "A completed baseline receipt does not qualify another pairing.",
        ],
        "guard_command_template": "uv run python tools/card_guard.py watch --watch-pid <APP_PID> "
        "--need-ram-gib 12 --busy-card-mib <MAX_IDLE_CARD_MIB> --card-mib 15500 "
        "--adapter-shared-mib 1024 --process-shared-mib 512 "
        "--ram-floor-gib 4 --commit-floor-gib 4 --out <NEW_GUARD_JSONL>",
        "limits": dict(LIMITS), "attempt_budgets_seconds": {"session": 1800, "request": 600},
        "cases": cases(), "safety_checks": dict(SAFETY), "later_image_integration_checks": dict(INTEGRATION),
        "owner": "docs/maintainer/plan-prompt-assistant.md (PA#18)",
        "receipt_command": "uv run python tools/prompt_assistant_pack.py receipt --output <NEW_RECEIPT_JSON>",
        "check_command": "uv run python tools/prompt_assistant_pack.py check <RECEIPT_JSON>",
        "qualification": "not-decided; this is a protocol, never a run or a measurement",
    }


def receipt() -> dict[str, Any]:
    """Fresh, intentionally incomplete receipt; no sample evidence is fabricated."""
    return {
        "schema": SCHEMA, "session": {key: None for key in SESSION_FIELDS},
        "runtime": {"selection": None, "model_label": None, "model_sha256": None,
                    "server_build": None, "server_sha256": None,
                    "model_override": None, "server_override": None},
        "limits": dict(LIMITS), "llm_preflight_available_ram_gib": None,
        "llm_preflight_card_mib": None, "llm_preflight_busy_card_mib": None,
        "integration": {"status": "not-run", "reason": None, "checkpoint": None, "ram_budget_gib": None,
                        "checks": {key: {"passed": None, "evidence": None} for key in INTEGRATION}},
        "stopped_reason": None,
        "cases": [{"id": row["id"], "spec": row, "status": "not-run", "reason": None, "cache_state": None,
                   "request_receipt": None, "response_receipt": None, "sent_history": [],
                   "metrics": {key: None for key in METRICS}, "guard_ok": None,
                   "process_released": None, "quality": {"passed": None, "notes": None,
                   "checks": [{"criterion": text, "passed": None, "notes": None} for text in row["quality_checks"]]}}
                  for row in cases()],
        "safety": {name: {"passed": None, "evidence": None} for name in SAFETY},
        "lookup_unavailable": {"passed": None, "evidence": None},
    }


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _number(value: Any) -> TypeGuard[int | float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def check(value: Any) -> list[str]:
    """Check recorded completeness/consistency, without reading evidence paths."""
    if not isinstance(value, dict):
        return ["receipt must be an object"]
    errors: list[str] = []
    if value.get("schema") != SCHEMA:
        errors.append("unknown receipt schema")
    session = _object(value.get("session"))
    for key in SESSION_FIELDS:
        item = session.get(key)
        valid = _number(item) and item > 0 if key in ("gpu_total_mib", "system_ram_gib") else _nonempty(item)
        if not valid:
            errors.append(f"session: missing/invalid {key}")
    if _number(session.get("gpu_total_mib")) and session["gpu_total_mib"] < LIMITS["card_mib"]:
        errors.append("session: this bounded 16 GB protocol is not for a smaller card")
    revision = session.get("git_revision")
    if not (isinstance(revision, str) and len(revision) in (40, 64)
            and all(c in "0123456789abcdef" for c in revision)):
        errors.append("session: git_revision must be a full source revision")
    runtime = _object(value.get("runtime"))
    if runtime.get("selection") not in ("pinned", "custom"):
        errors.append("runtime: record pinned or custom selection")
    for key in ("model_label", "server_build"):
        if not _nonempty(runtime.get(key)):
            errors.append(f"runtime: missing {key}")
    for key in ("model_sha256", "server_sha256"):
        if not _digest(runtime.get(key)):
            errors.append(f"runtime: missing/invalid {key}")
    for key in ("model_override", "server_override"):
        if key not in runtime or (runtime[key] is not None and not _nonempty(runtime[key])):
            errors.append(f"runtime: record {key} as null or the explicit override path")
        elif isinstance(runtime[key], str) and not (
            PureWindowsPath(runtime[key]).is_absolute() or PurePosixPath(runtime[key]).is_absolute()
        ):
            errors.append(f"runtime: {key} must be an absolute path")
    overrides = any(_nonempty(runtime.get(key)) for key in ("model_override", "server_override"))
    if runtime.get("selection") == "pinned":
        if overrides or runtime.get("model_label") != BASELINE["model"] or runtime.get("server_build") != BASELINE["llama_cpp"]:
            errors.append("runtime: pinned label/build/overrides differ from the baseline")
        if runtime.get("model_sha256") != BASELINE["model_sha256"]:
            errors.append("runtime: pinned model hash differs from the baseline")
    elif runtime.get("selection") == "custom" and not overrides:
        errors.append("runtime: custom selection must record its override path")
    if value.get("limits") != LIMITS:
        errors.append("limits differ from the bounded protocol; do not loosen guard limits")
    available = value.get("llm_preflight_available_ram_gib")
    if not _number(available) or available < 12:
        errors.append("missing/refused LLM-only 12 GiB available-RAM preflight")
    card = value.get("llm_preflight_card_mib")
    busy = value.get("llm_preflight_busy_card_mib")
    if not _number(card) or not _number(busy) or not card <= busy < LIMITS["card_mib"]:
        errors.append("missing/refused explicit card-busy preflight baseline and ceiling")
    integration = _object(value.get("integration"))
    if integration.get("status") == "not-run":
        if not _nonempty(integration.get("reason")):
            errors.append("integration: record why the later image stage is not-run")
        checks = _object(integration.get("checks"))
        if any(_object(checks.get(key)).get("passed") is not None for key in INTEGRATION):
            errors.append("integration: not-run stage must not claim a check passed or failed")
    elif integration.get("status") == "completed":
        if not _nonempty(integration.get("checkpoint")):
            errors.append("integration: missing checkpoint")
        budget = integration.get("ram_budget_gib")
        if not _number(budget) or budget <= 4:
            errors.append("integration: missing image-load RAM budget above the guard floor")
        checks = _object(integration.get("checks"))
        for key in INTEGRATION:
            item = _object(checks.get(key))
            if item.get("passed") is not True or not _nonempty(item.get("evidence")):
                errors.append(f"integration {key}: missing/pending/failed recorded check")
    else:
        errors.append("integration: record completed or not-run; retain failed stage evidence separately")
    if value.get("stopped_reason") is not None:
        errors.append("session stopped; retain partial evidence, do not mark complete")
    rows = value.get("cases")
    specs = cases()
    if not isinstance(rows, list) or [r.get("id") if isinstance(r, dict) else None for r in rows] != [s["id"] for s in specs]:
        return [*errors, "case ids/order differ (missing, extra or duplicate rows)"]
    for row, spec in zip(rows, specs):
        name = spec["id"]
        if row.get("spec") != spec:
            errors.append(f"{name}: fixed case differs")
        if spec.get("optional") and row.get("status") in ("not-run", "skipped") and _nonempty(row.get("reason")):
            if (row.get("guard_ok") is not None or row.get("process_released") is not None
                    or row.get("request_receipt") is not None or row.get("response_receipt") is not None
                    or any(v is not None for v in _object(row.get("metrics")).values())
                    or _object(row.get("quality")).get("passed") is not None):
                errors.append(f"{name}: not-run case must not claim GPU evidence")
            continue
        if row.get("status") != "completed":
            errors.append(f"{name}: not completed")
        if row.get("cache_state") != spec["cache_state"]:
            errors.append(f"{name}: missing/incorrect cache state; cold and cached timings are separate")
        for key in ("request_receipt", "response_receipt"):
            if not _nonempty(row.get(key)):
                errors.append(f"{name}: missing {key}")
        if spec.get("history_from"):
            history = row.get("sent_history")
            if not (isinstance(history, list) and len(history) >= 2
                    and all(isinstance(turn, dict) and turn.get("role") in ("user", "assistant")
                            and _nonempty(turn.get("content")) for turn in history)
                    and any(turn["role"] == "assistant" for turn in history)
                    and any(turn["role"] == "user" and turn["content"] == CREATE for turn in history)):
                errors.append(f"{name}: missing exact previous user/assistant history")
        elif row.get("sent_history") != []:
            errors.append(f"{name}: unexpected history changes the fixed case")
        for key in ("guard_ok", "process_released"):
            if row.get(key) is not True:
                errors.append(f"{name}: {key} did not explicitly pass")
        quality = _object(row.get("quality"))
        if quality.get("passed") is not True or not _nonempty(quality.get("notes")):
            errors.append(f"{name}: missing/pending/failed manual quality judgment against every listed check")
        judgments = quality.get("checks")
        if not isinstance(judgments, list) or [j.get("criterion") if isinstance(j, dict) else None for j in judgments] != spec["quality_checks"]:
            errors.append(f"{name}: manual quality criteria differ or are missing")
        elif any(j.get("passed") is not True or not _nonempty(j.get("notes")) for j in judgments):
            errors.append(f"{name}: each manual quality criterion needs its own verdict and notes")
        metrics = _object(row.get("metrics"))
        for key in METRICS:
            if not _number(metrics.get(key)):
                errors.append(f"{name}: missing/invalid metric {key}")
        for key, limit_key in (("whole_card_peak_mib", "card_mib"), ("adapter_shared_rise_mib", "adapter_shared_rise_mib"),
                               ("process_shared_rise_mib", "process_shared_rise_mib")):
            if _number(metrics.get(key)) and metrics[key] > LIMITS[limit_key]:
                errors.append(f"{name}: {key} crossed the guard limit")
        for key, limit_key in (("minimum_free_ram_gib", "ram_floor_gib"),
                               ("minimum_commit_headroom_gib", "commit_floor_gib")):
            if _number(metrics.get(key)) and metrics[key] < LIMITS[limit_key]:
                errors.append(f"{name}: {key} crossed the guard floor")
        if _number(metrics.get("elapsed_seconds")) and metrics["elapsed_seconds"] > 600:
            errors.append(f"{name}: elapsed_seconds exceeded the request time budget")
    safety = _object(value.get("safety"))
    for key in (*SAFETY, "lookup_unavailable"):
        item = _object(value.get(key) if key == "lookup_unavailable" else safety.get(key))
        if item.get("passed") is not True or not _nonempty(item.get("evidence")):
            errors.append(f"{key}: missing/pending/failed recorded check")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="print future GPU protocol; perform no run or measurement")
    template = sub.add_parser("receipt", help="print a blank receipt, or write it to a new file")
    template.add_argument("--output", type=Path, help="create this new file; refuse an existing path")
    audit = sub.add_parser("check", help="audit a recorded receipt; never grant qualification")
    audit.add_argument("receipt", type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        print(json.dumps(plan(), indent=2, allow_nan=False))
        return 0
    if args.command == "receipt":
        encoded = json.dumps(receipt(), indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(encoded, end="")
            return 0
        try:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(encoded)
        except OSError as exc:
            print(json.dumps({"written": False, "problem": str(exc)}))
            return 1
        print(json.dumps({"written": True, "path": str(args.output), "receipt_complete": False}))
        return 0
    doc: Any = None
    try:
        doc = json.loads(args.receipt.read_text(encoding="utf-8-sig"))
        errors = check(doc)
    except (OSError, ValueError) as exc:
        errors = [str(exc)]
    recorded = _object(doc)
    rows = recorded.get("cases")
    cold = next((r for r in rows if isinstance(r, dict) and r.get("id") == "create-cold"), {}) if isinstance(rows, list) else {}
    print(json.dumps({"receipt_complete": not errors, "problems": errors,
                      "receipt_scope": "prompt-only with explicit optional-stage status",
                      "recorded_stages": {"missing_cache_download": cold.get("status", "unknown"),
                                          "image_integration": _object(recorded.get("integration")).get("status", "unknown")},
                      "qualification": "not-decided; inspect raw evidence and manual quality judgments"}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
