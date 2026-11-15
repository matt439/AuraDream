"""Print an offline Turbo Refine session pack; check receipt completeness, never run it.

No backend, torch, counters, network or model imports. ``plan`` prints JSON;
``check FILE`` audits a manually completed receipt, not GPU truth or quality.
The private runner and owner-authorized card session remain separate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

CHECKPOINT = "circlestone-labs/Anima"
SEED = 4471
# Synthetic qualification prompt, not an author's example or recommendation.
PROMPT = "anime illustration of a woman in a blue coat beside a red bicycle, leafy park, daylight"
SETTINGS = {"prompt": PROMPT, "negative_prompt": "", "steps": 8, "cfg": 1.0,
            "mu": None, "batch_call_size": 1}
METRICS = (
    "elapsed_seconds", "allocated_peak_mib", "reserved_peak_mib", "whole_card_peak_mib",
    "adapter_shared_rise_mib", "process_shared_rise_mib", "process_ram_peak_gib",
    "minimum_free_ram_gib", "minimum_commit_headroom_gib", "resident_allocated_mib",
)
SESSION_FIELDS = (
    "git_revision", "utc_started", "checkpoint_revision", "loader_recipe", "gpu_name",
    "driver", "torch_version", "diffusers_version", "torchao_version", "guard_log",
    "limits_receipt", "preflight_receipt", "load_receipt", "final_residency_receipt",
)


def cases() -> list[dict[str, Any]]:
    """Ordered one-load protocol; optional enlargement never gates same-size evidence."""
    def case(name: str, strength: float | None, seed: int = SEED, **extra: Any) -> dict[str, Any]:
        return {"id": name, "width": 1024, "height": 1024, "seed": seed,
                "strength": strength, "source": "baseline" if strength is not None else None, **extra}
    return [
        case("baseline", None),
        case("strength-0125", .125), case("strength-025", .25), case("strength-04", .4),
        case("repeat-04", .4, compare_to="strength-04"),
        case("batch-first", .4, compare_to="strength-04", batch_index=0),
        case("batch-second", .4, SEED + 1, batch_index=1),
        case("solo-second", .4, SEED + 1, compare_to="batch-second"),
        case("cancel-04", .4, cancel_after_completed_steps=1),
        case("recovery-04", .4, compare_to="strength-04"),
        case("direct-after", None, compare_to="baseline"),
        case("upscale-125", .4, width=1280, height=1280, optional=True,
             requires="same-size passed, guard projection and measured headroom permit"),
    ]


def plan() -> dict[str, Any]:
    rows = cases()
    return {
        "schema": "anima-refine-session-v1", "mode": "offline-plan-not-a-run",
        "selection": {"checkpoint": CHECKPOINT, "variant": "Turbo v1.1", "quantization": "fp8",
                      "offload": "model", "platform": "Windows", "card": "RTX 5080 16 GB"},
        "settings": dict(SETTINGS),
        "constraints": [
            "Owner-authorized quiet card window; private runner only; public Refine stays held.",
            "One load; use existing card_guard limits and sampler, no copied or loosened limits.",
            "Set checkpoint-specific preflight RAM and busy-card limits before loading.",
            "Stop at first spill, missing guard coverage, OOM, headroom refusal or timeout; retain failed receipts.",
            "Batch members are consecutive one-image calls, seeded seed+i, then solo comparison.",
            "All Refine cases use the immutable baseline source, never a previous refined result.",
            "Guard can refuse the optional 1.25x step; no larger target or auto-retry.",
            "Aesthetic, Preview, 3070, WSL, alternative loaders and public UI are not qualified here.",
        ],
        "attempt_budgets_seconds": {"session": 1800, "load": 600, "pass": 120},
        "cases": rows,
        "receipt": {
            "schema": "anima-refine-receipt-v1", "selection": {"checkpoint": CHECKPOINT,
            "variant": "Turbo v1.1", "quantization": "fp8", "offload": "model", "platform": "Windows"},
            "settings": dict(SETTINGS),
            "session": {name: None for name in SESSION_FIELDS},
            "load_count": None, "stopped_reason": None,
            "cases": [{"id": row["id"], "status": "not-run", "reason": None,
                       "request": row, "metrics": {key: None for key in METRICS},
                       "completed_steps": None, "start_sigma": None, "pixel_sha256": None,
                       "png_sha256": None, "metadata_sha256": None, "source_sha256": None,
                       "guard_ok": None, "visual_review": None} for row in rows],
        },
    }


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _nonnegative(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def check(receipt: Any) -> list[str]:
    """Missing/inconsistent recorded evidence; never asserts that a session occurred."""
    errors: list[str] = []
    if not isinstance(receipt, dict):
        return ["receipt must be an object"]
    expected = plan()["receipt"]
    if receipt.get("schema") != expected["schema"]:
        errors.append("unknown receipt schema")
    if receipt.get("selection") != expected["selection"]:
        errors.append("selection differs from the bounded Windows fp8/model Turbo session")
    if receipt.get("settings") != expected["settings"]:
        errors.append("settings differ from the fixed prompt/step/CFG protocol")
    session = receipt.get("session")
    if not isinstance(session, dict):
        session = {}
    for name in SESSION_FIELDS:
        if not _nonempty(session.get(name)):
            errors.append(f"session: missing {name}")
    if type(receipt.get("load_count")) is not int or receipt["load_count"] != 1:
        errors.append("session must record exactly one load")
    if receipt.get("stopped_reason") is not None:
        errors.append("session stopped; retain evidence but do not qualify")
    rows = receipt.get("cases")
    if not isinstance(rows, list):
        return [*errors, "cases must be an ordered list"]
    specs = cases()
    if [r.get("id") if isinstance(r, dict) else None for r in rows] != [s["id"] for s in specs]:
        return [*errors, "case ids/order differ (missing, extra or duplicate rows)"]
    by_id = {r["id"]: r for r in rows}
    for row, spec in zip(rows, specs):
        name = spec["id"]
        if row.get("request") != spec:
            errors.append(f"{name}: request differs from the fixed case")
        status = row.get("status")
        if spec.get("optional") and status == "skipped" and _nonempty(row.get("reason")):
            continue
        cancelled = "cancel_after_completed_steps" in spec
        if status != ("cancelled" if cancelled else "completed"):
            errors.append(f"{name}: not completed as planned")
            continue
        if row.get("guard_ok") is not True:
            errors.append(f"{name}: guard did not explicitly pass")
        metrics = row.get("metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        for key in METRICS:
            value = metrics.get(key)
            if not _nonnegative(value):
                errors.append(f"{name}: missing/invalid metric {key}")
        steps = row.get("completed_steps")
        expected_steps = 1 if cancelled else ({.125: 1, .25: 2, .4: 4}.get(spec["strength"], 8))
        if type(steps) is not int or steps != expected_steps:
            errors.append(f"{name}: completed steps differ from the planned tail")
        if spec["strength"] is not None:
            sigma = row.get("start_sigma")
            expected_sigma = {.125: .3, .25: .5, .4: .75}[spec["strength"]]
            if not _nonnegative(sigma) or not math.isclose(sigma, expected_sigma, abs_tol=1e-6):
                errors.append(f"{name}: missing/invalid actual start sigma")
            baseline = by_id["baseline"].get("pixel_sha256")
            if not _digest(row.get("source_sha256")) or row["source_sha256"] != baseline:
                errors.append(f"{name}: source does not match immutable baseline pixels")
        if cancelled:
            if any(row.get(key) is not None for key in ("pixel_sha256", "png_sha256", "metadata_sha256")):
                errors.append(f"{name}: cancelled pass must not retain a decoded output")
            continue
        for key in ("pixel_sha256", "png_sha256", "metadata_sha256"):
            if not _digest(row.get(key)):
                errors.append(f"{name}: missing/invalid {key}")
        if not _nonempty(row.get("visual_review")):
            errors.append(f"{name}: missing composition/detail/artefact review")
        compare = spec.get("compare_to")
        if compare and row.get("pixel_sha256") != by_id[compare].get("pixel_sha256"):
            errors.append(f"{name}: pixels differ from {compare}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="print offline fixed cases and blank receipt; never load or render")
    audit = sub.add_parser("check", help="check recorded receipt completeness, not card qualification")
    audit.add_argument("receipt", type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        print(json.dumps(plan(), indent=2, allow_nan=False))
        return 0
    try:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8-sig"))
        errors = check(receipt)
    except (OSError, ValueError) as exc:
        errors = [str(exc)]
    print(json.dumps({"receipt_complete": not errors, "problems": errors,
                      "qualification": "not-decided; inspect source receipts and images manually"}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

