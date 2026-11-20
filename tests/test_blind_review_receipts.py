"""Independent fixtures for selecting the recorded occurrence and authenticating its measurement."""
import hashlib
import json
from pathlib import Path

import pytest

from tools import blind_review_receipts as receipts, perf_cases


def cache_case(case_id):
    return {
        "kind": "image", "status": "completed", "seconds": 5.5, "pixel_sha256": "b" * 64,
        "seed": 4471,
        "case": {"id": case_id, "shape": "A", "prompt": "scene", "seed": 4471, "cached": True},
        "execution": {"compilation": None, "attention_backend": None, "step_cache": "fbc-005-v1"},
        "request_applied": {
            "execution": {"compilation": None, "attention_backend": None, "step_cache": "fbc-005-v1"},
            "prompt": perf_cases.PROMPTS["scene"], "width": 1024, "height": 1024,
            "steps": 28, "guidance_scale": 3.5, "num_images": 1,
        },
    }


def read_cache(images):
    suite = receipts._Suite("qualification.json", "cache-suite")
    arm = receipts._Arm("On", suite, "step_cache", "fbc-005-v1", "-on")
    return receipts._image(Path("artifacts"), arm, images, "A", "scene", 4471)


def test_first_case_cannot_be_substituted_with_same_seed_warm_repeat():
    repeat = cache_case("A-repeat-on")
    repeat["seconds"] = 3
    images = {"A-repeat-on": repeat}
    with pytest.raises(ValueError, match="Missing measured fixed case"):
        read_cache(images)
    images["A-first-on"] = cache_case("A-first-on")
    side, request = read_cache(images)
    assert side.path == Path("artifacts/cache-suite/A-first-on.png")
    assert side.seconds == 5.5 and side.method == "On"
    assert request["prompt"] == perf_cases.PROMPTS["scene"]


@pytest.mark.parametrize(("path", "replacement"), [
    (("seed",), 8823), (("case", "seed"), 8823), (("case", "cached"), False), (("execution",), None),
    (("request_applied", "execution"), None), (("request_applied", "prompt"), "different prompt"),
    (("request_applied", "width"), 512), (("status",), "cancelled"), (("pixel_sha256",), "invalid"),
])
def test_unmatched_or_unmeasured_cache_row_is_rejected(path, replacement):
    row = cache_case("A-first-on")
    obj = row
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = replacement
    with pytest.raises(ValueError):
        read_cache({"A-first-on": row})


def test_measurements_are_authenticated_against_the_tracked_suite_receipt(tmp_path, monkeypatch):
    suite = receipts._Suite("qualification.json", "cache-suite")
    directory = tmp_path / suite.directory
    directory.mkdir()
    row = cache_case("A-first-on")
    measurements = directory / "measurements.jsonl"
    data = (json.dumps(row) + "\n").encode()
    measurements.write_bytes(data)
    (tmp_path / suite.receipt).write_text(json.dumps({
        "runtime": {"loader": {"checkpoint": perf_cases.CHECKPOINT, "quantization": "fp8", "offload": "model"}},
        "sessions": [{
            "session": ".artifacts/gpu-20261118/cache-suite",
            "raw_artifact_sha256": {"measurements.jsonl": hashlib.sha256(data).hexdigest()},
        }],
    }), encoding="utf-8")
    monkeypatch.setattr(receipts, "_RECEIPTS", tmp_path)
    assert suite.images(tmp_path)["A-first-on"] == row
    measurements.write_bytes(data.replace(b"5.5", b"9.5"))
    with pytest.raises(ValueError, match="Measurement SHA256 differs"):
        suite.images(tmp_path)
