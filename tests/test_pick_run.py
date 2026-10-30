"""Pick this run (UI#37): gallery-select.js's pickRun in a Node VM.

The browser half is smoke-select's; this holds the decisions - one read at
the cap, the server's order beside existing picks, the cap's refusal, and the
late replies dropped - without a server.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "pick_run_fixture.mjs"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")


def run(scenario: str) -> dict:
    done = subprocess.run(["node", str(FIXTURE), str(ROOT), scenario],
                          capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads(done.stdout)


def test_adds_the_run_in_server_order_beside_existing_picks():
    out = run("plain")
    assert out["picks"] == [1, 10, 11, 12, 13, 14, 15]
    assert out["asked"] == ["/api/gallery?search=run%3Ar1&all_models=true&limit=1024"]
    assert out["said"] == ["Added 6 of the run's 6 images; 7 images selected."]


def test_cap_picks_only_the_room_in_refuse_over_caps_words():
    out = run("cap")
    assert out["picks"] == [1, 10, 11, 12]
    assert out["asked"][0].endswith("limit=4")
    assert out["said"] == ["4 images is the most one action takes; only 3 more were picked."]


def test_row_of_no_run_says_why_and_asks_nothing():
    out = run("none")
    assert out == {"picks": [1], "asked": [],
                   "said": ["That image predates runs being grouped, so it belongs to no run."]}


@pytest.mark.parametrize("scenario", ["cleared", "bin", "replaced"])
def test_reply_after_a_clear_a_replacement_or_the_bin_is_dropped(scenario):
    out = run(scenario)
    assert out["picks"] == ([2] if scenario == "replaced" else [])
    assert not any("of the run" in text for text in out["said"])


def test_an_id_forgotten_meanwhile_is_left_out():
    out = run("forgotten")
    assert out["picks"] == [1, 10, 11, 13, 14, 15]


def test_failure_says_so_and_picks_nothing():
    out = run("fails")
    assert out["picks"] == [1]
    assert out["said"] == ["Could not pick the run: fixture refusal"]
