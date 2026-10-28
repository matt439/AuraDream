"""Scientific scalar filters emitted by facts stay exact across SQL and HTTP."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest
from PIL import Image

from backend import main, search
from backend.schemas import GenerateRequest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(("key", "column"), [
    ("cfg", "guidance_scale"), ("mu", "mu"),
    ("tcfg", search.TRUE_CFG_SQL), ("strength", "strength"),
])
@pytest.mark.parametrize(("raw", "expected"), [
    ("1e-7", 1e-7), ("+1e-7", 1e-7), ("1E-07", 1e-7), ("1e-0", 1.0),
])
def test_whole_scientific_float_is_an_exact_parameter(key, column, raw, expected):
    assert search.filter_clause(key, raw) == (f"{column} = ?", [expected])


@pytest.mark.parametrize(("key", "raw", "clause", "parameters"), [
    ("cfg", "0.0000001", "guidance_scale = ?", [1e-7]),
    ("mu", "0.0000001", "mu = ?", [1e-7]),
    ("cfg", ">=1e-7", "guidance_scale >= ?", [1e-7]),
    ("mu", "<1e-7", "mu < ?", [1e-7]),
    ("cfg", "-1e-7", "guidance_scale = ?", [-1e-7]),
    ("cfg", "1e+2", "guidance_scale = ?", [100.0]),
])
def test_existing_decimal_comparator_and_signed_scalar_controls_are_unchanged(key, raw, clause, parameters):
    assert search.filter_clause(key, raw) == (clause, parameters)


@pytest.mark.parametrize(("key", "raw", "clause", "parameters"), [
    ("cfg", "0.5-1", "guidance_scale BETWEEN ? AND ?", [0.5, 1.0]),
    ("mu", "0-1.15", "mu BETWEEN ? AND ?", [0.0, 1.15]),
    ("cfg", "1e+2-2e+2", "guidance_scale BETWEEN ? AND ?", [100.0, 200.0]),
    ("cfg", "0-1e-7", "guidance_scale BETWEEN ? AND ?", [0.0, 1e-7]),
    ("strength", "0.1-0.5", "strength BETWEEN ? AND ?", [0.1, 0.5]),
])
def test_existing_dash_range_endpoints_and_parameters_are_unchanged(key, raw, clause, parameters):
    assert search.filter_clause(key, raw) == (clause, parameters)


@pytest.mark.parametrize(("key", "raw", "message"), [
    ("cfg", "1e-bad", "cfg: '1e' is not a number"),
    ("cfg", "0-1e-bad", "cfg: '1e-bad' is not a number"),
    ("cfg", "1e-7-1e-6", "cfg: '1e' is not a number"),
    ("cfg", "-1-2", "cfg: '-1-2' is not a number"),
    ("steps", "1e-7", "steps: '1e' is not a number"),
    ("seed", "-1e-7", "seed: '-1e-7' is not a number"),
    ("seed", "9223372036854775808", "seed: '9223372036854775808' is out of range"),
])
def test_malformed_range_operands_and_integer_refusals_keep_their_existing_messages(key, raw, message):
    with pytest.raises(ValueError) as caught:
        search.filter_clause(key, raw)
    assert str(caught.value) == message


@pytest.mark.parametrize(("raw", "expected"), [
    ("nan", float("nan")), ("inf", float("inf")), ("-inf", float("-inf")),
])
def test_existing_float_conversion_policy_is_not_broadened_or_narrowed(raw, expected):
    # Preserve parser conversion only. This does not claim that such a value
    # is valid source data or establish SQLite matching behavior for NaN.
    clause, parameters = search.filter_clause("cfg", raw)
    assert clause == "guidance_scale = ?" and len(parameters) == 1
    assert math.isnan(parameters[0]) if math.isnan(expected) else parameters[0] == expected


def test_null_zero_and_integer_terms_keep_their_existing_meanings():
    assert search.strict_scope("mu:none cfg:0 seed:2-4") == (
        ["mu IS NULL", "guidance_scale = ?", "seed BETWEEN ? AND ?"], [0.0, 2, 4],
    )
    assert search.filter_clause("mu", "0") == ("mu = ?", [0.0])
    assert search.filter_clause("id", "-1") == ("id = ?", [-1])


def test_scientific_terms_keep_strict_and_free_text_composition():
    clauses = ["guidance_scale = ?", "mu = ?"]
    assert search.split_search("harbor CFG:1e-7 mu:1e-7") == (clauses, [1e-7, 1e-7], "harbor")
    assert search.strict_scope("CFG:1e-7 mu:1e-7") == (clauses, [1e-7, 1e-7])
    with pytest.raises(ValueError, match="free text"):
        search.strict_scope("harbor cfg:1e-7")


def _keep(store, seed, guidance, mu):
    request = GenerateRequest(
        prompt="harbor numeric-search fixture", width=256, height=256, steps=8,
        guidance_scale=guidance, mu=mu,
    ).applied(negative_prompt_supported=False, cfg_threshold=1.0, mu_supported=True)
    assert request.guidance_scale == guidance and request.mu == mu
    with Image.new("RGB", (256, 256), (seed, 40, 90)) as image:
        return store.save_image(
            image, request, seed=seed, duration_ms=1,
            model_id=store.settings.model_id, run_id=f"numeric-{seed}",
        )


@pytest.fixture
def scalar_rows(store):
    return {
        "tiny": _keep(store, 11, 1e-7, 1e-7),
        "neighbor": _keep(store, 12, 1e-6, 1e-6),
        "zero": _keep(store, 13, 0.0, 0.0),
        "none": _keep(store, 14, 2.5, None),
    }


@pytest.mark.parametrize("key", ["cfg", "mu"])
def test_scientific_exact_search_selects_the_same_scratch_row_as_decimal(store, scalar_rows, key):
    expected = scalar_rows["tiny"].id
    # This equivalent exact spelling is an original-source positive control.
    decimal, total, _others = store.list_images(search=f"{key}:0.0000001")
    assert total == 1 and [record.id for record in decimal] == [expected]
    scientific, total, _others = store.list_images(search=f"{key}:1e-7")
    assert total == 1 and [record.id for record in scientific] == [expected]
    compared, total, _others = store.list_images(search=f"{key}:>=1e-7 {key}:<=1e-7")
    assert total == 1 and [record.id for record in compared] == [expected]


@pytest.fixture
def cold_reads_only(client, monkeypatch):
    attempts = []

    def forbidden(*_args, **_kwargs):
        attempts.append("model-or-queue")
        pytest.fail("a numeric search entered model loading, generation or queue submission")

    monkeypatch.setattr(main, "ensure_loaded", forbidden)
    monkeypatch.setattr(main.backend, "load", forbidden)
    monkeypatch.setattr(main.backend, "generate", forbidden)
    monkeypatch.setattr(main.jobs, "wait_for_gate", forbidden)
    monkeypatch.setattr(main.jobs, "submit", forbidden)
    yield attempts
    assert attempts == []
    assert client.get("/api/jobs").json() == []


def _rows(store):
    with store._session() as connection:
        return [tuple(row) for row in connection.execute("SELECT * FROM images ORDER BY id").fetchall()]


@pytest.mark.parametrize("key", ["cfg", "mu"])
def test_gallery_scientific_exact_read_returns_200_and_the_matching_row(
    client, store, scalar_rows, cold_reads_only, key,
):
    before = _rows(store)
    expected = scalar_rows["tiny"].id
    base = {"all_models": True}
    # Both controls already succeed against the original parser.
    for query in (f"{key}:0.0000001", f"{key}:>=1e-7 {key}:<=1e-7"):
        control = client.get("/api/gallery", params={**base, "search": query})
        assert control.status_code == 200, control.text
        assert control.json()["total"] == 1 and [row["id"] for row in control.json()["items"]] == [expected]
    response = client.get("/api/gallery", params={**base, "search": f"{key}:1e-7"})
    assert response.status_code == 200, response.text
    assert response.json()["total"] == 1 and [row["id"] for row in response.json()["items"]] == [expected]
    # Strict preview shares the parser; this is a GET only, without deletion.
    preview = client.get("/api/gallery/unstarred", params={**base, "search": f"{key}:1e-7"})
    assert preview.status_code == 200, preview.text
    assert preview.json()["records"] == 1 and preview.json()["unstarred"] == 1
    assert _rows(store) == before and cold_reads_only == []


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")
def test_actual_sampling_fact_terms_find_their_serialized_scratch_record(store, scalar_rows):
    record = scalar_rows["tiny"]
    script = r'''
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({ record: JSON.parse(process.argv[2]) });
// The real classic script only declares helpers. No DOM, page boot, network,
// image generation or browser is supplied by this isolated function probe.
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const terms = vm.runInContext('factSearchTerms(record).Sampling', context);
console.log(JSON.stringify(terms));
'''
    done = subprocess.run(
        ["node", "-e", script, str(ROOT / "web/gallery-search.js"), json.dumps(record.model_dump(mode="json"))],
        capture_output=True, text=True, timeout=15,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    terms = json.loads(done.stdout)
    assert terms == ["steps:8", "cfg:1e-7", "mu:1e-7"]
    found, total, _others = store.list_images(search=" ".join(terms))
    assert total == 1 and [item.id for item in found] == [record.id]
