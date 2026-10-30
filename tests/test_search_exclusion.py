"""Search exclusion: `-word` drops what the index matches, `-key:value` is the
complement of `key:value` (docs/plan-search-exclusion.md, GAL#21)."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest
from PIL import Image

from backend import clear, gallery_delete, gallery_reports, saved_searches, search
from backend.schemas import GenerateRequest
from backend.schemas_saved_searches import SavedSearchFields

ROOT = Path(__file__).resolve().parent.parent


def keep(store, prompt="a fox in the snow", *, seed=1, model_id="krea/Krea-2-Turbo", run_id=None, **fields):
    request = {"prompt": prompt, "width": 256, "height": 256, "steps": 8, "guidance_scale": 0.0, **fields}
    return store.save_image(
        Image.new("RGB", (request["width"], request["height"])), GenerateRequest(**request),
        seed=seed, duration_ms=1, model_id=model_id, job_id=run_id, run_id=run_id,
    )


def ids(store, query, **options):
    return sorted(r.id for r in store.list_images(search=query, limit=500, **options)[0])


@pytest.fixture
def mixed(store):
    """Rows that differ in each value an exclusion is checked on."""
    rows = {
        "blonde": keep(store, "a blonde girl reading", seed=1, steps=8, guidance_scale=0.5, mu=0.5,
                       lora="ink.safetensors", run_id="night"),
        "fox": keep(store, "a red fox in the snow", seed=2, steps=12, guidance_scale=1.5, negative_prompt="blurry",
                    width=256, height=384),
        "both": keep(store, "a blonde fox, ink style", seed=3, steps=8, guidance_scale=1.5, mu=1.0,
                     lora="other.safetensors", width=384, height=256),
        "plain": keep(store, "a quiet harbor", seed=4, steps=20, guidance_scale=0.0, run_id="night"),
    }
    with store._session() as conn:
        conn.execute("UPDATE images SET mask = ? WHERE id = ?", ("ab" * 32, rows["both"].id))
        conn.execute("UPDATE images SET created_at = ? WHERE id = ?",
                     ("2020-01-02T12:00:00+00:00", rows["plain"].id))
    return {name: record.id for name, record in rows.items()}


def test_an_excluded_word_drops_what_the_index_matches_and_keeps_the_rest(store, mixed):
    assert ids(store, "-blonde") == sorted([mixed["fox"], mixed["plain"]])
    # The tokenizer folds case, so the exclusion does as the positive word does.
    assert ids(store, "-BLONDE") == ids(store, "-blonde")
    assert ids(store, "fox -blonde") == [mixed["fox"]]
    assert ids(store, "-blonde -harbor") == [mixed["fox"]]
    assert ids(store, "a -fox") == sorted([mixed["blonde"], mixed["plain"]])
    # Before GAL#21 `-blonde` was the phrase "blonde": the images meant to go.
    assert set(ids(store, "-blonde")).isdisjoint(ids(store, "blonde"))


@pytest.mark.parametrize("term", [
    "seed:2", "steps:8", "steps:>=12", "cfg:1.5", "cfg:0-1", "mu:0.5", "tcfg:1.5", "tcfg:none",
    "lora:ink", "lora:none", "run:night", "date:2020-01-02", "date:2020-01-01..2020-01-03",
    "size:256x256", "size:portrait", "size:landscape", "size:square", "size:>0.07mp",
    "inpaint:yes", "inpaint:no", "mask:abab", "mu:none", "id:1-2",
])
def test_an_exclusion_and_its_term_count_every_row_exactly_once(store, mixed, term):
    # NULL rows - no LoRA, no mu, no true CFG, no mask - did not match the
    # term, so the exclusion keeps them.
    everything = ids(store, "")
    positive, negative = ids(store, term), ids(store, f"-{term}")
    assert set(positive).isdisjoint(negative)
    assert sorted(positive + negative) == everything


def test_null_clause_rows_stay_in_an_exclusion(store, mixed):
    assert ids(store, "-lora:ink") == sorted([mixed["fox"], mixed["both"], mixed["plain"]])
    assert ids(store, "-mu:0.5") == sorted([mixed["fox"], mixed["both"], mixed["plain"]])
    # "both" ran at CFG 1.5 with no negative prompt: no true CFG at all.
    assert ids(store, "-tcfg:1.5") == sorted([mixed["blonde"], mixed["both"], mixed["plain"]])


def test_exclusions_compose_with_terms_and_contradictions_match_nothing(store, mixed):
    assert ids(store, "seed:>1 -seed:3 fox -blonde") == [mixed["fox"]]
    assert ids(store, "seed:1 -seed:1") == []
    assert ids(store, "-run:night -inpaint:yes") == [mixed["fox"]]


def test_unchanged_spellings_stay_what_they_were():
    for token in ("-", "--ar", '"-blonde"'):
        assert search.parse_search(token) == ([], [], [token], [])
    assert search.split_search("cfg:0-1 seed:-5") == (
        ["guidance_scale BETWEEN ? AND ?", "seed = ?"], [0.0, 1.0, -5], "",
    )
    assert search.parse_search("-seed:-5").clauses == ["COALESCE((seed = ?), 0) = 0"]
    # An unknown key or a missing value is an excluded word, not a filter.
    assert search.parse_search("-portrait: -seed:") == ([], [], [], ["portrait:", "seed:"])


def test_a_bad_excluded_value_is_refused_like_the_term(store, client):
    with pytest.raises(ValueError, match="not a number"):
        store.list_images(search="-seed:abc")
    refused = client.get("/api/gallery", params={"search": "-seed:abc"})
    assert refused.status_code == 422 and "not a number" in refused.json()["detail"]


def test_the_fallback_moves_both_text_predicates_to_like_together(store):
    live = keep(store, 'a 26" monitor on a desk', seed=1)
    keep(store, 'a 26" monitor in a shop', seed=2)
    binned = keep(store, 'a 26" monitor on a stand', seed=3)
    other = keep(store, 'a 26" monitor on a shelf', seed=4, model_id="other/model")
    starred = keep(store, 'a 26" monitor on a desk, starred', seed=5)
    store.set_favorite(starred.id, True)
    store.bin_images([binned.id])

    # `26"` breaks MATCH; the excluded word then goes to NOT LIKE beside it.
    items, total, others = store.list_images(search='26" -shop', model_id="krea/Krea-2-Turbo")
    assert (sorted(r.id for r in items), total, others) == (sorted([live.id, starred.id]), 2, 1)
    # An excluded word with the odd quote moves the positive word with it.
    keep(store, 'a 27" monitor', seed=6)
    assert ids(store, 'monitor -26"') == [r.id for r in store.list_images(search="27")[0]]
    # Favorites, the bin and the scope still apply after the swap.
    assert ids(store, '26" -shop', favorites_only=True) == [starred.id]
    assert ids(store, '26" -shelf', binned=True) == [binned.id]
    assert ids(store, '-shop 26"', model_id="other/model") == [other.id]


def test_favorites_bin_and_checkpoint_scope_hold_under_exclusions(store):
    a = keep(store, "a blonde girl", seed=1)
    b = keep(store, "a red fox", seed=2)
    c = keep(store, "a red fox elsewhere", seed=3, model_id="other/model")
    d = keep(store, "a binned fox", seed=4)
    store.set_favorite(b.id, True)
    store.bin_images([d.id])
    items, total, others = store.list_images(search="-blonde", model_id="krea/Krea-2-Turbo")
    assert ([r.id for r in items], total, others) == ([b.id], 1, 1)
    assert ids(store, "-seed:2", favorites_only=True) == []
    assert ids(store, "-blonde", binned=True) == [d.id]
    assert ids(store, "-seed:4") == sorted([a.id, b.id, c.id])


def test_an_old_saved_query_that_is_now_invalid_still_reads(tmp_path):
    database = tmp_path / "gallery.db"
    path = saved_searches.path_for(database)
    old = {"id": "a" * 32, "revision": 1, "name": "Old", "query": "-seed:abc", "favorites": False,
           "all_models": False}
    path.write_text(json.dumps({"version": 1, "items": [old]}), encoding="utf-8")
    assert [item.query for item in saved_searches.read(database)] == ["-seed:abc"]
    # Writes still check the grammar, and another write keeps the old text.
    with pytest.raises(ValueError, match="invalid filter value"):
        SavedSearchFields(name="New", query="-seed:abc", favorites=False, all_models=False)
    fields = SavedSearchFields(name="New", query="fox -blonde -seed:1", favorites=False, all_models=False)
    items = saved_searches.change(database, fields)
    assert [item.query for item in items] == ["-seed:abc", "fox -blonde -seed:1"]
    assert json.loads(path.read_text(encoding="utf-8"))["items"][0]["query"] == "-seed:abc"


def test_the_saved_list_route_serves_an_old_query_that_is_now_invalid(client, store):
    old = {"id": "b" * 32, "revision": 1, "name": "Old", "query": "-seed:abc", "favorites": False,
           "all_models": False}
    saved_searches.path_for(store.settings.db_path).write_text(json.dumps({"version": 1, "items": [old]}), encoding="utf-8")
    listed = client.get("/api/gallery/saved-searches")
    assert listed.status_code == 200 and [item["query"] for item in listed.json()["items"]] == ["-seed:abc"]
    # Its Apply is the gallery's own refusal, in the grammar's words.
    refused = client.get("/api/gallery", params={"search": "-seed:abc"})
    assert refused.status_code == 422 and "not a number" in refused.json()["detail"]


def test_strict_scopes_take_excluded_terms_and_refuse_excluded_words():
    assert search.strict_scope("-seed:1 run:x") == (
        ["COALESCE((seed = ?), 0) = 0", r"run_id LIKE ? ESCAPE '\'"], [1, "%x%"],
    )
    with pytest.raises(ValueError, match=r"'-blonde' is free text; strict scopes take key:value terms and their exclusions"):
        search.strict_scope("run:x -blonde")
    with pytest.raises(ValueError, match="key:value terms only; 'fox' is free text"):
        search.strict_scope("fox -blonde")


def test_a_strict_preview_of_an_exclusion_and_its_term_cover_the_scope(store):
    for seed, run in enumerate(["x", "x", "y", None], start=1):
        keep(store, seed=seed, run_id=run)
    keep(store, seed=9, run_id="x", model_id="other/model")
    scope = "krea/Krea-2-Turbo"
    whole = gallery_delete.matching_preview("seed:>0", model_id=scope)["records"]
    plus = gallery_delete.matching_preview("run:x", model_id=scope)["records"]
    minus = gallery_delete.matching_preview("-run:x", model_id=scope)["records"]
    assert (plus, minus, whole) == (2, 2, 4)


def test_a_delete_with_an_exclusion_is_still_pinned_to_its_count(store):
    for seed in (1, 2, 3):
        keep(store, seed=seed)
    with pytest.raises(gallery_delete.SelectionChanged):
        gallery_delete.delete_matching("-seed:1", expected=1)
    assert store.list_images()[1] == 3
    done = gallery_delete.delete_matching("-seed:1", expected=2, to_bin=True)
    assert done["records"] == 2 and ids(store, "") == ids(store, "seed:1")


def test_the_unstarred_route_takes_an_exclusion_and_refuses_an_excluded_word(client, store):
    from backend.config import settings

    for seed in (1, 2, 3):
        keep(store, seed=seed, model_id=settings.model_id)
    assert client.get("/api/gallery/unstarred", params={"search": "-seed:1"}).json()["unstarred"] == 2
    refused = client.get("/api/gallery/unstarred", params={"search": "-fox"})
    assert refused.status_code == 422
    assert "their exclusions or an empty search" in refused.json()["detail"]
    assert "'-fox' is free text" in refused.json()["detail"]


def test_keep_rates_under_an_exclusion_leave_the_bin_and_other_models_out(store):
    plain = keep(store, seed=1)
    keep(store, seed=2, lora="ink.safetensors")
    # Both carry a LoRA, so `-lora:none` alone would keep them: what leaves
    # them out is the bin and the model term.
    binned = keep(store, seed=3, lora="ink.safetensors")
    keep(store, seed=4, lora="ink.safetensors", model_id="other/model")
    store.set_favorite(plain.id, True)
    store.bin_images([binned.id])
    rows = gallery_reports.keep_rates("-lora:none model:krea", by=("lora",))
    assert [(row["lora"], row["made"], row["starred"]) for row in rows] == [("ink.safetensors", 1, 0)]
    rows = gallery_reports.keep_rates("-lora:none", by=("lora",))
    assert [(row["lora"], row["made"], row["starred"]) for row in rows] == [("ink.safetensors", 2, 0)]
    with pytest.raises(ValueError, match="free text"):
        gallery_reports.keep_rates("-blonde")


def test_clear_where_previews_an_exclusion_and_refuses_unconfirmed(store, capsys, monkeypatch):
    monkeypatch.setattr(clear, "settings", store.settings)
    for seed in (1, 2, 3):
        keep(store, f"prompt {seed}", seed=seed)
    monkeypatch.setattr(clear, "_confirm", lambda _yes: False)
    assert clear.main(["--where=-seed:1"]) == 1
    out = capsys.readouterr().out
    assert "2 records match" in out and "Would delete 2 record(s)" in out
    assert store.list_images()[1] == 3
    # argparse reads a leading `-` as an option, so an exclusion first is
    # written `--where=...`; an excluded word is refused, confirmed or not.
    assert clear.main(["--where=-blonde", "--yes"]) == 2
    assert "'-blonde' is free text" in capsys.readouterr().err
    assert store.list_images()[1] == 3


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")
def test_the_page_reads_exclusions_as_the_server_does():
    script = r'''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({ assert });
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
vm.runInContext(`
  assert.equal(searchKey('-seed:1'), 'seed');
  assert.equal(searchKey('-SEED:1'), 'seed');
  assert.equal(searchKey('-blonde'), null);
  assert.equal(searchKey('--seed:1'), null);
  assert.equal(searchExclusion('-blonde'), 'blonde');
  assert.equal(searchExclusion('-'), null);
  assert.equal(searchExclusion('--ar'), null);
  assert.equal(mergeSearch('-seed:1 fox', ['seed:5']), 'seed:5 fox');
  assert.equal(mergeSearch('fox seed:2 -seed:1', ['seed:5']), 'fox seed:5');
  assert.equal(mergeSearch('cat -Fox run:x', ['fox']), 'cat fox run:x');
  assert.equal(mergeSearch('fox -fox', ['FOX']), 'fox');
  assert.equal(mergeSearch('-blonde', ['fox']), '-blonde fox');
  assert.equal(mergePromptSearch('old -blonde -seed:1 seed:3 words', ['moon']),
    '-blonde -seed:1 seed:3 moon');
  assert.equal(mergePromptSearch('-- - x', ['moon']), 'moon');
  assert.deepEqual(promptSearchWords('a -style cat --ar 2:3 -seed:4 -'), ['a', 'style', 'cat', 'ar', '2:3']);
  assert.ok(promptSearchWords('a -style cat').every((w) => !w.includes('-')));
`, context);
'''
    done = subprocess.run(
        ["node", "-e", script, str(ROOT / "web/gallery-search.js")],
        capture_output=True, text=True, timeout=15,
    )
    assert done.returncode == 0, done.stdout + done.stderr
