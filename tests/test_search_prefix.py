"""Free words match as prefixes; `-word` exclusions stay exact (GAL#25,
docs/maintainer/review-2026-11-12-todo.md)."""
from __future__ import annotations

import pytest

from backend import search
from tests.test_search_exclusion import ids, keep


@pytest.fixture
def rows(store):
    made = {
        "dragon": keep(store, "a red dragon over the sea", seed=1),
        "snap": keep(store, "snapdragon flowers in a vase", seed=2),
        "eyes": keep(store, "blue-eyes white knight", seed=3),
        "ops": keep(store, "AND or NOT near the gate", seed=4),
        "cafe": keep(store, "a café on the corner", seed=5),
        "stars": keep(store, "x*y stars (bright) ^up", seed=6),
        "drag": keep(store, "a drag race at night", seed=7),
    }
    return {name: record.id for name, record in made.items()}


def test_a_free_word_matches_the_start_of_a_word(store, rows):
    assert ids(store, "drag") == sorted([rows["dragon"], rows["drag"]])
    assert ids(store, "dra") == ids(store, "drag")
    # A prefix of a word, never its middle: the index's tokens, not a substring.
    assert rows["snap"] not in ids(store, "drag")
    assert ids(store, "DRAG") == ids(store, "drag")
    # Every word is a prefix and they still AND, in any order.
    assert ids(store, "sea dra") == [rows["dragon"]]
    assert ids(store, "nig dra") == [rows["drag"]]
    # A finished word still finds itself.
    assert ids(store, "dragon") == [rows["dragon"]]


def test_an_excluded_word_stays_exact(store, rows):
    # `-drag` leaves out the word drag, not every word it starts.
    assert ids(store, "dra -drag") == [rows["dragon"]]
    assert ids(store, "-dra") == ids(store, "")


@pytest.mark.parametrize("word,expected", [
    ("AND", "ops"), ("NOT", "ops"), ("NEA", "ops"), ("OR", "ops"),
    ("blue-ey", "eyes"), ("caf", "cafe"), ("x*", "stars"), ("(bri", "stars"), ("^u", "stars"),
])
def test_operator_words_and_special_characters_stay_literal(store, rows, word, expected):
    assert rows[expected] in ids(store, word)


def test_punctuation_alone_matches_nothing_and_raises_nothing(store, rows):
    assert ids(store, "...") == []
    assert ids(store, "( ^") == []


def test_the_quote_fallback_keeps_prefixes_and_exact_exclusions(store, rows):
    quoted = keep(store, 'a 26" dragon statue', seed=8)
    lion = keep(store, 'a 26" lion statue', seed=9)
    drag = keep(store, 'a 26" drag racer', seed=10)
    # The odd quote breaks MATCH; each word is then matched literally, the
    # quote included, but from where a word starts: a free word from its
    # first letters, never the middle of a word.
    assert ids(store, '26" drag') == sorted([quoted.id, drag.id])
    assert ids(store, '26" rag') == []
    assert ids(store, 'dra 26"') == sorted([quoted.id, drag.id])
    # An exclusion stays exact: `-drag` leaves out the drag racer, not the
    # dragon, and a part of a word leaves out nothing.
    assert ids(store, '26" -drag') == sorted([quoted.id, lion.id])
    assert ids(store, '26" -lio') == sorted([quoted.id, lion.id, drag.id])
    assert ids(store, '26" -lion') == sorted([quoted.id, drag.id])


@pytest.mark.parametrize("prompt,word,prefix,expected", [
    ("a red dragon", "drag", 1, 1),
    ("a red dragon", "drag", 0, 0),
    ("a red dragon", "dragon", 0, 1),
    ("A Red DRAGON", "dragon", 0, 1),
    ("a snapdragon", "drag", 1, 0),
    ("a snapdragon and a dragon", "drag", 1, 1),
    ("blue-eyes white", "blue-ey", 1, 1),
    ("blue-eyes white", "blue-eyes", 0, 1),
    ("blue white eyes", "blue-eyes", 0, 0),
    ('a 26" monitor', '26"', 0, 1),
    ('a 26" monitor', '6"', 0, 0),
    ("a 26 monitor", '26"', 0, 0),
    ('a 26"monitor', '26"', 0, 1),
    ("snake_case word", "case", 0, 1),
    ("anything", '"', 1, 0),
    (None, "x", 1, 0),
    ("anything", "", 1, 0),
])
def test_word_match_is_literal_and_word_bounded(prompt, word, prefix, expected):
    assert search.word_match(prompt, word, prefix) == expected
