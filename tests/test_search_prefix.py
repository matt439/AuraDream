"""Free words match as prefixes; `-word` exclusions stay exact (GAL#25,
docs/maintainer/review-2026-11-12-todo.md)."""
from __future__ import annotations

import pytest

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


def test_the_like_fallback_matches_prefixes_too(store, rows):
    quoted = keep(store, 'a 26" dragon statue', seed=8)
    keep(store, 'a 26" lion statue', seed=9)
    # The odd quote breaks MATCH; every word then becomes a LIKE substring,
    # which still finds a word from its first letters.
    assert ids(store, '26" drag') == [quoted.id]
    assert ids(store, '26" -lio') == [quoted.id]
    assert ids(store, 'dra 26"') == [quoted.id]
