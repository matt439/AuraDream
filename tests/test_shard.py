"""`pytest --shard K/N` (tests/shard.py): CI's Windows legs each run one part."""
from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from tests import shard

ROOT = Path(__file__).resolve().parent.parent
NODES = [f"tests/test_x.py::test_{n}[{p}]" for n in range(400) for p in ("a", "b")]


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_every_test_lands_in_one_part_and_the_parts_are_near_even(count):
    parts = Counter(shard.part(node, count) for node in NODES)
    assert set(parts) == set(range(1, count + 1))
    assert sum(parts.values()) == len(NODES)
    assert max(parts.values()) - min(parts.values()) <= len(NODES) // count // 4


def test_a_part_is_the_same_in_every_process():
    # Never hash(): its salt differs per process, and per xdist worker.
    code = "from tests import shard; print([shard.part(n, 3) for n in %r])" % NODES[:50]
    answers = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              check=True, cwd=ROOT).stdout for _ in range(2)}
    assert answers == {str([shard.part(n, 3) for n in NODES[:50]]) + "\n"}


@pytest.mark.parametrize("text", ["0/3", "4/3", "3", "x/3", "1/0"])
def test_a_bad_shard_is_refused(text):
    with pytest.raises(ValueError):
        shard.parse(text)


def test_the_option_runs_only_its_part(collected):
    together = set()
    for index in (1, 2, 3):
        chosen = collected(f"{index}/3")
        assert chosen
        assert not together & chosen
        together |= chosen
    assert together == collected("1/1")


@pytest.fixture
def collected():
    """The ids `pytest --collect-only --shard K/N` keeps of this very file."""
    def run(value: str) -> set[str]:
        answer = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
             "--shard", value, "tests/test_shard.py"],
            capture_output=True, text=True, check=True, cwd=ROOT,
        )
        return {line for line in answer.stdout.splitlines() if "::" in line}
    return run
