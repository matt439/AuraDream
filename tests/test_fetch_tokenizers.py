"""`tools/fetch_tokenizers.py`: retries, then fails only where HF_TOKEN says
every tokenizer should have come down. The Hub is never reached."""
from __future__ import annotations

import pytest

from tools import fetch_tokenizers


def _hub(monkeypatch, failures: dict[str, int]) -> list[str]:
    """A stand-in snapshot that fails each label its count of times."""
    calls: list[str] = []

    def snapshot(source, _local_only):
        calls.append(source.label)
        if failures.get(source.label, 0) > 0:
            failures[source.label] -= 1
            raise OSError(f"{source.label} unreachable\nsecond line")

    monkeypatch.setattr(fetch_tokenizers.pe, "_snapshot", snapshot)
    return calls


def _labels() -> list[str]:
    sources = {spec.source for model_id, _ in fetch_tokenizers.REAL
               for spec in fetch_tokenizers.pe.encoders_for(model_id) or ()}
    return sorted(source.label for source in sources)


@pytest.mark.parametrize("token", ["", "hf_x"])
def test_every_tokenizer_cached_passes(monkeypatch, capsys, token):
    monkeypatch.setenv("HF_TOKEN", token)
    calls = _hub(monkeypatch, {})
    assert fetch_tokenizers.main(sleep=lambda _: None) == 0
    assert calls == _labels()
    assert capsys.readouterr().out.splitlines()[-1] == f"{len(calls)} of {len(calls)} tokenizers cached"


def test_a_passing_retry_is_cached(monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    first = _labels()[0]
    waits: list[float] = []
    calls = _hub(monkeypatch, {first: fetch_tokenizers.ATTEMPTS - 1})
    assert fetch_tokenizers.main(sleep=waits.append) == 0
    assert calls.count(first) == fetch_tokenizers.ATTEMPTS
    assert waits == [fetch_tokenizers.RETRY_WAIT] * (fetch_tokenizers.ATTEMPTS - 1)
    assert "::" not in capsys.readouterr().out


def test_a_missing_tokenizer_fails_with_the_token(monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    first = _labels()[0]
    calls = _hub(monkeypatch, {first: fetch_tokenizers.ATTEMPTS})
    assert fetch_tokenizers.main(sleep=lambda _: None) == 1
    assert calls.count(first) == fetch_tokenizers.ATTEMPTS
    out = capsys.readouterr().out
    assert f"::error::{first}: OSError: {first} unreachable\n" in out
    assert "second line" not in out


def test_a_missing_tokenizer_only_warns_without_the_token(monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    first = _labels()[0]
    _hub(monkeypatch, {first: fetch_tokenizers.ATTEMPTS})
    assert fetch_tokenizers.main(sleep=lambda _: None) == 0
    assert f"::warning::{first}: OSError" in capsys.readouterr().out
