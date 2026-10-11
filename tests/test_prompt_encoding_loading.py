"""The tokenizer load lifecycle: failure, the cache and the lock.

Load and count errors become per-encoder details, a failed load is retried
and a good one kept until unload(), and the load source is a converted tree,
then the Hub cache, then the Hub. A stub load that never returns stands for a
stalled first-use fetch: what it may hold up - nothing but its own count -
is checked here, and through the app's Unload and Check routes in
tests/test_api_analysis_lifecycle.py.
"""
from __future__ import annotations

import fnmatch
import threading
import time
from pathlib import Path

import pytest

from backend import prompt_encoding as pe
from backend.backends import anima_loading, newbie_loading
from tests import fakes
from tests.prompt_encoding_helpers import (  # noqa: F401  (fixtures by name)
    FAILURES,
    QWEN_ADDED,
    FakeTokenizer,
    _kind,
    by,
    clip,
    fresh_cache,
    rules,
    tekken,
    tokenizers,
    words,
)


def test_anima_variants_share_tokenizers_but_read_their_own_converted_tree(monkeypatch, tmp_path):
    import huggingface_hub

    turbo_specs = pe.encoders_for(anima_loading.MODEL_ID)
    aesthetic_specs = pe.encoders_for(anima_loading.AESTHETIC_MODEL_ID)
    for turbo, aesthetic in zip(turbo_specs, aesthetic_specs):
        assert (aesthetic.source.repo, aesthetic.source.revision, aesthetic.source.subfolder) == (
            turbo.source.repo, turbo.source.revision, turbo.source.subfolder,
        )
        assert aesthetic.source.tree == anima_loading.AESTHETIC_MODEL_ID
        assert turbo.source.tree == anima_loading.MODEL_ID
        assert aesthetic.max_length == turbo.max_length == pe.ANIMA_SEQUENCE
    roots = {}
    for model, specs in (
        (anima_loading.MODEL_ID, turbo_specs),
        (anima_loading.AESTHETIC_MODEL_ID, aesthetic_specs),
    ):
        root = tmp_path / ("aesthetic" if "@" in model else "turbo")
        roots[model] = str(root)
        for spec in specs:
            (root / spec.source.tree_subfolder).mkdir(parents=True)
    monkeypatch.setattr(pe, "_converted_tree", roots.get)
    monkeypatch.setattr(pe, "_from_pretrained", lambda directory: directory)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("cached tree ignored"))
    for specs in (turbo_specs, aesthetic_specs):
        for spec in specs:
            assert pe._load_tokenizer(spec.source) == Path(roots[spec.source.tree]) / spec.source.tree_subfolder


def test_a_tokenizer_that_cannot_load_leaves_a_detail_not_an_exception(tokenizers):
    gated = RuntimeError("401 Client Error: gated repo\nmore")
    tokenizers.update(tokenizer=clip(), tokenizer_2=gated)
    reports, findings, detail = pe.measure("black-forest-labs/FLUX.1-dev", words(80), "blurry")
    assert [(r.name, r.field) for r in reports] == [
        ("CLIP-L", "prompt"), ("CLIP-L", "negative_prompt"),
    ]
    assert detail == "T5-XXL unchecked: RuntimeError: 401 Client Error: gated repo"
    # The pooled cut stands without claiming T5 reads on: T5 was not read.
    (cut,) = rules(findings, "encoder.cut")
    assert "reads on" not in cut.message
    # One failed load per analysis, not one per field.
    assert [s.subfolder for s in tokenizers["loads"]].count("tokenizer_2") == 1


@pytest.mark.parametrize("error", FAILURES, ids=_kind)
def test_any_error_a_load_raises_leaves_its_encoder_unchecked(tokenizers, error):
    tokenizers.update(tokenizer=clip(), tokenizer_2=error)
    reports, _, detail = pe.measure("black-forest-labs/FLUX.1-dev", "a cat", "blurry")
    assert [(r.name, r.field) for r in reports] == [
        ("CLIP-L", "prompt"), ("CLIP-L", "negative_prompt"),
    ]
    assert detail.startswith(f"T5-XXL unchecked: {_kind(error)}")


def test_hub_errors_are_said_in_words(tokenizers):
    from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError

    gated = GatedRepoError.__new__(GatedRepoError)
    tokenizers.update(tokenizer=clip(), tokenizer_2=gated)
    detail = pe.measure("black-forest-labs/FLUX.1-dev", "a cat", None)[2]
    assert detail == (
        "T5-XXL unchecked: the tokenizer's repo is gated - accept its terms on "
        "huggingface.co and log in"
    )
    pe.unload()
    tokenizers.update(tokenizer=LocalEntryNotFoundError("Cannot find the requested files"))
    detail = pe.measure("baidu/ERNIE-Image-Turbo", "a cat", None)[2]
    assert detail == (
        "Ministral-3B unchecked: the tokenizer is not cached and the Hub cannot be reached"
    )


@pytest.mark.parametrize(
    "error", [NotImplementedError("return_offset_mapping is not available"), *FAILURES], ids=_kind
)
def test_a_tokenizer_that_breaks_while_counting_is_unchecked_too(tokenizers, error):
    class Broken(FakeTokenizer):
        def __call__(self, *args, **kwargs):
            raise error

    tokenizers.update(tokenizer=Broken(max_length=2048, special=1))
    reports, _, detail = pe.measure("baidu/ERNIE-Image-Turbo", "a cat", None)
    assert reports == [] and detail.startswith(f"Ministral-3B unchecked: {_kind(error)}")

    # A chat template that breaks is the same: Z-Image renders one per field.
    class Unrendered(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            raise error

    tokenizers.update(tokenizer=Unrendered(added=QWEN_ADDED))
    reports, _, detail = pe.measure("Tongyi-MAI/Z-Image-Turbo", "a cat", None)
    assert reports == [] and detail.startswith(f"Qwen3-4B unchecked: {_kind(error)}")


def test_a_failed_load_is_retried_and_a_good_one_is_kept_until_unload(tokenizers):
    ernie = "baidu/ERNIE-Image-Turbo"
    tokenizers.update(tokenizer=OSError("offline"))
    assert pe.measure(ernie, "a cat", None)[2] == "Ministral-3B unchecked: OSError: offline"
    tokenizers.update(tokenizer=tekken())
    assert pe.measure(ernie, "a cat", None)[2] is None
    assert pe.measure(ernie, "a dog", None)[2] is None
    assert len(tokenizers["loads"]) == 2
    pe.unload()
    assert pe._tokenizers == {}
    pe.measure(ernie, "a cat", None)
    assert len(tokenizers["loads"]) == 3


def test_concurrent_analyses_load_each_tokenizer_once(monkeypatch):
    loads = []

    def load(source):
        loads.append(source.subfolder)
        time.sleep(0.05)
        return clip()

    monkeypatch.setattr(pe, "_load_tokenizer", load)
    results = []

    def analyse():
        results.append(pe.measure("KBlueLeaf/Kohaku-XL-Zeta", words(80), None))

    threads = [threading.Thread(target=analyse) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(loads) == ["tokenizer", "tokenizer_2"]
    assert len(results) == 4
    assert all(detail is None and len(reports) == 2 for reports, _, detail in results)


class Stalled:
    """A `_load_tokenizer` that blocks until released, as a first-use fetch
    does on a stalled connection (the Hub's client has no timeout)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.entered = threading.Event()
        self.release = threading.Event()
        self.loads = 0

    def __call__(self, source):
        self.loads += 1
        self.entered.set()
        self.release.wait(30)
        return self.tokenizer


@pytest.fixture
def stalled(monkeypatch):
    load = Stalled(tekken())
    monkeypatch.setattr(pe, "_load_tokenizer", load)
    yield load
    load.release.set()


def _in_thread(fn):
    """`fn` started on a daemon thread; the list gets its result."""
    result: list = []
    thread = threading.Thread(target=lambda: result.append(fn()), daemon=True)
    thread.start()
    return thread, result


def test_unload_never_waits_for_a_load_and_a_late_one_is_not_kept(stalled):
    ernie = "baidu/ERNIE-Image-Turbo"
    check, answers = _in_thread(lambda: pe.measure(ernie, "a cat", None))
    assert stalled.entered.wait(5)
    unloading, _ = _in_thread(pe.unload)
    unloading.join(5)
    assert not unloading.is_alive(), "unload() waited on the tokenizer fetch"
    stalled.release.set()
    check.join(5)
    # The check that asked still reads what the load brought back ...
    ((reports, _, detail),) = answers
    assert detail is None and by(reports)["Ministral-3B"].budget == 2047
    # ... but a load that lands after unload() is not cached.
    assert pe._tokenizers == {} and pe._loading == {}
    pe.measure(ernie, "a cat", None)
    assert stalled.loads == 2 and len(pe._tokenizers) == 1


def test_a_stalled_load_costs_the_check_its_count_and_lands_for_the_next(
    stalled, monkeypatch, caplog
):
    ernie = "baidu/ERNIE-Image-Turbo"
    monkeypatch.setattr(pe, "LOAD_WAIT", 0.2)
    started = time.monotonic()
    with caplog.at_level("INFO", logger=pe.log.name):
        reports, findings, detail = pe.measure(ernie, "a cat BREAK a dog", None)
    assert time.monotonic() - started < 5
    assert reports == [] and detail == (
        "Ministral-3B unchecked: the tokenizer is still loading (a first use downloads it) - "
        "check again in a moment"
    )
    # The log says what timed out, not just which files.
    assert "tokenizer still loading when the check's 0.2 s wait ran out" in caplog.text
    # The rest of the analysis stands.
    assert [f.rule for f in findings] == ["encoder.weighting"]
    # A second check waits on the same load rather than starting its own ...
    assert pe.measure(ernie, "a cat", None)[2] == detail
    assert stalled.loads == 1
    # ... and once it lands the next one reads it.
    stalled.release.set()
    monkeypatch.setattr(pe, "LOAD_WAIT", 5.0)
    assert pe.measure(ernie, "a cat", None)[2] is None
    assert stalled.loads == 1


def test_a_load_still_in_flight_does_not_speak_for_one_that_failed(monkeypatch):
    # "Check again in a moment" is true of the load still going only: a
    # gated tokenizer beside it keeps its own reason, and a shared reason
    # names both.
    gemma, jina = pe.encoders_for(newbie_loading.MODEL_ID)
    release = threading.Event()

    def load(source):
        if source == jina.source:
            raise PermissionError("403 Client Error: gated")
        release.wait(30)
        return fakes.fake_tokenizer(source)

    monkeypatch.setattr(pe, "_load_tokenizer", load)
    monkeypatch.setattr(pe, "LOAD_WAIT", 0.2)
    try:
        detail = pe.measure(newbie_loading.MODEL_ID, "a cat", None)[2]
    finally:
        release.set()
    assert detail == (
        "Gemma 3 4B unchecked: the tokenizer is still loading (a first use downloads it) - "
        "check again in a moment; Jina CLIP v2 unchecked: PermissionError: 403 Client Error: gated"
    )
    pe.unload()
    monkeypatch.setattr(pe, "_load_tokenizer", lambda source: _raise(OSError("offline")))
    assert pe.measure(newbie_loading.MODEL_ID, "a cat", None)[2] == (
        "Gemma 3 4B and Jina CLIP v2 unchecked: OSError: offline"
    )


def _raise(error):
    raise error


def test_loading_prefers_a_converted_tree_then_the_cache_then_the_hub(monkeypatch, tmp_path):
    import huggingface_hub

    calls = []
    snapshot = tmp_path / "snapshot"
    (snapshot / "text_encoder").mkdir(parents=True)

    def snapshot_download(repo, *, revision, allow_patterns, local_files_only):
        calls.append((repo, revision, tuple(allow_patterns), local_files_only))
        if local_files_only:
            raise FileNotFoundError("not in the cache")
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(pe, "_from_pretrained", lambda directory: directory)
    monkeypatch.setattr(pe, "_converted_tree", lambda model_id: None)
    source = pe.encoders_for(newbie_loading.MODEL_ID)[0].source
    assert pe._load_tokenizer(source) == snapshot / "text_encoder"
    assert [call[3] for call in calls] == [True, False]
    repo, revision, patterns, _ = calls[0]
    assert (repo, revision) == (newbie_loading.MODEL_ID, newbie_loading.SOURCE["revision"])
    # Tokenizer files only: never the encoder's weights beside them.
    for wanted in ("tokenizer.json", "tokenizer.model", "merges.txt", "chat_template.jinja"):
        assert any(fnmatch.fnmatch(f"text_encoder/{wanted}", pattern) for pattern in patterns)
    for weights in ("gemma3-4b-it.safetensors", "model.bin"):
        assert not any(fnmatch.fnmatch(f"text_encoder/{weights}", pattern) for pattern in patterns)

    # A converted tree answers without touching the Hub.
    tree = tmp_path / "tree"
    (tree / "tokenizer").mkdir(parents=True)
    trees = {newbie_loading.MODEL_ID: str(tree)}
    monkeypatch.setattr(pe, "_converted_tree", trees.get)
    calls.clear()
    assert pe._load_tokenizer(source) == tree / "tokenizer"
    assert calls == []

    # A local checkpoint directory is read in place; a missing folder is a
    # plain not-found, never handed on as a repo id.
    local = pe.TokenizerSource(str(tmp_path), "tokenizer_2")
    with pytest.raises(FileNotFoundError, match="no tokenizer at"):
        pe._load_tokenizer(local)
    (tmp_path / "tokenizer_2").mkdir()
    assert pe._load_tokenizer(local) == tmp_path / "tokenizer_2"
    assert calls == []
