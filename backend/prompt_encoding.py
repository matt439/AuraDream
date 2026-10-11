"""What each checkpoint's own tokenizers make of a prompt, measured on the CPU.

Every pipeline here cuts an over-long prompt from the right, and most do it in
silence: the SDXL negative prompt, every Qwen-encoded pipeline, ERNIE and
Anima never say so, and the warnings the others print go to the server console.
Nothing reaches the page or the saved row. This module answers the question
exactly, with the checkpoint's real tokenizers: how many tokens the text is,
how many of them the pipeline keeps, and the character where reading stops.

**Measured over the string the pipeline builds.** Krea 2 and Qwen-Image put a
34-token system turn in front of the prompt, Z-Image and klein render the
tokenizer's chat template around it, Lumina 2 prepends NetaYume's instruction
and `<Prompt Start>`; the budget is what is left of the pipeline's truncation
length after all of that, and a boundary merge (Gemma's trailing space
joining the first word) is counted the way the tokenizer counts it. The fast
tokenizers' offsets then carry the cut back into the user's own text.


**Tokenizers load lazily, on the CPU, and failure is an answer.** A gated
repo or a first use with no network leaves that encoder unchecked with a
one-line reason; `measure` never raises. A load runs on a daemon thread of its
own, outside the cache's lock, and a check waits `LOAD_WAIT` seconds for it at
most: the Hub's client has no timeout, and a stalled fetch must cost the
checks their counts, never hold Unload or shutdown. A loaded tokenizer stays
cached until `unload()`, which the model's Unload also calls - but no status
route ever loads one, because the first load can be a download.

The table of what each pipeline feeds its encoders is `encoder_table`
(re-exported here); literal weighting syntax is `prompt_weighting`.
"""
from __future__ import annotations

import gc
import logging
import os
import re
import threading
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

from . import prompt_weighting
from .backends import anima_loading, netayume_loading, newbie_loading, single_file
from .encoder_table import (  # noqa: F401 - re-exported for callers and tests
    ANIMA_SEQUENCE,
    FLUX_T5_SEQUENCE,
    KLEIN_SEQUENCE,
    KREA2_PREFIX_TOKENS,
    KREA2_SEQUENCE,
    KREA2_SUFFIX_TOKENS,
    LUMINA_SEPARATOR,
    LUMINA_SEQUENCE,
    QWEN_ASSISTANT_SUFFIX,
    QWEN_IMAGE_PREFIX_TOKENS,
    QWEN_IMAGE_SEQUENCE,
    QWEN_IMAGE_TOKENIZER_MAX,
    QWEN_SYSTEM_PREFIX,
    Z_IMAGE_SEQUENCE,
    EncoderSpec,
    Field,
    TokenizerSource,
    encoders_for,
)
from .schemas import EncoderReport, PromptFinding

log = logging.getLogger(__name__)

# The chat template is rendered around this private-use character once, to
# find the text the template puts before and after the prompt.
_SLOT = "\ue000"
# Only the files a tokenizer is made of: a tokenizer subfolder may sit beside
# the weights (NewBie's text_encoder/), and a pattern that let those through
# would turn a token count into an 8 GB download.
_TOKENIZER_FILES = ("*.json", "*.txt", "*.model", "*.jinja")


# --- loading ------------------------------------------------------------------


class _Loaded(NamedTuple):
    tokenizer: Any
    # Token id -> the literal text of a control token this tokenizer parses
    # out of typed text: its own special tokens, and the chat protocol markers
    # its template uses (`<think>` on Qwen3), but not Gemma's HTML-tag or
    # whitespace-run tokens, which are vocabulary.
    controls: dict[int, str]


class _Pending:
    """One tokenizer load in flight, for every check that wants it."""

    def __init__(self, source: TokenizerSource) -> None:
        self.source = source
        self.done = threading.Event()
        self.loaded: _Loaded | None = None
        self.error: BaseException | None = None


class StillLoading(TimeoutError):
    """A load that outlasted a check's wait. It goes on in the background,
    and the next check finds it cached."""


# How long one check waits for the tokenizers it starts loading, all together.
LOAD_WAIT = 30.0

# Guards the two dicts only, never a load: `unload()` must not wait on I/O.
_lock = threading.Lock()
_tokenizers: dict[TokenizerSource, _Loaded] = {}
_loading: dict[TokenizerSource, _Pending] = {}

_UNKNOWN_PLACES = 8
_MARKUP = re.compile(r"<\|?/?[^<>\s|]+\|?>|\[/?[A-Z_]+\]")
_PIPE_SHAPE = re.compile(r"<\|[^<>|]+\|>")


def _converted_tree(model_id: str) -> str | None:
    for module in (single_file, netayume_loading, newbie_loading, anima_loading):
        tree = module.converted_tree(model_id)
        if tree:
            return tree
    return None


def _directory(path: Path, where: str) -> Path:
    # A path that is not a directory would reach from_pretrained as a repo
    # id, and fail with a message about repo-id characters.
    if not path.is_dir():
        raise FileNotFoundError(f"no tokenizer at {where}")
    return path


def _snapshot(source: TokenizerSource, local_only: bool) -> Path:
    import huggingface_hub

    snapshot = huggingface_hub.snapshot_download(
        source.repo,
        revision=source.revision,
        allow_patterns=[f"{source.subfolder}/{pattern}" for pattern in _TOKENIZER_FILES],
        local_files_only=local_only,
    )
    where = f"{source.label} in the Hub cache" if local_only else source.label
    return _directory(Path(snapshot) / source.subfolder, where)


def _tree(source: TokenizerSource) -> Path:
    tree = _converted_tree(source.tree or "")
    if tree is None:
        raise FileNotFoundError(f"no converted tree for {source.tree}")
    subfolder = source.tree_subfolder or source.subfolder
    return _directory(Path(tree) / subfolder, f"{tree}/{subfolder}")


def _locations(source: TokenizerSource) -> list[Callable[[], Path]]:
    """Where to look, cheapest first: a converted tree, the Hub cache, the Hub."""
    places: list[Callable[[], Path]] = []
    if source.tree:
        places.append(partial(_tree, source))
    if os.path.isdir(source.repo):
        places.append(partial(_directory, Path(source.repo) / source.subfolder, source.label))
        return places
    # The cache first: from_pretrained on a Hub id asks the network about
    # every optional file even when the snapshot is complete.
    places.append(partial(_snapshot, source, True))
    places.append(partial(_snapshot, source, False))
    return places


def _from_pretrained(directory: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(directory))


def _load_tokenizer(source: TokenizerSource) -> Any:
    """One tokenizer from the first place that has it; raises the last error."""
    error: Exception | None = None
    for locate in _locations(source):
        try:
            tokenizer = _from_pretrained(locate())
        except Exception as exc:  # noqa: BLE001 - the next place may have it
            error = exc
            continue
        # The count runs past model_max_length on purpose; transformers'
        # one-time "longer than the maximum" warning would say otherwise.
        warned = getattr(tokenizer, "deprecation_warnings", None)
        if isinstance(warned, dict):
            warned["sequence-length-is-longer-than-the-specified-maximum"] = True
        return tokenizer
    raise error or FileNotFoundError(source.label)


def _control_tokens(tokenizer: Any) -> dict[int, str]:
    template = getattr(tokenizer, "chat_template", None) or ""
    if isinstance(template, dict):
        template = "\n".join(str(text) for text in template.values())
    controls: dict[int, str] = {}
    for token_id, added in (getattr(tokenizer, "added_tokens_decoder", None) or {}).items():
        content = getattr(added, "content", str(added))
        if not _MARKUP.fullmatch(content):
            continue
        special = getattr(added, "special", False)
        if special or _PIPE_SHAPE.fullmatch(content) or content in template:
            controls[int(token_id)] = content
    special_ids = list(getattr(tokenizer, "all_special_ids", []) or [])
    for token_id, content in zip(special_ids, tokenizer.convert_ids_to_tokens(special_ids)):
        if isinstance(content, str) and _MARKUP.fullmatch(content):
            controls.setdefault(int(token_id), content)
    return controls


def _fetch(pending: _Pending) -> None:
    """The load itself, on its own daemon thread. Cached only after success,
    so a failed first load (gated, offline) is retried by the next check, and
    only while still registered: one that lands after `unload()` is handed to
    the checks waiting on it and then dropped."""
    source = pending.source
    try:
        tokenizer = _load_tokenizer(source)
        pending.loaded = _Loaded(tokenizer, _control_tokens(tokenizer))
    except BaseException as exc:  # noqa: BLE001 - raised again in the waiting check
        pending.error = exc
    finally:
        with _lock:
            if _loading.get(source) is pending:
                del _loading[source]
                if pending.loaded is not None:
                    _tokenizers[source] = pending.loaded
        pending.done.set()


def _start(source: TokenizerSource) -> _Loaded | _Pending:
    """The cached tokenizer, or the load in flight - begun here if none is,
    so concurrent checks share one load. Never waits. (After `unload()` the
    next check begins a load of its own: a stalled one is forgotten, not
    stopped, and its thread lives on until it returns or the process ends.)"""
    with _lock:
        loaded = _tokenizers.get(source)
        if loaded is not None:
            return loaded
        pending = _loading.get(source)
        if pending is None:
            pending = _loading[source] = _Pending(source)
            thread = threading.Thread(
                target=_fetch, args=(pending,), daemon=True, name="tokenizer-load"
            )
            try:
                thread.start()
            except RuntimeError as exc:  # no new threads while the process exits
                del _loading[source]
                pending.error = exc
                pending.done.set()
        return pending


def _wait(state: _Loaded | _Pending, deadline: float) -> _Loaded:
    """The tokenizer, once its load lands, waiting until `deadline`
    (monotonic) at most; raises what the load raised."""
    if isinstance(state, _Loaded):
        return state
    if not state.done.wait(max(0.0, deadline - time.monotonic())):
        # For the log; the page's words key on the class (`_HUB_REASONS`).
        raise StillLoading(
            f"{state.source.label} still loading when the check's {LOAD_WAIT:g} s wait ran out"
        )
    if state.loaded is None:
        raise state.error or FileNotFoundError(state.source.label)
    return state.loaded


def unload() -> None:
    """Drop every cached tokenizer - along with the pipeline on Unload.

    Never waits: a load in flight is forgotten, not joined, and what it
    brings back is not cached.
    """
    with _lock:
        loaded = bool(_tokenizers)
        _tokenizers.clear()
        _loading.clear()
    if loaded:
        gc.collect()


# --- reading one field --------------------------------------------------------


@dataclass
class _Reading:
    spec: EncoderSpec
    field: Field
    text: str
    tokens: int
    budget: int
    cut: int | None
    # The budget that keeps the template's closing tokens, and where in the
    # text they start being lost; None where there is no tail to lose.
    clean: int | None
    tail_start: int | None
    # (start, end, the control token as the tokenizer stores it)
    controls: list[tuple[int, int, str]]
    unknown: list[tuple[int, int]]


def _frame(spec: EncoderSpec, tokenizer: Any, text: str, negative: bool) -> tuple[str, str]:
    """The text the pipeline puts before and after this field's text."""
    if negative and spec.negative == "bare":
        return "", ""
    if spec.chat_thinking is None:
        return spec.prefix, spec.suffix

    def render(content: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=spec.chat_thinking,
        )

    prefix, slot, suffix = render(_SLOT).partition(_SLOT)
    if not slot or render(text) != prefix + text + suffix:
        raise ValueError(f"{spec.name}'s chat template does not pass the prompt through as typed")
    return prefix, suffix


def _read(spec: EncoderSpec, loaded: _Loaded, field: Field, text: str) -> _Reading:
    tokenizer = loaded.tokenizer
    prefix, suffix = _frame(spec, tokenizer, text, negative=field == "negative_prompt")
    built = prefix + text + suffix
    start, stop = len(prefix), len(prefix) + len(text)
    encoding = tokenizer(built, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoding["input_ids"])
    offsets = [(int(s), int(e)) for s, e in encoding["offset_mapping"]]

    # Content tokens of `built` that survive: the truncation length less the
    # special tokens every pipeline here lets the tokenizer add, then
    # Qwen-Image's output window.
    limit = spec.max_length if spec.max_length is not None else int(tokenizer.model_max_length)
    cap = limit - int(tokenizer.num_special_tokens_to_add())
    if spec.window is not None:
        cap = min(cap, spec.window[0] + spec.window[1])

    # The field's own tokens: any that cover one of its characters, so a
    # token merged across the template boundary counts as the text's.
    inside = [
        j for j, (s, e) in enumerate(offsets)
        if (s < stop and e > start) or (s == e and start <= s < stop)
    ]
    if inside:
        first, last = inside[0], inside[-1]
        tokens = last - first + 1
        after = len(ids) - last - 1
    else:
        first = sum(1 for _, e in offsets if e <= start)
        last, tokens = first - 1, 0
        after = len(ids) - first
    budget = max(0, cap - first)

    def at(j: int) -> int:
        return min(max(offsets[j][0] - start, 0), len(text))

    cut = at(max(first, cap)) if tokens and last >= cap else None
    # Tokens after the text inside the truncated string are the template's
    # closing turn: they go first, as soon as the text passes `clean`.
    clean = max(0, budget - after) if after else None
    tail_start = at(first + clean) if clean is not None and tokens > clean else None

    controls: list[tuple[int, int, str]] = []
    unknown: list[tuple[int, int]] = []
    unk_id = getattr(tokenizer, "unk_token_id", None)
    unk_text = getattr(tokenizer, "unk_token", None)
    normal = _normalizer(tokenizer)
    for j in range(first, last + 1):
        s, e = offsets[j]
        piece = built[s:e]
        stripped = piece.strip()
        if not stripped:
            continue
        lead = s + len(piece) - len(piece.lstrip()) - start
        span = (max(lead, 0), min(lead + len(stripped), len(text)))
        # Compared as the tokenizer compares them: CLIP lowercases before it
        # matches its added tokens, so "<|EndOfText|>" is its EOS - and, EOS
        # being CLIP's unk id too, would otherwise read as an unknown character.
        control = loaded.controls.get(ids[j])
        if control is not None and normal(stripped) == normal(control):
            controls.append((*span, control))
        elif unk_id is not None and ids[j] == unk_id and stripped != unk_text:
            unknown.append(span)
    return _Reading(
        spec, field, text, tokens, budget, cut, clean, tail_start, controls, _runs(text, unknown)
    )


def _normalizer(tokenizer: Any) -> Callable[[str], str]:
    """The fast tokenizer's own normalizer, or the text unchanged."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    normalizer = getattr(backend, "normalizer", None)
    return normalizer.normalize_str if normalizer is not None else str


def _runs(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Adjacent spans as one, where only whitespace lies between them."""
    runs: list[tuple[int, int]] = []
    for s, e in sorted(set(spans)):
        if runs and not text[runs[-1][1]:s].strip():
            runs[-1] = (runs[-1][0], max(runs[-1][1], e))
        else:
            runs.append((s, e))
    return runs


# --- findings -----------------------------------------------------------------


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _noun(field: Field) -> str:
    return "prompt" if field == "prompt" else "negative prompt"


def _snippet(text: str, start: int, limit: int = 40) -> str:
    """The words from `start` on, whole from the first one, and short."""
    begin = start

    def inside_word(index: int) -> bool:
        # Han and kana, and their 、，。, are wide: Chinese and Japanese put no
        # spaces between words, so each such character stands alone. Hangul
        # is wide too, but Korean spaces its words.
        char = text[index]
        if char.isspace() or char in ",;":
            return False
        return (
            unicodedata.east_asian_width(char) not in ("W", "F")
            or unicodedata.name(char, "").startswith("HANGUL")
        )

    # A cut inside a word ("snow|-covered", T5's "forest|3") quotes the word
    # it splits; a cut at a comma quotes the tag after it. A "word" longer
    # than half the snippet (a URL, Thai, a run of marks) is quoted from the
    # cut instead: from its start the quote could end before reading stops.
    while 0 < begin < len(text) and inside_word(begin) and inside_word(begin - 1):
        begin -= 1
    if start - begin > limit // 2:
        begin = start
    rest = " ".join(text[begin:].split())
    rest = rest.lstrip(",;: 、，。；：") or rest
    if len(rest) <= limit:
        return rest
    space = rest.rfind(" ", 0, limit)
    return rest[: space if space > limit // 2 else limit].rstrip(" ,;") + "..."


def _cut_findings(readings: list[_Reading]) -> list[PromptFinding]:
    groups: dict[tuple[Any, ...], list[_Reading]] = {}
    for reading in readings:
        if reading.cut is not None:
            key = (
                reading.field, reading.cut, reading.budget, reading.tokens,
                reading.spec.pooled_only,
            )
            groups.setdefault(key, []).append(reading)
    findings = []
    for (field, cut, budget, tokens, pooled), group in groups.items():
        names = [reading.spec.name for reading in group]
        text, noun, many = group[0].text, _noun(field), len(names) > 1
        subject, snippet = _join(names), _snippet(group[0].text, cut)
        read, stop = ("read", "stop") if many else ("reads", "stops")
        opening = f"{subject} {read} the first {budget} of the {noun}'s {tokens} tokens"
        if pooled:
            onward = [
                reading.spec.name for reading in readings
                if reading.field == field and not reading.spec.pooled_only
                and (reading.cut is None or reading.cut > cut)
            ]
            message = f'{opening} into one pooled vector and {stop} at "{snippet}"'
            if onward:
                message += f"; {_join(onward)} {'read' if len(onward) > 1 else 'reads'} on"
            message += "."
            suggestion = f"Lead with the subject: {subject}'s one vector covers only this opening."
        else:
            message = (
                f'{opening} and {stop} at "{snippet}": the rest never reaches '
                f"{'them' if many else 'it'}."
            )
            suggestion = f"Put what matters before that point, or shorten the {noun}."
        findings.append(
            PromptFinding(
                rule="encoder.cut",
                severity="info" if pooled else "warn",
                message=message,
                field=field,
                start=cut,
                end=len(text),
                source="encoder",
                cite=group[0].spec.cite,
                suggestion=suggestion,
            )
        )
    return findings


def _tail_findings(readings: list[_Reading]) -> list[PromptFinding]:
    findings = []
    for reading in readings:
        if reading.tail_start is None or reading.clean is None:
            continue
        noun = _noun(reading.field)
        findings.append(
            PromptFinding(
                rule="encoder.template_tail",
                severity="warn",
                message=(
                    f"{reading.spec.name} sees the {noun} as a user turn that never closes: past "
                    f"{reading.clean} tokens the pipeline's truncation removes the end of its chat "
                    f"template, and this {noun} has {reading.tokens}."
                ),
                field=reading.field,
                start=reading.tail_start,
                end=len(reading.text),
                source="encoder",
                cite=reading.spec.cite,
                suggestion=f"Keep the {noun} within {reading.clean} tokens.",
            )
        )
    return findings


def _span_findings(readings: list[_Reading]) -> list[PromptFinding]:
    """Control tokens and unknown characters, one finding per place, naming
    every encoder that reads it that way."""
    controls: dict[tuple[Any, ...], list[_Reading]] = {}
    unknown: dict[tuple[Any, ...], list[_Reading]] = {}
    for reading in readings:
        for start, end, content in reading.controls:
            controls.setdefault((reading.field, start, end, content), []).append(reading)
        for start, end in reading.unknown:
            unknown.setdefault((reading.field, start, end), []).append(reading)
    findings = []
    for (field, start, end, content), group in controls.items():
        names = [reading.spec.name for reading in group]
        # As typed, which may differ in case from the token the cite names.
        typed = group[0].text[start:end]
        findings.append(
            PromptFinding(
                rule="encoder.control_token",
                severity="warn",
                message=(
                    f'{_join(names)} {"read" if len(names) > 1 else "reads"} "{typed}" as a '
                    "control token, not as text."
                ),
                field=field,
                start=start,
                end=end,
                source="encoder",
                cite=(
                    f"{group[0].spec.source.label} parses {content} as one of its control tokens, "
                    f"and {group[0].spec.file} tokenizes the prompt as typed"
                ),
                suggestion="Delete it, or spell it differently if it is meant as text.",
            )
        )
    shown: dict[str, int] = {}
    for (field, start, end), group in unknown.items():
        # A Chinese prompt on FLUX.1 is <unk> nearly throughout: past a few
        # places per field the last one says how many more, not each.
        index = shown[field] = shown.get(field, -1) + 1
        if index >= _UNKNOWN_PLACES:
            continue
        names = [reading.spec.name for reading in group]
        many = len(names) > 1
        message = (
            f'{_join(names)} {"have" if many else "has"} no token for '
            f'"{group[0].text[start:end]}" and {"read" if many else "reads"} <unk> in its place.'
        )
        rest = sum(1 for key in unknown if key[0] == field) - _UNKNOWN_PLACES
        if index == _UNKNOWN_PLACES - 1 and rest > 0:
            message += f" The same happens at {rest} more place{'s' if rest > 1 else ''}."
        findings.append(
            PromptFinding(
                rule="encoder.unknown_chars",
                severity="info",
                message=message,
                field=field,
                start=start,
                end=end,
                source="encoder",
                cite=(
                    f"{group[0].spec.source.label} maps characters outside its vocabulary "
                    "to <unk>"
                ),
            )
        )
    return findings


# huggingface_hub's errors, by class name, in words the page can show: their
# own first lines are an HTTP status and a request id. And this module's own
# wait, whose load goes on without the check.
_HUB_REASONS = {
    "StillLoading": "the tokenizer is still loading (a first use downloads it) - check again "
    "in a moment",
    "GatedRepoError": "the tokenizer's repo is gated - accept its terms on huggingface.co "
    "and log in",
    "RepositoryNotFoundError": "the tokenizer's repo is not on the Hub, or is private",
    "OfflineModeIsEnabled": "the tokenizer is not cached and HF_HUB_OFFLINE is set",
    "LocalEntryNotFoundError": "the tokenizer is not cached and the Hub cannot be reached",
}


def _reason(exc: Exception) -> str:
    known = (_HUB_REASONS.get(kind.__name__) for kind in type(exc).__mro__)
    reason = next((text for text in known if text), None)
    if reason is None:
        first = next(iter(str(exc).strip().splitlines()), "")
        reason = f"{type(exc).__name__}: {first}" if first else type(exc).__name__
    if len(reason) > 200:
        reason = reason[:197] + "..."
    return reason


def _detail(failures: list[tuple[str, Exception]]) -> str | None:
    """Each unchecked encoder under its own reason: "check again in a moment"
    must not speak for a tokenizer that is gated or broken."""
    by_reason: dict[str, list[str]] = {}
    for name, exc in failures:
        names = by_reason.setdefault(_reason(exc), [])
        if name not in names:
            names.append(name)
    return "; ".join(
        f"{_join(names)} unchecked: {reason}" for reason, names in by_reason.items()
    ) or None


def _one_line(lines: list[str | None]) -> str | None:
    """The unchecked reasons as the one line `measure` promises."""
    present = [line for line in lines if line]
    if len(present) < 2:
        return next(iter(present), None)
    return "; ".join(line.rstrip(".") for line in present) + "."


# --- the entry point ----------------------------------------------------------


def measure(
    model_id: str, prompt: str, negative_prompt: str | None
) -> tuple[list[EncoderReport], list[PromptFinding], str | None]:
    """Token counts, cut points and encoder findings for one request.

    `negative_prompt` is passed only when it will run - the caller decides
    through `applied()`. Returns the per-encoder reports, the `encoder.*`
    findings and a one-line reason for anything left unchecked. Never raises.
    """
    specs = encoders_for(model_id) or []
    statement = prompt_weighting.STATEMENT
    cite = f"{statement} ({specs[0].file})" if specs else statement
    fields: list[tuple[Field, str]] = [("prompt", prompt)]
    if negative_prompt:
        fields.append(("negative_prompt", negative_prompt))
    # The syntax pass runs outside the tokenizer loop's guard, so it gets its
    # own: whatever it trips on costs its findings, never the analysis.
    syntax: str | None = None
    try:
        weighting = [
            finding
            for field, text in fields
            for finding in prompt_weighting.findings(text, field, cite)
        ]
    except Exception as exc:  # noqa: BLE001 - unchecked, never an error
        log.info("prompt analysis: weighting syntax unchecked: %s", exc)
        weighting = []
        syntax = f"Weighting syntax unchecked: {type(exc).__name__}"
    if not specs:
        budgets = "Token budgets unchecked: no encoder table entry for this checkpoint."
        return [], weighting, _one_line([budgets, syntax])

    readings: list[_Reading] = []
    failures: list[tuple[str, Exception]] = []
    # Every first load starts before any is waited for, so they download side
    # by side within the one bound, and both fields read what one load gave.
    deadline = time.monotonic() + LOAD_WAIT
    started = {spec.source: _start(spec.source) for spec in specs}
    for field, text in fields:
        for spec in specs:
            if field == "negative_prompt" and spec.negative == "none":
                continue
            if any(name == spec.name for name, _ in failures):
                continue
            try:
                loaded = _wait(started[spec.source], deadline)
                readings.append(_read(spec, loaded, field, text))
            except Exception as exc:  # noqa: BLE001 - unchecked, never an error
                log.info("prompt analysis: %s unchecked: %s", spec.name, exc)
                failures.append((spec.name, exc))

    reports = [
        EncoderReport(
            name=reading.spec.name,
            field=reading.field,
            tokens=reading.tokens,
            budget=reading.budget,
            cut=reading.cut,
            role=reading.spec.role_for(reading.field),
            pooled_only=reading.spec.pooled_only,
            source=reading.spec.source.label,
        )
        for reading in readings
    ]
    findings = _cut_findings(readings) + _tail_findings(readings) + _span_findings(readings)
    return reports, findings + weighting, _one_line([_detail(failures), syntax])
