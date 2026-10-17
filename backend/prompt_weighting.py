"""Weighting syntax no diffusers pipeline parses, found in the typed text.

Weighting syntax (`(x:1.2)`, `((x))`, `BREAK`, `<lora:...>`) is reported by
the prompt analyser too: no pipeline in diffusers parses it, so it is a fact
about what reaches the encoder, not a style question. `\\(` and `\\)` escapes
are not flagged, though the tag autocomplete writes parentheses bare: the
authors' own examples carry escapes, and so do pasted A1111 prompts. Regex
only - no tokenizer; `prompt_encoding.measure` calls `findings` under a
guard of its own.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from .prompt_rules import QUOTED
from .schemas import PromptFinding

if TYPE_CHECKING:
    from .encoder_table import Field

# `\(` and `\)` are escapes, never syntax: a `(` or `[` preceded by a
# backslash opens nothing, and an escaped bracket inside is part of the words.
_WEIGHT = re.compile(
    r"(?<!\\)\((?P<words>(?:\\.|[^()\\])+?):\s*(?P<weight>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)"
)
_EMPHASIS = re.compile(r"(?<!\\)\({2,}(?P<words>(?:\\.|[^()\\])+)\){2,}")
_BRACKETS = re.compile(r"(?<!\\)\[+(?P<words>(?:\\.|[^\[\]\\])+)\]+")
_EDITING = re.compile(r"\||:\s*[+-]?(?:\d|\.\d)")
_BREAK = re.compile(r"\bBREAK\b")
_LORA = re.compile(r"<(?:lora|lyco|hypernet):[^<>]*>", re.IGNORECASE)
_EMBEDDING = re.compile(r"\bembedding:[^\s,()\[\]<>]+", re.IGNORECASE)
_ALTERNATION = re.compile(r"(?<!\\)\{[^{}]*\|[^{}]*\}")

STATEMENT = (
    "diffusers has no prompt-weighting parser: the pipeline tokenizes the prompt as typed"
)


def _whole_item(text: str, start: int, end: int) -> bool:
    """Whether a span is a comma item on its own, as A1111 brackets wrap tags."""
    left, right = text[:start].rstrip(" \t"), text[end:].lstrip(" \t")
    return (not left or left[-1] in ",\n") and (not right or right[0] in ",\n")


def _plain(words: str) -> str:
    return " ".join(words.replace("\\(", "(").replace("\\)", ")").split())


def _is_json(text: str) -> bool:
    stripped = text.strip()
    if stripped[:1] not in ("{", "["):
        return False
    try:
        json.loads(stripped)
    # A few thousand `[` - well inside the prompt's length limit - nest deeper
    # than the parser recurses, and that is not JSON anyone typed.
    except (ValueError, RecursionError):
        return False
    return True


def _weighting_spans(text: str) -> list[tuple[int, int, str, str]]:
    """(start, end, kind, words) per construct, outermost where they nest.

    Nothing inside quotation marks counts - `a sign reading "[OPEN]"` is text
    to render, read as the rules read it (`prompt_rules.QUOTED`) - and in a
    JSON prompt brackets and braces are its structure.
    """
    found: list[tuple[int, int, str, str]] = []
    for match in _WEIGHT.finditer(text):
        words, weight = match["words"].rstrip(), match["weight"]
        # "(16:9)" and "(at 3:30)" are a ratio and a time, not a weight.
        if words[-1:].isdigit() and "." not in weight:
            continue
        found.append((match.start(), match.end(), "weight", words))
    for match in _EMPHASIS.finditer(text):
        found.append((match.start(), match.end(), "emphasis", match["words"]))
    structured = _is_json(text)
    for match in () if structured else _BRACKETS.finditer(text):
        # A bracket in prose is usually a placeholder ("portrait of
        # [character]"); as a whole tag, or with editing syntax, it is A1111's.
        if _EDITING.search(match["words"]) or _whole_item(text, match.start(), match.end()):
            found.append((match.start(), match.end(), "brackets", match["words"]))
    patterns = [("break", _BREAK), ("lora", _LORA), ("embedding", _EMBEDDING)]
    if not structured:
        patterns.append(("alternation", _ALTERNATION))
    for kind, pattern in patterns:
        found.extend((m.start(), m.end(), kind, m.group()) for m in pattern.finditer(text))
    quoted = [m.span() for m in QUOTED.finditer(text)]
    kept: list[tuple[int, int, str, str]] = []
    for span in sorted(found, key=lambda item: (item[0], -item[1])):
        if kept and span[0] < kept[-1][1]:
            continue
        if any(start <= span[0] and span[1] <= end for start, end in quoted):
            continue
        kept.append(span)
    return kept


# A weight or a prompt-editing step at the end of bracketed words - the
# `:1.2` of `((x:1.2))`, the `:0.5` of `[cat:dog:0.5]` - is `_WEIGHT`'s number.
_TRAILING_NUMBER = re.compile(r":\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*$")


def _choose(options: list[str]) -> str:
    """What to write instead of a construct that offers these wordings."""
    options = list(dict.fromkeys(option for option in options if option))
    if not options:
        return "Delete it."
    if len(options) == 1:
        return f'Write "{options[0]}" plainly, or leave it out.'
    quoted = [f'"{option}"' for option in options]
    listed = ", ".join(quoted[:-1]) + " or " + quoted[-1]
    return f"Pick {listed}, or compare them on a grid's prompt axis."


def _pipes(text: str) -> str:
    return "the pipe" if text.count("|") == 1 else "the pipes"


def _bracket_message(typed: str, words: str) -> tuple[str, str]:
    """A1111's square brackets: alternation, prompt editing or de-emphasis.

    The suggestion never quotes the syntax back: the words it names are the
    ones the user meant, without the colons, the step or the pipes.
    """
    if "|" in words:
        return (
            f'"{typed}" is alternation syntax, but nothing here parses it: the encoder reads the '
            f"brackets and {_pipes(words)} as text.",
            _choose([_plain(part) for part in words.split("|")]),
        )
    step = _TRAILING_NUMBER.search(words)
    if step:
        # [to:when], [from:to:when], [from::when]: the wording changes partway.
        colons = "the colon" if words.count(":") == 1 else "the colons"
        return (
            f'"{typed}" is prompt-editing syntax, but nothing here parses it: the encoder reads '
            f"the brackets, {colons} and the number as text.",
            _choose([_plain(part) for part in words[: step.start()].split(":")]),
        )
    plain = _plain(words)
    message = (
        f'"{typed}" is de-emphasis syntax, but nothing here parses it: the encoder reads the '
        "brackets as text."
    )
    if typed.startswith("[["):
        return message, f'Write "{plain}" plainly.'
    # One bracket around a whole comma item is also how a template marks a
    # slot - `[character]` - which wants filling, not unwrapping.
    return message, (
        f'Write "{plain}" plainly - or, if it is a template slot, put what it stands for there.'
    )


def _weighting_message(kind: str, typed: str, words: str) -> tuple[str, str]:
    if kind == "brackets":
        return _bracket_message(typed, words)
    if kind == "emphasis":
        weight = _TRAILING_NUMBER.search(words)
        if weight:
            return (
                f'"{typed}" is emphasis and weighting syntax, but nothing here parses it: the '
                "encoder reads the brackets, the colon and the number as text.",
                f'Write "{_plain(words[: weight.start()])}" plainly; no pipeline here reads the '
                "number.",
            )
        return (
            f'"{typed}" is emphasis syntax, but nothing here parses it: the encoder reads the '
            "brackets as text.",
            f'Write "{_plain(words)}" plainly.',
        )
    if kind == "weight":
        return (
            f'"{typed}" is weighting syntax, but nothing here parses it: the encoder reads the '
            "brackets, the colon and the number as text.",
            f'Write "{_plain(words)}" plainly; no pipeline here reads the number.',
        )
    if kind == "break":
        return (
            "BREAK starts no new chunk here: the pipeline encodes the prompt as one sequence and "
            "reads BREAK as a word.",
            "Delete it, or use a comma.",
        )
    if kind == "lora":
        return (
            f'"{typed}" loads no adapter: the encoder reads it as text.',
            "Choose the adapter in the LoRA box and delete this text.",
        )
    if kind == "embedding":
        return (
            f'"{typed}" loads no embedding: the encoder reads it as text.',
            "Delete it; this app loads no textual-inversion embeddings.",
        )
    return (
        f'"{typed}" is not expanded: the encoder reads the braces and {_pipes(typed)} as text.',
        _choose([_plain(part) for part in typed[1:-1].split("|")]),
    )


def findings(text: str, field: Field, cite: str) -> list[PromptFinding]:
    findings = []
    for start, end, kind, words in _weighting_spans(text):
        message, suggestion = _weighting_message(kind, text[start:end], words)
        findings.append(
            PromptFinding(
                rule="encoder.weighting",
                severity="warn",
                message=message,
                field=field,
                start=start,
                end=end,
                source="encoder",
                cite=cite,
                suggestion=suggestion,
            )
        )
    return findings
