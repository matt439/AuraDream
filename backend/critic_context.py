"""Grounded critic questions, response schema and token budgets."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from . import critic_runtime, critic_server
from .prompt_order import places_every_item
from .prompt_sources import Sources, sources_for
from .schemas import GenerateRequest, PromptAnalysis, PromptFinding

# The least a critique may generate, thought included: the card's critiques
# generated 705-2,511 tokens on FLUX.1 Krea [dev] prompts and 2,568 on a tag
# prompt (docs/verified.md, "Prompt analyser"). A long prompt is given more
# (`max_tokens`).
MIN_TOKENS = 4096
# An answer's cost besides its copies of the prompt: six issues, their fixes
# and the JSON around them. A short prompt's answer measured under 450.
ANSWER_OVERHEAD = 800
# Qwen3's recommended sampling for thinking mode. Greedy decoding is what its
# card warns against there: it loops.
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}

PARTS = ("subject", "action", "setting", "style", "lighting", "camera", "text_to_render")
ISSUE_KINDS = ("order", "missing", "vague", "conflict", "other")
MAX_ISSUES = 6
MAX_TEXT = 4000
# What the system message may spend on the authors' examples and the rule
# findings. The whole message stays under 1.5k tokens (Qwen3's tokenizer
# averages over four characters a token on it): guide sentences and excerpts
# are the point, and everything else is trimmed around them.
SYSTEM_BUDGET_CHARS = 6000
_MAX_EXAMPLES = 3
_EXAMPLE_CHARS = 280
_MAX_FINDINGS = 8
_FINDING_CHARS = 160


# --- The question ------------------------------------------------------------


@dataclass(frozen=True)
class Citation:
    text: str
    url: str | None
    source: Literal["guide", "author"]
    # Whose words these are when not the checkpoint's own authors' - an
    # excerpt's `note` (a translation, a family guide, a parent's prompt
    # book), or the fallback's general advice - shown beside the cite.
    note: str | None = None


@dataclass(frozen=True)
class CriticContext:
    """Everything one critique asks with, and resolves its answer against."""

    sources: Sources
    citations: dict[str, Citation]
    # Whether the app checks an order for the checkpoint (`analysis.order`);
    # only then may `reordered` be anything but null.
    ordered: bool
    # How the prompt reads (`analysis.style`): what a piece of it is, for
    # telling a reorder from a rewrite.
    style: str
    # Whether the rules placed the prompt in that order themselves
    # (`analysis.order_checked`), and the reorder they made of it
    # (`analysis.reordered`, None where nothing had to move): there a
    # reorder of the critic's stands only as that one - unless some item the
    # order rule reads kept every slot a category could give it, where that
    # decides the order (`placed_partly`: `prompt_rules.places_every_item`
    # false with `analysis.categories` - an item never looked up, or one the
    # tag index did not know). Then theirs is one order the documented one
    # allows, not the only one, and the critic's stands if it keeps the
    # items the rule reads whole and the same rule, placing with the same
    # `categories`, reads it and flags nothing in it.
    order_checked: bool
    rules_reordered: str | None
    placed_partly: bool
    categories: dict[str, str] | None
    # The rule ids layer 1 reported on the prompt, and where in it each
    # finding about the words points, with how far past them a quote may
    # reach: an issue restating any of them is left out. Those reported on
    # the negative prompt, which only an issue naming it restates. And the
    # prompt's word count with the guide's range, (words, low, high), where
    # the rules measured it: an issue about the length is left out, reported
    # or not.
    reported: frozenset[str]
    reported_negative: frozenset[str]
    reported_spans: tuple[tuple[str, int, int, int], ...]
    measured_length: tuple[int, int, int] | None
    # The documented order's steps, which a "put X before Y" is held to.
    order_steps: tuple[str, ...]
    system: str
    user: str
    schema: dict[str, Any]
    # The prompt's tokens, and both messages': `estimate_tokens` until the
    # server has counted them (`_counted`).
    prompt_tokens: int
    input_tokens: int

    @property
    def copies(self) -> int:
        """How often the answer repeats the prompt: about once across
        `parts`, once in `rewrite`, and once more in `reordered`."""
        return 3 if self.ordered else 2


# Qwen2's pre-tokenizer, which Qwen3 keeps, in `re` (a letter is [^\W\d_]):
# a word with the one mark or space before it, a digit, a run of punctuation,
# whitespace. Its merges never cross these pieces, so each is a token at least.
_PIECES = re.compile(
    r"'(?i:[sdmt]|ll|ve|re)|((?:[^\r\n\w]|_)?)([^\W\d_]+)|\d"
    r"| ?(?:[^\s\w]|_)+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
# Tokens per letter past ASCII, by script: Qwen3-4B's tokenizer on ordinary
# text in each, with room over its mean. Other letters take 1.5.
_SCRIPT_RATES = (
    (0x00C0, 0x024F, 1.2),  # Latin-1 and Latin Extended
    (0x0370, 0x03FF, 1.2),  # Greek
    (0x0400, 0x052F, 0.7),  # Cyrillic
    (0x0530, 0x058F, 1.35),  # Armenian
    (0x0590, 0x05FF, 0.7),  # Hebrew
    (0x0600, 0x06FF, 0.85),  # Arabic
    (0x0750, 0x077F, 0.85),  # Arabic Supplement
    (0x0900, 0x0DFF, 2.0),  # Devanagari to Sinhala, split around their vowel signs
    (0x0E00, 0x0EFF, 1.0),  # Thai, Lao
    (0x10A0, 0x10FF, 1.35),  # Georgian
    (0x1100, 0x11FF, 1.3),  # Hangul Jamo
    (0x1E00, 0x1EFF, 1.2),  # Latin Extended Additional
    (0x1F00, 0x1FFF, 1.2),  # Greek Extended
    (0x3040, 0x309F, 1.1),  # Hiragana
    (0x30A0, 0x30FF, 0.8),  # Katakana
    (0x3130, 0x318F, 1.3),  # Hangul Compatibility Jamo
    (0x3400, 0x4DBF, 1.0),  # CJK Extension A
    (0x4E00, 0x9FFF, 0.8),  # CJK Unified Ideographs
    (0xAC00, 0xD7AF, 1.3),  # Hangul Syllables
    (0xF900, 0xFAFF, 1.0),  # CJK Compatibility Ideographs
)


def _letter_rate(char: str) -> float:
    code = ord(char)
    for first, last, rate in _SCRIPT_RATES:
        if first <= code <= last:
            return rate
    return 1.5


def _mark_rate(char: str) -> float:
    """Punctuation a third of a token; a symbol one, an emoji past the Basic
    Multilingual Plane a quarter more; a joiner or a combining accent two,
    since they seldom merge."""
    if char < "\x80":
        return 1 / 3
    code = ord(char)
    if 0x0300 <= code <= 0x036F or code in (0x200C, 0x200D):
        return 2.0
    return 1.25 if code > 0xFFFF else 1.0


# A mixed-case run's humps: a capital run, or a word with its one capital.
_HUMPS = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+")


def _ascii_word_cost(letters: str) -> float:
    """Under a third of a token per letter in a word, more in capitals; a
    mixed-case run - an identifier, a brand, base64 - a token at least per
    hump, since its humps seldom merge across a change of case."""
    if len(letters) > 1 and letters.isupper():
        return 0.45 * len(letters)
    if letters.islower() or letters.isupper() or letters[1:].islower():
        return 0.3 * len(letters)
    return sum(
        max(1.0, (0.45 if len(hump) > 1 and hump.isupper() else 0.3) * len(hump))
        for hump in _HUMPS.findall(letters)
    )


def estimate_tokens(text: str) -> int:
    """Qwen3's token count for `text`, estimated before a server runs to
    count it (`_counted`).

    Piece by piece as its pre-tokenizer splits the text (`_PIECES`, after
    the NFC its normalizer applies), each a token at least - so a digit is
    one, and so is each run of letters between a hex digest's digits - at a
    rate per character for the rest. Measured against Qwen3-4B's tokenizer
    (its GGUF's own vocabulary) on the guides, the system messages, the
    design notes, ordinary text in two dozen scripts and on hex, base64,
    URLs, JSON, XML and emoji with joiners, it is long by a quarter to two
    fifths on English prose, about a fifth on tags and simplified Chinese,
    and up to half on Cyrillic, Arabic and Thai - three fifths at most on
    ordinary text (Vietnamese), which `_ESTIMATE_FLOOR` relies on; a list of
    camel-case identifiers or brands, whole words in the vocabulary, comes
    out two and a half times over. It is short on what it cannot see,
    vocabulary: by a fifth on Turkish, Polish or Icelandic words, romanised
    names and base64, a tenth on traditional Chinese, a quarter on kanji
    names, two fifths on Cantonese and half on rare Han characters. So it is
    a bound in neither direction: scaled down, it refuses before a server
    starts only what cannot fit (`check_fits`), and it sizes a question only
    for a server that will not count it.
    """
    text = unicodedata.normalize("NFC", text)
    total = 0.0
    for match in _PIECES.finditer(text):
        piece, mark, letters = match.group(0), match.group(1), match.group(2)
        if letters:
            plain = "".join(char for char in letters if char < "\x80")
            cost = _ascii_word_cost(plain) if plain else 0.0
            cost += sum(_letter_rate(char) for char in letters if char >= "\x80")
            if mark and mark != " ":
                cost += _mark_rate(mark)
        elif piece.isdigit():
            cost = 1.0
        else:
            cost = sum(_mark_rate(char) for char in piece if char != " ")
        # A space seldom merges with what follows it past ASCII: two more
        # tokens before kana or Han, one before a symbol.
        if piece[:1] == " " and piece[1:2] >= "\x80":
            if not piece[1].isalpha():
                cost += 1.0
            elif 0x3040 <= ord(piece[1]) <= 0x9FFF:
                cost += 2.0
        total += max(1.0, cost)
    return math.ceil(total)


def max_tokens(context: CriticContext) -> int:
    """What one critique may generate: its thought, then an answer that
    repeats the prompt `copies` times, then the issues - so the thought keeps
    its whole `REASONING_BUDGET` wherever the window has room for it. Sized
    by the server's own count of the prompt (`_counted`) once it runs, and by
    the estimate only for a server that cannot count.

    Not cut to the window the input leaves: b11206 stops at the window's end
    itself (context shift is off by default; finish_reason "length"), and a
    count that errs long would stop an answer the window could hold.
    """
    need = critic_runtime.REASONING_BUDGET + context.copies * context.prompt_tokens + ANSWER_OVERHEAD
    return min(critic_runtime.CONTEXT, max(MIN_TOKENS, need))


# The estimate's longest overcount of ordinary text is three fifths
# (`estimate_tokens`); scaled by this, a question it refuses cannot fit
# however that text tokenizes.
_ESTIMATE_FLOOR = 0.6


def check_fits(context: CriticContext, *, estimated: bool = False) -> None:
    """Refuse a question whose answer cannot fit the window even with no
    thought at all - 4000 characters of Chinese, say, which the answer would
    repeat to well past 8192 tokens.

    Twice. Before any server starts, on the estimate (`estimated`) scaled
    down to what it can promise, so what surely cannot fit is refused
    without a server's start (3 s on the card, half a minute under WSL with
    the model on /mnt/c). Then on the server's own count (`_counted`),
    before the question is sent: what is admitted fits, but for an answer
    longer than `copies` of the prompt and `ANSWER_OVERHEAD`, and what does
    not is refused rather than left to run to the window's end.
    """
    need = context.input_tokens + context.copies * context.prompt_tokens + ANSWER_OVERHEAD
    if (need * _ESTIMATE_FLOOR if estimated else need) > critic_runtime.CONTEXT:
        raise critic_runtime.CriticError(
            f"the prompt is too long for the critic: its answer repeats the prompt "
            f"{context.copies} times, which with the sources does not fit its "
            f"{critic_runtime.CONTEXT}-token window; shorten the prompt or the negative prompt"
        )


_COUNT_TIMEOUT = 30.0


def _count(base: str, api_key: str, text: str, *, special: bool) -> int | None:
    """The server's own token count of `text`, or None if it gives none."""
    status, payload = critic_server._request(
        "POST", base + "/tokenize", api_key,
        {"content": text, "add_special": special, "parse_special": special}, _COUNT_TIMEOUT,
    )
    tokens = payload.get("tokens") if status == 200 and isinstance(payload, Mapping) else None
    return len(tokens) if isinstance(tokens, list) else None


def _counted(
    base: str, api_key: str, body: dict[str, Any], context: CriticContext, prompt: str
) -> CriticContext | None:
    """`context` counted by the running server's tokenizer - the pinned
    GGUF's own vocabulary - in place of the estimate: the messages as its
    chat template renders them, which is exactly what it reads, and the
    prompt the answer repeats, as text it writes. None when the server does
    not say (a server of one's own without b11206's `/apply-template` or
    `/tokenize`), and the estimate stands. Both endpoints take the key, so
    this runs after the listener check."""
    status, rendered = critic_server._request(
        "POST", base + "/apply-template", api_key,
        {"messages": body["messages"], "chat_template_kwargs": body["chat_template_kwargs"]},
        _COUNT_TIMEOUT,
    )
    template = rendered.get("prompt") if status == 200 and isinstance(rendered, Mapping) else None
    if not isinstance(template, str):
        return None
    input_tokens = _count(base, api_key, template, special=True)
    prompt_tokens = _count(base, api_key, prompt, special=False)
    if input_tokens is None or prompt_tokens is None:
        return None
    return replace(context, input_tokens=input_tokens, prompt_tokens=prompt_tokens)


def _one_line(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# Qwen3's control and reasoning tokens as typed text: <|im_end|> and every
# other <|...|> marker, <think>, <tool_call>, <tool_response> and their
# closers. llama-server tokenizes the rendered chat with special tokens
# parsed, so a prompt holding <|im_end|> would end the turn it sits in and
# open whatever it typed next, and on a Qwen-encoder checkpoint the control
# token finding, which quotes it, would end the app's own system message. A
# zero-width joiner after the "<" leaves each one text the model reads and
# quotes back; one already there gets another, so `_restored` takes off
# exactly what `_defused` put in.
_JOINER = "\u200d"
_CONTROL_AHEAD = r"(?=\u200d*(?:\||/?(?:think|tool_call|tool_response)>))"
_CONTROL = re.compile("<" + _CONTROL_AHEAD)
_DEFUSED = re.compile("<" + _JOINER + _CONTROL_AHEAD)


def _defused(text: str) -> str:
    return _CONTROL.sub("<" + _JOINER, text)


def _restored(value: Any) -> Any:
    """The answer's strings with `_defused`'s joiners taken out: the model
    quotes the text it was sent, and its quotes are looked for in the
    prompt as typed."""
    if isinstance(value, str):
        return _DEFUSED.sub("<", value)
    if isinstance(value, Mapping):
        return {key: _restored(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restored(item) for item in value]
    return value


def response_schema(cite_ids: list[str], ordered: bool) -> dict[str, Any]:
    """The answer's shape, enforced by llama-server's grammar sampling.

    `cite_id` is an enum of the ids the message numbered, so a citation cannot
    be invented, only misapplied. Properties generate in the order listed:
    the parts first, as the model's own reading of the prompt.
    """
    nullable = {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": {
            "parts": {
                "type": "object",
                "properties": {name: nullable for name in PARTS},
                "required": list(PARTS),
                "additionalProperties": False,
            },
            "issues": {
                "type": "array",
                "maxItems": MAX_ISSUES,
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"enum": list(ISSUE_KINDS)},
                        "problem": {"type": "string"},
                        "quote": nullable,
                        "cite_id": {"enum": cite_ids},
                        "fix": {"type": "string"},
                    },
                    "required": ["kind", "problem", "quote", "cite_id", "fix"],
                    "additionalProperties": False,
                },
            },
            "rewrite": {"type": "string"},
            # Forced null where no order is documented: nothing to permute
            # into, and nothing for the model to spend tokens inventing.
            "reordered": nullable if ordered else {"type": "null"},
        },
        "required": ["parts", "issues", "rewrite", "reordered"],
        "additionalProperties": False,
    }


# Rules whose span is the words a finding is about: the guide's, the authors'
# and the app's own checks, and literal weighting syntax. Not the encoder's
# other spans, which mark characters and places whatever the words mean -
# where reading stops (a cut, a template's tail), a control token, what T5
# reads as <unk> - on FLUX.1 that is nearly all of a Chinese prompt, and
# every issue quoting it went as a restated finding.
def _about_words(rule: str) -> bool:
    return rule.startswith(("guide.", "author.", "heuristic.")) or rule == "encoder.weighting"


# The fallback's sentences are this app's, not an author's (`sources.url` is
# None), and a cite of one says so.
_GENERAL_NOTE = "this app's general advice: no author sources are known for this checkpoint"


def _finding_line(finding: PromptFinding) -> str:
    return _one_line(f"- {finding.rule} ({finding.severity}): {finding.message}", _FINDING_CHARS)


def build_context(
    model_id: str,
    request: GenerateRequest,
    applied: GenerateRequest,
    analysis: PromptAnalysis,
) -> CriticContext:
    """The system and user messages, the schema and the citation table."""
    sources = sources_for(model_id)
    citations: dict[str, Citation] = {}
    general = None if sources.url else _GENERAL_NOTE
    for index, sentence in enumerate(sources.sentences, 1):
        citations[f"G{index}"] = Citation(sentence.text, sentence.url, "guide", general)
    for index, excerpt in enumerate(sources.excerpts, 1):
        citations[f"A{index}"] = Citation(excerpt.text, excerpt.url, "author", excerpt.note)
    order = analysis.order
    ordered = order is not None
    negative = applied.negative_prompt or None
    quotes = "quot" in sources.guide["prompt"].lower()

    # The rewrite padded short prompts with scenery and mood in nearly every
    # run on the card, against the sources' own "Start short", and wrapped
    # whole prompts in quotation marks on "text in quotation marks".
    rewrite_rule = (
        "- rewrite: the prompt revised to obey the sources. Keep the user's intent: the "
        "same subjects, count, medium and any text to render. Add no object, scenery, "
        "mood or text the prompt lacks, and never pad it to reach a word range: a range "
        "is the guide's band for most images, not a minimum. Use the form the sources "
        "ask for (sentences, tags or XML) and fix the rule findings too. Return it "
        "unchanged if no edit is needed."
    )
    if analysis.style in ("tags", "hybrid"):
        rewrite_rule += " Keep tags' backslash escapes."
    if negative is None:
        rewrite_rule += (
            " No negative prompt runs, so describe what should be seen, never what should not."
        )
    if quotes:
        rewrite_rule += (
            " Put quotation marks only around words the image must show as lettering - "
            "never around the prompt or a description."
        )
    if ordered:
        order_rule = (
            "- reordered: the prompt's own phrases moved whole into the documented order "
            "below and nothing else - add, drop, change or split no word, and move no word "
            "from one phrase to another."
        )
    else:
        # Not "no order is documented": Anima's card, Neta Lumina's prompt
        # book and ERNIE's enhancer all give one in the excerpts, which an
        # order issue may cite. What is missing is the app's own check.
        order_rule = (
            "- reordered: null; this app checks no order for this checkpoint. An order "
            "a source below states may still be an issue that cites it."
        )
    # The fallback has no authors (`sources.url` is None): its sentences are
    # the app's general guidance, and its caveat says so.
    whose = ", from the checkpoint's authors" if sources.url else ""
    lines = [
        f"Review an image prompt for checkpoint {model_id}. Judge it only "
        f"against the numbered sources below{whose}; general habits and advice written "
        "for other models do not count.",
        "Use the given JSON schema:",
        "- parts: the prompt's own words for each part, null where it has none.",
        f"- issues: at most {MAX_ISSUES}, most important first, and none that no source "
        "supports. Each cites exactly one source id. quote: the prompt's exact words the "
        "issue is about, or null. Do not repeat the rule findings listed at the end.",
        "Order lists place present parts. Generic/original subjects need no "
        "character/series names; never invent them.",
        rewrite_rule,
        order_rule,
    ]
    if sources.caveat:
        lines += ["", sources.caveat]
    lines += ["", "GUIDE"]
    lines += [
        f"G{index}: {sentence.text}" for index, sentence in enumerate(sources.sentences, 1)
    ]
    if sources.excerpts:
        lines += ["", "AUTHORS"]
        # A note shared by a run of excerpts is written out once; the rest
        # point back to it, which is most of FLUX.2's and Krea [dev]'s list.
        first_with: dict[str, int] = {}
        for index, excerpt in enumerate(sources.excerpts, 1):
            note = ""
            if excerpt.note in first_with:
                note = f" (as A{first_with[excerpt.note]})"
            elif excerpt.note:
                first_with[excerpt.note] = index
                note = f" ({excerpt.note})"
            lines.append(f"A{index}: {excerpt.text}{note}")
    if order is not None:
        lines += [
            "",
            "DOCUMENTED ORDER: " + " > ".join(order.steps)
            + f' - "{_one_line(order.cite, 200)}"',
        ]
    examples = [example for example in sources.guide.get("examples", []) if example.strip()]
    if examples:
        # Their form only: one rewrite on the card ended in the sign example's
        # "Black Forest Labs", another in the owl example's colour grading.
        lines += ["", "AUTHORS' EXAMPLE PROMPTS (their form only; never copy their subjects or text)"]
        lines += [f"- {_one_line(example, _EXAMPLE_CHARS)}" for example in examples[:_MAX_EXAMPLES]]
    if analysis.findings:
        lines += ["", "RULE FINDINGS (already reported)"]
        lines += [_finding_line(finding) for finding in analysis.findings[:_MAX_FINDINGS]]

    user = f"Prompt:\n{request.prompt}"
    if negative is not None:
        user += f"\n\nNegative prompt (it runs):\n{negative}"
    # Both messages whole: the prompt and the negative, and the findings
    # that quote them.
    user = _defused(user)
    system = _defused("\n".join(lines))
    # What `prompt_rules._words` measured and found in range, which it
    # reports nothing about - not XML or JSON, which it leaves alone. Out of
    # range, `guide.words` is reported.
    words = (sources.guide.get("analysis") or {}).get("words")
    measured_length = None
    if words and analysis.style not in ("xml", "json"):
        low, high = words["range"]
        if low <= analysis.words <= high:
            measured_length = (analysis.words, low, high)
    return CriticContext(
        sources=sources,
        citations=citations,
        ordered=ordered,
        style=analysis.style,
        order_checked=analysis.order_checked,
        rules_reordered=analysis.reordered,
        # From the placement itself, not from `categories_detail`, which says
        # only that some item was never looked up: one the index was asked
        # about and did not know - an OC's name, a trigger word - is placed
        # no better. An unplaced item counts only where its slot decides the
        # order (`unplaced_items`): where none does, the rules' order is the
        # documented one whatever the categories. The same predicate sets
        # `analysis.order_detail`, so the page says "Partly checked" exactly
        # where this is true.
        placed_partly=analysis.order_checked and not places_every_item(
            model_id, request.prompt, analysis.categories
        ),
        categories=analysis.categories,
        # The prompt's own: an issue is about the prompt, and one asking for
        # emphasis is not restating weighting found in the negative prompt -
        # unless it says it is about the negative prompt.
        reported=frozenset(
            finding.rule for finding in analysis.findings if finding.field == "prompt"
        ),
        reported_negative=frozenset(
            finding.rule for finding in analysis.findings if finding.field == "negative_prompt"
        ),
        reported_spans=tuple(
            (finding.rule, finding.start, finding.end,
             _QUOTE_SLACK if len(request.prompt[finding.start:finding.end].split()) > 1 else 0)
            for finding in analysis.findings
            if finding.field == "prompt" and finding.start is not None
            and finding.end is not None and _about_words(finding.rule)
        ),
        measured_length=measured_length,
        order_steps=tuple(order.steps) if order is not None else (),
        system=system,
        user=user,
        schema=response_schema(list(citations), ordered),
        prompt_tokens=estimate_tokens(request.prompt),
        input_tokens=estimate_tokens(system) + estimate_tokens(user),
    )


def request_body(context: CriticContext) -> dict[str, Any]:
    """The chat completion request, in the shape measured against b11206."""
    return {
        "messages": [
            {"role": "system", "content": context.system},
            {"role": "user", "content": context.user},
        ],
        **SAMPLING,
        "max_tokens": max_tokens(context),
        "chat_template_kwargs": {"enable_thinking": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "critique", "schema": context.schema},
        },
    }


# How far past a finding's words an issue's quote may reach and still be
# about those words: "rain, no people" for "no people". Only a quote that
# takes in some of them, though - "a shop" or "rain" beside "no people" is
# about the shop or the rain - and none past a single word: "a black cat" is
# not about a repeated "cat".
_QUOTE_SLACK = 8
