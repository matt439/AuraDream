"""Validate critic answers against the prompt, guide and deterministic findings."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Literal

from . import critic_context, critic_reorder, critic_runtime
from .critic_issues import _MEASURED, _clash, _described, _order_wants, _restated, _reversed
from .prompt_order import order_checked, reorder_problem, scene_ahead
from .prompt_rules import order_finding
from .prompt_text import QUOTED, _WORD_NOT_CJK, _capitalised, _is_prose, _norm, rendered_targets
from .schemas import CritiqueIssue, PromptCritique, PromptFinding

# A letter or digit of a script that spaces its words. A quote that starts or
# ends in one must start or end a word of the text as well: "man" is not the
# end of "woman", nor "pen" the middle of "OPEN". Han, kana and Hangul run
# on without spaces (`prompt_text._WORD_NOT_CJK`, which `QUOTED` reads too,
# so the two cannot drift apart), and so do Thai, Lao, Myanmar and Khmer: a
# quote of them may begin and end anywhere. An underscore is a tag's space:
# "hair" is a word of "long_hair".
_SPACED = rf"(?![_\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff]){_WORD_NOT_CJK}"
_SPACED_CHAR = re.compile(_SPACED)


def locate(quote: str, text: str) -> tuple[int, int] | None:
    """Where `quote` occurs in `text` as whole words, ignoring case, runs of
    whitespace and how apostrophes, quotation marks and dashes are written.

    The model's quote is a claim; the span is found here or not at all, and
    what the page selects is the prompt's own characters, not the model's.
    """
    quote, text = quote.translate(critic_reorder._FOLD), text.translate(critic_reorder._FOLD)
    for candidate in (quote, quote.strip(critic_reorder._QUOTE_TRIM)):
        words = candidate.split()
        if not words:
            continue
        pattern = r"\s+".join(map(re.escape, words))
        if _SPACED_CHAR.match(words[0][0]):
            pattern = rf"(?<!{_SPACED}){pattern}"
        if _SPACED_CHAR.match(words[-1][-1]):
            pattern = rf"{pattern}(?!{_SPACED})"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.start(), match.end()
    return None


def _clip(value: Any) -> str:
    return value.strip()[:critic_context.MAX_TEXT] if isinstance(value, str) else ""


# The answer's top-level keys (`response_schema`). An object with none of them
# is some other object - an issue, or braces in the thinking - not the answer.
_ANSWER_KEYS = ("parts", "issues", "rewrite", "reordered")
_THINK_END = "</think>"


def _answer_json(content: str) -> dict[str, Any]:
    """The answer object in the reply's content.

    llama-server splits the thinking off into `reasoning_content`; a template
    that left it inline ends it with </think>. So the answer is looked for
    after each </think>, the last first, and then in the whole content - never
    only after the last one: the answer quotes the prompt back, and a prompt
    pasted from another model's output can hold a </think> of its own. What
    follows one of those is a piece of the answer, and its first object - an
    issue - lacks the answer's keys, so it is passed over rather than read as
    an answer with nothing in it. The first failure is the one reported.
    """
    pieces = content.split(_THINK_END)
    failure: tuple[str, Exception | None] | None = None
    for i in reversed(range(len(pieces))):
        text = _THINK_END.join(pieces[i:])
        start = text.find("{")
        if start < 0:
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            failure = failure or (f"the critic's answer was not valid JSON: {exc}", exc)
            continue
        if isinstance(value, dict) and any(key in value for key in _ANSWER_KEYS):
            return value
        failure = failure or ("the critic's answer was not the JSON object it was asked for", None)
    message, cause = failure or ("the critic's answer held no JSON object", None)
    raise critic_runtime.CriticError(message) from cause


def _issue(raw: Any, context: critic_context.CriticContext, prompt: str) -> CritiqueIssue | None:
    if not isinstance(raw, Mapping):
        return None
    cite_id = str(raw.get("cite_id") or "").strip().upper()
    citation = context.citations.get(cite_id)
    problem = _clip(raw.get("problem"))
    if citation is None or not problem:
        return None
    kind = raw.get("kind") if raw.get("kind") in critic_context.ISSUE_KINDS else "other"
    quote = start = end = None
    if isinstance(raw.get("quote"), str):
        span = locate(raw["quote"], prompt)
        if span is not None:
            start, end = span
            quote = prompt[start:end]
    return CritiqueIssue.model_validate({
        "kind": kind,
        "problem": problem,
        "fix": _clip(raw.get("fix")),
        "cite_id": cite_id,
        # The citation's own, whatever else the answer carries: a server of
        # one's own need not hold the model to the schema's keys.
        "cite": citation.text,
        "url": citation.url,
        "source": citation.source,
        "note": citation.note,
        "quote": quote,
        "start": start,
        "end": end,
    })


def _same(a: str, b: str) -> bool:
    return " ".join(a.split()) == " ".join(b.split())


# Sentence punctuation that closes a whole prompt.
_STOPS = ".!?…。！？"


def _unchanged(text: str, prompt: str) -> bool:
    """Whether `text` is the prompt as it stands: runs of whitespace, a
    closing stop and how apostrophes, quotation marks and dashes are typed
    aside."""
    return _same(
        text.translate(critic_reorder._FOLD).strip().rstrip(_STOPS),
        prompt.translate(critic_reorder._FOLD).strip().rstrip(_STOPS),
    )


def _prompt_text(value: Any) -> str | None:
    """A whole prompt from the answer, or None. Over the request's own limit
    it is dropped rather than cut: a grid variant that stops mid-word is worse
    than no variant."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > critic_context.MAX_TEXT:
        return None
    return text


def _too_long(value: Any) -> bool:
    return isinstance(value, str) and len(value.strip()) > critic_context.MAX_TEXT


def _flat(text: str) -> str:
    return " ".join(text.translate(critic_reorder._FOLD).split()).casefold()


# A clause's end, which lettering does not run past.
_CLAUSE_END = re.compile(r"[,.;:!?\n，。；：！？]")


def _capitals(text: str) -> bool:
    """Whether `text`, which follows lettering whose end the rules are sure
    of, is only more words starting in a capital or a digit, set off by a
    space: the rules end "OPEN IN 5 MINUTES" and "Cafe On Main" at their "in"
    and "on", whatever the case, where a name or a title goes on."""
    return text[:1] in ("", " ", "\t") and all(_capitalised(word) for word in text.split())


def _lettered(
    words: str, prompt: str, targets: list[tuple[int, int, bool, int]]
) -> bool:
    """Whether `words` are text the prompt asks to render: inside one of its
    `rendered_targets`, or starting in one and running on within its clause
    where the rules ended it at a guess ("says fish and chips": `ends` is
    False), since the critic may read further. Past an end the rules are sure
    of ("says OPEN in red letters") only more capitals run on. Either way
    they start before the target's `head`: where the rules stopped a
    participle's words at a negation, the words from it on ("without limits"
    in "reading love without limits") are the absence the negation rule
    reads. Never the prompt's description."""
    key = _flat(words.strip(critic_reorder._QUOTE_TRIM))
    if not key:
        return False
    # Whole words: "pen" is not in "OPEN".
    if any(_starts_in(key, prompt, start, end, head) for start, end, _, head in targets):
        return True
    at = locate(words, prompt)
    return at is not None and any(
        start <= at[0] < head
        and not _CLAUSE_END.search(prompt, end, at[1])
        and (not ends or _capitals(prompt[end:at[1]]))
        for start, end, ends, head in targets
    )


def _starts_in(key: str, prompt: str, start: int, end: int, head: int) -> bool:
    """Whether `key`, flattened, is whole words of `prompt[start:end]`
    starting before `head`."""
    whole = f" {_flat(prompt[start:end].strip(critic_reorder._QUOTE_TRIM))} "
    lead = len(_flat(prompt[start:head].strip(critic_reorder._QUOTE_TRIM)))
    at = whole.find(f" {key} ")
    while at != -1:
        # `at` is the space before the words, which start at `at` unpadded.
        if at < lead:
            return True
        at = whole.find(f" {key} ", at + 1)
    return False


# What a part says when the prompt has no such part.
_NO_PART = frozenset(("none", "null", "n/a", "na", "nothing", "not specified", "unspecified", "-"))
_PART_BREAK = re.compile(r"[,;，；、\n]")


def _named(value: Any) -> list[str]:
    """The comma pieces of a part as the model named it; none where it named
    nothing ("none", "n/a", blank)."""
    text = _clip(value)
    if not text or text.strip(critic_reorder._QUOTE_TRIM).casefold() in _NO_PART:
        return []
    return [piece for piece in _PART_BREAK.split(text) if piece.strip(critic_reorder._QUOTE_TRIM)]


def _part(
    name: str, value: Any, prompt: str, targets: list[tuple[int, int, bool, int]]
) -> str | None:
    """A part of the prompt as the model named it, if it is the prompt's own.

    Every comma piece must be in the prompt (`locate`); text to render must
    also be lettering the prompt asks for. On the card the model put the
    whole prompt under text to render, "none" under camera, and lighting the
    prompt never mentioned.
    """
    pieces = _named(value)
    if not pieces or not all(locate(piece, prompt) for piece in pieces):
        return None
    if name == "text_to_render" and not all(
        _lettered(piece, prompt, targets) for piece in pieces
    ):
        return None
    return _clip(value)


# A rewrite's outer quotation marks, and the marks that close each opener.
_WRAPS = {'"': '"', "“": "”", "„": "“”", "«": "»", "「": "」", "『": "』", "'": "'", "‘": "’"}
# What the critic's own sentence may put after a wrap's closing mark: a stop,
# spaced or not, or a comma, semicolon or colon as it goes on.
_WRAP_TAIL = re.compile(r"[\s.!?…。！？,;:，；：、]+\Z")


def _unwrapped(text: str) -> str | None:
    """`text` without quotation marks around the whole of it, or None when it
    has none: the marks inside must still open before they close, so
    `"OPEN" sign, "CLOSED"` is two quotations and not one wrap, while the
    apostrophe in `'A dog's tail'` is no mark at all. Sentence punctuation
    after the closing mark - `"A cat on a sofa".`, `"..." !`, `"...";` -
    still leaves a wrap: the text keeps a stop, closed up to it, and not a
    comma, semicolon or colon, which were the critic's sentence going on."""
    tail = _WRAP_TAIL.search(text)
    body = text[: tail.start()] if tail else text
    stop = "".join(char for char in tail.group() if char in _STOPS) if tail else ""
    opener = body[:1]
    closers = _WRAPS.get(opener)
    if closers is None or len(body) < 3 or body[-1] not in closers:
        return None
    inner = body[1:-1]
    depth = 0
    for index, char in enumerate(inner):
        if char != opener and char not in closers:
            continue
        before = inner[index - 1] if index else " "
        after = inner[index + 1] if index + 1 < len(inner) else " "
        if before.isalnum() and after.isalnum():
            continue  # an apostrophe inside a word
        # A straight mark opens after a space, a curly one by its shape.
        opening = (before.isspace() or before in "([{") if opener in closers else char == opener
        depth += 1 if opening else -1
        if depth < 0:
            return None
    if depth:
        return None
    inner = inner.strip()
    return inner + stop if inner and not inner.endswith(tuple(_STOPS)) else inner


# Words that carry no content of their own, for telling a rewrite's new
# content from its new wording.
_FILLER = frozenset(
    "a an the of with and or in on at to from by for into onto over under is are was "
    "were be been its their his her this that these those as it while which".split()
)
_CONTENT_WORD = re.compile(r"[^\W\d_][\w'’-]*")
_CJK_TEXT = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
# New content words past which a rewrite is said to add content: a handful
# is rewording ("no people" -> "empty"), more is scenery and mood.
_ADDED_WORDS = 6


def _stem(word: str) -> str:
    word = word.casefold().replace("’", "'")
    return re.sub(r"(?:'s|ing|ed|es|s|ly)$", "", word) if len(word) > 4 else word


def _added_words(rewrite: str, prompt: str) -> list[str]:
    """The rewrite's content words the prompt has in no form."""
    # A closing single quote is not part of the word: 'Open Late'.
    have = {_stem(word.rstrip("'’")) for word in _CONTENT_WORD.findall(prompt)}
    added: list[str] = []
    for word in _CONTENT_WORD.findall(rewrite):
        word = word.rstrip("'’")
        key = word.casefold()
        if len(key) < 3 or key in _FILLER or _stem(word) in have or key in added:
            continue
        added.append(key)
    return added


# A weight written into a tag - "(twintails:1.2)" - is how strongly it reads,
# not which words it has.
_TAG_WEIGHT = re.compile(r":\s*-?(?:\d+\.?\d*|\.\d+)\s*(?=[)\]>])")
# A tag's words: an underscore is its space, and escapes, brackets and
# weights are not words.
_TAG_WORD = re.compile(r"[^\W_]+")
# The prompt's prose content words the rewrite may lose before it is said to
# drop content: one is rewording ("no people" -> "empty"); a lighthouse
# rewrite on Z-Image Turbo lost "a dramatic and melancholic mood", three.
_DROPPED_WORDS = 3


def _tag_words(text: str) -> tuple[str, ...]:
    unweighted = _TAG_WEIGHT.sub(" ", text.translate(critic_reorder._FOLD))
    return tuple(_stem(word) for word in _TAG_WORD.findall(unweighted))


def _dropped(rewrite: str, prompt: str, style: str) -> tuple[list[str], list[str]]:
    """The prompt's tags the rewrite has in no form, and the content words of
    its prose the rewrite has in no form.

    A tag is kept where its words, in its order, run whole inside one of the
    rewrite's comma pieces: case, underscores for spaces, spacing (`twin
    tails` for `twintails`), parenthesis escapes, a weight, an inflection
    and where the tag sits do not lose it, and nor
    does a longer tag holding it ("long twintails"). A changed tag is lost:
    `zzkx oc` -> `zzkx (series)` keeps no `zzkx oc`. A hybrid prompt's
    phrases are counted as prose, word by word, since rewording one is the
    rewrite's job.
    """
    pieces = [_tag_words(rewrite[start:end]) for start, end, _ in critic_reorder._piece_spans(rewrite, "tags")]

    def kept(words: tuple[str, ...]) -> bool:
        # Spacing aside: `twintails` and `twin tails` (a Danbooru alias) are
        # one tag, so the words are compared run together.
        joined = "".join(words)
        return any(
            "".join(piece[start:end]) == joined
            for piece in pieces
            for start in range(len(piece))
            for end in range(start + 1, len(piece) + 1)
        )

    tags: list[str] = []
    prose: list[str] = []
    if style in ("tags", "hybrid"):
        for start, end, _ in critic_reorder._piece_spans(prompt, "tags"):
            item = prompt[start:end]
            if style == "hybrid" and _is_prose(item):
                prose.append(item)
                continue
            words = _tag_words(item)
            shown = _norm(item)
            if words and not kept(words) and shown not in tags:
                tags.append(shown)
    else:
        prose.append(prompt)
    # The prose's words the rewrite lacks are the rewrite's added words the
    # other way round.
    return tags, _added_words(", ".join(prose), rewrite) if prose else []


_UNCHANGED = "The critic's rewrite was your prompt as it stands, so there is no rewrite to show."
_TAG_ESCAPES_KEPT = (
    "Your parenthesis escaping was kept in tags the rewrite otherwise left unchanged."
)


def _keep_tag_escapes(rewrite: str, prompt: str) -> str:
    """Keep the original spelling of retained tags, ignoring only parenthesis escapes.

    The authors' examples and pasted A1111 prompts carry these escapes, and
    autocomplete writes the bare form; either is kept as typed. Removing
    them alone is not a useful rewrite. They do change the encoded
    text, so retain the original rather than calling the two spellings equal.
    Match whole comma items exactly otherwise; never alter lettering, or
    choose a spelling when the prompt has differently escaped copies.
    """
    originals: dict[str, set[str]] = {}
    protected = [(start, end) for start, end, _, _ in rendered_targets(prompt)]
    for start, end, _ in critic_reorder._piece_spans(prompt, "tags"):
        item = prompt[start:end]
        if not any(mark in item for mark in "()") or any(
            start < high and end > low for low, high in protected
        ):
            continue
        key = item.replace(r"\(", "(").replace(r"\)", ")")
        originals.setdefault(key, set()).add(item)
    protected = [(start, end) for start, end, _, _ in rendered_targets(rewrite)]
    for start, end, _ in reversed(critic_reorder._piece_spans(rewrite, "tags")):
        item = rewrite[start:end]
        key = item.replace(r"\(", "(").replace(r"\)", ")")
        choices = originals.get(key, set())
        if len(choices) == 1 and not any(
            start < high and end > low for low, high in protected
        ):
            rewrite = rewrite[:start] + next(iter(choices)) + rewrite[end:]
    return rewrite


def _check_rewrite(
    rewrite: str, prompt: str, targets: list[tuple[int, int, bool, int]], style: str
) -> tuple[str | None, list[PromptFinding], list[str]]:
    """The rewrite's quotation marks held to what the prompt asks to render.

    On FLUX a quotation is lettering, and the rules skip quoted text, so
    nothing after this would notice: four of 19 rewrites on the card were the
    whole prompt in quotes, one ended in the example prompts' "Black Forest
    Labs". A wrap comes off; lettering the prompt does not have drops the
    rewrite; the prompt's own description in quotes is flagged on the rewrite.
    """
    if _unchanged(rewrite, prompt):
        return None, [], [_UNCHANGED]
    notes: list[str] = []
    inner = _unwrapped(rewrite)
    if inner is not None:
        notes.append(
            "The rewrite came back wrapped in quotation marks, which ask for lettering; "
            "they were taken off."
        )
        if not inner:
            return None, [], notes
        rewrite = inner
    if style in ("tags", "hybrid"):
        kept = _keep_tag_escapes(rewrite, prompt)
        if kept != rewrite:
            notes.append(_TAG_ESCAPES_KEPT)
            rewrite = kept
    if _unchanged(rewrite, prompt):
        return None, [], [*notes, _UNCHANGED]
    if len(rewrite) > critic_context.MAX_TEXT:
        return None, [], [*notes, (
            f"The rewrite was left out: it is longer than the {critic_context.MAX_TEXT} characters a "
            "prompt may be."
        )]
    findings: list[PromptFinding] = []
    for match in QUOTED.finditer(rewrite):
        words = match.group()[1:-1]
        if not words.strip() or _lettered(words, prompt, targets):
            continue
        if locate(words, prompt) is None:
            notes.append(
                f'The rewrite was left out: it puts "{critic_context._one_line(words, 60)}" in quotation '
                "marks - lettering the prompt does not ask for."
            )
            return None, [], notes
        findings.append(
            PromptFinding(
                rule="heuristic.rewrite_quote",
                severity="warn",
                message=(
                    f'"{critic_context._one_line(words, 60)}" is in quotation marks here but is not text '
                    "the prompt asks to render; quoted words are drawn as lettering."
                ),
                source="heuristic",
                start=match.start(),
                end=match.end(),
            )
        )
    # Told not to, the 4B critic still padded nearly every short prompt with
    # scenery and mood on the card: said, so the rewrite is read as new
    # content, and a comparison against it as testing more than wording.
    # Chinese and Japanese put no spaces between words, so they are not
    # counted this way.
    if not _CJK_TEXT.search(prompt):
        added = _added_words(rewrite, prompt)
        if len(added) >= _ADDED_WORDS:
            shown = ", ".join(added[:8]) + ("…" if len(added) > 8 else "")
            findings.append(
                PromptFinding(
                    rule="heuristic.rewrite_adds",
                    severity="info",
                    message=(
                        f"The rewrite adds {len(added)} words the prompt does not have "
                        f"({shown}): new content as well as new wording, so a comparison "
                        "against it tests both."
                    ),
                    source="heuristic",
                )
            )
        # And the other way: 8 of 19 SDXL rewrites on the card lost or
        # changed a tag the user wrote (`twintails`, `green hair`, `zzkx oc`
        # as `zzkx (series)`), and nothing said so. Any tag lost is said; a
        # prose rewrite rewords, so its words count from `_DROPPED_WORDS`.
        tags, words = _dropped(rewrite, prompt, style)
        if len(words) < _DROPPED_WORDS:
            words = []
        if tags or words:
            lost = [
                f"{len(tags)} of your tags ({', '.join(tags)})" if tags else "",
                f"{len(words)} of the prompt's words ({', '.join(words[:8])}"
                f"{'…' if len(words) > 8 else ''})" if words else "",
            ]
            findings.append(
                PromptFinding(
                    rule="heuristic.rewrite_drops",
                    severity="info",
                    message=(
                        f"The rewrite drops {' and '.join(part for part in lost if part)} in "
                        "any form: content taken out as well as wording changed, so a "
                        "comparison against it tests both."
                    ),
                    source="heuristic",
                )
            )
    return rewrite, findings, notes


def validate(
    answer: str | Mapping[str, Any],
    context: critic_context.CriticContext,
    prompt: str,
    *,
    model: str,
    seconds: float,
) -> PromptCritique:
    """The model's answer as a `PromptCritique`: every citation resolved,
    every quote and part found in the prompt, and what cannot stand left out
    and said so (`dropped`, `set_aside`). The rewrite is still the model's
    suggestion; the route lints it with layer 1."""
    data = critic_context._restored(_answer_json(answer) if isinstance(answer, str) else dict(answer))
    targets = rendered_targets(prompt)
    raw_parts = data.get("parts") if isinstance(data.get("parts"), Mapping) else {}
    parts = {name: _part(name, raw_parts.get(name), prompt, targets) for name in critic_context.PARTS}

    issues: list[CritiqueIssue] = []
    dropped = 0
    set_aside: list[str] = []
    restated: list[str] = []
    measured = 0
    raw_issues = data.get("issues") if isinstance(data.get("issues"), list) else []
    for raw in raw_issues[:critic_context.MAX_ISSUES]:
        issue = _issue(raw, context, prompt)
        if issue is None:
            dropped += 1
        elif (rule := _restated(issue, context)) == _MEASURED:
            measured += 1
        elif rule is not None:
            restated.append(rule)
        else:
            issues.append(issue)
    if restated:
        count = len(restated)
        set_aside.append(
            f"{count} issue{'' if count == 1 else 's'} left out: "
            f"{'it repeated' if count == 1 else 'they repeated'} the rule findings above "
            f"({', '.join(dict.fromkeys(restated))})."
        )
    if measured:
        # No finding to point at: the length was in range.
        assert context.measured_length is not None, "a measured-length issue requires the measured range"
        words, low, high = context.measured_length
        set_aside.append(
            f"{measured} issue{'' if measured == 1 else 's'} left out: "
            f"{'it' if measured == 1 else 'they'} spoke to the prompt's length, which the "
            f"check measured ({words} words, inside the {low}-{high} the guide suggests)."
        )
    wants = {index: want for index, issue in enumerate(issues) if (want := _order_wants(issue))}
    against = {
        index: wish for index, want in wants.items()
        if (wish := _reversed(want, context.order_steps)) is not None
    }
    # The documented order settles what it can; what is left must agree.
    rest = [index for index in wants if index not in against]
    clashing = [
        index for index in rest
        if any(other != index and _clash(wants[index], wants[other]) for other in rest)
    ]
    left_out = set(against) | set(clashing)
    issues = [issue for index, issue in enumerate(issues) if index not in left_out]
    if against:
        count = len(against)
        set_aside.append(
            f"{count} order issue{'' if count == 1 else 's'} left out: "
            f"{'it' if count == 1 else 'they'} asked for "
            f"{', '.join(dict.fromkeys(against.values()))}, against the documented order."
        )
    if clashing:
        asked = dict.fromkeys(wish for index in clashing for wish in _described(wants[index]))
        set_aside.append(
            f"{len(clashing)} order issues left out: they asked for orders that cannot "
            f"{'both' if len(clashing) == 2 else 'all'} hold ({', '.join(asked)})."
        )

    # Parts the model named that are not the prompt's: the page shows no row
    # for them, and on the card text to render held the scene in most runs.
    left = [name for name in critic_context.PARTS if parts[name] is None and _named(raw_parts.get(name))]
    if others := [name for name in left if name != "text_to_render"]:
        named = ", ".join(others[:-1]) + (" and " if len(others) > 1 else "") + others[-1]
        set_aside.append(
            f"The critic's {named} {'was' if len(others) == 1 else 'were'} left out: "
            "not the prompt's own words."
        )
    if "text_to_render" in left:
        set_aside.append(
            "The critic's text to render was left out: not lettering the prompt asks for."
        )

    raw_rewrite = data.get("rewrite")
    rewrite = _prompt_text(raw_rewrite)
    rewrite_findings: list[PromptFinding] = []
    if _too_long(raw_rewrite):
        set_aside.append(
            f"The rewrite was left out: it is longer than the {critic_context.MAX_TEXT} characters a "
            "prompt may be."
        )
    if rewrite is not None:
        rewrite, rewrite_findings, notes = _check_rewrite(rewrite, prompt, targets, context.style)
        set_aside += notes
    raw_reordered = data.get("reordered") if context.ordered else None
    reordered = None
    # Which reordering was kept, said by the server that decided it: the
    # page's last check may have placed the tags with other categories.
    kind: Literal["check", "allowed", "unchecked"] | None = None
    if _too_long(raw_reordered):
        set_aside.append(
            f"The critic's reordered version was left out: it is longer than the {critic_context.MAX_TEXT} "
            "characters a prompt may be."
        )
    elif (candidate := _prompt_text(raw_reordered)) is not None:
        # Nothing moved says nothing: the same pieces in the same order, as
        # `is_reorder` reads them, is no reorder and no change of words.
        kept, moved = critic_reorder._reordered(candidate, prompt, context.style)
        if not kept:
            set_aside.append(
                "The critic's reordered version was left out: "
                f"{critic_reorder._not_a_reorder(candidate, prompt, context.style)}; a reordering of your "
                "own can go under Compare."
            )
        elif moved is not None and context.style == "xml" and (
            scene := scene_ahead(context.sources.model_id, moved)
        ):
            # Check reads no XML order, but the guide states one - each
            # character's block, then the shared scene - and the critic was
            # told to move pieces into it: the other way round is no reading
            # of it, and order issues asking for that are left out too.
            # Where the prompt has it there already, the critic only left it.
            verb = "keeps" if scene_ahead(context.sources.model_id, prompt) else "puts"
            set_aside.append(
                f"The critic's reordered version was left out: it {verb} <{scene}> ahead of a "
                "character block, against the documented order."
            )
        elif moved is not None and (problem := reorder_problem(
            context.sources.model_id, prompt, moved, context.categories
        )):
            set_aside.append(problem)
        elif moved is None or not context.order_checked:
            reordered = moved
            kind = None if moved is None else "unchecked"
        elif context.rules_reordered is not None and critic_reorder._reordered(
            candidate, context.rules_reordered, context.style
        ) == (True, None):
            # Where the rules placed the prompt, the documented order is
            # theirs, and so is the text: the one Check offers under Use.
            reordered, kind = context.rules_reordered, "check"
        elif not context.placed_partly:
            # Every item placed: any other permutation is some other order,
            # and a comparison row of it would claim the documented one. The
            # critic moved tags only where the prompt is tags; elsewhere its
            # pieces were sentences or clauses, and what Check placed was
            # the prompt's comma items, sentences and all.
            placed = "these tags" if context.style == "tags" else "this prompt"
            set_aside.append(
                f"The critic's reordered version was left out: Check places {placed} in the "
                "documented order itself, and the critic's order is not that one."
            )
        elif critic_reorder._order_items(moved) != critic_reorder._order_items(prompt):
            # The pieces a reorder moves are the critic's reading, and the
            # order rule reads its own items: outside lettering, "、", ";"
            # and line breaks part the critic's pieces but not the rule's
            # comma items. Set apart otherwise, the same pieces are other
            # items to the rule (two joined by "、" are one, which no order
            # can be wrong about), and its answer would be about a list the
            # prompt is not.
            # Tags only where the prompt is tags: a prose prompt's comma
            # items are clauses (`order_detail` names them so), a hybrid's
            # are both.
            items = {"tags": "tags", "hybrid": "tags and clauses"}.get(context.style, "clauses")
            set_aside.append(
                "The critic's reordered version was left out: it does not keep the prompt's "
                f"comma items whole, so Check would read it as other {items}."
            )
        elif not order_checked(context.sources.model_id, moved):
            # The same items, but read as XML where an opening tag now leads:
            # an order the rule never read is not one it passed.
            set_aside.append(
                "The critic's reordered version was left out: Check does not read the order "
                "of it, so nothing has checked that order."
            )
        elif (finding := order_finding(
            context.sources.model_id, moved, context.categories
        )) is not None:
            set_aside.append(
                f"The critic's reordered version was left out: in it, {finding.message}"
            )
        else:
            # Some tag was not placed, so Check's own order is one the
            # documented order allows, not the only one; the critic's is
            # another that the same rule, placing with the same categories,
            # finds nothing wrong with. Still the critic's reading, which
            # the comparison then tests on the user's seeds.
            reordered, kind = moved, "allowed"
    return PromptCritique(
        model=model,
        seconds=round(seconds, 2),
        parts=parts,
        issues=issues,
        rewrite=rewrite,
        reordered=reordered,
        reordered_kind=kind,
        rewrite_findings=rewrite_findings,
        dropped=dropped,
        set_aside=set_aside,
    )


def _content(response: Any) -> tuple[str, str | None]:
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise critic_runtime.CriticError("llama-server returned no answer") from exc
    if not isinstance(content, str):
        raise critic_runtime.CriticError("llama-server returned no answer text")
    return content, choice.get("finish_reason")
