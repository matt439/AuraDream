"""What a critic issue says: the restated-finding vocabulary and order wants.

Whether an issue restates a rule finding the critic was told of (`_restated`,
`_MEASURED`) and what an order issue wants first or before what (`_order_wants`,
`_reversed`, `_clash`, `_described`). Pure over the issue's text and the
answer's context; `critic_validation.validate` imports these names and this
module never imports it.
"""
from __future__ import annotations

import re

from . import critic_context, critic_reorder
from .prompt_text import QUOTED
from .schemas import CritiqueIssue


# What an issue reads like when it restates a rule finding the critic was
# told of - which it did in nearly every answer on the card - keyed by the
# rule, with the name the page gives it. `guide.words` counts whenever the
# rules measured the length against a range, reported or not: the critic
# miscounts. Its pattern is a claim about the prompt's own length - its
# count, the guide's band, a bound, a sentence opening "Too short" - read a
# sentence at a time (`_says`), not any length or number of words: "a focal
# length of 85mm", "the dress length is", "add 3 words on its glow", "keep
# the sign text under 5 words". A number of words about some part is that
# part's where the words tying it there are these: "at least 5 words for the
# lighting", "3 words to the fog", "only 2 words describe the setting", "only
# 3 words of the prompt are about the lighting", "describe the umbrella in
# (or with) more than 2 words", "the ideal length of the raincoat" - and
# "give the lighting at least 5 words", "the lighting has only 3 words",
# "expand the lighting to 5 words", "keep the lighting description under 5
# words" (`_PART_COUNT`). What names the prompt or its description,
# "it" or a total ("30 words in the prompt", "in it", "in all", "keep it in
# under 80 words"), the guide or its range ("for the guide", "to the
# recommended range" - not "the dynamic range"), a fit to them ("to reach the
# range", "to stay under 80" - not "to match the lighting"), or the length of
# the prompt or a length of so many words, is still its length.
_GIVEN = (
    r"(?:on|about|of|for|to|in|describ(?:e[sd]?|ing)|cover(?:s|ing)?"
    r"|(?:are|is|go|goes)\s+(?:about|for|on|to))"
)
_PROMPT_NP = r"(?:(?:the|this|your|a|an)\s+)?(?:[\w'’-]+\s+)?prompts?(?![\w'’-])"
# The guide, its range or a bound it sets, as a count's object: "the
# guide's 30-80 range", "the recommended range", never any "range".
_MOD = r"(?:(?:guide|model)(?:'s|’s)|recommended|suggested|\d+\s*(?:-|–|to)\s*\d+)"
_DET = r"(?:(?:the|this|your|its|a)\s+)?"
_HEAD = r"(?:range|guide|limit|minimum|maximum)s?(?![\w'’-])"
_TARGET = rf"{_DET}(?:{_MOD}\s+){{0,2}}{_HEAD}"
_FIT = rf"it\b|{_DET}(?:{_MOD}\s+){{0,2}}(?:{_HEAD}|\d)|(?:about|around|at\s+least)\s+\d"
_OWN = (
    rf"{_PROMPT_NP}|{_TARGET}|it\b|(?:(?:a|the)\s+)?total\b|all\b(?![ \t]+[^\W\d_])"
    r"|(?:reach|meet|hit|fit|match|stay|fall|satisfy|follow|get)\w*\s+"
    rf"(?:(?:in|within|inside|into|under|below)\s+)?(?:{_FIT})"
)
_SOME_PART = (
    rf"(?!\s+(?:(?:of|in)\s+{_PROMPT_NP}\s+)?{_GIVEN}\b"
    rf"(?!\s+(?:{_OWN})))"
)
# A description, or descriptions, is the prompt's where the word just ahead
# is a determiner ("the description", "your description", "her
# description") or a word for the whole or for the critic's own version of
# it ("the full description", "the image description", "the prompt's
# description", "the revised description", "the short description"); any
# other word there names a part ("the lighting description", "your
# background description", "the soft warm lighting description"; in a
# count, up to four words name it), as any object but the image, the
# picture or the prompt (`_WHOLE_NOUNS`) does in "the description of the
# lighting". The same test decides a count (`_PART_COUNT`) and a manner
# (`_MANNER`), and a word for the whole keeps a length the prompt's: "too
# long a full description", "the recommended length of the full
# description".
_DETERMINERS = (
    "the", "this", "that", "your", "my", "our", "its", "his", "her", "their", "each", "a", "an",
)
_WHOLE_NOUNS = ("image", "picture", "prompt")
_WHOLE_WORDS = (
    "full", "whole", "entire", "complete", "overall", "total", "general", "current", "existing",
    "original", "main", "new", "revised", "rewritten", "updated", "edited", "improved",
    "expanded", "suggested", "proposed", "final", "detailed", "short", "shorter", "brief",
    "long", "longer", *_WHOLE_NOUNS,
)
_DETERMINER = "(?:" + "|".join(_DETERMINERS) + ")"
_WHOLE = "(?:" + "|".join(_WHOLE_WORDS) + ")"
_DESCRIPTION = r"descriptions?\b"
_PROMPTS_DESCRIPTION = rf"(?:{_DETERMINER}|{_WHOLE}(?:['’]s)?)\s+{_DESCRIPTION}"
# A bound after "in" or "with" is how some part is to be described - "the
# umbrella in under 5 words", "described with only 4", "the lighting
# description in under 5 words" - unless the prompt's comes before it: "it"
# or "this" in so many words, the prompt, or its description (`_OWNER`).
_OWNER = rf"\b(?:it|this|prompt|{_PROMPTS_DESCRIPTION})\s+(?:in|with)\s"
_MANNER = rf"(?:(?<!\bin\s)(?<!\bwith\s)|{_OWNER})"
# How long or short the prompt is said to be: "a little long", "too short".
_SO = r"(?:too\s+|very\s+|rather\s+|quite\s+|a\s+(?:little|bit)\s+|slightly\s+)?"
_NEGATIVE = r"negative[- ]prompts?\b"
_RESTATES: dict[str, tuple[str, re.Pattern[str]]] = {
    "guide.words": (
        "word count",
        re.compile(
            r"(?i)\bword[- ]?(?:count|range|limit)s?\b|\b\d+\s*words?\s+long\b"
            r"|\b\d+\s*(?:-|–|to)\s*\d{2,}[- ]?words?\b|\(\s*\d+\s*words?\s*\)"
            r"|\bbetween\s+\d+\s+and\s+\d+\s*words?\b"
            r"|\b(?:guide(?:'s|’s)|recommended|suggested)\s+\d+\s*(?:-|–|to)\s*\d{2,}\b"
            r"(?![ \t]*[^\W\d_])"
            rf"|{_MANNER}\b(?:under|over|below|above|fewer than|more than|less than|at least"
            rf"|at most|only|just)\s+\d+\s*words?\b{_SOME_PART}"
            r"|\b\d+[- ]words?\s+(?:minimum|maximum)\b"
            r"|\b\d+\s*words?,?\s+(?:is\s+|are\s+)?(?:above|below|over|under|outside|short of"
            r"|within|inside)\s+the\s+(?:range|guide|recommend|suggest)"
            r"|^at\s+\d+\s*words?\b"
            r"|\b(?:cut|trim|shorten|expand|lengthen|reduce|extend|bring|grow)\w*"
            r"(?:\s+(?:it|this|the prompt))?(?:\s+(?:down|up|back|out))?\s+to\s+"
            r"(?:about\s+|around\s+|roughly\s+|at least\s+|under\s+)?\d+\s*words?\b"
            r"|\bprompt\b(?:(?!\b(?:add|with)\b)[^.;]){0,30}?\b\d+\s*words?\b"
            rf"{_SOME_PART}"
            r"|\bprompt(?:\s+(?:is|runs|reads|seems))?"
            rf"\s+{_SO}(?:short|long|brief)\b"
            # "Reads as long", not "as long as": a comparison or a condition.
            rf"|\bprompt\s+reads\s+as\s+{_SO}(?:short|long|brief)\b(?!\s+as\b)"
            r"|^(?:(?:it|this)(?:'s|’s|\s+is)\s+(?:too|very)|too)\s+(?:short|long|brief)\b"
            rf"(?!\s+an?\s+(?!(?:{_WHOLE}(?:['’]s)?\s+)?(?:prompt|description|list)\b))"
            r"|\bprompt(?:'s|’s)? length\b|\blength of (?:the|this|your) prompt\b"
            r"|\b(?:recommended|suggested|ideal|optimal|required|minimum|maximum)"
            r" (?:prompt )?length\b(?!\s+of\b"
            r"(?!\s+(?:(?:the|this|your|a|an)\s+)?"
            rf"(?:{_WHOLE}(?:['’]s)?\s+)?(?:prompts?|description)\b"
            r"|\s+(?:(?:about|around)\s+)?\d))"
        ),
    ),
    "guide.negation": (
        "negation", re.compile(r"(?i)\bnegat|\babsence\b|\bwhat (?:not|you do not|you don't)\b"),
    ),
    "guide.quote_text": ("quotation marks", re.compile(r"(?i)\bquot")),
    "guide.order": ("tag order", re.compile(r"(?i)\border\b|\bfirst\b|\bbefore\b|\bafter\b")),
    "encoder.weighting": (
        "weighting", re.compile(r"(?i)\bweight|\bemphasis\b|\bparenthes|\bbracket"),
    ),
    "heuristic.quality_boilerplate": (
        "quality boilerplate", re.compile(r"(?i)\bquality (?:tag|term|word|boilerplate)|\bmasterpiece\b"),
    ),
    # A tag guide's missing finish, its tier or its year modifier - the
    # card's critic kept "the quality finish is missing" (A2) beside it -
    # named, or by the recipes' own tags.
    "guide.quality": (
        "quality tags",
        re.compile(
            r"(?i)\bquality[- ](?:tags?|terms?|words?|tiers?|tokens?|keywords?|modifiers?"
            r"|finish|suffix|recipe|boilerplate|scores?)\b"
            r"|\b(?:recommended|quality)\s+(?:finish|suffix|ending|recipe)\b"
            r"|\byear[- ](?:modifiers?|tags?)\b"
            r"|\bmasterpiece\b|\babsurdres\b|\b(?:high|great)[- ]score\b"
            r"|\b(?:good|best|amazing)[- ]quality\b"
        ),
    ),
    # A negative prompt missing, not running or recommended - "Missing
    # negative prompt" (G5) and "No negative prompt" (A3) were kept beside
    # the finding on the card. Only `reported_negative` holds the rule, so an
    # issue must name the negative prompt too; one moving a word there ("Put
    # 'blurry' in the negative prompt") is about the prompt, and stays.
    "guide.negative_prompt": (
        "negative prompt",
        re.compile(
            rf"(?i)\b(?:no|missing|without|lacks?|lacking|empty|absent|blank)\s+"
            rf"(?:(?:a|an|the|any)\s+)?{_NEGATIVE}"
            rf"|\b(?:add(?:s|ed|ing)?|includ(?:e|es|ed|ing)|suppl(?:y|ies|ied|ying)"
            rf"|provid(?:e|es|ed|ing)|set(?:s|ting)?|specif(?:y|ies|ied|ying))\s+"
            rf"(?:(?:a|an|the|one|its|their"
            rf"|recommended|suggested|official|card(?:'s|’s)|model(?:'s|’s)"
            rf"|authors?(?:'s|’s|'|’)?)\s+){{0,3}}{_NEGATIVE}"
            rf"|\bus(?:e|es|ed|ing)\s+(?:a|an|one)\s+(?:(?:recommended|suggested)\s+)?{_NEGATIVE}"
            rf"|\b(?:recommended|suggested)\s+{_NEGATIVE}"
            rf"|{_NEGATIVE}\s+(?:(?:is|was|are)\s+)?(?:(?:missing|empty|absent|blank|unused|ignored"
            rf"|disabled)\b|not\s+(?:being\s+)?(?:set|used|provided|given|run|running|applied"
            rf"|active|specified|included|present)\b)"
            rf"|{_NEGATIVE}\s+(?:does\s+not|doesn['’]t|never|will\s+not|won['’]t)\s+"
            rf"(?:run|apply|take\s+effect)"
        ),
    ),
}
# Marks around a quote that say nothing: "no people," is "no people".
_QUOTE_EDGE = critic_reorder._QUOTE_TRIM + "。，、；：！？"
# `_restated`'s answer for a length the rules measured and found in range.
_MEASURED = "measured length"
# A sentence of an issue's problem or fix; lettering, whose length is the
# sign's, not the prompt's; and the prompt, named.
_SENTENCE = re.compile(r"(?<=[.;!?])\s+")
_LETTERING = re.compile(
    r"(?i)\b(?:signs?|signage|lettering|captions?|titles?|labels?|slogans?|logos?|headlines?"
    r"|banners?|(?<!prompt )text(?![- ]to[- ]image))\b"
)
_PROMPT = re.compile(r"(?i)\bprompt\b")
# A number of words given to a part named ahead of it - "Give the lighting at
# least 5 words", "The lighting has only 3 words", "Expand the lighting to 5
# words" - taken out of a sentence before `guide.words` reads it, as is one
# given to a part named with its description ("Keep the lighting description
# under 5 words", "Keep your lighting description under 5 words", "Give the
# description of the lighting at least 5 words"). The prompt, its
# description, the whole, the count or the problem given as many ("Give the
# prompt at least 30 words", "Keep the description under 80 words", "The
# full description has only 12 words", "Keep the revised description under
# 80 words", "It has only 12 words", "The whole prompt is 12 words", "The
# problem is only 12 words") is still its length. "A" or "an" names a part
# only with its description: "keep a single paragraph under 80 words" is the
# prompt's.
_BOUND = (
    r"(?:(?:only|just|at least|at most|more than|fewer than|less than|under|over|about|around)"
    r"\s+)?\d+\s*words?\b"
)
_NAMES = "(?:" + "|".join(word for word in _DETERMINERS if word not in ("a", "an")) + ")"
_NAME_WORD = r"[^\W\d_][\w'’-]*"
_PART_WORDS = (
    r"(?:\s+(?!(?:prompts?|descriptions?|whole|entire|count|length|total|problem|issue)\b)"
    rf"{_NAME_WORD}){{1,2}}"
)
_WHOLE_NOUN = "(?:" + "|".join(_WHOLE_NOUNS) + ")s?"
_NAMED = (
    rf"(?:{_NAMES}{_PART_WORDS}"
    rf"|{_DETERMINER}(?:\s+{_NAME_WORD}){{0,3}}"
    rf"\s+(?!(?:{_DETERMINER}|{_WHOLE})\b){_NAME_WORD}\s+{_DESCRIPTION}"
    rf"|{_DETERMINER}(?:\s+{_WHOLE})?\s+{_DESCRIPTION}\s+of\s+{_NAMES}"
    rf"(?!(?:\s+{_NAME_WORD})?\s+{_WHOLE_NOUN}\b){_PART_WORDS})"
)
_PART_COUNT = re.compile(
    rf"(?i)\b(?:give|devote)\w*\s+{_NAMED}\s+{_BOUND}"
    rf"|\b(?:cut|trim|shorten|expand|keep|limit)\w*\s+{_NAMED}\s+(?:to|under|below)\s+{_BOUND}"
    rf"|\b{_NAMED}\s+(?:gets?|receives?|has|have|is|are|uses?)\s+{_BOUND}"
)
# An issue that says it is about the negative prompt, the only way one can
# restate a finding there: its quote is looked for in the prompt alone.
_NEGATIVE_PROMPT = re.compile(rf"(?i)\b{_NEGATIVE}")


def _says(rule: str, pattern: re.Pattern[str], issue: CritiqueIssue) -> bool:
    if rule != "guide.words":
        return bool(pattern.search(f"{issue.problem} {issue.fix}"))
    # A sentence at a time, none about lettering - and in an issue about
    # lettering ("The text on the mug is long" / "Cut it to 2 words"), only
    # one that names the prompt - with a part's own count taken out.
    sentences = [part for field in (issue.problem, issue.fix) for part in _SENTENCE.split(field)]
    lettering = any(_LETTERING.search(sentence) for sentence in sentences)
    return any(
        pattern.search(_PART_COUNT.sub(" ", sentence)) and not _LETTERING.search(sentence)
        and (not lettering or _PROMPT.search(sentence))
        for sentence in sentences
    )


def _restated(issue: CritiqueIssue, context: critic_context.CriticContext) -> str | None:
    text = f"{issue.problem} {issue.fix}"
    for rule, (name, pattern) in _RESTATES.items():
        if not _says(rule, pattern, issue):
            continue
        if rule in context.reported:
            return name
        if rule in context.reported_negative and _NEGATIVE_PROMPT.search(text):
            return name
        if rule == "guide.words" and context.measured_length is not None:
            return _MEASURED
    # Or it quotes the very words a finding points at - the card's critic
    # raised "no people" / "Replace with 'empty'..." beside the negation
    # finding on the same words, in words none of the patterns name.
    if issue.quote is not None and issue.start is not None:
        start = issue.start + len(issue.quote) - len(issue.quote.lstrip(_QUOTE_EDGE))
        end = issue.start + len(issue.quote.rstrip(_QUOTE_EDGE))
        for rule, low, high, slack in context.reported_spans:
            overlaps = start < high and end > low
            if start < end and overlaps and low - slack <= start and end <= high + slack:
                return _RESTATES[rule][0] if rule in _RESTATES else rule.split(".", 1)[-1]
    return None


# What an order issue asks for: a part first, or one part before another.
# One the documented order the critic was given reverses - "Place style
# description first" where it says subject first - is not right, and two in
# one answer that cannot both hold - "Move subject to the beginning" beside
# "Place style description first", both seen on Krea's own example - cannot
# both be. "Put the camera settings before the lighting" says nothing of
# what comes first, so it stands beside "Start with the subject". Read from
# the fix, the instruction ("Place the subject first, then the style"), and
# only then from a problem that says what is not first; a problem describing
# the prompt ("the style is placed before the subject") names parts in the
# prompt's order, not the one wanted, and a negated instruction ("Avoid
# placing the style before the subject") asks for nothing it names.
_PART = (
    r"(subject|style|setting|location|environment|scene|lighting|light|camera|action"
    r"|context|image type|medium|mood|quality)"
)
_WANTS = (
    re.compile(rf"(?i)\b(?:start|begin|lead|open)\w*\s+with\s+(?:the\s+|a\s+|an\s+)?{_PART}\b"),
    re.compile(
        rf"(?i)\b{_PART}\b(?:\s+[\w'-]+){{0,3}}?\s+(?:first|early"
        r"|before (?:everything|anything|all|the rest)|(?:at|to) the (?:beginning|start|front))\b"
    ),
)
_BEFORE = re.compile(
    rf"(?i)\b{_PART}\b(?:\s+[\w'-]+){{0,3}}?\s+before\s+(?:[\w'-]+\s+){{0,2}}?{_PART}\b"
)
_NOT_FIRST = re.compile(
    rf"(?i)\b{_PART}\b(?:\s+is)?\s+not\s+(?:first|front-loaded|placed first"
    r"|at the (?:beginning|start|front))\b"
)
# BFL's first step is "the image type and subject": an image type or medium
# first is the subject's place, not a rival to it.
_SAME_PART = {
    "location": "setting", "environment": "setting", "scene": "setting", "light": "lighting",
    "image type": "subject", "medium": "subject",
}
_STEP_PART = re.compile(rf"(?i)\b{_PART}\b")
# A negation in an instruction's clause, ahead of it or inside it: "Do not
# put the lighting before the camera", "The style should not come before".
_NEGATED = re.compile(r"(?i)\b(?:not|never|cannot|avoid\w*|instead of|rather than)\b|n[’']t\b")
# A clause's end, as `critic_validation._CLAUSE_END` reads it for lettering; held
# equal to it by `tests/test_critic_issues.py`.
_CLAUSE_BREAK = re.compile(r"[,.;:!?\n，。；：！？]")

# The part an order issue wants first, if any, and each (earlier, later).
_Wants = tuple[str | None, frozenset[tuple[str, str]]]


def _same_part(name: str) -> str:
    part = name.lower()
    return _SAME_PART.get(part, part)


def _negated(text: str, found: re.Match[str]) -> bool:
    breaks = [mark.end() for mark in _CLAUSE_BREAK.finditer(text, 0, found.start())]
    return bool(_NEGATED.search(text[breaks[-1] if breaks else 0:found.end()]))


def _order_wants(issue: CritiqueIssue) -> _Wants | None:
    if issue.kind != "order":
        return None
    # The prompt's own words, quoted in the fix, name no part.
    fix, problem = QUOTED.sub(" ", issue.fix), QUOTED.sub(" ", issue.problem)
    match = next(
        (found for pattern in _WANTS for found in pattern.finditer(fix)
         if not _negated(fix, found)),
        None,
    ) or _NOT_FIRST.search(problem)
    first = _same_part(match.group(1)) if match else None
    pairs = frozenset(
        (earlier, later) for found in _BEFORE.finditer(fix)
        if not _negated(fix, found)
        and (earlier := _same_part(found.group(1))) != (later := _same_part(found.group(2)))
    )
    return (first, pairs) if first or pairs else None


def _step(part: str, steps: tuple[str, ...]) -> int | None:
    """Where a part stands in the documented order: the first step naming it
    ("subject first", "Subject and scene"), or None."""
    for index, step in enumerate(steps):
        if any(_same_part(found.group(1)) == part for found in _STEP_PART.finditer(step)):
            return index
    return None


def _against_order(pair: tuple[str, str], steps: tuple[str, ...]) -> bool:
    """Whether the documented order puts `later` ahead of `earlier`: on an
    earlier step, or first when `earlier` is on none."""
    earlier, later = (_step(part, steps) for part in pair)
    return later is not None and (later == 0 if earlier is None else later < earlier)


def _reversed(wants: _Wants, steps: tuple[str, ...]) -> str | None:
    """What an order issue asks for that the documented order reverses: a
    part first that it puts some other part ahead of, or a pair."""
    first, pairs = wants
    named = {_same_part(found.group(1)) for step in steps for found in _STEP_PART.finditer(step)}
    if first and any(_against_order((first, part), steps) for part in named - {first}):
        return f"{first} first"
    return next((f"{a} before {b}" for a, b in sorted(pairs) if _against_order((a, b), steps)), None)


def _clash(a: _Wants, b: _Wants) -> bool:
    """Whether two order issues' wants cannot both hold."""
    (a_first, a_pairs), (b_first, b_pairs) = a, b
    if a_first and b_first and a_first != b_first:
        return True
    # One wants a part first, the other something ahead of it.
    for first, pairs in ((b_first, a_pairs), (a_first, b_pairs)):
        if any(later == first for _, later in pairs):
            return True
    return any((later, earlier) in b_pairs for earlier, later in a_pairs)


def _described(wants: _Wants) -> list[str]:
    first, pairs = wants
    return ([f"{first} first"] if first else []) + [f"{a} before {b}" for a, b in sorted(pairs)]
