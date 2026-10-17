"""Shared prompt reading: comma items, prose shape, quotation and rendered lettering."""

from __future__ import annotations

import json
import re
from itertools import pairwise
from typing import Literal, NamedTuple


Style = Literal["prose", "tags", "hybrid", "xml", "json"]


# ------------------------------------------------------------------ reading the text


class Item(NamedTuple):
    """One comma item: its text as typed, stripped, and where it sits."""

    text: str
    start: int
    end: int


def comma_items(prompt: str) -> list[Item]:
    """The prompt's comma items, split the way the page's tag chips split it.

    `categories` is keyed by these texts, so the caller that builds it from
    the tag lookup and the rules that read it agree on what an item is. The
    rules read a stretch of lettering as one item (`_units`); every other
    item they read is one of these.
    """
    return _split(prompt, r"[^,]+")


# A comma item that is a grid placeholder whole - `{hair}`, `grid.PLACEHOLDER`'s
# shape, held equal to it by a test - is the grid's: Run grid fills it with
# its values, and Generate refuses it while the fold declares it. No tag is
# spelled so, so the index is never asked about one, and the order rules
# place it nowhere (`prompt_order._place`).
GRID_VARIABLE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


def is_grid_variable(text: str) -> bool:
    return GRID_VARIABLE.fullmatch(text.strip()) is not None


def _split(prompt: str, piece: str) -> list[Item]:
    """`prompt`'s stretches matching `piece`, stripped, where each one sits."""
    items = []
    for match in re.finditer(piece, prompt):
        raw = match.group()
        text = raw.strip()
        if text:
            start = match.start() + len(raw) - len(raw.lstrip())
            items.append(Item(text, start, start + len(text)))
    return items


# Ideographs and kana count one word each, the way word processors count
# them: Chinese and Japanese put no spaces between words. Hangul does, so it
# counts like Latin.
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*")


def word_count(prompt: str) -> int:
    return cjk_count(prompt) + len(_WORD.findall(_CJK.sub(" ", prompt)))


def cjk_count(prompt: str) -> int:
    """How many of `word_count`'s words are Han or kana characters, one
    each: the page calls those characters, not words."""
    return len(_CJK.findall(prompt))


_XML_ELEMENT = re.compile(r"<([A-Za-z_][\w.\-]*)>.*?</\1\s*>", re.S)
_XML_START = re.compile(r"<[A-Za-z_][\w.\-]*>")
# One sentence ending and another starting inside a comma item - "embroidery.
# Impeccable makeup" - which no tag does. Only before a capital: "dr. stone"
# and "love live! school idol project" are Danbooru series. A decimal point
# and a one- or two-letter abbreviation ("Dr. Stone", "St. Ives") are not one.
_SENTENCE_BREAK = re.compile(
    r"(?<!\d)(?<!\b[^\W\d_])(?<!\b[^\W\d_]{2})[.!?]\s+(?=[A-ZÀ-ÖØ-Þ])|[。！？](?=\S)"
)
_SENTENCE_END = re.compile(r"[.!?。！？][\"'”’)\]]*\s*$")
_ARTICLE = re.compile(r"(?i)^(?:a|an|the)\b")
# Words a sentence cannot do without and a tag list almost never has.
_FUNCTION_WORDS = frozenset(
    "a an the of with and or is are was on in at to from by into onto its their".split()
)
# What makes a comma item read as a sentence. Articles, which a Danbooru
# tag drops ("holding sign") but a title can keep ("the legend of zelda"),
# and clause words - pronouns, be and have - which tags never carry: they
# write "hand on own chest" where a caption says "her". A preposition marks
# a phrase only in a longer piece, since short tags use them freely (looking
# at viewer, hand on hip); "to" is left out for the romanised particle that
# opens "to aru majutsu no index".
_ARTICLES = frozenset(("a", "an", "the"))
_CLAUSE_WORDS = frozenset(
    "he she it they him her his its their them is are was were be been being "
    "has have had".split()
)
_LINKING_WORDS = frozenset(
    "of with and or on in at from by into onto over under through toward towards "
    "across along behind beneath above below between among around near inside while".split()
)
# Words past which an item is no Danbooru name: the longest romanised
# series titles run to about ten ("ore no imouto ga konna ni kawaii wake ga nai").
_LONGEST_TAG = 10


def _lower_words(text: str) -> list[str]:
    return [word.lower() for word in _WORD.findall(_CJK.sub(" ", text))]


def _is_prose(text: str) -> bool:
    """A comma item that reads as a phrase or a sentence rather than a tag.

    Not by length alone: "jojo no kimyou na bouken" and "march 7th (honkai:
    star rail)" are five-word Danbooru names. It takes a sentence break, the
    article or clause word a sentence is built with, or - in a piece of five
    words or more - a preposition or conjunction linking them ("expressive
    eyes with detailed reflections"). Ideographs carry none of those signs,
    so a Chinese or Japanese piece counts by length.
    """
    if _SENTENCE_BREAK.search(text):
        return True
    count = word_count(text)
    if _CJK.search(text):
        return count >= 5
    words = _lower_words(text)
    if count >= 3 and any(word in _ARTICLES or word in _CLAUSE_WORDS for word in words):
        return True
    return count >= 5 and any(word in _LINKING_WORDS for word in words)


def _is_sentence(text: str) -> bool:
    """Prose no Danbooru name could be, so a tag lookup could not know it.

    Stricter than `_is_prose`: a title keeps its articles ("the king of
    fighters"), and only a clause word, a sentence break, a closing full stop
    after an opening article or more words than any tag runs to rule a name out.
    """
    if not _is_prose(text):
        return False
    return (
        bool(_SENTENCE_BREAK.search(text))
        or word_count(text) > _LONGEST_TAG
        or any(word in _CLAUSE_WORDS for word in _lower_words(text))
        or bool(_ARTICLE.match(text) and _SENTENCE_END.search(text))
    )


def _pieces(text: str) -> list[str]:
    """The pieces the style is read from: split at commas, fullwidth commas
    and line breaks, where `comma_items` - the chips' split - takes commas
    alone. A list written one tag per line or with "，" reads as tags here and
    is one comma item there."""
    return [piece.strip() for piece in re.split(r"[,，\n]", text) if piece.strip()]


def detect_style(prompt: str, categories: dict[str, str] | None = None) -> Style:
    """How the text reads: sentences, a comma tag list, both, XML or JSON.

    By the share of its words that sit in phrase-length comma pieces: a
    caption's list of attributes ("black horns, and a long, sinuous tail")
    makes short pieces too, so a mixed text that ends like a sentence and is
    full of articles and prepositions is still prose. With `categories` -
    `check`'s own - a piece the tag index knows reads as a tag, so the style
    reported is the one the rules applied.
    """
    return _detect_style(prompt, _category_lookup(categories))


def _detect_style(prompt: str, known: dict[str, str] | None) -> Style:
    """`detect_style`, with the pieces the tag index knows read as tags
    whatever their shape: "the legend of zelda" is a series, not a phrase."""

    def listed(piece: str) -> bool:
        return bool(known) and (piece in known or _norm(piece) in known)

    def phrase(piece: str) -> bool:
        return _is_prose(piece) and not listed(piece)

    text = prompt.strip()
    if text[:1] in ("{", "["):
        # A few thousand nested brackets fit inside the 4000-character limit
        # and exhaust the parser's recursion before it can say "not JSON".
        try:
            parsed = json.loads(text)
        except (ValueError, RecursionError):
            parsed = None
        if isinstance(parsed, (dict, list)) and parsed:
            return "json"
    if _XML_ELEMENT.search(text) or _XML_START.match(text):
        return "xml"
    pieces = _pieces(text)
    if len(pieces) <= 1:
        single = pieces[0] if pieces else ""
        tag = listed(single) or (
            single
            and not _is_prose(single)
            and word_count(single) <= 3
            and not _ARTICLE.match(single)
        )
        return "tags" if tag else "prose"
    phrase_words = tag_words = 0
    for piece in pieces:
        words = word_count(piece)
        if phrase(piece):
            phrase_words += words
        else:
            tag_words += words
    total = phrase_words + tag_words
    if not total:
        return "prose"
    share = phrase_words / total
    if share >= 0.75:
        return "prose"
    if share <= 0.25:
        return "tags"
    words = [word.lower() for word in _WORD.findall(_CJK.sub(" ", text))]
    function = sum(1 for word in words if word in _FUNCTION_WORDS)
    if words and function / len(words) >= 0.15 and _SENTENCE_END.search(text):
        return "prose"
    return "hybrid"


# Quoted stretches, in the marks a prompt uses for text to render. A single
# quote counts only as a pair that does not touch a Latin letter or a digit,
# so "animal's" is an apostrophe and 'Black Forest Labs' a quotation. Han,
# kana and Hangul may touch it: Chinese, Japanese and Korean put no space
# around a quotation (写着'营业中'的招牌), and ERNIE's own rewriter wraps
# lettering in English single quotes. The one definition: the critic's
# reorder and rewrite checks and the encoder layer's weighting scan read
# quoted text with it too.
_WORD_NOT_CJK = (
    r"[^\W\u1100-\u11ff\u3040-\u30ff\u3130-\u318f\u3400-\u4dbf\u4e00-\u9fff"
    r"\uac00-\ud7af\uf900-\ufaff]"
)
QUOTED = re.compile(
    r'"[^"\n]*"|“[^”\n]*”|„[^“”\n]*[“”]|「[^」\n]*」|『[^』\n]*』|«[^»\n]*»|‘[^’\n]*’'
    rf"|(?<!{_WORD_NOT_CJK}|')'[^'\n]+'(?!{_WORD_NOT_CJK}|')"
)
_QUOTE_OPEN = "\"'“„「『«‘"


def _quoted_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in QUOTED.finditer(text)]


def _inside(spans: list[tuple[int, int]], position: int) -> bool:
    return any(start <= position < end for start, end in spans)


def _units(text: str, items: list[Item], lettering: list[tuple[int, int]]) -> list[Item]:
    """`items` with each stretch of lettering one item, its commas and all.

    "hello, world" on a sign is one text to render, not two tags: split, the
    order rule placed its halves apart and a rating or artist rule rewrote a
    piece that happened to be a tag ("hi, safe, bye"). As one item it moves
    whole, keeps the spacing inside it, and is read as no tag (`_Context.norms`).
    """
    spans = sorted(lettering)
    units: list[Item] = []
    reach = index = 0
    for item in items:
        # How far the lettering that starts before this item reaches. It
        # starts and ends on a mark or a word, never in the commas and spaces
        # between two items: past the item before, it runs on into this one.
        while index < len(spans) and spans[index][0] < item.start:
            reach = max(reach, spans[index][1])
            index += 1
        if units and reach > units[-1].end:
            first = units[-1]
            units[-1] = Item(text[first.start:item.end], first.start, item.end)
        else:
            units.append(item)
    return units


def _held(items: list[Item], spans: list[tuple[int, int]]) -> list[bool]:
    """Whether each of `items`, in order, holds any of `spans` - lettering -
    in one pass, as `_units` merges: a prompt can be a thousand quotations."""
    ordered = sorted(spans)
    held: list[bool] = []
    reach = index = 0
    for item in items:
        while index < len(ordered) and ordered[index][0] < item.end:
            reach = max(reach, ordered[index][1])
            index += 1
        held.append(reach > item.start)
    return held


def _blank(text: str, spans: list[tuple[int, int]], fill: str = " ") -> str:
    """`text` with `spans` blanked, every offset kept."""
    chars = list(text)
    for start, end in spans:
        chars[start:end] = fill * (end - start)
    return "".join(chars)


def _norm(tag: str) -> str:
    """A comma item as comparable text: case, underscores, the escaped
    parentheses the tag autocomplete writes and a sentence's closing full stop
    ("..., masterpiece.") do not make it another tag."""
    text = tag.strip().lower().replace(r"\(", "(").replace(r"\)", ")").replace("_", " ")
    return re.sub(r"(?<=\w)\.$", "", re.sub(r"\s+", " ", text).strip())


_YEAR_TAG = re.compile(r"year \d{4}")


def _separator(prompt: str) -> str:
    """The comma style the prompt already uses, for anything added to it."""
    commas = re.findall(r",([ \t]?)", prompt)
    spaced = sum(1 for gap in commas if gap)
    return ", " if not commas or spaced * 2 >= len(commas) else ","


def _rebuild(prompt: str, items: list[Item], texts: list[str]) -> str:
    """The prompt with its comma items replaced by `texts`, in that order.

    Whatever sat between the original items - ", ", ",\\n", a stray double
    space - stays where it was, position for position, so a reorder changes
    the order and nothing else. Items beyond the original count take the
    prompt's own separator; fewer items drop the trailing separators.
    """
    if not items:
        return prompt
    gaps = [prompt[a.end:b.start] for a, b in pairwise(items)]
    default = _separator(prompt)
    out = [prompt[: items[0].start]]
    for index, text in enumerate(texts):
        if index:
            out.append(gaps[index - 1] if index - 1 < len(gaps) else default)
        out.append(text)
    out.append(prompt[items[-1].end:])
    return "".join(out)


def _category_lookup(categories: dict[str, str] | None) -> dict[str, str] | None:
    if categories is None:
        return None
    lookup = {_norm(text): category for text, category in categories.items() if category}
    lookup.update({text: category for text, category in categories.items() if category})
    return lookup
_NEGATION = re.compile(
    r"\b(?P<head>do\s+not|does\s+not|don['’]t|doesn['’]t|without|never|avoid|no|not)"
    r"(?:[ \t]+[^\s,.;:!?()\[\]\"“”]+){1,3}",
    re.IGNORECASE,
)
# A name running on from where a negated phrase's window ends, a step at a
# time: a capitalised word ("without ropes on El Capitan" is not cut at
# "El"), one after a lowercase particle ("Rio de Janeiro", "Ludwig van
# Beethoven"), or one after the full stop of an abbreviation that opens a
# name (`_NAME_ABBREVIATION`: "St. Ives", "Dr. Smith").
_NAME_WORD = r"[A-ZÀ-ÖØ-Þ][^\s,.;:!?()\[\]\"“”]*"
_NAME_STEP = re.compile(
    r"[ \t]+(?:(?:de|da|do|dos|das|del|della|di|du|la|le|van|von|der|den)[ \t]+)?" + _NAME_WORD
)
_ABBREVIATION_STEP = re.compile(r"\.[ \t]*(" + _NAME_WORD + ")")
# St. as well as the titles `_MID_CLAUSE` keeps, since in a name it is
# Saint. Only where it opens the name - after the head or a lowercase word
# ("Avoid Dr. Smith", "no boats in St. Ives") - since after a capitalised
# word it is a street, which ends a sentence as often ("no cars on Elm Dr.
# Lovely day").
_NAME_ABBREVIATION = re.compile(r"(?<![\w'’-])(?:Dr|Mr|Mrs|Ms|Mt|Prof|St)$")
# Where a head sits inside a clause rather than opening one: right after a
# word ("trying to avoid the rain", "the film No") or an abbreviation that
# goes before a name, spaced or not ("Dr. No", "Dr.No"). Not one that comes
# after a name, which ends a sentence as often ("a photo taken on Main St.
# No cars.", "Martin Luther King Jr. No crowds."). Anything else opens one -
# the text's start, a line break, any other full stop or clause mark, a
# dash, a bullet, a slash - and so does a word that only softens an
# instruction ("please avoid text").
_MID_CLAUSE = re.compile(
    r"(?:(?<![\w'’-])(?!(?i:please|also)[ \t])[^\W\d_][\w'’-]*[ \t]+"
    r"|\b(?:Dr|Mr|Mrs|Ms|Mt|Prof)\.[ \t]*)$"
)
# Words a negated phrase does not end on: "no people in the background" is
# about the people.
_TRAILING = frozenset(
    "a an the in on at of to with and or for from by into onto over under any is are".split()
)
_IDIOMS = ("no longer", "no matter", "no doubt", "no less", "no more than", "not only", "not just")
# A lowercase "no" between two capitalised words is the Japanese particle of
# a romanised title or name - Sousou no Frieren, Kimi no Na wa - not a negation.
_WORD_BEFORE = re.compile(r"([^\W\d_][\w'’-]*)[ \t]+$")
_CAPITAL_AFTER = re.compile(r"[ \t]+[A-ZÀ-ÖØ-Þ]")


def _particle(text: str, match: re.Match[str]) -> bool:
    if match.group("head") != "no":
        return False
    before = _WORD_BEFORE.search(text, max(0, match.start() - 64), match.start())
    return bool(
        before and before.group(1)[0].isupper() and _CAPITAL_AFTER.match(text, match.end("head"))
    )


def _opens_clause(text: str, position: int) -> bool:
    return not _MID_CLAUSE.search(text, max(0, position - 64), position)


def _not_absence(text: str, match: re.Match[str]) -> bool:
    """A negation head that asks for no absence: "avoid" anywhere but as an
    instruction opening a clause ("a cat trying to avoid the rain" is drawn
    avoiding it), and a capital "No" in mid-clause, where only a name has one
    ("the film Dr. No", "a poster for No Country for Old Men"). A clause's
    own "No" is capitalised by grammar, so what follows it says nothing: "No
    English text" is an absence, and so, at a clause's start, is a title."""
    head = match.group("head")
    if head.lower() != "avoid" and head != "No":
        return False
    return not _opens_clause(text, match.start())


def _absence(text: str, match: re.Match[str]) -> tuple[int, int, str] | None:
    """The absence a `_NEGATION` match asks for, as (start, end, its words
    lowercased), or None where it asks for none: a particle, a name, "avoid"
    as an action, an idiom, or a head with only trailing words after it."""
    if _particle(text, match) or _not_absence(text, match):
        return None
    head = len(match.group("head").split())
    words = list(re.finditer(r"\S+", match.group()))
    while len(words) > head and words[-1].group().lower() in _TRAILING:
        words.pop()
    lowered = " ".join(word.group().lower() for word in words)
    if len(words) <= head or lowered.startswith(_IDIOMS):
        return None
    start, end = match.start(), match.start() + words[-1].end()
    if _capitalised(words[-1].group()):
        end = _name_end(text, start, end)
    return start, end, lowered


def _name_end(text: str, head: int, end: int) -> int:
    """Where a name that ends the window of a negated phrase starting at
    `head` runs on to, from `end` (`_NAME_STEP`, `_NAME_ABBREVIATION`)."""
    while True:
        step = _NAME_STEP.match(text, end)
        if step is None:
            abbreviation = _NAME_ABBREVIATION.search(text, max(head, end - 4), end)
            if abbreviation is None:
                return end
            before = _WORD_BEFORE.search(text, head, abbreviation.start())
            if before and before.start(1) != head and _capitalised(before.group(1)):
                return end
            step = _ABBREVIATION_STEP.match(text, end)
            # Nor into a word opening an absence of its own: "no boats in
            # St. No cars." is two, where "Avoid Dr. No" is the film.
            if step is None or _absence_at(text, step.start(1)):
                return end
        end = step.end()


def _absence_at(text: str, position: int) -> bool:
    """Whether the negation rule's own reading of `text` finds an absence
    starting at `position` - not inside a negation that starts earlier ("do
    not enter", "no pain no gain"), and not a head it reads nothing from
    ("now or never")."""
    for match in _NEGATION.finditer(text):
        if match.end() > position:
            return match.start() == position and _absence(text, match) is not None
    return False


# `verb` triggers are ordinary verbs that only now and then introduce
# lettering; "titled", "spells out" and the nouns always do.
_TEXT_TRIGGER = re.compile(
    r"\b(?:(?P<verb>reads|reading|says|saying|said|labell?ed(?:[ \t]+as)?"
    r"|spells(?![ \t]+out\b)|spelling(?![ \t]+out\b))"
    r"|titled|entitled|spells[ \t]+out|spelling[ \t]+out"
    r"|(?P<noun>(?:the|with)[ \t]+(?:words?|text|letters|lettering|phrase|slogan|caption"
    r"|headline|inscription)))\b[ \t]*:?[ \t]*",
    re.IGNORECASE,
)
_CJK_TEXT_TRIGGER = re.compile(r"(?:写着|写有|印着|印有|标着|标有)[:：]?")
# A verb trigger is text only on something that carries text, earlier in
# the same sentence: a sign reading OPEN, a mug that says MONDAY, a jar
# labelled SUGAR - not a girl reading a book, a mother saying goodbye or a
# wizard casting spells. Places a speaker stands (a door, a window, a wall)
# and clothes other than the shirt are left out: "a man in a hat says hello"
# is speech.
_TEXT_BEARER = re.compile(
    r"(?i)\b(?:signs?|signboard|signpost|board|banner|poster|flyer|leaflet|label|screen"
    r"|display|plaque|card|postcard|note|letter|envelope|message|text|tag|shirt|t-shirt"
    r"|mug|cup|bottle|jar|box|package|crate|vial|potion|cover|book|notebook|magazine"
    r"|newspaper|billboard|marquee|neon|letters|lettering|headline|caption|title|logo"
    r"|placard|chalkboard|whiteboard|blackboard|menu|sticker|badge|page|paper|scroll"
    r"|monitor|tattoo|cake|bubble|balloons?|flag|stamp|ticket|receipt|certificate|trophy"
    r"|tombstone|gravestone|headstone|doormat|graffiti)(?:e?s)?\b[^.!?\n]*$"
)
# "labeled" as an adjective - "a diagram with labeled parts", "a labeled
# map" - names no text; after a noun it names the text: a jar labeled SUGAR.
_ATTRIBUTIVE = re.compile(r"(?i)\b(?:a|an|the|with|and|or|of|its|their|some|many)[ \t]+$")
# A trigger that can be a noun - an -ing form, "spells" - is one after a
# determiner or a preposition: "a book of spells", "with reading glasses",
# "the reading". Not before a capital, which is lettering on something near
# a person ("a sign above her reading OPEN": "her" is also an object), and
# never a finite verb ("a sign above her reads OPEN").
_NOUN_USE = re.compile(
    r"(?i)\b(?:a|an|the|of|with|its|their|his|her|my|your|our|some|many)[ \t]+$"
)
# After "the text", a word that makes it a noun rather than a quotation:
# "the text is blurry".
_NOT_TEXT = frozenset(
    "is are was were be been appears appear reads says in on at of that which should "
    "will can has have and or with from to it its below above".split()
)
# Words that end the text to render: where it is and how it is written.
_TARGET_STOP = frozenset(
    "in on at above below under over beside next while written painted printed "
    "across along near behind inside appears glowing".split()
)
# Words that may end it or belong to it: "says OPEN and a cat sleeps" against
# "says Fish and Chips", "reads Welcome to Paradise". A capital after one
# carries the text on, as a name or a title would; anything else ends it
# there, and where it ends is then a guess.
_TARGET_JOIN = frozenset("and & with as from for to".split())
_TARGET_WORDS = 6
_TARGET_WORD = re.compile(r"[ \t]*([^\s,.;:!?\"“”()]+)")
# Words that open a description of the text rather than the text itself -
# "with the words printed across the chest", "the text carved into the
# wood", "a newspaper reading without glasses": how it is made, where it
# sits, whose it is. Only in lowercase: "reads Into the Woods" is a title.
_DESCRIBES = _TARGET_STOP | frozenset(
    "a an the this these those his her my your our their into onto upon about around "
    "through by without within beneath underneath between against toward towards via "
    "carved etched embossed engraved inscribed handwritten hand-written hand-painted "
    "stenciled stencilled scrawled scribbled stamped embroidered stitched sewn burned "
    "burnt spray-painted drawn displayed shown visible faded fading floating flashing "
    "overlay overlaid superimposed".split()
)
# An adverb, which describes too where it leads into one of those or ends
# the clause: "the words neatly printed", "reading quietly".
_ADVERB = re.compile(r"[a-z]{3,}ly")
# After a noun trigger, a participle with a preposition of place after it
# describes the words rather than being them: "with the words shimmering
# across the chest", "the text piped in blue icing", "the letters dripping
# down the bricks" - an irregular one too ("the headline set in bold type",
# "the words lit up at night", "the text woven into it"). So does an -ing
# form the clause ends on or an adverb follows ("with the words flickering",
# "flickering softly"), though not an adverb of when, which a sign says
# ("opening shortly", "closing early today"); not an -ed one, which a sign
# says as often ("with the words closed", "closed temporarily"). Not after
# "says", whose words are what it says.
_PARTICIPLE = re.compile(r"[a-z]{3,}(?:ing|ed)")
_IRREGULAR_PARTICIPLE = frozenset(
    "set lit woven spun cut hung strung wound bent knit torn worn held made laid spelt "
    "blown frozen hidden stuck struck cast split spread".split()
)
_PLACE = frozenset(
    "in on at above below under over beside next across along near behind inside into "
    "onto upon through around down out up against beneath underneath from".split()
)
_WHEN_ADVERB = frozenset("early shortly daily weekly nightly hourly monthly yearly".split())
# Heads of a negated phrase, which lowercase words after a participle
# trigger do not run on through where the negation rule reads an absence
# from one (`_absence_at`): "a boy on a box reading comics without pictures"
# is the negation rule's, "a sign reading do not enter" is lettering whole.
_NEGATING = frozenset("without never no not".split())
# Words whose phrase a negation right after them closes - "now or never",
# "better late than never", "ready or not", "yes or no". Lettering stopped
# there would end on "or" or "than", and the absence the negation rule reads
# from it is where the lettering sits ("never in bold type", "never on the
# door"): the negation is the lettering's own.
_CLOSED_BY_NEGATION = frozenset(("or", "than"))
# Words that make a negation right after them part of a sentence the
# lettering says, rather than an absence the prompt asks for: a verb it
# negates ("we are not open", "we will never forget", "it's not a drill"),
# "but" ("gone but not forgotten") and a verb of saying, whose object it is
# ("say no to war"). A negation among the words already read makes a later
# one the same lettering's too ("no shoes no shirt no service").
_NEGATED_VERB = frozenset(
    "am is are was were be been being will would can could shall should must may might "
    "do does did have has had but say says said saying".split()
)
_CONTRACTED_VERB = re.compile(r"['’](?:m|re|s|ve|ll|d)$")


def _owns_negation(words: list[re.Match[str]]) -> bool:
    """Whether the lowercase words a participle's lettering has so far make
    a negation right after them their own (`_CLOSED_BY_NEGATION`,
    `_NEGATED_VERB`, or a negation among them)."""
    last = words[-1].group(1).lower()
    return (
        last in _CLOSED_BY_NEGATION
        or last in _NEGATED_VERB
        or bool(_CONTRACTED_VERB.search(last))
        or any(word.group(1).lower() in _NEGATING for word in words)
    )
# Chinese lettering has no spaces to end it. What ends it for certain is
# whitespace, punctuation, a quotation mark or a bracket - though not an
# ASCII mark inside Latin lettering or digits (营业时间9:00-21:00, 3.14,
# www.shop.com). 的 and 之 link it to what it describes (写着欢迎光临的木质招牌,
# a wooden sign reading ...), so a target stopped there - or by the length
# cap, with more text after - is a guess, unless the script changes there
# (写着OPEN的招牌).
_CJK_BREAK = r"\s，。、；：！？,.;:!?\"'“”‘’「」『』（）()《》【】"
_CJK_TARGET = re.compile(rf"[^{_CJK_BREAK}的之]{{1,12}}")
_CJK_TARGET_END = re.compile(rf"[{_CJK_BREAK}]|$")
# Latin lettering runs on past a space (写着OPEN NOW的招牌) or an ASCII mark
# with a letter or a digit after it (营业时间9:00).
_LATIN_ON = re.compile(r"[ \t]+[A-Za-z0-9]|[,.;:!?'][A-Za-z0-9]")
# Where description starts inside a run that punctuation ends. 字样 names the
# lettering before it (写有欢迎字样, "bearing the word 欢迎"), and so does a
# count of its characters (写着福两个大字), so each ends it for certain - but a
# 等 before one is "and so on" or the end of a word (平等字样), a guess. A
# verb that puts the sign somewhere, with its 在 or 于, starts describing it
# (写着营业中挂在门口, hung by the door) - a guess like 的, since lettering can
# hold one - but not where a request or a prohibition asks for it
# (`_CJK_ASKS`), and never 在 alone: 正在营业, 现在, 在线.
_CJK_STOP = re.compile(
    r"(?P<etc>等)?(?:字样|[一两二三四五六七八九十几]个大?字)"
    r"|(?P<placed>(?:悬挂|张贴|摆放|停放|存放|安放|堆放|放置|陈列|矗立|竖立|镶嵌|悬浮"
    r"|[挂贴放摆立悬钉刻嵌插停站坐躺靠飘浮竖])[在于]|位于)"
)
# A request or a prohibition right before such a verb makes it the sign's
# own words (请站在黄线外, 请勿随意停放在此, 禁止停放在此), and so does one
# with 将 or 把 and a thing of one or two characters between (请将车辆停放在
# 指定区域). Not one further back: 请勿吸烟挂在墙上 and 请把门关好挂在门上 are
# signs on a wall and a door.
_CJK_ASKS = re.compile(
    r"(?:请您?|勿|别|禁止?|不要|不准|不得|不许)(?:随意|随便|随地|擅自|私自|乱)?"
    r"(?:[将把].{1,2})?$"
)
# A title in its marks is the lettering, marks and all: 写着《西游记》的封面.
_CJK_TITLE = re.compile(r"《[^《》\n]{1,30}》")


class _Target(NamedTuple):
    """Words that read as text to render, whether their extent is sure, and
    whether the lettering may run on past them."""

    start: int
    end: int
    # False where the text's extent is a guess - the words run on past a
    # joining word, a capitalised stop word, punctuation or the word limit,
    # a Chinese target stops at 的, at 等
    # before 字样, at a verb placing the sign (挂在), at its length cap or
    # inside Latin lettering, lowercase words follow a participle, or words
    # not in capitals follow a noun trigger ("with the words happy birthday
    # piped in icing"): the finding then offers no one-click quote, which
    # would letter half a name or a description.
    exact: bool
    # False where the lettering may run on past `end` - a joining word, a
    # capitalised stop word, the word or length cap, lettering going on past
    # a space or a mark, or a negation that stopped a participle's lowercase
    # words ("reading love without limits", which `rendered_targets` reads
    # on through) - so a
    # reader may take it further. True where what follows is
    # read as description, even where `exact` is not: at 的, 等 or 挂在 the
    # guess is whether it is the lettering's own (写着我的家), and after a
    # participle whether the words are lettering at all; neither makes what
    # follows lettering. After a noun trigger, words not in capitals are that
    # same guess, and their end is as sure as it was.
    ends: bool
    # Where the words the rules read as lettering end: `end` itself, but
    # for a participle's lowercase words read on through a negation the
    # rules stop them at ("reading love | without limits"), the end of the
    # words before it. The lettering starts in those: "without limits" on
    # its own is the absence the negation rule reads, not a piece of it.
    head: int


def _capitalised(word: str) -> bool:
    return word[:1].isupper() or word[:1].isdigit()


def _describes(text: str, word: re.Match[str], noun: bool) -> bool:
    """Whether the first word after a trigger starts describing the text
    rather than being it (`_DESCRIBES`, an adverb leading into one, or after
    a noun trigger a participle placing it)."""
    first = word.group(1)
    if not first[:1].islower():
        return False
    if first.lower() in _DESCRIBES:
        return True
    after = _TARGET_WORD.match(text, word.end())
    spaced = text[word.end():word.end() + 1] in (" ", "\t")
    following = after.group(1).lower() if spaced and after else None
    if noun and (_PARTICIPLE.fullmatch(first) or first in _IRREGULAR_PARTICIPLE):
        if following in _PLACE:
            return True
        if first.endswith("ing") and (
            following is None
            or (_ADVERB.fullmatch(following) and following not in _WHEN_ADVERB)
        ):
            return True
    if not _ADVERB.fullmatch(first.lower()):
        return False
    return following is None or following in _DESCRIBES


def _in_capitals(words: list[re.Match[str]]) -> bool:
    """Whether no word has a lowercase letter - "HAPPY BIRTHDAY", "2024" -
    a joining word between two such aside ("FISH and CHIPS")."""
    return all(
        not any(c.islower() for c in word.group(1))
        or (i > 0 and word.group(1).lower() in _TARGET_JOIN)
        for i, word in enumerate(words)
    )


def _punctuation_continues(text: str, end: int) -> bool:
    """A mark inside lettering, rather than a certain end to quote at.

    Times, prices and addresses run straight through marks (9:00, $9.99,
    www.shop.com). A comma, colon or semicolon followed by another capital
    or number may continue the words too (HELLO, WORLD). A full stop before
    a spaced sentence still ends them (OPEN. A cat sleeps).
    """
    mark = text[end:end + 1]
    if not mark or mark not in ",.;:!?":
        return False
    if text[end + 1:end + 2].isalnum():
        return True
    after = _TARGET_WORD.match(text, end + 1)
    return mark in ",;:" and after is not None and _capitalised(after.group(1))


def _cjk_stop(text: str, start: int, end: int) -> re.Match[str] | None:
    """The first `_CJK_STOP` that starts inside the run `text[start:end]`,
    skipping a placing verb the sign asks for (`_CJK_ASKS`). None is longer
    than four, so one that starts at the run's last character is read whole
    even where the length cap cut the run there."""
    for stop in _CJK_STOP.finditer(text, start + 1, end + 3):
        if stop.start() >= end:
            break
        if not (stop.group("placed") and _CJK_ASKS.search(text, start, stop.start())):
            return stop
    return None


def _unquoted_targets(
    text: str, quoted: list[tuple[int, int]], *, stop_at_negations: bool = True
) -> list[_Target]:
    """Words that read as text to render but carry no quotation marks.

    With `stop_at_negations`, the rules' reading: a participle's lowercase
    words stop at a negation the negation rule reads there (`_absence_at`),
    which that rule warns on. Without, they run on through it to where they
    would stop without it - "reading love without limits | above the stage" -
    which is how far the lettering may reach (`rendered_targets`), and the
    target's `head` keeps where the rules stopped."""
    targets: list[_Target] = []
    for match in _TEXT_TRIGGER.finditer(text):
        if _inside(quoted, match.start()):
            continue
        before = text[max(0, match.start() - 40):match.start()]
        verb = match.group("verb")
        if verb and not _TEXT_BEARER.search(before):
            continue
        noun = bool(
            verb
            and (verb.lower().endswith("ing") or verb.lower() == "spells")
            and _NOUN_USE.search(before)
        )
        if verb and verb.lower().startswith("label") and _ATTRIBUTIVE.search(before):
            continue
        position = match.end()
        if position >= len(text) or text[position] in _QUOTE_OPEN:
            continue
        participle = bool(verb and verb.lower().endswith("ing"))
        # To punctuation, to the preposition that starts describing where the
        # text is (says hello world | in red letters), or to a joining word
        # with no capital after it (says Fish and Chips | and a cat).
        words: list[re.Match[str]] = []
        cursor = position
        exact = True
        negated = False
        continued = False
        head: int | None = None
        while True:
            word = _TARGET_WORD.match(text, cursor)
            if word is None:
                break
            token = word.group(1).lower()
            if (words and token in _TARGET_STOP) or (not words and token in _NOT_TEXT):
                # "MADE IN JAPAN" may continue through a place word. Keep
                # the finding, but never quote just "MADE" in one click.
                continued = bool(words and _capitalised(word.group(1)))
                exact = not continued
                break
            if not words and _describes(text, word, match.group("noun") is not None):
                break
            # Lowercase words after a participle may be what someone reads,
            # and a negation among them is the negation rule's - where that
            # rule reads one from there, so that the words it stops are not
            # half of "do not enter" or "no pain no gain", and not where the
            # words before make it their own (`_owns_negation`): right after
            # "or" or "than", which it closes ("now or never in bold type",
            # "better late than never on the door"), a verb it negates ("we
            # are not open"), or after a negation of their own ("no shoes no
            # shirt no service").
            if (
                head is None
                and words
                and participle
                and token in _NEGATING
                and not _capitalised(words[0].group(1))
                and not _owns_negation(words)
                and _absence_at(text, word.start(1))
            ):
                if stop_at_negations:
                    negated = True
                    break
                head = words[-1].end(1)
            if len(words) == _TARGET_WORDS:
                exact = False
                break
            # A joining word right after a negation among a participle's
            # words goes on with it: "say no to war", "we are not for sale".
            if (
                words
                and token in _TARGET_JOIN
                and not (participle and words[-1].group(1) in _NEGATING)
            ):
                after = _TARGET_WORD.match(text, word.end())
                spaced = text[word.end():word.end() + 1] in (" ", "\t")
                if not (spaced and after and _capitalised(after.group(1))):
                    # Capitals up to here are the lettering (says OPEN and a
                    # cat); lowercase words may run on (says fish and chips).
                    exact = all(w.group(1).isupper() for w in words)
                    break
            words.append(word)
            cursor = word.end()
            if cursor < len(text) and text[cursor] not in " \t":
                continued = _punctuation_continues(text, cursor)
                exact = not continued
                break
        # A joining word never ends the text: "says Fish and" is not a name.
        while words and words[-1].group(1).lower() in _TARGET_JOIN:
            words.pop()
        if not words or (noun and not words[0].group(1)[:1].isupper()):
            continue
        ends = exact
        # A participle belongs to whichever noun it follows - "a boy on a box
        # reading comics" - so lowercase words after one may be what someone
        # reads or says rather than lettering. Where a negation stopped them,
        # the lettering may run on through it ("a banner reading love without
        # limits"); read without that stop, the words end where they would
        # have ended anyway, and what follows is description.
        if participle and not _capitalised(words[0].group(1)):
            exact, ends = False, not (negated or continued)
        # After a noun trigger, words not in capitals may describe the text
        # rather than be it - "with the words happy birthday piped in icing",
        # "the text closed temporarily" - so only capitals are sure to be the
        # lettering.
        # Where they end is no less sure: `ends` stands.
        elif match.group("noun") and not _in_capitals(words):
            exact = False
        end = words[-1].end(1)
        targets.append(
            _Target(words[0].start(1), end, exact, ends, end if head is None else min(head, end))
        )
    for match in _CJK_TEXT_TRIGGER.finditer(text):
        if _inside(quoted, match.start()):
            continue
        title = _CJK_TITLE.match(text, match.end())
        if title:
            targets.append(_Target(*title.span(), True, True, title.end()))
            continue
        target = _CJK_TARGET.match(text, match.end())
        if target and text[match.end()] not in _QUOTE_OPEN:
            end = target.end()
            stop = _cjk_stop(text, target.start(), end)
            if stop:
                end = stop.start()
            latin = text[end - 1].isascii() and text[end - 1].isalnum()
            # At 的, 等 or a verb placing the sign the end is a guess, unless
            # the script changes there (写着OPEN的招牌).
            guess = text[end:end + 1] in ("的", "之") or bool(
                stop and (stop.group("etc") or stop.group("placed"))
            )
            ends = (
                stop is not None or guess or bool(_CJK_TARGET_END.match(text, end))
            ) and not (latin and _LATIN_ON.match(text, end))
            lettering = text[target.start():end]
            exact = ends and (not guess or (lettering.isascii() and lettering.isalnum()))
            targets.append(_Target(target.start(), end, exact, ends, end))
    return sorted(targets)


def _unquoted_text(text: str, quoted: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Spans of words that read as text to render but carry no quotation marks."""
    return [(target.start, target.end) for target in _unquoted_targets(text, quoted)]


def rendered_targets(text: str) -> list[tuple[int, int, bool, int]]:
    """Where `text` asks for lettering, as (start, end, ends, head), in
    order: its quoted stretches, marks included, and the words a rule reads
    as unquoted text to render. `ends` is False only where the rules stopped
    at a guess and the lettering may run on past `end` (`_Target.ends`: a
    joining or capitalised stop word, the word or length cap, lettering going
    on past a space or a mark); where it is True, whatever follows is the
    prompt's description.

    A participle's lowercase words run on here through a negation the rules
    stop them at, to where they would stop without it: "a banner reading
    love without limits above the stage" asks for "love without limits",
    and "above the stage" is description, as after any participle's words.
    The quote and negation rules keep that stop - "without limits" warns as
    an absence, and no one-click quote letters a guess - but the lettering
    may hold the negation, and a guess that ran on to the clause's end would
    pass the description off as lettering. `head` is that stop (the end of
    "love"), and `end` everywhere else: a piece of the lettering starts
    before it, so "without limits" alone is the absence the negation rule
    reads, never lettering.

    It is not whether the quote rule offers its one-click quote
    (`_Target.exact`): a Chinese target stopped at 的, at 等 before 字样 or at
    a verb placing the sign (挂在), lowercase words after a participle and,
    where their end is sure, words not in capitals after a noun trigger get
    none, yet nothing past them reads as lettering."""
    quoted = _quoted_spans(text)
    unquoted = [
        (t.start, t.end, t.ends, t.head)
        for t in _unquoted_targets(text, quoted, stop_at_negations=False)
    ]
    return sorted([(start, end, True, end) for start, end in quoted] + unquoted)


def rendered_spans(text: str) -> list[tuple[int, int]]:
    """Where `text` asks for lettering: its quoted stretches, marks included,
    and the words a rule reads as unquoted text to render, through a
    negation among a participle's words - `rendered_targets` without its
    `ends` flag or its `head`."""
    return [(start, end) for start, end, _, _ in rendered_targets(text)]
