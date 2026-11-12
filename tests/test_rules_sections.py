"""`docs/rules.md` stays readable an area at a time.

AGENTS.md asks for an area's part of `docs/rules.md` to be read before the
area changes. That is practical only while the parts are headed and short:
Frontend and CLI and Tests once ran about 4,800 lines each under one heading
(DEV#60). So every section - the lines between one heading and the next,
fenced code left out - is held under a length here, and the anchors other
documents link to keep resolving.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Non-blank lines outside fenced code. The regrouped sections (DEV#60) run
# about 20 to 300 lines; the longest was 300 (Frontend's Gallery scope,
# Re-render, search and paging) when this was set. 350 leaves room for a few
# more rules before a section has to be split under a heading of its own,
# and is still a part that can be read in one sitting.
LIMIT = 350

# Sections DEV#60 did not regroup, held at their length when the limit was
# set: they may shrink, never grow. One that grows is split under ####
# headings by topic (move whole bullets; never reword a rule), and its entry
# goes when it fits the limit.
LONGER = {
    "### Model and loader": 791,
    "### Reproduction": 690,
    "### Jobs": 684,
    "### Prompt analyser": 412,
}

# Anchors into rules.md from documents whose changes do not run the tests
# (tools/ci_changes.py: only the tested prose does): README's Module map is
# read from the scan below; these are records that link here too.
LINKED_ELSEWHERE = {
    "layout": "docs/maintainer/review-2026-10-12.md",
    "rules": "docs/design/research.md",
}


def _rules() -> str:
    return (ROOT / "docs" / "rules.md").read_text(encoding="utf-8")


def _sections(text: str) -> list[tuple[str, int]]:
    """Each heading (the document's title first) with its section's length:
    the non-blank lines before the next heading, fenced code left out."""
    sections: list[tuple[str, int]] = []
    heading, count, fenced = "(before the first heading)", 0, False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        if line.startswith("#"):
            sections.append((heading, count))
            heading, count = line.strip(), 0
        elif line.strip():
            count += 1
    sections.append((heading, count))
    return sections


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, punctuation but hyphens
    dropped, spaces as hyphens."""
    words = heading.lstrip("#").strip().lower()
    return re.sub(r"[^\w\- ]", "", words).replace(" ", "-")


def test_no_section_runs_past_the_limit():
    long = {
        heading: count
        for heading, count in _sections(_rules())
        if count > LONGER.get(heading, LIMIT)
    }
    assert long == {}, (
        f"split these under headings of their own (limit {LIMIT} lines, "
        "fenced code aside; move whole bullets or paragraphs, never reword)"
    )


def test_the_longer_sections_are_still_there_and_still_longer():
    # An allowance outliving its section, or one the section now fits
    # without, would let the next section grow unseen.
    counts = dict(_sections(_rules()))
    assert [h for h in LONGER if h not in counts] == []
    assert [h for h in LONGER if counts[h] <= LIMIT] == []


def test_the_counting_skips_fenced_code_and_blank_lines():
    text = "# Title\n\nOne\n\n```\n# not a heading\ncode\n```\nTwo\n## Next\nThree\n"
    assert _sections(text) == [("(before the first heading)", 0), ("# Title", 2), ("## Next", 1)]


def test_headings_are_unique_so_anchors_are_stable():
    headings = [h for h, _ in _sections(_rules())[1:]]
    slugs = [_slug(h) for h in headings]
    assert sorted({s for s in slugs if slugs.count(s) > 1}) == []


def test_every_link_into_rules_resolves():
    anchors = {_slug(h) for h, _ in _sections(_rules())[1:]}
    tested = {
        "README.md": (ROOT / "README.md").read_text(encoding="utf-8"),
        "AGENTS.md": (ROOT / "AGENTS.md").read_text(encoding="utf-8"),
        "docs/verified.md": (ROOT / "docs" / "verified.md").read_text(encoding="utf-8"),
    }
    linked = [
        (anchor, name)
        for name, text in tested.items()
        for anchor in re.findall(r"rules\.md#([\w-]+)", text)
    ]
    # README's Module map: a scan finding nothing must not pass.
    assert ("layout", "README.md") in linked
    linked += [(anchor, name) for anchor, name in LINKED_ELSEWHERE.items()]
    assert [(a, where) for a, where in linked if a not in anchors] == []


def test_the_named_areas_have_headings():
    # The cross-references in rules.md and the phone review name these.
    headings = {h.lstrip("#").strip() for h, _ in _sections(_rules())[1:]}
    assert {"Layout", "Rules", "Frontend and CLI", "Phones and tablets", "Tests"} <= headings
