"""One read-only guide from the checkout, never a user-supplied file path."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from markdown_it import MarkdownIt
from markdown_it.token import Token

ROOT = Path(__file__).resolve().parent.parent
USAGE = ROOT / "docs" / "usage.md"
REPOSITORY_DOCS = "https://github.com/matt439/AuraDream/blob/main/docs/"
router = APIRouter()


def _plain(tokens: list[Token]) -> str:
    return "".join(
        _plain(token.children) if token.children else token.content
        for token in tokens if token.type in {"text", "code_inline", "image", "softbreak"}
    )


def heading_ids(tokens: list[Token]) -> list[str]:
    """GitHub-style ids for this guide's headings, including duplicate suffixes."""
    used: set[str] = set()
    result = []
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        inline = tokens[index + 1]
        words = _plain(inline.children or [])
        base = re.sub(r"[^\w\- ]", "", words.lower()).replace(" ", "-")
        anchor = base
        suffix = 0
        while anchor in used:
            suffix += 1
            anchor = f"{base}-{suffix}"
        used.add(anchor)
        token.attrSet("id", anchor)
        result.append(anchor)
    return result


def _guide_links(tokens: list[Token]) -> None:
    for token in tokens:
        if token.type == "link_open":
            href = str(token.attrGet("href") or "")
            parts = urlsplit(href)
            if not parts.scheme and not parts.netloc and parts.path:
                # The only local document route is /guide. Other repository
                # docs stay GitHub links, with no directory mount or file reader.
                target = urljoin(REPOSITORY_DOCS, href)
                if urlsplit(target).path == urlsplit(REPOSITORY_DOCS + "usage.md").path:
                    target = "/guide" + ("#" + parts.fragment if parts.fragment else "")
                token.attrSet("href", target)
        if token.children:
            _guide_links(token.children)


def render_guide(markdown: str) -> str:
    """Render curated Markdown with tables, escaped HTML and stable anchors."""
    parser = MarkdownIt("commonmark", {"html": False}).enable("table")
    # The generated capability table's comments are source markers, not copy.
    source = re.sub(r"^<!-- /?capability-table(?::[^\n]*)? -->\r?\n?", "", markdown, flags=re.MULTILINE)
    tokens = parser.parse(source)
    heading_ids(tokens)
    _guide_links(tokens)
    body = parser.renderer.render(tokens, parser.options, {})
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AuraDream guide</title>
<link rel="icon" href="/static/icons/icon-192.png">
<link rel="stylesheet" href="/static/style.css">
<link rel="stylesheet" href="/static/guide.css">
</head><body class="guide-page">
<nav aria-label="Guide navigation"><a href="/">AuraDream</a><a href="#using-auradream">Top</a></nav>
<main>""" + body + "</main></body></html>"


@router.get("/guide", response_class=HTMLResponse, include_in_schema=False)
def guide() -> HTMLResponse:
    # Read only on an explicit visit, so code/docs updates are reflected on
    # reload. This route never imports a model or touches the gallery.
    return HTMLResponse(render_guide(USAGE.read_text(encoding="utf-8")), headers={"Cache-Control": "no-cache"})
