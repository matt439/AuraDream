"""The actual guide renders the checked-in source without exposing files or HTML."""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt

from backend import api_guide, main, passcode


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.ids = []
        self.links = []
        self.tags = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(values["id"])
        if tag == "a":
            self.links.append(values.get("href", ""))


def test_guide_renders_usage_with_tables_code_and_anchors(client, monkeypatch):
    monkeypatch.setattr(main.backend, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model load")))
    response = client.get("/guide")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-content-type-options"] == "nosniff"
    page = Page(response.text)
    assert {"table", "pre", "code", "main", "nav"} <= set(page.tags)
    assert {"using-auradream", "quick-questions", "a-short-glossary", "edit-by-instruction"} <= set(page.ids)
    assert len(page.ids) == len(set(page.ids))
    assert "capability-table:" not in response.text
    assert "/guide?file=" not in response.text
    assert "https://github.com/matt439/AuraDream/blob/main/docs/setup.md" in page.links
    assert client.get("/guide/README.md").status_code == 404
    assert client.post("/guide").status_code == 405


def test_guide_is_behind_the_passcode(client, monkeypatch):
    monkeypatch.setattr(main, "_passcode_gate", passcode.Gate("guide-check-9", port=main.settings.port))
    assert client.get("/guide").status_code == 401
    assert client.get("/guide", headers={"Authorization": "Bearer guide-check-9"}).status_code == 200


def test_renderer_escapes_html_and_unsafe_links_and_supports_duplicate_headings():
    page = api_guide.render_guide("# Title\n\n## Same `word`\n\n## Same word\n\n<script>alert(1)</script>\n\n[x](javascript:alert(1))\n\n[x](../README.md#settings)")
    parsed = Page(page)
    assert "script" not in parsed.tags
    assert "&lt;script&gt;" in page
    assert not any(link.startswith("javascript:") for link in parsed.links)
    assert {"same-word", "same-word-1"} <= set(parsed.ids)
    assert "https://github.com/matt439/AuraDream/blob/main/README.md#settings" in parsed.links


def test_every_repository_document_anchor_linked_by_the_usage_guide_exists():
    parser = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = parser.parse(api_guide.USAGE.read_text(encoding="utf-8"))
    checked = []

    def check(items):
        for token in items:
            if token.type == "link_open":
                href = unquote(str(token.attrGet("href") or ""))
                parts = urlsplit(href)
                if not parts.scheme and not parts.netloc and parts.fragment:
                    target = (api_guide.USAGE.parent / parts.path).resolve() if parts.path else api_guide.USAGE
                    assert target.is_relative_to(api_guide.ROOT) and target.is_file(), href
                    assert target.suffix == ".md", href
                    anchors = api_guide.heading_ids(parser.parse(target.read_text(encoding="utf-8")))
                    assert parts.fragment in anchors, href
                    checked.append(href)
            if token.children:
                check(token.children)

    check(tokens)
    assert len(checked) >= 12


def test_help_links_the_guide_without_navigating_the_composer():
    page = Page((Path(__file__).resolve().parent.parent / "web" / "index.html").read_text(encoding="utf-8"))
    assert "/guide" in page.links
