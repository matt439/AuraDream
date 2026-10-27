"""Check the page's ordered classic scripts without running their bodies.

HTMLParser owns the script inventory; Node's JavaScript parser and global
declaration instantiation own syntax and declaration-collision semantics.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent


class _Scripts(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sources: list[str] = []
        self.in_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "script":
            return
        if self.in_script:
            raise ValueError("nested script element")
        names = [name for name, _ in attrs]
        if len(names) != len(set(names)):
            raise ValueError("duplicate script attribute")
        options = dict(attrs)
        if set(options) - {"src", "type"}:
            raise ValueError(f"unexpected script attributes: {names}")
        if (options.get("type") or "").strip().lower() not in {"", "text/javascript", "application/javascript"}:
            raise ValueError(f"expected a classic script, got type={options['type']!r}")
        source = options.get("src")
        if not source:
            raise ValueError("expected a local external script, got an inline script")
        self.sources.append(source)
        self.in_script = True

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script":
            raise ValueError("a script must have a closing tag")

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self.in_script = False

    def handle_data(self, data: str) -> None:
        if self.in_script and data.strip():
            raise ValueError("an external script must not contain inline code")


def script_paths(index: Path) -> list[Path]:
    """All blocking local classic scripts, in the HTML's document order.

    Reject a changed script kind instead of quietly excluding it from the
    guard. `/static/` is the backend's mount of the index's own directory.
    """
    parser = _Scripts()
    parser.feed(index.read_text(encoding="utf-8"))
    parser.close()
    if parser.in_script or not parser.sources:
        raise ValueError("expected at least one complete classic script element")
    web = index.resolve().parent
    paths: list[Path] = []
    for source in parser.sources:
        url = urlsplit(source)
        if url.scheme or url.netloc or url.query or url.fragment or not url.path.startswith("/static/"):
            raise ValueError(f"expected a local /static/ script: {source!r}")
        relative = unquote(url.path.removeprefix("/static/"))
        if "\\" in relative:
            raise ValueError(f"nonportable script path: {source!r}")
        path = (web / relative).resolve()
        if not path.is_relative_to(web) or path.suffix != ".js":
            raise ValueError(f"script is outside the web directory or is not JavaScript: {source!r}")
        if not path.is_file():
            raise ValueError(f"script does not exist: {path}")
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", nargs="?", type=Path, default=ROOT / "web" / "index.html")
    args = parser.parse_args()
    try:
        paths = script_paths(args.index)
        node = shutil.which("node")
        if node is None:
            raise ValueError("Node is required for the classic-script check; install it and retry")
        scripts = [{"filename": str(path), "source": path.read_text(encoding="utf-8")} for path in paths]
        done = subprocess.run(
            [node, str(ROOT / "tools" / "check_classic_scripts.mjs")],
            input=json.dumps(scripts), text=True, encoding="utf-8", timeout=30,
        )
        return done.returncode
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"Classic-script check failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
