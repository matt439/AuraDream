"""The load-time guard uses JavaScript's own parser; it never boots the page."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_classic_scripts import script_paths

ROOT = Path(__file__).resolve().parent.parent


def _run(index: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_classic_scripts.py"), str(index)],
        capture_output=True, text=True, encoding="utf-8", timeout=40,
    )


def _page(tmp_path: Path, sources: dict[str, str], tags: str | None = None) -> Path:
    for name, source in sources.items():
        (tmp_path / name).write_text(source, encoding="utf-8")
    index = tmp_path / "index.html"
    index.write_text(
        tags if tags is not None else "\n".join(f'<script src="/static/{name}"></script>' for name in sources),
        encoding="utf-8",
    )
    return index


def test_actual_page_scripts_parse_and_share_one_scope():
    done = _run(ROOT / "web" / "index.html")
    assert done.returncode == 0, done.stdout + done.stderr
    count = len(script_paths(ROOT / "web" / "index.html"))
    assert f"{count} classic scripts" in done.stdout


def test_allowed_sharing_and_app_boot_are_never_executed(tmp_path):
    marker = tmp_path / "booted"
    index = _page(tmp_path, {
        "state.js": "let shared; var repeated; function helper() {}",
        "other.js": "var repeated; function helper() {} function readsShared() { return shared; }",
        "app.js": f"let appState = helper(); throw Error('booted'); require('node:fs').writeFileSync({str(marker)!r}, 'x');",
    })
    done = _run(index)
    assert done.returncode == 0, done.stdout + done.stderr
    assert not marker.exists()


@pytest.mark.parametrize("declaration", ["let name", "const name = 2", "class name {}", "var name", "function name() {}"])
def test_a_later_script_cannot_redeclare_an_earlier_lexical_binding(tmp_path, declaration):
    done = _run(_page(tmp_path, {"state.js": "let name;", "app.js": declaration}))
    assert done.returncode == 1
    assert "state.js" in done.stderr and "app.js" in done.stderr
    assert "already been declared" in done.stderr


@pytest.mark.parametrize("declaration", ["var name", "function name() {}"])
def test_a_later_lexical_cannot_redeclare_an_earlier_var_or_function(tmp_path, declaration):
    done = _run(_page(tmp_path, {"state.js": declaration, "app.js": "const name = 3;"}))
    assert done.returncode == 1
    assert "app.js" in done.stderr and "already been declared" in done.stderr


@pytest.mark.parametrize("bad_source", ["let broken = ;", '"use strict"; function bad(a, a) {}'])
def test_original_syntax_errors_name_the_source(tmp_path, bad_source):
    done = _run(_page(tmp_path, {"state.js": "let fine;", "app.js": bad_source}))
    assert done.returncode == 1
    assert "app.js" in done.stderr and "SyntaxError" in done.stderr


@pytest.mark.parametrize("strict", [False, True])
def test_each_scripts_own_strictness_controls_block_function_scope(tmp_path, strict):
    # Annex B's sloppy block function also declares a global var. A strict
    # script's block function stays block-scoped and may share the later name.
    source = ('"use strict"; ' if strict else "") + "if (false) { function name() {} }"
    done = _run(_page(tmp_path, {"state.js": source, "app.js": "let name;"}))
    assert done.returncode == (0 if strict else 1), done.stdout + done.stderr


def test_a_strict_first_script_does_not_make_the_next_script_strict(tmp_path):
    done = _run(_page(tmp_path, {
        "state.js": '"use strict"; let fine;',
        "other.js": "if (false) { function name() {} }",
        "app.js": "let name;",
    }))
    assert done.returncode == 1
    assert "app.js" in done.stderr and "already been declared" in done.stderr


def test_ordinary_strings_comments_and_hashbang_do_not_set_strict_mode(tmp_path):
    done = _run(_page(tmp_path, {
        "state.js": '#! a script\n/* "use strict" */ 0; "use strict"; if (false) { function name() {} }',
        "app.js": "let name;",
    }))
    assert done.returncode == 1
    assert "already been declared" in done.stderr


@pytest.mark.parametrize("tags,reason", [
    ('<script src="/static/missing.js"></script>', "does not exist"),
    ('<script type="module" src="/static/state.js"></script>', "classic script"),
    ('<script defer src="/static/state.js"></script>', "unexpected script attributes"),
    ('<script src="https://example.com/state.js"></script>', "local /static/"),
    ('<script src="/static/../outside.js"></script>', "outside the web directory"),
    ('<script src="/static/%2e%2e/outside.js"></script>', "outside the web directory"),
    ('<script>let inline;</script>', "inline script"),
    ('<script src="/static/state.js">throw 1</script>', "inline code"),
    ('<script src="/static/state.js" src="/static/other.js"></script>', "duplicate script attribute"),
    ('<script src="/static/state.js"/>', "closing tag"),
    ('<script src="/static/state.js">', "complete classic script"),
    ('<!-- <script src="/static/state.js"></script> -->', "complete classic script"),
])
def test_missing_or_unexpected_scripts_cannot_silently_reduce_coverage(tmp_path, tags, reason):
    done = _run(_page(tmp_path, {"state.js": "let fine;"}, tags))
    assert done.returncode == 1
    assert reason in done.stderr


def test_script_order_comes_from_html_with_comments_and_attribute_variations(tmp_path):
    index = _page(tmp_path, {"a.js": "", "b.js": ""}, """
        <!-- <script src="/static/missing.js"></script> -->
        <SCRIPT type='text/javascript' src='/static/b.js'></SCRIPT>
        <script type\n src = "/static/a.js" ></script>
    """)
    assert [path.name for path in script_paths(index)] == ["b.js", "a.js"]


def test_missing_node_fails_instead_of_skipping(tmp_path):
    index = _page(tmp_path, {"state.js": "let fine;"})
    done = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_classic_scripts.py"), str(index)],
        env={**os.environ, "PATH": ""}, capture_output=True, text=True, encoding="utf-8", timeout=40,
    )
    assert done.returncode == 1
    assert "Node is required" in done.stderr
