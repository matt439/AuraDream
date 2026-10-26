"""CI skips the CPU suite only when the entire event changes prose/assets.

Real throwaway Git histories cover the range and rename behavior: checking
only HEAD, or detecting a code-to-docs move as one documentation path, could
otherwise hide a code change behind a documentation-only tip.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.ci_changes import full_tests_required, is_documentation, main


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
        encoding="utf-8",
    ).stdout.strip()


def _commit(repo: Path, changes: dict[str, str | None]) -> str:
    for name, content in changes.items():
        path = repo / name
        if content is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    _git(repo, "add", "--all")
    _git(repo, "commit", "-m", "test change")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def history(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main", "--template=")
    _git(repo, "config", "user.name", "CI range tests")
    _git(repo, "config", "user.email", "ci-tests@example.invalid")
    _git(repo, "config", "commit.gpgsign", "false")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "config", "core.hooksPath", str(repo / ".git" / "no-hooks"))
    base = _commit(repo, {
        "README.md": "Original readme\n",
        "docs/guide.md": "Original guide\n",
        "backend/main.py": "print('original')\n",
    })
    return repo, base


def _event(kind: str, base: str, head: str) -> dict:
    if kind == "pull_request":
        return {"pull_request": {"base": {"sha": base}, "head": {"sha": head}}}
    return {"before": base, "after": head}


@pytest.mark.parametrize("path", [
    "README.md", "TODO.md", "AGENTS.md", "CONTRIBUTING.rst",
    "docs/usage.md", "docs/design/notes.rst", "docs/design/a spaced name.md",
    "docs/design/日本語.md", "docs/figure.png", "docs/figure.jpg",
    "docs/figure.jpeg", "docs/figure.gif", "docs/figure.svg",
    "docs/figure.webp", "docs/figure.avif", "docs/figure.pdf",
])
def test_prose_and_documentation_assets_are_documentation(path):
    assert is_documentation(path)


@pytest.mark.parametrize("path", [
    "backend/main.py", "tools/ci_changes.py", "tests/test_ci_changes.py",
    "web/app.js", "docs/prototype.js", "docs/prototype/index.html",
    "docs/verify.py", "docs/check.json", "docs/README.yml", "docs/env.sh",
    ".github/workflows/ci.yml", ".github/PULL_REQUEST_TEMPLATE.md",
    "pyproject.toml", "uv.lock", "README.png", "backend/README.md",
])
def test_executable_docs_and_other_paths_require_full_tests(path):
    assert not is_documentation(path)


@pytest.mark.parametrize("kind", ["push", "pull_request"])
def test_documentation_only_event_skips_full_tests(history, kind):
    repo, base = history
    head = _commit(repo, {"docs/design/a spaced name.md": "New prose\n"})
    assert not full_tests_required(kind, _event(kind, base, head), cwd=repo)


@pytest.mark.parametrize("kind", ["push", "pull_request"])
def test_code_in_earlier_commit_is_not_hidden_by_documentation_tip(history, kind):
    repo, base = history
    _commit(repo, {"backend/main.py": "print('changed')\n"})
    head = _commit(repo, {"README.md": "Documentation-only tip\n"})
    assert full_tests_required(kind, _event(kind, base, head), cwd=repo)


@pytest.mark.parametrize("path, full", [("docs/guide.md", False), ("backend/main.py", True)])
def test_deleted_paths_are_classified(history, path, full):
    repo, base = history
    head = _commit(repo, {path: None})
    assert full_tests_required("push", _event("push", base, head), cwd=repo) is full


def test_code_renamed_to_documentation_still_requires_full_tests(history):
    repo, base = history
    _git(repo, "mv", "backend/main.py", "docs/moved.md")
    head = _commit(repo, {})
    assert full_tests_required("push", _event("push", base, head), cwd=repo)


def test_pr_does_not_include_changes_only_on_the_base_branch(history):
    repo, common = history
    _git(repo, "checkout", "-b", "docs-pr")
    head = _commit(repo, {"docs/guide.md": "Changed on the PR\n"})
    _git(repo, "checkout", "main")
    base = _commit(repo, {"backend/main.py": "print('base only')\n"})
    assert base != common
    assert not full_tests_required("pull_request", _event("pull_request", base, head), cwd=repo)


@pytest.mark.parametrize("kind", ["workflow_dispatch", "schedule", "unknown"])
def test_manual_and_unknown_events_always_run_full_tests(history, kind):
    repo, base = history
    head = _commit(repo, {"README.md": "Docs only\n"})
    assert full_tests_required(kind, _event(kind, base, head), cwd=repo)


@pytest.mark.parametrize("kind,event", [
    ("push", {}),
    ("push", {"before": "0" * 40, "after": "1" * 40}),
    ("push", {"before": "missing", "after": "also-missing"}),
    ("pull_request", {}),
    ("pull_request", {"pull_request": {"base": {}, "head": {}}}),
])
def test_missing_or_unavailable_history_runs_full_tests(history, kind, event):
    repo, _ = history
    assert full_tests_required(kind, event, cwd=repo)


def test_empty_diff_runs_full_tests(history):
    repo, base = history
    assert full_tests_required("push", _event("push", base, base), cwd=repo)


def test_git_failure_runs_full_tests(tmp_path):
    assert full_tests_required(
        "push", {"before": "1" * 40, "after": "2" * 40}, cwd=tmp_path,
    )


def test_main_appends_the_gate_output(history, tmp_path, monkeypatch):
    repo, base = history
    head = _commit(repo, {"README.md": "Docs only\n"})
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(_event("push", base, head)), encoding="utf-8")
    output = tmp_path / "output.txt"
    output.write_text("existing=value\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert main() == 0
    assert output.read_text(encoding="utf-8").splitlines() == [
        "existing=value", "full_tests=false",
    ]


def test_main_with_unreadable_event_keeps_full_tests_enabled(tmp_path, monkeypatch):
    output = tmp_path / "output.txt"
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(tmp_path / "missing-event.json"))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert main() == 0
    assert output.read_text(encoding="utf-8").strip() == "full_tests=true"
