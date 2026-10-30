"""CI runs a commit once: a branch push leaves it to its open pull request's
run, and main's merge push skips a tree a passed pull request run tested.

The GitHub API is a dict of canned answers; a path it lacks raises, as an
unreachable API would, so every unasked call is a test failure.
"""
from __future__ import annotations

import http.client
import json
import urllib.error

import pytest

from tools import ci_duplicates
from tools.ci_duplicates import FULL_JOBS, covering_run, run_covers

REPO = "owner/repo"
BEFORE_SHA = "9" * 40
MAIN_SHA = "a" * 40
HEAD_SHA = "b" * 40
BRANCH_SHA = "c" * 40
RUN_URL = "https://github.com/owner/repo/actions/runs/7"
PR_URL = "https://github.com/owner/repo/pull/3"


class _Polls(list):
    """Successive answers to one path, as repeated polls see them."""


def _api(answers: dict):
    calls = []

    def get(path):
        calls.append(path)
        if path not in answers:
            raise urllib.error.URLError(f"unexpected {path}")
        answer = answers[path]
        return answer.pop(0) if isinstance(answer, _Polls) else answer
    get.calls = calls
    return get


def _push(ref: str, sha: str) -> dict:
    return {"ref": ref, "before": BEFORE_SHA, "after": sha,
            "repository": {"full_name": REPO, "owner": {"login": "owner"}}}


def _commit(tree: str) -> dict:
    return {"commit": {"tree": {"sha": tree}}}


def _jobs(*names: str, conclusion: str = "success") -> dict:
    return {"jobs": [{"name": name, "conclusion": conclusion} for name in names]}


# A merged pull request's runs list no pull requests (the API's own answer).
RUN = {"id": 7, "path": ".github/workflows/ci.yml", "event": "pull_request",
       "head_sha": HEAD_SHA, "head_branch": "feat/x",
       "head_repository": {"full_name": REPO}, "pull_requests": [], "html_url": RUN_URL}


def _merged(*, head_tree="t1", compare="ahead", run=None, jobs=None) -> dict:
    return {
        f"/repos/{REPO}/commits/{MAIN_SHA}": _commit("t1"),
        f"/repos/{REPO}/commits/{MAIN_SHA}/pulls?per_page=100": [
            {"number": 3, "head": {"sha": HEAD_SHA, "ref": "feat/x"}}],
        f"/repos/{REPO}/commits/{HEAD_SHA}": _commit(head_tree),
        f"/repos/{REPO}/compare/{BEFORE_SHA}...{HEAD_SHA}": {"status": compare},
        f"/repos/{REPO}/actions/runs?event=pull_request&status=success"
        f"&per_page=100&head_sha={HEAD_SHA}": {"workflow_runs": [run or RUN]},
        f"/repos/{REPO}/actions/runs/7/jobs?per_page=100": jobs or _jobs(*FULL_JOBS),
    }


def _main(get, full_tests=True):
    return covering_run("push", _push("refs/heads/main", MAIN_SHA), get, full_tests=full_tests)


def test_merge_push_skips_a_tree_its_pull_request_run_passed():
    assert _main(_api(_merged())) == RUN_URL


def test_merge_push_runs_when_main_moved_under_the_pull_request():
    assert _main(_api(_merged(head_tree="t2"))) is None


@pytest.mark.parametrize("compare", ["identical", "ahead"])
def test_merge_push_skips_only_a_head_that_contains_main(compare):
    assert _main(_api(_merged(compare=compare))) == RUN_URL


@pytest.mark.parametrize("compare", ["behind", "diverged"])
def test_merge_push_runs_when_the_pull_request_run_tested_another_merge(compare):
    # Same tree, but main's previous commit is outside the head: the run
    # tested the head merged with an older main, which may differ.
    assert _main(_api(_merged(compare=compare))) is None


@pytest.mark.parametrize("missing", sorted(FULL_JOBS))
def test_merge_push_runs_unless_every_full_job_passed(missing):
    jobs = _jobs(*(FULL_JOBS - {missing}))
    jobs["jobs"].append({"name": missing, "conclusion": "skipped"})
    assert _main(_api(_merged(jobs=jobs))) is None


def test_documentation_merge_needs_only_lint_to_have_passed():
    answers = _merged(jobs=_jobs("lint"))
    assert _main(_api(answers)) is None
    assert _main(_api(answers), full_tests=False) == RUN_URL


@pytest.mark.parametrize("change", [
    {"path": ".github/workflows/other.yml"},
    {"event": "push"},
    {"head_sha": "d" * 40},
    {"head_branch": "feat/other"},
    # A fork's run had no secrets.
    {"head_repository": {"full_name": "fork/repo"}},
    {"head_repository": None},
])
def test_merge_push_counts_only_this_workflows_run_for_that_pull_request(change):
    assert _main(_api(_merged(run={**RUN, **change}))) is None


def test_direct_push_to_main_without_a_pull_request_runs():
    get = _api({f"/repos/{REPO}/commits/{MAIN_SHA}": _commit("t1"),
                f"/repos/{REPO}/commits/{MAIN_SHA}/pulls?per_page=100": []})
    assert _main(get) is None


def _branch(pulls: list, *mergeable, head=BRANCH_SHA) -> dict:
    return {
        f"/repos/{REPO}/pulls?state=open&per_page=100&head=owner%3Afeat%2Fx": pulls,
        f"/repos/{REPO}/pulls/3": _Polls(
            {"head": {"sha": head}, "mergeable": value} for value in mergeable),
    }


def _branch_push(get, sleeps=None):
    return covering_run("push", _push("refs/heads/feat/x", BRANCH_SHA), get,
                        full_tests=True, sleep=([] if sleeps is None else sleeps).append)


def test_branch_push_with_an_open_pull_request_leaves_it_to_that_run():
    assert _branch_push(_api(_branch([{"number": 3, "html_url": PR_URL}], True))) == PR_URL


def test_branch_push_without_a_pull_request_runs():
    assert _branch_push(_api(_branch([]))) is None


def test_conflicted_pull_request_runs_no_workflow_so_the_push_runs():
    sleeps = []
    get = _api(_branch([{"number": 3, "html_url": PR_URL}], None, None, False))
    assert _branch_push(get, sleeps) is None
    assert sleeps == [ci_duplicates.MERGEABLE_WAIT] * 2


def test_mergeability_known_after_polls_leaves_it_to_the_pull_request():
    sleeps = []
    get = _api(_branch([{"number": 3, "html_url": PR_URL}], None, True))
    assert _branch_push(get, sleeps) == PR_URL
    assert sleeps == [ci_duplicates.MERGEABLE_WAIT]


def test_mergeability_still_unknown_after_the_polls_runs():
    # A conflicted pull request may still be computing: run, never neither.
    sleeps = []
    get = _api(_branch([{"number": 3, "html_url": PR_URL}], *[None] * ci_duplicates.MERGEABLE_POLLS))
    assert _branch_push(get, sleeps) is None
    assert len(sleeps) == ci_duplicates.MERGEABLE_POLLS - 1


def test_pull_request_still_showing_an_older_head_runs():
    # Its mergeable answers for the old head, not this push.
    get = _api(_branch([{"number": 3, "html_url": PR_URL}],
                       *[True] * ci_duplicates.MERGEABLE_POLLS, head="e" * 40))
    assert _branch_push(get) is None


@pytest.mark.parametrize("kind", ["pull_request", "workflow_dispatch", "schedule"])
def test_other_events_never_ask_the_api(kind):
    get = _api({})
    assert covering_run(kind, _push("refs/heads/main", MAIN_SHA), get, full_tests=True) is None
    assert get.calls == []


def test_tag_push_runs():
    get = _api({})
    assert covering_run("push", _push("refs/tags/v1", MAIN_SHA), get, full_tests=True) is None


def test_run_covers_requires_success():
    assert run_covers(_jobs(*FULL_JOBS)["jobs"], full_tests=True)
    assert not run_covers(_jobs(*FULL_JOBS, conclusion="failure")["jobs"], full_tests=True)
    assert not run_covers(_jobs("lint", conclusion="cancelled")["jobs"], full_tests=False)


def _run_main(tmp_path, monkeypatch, event: dict, get) -> list[str]:
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    output = tmp_path / "output.txt"
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GH_TOKEN", "token")
    monkeypatch.setenv("FULL_TESTS", "true")
    monkeypatch.setattr(ci_duplicates, "_api", lambda base, token: get)
    assert ci_duplicates.main() == 0
    return output.read_text(encoding="utf-8").splitlines()


def test_main_appends_the_duplicate_output(tmp_path, monkeypatch):
    lines = _run_main(tmp_path, monkeypatch, _push("refs/heads/main", MAIN_SHA), _api(_merged()))
    assert lines == ["duplicate=true"]


@pytest.mark.parametrize("event", [{}, {"ref": "refs/heads/main"}, _push("refs/heads/main", MAIN_SHA)])
def test_main_runs_when_the_event_or_api_fails(tmp_path, monkeypatch, event):
    assert _run_main(tmp_path, monkeypatch, event, _api({})) == ["duplicate=false"]


def test_main_runs_when_a_response_is_cut_short(tmp_path, monkeypatch):
    # http.client errors are neither OSError nor URLError.
    def get(path):
        raise http.client.IncompleteRead(b"")
    lines = _run_main(tmp_path, monkeypatch, _push("refs/heads/main", MAIN_SHA), get)
    assert lines == ["duplicate=false"]


def test_main_without_a_token_runs(tmp_path, monkeypatch):
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(_push("refs/heads/main", MAIN_SHA)), encoding="utf-8")
    output = tmp_path / "output.txt"
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert ci_duplicates.main() == 0
    assert output.read_text(encoding="utf-8").strip() == "duplicate=false"
