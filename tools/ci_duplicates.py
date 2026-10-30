"""Skip a CI run whose commit another run of this workflow already covers.

Two events used to test the same work twice: a push to a branch with an
open pull request, which that pull request's own run tests as well, and the
push to main that merges a pull request whose run already passed the same
tree. Pull requests, manual runs, other events and any API failure run.
Standard library only: the lint job never installs the application.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

Get = Callable[[str], Any]

WORKFLOW = ".github/workflows/ci.yml"
# Every job a full run on a pull request into main passes; a documentation-
# only run passes lint alone.
FULL_JOBS = frozenset({"lint", "browser", "test (ubuntu-latest)", "test (windows-latest)"})
# GitHub computes a pull request's mergeability after a push; until then
# `mergeable` is null.
MERGEABLE_POLLS = 5
MERGEABLE_WAIT = 3.0


def _q(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def run_covers(jobs: list[dict], *, full_tests: bool) -> bool:
    passed = {job.get("name") for job in jobs if job.get("conclusion") == "success"}
    return FULL_JOBS <= passed if full_tests else "lint" in passed


def open_pull_request(get: Get, repo: str, owner: str, branch: str, sha: str, *,
                      sleep: Callable[[float], None] = time.sleep) -> str | None:
    """A branch push's open pull request whose own run will test it."""
    pulls = get(f"/repos/{repo}/pulls?state=open&per_page=100&head={_q(f'{owner}:{branch}')}")
    for pull in pulls:
        # Workflows never run on a pull request with a merge conflict, so it
        # covers the push only once it shows this commit as its head and
        # answers mergeable; stale or unknown after the polls, the push runs.
        for attempt in range(MERGEABLE_POLLS):
            answer = get(f"/repos/{repo}/pulls/{pull['number']}")
            current = answer["head"]["sha"] == sha
            if current and answer.get("mergeable") is not None:
                break
            if attempt < MERGEABLE_POLLS - 1:
                sleep(MERGEABLE_WAIT)
        if current and answer.get("mergeable") is True:
            return pull["html_url"]
    return None


def passed_pull_request_run(get: Get, repo: str, before: str, sha: str, *,
                            full_tests: bool) -> str | None:
    """A passed pull request run of this workflow over the same tree as `sha`."""
    tree = get(f"/repos/{repo}/commits/{sha}")["commit"]["tree"]["sha"]
    for pull in get(f"/repos/{repo}/commits/{sha}/pulls?per_page=100"):
        head = pull["head"]["sha"]
        if get(f"/repos/{repo}/commits/{head}")["commit"]["tree"]["sha"] != tree:
            continue
        # The run tested GitHub's merge of the head with main as it was then.
        # With main's previous commit inside the head, every earlier main is
        # too, so that merge was the head's own tree.
        if get(f"/repos/{repo}/compare/{before}...{head}")["status"] not in {"ahead", "identical"}:
            continue
        runs = get(f"/repos/{repo}/actions/runs?event=pull_request&status=success"
                   f"&per_page=100&head_sha={head}")["workflow_runs"]
        for run in runs:
            # A merged pull request's runs list no pull requests, so the run
            # is matched by its head. A fork's run had no secrets: its gated
            # tokenizer tests skipped.
            if (run.get("path") != WORKFLOW or run.get("event") != "pull_request"
                    or run.get("head_sha") != head
                    or (run.get("head_repository") or {}).get("full_name") != repo
                    or run.get("head_branch") != pull["head"]["ref"]):
                continue
            jobs = get(f"/repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100")["jobs"]
            if run_covers(jobs, full_tests=full_tests):
                return run["html_url"]
    return None


def covering_run(event_name: str, event: dict, get: Get, *, full_tests: bool,
                 sleep: Callable[[float], None] = time.sleep) -> str | None:
    """The URL of what already covers this event, or None to run."""
    if event_name != "push":
        return None
    repo = event["repository"]["full_name"]
    ref, sha = event["ref"], event["after"]
    if ref == "refs/heads/main":
        return passed_pull_request_run(get, repo, event["before"], sha, full_tests=full_tests)
    if ref.startswith("refs/heads/"):
        owner = event["repository"]["owner"]["login"]
        return open_pull_request(get, repo, owner, ref.removeprefix("refs/heads/"), sha,
                                 sleep=sleep)
    return None


def _api(base: str, token: str) -> Get:
    def get(path: str) -> Any:
        request = urllib.request.Request(base.rstrip("/") + path, headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)
    return get


def main() -> int:
    try:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
        get = _api(os.environ.get("GITHUB_API_URL", "https://api.github.com"), os.environ["GH_TOKEN"])
        covered = covering_run(
            os.environ.get("GITHUB_EVENT_NAME", ""), event, get,
            full_tests=os.environ.get("FULL_TESTS", "true") != "false",
        )
    # Any failure, network or answer, runs.
    except Exception as exc:
        print(f"::warning::Duplicate check failed, running: {exc!r}")
        covered = None
    if covered:
        print(f"::notice::Skipped: {covered} covers this commit")
    result = f"duplicate={str(bool(covered)).lower()}"
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a", encoding="utf-8") as stream:
            stream.write(result + "\n")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
