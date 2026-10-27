"""Whether this sign-in can download a checkpoint, asked before a switch
releases the working one (`main.switch_model`).

The catalogue's gate word (`launch.CHECKPOINTS`, `model_info.details_for`) is
static: it says a repo is gated, not whether this user's token has been let
in. Without this a switch to a gated checkpoint the user had not accepted
unloaded the working model and then failed its load. `check` answers a
`Refusal` only when the Hub says no; everything it cannot settle - offline,
`HF_HUB_OFFLINE`, a timeout, a server error, a missing file - is None, and
the switch goes ahead as before, its load reporting whatever follows.

No answer comes from the network when the files are already cached
(`model_info.download_facts`' "complete"); otherwise each Hub source the
checkpoint's download recipe (`model-downloads.json`) names is asked once:
a HEAD on one of its required files at the recipe's revision, so a pinned
commit and a single-file repo are checked as the loader will fetch them,
or the repository's auth check where the recipe lists no files. A custom id
off the recipe gets the auth check; a local path gets nothing.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from .launch import CHECKPOINTS
from .model_info import download_facts, download_references

log = logging.getLogger("auradream")

# Each request's timeout. The route awaits the whole check for at most
# `CHECK_SECONDS` (two sources, each its own request) and goes ahead without
# an answer after it.
REQUEST_SECONDS = 4.0
CHECK_SECONDS = 9.0

LOGIN = "run `uv run hf auth login` once with a read token"


@dataclass(frozen=True)
class Refusal:
    """The Hub's no: `kind` is "gated" or "not_found", as a failed load's
    `detail_kind`; `detail` the sentence with what to do; `url` the page of
    the repository that refused."""

    kind: str
    detail: str
    url: str


@dataclass(frozen=True)
class Source:
    repo_id: str
    revision: str | None
    # A file the loader fetches from it, for the HEAD; None asks the
    # repository's auth check instead.
    filename: str | None


def sources(model_id: str) -> list[Source]:
    """The Hub repositories a checkpoint's first load reads, from its download
    recipe; the id itself, unpinned, off the recipe; none for a local path."""
    recipe = download_references()["models"].get(model_id)
    if recipe is None:
        if os.path.exists(model_id):
            return []
        return [Source(model_id, None, None)]
    found = []
    for item in recipe["sources"]:
        source = download_references()["sources"][item["source"]]
        files = item["files"] or []
        # A small descriptor where there is one; any listed file otherwise
        # (a single-file repo lists only its checkpoint). A HEAD downloads
        # nothing either way.
        filename = next((name for name in files if name.endswith(".json")), files[0] if files else None)
        found.append(Source(source["repo_id"], source["revision"], filename))
    return found


def _ask(source: Source) -> None:
    """Raise the Hub's error for this source, or return when it answers."""
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

    if source.filename is None:
        HfApi().auth_check(source.repo_id)
        return
    url = hf_hub_url(source.repo_id, source.filename, revision=source.revision)
    get_hf_file_metadata(url, timeout=REQUEST_SECONDS)


def _refused_kind(exc: BaseException) -> str | None:
    """"gated", "not_found", "denied" for an answer that refuses this sign-in,
    or None for one that settles nothing. By class, narrowest first:
    GatedRepoError is a RepositoryNotFoundError."""
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        OfflineModeIsEnabled,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    if isinstance(exc, OfflineModeIsEnabled):
        return None
    if isinstance(exc, GatedRepoError):
        return "gated"
    if isinstance(exc, (RepositoryNotFoundError, RevisionNotFoundError)):
        return "not_found"
    if isinstance(exc, HfHubHTTPError):
        response = getattr(exc, "response", None)
        if getattr(response, "status_code", None) in (401, 403):
            return "denied"
    return None


def _entry(model_id: str):
    exact = next((checkpoint for checkpoint in CHECKPOINTS if checkpoint.repo == model_id), None)
    return exact or next((
        checkpoint for checkpoint in CHECKPOINTS if checkpoint.repo == model_id.split("@", 1)[0]
    ), None)


def _refusal(model_id: str, source: Source, kind: str) -> Refusal:
    entry = _entry(model_id)
    name = entry.name if entry else model_id
    page = f"https://huggingface.co/{source.repo_id}"
    # The checkpoint's own repository, or another one its load reads too.
    own = source.repo_id == model_id.split("@", 1)[0]
    where = f"{name}" if own else f"{name} needs {source.repo_id}, which"
    if kind == "gated":
        gate = entry.gate if entry and own else ""
        if gate.startswith("click-through"):
            how = f"accept its licence on {page} (a click-through)"
        elif gate.startswith("manual"):
            how = f"ask for access on {page} and wait for its authors to approve it"
        else:
            how = f"accept its terms, or ask for access, on {page}"
        lead = f"{where} is gated on Hugging Face and this sign-in has no access yet"
        return Refusal("gated", f"{lead}: {how}, {LOGIN}, then switch again.", page)
    if kind == "denied":
        lead = f"Hugging Face refused this sign-in for {source.repo_id}"
        return Refusal("gated", f"{lead}: check its access on {page}, {LOGIN}, then switch again.", page)
    revision = f" at revision {source.revision}" if source.revision and source.revision != "main" else ""
    return Refusal(
        "not_found",
        f"{source.repo_id}{revision} is not on Hugging Face, or is private to this sign-in: "
        f"check the checkpoint id; for a private repo, {LOGIN} that can read it.",
        page,
    )


def check(model_id: str) -> Refusal | None:
    """Blocking: the Hub's refusal of this checkpoint's download, or None when
    it is cached, allowed, or the answer is not known. Run off the event loop."""
    try:
        if download_facts(model_id)["cache_state"] == "complete":
            return None
    except Exception:  # noqa: BLE001 - an unreadable cache settles nothing
        log.debug("cache check failed for %s", model_id, exc_info=True)
    for source in sources(model_id):
        try:
            _ask(source)
        except Exception as exc:  # noqa: BLE001 - classified below
            kind = _refused_kind(exc)
            if kind is None:
                # Not known for this source; another may still say no.
                log.info("access check for %s inconclusive (%s)", source.repo_id, type(exc).__name__)
                continue
            log.info("access check: %s refused (%s)", source.repo_id, type(exc).__name__)
            return _refusal(model_id, source, kind)
    return None
