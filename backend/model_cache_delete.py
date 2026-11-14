"""Deleting one whole catalogue checkpoint from this server's caches: the
planner, read-only (`GEN#72` stage 1), and the delete's worker, which so far
removes converted trees only (stage 2; no route reaches it yet).

docs/maintainer/plan-model-delete.md is the design. The unit is one catalogue
entry and its group: the entry's own id, its Nunchaku lite ids and any
cached lite-infer alias of them. `plan()` reads the inventory's own scan
(`model_cache_storage._scan` and its per-file index) under the same scan
reservation and answers, without changing anything on disk:

- a repository the group alone names (no other entry's recipe does): the
  whole `models--org--name` folder - every revision, unselected files,
  `refs/`, `.no_exist/` - and each lite repository the same way;
- a repository other entries' recipes also name: file by file, the group's
  selected files at its recorded commits, a file another entry selects too
  kept while that sharer is present (the last-holder rule) and taken with
  the last holder otherwise; deleting inside such a repository is stage 4,
  so until `SHARED_REPOSITORY_DELETE` a plan that would is held;
- the entry's converted tree and its work directories, and older converted
  trees whose marker names the entry's own source repository and file.

What the server cannot prove safe is `held` (`refusal_codes.MODEL_DELETE_HELD`):
a shared Xet store, a link out of its repository, an unrecorded revision of a
repository several entries name, an older converted tree two entries' source
repository could own, a path outside the configured roots. Every path is
checked inside the Hub cache or the converted root before anything is
planned on it, and answers name repositories, revisions and names relative
to those roots only, never an absolute path.

`delete()` re-plans under the same reservation, refuses a different pin, and
removes the converted parts: each tree under its loaders' own lock, renamed
first into exactly `single_file._is_work`'s shape (`<target>-<8>/tree`), so
it stops being a hit at once, then removed with every stale work directory
of the target. A refusal before anything changed is `DeleteRefused` with its
code; a failure after something changed is the outcome's `remaining`. Until
`HUB_DELETE` (stage 3), a plan with anything to take from the Hub is refused
with nothing changed.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import shutil
import stat
import string
from collections import defaultdict
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import keep_awake, model_cache_storage, model_info
from .refusal_codes import (
    MODEL_CACHE_CHANGED,
    MODEL_DELETE_BUSY,
    MODEL_DELETE_HELD,
    MODEL_DOWNLOADING,
    MODEL_FILES_IN_USE,
    MODEL_SELECTED,
)

log = logging.getLogger(__name__)

# Deleting files inside a repository other entries' recipes also name is
# the plan's stage 4. Until it is built, a plan with such a removal is held;
# the parts it would take carry `shared_repository: true`.
SHARED_REPOSITORY_DELETE = False
# huggingface_hub 1.33's cache-wide shared Xet store marks `<hub>/blobs`
# with this file: its payloads serve several repositories (held, never planned).
SHARED_STORE_MARKER = ".huggingface-shared-blobs"
# An older converted tree's marker is read only this far.
_MARKER_BYTES = 64 * 1024
# Removing a repository from the Hub cache (the trash rename) is the plan's
# stage 3. Until it is built, `delete()` refuses a plan with a Hub or lite
# part, with nothing changed; the planner answers as before.
HUB_DELETE = False
# How long a delete waits for a converted tree's lock. A conversion holds it
# for minutes and a load only while it sweeps and checks, so a wait past this
# is one running: `model_delete_busy`.
LOCK_TIMEOUT = 2.0
# What a work directory's suffix is drawn from: `_is_work` accepts these.
_WORK_ALPHABET = string.ascii_lowercase + string.digits
# The keep-awake hold the worker takes for its whole run.
DELETING = "deleting a checkpoint"


class NotOnDisk(LookupError):
    """No catalogue entry by that id, or nothing of it is cached: one answer."""


@dataclass(frozen=True)
class PlanContext:
    """The server's state a plan reads, by catalogue entry id (`entry.repo`):
    the selected checkpoint, the loaded one, the one saved to load on the
    next start, and Check's tokenizer fetches still in flight
    (`prompt_encoding.pending_sources`). Each counts as a present sharer.
    An id outside the catalogue stays itself (`entry_of`): `Plan.names`
    reads it by its repository."""

    selected: str | None = None
    loaded: str | None = None
    saved: str | None = None
    pending: frozenset[Any] = frozenset()

    def present(self) -> set[str]:
        return {entry for entry in (self.selected, self.loaded, self.saved) if entry is not None}


@dataclass(frozen=True)
class Removal:
    """One file or link a delete would remove; `path` is never answered."""

    kind: str
    root: str  # "hub" or "converted"
    relative: str
    path: Path
    size: int
    mtime_ns: int


@dataclass
class Part:
    kind: str  # "hub", "lite", "converted", "converted-old"
    repo: str | None = None
    revision: str | None = None
    name: str | None = None
    whole: bool = False
    shared_repository: bool = False
    bytes: int = 0
    files: int = 0
    # A converted part's current target, never answered: its lock is the one
    # the loaders take, and its work shape (`<target>-<8>/tree`) the one a
    # removal renames into - an older tree's too, so any leftover is swept
    # by the next load or delete of the entry.
    target: Path | None = None
    # What a delete would act on first: whole folders (a repository, a
    # converted tree, a work directory) or, inside a shared repository, the
    # snapshot entries and the blobs no surviving entry resolves to.
    units: list[Path] = field(default_factory=list)

    def answer(self) -> dict:
        answer: dict = {"kind": self.kind}
        if self.repo is not None:
            answer["repo"] = self.repo
        if self.revision is not None:
            answer["revision"] = self.revision
        if self.name is not None:
            answer["name"] = self.name
        answer.update(whole=self.whole, shared_repository=self.shared_repository,
                      bytes=self.bytes, files=self.files)
        return answer


@dataclass
class Plan:
    entry: str
    label: str
    # The group's catalogue ids (the entry's own, its lite ids): the gallery
    # records images under these.
    model_ids: tuple[str, ...]
    context: PlanContext
    parts: list[Part] = field(default_factory=list)
    kept_shared: list[dict] = field(default_factory=list)
    also_taken_shared: list[dict] = field(default_factory=list)
    held: dict | None = None
    downloading: list[dict] = field(default_factory=list)
    # Every repository the group's recipes name (shared ones included) and
    # the group's lite aliases: Check's fetch from any of them holds a delete.
    repositories: frozenset[str] = frozenset()
    removals: list[Removal] = field(default_factory=list)
    pin: str = ""
    # The roots the plan's scan read: the delete re-asserts them against the
    # configured ones before its first rename.
    hub: Path | None = None
    converted_root: Path | None = None

    @property
    def frees_bytes(self) -> int:
        return sum(part.bytes for part in self.parts)

    def fetching(self, sources: Iterable[Any]) -> bool:
        """Whether a tokenizer fetch in flight reads a repository or a
        converted tree this plan touches."""
        return any(getattr(source, "repo", None) in self.repositories
                   or getattr(source, "tree", None) in self.model_ids for source in sources)

    def names(self, model_id: str | None) -> bool:
        """Whether a `PlanContext` id (selected, loaded, saved) is what this
        plan would delete: its own entry, or an id outside the catalogue
        whose repository the plan names. A selected `Tongyi-MAI/Z-Image` is
        Z-Anime's recipe repository, owned and taken whole; a case variant
        of a catalogue id (`AURADREAM_MODEL_ID=black-forest-labs/flux.1-krea-dev`)
        is no catalogue entry but the same folder on NTFS, so the comparison
        ignores case, as the Hub's ids do. Another catalogue entry is never
        this plan's: a sharer's files are kept by the last-holder rule."""
        if model_id is None:
            return False
        if model_id == self.entry:
            return True
        if model_info.catalogue_entry(model_id) is not None:
            return False
        repo = model_id.split("@", 1)[0].casefold()
        return any(repo == named.casefold() for named in self.repositories)

    def answer(self) -> dict:
        return {
            "entry": self.entry, "label": self.label,
            "parts": [part.answer() for part in self.parts],
            "kept_shared": self.kept_shared, "also_taken_shared": self.also_taken_shared,
            "frees_bytes": self.frees_bytes, "held": self.held, "downloading": self.downloading,
            "selected": self.names(self.context.selected),
            "loaded": self.names(self.context.loaded),
            "load_on_startup": self.names(self.context.saved),
            "pin": self.pin,
        }


def entry_of(model_id: str) -> str:
    """The catalogue entry id a model id belongs to (a lite id or alias to
    its entry's), or the id itself where no entry names it."""
    entry = model_info.catalogue_entry(model_id)
    return entry.repo if entry is not None else model_id


def _label(entry_id: str) -> str:
    entry = model_info.catalogue_entry(entry_id)
    return entry.name if entry is not None else entry_id


def _repo_folder(hub: Path, repo_id: str) -> Path:
    return hub / ("models--" + repo_id.replace("/", "--"))


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _lexically_inside(path: Path, root: Path) -> bool:
    """Inside `root` by name alone: never touches a path that is not. Paths
    on two drives (Windows' `commonpath` raises) are not inside."""
    path_text = os.path.normcase(os.path.abspath(path))
    root_text = os.path.normcase(os.path.abspath(root))
    try:
        return path_text != root_text and os.path.commonpath([path_text, root_text]) == root_text
    except ValueError:
        return False


def _inside(path: Path, root: Path) -> bool:
    """`path` itself (a link is removed, never what it names) lies under
    `root`, by name and through every real directory above it."""
    if not _lexically_inside(path, root):
        return False
    try:
        parent = path.parent.resolve(strict=True)
        real_root = root.resolve(strict=True)
    except OSError:
        return False
    return parent == real_root or parent.is_relative_to(real_root)


class _Collector:
    def __init__(self, hub: Path, converted: Path) -> None:
        self.roots = {"hub": hub, "converted": converted}
        self.removals: list[Removal] = []
        self.held: list[tuple[int, str]] = []
        # Whether a hold was found in something of the entry's on disk. A
        # hold that is not (a converted target outside the root, never looked
        # at) leaves an entry with nothing cached the plan's one 404.
        self.found = False

    def hold(self, priority: int, detail: str, *, on_disk: bool = True) -> None:
        self.held.append((priority, detail))
        self.found = self.found or on_disk

    def item(self, kind: str, root: str, path: Path) -> Removal | None:
        """One file or link, measured without following it."""
        try:
            info = os.lstat(path)
        except OSError:
            self.hold(3, "Some of its files changed or could not be read; refresh and try again.")
            return None
        size = info.st_size if stat.S_ISREG(info.st_mode) else 0
        relative = Path(os.path.relpath(path, self.roots[root])).as_posix()
        removal = Removal(kind, root, relative, path, size, info.st_mtime_ns)
        self.removals.append(removal)
        return removal

    def tree(self, kind: str, root: str, unit: Path) -> tuple[int, list[Removal]]:
        """Every file and link under one folder a delete would remove whole:
        (logical bytes, the items). A link whose target leaves the folder is
        held: what it names is someone else's."""
        found: list[Removal] = []
        failed: list[OSError] = []
        real_unit = unit.resolve()
        for directory, dirs, files in os.walk(unit, followlinks=False, onerror=failed.append):
            here = Path(directory)
            links = [name for name in dirs if _is_link(here / name)]
            dirs[:] = [name for name in dirs if name not in links]
            for name in [*files, *links]:
                path = here / name
                if _is_link(path):
                    try:
                        target = Path(os.path.realpath(path))
                    except OSError:
                        target = None
                    if target is None or not (target == real_unit or target.is_relative_to(real_unit)):
                        self.hold(0, "A link inside its files points outside them, as a shared Xet store's "
                                     "do; deleting it is not built yet.")
                        continue
                removal = self.item(kind, root, path)
                if removal is not None:
                    found.append(removal)
        if failed:
            self.hold(3, "Some of its folders could not be read; refresh and try again.")
        return sum(removal.size for removal in found), found


def plan(model_id: str, context: PlanContext | None = None, cache_dir: Path | None = None) -> Plan:
    """The plan for `model_id`'s catalogue entry, from one scan under the
    scan reservation. Raises `NotOnDisk` for an id no entry names or an
    entry with nothing cached, `model_cache_storage.ScanBusy` while another
    scan (or a delete) holds the reservation."""
    entry = model_info.catalogue_entry(model_id)
    if entry is None or entry.repo != model_id:
        # Exact entry ids only: a lite id is its entry's sub-line, not a unit.
        raise NotOnDisk(model_id)
    if not model_cache_storage._scan_lock.acquire(blocking=False):
        raise model_cache_storage.ScanBusy
    try:
        return _plan(entry.repo, context or PlanContext(), cache_dir)
    finally:
        model_cache_storage._scan_lock.release()


def _plan(entry_id: str, context: PlanContext, cache_dir: Path | None) -> Plan:
    """`plan()`'s work; the caller holds `model_cache_storage._scan_lock`."""
    from .backends.single_file import CONVERSION_MARKER, _is_work

    entry = model_info.catalogue_entry(entry_id)
    assert entry is not None
    _, index = model_cache_storage._scan(cache_dir)
    hub, converted_root = index.hub, index.converted_root
    lite_ids = [repo for fmt, _ in model_info.FORMATS if entry.lite and (repo := entry.lite_repo(fmt))]
    aliases = {alias for fmt, _ in model_info.FORMATS
               if entry.lite and (alias := entry.lite_alias(fmt)) and alias not in lite_ids}
    group = {entry.repo, *lite_ids}
    result = Plan(entry.repo, entry.name, tuple(sorted(group)), context)
    result.hub, result.converted_root = hub, converted_root
    collect = _Collector(hub, converted_root)

    named = {repo for repo, models in index.recipe_repos.items() if models & group}
    result.repositories = frozenset(named | aliases | set(lite_ids))
    owned = {repo for repo in result.repositories if not index.recipe_repos.get(repo, frozenset()) - group}
    shared = result.repositories - owned

    touched: set[str] = set()
    # Repositories the group alone names: whole.
    for repo in sorted(owned):
        folder = _repo_folder(hub, repo)
        if not os.path.lexists(folder):
            continue
        if not _inside(folder, hub) or _is_link(folder) or not folder.is_dir():
            collect.hold(1, f"{repo}'s cache folder is a link or lies outside the Hub cache; "
                            "deleting it is not built.")
            continue
        kind = "lite" if entry.lite_format(repo) is not None else "hub"
        size, items = collect.tree(kind, "hub", folder)
        snapshot_entries = sum(1 for item in items if item.relative.split("/")[1:2] == ["snapshots"])
        result.parts.append(Part(kind, repo=repo, whole=True, bytes=size, files=snapshot_entries, units=[folder]))
        touched.add(repo)

    # Repositories other entries' recipes name too: file by file.
    present = context.present() | _present_entries(index)
    pending_repos = {getattr(source, "repo", None) for source in context.pending}
    taken: dict[Path, tuple[model_cache_storage.IndexedFile, frozenset[str]]] = {}
    kept: dict[Path, tuple[model_cache_storage.IndexedFile, frozenset[str]]] = {}
    for file in index.files:
        if file.repo_type != "model" or file.repo_id not in shared:
            continue
        if file.owners:
            owners = {entry_of(model) for model in file.owners}
            if entry.repo not in owners:
                continue  # another entry's file in a repository both name
            sharers = frozenset(owners - {entry.repo})
            if sharers and (sharers & present or file.repo_id in pending_repos):
                # Named by the sharers that keep it (all of them for a fetch).
                kept[file.entry] = (file, frozenset(sharers & present) or sharers)
            else:
                taken[file.entry] = (file, sharers)
        elif file.other_of:
            others = {entry_of(model) for model in file.other_of}
            if entry.repo in others and others != {entry.repo}:
                collect.hold(2, f"{file.repo_id} is cached at a revision no download record names, which "
                                f"{', '.join(sorted(_label(other) for other in others - {entry.repo}))} "
                                "could also own; deleting it is not built.")
    survivors = [file for file in index.files if file.entry not in taken]
    surviving_blobs = {file.blob for file in survivors}
    counted: set[Path] = set()
    parts: dict[tuple[str, str], Part] = {}
    shares: dict[frozenset[str], list[int]] = defaultdict(list)
    for path, (file, sharers) in sorted(taken.items(), key=lambda item: str(item[0])):
        folder = _repo_folder(hub, file.repo_id)
        if not (_inside(file.entry, folder) and _inside(file.blob, folder)) or (
                file.blob != file.entry and _is_link(file.blob)):
            collect.hold(0, f"A file of {file.repo_id} resolves outside its repository, as a shared Xet "
                            "store's do; deleting it is not built yet.")
            continue
        part = parts.get((file.repo_id, file.revision))
        if part is None:
            kind = "lite" if entry.lite_format(file.repo_id) is not None else "hub"
            part = parts[(file.repo_id, file.revision)] = Part(
                kind, repo=file.repo_id, revision=file.revision, shared_repository=True)
        part.files += 1
        part.units.append(path)
        collect.item(part.kind, "hub", path)
        freed = 0
        if file.blob != file.entry and file.blob not in surviving_blobs and file.blob not in counted:
            part.units.append(file.blob)
            collect.item(part.kind, "hub", file.blob)
        if file.blob not in surviving_blobs and file.blob not in counted:
            freed = file.size
            counted.add(file.blob)
        part.bytes += freed
        if sharers:
            shares[sharers].append(freed)
        touched.add(file.repo_id)
    result.parts.extend(parts[key] for key in sorted(parts))
    result.also_taken_shared = [
        {"bytes": sum(sizes), "with": sorted(_label(sharer) for sharer in sharers)}
        for sharers, sizes in sorted(shares.items(), key=lambda item: sorted(item[0]))]
    kept_by: dict[frozenset[str], set[tuple]] = defaultdict(set)
    for file, sharers in kept.values():
        kept_by[sharers].add(file.identity)
    result.kept_shared = [
        {"bytes": sum(index.sizes.get(identity, 0) for identity in identities),
         "with": sorted(_label(sharer) for sharer in sharers)}
        for sharers, identities in sorted(kept_by.items(), key=lambda item: sorted(item[0]))]

    # The converted copy, its work directories, and older recipes of it.
    targets = {model: index.converted_targets[model] for model in sorted(group) if model in index.converted_targets}
    for target in targets.values():
        # By name first: a target resolved through another HF_HOME is never
        # even looked at.
        if not _lexically_inside(target, converted_root):
            # Unlooked at, it says nothing of what the entry has on disk.
            collect.hold(1, "Its converted copy's folder lies outside this server's converted cache; "
                            "deleting it is not built.", on_disk=False)
            continue
        units = []
        if os.path.lexists(target):
            if _is_link(target) or not target.is_dir():
                collect.hold(1, "Its converted copy is a link, not a folder; deleting it is not built.")
                continue
            units.append(target)
        units.extend(path for path in _work_directories(converted_root, target, _is_work))
        if not units:
            continue
        if not _inside(target, converted_root):
            collect.hold(1, "Its converted copy's folder lies outside this server's converted cache; "
                            "deleting it is not built.")
            continue
        part = Part("converted", name=target.name, units=units, target=target)
        for unit in units:
            size, items = collect.tree("converted", "converted", unit)
            part.bytes += size
            part.files += len(items)
        result.parts.append(part)
    sources = _conversion_sources(index)
    # Each of the group's current sources and the model converting from it.
    mine = {sources[model]: model for model in sorted(group) if model in sources}
    mine_repos = {repo for repo, _ in mine}
    current = {target.name for target in index.converted_targets.values()}
    for tree in _older_trees(converted_root, current, index.converted_targets.values(), _is_work):
        source = _marker_source(tree / CONVERSION_MARKER)
        if source is None:
            continue  # without a readable marker it stays "not referenced"
        if source in mine:
            if not _inside(tree, converted_root):
                collect.hold(1, "An older converted copy lies outside this server's converted cache.")
                continue
            size, items = collect.tree("converted-old", "converted", tree)
            result.parts.append(Part("converted-old", name=tree.name, bytes=size, files=len(items), units=[tree],
                                     target=index.converted_targets[mine[source]]))
        elif source[0] in mine_repos:
            owners = {model for model, (repo, _) in sources.items() if repo == source[0]}
            claimed = {sources[model][1] for model in owners}
            if len({entry_of(model) for model in owners}) > 1 and source[1] not in claimed:
                collect.hold(2, f"The older converted copy {tree.name} comes from {source[0]}, which "
                                f"{' and '.join(sorted(_label(entry_of(model)) for model in owners))} share, "
                                "from a file neither uses now; it is not attributed to either.")

    # Partial downloads in what the plan would touch: another process may be
    # writing them (refused by the delete with `model_downloading`).
    for repo in sorted(touched):
        blobs = _repo_folder(hub, repo) / "blobs"
        try:
            partial = sorted(blobs.glob("*.incomplete"))
        except OSError:
            partial = []
        result.downloading.extend(
            {"repo": repo, "file": Path(os.path.relpath(path, hub)).as_posix()} for path in partial)

    # A cache-wide shared Xet store serves the Hub parts' bytes to other
    # repositories: held where the plan takes something from the Hub, never
    # for an entry with nothing there (its converted copy is not the store's).
    if (any(part.kind in ("hub", "lite") for part in result.parts)
            and os.path.lexists(hub / "blobs" / SHARED_STORE_MARKER)):
        collect.hold(0, "This Hugging Face cache keeps a shared Xet store across repositories; "
                        "deleting from it is not built yet.")
    if any(part.shared_repository for part in result.parts) and not SHARED_REPOSITORY_DELETE:
        repos = sorted({part.repo for part in result.parts if part.shared_repository and part.repo})
        others = sorted({_label(entry_of(model)) for repo in repos for model in index.recipe_repos.get(repo, ())}
                        - {entry.name})
        collect.hold(4, f"Its files in {', '.join(repos)} sit in a repository "
                        f"{', '.join(others)} also use{'s' if len(others) == 1 else ''}; deleting files "
                        "inside a shared repository is not built yet.")
    for removal in collect.removals:
        if not _inside(removal.path, collect.roots[removal.root]):
            collect.hold(1, "A planned file lies outside this server's caches; nothing can be deleted.")
            break
    if collect.held:
        result.held = {"code": MODEL_DELETE_HELD, "detail": min(collect.held)[1]}
    result.removals = sorted(collect.removals, key=lambda removal: (removal.root, removal.relative))
    result.pin = _pin(result, kept, targets)
    if not result.parts and not result.kept_shared and not result.downloading and not collect.found:
        raise NotOnDisk(entry_id)
    return result


def _present_entries(index: model_cache_storage.CacheIndex) -> set[str]:
    """Entries with something of their own on disk: exclusive Hub bytes,
    other-revision bytes or a converted tree (the last-holder rule)."""
    present = set()
    for model, row in index.rows.items():
        if row.get("exclusive_bytes") or row.get("other_revision_bytes") or row.get("converted_bytes"):
            present.add(entry_of(model))
    return present


def _work_directories(root: Path, target: Path, is_work) -> list[Path]:
    try:
        return sorted(path for path in root.iterdir() if path.name.startswith(target.name + "-") and is_work(path, target))
    except OSError:
        return []


def _older_trees(root: Path, current: set[str], targets: Iterable[Path], is_work) -> list[Path]:
    """Folders under the converted root that are no catalogue entry's
    current target, nor one's work directory, nor a link."""
    targets = list(targets)
    try:
        candidates = sorted(root.iterdir())
    except OSError:
        return []
    return [path for path in candidates
            if path.name not in current and not _is_link(path) and path.is_dir()
            and not any(is_work(path, target) for target in targets)]


def _marker_source(marker: Path) -> tuple[str, str | None] | None:
    """(source repository, file) from a converted tree's marker, or None
    when it is missing, unreadable or names no source repository."""
    try:
        if _is_link(marker) or not marker.is_file():
            return None
        with marker.open("rb") as stream:
            data = stream.read(_MARKER_BYTES + 1)
        if len(data) > _MARKER_BYTES:
            return None
        recipe = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError):
        return None
    source = recipe.get("source") if isinstance(recipe, dict) else None
    if not isinstance(source, dict) or not isinstance(source.get("repo_id"), str):
        return None
    filename = source.get("filename")
    return source["repo_id"], filename if isinstance(filename, str) else None


def _conversion_sources(index: model_cache_storage.CacheIndex) -> dict[str, tuple[str, str | None]]:
    """Each converting catalogue id's current source (repository, file)."""
    sources = {}
    for model in index.converted_targets:
        conversion = model_info._conversion_target(model)
        if conversion is None:
            continue
        source = json.loads(conversion[2]).get("source") or {}
        if isinstance(source.get("repo_id"), str):
            filename = source.get("filename")
            sources[model] = (source["repo_id"], filename if isinstance(filename, str) else None)
    return sources


def _pin(result: Plan, kept: dict, targets: dict[str, Path]) -> str:
    """sha256 over every planned removal (kind, path relative to its root,
    size, mtime_ns), the kept and also-taken sets and the converted target's
    name: the delete re-plans and refuses a different one."""
    def kept_stamp(file: model_cache_storage.IndexedFile, sharers: frozenset[str]) -> list:
        try:
            when = os.lstat(file.entry).st_mtime_ns
        except OSError:
            when = -1
        return [file.repo_id, file.revision, file.name, file.size, when, sorted(sharers)]

    payload = {
        "entry": result.entry,
        "remove": [[removal.kind, removal.root, removal.relative, removal.size, removal.mtime_ns]
                   for removal in result.removals],
        "kept": sorted(kept_stamp(file, sharers) for file, sharers in kept.values()),
        "also_taken": sorted([share["bytes"], share["with"]] for share in result.also_taken_shared),
        "converted": sorted(target.name for target in targets.values()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()



# ---------------------------------------------------------------- the delete


class DeleteRefused(Exception):
    """A delete refused with nothing changed: `code` (a `refusal_codes`
    code) and `detail`, the person's sentence. A route raises it as
    `refusals.refusal(409, error.detail, error)`."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass
class Outcome:
    """What one delete did: whether anything changed yet, and what stays
    (`remaining`: each part's kind, its bytes left, the reason in words)."""

    label: str
    planned_bytes: int
    changed: bool = False
    freed_bytes: int | None = None
    remaining: list[dict] = field(default_factory=list)

    def stop(self, part: Part, code: str, detail: str, reason: str, size: int | None = None) -> None:
        """Before anything changed, refuse with `code`; after, record what
        stays and go on: something did change, so no refusal code may say
        nothing happened."""
        if not self.changed:
            raise DeleteRefused(code, detail)
        self.remaining.append({"kind": part.kind, "bytes": part.bytes if size is None else size, "reason": reason})

    def answer(self) -> dict:
        return {"label": self.label, "planned_bytes": self.planned_bytes,
                "freed_bytes": self.freed_bytes, "remaining": self.remaining}


def delete(model_id: str, pin: str, context: PlanContext | None = None, cache_dir: Path | None = None,
           *, lock_timeout: float = LOCK_TIMEOUT) -> Outcome:
    """Delete `model_id`'s catalogue entry as the plan with `pin` said, on
    the calling (daemon) thread. Holds the scan reservation for its whole run
    (`model_cache_storage.ScanBusy` while another scan or delete has it) and
    `keep_awake.busy`; re-plans with `context`, refuses a different pin, a
    selected, loaded or saved-for-start entry, a fetch it touches, a partial
    download or a held part, and re-asserts every path inside the configured
    roots, all before the first rename (`DeleteRefused`, nothing changed).
    Raises `NotOnDisk` as `plan()` does. The model cache facts are
    invalidated on the way out, whatever happened."""
    entry = model_info.catalogue_entry(model_id)
    if entry is None or entry.repo != model_id:
        raise NotOnDisk(model_id)
    if not model_cache_storage._scan_lock.acquire(blocking=False):
        raise model_cache_storage.ScanBusy
    try:
        with keep_awake.busy(DELETING):
            result = _plan(entry.repo, context or PlanContext(), cache_dir)
            _refuse_before_change(result, pin)
            outcome = Outcome(result.label, result.frees_bytes)
            volumes = _volumes(result)
            before = _free(volumes)
            try:
                remove_converted(result, outcome, lock_timeout=lock_timeout)
            finally:
                after = _free(volumes)
                if before is not None and after is not None:
                    # Another writer can move it either way: "about".
                    outcome.freed_bytes = max(0, after - before)
            return outcome
    finally:
        model_cache_storage._scan_lock.release()
        model_info.invalidate_cache_facts()


def _refuse_before_change(result: Plan, pin: str) -> None:
    """The worker's own checks on its fresh plan; the route's guards ran
    first (`main._model_delete_refusal`), these hold whatever changed since."""
    label = result.label
    if not hmac.compare_digest(result.pin.encode(), str(pin).encode()):
        raise DeleteRefused(MODEL_CACHE_CHANGED, f"{label}'s files changed since the plan was read; "
                                                 "read it again.")
    context = result.context
    if any(result.names(current) for current in (context.selected, context.loaded, context.saved)):
        raise DeleteRefused(MODEL_SELECTED, f"{label} is the selected checkpoint or the one saved to load "
                                            "on the next start. Choose another checkpoint first.")
    if result.fetching(context.pending):
        raise DeleteRefused(MODEL_DELETE_BUSY, f"Check is reading a tokenizer from its files; delete {label} "
                                               "once it is done.")
    if result.downloading:
        partial = result.downloading[0]
        raise DeleteRefused(MODEL_DOWNLOADING, f"A download is still writing {partial['file']} in "
                                               f"{partial['repo']}; let it finish, or stop it, first.")
    if result.held is not None:
        raise DeleteRefused(MODEL_DELETE_HELD, result.held["detail"])
    if not HUB_DELETE and any(part.kind in ("hub", "lite") for part in result.parts):
        raise DeleteRefused(MODEL_DELETE_HELD, f"Deleting {label}'s downloaded files is not built yet; "
                                               "nothing was deleted.")
    assert_inside_roots(result)


def _same_folder(one: Path, other: Path) -> bool:
    return os.path.normcase(os.path.abspath(one)) == os.path.normcase(os.path.abspath(other))


def assert_inside_roots(result: Plan) -> None:
    """Every path the plan acts on - each removal, each unit, each converted
    target, its lock and the folder its work directories go in - lies inside
    this server's configured roots, read now (the Hub cache as the scan
    resolves it, the converted root), and the plan's roots are those. Raises
    `DeleteRefused` (`model_delete_held`) otherwise; called before the
    first rename, so nothing has changed."""
    hub = model_cache_storage.hub_cache_dir().expanduser().resolve()
    converted = model_cache_storage.hf_home_dir("converted")
    refused = DeleteRefused(MODEL_DELETE_HELD, "A planned file lies outside this server's caches; "
                                               "nothing was deleted.")
    if (result.hub is None or result.converted_root is None or not _same_folder(result.hub, hub)
            or not _same_folder(result.converted_root, converted)):
        raise refused
    roots = {"hub": hub, "converted": converted}
    paths = [(removal.root, removal.path) for removal in result.removals]
    for part in result.parts:
        root = "converted" if part.kind in ("converted", "converted-old") else "hub"
        paths.extend((root, unit) for unit in part.units)
        if root == "converted":
            target = part.target
            # The loaders lock and work beside the target, in the root itself.
            if target is None or not _same_folder(target.parent, converted):
                raise refused
            paths.append((root, target))
            # Its lock, and an older tree's own.
            paths.extend((root, converted / f"{name}.lock") for name in
                         {target.name, *(unit.name for unit in part.units if part.kind == "converted-old")})
    for root, path in paths:
        if not _inside(path, roots[root]):
            raise refused


def _volumes(result: Plan) -> list[Path]:
    """The plan's roots, one per volume, for the free-space measure."""
    found: dict[object, Path] = {}
    for root in (result.hub, result.converted_root):
        if root is None:
            continue
        try:
            device: object = os.stat(root).st_dev
        except OSError:
            device = root
        found.setdefault(device, root)
    return list(found.values())


def _free(volumes: list[Path]) -> int | None:
    sizes = [model_cache_storage.free_bytes(volume) for volume in volumes]
    known = [size for size in sizes if size is not None]
    return sum(known) if known and len(known) == len(sizes) else None


def remove_converted(result: Plan, outcome: Outcome, *, lock_timeout: float = LOCK_TIMEOUT) -> None:
    """Remove the plan's converted parts: the entry's tree with its work
    directories, then each older recipe's tree. The caller holds the scan
    reservation and has checked the plan (`delete()`)."""
    for part in sorted((part for part in result.parts if part.kind in ("converted", "converted-old")),
                       key=lambda part: part.kind != "converted"):
        _remove_converted_part(result, part, outcome, lock_timeout)


def _remove_converted_part(result: Plan, part: Part, outcome: Outcome, lock_timeout: float) -> None:
    """One converted part, under the loaders' lock (docs/maintainer/plan-model-delete.md,
    "How files are removed > Converted tree"):

    1. the target's `.lock` (an older tree's own too), with a short timeout:
       a timeout is a conversion running;
    2. the tree as planned, re-read under the lock;
    3. `os.rename(tree, <target>-<8>/tree)`: the tree stops being a hit at
       once, and a refused rename changes nothing;
    4. every work directory of the target removed (the renamed tree's and
       stale ones), a failure midway leaving only `_is_work`'s shape, which
       the next load's `sweep_work` and the next delete remove.

    The `.lock` stays: zero bytes, recreated by the loaders, and unlinking a
    lock another process may take after the release would let two holders
    sweep each other's work directories."""
    from filelock import Timeout
    from huggingface_hub.utils._fixes import WeakFileLock

    from .backends.single_file import _is_work

    target = part.target
    assert target is not None
    root = target.parent
    label = result.label
    # By what the plan made each unit, not by the disk: a work directory a
    # load swept since is no tree that changed.
    trees = [unit for unit in part.units if unit == target or part.kind == "converted-old"]
    locks = [root / f"{target.name}.lock", *(root / f"{tree.name}.lock" for tree in trees if tree != target)]
    changed = f"{label}'s converted files changed since the plan was read; read it again."
    with ExitStack() as held:
        for lock in locks:
            try:
                held.enter_context(WeakFileLock(lock, timeout=lock_timeout))
            except Timeout:
                outcome.stop(part, MODEL_DELETE_BUSY,
                             f"{label} is being converted or loaded by another run; delete it once that is done.",
                             "Another run was converting or loading it.")
                return
        if part.kind == "converted" and target not in trees and os.path.lexists(target):
            outcome.stop(part, MODEL_CACHE_CHANGED, changed, "Its converted copy appeared after the plan was read.",
                         size=0)
            return
        for tree in trees:
            if not _as_planned(result, tree):
                outcome.stop(part, MODEL_CACHE_CHANGED, changed, "It changed after the plan was read.")
                return
        for tree in trees:
            if not _move_aside(tree, target, part, outcome, label):
                return
        _sweep_work(root, target, part, outcome, label, _is_work)


def _as_planned(result: Plan, tree: Path) -> bool:
    """Whether `tree` is still the real folder the plan measured, file for
    file (path, size, mtime_ns)."""
    if _is_link(tree) or not tree.is_dir():
        return False
    assert result.hub is not None and result.converted_root is not None
    collect = _Collector(result.hub, result.converted_root)
    _, items = collect.tree("converted", "converted", tree)
    prefix = tree.name + "/"
    planned = {(removal.relative, removal.size, removal.mtime_ns) for removal in result.removals
               if removal.root == "converted" and removal.relative.startswith(prefix)}
    return not collect.held and {(item.relative, item.size, item.mtime_ns) for item in items} == planned


def _new_work(root: Path, target: Path) -> Path:
    """A fresh, empty `<target>-<8 [a-z0-9]>` folder in `root`."""
    for _ in range(64):
        work = root / f"{target.name}-{''.join(secrets.choice(_WORK_ALPHABET) for _ in range(8))}"
        try:
            work.mkdir()
        except FileExistsError:
            continue
        return work
    raise FileExistsError(f"no free work directory name beside {target.name}")


def _move_aside(tree: Path, target: Path, part: Part, outcome: Outcome, label: str) -> bool:
    """Rename `tree` into a new work directory of `target`; False (the stop
    recorded, or raised while nothing has changed) when it would not go."""
    in_use = (f"A file in {label}'s converted copy is open, perhaps in another program, so it could not "
              "be moved aside; nothing was deleted.")
    try:
        work = _new_work(tree.parent, target)
    except OSError:
        log.warning("could not make a work directory to delete the converted copy %s", tree.name)
        log.debug("work directory for %s", tree.name, exc_info=True)
        outcome.stop(part, MODEL_FILES_IN_USE, in_use, "It could not be moved aside.")
        return False
    try:
        os.rename(tree, work / "tree")
    except OSError:
        log.warning("could not move the converted copy %s aside: a file in it is open", tree.name)
        log.debug("rename of %s", tree.name, exc_info=True)
        try:
            work.rmdir()
        except OSError:
            pass  # empty: `_is_work`'s shape, swept by the next load or delete
        outcome.stop(part, MODEL_FILES_IN_USE, in_use, "A file in it is open, so it could not be moved aside.")
        return False
    outcome.changed = True
    return True


def _contents(folder: Path) -> tuple[int, int]:
    """(entries, logical bytes) under `folder`, never following a link."""
    entries = size = 0
    for directory, dirs, files in os.walk(folder, followlinks=False):
        for name in [*dirs, *files]:
            entries += 1
            try:
                info = os.lstat(Path(directory) / name)
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                size += info.st_size
    return entries, size


def _sweep_work(root: Path, target: Path, part: Part, outcome: Outcome, label: str, is_work) -> None:
    """Remove every work directory of `target`, under its lock: each is
    stale, the renamed tree's included."""
    for work in _work_directories(root, target, is_work):
        if not _inside(work, root):
            continue
        before = _contents(work)
        try:
            shutil.rmtree(work)
        except OSError:
            log.warning("could not remove all of a deleted converted copy yet (%s): a file in it is open",
                        work.name)
            log.debug("rmtree of %s", work.name, exc_info=True)
            left = _contents(work) if os.path.lexists(work) else (0, 0)
            if left != before:
                outcome.changed = True
            outcome.stop(part, MODEL_FILES_IN_USE,
                         f"A file in {label}'s converted copy is open, perhaps in another program; "
                         "nothing was deleted.",
                         "A file in it is still open; the next load or delete removes the rest.", size=left[1])
            continue
        outcome.changed = True
