"""What deleting one whole catalogue checkpoint from this server's caches
would take: the planner, read-only (`GEN#72` stage 1).

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
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import model_cache_storage, model_info
from .refusal_codes import MODEL_DELETE_HELD

# Deleting files inside a repository other entries' recipes also name is
# the plan's stage 4. Until it is built, a plan with such a removal is held;
# the parts it would take carry `shared_repository: true`.
SHARED_REPOSITORY_DELETE = False
# huggingface_hub 1.33's cache-wide shared Xet store marks `<hub>/blobs`
# with this file: its payloads serve several repositories (held, never planned).
SHARED_STORE_MARKER = ".huggingface-shared-blobs"
# An older converted tree's marker is read only this far.
_MARKER_BYTES = 64 * 1024


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
        part = Part("converted", name=target.name, units=units)
        for unit in units:
            size, items = collect.tree("converted", "converted", unit)
            part.bytes += size
            part.files += len(items)
        result.parts.append(part)
    sources = _conversion_sources(index)
    mine = {sources[model] for model in group if model in sources}
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
            result.parts.append(Part("converted-old", name=tree.name, bytes=size, files=len(items), units=[tree]))
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
