"""Private startup-only compiler artifact directories; no compiler imports.

Only the owned ``python -m backend.main`` bootstrap may initialize this
contract, before project workers/requests exist. The module-name observation
is conservative, not proof of foreign-thread quiescence. Embedded/preloaded
hosts cannot establish that precondition by calling a hot model loader.
No public execution recipe is enabled by a binding. None does no cache I/O.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from .execution import CompilerMetadata


_KEYS = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
_CONSUMERS = ("torch._inductor", "torch._dynamo", "triton")
_lock = threading.RLock()
_sealed = False
_startup_thread: int | None = None
_attempt: _Attempt | None = None


def _absolute(path: Path) -> Path:
    # Lexical only: planning must not touch gallery/cache files.
    return Path(os.path.abspath(path))


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


@dataclass(frozen=True)
class CachePlan:
    db: Path
    outputs: Path
    bundles: Path
    web: Path
    root: Path
    facts: CompilerMetadata
    recipe: str
    namespace: str

    @property
    def directory(self) -> Path:
        return self.root / self.namespace

    @property
    def inductor(self) -> Path:
        return self.directory / "inductor"

    @property
    def triton(self) -> Path:
        return self.directory / "triton"


def cache_plan(db: Path, outputs: Path, bundles: Path | None, facts: CompilerMetadata,
               *, root: Path | None = None, web: Path | None = None) -> CachePlan:
    """Pure fixed-recipe namespace, supplied facts only; no metadata/FS reads."""
    db, outputs = _absolute(db), _absolute(outputs)
    bundles = _absolute(db.parent / "bundles" if bundles is None else bundles)
    web = _absolute(Path(__file__).parent.parent / "web" if web is None else web)
    root = _absolute(db.parent / "auradream.compile-cache" if root is None else root)
    if any(_overlaps(root, other) for other in (outputs, bundles, web)) or _overlaps(root, db):
        raise ValueError("Compiler artifacts need a separate directory outside served files, gallery data and bundles.")
    recipe = "regional-v1"
    material = json.dumps({"db": str(db), "outputs": str(outputs), "bundles": str(bundles), "web": str(web),
                           "runtime": asdict(facts), "recipe": recipe}, sort_keys=True, separators=(",", ":"))
    namespace = "regional-v1-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
    return CachePlan(db, outputs, bundles, web, root, facts, recipe, namespace)


@dataclass(frozen=True)
class CacheBinding:
    plan: CachePlan
    identities: tuple[tuple[Path, tuple[int, int]], ...]
    environment: tuple[tuple[str, str], ...]


@dataclass
class _Attempt:
    """Retained even on uncertain failure; never infer permission to retry."""
    plan: CachePlan
    status: str = "initializing"
    operation: str | None = None
    target: Path | None = None
    created: list[Path] = field(default_factory=list)
    descriptor: int | None = None
    probe: Path | None = None
    probe_identity: tuple[int, int] | None = None
    assigned: list[str] = field(default_factory=list)
    binding: CacheBinding | None = None


def _identity(info: os.stat_result) -> tuple[int, int]:
    if info.st_ino <= 0:
        raise RuntimeError("A stable compiler cache file identity is unavailable; startup binding was refused.")
    return info.st_dev, info.st_ino


def _is_link(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _check_path(path: Path, *, directory: bool = False) -> None:
    for part in (*reversed(path.parents), path):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if _is_link(info) or (part != path and not stat.S_ISDIR(info.st_mode)):
            raise ValueError("Compiler cache paths must not use links or non-directory ancestors.")
        if part == path and directory and not stat.S_ISDIR(info.st_mode):
            raise ValueError("Compiler cache path is not a directory.")


def _no_consumers() -> None:
    if any(name == prefix or name.startswith(prefix + ".")
           for name in tuple(sys.modules) for prefix in _CONSUMERS):
        raise RuntimeError("Compiler consumers are already imported; an owned fresh startup is required.")


def _environment(plan: CachePlan) -> tuple[tuple[str, str], ...]:
    expected = tuple(zip(_KEYS, (str(plan.inductor), str(plan.triton)), strict=True))
    for key, _value in expected:
        prior = os.environ.get(key)
        if prior is not None:
            raise RuntimeError("An external compiler cache setting is already present; it was preserved.")
    return expected


def _mkdir(path: Path, attempt: _Attempt) -> None:
    _check_path(path, directory=True)
    if path.exists():
        return
    if not path.parent.exists():
        _mkdir(path.parent, attempt)
    attempt.operation, attempt.target = "mkdir", path
    path.mkdir()  # Never adopt a concurrent writer or retry an uncertain result.
    attempt.created.append(path)
    attempt.operation = None


def _probe(path: Path, attempt: _Attempt) -> None:
    attempt.operation, attempt.target = "create-probe", path
    if os.name == "nt":
        # CRT delete-on-close is tied to this exclusive file handle. Never
        # lstat-then-unlink a pathname that another writer could replace.
        attempt.probe = path / (".auradream-probe-" + secrets.token_hex(16))
        descriptor = os.open(attempt.probe, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_BINARY | os.O_TEMPORARY, 0o600)
    elif hasattr(os, "O_TMPFILE"):
        # No named fallback: unsupported filesystems refuse this private stage.
        descriptor = os.open(path, os.O_RDWR | os.O_TMPFILE, 0o600)
    else:
        raise RuntimeError("A handle-owned compiler cache probe is unavailable on this platform.")
    attempt.descriptor = descriptor
    try:
        attempt.probe_identity = _identity(os.fstat(descriptor))
        attempt.operation = "write-probe"
        data = b"AuraDream compiler cache probe\n"
        if os.write(descriptor, data) != len(data):
            raise OSError("Compiler cache probe write was incomplete.")
        os.fsync(descriptor)
    except BaseException as primary:
        try:
            _close_probe(attempt)
        except BaseException as cleanup:
            primary.add_note(f"Compiler probe cleanup also failed: {type(cleanup).__name__}: {cleanup}")
        raise
    _close_probe(attempt)


def _close_probe(attempt: _Attempt) -> None:
    assert attempt.descriptor is not None
    attempt.operation = "close-probe"
    os.close(attempt.descriptor)
    attempt.descriptor = None  # An exception leaves the uncertain descriptor receipt.
    attempt.probe = None
    attempt.probe_identity = None
    attempt.operation = None


def _verify(binding: CacheBinding, plan: CachePlan) -> CacheBinding:
    if binding.plan != plan:
        raise RuntimeError("Compiler cache binding changed; restart with the intended startup context.")
    if any(os.environ.get(key) != value for key, value in binding.environment):
        raise RuntimeError("Compiler cache environment changed; restart is required.")
    _validate_paths(plan)
    for path, identity in binding.identities:
        _check_path(path, directory=True)
        if _identity(path.lstat()) != identity:
            raise RuntimeError("Compiler cache directory identity changed; restart is required.")
    return binding


def _validate_paths(plan: CachePlan) -> None:
    for path in (plan.db, plan.outputs, plan.bundles, plan.web, plan.root):
        _check_path(path)
    canonical = cache_plan(plan.db.resolve(), plan.outputs.resolve(), plan.bundles.resolve(),
                           plan.facts, root=plan.root.resolve(), web=plan.web.resolve())
    if canonical != plan:
        raise ValueError("Compiler cache plan uses an aliased path; choose its canonical startup location.")


def _initialize(plan: CachePlan) -> CacheBinding:
    global _attempt
    if _attempt is not None:
        if _attempt.status == "bound" and _attempt.binding is not None:
            return _verify(_attempt.binding, plan)
        raise RuntimeError("Compiler cache initialization did not complete; retain its receipt and restart.")
    if _sealed or _startup_thread != threading.get_ident():
        raise RuntimeError("Compiler cache initialization requires the owned startup window.")
    _attempt = attempt = _Attempt(plan)
    try:
        if cache_plan(plan.db, plan.outputs, plan.bundles, plan.facts, root=plan.root, web=plan.web) != plan:
            raise ValueError("Compiler cache plan is malformed.")
        _no_consumers()
        expected = _environment(plan)
        _validate_paths(plan)
        for path in (plan.root, plan.directory, plan.inductor, plan.triton):
            _mkdir(path, attempt)
        for path in (plan.inductor, plan.triton):
            _probe(path, attempt)
        identities = tuple((path, _identity(path.lstat())) for path in
                           (plan.root, plan.directory, plan.inductor, plan.triton))
        _no_consumers()
        _environment(plan)
        for key, value in expected:
            attempt.operation = "check-environment:" + key
            if key in os.environ:
                raise RuntimeError("An external compiler cache setting appeared during startup; it was preserved.")
            attempt.operation = "set-environment:" + key
            os.environ[key] = value
            attempt.assigned.append(key)
        binding = CacheBinding(plan, identities, expected)
        _verify(binding, plan)
        attempt.binding, attempt.status, attempt.operation = binding, "bound", None
        return binding
    except BaseException:
        attempt.status = "failed"
        # No env rollback, close retry, directory deletion or content sweep.
        raise


def _startup_plan() -> CachePlan | None:
    """Private future experiment hook; no public setting or recipe activation."""
    return None


def _owned_startup(factory: Callable[[], CachePlan | None] | None = None) -> CacheBinding | None:
    """Owned entrypoint only, before its project imports/workers/requests.

    Calling this from an embedded host does not prove foreign-thread quiescence
    and is unsupported. No caller-supplied boolean can waive the precondition.
    The one startup window is sealed on success, None and every failure.
    """
    global _sealed, _startup_thread
    with _lock:
        if _sealed:
            raise RuntimeError("The owned compiler cache startup window is closed; restart is required.")
        try:
            if _startup_thread is not None:
                raise RuntimeError("The owned compiler cache startup window is already active.")
            _startup_thread = threading.get_ident()
            plan = (factory or _startup_plan)()
            return None if plan is None else _initialize(plan)
        finally:
            _startup_thread = None
            _sealed = True


def _seal_startup() -> None:
    global _sealed
    with _lock:
        _sealed = True


def require_binding(db: Path, outputs: Path, bundles: Path | None, facts: CompilerMetadata,
                    *, web: Path | None = None) -> CacheBinding:
    """Read/verify only. Hot model loading must never create or redirect caches."""
    with _lock:
        if _attempt is None or _attempt.status != "bound" or _attempt.binding is None:
            raise RuntimeError("Regional compilation needs an owned compiler cache binding from fresh startup.")
        plan = cache_plan(db, outputs, bundles, facts, root=_attempt.plan.root, web=web)
        return _verify(_attempt.binding, plan)
