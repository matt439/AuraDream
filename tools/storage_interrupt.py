"""Kill owned storage workers at deterministic boundaries, then restart and retry.

Only an explicitly empty --data directory is accepted. Each scenario lives in
a fresh child directory which is removed after its workers have stopped.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import queue
import random
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import closing
from pathlib import Path
from zipfile import ZipExtFile, ZipFile


BOUNDARIES = ("export-write", "export-verified", "export-removed", "import-installed",
              "import-before-publish", "import-published")
ROWS = 12
SIDE = 256
MIN_FREE = 256 * 1024**2


def _app(gallery: Path) -> None:
    # No inherited model or gallery setting can direct this check elsewhere.
    for name in list(os.environ):
        if name.startswith("AURADREAM_"):
            del os.environ[name]
    os.environ["AURADREAM_DB_PATH"] = str(gallery / "auradream.db")
    os.environ["AURADREAM_OUTPUTS_DIR"] = str(gallery / "outputs")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _files(directory: Path) -> dict[str, str]:
    return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


def _rows(database: Path) -> list[dict]:
    with closing(sqlite3.connect(database)) as conn:
        conn.row_factory = sqlite3.Row
        assert [r[0] for r in conn.execute("PRAGMA integrity_check")] == ["ok"]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        return [dict(row) for row in conn.execute("SELECT * FROM images ORDER BY id")]


def _size(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _fixture(root: Path, boundary: str):
    from PIL import Image

    _app(root / "source")
    from backend import bundles, storage
    from backend.schemas import GenerateRequest

    storage.init_db()
    pixels = {}

    def save(seed: int, **changes):
        raw = random.Random(seed).randbytes(SIDE * SIDE * 3)
        request = GenerateRequest(prompt=f"interruption image {seed}", width=SIDE, height=SIDE,
                                  steps=28, guidance_scale=3.5, mu=None, **changes).applied(
                                      negative_prompt_supported=True, cfg_threshold=1.0,
                                      reference_supported=True, refine_supported=True, inpaint_supported=True)
        with Image.frombytes("RGB", (SIDE, SIDE), raw) as image:
            row = storage.save_image(image, request, seed=seed, duration_ms=100 + seed,
                                     model_id="test/interruption", quantization="fp8",
                                     job_id="interruption-job", run_id="interruption-run", batch=3)
        pixels[row.filename] = hashlib.sha256(raw).hexdigest()
        return row

    with Image.new("L", (SIDE, SIDE), 0) as image:
        image.paste(255, (8, 8, 192, 192))
        mask, _ = storage.store_mask(image)
    first, second = save(1), save(2)
    made = [first, second, save(3, reference=[second.id, first.id]),
            save(4, refine=first.id, strength=0.35),
            save(5, refine=second.id, strength=0.6, mask=mask)]
    made += [save(seed, reference=[first.id]) for seed in range(6, ROWS + 1)]
    for row in made[::3]:
        storage.set_favorite(row.id, True)
    storage.bin_images([made[-1].id])
    original = _rows(storage.settings.db_path)
    original_files = _files(storage.settings.outputs_dir)
    settings = storage.settings
    archive = None
    existing, existing_files = [], {}
    if boundary.startswith("import"):
        archive = bundles.export_bundle(keep_gallery=True)["path"]
        storage.settings = dataclasses.replace(
            settings, db_path=root / "destination" / "auradream.db", outputs_dir=root / "destination" / "outputs",
        )
        storage.init_db()
        save(1001)
        save(1002)
        existing = _rows(storage.settings.db_path)
        existing_files = _files(storage.settings.outputs_dir)
    return storage, {"archive": archive, "gallery": str(storage.settings.db_path.parent),
                     "original": original, "original_files": original_files, "pixels": pixels,
                     "mask": mask, "existing": existing, "existing_files": existing_files}


def _signal(event: str, state: dict) -> None:
    print(json.dumps({"event": event, "pid": os.getpid(), "token": state["token"],
                      "parent_pid": os.getppid(), "executable": sys.executable,
                      "boundary": state["boundary"]}), flush=True)


def _worker(root: Path, token: str, recover: bool) -> None:
    state = json.loads((root / "owner.json").read_text(encoding="utf-8"))
    assert state["token"] == token and Path(state["root"]).resolve() == root
    gallery = Path(state["gallery"]).resolve()
    assert gallery.is_relative_to(root)
    _signal("started", state)
    _app(gallery)
    from backend import bundles, gallery_lock, storage
    storage.init_db()
    boundary = state["boundary"]

    def pause():
        _signal("boundary", state)
        sys.stdin.readline()
        raise AssertionError("the controller must terminate this paused worker")

    if not recover:
        def progress(phase: str, done: int, total: int) -> None:
            if boundary == "export-write" and phase == "writing" and done == 2:
                pause()  # database plus one real image member has been written
            elif boundary == "export-verified" and phase == "clearing" and done == 0:
                pause()  # verified archive published; no cleanup has started
        if boundary == "export-removed":
            real_remove = bundles._remove_file

            def remove(path):
                real_remove(path)
                if path.parent == storage.settings.outputs_dir and path.suffix == ".png":
                    pause()  # actual unlink, inside cleanup's open transaction
            bundles._remove_file = remove
        if boundary == "import-installed":
            real_install = bundles._install

            def install(source, target, installed):
                real_install(source, target, installed)
                if target.parent == storage.settings.outputs_dir and target.suffix == ".png":
                    pause()  # actual exclusive install, prior database still live
            bundles._install = install
        if boundary in {"import-before-publish", "import-published"}:
            real_replace = bundles._replace_database

            def replace(staged):
                if boundary == "import-before-publish":
                    pause()  # every install finished; merged database still private
                real_replace(staged)
                pause()  # live database has been replaced; finally has not run
            bundles._replace_database = replace
        if boundary.startswith("export"):
            bundles.export_bundle(expected=ROWS, progress=progress)
        else:
            bundles.import_bundle(Path(state["archive"]))
        raise AssertionError("the requested boundary was not reached")

    # A genuinely fresh interpreter must be able to take the dead worker's
    # OS reservation, recover SQLite, and sweep its actual aged leftovers.
    with gallery_lock.hold(storage.settings.db_path, wait=1.0):
        pass
    swept = storage.settle_partials()
    assert swept["discarded"] == len(state["leftovers"])
    assert all(not Path(path).exists() for path in state["leftovers"])
    if boundary in {"export-write", "export-verified"}:
        exported = bundles.export_bundle(expected=ROWS)
        assert exported["cleared"] and len(exported["deleted_ids"]) == ROWS
        archive = Path(exported["path"])
    else:
        archive = Path(state["archive"])
    imported = bundles.import_bundle(archive)
    repeated = bundles.import_bundle(archive)
    assert (repeated["imported"], repeated["present"], repeated["masks_missing"]) == (0, ROWS, 0)
    assert storage._bundle_leftovers() == []
    _signal("recovered", state)
    print(json.dumps({"swept": swept["discarded"], "imported": imported,
                      "repeated": repeated, "reservation_reacquired": True,
                      "archive_bytes": archive.stat().st_size}), flush=True)


def _read_line(child) -> dict:
    answer = queue.Queue()
    thread = threading.Thread(target=lambda: answer.put(child.stdout.readline()), daemon=True)
    thread.start()
    try:
        line = answer.get(timeout=30)
    except queue.Empty as exc:
        raise AssertionError("owned storage worker did not reach its boundary within 30 seconds") from exc
    if not line:
        _, error = child.communicate(timeout=5)
        raise AssertionError(f"storage worker exited before its boundary: {error}")
    return json.loads(line)


def _command(root: Path, token: str, recover: bool = False) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "--data", str(root),
            "--worker", "recover" if recover else "interrupt", "--token", token]


def _identify(child, command: list[str], state: dict) -> dict:
    started = _read_line(child)
    assert {key: started[key] for key in ("event", "token", "boundary")} == {
        "event": "started", "token": state["token"], "boundary": state["boundary"]}, started
    # Windows venv python.exe can be a redirector whose runtime is its child.
    assert started["pid"] == child.pid or started["parent_pid"] == child.pid, (child.pid, started)
    assert Path(started["executable"]).samefile(command[0]), started
    assert started["pid"] != os.getpid() and child.args == command
    return started


def _close_owned(child, runtime_pid: int | None, *, abort: bool, terminated: bool = False) -> None:
    if abort and runtime_pid is not None and not terminated:
        try:
            os.kill(runtime_pid, signal.SIGTERM if os.name == "nt" else signal.SIGKILL)
        except ProcessLookupError:
            pass
    if child.poll() is None:
        assert Path(child.args[1]).samefile(Path(__file__))
        child.kill()  # only this launcher's exact Popen handle
        child.wait(timeout=10)
    for stream in (child.stdin, child.stdout, child.stderr):
        stream.close()


def _kill_at_boundary(root: Path, state: dict) -> dict:
    command = _command(root, state["token"])
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    runtime_pid = None
    terminated = False
    try:
        started = _identify(child, command, state)
        runtime_pid = started["pid"]
        reached = _read_line(child)
        assert reached == {**started, "event": "boundary"}
        assert child.poll() is None
        from backend import gallery_lock

        blocked = False
        try:
            with gallery_lock.hold(Path(state["gallery"]) / "auradream.db", wait=0.0):
                pass
        except gallery_lock.GalleryBusy:
            blocked = True
        assert blocked  # the worker really holds the cross-process reservation
        os.kill(runtime_pid, signal.SIGTERM if os.name == "nt" else signal.SIGKILL)
        terminated = True  # only this positively identified, owned runtime
        assert child.wait(timeout=10) != 0
        return {"killed_pid": runtime_pid, "launcher_pid": child.pid, "reservation_held_before_kill": True}
    finally:
        _close_owned(child, runtime_pid, abort=True, terminated=terminated)


def _restart(root: Path, state: dict) -> list[dict]:
    command = _command(root, state["token"], recover=True)
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    runtime_pid = None
    complete = False
    try:
        started = _identify(child, command, state)
        runtime_pid = started["pid"]
        output, error = child.communicate(input="", timeout=60)
        complete = True
        assert child.returncode == 0, output + error
        return [started, *(json.loads(line) for line in output.splitlines())]
    finally:
        # subprocess.run(timeout=...) alone may stop only the Windows venv
        # redirector. Pin and stop the acknowledged runtime on timeout/error.
        _close_owned(child, runtime_pid, abort=not complete)


def _check_archive(path: Path) -> None:
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["records"] == ROWS
        for name, expected in manifest["files"].items():
            with archive.open(name) as member:
                assert isinstance(member, ZipExtFile), "reading a ZIP member requires its binary stream"
                assert hashlib.file_digest(member, "sha256").hexdigest() == expected["sha256"]
            assert archive.getinfo(name).file_size == expected["size"]


def _verify_recovered(storage, state: dict) -> None:
    from PIL import Image
    from backend.metadata import PNG_MASK_KEY, PNG_METADATA_KEY

    actual = {row["filename"]: row for row in _rows(storage.settings.db_path)}
    original = {row["filename"]: row for row in state["original"]}
    by_old_id = {row["id"]: row for row in state["original"]}
    assert set(actual) == set(original) | {row["filename"] for row in state["existing"]}
    mapping = {row["id"]: actual[name]["id"] for name, row in original.items()}
    for name, before in original.items():
        after = actual[name]
        ignored = {"id", "reference", "refine"}
        assert {k: v for k, v in after.items() if k not in ignored} == {
            k: v for k, v in before.items() if k not in ignored}
        references = [mapping[i] for i in json.loads(before["reference"])] if before["reference"] else None
        refine = mapping[before["refine"]] if before["refine"] else None
        assert (json.loads(after["reference"]) if after["reference"] else None) == references
        assert after["refine"] == refine
        with Image.open(storage.settings.outputs_dir / name) as image:
            assert image.mode == "RGB" and image.size == (SIDE, SIDE)
            assert hashlib.sha256(image.tobytes()).hexdigest() == state["pixels"][name]
            metadata = json.loads(image.info[PNG_METADATA_KEY])
            assert metadata["hash"] == after["content_hash"] == state["pixels"][name][:16]
            assert metadata["reference"] == references and metadata["refine"] == refine
            assert metadata["mask"] == after["mask"]
            def provenance(old_id):
                source = by_old_id[old_id]
                return {"filename": source["filename"], "hash": source["content_hash"],
                        "width": SIDE, "height": SIDE}

            assert metadata["reference_sources"] == (
                [provenance(i) for i in json.loads(before["reference"])] if before["reference"] else None)
            assert metadata["refine_source"] == (provenance(before["refine"]) if before["refine"] else None)
            if after["mask"]:
                assert PNG_MASK_KEY in image.info and metadata["mask_blend"] == "overlay"
                assert metadata["mask_source"] == {"filename": f"{after['mask']}.png", "hash": after["mask"],
                                                   "width": SIDE, "height": SIDE}
    for row in state["existing"]:
        assert actual[row["filename"]] == row
    files = _files(storage.settings.outputs_dir)
    assert set(files) == set(state["original_files"]) | set(state["existing_files"])
    for name, digest in {**state["original_files"], **state["existing_files"]}.items():
        if name.startswith(("thumbnails/", "masks/")) or name in state["existing_files"]:
            assert files[name] == digest
    assert storage._tracked_masks() == {state["mask"]}
    with Image.open(storage.mask_path(state["mask"])) as mask:
        assert storage.mask_hash(mask) == state["mask"]


def _run(root: Path, boundary: str) -> dict:
    storage, state = _fixture(root, boundary)
    state.update(root=str(root), boundary=boundary, token=uuid.uuid4().hex)
    owner = root / "owner.json"
    owner.write_text(json.dumps(state), encoding="utf-8")
    before_rows = _rows(storage.settings.db_path)
    before_files = _files(storage.settings.outputs_dir)
    killed = _kill_at_boundary(root, state)
    after_rows = _rows(storage.settings.db_path)  # a new connection recovers a killed transaction
    after_files = _files(storage.settings.outputs_dir)
    if boundary in {"export-write", "export-verified"}:
        assert (after_rows, after_files) == (before_rows, before_files)
    elif boundary == "export-removed":
        assert after_rows == before_rows
        assert len(set(before_files) - set(after_files)) == 1
    elif boundary == "import-published":
        assert len(after_rows) == ROWS + len(state["existing"])
        assert set(after_files) == set(state["original_files"]) | set(before_files)
        _verify_recovered(storage, state)  # publication must already be consistent, before a retry
    else:
        assert after_rows == before_rows
        assert set(after_files) > set(before_files)
        if boundary == "import-before-publish":
            assert set(after_files) == set(state["original_files"]) | set(before_files)
    for name in set(before_files) & set(after_files):
        assert after_files[name] == before_files[name]
    if boundary.startswith("export"):
        archives = list((storage.settings.db_path.parent / "bundles").glob("*.zip"))
        if boundary == "export-write":
            assert archives == []
            partials = list((storage.settings.db_path.parent / "bundles").glob("*.part"))
            assert len(partials) == 1 and partials[0].stat().st_size > 0
        else:
            assert len(archives) == 1
            _check_archive(archives[0])
            state["archive"] = str(archives[0])
    leftovers = storage._bundle_leftovers()
    assert leftovers  # actual artifacts from the killed worker, never planted stand-ins
    old = time.time() - storage.PARTIAL_SETTLE_SECONDS - 60
    for path in leftovers:
        resolved = path.resolve()
        assert resolved.is_relative_to(root) and resolved != root
        os.utime(path, (old, old))  # exercise the ordinary stale cutoff without sleeping an hour
    state["leftovers"] = [str(path) for path in leftovers]
    owner.write_text(json.dumps(state), encoding="utf-8")
    scratch_bytes = _size(root)
    assert scratch_bytes < 150 * 1024**2
    lines = _restart(root, state)
    assert [line["event"] for line in lines[:2]] == ["started", "recovered"]
    assert lines[0]["pid"] == lines[1]["pid"] and lines[0]["pid"] != killed["killed_pid"]
    assert all(line["token"] == state["token"] for line in lines[:2])
    _verify_recovered(storage, state)
    report = lines[2]
    assert report["reservation_reacquired"] and report["swept"] == len(leftovers)
    return {"boundary": boundary, **killed, "restart_pid": lines[0]["pid"],
            "rows_after_recovery": len(_rows(storage.settings.db_path)),
            "scratch_bytes_before_recovery": scratch_bytes, **report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="empty disposable scratch directory")
    parser.add_argument("--boundary", choices=BOUNDARIES)
    parser.add_argument("--worker", choices=("interrupt", "recover"), help=argparse.SUPPRESS)
    parser.add_argument("--token", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.data.resolve()
    if args.worker:
        _worker(root, args.token, recover=args.worker == "recover")
        return 0
    if args.boundary is None:
        parser.error("--boundary is required")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        parser.error("--data must be empty; existing galleries are never reused")
    parent = root if root.exists() else next(path for path in root.parents if path.exists())
    if shutil.disk_usage(parent).free < MIN_FREE:
        parser.error("the scratch volume needs at least 256 MiB free")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="storage-interrupt-", dir=root) as scratch:
        work = Path(scratch).resolve()
        assert work.is_relative_to(root)
        report = _run(work, args.boundary)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
