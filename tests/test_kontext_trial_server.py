"""The INP#13 candidate is separately built, fenced and observable."""
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from tools import kontext_trial_runtime as runtime
from tools import kontext_trial_server as server
from tools.anima_refine_card import atomic_json


def test_build_digest_matches_lf_and_mixed_checkout_line_endings(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    patch = (server.ROOT / "tools/kontext_trial.patch").read_bytes().replace(b"\r\n", b"\n")
    sources = {"tools/kontext_trial.patch": patch, "tools/kontext_trial_runtime.py": b"# runtime\n"}
    for line in patch.decode().splitlines():
        if line.startswith("--- a/"):
            name = line.removeprefix("--- a/")
            sources[name] = (server.ROOT / name).read_bytes().replace(b"\r\n", b"\n")
    for suffix in ("css", "js", "html", "json", "md", "py", "svg", "webmanifest"):
        sources[f"web/fixture.{suffix}"] = b"\xef\xbb\xbffirst\nsecond\n"
    binary = b"\x89PNG\r\n\x1a\n\x00"
    sources["web/image.png"] = binary
    for name, content in sources.items():
        file = checkout / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content)
    subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
    subprocess.run(["git", "-c", "core.autocrlf=false", "add", "."], cwd=checkout, check=True)
    monkeypatch.setattr(server, "ROOT", checkout)
    lf = server.build_candidate(tmp_path / "lf")
    for name, content in sources.items():
        if name.endswith(".png"):
            continue
        # Leave some files entirely CRLF and others mixed in the same checkout.
        changes = 1 if "/fixture." in name else -1
        (checkout / name).write_bytes(content.replace(b"\n", b"\r\n", changes))
    mixed = tmp_path / "mixed"
    assert server.build_candidate(mixed) == lf
    assert (mixed / "web/image.png").read_bytes() == binary
    assert (mixed / "web/fixture.py").read_bytes() == sources["web/fixture.py"]


def test_build_selects_trial_tuples_without_changing_loader_or_instance_qualification(tmp_path):
    target = tmp_path / "build"
    digest = server.build_candidate(target)
    assert len(digest) == 64
    # Import in a fresh process: pytest may already have the shipping modules
    # cached. Using the copy must not silently select a live backend on import.
    subprocess.run([sys.executable, "-c", """
from backend import instruction_edit as e
from backend.backends import get_backend
from backend.schemas import InstructionEditSubmission
from pydantic import ValidationError
for recipe, w, h, ow, oh in [
    ('kontext-v1', 1024, 1024, 512, 512), ('kontext-v1', 1024, 576, 512, 288),
    ('kontext-v1', 576, 1024, 288, 512), ('kontext-v1', 1024, 768, 512, 384),
    ('kontext-v1', 768, 1024, 384, 512), ('kontext-v1', 1152, 768, 576, 384),
    ('kontext-v1', 768, 1152, 384, 576), ('kontext-1024-v1', 1024, 1024, 1024, 1024),
]:
    geometry = e.geometry(w, h, recipe=recipe)
    assert geometry.output_size == (ow, oh) and e.geometry_refusal(geometry) is None
    spec = e.InstructionEditPreview(source=1, recipe=recipe)
    req = InstructionEditSubmission(**spec.model_dump(), prompt='edit', width=ow, height=oh).request()
    for closed in (False, 1):
        try: e.require_execution_available(req.edit, qualified=closed)
        except ValueError as exc: assert str(exc) == e.INSTRUCTION_EDIT_UNAVAILABLE
        else: raise AssertionError('closed instance qualified')
    e.require_execution_available(req.edit, qualified=True)
assert e.loader_refusal(e.KONTEXT_V1.model_id, quantization='none', offload='model', device='cuda:0',
                        platform='win32', loader_execution=None, diffusers_version='0.42.1')
try: e.geometry(1024, 576, recipe='kontext-1024-v1')
except ValueError: pass
else: raise AssertionError('non-square larger recipe admitted')
try: get_backend()
except RuntimeError as exc: assert 'private build' in str(exc)
else: raise AssertionError('unguarded private backend')
try: e.InstructionEditPreview(source=1, recipe='other')
except ValidationError: pass
else: raise AssertionError('unknown recipe accepted')
"""], cwd=target, check=True)
    from backend import instruction_edit as release
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        release.InstructionEditPreview(source=1, recipe="kontext-1024-v1")
    assert release.geometry_refusal(release.geometry(1024, 1024, recipe="kontext-1024-v1"))
    assert release.geometry_refusal(release.geometry(1024, 576))
    with pytest.raises(ValueError, match="empty"):
        server.build_candidate(target)


@pytest.mark.parametrize("approved", [True, False])
def test_guard_matches_only_this_process_action_and_nonce(tmp_path, monkeypatch, approved):
    permit = tmp_path / "permit.json"
    atomic_json(permit, {"pid": os.getpid(), "nonce": "stale", "action": "edit", "approved": True})
    collision = threading.Event()
    original_open = Path.open

    def open_during_replace(path, mode="r", *args, **kwargs):
        if path == permit and mode == "rb" and not collision.is_set():
            collision.set()
            raise PermissionError(errno.EACCES, "Windows mailbox sharing violation")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(runtime.sidecar_io, "WINDOWS", True)
    monkeypatch.setattr(Path, "open", open_during_replace)
    outcomes = []

    def run():
        try:
            runtime.await_permit(tmp_path, "edit", seed=12346)
        except RuntimeError as exc:
            outcomes.append(str(exc))
        else:
            outcomes.append("accepted")

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    request = tmp_path / "request.json"
    deadline = time.monotonic() + 30
    while not request.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    data = runtime.read_mailbox(request)
    assert data is not None
    assert data["pid"] == os.getpid() and data["nonce"] != "stale" and data["action"] == "edit" and data["seed"] == 12346
    assert collision.wait(30) and worker.is_alive()
    atomic_json(permit, {**data, "approved": approved})
    worker.join(30)
    assert outcomes == (["accepted"] if approved else ["The live guard refused this edit."])


def test_real_mode_requires_supervision_and_trial_writes_are_bounded():
    with pytest.raises(ValueError, match="guard"):
        runtime.Trial("real", None)
    for forbidden in ("/api/model/switch", "/api/prompt/expand", "/api/prompt/critique", "/api/models/storage/delete"):
        assert not server.permitted("POST", forbidden)
    assert not server.permitted("DELETE", "/api/images/1")
    for allowed in ("/api/model/load", "/api/instruction-edit", "/api/jobs/abc/cancel", "/api/runs/xyz/finish"):
        assert server.permitted("POST", allowed)


def test_bound_proof_must_match_the_planted_original_at_this_id():
    from backend.instruction_edit import StoredInstructionEdit

    original = {"filename": "owned.png", "hash": "a" * 64, "width": 1024, "height": 1024}
    def bound(source, **changes):
        return StoredInstructionEdit.model_validate({
            "source": source, "recipe": "kontext-1024-v1", "edit_source": {**original, **changes},
        })

    runtime.require_fixture(bound(4), {4: original})
    for substituted in (bound(4, hash="b" * 64), bound(4, filename="another.png"), bound(5)):
        with pytest.raises(ValueError, match="differs from the fixture"):
            runtime.require_fixture(substituted, {4: original})
