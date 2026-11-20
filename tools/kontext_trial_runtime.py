"""Runtime of the isolated INP#13 build. Only the trial launcher installs a backend."""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.backends.base import GenerationBackend
    from backend.instruction_edit import StoredInstructionEdit


_backend: GenerationBackend | None = None


def require_fixture(edit: StoredInstructionEdit, proofs: dict[int, dict]) -> None:
    if edit.source is None or proofs.get(edit.source) != edit.edit_source.model_dump():
        raise ValueError("The bound source differs from the fixture planted for this trial.")


def trial_backend() -> GenerationBackend:
    if _backend is None:
        raise RuntimeError("Run the private build through kontext_trial_server.py.")
    return _backend


def await_permit(directory: Path, action: str, **details: object) -> None:
    """Correlate the supervisor's fresh decision to this one synchronous call."""
    identity = {"pid": os.getpid(), "nonce": uuid.uuid4().hex, "action": action}
    temporary = directory / "request.tmp"
    temporary.write_text(json.dumps({**identity, **details}), encoding="utf-8")
    temporary.replace(directory / "request.json")
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        try:
            permit = json.loads((directory / "permit.json").read_bytes())
        except FileNotFoundError:
            permit = {}
        if all(permit.get(key) == value for key, value in identity.items()):
            if permit.get("approved") is True:
                return
            raise RuntimeError(f"The live guard refused this {action}.")
        time.sleep(0.02)
    raise RuntimeError(f"No fresh guard decision arrived for this {action}.")


class Trial:
    def __init__(self, mode: str, guard: Path | None):
        if mode not in {"real", "fake"} or (mode == "real" and guard is None):
            raise ValueError("The real trial needs its live guard.")
        self.mode = mode
        self.guard = guard
        self.load_attempted = False
        self.loads = 0
        self.accepted_unloads = 0

    def install(self, fixture_proofs: dict[int, dict]) -> None:
        global _backend
        if _backend is not None:
            raise RuntimeError("The trial backend is already installed.")
        from backend.config import settings
        from backend import card_facts
        from backend.instruction_edit import StoredInstructionEdit

        fake = None
        if self.mode == "real":
            from backend.backends.diffusers_backend import DiffusersBackend
            backend: GenerationBackend = DiffusersBackend()
        else:
            from backend import prompt_encoding
            from tests.fakes import FakeBackend, fake_tokenizer

            # As in the ordinary smoke server, never probe a card or fetch a tokenizer.
            card_facts.card = lambda device="cuda": None
            prompt_encoding._load_tokenizer = fake_tokenizer
            fake = FakeBackend(loaded=False, step_seconds=0.025)
            fake.edit_qualified = True
            # Synthetic allocator readings for metadata/carry agreement only.
            fake.peak = (96 * 1024**2, 128 * 1024**2)
            backend = fake

        load = backend.load
        generate = backend.generate
        try_unload = backend.try_unload

        def guarded_load():
            if self.load_attempted:
                raise RuntimeError("The one-load trial has already attempted its load.")
            self.load_attempted = True
            if self.mode == "real":
                if settings.device != "cuda:0" or (found := card_facts.card(settings.device)) is None or (
                    found.name != "NVIDIA GeForce RTX 5080"
                ):
                    raise RuntimeError("The real trial requires the Windows RTX 5080 on CUDA 0.")
            if self.guard is not None:
                await_permit(self.guard, "load")
            load()
            self.loads += 1

        def guarded_generate(request, seed, on_progress=None, should_cancel=None, on_preview=None):
            edit = request.validate_edit(require_bound=True)
            if not isinstance(edit, StoredInstructionEdit) or self.loads != 1 or self.accepted_unloads:
                raise ValueError("The trial accepts only bound instruction edits on its one live load.")
            require_fixture(edit, fixture_proofs)
            if self.guard is not None:
                await_permit(
                    self.guard, "edit", source=edit.source, recipe=edit.recipe, seed=seed,
                    width=request.width, height=request.height, prompt=request.prompt,
                    steps=request.steps, guidance_scale=request.guidance_scale,
                )
            if fake is not None:
                fake.step_seconds = 0.4 if seed == 777 else 0.025
            return generate(request, seed, on_progress, should_cancel, on_preview)

        def observed_unload():
            accepted = try_unload()
            if accepted:
                self.accepted_unloads += 1
            return accepted

        backend.load = guarded_load
        backend.generate = guarded_generate
        backend.try_unload = observed_unload
        _backend = backend

    def status(self, digest: str) -> dict:
        allocated, reserved = 0, 0
        if self.mode == "real" and self.load_attempted:
            import torch
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
        return {"pid": os.getpid(), "mode": self.mode, "build_digest": digest, "loads": self.loads,
                "accepted_unloads": self.accepted_unloads, "ready": trial_backend().is_ready(),
                "allocator": {"allocated_bytes": allocated, "reserved_bytes": reserved}}
