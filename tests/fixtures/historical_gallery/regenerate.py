"""Rebuild synthetic galleries using the storage and export writers from Git history.

Run with the project's Python and an output directory. The normal test suite reads
the checked-in ZIPs and needs neither Git history nor these old imports.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID
import zipfile


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
WRITERS = {
    "legacy-v1": "16e9f059ff1ef602545e1cec2f1458222b641e1c",
    "edit-v2": "613ca303ff72a3aa6db701c6ff4bc547107493da",
}
STAMP = datetime(2026, 1, 2, 3, 4, 6, tzinfo=UTC)
KREA = "black-forest-labs/FLUX.1-Krea-dev"


def write_gallery(edition, destination):
    # This subprocess is started with -I and its cwd is the detached checkout.
    sys.path.insert(0, str(Path.cwd()))
    from PIL import Image
    from backend import bundles, gallery_delete, grid, storage
    from backend.schemas import GenerateRequest

    if Path(storage.__file__).resolve().parent.parent != Path.cwd():
        raise RuntimeError("The writer must come from the detached historical checkout")
    storage.init_db()

    def save(seed, prompt, rgb, *, model_id=KREA, variables=None, **fields):
        request = GenerateRequest(prompt=prompt, width=512, height=512, seed=seed, steps=28,
                                  mu=None, guidance_scale=2.5 if "edit" in fields else 3.5,
                                  **fields)
        run = grid.run_id([grid.Cell(request, seed)], model_id, "fp8")
        fixed_uuid = UUID(hex=f"{seed:06d}" + "0" * 26)
        with (Image.new("RGB", (512, 512), rgb) as image,
              patch.object(storage, "uuid", SimpleNamespace(uuid4=lambda: fixed_uuid)),
              patch.object(storage, "datetime", SimpleNamespace(now=lambda zone: STAMP))):
            return storage.save_image(image, request, seed, duration_ms=seed + 10, model_id=model_id,
                                      quantization="fp8", job_id=f"fixture-job-{seed}", run_id=run,
                                      batch=1, variables=variables)

    def backup(name, ids=None):
        # ZipFile.writestr uses the local clock for the manifest member's header.
        # Freeze only zipfile's time binding. The application still writes and
        # verifies the entire archive, including the manifest and SQLite backup.
        archive_clock = SimpleNamespace(time=STAMP.timestamp, localtime=time.gmtime)
        with patch.object(zipfile, "time", archive_clock):
            result = bundles.export_bundle(keep_gallery=True, ids=ids)
        shutil.copyfile(result["path"], destination / f"{name}.zip")

    if edition == "legacy-v1":
        source = save(101, "red study", (110, 20, 30), negative_prompt="glare", true_cfg=1.75,
                      variables={"color": "red"})
        save(102, "blue refinement", (30, 40, 130), refine=source.id, strength=0.375)
        discard = save(103, "green discard", (11, 95, 23))
        with patch.object(gallery_delete, "datetime", SimpleNamespace(now=lambda zone: STAMP)):
            storage.bin_images([discard.id])
    else:
        from backend.instruction_edit import KONTEXT_V1, StoredInstructionEdit, descriptor_from_image

        source = save(201, "teal square", (17, 80, 93))
        with Image.open(storage.settings.outputs_dir / source.filename) as image:
            proof = descriptor_from_image(source.filename, image)
        edit = StoredInstructionEdit(source=source.id, recipe="kontext-v1", edit_source=proof)
        child = save(202, "paint it ochre", (118, 62, 31), model_id=KONTEXT_V1.model_id, edit=edit)
        backup("edit-only-v2", [child.id])
    storage.set_favorite(source.id, True)
    backup(edition)


def regenerate(destination):
    destination.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(prefix="auradream-historical-writer-") as folder:
        checkout = Path(folder) / "checkout"
        subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO),
                        str(checkout)], check=True)
        for edition, revision in WRITERS.items():
            subprocess.run(["git", "checkout", "--quiet", "--detach", revision], cwd=checkout, check=True)
            subprocess.run([sys.executable, "-I", __file__, str(destination), "--writer", edition],
                           cwd=checkout, check=True)


def run_writer(edition, destination):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if revision != WRITERS[edition]:
        raise RuntimeError(f"Expected the pinned historical checkout for {edition}")
    with TemporaryDirectory(prefix="auradream-synthetic-gallery-") as folder:
        scratch = Path(folder)
        prefix = "LOCAL_DIFFUSION" if edition == "legacy-v1" else "AURADREAM"
        for key in list(os.environ):
            if key.startswith(("LOCAL_DIFFUSION_", "AURADREAM_")):
                del os.environ[key]
        os.environ.update({
            f"{prefix}_DB_PATH": str(scratch / "gallery.db"),
            f"{prefix}_OUTPUTS_DIR": str(scratch / "images"),
            f"{prefix}_BUNDLES_DIR": str(scratch / "archives"),
            f"{prefix}_MODEL_ID": KREA,
            "HF_HUB_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        })
        write_gallery(edition, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--writer", choices=WRITERS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.writer:
        run_writer(args.writer, args.destination.resolve())
    else:
        regenerate(args.destination.resolve())
