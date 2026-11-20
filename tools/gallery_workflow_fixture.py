"""Plant the 5,000-row gallery workflow benchmark into a new scratch directory."""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "black-forest-labs/FLUX.1-Krea-dev"


def copy_database(source: Path, target: Path) -> None:
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as reader:
        with closing(sqlite3.connect(target)) as writer:
            reader.backup(writer)


def plant(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    os.environ.update(AURADREAM_DB_PATH=str(directory / "auradream.db"),
                      AURADREAM_OUTPUTS_DIR=str(directory / "outputs"), AURADREAM_MODEL_ID=MODEL)
    sys.path.insert(0, str(ROOT))
    from PIL import Image

    from backend import storage
    from backend.schemas import GenerateRequest

    storage.init_db()
    favorites = []
    for group in range(250):
        subject = ("mountain", "lake", "forest", "orchard", "lighthouse")[group % 5]
        run = f"{group + 1:012x}"
        request = GenerateRequest(prompt=f"A {subject} in evening light, study {group}",
                                  steps=4, width=256, height=256).applied(
                                      negative_prompt_supported=False, cfg_threshold=1, mu_supported=False)
        for i in range(20):
            seed = 20 * group + i + 1
            image = Image.new("RGB", (256, 256), (group % 256, i * 12, seed % 256))
            row = storage.save_image(image, request, seed, 1, MODEL, job_id=run, run_id=run,
                                     quantization="fp8", batch=1)
            if i % 4 == 0:
                favorites.append(row.id)
    for start in range(0, len(favorites), 100):
        storage.set_favorites(favorites[start:start + 100], True)
    print(f"Planted 5,000 images in {directory}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    plant(parser.parse_args().directory.resolve())
