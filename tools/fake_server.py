"""Serve the web app on a fake backend against a throwaway gallery.

    uv run python tools/fake_server.py                 # http://127.0.0.1:7861
    uv run python tools/fake_server.py --step 0.3      # 0.3 s per denoising step
    uv run python tools/fake_server.py --fresh         # wipe the throwaway gallery first

No GPU, no weights: the backend is the tests' FakeBackend with a drawing on
top, so every "image" is a gradient stamped with its seed and steps and the
cells of a grid can be told apart on a sheet. The gallery lives under the
system temp directory (or --data), never in the checkout, so nothing here can
reach the real krea.db. Everything else - the job queue, SSE, resume, the
runs summary, pause, the budget - is the real code, which is the point:
`tools/browser.mjs` drives a headless browser against this to check the UI.

Port 7861, so it can run beside the real server on 7860.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--step", type=float, default=0.0, help="seconds per denoising step")
    parser.add_argument(
        "--data", default=str(Path(tempfile.gettempdir()) / "krea-fake"),
        help="where the throwaway gallery lives (default: <temp>/krea-fake)",
    )
    parser.add_argument("--fresh", action="store_true", help="delete the throwaway gallery first")
    parser.add_argument("--model", default="krea/Krea-2-Turbo", help="the checkpoint id to claim")
    args = parser.parse_args()

    data = Path(args.data)
    if args.fresh and data.exists():
        shutil.rmtree(data)
    data.mkdir(parents=True, exist_ok=True)
    # Before backend.config is imported: its settings freeze at import.
    os.environ["KREA_DB_PATH"] = str(data / "krea.db")
    os.environ["KREA_OUTPUTS_DIR"] = str(data / "outputs")
    os.environ["KREA_PORT"] = str(args.port)
    os.environ["KREA_MODEL_ID"] = args.model
    # An adapter directory of its own, with one file in it, so the LoRA select
    # has something to offer and the panel can be driven end to end. The fake
    # never opens it - it draws gradients - but every path up to the load is
    # the real one: the listing, the select, the axis, the recorded column.
    loras = data / "loras"
    loras.mkdir(exist_ok=True)
    demo = loras / "demo-style.safetensors"
    if not demo.exists():
        demo.write_bytes(b"not a real adapter; the fake backend never loads it")
    os.environ["KREA_LORAS_DIR"] = str(loras)

    from PIL import Image, ImageDraw

    from backend import main as app_main
    from backend.jobs import JobManager
    from tests.test_api import FakeBackend

    class DrawingBackend(FakeBackend):
        """A fake whose images carry their seed and steps as pixels."""

        def generate(self, request, seed, on_progress=None, should_cancel=None, on_preview=None):
            # Drawn first, so the steps can hand out ever-sharper previews of
            # it the way the real backend hands out latent previews.
            out = self._draw(request, seed)

            def progress(step, total):
                if on_progress is not None:
                    on_progress(step, total)
                if on_preview is not None and step < total:
                    import io

                    from PIL import ImageFilter

                    small = out[0].copy()
                    small.thumbnail((256, 256))
                    small = small.filter(ImageFilter.GaussianBlur(4 * (total - step) / total))
                    buffer = io.BytesIO()
                    small.save(buffer, format="JPEG", quality=70)
                    on_preview(buffer.getvalue(), step, total)

            super().generate(request, seed, progress, should_cancel)
            return out

        def _draw(self, request, seed):
            out = []
            for offset in range(request.num_images):
                image = Image.new("RGB", (request.width, request.height))
                draw = ImageDraw.Draw(image)
                w, h = image.size
                base = (seed * 37 + request.steps * 11) % 255
                band = max(1, h // 64)
                for y in range(0, h, band):
                    shade = (base + y * 255 // h) % 255
                    draw.rectangle([0, y, w, y + band], fill=(shade, (shade * 3) % 255, 255 - shade))
                for x in range(0, w, max(8, w // 16)):
                    draw.line([x, 0, x, h], fill=(255, 255, 255), width=1)
                draw.rectangle([10, 10, 260, 70], fill=(0, 0, 0))
                draw.text((16, 16), f"seed {seed + offset}", fill=(255, 255, 255))
                draw.text((16, 36), f"steps {request.steps} cfg {request.guidance_scale}", fill=(255, 255, 255))
                out.append(image)
            return out

    fake = DrawingBackend(step_seconds=args.step)
    # Behave as the claimed checkpoint's profile predicts: drop the negative
    # prompt below the threshold, keep real CFG apart where FLUX.1 does. The
    # drawing ignores the negative prompt anyway, so a grid that varies only
    # it renders identical images - which is what the duplicate warning and
    # the distinct count exist to show.
    from backend.model_defaults import defaults_for

    profile = defaults_for(args.model)
    fake.cfg_threshold = profile["cfg_threshold"]
    fake.true_cfg_supported = profile["true_cfg_supported"]
    # klein takes reference images and nothing else here does, so the greyed-out
    # panel the browser check drives is the one the real checkpoint would show.
    fake.reference_supported = profile["reference_supported"]
    app_main.backend = fake
    app_main.jobs = JobManager(fake, app_main.ensure_loaded)

    import uvicorn

    print(f"fake server on http://127.0.0.1:{args.port}  gallery: {data}", flush=True)
    uvicorn.run(app_main.app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
