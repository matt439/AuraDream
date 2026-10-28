"""Serve the web app on a fake backend against a throwaway gallery.

    uv run python tools/fake_server.py                 # http://127.0.0.1:7861
    uv run python tools/fake_server.py --step 0.3      # 0.3 s per denoising step
    uv run python tools/fake_server.py --fresh         # new retained scratch gallery
    uv run python tools/fake_server.py --critic-delay 2   # a slower prompt critic
    uv run python tools/fake_server.py --no-critic     # as a platform with no critic build

No GPU, no weights: the backend is the tests' FakeBackend with a drawing on
top, so every "image" is a gradient stamped with its seed and steps and the
cells of a grid can be told apart on a sheet. A cell whose prompt holds
`[fail]` raises, as an out-of-memory cell would, so a grid's skipped cells
and the failure rule can be driven from the page; and `POST /api/fake/load`
(the fake's alone) makes the loads after it slow (`seconds`) or gated
(`fail=gated`, raised as huggingface_hub raises it), for the load's wait and
a failed load's card; `POST /api/fake/step` sets the seconds per step for
the jobs after it and answers the pace it replaced; `POST /api/fake/expand`
slows ERNIE's Expand on the server (`seconds`), holds it until the next call
(`hold`), makes it fail (`fail`, the message of a RuntimeError) or puts its
rewriter and translator back off disk (`fresh`); `POST /api/fake/model`
sets the startup overrides the status reports (`overrides`), a critic
that will not stop, failing a switch's teardown (`stuck_critic`), and the
Hub's answer to a switch's access check (`access`: "gated" or "not_found"
refuses every switch to another checkpoint, as the real check refuses one
this sign-in cannot download; the fake never asks the Hub); `POST
/api/fake/lora` moves an adapter out of the directory and back
(`present`); `POST /api/fake/bin-age` backdates binned images' `binned_at`
by `days`, for Empty recycle bin's age scope; `POST /api/fake/peak` sets the
(`allocated`, `reserved`) bytes the jobs after it save as their peak;
`POST /api/fake/instruction-edit` closes or reopens the instruction-edit
gate (`available`), which is open on a fake claiming FLUX.1 Kontext [dev]
alone - there an edit is the source proved and prepared, tinted and
stamped, and two more sources (3:2 and a refused 2:1) are planted. The
prompt check reads the tests' word tokenizers - a token per word or mark, at
each encoder's real budget - and the critic is the tests' FakeCritic, whose
answer is built from the prompt. The default gallery lives under the system
temp directory. --fresh creates a new scratch directory, prints its path
and retains it after shutdown. Explicit --data selects and may reuse the
supplied directory, including its saved settings; use genuinely new scratch
data for isolated runs. Before importing the application, the fake installs
a fixed refusal for Windows backup credential access. Cached credential
metadata and platform facts remain unchanged; this does not isolate reused
data from other filesystem or backup operations.
Everything else - the job queue, SSE, resume, the runs summary, pause, the
budget, the analyser's rules - is the real code, which is the point:
`tools/browser.mjs` drives a headless browser against this to check the UI.

Port 7861, so it can run beside the real server on 7860.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1] if __doc__ else "Serve an isolated fake backend.")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--step", type=float, default=0.0, help="seconds per denoising step")
    data_args = parser.add_mutually_exclusive_group()
    data_args.add_argument(
        "--data", default=None,
        help="use this gallery directory, preserving existing data (default: <temp>/auradream-fake)",
    )
    data_args.add_argument(
        "--fresh", action="store_true",
        help="create a new temporary gallery and retain its reported directory after shutdown",
    )
    parser.add_argument("--model", default="krea/Krea-2-Turbo", help="the checkpoint id to claim")
    parser.add_argument(
        "--critic-delay", type=float, default=0.4,
        help="seconds each prompt critique takes, so its in-flight state can be seen (default 0.4)",
    )
    parser.add_argument(
        "--no-critic", action="store_true",
        help="answer as a server with no prompt critic: the page hides its section",
    )
    return parser.parse_args(argv)


def _data_directory(args: argparse.Namespace) -> Path:
    """Select existing data or allocate one new, retained fresh directory."""
    if args.fresh:
        return Path(tempfile.mkdtemp(prefix="auradream-fake-"))
    return Path(args.data) if args.data is not None else Path(tempfile.gettempdir()) / "auradream-fake"


def _refuse_credential_api() -> NoReturn:
    raise RuntimeError("Windows backup credentials are unavailable in the fake server.")


def _install_credential_refusal() -> None:
    """Keep the fake's credential factory refused for all later workers."""
    from backend import backup_credentials

    backup_credentials._credential_api = _refuse_credential_api


def main() -> int:
    args = _parse_args()
    data = _data_directory(args)
    if args.fresh:
        print(f"fresh fake gallery retained: {data}", flush=True)
    data.mkdir(parents=True, exist_ok=True)
    # Before backend.config is imported: its settings freeze at import.
    os.environ["AURADREAM_DB_PATH"] = str(data / "auradream.db")
    os.environ["AURADREAM_OUTPUTS_DIR"] = str(data / "outputs")
    os.environ["AURADREAM_BUNDLES_DIR"] = str(data / "bundles")
    os.environ["AURADREAM_PORT"] = str(args.port)
    os.environ["AURADREAM_MODEL_ID"] = args.model
    # An adapter directory of its own, with one file in it, so the LoRA select
    # has something to offer and the panel can be driven end to end. The fake
    # never opens it - it draws gradients - but every path up to the load is
    # the real one: the listing, the select, the axis, the recorded column.
    loras = data / "loras"
    loras.mkdir(exist_ok=True)
    demo = loras / "demo-style.safetensors"
    if not demo.exists():
        demo.write_bytes(b"not a real adapter; the fake backend never loads it")
    # And one for another checkpoint, which the listing carries with the
    # reason and the select must not offer.
    stranger = loras / "other-family.safetensors"
    if not stranger.exists():
        stranger.write_bytes(b"not a real adapter; addressed to another denoiser")
    os.environ["AURADREAM_LORAS_DIR"] = str(loras)

    from PIL import Image, ImageDraw

    # Match the tests' factory boundary before app startup can create workers.
    _install_credential_refusal()
    from backend import main as app_main
    from backend import prompt_encoding
    from backend.jobs import JobManager
    from tests.fakes import FakeBackend, FakeCritic, fake_tokenizer

    class DrawingBackend(FakeBackend):
        """A fake whose images carry their seed and steps as pixels."""

        load_seconds = 0.0
        load_failure: str | None = None

        def load(self) -> None:
            # On the load's own daemon thread (detached.detached), as the real one.
            import time

            time.sleep(self.load_seconds)
            if self.load_failure == "gated":
                from huggingface_hub.errors import GatedRepoError

                gated = GatedRepoError.__new__(GatedRepoError)
                gated.args = (
                    f"Cannot access gated repo for url https://huggingface.co/{args.model}"
                    "/resolve/main/model_index.json. Access to model is restricted. You must "
                    "have access to it and be authenticated to access it. Please log in.",
                )
                raise gated
            super().load()

        # ERNIE's Expand, slowed, held or failing on request
        # (`POST /api/fake/expand`): a first use's minutes and the route's
        # failures, on the server's side. Held, an expansion waits - inside
        # the generation gate, as a real one runs - until the next call lets
        # it go, or 30 s, so a smoke's steps under a running Expand never race
        # its end however slow the machine.
        expand_seconds = 0.0
        expand_gate: threading.Event | None = None

        def expand_prompt(self, prompt, width, height, should_cancel=None):
            import time

            time.sleep(self.expand_seconds)
            if self.expand_gate is not None:
                self.expand_gate.wait(timeout=30)
            return super().expand_prompt(prompt, width, height, should_cancel)

        def generate(self, request, seed, on_progress=None, should_cancel=None, on_preview=None):
            self._last_peak = None
            if "[fail]" in request.prompt:
                raise RuntimeError("a [fail] cell, as the prompt asked")
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

            ran = super().generate(request, seed, progress, should_cancel)
            if request.edit is not None:
                # The tests' fake hands back the prepared source it proved;
                # the drawing above is what this fake saves.
                for image in ran:
                    image.close()
            return out

        def describe(self) -> dict:
            # Where an inpaint starts, as the real backend says it (the profile's
            # own `inpaint_start`, else `refine.library_strength`), so the Inpaint dialog
            # opens where the real one would on this fake too.
            return {**super().describe(), "inpaint_strength": inpaint_strength if self.inpaint_supported else None}

        def _draw(self, request, seed):
            # An instruction edit starts from its source as the real branch
            # does: the gate asked again, the pixels proved again and prepared
            # at the working size (`load_bound_source`), then tinted and
            # stamped with whose edit it is, so a smoke can tell it from its
            # source by eye and by pixel. One image, at the output size.
            if request.edit is not None:
                return [self._edit_image(request, seed)]
            # An inpaint changes only what its mask covers, through the real
            # composite (the tests' fake fill by seed): no stamp, so a smoke
            # can read every unmasked pixel against the source's.
            if request.refine is not None and request.mask is not None and self.inpaint_supported:
                return FakeBackend._inpaint(request, seed)
            # A refine pass starts from its source, as the real one does: the
            # gallery image at the requested size - cropped where the shape
            # differs - under a stamp saying whose refinement it is and at
            # what strength, so a strength sweep's cells can be told apart.
            refining = request.refine is not None and self.refine_supported
            out = []
            for offset in range(request.num_images):
                if refining:
                    from backend import refine

                    image = refine.source_image(request.refine, request.width, request.height).copy()
                    draw = ImageDraw.Draw(image)
                else:
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
                draw.rectangle([10, 10, 260, 70 + 20 * refining], fill=(0, 0, 0))
                draw.text((16, 16), f"seed {seed + offset}", fill=(255, 255, 255))
                draw.text((16, 36), f"steps {request.steps} cfg {request.guidance_scale}", fill=(255, 255, 255))
                if refining:
                    draw.text(
                        (16, 56), f"refine of {request.refine} @ {request.strength:g}", fill=(255, 255, 255)
                    )
                out.append(image)
            return out

        def _edit_image(self, request, seed):
            from backend.instruction_edit import require_execution_available
            from backend.instruction_edit_admission import load_bound_source

            require_execution_available(request.validate_edit(), qualified=self.instruction_edit_qualified())
            stored, resolved, prepared = load_bound_source(request.validate_edit(require_bound=True))
            with prepared:
                tint = Image.new("RGB", prepared.size, (255, 140, 0))
                image = Image.blend(prepared, tint, 0.35)
                tint.close()
            if image.size != resolved.output_size:
                raise RuntimeError("the fake edit's image is not at the recipe's output size")
            draw = ImageDraw.Draw(image)
            draw.rectangle([10, 10, 260, 50], fill=(0, 0, 0))
            draw.text((16, 16), f"edit of #{stored.source} · seed {seed}", fill=(255, 255, 255))
            draw.text((16, 32), f"steps {request.steps} cfg {request.guidance_scale}", fill=(255, 255, 255))
            return image

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
    # And a refine pass only where diffusers has an image-to-image pipeline for
    # the claimed checkpoint: greyed on a Krea 2 or klein fake, live on a
    # FLUX.1, SDXL, Z-Image or Qwen-Image one.
    fake.refine_supported = profile["refine_supported"]
    # And an inpaint only where the profile enables one (SDXL, Z-Image and
    # FLUX.1), starting where the real backend does: the profile's own
    # measured start (FLUX.1's 0.95), else the strength the installed
    # inpaint class of the profile's family defaults to.
    fake.inpaint_supported = profile["inpaint_supported"]
    inpaint_strength = None
    if fake.inpaint_supported:
        import diffusers

        from backend import refine
        from backend.model_defaults import inpaint_family, inpaint_start

        inpaint_strength = inpaint_start(args.model)
        if inpaint_strength is None:
            family = inpaint_family(args.model)
            assert family is not None, "an inpaint-capable profile requires a pipeline family"
            sibling = refine.sibling(getattr(diffusers, family), "Inpaint")
            assert sibling is not None, "an inpaint-capable profile requires its inpaint pipeline"
            inpaint_strength = refine.library_strength(sibling)
    # A Mu only where the claimed checkpoint's schedule can take one: on an
    # Anima or SDXL fake the row drops it, as the real applied() does.
    fake.mu_supported = profile["mu_supported"]
    # And no adapter where the real backend refuses one by the id alone
    # (Anima Turbo, NewBie): the same sentence, from the same function, so
    # /api/loras, the greyed select and a submit's refusal are the real ones.
    from backend.backends.loading import lora_refusal

    fake.lora_reason = lora_refusal(args.model)
    # ERNIE's browser check drives the real button and endpoint without the
    # multi-GB rewriter. FakeBackend's deterministic sentence makes it clear
    # that the chosen canvas size reached the operation.
    fake.prompt_expander_supported = "ernie-image" in args.model.lower().replace("_", "-")
    fake.lora_mismatches = {
        stranger.name: f"{stranger.name} attaches to 494 modules the fake's denoiser "
        "has not got, such as single_transformer_blocks.0.attn.to_k",
    }
    # Instruction editing: qualified on a fake claiming FLUX.1 Kontext [dev]
    # alone, so the edit route, queue and dialog run end to end there while
    # every other fake - and the real server, whose backend answers the
    # literal False - holds them with the reason. `POST
    # /api/fake/instruction-edit` closes and reopens the gate.
    from backend.instruction_edit import KONTEXT_V1

    fake.edit_qualified = args.model == KONTEXT_V1.model_id
    app_main.backend = fake
    app_main.jobs = JobManager(fake, app_main.ensure_loaded)

    # Not "downloaded" until a first translation, as the fake's rewriter is not
    # until a first expansion (FakeBackend), so Expand's first-use line is on
    # show on a newly started server and gone after its first expansion.
    class FakeTranslator:
        downloaded = False

        def translate(self, text: str, should_cancel=None) -> str:
            if should_cancel is not None and should_cancel():
                from backend.backends.base import Cancelled

                raise Cancelled("prompt expansion stopped")
            self.downloaded = True
            return f"English translation of: {text}"

        def cached(self) -> bool:
            return self.downloaded

        def unload(self) -> None:
            pass

    translator = FakeTranslator()
    app_main.prompt_translator = translator

    # The critic without llama.cpp or 4 GB of model: the real route, gate and
    # busy refusals around a deterministic answer. Not "downloaded" until the
    # first critique prepares it, so the page's first-use line is on show
    # before and its timing line after, as they would be on a new install.
    critic = FakeCritic(delay=args.critic_delay)
    critic.available = not args.no_critic
    app_main.prompt_critic = critic
    # And the check's tokenizers without the Hub: realistic counts and real
    # offsets, so a long prompt on a FLUX-claiming fake shows its CLIP cut.
    prompt_encoding._load_tokenizer = fake_tokenizer

    # Deterministic stand-in for Danbooru's live tag API. Browser smoke tests
    # exercise autocomplete, caret replacement and per-item validation without
    # making the test suite depend on the network or the current tag database.
    from backend.tags import display_tag, is_model_tag, normalize_tag

    class FakeTagVocabulary:
        records = {
            "1girl": ("general", 8_000_000),
            "green_hair": ("general", 590_000),
            "green_hat": ("general", 41_000),
            "green_eyes": ("general", 900_000),
            "firefly_(honkai:_star_rail)": ("character", 16_000),
            "honkai:_star_rail": ("copyright", 160_000),
            # Artists, for Anima's "put @ before an artist name".
            "kantoku": ("artist", 9_000),
            "wlop": ("artist", 31_000),
        }

        def suggest(self, query, cutoff, limit=20):
            wanted = normalize_tag(query)
            return [
                {
                    "text": display_tag(name),
                    "canonical": name,
                    "category": category,
                    "post_count": count,
                }
                for name, (category, count) in self.records.items()
                if wanted in name
            ][:limit]

        # The finder's wiki: each tag's page text, searched for the words
        # typed as the real one is (a tag named them is asked for too), and
        # its first sentence. "fake offline" is the index unreachable.
        wiki = {
            "ahoge": ("general", 994_000, 'Literally "stupid hair", an ahoge is a single long lock '
                      "of hair that sticks out of a character's head.", ""),
            "cowlick": ("general", 11_000, "An unruly tuft of hair that sticks out from the head; "
                        "use ahoge for a single long lock.", ""),
            "flipped_hair": ("general", 43_000, None, "See also: ahoge, cowlick, hair flaps."),
        }

        def describe(self, query, cutoff, limit=12):
            from backend.tags import TagLookupUnavailable

            wanted = " ".join(query.split()).lower()
            if wanted == "fake offline":
                raise TagLookupUnavailable("fake Danbooru offline")
            found = [
                (name, category, count, definition)
                for name, (category, count, definition, rest) in self.wiki.items()
                if wanted in f"{definition or ''} {rest}".lower() or name == normalize_tag(wanted)
            ]
            found.sort(key=lambda row: (row[0] != normalize_tag(wanted), -row[2]))
            return [
                {
                    "text": display_tag(name),
                    "canonical": name,
                    "category": category,
                    "post_count": count,
                    "definition": definition,
                }
                for name, category, count, definition in found
            ][:limit]

        def validate(self, tags, special_tags, cutoff, allow_year_tag=True):
            results = []
            for tag in tags:
                name = normalize_tag(tag)
                if is_model_tag(tag, special_tags, allow_year_tag):
                    results.append(
                        {
                            "tag": tag,
                            "valid": True,
                            "canonical": tag,
                            "source": "model",
                        }
                    )
                elif name in self.records:
                    category, count = self.records[name]
                    results.append(
                        {
                            "tag": tag,
                            "valid": True,
                            "canonical": display_tag(name),
                            "source": "danbooru",
                            "category": category,
                            "post_count": count,
                        }
                    )
                else:
                    results.append(
                        {
                            "tag": tag,
                            "valid": False,
                            "reason": "not found in Danbooru",
                        }
                    )
            return results

    from backend import tag_lookup

    tag_lookup.tag_vocabulary = FakeTagVocabulary()

    # Two images made on another checkpoint, so the gallery's scope has
    # something to hide: by default the page shows the claimed checkpoint's
    # images alone, and the All models tick reads "(+2)". Planted once per
    # throwaway gallery, under the id the fake is not claiming, and one of
    # them starred so Delete unstarred has a keeper it must not count.
    from backend import storage
    from backend.schemas import GenerateRequest

    storage.init_db()
    other = "krea/Krea-2-Turbo" if args.model != "krea/Krea-2-Turbo" else "Tongyi-MAI/Z-Image-Turbo"
    # Planted once, the recycle bin's included: a smoke that binned them must
    # not find a second pair on the next start.
    if not any(storage.list_images(limit=1, model_id=other, binned=binned)[1] for binned in (False, True)):
        request = GenerateRequest(prompt="a lighthouse on another checkpoint", width=256, height=256)
        planted = [
            storage.save_image(
                fake._draw(request, seed)[0], request, seed=seed, duration_ms=1,
                model_id=other, job_id="planted", run_id="planted",
            )
            for seed in (1, 2)
        ]
        storage.set_favorite(planted[0].id, True)
    # And on a Kontext fake, two more sources for Edit by instruction…, under
    # the same other id: a 3:2 one (768x512, edited at 576x384) and a 2:1 one
    # (512x256), whose aspect the recipe refuses. The 256x256 pair above are
    # the 1:1 sources (edited at 512x512). Planted once, binned rows counted.
    if fake.edit_qualified:
        with storage._session() as conn:
            edit_sources = conn.execute("SELECT 1 FROM images WHERE job_id = 'planted-edit' LIMIT 1").fetchone()
        if edit_sources is None:
            for width, height in ((768, 512), (512, 256)):
                request = GenerateRequest(
                    prompt=f"a harbour at {width}x{height} on another checkpoint", width=width, height=height,
                )
                storage.save_image(
                    fake._draw(request, 3)[0], request, seed=3, duration_ms=1,
                    model_id=other, job_id="planted-edit", run_id="planted-edit",
                )

    # The fake's own knob for the load, which no flag can reach once the
    # smokes' runner has started the server: how long the loads after it take,
    # and whether they fail as a gated repo does.
    from fastapi import Body

    @app_main.app.post("/api/fake/load", include_in_schema=False)
    async def fake_load(seconds: float = Body(0.0), fail: str | None = Body(None)) -> dict:
        fake.load_seconds, fake.load_failure = seconds, fail
        return {"seconds": seconds, "fail": fail}

    # And the steps' pace, for a smoke step that needs a job to outlast its
    # own round trips - a cancel landing before the grid ends, a stream
    # attaching before the pass is done - at any speed the server was started
    # with. Answers the pace it replaces, for the step to put back.
    @app_main.app.post("/api/fake/step", include_in_schema=False)
    async def fake_step(seconds: float = Body(0.0, embed=True)) -> dict:
        previous, fake.step_seconds = fake.step_seconds, max(0.0, seconds)
        return {"seconds": fake.step_seconds, "previous": previous}

    # ERNIE's Expand: the seconds each expansion takes on the server, inside
    # the generation gate as the real one's are, and an error the expansions
    # after it raise (None for none) - "RuntimeError" for the class name the
    # route strips. `hold` keeps the expansions after it waiting until a
    # call without it (any call: each sets every knob); `fresh` puts the
    # rewriter and the translator back off disk, so a first use's line shows
    # again on a server a smoke has already run against. Answers what it set.
    expand_gate = threading.Event()
    fake.expand_gate = expand_gate
    expand_gate.set()

    @app_main.app.post("/api/fake/expand", include_in_schema=False)
    async def fake_expand(
        seconds: float = Body(0.0),
        fail: str | None = Body(None),
        hold: bool = Body(False),
        fresh: bool = Body(False),
    ) -> dict:
        fake.expand_seconds = max(0.0, seconds)
        fake.expand_error = RuntimeError(fail) if fail else None
        if hold:
            expand_gate.clear()
        else:
            expand_gate.set()
        if fresh:
            fake.expander_cached = fake.expander_loaded = False
            translator.downloaded = False
        return {"seconds": fake.expand_seconds, "fail": fail, "hold": hold, "fresh": fresh}

    # The Models picker's server facts no page can cause: which saved choices
    # this start's flags or environment override (`overrides`, as
    # launch.configure records them, e.g. {"model_id": "--model"}), a
    # critic that will not stop (`stuck_critic`), so a switch stops before
    # selecting its checkpoint, and the Hub's answer to a switch's access
    # check (`access`: None lets every switch through; "gated" or
    # "not_found" refuses it in hub_access's own words). Each call sets all
    # three; answers what it set.
    from backend import hub_access
    from backend.critic import CriticError

    critic_unload = critic.unload

    def stuck_unload():
        raise CriticError("the fake critic will not stop")

    # The fake never asks the Hub: the real check would reach it for any
    # catalogue id not cached on this machine.
    fake_access: dict[str, str | None] = {"kind": None}

    def fake_check(model_id: str):
        kind = fake_access["kind"]
        if kind is None:
            return None
        found = hub_access.sources(model_id) or [hub_access.Source(model_id, None, None)]
        return hub_access._refusal(model_id, found[0], kind)

    hub_access.check = fake_check

    @app_main.app.post("/api/fake/model", include_in_schema=False)
    async def fake_model(
        overrides: dict[str, str] = Body({}), stuck_critic: bool = Body(False),
        access: str | None = Body(None),
    ) -> dict:
        app_main._startup_overrides = overrides
        critic.unload = stuck_unload if stuck_critic else critic_unload
        fake_access["kind"] = access if access in ("gated", "not_found") else None
        return {"overrides": overrides, "stuck_critic": stuck_critic, "access": fake_access["kind"]}

    # An adapter taken out of the directory while a page holds it, and put
    # back (`present`): moved aside under a name the listing skips, as a
    # rename or a delete by hand would leave it. Answers whether it is there.
    @app_main.app.post("/api/fake/lora", include_in_schema=False)
    async def fake_lora(name: str = Body(...), present: bool = Body(True)) -> dict:
        path = loras / Path(name).name
        aside = path.with_name(path.name + ".aside")
        if present and aside.exists():
            aside.replace(path)
        elif not present and path.exists():
            path.replace(aside)
        return {"name": path.name, "present": path.exists()}

    # The instruction-edit gate, as a backend instance answers it: `available`
    # true opens it (the edit route queues, the dialog's Run is live where
    # FLUX.1 Kontext [dev] is selected), false closes it with the real
    # server's reason. Answers what it set and what it replaced.
    @app_main.app.post("/api/fake/instruction-edit", include_in_schema=False)
    async def fake_instruction_edit(available: bool = Body(..., embed=True)) -> dict:
        previous, fake.edit_qualified = fake.edit_qualified is True, available is True
        return {"available": fake.edit_qualified, "previous": previous}

    # A measured peak, which a fake never has: the (allocated, reserved)
    # bytes the jobs after it save, or null for none again. Answers it.
    @app_main.app.post("/api/fake/peak", include_in_schema=False)
    async def fake_peak(allocated: int | None = Body(None), reserved: int | None = Body(None)) -> dict:
        fake.peak = (allocated, reserved) if allocated is not None and reserved is not None else None
        return {"peak": fake.peak}

    # Time in the recycle bin, which no page can make: these binned images
    # backdated to `days` ago, for Empty recycle bin's age scope. Live rows
    # are left live. Answers the ids it moved.
    @app_main.app.post("/api/fake/bin-age", include_in_schema=False)
    async def fake_bin_age(ids: list[int] = Body(...), days: float = Body(...)) -> dict:
        from datetime import datetime, timedelta, timezone

        stamp = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with storage._write_lock, storage._session() as conn:
            moved = [
                row[0] for row in conn.execute(
                    f"UPDATE images SET binned_at = ? WHERE binned_at IS NOT NULL"
                    f" AND id IN ({','.join('?' * len(ids))}) RETURNING id",
                    (stamp, *ids),
                )
            ] if ids else []
        return {"ids": sorted(moved), "binned_at": stamp}

    import uvicorn

    print(f"fake server on http://127.0.0.1:{args.port}  gallery: {data}", flush=True)
    uvicorn.run(app_main.app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
