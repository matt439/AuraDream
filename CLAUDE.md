# Local Diffusion

Local text-to-image generator on an RTX 5080. FastAPI backend + vanilla JS
frontend, no build step. The default checkpoint is **FLUX.1 Krea [dev]**
(`black-forest-labs/FLUX.1-Krea-dev`), which is click-through gated; **Krea 2
Turbo** (`krea/Krea-2-Turbo`) needs manual approval on HuggingFace and is what
the sizing notes in `docs/design-notes.md` are written around. The catalogue
in `backend/launch.py` lists the rest; any of them runs unchanged - set
`LOCAL_DIFFUSION_MODEL_ID`, or answer the start-up menu.

## Run

```powershell
uv run python -m backend.main      # http://127.0.0.1:7860; bare on a terminal, asks
uv run python -m backend.main --defaults   # LOCAL_DIFFUSION_* as set, no questions
uv run --group dev pytest          # no GPU, no weights, a few seconds
```

```bash
bash wsl/setup.sh                  # WSL 2, once: Linux venv + CUDA check
source wsl/env.sh && bash wsl/serve.sh   # 4-bit Nunchaku, same URL
```

The model is *not* loaded at startup. The first `POST /api/generate` (or
`POST /api/model/load`) triggers a load that takes a few minutes.

## Layout

- `backend/config.py` — settings, most overridable by `LOCAL_DIFFUSION_*` env vars
- `backend/launch.py` — the start-up menu and the `--model/--quant/--offload`
  flags, run before `config` is imported; the checkpoint catalogue lives here
- `backend/schemas.py` — pydantic request/response models
- `backend/model_defaults.py` — recommended steps/CFG/mu per checkpoint and
  the control tooltips; each profile hangs its prompting guide from
- `backend/guides.py` — the guides themselves, one dict per checkpoint: prompt
  and size prose, the documented megapixel range with its word, the authors'
  example prompts
- `backend/seeds.py` — the seed range and the wrap, shared by server and CLI
- `backend/loras.py` — the adapter directory: what is in it, and the file
  behind a name (the only place a request's name becomes a path)
- `backend/sizes.py` — the preset table; the CLI's `--aspect/--mp`, config's
  `max_side`/`max_pixels`, `web/state.js`'s copy, `index.html`'s `max=` and the
  README table are all held to it by `tests/test_profiles.py`
- `backend/backends/base.py` — the `GenerationBackend` ABC
- `backend/backends/diffusers_backend.py` — in-process diffusers implementation
- `backend/backends/loading.py` — everything decided before the weights
  move: the quantization config, the Nunchaku preflight and the
  prequantized refusal, the components left out of the download, the
  device placement; takes `settings` as a parameter
- `backend/backends/schedule.py` — how a requested mu becomes the whole
  shift: `forced_mu`, `shifted_sigmas`, the static-shift fallback
- `backend/grid.py` — axis specs (`6-12x2`) and the cell expansion, shared
- `backend/latent_preview.py` — the latent-to-RGB maps the live preview
  draws with, one per VAE, fitted on the card
- `backend/jobs.py` — single-worker queue, SSE progress
- `backend/storage.py` — SQLite index + PNGs with embedded metadata
- `backend/search.py` — the search box's grammar: `key:value` terms as SQL
  predicates, the LIKE escape, `strict_scope`
- `backend/main.py` — routes
- `backend/generate.py` — headless CLI; drives the backend directly, no server
- `backend/clear.py` — deletes rows, their PNGs and any orphans
- `backend/stats.py` — the starred *rate* per setting (`storage.keep_rates`),
  scoped to a run by default
- `backend/reindex.py` — rows rebuilt from the PNGs' `local-diffusion` chunk, in place or
  copied in from another gallery; idempotent on the UNIQUE filename
- `web/` — served as-is from `/static`; the page script is ten classic
  scripts `index.html` loads in order (`state`, `sizes`, `status`,
  `generation`, `panel`, `gallery`, `sheet`, `lightbox`, `tags`, `app`),
  sharing one global scope; only `app.js`, the last, runs anything at load
- `tools/` — `fake_server.py` (the app on the tests' fake backend, port 7861)
  and `browser.mjs` + `smoke.json`, `smoke-flux.json`, `smoke-preview.json`
  (headless browser over DevTools): the UI's stand-in for a test suite, see
  Tests
- `wsl/` — `env.sh` (Linux-side paths and the Nunchaku defaults, to source),
  `setup.sh`, `serve.sh`: running the same checkout from WSL 2
- `docs/design-notes.md` — why each part is the way it is, one section per
  area, with the history behind every rule below; read the section for an
  area before changing it
- `docs/verified.md` — what has been measured on the card, per checkpoint
  and per feature, and when

## Rules

The reasoning behind each of these, and what went wrong before it existed, is
in `docs/design-notes.md` under a heading of the same subject; read that
section before changing the area. `docs/verified.md` lists what has been
measured on the card and when.

Model and loader:

- App-owned names use Local Diffusion: `LOCAL_DIFFUSION_*`,
  `local-diffusion.db`, `~/local-diffusion`, `local-diffusion.*` browser keys
  and the `local-diffusion` PNG chunk. Keep Krea only in checkpoint/vendor
  names and explicit compatibility reads for pre-rename data.
- Quantize through `PipelineQuantizationConfig(quant_mapping=...)`, one
  `TorchAoConfig` per component from the library that owns it, each built from
  a torchao `AOBaseConfig` *instance*. The `quant_backend="torchao"` shorthand
  raises on a transformers text encoder.
- `torch.no_grad()`, never `torch.inference_mode()`, around the pipeline call:
  model offload moves components inside `forward()` and torchao's tensor
  subclasses trip torch 2.13's storage-aliasing guard under inference_mode.
- Loading happens only in `main.ensure_loaded()` under its asyncio lock; jobs
  await it, `generate()` raises on a missing pipeline, and
  `POST /api/model/unload` answers 409 while a job or a load is live.
- `load()` refuses a pipeline whose `__call__` lacks `generator` or
  `callback_on_step_end` (`_REQUIRED_PARAMS`): the seed and the cancel depend
  on them.
- `LOCAL_DIFFUSION_QUANT=nunchaku` means a pre-quantized lite-infer repo, Linux only
  (WSL 2 here), and no quantization config; `loading.nunchaku_preflight` and `loading.refuse_prequantized` refuse the other pairings before any download. Under
  WSL, `source wsl/env.sh` before `uv`.
- `config.py` sets `DIFFUSERS_TRUST_REMOTE_KERNELS` and must be imported
  before diffusers everywhere, the CLI included. The offload hooks are handed
  `LOCAL_DIFFUSION_DEVICE` explicitly; given nothing, diffusers takes GPU 0.
- `loading.UNUSED_COMPONENTS` names what a pipeline lists but the backend never runs
  (ERNIE-Image's prompt rewriter). `load()` reads `model_index.json` ahead of
  the weights and hands them as None to the *concrete* class's
  `from_pretrained`, which keeps them out of the download and the load; the
  base class takes `pe=None` for a pipeline kwarg and pulls the 7.7 GB.
  `generate()` sets `use_pe=False` wherever the switch exists. The tests'
  conftest refuses `loading.hub_file` for every test; one that needs an index stubs it.

Schedule and guidance:

- A requested `mu` is the whole shift. `schedule.forced_mu` overrides the value
  on its way into a dynamically shifting scheduler; a static one gets
  `schedule.shifted_sigmas` pre-shifted by `mu - ln(shift)`. Assert on
  `scheduler.sigmas`, never on the list the backend computed.
- Where guidance switches on is the pipeline's: `cfg_threshold()` is 1.0 with
  `true_cfg_scale`, 0.0 for a `negative_prompt` pipeline in
  `_GUIDES_FROM_ZERO`, else 1.0. The profile guesses until a load;
  `/api/status` merges the live answer, as it does `negative_prompt_supported`,
  `true_cfg_supported`, `reference_supported` and `lora_supported`.
- `GenerateRequest.applied()` is the one normalisation - negative prompt, mu,
  references, `lora_scale` - and `backend.applied()` answers it from the
  parameter list or the profile. Every save path, the resume prune and the
  duplicate count go through it, never through the raw request.
- `true_cfg` on a row means the scale the negative prompt ran at, or None, on
  every checkpoint.
- `model_defaults._PROFILES` is first-match on the id, so a distilled variant
  sits ahead of its family's token (klein and schnell before flux, z-image-turbo
  before z-image, ernie-image-turbo before ernie-image); a catalogue entry
  that reaches the fallback is a test failure. Steps, CFG and mu are the
  checkpoint's; `mu` is Turbo's alone. `krea-dev` sits ahead of `flux` for
  its guide alone and is the flux profile in every number and control hint.
- Every profile's `guide` (`backend/guides.py`) is sourced from the checkpoint's authors - the
  model card, README, prompting guide or report - never invented; the
  design notes keep the URLs. `pixels` is the range they put in writing and
  `pixels_word` what that writing was (documented, recommended, trained);
  the range must admit the 1 MP presets. Its prose rides in `hints` as
  `prompt` and `size`, its data in `ModelStatus.guide`.

Reproduction:

- Seeds are drawn from `2**53 - 1` and wrapped at `2**63 - 1`; both live in
  `seeds.py`. One generator per image, seeded `seed + i`; runs take base
  `seed + i * num_images`.
- `grid.expand()` is the one cell loop for the CLI and `/api/grid`;
  `jobs.submit()` keeps its inline copy of the seed loop deliberately, asserted
  equal to the CLI's in `tests/test_cli.py`.
- The reproduction set - prompt pair, size, steps, CFG, true CFG, mu, seed,
  checkpoint, quantization, references, LoRA and weight, variables, hash - is
  on the row, in the PNG's `local-diffusion` chunk and in `CellKey`. `run_id` names a
  cell by the shortest key carrying its non-None optional elements, so every
  existing run id survives a widening.
- Resume keys on settings (`storage.completed_cells`, six-decimal rounding,
  scoped by checkpoint and quantizer), requires an explicit seed unless the
  seed axis is a list, and answers 409 when nothing is left.
- A new column needs a `_migrate` entry, not a `"x" in row.keys()` read guard:
  `_row_to_record` reads every column unguarded. `save_image` and
  `reindex.index_file` share `_insert_row` over `_COLUMNS`.
- Reference images are gallery ids, never paths. A LoRA is a bare name that
  only `loras.py` turns into a path, `.safetensors` only, and it is
  hot-swapped in `generate()`, never baked into `load()`. Whether it fits is
  `loras.mismatch` - the adapter's module names and widths against the
  denoiser's, PEFT's own lookup run early - from the checkpoint's configs on
  the meta device before a load (`loading.denoiser`) and the live pipeline
  after; a misfit is listed with its `detail`, hidden by the page and
  refused at submit and by the CLI, unreadable configs mean unchecked rather
  than refused, and a listing never converts a single-file checkpoint.
- Under Nunchaku (WSL) a seed does not reproduce bit-for-bit - Z-Image
  within noise, ERNIE visibly - so a duplicate count or a re-render's hash is
  the fp8 path's guarantee only; `docs/verified.md` has the numbers.

Jobs:

- Cancel and pause are cooperative and polled: cancel from
  `callback_on_step_end` (raise `Cancelled`), pause at the cell boundary
  every quarter second. Never `Event.wait()`.
- A job is a list of cells; seeds are the outermost loop, prompts next. A
  cell's failure skips the cell; three in a row or more than a third of the
  grid ends the job. Every terminal transition goes through `_finish`, and a
  budget stop returns through the cancel path rather than a fourth status.
- Each SSE stream has its own queue; `events()` opens with `done` for a job
  that has finished; the per-cell event and `/api/jobs` send
  `snapshot(images=False)`. Progress events carry `elapsed`; `waiting_on`
  says `load` or `queue`.
- `handle_exit` cancels the active jobs on the signal itself, the load runs
  on a daemon thread, and lifespan calls `try_unload()`.
- `save_image` writes `.png.part` and renames after the row commits, retrying
  the rename; the orphan sweep leaves `.part` files younger than an hour.
- The preview callback (`_step_end` -> `latent_preview`) never raises - an
  exception there is how `Cancelled` ends a job - and `jobs.on_preview` checks
  for subscribers per frame. A preview needs a matrix for the VAE and either
  `_unpack_latents` or already-unpacked 4-D latents.
- Generation is serialized through one worker thread; the GPU cannot usefully
  run two.

Frontend and CLI:

- `web/` has no build step and no test suite; check it in a real browser with
  `tools/fake_server.py` and `tools/browser.mjs` (see Tests). The page script
  is classic scripts sharing one global scope, loaded in the order
  `index.html` lists them; their top-level bindings stay top-level, and only
  `app.js`, the last, runs anything at load.
- The size line compares the request against `guide.pixels` a tenth either
  way and quotes `pixels_word`; the Examples menu is built on open from the
  last status, and a pick tells the preview and the panel the way `choose`
  does.
- Profile defaults apply per control (`owned`); an empty Steps or CFG box is
  sent absent and 0 is a valid CFG. The panel persists to `localStorage`
  with every access in try/catch; `resetPanel` is the one thing that forgets
  it, since clearing the gallery never reaches the browser.
- The sheet's layout contract - axes, prompt labels, pins, variables - lives in
  the client, and the export is client-side, not a route. The sheet asks for
  the whole run in one page.
- `generate.py` duplicates `jobs._generate`'s loop deliberately. It applies
  `--model/--quant/--offload` to `os.environ` before importing
  `backend.config`, as `launch.configure()` does for the server; both take the
  flags from `launch.add_model_arguments`.
- The gallery is the running checkpoint's: `GET /api/gallery`, `/api/runs`,
  `GET`/`DELETE /api/gallery/unstarred` and `POST /api/rerender` scope to
  `settings.model_id` (`main._model_scope`, an exact predicate, never the
  `model:` LIKE) unless `all_models=true`; the page's All models tick is the
  one switch, unpersisted like Favorites, and `GalleryPage.other_models` is
  what the same filter matches elsewhere. Delete unstarred takes the scope
  because its count must be the grid's.
- Search terms are `key:value` predicates plus FTS text; `date:` is a local
  day, `mu:none` is `IS NULL`, and `search.strict_scope` is the reading for
  `clear --where`, `keep_rates` and the gallery's Delete unstarred
  (`GET`/`DELETE /api/gallery/unstarred`): `clear --where <box>
  --keep-favorites` over HTTP, the search box strict or empty, gated on the
  count typed back and pinned to it as `expected`, which `delete_matching`
  checks under the write lock and answers 409 on a mismatch; rows and files
  only, no orphan sweep. `forgetImages` is the one place the page drops
  deleted ids. The gallery pages by id, never by offset,
  and pages itself as "Load more" scrolls near the viewport (`moreWatcher`,
  re-armed by `watchMore` after each page that lands, never after one that
  failed - the button is the retry).

## Adding a backend

Implement `GenerationBackend` (`load`/`unload`/`is_ready`/`generate`/`describe`)
and register it in `backend/backends/__init__.py:get_backend`. `generate()` is
handed a `should_cancel` predicate and must poll it at every step boundary,
raising `Cancelled` when it answers True. Nothing in the HTTP layer or the UI
needs to change.

## Tests

`uv run --group dev pytest`. They stub the pipeline (`tests/fakes.py`'s
`FakeBackend`, wired by the `fake` and `client` fixtures in `tests/conftest.py`
and driven through `tests/api_helpers.py`; the HTTP tests are `tests/test_api*.py`,
one file per area) and the scheduler (`tests/test_schedule.py`'s `StubPipeline`), so
nothing downloads weights or touches the GPU, and they run against a throwaway
database and outputs directory (`tests/conftest.py`). They cover the invariants
that are expensive to get wrong: the schedule the scheduler ends up with, seeds
surviving a JSON round trip, the search fallback keeping its filters, and the
HTTP guards.

There is no JavaScript test suite. `web/` is checked in a real browser
instead: `uv run python tools/fake_server.py` serves the app on 7861 on the
tests' fake backend (gradient images stamped with seed and steps, a throwaway
gallery under the system temp dir, `--step 0.3` to make pause and queueing
observable), and `node tools/browser.mjs tools/smoke.json` drives headless
Edge or Chrome over the DevTools protocol with Node's own WebSocket - no npm
install - through the lightbox zoom, the sheet, the export, the re-render
button, the panel's persistence and the duplicate/distinct counts, exiting 1
on a failed `expect`. `smoke-conditioning.json` runs against a klein-claiming fake on 7864 for the
reference chips, the adapter select and a weight sweep;
`smoke-flux.json` runs against a second fake server
claiming FLUX.1 (`--port 7862 --model black-forest-labs/FLUX.1-Krea-dev`)
for the True CFG box, `smoke-preview.json` against one slowed to
`--step 0.3` on 7863 for the live preview, and `smoke-paging.json` against a
`--fresh` one on 7867 for the gallery paging itself on scroll (it builds a
60-image grid, so it holds once per gallery); the fake takes the claimed
profile's threshold and true-CFG flag so it behaves as the real one would.
Static files are served from disk, so a `web/` edit needs no server restart;
a `backend/` edit does. The app's top-level bindings (`els`, `lbView`,
`sheetRecords`) are reachable from a step's `eval` because the page's scripts
are classic scripts sharing one global scope - which is also why they stay
top-level. `browser.mjs` gives
each run its own profile directory: a shared one stays locked by an Edge a
crashed run left behind, and every later launch then hands its command line
to that instance and reports "the browser did not come up".

## Environment

Python 3.13 (not 3.14 — torchao/diffusers wheels lag there).
torch 2.13.0+cu130, diffusers 0.40.0, torchao 0.18.0.
