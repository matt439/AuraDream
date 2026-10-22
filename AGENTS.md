# Local Diffusion

Local text-to-image generator on an RTX 5080. FastAPI backend + vanilla JS
frontend, no build step. The default checkpoint is **FLUX.1 Krea [dev]**
(`black-forest-labs/FLUX.1-Krea-dev`), which is click-through gated; **Krea 2
Turbo** (`krea/Krea-2-Turbo`) needs manual approval on HuggingFace and is what
the sizing notes in `docs/design-notes.md` are written around. The catalogue
in `backend/launch.py` lists the rest; any of them runs unchanged - set
`LOCAL_DIFFUSION_MODEL_ID`, or choose in the browser's Models panel.

This file is the short version. Before changing an area, read its part of
**`docs/rules.md`** (every rule in full, the module map, what each smoke
covers) and its section of **`docs/design-notes.md`** (why, and what went
wrong before). `docs/verified.md` lists what has been measured on the card.
**`TODO.md`** indexes open work, one row per item pointing to its owner; a
fresh thread starts at its NOW.
Edit `AGENTS.md` only: `CLAUDE.md` just imports it (`@AGENTS.md`).

## Run

```powershell
uv run python -m backend.main      # http://127.0.0.1:7860; restores saved browser choices
uv run python -m backend.main --defaults   # ignore saved choices; environment/defaults
uv run python -m backend.main --menu       # optional terminal picker
uv run --group dev pytest          # no GPU, no weights, about three minutes
```

```bash
bash wsl/setup.sh                  # WSL 2, once: Linux venv + CUDA check
source wsl/env.sh && bash wsl/serve.sh   # 4-bit Nunchaku, same URL
```

`bin\` (on the user PATH) holds `local-diffusion` and `local-diffusion-wsl`:
the two server starts above from any folder, arguments passed through. Each
is a `.ps1`, which PowerShell picks, beside a `.cmd` for cmd.exe; under a
`.cmd`, Ctrl-C asks "Terminate batch job (Y/N)?" however idle the server.

The model is *not* loaded at startup unless **Load automatically on startup**
was saved (or `LOCAL_DIFFUSION_PRELOAD=true`). The first `POST /api/generate` (or
`POST /api/model/load`) triggers a load that takes a few minutes.
**Models** changes checkpoint, quantization and offload while idle without
restarting. **Apply & save** selects without loading changed weights;
**Switch & load** loads too. Both save the choices beside the gallery database
in `local-diffusion.settings.json`; explicit environment/flags override them.

## Layout

Backend (`backend/`):

- `config.py` settings (`LOCAL_DIFFUSION_*`); `launch.py` start-up menu,
  `--model/--quant/--offload`, the checkpoint catalogue;
  `loader_choices.py` the quantization/offload wording the menu and picker share
- `model_selection.py` current server checkpoint/loading options/revision,
  separate from frozen startup settings; runtime readers use
  `current_model_id(settings)` / `current_settings(settings)`;
  `startup_preferences.py` atomically saves loader/startup choices;
  `model_info.py` supplies cold catalogue details from existing sourced guides;
  `hub_access.py` asks the Hub whether a switch's checkpoint can download;
  `model_benchmarks.py` supplies retained timing references and speed categories
- `schemas.py` re-exports the request/response models, each defined in
  `schemas_generate.py` (the generation request and `applied()`, grids, job
  answers, Load settings from PNG's), `schemas_prompt.py` (Expand, Check, critic, tag lookup),
  `schemas_gallery.py` (records, runs, bin), `schemas_rerun.py` (Re-render,
  Swap tags, Mass refine, More seeds) or `schemas_model.py` (model status,
  guide, switch); `model_defaults.py` per-checkpoint profiles; `guides.py` each checkpoint's sourced prompting guide and its
  `analysis` data
- `main.py` lifecycle, model state, switching and the model routes;
  `app_context.py` the `AppServices` getters routers read main through;
  `detached.py` daemon-thread calls; `api_prompt.py` tag helper and prompt
  routes, `prompt_check.py` Check's layer 1, `tag_lookup.py` the Danbooru
  lookup (`tag_vocabulary`); `submission.py` submit checks and the gallery
  selection; `api_generate.py` generate/grid/Re-render, `api_swap.py`,
  `api_refine_batch.py`, `api_more_seeds.py` the gallery actions' routes;
  `api_png.py` Load settings from PNG (a PNG's text chunks as the panel's
  settings, after `applied()`); `api_masks.py` an inpaint's mask stored
  and served by its hash, and an outpaint's presets
  (`GET /api/images/{id}/outpaint-presets`); `outpaint.py` an outpaint's
  pure parts (the padded source and its fills, the built mask, the presets);
  `api_gallery.py`, `api_jobs.py` the gallery and job routes; `api_bundles.py` browser bundle
  routes and maintenance admission; `bundles.py` verified export/merge,
  `bundle_archive.py` the archive format and its verifying read,
  `bundle_images.py` a tracked PNG's identity, row checks and metadata
  rewrite, `bundle_common.py` their errors and database reads,
  `bundle_crypto.py` its optional password layer;
  `gallery_lock.py` reservations across storage threads and processes
- `jobs.py` single-worker queue and SSE; `storage.py` SQLite + PNGs;
  `metadata.py` the PNG chunk's key and its legacy `krea` reader;
  `png_settings.py` a PNG's record and A1111 'parameters' block read back
  (`--from-image` and `api_png.py` share it);
  `gallery_reports.py` estimates, keep rates, runs; `gallery_delete.py`
  scoped delete, recycle bin, clear;
  `search.py` search grammar; `grid.py` axes, variables, variants, cells,
  records as cells (`cell_from_record`); `tag_swap.py` Swap tags' rewrite;
  `refine_batch.py` Mass refine's per-image plan; `more_seeds.py` More
  seeds' batches and their next seeds;
  `seeds.py`; `sizes.py` presets and size guards; `loras.py` adapter names to
  paths; `refine.py` the image-to-image pass and the inpaint (`sibling(kind)`,
  `composite`); `latent_preview.py`
- Prompt analyser: `prompt_encoding.py` (tokenizers), `encoder_table.py`
  (each checkpoint's encoders and copied pipeline templates),
  `prompt_weighting.py` (literal weighting syntax), `prompt_text.py`,
  `prompt_order.py`, `prompt_rules.py` (rule checks), `prompt_sources.py`
  (what the critic may cite), `critic.py` + `critic_runtime.py`,
  `critic_server.py`, `critic_context.py`, `critic_reorder.py`,
  `critic_validation.py`; `tags.py` Danbooru lookup; `translation.py`
  ERNIE's Marian translation
- `backends/`: `base.py` (the ABC), `diffusers_backend.py`, `loading.py`
  (everything decided before the weights move), `schedule.py` (mu, sigmas,
  refine start), per-checkpoint converters and pipelines (`single_file`,
  `anima_*`, `netayume_loading`, `newbie_*`)
- CLIs: `generate.py`, `clear.py`, `stats.py`, `reindex.py`

Frontend (`web/`, served from `/static`): `index.html` loads classic scripts
in order, sharing one global scope; only `app.js`, the last, runs anything.
`style.css` (tokens, base) then `styles/*.css` in link order - keep that
cascade. `docs/rules.md`'s Layout maps every file; `docs/source-audit.md`
records the split audits.

Also: `tools/` (fake server, headless-browser smokes), `tests/`, `wsl/`,
`docs/`.

## Rules that break things silently

Model and loader:

- Quantize only through `PipelineQuantizationConfig(quant_mapping=...)`: one
  `TorchAoConfig` per component, from the library that owns it, built from an
  `AOBaseConfig` *instance*. Never `quant_backend="torchao"`.
- `torch.no_grad()`, never `torch.inference_mode()`, around the pipeline call.
- `config.py` is imported before diffusers everywhere, the CLI included.
- Loading happens only in `main.ensure_loaded()` under its asyncio lock, after
  `jobs.wait_for_gate`. `generate()` raises on a missing pipeline; unload is
  409 while a job or load is live. Load failures go through
  `main._load_failure`.
- Switching reserves the idle server through cleanup and `select()`; its load
  is then an ordinary `ensure_loaded` (Check, tags and LoRAs answer, Generate
  queues, a second switch is refused).
  Model-dependent requests are counted before validation; stale browser
  revisions are refused before submission. Never mutate frozen startup settings.
- A switch to another checkpoint asks `hub_access.check` before it saves,
  reserves or unloads: only the Hub's no refuses; offline or a timeout goes
  on. Tests and the fake server never reach the Hub.
- `load()` refuses a pipeline whose `__call__` lacks `generator` or
  `callback_on_step_end`.
- Per-checkpoint loader rules (pinned revisions, single-file conversions,
  scheduler overrides, exact-id profiles, what is unavailable) are in
  `docs/rules.md` under "Model and loader". Keep official/public ids in
  settings and saved metadata; never silently drop weights or scales.
- `LOCAL_DIFFUSION_QUANT=nunchaku` is Linux (WSL) only; the preflight refuses
  other pairings before any download.
- App-owned names say Local Diffusion (`LOCAL_DIFFUSION_*`,
  `local-diffusion.*`); Krea only in checkpoint/vendor names.

Schedule and profiles:

- A requested `mu` is the whole shift (`schedule.forced_mu`,
  `shifted_sigmas`, `requested_shift`), and only on a `FlowMatch*`
  scheduler. Tests assert on `scheduler.sigmas`.
- `GenerateRequest.applied()` is the one normalisation; every save path,
  resume and duplicate count goes through it, never the raw request.
- `model_defaults._PROFILES` is first-match: distilled variants before their
  family; `_EXACT_PROFILES` first; never match Anima with a bare `anima`.
  A catalogue entry reaching the fallback is a test failure.
- Every guide is sourced from the checkpoint's authors, never invented; cites
  are verbatim (`test_every_cite_is_the_guides_own_words`).

Reproduction and storage:

- Seeds are drawn from `2**53 - 1` and wrapped at `2**63 - 1` (`seeds.py`);
  one generator per image, seeded `seed + i`.
- `grid.expand()` is the one cell loop; `jobs.submit()` and `generate.py`
  keep deliberate copies, held equal by `tests/test_cli.py`. Gallery records
  run again (Re-render, Swap tags, Mass refine, More seeds) become cells only
  through `grid.cell_from_record`, their images through
  `submission.resolve_selection`.
- The reproduction set lives on the row and in the PNG's `local-diffusion`
  chunk; `CellKey`/`run_id` widen without changing existing ids.
- A new column needs a `_migrate` entry; `_row_to_record` reads unguarded.
- `save_image` writes `.png.part` and renames after the row commits.
- Page deletes bin (`binned_at`); live reads exclude it,
  `_tracked_files`/`existing_*`/bundles/`clear`/`completed_cells`/`lineage_images`/
  `newest_image_id` read every row; only Empty bin, `DELETE /api/images/{id}` and `clear`
  remove PNGs.
- Gallery bundles are verified before source cleanup; imports are staged
  before merging. Keep gallery ids monotonic when replacing a cleared DB.
  Whole storage operations take the gallery reservation before the write
  lock; maintenance stays reserved until its worker has finished.
- References and refine sources are gallery ids, never paths. A LoRA is a
  bare name only `loras.py` turns into a path, hot-swapped in `generate()`.
- Load settings from PNG sends a PNG's text chunks only, never its image,
  which is never stored or made a reference. A file's reference ids carry
  only where this gallery's row matches the record's provenance hash; an
  A1111 block gives the prompt, negative and size, never a mapped sampler.
- An inpaint is a refine pass with `mask`: a stored mask's 64-hex hash (size
  and `L` levels, `storage.mask_hash`), never a path - `storage.mask_path`
  resolves it - and never a gallery image, reference or source. It runs at
  the source's size and is always composited over the source
  (`refine.composite`; chunk `mask_blend: "overlay"`); the mask's PNG rides
  in `local-diffusion-mask`. `grid.mask_key` is `CellKey`'s fifth extension
  (existing run ids unchanged); Mass refine drops the mask. An outpaint is
  the exception to the size: it runs at the source plus its padding
  (`refine.outpaint_source`).
  `gallery_delete.sweep_masks` spares a mask any row (live or binned) or
  live job names, or younger than the settle window. Only profiles with
  `inpaint_supported` run one; `model_defaults.INPAINT_DEFERRED` families
  are refused though diffusers has the class.
- An outpaint is an inpaint over a padded source whose mask the server
  builds (`outpaint.mask`): stamped - built and stored - first in
  `submit_cells`, before `Job(...)`, and a stamped request (mask and
  outpaint together, which the validator refuses) is never revalidated. It
  is never on a row without `refine` and `mask`; it keys by its spec
  (`outpaint:` + `OutpaintSpec.key()`), and a row whose spec does not build
  its own mask is refused by every rerun. Each `FILLS` entry and
  `outpaint.mask` are frozen: a changed fill gets a new name.

Jobs:

- One worker thread. Cancel and pause are cooperative and polled (cancel by
  raising `Cancelled` from `callback_on_step_end`); never `Event.wait()`.
- Every terminal transition goes through `_finish`.
- The preview callback never raises.
- GPU work outside jobs (Expand, the critic) goes through
  `jobs.run_when_idle`; nothing holding the gate awaits the load lock.

Prompt analyser:

- Check works cold: CPU tokenizers and rules on a daemon thread, never
  `ensure_loaded`, never the gate.
- Every finding carries `source`/`cite`/`url`; rule config lives in the
  guide's `analysis`, never a profile key. A check no author sentence
  supports is `heuristic`. Never invent a documented order.
- `prompt_encoding` measures the exact string the pipeline builds; the
  strings it copies live in `encoder_table`, held to the installed diffusers
  source by `tests/test_prompt_encoding.py`.
- The critic is pinned (llama.cpp build, Qwen3-4B GGUF, sizes and sha256),
  runs one loopback `llama-server` per call with its key in the environment,
  stopped as a whole process tree; its answer passes `critic.validate` before
  it is shown, and the docs never call it trusted.

Frontend:

- No build step, no JS test suite: check `web/` in a real browser with the
  fake server and `tools/browser.mjs`. Top-level bindings stay top-level.
- CSS uses the `:root` tokens; wide-tier rules go in an
  `@media (min-width: 1800px)` block in the component's section; nothing
  changes under 1800 px, where the smokes run, but the panel's own gap and
  field height under 900 px tall (`docs/rules.md`).
- A button held for a reason is `aria-disabled` with the reason as its title,
  never `disabled` (documented exceptions in `docs/rules.md`); a press says
  why.
- Whole-box writes to the prompt or negative go through `writeBox` /
  `rewriteBox` / `replaceBoxText`, never `.value`, so Ctrl+Z works.
- `setReferences` is the one writer of references; `forgetImages` is the one
  place the page drops deleted or binned ids; `setQueue` the queue rows'.
- The per-image delete never confirms; Empty bin and Delete permanently do.
- `localStorage`/`sessionStorage` access is always in try/catch.
- The page never scrolls by itself and never moves a control under the
  pointer; never `overflow-y: auto` on the panel.
- The server decides (order slots, start sigma, reorders, grid expansion);
  the page adds presentation only.

CLI: `generate`, `clear`, `stats`, `reindex` reconfigure stdout to UTF-8 at
the top of `main()`; `generate.py` applies `--model/--quant/--offload` to
`os.environ` before importing `backend.config`.

## Adding a backend

Implement `GenerationBackend` (`load`/`unload`/`is_ready`/`generate`/`describe`)
and register it in `backend/backends/__init__.py:get_backend`; no setting
picks one, so select it where the server and the CLI construct theirs
(`get_backend("diffusers")` in `main.py` and `generate.py`). `generate()` is
handed a `should_cancel` predicate and must poll it at every step boundary,
raising `Cancelled` when it answers True. Optionally, `start_sigma(request)`
answers the Refine preview's start sigma for a pass as `applied()` runs it,
and `prompt_expander_status()` what an expansion would wait on and download
(file checks only, for the status poll); both default to None. Nothing else
in the HTTP layer or the UI needs to change.

## Tests

`uv run --group dev pytest` - no GPU, no weights, a throwaway database, and
`LOCAL_DIFFUSION_*` cleared by `tests/conftest.py`. The fake backend is
`tests/fakes.py`; HTTP tests are `tests/test_api*.py`, one file per area.

The UI is checked with smokes: start a fake server, then run the smoke.
What each smoke covers is listed in `docs/rules.md` under Tests.

```powershell
uv run python tools/fake_server.py --port <port> [--model <id>] [--step 0.3]
node tools/browser.mjs tools/<smoke>.json
```

| Port | `--model` (default Krea 2 Turbo) | Smokes |
|---|---|---|
| 7861 | default | `smoke.json` (needs a fake of its own) |
| 7862 | FLUX.1-Krea-dev | `smoke-flux` |
| 7863 | default, `--step 0.3` | `smoke-preview` |
| 7864 | black-forest-labs/FLUX.2-klein-4B | `smoke-conditioning` |
| 7865 | baidu/ERNIE-Image-Turbo | `smoke-ernie` |
| 7866 | animagine-xl-4.0 | `smoke-animagine` |
| 7867 | OnomaAIResearch/Illustrious-XL-v2.0 | `smoke-illustrious` |
| 7867 | `--fresh` (run separately) | `smoke-paging` |
| 7868 | SeeSee21/Z-Anime | `smoke-z-anime` |
| 7869 | circlestone-labs/Anima | `smoke-anima` |
| 7870 | KBlueLeaf/Kohaku-XL-Zeta | `smoke-kohaku` |
| 7871 | duongve/NetaYume-Lumina-Image-2.0 | `smoke-netayume` |
| 7872 | NewBie-AI/NewBie-image-Exp0.1 | `smoke-newbie` |
| 7873 | LyliaEngine/waiIllustriousSDXL_v170 | `smoke-wai` |
| 7874 | FLUX.1-Krea-dev | `smoke-analysis`, `-rewrite`, `-comparison`, `-comparison-batch` |
| 7875 | FLUX.1-Krea-dev | `smoke-refine`, `smoke-refine-unavailable` |
| 7876 | FLUX.1-Krea-dev | `smoke-wide` |
| 7877 | animagine-xl-4.0 | `smoke-refine-sdxl` |
| 7878 | animagine-xl-4.0 | `smoke-tags` |
| 7879 | FLUX.1-Krea-dev | `smoke-grid` |
| 7880 | FLUX.1-Krea-dev | `smoke-sheet` |
| 7881 | FLUX.1-Krea-dev | `smoke-delete` |
| 7882 | default, `--step 0.3` | `smoke-jobs` |
| 7883 | FLUX.1-Krea-dev, `--step 0.3` | `smoke-results` |
| 7884 | FLUX.1-Krea-dev | `smoke-lightbox` |
| 7885 | default | `smoke-gallery` |
| 7886 | black-forest-labs/FLUX.2-klein-4B | `smoke-sheet-klein` |
| 7887 | animagine-xl-4.0 | `smoke-analysis-tags` |
| 7888 | FLUX.1-Krea-dev | `smoke-stage` |
| 7889 | FLUX.1-Krea-dev, `--step 0.3` | `smoke-narrow` |
| 7890 | Minthy/RouWei-0.8 | `smoke-rouwei` |
| 7891 | Laxhar/noobai-XL-Vpred-1.0 | `smoke-noobai` |
| 7895 | circlestone-labs/Anima@aesthetic-v1.1 | `smoke-anima-aesthetic` |
| 7896 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-model-switch`, `smoke-model-startup`, `smoke-model-hover` |
| 7896 | FLUX.1-Krea-dev, its own fresh scratch `--data` (run separately) | `smoke-model-gallery` |
| 7897 | default, fresh scratch `--data` | `smoke-bundles` |
| 7898 | default, fresh scratch `--data` | `smoke-select` |
| 7899 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-compare-cfg` |
| 7900 | animagine-xl-4.0, fresh scratch `--data` | `smoke-swap` |
| 7901 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-refine-batch` |
| 7902 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-more-seeds` |
| 7903 | default, fresh scratch `--data` | `smoke-bin` |
| 7904 | default, fresh scratch `--data` | `smoke-png-settings` |
| 7905 | animagine-xl-4.0, fresh scratch `--data` | `smoke-inpaint` |
| 7906 | default | `smoke-inpaint-unavailable` |
| 7907 | animagine-xl-4.0, fresh scratch `--data` | `smoke-outpaint` |
| 7908 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-context-menu` |
| 7909 | animagine-xl-4.0 | `smoke-tag-finder` |

FLUX.1-Krea-dev is `black-forest-labs/FLUX.1-Krea-dev`; animagine is
`cagliostrolab/animagine-xl-4.0`.

Gotchas:

- Fake servers share one throwaway gallery unless `--data` gives each its
  own; `smoke.json` counts images planted under another checkpoint.
- Smokes are written at 1400x900; `LOCAL_DIFFUSION_VIEWPORT`,
  `LOCAL_DIFFUSION_DPR` and `LOCAL_DIFFUSION_PORT_OFFSET` move them.
- `browser.mjs` exits 1 on a failed `expect`, 2 on a script it refuses
  (an async helper joined with `&&`/`||` is never awaited: put the whole
  condition inside the predicate).
- A `web/` edit needs no server restart; a `backend/` edit does.
- Each run gets its own browser profile directory; a shared one stays locked
  by a crashed Edge.
- Fake-server knobs: `POST /api/fake/load`, `/api/fake/step`,
  `/api/fake/expand`, `/api/fake/lora`, `/api/fake/bin-age`; a prompt containing `[fail]` fails its cell;
  `--critic-delay`, `--no-critic`.

GPU behaviour, pixels, timing and memory are the card's to say, recorded in
`docs/verified.md`; CPU and browser checks alone do not establish them.

## Environment

Python 3.13 (not 3.14 — torchao/diffusers wheels lag there).
torch 2.13.0+cu130, diffusers 0.40.0, torchao 0.18.0.
