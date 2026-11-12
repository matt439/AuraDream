# AuraDream

Local text-to-image generator on an RTX 5080. FastAPI backend + vanilla JS
frontend, no build step. The default checkpoint is **FLUX.1 Krea [dev]**
(`black-forest-labs/FLUX.1-Krea-dev`), which is click-through gated; **Krea 2
Turbo** (`krea/Krea-2-Turbo`) needs manual approval on HuggingFace and is what
the sizing notes in `docs/design-notes.md` are written around. The catalogue
in `backend/launch.py` lists the rest. With no saved or explicit checkpoint
and no Hugging Face sign-in, a fresh visit suggests the open **Animagine
XL 4.0**, matching the installer.
Any text-to-image one runs unchanged -
set `AURADREAM_MODEL_ID`, or choose in the browser's Models panel. The last,
FLUX.1 Kontext [dev], only edits, and its loads are held (the edit gate below).

This file is the short version. Before changing an area, read its part of
**`docs/rules.md`** (every rule in full, the module map, what each smoke
covers; its headings name the areas, `grep -nE "^##+ " docs/rules.md`) and its section of **`docs/design-notes.md`** (why, and what went
wrong before). `docs/verified.md` lists what has been measured on the card.
**`TODO.md`** indexes open work, one row per item pointing to its owner; a
fresh thread starts at its NOW.
Edit `AGENTS.md` only: `CLAUDE.md` just imports it (`@AGENTS.md`).

## Git workflow

Work on a local branch and commit completed changes there. Keep related work
together on the same branch across commits. Create a pull request only when
Matt explicitly requests one, collecting the accumulated commits into that
PR. Do not create a PR for each commit or split related work into stacked
PRs unless requested. Keep commits local until Matt requests a push or a PR.

A pull request targets main from the start: a retargeted one starts no CI
run and has no `gate` until a new commit. main accepts only pull requests
whose `gate` check passed on a branch up to date with main. When asked to
merge, run `gh pr update-branch` if the branch is behind, then
`gh pr merge --auto --merge`, which merges once `gate` passes; never merge
around a red or pending gate. Watch it until it has merged: if main moves
first, the pull request waits behind it until `gh pr update-branch` again. On a red smoke, read
its rerun verdict: "passed alone" is a race to fix where it comes from (a
press before the page is idle, a state that passes between polls), not a
reason to rerun until green. Rerun only the failed jobs
(`gh run rerun <id> --failed`), and only for an infrastructure failure.

## Run

```powershell
uv run python -m backend.main      # http://127.0.0.1:7860; restores saved browser choices
uv run python -m backend.main --defaults   # ignore saved choices; environment/defaults
uv run python -m backend.main --menu       # optional terminal picker
uv run python -m backend.main --open       # also open the browser once listening
uv run --group dev pytest          # no GPU, no weights; add -n auto for parallel
uv run --group dev ruff check .    # correctness lint
uv run --group dev python tools/typecheck.py  # pinned Pyright standard; Node required
```

```bash
bash wsl/setup.sh                  # WSL 2, once: Linux venv + CUDA check
source wsl/env.sh && bash wsl/serve.sh   # 4-bit Nunchaku, same URL
bash linux/setup.sh                # native Linux, once: driver/card checks, uv, uv sync, doctor
bash linux/install-service.sh      # systemd user service on the LAN (--uninstall, --dry-run)
bash linux/update.sh --restart     # stop when idle, pull + sync, start again
```

`bin\` (on the user PATH) holds `auradream` and `auradream-wsl`:
the two server starts above from any folder, arguments passed through. Each
is a `.ps1`, which PowerShell picks, beside a `.cmd` for cmd.exe; under a
`.cmd`, Ctrl-C asks "Terminate batch job (Y/N)?" however idle the server.
`auradream-server` updates and controls a native Linux server over ssh
(`linux/control.sh`, `linux/update.sh`; its `AURADREAM_SERVER*` variables
are the script's, never the app's).

The model is *not* loaded at startup unless **Load automatically on startup**
was saved (or `AURADREAM_PRELOAD=true`). The first `POST /api/generate` (or
`POST /api/model/load`) triggers a load that takes a few minutes.
**Models** changes checkpoint, quantization and offload while idle without
restarting. **Apply & save** selects without loading changed weights;
**Switch & load** loads too. Both save the choices beside the gallery database
in `auradream.settings.json`; explicit environment/flags override them.

## Layout

Every file is mapped in [`docs/rules.md`'s Layout](docs/rules.md#layout); read the area's part before changing it. One line per area:

- Backend core - settings, launch and the catalogue, model selection and facts, backups, setups, snippets, saved searches, favourites: [settings, libraries and model facts](docs/rules.md#backend-settings-libraries-and-model-facts)
- Schemas, profiles, guides and the prompt analyser (`prompt_*`, `critic*`, `tags`, `translation`): [schemas, guides and the prompt analyser](docs/rules.md#backend-schemas-guides-and-the-prompt-analyser)
- Generation passes, `backends/`, the grid and the gallery actions' plans (refine, inpaint/outpaint, instruction edit, More seeds): [passes, backends and the grid](docs/rules.md#backend-passes-backends-and-the-grid)
- Jobs, storage, PNG metadata, gallery reports/delete and bundles: [jobs, storage and bundles](docs/rules.md#backend-jobs-storage-and-bundles)
- The app and its routes (`main`, `app_context`, keep-awake, response headers, passcode, `api_*`): [the app and its routes](docs/rules.md#backend-the-app-and-its-routes)
- CLIs, the doctor, the server probe, build info, the log file and refusal codes: [CLIs, the doctor and support modules](docs/rules.md#backend-clis-the-doctor-and-support-modules)
- Windows installer/updater and `linux/` scripts: [install and update scripts](docs/rules.md#install-and-update-scripts)
- The page (`web/`, classic scripts in `index.html`'s order, one global scope, only `app.js` runs; `style.css` then `styles/*.css` in link order), its phone/touch hosts (`touch.js`, `phone*.js`), `sw.js`, `offline.html` and `passcode.html`: [the page](docs/rules.md#the-page-web), with the phone's rules under [Phones and tablets](docs/rules.md#phones-and-tablets)
- `tools/`, `tests/`, `bin/`, `wsl/` and `docs/`: [tools, tests, bin, WSL and docs](docs/rules.md#tools-tests-bin-wsl-and-docs)

## Rules that break things silently

Model and loader:

- Quantize only through `PipelineQuantizationConfig(quant_mapping=...)`: one
  `TorchAoConfig` per component, from the library that owns it, built from an
  `AOBaseConfig` *instance*. Never `quant_backend="torchao"`.
- `torch.no_grad()`, never `torch.inference_mode()`, around the pipeline call.
- A quantizing `from_pretrained` runs inside
  `loading.checkpoint_files_released`, which copies unquantized tensors out
  of their shards; without it the whole bf16 checkpoint stays mapped.
- `config.py` is imported before diffusers everywhere, the CLI included.
- Loading happens only in `main.ensure_loaded()` under its asyncio lock, after
  `jobs.wait_for_gate`. `generate()` raises on a missing pipeline; unload is
  409 while a job or load is live. Load failures go through
  `main._load_failure`; an edit-only checkpoint's refusal there
  (`require_model_available`, given the instance's `qualified` and
  `loader` at every load entry) is an error in its own words.
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
- `AURADREAM_QUANT=nunchaku` is Linux only (native or WSL); the preflight refuses
  other pairings before any download.
- App-owned names say AuraDream (`AURADREAM_*`,
  `auradream.*`); Krea only in checkpoint/vendor names. Pre-rename
  `local-diffusion` names (and the older `krea` chunk) are legacy: read, or
  an existing file used where it is (`legacy_names.py`), never given to
  anything new; one feeding a hash, key derivation or outside store stays
  frozen with a comment. Only `AURADREAM_*` variables are read.

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
- `grid.expand()` is the one cell loop, the CLI's too; `jobs.submit()` keeps
  a deliberate copy of its seed loop, held equal by `tests/test_jobs.py`
  (and the CLI's seeds by `tests/test_cli.py`). Gallery records
  run again (Re-render, Swap tags, Mass refine, More seeds) become cells only
  through `grid.cell_from_record`, their images through
  `submission.resolve_selection`.
- The reproduction set lives on the row and in the PNG's `auradream`
  chunk; `CellKey`/`run_id` widen without changing existing ids. The peak
  GPU memory pair beside it is a result, never in `CellKey`, `run_id` or
  `applied()`.
- A new column needs a `_migrate` entry (`storage._ADDED_COLUMNS`, which a
  bundle merge's types also come from); `_row_to_record` reads unguarded.
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
  in `auradream-mask`. `grid.mask_key` is `CellKey`'s fifth extension
  (existing run ids unchanged); Mass refine drops the mask. An outpaint is
  the exception to the size: it runs at the source plus its padding
  (`refine.outpaint_source`). Only the masked area (`mask_crop`) is the
  app's crop by the frozen rule `mask_crop.region`, never the class's
  `padding_mask_crop`: the request keeps the source's size, its box and
  working size are results (never keyed), and its key rides `mask_key` as
  `+bbox16:32@1`.
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
- Instruction edits retain a distinct `edit` spec and full original-RGB source
  proof through rows, PNGs, keys, reruns and recovery; unresolved sources keep
  their intent and refuse replay. Edit bundles use v2; edit-free bundles stay
  v1.
- The one edit gate is the backend instance's `instruction_edit_qualified()`
  (literal `True` only), never a setting; `DiffusersBackend` returns the
  literal True (stage 4's release), every other gate the literal False.
  `QUALIFIED_ENVELOPE` then decides loads and edits: the exact Kontext id,
  fp8, model offload, CUDA, Windows, Eager, diffusers 0.42.1 (re-measured
  2026-11-10, pixel-identical to the first-measured 0.40.0), and 1:1
  sources at 512x512. The editor load holds open only for the exact id inside
  the envelope (the pinned configs-then-weights loader); Qwen-Image-Edit is
  always refused. Kontext is an edit-only catalogue entry, listed last.
  Edits queue through `POST /api/instruction-edit` and More seeds (new seeds
  of the original source inside the envelope, its recorded proof checked
  first), and each cell's source is re-proved as it begins. Reuse opens the
  Edit by instruction dialog. Re-render refuses an edit with words naming
  it; Generate and Finish stay held. The plan and the
  carry/bundle checklists own the remaining integration and the card step.

Jobs:

- One worker thread. Cancel and pause are cooperative and polled (cancel by
  raising `Cancelled` from `callback_on_step_end`); never `Event.wait()`.
- Every terminal transition goes through `_finish`.
- The preview callback never raises.
- A save the disk has no room for (`jobs.disk_full`: `ENOSPC`, SQLite's
  "database or disk is full") pauses the job with `pause_reason`, the render
  held in memory and saved at Resume; never a failed cell. Every other save
  error stays the cell's failure. The CLI's own loop holds the image the
  same way but polls (`generate._save_held`): it says the reason once and
  saves when there is room; Ctrl-C ends the wait as a cancel.
- GPU work outside jobs (Expand, the critic) goes through
  `jobs.run_when_idle`; nothing holding the gate awaits the load lock.
- A failure site logs one ERROR line and its traceback at DEBUG with
  `exc_info`: the log file (`log_file.py`) takes every `exc_info` record,
  the console only at DEBUG. The file is installed in `main.run`, never at
  import, and is never bundled, exported or cleared as gallery data.
- Keep-awake providers are read on the event loop only; work on a daemon
  thread holds through `keep_awake.busy`. A new timed poll joins
  `counts_as_activity`'s exclusions, or an idle tab holds the PC awake.

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
  field height under 900 px tall (`docs/rules.md`). Phone and tablet rules
  go in `styles/touch.css` under `(pointer: coarse)` (the phone's own
  hosts, filled by `phone.js`, in `styles/phone.css`, its gallery head in
  `styles/phone-gallery.css`); a one-column rule
  keyed on 900 px excludes the tablet's two columns with the same query.
- A button held for a reason is `aria-disabled` with the reason as its title,
  never `disabled` (documented exceptions in `docs/rules.md`); a press says
  why.
- What the page does about a refusal is decided by its `code`
  (`refusal_codes.py`; `imageGone`, `imageBinned`, `refusedIds`,
  `loraRefused` in `state.js`), never by testing `err.message`: the
  sentence is only shown. A refusal the page must act on gets a code first.
- Whole-box writes to the prompt or negative go through `writeBox` /
  `rewriteBox` / `replaceBoxText`, never `.value`, so Ctrl+Z works.
- `setReferences` is the one writer of references; `forgetImages` is the one
  place the page drops deleted or binned ids; `setQueue` the queue rows'.
- The per-image delete never confirms; Empty bin, Delete permanently and a
  saved bundle's Delete do.
- `localStorage`/`sessionStorage` access is always in try/catch.
- The page never scrolls by itself and never moves a control under the
  pointer; never `overflow-y: auto` on the panel.
- The server decides (order slots, start sigma, reorders, grid expansion);
  the page adds presentation only.
- `response_headers.ResponseHeadersMiddleware` stays the last
  `add_middleware` in `main` (outermost), and its source stays `'self'`:
  the smokes frame `/`. Under nosniff a script or stylesheet must be served
  as one. It answers an unhandled error itself (Starlette's own 500 would
  pass through no layer) and raises it on.

Passcode (`backend/passcode.py`; docs/design-notes.md, "A passcode beyond
loopback"):

- `AURADREAM_PASSCODE` is read from the environment only: never a flag, the
  saved settings, `/api/status`, a log line, an error, a process argument
  or test output. The gate keeps a digest and a derived key, never the
  passcode.
- It is asked when it is set and the bind or an allowed host is not
  loopback, and then of every request, 127.0.0.1's too (a proxy on the
  same machine arrives from there). Off, the layer passes everything
  through and every answer is what it was.
- `PasscodeMiddleware` stays inside the response headers, the cross-site
  write block and the Host check, and outside `hold_model_for_request`,
  activity and gallery admission. A new route is closed by default
  (`tests/test_api_passcode_routes.py` walks the route table); only
  `OPEN_PATHS` (the manifest and its icons, by exact path) answer signed
  out, and `/` is the sign-in page as a 200, never a redirect or a 401
  (the service worker and the title probes).
- `web/passcode.html` loads nothing under `/static/` and keeps
  `<title>AuraDream</title>`.
- The page never reloads by itself when a session ends: Sign in is the
  user's press. Every caller that is not a browser sends
  `Authorization: Bearer` from its own environment, through an opener that
  follows no redirect, and reads the gate's 401 and 429 as refusals in
  their own words, never as a server that is down or idle.
- The docs say what plain http leaves readable wherever they name the
  setting (README, `docs/usage.md`, `docs/setup.md`): it keeps out the
  household, not anyone who can capture the network's traffic.

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
`AURADREAM_*` cleared by `tests/conftest.py`. The fake backend is
`tests/fakes.py`; HTTP tests are `tests/test_api*.py`, one file per area.
Node is required for the classic-script syntax/shared-scope guard; it
checks HTML script order without executing the page's boot, and that every
name a script uses is declared (a new browser global joins `BROWSER_GLOBALS`
in `tools/check_classic_scripts.mjs`).
The pinned Pyright gate checks every Python source in `backend/` and
`tools/`; missing tools and empty scope fail. Its wrapper uses installed
Node and bundled checker files, without automatic updates or downloads.

CI (`.github/workflows/ci.yml`, detail in `docs/rules.md` under Tests): ruff
on Linux for every push and pull request. Documentation-only changes run
just lint, without app/tokenizer installs; executable files under `docs/`
and the prose tests read (this file, `docs/rules.md`, `docs/verified.md`)
still get tests. Other changes, manual and weekly runs get the pinned type
gate and `pytest -n auto --durations=25` (the 25 slowest tests) on Linux,
and on Windows too for main (pull requests into it, pushes, by hand and
weekly), split in five runners by `pytest --shard K/N`, each test under
`--timeout=600`; five `browser` jobs run a fifth of the smoke table each on
Linux Chrome (`run_smoke.py --shard K/5 --jobs 2 --rerun-failed-alone`,
balanced by `tools/run_smoke_costs.json`; fonts from `tools/smoke_fonts.py`;
a smoke passing alone after failing still fails), printing their timings
and rerun verdicts (`tools/smoke_timings.py --annotate`) and uploading
diagnostics on failure. Node is pinned to 24 (`actions/setup-node`).
main's ruleset requires a pull request, an up-to-date branch and two
checks, with no bypass: `gate`, and `ci-definition`
(`.github/workflows/ci-guard.yml`, run by `pull_request_target` as main
defines it), which fails a pull request that changes `.github/` or
`tools/ci_changes.py`/`ci_duplicates.py`: a CI change lands only through a
deliberate ruleset edit (`docs/rules.md`, Tests). A commit
runs once (`tools/ci_duplicates.py`): a branch push with an open pull
request (or one opened within 45 seconds) leaves it to that pull request's
run, and main's merge push skips when a passed pull request run tested the
same tree. Its runners are
slow and have no GPU driver: a test that races a clock (a time budget, a
cell to skip or pause mid-run) needs seconds of margin, not milliseconds.
`tools/fetch_tokenizers.py` fetches the real-tokenizer tests' files first,
with the `HF_TOKEN` repository secret for the gated repos its list names
(FLUX.1-Krea-dev, FLUX.1-Kontext-dev, Krea 2 Turbo).

The UI is checked with smokes: start a fake server, then run the smoke.
What each smoke covers is listed in `docs/rules.md` under Tests.

```powershell
uv run python tools/run_smoke.py <smoke> [...]   # starts the row's fake, runs, stops it
uv run python tools/run_smoke.py --all --jobs 4  # every row, longest first; --offset N moves ports; --shard K/N a part
uv run python tools/run_smoke.py --all --jobs 4 --rerun-failed-alone  # or --from-summary <summary.json>
uv run python tools/fake_server.py --port <port> [--model <id>] [--step 0.3]   # by hand
node tools/browser.mjs tools/<smoke>.json
```

Use `{"expect": "...", "fatal": true}` for a fixture prerequisite that must
stop later steps when it fails. Async setup expressions must explicitly return
true. Ordinary expectations keep aggregate diagnostics; fatal failure still
uses the driver's graceful close/cleanup. See the full Tests rules.

`run_smoke.py` reads the table below, so a new smoke needs only its row.

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
| 7892 | default, fresh scratch `--data` | `smoke-batch-mates` |
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
| 7910 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-sheet-zoom` |
| 7911 | Gazingstars123/Anima-2.9B, fresh scratch `--data` | `smoke-anima-preview` |
| 7911 | black-forest-labs/flux-3-image-klein, its own fresh scratch `--data` | `smoke-flux3` |
| 7912 | FLUX.1-Krea-dev, fresh scratch `--data`, `--step 0.3` | `smoke-then-refine` |
| 7913 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-shortcuts` |
| 7914 | FLUX.1-Krea-dev, fresh scratch `--data`, `--step 0.3` | `smoke-queue-controls` |
| 7915 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-snippets` |
| 7916 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-random-prompts` |
| 7917 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-size-crop` |
| 7918 | animagine-xl-4.0, fresh scratch `--data` | `smoke-button-states` |
| 7919 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-more-seeds-tabs` |
| 7920 | KBlueLeaf/Kohaku-XL-Zeta, fresh scratch `--data` | `smoke-mixed-prompts` |
| 7921 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-numeric-inputs` |
| 7922 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-queue-outcomes` |
| 7923 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-control-sizing` |
| 7924 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-shortcut-dispatch` |
| 7925 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-compact-setup` |
| 7925 | FLUX.1-Krea-dev, its own fresh scratch `--data` | `smoke-sampling-fold` |
| 7926 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-execution-reuse` |
| 7927 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-automatic-backup` |
| 7928 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-retention` |
| 7929 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-credential` |
| 7930 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-execution-selector` |
| 7931 | black-forest-labs/FLUX.1-Kontext-dev, fresh scratch `--data` | `smoke-instruction-edit` |
| 7932 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-generation-presets` |
| 7933 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-saved-searches` |
| 7934 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-snippet-portability` |
| 7935 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-library-write-recovery` |
| 7936 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-saved-name-unicode` |
| 7937 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-picked-favorites` |
| 7938 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-setup-list-recovery` |
| 7939 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-saved-search-write-recovery` |
| 7940 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-favorite-write-order` |
| 7941 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-bin-restore-order` |
| 7942 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-broad-bin-order` |
| 7943 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-bundle-clear-order` |
| 7944 | FLUX.1-Krea-dev, fresh scratch `--data`, `--step 0.3` | `smoke-queue-cancel-recovery` |
| 7945 | FLUX.1-Krea-dev, fresh scratch `--data`, `--step 0.3` | `smoke-watched-action-replies` |
| 7946 | FLUX.1-Krea-dev, fresh scratch `--data`, `--step 0.3` | `smoke-reuse-pass-opening` |
| 7947 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-runs-scope-recovery` |
| 7948 | default, fresh scratch `--data` | `smoke-grid-preview-close` |
| 7949 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-runs-favorite-counts` |
| 7950 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-copy-feedback-ownership` |
| 7951 | FLUX.1-Krea-dev, fresh scratch `--data`, `--step 0.3` | `smoke-split-extra-queue` |
| 7952 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-check-dismissal` |
| 7953 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-setup-portability` |
| 7954 | FLUX.1-Krea-dev, fresh scratch `--data`, `--step 0.3` | `smoke-queue-estimates` |
| 7955 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-gallery-top` |
| 7955 | FLUX.1-Krea-dev, its own fresh scratch `--data` | `smoke-gallery-maintenance` |
| 7956 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-phone` |
| 7956 | FLUX.1-Krea-dev, its own fresh scratch `--data` | `smoke-phone-create` |
| 7957 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-tablet` |
| 7957 | FLUX.1-Krea-dev, its own fresh scratch `--data` | `smoke-phone-landscape` |
| 7958 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-offline` |
| 7958 | FLUX.1-Krea-dev, its own fresh scratch `--data` | `smoke-passcode` |
| 7958 | FLUX.1-Krea-dev, its own fresh scratch `--data` | `smoke-idle-activity` |
| 7959 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-catalogue-stale` |
| 7959 | FLUX.1-Krea-dev, its own fresh scratch `--data` | `smoke-download-progress` |
| 7959 | FLUX.1-Krea-dev, its own fresh scratch `--data` | `smoke-model-storage` |
| 7960 | FLUX.1-Krea-dev, fresh scratch `--data` | `smoke-phone-dialogs` |
| 7960 | FLUX.1-Krea-dev, its own fresh scratch `--data` | `smoke-phone-gallery` |
| 7960 | FLUX.1-Krea-dev, its own fresh scratch `--data` | `smoke-phone-keyboard` |

FLUX.1-Krea-dev is `black-forest-labs/FLUX.1-Krea-dev`; animagine is
`cagliostrolab/animagine-xl-4.0`. The ports are full: `--jobs` puts each
worker 100 ports on, so 7961 would be `smoke.json`'s 7861 in the next
band; a new row shares a port (its own fake, one at a time per band).

Gotchas:

- Fake servers share one throwaway gallery unless `--data` gives each its
  own; `smoke.json` counts images planted under another checkpoint.
- Smokes are written at 1400x900; `AURADREAM_VIEWPORT`,
  `AURADREAM_DPR` and `AURADREAM_PORT_OFFSET` move them.
- A phone or tablet smoke's viewport steps add `"mobile": true` (overlay
  scrollbars, no 15 px classic one), and `{"inject": "phone_audit.js"}`
  gives it `__phoneAudit()` (controls under 44 px, off the screen, dialogs
  outside the window, sideways overflow); `docs/rules.md` Tests has both.
- A smoke that opens dialogs follows each navigate with
  `{"inject": "a11y_audit.js"}` and `{"eval": "__a11yWatch()"}` and ends
  (and precedes each later navigate) with `{"expect": "__a11yCheck()"}`:
  unnamed controls, duplicate ids, `aria-disabled` with no reason on the
  element itself, unnamed dialogs. `tests/test_a11y_audit.py` lists the
  smokes; a regression guard, never `UI#7`'s evidence.
- `browser.mjs` exits 1 on a failed `expect`, 2 on a script it refuses
  (an async helper joined with `&&`/`||` is never awaited: put the whole
  condition inside the predicate). It also exits 1 when the page threw: an
  uncaught exception or unhandled rejection in the page's code is printed
  as `! page error:` and fails the run though every `expect` passed. Fix
  the page (or the promise an `eval` dropped); no step allows one.
- A `web/` edit needs no server restart; a `backend/` edit does.
- Each run gets its own browser profile directory; a shared one stays locked
  by a crashed Edge. It is removed on every exit (failures and Ctrl-C too),
  and a later run sweeps any left over six hours.
- Fake-server knobs: `POST /api/fake/load`, `/api/fake/step`,
  `/api/fake/expand`, `/api/fake/model` (`overrides`, `stuck_critic`,
  `access`), `/api/fake/sign-in` (`token`, `unsourced`), `/api/fake/lora`,
  `/api/fake/bin-age`, `/api/fake/peak`, `/api/fake/card` (a claimed card),
  `/api/fake/refine` (`supported`: the claimed checkpoint's refine pass),
  `/api/fake/instruction-edit` (the edit gate, open by default only on a
  Kontext fake), `/api/fake/unreachable` (`on`: every other request held
  unanswered, a PC asleep), `/api/fake/disk-full` (`on`: every image save
  refused as a full disk), `/api/fake/bundle` (`hold`: each bundle
  export or import waits, so the gallery's admission refuses counted
  routes; the knobs pass it), `/api/fake/passcode` (`passcode`: a string
  turns the passcode's gate on, null off; `reset`: clears the limiter; a
  fake starts with none), `/api/fake/activity` (every request since the
  last `reset` and those keep-awake's rule counts as use, the fake's own
  keep-awake being off); a prompt containing `[fail]` fails its cell;
  `--critic-delay`, `--no-critic`.

GPU behaviour, pixels, timing and memory are the card's to say, recorded in
`docs/verified.md`; CPU and browser checks alone do not establish them.

## Environment

Python 3.13 (not 3.14 — torchao/diffusers wheels lag there).
torch 2.13.0+cu130, diffusers 0.42.1, torchao 0.18.0 (the execution
candidates stay held on their measured diffusers 0.40; Kontext's envelope
was re-measured on 0.42.1).
