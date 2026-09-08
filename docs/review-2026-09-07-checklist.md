# Review 2026-09-07 — implementation checklist

Companion to `review-2026-09-07.md`. One line per entry of the review, in the
review's own order and with its own `file:line` anchors (which refer to commit
`c7331fb`, the tree the review was made against - the lines have moved since).
Status as of 2026-09-08, after the commits listed at the end.

| Section | Entries | Done | Partial | Not done |
| --- | --- | --- | --- | --- |
| HIGH | 3 | 3 | 0 | 0 |
| MEDIUM | 37 | 37 | 0 | 0 |
| LOW | 41 | 41 | 0 | 0 |
| Disputed | 3 | 3 | 0 | 0 |
| Unverified leads | 9 | 9 | 0 | 0 |
| Possible improvements | 33 | 31 | 0 | 2 |

Legend: `[x]` done, `[~]` partial (what is missing is in the note), `[ ]` not
done. The commit named is the one that closed the entry; the note says where
the fix departs from what the review suggested. The test suite went from 135
to 242 passing tests over the series, and `web/` - which has no test suite -
was exercised in headless Edge against the fake backend for every UI change
(the harness is `tools/`; three smoke scripts now).

Verified on the GPU, 2026-09-08 (FLUX.1 Krea dev, fp8, model offload, from
the cache): a generation on FLUX - the `applied()` normalisation, since a
negative prompt with no true CFG produced the byte-identical image to none,
and the thumbnail write; the same request twice is the same pixel hash;
true CFG 1.5 gives a different, reproducible image at ~1.65x; a Ctrl-C
(SIGINT to uvicorn's handler) 12 s into a load exits in 3 s with the
lifespan run; mid-sweep with a stream attached, `done` reaches the stream
0.2 s after the signal and the process is gone 3 s later; Pause is
acknowledged at the cell boundary and the card drops from 14.95 GB during a
1 MP denoise to 2.1 GB held; `reindex --from` over the real gallery rebuilds
every row with the same hashes. The console Ctrl-C itself could not be
generated from the tool's shell, so the signal was raised in-process - the
same Python handler a terminal Ctrl-C reaches. Still not verified:
`KREA_DEVICE=cuda:1` (one card here).

## Confirmed findings

### HIGH

- [x] `web/app.js:531` appendResult loads every image at full resolution, uncapped — `c3b8da5`. Lazy + async decode, DOM capped at 48 with a note outside the strip; the lightbox still walks every record.
- [x] `web/app.js:918` renderSheet crosses two unbounded axes and rescans records per cell — `c3b8da5`. One-pass (row, column) map, 4096-cell ceiling with a pointer at the card's run button, pins hoisted, a one-value-per-image axis dropped when anything else varies.
- [x] `backend/grid.py:348` the drawn base seed is published nowhere — `a56431b`. `JobAccepted` and `Job.snapshot()` carry it, the UI writes it into a blank Seed box after Run grid (grid only; a plain Generate left blank still draws fresh), all three instructions say the *oldest* image. The optional 422 for a resume whose seed matches no cell was not added.

### MEDIUM

- [x] `backend/grid.py:227` prompt product materialised before any ceiling, on the loop — `df75e7e`. `math.prod` against `max_grid_images` before a dict is built; `expand()` no longer resolves the product twice. `_expand` still runs the (now bounded) expansion on the loop rather than in `to_thread`.
- [x] `backend/backends/diffusers_backend.py:411` static shift composes into `mu + ln(shift)` — `e51e6e4`. Pre-shift by `mu - ln(shift)` read off the instance; test parametrised over shift 1.0/3.0 asserting on `scheduler.sigmas`; warning for `shift_terminal`/karras/exponential/beta; CLAUDE.md bullet corrected.
- [x] `backend/grid.py:216` placeholders with an empty variables box escape the guard — `df75e7e`. Refused; a prompt without placeholders still expands to itself. No escape syntax for a literal `{word}` was added.
- [x] `backend/grid.py:298` a batch cell is done when its first image committed — `41732be`. `cell_keys()` yields one key per image; `prune_done` keeps a cell unless all are present (reads `num_images` off the cell, so callers pass nothing).
- [x] `web/app.js:614` a resumed night is split across two job ids — `a56431b`. `run_id` = hash of checkpoint, quantization, size, batch and every cell key of the full grid, computed before pruning in `_expand` and the CLI; column + index + migration, `run:` filter, PNG field, the sheet button uses it with `job:` fallback.
- [x] `backend/storage.py:388` quantization not in the record or the resume key — `41732be`. Column, migration, PNG chunk, A1111 line, `ImageRecord`, `quant:` filter, `completed_cells` scope (NULL rows match any quantizer).
- [x] `backend/backends/diffusers_backend.py:403` the recorded negative prompt is not the one that ran — `41732be`. The complete version: `GenerateRequest.applied()` (negative prompt and mu), `backend.applied()` from the loaded signature or the profile, both save paths, both resume prunes, mu in the A1111 block, CLAUDE.md 77-79 corrected.
- [x] `web/app.js:172` negative-prompt gate from the panel's single CFG — `41732be`. Gated on the max of the open grid's resolved CFG axis; `collect()` sends null for a disabled field.
- [x] `web/app.js:695` Reuse restores checkpoint-specific settings onto another checkpoint — `496bfb1`. Loaded model kept in module state; a muted note names both checkpoints. Steps/CFG/mu are still restored (the review's "warn first" option).
- [x] `web/app.js:198` touching Steps/CFG/Mu/Seed freezes all three at the fallback — `496bfb1`. Per-control ownership; Seed owns nothing. `import torch` on the first status call was not moved.
- [x] `web/app.js:264` a cleared CFG box generates at 0 — `496bfb1`. Empty Steps/CFG sent as absent; size stays coerced.
- [x] `web/app.js:1028` the Resume checkbox fires no preview — `463ece1`. `change` on Resume, `input` on the negative prompt; `completed_cells` off the loop in the same commit.
- [x] `web/app.js:857` a promoted axis stays in the pins — `c3b8da5`. `reconcileAxes` on every render; `setAxis` swaps on collision.
- [x] `web/app.js:846` pins survive a change of run — `c3b8da5`. Pins clamped to present values each render; `setSheet(false)` resets the layout.
- [x] `web/app.js:718` no negative-prompt sheet axis — `c3b8da5`. Axis added with clipped labels. `axis_values` still reports only the prompt list, so a negative-only variable shows duplicate prompts in the preview.
- [x] `web/app.js:555` a reset during an in-flight page is dropped — `c3b8da5`. Re-issued after the pending page settles. Covered by the browser smoke since `9fd0480`, not by a unit test.
- [x] `backend/main.py:92` Ctrl-C cannot stop a running sweep — `463ece1`. `uvicorn.Server` subclass cancels active jobs in `handle_exit`; `timeout_graceful_shutdown=30`.
- [x] `backend/main.py:93` Ctrl-C during a load freezes the loop — `463ece1`. `try_unload()` (non-blocking) off the loop in lifespan; the load runs on a daemon thread (`main._detached`) so the process can exit under a download. `load()` itself still cannot be abandoned mid-`from_pretrained`.
- [x] `backend/main.py:364` four storage calls on the event loop — `463ece1`. All through `asyncio.to_thread`; `_expand` is async.
- [x] `backend/jobs.py:258` per-cell status ships the accumulated images — `463ece1`. `snapshot(images=False)` for per-cell and running events and `/api/jobs`; a copied list where images are sent.
- [x] `backend/jobs.py:265` one cell's exception abandons the grid — `463ece1`. Cell-level catch, `failures` in the snapshot, cap of 3 consecutive or a third of the grid; `done` with an error line the UI shows; the CLI mirrors it and exits 1.
- [x] `backend/jobs.py:43` `short_error` cuts the OOM remedy — `463ece1`. Middle cut; the full message logged at ERROR; `KREA_LOG_LEVEL`.
- [x] `backend/storage.py:211` cleanup deletes a committed image on rename failure — `8f947a9`. `committed` flag plus a retried rename.
- [x] `backend/backends/diffusers_backend.py:415` `callback_on_step_end`/`generator` dropped silently — `8f947a9`. Refused at the end of `load()`; optional arguments still drop at DEBUG.
- [x] `backend/launch.py:322` an unknown `KREA_MODEL_ID` is discarded by the menu — `e3489de`. Gets a real row that Enter keeps; test added.
- [x] `wsl/env.sh:22` pre-quantized default regardless of quant — `e3489de` (env.sh conditioned, format interpolated) and `799e67e` (the symmetric preflight in `_quantization_config`); README:131 reworded.
- [x] `web/app.js:597` no downscaled derivative — `c3b8da5`. 384 px WEBP after the rename, `thumb_url` when the file exists, cards and sheet use it, delete/clear/orphan sweep follow it.
- [x] `web/app.js:600` lightbox unreachable by keyboard — `c3b8da5`. One `thumb()` builder: focusable, `role=button`, Enter/Space, focus ring.
- [x] `web/style.css:338` card tools gated on hover — `496bfb1`. `:focus-within`, coarse pointer, focus ring.
- [x] `web/style.css:75` sticky panel has no bound — `496bfb1`. Static below 900 px of height and in the one-column layout (the side-effect-free option).
- [x] `web/style.css:90` note at 3.0:1 / 1.55:1 — `496bfb1`. Colour token; only the textarea dims.
- [x] `tests/test_api.py:42` nothing cancels a job — `463ece1`. Fake honours `should_cancel`; mid-sweep and queued-cancel tests; a JobManager-level `done_sent` test in `tests/test_jobs.py`.
- [x] `tests/test_api.py:75` load-on-first-generate untested — `e3489de`. Fixture wires `ensure_loaded`; fake raises when unloaded; one-load test.
- [x] `tests/test_api.py:147` literal `negative_prompt_supported is True` — `e3489de`. Derived from the profile. `settings.model_id` is not pinned in the fixture.
- [x] `tests/test_schedule.py:39` one generator per image asserted nowhere — `e3489de`. Stub keeps the generators; asserted near `MAX_SEED`.
- [x] `README.md:14` disk and download sizes — `8f947a9`.
- [x] `README.md:317` sheet-label example — `8f947a9`. Prose narrowed to what the code does; the LCS-alignment widening was not built. Superseded in practice by `57bf669`: a prompt grid's cells now carry their variable assignment and the sheet uses one axis per variable, so the word-diff labels only apply to rows from before that.

### LOW

- [x] `backend/backends/diffusers_backend.py:446` "has `negative_prompt` -> 0.0" applied to every pipeline — `8f947a9`. Literal family set (`Krea2Pipeline`, `ZImage*`); checked against the real classes in `tests/test_profiles.py`; CLAUDE.md corrected.
- [x] `backend/storage.py:184` a stale `.png.part` is invisible and swept by nothing — `8f947a9`. Encode inside the guard; counted in `summarise()`, named in `clear.py`'s preview and its "Nothing to clear" check, swept past an hour; CLAUDE.md and README qualified.
- [x] `backend/storage.py:505` `keep` snapshotted before the orphan glob — `c3b8da5`. Glob first, in `clear_history` and `summarise()`.
- [x] `backend/storage.py:231` `step_cost_ms` measures the previous checkpoint — `8f947a9`. `model_id` predicate; CLAUDE.md sentence updated.
- [x] `web/app.js:978` lightbox has no dialog role, trap or restore — `496bfb1`. A `<dialog>` with `showModal()`; `alt` set.
- [x] `web/index.html:104` no live region — `496bfb1`. `role=alert` (unhidden before writing), `role=status`, `progressbar` with `aria-valuenow`, `aria-live` on the status text.
- [x] `web/app.js:626` icon buttons named by their glyph — `496bfb1`.
- [x] `web/app.js:8` `SHEET_PAGE` hard-wired to the default ceiling — `c3b8da5`. `/api/status` publishes `max_grid_images`; the gallery's `le` derives from settings.
- [x] `backend/jobs.py:217` a complete sweep reported "cancelled" — `463ece1`. Status from whether the loop stopped early.
- [x] `backend/jobs.py:202` `_run` overwrites a terminal status — `463ece1`. `_finish` owns every terminal transition.
- [x] `backend/seeds.py:19` span excludes the grid's seed axis — `e3489de`. Caps live in `seeds.py`; schemas read them.
- [x] `backend/backends/diffusers_backend.py:116` unknown `KREA_QUANT`/`KREA_OFFLOAD` silent — `8f947a9`. Validated at import against one vocabulary in `model_defaults`.
- [x] `backend/storage.py:70` `_migrated` flag defeats per-connection recovery — `8f947a9`. Keyed on the database path.
- [x] `backend/clear.py:109` unguarded `vacuum()` before the report — `8f947a9`. Report first, best-effort vacuum, exit 3 after a failed deletion; README exit sentence covers both tools.
- [x] `backend/generate.py:287` negative-prompt note from the base CFG — `41732be`. Counted over the cells: none / some / all.
- [x] `backend/generate.py:200` `--no-mu` swallows `--mu` — `463ece1`. Mutually exclusive group.
- [x] `backend/generate.py:350` Ctrl-C during the load promises a step — `463ece1`. First Ctrl-C ends the load outright.
- [x] `backend/generate.py:254` raw pydantic blob — `463ece1`. `schemas.error_detail` shared with the server.
- [x] `backend/clear.py:108` cannot report or remove a stale `.part` — `8f947a9`.
- [x] `web/app.js:509` `resumeActiveJob` attaches to the newest job — `496bfb1`, then `5cdba2b`. Running first; failing one, the *oldest* queued (the one about to run), and the rest are listed as the queue.
- [x] `web/app.js:142` presets, Reuse, dice leave a stale preview — `496bfb1`.
- [x] `backend/storage.py:251` OFFSET paging duplicates and skips rows — `c3b8da5` (dedupe by id) and `c713330` (the full fix: keyset paging). `GET /api/gallery?before=<id>` with an exact `has_more` from one row fetched past the page; the browser anchors on the last record it holds, so a delete moves nothing either; `offset` stays for scripts.
- [x] `backend/main.py:177` unload guard covers jobs but not loads — `463ece1`. 409 while a load is in flight.
- [x] `backend/launch.py:229` repo-id prompt spins on EOF — `e3489de`.
- [x] `backend/schemas.py:58` `check_budget` is per-image — `e3489de`. `max_batch_pixels` (`KREA_MAX_BATCH_PIXELS`), message points at Runs.
- [x] `backend/main.py:104` Origin always appends the port — `e3489de`. Hostname and effective port, http or https; port-80 test.
- [x] `uv.lock:1323` torch wheels locked by URL without hashes — `122d1a5`. The index publishes no digest for those four wheels, so a lock re-run could not add them; they were downloaded, hashed and their RECORD manifests compared entry for entry with the installed Windows and WSL copies (no mismatches), then written in by hand. `uv lock --check`/`--dry-run` accept them; README says a `--upgrade` that moves torch drops them. No move to PyPI torch: its Windows wheel is CPU-only.
- [x] `web/app.js:669` errors raised under the lightbox are hidden — `496bfb1`. Mirrored into a slot inside the dialog.
- [x] `web/app.js:274` Generate/Cancel destroy focus — `496bfb1`. `aria-disabled` plus the existing early returns.
- [x] `web/style.css:49` no `prefers-reduced-motion` — `496bfb1`.
- [x] `README.md:19` "every checkpoint is gated" — `8f947a9`.
- [x] `backend/launch.py:97` menu restates config's defaults — `e3489de`. One source in `model_defaults`, read by both; no test needed.
- [x] `backend/generate.py:192` `generate.main()` never executed — `f66faf7`. `tests/test_cli.py`: seeds match the server's, dry run, failed cell, resume, exclusive mu, one-line error, durations.
- [x] `wsl/setup.sh:29` uv installed into a PATH only the script sees — `e3489de`.
- [x] `wsl/setup.sh:55` relative `source` for `.bashrc` — `e3489de`.
- [x] `backend/backends/diffusers_backend.py:185` `KREA_DEVICE=cpu` skips the CUDA preflight — `8f947a9`. Offload considered; `torch.device(...).type`.
- [x] `backend/jobs.py:138` `submit()` re-implements the seed loop — `f66faf7`. The review's second option: CLAUDE.md names it as the deliberate second copy (routing it through `expand()` would apply the placeholder check to plain prompts) and `tests/test_cli.py` asserts the arithmetic agrees.
- [x] `backend/grid.py:87` fractional bound on an integer axis — `df75e7e`.
- [x] `backend/config.py:83` `outputs_dir.mkdir()` at import — `8f947a9`, and the checklist commit for the second half. The directory is owned by `storage.ensure_outputs_dir()`, and the CLI's resume lookup runs only against a database that already exists, so `--dry-run --resume` creates nothing.
- [x] `backend/model_defaults.py:209` `hints_for` ignores the live negative flag — `8f947a9`.
- [x] `web/index.html:55` `aria-label` on the `?` badges hides the tip — `496bfb1`. `aria-describedby` from every control; badges named "Help".

## Disputed

- [x] `backend/main.py:93` lifespan's synchronous unload — `463ece1` (see MEDIUM).
- [x] `backend/storage.py:184` a killed encode leaves a `.part` — `8f947a9`. Encode inside `storage`'s own `except BaseException`.
- [x] `web/app.js:373` preview has no request-ordering guard — `f66faf7`. Epoch token, as the gallery has.

## Unverified leads

- [x] `diffusers_backend.py:87` `_hub_file` on a local directory — `f66faf7`.
- [x] `storage.py:184` `outputs_dir` never recreated — `f66faf7`. `ensure_outputs_dir()` before every write and before the mount.
- [x] `diffusers_backend.py:199` `KREA_DEVICE=cuda:1` loads onto GPU 0 under offload — `c713330`. Both offload hooks are handed the configured CUDA device and `describe()` reads that card's name and memory; tested with a recording stub. Not run on a two-card machine.
- [x] `app.js:994` `stepLightbox` overwrites a concurrent back-step — `f66faf7`.
- [x] `app.js:965` lightbox image keeps `alt=""` — `496bfb1`.
- [x] `generate.py:327` `--quiet` stdout carries prose — `8f947a9`.
- [x] `generate.py:48` size table in four places — `f66faf7` (app.js vs generate.py) and `c713330` (the rest). `backend/sizes.py` is the one source; `config.py` derives `max_side`/`max_pixels` from it (2736 and 1776x2368, where 2816 and 4,300,000 described nothing), `index.html`'s `max=` follows, and tests hold the README table and `app.js` to it.
- [x] `diffusers_backend.py:72` `describe()` dead keys and `_error` — `799e67e`. Removed; the ABC names the required and optional keys.
- [x] `storage.py:125` the `"mu" in row.keys()` guard's stale justification — `c713330`. The guards are gone: every row comes out of a connection `_migrate` has run on, and the docstring says a new column wants a `_migrate` entry, not a read guard.

## Possible improvements

Not defects. The review ranked them by value in this order. The two left need
weights or a file this machine does not have; each says which.

- [x] 1. `backend/main.py:291` nothing says what happened to last night's run — `98c5ab6`. `GET /api/runs` (`storage.runs`: images, favourites, prompts, jobs, timestamps, the oldest image's prompt and seed) outlives the process; `GET /api/jobs?include_finished=true` adds the terminal jobs from memory with `image_count`. UI: a Runs picker beside the search box (pick one: `run:` search plus the sheet) and a dismissable banner under the panel when nothing is live, gated to 24 h and once per job per browser.
- [x] 2. `web/style.css:371` the lightbox cannot zoom — `5cd6508`. Explicit-size layout in a `flex: 1 1 auto; min-height: 0` stage (not a transform); wheel about the pointer (non-passive), drag, `+`/`-`/`0`/`1`, double-click; 1:1 divides by `devicePixelRatio`; the view is keyed on the record's size and carried across cells; the backdrop click closes only at fit.
- [x] 3. `backend/generate.py:317` CLI grids print no time estimate — `f66faf7`. `grid.estimate_seconds` shared with the preview; `model_id` predicate; guarded on `db_path.exists()`; hours branch in `_duration`.
- [x] 4. `backend/main.py:254` re-render a night's winners at full size — `b26ee19`. `POST /api/rerender`: the gallery's filter server-side, one cell per record with its own settings and seed, `check_budget`, refused over the ceiling and for another checkpoint's records, pruned like a resume, 409 when nothing is left; a "Re-render N at WxH" button beside the filters. The README says the same seed at another size is a new sample.
- [x] 5. `tests/test_schedule.py:40` the scheduler stub never calls back — `e3489de`. Progress, cancel and generator code now run under test.
- [x] 6. `backend/grid.py:78` the prompt-variable assignment is thrown away — `57bf669`. `variables TEXT` (JSON, `_migrate`), the `krea` chunk, `ImageRecord.variables`, both writers; the sheet builds one axis per variable and suppresses the flat prompt and negative axes when any record carries variables. No `var:` search term, as the review advised.
- [ ] 7. `backend/backends/diffusers_backend.py:376` klein's reference-image conditioning unreachable *(medium)*. Not done: the klein weights are not in the Windows cache (only the Nunchaku re-uploads under WSL), so sizing the 2-3x cost and checking that the resized reference conditions means a download first; and it widens `CellKey`, `completed_cells`, the `krea` chunk and the CLI in one change.
- [x] 8. `backend/jobs.py:225` no log lines and no timestamps — `f66faf7` (lines and timestamps) and `079f647` (the UI reads them: elapsed and an ETA from the job's own pace under the bars, a summary line when it finishes).
- [x] 9. `web/app.js:1068` the grid spec lives only in the DOM — `8b5c4c6`. `localStorage` in two phases (model-independent fields at start-up, steps/CFG/mu once the checkpoint is known and matches), delegated `input`/`change` plus the programmatic writes, Resume never stored, every access in try/catch.
- [x] 10. `backend/generate.py:69` no `--from-image` — `0a164fc`. Reads `.info["krea"]` before `apply_env` so the checkpoint reaches `KREA_MODEL_ID`; flags win; quantization not carried (named when it differs); job/run displayed, never reused.
- [x] 11. `backend/clear.py:55` deletion is all-or-nothing — `6b8c4ee`. `--where` with `key:value` terms only, a preview naming what substring terms matched, `DELETE ... RETURNING filename` in one statement, no orphan sweep on that path.
- [x] 12. `backend/storage.py:143` no reindex from PNG metadata — `6b8c4ee`. `backend/reindex.py`: `.info` not `.text`, idempotent on the UNIQUE filename, `.part`-then-rename copy-in, never over a different file of the same name, timestamp from the filename as UTC, through the shared INSERT so FTS follows. Favourite and duration do not come back.
- [x] 13. `backend/jobs.py:198` "waiting for the job ahead" during the download; no Load button — `079f647`. `Job.waiting_on` (`load`/`queue`) beside the status, set only when a load is actually needed; a Load button in the top bar while unloaded or after a failed load.
- [x] 14. `backend/grid.py:351` the seed axis is a count, not a list — `079f647`. `seed_axis` / `--seed-list` through `parse_axis`; wins over the count and the base; counted by the ceiling; not wrapped (the schema's bounds refuse `-5`); resume waives the base seed for it; the preview lists the seeds.
- [x] 15. `backend/backends/diffusers_backend.py:405` real CFG welded to the distilled guidance value on FLUX.1 — `8616111`. `GenerateRequest.true_cfg` (None = off), `true_cfg_supported` from `describe()` and the profile, merged on `/api/status` like the threshold; `applied()` compares the threshold against whichever knob decides. The recorded value is the scale the negative prompt ran at on *every* checkpoint (so the column, the `tcfg:` filter, the stats grouping and the resume key agree, and pre-column rows read as the CFG they ran at); `run_id` keeps the six-element key where the field is unset, so no existing run id changes. The True CFG box, greyed where CFG is real CFG; `--true-cfg`. Verified on the card: no true CFG with a negative prompt is the byte-identical image to none; 1.5 is a different, reproducible image. The fourth grid axis was not added, as the review advised.
- [x] 16. `web/app.js:307` the browser holds exactly one job — `5cdba2b`. Submitting during a job queues it; the buttons are held only for the POST; a queue line with "Cancel queued"; `watchNext()` attaches to the next job; `resumeActiveJob` prefers the running job, then the oldest queued.
- [x] 17. `web/app.js:920` the sheet shows no favourites and crops every cell — `8b5c4c6`. `.fav` star per cell toggled by class (never a re-render), the commonest aspect ratio on the sheet with `object-fit: contain`.
- [x] 18. `backend/storage.py:258` `created_at` never used; no `date:` filter — `079f647`. A local day (`today`, `yesterday`, `YYYY-MM-DD`) as a half-open range on `created_at`, converted to UTC and formatted like the rows.
- [x] 19. `backend/storage.py:163` no content hash — `8616111`. Measured first: the same request twice through the fp8 FLUX path under model offload is the same hash. 16 hex of the sha256 of the pixels in the row, the PNG and the record; `hash:` filter; `distinct_count` on the job snapshot and `distinct` on `RunSummary` (only when every image of the run is hashed); the UI, the Runs picker and the CLI say "only N distinct". The `stddev` half was dropped as advised. The companion warning: `grid.duplicate_count` in the preview and the dry run, through `backend.applied()`; the cells are not dropped.
- [x] 20. `backend/generate.py:338` CLI and server share a GPU unknowingly — `0a164fc`. `_server_holds_gpu` probes `/api/status` after the `--dry-run` guard; `ready`/`loading` or a timeout refuses with exit 4; no listener is the ordinary case; `--ignore-server`. Advisory, blind across WSL/Windows, and stubbed in the test fixture.
- [x] 21. `backend/main.py:177` no way to hand the GPU back mid-night — `5cdba2b`. The cheap half the judges recommended: `POST /api/jobs/{id}/pause` holds at the cell boundary with a worker-written `paused` acknowledgement, `backend.free_cache()` returns the allocator reserve, the hold is polled (never `Event.wait()`), unload still 409s. Not measured on the card.
- [ ] 22. `backend/backends/diffusers_backend.py:199` no LoRA loading *(medium)*. Not done: a new `peft` dependency, torchao + PEFT is unverified, there is no LoRA file on this machine to try one against, and the columns and resume key have to land with the first working generation.
- [x] 23. `backend/clear.py:49` no gallery size or free space on a running surface — `6b8c4ee`. `disk_free_mb` on `/api/status` (one syscall; the directory walk stays off the poll), `bytes` on the preview and a disk line in the CLI from the median PNG size at that size on that checkpoint; the preview turns amber past the free space.
- [x] 24. `backend/storage.py:325` which settings produced the favourites — `0a164fc`. `python -m backend.stats`: the starred *rate* per setting over a scope, the newest run by default, `--all` labelled indicative, `--min` 5, grouped on the six-decimal rounding resume uses, size always included.
- [x] 25. `tests/test_klein.py:45` pipeline facts never checked against the real classes — `f66faf7`. `tests/test_profiles.py`, five id -> class cases, mapping marked as the unverifiable assumption.
- [x] 26. `tests/test_gallery.py:22` PNG metadata never read back — `799e67e`. Key-set equality (now including `variables`).
- [x] 27. `backend/launch.py:62` catalogue not joined to the profile table — `f66faf7`. By identity against `_FALLBACK`, over every catalogue repo and lite re-upload.
- [x] 28. `web/app.js:450` no elapsed time or ETA in the UI — `079f647`. From the job's own finished-cell pace (`elapsed` on the progress event, server clock), in the cell line under the bars; a "Finished N images in T" line when done, outside the element `setBusy` hides.
- [x] 29. `backend/backends/diffusers_backend.py:354` live latent preview — `8616111`. The matrices were fitted on the card rather than copied: least squares from pipeline-normalised latents of the real FLUX `AutoencoderKL` and Krea 2 `AutoencoderKLQwenImage` (R² 0.96 / 0.95 on real images), keyed on VAE class and channel count as advised, FLUX.1 and Krea 2 only (klein's callback has no unpacker), x_t handled with a gain and a blur, the whole callback in a try/except that switches itself off. Frames from a real render were looked at. Found on the card and fixed: the per-cell subscriber check skipped every first cell.
- [x] 30. `backend/jobs.py:251` no wall-clock budget — `57bf669`. `budget_minutes` / `--budget` as an absolute deadline from submission (the load counts), checked at the top of every cell, through the cancel path with `stopped_by: "budget"` so the UI reports the plan working and points at Resume.
- [x] 31. `web/app.js:839` no contact-sheet export — `b4e5aad`. Client-side canvas from the thumbnails and the sheet's own layout, stars and a caption (search, pins, checkpoint, size, run, time), downloaded as a PNG.
- [x] 32. `backend/backends/diffusers_backend.py:72` dead state — `799e67e`.
- [x] 33. `tests/test_launch.py:38` `clean_env` leaks `KREA_*` — `e3489de`. Save/restore fixture.

## Commits

In order, oldest first, each corresponding to a step of the review's fix order
or to one pass over a list:

| Commit | Fix order | What |
| --- | --- | --- |
| `a56431b` | 1 | Base seed published; `run_id` |
| `41732be` | 2 | Resume key: batch cells, quantization, `applied()` |
| `c3b8da5` | 3, 7 | Viewer cap, sheet index and pins, thumbnails, gallery reset |
| `e51e6e4` | 4 | Mu fallback subtracts the static shift |
| `463ece1` | 5, 6 | Ctrl-C paths, per-cell failures, terminal status, loop hygiene |
| `df75e7e` | 8 | Prompt-axis guard, empty variables, fractional int axis |
| `496bfb1` | 9 (+ UI MEDIUMs) | Per-control ownership, empty CFG, Reuse note, accessibility |
| `e3489de` | 10 (+ LOW) | Menu unknown id, seed span, batch budget, Origin, WSL scripts, test gaps |
| `8f947a9` | LOW | Save path, stale partials, step-cost scope, threshold family, mode vocabulary, clear.py, README |
| `f66faf7` | LOW, improvements | CLI estimate, job timestamps/logs, preview epoch, leads, new test files |
| `799e67e` | MEDIUM tail | Symmetric pre-quantized preflight, dead describe state |
| `c77f757` | — | This checklist |
| `c713330` | LOW tail, leads | Keyset paging, `KREA_DEVICE` under offload, `sizes.py`, the read guards |
| `98c5ab6` | improvement 1 | `/api/runs`, finished jobs, the banner and the Runs picker |
| `5cd6508` | improvement 2 | Lightbox zoom, pan, 1:1, the carried view |
| `b26ee19` | improvement 4 | `/api/rerender` and its button |
| `079f647` | improvements 8, 13, 14, 18, 28 | Seed list, `waiting_on`, elapsed/ETA, `date:`, Load button |
| `8b5c4c6` | improvements 9, 17 | Panel persistence, sheet stars and aspect |
| `0a164fc` | improvements 10, 20, 24 | `backend.stats`, `--from-image`, the server probe |
| `6b8c4ee` | improvements 11, 12, 23 | `clear --where`, `backend.reindex`, disk free and bytes |
| `57bf669` | improvements 6, 30 | Time budget, the variable assignment recorded and on the sheet |
| `5cdba2b` | improvements 16, 21 | Pause between cells, the browser's queue |
| `b4e5aad` | improvement 31 | Sheet export |
| `9fd0480` | — | `tools/`: the fake server and the headless-browser driver |
| `b5b21b5` | — | The checklist brought up to date |
| `122d1a5` | LOW tail | The PyTorch wheels pinned by hash |
| `8616111` | improvements 15, 19, 29 | True CFG apart from the guidance value, the pixel hash and the duplicate warning, the live preview; the GPU pass |
