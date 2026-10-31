# Source-file audit, 2026-10-11

Audited the tracked Python, JavaScript, CSS and HTML in `backend/`, `web/`,
`tests/` and `tools/` at `d7c71db` (200 files). Excluded dependencies,
generated assets, review prototypes and historical documentation. Of these
files, 18 exceeded 1,000 lines: 16 application source files and two test
modules. Line count was a screening tool: each file was mapped by
responsibility first, and a file was split only where a role had grown that
other files call or that shares no state with the rest. Splits move code
verbatim, with no new build step or runtime feature. This is the second pass;
the first, at `21afcdf` on 2026-10-07, is kept below and marked as such. The
third, on 2026-10-31, is [its own section](#third-pass-2026-10-31) after
this one's verification; its splits are the 2026-10-31 rows of the
boundaries table, and its retained files have a table of their own.

After the second pass 8 files were above 1,000 lines; the table of retained
files says why each stayed whole. Its figures are as audited then.

The 2026-10-28 saved gallery search addition follows the same boundary:
`saved_searches.py`, `schemas_saved_searches.py` and `api_saved_searches.py`
own definitions and cold CRUD outside gallery SQL readers; `saved-searches.js`
and `styles/saved-searches.css` own the dialog. Its explicit filter transition
uses the existing gallery epoch, sheet and bin paths. `saved-searches.js` owns
the dialog's declarations and its wiring (`initSavedSearches`), and `app.js`
calls that init once at boot - the convention every feature file with an
init function follows.

## Implemented boundaries

| Pass | Original file | Lines before | Responsibilities after the split |
| --- | --- | ---: | --- |
| 2026-10-31 | `backend/storage.py` | 2,112 | Schema, migrations, connections, record conversion, writers, the masks table and its holds, partial files and live reads (1,808). `run_jobs.py` (162) retained run jobs (`retain_run_job` .. `finish_run_job`), reached as `storage.<name>` through `_MOVED` and reading `storage.settings`, `_write_lock` and `_session()` at call time, as `gallery_delete` does; `record_fields.py` (207) the pure row and chunk field codecs (mask hash and provenance, the outpaint chunk, the peaks), imported back by name. The mask writers, holds, `mask_path` and `stamp_outpaints` stay, since tests patch them on `storage`. |
| 2026-10-31 | `backend/reindex.py` | 1,150 | Not a length split: the bundle layer imported the CLI for private names. The PNG chunk codec (`PNG_SIGNATURE`, `text_keyword`, `metadata_chunk`) and `execution_from_sources`/`_info` moved to `metadata.py`, `_DB_REQUIRED` to `bundle_common`; `reindex` (1,110) keeps its old names as aliases. |
| 2026-10-31 | `backend/generate.py` | 2,084 | The local run, `main()` and the helpers that reach the gallery (1,262). `generate_args.py` (591) the parser, `--from-image` settings, the outpaint, mask and size reads and the flag-only checks, imported back as globals so `generate._image_settings`/`_mask_file` patches intercept; `generate_remote.py` (284) the `--via-server` client, handed the three shared helpers by `main()`; `submission_rules.py` (202, from `submission.py`, which re-exports it) the service-free submit checks. `test_profiles.py` holds `generate_args.SIZES is sizes.SIZES`. |
| 2026-10-31 | `tests/test_cli.py` | 1,854 | Seeds, the loop equality AGENTS.md names, budgets, `--from-image`, conditioning, Then refine and the Kontext refusal (805). `cli_helpers.py` (103) the shared fixtures, the autouse isolation included; `test_cli_passes.py` (592) refine, inpaint, outpaint and the local GAN; `test_cli_via_server.py` (421) the client. 122 node ids before and after. |
| 2026-10-31 | `tests/test_api_grid.py` | 1,298 | The grid routes (771). `test_api_grid_variants.py` (323) prompt-variant comparisons; `test_api_rerender.py` (219) Re-render; `VARIANTS`, `TEXTS` and `LABELS` in `api_helpers.py`. 66 node ids before and after. |
| 2026-10-31 | `tests/test_api_model_switch.py` | 1,044 | The switch (836). `test_api_model_catalogue.py` (204) the eleven tests that only read `/api/models` and `/api/status`; `model_switch_helpers.py` (27) the constants and the autouse `fresh_selection`. 54 node ids before and after. |
| 2026-10-31 | `web/snippet-import.js`, `web/setup-import.js` | 339, 407 | Not a length split: the setup import was a function-for-function copy of the snippet import. `import-merge.js` (170, loaded before both) the stateless merge machinery; each dialog keeps its state lets, preview, commit and uncertain-import flow (256 and 321), so Saved setups still never calls `setupSummary` and neither resends an uncertain import. |
| 2026-10-31 | `web/analysis.js`, `web/grid-preview.js`, `web/shortcuts.js` | 839, 314, 312 | Not a length split: page-wide pure helpers that lived in late, feature-specific files moved verbatim, names unchanged, to `page-helpers.js` (45), loaded right after `state.js`: `plural` (called in 21 files), `excerpt` and `units` from `analysis.js`, `humanTime` (11 files) from `grid-preview.js`, and `keyboardEventHeld`, `keyboardEditingTarget` and `keyboardCommandModifier` (up to 23 files, 19 of them loaded before `shortcuts.js`) from `shortcuts.js`. Every call was already at run time, so nothing changes but where the names are declared. `span` and `note` stay in `analysis.js`, and `keyboardSubmit` in `shortcuts.js`, used only there and in `app.js`. `test_check_dismissal.py` loads the new file before `analysis.js`. The page loads 66 classic scripts. |
| 2026-10-31 | `web/gallery.js` | 982 | Gallery paging, cards and keyboard navigation (799). Not a new file: the tail, which reads no gallery state, went to the files whose roles it is. The Reuse button (`modelsDiffer`, `syncReuseButton`, `notCarried`, `reuseButton`, `syncReuseButtons`) moved to the end of `reuse.js` (621), and a record's title words (`cardTitle`, `passWord`, `outpaintOf`, `goneWord`, `editGoneWord`) to the end of `facts.js` (444). Both load after `gallery.js` and every use is at call time, so `index.html` is unchanged. |
| 2026-10-31 | `web/gallery-actions.js` | 1,180 | Starring, Bin picked, deleting, forgetting and taking back, Delete unstarred's sweep and the ask dialogs (1,028). `gallery-head.js` (157, loaded right after it) the gallery head's pointer-stable layout (`headPointer` through `libraryPointerLeft`), which shares no state with the rest and which `app.js`, `generation.js`, `gallery.js`, `job-results.js` and `bin.js` call; the rest reaches it only through two run-time `syncDeleteUnstarred()` calls. The ask dialogs (`openAsk` through `askBackdrop`) stay: a further split was optional and is not taken. `test_runs_favorite_counts.py` still evaluates `gallery-actions.js` alone, whose favourite code never names the head. The page loads 65 classic scripts. |
| 2026-10-31 | `web/tags.js` | 788 | After the note's split above, autocomplete, the chip editor and the tray (596). `tag-lists.js` (196, loaded right after it) the suggestion lists the chip editor, the Refine dialog's prompt and the panel's negative box share (`selectEditorSuggestion` through `editorListKeydown`) and the negative box's own (`suggestNegativeTags` through `negativeCaret`): function declarations whose state is `state.js`'s, reaching back into `tags.js` (`commitTagEdit`, `tagKind`) only at call time. The page loads 64 classic scripts. |
| 2026-10-31 | `web/tags.js` | 904 | Prompt writes (`writeBox`, which clears the tag typing state and stays), tag autocomplete and editing (788). `prompt-note.js` (104, loaded right before it) the note under the prompt and its Undo: `promptUndo`, `PROMPT_NOTE_HOME`, `showPromptNote`, `offerPromptUndo`, `dropPromptUndo`, `syncPromptUndo`, `undoPromptWrite` and `discardPromptNote`, which use no tag state and which eight files call. Its comment keeps the object's shape and the `moveBefore` reason and points to docs/rules.md for the rest, which it repeated. The page loads 63 classic scripts. |
| 2026-10-31 | `web/tag-chips.js` | 1,012 | The chips, the verdict cache, validation scheduling, placement and `writePromptTags` (660). `tag-helper.js` (356, loaded right after it) the Tags helper the 2026-10-07 pass placed here, grown past 1,000 lines with the mixed-prompt annotations, Anima's Start with and the grid-variable chips: `togglePromptTag` through `syncTagHelper`, all function declarations, whose one state (`tagHelper`) is `state.js`'s. The page loads 62 classic scripts. |
| 2026-10-31 | `web/inpaint.js` | 1,068 | The Inpaint dialog (1,013). Not a new file: `renderPassCheck`, `syncPassCheck` and `selectPassFrom`, which the outpainting pass generalised in place here, moved verbatim to `refine-prompt.js` beside `refineCutLine`, whose own comment already named it their owner; `refine-prompt.js` loads before `inpaint-mask.js`, `inpaint.js` and `outpaint.js`, so Outpaint no longer reaches into the Inpaint dialog's file for them. `sourceGoneKind`/`sourceGoneWords` went to `refine-rules.js` in the split above. |
| 2026-10-31 | `web/refine.js` | 1,178 | The Refine dialog's state machine around its shared lets (1,089). `refine-rules.js` (110, loaded right before it) the seam the 2026-10-11 pass retained as "not worth a file yet", whose callers grew from four files to ten (generation, panel, state, refine-batch, lightbox, lightbox-compare, sheet-render, inpaint, outpaint, then-refine): the refine bounds (`REFINE_MAX_SIDE`/`REFINE_MAX_PIXELS`, byte-identical for `tests/test_profiles.py`), `REFINE_KEY`, `SINGLE_STRENGTH`, `refinePrefs`, `refineScaled`, `refineFits`, `refineDefaultStrength`, `refineTight`, `openingRefineSize`, `refineKept` and `refineSizeProblem`. `REFINE_ANIME6B` and `refineUpscalerAvailable` stay, used only in `refine.js`. `sourceGoneKind`/`sourceGoneWords` moved from `inpaint.js` beside them, and `refineSourceGone` now uses them, as `inpaintSourceGone` does, instead of an inline copy whose regexes had drifted (no optional full stop, no `/i`); `sourceGoneWords` makes "refine" "refined". The retained row for `refine.js` is superseded. The page loads 61 classic scripts. |
| 2026-10-31 | `web/sheet.js` | 1,069 | Axes, labels and controls (765). `sheet-zoom.js` (307, loaded right after it) the zoom and pan (2a8c998), which nothing in the axes and labels reads; `sheet-render.js`, `app.js` and `sheet-export.js` reach it at run time. `let sheetLayout` moved from `sheet-export.js` to `sheet-render.js`, whose `drawSheet` is its only writer. The page loads 60 classic scripts. |
| 2026-10-31 | `web/bundles.js` | 1,205 | The bundle dialog: export, Back up, import, Bundle & clear's lifecycle and status polling (846). `automatic-backup.js` (365, loaded right after it) the automatic startup backup settings and their state, which reach back only to `bundleActionHold`, `focusBundleField` and `els.bundlesDialog`; `bundles.js` still calls `initAutomaticBackup`, `refreshAutomaticBackup`, `syncAutomaticBackupControls` and the two result-text helpers at run time. The page loads 59 classic scripts. |
| 2026-10-31 | `backend/api_generate.py` | 881 | Generate, Run grid and its preview (267): `expand_grid` and `then_refine_spec`. `api_rerender.py` (309) Re-render and its preview, `foreign_refusal` and the plan types, with `rerender_plan`/`rerender_selection` still nested in its `create_router`, moved unchanged rather than lifted; `api_runs.py` (345) run details and Finish run, `FinishPlan`, `_finish_key` and `finish_plan`. The three families shared no helper. `main` includes the runs router before generate and the Re-render one after, and the sorted OpenAPI document is byte-identical. Run details call More seeds' planner, so `more_seeds_plan` and its helpers moved from `api_more_seeds.py` to `more_seeds_planner.py` (216) and no router imports another. |
| 2026-10-31 | `backend/schemas_generate.py` | 1,077 | The request, `applied()`, grids, the submissions, `JobAccepted` and Load settings from PNG's models (925). `schemas_outpaint.py` (167) an outpaint's spec (`OutpaintSpec`, `Outpaint`, `_on_grid`, the key pattern) and the mask and preset answers, a leaf reading no settings that storage, `schemas_gallery` and the CLI use; `schemas_generate` imports `Outpaint` back and `schemas.py` re-exports all five. The `Png*` models stay: about 60 lines that are the request's own answer, not worth a file. |
| 2026-10-31 | `backend/critic_validation.py` | 1,118 | Locating quotes, the answer's JSON, the unchanged/part/lettering helpers, the rewrite checks and `validate` (747). `critic_issues.py` (389) what an issue says: the restated-finding vocabulary (`_GIVEN` .. `_PART_COUNT`, `_says`, `_restated`, `_MEASURED`) and the order wants (`_PART` .. `_described`), moved verbatim; it uses nothing of the rest, and `critic_validation` imports the six names `validate` calls under their private names (renaming them is outside this pass). The rewrite checks stay: they use `locate`, `_lettered` and `_unchanged`, and `validate` calls them. The twin clause regexes stay one per module, held equal by a test. |
| 2026-10-31 | `backend/api_bundles.py` | 441 | Not a length split: the bundle routes (240) kept service objects the automatic backup imports, so `backup_service` and `backup_coordinator` imported a router and `api_backup` another router. `bundle_manager.py` (212) `BundleManager`, `GalleryAdmission` and the prefixes, moved verbatim; `schemas_bundles.py` (34) the three request bodies, as every other area keeps its models in `schemas_*`. `api_bundles` re-exports the names main and the tests reach through it. |
| 2026-10-31 | `backend/jobs.py` | 1,711 | `JobManager`, the queue, its gate, submission, the worker and Then refine's stage (1,318), kept whole: the Then refine stage calls `_emit`, `_save`, `_hold` and the backend, and `submit_cells`/`_generate` close over the job and loop. `job_state.py` (387) the `Job` dataclass and `_TERMINAL`, which read no gallery, settings or backend; `error_text.py` (28) `short_error`, standard library only, so `prompt_check` and `tag_lookup` no longer import the queue. `jobs` imports all three names back; the tests' patches (`jobs._run_id`, `jobs.current_settings`) stay on `jobs`. |
| 2026-10-11 | `backend/main.py` | 2,629 | App lifecycle and model ownership (856): load state and lock, `ensure_loaded`, `_load_failure`, the switch, lifespan, status, model and LoRA routes. The service context the 2026-10-07 audit asked for is `app_context.py` (52): `AppServices`, late-bound getters over main's own globals, so tests that patch `main.backend`/`main.jobs`/`main.settings` still reach every route. `detached.py` (40) runs work on a daemon thread; `tag_lookup.py` (173) the Danbooru lookup and tag categories; `prompt_check.py` (262) the analyser's blocking layer (`analyse`, `lint`, `order_slots`, the order detail); `api_prompt.py` (368) the expand/analyze/measure/critique/tags routes; `submission.py` (168) shared submission checks and `resolve_selection`; `api_generate.py` (263) generate, grid, grid preview and Re-render; `api_swap.py` (214), `api_refine_batch.py` (233) and `api_more_seeds.py` (241) Swap tags, Mass refine and More seeds. |
| 2026-10-11 | `backend/storage.py` | 1,866 | Schema, migrations, connections, record conversion, writers, partial files and live reads (1,170). `gallery_reports.py` (306) holds the read-only aggregates (estimates, keep rates, runs); `gallery_delete.py` (461) takes images out (scoped deletes, the recycle bin, clear and vacuum). Both import storage, never the reverse, so a patched `storage.settings` is read at call time; moved names still resolve as `storage.X` through a PEP 562 `__getattr__`. |
| 2026-10-11 | `backend/schemas.py` | 1,586 | A re-export facade (156): `settings`, explicit imports and `__all__`. The models live by role in `schemas_generate.py` (476, `GenerateRequest` and `applied()`), `schemas_prompt.py` (375), `schemas_gallery.py` (221), `schemas_rerun.py` (341, Re-render, Swap tags, Mass refine, More seeds) and `schemas_model.py` (224). A runtime patch of the settings `applied()` reads targets `schemas_generate.settings`. |
| 2026-10-11 | `backend/bundles.py` | 1,162 | Verified export, merge and import (590). `bundle_common.py` (76) the shared vocabulary (errors, progress, chunking, read connections); `bundle_images.py` (230) the tracked-image checks (pixels, digests, metadata rewrite); `bundle_archive.py` (334) the archive format, staging and the space check. |
| 2026-10-11 | `backend/prompt_encoding.py` | 1,264 | Measuring a prompt with the checkpoint's own tokenizers (716), re-exporting the table. `encoder_table.py` (379) the per-checkpoint encoder recipes and their source contracts; `prompt_weighting.py` (219) the literal weighting syntax and its finding. |
| 2026-10-11 | `web/lightbox.js` | 1,326 | Navigation, view and layout state machine (800). `facts.js` (239, loaded before it) a record's settings line and facts list, which the lightbox and the viewer share, with its reference thumbs and Refine-source link; `lightbox-compare.js` (304, loaded after it) the comparison's controls, toggles, divider and pass stepping; its geometry and state block stay in `lightbox.js`. |
| 2026-10-11 | `web/model-switch.js` | 1,183 | The Models picker's table, favourites, selection and the switch (912). `model-shots.js` (277, loaded after it) the strip of the chosen checkpoint's gallery images. |
| 2026-10-11 | `web/app.js` | 1,072 | Event wiring and boot, still the last script (620). `prompt-examples.js` (104) the Examples menu; `prompt-expand.js` (356) ERNIE's Expand. Both load just before `analysis.js`. |
| 2026-10-11 | `web/generation.js` | 1,010 | Submission and the job lifecycle (495). `job-progress.js` (264) the watched job's progress, its mirrors and Pause; `job-queue.js` (257) the queue rows (`setQueue`) and which job the page watches. |
| 2026-10-11 | `tests/test_prompt_encoding.py` | 1,209 | The encoder table and its source contracts against installed diffusers (344). `prompt_encoding_helpers.py` (180) fixtures and the failure table; `test_prompt_encoding_budgets.py` (295), `test_prompt_encoding_syntax.py` (157), `test_prompt_encoding_loading.py` (324). |
| 2026-10-11 | `tests/test_bundles.py` | 1,206 | Export, verification and safe clearing (361). `bundle_helpers.py` (77); `test_bundles_merge.py` (316), `test_bundles_repeat.py` (210), `test_bundles_password.py` (90), `test_bundles_staging.py` (130) and `test_gallery_lock.py` (91). |
| 2026-10-07 | `backend/critic.py` | 3,320 | Coordinator; `critic_runtime.py` installs/verifies the pinned runtime; `critic_server.py` owns processes and loopback requests; `critic_context.py` builds the question and budgets; `critic_reorder.py` checks pure permutations; `critic_validation.py` validates answers. |
| 2026-10-07 | `backend/prompt_rules.py` | 2,422 | Rule context and dispatch; `prompt_text.py` parses text and lettering; `prompt_order.py` places tags and checks category coverage. |
| 2026-10-07 | `backend/main.py` | 1,917 | App lifecycle and model/prompt/generation routes; `api_gallery.py` owns browsing, runs, the recycle bin and deletion; `api_jobs.py` owns status, cancel, pause and SSE. |
| 2026-10-07 | `web/style.css` | 3,032 | Tokens/base layout plus seven consecutive component sections in `web/styles/`, linked in the original cascade order. Later: `bundles.css` (gallery bundles) and `model-picker.css` (the Models picker, moved 2026-10-09 from between the topbar and layout rules and linked right after `style.css`); `style.css` is now 390 lines. |
| 2026-10-07 | `web/generation.js` | 2,209 | Submission/queue; `grid-preview.js`, `job-stream.js`, `job-results.js`. |
| 2026-10-07 | `web/sheet.js` | 1,822 | Sheet model/axes; `sheet-render.js`, `sheet-export.js`. |
| 2026-10-07 | `web/gallery.js` | 1,768 | Gallery layout/cards; `gallery-actions.js`, `reuse.js`. |
| 2026-10-07 | `web/tags.js` | 1,604 | Suggestions/editing; `tag-tokens.js`, `tag-chips.js` (chips, order slots, the Tags helper and its caret line, which presents the server's order and slots). |
| 2026-10-07 | `web/analysis.js` | 1,593 | Prompt check; `analysis-critic.js`, `comparison.js`. |
| 2026-10-07 | `web/panel.js` | 1,533 | Settings/persistence; `runs.js`, `viewer.js`. |
| 2026-10-07 | `web/refine.js` | 1,444 | Dialog/source/request lifecycle; `refine-prompt.js` owns prompt controls. |
| 2026-10-07 | `tests/test_critic.py` | 4,696 | Runtime/install/process/server/context/tokens and answer validation, lettering, issues, reordering and sources. |
| 2026-10-07 | `tests/test_prompt_rules.py` | 2,175 | Guide consistency, text/prose, lettering, order, slots and checkpoint-specific rules. |
| 2026-10-07 | `tests/test_api_analysis.py` | 1,913 | Analysis routes, tags, guides, critic and lifecycle. |

Added between the two passes, with no split: `model-switch.js` owns the
Models picker, styled by `styles/model-picker.css`; `bundles.js` and
`styles/bundles.css` own gallery bundles (the backup and its scope, picked or
starred, included; the picked ids are read from `gallery-select.js`). `gallery-select.js` owns the
gallery's selection (Select, the picked ids, and `galleryActionTarget` /
`galleryActionScopeLine`, which the three footer actions and Re-render word
alike) and, since 2026-10-12, the Picked menu, whose star and bin
(`starPicked`, `binPicked`) are `gallery-actions.js`'s beside a card's;
`gallery-batch.js` a card's batch outlined on hover; `lightbox-cfg.js` (531) the
lightbox's comparison across CFG and across steps. `tag-swap.js` + `styles/swap.css`,
`refine-batch.js` + `styles/refine-batch.css`, `more-seeds.js` +
`styles/more-seeds.css` and `bin.js` + `styles/bin.css` own Swap tags, Mass
refine, More seeds and the recycle bin's page side. What each of those
actions does is the server's: `backend/tag_swap.py`, `refine_batch.py` and
`more_seeds.py`, reached since this pass through `api_swap.py`,
`api_refine_batch.py` and `api_more_seeds.py`, the selection resolver
`submission.resolve_selection` and `grid.cell_from_record`. Which reads leave
the bin out is `storage._LIVE`; the bin's writes are `gallery_delete.py`'s and
its routes `api_gallery.py`'s. `forgetImages` and `unforgetImages` stay in
`gallery-actions.js`.

Added since this pass, with no split: `png-settings.js` (loaded right
after `reuse.js`) owns Load settings from PNG's page side - reading a PNG's
text chunks, the drop (never a drag that began on the page) and the
picker's file - and fills the panel through
`reuse.js`'s `reuseIntoPanel`, which now takes the note's name and whether
the sampling values carry. The route is `api_png.py`; the reading
`--from-image` did in `generate.py` (`_image_settings`, now a wrapper that
turns refusals into argument errors) moved to `png_settings.py`, pure like
`metadata.py`, so the parser/service seam below has one function less.

Added since this pass, with no split (inpainting, `feat-inpainting`):
`inpaint.js` (882) owns the Inpaint dialog - opening, Reuse, the mask's
upload, the preview, submission and close - and `inpaint-mask.js` (527,
loaded before it) its canvas: the tools, the stroke list and its replay,
and the export; `styles/inpaint.css` (139) styles both. The dialog reuses
`refine-prompt.js`'s check lines rather than copying them (`refineCutLine`
takes the box to select). The server side is `backend/api_masks.py` (131),
the upload and read routes, model-free, over storage's mask functions; the
inpaint runner stays in `refine.py` (333) and `DiffusersBackend` beside the
refine pass it extends, and the sweep in `gallery_delete.py`. The growth
lands in retained files: `storage.py` 1,498 (the `masks` table, its
writer, resolver and partials, the chunk), `generate.py` 1,255 (`--mask`
and the record's mask), `model_defaults.py` 1,319 (`inpaint_supported` per
profile, `INPAINT_DEFERRED`) and `index.html` 1,518 (the dialog's DOM).
None has gained a new role; their table rows below are as audited. The page
loads 46 classic scripts.

Added since, with no split (outpainting, `feat-outpainting`): `outpaint.js`
(994) owns the Outpaint dialog - the extend controls and frame, the
presets, the preview, submission, close and Reuse - and
`styles/outpaint.css` (99) its frame, the box and panes being
`inpaint.css`'s. What it shares with the Inpaint dialog was generalised in
place in `inpaint.js` (1,070) rather than copied: `syncPassSampling`,
`renderPassCheck`, `syncPassCheck`, `selectPassFrom`, `libraryStrengthInto`,
`sourceGoneKind`/`sourceGoneWords`, and `inpaintNone(outpainting)`. The page
loads 47 classic scripts. On the server, `backend/outpaint.py`
(298) is a leaf - numpy, Pillow and `sizes` only - holding an outpaint's
pure parts: the padded source and its frozen fills, the built mask, the
kept rectangle and `band_too_wide`'s words, and the presets' integer
arithmetic; the stamp that stores a mask is `storage.stamp_outpaints`, the
submit checks (`outpaint_unusable`, `mismatched_outpaints`) are
`submission.py`'s, and the presets route is `api_masks.py`'s.

How the boundaries hold:

- The new route modules take one `AppServices` built by `main` over its own
  globals with late-bound getters. They hold no second worker, load state or
  settings snapshot; `main.ensure_loaded()`, its lock and the generation gate
  keep their ownership and ordering. Prompt routes call `detached.detached`
  directly, so a test patching `main._detached` would not reach them; nothing
  does today.
- Where tests replace a module's `settings` wholesale (`storage`), the new
  modules import the old one and never the reverse, and moved names stay
  reachable from the old owner (`storage.__getattr__`, the `schemas` facade,
  `prompt_encoding`'s re-exports). Tests that patch an implementation detail
  target its new owner; `bundles` keeps unqualified from-imports so existing
  patches on `bundles._verify_archive`, `_install` and `_replace_database`
  still intercept.
- Test families import from small helper modules (`prompt_encoding_helpers`,
  `bundle_helpers`), never from one another, and re-collect nothing through
  wildcard imports.
- The browser still runs classic scripts sharing one global scope: 43 at
  this pass, 44 since `png-settings.js`, each loaded exactly once, `app.js`
  last. No binding was wrapped in a module or
  closure; each moved block keeps its comment and source order.

## Long files retained, 2026-10-11

| File | Lines | Audit decision |
| --- | ---: | --- |
| `backend/guides.py` | 1,455 | Data only: 22 guide dicts and the private pieces each family shares, with no functions or state. Grown from 1,250 by new checkpoints (RouWei 0.8, Anima Aesthetic v1.1, NoobAI-XL V-pred) and sourced examples, not by a new role. Guides are compared side by side and share one schema and cite convention. A split by family would need explicit re-exports of every `*_GUIDE` (`prompt_sources.guide_constants()` and the tests' `_all_guides()` enumerate the module) and of private names tests read (`_NOOBAI_PREFIX`, `_BFL_TECHNICAL`). If ever done, keep `guides.py` as the facade with explicit imports, never a star import. |
| `backend/prompt_sources.py` | 1,358 | What the critic may cite: about 1,130 lines (83%) are per-checkpoint author excerpts and `GROUNDING`, beside about 170 lines of types, sentence splitting and resolution. Growth from 1,206 is new groundings; `author_orders` reads `GROUNDING`, so it belongs with it. Moving the resolver out would leave a 1,190-line data file, split the docstring that states the list's rules, and add an edge to the lazy `model_defaults` <-> `prompt_sources` cycle. Splitting the data by family would separate groundings that borrow from each other. |
| `web/index.html` | 1,340 | The page's static DOM, no code. `state.js` builds `els` with about 145 lookups at parse time, so every dialog must be inline before the first script; a fragment split needs a build step, a server-side template or async fetches that race `app.js`. Grown from 1,015 by the DOM of features since added (Swap tags, Mass refine, More seeds, the bin, the CFG comparison, the Models table, bundles), each next to its trigger or as its own `<dialog>`; about 31% is comments recording why. `tests/test_profiles.py` reads literal markup from it. |
| `backend/model_defaults.py` | 1,256 | Each checkpoint's control profile (about 840 lines of data in first-match order) and the views served from it (`defaults_for`, `guide_for`, `hints_for`, `strength_sweep`); the docstring is the schema of both. Grown from 1,135 by three profiles and `_served_tag_helper`. A data/views split would halve one role, and tests and `prompt_order`/`prompt_sources` import `_PROFILES`, `_FALLBACK` and `_profile_for` and assert identity. Revisit when a profile key gains per-checkpoint logic. |
| `backend/storage.py` | 1,170 | Split this pass (from 1,866). What remains is the core that tests patch through `storage.settings`: schema and migrations, connections and the write lock, record conversion, writers, thumbnails, partial files and live reads. The thumbnail, partial and staging helpers are called from `save_image`, `_row_to_record` and `init_db`, so they cannot move to a module storage imports. |
| `backend/generate.py` | 1,089 | Grown 7 lines between the audits (`include_binned=True` for the bin) and 1 in this pass (reaching `gallery_reports` through `_project()`). A parser/service seam is identified: `_build_parser`, `_image_settings`, `_apply_image_settings`, `_image_variables` and `_resolve_size` (about 350 lines, pure over argparse and the PNG's metadata, never importing config) could become `generate_args.py`. Not taken: the old trigger, CLI work growing, has not happened, and `main()` stays about 650 lines either way because its report and run loop share 15+ locals. `tests/test_profiles.py` asserts `generate.SIZES is sizes.SIZES`, which a split must retarget. |
| `backend/prompt_rules.py` | 1,064 | What the 2026-10-07 split left (context, `check()` dispatch, the rules and their `_RULES` order); grown 4 lines since. Every rule has the same shape over one shared `_Context`; splitting guide, tag and heuristic rules would scatter one dispatch table. A `prompt_heuristics.py` of about 65 lines is the one clean seam, not worth a file yet. |

No split planned for this pass failed. The 2026-10-07 retains of `main.py`,
`storage.py`, `prompt_encoding.py`, `schemas.py`, `lightbox.js` and
`test_prompt_encoding.py` are superseded by the splits above.

## Verification

Run on `2659e36` plus this documentation change, after all eleven split
commits (`acf2013` .. `2659e36`).

- `uv run --group dev pytest -q`: **2,588 passed, 6 skipped**, 49 warnings,
  in 186 seconds; no GPU or weights. The first split's baseline at
  `d7c71db` was 2,590 passed, 4 skipped. The two extra skips are
  `test_critic_tokens.py:115` (Qwen3-4B's tokenizer) and the FLUX.2 klein
  tokenizer check now in `test_prompt_encoding.py:277`, each "not in the Hub
  cache"; both skip the same way on a `d7c71db` worktree today, so the
  environment changed, not the code. Every split reported the same counts
  before and after, and the split test families collect the same node ids
  (157 for `test_prompt_encoding*`, 89 for the bundle families).
- AST comparison per split: every top-level definition of each original file
  exists exactly once across its successors (62 moved from `main.py`, 85
  from `storage.py`, 61 from `schemas.py`, 58 from `bundles.py`, 100 from
  `prompt_encoding.py`, 85 and 58 from the two test files). Bodies match
  after normalising module qualifiers and the injected `services`; the
  differences that remain are the planned ones (`model_id` passed into
  `chip_verdicts`/`order_slots`, private-to-public renames, docstrings that
  named a renamed symbol).
- Sorted OpenAPI output from `backend.main.app.openapi()` is byte-identical
  before and after the `main.py`, `schemas.py` and `storage.py` splits.
- Every `web/*.js` passes `node --check`; the 43 scripts concatenated in
  `index.html` order parse, so no top-level name is declared twice; the
  scripts `index.html` loads are exactly the 43 files in `web/`, each once,
  `app.js` last. Each JavaScript split was checked to rebuild the original
  statements in order.
- **Browser smokes: 47 of 49 pass.** Each ran against its own fresh fake
  server on the port and with the `--model`/`--step` flags of the table in
  `CLAUDE.md`, with a fresh scratch `--data` gallery, its own browser profile
  and, for ports shared by several smokes, a new server per smoke; every
  server was stopped afterwards. `smoke-compare-cfg.json` fails 3
  expectations (the "CFG 6" option in Against, `lbCompare.source` being the
  CFG 6 image, and the "1 of 2 CFG values" note) identically on a `d7c71db`
  worktree, so it predates the splits and needs its own investigation.
  `smoke-jobs.json` is intermittent: it passed 2 of 6 runs here and 0 of 3 on
  `d7c71db`, failing the same two ways on both (the cancelled-load note
  "Cancelled — the weights keep loading..." not appearing in time, or tab
  B's "a tractor" job not landing within 40 s, which cascades through the
  rest of the script), so that is pre-existing timing, not a split.
- `cmp CLAUDE.md AGENTS.md`: identical. `git diff --check`: clean. References
  in `CLAUDE.md`, `docs/rules.md` and `docs/design-notes.md` to a moved
  function name its new owner; historical reviews and `docs/verified.md` are
  left as written.

These checks establish refactor equivalence at the source, API and
fake-backend levels. This pass adds no GPU quality or performance
measurements to `docs/verified.md`.

## Third pass, 2026-10-31

Screened the 543 tracked Python, JavaScript, CSS and HTML files in
`backend/`, `web/`, `tests/` and `tools/` at `be46506`, then worked on the
`chore/code-hygiene` branch. 26 files were over 1,000 lines (22 application,
4 test), up from 8; 18 had no row here, and three split successors had grown
back past their pre-split size (`main.py` 856 to 1,181, `bundles.py` 590 to
1,191, `model-switch.js` 912 to 1,236). The method is the second pass's: map
each file by role first, and split only a role other files call or one that
shares no state with the rest, moving code verbatim.

After the pass 18 files are over 1,000 lines: 17 application files and
`tests/test_profiles.py`. Each has a row in the table below. Figures are
`wc -l` after the pass.

What the pass did, by package (each split is a 2026-10-31 row in
[Implemented boundaries](#implemented-boundaries)):

- **Storage and the bundle layer.** The bundle layer no longer imports the
  `reindex` CLI: the PNG chunk codec and the recipe reconciliation moved to
  `metadata.py` (143 to 192), `_DB_REQUIRED` to `bundle_common`, and
  `reindex` keeps aliases its tests read. Retained run jobs went to
  `run_jobs.py` (162) and the pure field codecs to `record_fields.py` (207),
  taking `storage.py` from 2,112 to 1,808. The migration list is hoisted to
  `storage._ADDED_COLUMNS`, from which a bundle merge takes its column
  types (held to `_SCHEMA` by `tests/test_storage_columns.py`), and
  `tests/test_chunk_builders.py` holds the three PNG chunk builders to one
  key set. Six call sites stopped reaching through `storage._crop_anchor`.
- **The CLI.** `generate.py` went from 2,084 to 1,262: `generate_args.py`
  (591) the parser, `--from-image` settings and the flag-only checks,
  `generate_remote.py` (284) the `--via-server` client, and
  `submission_rules.py` (202) the service-free submit checks, so the CLI no
  longer imports FastAPI. The two paths share one `_grid_flags`, and the
  local run now refuses `--grid-seeds 0` as `--via-server` did.
  `generate.main` is still 921 lines: its report and run loop share the
  locals the 2026-10-11 row named, and extracting the report as a
  many-parameter function would move the coupling, not remove it.
- **Smaller backend splits.** `job_state.py` and `error_text.py` from
  `jobs.py`; `api_rerender.py`, `api_runs.py` and `more_seeds_planner.py`
  from `api_generate.py` and `api_more_seeds.py`, so no router imports
  another; `schemas_outpaint.py`; `critic_issues.py`; `bundle_manager.py`
  and `schemas_bundles.py` from the bundle router; and `bounded_route.py`
  (60), the import routers' one bounded-body route class. The rerun planners
  share `submission.prune_held`, `input_spec` and `status_counts`, and seven
  routes build their answer with `JobAccepted.of`.
- **The page.** Verbatim moves into `automatic-backup.js`, `sheet-zoom.js`,
  `refine-rules.js`, `tag-helper.js`, `prompt-note.js`, `tag-lists.js`,
  `gallery-head.js` and `page-helpers.js`, `gallery.js`'s tail into
  `reuse.js` and `facts.js`, and the pass-check lines into
  `refine-prompt.js`; and one shared-code file, `import-merge.js` (170), the
  snippet and setup import dialogs' merge machinery. The page loads 67
  classic scripts, up from 58. Duplicates removed in place: one
  `applyGalleryFilter` for the four filter transitions, `.sr-only` for the
  screen-reader lines outside the stage, one result row for Swap tags, Mass
  refine and More seeds (`page-helpers.js`), and `dropPassLora`,
  `stepStrength` and `PASS_IN_FLIGHT` in `refine-rules.js`.
- **Tests.** `test_cli.py` (1,854), `test_api_grid.py` (1,298) and
  `test_api_model_switch.py` (1,044) split, with their node-id counts equal
  before and after; shared helpers moved to `cli_helpers.py`,
  `backend_helpers.py`, `gallery_helpers.py`, `browser_helpers.py` and
  `model_switch_helpers.py`, imported by name; the version-1 backup policy
  writer, which no app code called, to `backup_helpers.py`; and the three
  converter caches' copied tests became one parametrized file,
  `test_converter_cache.py`.

### Long files retained, 2026-10-31

| File | Lines (last row or base, then now) | Audit decision |
| --- | ---: | --- |
| `web/index.html` | 1,340; 2,135 | The page's static DOM, still no code: no inline script, 18 `<dialog>`s, 67 scripts. `state.js` now builds `els` with 593 lookups at parse time, so the reason a split needs a build step, a server template or fetches that race `app.js` is stronger than before. Growth is the DOM of Inpaint, Outpaint, Edit by instruction, Then refine, the automatic backup settings, Shortcuts, Saved searches, Snippets and Setups and their imports, each beside its trigger or in its own dialog. About 25% of the lines are comments. |
| `backend/backends/diffusers_backend.py` | 862 (`2659e36`); 1,976 | `DiffusersBackend` (1,605 lines) and the private pipeline-ownership classes (`_PipelineCandidate`, `_CacheDisposal`, `_CallDisposal`, about 270 lines). The classes use no module global tests patch, so a move is mechanically safe, but they are the backend's own lock protocol, touching `_lock`, `_cache_owner`, `_cache_disposal`, `_pending_cache_disposal`, `_loaded_execution` and `_pipe` 30+ times; a split would spread one protocol over two files. If ever split, move them verbatim and import them back so `type(x) is _CacheDisposal` still resolves. |
| `backend/storage.py` | 1,170; 1,808 | Split this pass (from 2,112). What remains is the 2026-10-11 core, grown by the masks table, its writers and holds, the outpaint stamp and the edit columns, which tests patch on `storage`. |
| `backend/guides.py` | 1,455; 1,735 | Still one role: 25 guides (22 at the last row) and two pure import-time builders for the Anima dicts (`_anima_helper`, `_anima_order`), so "no functions" no longer holds but the data-only argument does. A family split would need re-exports of 25 `*_GUIDE` names and of the private names read from outside: `_NOOBAI_PREFIX`, `_BFL_TECHNICAL` and now `_KONTEXT_DRIFT` (`prompt_sources` and two instruction-edit tests). |
| `backend/model_defaults.py` | 1,256; 1,554 | Growth is inpaint keys and views (`inpaint_family`, `inpaint_start`, `INPAINT_DEFERRED`, the measured labels) and the Kontext exact profile, not per-checkpoint logic. An `inpaint_profile.py` would need `_profile_for` from here while `hints_for` needs `_inpaint_strength_hint` from it: a new cycle to move about 100 lines. The 2026-10-11 reasons and revisit trigger stand. |
| `backend/prompt_sources.py` | 1,358; 1,469 | Growth is the Anima Preview and Kontext groundings; about 85% is still excerpts and `GROUNDING`, which the resolver reads directly. The 2026-10-11 reasons stand. |
| `backend/jobs.py` | 910 (`2659e36`); 1,319 | Split this pass (from 1,711). `JobManager` (1,230 lines) stays whole: Then refine's stage calls `_emit`, `_save`, `_hold` and the backend, and `submit_cells`/`_generate` close over the job and loop. |
| `backend/generate.py` | 1,089; 1,262 | Split this pass (from 2,084); the 2026-10-11 trigger, CLI work growing, had fired. `main()` is 921 lines of the rest. |
| `web/model-switch.js` | 912; 1,236 | The Models table, favourites, selection and the switch, kept together as the 2026-10-11 row put them. The table (about 370 lines) is called from `execution.js`, `model-shots.js` and `app.js`, but it reads the switch's catalogue state (`modelCatalogueEntries`, `modelCatalogueDetails`, `modelRuntimeSettings`, `modelState`) and calls back into `syncModelSwitch`, so a `model-table.js` would share most of its state with this file. |
| `backend/bundles.py` | 590; 1,193 | Export and import, the two operations tests patch through (`bundles._verify_archive`, `_install`, `_replace_database`, `bundles.datetime`), `_import_bundle` about 350 lines and `export_bundle` about 265. Moving the five import-only recovery helpers to a `bundle_recovery.py`, and `_read_whole`/`_ReadAhead` to `bundle_archive.py`, is safe and optional, not taken; revisit if import grows again. |
| `backend/grid.py` | 951 (`2659e36`); 1,182 | Every role works on the one `Cell`/`CellKey` vocabulary AGENTS.md anchors here (`expand`, `cell_from_record`, `run_id`), and tests patch `grid.settings` and `grid.random_seed`, which both `expand` and the prompt-variable code read. Seams recorded: a `grid_prompts.py` of about 220 lines (variables, variants, random assignment), which would need `PLACEHOLDER`/`VARIANT` re-exported and would stop following `grid.settings` patches; and about 80 lines of pricing (`estimate_seconds`, `_cell_ms`, `seeds_within_budget`) beside `gallery_reports.execution_estimates`. Revisit if either grows. |
| `backend/main.py` | 856; 1,180 | Growth is the model routes, bound to module globals tests rebind (26 distinct names, among them `settings`, `_switching`, `details_for` and `_merged_defaults`), which `AppServices` reaches late-bound. Seam recorded, not taken: `_merged_defaults`, `_expander_status`, `_listed_loras`, `_disk_free_mb` and `_gallery_timing` (about 130 lines) as a `model_views.py`, imported back under main's names so the patches still intercept. Revisit if `/api/models` grows again. |
| `backend/reindex.py` | 584 (`2659e36`); 1,110 | The in-place reindex and `merge_database` share the PNG reading, the mask and edit checks and about 50 patch sites (`reindex.settings` 44 times, `_copy_in`, `_Masks.resolve`). This pass moved out what the bundle layer read, so only `bundles.py`'s one lazy `_embedded_mask`, which needs storage's mask hash, still imports the CLI. |
| `web/refine.js` | 1,018; 1,080 | Split this pass (from 1,178): the 2026-10-11 retain's reason, four callers of the rules seam, no longer held. The dialog's state machine around its shared lets remains. |
| `tests/test_profiles.py` | 936 (`d7c71db`); 1,080 | One role, profile and size facts held to the page, the CLI, the README and verified.md, about 8% over the screen. Seam recorded: the size-table tests (`test_the_page_writes_the_cfg_label_on_both_cfg_boxes` through `test_the_batch_budget_names_what_to_use_instead`, about 177 lines) could join `tests/test_sizes.py` if the file passes about 1,200 lines or gains another size-table check; rules.md, design-notes and the 2026-10-11 rows cite this file for them, so retarget those then. |
| `backend/prompt_rules.py` | 1,064; 1,067 | Unchanged in kind: one `_Context`, one `_RULES` table of 26 rules, and the heuristics seam still about 67 lines. |
| `web/gallery-actions.js` | 826 (`2659e36`); 1,029 | Split this pass (from 1,180); the ask dialogs stayed, an optional move not taken. |
| `web/inpaint.js` | 1,070; 1,004 | Trimmed this pass (from 1,083) by the moves to `refine-prompt.js` and `refine-rules.js`. Still the Inpaint dialog's one role; Outpaint shares its capability helpers (`inpaintSupported`, `syncInpaintCapability`, `inpaintNone`). |

Screened and retained under 1,000 lines, without a row of their own:
`web/outpaint.js` 989 (its extend controls are its own role but have no
outside caller, and it shrank through the duplicate removals),
`backend/prompt_text.py` 933 (negation and unquoted lettering, about 580
lines, depend only on `_QUOTE_OPEN`, `_quoted_spans` and `_inside`, and
would become `prompt_lettering.py` past 1,000), `web/sheet-render.js` 929
(one role, drawing and navigating the sheet; lifting the controls bar out of
`drawSheet` as `drawSheetControls(values)` is optional), `web/lightbox.js`
909 (one state machine; download is the only new role, about 35 lines),
`web/app.js` 902 (boot and wiring, its documented role; the document
keydown dispatcher could become a `documentKeydown` in `shortcuts.js`, with
`test_prompt_route.py` and five rules.md mentions retargeted) and
`web/status.js` 867 (one `/api/status` role; `receiveStatus` is its one
dispatcher).

Not taken, by decision: renaming underscore-private names read across
modules (a project-wide convention, and tests bind and patch many of them),
one atomic-write helper for every sidecar, a shared converted-tree cache in
the loaders, and splitting `outpaint.js`, `sheet-render.js`, `lightbox.js`,
`status.js` or `app.js` by size alone.

### Verification, 2026-10-31

Each split commit records its own checks in its message: `pytest
--collect-only` node-id counts equal before and after each test split (122,
66 and 54; the converter caches went from 70 to 71, one case added on
purpose); the sorted OpenAPI document byte-identical across the router and
schema moves; computed styles compared in a browser before and after the
CSS changes; and the classic-script guard and the docs-map tests for every
page move. At the end of the pass, on Windows with this documentation
change: `uv run --group dev pytest -q -n auto` **8,141 passed, 7 skipped**
in 119 seconds, `ruff check .` clean, and the pinned type gate 0 errors over
166 files; every relative link and anchor in the tracked Markdown resolves.
The page splits ran the whole smoke table (100 of 100 passed), and the page's
duplicate removals the 20 smokes over what they touched. As before, these
establish equivalence at the source, API and fake-backend levels, and add no
GPU measurement to `docs/verified.md`.
