# Source-file audit, 2026-10-11

Audited the tracked Python, JavaScript, CSS and HTML in `backend/`, `web/`,
`tests/` and `tools/` at `d7c71db` (200 files). Excluded dependencies,
generated assets, review prototypes and historical documentation. Of these
files, 18 exceeded 1,000 lines: 16 application source files and two test
modules. Line count was a screening tool: each file was mapped by
responsibility first, and a file was split only where a role had grown that
other files call or that shares no state with the rest. Splits move code
verbatim, with no new build step or runtime feature. This is the second pass;
the first, at `21afcdf` on 2026-10-07, is kept below and marked as such.

After this pass 8 files are above 1,000 lines; the table of retained files
says why each stays whole.

## Implemented boundaries

| Pass | Original file | Lines before | Responsibilities after the split |
| --- | --- | ---: | --- |
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
`gallery-batch.js` a card's batch on hover; `lightbox-cfg.js` (416) the
lightbox's comparison across CFG. `tag-swap.js` + `styles/swap.css`,
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
(932) owns the Outpaint dialog - the extend controls and frame, the
presets, the preview, submission, close and Reuse - and
`styles/outpaint.css` (99) its frame, the box and panes being
`inpaint.css`'s. What it shares with the Inpaint dialog was generalised in
place in `inpaint.js` (1,067) rather than copied: `syncPassSampling`,
`renderPassCheck`, `syncPassCheck`, `selectPassFrom`, `libraryStrengthInto`,
`sourceGoneKind`/`sourceGoneWords`, and `inpaintNone(outpainting)`. The page
loads 47 classic scripts.

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

## Long files retained

| File | Lines | Audit decision |
| --- | ---: | --- |
| `backend/guides.py` | 1,455 | Data only: 22 guide dicts and the private pieces each family shares, with no functions or state. Grown from 1,250 by new checkpoints (RouWei 0.8, Anima Aesthetic v1.1, NoobAI-XL V-pred) and sourced examples, not by a new role. Guides are compared side by side and share one schema and cite convention. A split by family would need explicit re-exports of every `*_GUIDE` (`prompt_sources.guide_constants()` and the tests' `_all_guides()` enumerate the module) and of private names tests read (`_NOOBAI_PREFIX`, `_BFL_TECHNICAL`). If ever done, keep `guides.py` as the facade with explicit imports, never a star import. |
| `backend/prompt_sources.py` | 1,358 | What the critic may cite: about 1,130 lines (83%) are per-checkpoint author excerpts and `GROUNDING`, beside about 170 lines of types, sentence splitting and resolution. Growth from 1,206 is new groundings; `author_orders` reads `GROUNDING`, so it belongs with it. Moving the resolver out would leave a 1,190-line data file, split the docstring that states the list's rules, and add an edge to the lazy `model_defaults` <-> `prompt_sources` cycle. Splitting the data by family would separate groundings that borrow from each other. |
| `web/index.html` | 1,340 | The page's static DOM, no code. `state.js` builds `els` with about 145 lookups at parse time, so every dialog must be inline before the first script; a fragment split needs a build step, a server-side template or async fetches that race `app.js`. Grown from 1,015 by the DOM of features since added (Swap tags, Mass refine, More seeds, the bin, the CFG comparison, the Models table, bundles), each next to its trigger or as its own `<dialog>`; about 31% is comments recording why. `tests/test_profiles.py` reads literal markup from it. |
| `backend/model_defaults.py` | 1,256 | Each checkpoint's control profile (about 840 lines of data in first-match order) and the views served from it (`defaults_for`, `guide_for`, `hints_for`, `strength_sweep`); the docstring is the schema of both. Grown from 1,135 by three profiles and `_served_tag_helper`. A data/views split would halve one role, and tests and `prompt_order`/`prompt_sources` import `_PROFILES`, `_FALLBACK` and `_profile_for` and assert identity. Revisit when a profile key gains per-checkpoint logic. |
| `backend/storage.py` | 1,170 | Split this pass (from 1,866). What remains is the core that tests patch through `storage.settings`: schema and migrations, connections and the write lock, record conversion, writers, thumbnails, partial files and live reads. The thumbnail, partial and staging helpers are called from `save_image`, `_row_to_record` and `init_db`, so they cannot move to a module storage imports. |
| `backend/generate.py` | 1,089 | Grown 7 lines between the audits (`include_binned=True` for the bin) and 1 in this pass (reaching `gallery_reports` through `_project()`). A parser/service seam is identified: `_build_parser`, `_image_settings`, `_apply_image_settings`, `_image_variables` and `_resolve_size` (about 350 lines, pure over argparse and the PNG's metadata, never importing config) could become `generate_args.py`. Not taken: the old trigger, CLI work growing, has not happened, and `main()` stays about 650 lines either way because its report and run loop share 15+ locals. `tests/test_profiles.py` asserts `generate.SIZES is sizes.SIZES`, which a split must retarget. |
| `backend/prompt_rules.py` | 1,064 | What the 2026-10-07 split left (context, `check()` dispatch, the rules and their `_RULES` order); grown 4 lines since. Every rule has the same shape over one shared `_Context`; splitting guide, tag and heuristic rules would scatter one dispatch table. A `prompt_heuristics.py` of about 65 lines is the one clean seam, not worth a file yet. |
| `web/refine.js` | 1,018 | The Refine dialog's state machine around shared lets. A `refine-rules.js` seam of about 85 lines is identified: the refine bounds, `refineScaled`, `refineFits`, `refineTight`, `openingRefineSize`, `refineKept`, `refineSizeProblem`, `refinePrefs`, `REFINE_KEY`, `SINGLE_STRENGTH` and `refineDefaultStrength`, which read only `state.js`/`sizes.js`/`status.js` globals and which `refine-batch.js`, `lightbox.js`, `generation.js` and `panel.js` call. Not worth a file yet; `REFINE_MAX_SIDE`/`REFINE_MAX_PIXELS` must stay byte-identical for `tests/test_profiles.py` wherever they live. |

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
