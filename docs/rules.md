# Rules in full

The complete rule set, moved here from `AGENTS.md` on 2026-10-08 so it stays
short (`CLAUDE.md` only imports it). `AGENTS.md` keeps the rules that break things
silently; this file keeps every rule with its identifiers and exact wording,
the full module map, and what each smoke covers. `docs/design-notes.md` keeps
the reasoning, under headings of the same subject. Read the part for an area
before changing it.

## Layout

- `backend/config.py` — settings, most overridable by `AURADREAM_*` env vars
- `backend/legacy_names.py` — where files made under the old name (Local
  Diffusion) keep working: the default database, sidecars beside it and
  `HF_HOME` folders resolve to an existing `local-diffusion` one while no
  `auradream` one exists; it renames nothing
- `backend/launch.py` — the start-up menu and the `--model/--quant/--offload`
  flags, run before `config` is imported; the checkpoint catalogue lives here;
  `warn_legacy_env` names any `LOCAL_DIFFUSION_*` still set, which is not read
- `backend/loader_choices.py` — quantization, offload and Nunchaku format
  choices, worded once for the `--menu` and the browser picker
- `backend/model_selection.py` — the server's selected checkpoint, loader options
  and revision, separate from immutable startup settings; `current_model_id`
  and `current_settings` supply runtime reads, and `available` limits choices
  to the selected quantization/format/platform
- `backend/execution.py` - frozen named execution recipes, canonical empty/key
  contracts, composer/replay resolution, distribution metadata and pure compilation/attention/cache candidate
  matrices; every nonempty recipe is currently refused before execution
- `backend/compiler_cache.py` - private one-use startup binding for isolated
  compiler artifact directories, immutable process ownership and drift checks;
  Eager is cold and no public recipe is activated
- `backend/startup_preferences.py` — validated atomic loader/startup persistence;
  the server launcher restores it before config imports
- `backend/backup_policy.py` - pure automatic-backup policy, verified-success
  age and managed-retention predicates; no startup scheduling or archive I/O
- `backend/backup_retention.py` - explicitly consented post-success managed cleanup;
  durable pending-intent stops, protected copies and separate cleanup outcomes
- `backend/backup_credentials.py` - Windows generic credential adapter, exact
  opaque scope references and fixed provider errors; vault APIs are lazy
- `backend/failure_state.py` - traceback/cause/context-free ownership failure
  snapshots; process-lifetime quarantine never stores caller exceptions
- `tools/backup_credential.py` - explicit inactive/pending credential setup,
  local profile inspection and independent-password archive recovery verification
- `backend/backup_preferences.py` - strict bounded, atomic database-scoped
  backup policy sidecars; no lifecycle activation or credential fields
- `backend/backup_settings.py` - explicit v2 plaintext activation and durable
  gallery/destination scope; older draft policies never activate implicitly
- `backend/backup_receipts.py` - internal managed-export marker, outer-byte
  proof, strict atomic ownership journal and current-file identity checks;
  no startup, age advancement or pruning callers
- `backend/backup_coordinator.py` - explicitly invoked one-attempt
  due/export worker under bundle admission and gallery reservation; supplied
  policy, identities, clock and transient protection
- `backend/backup_service.py` - default-off startup/configuration/status
  lifecycle over the coordinator, with cached status during maintenance
- `backend/api_backup.py` - explicit backup configuration and cached status
  routes; `backend/schemas_backup.py` owns their validated models
- `backend/bundle_leases.py` - per-process archive reader/exclusive claims and
  leased FileResponse lifetime; shared process state and explicit native-lock mode
- `backend/bundle_file_lock.py` - opt-in nonblocking shared/exclusive native
  locks on persistent archive companions, with identity and cleanup ownership checks
- `backend/bundle_mode.py` - explicit Windows/NTFS shared-download installation,
  immutable destination marker and sticky discovery; no retention activation
- `backend/bundle_delete.py` - Windows same-handle managed archive
  verification/disposition, mandatory installed-mode exclusion and uncertainty ownership
- `backend/bundle_capacity.py` - conservative current-format managed ZIP,
  encryption framing and receipt-temp logical-byte budgets; no allocation guarantee
- `backend/snippets.py` - bounded, revisioned personal prompt snippets in an
  atomic sidecar beside the settings, under an independent cross-process lock
- `backend/snippet_portability.py` - cold bounded UTF-8 portable snippet codec,
  collection/input digests and pure file-index merge planning; strict portable
  documents and choices live in `backend/schemas_snippet_portability.py`
- `backend/api_snippets.py` - model-free snippet CRUD routes, independent of
  gallery maintenance; bounded export/merge preview/import use the existing
  snippet writer and independent reservation. `backend/schemas_snippets.py`
  owns CRUD models; portability request/response models live beside the codec
- `backend/saved_searches.py` - bounded atomic database-scoped gallery filter
  definitions, unique normalized names and optimistic item revisions under an
  independent lock; `backend/schemas_saved_searches.py` owns strict fields and
  `backend/api_saved_searches.py` model-free CRUD independent of maintenance
- `backend/presets.py` - bounded, revisioned named generation/grid drafts in
  their own atomic sidecar; `backend/schemas_presets.py` owns the strict raw
  payload and API capture types
- `backend/preset_validation.py` - cold Generate/Grid validation and effective
  summaries through ordinary schemas, grid helpers and `applied`; disposable
  random-grid examples never supply saved seeds
- `backend/api_presets.py` - setup CRUD and explicit validation, separate from
  submission; capture/validation holds model selection, while reads/deletes
  ignore it and every route bypasses gallery maintenance
- `backend/model_info.py` — cold catalogue details from existing sourced guides,
  plus each entry's Hugging Face access from its catalogue gate word
- `backend/hub_access.py` — the Hugging Face access check a switch asks before
  it releases the working checkpoint: cache first, then one HEAD per recipe
  source at its revision (or the repo's auth check); a `Refusal` only when the
  Hub says no
- `backend/model_benchmarks.py` — retained RTX 5080 timing references, exact
  checkpoint/loading-option matching and coarse speed categories; no live benchmark
- `backend/schemas.py` — re-exports the pydantic request/response models;
  `settings` there is a binding for readers, so runtime patches target
  `schemas_generate`
- `backend/schemas_generate.py` — the generation request and its one
  normalisation `applied()`, grids and their preview, `JobAccepted`,
  `default_strength`, `effective_true_cfg`, `error_detail`; the leaf the
  other schema modules build on
- `backend/schemas_prompt.py` — Expand, Check (analysis, findings, orders),
  the chip count, the critic's critique, Danbooru tag suggestion/validation
- `backend/schemas_gallery.py` — `ImageRecord`, run summaries, gallery pages,
  Delete unstarred's preview, the recycle bin's bodies and answers, the
  picked cards' stars (`FavoritesBody`, `FavoritesResult`)
- `backend/schemas_rerun.py` — Re-render, Swap tags, Mass refine and More
  seeds: requests, statuses, results, previews, accepted answers
- `backend/schemas_model.py` — the model status poll, the prompting guide and
  its tag helper, critic/Expand status, the switch request
- `backend/model_defaults.py` — recommended steps/CFG/mu per checkpoint and
  the control tooltips; each profile hangs its prompting guide from
- `backend/guides.py` — the guides themselves, one dict per checkpoint: prompt
  and size prose, the documented megapixel range with its word, the authors'
  example prompts, and `analysis` - the prose's checkable statements as data
  for the rule checks, each citing the prose verbatim
- `backend/encoder_table.py` — what each checkpoint's pipeline feeds its text
  encoders: the template strings and truncation lengths copied from the
  installed pipelines, `TokenizerSource`/`EncoderSpec`, the per-family
  builders and the first-match lookup `encoders_for`; pure data, no tokenizer
- `backend/prompt_weighting.py` — literal A1111/ComfyUI weighting syntax no
  pipeline parses, found by regex in the typed text (`encoder.weighting`)
- `backend/prompt_encoding.py` — the prompt analyser's encoder layer: each
  checkpoint's own tokenizers, on the CPU, over the exact string its pipeline
  builds - token counts, where reading stops, template-tail loss, typed
  control tokens, T5's unknown characters; `measure` adds
  `prompt_weighting`'s findings and re-exports `encoder_table`'s names
- `backend/prompt_text.py` — shared prompt parsing, prose/tag detection and lettering spans
- `backend/prompt_parts.py` - contextual tag/prose passages beside unchanged
  dictionary verdicts, exact UTF-16 display spans, and the guard against
  reorders that split a comma-containing prose passage
- `backend/prompt_order.py` — documented tag placement, ordering and category coverage
- `backend/prompt_rules.py` — the analyser's rule checks against the guide's
  `analysis`: style, documented order, `reorder_tags`, the items left
  unplaced where that decides the order (`unplaced_items`), each comma
  item's step for the chips (`order_slots`, answering only where
  `places_items` - `reads_order` but a lone item too; these and the order rule itself place through one
  `_places`) and what a critic's reorder is held to
  (`places_every_item`, `order_items`, `order_finding`, `scene_ahead`),
  what a check asks the tag index about (`lookup_items`) and whether a
  rewrite's artists were all read (`artists_read`), negations, unquoted
  text to render, the tag checkpoints' specifics; `heuristic` ones labelled
- `backend/prompt_sources.py` — what the critic may cite, per guide constant:
  the guide's own sentences (G) and short verbatim author excerpts (A) with
  their URLs, notes and caveats
- `backend/critic.py` — the local prompt critic: pinned llama.cpp
  `llama-server` and Qwen3-4B GGUF, fetched and verified by `prepare()`, one
  server per `critique()`, the answer validated against its sources;
  `critic_runtime.py` owns pinned installation, `critic_server.py` the process
  tree and loopback requests, `critic_context.py` the question and token budget,
  `critic_reorder.py` pure prompt permutations, and `critic_validation.py`
  answer grounding and rewrite validation
- `backend/translation.py` — lazy CPU-only Marian translation of ERNIE's
  expanded Chinese prompt for the UI; the Chinese prompt remains canonical.
  One sentence per sequence (`_chunks`), never packed: Marian drops most of
  a packed one silently; only a sentence past the token limit is cut.
  `cached()` is a file check (a local tree or the Hub cache, no network) for
  the status's first-use line. The load sets `DISABLE_SAFETENSORS_CONVERSION`
  for itself alone (`_without_conversion`, the previous value put back),
  since Transformers would otherwise fetch the conversion bot's safetensors
  copy of the `.bin` it loaded, a second 312 MB
- `backend/tags.py` — cached Danbooru tag suggestions, their order
  (`rank_suggestions`: exact match, the query at the start of any word -
  "hair" finds `short hair` - then inside a word or by alias; post count
  within each, 20 at most, the index asked for `*query*`; a guide tag ahead once half of it is typed;
  a tag of the recommended negative never first unless typed whole), Find
  tag's wiki search (`describe`, `wiki_definition`) and exact
  validation with a reason `code` for tag-caption checkpoints; remote failure
  means unchecked, never invalid - except an item longer than any tag
  (`TAG_NAME_MAX`, 200), answered unknown without the index, reachable or
  not
- `backend/seeds.py` — the seed range and the wrap, shared by server and CLI
- `backend/loras.py` — the adapter directory: what is in it, and the file
  behind a name (the only place a request's name becomes a path)
- `backend/refine.py` — the refine pass: the image-to-image class diffusers
  ships beside a pipeline's own (`sibling`), that class built over the loaded
  components (`build`), the steps a strength leaves, and the source scaled to
  cover the requested size and cropped; a leaf module like `seeds`
- `backend/instruction_edit.py` — standalone `kontext-v1` recipe, strict
  request/stored-source contracts, original-RGB provenance and exact-aspect
  Lanczos geometry, canonical bound identity, the held-reason sentences and
  the execution holds that open only for `qualified=True`
  (`edit_unavailable_reason`); this module does not enable an editing checkpoint.
- `backend/instruction_edit_admission.py` — prove live gallery source pixels,
  descriptor and exact recipe geometry before key/admission work
  (`bind_source`, `bind_request`); `load_bound_source` proves and decodes
  through one open handle and returns the prepared working-size RGB; no loading.
- `backend/api_instruction_edit.py` — the cold single-source preview and
  `POST /api/instruction-edit`, the one route that queues an edit, both behind
  the backend's `instruction_edit_qualified()` and the selected checkpoint.
- `backend/instruction_edit_preflight.py` — cold, supplied-config Kontext
  inventory/class/shape checks and canonical config digests;
  `QUALIFIED_CONFIG_HASHES` (the six configs at the revision, qualified
  2026-10-27; a config pin, not a weights pin) and `require_pinned`; performs
  no file/Hub access and never grants execution.
- `backend/outpaint.py` — an outpaint's pure parts: the fills (`FILLS`,
  `padded`), the server-built mask (`mask`), the presets (`presets`, the
  refine guard's limits passed in) and the off-grid words
  (`source_off_grid`); numpy, Pillow and `sizes` only. Each fill and the
  mask are frozen, their bytes pinned by `tests/test_outpaint.py`
- `backend/sizes.py` — the preset table; the CLI's `--aspect/--mp`, config's
  `max_side`/`max_pixels` and a refine pass's `refine_max_side`/
  `refine_max_pixels` (any 2 MP preset at 2x), `web/state.js`'s copy,
  `web/refine.js`'s two refine bounds, `index.html`'s `max=` and the README
  table are all held to it by `tests/test_profiles.py`
- `backend/backends/base.py` — the `GenerationBackend` ABC
- `backend/backends/diffusers_backend.py` — in-process diffusers implementation
- `backend/backends/regional_compile.py` - exact-region discovery, private lazy
  registration, candidate disposal, owned-call validation and scoped compiler
  policy; public nonempty execution remains unavailable
- `backend/backends/attention_selection.py` - private complete FLUX processor
  coverage, shared ownership, public-context configuration, restoration and
  read-only experimental Native environment/dispatcher policy;
  None is import-cold and public selection/admission remains closed
- `backend/backends/fbc_cache.py` - private exact FLUX cache-layout checks,
  per-call hook/state cleanup and shared-denoiser ownership/invalidation;
  library imports are lazy and render admission remains closed
- `backend/backends/loading.py` — everything decided before the weights
  move: the quantization config, the Nunchaku preflight and the
  prequantized refusal, the components left out of the download, the
  device placement, and the guard that keeps the adapter fit's meta-device
  build apart from a load (`weights_loading`, `empty_weights`), and the
  release of a quantizing load's bf16 files (`checkpoint_files_released`,
  `own_tensors`); takes `settings` as a parameter
- `backend/backends/single_file.py` — pinned, CPU-only first-load conversion
  for Illustrious XL v2.0-STABLE, WAI v17 and RouWei 0.8 epsilon; atomic cached
  trees feed the normal loader; `sweep_work`, which every converter calls first under its
  target's lock, removes interrupted conversions' work directories (exactly
  `<tree>-` plus tempfile's 8 characters, a real directory holding only
  `tree`), and each converter's work directory ignores its own cleanup
  errors so a failed conversion reports its own error
- `backend/backends/anima_loading.py` — pinned CPU conversion of full Anima
  Turbo, Aesthetic v1.1 and 2.9B Preview v1 and their conditioners, with official shared components
  and separate cached trees
- `backend/backends/anima_pipeline.py` — upstream Anima modular blocks behind
  the standard pipeline interface, with per-step progress and cancellation
- `backend/backends/netayume_loading.py` — pinned CPU conversion of NetaYume
  v4's split denoiser, with the official Lumina 2 parent's shared components
- `backend/backends/newbie_loading.py` — pinned CPU conversion of NewBie Exp0.1;
  `newbie_pipeline.py`, `newbie_transformer.py` and `newbie_text_encoder.py`
  provide its callback-capable pipeline, pooled-CLIP denoiser and native Jina tower
- `backend/backends/schedule.py` — how a requested mu becomes the whole
  shift: `forced_mu`, `shifted_sigmas`, the static-shift fallback, and
  `requested_shift`, which says which of the two a pipeline takes, for
  `generate()` and the Refine preview alike; and `start_sigma`, where a
  refine pass starts, read off a copy of the loaded scheduler, with
  `START_RECIPES` (each flow image-to-image and inpaint class's
  sequence-length and default-schedule statements) and `SHIFT_CALL`, held to the installed
  diffusers source by `tests/test_refine_schedules.py`
- `backend/grid.py` — axis specs (`6-12x2`), prompt variables and prompt
  variants, and the cell expansion, shared; `cell_from_record` /
  `cells_from_records`, a gallery record as a cell
- `backend/tag_swap.py` — Swap tags' rewrite of a record's prompt or
  negative prompt (pure; the routes are `api_swap.py`'s)
- `backend/refine_batch.py` — Mass refine's plan of one record: the pass
  it becomes or why not (pure; the routes are `api_refine_batch.py`'s)
- `backend/then_refine.py` — Then refine's pure parts: the strength
  resolved, a cell's stand-in pass for the submit's checks and prices, its
  refusals in words and the pass run id (the passes themselves are
  `refine_batch.plan`'s, built in the worker, `JobManager._passes`)
- `backend/more_seeds.py` — More seeds' batches of a selection, each one's
  next seeds, Batch and cells (pure; the routes are `api_more_seeds.py`'s)
- `backend/more_seeds_intents.py` - bounded preview-issued More seeds tokens,
  one owned admission per intent, replay receipts, expiry and shutdown drain
- `backend/latent_preview.py` — the latent-to-RGB maps the live preview
  draws with, one per VAE, fitted on the card
- `backend/jobs.py` — single-worker queue, SSE progress
- `backend/storage.py` — SQLite index + PNGs with embedded metadata
- `backend/metadata.py` — the PNG chunk's key (`auradream`) and
  `embedded_metadata`, which still reads the legacy `local-diffusion` and
  `krea` keys (and `auradream-mask`'s legacy `local-diffusion-mask`); shared
  by storage, `png_settings.py` (so `generate.py` and `api_png.py`) and
  `reindex.py`
- `backend/png_settings.py` — settings read back from a PNG's text chunks:
  this app's record (`record`, and `image_settings` for a file on disk,
  which `--from-image` reads through) and A1111's 'parameters' block
  (`parse_parameters`, A1111's own reading of it); pure, no config, Pillow
  only where a file is opened
- `backend/gallery_reports.py` — read-only aggregate queries: the preview's
  cost and size estimates, keep rates, runs (reads `storage` at call time)
- `backend/gallery_delete.py` — taking images out: the keep-favourites
  predicates, scoped delete and its preview, per-image delete, recycle bin,
  clear and VACUUM (under `storage._write_lock`, read at call time)
- `backend/gallery_lock.py` — reentrant reservation of complete gallery
  operations across threads and cooperating server/CLI processes
- `backend/bundles.py` — verified ZIP snapshots (of the whole gallery, or
  of chosen rows cut from a private copy: `_chosen`, `_restrict`), source
  cleanup, skipped for a backup, and staged database/image merges; `bundle_archive.py` the archive format, its one
  verifying read (`_verify_archive`), the decode pool, the staging writer and
  the free-space check; `bundle_images.py` a tracked PNG's pixel identity,
  row validation, embedded metadata and tEXt rewrite; `bundle_common.py` the
  bundle errors, progress type and read-only database reads every layer
  shares; `bundle_crypto.py` the optional password layer
  (streamed chunked AES-256-GCM, scrypt); `api_bundles.py` owns browser routes
  (the export's scope resolved through `submission.resolve_selection`)
  and exclusive maintenance admission through background completion
- `backend/search.py` — the search box's grammar: `key:value` terms as SQL
  predicates (`size:WxH` both sides exact; `lora:none` is IS NULL, every
  other text key a substring; an integer past SQLite's 64 bits a
  ValueError, so the routes' 422), the LIKE escape, `strict_scope`
- `backend/main.py` — app lifecycle and model ownership: load state and
  lock, `ensure_loaded`, the load-failure wording, model switching, status,
  the model and adapter routes, the middlewares, and the `services`
  (`app_context.AppServices`) every router factory reads main's state
  through; `api_gallery.py` owns gallery browsing, runs, stars, the recycle
  bin and deletion, and `api_jobs.py` job inspection, cancellation, pause and SSE
- `backend/app_context.py` — `AppServices`, late-bound getters over main's
  globals (settings, backend, jobs, critic, translator, `ensure_loaded`,
  the load lock, the merged defaults), so a test or the fake server that
  rebinds one on main is seen by every route; `model_scope` is the
  gallery's checkpoint scope
- `backend/detached.py` — `detached`, a blocking call on a daemon thread
  that interpreter exit never joins (the model load, the prompt routes)
- `backend/api_prompt.py` — the tag helper and prompt routes (suggest,
  validate, Expand, Check, measure, the critic) and their refusal wording;
  `backend/prompt_check.py` layer 1 of Check (encoder reports and rule
  findings, the order detail, the chips' order slots, the rewrite's lint),
  blocking and CPU only; `backend/tag_lookup.py` the Danbooru lookup the
  chips and Check share, and owner of the patchable `tag_vocabulary`
- `backend/submission.py` — what every cell producer checks before it
  queues (`refuse_unusable_inputs`, `unusable_input`, `refine_unavailable`,
  `inpaint_unavailable`, `mask_unusable`, `not_yet_rendered`) and the gallery selection (`resolve_selection`);
  `backend/api_generate.py` Generate, Run grid, its preview and Re-render;
  `api_swap.py`, `api_refine_batch.py`, `api_more_seeds.py` the Swap tags,
  Mass refine and More seeds plans, previews and submissions
- `backend/api_png.py` — Load settings from PNG (`POST /api/png/settings`):
  a PNG's text chunks, as the page read them, answered as the panel's
  settings after `applied()`, with the notes on what did not carry; the
  body capped at `MAX_BODY`
- `backend/api_masks.py` — an inpaint's mask: `POST /api/masks` (a PNG
  body over gallery image `source`, decoded under `refine_max_pixels`,
  read by `refine.canonical_mask` and kept as `L` by `storage.store_mask`,
  answered by its hash), `POST /api/masks/{hash}/keep` (`storage.touch_mask`:
  the settle window restarted, as a re-upload would; 404 where it has gone)
  and `GET /api/masks/{hash}` (through `storage.mask_path`); and an
  outpaint's presets, `GET /api/images/{id}/outpaint-presets`
  (`outpaint.presets` under the refine guard's limits, with `off_grid`,
  the fill, the band's default, its note (`outpaint.band_note`, naming the
  card only while the default is the width it judged) and bound and
  `min_kept`; a binned source a 404 that says it is in the recycle bin);
  model-free
- `backend/generate.py` — headless CLI; drives the backend directly, no server
- `backend/clear.py` — deletes rows, their PNGs and any orphans, the recycle
  bin's with the rest; `--bin` empties the bin alone, `--bin --older-than
  DAYS` only what was binned more than DAYS days ago
- `backend/stats.py` — the starred *rate* per setting (`gallery_reports.keep_rates`),
  scoped to a run by default; `--by variant` gives a comparison's rate
  per version
- `backend/reindex.py` — rows rebuilt from the PNGs' `auradream` chunk, in place or
  copied in from another gallery; idempotent on the UNIQUE filename
- `web/` - served as-is from `/static`; `index.html` loads 57 classic
  scripts in order, sharing one global scope; only `app.js`, the last, runs
  the wiring and boot. `state.js` owns shared state; `generation.js`,
  `grid-preview.js`, `job-stream.js` and `job-results.js` handle submission,
  previews and progress, with `job-progress.js` the watched job's progress,
  mirrors and Pause, and `job-queue.js` the queue and which job the page
  watches; `panel.js`, `runs.js` and `viewer.js` the settings,
  run history and results; `gallery.js`, `gallery-actions.js` and `reuse.js`
  the cards and their actions; `png-settings.js` Load settings from PNG
  (a dropped or picked PNG's text chunks read, sent and filled in through
  Reuse); `gallery-search.js` the lightbox's search
  buttons; `sheet.js`, `sheet-render.js` and
  `sheet-export.js` the contact sheet; `facts.js` a record's settings line
  and facts list (`settingsLine`, `factsList`), shared by the lightbox and
  the viewer; `lightbox.js` the lightbox's view, layout, walk and gestures,
  `lightbox-compare.js` its comparison's controls, divider and pass
  stepping, `lightbox-cfg.js` its comparison across CFG and across steps; `refine.js` and `refine-prompt.js` the Refine dialog; `inpaint.js` the Inpaint dialog (its mask's
  upload, preview, submit and Reuse) and `inpaint-mask.js` its canvas, tools, strokes and export
  (`styles/inpaint.css`); `outpaint.js` the Outpaint dialog (the side, band and preset
  controls, the frame, the preview, submit, close and Reuse; `styles/outpaint.css`);
  `instruction-edit.js` Edit by instruction… (the capability, the menu item, the
  dialog's preview, submit, close and Reuse, and an edit's lineage; `styles/instruction-edit.css`);
  `tags.js`, `tag-tokens.js` and `tag-chips.js` the prompt helper, `tag-finder.js` its Find
  tag dropdown; `prompt-examples.js` the
  Examples menu, `prompt-expand.js` ERNIE's Expand; `analysis.js`,
  `analysis-critic.js` and `comparison.js` the check and critic. `sizes.js`
  and `status.js` retain their control/profile roles; `model-switch.js` owns
  checkpoint selection and retiring state prepared under the old model;
  `execution.js` owns inspectable unavailable performance choices, composer
  cache intent and shared execution holds;
  `model-shots.js` the picker's strip of the chosen checkpoint's images;
  `bundles.js` owns the gallery bundle dialog (Back up and its scope
  included) and background-operation polling;
  `gallery-select.js` the picked cards and the Picked menu, `gallery-batch.js` a
  card's batch outlined on hover, `tag-swap.js` Swap tags,
  `refine-batch.js` Mass refine, `then-refine.js` Then refine's opener
  on the Notify row and its dialog, `more-seeds.js` More seeds and `bin.js` the
  recycle bin's view, undo note and Empty; `context-menu.js` the right-click
  menu on an image (`styles/context-menu.css`); `shortcuts.js` the keyboard
  help dialog; `prompt-snippets.js` the personal snippet dialog and insertion;
  `snippet-import.js` the explicit portable export and file-index merge dialog;
  `generation-presets.js` the named generation/grid setup dialog, raw capture,
  explicit validated Restore and whole-draft Undo.
  `saved-searches.js` owns named gallery filters, their native dialog and the
  single explicit filter transition out of sheets and the bin.
  The stylesheet is
  `style.css` (tokens and base layout), followed by `styles/*.css` in the
  explicit link order: `model-picker.css` the Models picker (first, at its
  old place in `style.css`), `fields.css` the panel's fields,
  `analysis.css` the prompt check, critic and comparison, `generation.css`
  the grid search, the buttons and progress, `gallery.css` the stage and
  Gallery beside, `sheet.css` the contact sheet, `lightbox.css` the
  lightbox, `refine.css` the Refine dialog and motion, `inpaint.css` and
  `outpaint.css` their dialogs, `instruction-edit.css` the Edit by
  instruction dialog and an edit's badge, `bundles.css` the bundle dialog,
  `swap.css`, `refine-batch.css` and `more-seeds.css` those dialogs,
  `bin.css` the recycle bin, `context-menu.css` the right-click menu and
  `shortcuts.css` the keyboard help dialog, `snippets.css` the snippet dialog
  and `setups.css` the named setup dialog, followed by `saved-searches.css`
  for the saved gallery search dialog and its footer opener;
  preserve that cascade (`tests/test_docs_maps.py` holds this list to
  `index.html`'s links, in order). `docs/source-audit.md` records the
  split audits and why each long file it kept stays whole.
- `tools/` — `fake_server.py` (the app on the tests' fake backend, port 7861)
  and `browser.mjs` (portable screenshot paths in `browser-output.mjs`) +
  `smoke.json` and a `smoke-*.json` per checkpoint or
  feature (headless browser over DevTools; among the features
  `smoke-stage.json`, Gallery beside, and `smoke-narrow.json`, one column
  and narrow windows): the UI's stand-in for a test suite, and
  `run_smoke.py`, which starts each smoke's fake from `AGENTS.md`'s port
  table and runs it, with `smoke_fonts.py` its Linux Segoe UI stand-in;
  `ci_changes.py` classifies GitHub push/PR ranges for
  the documentation-only lint path; `typecheck.py` runs the pinned,
  locally bundled Pyright checker over the configured scope; see Tests
- `tests/` — the pytest suite, no GPU and no weights: `conftest.py`,
  `fakes.py` and `api_helpers.py` (the fake backend and its fixtures),
  `test_api*.py` (the HTTP routes, one file per area), and a file per module
  or invariant beside them - `test_refine_schedules.py`, for one, holds a
  pass's start index and start sigma to the installed diffusers; see Tests
- `bin/` — `auradream` and `auradream-wsl`, for the user PATH:
  the two server starts (`backend.main`, `wsl/serve.sh`) from any folder,
  arguments passed through. Each is a `.ps1` (PowerShell picks it, so
  Ctrl-C stops the server without cmd.exe's "Terminate batch job (Y/N)?")
  beside a `.cmd` for cmd.exe, which always asks; keep each pair in step
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

- `KBlueLeaf/Kohaku-XL-Zeta` uses its ordinary Diffusers SDXL tree and
  supplied epsilon/Euler scheduler. Start at 28 steps/CFG 5 and disable Mu.
  Keep its hybrid prose/tag guide, 2024-07-10 vocabulary cutoff and
  nonexclusive rating buttons; its explicit caption uses both `nsfw` and
  `explicit`. The published tokenizers limit this pipeline to 77 tokens,
  despite the card's longer training context. The size guidance is around
  1 MP, not a promise of large-square output. No Nunchaku build is mapped.

- Illustrious XL v2.0-STABLE is single-file-only upstream. Resolve it through
  `single_file.resolve_model` before the normal component scan, keeping the
  official id in runtime settings and saved metadata. The converted tree has
  a fixed epsilon/Euler Ancestral scheduler; do not pass pipeline-level
  quantization into `from_single_file` or silently skip the CLIP encoders.

- `LyliaEngine/waiIllustriousSDXL_v170` selects WAI0731's v17 through a pinned
  community mirror, not an official repository or a LoRA. `single_file` checks
  the source SHA-256 against the author's archived release before converting
  on CPU, retaining the embedded VAE and both CLIP encoders. Use pinned SDXL
  descriptors only, epsilon/Euler Ancestral and normal FP8/int8/bfloat16 loading;
  preserve the public mirror id in settings and metadata. Keep the known NaNs
  in the unused final CLIP-L layer unchanged: SDXL consumes that tower's
  penultimate states and the other tower's pooled output. Preserve the regression
  checking the used conditioning. The exact-id profile starts at 28 steps/CFG 6,
  disables Mu and has no Nunchaku mapping. The author's reference sizes and
  separate hires pass are not a published maximum or direct-generation guarantee.

- `Laxhar/noobai-XL-Vpred-1.0` selects its pinned official Diffusers tree.
  `loading.resolve_model` reads configs before quantization checks and then
  only component weights/tokenizers, excluding the duplicate single file.
  Keep the public id in settings/metadata. The published scheduler incorrectly
  declares epsilon: `loading.component_overrides` installs Euler with
  `v_prediction`, zero-terminal-SNR rescaling and trailing timesteps (999
  first). Trailing is the app's Diffusers-backed integration choice; the
  author explicitly specifies the other two overrides. Use exact-id 28
  steps/CFG 5, disabled Mu and ordinary SDXL Refine at strength 0.4.
  No Nunchaku build is mapped. Preserve the author's quality/date vocabulary,
  approximate 2024-10-23 Danbooru cutoff, separate prefix and caption order,
  77-token CLIP limit and approximately 1 MP size advice.

- `Minthy/RouWei-0.8` selects the author's pinned epsilon single file through
  `single_file`, with its SHA-256 checked before CPU conversion. Preserve its
  VAE, both CLIP encoders and public id in settings/metadata. Use the exact-id
  28-step/CFG-6 profile, epsilon/Euler Ancestral, disabled Mu and the ordinary
  SDXL Refine pass (strength 0.4); no Nunchaku mapping. Its hybrid tag/prose
  guide uses the 2025-04-30 cutoff, `by ` artist prefixes and artist-last
  fallback. The author prefers separate CLIP chunks, but this app has only
  77 tokens per CLIP and no chunking or `BREAK` operator. The author's
  1-1.5 MP recommendation and separate upscale recipes are not a direct
  large-image guarantee. The same repository's V-pred file is a different
  checkpoint and must not replace this epsilon recipe.

- Z-Anime Base has a pinned nested `diffusers/` tree. `loading.resolve_model`
  resolves configs before quantization checks and weights after them; both
  passes use the same revision and keep the official id in settings/metadata.
  Diffusers 0.40 does not support pipeline-level `subfolder`. Keep the supplied
  FlowMatch Euler scheduler (shift 6); the card's ComfyUI Euler Ancestral/beta
  recommendation is a different setup, not a verified equivalent.
  Its bundled encoder has scaled FP8 tensors without quantization metadata.
  Skip those weights: `loading.component_overrides` loads the standard Qwen3
  encoder from pinned `Tongyi-MAI/Z-Image`, using the same Transformers-owned
  config from `quant_mapping`. Never silently discard the bundled scales.

- `circlestone-labs/Anima` selects the pinned full Turbo v1.1 split file.
  `anima_loading` converts its denoiser and conditioner on CPU, uses the pinned
  official Base repository only for shared components/configs, and caches an
  atomic tree for normal FP8/int8/bfloat16 loading. Keep the official id in
  settings/metadata. `AnimaTurboPipeline` wraps upstream modular blocks with
  the required callback; preserve its fixed flow schedule and exact-id profile
  (8 steps, CFG 1), with Mu, references and user LoRAs disabled. No Nunchaku
  build is mapped to this entry.

- `circlestone-labs/Anima@aesthetic-v1.1` is the app's distinct selector for
  the official Aesthetic v1.1 split file in `circlestone-labs/Anima`.
  `anima_loading` pins its own revision and cache, sharing the strict CPU
  conversion and official shared components with Turbo. Keep the selector
  in settings and saved metadata, but use the actual repository for Hub
  downloads and links. Never change the bare id's Turbo meaning or cache.
  `AnimaAestheticPipeline` shares the inference blocks with 30-step/CFG-4
  defaults and fixed FlowMatch Euler shift 3. Mu, references, user LoRAs,
  Refine and Nunchaku are unavailable. Its own guide and critic grounding
  omit Turbo's CFG-1/distillation claims and the generic `score_7` prefix:
  the authors discourage `score_*` in both Aesthetic prompts. Both guides
  state the card's tag order verbatim, which Check reads
  (`guides._anima_order`: Danbooru's meta category in General, since the
  card's full tag example writes `twitter username` there; an @ name in
  Artist; sentences anywhere), and serve a Tags menu of the card's named
  tags (`_anima_helper`, Danbooru vocabulary with no cutoff, `@` artist
  prefix); Turbo's adds the score tags and the recommended prefix,
  Aesthetic's places the score tags (`order.tags`) without offering them.
  The card's full tag example passes Check clean
  (`test_anima_cards_full_tag_example_passes_check_clean`).

- `Gazingstars123/Anima-2.9B` selects the pinned community
  `Anima-2.9B-preview-v1.safetensors`. Keep its separate conversion recipe/cache,
  `net.` checkpoint prefix and 40-layer transformer config; strictly load all
  denoiser and conditioner tensors, sharing only the pinned official Anima
  configs, Qwen3 encoder, Qwen-Image VAE and tokenizers. Preserve the public id
  in settings and metadata. `AnimaPreviewPipeline` uses the existing fixed
  FlowMatch Euler shift 3 integration with 30-step/CFG-4 defaults; this is not
  the author's ComfyUI sampler/scheduler recipe. Mu, references, user LoRAs,
  Refine, inpaint and Nunchaku are unavailable. Its own guide/critic grounding
  describes detailed mixed prompts and the preview's size advice without
  inheriting Turbo distillation or Aesthetic score prohibitions. Do not infer
  measured speed, memory or quality from the other Anima selections.

- `duongve/NetaYume-Lumina-Image-2.0` selects the pinned v4 split denoiser.
  `netayume_loading` strictly converts it on CPU and uses the pinned official
  Lumina 2 parent for configs, Gemma 2, tokenizers and FLUX VAE, excluding the
  parent's denoiser weights. Keep its public id in settings/metadata. Normal
  per-component quantization loads the atomic cached tree. Its fixed anime
  system instruction is passed explicitly per generation; keep FlowMatch
  Euler shift 6 and the exact-id profile (40 steps, CFG 5). The author's
  v2 family guidance and ComfyUI samplers are not v4 runtime equivalence.
  No Nunchaku build is mapped to this entry.

- `NewBie-AI/NewBie-image-Exp0.1` selects pinned official components, strictly
  converted on CPU by `newbie_loading`. Keep its public id in status/metadata.
  Diffusers 0.40 has no native NewBie pipeline; use the local classes, Gemma 3
  penultimate hidden states and Jina's unnormalized masked mean. Keep both
  prompts free of system prefixes, FlowMatch Euler shift 6, normalized reverse
  transformer time, and whole-sample CFG norm limiting. The exact-id profile
  starts at the card's 28 steps and the author's sample CFG 4. XML is text,
  never comma chips; 1024x1024 is an example, not a published size ceiling.
  References, user LoRAs and Nunchaku are unavailable for this integration.


- App-owned names use AuraDream: `AURADREAM_*`,
  `auradream.db`, `~/auradream`, `auradream.*` browser keys
  and the `auradream` PNG chunk. Keep Krea only in checkpoint/vendor
  names and explicit compatibility reads for pre-rename data. The
  project was Local Diffusion until 2026-10-27: its `local-diffusion` names
  are legacy - PNG chunks and the bundle format are read, never written; an
  existing database, sidecar or `HF_HOME` folder is used where it is
  (`legacy_names.py`); browser keys move once on first read - and only
  `AURADREAM_*` variables are read. An old name whose value feeds a hash, a key derivation or a store
  outside the app stays frozen, with a comment saying so (see "The
  project-owned namespace follows AuraDream" in `design-notes.md`).
- Quantize through `PipelineQuantizationConfig(quant_mapping=...)`, one
  `TorchAoConfig` per component from the library that owns it, each built from
  a torchao `AOBaseConfig` *instance*. The `quant_backend="torchao"` shorthand
  raises on a transformers text encoder.
- A quantizing `from_pretrained` runs inside
  `loading.checkpoint_files_released`: the tensors the load leaves
  unquantized are copied out of their safetensors shard (per shard for a
  diffusers model, per component for a transformers one), or each one keeps
  its whole bf16 shard mapped beside the quantized copy - about 46 GiB at
  FLUX.1's second transformer shard (`GEN#22`). A bf16 load keeps its
  mapping. `tests/test_quantized_load_ram.py` loads a sharded two-library
  pipeline and reads `/proc/self/maps`; it also asserts that an unreleased
  load still keeps its files, so a library that stops holding them is
  noticed.
- `torch.no_grad()`, never `torch.inference_mode()`, around the pipeline call:
  model offload moves components inside `forward()` and torchao's tensor
  subclasses trip torch 2.13's storage-aliasing guard under inference_mode
  (`test_the_pipeline_runs_under_no_grad_not_inference_mode`; the refine
  sibling's calls are held to it too).
- Loading happens only in `main.ensure_loaded()` under its asyncio lock; jobs
  await it, `generate()` raises on a missing pipeline, and
  `POST /api/model/unload` answers 409 while a job or a load is live; once
  `try_unload` succeeds, the route sets the state to `unloaded` before the
  translator, critic and tokenizers are let go. A load
  first waits out whatever holds the generation gate (`jobs.wait_for_gate`,
  inside the lock, without keeping the gate): a job queued during a critique
  must not move weights onto the card beside the critic's server. A failed
  load is mapped by `main._load_failure` to a sentence with the steps, with
  `ModelStatus.detail_kind` and `detail_url` beside `detail`, for
  huggingface_hub's `GatedRepoError`, `RepositoryNotFoundError`,
  `OfflineModeIsEnabled` and `LocalEntryNotFoundError` (by class name over
  the MRO and the `__cause__`/`__context__` chain, the first found except
  that `OfflineModeIsEnabled` wins wherever it is - huggingface_hub raises
  `LocalEntryNotFoundError` from it under `HF_HUB_OFFLINE`; the gate's kind
  from `launch.CHECKPOINTS`); anything else stays `short_error`, and the log
  keeps the library's words. It is raised on as `jobs.LoadFailed(detail)`,
  so a job that ended at the load and Expand (a 503 from its route) carry
  the status's sentence.
  `ModelStatus.started_at` is set once, at import.
  Status derives ready/unloaded from actual backend readiness without changing
  the load supervisor's cached state; loading/error and their details remain
  authoritative. An explicit Load also consults actual readiness, so an unsafe
  private pipeline disposal cannot leave a cached ready flag blocking reload.
  GET starts no load; retries still use the ordinary load lock and generation gate.
- `POST /api/model/switch` accepts only `GET /api/models` choices for the selected
  quantization/format; Nunchaku remains Linux-only. Checkpoint, quantization and
  offload form one immutable runtime snapshot. `load=false` applies and saves
  without loading changed weights; an unchanged selection keeps its weights.
  Its typed `execution` extension is never silently ignored: omission preserves
  the current loader intent, while explicit null/empty chooses Eager/native.
  `load=false` may save a structurally compatible unavailable recipe, without a
  Hub check or a load. Invalid placement/combinations and per-render step cache
  are refused as loader configuration. `load=true` and explicit Load refuse
  unavailable intent before Hub, save, unload or load-state effects; the runtime
  backend and queue also refuse an omitted/Eager request under that selection.
  No nonempty recipe is activated. Both actions atomically save loader/startup preferences beside the
  database before releasing existing weights; a save failure is an HTTP error
  and leaves the selection and existing preference file intact.
  A switch to another checkpoint first asks `hub_access.check` (through
  `main._refuse_hub_access`, on a daemon thread, at most
  `hub_access.CHECK_SECONDS`), after the busy refusals and before saving,
  reserving or unloading: a cached checkpoint asks nothing; a gated, 401/403,
  missing repo or missing pinned revision refuses with 403/404 and "<running>
  is still selected"; anything inconclusive (offline, `HF_HUB_OFFLINE`,
  timeout, 5xx, a missing file) goes on as before, never blocks. The check
  awaits, so the revision header, `_switching`/draining and
  `_refuse_busy_switch` are checked again after it. Loading options alone on
  the same checkpoint ask nothing. Tests never reach the Hub (`no_hub` stubs
  `hub_access._ask`), and neither does the fake server
  (`test_hub_access.py`).
  The server reserves `_switching` before any await and holds it through
  prompt-helper cleanup (critic, translator, tokenizers - before the weights, so
  one that will not stop leaves them loaded), unload and selection; it clears
  it under the load lock once `select()` has run and the state says loading.
  The load after it is an ordinary `ensure_loaded`: Check, tag lookup and
  LoRAs answer and a Generate queues behind it, as after Apply & save then
  Load, while a second switch is still refused by `_refuse_busy_switch` (the
  lock, then the loading state) and Unload by the lock
  (`test_switch_reserves_teardown_and_loading_against_new_model_work`). Any failure before `select()` restores the previous
  preference file (or removes the new one) and `load_on_startup`, leaves the
  load state alone and reports `switch_error`, never a load error of the old
  checkpoint; the next switch, load or unload clears it. The background task owns
  this even if the browser disconnects. Refuse queued/running/paused jobs, loads,
  card holders and in-flight model-dependent requests. The middleware counts
  those requests (by path, `_model_request_paths`) before Pydantic fills
  defaults and holds the selection until their work finishes; new requests
  during a switch, and during its wait for counted requests to drain, and
  stale browser revision headers receive 409. The switch waits up to
  `SWITCH_DRAIN_SECONDS` for counted requests already in flight, then
  rechecks jobs, loads and card holders without an await before reserving;
  a refusal after the wait names the counted paths
  (`test_switch_waits_out_a_model_request_already_in_flight`,
  `test_switch_names_a_model_request_that_outlasts_its_wait`). The page's `api()` sends its revision on every `/api/`
  request; the middleware alone lists the guarded routes and ignores the
  header elsewhere (`test_routes_that_read_no_checkpoint_ignore_the_revision_header`).
  Include scoped bulk deletion in this guard; the recycle bin's routes
  (bin, restore, the bin's listing, summary and Empty) read no checkpoint
  and stay out of it. Startup
  `Settings` stays frozen; runtime model reads use `current_model_id(settings)`
  and loader/quantization/offload reads use `current_settings(settings)`.
  Model revisions distinguish A-to-B-to-A switches and start from the
  process's start in milliseconds, so a page holding a previous process's
  revision is refused; status never pairs an old threaded description with the
  new revision. Every threaded read of `/api/status` (the description, the
  gallery's newest id) is inside its revision/`switching` retry loop, and
  nothing awaits between that loop and the answer: an await after it let a
  switch finish meanwhile, and the old checkpoint and revision went out with
  `switching` false, which the page took for a switch with no revision of its
  own to come - the next revision wiped the Models note's outcome
  (`test_status_retries_a_newest_id_read_that_crossed_a_switch`). LoRA denoiser caches must be
  invalidated when the selected checkpoint changes.
- Server launch restores `startup_preferences` before config imports, without
  terminal input unless `--menu` is given. Flags override explicit environment,
  then saved preferences, then defaults; `--defaults` skips preferences. The
  preferences are one choice: an explicit model (flag or environment, not env.sh's
  marked default) other than the saved one sets all of them aside, with a
  warning. The saved choice fills `--menu`'s marked defaults; a menu pick of
  another checkpoint than the saved one sets the saved `load_on_startup`
  aside too (recorded as `--menu`), so the pick never auto-loads under the
  other model's settings and the menu's printed command starts the same way
  (`test_a_menu_pick_of_another_model_sets_the_saved_auto_load_aside`).
  `launch.configure` records what was explicit in
  `AURADREAM_STARTUP_OVERRIDES`; status and `/api/models` report it as
  `startup_overrides` (empty when `backend.main` is imported). Malformed
  legacy preferences retain their existing fallback behavior. V2 stores the
  canonical loader recipe as well; invalid execution-bearing files/env values
  retain a fixed diagnostic and refuse loading/rendering until explicit repair.
  Save/rollback preserves exact prior bounded bytes, including an invalid file.
  Only model id, quantization, offload, `load_on_startup` and loader intent are
  saved, never secrets, per-render step cache or unrelated settings. Named
  `--compilation`/`--attention` and their `AURADREAM_COMPILATION`/
  `AURADREAM_ATTENTION` variables follow existing explicit override rules.
  The browser's
  compact model facts reuse sourced guides, catalogue data and recorded local
  evidence. The size fact shows upper guidance and labels recommendations,
  native/documented maxima and examples distinctly; an example never becomes
  a guaranteed maximum. Model favourites are browser preferences, canonical across Nunchaku
  weight formats, and never mutate the selected model or loader settings.
- The checkpoint is a radio table (`#model-table`) showing every model's
  facts; read and set it only through `modelChoice()` / `chooseModel()`, with
  no row checked when the runtime id is missing. It is rebuilt only when its
  key changes, keeping focus on the same model's control; a star toggles in
  place while the picker is open and the grouping follows at the next opening.
  The server's row state (`data-state`, the name line's badge) is written in
  place by `syncModelRowStates`, never through the rebuild key; that line is
  always present, so the badge never resizes a row. Stars and links are one
  roving tab stop (`syncModelTabStops`): the checked row's, else the first
  row's, plus the row holding the focus. Each radio's `aria-describedby`
  points at a hidden span of its cells' notes; a favourite's hidden prefix
  follows `modelTableGrouping`. The heading row is sticky in the picker body,
  whose `scroll-padding-top` keeps a focused row out from under it. A rebuild
  checks the wanted row before restoring focus, so a first opening (the body
  held the focus) lands on the checked radio, not the first. Opening, and a
  rebuild's refocus unless the body is hovered, scroll only
  `#model-picker-body` to the focused row (`revealModelControl`, below the
  sticky head); never `scrollIntoView`, never the page.
- The strip of the chosen checkpoint's images (`#model-shots`, `modelShots`)
  is a column on the picker's left from 1280 px wide, under the table
  below that; it shows the chosen row's newest images and count from `GET /api/gallery?model=<id>`, the
  gallery's exact scope chosen by the server, never a page-side match. It
  is a gallery read, in no model-route list, so it works cold, loading or
  switching; a refused read (bundle maintenance, offline) is said in its
  own box and retried from status polls (`retryModelShots`). It asks for
  only the whole rows its grid holds (`modelShotsFit`: columns times rows,
  at most 24, measured when the request goes and again on a window resize
  while the picker is open: `refitModelShots` drops the oldest at once when
  fewer fit, so no row is left cut in half or hidden in the arrow keys'
  walk, and asks again in place when more do), after a pause on a new
  choice; a newer choice aborts the older request and an epoch drops its
  reply, and the old pictures wait dimmed and `inert`. Under the table its
  height is fixed by `--model-shot` (64 px, 40 under 640 px tall) whatever
  it holds, so nothing under it moves. The column is absolutely positioned
  in the picker's padding box, which `padding-left` reserves, so the table
  side alone sets the picker's height and keeps its width
  (`--model-table-w`) and layout; the column clips to that height whatever
  it holds. Its pictures are `--model-shot-side` (the gallery's S cards:
  150 px, `clamp(150px, 6.5vw, 180px)` from 1800 px), two columns, three
  from 1800 px. The picker's width keeps a classic scrollbar's 17 px back
  from `100vw`, so its left edge never passes the gutter: from 1294 px the
  900 px table side, the column and the gutters fit whole, and from 1280 (a
  1920 px screen at 150%) the table side gives up to 14 px. This is the
  picker's own layout breakpoint, not a scale tier. The DOM, and so the tab
  order, is the same in both places. The footer, which holds the column,
  is `tabindex="-1"` like `#model-picker-body`: a press on its blank space
  (the column's under a short set of pictures is most of its height) keeps
  the focus in the picker, whose focusout would otherwise close it and
  drop an unsaved choice. While the picker is open, `#library` (the
  gallery, its Gallery beside column, the sheet) and `#recent` are
  `visibility: hidden` by a `:has()` rule in `model-picker.css`, never
  `display: none`: their layout and scroll stay, and they are back as it
  closes, so the picker's pictures are never read as the gallery's. The
  lightbox is outside them and still shows. Its pictures are one roving
  tab stop (`syncModelShotStops`, arrow keys/Home/End); a picture opens the
  lightbox with origin `models`, which the picker's pointerdown and
  focusout closes ignore while it is open: `dismissLightbox` focuses the
  picture again (`focusModelShot`), any other `closeLightbox` closes the
  picker. See all is hidden at count 0 (the empty line says why) and closes
  the picker, ticks All models, clears Starred, closes the sheet and
  searches `model:=<id>`. `forgetImages` calls `forgetModelShots`.
- Picker edits are per field (`modelPickerEditedKeys`, filled by the change
  handlers); polls and catalogue replies write every other field, so another
  tab's checkbox-only save still lands. A revision change or a successful
  switch clears them. While a loading option is edited, polls ask for no
  catalogue. `modelChoiceIntent` (a row, the server's selection, a switch)
  re-picks a choice a catalogue for other loading options lost: the exact id,
  then the listed id with its canonical `favourite_id`, else no row.
  `modelWorkHold` is the one list of page work a switch waits for;
  `offerModelPicker` uses it too.
- The picker note (`modelPickerNote`, a live region) holds outcomes and
  failures, then the server's state: a request error, a press's refusal
  (`modelPressReason`, dropped once that action's hold changes or the picker
  reopens), `switch_error`, a failed load's detail, then
  `modelPickerMessage`; after it Loading, the unsaved edits with Revert, or
  the loaded/not-loaded line. Holds go in the actions' titles and
  `aria-disabled` only, never the note. `modelPickerMessage` is set by an
  accepted press, naming what it did and the `startup_overrides` it cannot
  beat, or by a revision from elsewhere that retires unsaved edits the new
  settings do not match (`modelDroppedEdits`: "<new> was selected in another
  tab or window; your unsaved choice (<old>) was dropped."); a new failure
  clears it, and the revision reset keeps a press's only for this page's own
  switch (`modelOwnSwitch`). Revert clears the edited keys,
  reapplies the server's settings and focuses the chosen radio. The help
  describing the table is `.sr-only` and not live. A failed catalogue keeps
  its rows, dimmed by the shared stale rule (0.8, `--muted` raised) and
  `aria-busy`, through its retries until one succeeds, and holds the actions
  with its own reason.
- The loader selects are `appearance: base-select`, so their lists draw in
  the page; never return them to the native popup, a separate window Chromium
  re-anchors under a still pointer. While any picker list is open
  (`modelListOpen()`, `:open`), the page's key handler leaves every key to it
  (Escape closes only the list), its own value and option rebuilds (catalogue
  replies, other tabs' favourites) wait for it to close
  (`whenModelListsClosed`), and the closed controls still take a server
  change at once. Options are replaced only when they differ
  (`setModelOptions`); polls write only fields changed since the last applied
  snapshot. A runtime id missing from its own catalogue is requested once.
- Picker speed facts use retained `docs/verified.md` measurements at their
  recorded size and steps. Only a warm sample matching quantization, format,
  platform and offload gets Fast (at most 15 s), Medium (over 15 through
  45 s) or Slow (over 45 s), from the sample's upper time. A first image of
  a process on a busy PC (`Sample.warm=False`) matched is Reference with its
  basis "First run, busy PC" and its per-step figures in the note. Any other is
  Reference, or Unmeasured for BF16 or sequential offload, with a "timed
  with/on ..." basis naming what differs; the closest sample is chosen by
  exact quantization/format, then 4-bit or torchao family, then platform,
  then offload. Unmeasured variants never inherit a family timing. These are
  not ETAs or new GPU evidence.
- Picker access (`details_for`'s `access`/`access_note`) comes from
  `Checkpoint.gate`: click-through is Accept licence, manual is Needs approval,
  anything else and every lite-infer build Open, a custom id Unknown. Nothing
  checks the Hub or the cache for it; only a press does (`hub_access`).
- A checkpoint whose torchao builds are larger than the card has a
  `details_for` `fit`/`fit_note` from `model_info._FIT` (Qwen-Image only,
  from the catalogue's own record); its lite-infer build and every other
  entry have empty ones. The row shows `fit` beside the access label
  (`.model-fit`, the note its title and part of the radio's description),
  and the picker note says it, before the state line, while that row is
  chosen. It does
  not hold Switch & load: sequential offload is not refused. FP8 with it
  stopped at the RAM guard during the load (2026-10-25, a 64 GB machine,
  transformer shard 6 of 9); BF16 with it is untried.
- Quantization and offload descriptions live in `loader_choices.py` and are
  shared by the menu and the picker; only figures true of every catalogue
  checkpoint go there, and no unmeasured quality claim (INT8 is "the same
  size as FP8", and a little faster and usually nearer BF16 only "where
  compared", naming FLUX.1, Krea 2, klein 9B and Qwen-Image as not compared:
  `verified.md`, "Quantization and offload (2026-10-25)").
- On Linux, the server launcher prepares `DIFFUSERS_TRUST_REMOTE_KERNELS=true`
  before diffusers imports for optional browser Nunchaku selection; an explicit
  false remains authoritative and refuses Nunchaku before unload. The standalone
  CLI still enables it only when launched with Nunchaku.
- `load()` refuses a pipeline whose `__call__` lacks `generator` or
  `callback_on_step_end` (`_REQUIRED_PARAMS`): the seed and the cancel depend
  on them.
- A non-catalogue Hub id or local tree must expose a readable
  `model_index.json` with a concrete pipeline class before weights load.
  `loading.require_custom_pipeline_class` refuses unavailable editor classes
  and unreadable/malformed class identity; it never falls through to
  `from_pretrained` after an inconclusive read. The catalogue and its listed
  Nunchaku formats retain their existing load behavior. The optional
  `unused_components` optimization is not an admission decision.
- Instruction editing has one gate: the running backend instance's
  `instruction_edit_qualified()`, and only the literal `True` opens it.
  It is never a setting, environment variable or flag. The base class
  answers False, and `DiffusersBackend` overrides it with a body of only
  `return False` until stage-3 card qualification
  (`test_instruction_edit_gate.py` scans every definition under `backend/`
  for that literal). Every edit check takes `qualified=` from that instance:
  `require_execution_available`, `require_model_available`,
  `edit_unavailable_reason`, `applied()`, `check_refine_supported()`,
  `ensure_loaded`, `/api/model/load`, `jobs.submit_cells` and `_run`, and
  the per-cell recheck.
  The diffusers editor load holds take no `qualified` argument and stay
  unconditional: `load`, `_load_prepared`, both editor-class refusals and
  `require_custom_pipeline_class`. Kontext stays out of `launch.py`
  `CHECKPOINTS` until stage 4. A future editor load runs
  `require_pinned(validate_configs(...))` on the six `CONFIG_PATHS` fetched
  at the revision before any weights move.
- The edit call (`DiffusersBackend._edit_images`) branches off
  `_generate_images_prepared` right after the generator list. It never
  reaches the text-to-image kwargs, sigmas or the step that drops
  unsupported parameters. It passes exactly `_EDIT_PARAMS`, and a parameter
  missing from the loaded call is refused, never dropped. The source comes
  through `load_bound_source`. The call runs under `no_grad` and
  `forced_mu(pipe, None)`; the hooks are freed and the source closed in
  `finally`; and the result must be exactly one image at the recipe's output
  size, with no composite.
- Backend loading owns returned component overrides and the unpublished pipeline
  in one candidate through shard/weights-context exit, placement, callback checks,
  console configuration and diagnostics. Publish the complete call parameters,
  effective recipe and owned registration before assigning the live pipeline
  last. Failed candidates drop owned application containers before allocator
  release, outside backend/weights locks, preserving the original exception and
  attaching cleanup errors. Both unload paths detach base/sibling/adapter/recipe
  ownership together, retaining compatibility caches and console preference;
  contention in `try_unload` changes nothing. Private nonempty loaded state must
  refuse an ordinary eager request before device or pipeline work. This opens no
  selection/render gate and proves no physical GPU memory release.
- `AURADREAM_QUANT=nunchaku` means a pre-quantized lite-infer repo, Linux only
  (WSL 2 here), and no quantization config; `loading.nunchaku_preflight` and `loading.refuse_prequantized` refuse the other pairings before any download. Under
  WSL, `source wsl/env.sh` before `uv`.
- `config.py` sets `DIFFUSERS_TRUST_REMOTE_KERNELS` and must be imported
  before diffusers everywhere, the CLI included. The offload hooks are handed
  `AURADREAM_DEVICE` explicitly; given nothing, diffusers takes GPU 0.
- `loading.UNUSED_COMPONENTS` names what ordinary image loading leaves out
  (ERNIE-Image's prompt rewriter). `load()` reads `model_index.json` ahead of
  the weights and hands them as None to the *concrete* class's
  `from_pretrained`, which keeps them out of the download and the load; the
  base class takes `pe=None` for a pipeline kwarg and pulls the 7.7 GB.
  The explicit Expand endpoint loads those components on first use, rebuilds
  the offload chain, and puts the result in the prompt editor.
  `ModelStatus.prompt_expander` (`ExpanderStatus`: `loaded`, `fetch`
  [{name, bytes, step}], `fetch_bytes`; None without an expander) says,
  from file checks only (`backend.prompt_expander_status()`,
  `loading.prompt_expander_cached`, `MarianTranslator.cached()`), whether
  the rewriter is on the live pipeline and what a first use downloads, each
  at its step: the rewriter's `pe/model.safetensors`
  (`loading.PROMPT_EXPANDER_BYTES`, step `expand`) and the translator's
  `.bin` (`translation.MODEL_BYTES`, step `translate`). It is the
  expander's counterpart of the critic's `describe()`. The route answers a
  load that failed with 503 and the status's sentence; its other failures,
  and a translation's, come without the class name in front
  (`api_prompt.expand_detail`) - a message of one word keeps the class, and the
  log always does. `generate()`
  still sets `use_pe=False` wherever the switch exists. The tests'
  conftest refuses `loading.hub_file` for every test; one that needs an index stubs it.

Schedule and guidance:

- A requested `mu` is the whole shift. `schedule.forced_mu` overrides the value
  on its way into a dynamically shifting scheduler; a static one gets
  `schedule.shifted_sigmas` pre-shifted by `mu - ln(shift)`;
  `schedule.requested_shift` decides which of the two a pipeline takes, for
  `generate()` and the Refine preview's `start_sigma` alike. Assert on
  `scheduler.sigmas`, never on the list the backend computed. Mu requires a
  `FlowMatch*` scheduler first: SDXL's Euler scheduler accepts custom sigmas
  too, but is not flow matching. `mu_supported` is profiled before load and
  read from the scheduler after it, and disables both Mu controls when false.
  `mu_static` (in every status's defaults) is the shift a blank Mu leaves
  where the scheduler's shift is a fixed number: ln 6 on Z-Anime, NetaYume
  and NewBie, ln 3 on Z-Image and Z-Image Turbo (lite repos included), None
  elsewhere; it is set only where Mu applies and the profile sends no mu of
  its own (`test_a_fixed_shift_is_the_profiles_mu_static`). The panel's and
  the Refine dialog's Mu placeholder is that number (`muPlaceholder`: "1.79"
  where FLUX's says "auto", nothing on a dead box). The grid preview and the
  sheet read Mu's none as "none (1.79)" - on the sheet its heads, the Fixed
  line, pins and the export - and the sheet sorts it at the printed 1.79,
  just ahead of an explicit 1.79, only while every image on the sheet is the
  running checkpoint's. The stored mu stays null, `none` stays the grid's
  word, the lightbox's "no Mu" is unchanged, and the Mu tips format it from
  the profile.
- Where guidance switches on is the pipeline's: `cfg_threshold()` is 1.0 with
  `true_cfg_scale`, 0.0 for a `negative_prompt` pipeline in
  `_GUIDES_FROM_ZERO`, else 1.0. The profile guesses until a load;
  `/api/status` merges the live answer, as it does `mu_supported`,
  `negative_prompt_supported`, `true_cfg_supported`, `reference_supported`,
  `refine_supported`, `inpaint_supported` and `lora_supported`. It also says whether
  `refine_supported` is final (`refine_settled`: a live answer, or one of
  `model_defaults._OWN_PIPELINES` - Anima Turbo/Aesthetic and NewBie, whose pipelines
  are this repo's own classes, beside which diffusers exports no sibling),
  and whether `refine_strength` is the profile's own
  (`refine_strength_profiled`) or the general default filled in for an
  unprofiled pipeline with a pass. `inpaint_settled` is the same for
  `inpaint_supported`: a live answer, or any profile's False, since a load
  can only take an inpaint away.
- `GenerateRequest.applied()` is the one normalisation - negative prompt, mu,
  references, `lora_scale`, the refine source and its `strength` - and
  `backend.applied()` answers it from the parameter list or the profile. Every save path, the resume prune and the
  duplicate count go through it, never through the raw request.
- `true_cfg` on a row means the scale the negative prompt ran at, or None, on
  every checkpoint.
- `model_defaults._PROFILES` is first-match on the id, so a distilled variant
  sits ahead of its family's token (klein and schnell before flux, z-image-turbo
  before z-image, ernie-image-turbo before ernie-image); a catalogue entry
  that reaches the fallback is a test failure. Steps, CFG and mu are the
  checkpoint's; `mu` is Turbo's alone. `krea-dev` sits ahead of `flux` for
  its guide alone and is the flux profile in every number and control hint,
  `cfg_label` included. That is the one hint that is a name, "CFG
  (distilled)": `applyHints` writes it on `#cfg-label`, `#grid-cfg-label`
  and the Refine dialog's `#refine-cfg-label` (plain "CFG" where a status
  has none), and `hints_for`
  drops it where a loaded pipeline's negative-prompt answer replaces the CFG
  tip. No other profile has one: schnell's box is ignored and klein's
  guidance is baked in. `cfg_ignored` (in every status's defaults) marks a
  profile whose CFG box never reaches the model - the distilled klein,
  schnell and Qwen-Image - exactly where its CFG tip opens "Ignored"
  (`test_cfg_ignored_is_exactly_where_the_cfg_tip_says_ignored`).
  `_EXACT_PROFILES` takes precedence for pinned releases such as WAI v17
  and Anima Turbo; never match Anima with a broad `anima` token, which also
  occurs in Animagine.
- Every profile's `guide` (`backend/guides.py`) is sourced from the checkpoint's authors - the
  model card, README, prompting guide or report - never invented; the
  design notes keep the URLs. `pixels` is the range they put in writing and
  `pixels_word` what that writing was (documented, recommended, trained),
  or `example` where the authors publish no range and `pixels` is only the
  span of their own example sizes (NetaYume, NewBie, WAI). `pixels_source`
  names whose sizes those are, in the size prose's words ("the authors'
  example sizes", "the author's example", "the author's reference sizes"),
  set exactly where the word is `example`
  (`test_the_example_word_is_exactly_where_the_prose_says_no_bound`, which
  also ties the word to the size tips that say no maximum). The range must
  admit the 1 MP presets. `aspects` and `resolutions` are a card's finite
  table of shapes: Animagine's, and ERNIE's seven "Recommended Parameters"
  sizes on both ERNIE guides, where the 9:16/16:9 preset (768x1360) stands
  for 768x1376. Every listed aspect is a preset name, and every listed size
  takes one of the guide's framings by the page's nearest-aspect rule
  (`test_a_guides_aspects_are_presets_and_its_sizes_their_framings`). Its
  prose rides in `hints` as
  `prompt` and `size`, its data in `ModelStatus.guide`.

Reproduction:

- The reserved `execution` request field uses `execution.ExecutionSpec`:
  frozen named compilation, attention and step-cache recipes, with unknown
  fields/names refused. Every nonempty recipe is unavailable in this build,
  refused at validation, submission, queue admission, `applied()` and direct
  generation before model work; the queue checks every cell before stamping
  masks or retaining work, even with an explicit run id. The fake shares the
  backend guards. Runtime request resolution distinguishes omission (a new
  composer inherits selected loader intent) from a present complete replay
  (null/empty means legacy Eager, whose loader must match exactly).
  Submission-only `step_cache` belongs to the composer; its presence cannot
  accompany `execution`, including explicit Off. Resolve before run/planned
  identities, using narrow copies for stamped masks. Portable codecs never
  consult current selection. Resolution and `applied()` canonicalize empty
  intent to None without granting availability. Stored records carry the
  nullable recipe through canonical JSON in `images.execution` and named
  PNG metadata; empty execution is SQL NULL and absent from the PNG chunk.
  Its CellKey slot is immediately before canvas/crop, preserving shape[-2]
  and the final seed. `run_id` explicitly omits an absent execution slot
  even where a later shape is nonempty, then applies legacy shape/extension
  compaction. Loader choices can be saved, but no nondefault runtime is enabled.
- `backends.regional_compile` validates a fresh exact FLUX transformer,
  its uncompiled root and both installed region families before registration.
  Setup uses fullgraph/static Inductor registration under scoped strict
  compiler flags; failure discards the unpublished candidate and preserves
  its original exception. Closing a registration only drops owned references;
  it never unwraps modules, resets global compiler state or deletes caches.
  Public `load()` delegates to private eager preparation. Only an explicitly
  supplied private regional recipe reaches cold dependency/override preflight
  and registration after ordinary placement/callback checks. Invalidate a
  failed candidate before publication; drop/release through the outer owner
  outside backend/weights locks. Actual direct and refine pipeline calls share
  one invocation boundary; Eager enters no compiler policy. Private regional
  calls require the exact owned live root/fixed facts, one image and no adapter,
  then strict policy around one forward without retry. Public request, applied
  and direct-generation guards remain closed. Registration facts establish no
  compiled kernels, warm image or fallback qualification.
  Per-load diagnostics distinguish registration from the first attempted call;
  preflight/entry refusal stays pending, and returned follows verification and
  policy exit. Retire tokens at detach/unload; cold readers and frozen terminal
  job snapshots retain no pipeline. Observations never enter saved recipes,
  CellKey, run ids or image metadata and imply no qualification.
- Private attention configuration owns one exact FLUX transformer's complete
  joint/single processor layout. Validate types, config counts and every path
  before mutation; retain block/attention/processor identities and prior
  backend values. The public native context restores the setter's incidental
  global change. Roll back all snapshots on failed setup, preserve the cause,
  and never publish a failed candidate. A shared sibling uses the same handle;
  a second transformer cannot claim already-owned processors. Verify detects
  later replacement/backend changes, and restore releases owned references
  only after restoration succeeds. Callers must retain the handle, configure
  only while idle and discard failed candidates. None changes nothing.
  Private preparation now transfers that handle with the unpublished pipeline;
  both unload paths restore/dispose outside locks. Verify the exact owner/root
  before and after one direct base call. Preserve a primary forward failure or
  cancellation with verification notes; never retry or save a changed result.
  Terminal discard/failed setup weakly excludes original identities from reuse
  without retaining weights or poisoning unrelated replacement processors.
  Ordinary retained-owner restoration repair remains; every public nonempty
  request and generation guard stays closed.
- `config.ATTENTION_ENVIRONMENT_WARNING` snapshots the external
  `DIFFUSERS_ATTN_BACKEND` before framework imports, matching the installed
  verbatim default of `native`. Missing/exact-native values are silent;
  every other value, including empty, gets a fixed bounded config log and
  `ModelStatus.attention_environment_warning`. Never echo or mutate its value,
  reread it from status, infer an effective recipe or change admission. This
  diagnostic chooses no app/environment precedence or acceptance policy.
- Cold attention/cache candidate checks use supplied request, distribution and
  external-attention observations only. They reject unsupported combinations,
  adapters/passes and nonnative overrides without importing a framework or
  changing the environment. FBC additionally refuses true CFG and an existing
  cache. A matching candidate grants no runtime availability or loaded-API,
  processor, shape, kernel or device qualification.
- Private FBC ownership validates exact eager FLUX block identities, order,
  counts and output metadata. One retained owner covers the shared denoiser;
  each call installs fresh fixed-threshold hooks, resets state/context before
  and after use, and restores only its own hooks, forwards and registry caches.
  A failed cleanup invalidates that model identity before discarding every
  shared pipeline; invalidation survives owner close and requires a fresh
  model. Private preparation retains its owner without hooks; one direct base
  call gets fresh hooks/state after exact ownership, ordinary-guidance and
  read-only effective Native checks. Cleanup failure detaches the expected
  unsafe pipeline, then explicit disposal drains after call/sigma/no_grad scopes
  outside locks. Pending disposal holds load/forward; stale callbacks cannot
  detach replacements. Terminal discard drops refs without touching foreign
  hooks. Public nonempty guards remain closed; actual runtime/card qualification
  remains separate.
- Seeds are drawn from `2**53 - 1` and wrapped at `2**63 - 1`; both live in
  `seeds.py`. One generator per image, seeded `seed + i`; runs take base
  `seed + i * num_images`. The page refuses a typed seed past
  `Number.MAX_SAFE_INTEGER` (`startJob`; the Seed box's `max`), which a
  double cannot send exactly; the CLI takes the schema's whole range.
- `grid.expand()` is the one cell loop for the CLI and `/api/grid`;
  `jobs.submit()` keeps its inline copy of the seed loop deliberately, asserted
  equal to the CLI's in `tests/test_cli.py`.
- Grid prompt variables default to `prompt_mode="cross"`. In `random` mode,
  each variable is drawn from the image seed and variable name by the frozen
  `grid.random_assignment` hash recipe. Numeric axes share that draw; Batch N
  expands to N one-image cells so each saved row and PNG carries its actual
  prompt and `variables`. Random mode never constructs the variables' Cartesian
  product, and adds no reproduction field or key extension. Resume and
  duplicates use the resolved cells. Preview expands once and reports
  `prompt_sample` when neither a base seed nor a seed list fixes its choices;
  the page labels those prompts as examples and says Batch images run
  separately. The CLI uses `--grid-prompt-mode random`.
- Grid sizes are paired `WIDTHxHEIGHT` values in `size_axis`, never independent
  width/height products. Each concrete cell keeps its image seed and random
  assignment across sizes, passes the existing guards and contributes to the
  grid ceiling. Resume/planned work and estimates use each actual size. A
  mixed run reports all sizes, and Then refine reports all `pass_sizes`, with
  the old scalar pass dimensions only for a uniform result.
- `crop_anchor` is ordinary Refine reproduction data: center by default, or
  an edge/corner. It is refused beside a mask/outpaint or without a source,
  and resets when a source is dropped or Mass refine chooses another. Center
  keeps the old resize/rounding/crop path. Noncenter anchors travel through
  rows, PNGs, reruns, reindex, retained work and bundles; center remains NULL
  in storage and absent in legacy metadata. One shape extension carries size
  and anchor together in `CellKey`. Uniform centered run hashes keep the
  legacy definition; mixed/noncenter runs name their concrete shapes. Source
  frames, comparisons and sheet/export thumbnails must show that exact crop.
- A producer that runs gallery records again builds its cells with
  `grid.cell_from_record` (one record, naming only what changes: a size, a
  prompt, a negative prompt, variables) or `grid.cells_from_records` (a list
  at one optional size, in the order given - oldest first), never a
  `GenerateRequest` of its own: the reproduction set carried (prompt pair,
  steps, CFG, effective true CFG, mu, references, LoRA and weight, refine
  source and strength - a strength only beside a source - seed, variables)
  is one list. Any other `GenerateRequest` field is named through
  `**changes` (validated with the rest; a name that is no field is a
  TypeError; a `seed` there is the cell's base seed as well as the
  request's) and the cell's caption through `label`. Re-render, Swap tags,
  Mass refine and More seeds use it. Their
  images are `submission.resolve_selection`'s: with `ids` (the picked cards),
  exactly those, oldest first, and the ids gone (deleted, or binned since
  they were picked: the third list) reported per image, never a
  refusal; without, the gallery's filter (search, Starred, `AppServices.model_scope`),
  404 for none, 422 past `max_grid_images` by count. What to do with
  another checkpoint's records is the caller's (Re-render refuses the
  selection, `api_generate.foreign_refusal`, naming picked ones to unpick;
  Swap tags reports each image `foreign`;
  Mass refine reports it `foreign` unless `include_foreign`, then
  substitutes the running profile; More seeds reports each batch
  `foreign`).
  `submission.unusable_input` is `submission.refuse_unusable_inputs`' finding without the
  refusal, (kind, why), for a per-image status.
- The reproduction set - prompt pair, size, steps, CFG, true CFG, mu, seed,
  checkpoint, quantization, references, LoRA and weight, refine source and
  strength, variables, hash - is on the row and in the PNG's
  `auradream` chunk. So is `batch`, the image count of the call that
  made the image (`cell.request.num_images`, from `jobs._save` and the CLI;
  NULL on older rows), which is no part of it - one generator per image -
  but what More seeds runs a batch's next images at; a reindex, a merge
  and a bundle carry it, and a chunk's value that is no positive integer
  is read as NULL (`storage._batch`). So is the peak pair,
  `vram_peak_allocated` and `vram_peak_reserved`: the bytes PyTorch's
  allocator held at most during the `generate()` call that made the image
  (`GenerationBackend.last_peak()`, reset as `DiffusersBackend.generate`
  begins and read as it returns, None on a call that raised, with no CUDA
  device and from the fakes), saved whole on every image of a batch by
  `jobs._save` and the CLI, never divided as the duration is. A result, no
  part of the reproduction set, `CellKey`, `run_id`, `applied()` or a search
  key; absent from the chunk when unmeasured, and a chunk's or a bundle's
  value that is no non-negative integer is NULL (`storage._peak`), never a
  refused row. Reindex (both paths) and bundles carry it; the bundle
  staging table adds both as INTEGER. `CellKey` holds the prompt pair, steps, CFG, mu,
  true CFG, references, LoRA and weight, refine source and strength, execution, size/crop
  shape, and the seed; checkpoint and quantization scope the resume lookup
  (`storage.completed_cells`) and, with the batch, `run_id`'s definition;
  variables and hash are the row's and the chunk's alone. `run_id` names a
  cell by the shortest key carrying its non-None optional elements, so every
  existing run id survives a widening - except a grid whose `none` value
  sat beside doubled spaces, indentation or trailing spaces, or right after
  `(`, `<tag>` or a quote: `grid.substitute` closes only the hole
  (`_close_hole`), so its none cells resolve to another prompt, and a
  Resume of a run begun before that rendered them again under a new run id.
  `/api/rerender` and `--from-image`
  carry a row's variables. The CLI carries them only while the prompt pair is
  the image's own and never beside a `--grid-prompt` axis; it rebuilds a
  comparison version through `expand(variants=)`, verbatim, and attaches any
  other assignment, including a `variant` value no `PromptVariant` label
  admits.
- Resume keys on settings (`storage.completed_cells`, six-decimal rounding,
  scoped by size, checkpoint and quantizer), requires an explicit seed unless
  the seed axis is a list, and answers 409 when nothing is left. Resume also
  drops the cells a live job already plans (`submission.not_yet_planned` over
  `jobs.planned_keys()`, after `not_yet_rendered`), counted apart as
  `GridPreview.queued`, never in `skipped` ("already done"); it is opt-in,
  never inside `not_yet_rendered`, whose other callers speak of the gallery
  only. Without
  Resume, `/api/grid/preview` asks the same lookup (`submission.not_yet_rendered`)
  only for an explicit seed or a seed list, reports it as
  `GridPreview.in_gallery` and drops nothing. A cell in the recycle bin
  counts as rendered (a rejected render, not a missing one); Empty recycle
  bin makes it eligible again, as a delete does.
- `_connect` runs `_migrate` on every connection (`PRAGMA table_info`, no
  once-per-process flag): an older file restored over the same path under a
  running server gains its columns at its next connection.
- A new column needs a `_migrate` entry, not a `"x" in row.keys()` read guard:
  `_row_to_record` reads every column unguarded. `save_image` and
  `storage.index_file` (`reindex`'s writer) share `_insert_row` over `_COLUMNS`, and both build
  their dict explicitly, so a new column goes in both (`binned_at: None`).
- The recycle bin is the `binned_at` column (ISO UTC text like `created_at`,
  NULL while live, never NOT NULL: an older database or bundle reads None and
  is live). Every page-side delete bins (`POST /api/images/bin`,
  `gallery_delete.bin_images`); only Empty recycle bin (`DELETE /api/gallery/bin`,
  `gallery_delete.empty_bin`), `DELETE /api/images/{id}` (the bin view's Delete
  permanently, permanent for live and binned rows alike) and `clear` remove
  PNGs. Files and thumbnails never move: a bundle's inventory takes only
  `images/*` and `images/thumbnails/*`, and a Restore is one UPDATE. A binned
  image is gone for every live-facing read and present for every read about
  what exists or was made. `storage._LIVE` (`binned_at IS NULL`) is added by
  `list_images` (before the search predicate, which the LIKE fallback swaps
  as the last; `binned=True` lists the bin alone), `get_image`/`get_images`
  (unless `include_binned`), `reference_paths` (a binned id is refused,
  "is in the recycle bin; restore it to use it", before the file check - so a
  submission, a queued cell and `generate.py` alike), `set_favorite` (404
  while binned) and `set_favorites` (a binned id answered `binned`, its flag
  as it was), `runs`, `keep_rates`, `_STARRED_SOURCE` (only a live star
  protects) and `_UNKEPT` (a binned star is unkept), and by
  `matching_preview(live_only=True)` / `delete_matching(to_bin=True)` for
  Delete unstarred. `_tracked_files`, `existing_filenames`,
  `existing_file_ids`, bundles, `clear`, `completed_cells`, `lineage_images`,
  `image_provenance`, `newest_image_id` and the cost/size samples read every
  row: filtered to live rows, the orphan sweep would delete binned PNGs and a
  reindex would add them back as new rows, and the status's newest id would
  fall at a bin and rise at its restore - an image the page already knew. There is no `bin:` search key - it would reach
  `strict_scope`, and through it Delete unstarred, `clear --where` and
  `keep_rates`; the bin's view is its own route. A starred image binned
  keeps `favorite = 1`, so a Restore brings the star back.
  `bin_images`/`restore_images` look up and write under one lock (a double
  press answers `binned` once and `already` once) and answer
  `{binned|restored, already|not_binned, missing}`.
- Empty recycle bin is pinned to the images its confirmation covered, not
  only their count: `bin_summary` answers `pin` (`gallery_delete.bin_pin`,
  16 hex digits of the sorted binned ids), the page and `clear --bin` send
  it back with `expected`, and `empty_bin` compares both under the write
  lock. A Restore and a bin of another image in another tab keep the count
  and change the pin: 409 "The recycle bin changed: other images are in it
  now than the N confirmed. Nothing was deleted; look again." `clear
  --where` pins the count its prompt showed (`expected`, net of what
  `--keep-favorites` keeps) and answers a `SelectionChanged` with exit 1 and
  nothing deleted, as `--bin` does.
- An age-scoped Empty (`GAL#5`: the bin's `#bin-age` select, `clear --bin
  --older-than DAYS`) is pinned to a cutoff the server fixed:
  `GET /api/gallery/bin/summary?older_than_days=N` computes it once
  (`gallery_delete.bin_cutoff`) and answers it as `before`, its `records`,
  `bytes`, `sources_in_use` and `pin` being those of the rows binned earlier
  (`bin_records` the whole bin). `DELETE /api/gallery/bin` takes that
  `before` back unchanged - never a day count recomputed at delete time -
  and `empty_bin(before=...)` re-selects by it under `BEGIN IMMEDIATE`,
  compares `expected` and `pin`, and deletes only those ids; files,
  `forget_images` and `sweep_masks` follow as for the whole bin. Stamps are
  compared parsed (no zone is UTC), never as text; an unparseable
  `binned_at` is not old enough for a scoped empty. A malformed `before` is
  422. The footer's count is the whole bin's: the page never shows a scoped
  summary's `records` there.
- Reference images are gallery ids, never paths. The page's chips are
  numbered in `referenceIds` order, since that order is what a prompt's
  "image 2" means; every reorder or removal (a drag, Alt+Left/Right or Delete
  on a chip's ×) builds a new array through `setReferences`, the one writer,
  and chip nodes are never moved. A chip whose picture fails is marked
  `.gone` and asks `GET /api/images/{id}` (`dropGoneReferences`); only
  "unknown image" forgets it (`forgetImages` through `forgetReferences`,
  binned where the detail says "in the recycle bin", with its note in
  `#error`; the lightbox's reference slot likewise reads "in the bin" or
  "deleted"). A
  submit refused with "no gallery image with id N", or with "reference image
  N is in the recycle bin; …" (N the gallery id, not the chip's number), for
  one of the payload's references does the same (`dropRefusedReferences`)
  and says "Nothing was queued: reference #N has been deleted, so it was
  taken out of the references. Generate again to run without it." ("has
  been moved to the recycle bin" for a binned one; "Run the grid again",
  "Run the comparison again" after those, `pressAgain`). A submit from the
  panel refused with a sentence naming its chosen adapter reads `/api/loras`
  again (`dropRefusedLora`, through `refreshLoras`, which answers the
  listing it applied); one no longer offered has left LoRA, the panel is
  saved, and the refusal reads "Nothing was queued: NAME is no longer in the
  LoRA folder, so it was taken out of LoRA. Generate again to run without
  it." ("no longer fits this checkpoint (the LoRA tip says why)" where the
  file is still there for another checkpoint) - never the server's sentence
  with the directory's path and `AURADREAM_LORAS_DIR`. One still
  offered keeps the server's sentence. After a removal focus goes to the next ×,
  else the previous, else the emptied `#reference-list` (tabindex -1), never
  the field's badge, whose tip a focus opens. A LoRA is a bare name that
  only `loras.py` turns into a path, `.safetensors` only, and it is
  hot-swapped in `generate()`, never baked into `load()`.
  `loading.lora_refusal(model_id)` is the one id-based refusal (Anima Turbo
  and NewBie, from the loaders' `MODEL_ID`s, case-insensitive):
  `DiffusersBackend.lora_unavailable_reason` asks it after the Nunchaku
  check and ahead of the peft and loader checks, and `tools/fake_server.py`
  sets the fake's `lora_reason` from it. Whether it fits is
  `loras.mismatch` - the adapter's module names and widths against the
  denoiser's, using the pipeline's adapter reader and PEFT key conversion -
  from the checkpoint's configs on the meta device before a load
  (`loading.denoiser`) and the live pipeline
  after. That build is accelerate's `init_empty_weights`, which patches
  `Module.register_parameter` for the whole process, so it goes through
  `loading.empty_weights` and never overlaps a load: `load()` holds
  `loading.weights_loading()` throughout (its conversions inside) and waits
  out a build in progress, a build asked for during a load raises
  `LoadInProgress` and the fit goes unchecked, and builds go one at a time.
  A misfit is listed with its `detail`, hidden by the page and
  refused at submit and by the CLI, unreadable configs mean unchecked rather
  than refused, and a listing never converts a single-file checkpoint.
- A refine pass is `GenerateRequest.refine` - a gallery id, like a reference -
  and `strength`: the source resized to the request's size (`refine.fitted`:
  scaled to cover and centre-cropped, never stretched) and re-denoised from
  the point on the schedule the strength names. It runs the image-to-image
  class diffusers exports beside the loaded pipeline's own (`refine.sibling`:
  by name, only for the class diffusers itself exports, and only where its
  call takes `refine.REQUIRED` and every `refine.CARRIED` argument the
  text-to-image call takes; `CARRIED` is held to `generate()`'s own kwargs by
  `tests/test_refine.py`), built once over the pipeline's own components by
  `refine.build` - never `from_pipe`, which ends by casting the modules it
  was handed - and let go on unload. `refine_supported` is profiled before a
  load, each guess held to the installed diffusers, and the class's answer
  after it; unsupported, or a source that is gone, is refused at submit and
  by the CLI. Both recheck after loading and fail before rendering if the
  live capability would drop the requested source. `applied()` drops the source, puts `strength` back to
  the checkpoint's default with it, and drops references under a pass. That
  default is the profile's `refine_strength` (`schemas_generate.default_strength`,
  `refine.DEFAULT_STRENGTH` where the profile names none), filled in for a
  request that gives no strength and returned by live status when an
  unprofiled pipeline supports the pass, and where the Refine dialog's
  Strength starts, because a strength is an index into the
  checkpoint's own schedule: 0.4 on SDXL, 0.1 on FLUX.1 [dev], where 0.4
  repaints a face. Each value must leave a step of its profile's own step
  count; `docs/design-notes.md` has each one's basis and `docs/verified.md`
  says which were measured. One call
  per image of a batch, each with that image's own generator
  (`DiffusersBackend._refine`): the VAE's encoding of the source is sampled,
  and `FluxImg2ImgPipeline` samples a batch's from the first generator alone.
  The loaded pipeline's `maybe_free_model_hooks()` runs in that method's
  `finally`: the sibling has no hook list of its own. `steps` on the row is
  the schedule's length, of which about `steps * strength` run
  (`refine.steps_run`, SDXL/flow rounding from `refine.start_index`; unknown
  profiles use the floor until the callback supplies the pipeline's count).
  A job's initial `total_steps` uses that count. `gallery_reports.step_cost_ms` prices
  direct rows only; `gallery_reports.refine_cost_ms` prices a pass from full per-image
  durations at the same size, checkpoint and denoised step count, with no
  estimate where matching history is absent. The duplicate advisory compares
  schedule start indices; reproduction keys retain the requested strength.
  `GenerateRequest.check_refine_steps()` refuses less than one step after
  the capability check and on each expanded grid cell, never the unused base.
  `refine_key`
  (`id@strength`) is `CellKey`'s fourth optional element;
  `GridRequest.strength_axis` is an axis only beside a source, as the LoRA
  weight is only beside an adapter. Re-render, `--from-image` and Reuse carry
  both; a database merge remaps the source id as it does a reference's.
  PNGs also carry `reference_sources`/`refine_source` descriptors (filename,
  pixel hash and dimensions): PNG-only recovery verifies these before
  assigning new ids, clears/counts unproven legacy or missing links, and
  rewrites changed ids in PNG metadata even during an in-place rebuild. The
  CLI's `--scale` and the dialog's scale buttons share one rounding
  (`refine.scaled`, half up). A pass has a size guard of its own,
  `refine_max_side`/`refine_max_pixels` (any 2 MP preset at 2x: 4448 and
  8,411,136): the schema's `width`/`height` bound is that,
  `_direct_renders_stay_in_the_presets` holds a request with no source to
  `max_side`, and `check_budget` holds a pass to `refine_max_pixels` and
  skips the batch budget, since a pass is one call per image. A typo guard,
  not a VRAM model: what fits is the checkpoint's (`docs/verified.md`). The
  dialog's tight line - where its size line warns and past which it does not
  open - is the profile's `refine_measured` (`model_defaults.Measured`: the
  checkpoint and quantization of the largest pass verified.md's "Past 4 MP"
  section ran without spilling, and its size: fp8, or NVFP4 for Qwen-Image,
  which fits the card only under Nunchaku; `tests/test_profiles.py` holds
  each to its row), reaching the page as `defaults.refine_tight_pixels` and,
  in words, `hints.refine_tight`; None, the generic 1600x1600 line, for
  every profile with no row - since 2026-10-05 every catalogued profile
  with a pass has one, so only an unprofiled pipeline gets it. A shared
  profile shares its row (FLUX.1 dev and the lite repos quote Krea dev's
  fp8 run), so the size line names the run it quotes, never "this
  checkpoint". FLUX.1's `hints.refine_large` is shown
  under Strength past it (`applyHints` clears it where a status has none).
  The preview reports where a pass of one strength starts
  (`GridPreview.start_sigma`, the dialog's "starts at N% noise", and the
  Inpaint and Outpaint dialogs' alike), read by
  `backend.start_sigma` and never computed by the page.
  `schedule.start_sigma` builds a copy of the loaded scheduler from its
  config, with the loaded one's static shift, and hands it what the
  sibling's call would - its module's own `calculate_shift` over the
  sequence length that call measures, with the scheduler config's base and
  max shift; its default schedule; a requested mu through `requested_shift`
  and `forced_mu`, as the pass applies it - through the module's own
  `retrieve_timesteps`, and reads `sigmas[refine.start_index * order]`. The
  loaded scheduler is never touched. `START_RECIPES` copies
  `FluxImg2ImgPipeline`'s, `QwenImageImg2ImgPipeline`'s and
  `ZImageImg2ImgPipeline`'s statements, and the inpaint classes' beside
  them (`ZImageInpaintPipeline` and `FluxInpaintPipeline`, and
  `QwenImageInpaintPipeline` for a family still deferred, which nothing
  asks; klein's inpaint class uses `compute_empirical_mu` and has none);
  `tests/test_refine_schedules.py` holds them, `SHIFT_CALL`, the default
  schedule and the sibling's VAE scale to the installed source, and every
  flow refine profile and every flow inpaint profile must have a recipe. The reading is None before a load, for a scheduler that is not
  `FlowMatch*`, a class not in the table, an unprofiled start index, a
  sweep, or a reading that raises (it is advice, and a preview it broke
  would refuse the pass). An open dialog's preview is asked again when the
  model becomes ready or stops being so (status.js, beside the grid
  preview's re-ask), so a load that lands with the dialog open adds the
  reading and an unload takes it away. The fake backend reads none.
  `ImageRecord.refine_missing` is read, not stored (`storage._missing_sources`,
  from `list_images`, `get_image` and `save_image`, which asks under the write
  lock, so a pass whose source was deleted while it ran lands saying so;
  `jobs.forget_images` marks a job's replayed passes), and is not in the
  reproduction set. A binned source is missing too, since it cannot be used
  until restored; `refine_binned` (read, not stored, from the same lookup)
  says which, so the page reads "refine of #N (in the recycle bin)" and
  offers Restore, which clears both.
- An inpaint is a refine pass with `GenerateRequest.mask`: a stored mask
  named by `storage.mask_hash` (all 64 hex digits of the SHA-256 of its
  size and `L` levels), never a path; `storage.mask_path` is the one
  resolver, through the `masks` table (created on every connection beside
  `_migrate`), naming an unknown hash and a missing file. Masks live in
  `outputs_dir/masks/`, written by `storage.store_mask` (`.png.part`,
  renamed after the row commits; the stale-partial sweep finishes one whose
  row exists and that decodes to the mask its name says, and discards any
  other), uploaded through `POST /api/masks` (a PNG over a live gallery
  `source`, at its size, under `refine_max_pixels`, refused when it paints
  nothing; an `OSError` writing it is a worded 503) and served by `GET
  /api/masks/{hash}`; both model-free. `refine.canonical_mask` is the one
  reading of a picture as a mask, for the upload and `--mask`: 16-bit
  levels scaled to 8, a picture with alpha laid on black, `L`/`1`/`RGB`/
  opaque `P` converted, any other mode refused by name.
  A mask is no gallery image, never a reference or a source. The validator
  refuses a mask without `refine` ("a mask needs the image it covers");
  `applied()` asks `inpaint_supported` with a mask and `refine_supported`
  without, and drops the mask with its source. At submit
  (`submission.unusable_input`: kind `mask_missing` where `mask_path`
  refuses, which Swap tags and More seeds report per image as status
  `mask_missing`, "the mask of #N is missing" (`unusable_detail`); kind
  `mask` for a size mismatch) the source is live, the mask held, and mask,
  source and request are one size; a cell checks its mask
  file as it begins (`jobs`) and fails in words. `images.mask` is a
  `_migrate` column beside `refine`; the chunk carries `mask`,
  `mask_source` (`storage.mask_provenance`) and `mask_blend: "overlay"`,
  null together otherwise, the A1111 block `, Inpaint mask: <16 hex>`, and
  the mask's own PNG rides in the `auradream-mask` text chunk
  (`metadata.PNG_MASK_KEY`; best effort, the hash still says what ran).
  `grid.mask_key` is `CellKey`'s fifth optional element, after refine;
  `run_id` is unchanged for every run without one (pinned in
  `tests/test_conditioning.py`), and `duplicate_count` keeps every element
  after the refine one. `gallery_reports.refine_cost_ms` splits on
  `mask IS NULL` (`inpaint=`), and the estimates price an inpaint cell
  from inpaints alone (`inpaint_costs`). `cell_from_record` carries the
  mask with its source; Mass refine drops it, since the record is the new
  source. Load settings from PNG carries no mask, as it carries no source.
- Masks travel with their images. A bundle carries `masks/<hash>.png` as
  its own top-level prefix (manifest still `version: 1`; an older build
  skips it) and the `masks` table in its database; a selection carries only
  its rows' masks. The import holds each mask member to its name
  (`bundle_images._inspect_mask`: `L`, `storage.mask_hash` equal to the
  name, the size its row records), merges rows by hash (`bundles._merge_masks`;
  one hash at another size is refused), installs absent files, and counts a
  row whose mask neither the bundle nor its result's own embedded copy
  (`bundles._embedded_member`, held to the hash and the row's size) holds
  as `masks_missing`, keeping the hash; an unresolved source takes the
  row's mask with it. A whole-gallery inventory carries only files in
  `masks/` named by 64 lowercase hex digits plus `.png` that decode to that
  mask (`bundles._carried_mask`), leaving the rest where they are and
  naming them in the preview's `masks_left_out`. Re-render refuses such a
  row "the mask of #N is missing" (`submission.missing_masks`), and an
  inpaint at another size than its own naming it and its source
  (`submission.resized_inpaint_detail`). Reindex
  rebuilds masks from `masks/*.png` (a file not named by its pixels is
  counted `bad_masks`), from the `--from` gallery's `masks/` with
  `--database`, and from each result's embedded copy, counting
  `unresolved_masks`. `gallery_delete.sweep_masks` is the one remover:
  where PNGs are removed for good (Empty recycle bin, `DELETE
  /api/images/{id}`, permanent `delete_matching`, `clear`), it spares a
  mask any row names, live or binned, any live job's cell names (the
  `mask_holds` table, which `JobManager.submit_cells` writes through
  `storage.hold_masks` before a job is queued - a mask gone by then is a
  422, `jobs.MaskGone` - `_finish` releases and a server's start empties,
  so a CLI sweep sees a server's queue; and `JobManager.masks_in_flight`,
  passed by the routes, both read inside the sweep's `BEGIN IMMEDIATE`),
  or that is younger than `PARTIAL_SETTLE_SECONDS` (a re-upload or a
  `/keep` restarts the clock); the
  whole `clear` takes every mask no kept row names and the directory, and
  its orphan sweep a mask file with no row once settled. The bin never
  touches a mask. `generate.py --mask FILE` (beside `--refine`, at the
  source's size) stores through `storage.store_mask`; `--from-image`
  carries the record's mask, from the PNG's own copy where the gallery
  lacks it, and its source and mask only where the row under the id has the
  pixel hash and size the record's `refine_source` names
  (`_check_carried_source`: else a note and a plain generation, or a
  refusal where `--strength`, `--scale` or `--mask` asked for the pass).
  Search: `mask:<hex>`, `mask:none`, `inpaint:yes|no`; stats
  `--by mask`.
- An outpaint (`docs/plan-outpainting.md`, phases 1 and 2 built) is an inpaint
  over a padded copy of its source. Two types: `OutpaintSpec` is the stored
  one (`ImageRecord.outpaint`, the `_migrate` column `images.outpaint`
  holding `OutpaintSpec.key()`, `fill:L,T,R,B@band`), plain ints with grid
  checks only (sides multiples of 16, band of 8) and no bound read from
  settings; `Outpaint` is the request's (`GenerateRequest.outpaint`), with
  the default fill, `outpaint.MAX_BAND` and at least one side, its canvas
  bounded by `check_budget`. The validator refuses one without `refine`
  and one with a `mask`; `GenerateRequest.inpainting` (a source with a mask
  or an outpaint) chooses the inpaint class, its strength, its words and its
  costs, while the stored mask file's checks ask `mask is not None`.
  **Invariant:** an outpaint is never on a row without both `refine` and
  `mask`; every carry site drops it with either. The size is the source's
  plus the padding exactly (`submission.outpaint_unusable`, kind
  `outpaint`, status `outpaint` in Swap tags and More seeds: an off-grid
  source, an unknown fill by name, the size, the band's kept rectangle -
  under `outpaint.MIN_KEPT` either way, said by `outpaint.band_too_wide`
  with the short axis and the padded sides the band comes off). One
  well-formedness rule for a stored spec, `OutpaintSpec.wellformed()` (a
  side extended, the band within `MAX_BAND`): the chunk
  (`storage.chunk_outpaint`), a merged database (`reindex.merge_database`,
  counted `outpaint_malformed`) and a bundle row (`_validate_row`, refused)
  all hold to it, so a row's spec always runs again as an `Outpaint`;
  Re-render still answers a row the request models refuse as a worded 422.
  A bundle import stores the key in `key()`'s one spelling, and
  `completed_cells` keys the column through it, so a `0400` row keys as its
  re-render. A row whose spec does not build its own mask (a bundle's
  `masks_missing` row, a reindexed chunk, another build's mask code) is
  refused by every rerun (`submission.mismatched_outpaints`: Re-render's
  422, Swap tags' and More seeds' status `outpaint`), as `--from-image`
  refuses it; `--from-image` with a `--band` of its own builds the new
  band's mask and holds nothing to the record's.
  The mask is the server's: `storage.stamp_outpaints` builds
  (`outpaint.mask`) and stores it as the **first storage step** of
  `JobManager.submit_cells`, before `Job(...)` (`OSError`/`GalleryBusy` a
  worded 503, `jobs.OutpaintStampFailed`), and in `generate.py` after the
  resume prune and before the load (never under `--dry-run`; no
  `mask_holds` rows). A stamped request holds a mask and an outpaint by
  `model_copy`; never revalidate one. `grid.mask_key(mask, outpaint)` folds
  the outpaint into the fifth key element as `outpaint:<key>`, so stamped
  and unstamped cells key and name one run, and no existing `run_id`
  changes. **Each fill in `outpaint.FILLS` and `outpaint.mask` are frozen**
  (pinned by SHA-256 in `tests/test_outpaint.py`): a changed fill is a new
  name. The chunk carries `outpaint` and `outpaint_fill`, the A1111 block
  `, Outpaint: L/T/R/B, Outpaint band: B, Outpaint fill: NAME`; bundles and
  reindex carry them, build a lost mask again from the spec where it hashes
  to the row's, and recover a spec an older build stripped from the A1111
  line only where the rebuilt mask hashes to the row's. Presets are the
  server's (`GET /api/images/{id}/outpaint-presets`, `outpaint.presets`).
  Search `outpaint:yes|no`; stats `--by outpaint`. The page is the Outpaint
  dialog (phase 2, below under the frontend rules).
- Inpainting is a decision as well as a capability. `refine.sibling(pipe,
  "Inpaint")` finds the class under the exact-export rule, its call taking
  `refine.INPAINT_REQUIRED` (`REQUIRED` and `mask_image`) and `CARRIED`;
  `DiffusersBackend._inpaint_supported` also needs the pipeline outside
  `model_defaults.INPAINT_DEFERRED` (`Flux2KleinPipeline`,
  `QwenImagePipeline`: diffusers ships a class, this app has not enabled the
  family), the profile's `inpaint_supported` True, and the loaded class's
  name equal to the profile's `inpaint_family`, so a deferred family, an
  unprofiled SDXL id, or a hand-entered id the catch-all FLUX.1 profile
  takes but which loads as another class (FLUX.1 Kontext, Control) is
  refused with the class installed. True only on
  the six SDXL checkpoints, Z-Image Turbo, Z-Image and Z-Anime, and FLUX.1
  Krea dev, dev and schnell (from 2026-10-19; the Nunchaku Z-Image and FLUX.1
  repos share those profiles and nothing refuses them; the class ran over
  the Z-Image Turbo and Krea dev Nunchaku builds under WSL on 2026-10-25,
  the other Nunchaku repos not run); the fallback is False. A
  deferred profile carries `inpaint_deferred`, so the page says "not
  enabled yet" rather than "no inpaint pipeline in the installed
  diffusers". `tests/test_inpaint.py` holds every True to an installed,
  non-deferred class and every catalogue class left off to the list;
  `tests/test_inpaint_ui.py` holds `inpaint_deferred` to it. The Strength
  box starts at `defaults.inpaint_strength`, the loaded class's own
  signature default (`refine.library_strength`, read with `inspect`, None
  unless supported) and said as the library's number, since no catalogue
  author gives one; there is no `inpaint_strength` profile key. The one
  exception is a profile's `inpaint_start`, its own start where the card
  measured the class's default as the wrong one: FLUX.1's three profiles
  carry 0.95 (Krea dev, 2026-10-19: the library's 0.6 kept the old colour).
  `model_defaults.inpaint_start` is asked first by every place deciding a
  start - `default_inpaint_strength`, `DiffusersBackend.inpaint_strength`
  (and so `describe()`), the status merge (which gives it before a load
  too) and `tools/fake_server.py` - and `defaults.inpaint_start` tells the
  page to word the note as the checkpoint's start, not the library's. Only
  the default moves: an explicit strength runs, records and keys as before.
  Where the card measured the start as the right one (WAI v17, Animagine XL
  4.0, Z-Anime at the library's default, FLUX.1 Krea dev at its own:
  `verified.md`, Inpaint) the profile names itself in
  `inpaint_strength_measured` (in `_NOT_DEFAULTS`: a label), and
  `hints.inpaint_strength` - every enabled profile's, words from
  `hints_for`, never the page's - says it was measured there, or on the
  checkpoints running the same inpaint class and not on this one;
  `tests/test_profiles.py` holds the labels to that section. A request
  with a mask and no strength (`--mask`, the API) starts at the same
  start before any load
  (`schemas_generate.default_inpaint_strength`, through each enabled
  profile's `inpaint_family`, the text-to-image class it runs). An inpaint
  runs at its source's size (`refine.inpaint_source` refuses another; the
  mask through `refine.mask_image`), one call per image through `_refine`'s
  loop, and drops references. The class is handed `refine.pipeline_mask`
  of the painted mask - thresholded at > 0 and widened to whole latent
  cells (`vae_scale_factor`, 8) - since it binarises at 0.5 and samples
  each cell's corner pixel. After each decode the result is always
  composited over the source with the painted mask's levels as alpha
  (`refine.composite`, the library's `VaeImageProcessor.apply_overlay`), so
  every unmasked pixel is the source's; it is no setting, and the chunk's
  `mask_blend: "overlay"` records the rule. The `content_hash` is the
  composite's. `start_sigma` for an inpaint (or an outpaint, at its
  canvas's size) asks the inpaint class's own `START_RECIPES` entry, so
  Z-Image answers and SDXL's epsilon schedule does not; `start_index`/
  `steps_run` take `inpaint=True`, keyed on `inpaint_supported`. The live
  preview runs for an inpaint as for a refine, drawn from the inpaint
  class's callback latents (Z-Image's are the blended latents, unpacked).
  No profile has an `inpaint_measured` row: none measured differs from its
  `refine_measured`, and the dialog has no tight line.
- Under Nunchaku (WSL) a seed does not reproduce bit-for-bit - Z-Image
  within noise, ERNIE visibly - so a duplicate count or a re-render's hash is
  the fp8 path's guarantee only; `docs/verified.md` has the numbers.
- `GridRequest.prompt_variants` (2-8 `{label, prompt}`) is the prompt axis in
  place of the variables, never beside them: each variant is taken verbatim
  (nothing substituted; a `{`, `|` or newline is text), seeds stay outermost
  with the variants next in the order given, and each cell records
  `variables={"variant": label}`, whose starred rate per label is
  `stats --by variant`. `CellKey`, `run_id`, resume and the duplicate count
  read the resolved prompt, not the label. A placeholder in the negative
  prompt is still refused. The page refuses, in the comparison's preview
  and at Run comparison, a version or negative holding a `{name}` the open
  grid's variables declare (`compareUnfilled`, `declaredGridVariables`), as
  `generate()` refuses one; undeclared braces pass.

Jobs:

- The manager owns queued admission order. `active()` and both HTTP listings
  show running work first, then that order. Move to front changes only queued
  work and never preempts a holder. Loading stays outside the gate; admission
  is checked again after taking it. The page uses `setQueue` for the same
  order, preserves newer local submissions across delayed replies and keeps
  the focused row/action when reordered. A watched job that is queued has
  its own Move button beside progress.
- Skip names `current_call` in `POST /api/jobs/{id}/skip?call=N` and raises
  `Cancelled` cooperatively for that call only. Stale tokens are refused;
  Cancel wins, and a late Skip never discards a completed image or reaches
  the following call. Intentional skips are retained separately from failures,
  excluded from successful indices and recoverable by Finish. Skip is held
  between calls, while saving/paused/loading and during dynamic Then refine
  passes, which Finish cannot recover separately. The worker emits a compact
  status on ending a call so the page holds Skip before a slow save. Held
  controls keep their size, `aria-disabled` and the reason as their title.
- Cancel and pause are cooperative and polled: cancel from
  `callback_on_step_end` (raise `Cancelled`), pause at the cell boundary
  every quarter second. Never `Event.wait()`. The page answers a Pause press
  at once ("Pausing…"; a press on it withdraws the request, one on Resume
  reads Pause) and shows Resume only on the worker's `paused`, deriving the
  label from every status frame's `paused`/`pause_requested` but one that
  contradicts its own press while that POST is out (`pauseAsking`); the
  label is the state, so it has no `aria-pressed` (the toggle rule,
  Frontend). A
  running job of one cell has Pause `aria-disabled` with the reason. A queued
  job's pause is kept (`JobManager.pause`) and holds it before its first unit
  once its turn comes, keeping the gate; its cell line says "will hold before
  its first image" and Pause's title "Hold this job before it starts; Resume
  lets it run", never "pausing after this image". A pause stops a time
  budget's clock: `_hold` moves `Job.deadline` on by the time it held, before
  the resume's frame. `_hold` stamps `Job.paused_at` (Unix seconds) with
  `paused` and clears it with it; a job waiting behind a paused one carries
  that time as `ahead_paused_at` (`_wait_on`, wherever `waiting_on` is set).
  Past a minute the paused tab's cell line says "Paused after image 2/3 for 4
  min", its title "⏸ Paused 4 min", and the waiting tab's note "The job
  ahead has been paused for 4 min" (`pausedFor`), moved on by the status
  poll (`tickPauseClock`) and counted on the server's clock (the status's
  `server_time`, `serverClockOffset`), never the page's alone; a `note` that
  is a function is asked again at each redraw. A plain
  job's units are images, or batches where a repeat makes several
  (`unitWord`, from the snapshot's `grid` and `planned_images`; `Job.unit`
  on the server), in the
  progress, the finish line, the skipped units, the server's failure errors
  and the last-run banner ("stopped at image 2 of
  3", "(2/3 batches)"); "cell" is a grid's word. A finished job that made
  nothing holds the title "Nothing made", with no ✓.
- A job is a list of cells; seeds are the outermost loop, prompts next. A
  cell's failure skips the cell; three in a row or more than a third of the
  grid ends the job. Every terminal transition goes through `_finish`, and a
  budget stop returns through the cancel path rather than a fourth status,
  as does a shutdown's (`JobManager.stop_all("shutdown")`, from
  `shutdown()` and `Server.handle_exit`, `stopped_by` "shutdown"), which the
  page says as the server's stop, never as a cancel. `stop_all` finishes a
  job still queued at once, as `cancel()` does, through `_finish` on the
  jobs' loop (`call_soon_threadsafe`: the signal's thread), so a stream
  waiting on the load ends before uvicorn's drain.
  Each skipped cell is kept in `Job.failed_cells` ({label, seed, error}),
  sent only with the records (the full snapshot). When the failure rule ends
  a job, `stopped_by` is "failures" (status still `error`, through
  `_finish`) and its error reads "Stopped at cell X of Y after N cells
  failed; K images kept. Last: <short_error>" (a plain job's in images or
  batches, `Job.units`). The page's Retry ("Retry the
  N skipped cells") and Resume ("Resume the remaining N cells", for a grid
  `stopped_by` "failures" or "budget": N is `total_runs - (run - failures)`,
  since the budget is checked before a cell starts; a budget stop's title
  ends "under the same N-minute budget", the payload's `budget_minutes`
  sent again with it) send the
  job's own submitted payload (`submissions`) again with `resume: true` and
  `payload.seed ?? JobAccepted.seed` (none beside a seed list) - never the
  POST's seed over the payload's own, never the panel - each once ("Retry
  sent", "Resume sent", `aria-disabled`, focus kept). A sweep whose refine
  source has gone cannot run again: the press holds the button
  (`aria-disabled`, the sentence as its title, a press saying it again) and
  says "#N, this sweep's source, was deleted: the run cannot resume."
  (Retry: "its skipped cells cannot run again") - before sending where the
  page saw the delete (`deletedIds`), else on the server's refusal of that
  id (`startJob`'s `onRefused`).
- Each SSE stream has its own queue; `events()` opens with `done` for a job
  that has finished; the per-cell event and `/api/jobs` send
  `snapshot(images=False)`. Progress events carry `elapsed`; the page's
  estimate of the time left runs from the end of the job's first progress
  event once a second has landed (`etaFrom`; none before it: the first step
  carries a pass's VAE encode and the denoiser's move onto the card); `waiting_on`
  says `load`, or what holds the generation gate (`JobManager._holder`: a
  job's id, or `run_when_idle(fn, holder=)`'s "critic" / "expand", passed by
  the two routes, with the ticket an editor action was sent with,
  `_ticket`; `JobManager.card()` answers ("job" | "critic" | "expand" |
  None, ticket) for the status's `card_holder`/`card_ticket`): `queue`
  behind a running job, `paused` behind a paused
  one, `critic` or `expand` behind an editor action, and None for a queued
  job with nothing ahead. Every job past its load and waiting at the gate
  (`_at_gate`) is relabelled, with a status frame, whenever the label would
  change: when a job takes the gate (the second of two jobs queued during a
  critique said "critic" for the whole of the first's run), and from
  `_hold` after a pause and after a resume (`_relabel_queued`, on the event
  loop, which owns the job table). What a queued job waits on, in the
  page's words (the job ahead paused, another tab's critic or Expand, the
  first load), rides in `cellArgs` (`showCell`'s `note`), so the queue's
  redraw (`setQueue`) and a Pause press say it again rather than wiping it
  under a progress line still saying "waiting". A
  snapshot's `images` are the job's live records:
  the delete and bin routes call `jobs.forget_images` (`binned=True` from
  bin and Delete unstarred, which also sets a replayed pass's
  `refine_binned`), which takes those images out of every job's replay and
  counts them in `deleted_count` - gone from the gallery, deleted or binned;
  the restore route calls `jobs.restore_images`, which puts them back and
  clears a pass's `refine_missing`/`refine_binned`. Both run on the event
  loop. `image_count` stays what the job produced. Snapshots also carry `prompt` (its first 120
  characters), `refine`, `width`, `height`, `planned_images`, `grid`,
  `comparison` (`run_grid`'s `bool(prompt_variants)`) and `prompt_template`
  (a grid's prompt as submitted, its variables unfilled, from `/api/grid`;
  None for a plain request and a re-render), so a page that did not submit
  a job can name it - a grid by its own prompt, never its first cell's.
- `ModelStatus.active_jobs` (running first, then queued admission order) feeds
  `followServerJobs` on every status poll. A page watching nothing that sees
  a job it neither submitted nor watched sets `elsewhereBusy`: Unload and
  Expand are held aria-disabled with "A job started in another tab or
  window is running; Unload|Expand once it ends", and Ask the critic with
  "A job started in another tab or window is running. The critic needs the
  card to itself, so it waits for that job to finish." A press says it as a
  sentence until that job ends (`elsewhereNote`): in `#error`, and for
  Expand through `expandSay`. The topbar shows "· busy in another tab"
  beside the name while the model is ready (`#status-aside`, aria-hidden
  and outside the name's live region, hidden at 600 px and under), Generate
  stays live (the server queues it), and when the flag clears the gallery
  and the Runs picker catch up (`prependFinished(0)`, `loadRuns`) - never
  under a sheet (a run's or a search's), which would be laid out again
  whole at a moment no one chose; Sheet off resets the gallery, which takes
  the images in then. Where a job of this page's own took over while
  another tab's ran (`setBusy(true)` clears `elsewhereBusy` quietly), the
  catch-up is owed (`elsewhereCatchUp`). That job's end takes it, whatever
  the job made, a cancel before its first image included; a stream that
  gives up takes it too, and the server answering again clears it. It is
  never attached
  to: watching it would take this page's viewer. A busy page sets its queue
  rows to the server's jobs behind the watched one (`resyncQueue`, keeping
  ids it listed after the poll was asked, `queuedAt`) through `setQueue`,
  the rows' one writer, which keeps focus as a row's × does; a row this page
  did not submit reads "Another tab's job: …" ("Next, another tab's job: …"
  alone), its × "Cancel another tab's queued job: …", and Cancel queued's
  title says it takes another tab's too; a paused job's cell line counts the
  jobs waiting behind it. A row the poll adds or takes out waits while the
  pointer rests on the list or anything under it that it would move
  (`queuePointer`, from the page's pointerover: in the panel, and in one
  column the whole page below; never `:hover`), the latest change kept in
  `queueWaiting` and asked again as the pointer leaves (`releaseQueue`),
  and dropped by a poll that finds the queue back as it is shown;
  this page's own rows - a submission, an × - go in at once. A job `watchNext` picks up that this page did not
  submit, while its viewer holds this session's own results (`isForeign`),
  is shown in the progress row only - no `clearResults`, frame or stand-in -
  named "Another tab's job:" in the Now row, the mirrors and the title, and
  its end, "Another tab's job finished: N images.", is said once on
  `#job-said` alone, with no Use seed and no held title: the tab's own
  unseen outcome (`titleHeld`) gets the title back. The finish line stays
  this page's own last job, Use seed and all, with its lead as it was
  (`staleBefore`), and New image is left alone (a job withdraws it only
  once its first frame clears the strip, `landed`). A
  reload's attach, with nothing of this page's in the viewer, is shown as
  before. A cancel this page did not ask for (`cancelledHere`: Cancel, a
  row's ×, Cancel queued; a POST that got no answer is not one) reads
  "Cancelled from another tab or window — …" and holds the title "Cancelled
  elsewhere". `ownJobs` and `cancelledHere` are per-tab and kept across a
  reload in sessionStorage (`TabJobIds`: `auradream.own-jobs`,
  `auradream.cancelled-jobs`, read once at load, the newest 50, every
  access in try/catch).
- `ModelStatus.newest_image_id` is the gallery's highest id over every row,
  the bin's included (`storage.newest_image_id`: `MAX(id)` on the rowid, 0
  when empty), so only an insert raises it - a save, an import, a reindex;
  ids are AUTOINCREMENT and stay monotonic across a clear and a replaced
  database. The read waits `NEWEST_ID_WAIT` (0.2 s) for the gallery
  reservation and answers None rather than queue behind a bundle operation
  or a clear (`/api/status` is exempt from maintenance admission); the
  route reads `active_jobs` first, so a job missing from it has every image
  counted. `GalleryPage.newest_image_id` is the same number read under the
  page's own reservation (`storage.list_images_at_head`): the first gallery
  page sets `newestImageSeen` (state.js), never a status - a separate read
  could count an image saved after the page's SELECT. The page's
  `followGalleryHead` (job-queue.js) compares the status's with it on every
  status: a restart, the server back and a checkpoint change only set it,
  since that answer took everything in; a fall (a permanent delete of the
  newest, a clear) and None change nothing to show. Another tab's job's end
  (`setElsewhereBusy`, `elsewhereEnded`) catches up in the same status. A
  rise is decided later while this page submits or lists (`jobListing`);
  is explained, and only remembered, where every id up to it is one this
  page's watched jobs streamed or finished (`ownImageIds`, pruned at or
  below the value); is owed to this page's running job's end
  (`elsewhereCatchUp`; `noteUnseenImages(newest)` at once re-asks only the
  ladders read before `newest`, and the job's end, through
  `noteUnseenImages()` with no number, still drops every one); and
  otherwise - another
  tab's job running included, so a chain of its jobs comes in job by job -
  is taken in by `catchUpGallery(newest - seen, newest)`: `noteUnseenImages`
  (every ladder the lightbox holds that was read before `newest` asked
  again - each answer keeps its first page's `newest_image_id` as `head`,
  and one on its way that reads less is asked again when it lands -
  whatever the view takes in: a sheet, the bin, Starred and a search leave
  the images out of `fresh`), `prependFinished` with the rise as `made` so
  its page reaches back over an import to the head held (never under a
  sheet nor in the bin, cards held while the pointer rests on the
  gallery's cards, no duplicates: `joinFinished`), and `loadRuns`. A
  catch-up whose `newest` (`upTo`) the grid has read already
  (`galleryReadHead`, the newest id of this epoch's first page or a joined
  head page) fetches and resets nothing - a late status after a reset over
  those images; one that finds a reset's page on its way waits for it
  first, never resetting again over the line that reset's caller wrote
  (Delete unstarred's outcome or its 409). Where even that page does not
  join up, `prependFinished` resets the gallery only once the pointer has
  left the cards. A job queued, run and ended between two polls never
  showed in `active_jobs`, and its images otherwise stayed out of the
  gallery and the CFG and steps ladders.
- `main.Server.handle_exit` stops the active jobs (`jobs.stop_all("shutdown")`)
  and starts the critic's stop on the signal itself. The load, the critic's `prepare()`, Check's
  analysis and the critique's analysis and lint run on daemon threads
  (`detached.detached`), as does each tokenizer load, so interpreter exit joins none
  of them; a first use's tokenizer fetch holds a Check, and so a Ctrl-C's
  drain, for `prompt_encoding.LOAD_WAIT` at most. Lifespan stops the critic,
  then `jobs.shutdown()`, then calls `try_unload()`.
- `save_image` writes `.png.part` and renames after the row commits, retrying
  the rename; the orphan sweep leaves `.part` files younger than an hour.
  A stale `<name>.png.part` whose row exists and whose `<name>.png` does not
  is the only copy of a committed image (the rename failed, or the process
  died between the two): `storage._settle` renames it into place and
  discards every other stale partial (`settle_partials`, and `clear_history`
  before its thumbnail sweep). Never discard one; a failed rename leaves it
  for the next sweep. A stale `thumbnails/<name>.webp.part` is discarded even
  beside a committed row: `thumbnail_for` may have been cut short mid-encode,
  a placed thumbnail is never encoded again, and the card falls back to the PNG.
- Full-size PNGs stay in the output root; WebP previews live in its
  `thumbnails/` subfolder. `init_db` moves indexed legacy previews there,
  and `thumbnail_for` does the same during PNG-only recovery. A locked legacy
  file remains readable at its old URL; an existing destination is never
  replaced. Deletion, disk totals and stale-partial cleanup cover both
  locations. Read-only previews never create directories or move files.
- Gallery bundles cover all checkpoints and favourites, independent of the
  browser filter. Publish and verify the ZIP and its SHA-256 manifest before
  removing source files; recheck the source inventory and hashes before
  cleanup. Use SQLite backup for WAL databases. Report retained files and the
  durable bundle on incomplete cleanup. Cleanup takes SQLite's write lock
  (`BEGIN IMMEDIATE`) before removing any file, deletes rows only for files
  that really went, and reports `deleted_ids` only once that commits. Compact
  the cleared database while keeping its AUTOINCREMENT high-water mark: old
  browser ids must never name a new image after a clear. Import validates names, checksums, database rows,
  images and conflicts in staging before installing new files and replacing
  the database; references/refine ids and PNG metadata are remapped together.
  Never extract arbitrary archive paths or overwrite conflicting images.
  A present image is the same only by its pixel-deciding chunks or its full
  decoded identity (mode, size, palette, transparency, full SHA-256 of the
  raw bytes), never the 16-hex `content_hash`; a present image's thumbnail is
  kept whatever its bytes, a new image's is held to the bundle's. A known row
  whose file is missing is restored from the bundle (`restored`) only when
  the copy is plain RGB (no transparency) at the row's size, its pixel hash
  equals the row's `content_hash` and its identity metadata matches (the
  short hash covers raw bytes only); the row is left unchanged and its PNG chunk rewritten
  from it. Anything else is refused.
- A backup (`keep_gallery`, the dialog's Back up (keep gallery), operation
  `backup`) is an export that ends at the verified read-back: no typed
  count (`expected` is the clear's gate alone), no second inventory, no
  cleanup, `deleted_ids` empty and `kept` true. Never drop the read-back
  for it. A bundle of the picked `ids` or of the starred (`scope`) is
  always a backup; the route refuses one that would clear (422), and
  `export_bundle(ids=...)` refuses it too. Its ids come only from
  `submission.resolve_selection` (Starred over every checkpoint, no search,
  never the bin; picked ids whatever the filter, one binned or gone since
  in `left_out`), read before the reservation and refused at once while a
  bundle operation runs. Under the reservation the rows are cut from
  export's private copy (`_restrict`: DELETE, FTS `rebuild`, VACUUM, so no
  other prompt survives in the index or in free pages), held equal to the
  live rows, and only their PNGs and thumbnails are archived; a chosen row
  with no file refuses as `MissingImage`, an unchosen one does not.
  `links_outside` counts references and refine sources the bundle lacks;
  import clears and counts them as `unresolved_references`, said as a fact,
  not a failure. Every export checks free space (`_check_space`) for its
  files and database copy (with its `-wal`; twice for a chosen bundle's
  VACUUM) before it writes. A finished backup reloads the gallery views,
  as every bundle operation does, but keeps the typed count.
- Internal managed exports additionally require whole-gallery `keep_gallery`
  and strict gallery/destination/operation UUIDs. Their reserved manifest
  marker is verified during the existing read-back; a forward-only outer
  writer hashes ZIP bytes or ciphertext while writing. Bind the closed
  partial to the original writer identity before verification, publish
  without replacing an existing name, and clean a partial only while its
  device/inode still belongs to that writer. Append the database-specific
  atomic receipt journal only after verified publication under the same
  reservation. Journal failure leaves the archive unmanaged with a fixed
  warning and preserves prior receipts. Explicitly enabled startup backups use
  this path; separately consented v3 cleanup follows only a new success.
  `matches_current` checks regular-file identity,
  destination and full raw digest; its snapshot never authorizes deletion.
- Automatic startup backup defaults off. Only an explicit v2 save with complete
  age/count, displayed destination and fresh plaintext acknowledgement activates
  the next start; every older v1 draft remains inactive. Create durable namespace
  UUIDs only on explicit configuration, under reservation, and never adopt an
  existing foreign/damaged journal. Atomic saves use revision comparison and the
  reader's size bound. Cached GET status takes no gallery lock; explicit Reload
  reads bounded settings off-loop. Neither Save nor Reload creates an archive.
  A startup attempt reserves before lifespan yields, waits before optional
  preload and drains its owned writer at shutdown. Fixed errors carry no secrets.
  Older v2 settings remain export-only: their N is not enforced and they confer
  no deletion permission. No credential provider or automatic retry is installed.
- Retention requires explicit v3 configuration with fresh separate deletion
  acknowledgement, N/protection/scope and matching full installed shared mode.
  Save/reload never export or prune; cached GET is native-I/O-free. Disable clears
  permission without destination access, preserving local pending/unknown warnings.
  Only a fresh exact nonempty managed success can enter cleanup under the worker/
  gallery reservation. Protect new success and newest N current same-scope copies
  with shared claims; requalify older protected copies without fully rereading
  the new export. Manual/foreign/unknown/busy copies confer no deletion authority.
  Persist one exact intent before marking. Pending blocks cleanup/restart retry
  and is excluded from age/N. Known no-mark/refusal may restore only after normal
  entire bound close; accepted mark plus normal close retires without claiming
  physical removal. Unknown mark/cleanup/final-write failures preserve pending and
  stop. Journal v2 retains active/pending/retired proofs and all-record uniqueness/
  retry lookup within 20,000-record/16-MiB caps, with no eviction. Backup success
  and actual receipt time remain separate from cleanup warnings/counters.
- Managed export adds a final logical-byte capacity check before opening its
  partial: actual post-compaction snapshot and inventory sizes, exact manifest
  JSON names/sizes, maximal ZIP64 headers/descriptors, encryption framing and
  the bounded atomic receipt temporary. Group these by actual volume. Its
  early DB+WAL heuristic allows three equivalents for retained-cell VACUUM;
  allocation metadata, SQLite growth and other writers prevent a fit guarantee.
  Ordinary manual estimates and the single archive read-back remain unchanged.
- Saved archive downloads hold a per-process reader lease for the complete
  ASGI response, acquired before stat/open and released in finally through
  body/range/HEAD/error/cancellation/background paths. Disable pathsend handoff
  because its send completion does not prove transfer completion. Reader
  leases exclude an exclusive claim on the same resolved path; release is
  idempotent across threads. Ordinary bundle maintenance still permits these
  downloads. Default registries cover cooperating threads without sidecar
  writes. Explicit cross-process mode adds native shared/exclusive locks on
  persistent, identity-checked archive companions; never delete or replace
  those companions. Retain native ownership until the last local reader
  releases, including mixed-mode readers. Uncertain close retains exclusion;
  native access/identity errors never become an unlocked fallback. Shared-mode
  installation must be quiescent and complete across cooperating processes
  before any retention caller. Linux/fork and mixed old processes remain
  unqualified; no deletion caller is enabled.
- Shared download mode is installed only by the explicit Windows fixed-local-NTFS
  tool, independent of startup backup policy and N. The operator confirms every
  other server using that destination is stopped; this is not machine-verified.
  A common process guard refuses existing lease state and blocks new claims
  through immutable destination-bound marker publication. The default download
  registry discovers mode and acquires its reader atomically at ASGI entry,
  including responses constructed before cutover. Missing never-observed markers
  retain non-writing defaults; observed loss, replacement, named-path retarget,
  invalid/foreign data or native errors fail before archive headers/bytes. Never
  overwrite/adopt a bad marker or silently fall back. `--check` is read-only
  outcome inspection and makes no quiescence claim. Handle-bound mutation remains
  a prerequisite to future retention; v2 N is inert.
- Published archive imports take the destination gallery reservation before one
  source reader, held through verification, staging, merge and rollback. Listing,
  public encrypted readers/header checks and receipt hashing join the same
  mode-aware protocol. Keep native acquisition/release outside availability
  catches; wrong receipt scopes refuse before companion creation. Private owned
  uploads/export partials use explicit private seams, never a suffix-based or
  ambient bypass. Forward the read-owner capability rather than nesting claims.
  If an input close may leave a live file, retain its object and original reader
  claim until restart, without resolving a possibly retargeted name or retrying
  close. Preserve body failures with cleanup notes. No extra full archive pass
  or key derivation is introduced, and no retention deletion is connected.
- The Windows managed-deletion helper accepts only an exact receipt
  and expected scope/installed mode. Its caller must establish journal
  membership, gallery reservation, policy eligibility and explicit deletion
  consent. Require positive matching shared mode and native exclusive ownership;
  an ordinary optional-mode lease is insufficient. Pin directory and marker
  against replacement/writes. Open one archive READ|DELETE/share0 handle without
  delete-on-close, refuse links/reparse/directory/delete-pending states, and hash
  that same descriptor once immediately before final scope/identity checks and
  the native disposition mark. Never reopen/unlink by pathname. Mark only once;
  unknown mark or close retains original handles/exclusion until restart without
  retry. Distinguish accepted disposition and confirmed owned close from physical
  removal. The helper itself changes no journal or settings; only separately
  consented v3 orchestration may call it. V2 N grants no deletion permission.
- Bundles and the recycle bin: export carries every row, binned ones
  included (`preview()`'s `records` stays all rows, for the typed
  confirmation, and `binned` says how many are in the bin; the export
  result carries it too), and source cleanup deletes them with the rest,
  in `deleted_ids`. A new imported row keeps the `binned_at` it was exported
  with (through `_COLUMNS`; the ALTER fallback gives an older table TEXT); a
  known row keeps this gallery's state, as it keeps its star, and
  `binned_at` is not in the identity comparison. The import result's
  `binned` counts new rows arriving binned. A bundle from before the column
  imports live; `binned_at` is not in `reindex._DB_REQUIRED` and the
  manifest stays `version: 1`. Bin state is SQLite's alone, never the PNG
  chunk's: a PNG-only reindex, and an older build importing a newer bundle,
  bring binned images back live, as a PNG-only reindex loses a star;
  `reindex --database` carries `binned_at`.
- Every archive is read once: `bundle_archive._verify_archive` checks CRC, manifest SHA-256
  and (encrypted) every chunk tag, and tracked PNGs are decoded from the bytes
  it just read on the bounded decode pool, which also hashes them; errors are
  raised in row order. A present image whose pixel-deciding chunks equal the
  bundle copy's is not decoded, but its bundle row is still validated against
  the copy's IHDR size. Never add a second full pass, and never weaken a check
  for speed (no zlib-only check, no size/mtime in place of the pre-cleanup
  hash, no trusting the live `content_hash` for an existing image). Export's
  read-back reuses the writer's scrypt key only for its byte-identical header;
  import, upload and the password check always derive. Members whose
  destination exists are not unpacked: a present image is compared by its
  chunks, anything else (non-PNG) with its verified manifest size and
  SHA-256, and one that has vanished by installation refuses the import.
  Staged files are written, fsynced and closed off the reading thread
  (`_Stager`, bounded by count and bytes) and metadata rewrites run on a pool;
  every one finishes before installation, and on failure before the staging
  directory goes. Staged files are installed by an exclusive rename/link,
  never replacing one; the copy fallback writes `<name>.part` beside the
  target and only then claims the name. A tracked member over
  `_MAX_IMAGE_BYTES` (256 MiB) is refused before it is read into memory, and
  so is any member whose `compress_size` exceeds what deflate could make of
  its `file_size` (a whole `read()` fetches all of it in one call).
- Saved archives use `storage.bundle_directory()`: `AURADREAM_BUNDLES_DIR`
  when set, otherwise `bundles/` beside the database. Listing, downloads,
  export/import, the dialog's directory and archive `.part` cleanup must use
  the same location. Keep it outside the images directory and refuse a
  symbolic link or junction as the archive directory or an archive file.
  This setting does not move database work or image staging.
- Imports unpack, and uploads land, in `storage.bundle_staging_dir()`
  (`.auradream-staging` beside the images directory, or inside it when
  it is its own volume); the merged database is built in a `.bundle-merge-*`
  directory beside the live one. Free space is checked per volume. Bundle
  leftovers (`bundles/*.part`, `.bundle-*` work directories beside the
  database, staging entries) are stale partials: counted by `summarise`, and
  swept once older than `PARTIAL_SETTLE_SECONDS` at startup, at every bundle
  preview, before every bundle operation and by `clear_history`, always under
  the reservation - with the image and thumbnail partials, a committed
  image's PNG finished rather than discarded (`storage.settle_partials`). A
  `.part` the inventory still finds is a write in flight (or one another
  program holds open) and is refused with its name, its folder and what to
  do (`bundles._partial_refusal`). A row whose PNG is missing refuses the
  preview as `bundles.MissingImage`: "image #N (name) has no file in
  <folder>", with the way out (put the file back, or delete it permanently
  from the recycle bin, binning it first); its 422 carries `image_id` and
  `binned` beside `detail`, which the dialog's inline "Show #N" button reads
  (`api()` keeps the error's `body`) to close the dialog and search the
  gallery, or the bin, for `id:N` on every checkpoint, Starred off.
- A password bundle (`.ldbundle`) is the ordinary ZIP streamed through
  `bundle_crypto`; no plaintext archive ever reaches disk, and export verifies
  it by decrypting before anything is cleared. The password is never stored,
  logged, put in a URL or echoed in an error; a raw upload carries it
  base64-encoded in `X-Bundle-Password`. Every route's password passes
  `api_bundles._password` (length, and a UTF-8 form: JSON can carry a lone
  surrogate), refused with a fixed 422 before any check or worker;
  `bundle_crypto._secret` raises `CryptoError`, never a codec error that
  names a position. Header parameters are untrusted and
  bounded before key derivation. A wrong password, a bad header, or a
  truncated or extended file is refused before anything is staged; any other
  altered chunk is refused when it is read, before anything is installed, and
  the staging directory is discarded. Never add an up-front pass to
  authenticate every chunk.
- The dialog separates **Create a bundle** from **Import a bundle**.
  New password/Repeat apply to either creation action. Back up is the primary
  creation action and requires no typed count; the separate clear group names
  its safety purpose, entire-gallery scope and exact current record count,
  including all models and the recycle bin. That count is a confirmation,
  never an export quantity. A held clear press focuses a missing/wrong count
  only when that is its remaining gate, revealing it inside the dialog if
  clipped; all server safeguards stay in force.
  File and saved imports share one **Import password** field within the same
  import section. Protected rows visibly say **Uses Import password**, with
  keyboard navigation to that field; pressing an import held only for its
  password goes there too. There is no per-row secret. Feedback stays between
  file and saved controls, named Bundle operation status, so import failures
  and recovery remain near their actions. Downloads need no password and
  protected files stay encrypted. Secret lifetimes remain as described below.
- An upload whose body stops early (the page's Cancel, a reload, a closed tab:
  `ClientDisconnect`) or whose request is cancelled fails the reservation with
  `api_bundles.UPLOAD_STOPPED` and removes its `.bundle-upload-*.part` before
  anything is imported. `BundleManager.fail` never stores a bare class name,
  and stores a worker's lowercase `BundleError`/`CryptoError` clause as a
  sentence (`bundle_common.sentence`, as the password check's 400 and
  `main.invalid_bundle`'s 422/409 detail say it).
  The page's Cancel aborts only a request still being sent; once the server
  answers 202 the import is the server's. The route checks for a disconnect
  after the body ends, but a Cancel pressed after that check and before the
  202 arrives cannot stop the import, so the page says only "Upload
  cancelled." and leaves "nothing was imported" to the server's status.
- The dialog acts on a finished operation (forget deleted ids, reload the
  gallery) only when this page started it or saw it running. A result the
  page's first status already holds ended before it loaded: it is shown as
  "Last bundle operation:" and touches nothing; a result already read in the
  open dialog is labelled the same way when it is reopened. Export clears its
  password fields when pressed; the unlock field is kept after a failure (to
  retry) and cleared once an import is done. `#bundle-status` is the live
  region and says only the operation's sentence; phase and counts go in the
  non-live `#bundle-progress`.
- A bundle operation reserves HTTP admission before awaiting work, only when
  jobs, a model switch and existing gallery requests are idle (a load alone
  does not hold it: it never touches the gallery). Count complete
  image response streams. The reservation outlives request cancellation and
  lasts until its background worker finishes; graceful shutdown waits for it.
  `gallery_lock` also covers whole storage/reindex writes and SQLite sessions
  across processes. SQLite sessions and per-record writers (`save_image`,
  `thumbnail_for`, `import_file`, `index_file`, `set_favorite`,
  `set_favorites`, `delete_image`, `bin_images`, `restore_images`, `init_db`) wait up to
  `gallery_lock.SESSION_WAIT` for another process's reservation; whole
  maintenance operations (bundles, `delete_matching`, `empty_bin`,
  `clear_history`, `vacuum`, reindex/merge) refuse with `GalleryBusy`
  after `MAINTENANCE_GRACE` (0.25 s) of polling for the file lock, long
  enough to ride out another process's read (a gallery page, the status
  poll's newest id every 3 s per open tab), never queued behind a real
  holder. None of the bin's routes is in `GalleryAdmission.control`:
  a bundle operation refuses them, and they hold one off. The wait covers the in-process lock too (one
  deadline in `hold`), never `wait` per thread queued behind it; `jobs._save`
  retries a `GalleryBusy` save until the job is cancelled, and `generate.py`
  until a second Ctrl-C (saying once "gallery busy in another process; still
  waiting to save"; its `--resume` lookup refuses in one line, exit 1), never dropping a
  finished render. `save_image` and `thumbnail_for` encode their
  `.part` files before taking the reservation and reserve only the row and
  the renames. Keep lock ordering consistent: gallery reservation before
  storage's write lock. Status and bundle downloads remain available.
- The preview callback (`_step_end` -> `latent_preview`) never raises - an
  exception there is how `Cancelled` ends a job - and `jobs.on_preview` checks
  for subscribers per frame. A preview needs a matrix for the VAE and either
  `_unpack_latents` or already-unpacked 4-D latents.
- Generation is serialized through one worker thread; the GPU cannot usefully
  run two. Interactive accelerator work - ERNIE's Expand, the prompt critic -
  goes through `jobs.run_when_idle`, so it neither cuts ahead of queued images
  nor overlaps a newly submitted job; it checks the queue again once it has
  the gate. The critic's download and its rewrite's lint run outside the
  gate. Nothing that holds the gate awaits the load lock, which is why
  `wait_for_gate` cannot deadlock.
- An instruction edit queues only through `POST /api/instruction-edit`
  (`InstructionEditSubmission`). The submission is strict and refuses extra
  fields such as a negative prompt. It becomes one image and one run. Its
  admission order is fixed:
  1. the model-dependent middleware (409 for a switch or a stale revision);
  2. the selected loader and default execution;
  3. `edit_unavailable_reason` (a 422 before any file is read);
  4. `bind_request` in a thread, which proves the pixels, model and geometry;
  5. `check_budget`, `check_refine_steps` and `jobs.submit`.

  The route never asks `refuse_unusable_inputs`, whose text-to-image check
  refuses everything on the editor. `submission.unusable_input` still binds
  an edit, then answers UNAVAILABLE, or REPLAY_HELD with the gate open, so
  Generate, Re-render and More seeds never queue one. `submit_cells` takes
  only a bound edit, and `_run` fails a retained edit job when the backend is
  not qualified.

  Each edit cell runs `_recheck_edit` as it begins, after `begin_call` and
  before `generate`. It asks the gate again and re-proves the source with
  `bind_source`. A source that is missing, in the recycle bin or rewritten
  since the submit fails that cell in words, with no backend call and no
  row. `planned_keys` skips edit cells while the gate is closed.

Prompt analyser:

- Three layers over the body Generate sends (`PromptAnalysisRequest`): Check
  (`POST /api/prompt/analyze`), the critic (`POST /api/prompt/critique`) and
  the comparison grid (`prompt_variants`). Check works cold: CPU tokenizers
  and rules on a `detached.detached` daemon thread off the loop, never
  `ensure_loaded`, never the gate, `applied()` from the profile before a
  load. The negative prompt is measured and checked only as `applied()` runs
  it. `POST /api/prompt/measure` is its encoder layer alone
  (`prompt_encoding.measure`, no rules, no tag lookup), on the same terms
  and the same 600 ms pause, on every checkpoint. It serves the chip tray's
  token count and cut and, where no chips show, the line under the prompt
  box (`#prompt-tokens`; `syncPromptTokens` is its one writer). That line is
  hidden under three quarters of the tightest encoder's budget; from there
  it gives the count and every encoder that is as tight, and past the limit
  it is amber and names where reading stops (`tagCut`, the excerpt Check's
  cut link gives). An encoder left uncounted (a tokenizer still loading or
  unfetchable, the server not answering) is "cut unknown" in either place,
  never "no cut". The page measures the first 4000 code points
  (`PROMPT_MAX_CHARS`, tag-tokens.js, held equal to `PromptMeasureRequest`'s and
  `GenerateRequest`'s `max_length` by `tests/test_api_analysis_tags.py`). Past
  them the line leads with "Over 4000 characters: Generate and Check refuse
  it", in amber, then the count of what was counted and its cut (the chip
  tray's label says "over 4000 characters", the count of the first 4000
  after it), and the summary line says it once; a refused measure had read
  "Tokens: cut unknown" and taken the warning back. Where the encoder table has no entry
  for the checkpoint (`PromptMeasure.encoders_known` false) nothing will
  ever count the prompt, so the line stays hidden; Check's Encoders section
  says why. A measure that got no answer is asked again for the same text;
  `refreshStatus` asks again when `loadedModel` changes and when the server
  answers after an offline spell. The line is no live region: a change to
  past the limit is said once, in `#tag-summary`, and cleared on the way
  back under. It is inside `.prompt-editor`, under the box at the field's
  6 px gap, so the Example's Undo note, the Tags helper and the suggestions
  hang under it; a click on it closes the suggestions. Whether it shows
  moves everything under it in the panel by a line, so while the mouse is on
  a control at or below the prompt box's bottom (`promptTokensAimed`, set
  from the panel's pointerover) that change waits (`promptTokensHeld`) until
  the mouse leaves that control or the panel, or a key is pressed; its text
  and colour change in place. The Refine dialog checks its own request
  (never `collect()`) on Check's route (`POST /api/prompt/analyze`, the
  request without `strength`, `strength_axis` or `seeds`) under an epoch of
  its own
  (`checkRefinePrompt`): the encoders' non-pooled cuts with Show, and rule
  findings' whole-field Apply written through `replaceBoxText` into the
  dialog's boxes, never the panel's; stale while either box's text or the
  negative's activation differs from what it read, a negative-prompt Show or
  Apply dead (saying why) while the dialog's negative box is greyed; never a
  live region itself - a visually hidden polite line (`#refine-check-said`)
  carries its text, written only when that text changes; it never holds
  Refine and is silent on failure.
- Every finding says what it rests on: `source` (`guide`, `author`,
  `encoder`, `heuristic`), `cite` and `url`. Rule config lives in each guide's
  `analysis` key, never a profile key or `guide_for` (which passes on only
  the placeholder rule's patterns, as `guide.slots`, for the Examples menu);
  a `guide` cite is a verbatim substring of that guide's `prompt` prose and
  an `author` one of an example (`test_every_cite_is_the_guides_own_words`).
  An `author.placeholder` finding quotes its own slot: the entry's cite
  where it shows the slot, else the shortest example line holding it
  (`_slot_cite`; `test_each_slot_cites_the_authors_own_words`), and says
  what to put there (the entry's `fill`). One span, one finding:
  `prompt_check.checked` drops an `encoder.weighting` finding on exactly an
  `author.placeholder` span. The page's summary says characters for Han and
  kana (`PromptAnalysis.characters`, of `words`; `wordsLine`). The style
  note counts the pieces the style is read from (commas, "，", line breaks;
  lettering one piece), three or more. A check no author
  sentence supports is `heuristic`, with no cite. `documented_order` is None
  where the guide states none; never invent one, and never say the authors
  give none - the page says the app checks none and shows the authors' own
  (`Excerpt.order` -> `author_orders`). `order_checked` is True only where a
  rule placed the prompt (a tag checkpoint's comma items, each stretch of
  lettering one item with its commas: not XML or JSON, at most 256 items, and
  not where a line break or a fullwidth comma splits an item outside its
  lettering - multi-line prose included; `_placeable` is the one predicate
  `order_checked`, `reorder_tags` and the order rule share); the page says a
  documented order it was not is "Not checked". A rule's `suggestion` is a
  whole replacement for its field, offered only where it is unambiguous
  (lettering whose end is a guess gets none); an encoder's is advice text.
  `prompt_rules.QUOTED` is the one reading of quoted text, for the rules, the
  encoder layer and the critic; its single-quote pair may touch Han, kana
  and Hangul but no other letter or digit (写着'营业中'的 is a quotation,
  animal's an apostrophe). `prompt_rules.rendered_targets` gives each place
  the text asks for lettering as (start, end, ends, head): `ends` is False
  only where the rules stopped at a guess and the lettering may run on - a
  joining or capitalized stop word, six words, the 12-character cap, or
  lettering going on past a space or an ASCII mark (写着OPEN NOW, 9:00, www.shop.com) - and only
  there may the critic read on past `end` in any words of its clause; past
  an end the rules are sure of, it reads on only in words that start with a
  capital or a digit. The rules stop a participle's lowercase words at a negation
  the negation rule reads an absence from there, other than one the words
  before make their own - right after "or", "than", "but", a verb it negates
  or a verb of saying, or after a negation among them (`_absence_at`,
  `_owns_negation`, so "reading do not enter", "reading now or never in
  bold type", "reading we are not open" and "reading no shoes no shirt no
  service" are whole, and "reading love without limits" quotes "love" and
  warns on the rest); `rendered_targets` reads them on
  through it to where they stop without it, `ends` True ("love without
  limits", and "above the stage" after it is description), with `head` at
  the rules' stop (`end` elsewhere): the critic takes as lettering only
  words that start before `head`, so "without limits" alone is the
  absence, not a piece of the lettering. It is not the Apply's
  `_Target.exact`: a Chinese target stopped at 的/之, at 等 before 字样, or
  at a verb placing the sign (挂在, 位于) that no request or prohibition asks
  for gets no Apply, and neither do lowercase words after a participle, yet
  what follows them is description; nor do words not in capitals after a
  noun trigger ("with the words coming soon"), whose `ends` stands. After
  a noun trigger, a lowercase participle - irregular too ("set", "lit") -
  with a place word after it, or an -ing one the clause ends on or an
  adverb follows ("flickering softly", not one of when: "opening
  shortly"), is no lettering at all.
  `rendered_spans` is the same list without the flag or `head`; the rules'
  own reading of lettering (`_read`, and `_tag_items` outside `check`) keeps
  the negation's stop. The
  rules read each stretch of lettering, commas and all, as one comma item
  (`_units`): it moves whole and is placed by its shape alone, is never
  matched or rewritten as a tag (`_Context.norms` is None for it), is never
  asked of the tag index, and a reorder that would change what `QUOTED`
  reads is not offered. English punctuation inside times, prices, URLs or
  capitalized lists makes the quote boundary uncertain too: no partial Apply.
  NetaYume's prefix rule searches `ctx.bare` so prefix-shaped lettering is
  preserved and does not hide a genuine duplicate later in the prompt. It
  matches the loader's instruction in any case and spacing, its full stop
  optional. Where no such instruction precedes the first `<Prompt Start>`,
  and the text before the marker is one sentence (`_MORE_SENTENCES`) with no
  lettering that opens as an instruction does (`_INSTRUCTION`), that text
  and the marker are one span, worded as another workflow's instruction. An
  instruction opens "You are a/an", up to three words, then assistant,
  model, AI, generator or bot ending the opener: a stop or the end after
  it, or designed, that, which, who, built, trained, made, specialised,
  capable, for or to. A scene that opens "You are" ("You are looking at…",
  "You are a model student…", "You are an AI artist…"), or more than one
  sentence before the marker, loses the marker alone.
- `prompt_encoding` measures over the exact string the pipeline builds, and
  every template string, length and `enable_thinking` switch it copies
  (`encoder_table`) is held equal to the installed diffusers source (and the
  repo's Anima and NewBie pipelines) by `tests/test_prompt_encoding.py`. The
  table (`encoder_table.encoders_for`) is first-match like `_PROFILES`, exact
  ids first, reusing the loaders' pinned constants.
  Tokenizer files only. Each load runs on a daemon thread outside the cache's
  lock (`_start`/`_fetch`); concurrent checks share the load in flight until
  an `unload()` forgets it, and a load is kept only if it succeeds with no
  `unload()` since it began. A check waits `LOAD_WAIT` (30 s) for its loads
  at most, then reports the tokenizer as still loading while the load goes
  on; `unload()` never waits on a load. `measure` never raises, and
  `encoders_detail` says what went unchecked, each encoder under its own
  reason. Never from `/api/status`; `prompt_encoding.unload()` runs on
  Unload and shutdown. `\(`/`\)` escapes are not flagged
  (`prompt_weighting`): the authors' examples and pasted A1111 prompts carry
  them, though the autocomplete writes parentheses bare.
- Tag categories for the order and artist checks come from the chips' own
  lookup (`tag_lookup.lookup_tags`), only on a Danbooru-vocabulary guide or one whose
  rules read artists (`wants_artist_categories`: Anima, no cutoff, and
  RouWei), and only for the items `prompt_rules.lookup_items` names: a tag
  or hybrid prompt's `category_items`, or - where the rules read artists -
  a prose prompt's: its tag-shaped items alone where no order places by
  category (RouWei), and every `category_items` item, clauses included,
  where one does (Anima, `prompt_order.places_by_category`), since `_place`
  takes a clause the lookup answered nothing for as prose once any
  categories came back. The categories an order places with are therefore
  None or a lookup of every item a category could place
  (`test_prose_is_looked_up_only_where_the_rules_read_artists`); where an
  order stands without the artist rule, prose is never looked up. Capped at the chips' 64 items; any failure there is
  `None` - order placed from the guide's own groups, never guessed - not an
  error. What that cost is said in `PromptAnalysis.categories_detail`: the
  index unreachable (a connection error), no usable reply from it (an HTTP
  error, or a cut or unreadable body), or items past the 64. Whether a
  placed order was only partly checked is one predicate, the placement
  itself: `order_checked` and `prompt_rules.places_every_item` false with
  the analysis's categories - some item the order rule reads kept every
  slot where that decides the order (`unplaced_items`: some slot puts it
  out of order beside another item - not its own copy, written the same and
  so in the same slot - and some does not; a lone item, or one whose every
  slot gives the same answer, is not partly checked): never
  looked up for any of those reasons, a tag the index answered for with no
  usable category (unknown, deprecated, past the cutoff), a prose prompt's
  clause that is not a whole sentence (prose is never looked up where no
  rule reads artists), or lettering. An @ name on Anima is placed in the
  Artist step by its prefix (`_OrderSpec.prefix`, from `analysis.artist_prefix`
  where the order has an artist category), never looked up. A grid placeholder whole (`{hair}`,
  `prompt_text.is_grid_variable`) is none of these, and nor is a sentence
  where the guide's order sets `free_sentences` (Anima's alone: "You can
  mix tags and natural language in arbitrary order"; Kohaku's prose keeps
  its "general tags or prose" slot): either takes every slot
  (`prompt_order._anywhere`), is never looked up or listed unplaced, keeps
  its position in Check's reorder (the rest ordered round it; the critic's
  `placed_partly` is true beside it, `holds_grid_variable`), and a grid placeholder's chip is the grid's (code
  `grid_variable`, no verdict; its tip names the open fold's values and
  their verdicts). `PromptAnalysis.order_detail` is set exactly there, naming
  those items and why; the page then shows "Partly checked: <order_detail>" under the
  chain and "order partly checked" in the summary, never from
  `categories_detail`, and the critic's reorder other than Check's can stand
  exactly there (the critic's answer bullet below), since Check's own order
  is then one the documented order allows, not the only one. Where the
  guide states no order (RouWei), `categories_detail` alone makes the page
  say not every tag was checked for an artist name ("artists not all
  checked"); where the lookup gave nothing
  (`PromptAnalysis.categories_unread`: unreachable, no usable reply) it says
  no tag was ("artists not checked"), and the empty line above that note
  does not repeat its reason. The artist rule reads each comma item that is
  no sentence or lettering, whatever the whole prompt reads as, and names
  several with one verb, each once ("wlop and kantoku are artists"). The
  reported `style` is `detect_style(prompt, categories)`, the one the rules
  applied.
- The chips' order slots come from that same placement.
  `POST /api/tags/validate` takes an optional `prompt` and answers
  `order_steps` and `slots`; a blank prompt, or one longer than
  `TAG_SLOT_PROMPT_MAX` (Check's 4000), counts as no prompt, never a 422
  that would leave the chips unchecked. `slots` covers every comma item of
  the prompt, past the lookup's 64 too, each with the step
  `prompt_rules.order_slots` places it in - through `_places`, the one
  placement the order rule, `reorder_tags` and `unplaced_items` read, with
  `tag_lookup.tag_categories`' categories for that prompt. Those come from the cache
  the chips' lookup just filled, except names past the chips' 64 that
  Check's own 64 reach (guide tags are not asked about): the validate route
  fetches those once, as a Check would, and caches misses. Offline the
  categories are None and the index is not asked again. The route asks
  `prompt_rules.places_items` first, so a prompt it is false for costs the
  index nothing; where it is false (`order_checked` false, or no items)
  `slots` is None for the whole prompt. A lone item is placed, though no
  order rule or reorder reads one (`reads_order` keeps its two-item gate):
  the Tags helper's caret line needs its step. An item's slot is None
  where the item holds lettering or keeps more than one slot: an @ name, an
  item never looked up, or one with no usable category. A failure placing
  the slots costs the slots, never the verdicts answered beside them. The
  page never places a chip itself.
- Hybrid tag/prose presentation is separate from lexical validation.
  `chip_verdicts` adds `TagValidationResponse.parts` only for a guide whose
  helper explicitly sets `hybrid_prompt`, with a whole raw `prompt` whose
  comma items match the request's tag prefix. A capped prefix and grid-value
  lookups appended after all prompt items are allowed; missing, mismatched,
  XML/JSON or tag-only context supplies no parts. Strict `results` and the
  dictionary cache never change with the prompt context.
  `prompt_parts.contextual_parts` returns `kind` (`tag` or `prose`), inclusive
  original comma-item indexes `first`/`last`, and exact raw `start`/`end`
  offsets in JavaScript UTF-16 units. A prose-shaped dictionary miss anchors
  adjacent absent fragments as one passage: `1boy, masterpiece, a man
  wearing a red, old hat` has two tag units and one prose unit containing its
  internal comma. This does not depend on the whole prompt's majority style.
  Known/model tags, concrete deprecated/empty/no-date/cutoff findings,
  explicit artist prefixes (including malformed ones) and grid variables
  bound passages; quoted lettering stays text even when it contains a model
  tag. Unchecked items beyond the lookup cap or during an outage provide no
  evidence of absence; definite sentences and quoted text need none.
  Check's order finding's Apply suggestion, its `reordered`, and the critic's
  reordered candidate are withheld only when that candidate would split a
  multi-item prose passage. The finding remains advisory with no Apply;
  `PromptAnalysis.reordered_detail` and the critic's `set_aside` explain the
  app's precaution, without changing the author's order or cite.
- The critic is pinned: llama.cpp `b11206` archives (`critic.BUILDS`: Windows
  CUDA 13.4, and Ubuntu for WSL) and `Qwen/Qwen3-4B-GGUF` `Qwen3-4B-Q8_0.gguf`
  at a fixed revision, each held to its size and sha256 (`.part`, verify,
  extract beside, rename; traversal refused; a tree whose marker is missing
  or does not decode never used).
  `AURADREAM_CRITIC` (default on), `AURADREAM_CRITIC_SERVER` and
  `AURADREAM_CRITIC_MODEL` (absolute, resolved only on the server's
  command line: a link keeps the name answers are labelled with; neither
  held to the pins). The server setting should name llama-server itself: a
  wrapper is stopped with its tree, but a server it moves out of that tree
  (on POSIX, into a session of its own) escapes the stop, and on Windows and
  Linux the listener check refuses its port.
  `supported()`, `ready()` and `describe()` are file checks (and
  `supported()`'s one libc call), fit for `/api/status`: without a server of
  one's own, `supported()` is False where the Linux build's glibc 2.38 is
  missing. `describe()` (`ModelStatus.prompt_critic`) gives the page the
  model's label - the answer footer's name - whether it is the pinned pair,
  and what a first use would download; the line under the button promises
  the measured time and the GPU for the pinned pair only. For a pinned build
  `prepare()` runs `critic.preflight` ahead of any download - glibc, then
  libgomp.so.1/libssl.so.3/libcrypto.so.3 asked of the dynamic loader in a
  child interpreter - and refuses (503) naming the apt packages.
  `install_server` sweeps, under its lock, the work directories interrupted
  installs left (`critic_runtime._is_work`): exactly `<target>-` plus tempfile's 8
  characters (`[a-z0-9_]`), never another build's own directory, a real
  directory (no link or junction) holding only `tree` and `archive-<n>`
  (whole names) - never a user's copy named like one; best-effort, it never
  fails an install. A cached GGUF off its pin is downloaded again once
  (snapshot entry removed, `force_download`). A re-download that fails is a
  `CriticError` (502) quoting the Hub's reason; one still off its pin names
  the file and says the pin or the download path is wrong, never
  "delete it and retry".
- The critique route: 422 unsupported (`critic.pinned_build_refusal()` names
  the glibc a published build needs, where that is why `supported()` is
  False), 409 while a job is queued or running or the image model loads
  (checked before `prepare()` and again after it),
  `prepare()` on a daemon thread outside the gate, then `critique()` inside
  `jobs.run_when_idle`. One `llama-server` per call on 127.0.0.1, a free port
  and a fresh API key passed as `LLAMA_API_KEY` in its environment (never on
  the command line, which other local users can read), refused when
  `/v1/models` answers without it or when the port's listener is not its own
  process tree (`_listener_is_ours`; Windows and Linux, unchecked elsewhere).
  Only then is it asked to count the question itself (`_counted`:
  `/apply-template`, `/tokenize`); `check_fits` refuses the question unasked
  on that count, and `max_tokens` is sized from it. `estimate_tokens` only
  refuses, scaled by `_ESTIMATE_FLOOR` before a server starts, what cannot
  fit however it tokenizes, and sizes the question for a server that will
  not count. The server is stopped (terminate, wait, kill) as a whole tree in
  `finally`, before the gate opens (`_ServerProcess`): on Windows a job
  object, joined while the child is suspended and waited for by each
  process's handle; on POSIX its own session, and on Linux every process in
  that session. A spawn failure and a kill not waited out are `CriticError`s.
  `_process` is cleared and the log removed regardless (retried, and only
  logged if still held).
  `prompt_critic.unload()` on Unload, in lifespan before `jobs.shutdown()` and
  on a thread of `handle_exit`'s own, each through `main._stop_critic`, off
  the event loop: a `CriticError` (a kill not waited out) is logged and the
  rest still runs - Unload still drops the tokenizers and answers with the
  status, whose "unloaded" is the image model's and whose VRAM is read off
  the image model's card (the server is pinned to no card, so only on a
  one-card machine does what it still holds show there).
  `CriticUnavailable` is 503, `CriticError` 502; `api_prompt.prompt_failure` sends
  their messages without the class-name prefix (the log line keeps it).
- The critic's answer is checked before it is shown (`critic.validate`):
  `cite_id` is an enum of the numbered G and A sources in the answer's JSON
  schema and is resolved server-side (an unknown id drops the issue, counted
  in `dropped`); a quote is located in the prompt as whole words
  (`critic.locate`: never inside a longer word, except in Han, kana, Hangul,
  Thai, Lao, Myanmar or Khmer, with a tag's underscore as a space; curly and
  straight apostrophes, quotation marks and dashes folded one for one) or
  nulled; a part must be the prompt's own words and `text_to_render` its
  lettering (`prompt_rules.rendered_targets`, starting before a target's
  `head` and running on past an end the rules are sure of only in words
  that start with a capital or a digit);
  issues restating a rule finding reported on the prompt are left out - by
  wording, or by a quote that takes in a finding's words and runs at most 8
  characters past them (none past a single word), on guide, author,
  heuristic and weighting spans only, never a cut, template tail, unknown
  characters or control token - and one restating a negative-prompt finding
  only when it names the negative prompt (`reported_negative`;
  `guide.negative_prompt` only when it says one is missing, not running or
  recommended, not when it moves a word there; `guide.quality` by its
  finish, tier, year modifier or the recipes' tags); so are issues
  speaking to a length the rules measured in range (`measured_length`, not
  XML or JSON; `_says` reads a sentence at a time and skips lettering and a
  count worded as some part's - "at least 5 words for the lighting", "only
  3 words of the prompt are about the lighting", "describe the umbrella in
  under 5 words", "the lighting has only 3 words", "keep the lighting
  description under 5 words": `_SOME_PART`, `_MANNER`, `_PART_COUNT` -
  unless it names the prompt, its description (the word just ahead of
  "description" one of `_DETERMINERS` or `_WHOLE_WORDS`: "your
  description", "the revised description"; any other word there names a
  part, "your lighting description"), "it", a total, or the guide or its
  range as such, or fits them: "to reach the range", not "the dynamic
  range" or "to match the lighting"; a heuristic, every word of it pinned by a test row), an order issue the documented order reverses (a part first or a
  "put X before Y", `_reversed`; a negated instruction wants nothing), and
  order issues that cannot all hold.
  `reordered` survives only under a documented order and as a pure reorder
  of whole tags, clauses, sentences or XML blocks with quoted text unchanged
  (`is_reorder`, typed marks folded; every `rendered_targets` span is kept
  whole during piece splitting, even when tag order is unchecked, and initials
  or common abbreviation periods never split a name; for JSON, equal values
  with the keys sorted, each number as written - 0.50 is not 0.5 (`_json_forms`) - and no
  repeated key). On XML it never puts or keeps the guide's scene block ahead
  of a `<character_N>` block (`prompt_rules.scene_ahead`), the order the guide
  states, though Check reads no XML order. What is kept has the prompt's own pieces, each as the
  prompt writes it, in the critic's order with the critic's space and
  punctuation between them (JSON keeps the critic's text, whose values are
  then the prompt's writing). Where Check placed the tags
  (`order_checked`) it stands only as the rules' `reorder_tags` text -
  unless the rules could not place some item the order rule reads where
  that decides the order (`prompt_rules.places_every_item`, placing with
  the analysis's own categories, `PromptAnalysis.categories`, excluded from
  the response: an item never looked up or one the tag index had no usable
  category for - decided from the placement, never from `categories_detail`, which says only that some item
  was never looked up; the same predicate sets `order_detail`, so the page's
  "Partly checked" and this exception cover the same prompts). There it
  stands where it keeps the items the order rule reads whole
  (`prompt_rules.order_items`: lettering one item, commas and all), the
  rule reads it, and the rule, placing with the same categories, flags
  nothing in it (`prompt_rules.order_finding`).
  `PromptCritique.reordered_kind` says which was kept (`check`, `allowed`,
  or `unchecked` where Check reads no order); the page's Use reordered title
  follows it, and the page has no reorder rule of its own. A rewrite wrapped
  in quotation marks (sentence punctuation after the closing mark included,
  spaced or not: a stop kept, a comma, semicolon or colon dropped) is
  unwrapped; retained whole tags keep their original parenthesis escaping
  before the unchanged check, so removing those escapes alone is no rewrite.
  The critic is told that character and series names are optional: an order
  template positions supplied parts and never requires inventing identity.
  One that is the prompt unchanged (whitespace, a closing stop and typed
  marks aside) is dropped; one quoting words the prompt lacks is dropped; one
  quoting its description is flagged. What was left out besides `dropped` -
  the issues above, named parts that failed, a rewrite over-length,
  unchanged or quoting words the prompt lacks (and a wrap taken off), a
  reordered version too long, changed (an XML candidate with the prompt's
  own words says instead that it moved words inside or between blocks,
  changed only punctuation or spacing, or changed the text outside the
  blocks - `critic_reorder._not_a_reorder`), with `<general_tags>` ahead of a
  character block ("puts", or "keeps" where the prompt already had it
  there), not Check's, regrouped, unread or flagged by the order rule - is
  said in `set_aside`. Typed Qwen3 control
  text (`<|...|>`, `<think>`, `<tool_call>`, `<tool_response>` and closers)
  reaches llama-server defused by a zero-width joiner after the `<`, which
  `validate` takes out of the answer. The rewrite is linted by layer 1 in
  the route after the gate opens (`prompt_check.lint`, no tag lookup of its own: where
  the rules read artists it gets the analysis's artist categories), since a
  lint can fetch a tokenizer. `PromptCritique.rewrite_artists_checked` says
  whether that check read every artist the rewrite may name
  (`prompt_rules.artists_read`): every item a Danbooru name could be
  (`category_items`, a prose rewrite's clauses included) was among the
  analysis's `categories_asked`, excluded from the response, and no artist
  that answer named stands in the rewrite other than as a whole item the
  rule reads or after its @ ("in the style of wlop" keeps the hedge). What
  survives is still the model's suggestion; the docs
  must not call it trusted. Reasoning stays on. A G sentence listed
  `unsourced` in `prompt_sources` - the app's own menus, buttons, sampler,
  what it does not implement, or its own judgement - is never citable and
  no rule entry cites it; how the pipeline reads the prompt (a token limit,
  held to `prompt_encoding`) stays citable. A G sentence carries one page's
  words and links the page of the `analysis` entry whose cite it carries,
  else its `sentence_urls` page, else the guide's `url`; the fallback's
  sentences carry the app's own note, are labelled General, not Guide, and
  are never called the authors'. A borrowed or translated excerpt says whose
  it is, to the critic and on the page (`Citation.note` ->
  `CritiqueIssue.note`, under the issue and in its tooltip).

Frontend and CLI:

- Saved gallery searches store only Name, Query, Starred and All models in a
  versioned database-scoped sidecar, never image ids or picks. Names are unique
  after NFC normalization and case folding; duplicate-name replacement is
  explicit and revisioned. Reads are cold and first occur on opening the native
  dialog, not startup. CRUD bypasses gallery maintenance and model work. Select
  only fills the editor; Apply leaves the bin and sheet, cancels a pending
  debounce, writes all three filters and reloads gallery/Runs once. Old gallery
  epochs and sheet return filters cannot replace it. Picks remain selected and
  visibly counted, and Apply queues no generation or image mutation. Current
  checkpoint and relative dates are evaluated when applied. Errors, conflicts
  and late list replies preserve unsaved edits; modal keys and focus stay in
  the dialog, with connected-opener restoration using `preventScroll`.
- Native numeric inputs can display unfinished text such as `1e` while their
  DOM value is empty. `refuseBadNumber` checks `validity.badInput` at Generate,
  Run grid and comparison submission, names the field and focuses it without
  submitting. Intentional blanks retain profile defaults/random seeds;
  disabled inputs, sampling/seed values replaced by grid axes, and comparison's
  own Batch/Repeat values are ignored. Both the Run grid button and its keyboard
  path use `runGrid`.
  Stored reruns and More seeds do not validate unrelated composer inputs.
- Personal snippets are the final item of Examples, after the authors' prompts,
  and remain available with no author examples. They open a native dialog,
  with declarations in `prompt-snippets.js` and `initSnippets()` called by
  `app.js`. Writes go through `rewriteBox`/`writeBox`, keeping page scroll,
  with Append/Prepend and undo; a
  greyed negative refuses insertion. The modal owns its keys, with no
  Ctrl/Cmd+Enter submission underneath it. CRUD uses a bounded atomic
  `auradream.snippets.json` sidecar, optimistic item revisions and an
  independent lock; unreadable files are never replaced by an empty list.
  Snippets remain available during gallery bundle maintenance. The sidecar
  and temporary files are ignored by Git. No read is made at page startup.
- Named generation/grid setups open from Examples, before Personal snippets,
  in a native dialog (`generation-presets.js`, wired only by `app.js`). Their
  versioned payload retains raw fields and nullable
  profile overrides; never store `panelState`, expanded cells, conditioning or
  execution intent. Capture/validation requires the exact model revision and
  valid Eager/native loader intent with Step cache Off; GET/DELETE remains
  checkpoint-independent. Item revisions protect edits/deletes and validation
  rechecks after calculating. No route submits or restores, and no startup
  read occurs. Opening/selecting never writes the composer. Save current makes
  a new setup, Update from current replaces the selected payload, and Rename
  keeps it. Unsaved names survive failures and list reloads; selecting the
  refreshed item explicitly adopts its latest revision. Collection reads hold
  list selection without blocking name edits; successful Reload clears only
  visible selection, preserving the editor until deliberate selection. View
  toggles retain that visible selection or deliberate blank. Selection during
  validation still cancels the older check. Save names omitted
  active LoRA/reference/Then refine inputs as a hold. Restore requires the
  exact checkpoint, offers Models on a mismatch, and validates afresh before
  writing. Its reply must still match item/model revisions, the entire draft,
  performance intent and the edit epoch; otherwise a new press is needed.
  It closes the modal before undo-aware writes, uses `setReferences`, clears
  conditioning and Then refine, and keeps raw inactive text/overrides. Its
  summary shows saved/effective settings and what will clear. Grid restores
  Resume off and retains its ordinary seed-axis and request Repeat=1 rules.
  The one whole-draft Undo keeps raw sampling/ownership, grid, conditioning,
  Then refine and translation; first later edits retire it through panel and
  Then refine handlers. It shares the stable prompt-note overlay and retirement
  rules with Reset/Reuse; deleted/binned references leave its snapshot too.
  Restore/Undo do not scroll; native Ctrl+Z retains prompt history. Held actions
  explain their title reason in the dialog, and modal keys cannot submit work
  underneath it. Held/invalid performance intent is never cleared by refusal.
- Keyboard help (`shortcuts.js`) opens with `?` or the top bar's named
  button, never while typing in an input, textarea, select or editable
  content, during composition, on a repeat or with Ctrl/Meta/Alt. Its native
  dialog may sit above another dialog; while open, the page's global
  shortcut dispatcher returns before Ctrl/Cmd+Enter or Escape can act
  underneath it. Tab stays on help's sole Close control. Close, native Escape
  and an actual backdrop press restore the connected opener with
  `preventScroll`. The complete, grouped list is
  available even when the gallery is empty or a comparison is inactive.
  Opening snapshots the current context before a menu loses focus. Relevant
  groups lead, followed by the complete inventory; there are no hidden shortcut
  groups or extra help controls. Queue/Bundles help names native button/field
  navigation and explicitly promises no dedicated submit/queue accelerator.
  A menu item's hidden-but-connected node is not a valid return target: use its
  visible opener fallback, without reopening the menu or scrolling. Context-menu
  help owns `?` before the menu's propagation stop. Image shortcut hints apply
  after closing that menu; Enter/Space chooses within it. Sheet images retain
  their own documented keys, without promising the image actions menu.
- `web/` has no build step and no test suite; check it in a real browser with
  `tools/fake_server.py` and `tools/browser.mjs` (see Tests). The page script
  is classic scripts sharing one global scope, loaded in the order
  `index.html` lists them; their top-level bindings stay top-level, and only
  `app.js`, the last, runs anything at load.
- `web/style.css` takes its type, greys, spacing, control heights, radii and
  layout widths from the tokens on `:root`, whose table says what each is
  for; a size the page shares is a token, a one-off stays a literal, and a
  control's vertical padding is worked out from its height token
  (`--line-strong` is the controls' borders, `--topbar-h` the topbar's set
  height). The settings panel's sticky top is `panel.js`'s (`panelPos`,
  written inline): resting under the topbar while the panel fits, else
  between that and the top that holds its bottom a gutter above the window's,
  moved there by the user's own scrolling only (never by scroll anchoring or
  a page that shortens), placed on a change of its height or the window's
  but not while the pointer is over it (Generate must not move under the
  pointer that pressed it), and moved to bring a control focused out of
  sight into view, the browser's own scroll to it taken back. Never a
  window-height media query to unpin it or set its top, and never
  `overflow-y: auto` on the panel, which would clip the Examples menu, the
  tag menu and the tips; it stays `position: sticky` at z-index 10 above 900
  px and static at or below it, where the one column is `minmax(0, 1fr)`,
  never a bare `1fr`, whose least width is the panel's min-content (a
  running job's unbroken Now line widened the page under 680 px). A tip
  that would run past the window's bottom opens upwards; one that would run
  past either side moves in by as much, 8 px short of the edge and rounded
  away from it (`placeTip`: an inline `translate` cleared at each placement,
  never the markup's `.help.end`). A badge inside a box that clips its
  overflow (`tipClipper`: the prompt check's scroller, above the results
  and docked, and the Refine dialog) hangs its tip from the window instead
  (`floatTip`, `.tip.floating`, position fixed). The tip goes under the
  badge where it fits, else over it, moved in from the window's sides, and
  is placed again on any scroll while its badge holds the pointer or the
  focus (`followFloatingTip`). It is hidden while the badge is scrolled out
  of the box (`.help.clipped`). Placed inside the check, the comparison's
  Batch tip was cut off by the check's edge and lay under its sticky head.
- A window 1800 px and wider overrides the tokens in one media query (the
  wide tier): the type ramps by `--ramp` from its 1800 px size to its 2560
  one (body 14 to 15.5 px, captions 12 to 13.5), the panel is
  `clamp(400px, 18vw, 480px)`, the cards `clamp(200px, 9vw, 260px)` (M; the
  gallery head's S is `clamp(150px, 6.5vw, 180px)` on
  `#gallery[data-density="s"]`), and the
  prompt box grows with its text from five lines to twelve. That growth rule
  is `#prompt:not([style*="height"])`, so a box dragged by its grip keeps the
  height it was dragged to; nothing may write an inline height on `#prompt`
  for any other reason. A second query, for wide windows under 1150 px tall,
  sets compact control heights of its own (a field 40 px, the primary 44, the
  small and tiny buttons 28 and 26), today's panel padding and topbar, a 12 px
  gap and an eight-line prompt cap, and leaves the Output caption the
  group's name alone, as on a two-column page under 1800 px. In the wide
  tier the tiny buttons are 30 px at a caption's type (a finding's Show,
  inline in its sentence, is the line's height instead:
  `.analysis-message > .analysis-show`), tick boxes 18 px, a tag chip's ×
  and a reference chip's × 24 px targets, reference chips up to 96 px
  badged "image N", a card's (24 px wide at the gallery's S, 32 tall) and a
  long run's tile tools and a result's row `--h-tool` (32 px), and the
  lightbox's arrow circles `--lb-nav` (52 px; 40 under 1800 px). Nothing
  changes from this wide-tier scaling under 1800 px, where the
  smokes run, but the panel's gap and field height on a two-column window
  under 900 px tall (the panel's order, below); 1400x900 is not short. That is scale; a component's own layout breakpoint is not,
  and is named in its rule (the Models picker's column of pictures from
  1280 px, under "Model and loader"; the panel's two columns). A component's wide-tier rule goes in its own section, inside an
  `@media (min-width: 1800px)` block, and new CSS uses the tokens. Contrast
  and focus are not scale and apply at every width: `--line-strong`
  (#666680), `--accent-strong` (#6a4cf0), `--danger-strong` (#b4233a, the
  delete dialog's Delete fill), placeholders in `--muted`, and the
  page's own `:focus-visible` ring (2 px of `--accent-2`), which ships with
  `color-scheme: dark` because that turns the UA ring near black; a control
  with a focus style of its own outranks the ring. A disabled box is dimmed
  in the panel's field rows only (`.panel .sub > input:disabled`, `select`):
  under 901 px the prompt check sits in the panel, and its comparison's
  "original" box is disabled because it always runs, which its
  `aria-description` says.
- A tip lets clicks through (`.tip { pointer-events: none }`), wraps by its
  own `white-space: normal` (never inheriting a caption's or a label line's
  nowrap, which ran the sampling row's tips out past the window), and keeps
  its line breaks only where the code writing it sets `.tip.pre` (`applyHints`,
  `refreshLoras`), never because it is `.tip.wide`. Every `.help` badge has a
  static `aria-label` naming what it explains ("Steps help", "Steps axis
  help", "Grid seeds help", the dialog's "Refine size help") and
  `aria-describedby` pointing at its own tip, keeps role=note and
  tabindex=0, and stays outside its field's `<label>`, where a press ticked
  the box (Resume's). Red on hover is the `.ghost.danger` class (Cancel, a
  queue row's ×, Cancel queued, Delete unstarred, Reset), never every small
  ghost button. The page's one h1 is the word mark (`.brand`), one word
  ("Aura" and an accent `<span>` "Dream"), so it reads "AuraDream" with no
  spacer. The settings panel is a `<section>` named by its first
  child, an `.sr-only` h2 "Settings" (`aria-labelledby="panel-head"`).
  `#viewer` is a region named Results (`role="region"`, `aria-label`), never
  with an unseen heading before it: the docked stage's reading flow read
  such a heading after the gallery. `.sr-only` is out of the flow (no flex
  gap, no grid cell), so it is how a heading joins the panel without moving
  it, and it is never a child of `.stage`.
  A button whose words are its state (Pause / Pausing… / Resume) has no
  `aria-pressed`. A toggle with `aria-pressed` keeps one name in every
  state - "Star" for every star (the lightbox's words are "☆ Star" / "★ Star"
  with `.active`), "Reference image" for a card's ❐, "Fill width" for the
  sheet's - and its state is
  `aria-pressed`, the glyph and, in words, its title (`starTitle(on, button)`
  for every star; ❐'s "Reference image N - click to remove" or its reason,
  also its `aria-description`). A button held
  for a reason (Generate, Run grid, and Run comparison under the same
  `submitHold`, Unload, Pause, Refine, the Refine dialog's Compare 3 and From panel, Ask
  the critic, Expand, a card's ⤢ and ❐, a pass's Compare with source in the
  result column, the sheet's Fill width on a wrapped row and its export
  buttons while one runs, a negative-prompt finding's Show) is
  `aria-disabled` with the reason as its title, never `disabled`, so it keeps
  focus and a press can say why. The prompt check's dead Use, Apply and Run
  comparison (stale, or fewer than two versions), the Refine dialog's check
  buttons (Show, Apply) and its scale buttons past the guard are `disabled`
  with the reason as their title - the documented exceptions (below): each
  dies with a state its own panel says in words beside it.
- Secondary `.ghost` actions use the existing panel surface, readable text
  and strong edge; tiny caption tools stay tertiary. Enabled hover is scoped
  to enabled controls and has low specificity so selected component states
  keep their readable text and accent fill/edge. Held controls retain their
  reason-on-press behavior, reduced surface and keyboard focus ring, and
  never brighten as enabled on hover. Footer actions keep their existing
  geometry while sharing those states; inpaint and small selected tools use
  readable text rather than small purple labels on the panel.
- Frequent prompt-caption tools have a 24 px floor and `--fs-sm` type below
  1800 px (`--h-sm - 2px`, scoped in fields.css); their existing wide sizes
  stay intact. Gallery footer actions use `--h-sm` as a stable minimum height,
  including Bundles, with a 24 px width floor for collapsed Picked/Clear
  arrows. Primary Generate/Run grid sizes, seed width and layout order stay
  intact. These target-size changes apply in every held/selected/hover state,
  so a state change cannot move a control under the pointer. Narrow footer
  wrapping and the error line's reserved slot remain unchanged.
- Each of the panel's size strips (`#presets`, `#tiers`) is a `role="group"`
  with one tab stop (the chosen preset, else the first) and `aria-pressed`;
  Left and Right, Home and End move focus
  along `strip.querySelectorAll('button')` without choosing (`stepPreset`),
  and `syncSize` resets the stop. The gallery head's two card-view pairs
  (`#card-view`: S/M and Fit/Crop, shown from 1800 px only, hidden on a sheet
  by `setSheet`) follow the same pattern: one tab stop each (the chosen
  button), `aria-pressed`, and `stepPreset` arrows. The Refine dialog's
  scale buttons (`#refine-scales`, role=group) carry `aria-pressed` beside
  `.active`; past the guard they stay `disabled` (the documented exception),
  and they are not a roving tab stop. A strip whose every preset is in range
  carries no marks (`all-guided`). While the negative box is greyed and the
  checkpoint takes a negative, `#neg-enable` ("Set True CFG" or "Set CFG"),
  after the note in the field's caption, focuses and selects the deciding box
  and writes nothing (`focusNegativeKnob`); it never goes over the box, whose
  text is what a greyed negative shows, and the greyed box keeps the pointer,
  so a carried-over negative can still be scrolled. The negative's caption
  names a cost only where it is the field's own: greyed ("(needs CFG > N —
  ~2× slower)") and under True CFG, where live it reads "(~2× slower; one
  space counts)". Live where CFG is the knob, the uncond
  pass runs above the threshold whatever the box holds, so the caption reads
  "(runs at CFG > N, no extra cost)", or "(for grid cells above CFG N)"
  where only an open grid's axis reaches past the threshold; each fits on
  one line beside the label at 1400x900 and at 2048x1040 at 125%, as the
  greyed one does. The Refine dialog's negative caption uses the same words
  for the same state: "(runs at CFG > t, no extra cost)" where CFG is the
  knob and the box is live, "(needs CFG|True CFG > t — ~2× slower)" where
  it is greyed, "(~2× slower; one space counts)" (`TRUE_CFG_LIVE_NOTE`,
  the app's words) where True CFG is the knob and the box is live, and
  "(this checkpoint takes no negative prompt)" where it takes none; the
  Inpaint and Outpaint dialogs' (`syncPassSampling`) say the same. Every
  `collect()` (the panel's, `refineRequest`, `inpaintRequest`,
  `outpaintRequest`) sends the negative through `negativeSent(box)`
  (`sizes.js`): null from a greyed box, the text trimmed from a live one,
  and a whitespace-only live box as `" "` only where `trueCfgSupported`
  (Qwen-Image, FLUX.1) - a lone space is a negative there, so True CFG runs,
  as Qwen-Image's guide asks ("a single space if there is nothing to
  remove"). A truly empty box is null everywhere, and blank means none
  wherever CFG is the knob. No backend change: `applied()` keeps `" "`
  (truthy), and old rows and resume keys are unchanged. The facts list and
  the sheet's negative axis name a whitespace-only negative "a single space"
  (`negativeShown`), never a blank that reads as none. Where the checkpoint
  takes no negative prompt at all (klein) the field is its caption alone
  (`.unsupported` hides the box; its text is kept, and the note says so).
  `#negative-field` is a div with `<label for="negative">`: a button inside a
  label becomes the label's control. `#negative` is described by `#neg-note`
  (`aria-describedby`), so a greyed box says why.
- Custom dimensions and LoRA/reference setup use native details/summary
  disclosures (`#custom-size`, `#conditioning-section`). Inactive defaults
  start closed; summaries retain actual dimensions, adapter/weight and reference
  count, with incomplete/unavailable state named. Aspect/MP presets and size
  warnings stay outside. Native summaries retain `--h-sm` targets and the
  common focus ring. Explicit restore, Reuse/Undo and new reference activation
  reveal relevant setup; numeric-error and reference focus open the containing
  fold first. Polling/sync changes words only. Closing never changes values;
  no automatic close or poll-driven reopen, and no new persistence key.
  Pending restored LoRA names survive snapshots/Undo, but Reset or a replacement
  Reuse clears them so a late catalogue cannot restore stale conditioning.
- The sampling row (`.sampling-row`) keeps each name on one line, with
  `white-space: nowrap` on the `<label>` alone - never on the label line,
  which also holds the ? badge and its tip. No box is narrower than its name
  (`min-width: min-content`), and its boxes take the counters' padding.
  Seed, Batch and Repeat form the Output group (`#output-caption`, the
  Sampling caption's twin, role=group), one line at every width.
  `#run-hint`, what Batch and Repeat add up to (Batch and Repeat are
  described by it), sits under Generate after `#progress`, not in the group:
  above Generate its two lines moved Generate as Batch or Repeat passed 1
  and put a running batch's bars past the window's foot. The seed's dice (`#btn-dice`)
  sits in the Seed name's line, as the ? badges do, at that line's height,
  its target reaching past its frame (`::after`: 6 px up - to the caption
  where it shows - 4 px down to the box, 25 px or more in all); the seed
  box therefore has the row less the two counters (156 px at 1400, for a
  16-digit seed's 121 px). The Seed field is a div with
  `<label for="seed">`, because a button inside a label becomes the label's
  control. The caption shows in one column and in the wide tier on a window
  taller than 1150 px, where the Output group is exactly as tall as the
  Sampling one. Elsewhere it is the group's name alone, visually hidden
  (absolute, 1 px, clipped): a two-column page under 1800 px and the short
  wide windows the compact controls are for (`(900px < width < 1800px),
  (min-width: 1800px) and (max-height: 1150px)`, the complement of the
  one-column query and the wide tier, fractional widths included). There
  the group is its row's height, so Generate's top is in the window at
  scroll 0 (776.5 at 1400x900); smoke-wide holds
  1400x900, 1920x947 and 2048x1040@1.25.
- The panel's order is Prompt, Negative, Size, Sampling, Output, Generate
  and what a job adds under it (`#progress`, then the run hint, the finish
  line, the queue, Load settings from PNG… (`.png-load`), `#error`, the
  last-run banner), the grid fold, the Step cache row
  (`.execution-cache-row`), then Reference images
  (`#reference-field`) and LoRA (`#lora-field`) last. Nothing rarely used
  goes between Sampling and Generate: at scroll 0 the panel is in the
  page's flow and its sticky top cannot lift it, so every line above
  Generate pushes a running job's bars and its elapsed line (`#cell-note`)
  towards the window's foot - with References and LoRA there, at 2048x1040
  at 125% the batch bar sat at 1038 of 1040 and the elapsed line past it.
  A request reads every control wherever it sits; the grid's LoRA weight
  axis says the adapter is picked in LoRA below. With the run hint under
  the progress and the counters on the time's line, a running 4 x 3 batch's
  progress ends at 895 at 1400x900 (FLUX.1 fake, scroll 0), 901 at
  1920x947 and 907 at 2048x1040@1.25. A two-column window under 900 px tall
  (`(900px < width < 1800px) and (height < 900px)`) sets `--panel-gap`
  10 px and `--h-field` 37 px on `.panel` alone, so at 1536x864 the same
  progress ends at 851; the dialogs, gallery and stage keep their heights.
  A shorter window - a maximized browser at that scaling, about 758 px -
  still needs a scroll, or the topbar's mirror. smoke-wide holds the order;
  smoke-narrow holds a real batch's whole progress in the window at
  1400x900, 1920x947, 2048x1040@1.25 and 1536x864@1.25, and Generate
  unmoved as Batch and Repeat change.
- The size line leads with a side off the 16 grid, in the warning colour,
  naming its two neighbours ("1000 is not a multiple of 16: 992 or 1008",
  the schema's own words at Generate). It compares the request against
  `guide.pixels` a tenth either way and quotes `pixels_word`. An `example` span reads "above/below the N
  MP of <pixels_source>", never what the checkpoint is "for", and its
  in-range presets' titles read "within the span of <pixels_source>", or
  "about the size of <pixels_source>" where the span is one size. `fmtMp`
  quotes every bound to at most two decimals, trailing zeros dropped (0.98,
  1.05, 1.57, 0.26, 2.36, 2.25, 1.64, 1.77). Where the guide has `aspects`,
  a typed size's framing is the preset aspect nearest its ratio within 3%
  (`framingOf`: 1400x800 is 7:4, not the 16:9 1.6% away); a size in a
  guided framing that is not one of `resolutions` is "not one of the
  recommended sizes", and any other framing "not a recommended aspect". The
  Examples menu is built on open from the
  last status. A pick writes through `writeBox` (tags.js) and tells the
  preview and the panel the way `choose` does, and over a prompt of the
  user's own offers an Undo (`offerPromptUndo`) in the note under the prompt
  (`#prompt-note`, `showPromptNote`). The note is an overlay on two columns,
  and in the panel's flow under 901 px and while the negative box is
  greyed - except Reset's notes (`at: 'box'`, `data-at="box"`), which lie
  in the box itself at every width. It holds one thing at a time, by
  `promptUndo.kind`, which it also
  sets as `data-kind`: 'write' (an Example's or an expansion's Undo, gone
  once the box holds anything else), 'info' (gone the same way, with no
  button), 'reset' (Reset's Undo, under Reset's own rule, lying in the
  emptied box itself), and 'running' and
  'held' (Expand's running line and a held expansion: typing does not take
  them and they have no timer, so they always take a line of the panel's
  flow, never the overlay over a negative box that may be live). Each kind
  expires after its `ms` unless focus is on its buttons, and `onDrop` runs
  whatever takes it. A Reuse's Undo and this note retire each other. While
  shown, a note may sit at the end of another box (`host`): an expansion's
  note sits at the end of its English from 1800 px.
  The menu is a role=menu named by its button (`aria-labelledby="btn-examples"`)
  whose items stay the clickable `<li>` children:
  ArrowDown, Enter or Space on the button opens it on its first item,
  arrows, Home and End walk it, Escape hands focus back to the button, and
  focus leaving closes it. Where references are supported and none is
  attached, an example matching `EDIT_PROMPT` (prompt-examples.js, held equal to
  `prompt_rules.EDIT_PROMPT` by `tests/test_prompt_rules_prose.py`) says it needs a
  reference image. An example holding one of the authors' template slots
  (Z-Anime's; NewBie's anonymous character identifiers can stay as written)
  says which slots are left to fill. The slots are `ModelStatus.guide.slots`,
  the placeholder rule's patterns, read case-insensitively and held by a
  test to syntax both engines read alike. The note reads "fill in
  [character]", names two slots where there are two, and gives the first and
  a count where there are more. A pick selects the first slot
  (`templateSlots`, `setSelectionRange`, `scrollToSelection`); selecting
  writes nothing. `guide.example_notes` maps prompt text to an editing note,
  displayed alongside an author scene excerpt and never inserted into the
  prompt. Every catalogue guide has a substantive scene example; quality
  recipes remain in Tags. Preserve author wording and label excerpts. An
  example drawing any finding about its own words (every rule, notes
  included; not the negative box, references or slots, which the menu says
  itself) has a note naming the gap ("Author sample: under 30-80 words"),
  held by `test_the_authors_examples_draw_no_finding_unnoted`; never edit
  the example to pass.
- Prompt expansion is ERNIE-only and explicit. `syncExpandButton` is
  Expand's one writer: it is held while it runs (reading "Expand…", never
  the wider "Expanding…", in the width it had idle: `syncExpandButton` locks it and
  `.running` gives the ellipsis its side padding, so the caption row, full at 1400 px,
  never wraps Reset), during this page's job or another tab's
  (`EXPAND_ELSEWHERE`, after busy in its reasons), during a critique, or
  with the server offline - aria-disabled, with the reason as its title, so
  a keyboard user's focus stays; otherwise the title names a first use's
  cost from `ModelStatus.prompt_expander`. A press on it held says why,
  through `expandSay`: this page's job's reason is a job note gone at the
  job's end, another tab's is said until that job ends, and while a Reuse's
  Undo holds the panel's line it is said under the prompt, never over the
  Undo. While it runs a press says what the result will do as things stand
  (`expandOutcome`: written in, held, or dropped). The line under the
  prompt (`#prompt-note`, kind 'running') says what Expand waits on, in the
  server's order - first the card, while something else holds it (another
  tab's critic or Expand, or a job another tab queued during the load:
  `ModelStatus.card_holder`); then the model's load, the rewriter's first
  download or its load, the expansion, then the translator's first
  download - beside an aria-hidden clock. The press sends a `ticket` of its
  own (`PromptExpansionRequest.ticket`), which the status answers as
  `card_ticket` while that expansion holds the card, so the page tells its
  own press from another tab's ("Waiting for the card, which another tab's
  Expand has, then …"). `syncExpandRunning` follows the status, which is
  asked 250 ms after the press and again when Expand ends, and
  `refreshStatus` asks for the line again as the holder changes. Before sending, a
  prompt past `EXPAND_MAX_CHARS` (prompt-expand.js, held equal to
  `PromptExpansionRequest`'s `max_length` by `tests/test_ernie.py`), or a W
  or H that is not whole pixels in range, is refused at its box in the
  panel's words. The result is written through `rewriteBox` (Ctrl+Z gives
  back what it replaced), `promptReplaced()` follows, its English is shown
  after the write (whose input event clears one), the caret and both boxes'
  scroll go to their start, and "Prompt expanded: …" offers an Undo back to
  the replaced prompt and its English; the usual persistence and grid
  preview paths are told about the programmatic edit. The result never
  replaces a box that changed while the request was in flight: typed over
  (no `replaceBoxText` write meanwhile: `replacingBox`), it is held under
  the prompt (Use it, Discard); replaced whole by a Reset, Reuse, Example or
  Use, it goes without a word. An Example's Undo or a held expansion that
  the running line took is offered again where the expansion fails.
  Failures go to `#error` as "Expand did not finish: <reason>. The prompt is
  unchanged."; a failed load (503) is a pointer to the viewer's card where
  the card is shown (`loadFailureInViewer()`), elsewhere the sentence, once,
  with its commands in quotation marks. While a Reuse's Undo holds the
  panel's line, these are said under the prompt instead, as a 20 s note
  (`expandSay`). The English is cleared only by another expansion or a
  prompt change; it rides in `panelState()`, so Reset's and Reuse's Undo and
  a reload bring it back while the prompt is the one it translates. It sits
  under the prompt below 1800 px; from 1800 px it heads the results
  (`placeTranslation`), at its own height up to 40vh, with the expansion's
  note at its end, and beside the docked check it takes the first row of
  the columns left of it (`#prompt-translation-field`, the no-chips dock
  only, whose rules repeat the dock's selector and read "no chips" from the
  chips' host as the dock's do). Its hint (`#prompt-translation-hint`)
  describes the box.
  The critic holds what Expand holds (Generate, Run grid, Run comparison and
  the Refine dialog's Refine aria-disabled, Expand, Unload) from the click
  on, including the check it first runs on a
  changed prompt, and `startJob` refuses while it runs. Check and Check again
  are disabled during that check and re-enabled in a `finally`; if that check
  is replaced anyway the page says nothing was sent, if it is cleared (a
  Reset, another checkpoint) it says nothing, and Close pressed meanwhile
  focuses the prompt, not the disabled Check. Ask the critic is
  aria-disabled, never disabled, so it keeps focus. It is held
  (`criticHold`) while the server does not answer and while the image
  model loads (the route's 409), then for a job (this page's or another
  tab's) and for Expand, with the reason as its title and on
  `#critic-cost`; `goOffline`, the poll that answers again and a change of
  the model's state each sync it. A press says why: offline, the offline
  words (`OFFLINE_MESSAGE`), which the poll that answers takes back; during
  the load, a note taken back when the load ends (`criticLoadNote`); during
  this page's job, a job note (`showJobNote`); during another tab's job,
  said until it ends (`elsewhereNote`). A critique the server never
  answered reads "The critic did not answer: the server did not answer."
  (`critiqueOffline`) and is taken back by the status poll that answers
  again (`takeBackCriticOffline`); `critiqueFailureText` drops any closing
  stop, question mark or exclamation mark of the reason's own ("…still
  running?." came of keeping it).
  Its line is two spans:
  `#critic-progress` (aria-live polite) says only what answers a press
  (Downloading…, Thinking…), and `#critic-cost` (not live) says what a press
  costs or what holds it (no server, the load, a job). The clock
  (`#critic-clock`, aria-hidden, outside the live region) follows them in
  one run of text (`.critic-words`), never a flex item of its own, kept whole
  and joined to the last word by a word joiner in its `::before`. A size is
  binary and labelled so (`humanBytes`: KiB, MiB, GiB), as every size the
  page shows is - the topbar's VRAM and free disk too - and the CLI's
  `gallery_reports.human_bytes`; any answer, shown or
  set aside, sets `promptCritiqueReady` at once. An answer takes focus
  (`#critic-result`, tabindex=-1) while focus is still on the button; so does
  a failed critique's row (`critiqueFailure`, never a `critiqueResult`), which
  reads "The critic did not answer: <the server's reason>." with nothing
  generic added; `#error` gets that sentence only where the critic's block is
  not shown. The panel's own messages (words moved, an answer set aside or
  never asked) go to `#analysis-status` through `analysisMessage`, which falls
  back to `showError` where the critic's section is hidden, and stays at full
  strength while the rest of a stale panel is dimmed; refusals stay in
  `#error`. Check pressed on a two-column page moves focus to the panel
  heading (`checkFromButton`), and Check again hands focus to the heading
  when it hides; under 901 px focus stays on Check. Escape closes the check
  as Close does (`escapeAnalysis`, app.js's keydown): after the Examples
  menu, the Tags helper and anything that took the key
  (`defaultPrevented`), never under the lightbox, the Refine dialog, Mass
  refine's or Swap tags', during a composition or in a search box with
  text. The open Models picker's Escape (it closes the picker and takes the
  key) is skipped under those dialogs too: a picker opened under one
  (`offerModelPicker` on a status answer) stays, and the Escape is the
  dialog's cancel. Focus inside the check
  (or on `<body>`) goes to Check (to the prompt while Check is held), focus
  elsewhere stays.
- The prompt check describes one text: the prompt and negative prompt as
  `collect()` sent them, whether CFG or True CFG applies that negative, and
  the reference images it counted
  (`analysisResult.reference`; klein's references rule reads them). Any
  change to either box - typing, every programmatic write through
  `promptReplaced()`, `syncCfg()` greying the negative or changing its
  activation while a grid keeps the editor enabled - or to the references
  (`setReferences`, Reuse, a `reference_supported` flip) marks it stale:
  dimmed (opacity 0.8 with `--muted` raised to `--text-3`, one rule shared
  with the Refine dialog's `.rf-check`), every Use/Apply and Run comparison
  dead, a finding's words selected only while still where the check found
  them, scrolled into their box (`scrollToSelection`) and the box outlined
  for 600 ms (`flashBox`, `.box-flash`; none under prefers-reduced-motion).
  A finding or critic issue with a span does this through its Show
  (`makeSelectable`): a `.analysis-show` button after its words, described
  by them through `aria-describedby`, and titled with the box it selects in
  (the prompt, the negative prompt, or the rewrite for the rewrite's own
  findings). The item itself is no tab stop, and a click elsewhere on it
  does the same for the pointer. Show stays live while the panel is stale
  and says so when the words have moved. A negative-prompt finding's Show is
  `aria-disabled` with the reason as its title while the negative box is
  greyed (`syncNegativeShow`, from `syncNegativeApply`), and any selection
  in a greyed box (Show, a click on the finding, an encoder's cut link) says
  why in `#analysis-status` (`selectIn`). The badge says
  "references changed since" when only the references moved. A dead Use or
  Apply gives the reason in its title but stays `disabled`, not
  `aria-disabled` - the documented exception - since they die as a set while
  the stale badge and Check again, at full strength in the head, say why. A
  checkpoint change clears it. Raw negative text and effective activation
  are captured before Check and the critic await replies; inactive raw edits
  still make it stale. Apply is offered on rule findings only; an encoder
  suggestion is shown as text. In klein's references finding each "image N"
  it names is a link (`findingText`, `referenceLink`; the words unchanged)
  that focuses chip N's × and flashes the chip, or, past the last chip, says
  in `#analysis-status` that image N is not attached now; it stays live
  while the check is stale, and a press that finds its chip takes back only
  that sentence of its own (`notAttachedSaid`), never another the slot
  holds. Use and Apply write through `replaceBoxText`
  (so Ctrl+Z restores) and check again; every other whole-box write of the
  prompt or negative - the tag tools, Examples, Reuse's `fillPanel` and its
  Undo, Expand and its Undo, Reset and its Undo - goes through tags.js's
  `writeBox` (`replaceBoxText`, then no suggestion list or typing state over
  the new text), never `.value`, and Reuse's, Expand's and Reset's through
  `rewriteBox`, which focuses the box with `preventScroll` first;
  `replaceBoxText` sets `replacingBox` while it writes. Use rewrite, Use
  reordered (the critic's and Check's own) and a rule finding's Apply on the
  prompt first put the prompt they replace into an empty Your version
  (`useVersion`), with a note (a negative-prompt Apply has no Your version:
  Ctrl+Z is its way back); the order finding whose suggestion equals `reordered` says "Same as
  Use reordered above." instead of Apply. Server offsets count code points;
  the page converts them (`units()`) before selecting. Rewrites show
  character counts and an expandable exact diff against the checked source
  (`promptChanges`), using text nodes with additions/removals marked, and,
  outside the closed diff, a gist of its first three changes grouped as a
  reader names them (`changeHunks`, `gistItems`: a word replaced is one
  change, edits only spaces and marks keep apart are one, words put elsewhere
  are a move); no rewrite says so, with the server's set-aside explanations.
  The rewrite's findings are marked "also in your prompt" or "new", followed
  by "No longer flagged: …"; a prompt finding that rests on the tag index
  (`readsTagIndex`: the order, the style, an artist's @), which the rewrite's
  lint never asks of its own, is never called fixed but listed as possibly
  still applying ("Not flagged on the rewrite, whose check asks the tag
  index nothing of its own - these may still apply") - except an artist's @
  where the server says `rewrite_artists_checked`. The comparison has its own Batch
  select (1, 2 or 4; default 1) and Seeds (4 or 8 total per version):
  `num_images` is its batch size and grid `seeds` is Seeds divided by Batch,
  keeping the image count versions times Seeds. Its preview states the total
  seeds per version and batch size. The main panel's Batch and Repeat do not
  apply; its Seed still does, and a blank one stays blank: the comparison is
  not the panel's grid (`startJob`'s `panel: false`), so its base seed is
  never written back for a Resume it does not take. The mechanics are a ? tip after the
  comparison's Batch select; the 4-vs-8 caveat stays visible. Preview and Run
  read both comparison selects directly (they and Your version are
  `autocomplete=off`). It takes Check's `reordered` as its Reordered row, else
  the critic's as the server kept it (`critic.validate` decides; the page
  adds no rule of its own), never writes `els.prompt` or `els.gridPrompt`,
  drops a version identical to another, and opens its submission
  (`job:<id>`) as the sheet when done - Starred unticked first, and the
  sheet's note says so ("Starred unticked: nothing in a new comparison is
  starred yet"; `showJob(jobId, { fresh: true })`) - with the page scrolled
  to it (`showSheetScope`); its Your version box (`yours`) is the one way to
  test order on prose. A negative-prompt Apply is dead while the negative box
  is greyed.
- Tag autocomplete and comma-item validation use the fixed Danbooru
  tag API through `backend/tags.py`, filtered when a checkpoint documents a
  knowledge cutoff. WAI v17 publishes none: keep its cutoff `None` and the
  live lookup advisory rather than inventing a date. Model-card
  special tags validate locally; lookup failure stays advisory and never
  blocks. Suggestions and validation's `canonical` write parentheses bare
  (`tags.display_tag`: `hatsune miku (vocaloid)`), never A1111's `\(`, which
  diffusers never unescapes: two CLIP tokens for every qualified tag. A typed
  escape is the same tag to the chips (no "canonical:" note for escapes
  alone), the order rule and Swap tags, and is never rewritten. The page
  selects the first suggestion only from three typed characters it starts
  with (Tab otherwise moves on). Every suggestion list
  (the prompt box's, a chip editor's, the Refine dialog's) is a listbox
  named "Tag suggestions"; its options have ids, and the box's
  `aria-activedescendant` names the selected one and is dropped when none
  is selected. Validation asks about
  the first 64 items (`TAG_LOOKUP_CAP`, the request's ceiling; the order
  slots' category lookup can ask about names past it, as a Check would) and
  marks the rest unchecked. An item longer than any tag
  (`tags.TAG_NAME_MAX`, 200 characters: a hybrid prompt's sentence) is
  answered unknown (`tags.too_long_result`: `code` not_found, "longer than
  any Danbooru tag") without asking the index, reachable or not (`tag_lookup.lookup_tags`,
  `tags.unchecked_results`); only a blank item or one past 4000 characters
  is refused (`TagValidationRequest`). One 213-character sentence used to
  make the whole tray read "Danbooru unreachable" (a 422). Verdicts are
  cached per checkpoint, and the
  tray's state is pending, checked or offline (amber, never red) in
  `data-state`. Each chip that the server's last answer gives a step
  carries it as `data-slot`. That answer is `tagSlots`, sent with the
  verdict request's `prompt`. `.tag-check-label[data-slot]::before` draws
  the step as a lowercase caption at the chip's start, set off from the mark
  by a 1 px `--line-strong` rule, never as text in the chip. The step is
  also in the chip's title and aria-label, and the steps form the tray
  title's legend only while some chip has one. For a text not answered yet,
  a chip keeps the last answer's step for the same tag until its own answer
  lands (`chipSlots`). The page works out no slot itself. The Refine
  dialog lists only tag units `/api/tags/validate` answers `valid: false`
  for under its Prompt box, each once, only where `tagVocabularyEnabled`,
  excluding its exact matching contextual prose parts and silent when the
  lookup is unavailable; it has no chips and never calls
  `writePromptTags`. Its Prompt box suggests tags there too, from tags.js's
  editor list: `scheduleEditorSuggestions` and `renderEditorSuggestions`
  with `query` set to the comma item at the caret
  (`currentTagRange(els.refinePrompt)`), `pick` to `applyRefineSuggestion`,
  `place` to `placeRefineSuggestions` and `keep`. The list is
  `#refine-tag-suggestions`, under the box, or over it where the dialog
  shows no room below for the list as drawn. The same three-character
  preselect applies, and Enter or Tab takes the selected item as an insert,
  one Ctrl+Z from what was typed. A list already up stays while the next
  key asks, nothing in it selected until the answer
  (`deselectEditorSuggestions`), and a click or a caret key that stays in
  the item the list answers (`menu._query`) asks nothing. It never touches
  `els.prompt`, `#tag-suggestions`, the chips or `tagTyping`. The chip
  editor and the dialog share `tagEditorSuggestionTimer` and
  `tagEditorSuggestionEpoch`, which is safe only because the dialog is
  modal. Find tag: from three typed characters the prompt box's list (not
  a chip editor's or the dialog's) ends with a `find` row, "Tags described
  as “…”…", there too when no name matches and never preselected
  (`renderTagSuggestions`, `TAG_FIND_MIN`). Choosing it
  (`applyTagSuggestion` sees `find`) opens `#tag-finder` under the prompt,
  searched, with the typed range kept (`openTagFinderFor`). It is never a
  section of the Tags helper, which must stay whole at 1400x900, nor a
  button in the Prompt caption, which has no width left at 1400 px and
  must stay one line; it, the helper and Examples are one at a time. It
  asks `GET /api/tags/describe` (`tags.describe`): the Danbooru wiki pages
  whose text matches the words, ordered by their tag's post count (the
  index's default is the pages edited last), 100 at most, plus the tag
  named the words whether or not its page uses them, kept to general tags
  with posts at the checkpoint's cutoff, that tag first, then by post
  count, 12 at most, each with its page's first sentence
  (`wiki_definition`), or null. Offline is `available: false`, said in its
  status line, never an error. The first result pressed replaces the typed
  item through `applyTagSuggestion` while that range is still the whole
  comma item it was (`pickFoundTag`); otherwise, and after, a result is
  `togglePromptTag`'s, as the helper's tags are. A press keeps focus on the
  result (`writePromptTags` hands focus back inside the finder too), which
  shows pressed while in the prompt (`syncTagFinderSelection`, from
  `syncTagSelection`). Its last search survives closing and is dropped for
  another `tagHelper`. Its box is `type="text"`: Escape closes the finder,
  focus back in the prompt. The page's click-outside close ignores a target
  no longer in the page: the row is gone from the list by then. On a two-column page validation lives above the
  results, not in the panel; under 901 px the same node returns below Prompt.
  The prompt check's panel follows it, in that order. On a two-column page
  each has a `display: contents` host of its own in `.stage`, the chips'
  first: `#tag-chips-wide`, then `#tag-validation-wide`, which keeps the id
  the check's selectors use; under 901 px both go in
  `#tag-validation-home`, chips first. `placeTagValidation` moves each to
  its host, keeping focus and the selection (`moveBefore` where the browser
  has it, restored by hand elsewhere). Above the results below 1800 px the
  check is capped at 55vh less the chips above it (`--chips-h`; never under
  `clamp(8rem, 15vh, 12rem)`) and scrolls inside itself, its head
  (`.tag-validation-wide .analysis-head`) sticky at the scroller's top as in
  the dock; in the panel under 901 px it is uncapped. From 1800 px the
  stylesheet docks the open check beside the results instead: a `:has()`
  grid on `.stage` (`--dock-w`) puts the chips across both columns in the
  first row, then the viewer and its note in the left column and the check
  in the right beside them. This is CSS but for the chips' height
  (`--chips-h`, written on `.stage` by tags.js's `chipsResized` from a
  ResizeObserver): the DOM order is unchanged, and without `:has()` the
  check stays above the results. CSS that reads "no chips" reads the chips'
  host, `.stage:has(> #tag-chips-wide > #tag-validation[hidden]):has(>
  #tag-validation-wide > #prompt-analysis:not([hidden]))`, never
  `#tag-validation[hidden] + #prompt-analysis`, which can no longer match.
  The docked grid reads in source order (`reading-flow: source-order`) with
  the check's host after the stage's own items (`reading-order: 1` on
  `#tag-validation-wide`, 2 on `.library`), so Tab and the accessibility
  tree go: the chips, the image just made and its tools, `#viewer-note`,
  `#recent` where it docks between, the check, the gallery. The chips and
  the check need a host each because a `display: contents` host is ordered
  as one item, placed where its first child is drawn. Nothing out of the
  flow (absolute, fixed, `.sr-only`) is a child of `.stage`: a reading-flow
  container reads such a child after all its items. `reading-flow` is
  Chromium's; elsewhere the DOM order stands. The check takes the left
  column's height rather than setting it (`contain: size`, stretched, at
  most the window under the topbar and the chips), and the viewer holding
  something is then at least `min(36rem, that window)`; idle, it stays its
  one line and the gallery comes up beside the content-sized check (the
  results viewer below). The check scrolls
  inside itself, its head sticky at the scroller's very top (`top: 0`, the
  column's top padding in the head's own, so nothing scrolls past above
  it) with Close absolute in its top right corner, never wrapped onto a line
  of its own, and `focusAnalysisHeading` resets that scroll. Never `position: sticky` or a
  min-height on the column itself. The image does not move as the check
  opens, chips or none (with `#recent` docked the chips span all three
  columns). With Gallery beside (below) the check is not docked in the
  stage: every dock rule that places or sizes the check carries
  `:not(.beside)` - its grid row, `contain`, `align-self` and cap, idle or
  not - and their negations are written `:is(.beside, :not(:has(...)))`.
  The open check takes the gallery's column instead: the gallery-beside
  section gives it the column's whole box (its insets, `overflow-y: auto`,
  `overscroll-behavior: contain`, `max-height: none`) whatever the viewer
  holds, nothing made yet included, with `#library` `visibility: hidden`
  under it until Close and its scroll kept, and `.stage.beside` takes
  `reading-flow: flex-visual` while the check is open, so its fixed column
  comes after the image, its tools, the note and `#recent`. Only the
  sections' single column (`.analysis-body`) and the sticky head
  (`.tag-validation-wide .analysis-head`) are the check's in both places.
  Nothing in the stage moves as it opens, the chips included.
- On hybrid guides the annotation tray draws server-classified prose as
  selectable `.prompt-prose` text, grouped with its exact internal commas
  and spacing, without tag marks, drag/edit/delete tools, duplicate flags,
  order captions or unknown/tag counts. A paragraph made entirely of prose
  keeps the tray hidden; its encoder readout remains under the prompt.
  Token cuts can still mark unread prose using the original offsets.
  `tagPromptLayout` is valid only for the same model and exact raw prompt;
  it never enters the lexical `tagVerdicts` cache. While changed hybrid text
  awaits classification, raw text replaces stale chip controls. A model
  change or prompt edit cannot reuse another context's layout. Panel and
  Refine suggestions exclude a classified prose caret and recheck their
  text/model/caret after delayed replies. Prose keeps the native selection
  and context menu; tag edits operate at whole-passage boundaries through
  `promptEditingUnits`/`writePromptUnits`, preserving internal commas and
  ordinary Ctrl+Z. Tag-only models and pure tag lists retain strict chips.
- Validation chips edit the prompt: drag (or Alt+Left/Right) reorders, × or
  Delete removes, and a click or right-click in a gap inserts there (a
  chip's right-click or Menu key inserts after itself, + tag at the end) with
  autocomplete; a plain click and the hover marker count only within a row
  of chips (`tagInsertionIndexAtPoint(..., { gap: true })`, null on the label
  and the hint), a right-click takes the nearest place; the marker is drawn
  again at the end of every tray render while the pointer rests in a gap
  (`tagHoverPoint`, not under an open editor), since a lookup or a count
  landing rebuilt the chips without it. All mutations must go
  through `writePromptTags`, which writes through `writeBox` and hands focus
  back to the chip the edit leaves in place (`tagFocusIndex`); a chip's write
  visits the prompt in passing - focused with `preventScroll` and exempt from
  `revealInPanel` through `panelPos.press` - so neither the panel nor the
  page moves. The tray is one roving tab stop and no live region
  (`#tag-summary` is, not counting the item being typed, and counting with
  `plural`: "1 prompt tag"); its children are
  only chips, the editor and + tag - the label, state and token count live in
  `data-*` attributes its `::before` reads, and an unknown chip's reason in
  the shared `#tag-tip`. A focused chip's ring is its own `:focus-visible`
  outline; no state rule may use the chip's outline (a repeat is marked on
  its text). The Tags helper drops under the prompt box (`#tags-menu` inside
  `.prompt-editor`, capped by the window alone), a click in the box moves its
  caret without closing it, and it names where the next tag goes; it inserts
  at the caret and never slots tags itself. Its order lines are Check's
  Order section's, served in `tag_helper` by `guide_for`
  (`model_defaults._served_tag_helper`): `order` is
  `prompt_order.documented_order`'s steps with `order_cite` as the line's
  title, or, where no order is checked (NoobAI, RouWei), "This app checks
  no tag order" and `author_orders`, the list `prompt_sources.author_orders`
  gives Check's `PromptAnalysis.author_orders` too - never an empty
  "Recommended order:". The caret line names the place and, under an order,
  the steps a tag there fits (`tagCaretSteps`): from the step of the
  nearest tag before the caret to the nearest after it that the server's
  chip slots place (`chipSlots`; unplaced tags passed over, refreshed when
  the slots land), "start with" the first step in an empty prompt, and the
  tags around it out of order where those two are reversed. Where no tag
  around the caret is placed it tells the cases apart (`tagCaretSteps`'
  `answer`): "still being looked up" only while the answer for this very
  prompt is pending; otherwise it names the whole order and why - Check
  reads no order in this prompt (`slots` None, or no request was made),
  the lookup failed (`tagSlots.failed`), or Check places none of those
  tags in one step - never a "yet" that will not resolve. Those steps
  are marked (`.fits`) in the order line and on the groups whose tags go
  there; a group heading names its step where that is not its own name
  (`group_steps`, `year_step`: Animagine's "Score · Quality step"). A
  guide's `recommended_prefix` (Anima Turbo's card prefix alone; Aesthetic
  has none, its card advising against `score_*`) is a "Start with ..."
  button, offered and never a finding: it writes the prefix tags the
  prompt has not got ahead of it through `writePromptTags` (so `writeBox`,
  and Ctrl+Z), skipping a tag whose exclusive group already holds another
  (`safe` beside `sensitive`, `nsfw` or `explicit`; `prefixMissing`), and
  is held with the reason as its title where it would add nothing. Check's
  no-order note says "its guide states none" only where no author order
  follows it. `tests/test_tag_order_agreement.py` holds the ? text, the
  menu and Check to one order: the served order and author orders equal
  Check's, every group placed in a step, distinct headings, every step
  named in the ? prose. An Escape the suggestion list or
  a chip editor took closes that alone (the document's Escape skips
  `defaultPrevented`); the next closes the helper, and the one after that an
  open prompt check. In the Refine dialog the list's Escape is prevented on
  the box's keydown, which keeps the modal dialog's cancel from firing: it
  closes the list, and the next one closes the dialog.
- Keyboard events already prevented, composing (including keyCode 229), or
  carrying unsupported extra modifiers never become a second page command.
  Discrete activation does not repeat; navigation, zoom, numeric stepping and
  the mask's held stroke retain repeat. Native repeated Enter/Space activation
  is held for discrete buttons too. A focused input, select or editable owns
  its text/navigation keys. The event's native modal owns dispatch; settings,
  help, Bundles, menus and the lightbox cannot generate behind themselves.
- Ctrl/Cmd+Enter (exactly one primary modifier, no Alt/Shift) is routed by focus (app.js's keydown): in the Refine dialog it
  runs the pass, and so does a plain Enter in the dialog's one-line boxes
  (`refineKeydown`); in Mass refine's dialog it runs that
  (`runRefineBatch`) and in Swap tags' it queues (`queueSwap`); from a box
  of the open grid fold it clicks Run grid;
  on the main workspace, a focused image included, it generates - `thumb()`'s Enter
  and Space open the lightbox only without Ctrl, Meta or Alt. Generate
  itself refuses in the page alone (never the server, where braces can be
  text or {tag} emphasis) while the fold is open, its variables box has text
  and either prompt holds a `{name}`. Generate, Run grid and Refine show the
  hint through one `::after` rule (empty alt text, smaller and lighter, never
  opacity, since `setBusy` rewrites the buttons' text) plus
  `aria-keyshortcuts`; the lightbox's Compare with source shares that rule,
  its content S (style.css's lightbox section).
- `/` focuses and selects gallery search on the workspace, preserving its
  query, filters, bin/run and picks. A literal slash may carry Shift on a
  localized keyboard. Fields, menus, popups and modals own their keys;
  Ctrl/Meta/Alt, composition, handled events and repeats never take this route.
  Explicit navigation reveals a clipped search, without moving a visible one.
  Unshifted M opens More seeds for the focused gallery/result/sheet/session
  image or unobstructed lightbox image, using its one id and the existing holds
  and preview. It never queues or widens to hidden picks. Its actual visible
  opener owns focus and pending-preview validity, including narrow lightboxes.
  Visible M hints stay in image context and help; the More seeds entry points
  remain. A narrow lightbox menu uses its visible launcher instead of the hidden
  button for pending-preview validity. Lightbox redraws preserve valid focus;
  when their owned control disappears or becomes hidden/disabled, Close receives
  focus without scrolling, unless another dialog is open. Comparison controls and S retain their existing behavior.
- The grid preview is the server's expansion; the page adds only
  presentation and advice. It trims factors of one other than the seeds
  from the totals and writes ×, since `grid.describe` keeps its ASCII x and
  its ones for the CLI's log. It lists the resolved prompt whenever the
  variables box has text, one included, in a `.prompts` box of its own
  after the axes line, at most eight lines tall and scrolling inside itself
  (a nested scroller, never the panel's overflow). A prompt written over
  lines is listed by the lines a `{variable}` is on, whole and numbered
  (`promptListing`: "prompts, line 4 of 29:"), which rests on
  `grid.substitute` keeping every line in place. Its client-side warning lines (a
  variable of one value, a budget that completes no seed, the panel's
  Repeat above one, cells already in the gallery) never refuse anything, and
  are appended to `#grid-preview` as nodes, never by rewriting `textContent`,
  since `.totals` is its first child. The budget line's count is the
  server's (`GridPreview.budget_seeds`, `grid.seeds_within_budget`: the
  job's stop rule, a cell starting until the deadline has passed, so a seed
  whose last cell starts in time completes); it needs the preview's
  estimate, is left out under Resume, and names what else spends a budget
  that runs from submission: the load while the topbar's dot is not ready,
  the jobs ahead while the page watches one. The preview is asked again by
  `setBusy` (a job's start and end) and by `refreshStatus` when the model
  becomes ready or stops being so, not only by the boxes; and the first
  preview after the fold is opened by hand reveals Run grid again
  (`gridRevealPending`, set by the summary's click), since its reply lands
  under the pointer where the panel holds still. Axis messages lead with the
  box's name (`Steps`, `CFG`, `Mu`, `LoRA weight`, `Strength`, `Seed list`),
  a value past the request's own bounds included (`grid._within`, read off
  `GenerateRequest`: "CFG: 25 is above 20, the most it takes");
  `parse_axis` closes up a spaced range between two numbers, leaving a lone
  `-` as Mu's none, and refuses a stepless range that would stop at its
  first value (`0.8-1.2` needs a step). The grid's Steps and CFG
  placeholders are the profile's (`gridExamples`, from `applyDefaults`):
  Steps round(.75s), s, round(1.25s), clamped to the Steps box; CFG c-1, c,
  c+1, or c, c+1, c+2 where the profile's CFG is at or below a threshold
  that is CFG's own, not True CFG's (guidance off: Anima Turbo and ERNIE
  Turbo give 1,2,3), 0,0.5,1 for a CFG-0 profile, and nothing where
  `cfg_ignored`; Mu `none,<profile mu>` where the profile has one. The tips
  stay syntax help. A grid axis dead for the checkpoint or the installation
  reads "unavailable", and its reason is its tip and its wrapper's title -
  for Mu `hints.mu` (`syncMuWords`), for the LoRA weight `/api/loras`'
  detail once that list answers (`loraRefusal`, `syncConditioning`); the
  syntax help is kept as nodes (`gridMuTipDefault`, `gridLoraTipDefault`)
  and comes back where the axis applies.
- A refine pass is the Refine dialog's (`web/refine.js`), never the
  panel's: `collect()` sends `refine: null`, so Generate, Batch, Repeat, the
  grid and the comparison always render from the prompt, and the grid has no
  strength box. `openRefine(rec, pass)` - a card's ⤢, a viewer result's ⤢ and
  its column's Refine…, the lightbox's Refine (which closes the lightbox; a
  close that runs nothing gives it back on the same image of the same strip
  at the same view and comparison - `refineReturn` from `lightboxPlace()`,
  resolved by id in the current gallery or viewer strip, the comparison put
  back before the view, whose scale is relative to the fit of half the stage
  side by side - unless the passes strip has opened it on a pass already)
  and `reuse` on a refined record, which first asks `GET /api/images/{id}`
  for its source - fills the dialog from the record: prompt, negative, steps,
  CFG, Mu, True CFG and an adapter the list still offers, or this
  checkpoint's profile where the record is another checkpoint's (the fold
  opened, saying so, naming only the knobs this checkpoint has: mu only
  where Mu is supported, as Reuse's note does; a pass's strength is
  replaced too, and
  `#refine-reused` says whose the old strength was and what Strength starts
  at - the strength last run on this checkpoint, else its own where
  `defaults.refine_strength_profiled`, else "the general starting
  strength"); the seed; the remembered scale or 1.5x where that fits
  the pass's guard and is not past the tight line (`refineTight`: the
  profile's `refine_tight_pixels`, else `sizeIsTight`), else the largest
  factor that is, else 1x - a scale passed over says so on its button
  ("(last run at 2×)" for a remembered one, "(the usual opening size)" for
  the 1.5x default; `refineSkipped`), and while W and H are the size it
  opened at the size line names the next scale up and the line it is past
  (`refineCapNote`: "1.25× and larger go past the 1536×1536 (2.36 MP)
  measured on Z-Anime at fp8", or the generic "may run tight on 16GB
  VRAM"); a run at the size it opened at in place of the scale asked for
  leaves the remembered scale alone (`refineCapped`) - unless settings a
  close left unsent on the same image and checkpoint are restored
  (`unsentRefine`, page memory only, never `auradream.refine.v1`),
  which it says, with Start over; a run, another image, Reset and a
  checkpoint change forget them. It opens focused on Strength, its value
  selected. The fold's Prompt field is a div with `<label for="refine-prompt">`.
  From panel, at its caption's end, puts the panel's prompt and negative
  into the dialog's boxes (the negative first, so the prompt's Ctrl+Z is on
  top) and never writes the panel; every whole-box write into the dialog's
  prompt or negative (Apply, From panel) goes through `writeRefineBox`
  (`replaceBoxText`, then no suggestion list over the new text), which
  leaves a greyed box alone, since Chromium keeps no undo step for an edit
  to a box greyed again straight after it. While the dialog's negative is greyed
  the press copies the prompt alone and says to set True CFG (or CFG) above
  the threshold and press again, as its title says beforehand
  (`refineFromPanelPlan`); on a checkpoint with no negative prompt the
  negative is not compared. It is aria-disabled with the reason as its
  title (`syncRefineFromPanel`, from `syncRefine`) while there is nothing it
  would write: the boxes hold the panel's texts already, only a greyed
  negative differs, or the panel's prompt is empty. What a press did not do
  is said on a status line of the caption's own under the button
  (`#refine-from-note`, `sayRefineFromPanel`): empty, it has no height; it
  is scrolled into the dialog's view and cleared by `refineChanged`, an open
  and a close; it is never `#refine-error`, which sits at the dialog's foot,
  out of view with the fold open at 1400x900. `refineSource` is that record
  while the dialog is open and null once it closes. A card's or viewer
  result's ⤢ on a checkpoint without a
  pass is aria-disabled with the reason (`REFINE_NONE`, as the lightbox's
  Refine), and a press says so in `#gallery-error` and for two seconds over
  the card's tools (`refuseInCard`), opening nothing - decided in `card()`
  and `refineResult`, not in `openRefine`, whose own refusal (the lightbox's)
  is `showError(REFINE_NONE, true)`, mirrored into the lightbox; a refined
  record's Reuse opens it anyway, Refine aria-disabled with the reason, until
  status says the pipeline has one; `#refine-unavailable` hedges ("A load
  answers for certain") only while `defaults.refine_settled` is false, and
  is `REFINE_NONE`'s sentence where the answer is final - a loaded
  pipeline's, or one of `model_defaults._OWN_PIPELINES` (Anima Turbo/Aesthetic,
  NewBie). A source gone at Reuse fills the panel
  instead, its size unchanged, with a note. Versions is `num_images` on
  consecutive seeds; a Strength holding a list or a range is a `/api/grid`
  strength axis with Versions as `seeds`, opened as its sheet. A pass starts
  through `startJob(path, body, { sheet, panel: false, slot: els.refineError,
  placeholder, onAccepted })`, which never writes the panel's Seed box; the
  dialog closes in `onAccepted`, so a refusal at submit (a source deleted
  elsewhere) stays in `#refine-error` with the dialog's edits - a gone
  source in the page's words, "#N has been deleted, so it cannot be
  refined." (`refineSourceGone`, the preview's refusal and Refine's title
  too), the source line ending "#N (deleted)"; a binned one "#N is in the
  recycle bin; restore it to refine it.", as bin.js says it, the line ending
  "#N (in the recycle bin)". The carried adapter (`refineLora`, no control
  of its own) refused by name, at the preview or at submit, reads
  `/api/loras` again (`dropRefineLora`); one no longer offered leaves the
  pass, `#refine-carried` reads "LoRA NAME is not applied: it is not in the
  adapter list now." and `#refine-error` "NAME is no longer in the LoRA
  folder, so it was taken out of this pass; Refine runs it without the
  adapter." The preview is asked again without it, and a press that was
  waiting on the preview that took it out runs nothing (`refineLoraDrops`).
  The placeholder is the source's full image (`url`). While the POST is out the
  dialog holds (`refineSending`): Refine reads "Sending…", and `closeRefine`
  (Cancel, ✕), the dialog's prevented `cancel` (Escape) and the passes strip
  wait for the answer; a close the browser will not hold counts as a run in
  `refineClosed`, and a refusal after it is said in `#error`. A pass run from
  a dialog the lightbox opened puts focus on the source's gallery card where
  it is shown (`isShown`), else Generate, with `preventScroll`, through
  `handJobFocus`; any other run whose close leaves focus on
  `<body>` (an open from script, Reuse of a pass) puts it on Generate, or New
  image where that is shown, on the next frame. Every change asks `/api/grid/preview` with
  the grid-shaped request; its refusal, a size past the guard or no
  capability makes Refine aria-disabled with the reason as its title, and a
  press says it in the dialog. Every way of running a pass first waits for
  the preview of what the boxes hold (`refinePreviewed`, the request the last
  preview answered), since the debounced preview is what refuses a strength
  past 1, one that leaves no step and a source deleted meanwhile; a refused
  press moves the preview's sentence to the error line in the preview's
  place and at its size (the preview hidden until the next change) and
  focuses the field it is about, and the dialog's column never shrinks its
  items. Its Strength help is the profile's: the placeholder, Compare 3
  (`compareStrengths`) and the tip's example (`model_defaults.strength_sweep`)
  are the same three values - half, the profile's strength, half again on
  the 0.05 grid, half rounding up on the quotient snapped to six places
  (Z-Anime's 0.15 is 0.1,0.15,0.25), less any that leaves no step - one
  arithmetic in the same order in both, held by `tests/test_profiles.py`;
  Compare 3 only fills the box, and where one value is left out it reads
  Compare 2, its title naming the values it writes and the one left out and
  why (`compareSweep`), the label written only when it changes. Where Steps
  leaves fewer than two of them a step (a source made at 2
  steps), Compare 3 is aria-disabled, its title naming the Steps that would
  give a sweep (`compareStepsNeeded`), a press says so and writes nothing,
  the placeholder is the start alone and the sweep line names no values.
  From 1600 px the dialog is two panes (the source's PNG, the kept
  region of a reshaped size framed with `refine.fitted`'s arithmetic,
  `refineKept`), with the same DOM and tab order as the one column; Keep
  shape moves the other side on trusted input only. The dialog remembers the
  scale, the versions and, per checkpoint, a single strength the user ran
  (`auradream.refine.v1`); `resetPanel` forgets it.
- An inpaint is the Inpaint dialog's (`web/inpaint.js`, its canvas
  `web/inpaint-mask.js`, `styles/inpaint.css`), never the panel's and never
  a mode of the Refine dialog: `collect()` sends no mask. `openInpaint(rec,
  from)` - a card's ✎ (last before Delete), a viewer result's ✎ and its
  column's ✎ Inpaint…, the lightbox's Inpaint… beside Refine (closing the
  lightbox and given back on a close that runs nothing, as Refine's is) and
  Reuse on an inpainted record ("Reuse inpaint…", on its source with its
  stored mask from `GET /api/masks/{hash}`, its strength and seed). Where
  the checkpoint has none each entry point is `aria-disabled` with
  `inpaintNone()`'s sentence as its title (guess or final, deferred or
  absent) and a press says it and opens nothing (`syncInpaintTool`,
  `syncInpaintCapability`). Requests carry `refine`, `mask` and an explicit
  `strength` at the source's size; a strength list or range is `/api/grid`
  with `strength_axis` over the one mask, opened as a sheet. Strength
  starts at `defaults.inpaint_strength`, said as the installed diffusers'
  default followed by `hints.inpaint_strength` (where it was measured as
  the right start) - or, where it is `defaults.inpaint_start` (FLUX.1's
  0.95, known before a load), as this checkpoint's own start in place of
  that default; otherwise before a load, the checkpoint's refine strength, said as a
  stand-in, and replaced by the library's when a load answers and the box
  is untouched (`takeLibraryStrength`). The canvas's bitmap is the source's
  pixel size over the source's `<img>`; the mask shows as a tint and an
  outline. Undo replays strokes (`MASK_UNDO`, 200; older strokes are drawn
  into a base for good), never bitmap snapshots. Brush, eraser, size (`[`
  `]`, the slider, Alt+wheel; a plain wheel scrolls), feather (shown live,
  baked into the levels at export), invert, clear, undo and redo act only
  while the canvas has focus, so a box's Ctrl+Z stays the box's; the
  canvas is named "Mask over #N", and arrow keys move a keyboard brush that
  Space paints. Each tool's key is in `aria-keyshortcuts`. The mask exports
  as a grey PNG (`exportMask`: the feather blurred from a copy padded with
  the layer's own edge pixels, `featheredDraw`, so a mask reaching the
  border does not fade there; levels under `MASK_FLOOR` sent as black; a
  mask with no level from 128 up is nothing masked) and uploads 300 ms
  after a stroke ends (`POST /api/masks?source=ID`); a stroke waits while a
  Reuse's mask loads. A stored mask is kept with `POST
  /api/masks/{hash}/keep` on a Reuse, every 20 minutes while the dialog is
  open and before a press past that, and is uploaded again from the page's
  strokes where it has gone (a keep's 404, or a preview or submit refusal
  naming it). An adapter gone meanwhile is taken out as Refine's is
  (`dropInpaintLora`). Inpaint is `aria-disabled` with its reason
  while nothing is painted, while the mask uploads or loads, and while an
  upload or the preview refuses, and a press says why; "wait for the mask"
  clears once it lands. Every fill of the dialog (Start over included)
  ends what the last one had in flight (`inpaintOpenEpoch`), and only a
  first open scrolls it to the top. The tools row and
  the status line have fixed heights, the dialog is anchored at a fixed top
  and grows downward, and nothing scrolls it or moves a control under the
  pointer. A close that ran nothing keeps the strokes and the boxes in page
  memory (`inpaintKept`, four images, never storage) for the next open of
  that image on the same checkpoint; Start over forgets them. The prompt
  check is Refine's lines (`refine-prompt.js`; `refineCutLine` takes the
  box to select). The two-pane layout is only in its `@media (min-width:
  1800px)` block. Where it does not fit, the card's ✎ hides in the wide
  tier (a container query) and at S density, the docked result row drops
  ✎ Inpaint…, and the lightbox's Inpaint… hides under 901 px; another entry
  point remains in each case. Cards, facts, the settings line and job
  labels read "inpaint of #N @ 0.6", the facts list has a Mask row (the
  mask as a tint and an outline over the source's thumbnail, in an 80 px
  box of the mask's shape, then the hash's first 16 digits; `maskOutline`
  in `facts.js`) and an inpainted card a ✎ badge. The outline reads the
  stored PNG (`GET /api/masks/{hash}`) to the canvas's device pixels, a
  level above 0 counting as repainted, and draws no line along the
  picture's border; a deleted source (or one whose thumbnail will not
  load) leaves the record's own picture under it, a binned one is asked
  for with `?binned=true`.
- An outpaint is the Outpaint dialog's (`web/outpaint.js`,
  `styles/outpaint.css`, wearing `.inpaint-dialog`'s box and panes), never
  the panel's and never a mode of the Inpaint dialog: `collect()` sends no
  outpaint. `openOutpaint(rec, from)` (async: it asks `GET
  /api/images/{id}/outpaint-presets` first) - the lightbox's Outpaint… in the
  Refine/Inpaint row (one row, never wrapping; under 901 px Inpaint… hides
  and Outpaint… stays, the pair at their own widths, since a gallery image
  has no card glyph for it), a viewer result column's ⬚ Outpaint… and Reuse on an
  outpainted record ("Reuse outpaint…", asked before the inpaint's since the
  row has a mask too; `dataset.reuseOutpaint` holds `OutpaintSpec.key()`, so
  `syncReuseButtons` rebuilds it). No card glyph. Where the checkpoint has
  no inpaint class each entry point is `aria-disabled` with
  `inpaintNone(true)` (", which an outpaint runs through") as its title and
  a press says it. Requests carry `refine`, `outpaint` {fill, four sides,
  band} and the canvas as width x height, never a mask; the page draws no
  fill and does no preset arithmetic - the server answers every preset's
  sides, canvas and `held` words, `off_grid` and the guard's limits. A
  side box is committed (change, Enter, blur) rounded to 16 half up, and
  the band to 8 within `max_band`, the status line saying the rounding; a
  number already on the grid is taken as typed, and a typed side that
  moves the canvas clears what a preset or a rounding said. A preset
  replaces the four sides and keeps the band. To shape… only picks - an
  arrow key on a closed select fires change on every press - and its
  Apply (or Enter in the list) applies the pick, held with "Pick a shape
  first" or the preset's `held` words. Outpaint is `aria-disabled` with its
  reason while no side is extended, the source is off the grid, the canvas
  passes `max_side` or `max_pixels`, the band leaves less than `min_kept`
  kept (in the server's words), the preview refuses (an unknown fill by
  name; a source gone or binned, said as `sourceGoneWords`), a submit is in
  flight ("Wait for the last submit to be taken") or the checkpoint has
  none. A presets answer that lands after the user moved on - the lightbox
  it was pressed in dismissed or on another image, or another dialog
  opened - opens nothing; Start over reopens past the support gate, so the
  dialog resets and its unavailable line says why. A presets 404 or a
  Reuse whose source is binned says "restore it to outpaint it" (Reuse
  reads the source with `?binned=true`, as Reuse of an inpaint does).
  The frame is a box of a fixed size with the side boxes round it at fixed
  places; the status, frame, presets and band rows have fixed heights, the
  empty preview line holds its height and a refusal takes its place, and
  the dialog has `scrollbar-gutter: stable`, so nothing moves under the
  pointer. A close that ran nothing keeps the sides, band and boxes in page
  memory (`outpaintKept`) for that image on the same checkpoint. Its
  shared helpers are inpaint.js's (`inpaintStartStrength`,
  `libraryStrengthInto`, `syncPassSampling`, `renderPassCheck`,
  `syncPassCheck`, `selectPassFrom`, `sourceGoneWords`). Cards, the
  settings line and job labels read "outpaint of #N +L/+T/+R/+B @ S", the
  facts "Outpainted from" and an Outpaint row, an outpainted card a ⬚
  badge; Against source draws the source at (left, top) at the result's
  scale, and the Mask row's box (the canvas's shape) the source's
  thumbnail at its place by the padding.
- `startJob(path, body, { sheet, panel, slot, placeholder, onAccepted,
  onRefused })`: `slot` (default `els.error`) is where its refusals go.
  `onAccepted(jobId)` runs once the server has taken the job, before
  startJob watches it, though a listing between jobs (`jobListing`) may
  already be streaming it; a refusal never calls it, and calls
  `onRefused(err)` once said in `slot`, for the caller's own words. `placeholder` ({src, width, height, label}) is
  keyed to the job (`jobPlaceholders`); the stream reads it again at each
  progress event until it is there, and shows it from the first such event
  with no frame as the viewer's `figure.pending` (the source dimmed and
  blurred, in the result's own tile, its caption the tile's `p.foot`) until
  a frame or the result replaces it. startJob resolves on a refusal, once
  the job has ended, or, for a job queued behind the watched one, once the
  server has taken it. A submission queues, with no `setBusy`, while the
  page watches a job or while `jobListing` is out. `jobListing` is the
  request a page between jobs makes to pick its next job (`adoptListedJob`,
  behind `watchNext` and `resumeActiveJob`); it checks `busy` again once it
  answers and opens the adopted job's stream itself. After its POST, a
  queued submission waits for that listing, then: a job the listing made
  the watched one is awaited through its stream (`jobStreams`), or
  forgotten (`forgetJob`) if that stream has already ended; otherwise it
  takes the page if nothing is watched, and else queues an id that is not
  queued already. So no job is streamed twice (`watchedJobs`, every id a
  stream was opened for) or listed as its own Next: row. An accepted in-memory
  receipt not yet watched (`unwatchedSubmission`) also owes an outcome if its
  job finished between watches. The adopter queries finished history only
  while such a receipt is absent from the active list, then replays terminal
  snapshots one at a time through the ordinary stream/finish path in receipt
  order. Queue resync preserves those unseen receipts, and idle polling retries
  transient listing failures. `ownJobs` alone never authorizes historical
  replay after reload; foreign jobs and explicitly cancelled/forgotten rows
  are excluded. A receipt missing from server history is forgotten.
  A stream that gives up probes
  `GET /api/jobs/<id>`: alive, it streams again (three times at most); no
  answer, an offline message counting the saved images; a 404, "The server
  restarted, and this job ended with it after N of M images". The stale
  preview is dropped, and where nothing of the job reached the strip the
  viewer holds `showLostRun`'s card (the sentence, Run it again where this
  page sent it, `lostRunBack` rewording it once the server answers). While
  it reconnects the job line reads `LOST_PHRASE` (the topbar's "server not
  answering — retrying") with the count, no steps or time left, no topbar
  mirror (`streamLost`), the title and the strip's note the same phrase. The
  progress block is a one-line Now row
  (`#progress-now`, count · size · what - "Now: 3 images · 1024×1024 · a fox
  in the snow" (`describeJob`), the prompt last so the ellipsis takes it,
  whole in its title; an XML prompt is named by its words, its
  `<name>`/`</name>` tags dropped (`promptGist`), here, in the queue rows,
  in Cancel's title and in the Runs picker's labels, the title keeping it as
  written) with Pause and Cancel at its end, the bars at the
  panel's width, and under them one line holding the counters
  (`#progress-text`, then `#step-text` after a dot) and the live cell line
  (`#cell-note`), all inside `#progress` - the rows are `display: contents`
  and `.progress` a wrapping flex line, so a long cell line wraps whole to a
  line of its own; a counter beside each bar put the time on a fourth line
  past the foot of a 900 px window at scroll 0. Never a counter back beside
  its bar, nor anything between the bars and Generate but the Now line;
  queue rows read the same, and a single queued job is one "Next:" row. A
  finished job's outcome and actions (Use seed, More seeds…, Show sheet,
  More seeds of this run…, Retry/Resume) are the finish line, `#finish-note`, outside `#progress` (`setFinish`): only
  the next job's end replaces it, never a queued job's start; it is led by
  "Previous job:" while another runs or after a lost stream (`finishStale`).
  Its sentence, without buttons, is said once on the visually hidden polite
  `#job-said` (`sayJob`) unless `#error` says the job's outcome (a cancel,
  an error, skipped cells, a load's detail where the viewer holds images),
  and so is "Queued: <job>" when a press queues one behind the watched job.
  A job's outcome in `#error` - cancelled, its budget, an error, skipped
  cells (`showOutcome`, `jobOutcome`) - lasts as the finish line does: led by
  "Previous job: " while another runs (`#error.previous`, a generated
  `::before`; `showError` drops the class) and cleared when the next job's
  end is drawn (`finishJob`). A press on a held Pause, Unload, Expand or Ask
  the critic says why until the job ends (`showJobNote`, cleared by
  `setBusy(false)`; Expand's goes through `expandSay`). Every held press
  says its reason as a sentence with a closing stop; Pause's, Unload's and
  Expand's titles are the same reason without it (the critic's titles are
  whole sentences already). A job that
  ends with focus in `#progress` or `#queue-note` hands it to Generate and
  the finish line takes it on to its first button (`finishTakesFocus`);
  replaced with focus inside it, the finish line hands focus to its new first
  button, else Generate. A finished job's images are prepended to the
  gallery (`prependFinished`: same filters, `made` + PAGE, chained after
  `galleryPending`, dropped on an epoch change, reset where it does not join
  up or the gallery is empty, never in sheet mode; the head's count,
  `other_models`, Re-render, Delete unstarred and a gallery lightbox's
  counter are drawn as the page lands, and `prependFinished` answers the
  page's ids, so `noteOutsideView(images, coming)` says its line at once.
  The cards go in through `joinFinished`: only where some cards are new,
  it first waits until the pointer is off the gallery's cards, in one
  column as beside the viewer (`galleryPointerGone`: the `galleryPointer`
  flag from `#library-scroll`'s pointerenter/pointerleave, which fire under
  the viewer too, where it is `display: contents`; as `panelPos.pointer`,
  never `:hover`, which a tap leaves behind; beside the viewer that means
  the column's cards, gaps, scrollbar and Load more), not while `#gallery`
  is not shown (the column hidden under the check), released by
  `galleryPointerLeft` or by `syncBeside` changing columns, then checks again that the page
  joins up; a column scrolled down keeps its view (`holdGalleryView`: the
  first child in its view, measured across the insert); `now` skips the
  wait), never `resetGallery`,
  drawing the day's lines as `loadGallery` does - over the first new card
  where its day is not today's, between two new cards, and between the last
  of them and the old head read off the grid's first card (`data-made`, a
  tombstone's too; none where the grid opens with a line) - and giving the
  grid's tab stop to the newest image where it was the default
  (`galleryChosen` false: set by focus arriving, cleared by `resetGallery`)
  or the old head with focus outside the grid, else `keepCurrentCard()`, as
  anything adding cards must. "New image ↑" or "↓" (`#btn-new-image`)
  appears when a job's first image or frame lands while the viewer is out of
  sight, above or below the window: `viewerOutOfSight` answers above or
  below at any width, and `syncNewImage` writes the arrow and its
  aria-label and follows a scroll that jumps past the viewer. Under 901 px
  it is a pill fixed and centred under the sticky topbar, never in the
  window's foot, where it covered Generate's right end and took a press
  meant for Generate. Its press scrolls the viewer into view and focuses the
  newest image; before one has landed, the live preview's tile (or a pass's
  stand-in), which `streamJob` hands on to the image that replaces it, or to
  Generate where nothing does; with nothing in the viewer, the viewer itself
  (tabindex -1, let go on blur) or its held line. It goes
  (`withdrawNewImage`, with focus on it handed on, never to `<body>`) once
  the viewer is in view, when a job's first frame or stand-in clears the
  strip (`landed`, never at `setBusy(true)`: a job that never reaches the
  strip, such as another tab's job picked up over this page's results,
  leaves it pointing at this page's image), when a job whose first frame
  cleared the strip ends or gives up with nothing of it in the strip,
  and - the pill only, once the job has ended - at the
  user's own press in the panel or typing in one of its boxes
  (`newImageBackToPanel`), since it lies over whatever has scrolled under
  the topbar; never at a focus arriving, which the finish line makes
  itself. The page never scrolls by itself. `setBusy(true)` hides the
  last-run banner without `remember()`, and `announceLastRun` checks `busy`
  again after its requests, so the banner never comes up over a job
  started while they were out; the done branch remembers the job. The tab's title
  and the aria-hidden topbar line (`#job-mirror`, `#topbar-line`) mirror the
  job, and the finished title is held until the tab is looked at again.
- `api()` turns fetch's own TypeError (never an HTTP error or an abort) into
  an Error with `offline: true` and `OFFLINE_MESSAGE`; HTTP errors carry
  `.status`. `showError(message, note, { slot, mirror, offline })` records
  the slots holding such a message (`offline` defaults to whether the text
  contains `OFFLINE_MESSAGE`), and `clearOffline()` empties them on the first
  answered status poll. A poll that gets an offline error calls `goOffline`:
  dot error, "server not answering — retrying", Load hidden, Generate and Run
  grid aria-disabled with the reason (`syncSubmitButtons`, their one writer,
  its reasons from `submitHold()` - no server, Expand, the critic; the
  POST's own moment keeps the plain title - which has the Refine dialog's
  Refine follow through `syncRefineRun`, ahead of its own reasons, a press
  saying it in the dialog as a sentence, and the comparison's Run through
  `syncCompareRun`), Unload, Expand and Ask the critic held. A changed `started_at` writes the restart note into an empty
  `#error`. The lightbox's picture is one such slot: `els.lbImg.onerror` asks
  for the record and, where the server is not answering, writes
  `OFFLINE_MESSAGE` into `#lb-error` as an offline message, which the poll
  that answers clears before asking for the picture again
  (`retryLightboxImage`); otherwise "#N is no longer in the gallery." (an
  unknown image), "The file of #N could not be loaded." (the record exists)
  or "This image could not be loaded." (any other refusal), cleared by the
  next picture that loads. `syncUnload` is Unload's one writer (busy,
  another tab's job (`elsewhereBusy`), Expand, the critic, a load, offline):
  `aria-disabled` with the reason as its title,
  never `disabled`, and a press says why. A press held by this page's own
  reason (a load, Expand, the critic, a model change: `unloadHeldNote`) is
  taken back from `#error` once `syncUnload` finds that reason gone, as the
  job's and another tab's are. A press a model change holds (Generate and
  every `startJob`, Load, Check, the critic) says so as a note through
  `sayModelChangeHold(slot)`, which `syncSubmitButtons` takes back
  (`clearModelChangeHolds`) once `modelChangeHold()` is empty. Unload is hidden while Load is
  shown, and the press that hides one of the two hands focus to the other. A
  failed load is a card in the viewer's empty state (`syncLoadFailure`:
  heading, detail, model page link, a primary Retry load calling
  `loadModel`; `loadFailureInViewer()` is whether it is shown). `loadModel`
  takes back only what was said about the load - in the panel's line its
  own refusal, Expand's pointer (`loadLine`) or the status's sentence
  (`shownLoadDetail`), and under the prompt Expand's words about the load
  (`aboutLoad`) - never the line's other messages; a card that goes with
  focus in it hands focus to the new card's Retry load where one replaces
  it, else to Generate.
  Where the viewer holds images the detail goes to `#error`, once per
  detail, with `mirror: false`, and again after a job that ended at the load
  (`finishJob` forgets `shownLoadDetail`, since the press that queued it
  cleared the line); that job's finish line says "Not made: the model could
  not load" and where the reason is, in a `.not-made` span (`notMadeText`),
  said once on `#job-said` where the viewer's card holds the reason (the
  card is no live region; where the viewer holds images `#error` says it).
  Once the state leaves error, in every tab, `syncLoadFailure` rewords it
  ("…could not load then; it is loading now" / "it has loaded since" / "it
  is not loaded now") and takes back the detail it posted to `#error`
  (`loadDetailInError`). The card is centred in the viewer. The topbar's
  Load reads Retry load meanwhile, and the status text keeps the
  checkpoint's name in every state. That name is the start-up catalogue's
  (`ModelStatus.model_label`, by exact id, through `checkpointName`, which
  the prompt check's head, the facts' Model row, the load card, a Reuse's
  note and its ⚠ title, Expand's running line and the Refine dialog's lines
  share), or the id's last part off the catalogue, with the id in their
  titles; another checkpoint's records are named by their id's last part. The
  status text is a live region, so it is written only when it changes. At
  700 px and below, while `#job-mirror` is shown, the free-space reading
  (`.vram`) is hidden so the mirror and its load clock fit; the mirror's
  `margin-right: auto` keeps Load and Unload at the bar's end; wider, the
  free-space reading stays. While a job waits on the load, a client ticker
  writes "loading weights" with an `aria-hidden` clock span ("· m:ss") in
  `#progress-text` over an indeterminate bar; while the strip is empty (its
  idle or held `.placeholder`), `viewerWaiting` (panel.js) also puts
  "Loading weights · m:ss - the first load of a session takes a few
  minutes." in that line - the clock aria-hidden, one line, `.waiting` -
  and the line gets its own nodes back when the ticker stops. The stream's
  error handler stops the ticker. `#progress-text` is otherwise written
  through `setProgressText`, only when its spoken text changes (`tick`
  writes the ticker's words and clock itself); a single image's step count
  is an aria-hidden shown half (`.seen`) beside a visually hidden
  "denoising…" (`.said`), and the mirrors and the title read
  `progressShown()`.
- `showError(message, note, { slot, mirror, offline })` says each message in
  one slot (both are role=alert). What a gallery control did - Delete
  unstarred, a failed gallery page, Re-render's submit errors, a card's ❐,
  ⊞ and ⤢ refusals and a pass's Compare with source refusal, deletes and
  the recycle bin's note with its Restore (`noteBinned`), Empty recycle bin, a
  star's failure (a card's, a viewer result's, a sheet cell's or the
  lightbox's), the sheet's Resume this run refusals, Fill width pressed on
  a wrapped row, an export pressed while one runs, Export all's outcome and
  refusals, and its export failure - is said in `#gallery-error` (`els.galleryError`), which
  `resetGallery` clears; Reuse's notes stay in the panel's `#error`.
  `#gallery-error` is the first item of the gallery head's last row
  (`.gallery-footer`, a full-width flex item inside `.gallery-head`, styled
  in gallery.css with the footer's buttons; bundles.css holds only
  Bundles…), taking what the buttons leave, and its line is kept while it
  is empty (`#gallery-error[hidden]` is `visibility: hidden`, not
  `display: none`), so a message never moves the grid under the pointer.
  It is one line whatever the message (`white-space: nowrap` with an
  ellipsis): a long one is cut, and `showError` makes the whole text the
  title of a `.one-line` slot.
  Nothing in the footer row it shares may let it wrap.
- Profile defaults apply per control (`owned`); an empty Steps or CFG box is
  sent absent and 0 is a valid CFG. The panel persists to `localStorage`
  with every access in try/catch. A write is a merge (`writePanel`): only
  the keys this page changed since it last read or wrote the store
  (`panelBase`, seeded from the store by `restorePanel`) go onto what the
  store holds now - each field, the fold, the references, the translation
  and each sampling value - and an empty store takes the whole panel.
  Nothing listens to `storage`. `resetPanel` is the one thing that forgets
  it, since clearing the gallery never reaches the browser. The cards' S/M
  and Fit/Crop (`cardView`, gallery.js) belong to the gallery: they are
  stored under `auradream.gallery.v1`, every access in try/catch,
  read back at load (`restoreCardView`), and Reset leaves them. Reset first
  keeps an in-memory snapshot read off the page (`panelState()`, the raw
  sampling values, `owned`, the size preset, `REFINE_KEY`'s value and
  `unsentRefine`). It empties the prompt and negative through `rewriteBox`,
  so Ctrl+Z in the prompt brings the text back, forgets the store after
  those writes and keeps it forgotten until the panel changes (`resetLeft`:
  a write of the panel as Reset left it, such as the prompt's later blur,
  is skipped). It says "Panel reset." with an Undo in the top right of the
  prompt box it has just emptied, under Reset (`#prompt-note`, kind
  'reset', `offerResetUndo`, `showPromptNote`'s `at: 'box'`,
  `data-at="box"`). This holds at every width, never in the panel's flow:
  in the flow (one column, a greyed negative) it moved Generate down a
  line, and back up under the pointer when its 20 s ran out. The box is
  empty while the note shows, since the first panel change takes it, and
  its placeholder is transparent under it. It is retired by that first
  change (`syncResetUndo` from `persistPanel`) or after 20 s. The Undo
  writes the boxes through `rewriteBox`, restores the rest (the English
  included where the prompt is its source), runs the sync functions again
  and says "Reset undone: the panel is as it was." in the same place for
  6 s (`RESET_UNDONE_MS`): an 'info' note with no button and
  `pointer-events: none` over the text it put back, which typing takes.
  Nothing either note does moves Generate. Resume is never restored, and
  nothing of it survives a reload. Before it fills, Reuse keeps
  `panelState()` (what `persistPanel` stores), the raw steps/CFG/mu/true-CFG
  values and `owned`. `fillPanel` writes a record's Mu only where
  `muSupported`, as it writes True CFG only where that is a knob; the note
  for another checkpoint's record names "steps and CFG" alone where Mu is
  greyed, and `collect()` sends no mu from a greyed box, nor `gridPayload` a
  `mu_axis` from a greyed axis, as with the LoRA weight. Its note in the
  panel's `#error` opens "Settings from
  #N loaded" (" - your prompt was replaced" where it was) with an Undo right
  after that sentence, which restores them; the rest follows: the checkpoint
  mismatch, references on a checkpoint that takes none, an adapter with
  `/api/loras`' reason. The Undo is offered only while the panel holds what
  the Reuse left (`reuseNote.after`; `syncReuseUndo` from `persistPanel` and
  `resetPanel`): the first change removes the button and keeps the text, and
  a reference dropped from what it left (deleted, or "unknown image") is no
  change. A Reset takes the words too, while they are still what the line
  says (`clearReuseNote`, `reuseSaid`): "Settings from #N loaded - your
  prompt was replaced" had stood over the emptied panel. It drops a
  restored reference only on the server's "unknown
  image". Where the page goes is `revealReused`'s: the panel is placed and
  moved (`placePanel`, `revealInPanel`), and the page scrolls only where the
  panel cannot move (one column, the page's end), and only as far as the
  panel's resting place. A stuck panel is then lifted to bring the Undo into
  the window only as far as the prompt's field stays clear of the topbar
  (`showUndo`); at 1400x900 the prompt wins. The lightbox's Reuse closes the
  lightbox first. `syncReuseButton(button, rec, compact)` takes the record
  and keeps its checkpoint, refine source, references and adapter in data
  attributes for `syncReuseButtons`. A refined record's button reads "Reuse
  pass…" (the card keeps ↻/⚠), named "Reuse pass in the Refine dialog" (the
  visible words first, for voice control; a warning suffix after) and titled
  "Open the Refine dialog on #N with this pass's size, strength and seed"; a
  pass whose source is gone
  (`refine_missing`, kept in `data-reuse-pass-gone`) reads "Reuse settings",
  titled why the pass cannot be made again, since `reusePass` fills the panel
  naming the source "#N". A ⚠ on another checkpoint's record adds which
  references or adapter will not carry over (`notCarried`, `fillPanel`'s own
  test). That verdict reads the status, so status.js runs
  `syncReuseButtons` where the checkpoint changes, where
  `reference_supported` or `lora_supported` flips, and at the end of
  `refreshLoras`.
- Reuse refuses every nonempty recorded execution recipe before writing
  the composer or looking up pass sources. Its button is `aria-disabled`
  with the reason as title; a press says it in the gallery error slot and
  mirrors it into an open lightbox without closing it. The composer's prior
  Reuse note and Undo stay intact. `syncReuseButtons` preserves the recipe
  in its dataset so a capability refresh cannot clear this hold.
  An instruction-edit record never populates ordinary Generate:
  `reuseIntoPanel` refuses every edit. A resolved edit's Reuse
  ("Reuse instruction edit…", no checkpoint ⚠) opens the Edit by instruction
  dialog on its source instead (`reuseEdit`, which looks the source up with
  `?binned=true`). It writes the recorded instruction with `replaceBoxText`
  and the recorded seed, steps and guidance, and the composer's draft, Undo
  and references stay untouched. An unresolved (`source: null`), deleted (`edit_missing`) or
  binned (`edit_binned`) source holds the button with the reason
  (`editReuseReason`, through `executionReuseReason`). Edit presence is kept in the dataset across
  capability refreshes, and binning, deleting or restoring the source updates
  the edits' records, titles and buttons.
- Edit by instruction… (`instruction-edit.js`, `styles/instruction-edit.css`;
  `smoke-instruction-edit.json`, `tests/test_instruction_edit_ui.py`):
  - **Entry.** It is a right-click menu item after Outpaint… on a card, a
    result tile and the lightbox (`syncedItem` over `syncEditTool`, running
    `editFromCard`). It is live wherever its image is live, the real server
    included, and is held only for an image in the recycle bin ("restore it
    first"). There is no card glyph, lightbox button or wide viewer tool.
  - **Opening.** It opens `#edit-dialog`, the Inpaint dialog's box with a
    read-only picture. Opening from the lightbox closes the lightbox, and
    Escape gives it back where it was. Focus goes to `#edit-instruction`
    with `preventScroll`.
  - **Contents.** The source line is the server's "#N · W×H → w×h". The held
    reason `#edit-unavailable` (`defaults.instruction_edit_reason`) comes with
    **Open Models…**, which opens the picker and never switches or queues.
    Then the instruction, the checkpoint line, seed with its random button,
    steps and guidance from the preview, and one fixed-height line shared by
    the preview and the error. There is no Strength control and no canvas.
  - **Run.** Edit (`#edit-run`) is `aria-disabled` with its reason as the
    title. The reason is, in order (`editHold`): `submitHold()`, a submit
    in flight (`EDIT_IN_FLIGHT`), the capability reason, the preview's
    refusal (shown in place), "Wait for the preview of the source", a bad
    seed, steps or guidance (`editNumberProblem`), or "Write an instruction
    first." (`EDIT_NO_INSTRUCTION`). A held press writes the reason in place, with
    no scroll and no job. Run asks the preview again when it is stale, then
    queues through `startJob('/api/instruction-edit', …)`. Ctrl+Enter, and
    Enter in a number box, run it.
  - **Drafts.** Close keeps an unsent draft, which comes back only for the
    same source and checkpoint. A checkpoint change clears the drafts and
    asks the preview again.
  - **Never written:** `els.prompt`, `els.negative` and the references.
  - **On Kontext.** Generate and Run grid are held with
    `defaults.text_to_image_reason`, and pressing Generate says it and sends
    nothing. The reason comes from the gate (`text_to_image_refusal`): it
    points at Edit by instruction… only where an edit can queue, and
    otherwise says the edit gate's own reason.
  - **Lineage.** Jobs read "instruction edit of #N" in the queue, the
    progress line and Cancel's title. The settings line says "edit of #N".
    The facts have "Edited from #N" (marked deleted or in the recycle bin,
    with Restore for the bin), steps and guidance as the sampling, and the
    recipe. Cards get the `edited` class and badge, and the viewer labels the
    prompt "Instruction". Models adds an "Edit" note (`model_info`'s
    `instruction_edit_note`) and no fact cell.
- Load settings from PNG (`png-settings.js`, `api_png.py`,
  `png_settings.py`; `smoke-png-settings.json`, `tests/test_api_png.py`).
  A file dragged over the page outlines the window (`body.png-drop::after`,
  "Drop a PNG to load its settings", `pointer-events: none`, dropEffect
  `copy`) and a drop anywhere reads it; over a modal dialog
  (`dialog[open]:modal`) the dropEffect is `none` and nothing is read, as
  every file drop was before. A drag that starts on the page - a gallery
  card's image, whose types Chromium lists with 'Files' - is none of this
  (`pageDrag`, set at `dragstart`, cleared at `dragend`, the drop or the
  next mouse move): no outline, nothing read. "Load settings from PNG…"
  (`#btn-png`, `aria-describedby` its hint "or drop a PNG on the page")
  sits under Generate directly over `#error`, where its note is written,
  so a load grows the panel under the button and never moves it there -
  not at the panel's end, where the note above pushed it out of a
  2560x1305 window, nor in the prompt's caption, which has no room beside
  Tags or Expand at 330 px - and clicks the hidden
  `#png-file`, emptied once read so the same file reads again; its
  `change`/`input` never reach `persistPanel`. Two files are refused
  ("drop one PNG at a time"), a file without the PNG signature in words
  ("x.jpg is not a PNG: …") and nothing is sent. `pngTextChunks` walks the
  chunk headers to IEND and keeps tEXt (Latin-1), zTXt (zlib through
  `DecompressionStream('deflate')`) and iTXt (UTF-8, compressed or not),
  the last of a repeated keyword as Pillow's `.info` does; a chunk that
  does not parse is skipped. Only `{name, chunks}` is posted - never the
  image, which is never stored or made a reference (U61 B stays
  deferred); a keyword the server does not read (anything but
  `auradream`, `local-diffusion`, `krea`, `parameters`, `workflow`,
  `prompt`) goes by
  its name with empty text, so an XMP packet beside a settings block does
  not take the body past the cap. The page sends `panel`, `collect()`'s
  steps, CFG, true CFG and Mu, which an A1111 block's request takes
  (`_panel_sampling`; a value Generate would refuse is the profile's), so
  `applied()` judges the negative prompt at the CFG the panel holds. The
  route is model-dependent (`hold_model_for_request`,
  advisory in `api()`), caps the body at 1 MiB (413, before or while it
  arrives) and 64 keywords of 1-79 characters, and answers `PngSettings`:
  `settings` after `backend.applied()`, `size` (the PNG's where a direct
  render takes it - else the reason as a note and the panel's own),
  `sampling`, `model_id`, the PNG's own `negative_prompt` (the box keeps
  one `applied()` leaves out, greyed, with a note) and `notes`. This
  app's record (`auradream`, or the legacy `local-diffusion` or `krea`)
  wins over its
  own 'parameters' line and carries what Reuse does; its reference ids
  carry only where this gallery's live row under the id has the
  record's `reference_sources` hash and size, else a note; a refine pass
  fills the panel as a direct render at the panel's size, with a note;
  an adapter `applied()` drops is named with the installation's reason.
  An A1111/Forge/Civitai 'parameters' block is read as A1111 reads it
  (`parse_parameters`: the prompt to "Negative prompt:", the last line
  settings when it holds three `Key: value` pairs, JSON-quoted values)
  and gives the prompt, the negative and Size only (`sampling` false, the
  seed None): its Steps, Sampler, Schedule type, CFG scale, Seed and
  Model are named in a note as not carrying over - another sampler and
  checkpoint do not make the same picture from them - and never mapped,
  and its other keys are listed "Not read". A ComfyUI workflow and a PNG
  with neither block are refusals in words (422), naming the file. The
  page builds the record Reuse takes (the panel's seed where none
  carries) and calls `reuseIntoPanel(rec, size, notes, {name, sampling})`,
  so the note reads "Settings from <file> loaded", its Undo "Reuse of
  <file> undone", Ctrl+Z walks back the boxes, and the other-checkpoint
  sentence is Reuse's own - only where the record names a checkpoint.
- The results viewer (panel.js). `clearResults(job)` runs at a job's first
  frame or image: it moves what the strip held into `sessionRecords` (newest
  first, not the same job's) and sets the tier from the job's
  `planned_images` (`viewerJob`), never the running count; focus inside the
  strip it empties goes to Generate. One image is shown
  at 68vh; two to four are best-fit (`fitViewer`: `--cols`, `--rows`,
  `--tile-h`, `--col-w`, `.fitted`, the grid's rows and columns reserving
  every row and column the job will fill from its first frame, so a lone
  preview or stand-in holds the first image's slot and nothing moves
  sideways as the images land; sized as if the page's scrollbar were there,
  `pageScrollbarWidth()` off the width while the page has none, since a
  gallery too short to scroll takes one with a job's progress block and
  gives it back at its end - never a stable gutter on the root, which in
  Chrome shrinks every vw size on every page), refit on the next frame from a
  ResizeObserver and on the window's resize; more than four are small tiles,
  whose tools are out of the tab order, and from 2000 px those are
  `auto-fill minmax(--strip-min, 1fr)` under `min(560px, 45vh)` with their
  own scroll and `scrollbar-gutter: stable both-edges` - never
  `overscroll-behavior: contain`, so the wheel over a strip at its end, or
  one with nothing to scroll, scrolls the page - no 380 px floor, and
  every row the job will fill reserved from its first tile (`reserveStrip`:
  an inline min-height from that tile and the strip's column count, up to
  the cap, never under the stylesheet's own minimum, recomputed when the
  strip's width moves, let go by `clearResults`). Every tile appended
  goes through `stripAppend`, the preview and the stand-in included, so a
  strip at its end stays there, and `settleStrip` lets the strip grow by an
  overflow no bigger than its bottom padding. Every tile is a `viewerTile`
  with `--w`/`--h`, `--tile-h` and a foot of `--h-tool` (`fitViewer` reads
  the same token): a result, the live
  preview `figure.live`, and a pass's stand-in `figure.pending`. A result is
  `figure.result[data-id]`, never `.card`, a group named "#N" as a card is
  (`resultFigure`), holding `thumb()`'s image and the
  card's tools: `reuseButton`, ⤢ (`refineResult`), ❐ (`toggleReference`,
  hidden where the checkpoint takes no references), a `.star` via
  `setFavorite`, and ⌦ via `deleteImage` (to the recycle bin); no ⊞. Alone in the strip, a
  result carries `.result-side` from 1800 px: the lineage chip
  (`passLineage`), Star / Refine… / Reuse / ❐ Reference / on a pass Compare
  with source (`compareButton`: the lightbox through `openViewerImage`, then
  `toggleCompare`; aria-disabled with the reason once the source has gone,
  kept by `syncCompareResult` from `viewerSourcesGone`, a press refused
  through `refuseInCard`) / Use seed / More seeds… (`syncMoreSeedsFor`) and
  Delete (`.ghost.danger`, at the row's end) in words, the prompt (its line breaks kept, `white-space:
  pre-line`, at most `--prompt-cap` - twelve lines - scrolling inside
  itself, as the lightbox's `#lb-prompt` does) and `factsList(rec)`
  (without the lightbox's Use seed: the row has one); its tile's tools row
  stays, empty, at its height. With
  the check docked the side is a row under the picture in the tools row's
  place, at `--h-tool`: its actions and `.result-brief` (seed, size, steps,
  CFG; `settingsLine` in its title) - one set of words for the image just
  made either way. Compare with source is left out of the row: in a row the
  picture's width it cut a portrait pass's every word to an ellipsis, and
  in a row its buttons' width it would take another 150 px from a docked
  `#recent`; the picture's lightbox and S compare. The row is the
  picture's width, or its buttons' `max-content` where that is more,
  centred under the picture and never wider than the strip, so neither the
  picture nor the gallery moves as the check opens and a narrow portrait's
  words stay whole. The brief is left out of that measure (`contain:
  inline-size`) and takes what is left: under a 5:12 picture that is
  nothing, and the row is its actions alone (the seed is in Use seed's
  title and the image's). With Gallery beside the check takes the gallery's
  column instead, and the side stays a column.
  `#viewer-note` names a pass's source ("Refined from #N at s · W×H",
  `sourceLink` → `openRefineSource`). It names a single direct result as
  the card's caption does, "#N · W×H · seed N · Use seed" (`.seed-line`,
  hidden from 1800 px, where the column or the docked row says both). It
  keeps one line's height while empty, and a delete from the strip writes
  it again (`forgetImages` → `noteViewer`). `#recent` (from 2000 px, single
  tier, in the viewer's row only - with Gallery beside, a row of its own
  under the viewer's note, check open or not, 88 px thumbnails beside its
  heading) shows up to six `sessionRecords` - its own scroll never
  `overscroll-behavior: contain`, as the strip's - and opens the
  lightbox on [current, ...sessionRecords]; with the check docked, `fitRecent`
  keeps it between the image and the check in the width the image - or the
  row under it, where that is wider - leaves at its own size
  (`.recent-docked`, `--recent-room`), hides it under `--recent-min`; a tag
  checkpoint's chips span all three columns above. The docked test is
  written twice, and a change to the dock's selector changes both, and the
  translation's rows too (`#prompt-translation-field` beside the no-chips
  dock), which repeat its selector: the CSS `:has()`, on
  `.stage:not(.beside)` (or its negation `:is(.beside, :not(:has(...)))`)
  wherever a rule lays the stage out around the check, and `fitRecent`'s
  `matchMedia('(min-width: 2000px)')` with the check open in
  `#tag-validation-wide` and `!besideOn`.
  A single viewer image opens the lightbox on `galleryRecords` from its copy
  there, else on [rec, ...sessionRecords]; `openLightbox(rec, list, origin)`'s
  `origin` of "viewer" or "recent" is where `dismissLightbox` returns focus.
  Idle, the viewer is one line, keyed on `:has(> .placeholder:not(.held))`,
  never on `viewerRecords`; `.empty-install` (`syncEmptyInstall`, from an
  unfiltered gallery page with total 0 and no `other_models`) keeps the tall
  box with an Examples button. Idle with the check docked, the viewer stays
  that line, the gallery comes up in the first column beside the check, and
  the check is as tall as what it holds up to the window under the topbar
  and the chips (`contain: none`, spanning the rows down to the gallery's
  1fr); the docked 36rem minimum is
  only for a viewer holding something. A gallery with nothing to show says
  so under its head (`syncGalleryEmpty`, `#gallery-empty`: what fills it and
  the authors' Examples, or that nothing matches the filter, and what All
  models would add), never over tombstones or in sheet mode. A finished
  job's images the gallery's filter leaves out - a sheet's `job:` scope, a
  search, Starred - are counted in the gallery's line with Show all
  (`noteOutsideView`, per filter; `showAllImages` empties the search,
  unticks Starred and closes the sheet). The strip's run is `.card.this-run` in the
  gallery. When ⌦ deletes the strip's last image, focus moves to the held
  placeholder, never `<body>`. `forgetImages` splices `sessionRecords` and
  re-renders `#recent`, retitles every `img[data-id]` of a pass whose source
  it deleted (`cardTitle`) and calls `viewerSourcesGone` (the note, the
  lineage and the facts read "#N (deleted)"); `recordsWithId` includes
  `sessionRecords`, and `syncCardTools` also syncs the viewer's
  `.refine-tool`s and `.ref-tool`s.
- Gallery beside (`syncBeside`, gallery.js; style.css's "gallery beside"
  section; `tools/smoke-stage.json`). The tick: from 2000 px, "Gallery
  beside" in the gallery head (`#gallery-beside` in
  `label.check#beside-label`, after Sheet and before Delete unstarred,
  hidden under 2000 px, no ? badge), off by default and remembered as
  `auradream.gallery-beside` through `remember`/`forget`; not the
  panel's, so Reset leaves it. While a sheet is open it stays live, and its
  title says it takes effect when the sheet closes (`syncBesideTitle`). The
  gallery's blocks sit in `#library` (the head, then `#library-scroll`,
  which holds the empty line, the grid, the sheet and Load more); unticked,
  both are `display: contents` and pass their `grid-column` on
  (`grid-column: inherit`). It applies (`besideOn`, `.stage.beside`) only
  while the tick is on, the window is 2000 px or wider (`BESIDE_MEDIA`, with
  a `change` listener) and no sheet is open. `#library` is then
  `position: fixed` under the topbar at the window's right, `--library-w`
  wide, and the stage's `margin-right` is that width plus a gutter;
  `#library-scroll` scrolls by itself (`overscroll-behavior: contain`,
  `overflow-anchor: none`, scrollbar room kept), and the page keeps its
  scrollbar's room (`html:has(.stage.beside)`). Never sticky: a sticky
  column is held only within the stage's height. In the column cards are at
  least `--thumb-min-beside` wide (200 px, the five `--h-tool` tools' room,
  set as `--thumb-min` on `.stage.beside .library`), three across at QHD and
  two at 125% (M); at the head's S, 150 px
  (`.stage.beside #gallery[data-density="s"]`), four across at QHD and
  three at 125% - the S rule's own 166 px fitted three there, and S showed
  pressed with nothing changed. The search has its own row, its tip
  hanging left; a card
  brought into view there has no topbar margin. The stage keeps its own
  layouts in the width left: the result column and the strip, and
  `#recent` as a row under the note. The prompt check, open, takes the
  column's box whatever the viewer holds, and scrolls inside it; the column
  is `visibility: hidden` under it (scroll kept; no focus or pointer reaches
  it) until Close. `setSheet` calls `syncBeside` first, so a sheet always
  opens with the page in one column, across the stage's width, and is
  landed on; Sheet off puts the column back. `armMore` re-roots the paging
  trigger on every change. A job landing: only the cards wait (the
  finish-line bullet). The gallery's keyboard rules are unchanged:
  `moveCard`'s `scrollIntoView` and a focused card scroll the column, not
  the window, and `isShown` (state.js: `checkVisibility` with
  `visibilityProperty`) keeps focus out of the hidden column.
- The lightbox (`web/lightbox.js`). `lbView` is a scale relative to fit,
  never below `min(1, oneToOneScale)` nor above 8x device pixels
  (`clampScale`), carried across records whose aspect agrees within 1%
  (`lbView.aspect`, a tolerance, never a rounded key) and clamped to the new
  record's limits. The zoom line reads "1:1", "fit · N%" or "N%", tested in
  that order. It is written only when its words change (`writeZoom`) and is
  aria-hidden; what is said is its hidden copy `#lb-zoom-said` (polite),
  written 400 ms after the last change - a wheel's burst, a resize, a step -
  or at once for a zoom a key, a button or a double-click asked for
  (`sayZoom`). `#lb-count` is likewise written only when its words change,
  and `closeLightbox` empties all three. A double-click returns to fit from any other scale, and a
  backdrop click below fit waits out a double-click (`lbCloseTimer`). F, S,
  Delete and the zoom keys skip a key with Ctrl, Meta or Alt held (the
  browser's find and zoom), and a key a lightbox control already handled
  (`defaultPrevented`: the swipe divider's keys) is not walked. On a record
  with `refine`, S or Compare with source shows the source under the pass
  left of a divider (swipe), or side by side with each half fitted inside
  the arrows' circles (`lbPorts`, `lbNavRoom`, measured off the circle as
  drawn - read once per `layoutLightbox`, before its writes), on one
  `lbView`; switching keeps the magnification. The halves go left and
  right, or the source above the pass: stacked only where the window is
  under 901 px or the stage is taller than wide, and only where stacked
  halves fit the picture at least an eighth larger (`lbStacked(rec)`,
  `LB_STACK_GAIN`), because stacking takes the letterbox the facts use; a
  landscape stage from 901 px is always left and right. The split is
  decided once per layout, on the stage with the facts at six lines
  (`lbBaseStage`), before `lbFactsRoom`, and both use that decision; every
  reader of the ports takes a port's y as well as its x. Side by side keeps
  its name, and its title says which way a press puts the halves, before
  the press as well as after it (`lbStacked`). The layout also runs again
  on the stage's own resize (a ResizeObserver in app.js), not only the
  window's: under 901 px a line in `#lb-error` (Use seed's, a delete's) is
  a row of its own that takes height from the stage, and the picture laid
  out before had kept its height, its foot cut off under a readout saying
  "fit". The Refine dialog's
  passes strip and a pass's Compare with source in the result column open
  the lightbox with it on (`toggleCompare` after the open). The
  source is fetched by id once a session (`lbSources`) and drawn with
  `refineKept` and a clip-path. A step to another record ends the
  comparison; a re-render of the same record does not - except ← → while
  comparing, where the strip holds another pass of the same source (the
  Refine dialog's passes strip, a Compare 3 run's viewer, a `refine:`
  search): they swap to the next such pass that way, wrapping round and
  skipping the source and anything else, the comparison carried with its
  mode, divider and focus (`stepPass`, `lbSiblingPass`; the strip as
  loaded, no page pulled for it). Then the divider moves by Shift+← →
  wherever focus is (`nudgeSplit`, 2%), and on the focused divider by
  Shift+← →, PageUp/PageDown (10%), Home and End, its `aria-keyshortcuts`
  and title saying so; the keys line reads "← → passes · Shift+← → divider".
  With no other pass in the strip the arrows on the divider move it (2%,
  Shift+← → the same) and elsewhere walk, ending the comparison, as before;
  Shift+← → off the divider still move it while a swipe is up. Shift+← → is
  2% everywhere, on the divider or off it, passes or not; PageUp/PageDown is
  the 10% step. The nav
  circles always walk. A deleted source
  leaves the controls aria-disabled with the reason ("source #N deleted"). S
  pressed on the focused divider hands the focus to the toggle of the kind
  that was on.
- The comparison across CFG (`web/lightbox-cfg.js`; across steps, the same
  machinery, after this item). `lbCompare.kind` is `'source'` (a pass and
  its source, above), `'cfg'` or `'steps'`; `lbCompare.source` is the
  partner either way. An image's CFG ladder (`ladderOf(LADDER_AXES.cfg, …)`,
  `cfgLadder`) is the images
  that agree with it on every sheet axis but CFG and True CFG and on size,
  checkpoint, quantization and refine source (`sameExcept(a, b, ['cfg',
  'tcfg'])`, beside `sameSettings` in sheet.js, which is
  `sameExcept(a, b)`), filtered to the image's True CFG where True CFG is
  not an echo of CFG across the set (`trueCfgEchoesCfg`), one image a CFG
  value (the image itself, then the partner, then one in `lbList`, then the
  newest), sorted ascending. It is worked out in the page from records the
  server made - no route, no schema field - from `lbList`,
  `galleryRecords` (the sheet's records are the same array, never added
  twice), `viewerRecords`, `sessionRecords`, the gallery's answer and the
  comparison's own ladder, deleted ids left out; recomputed on every sync
  while off, built once when a comparison is turned on (`compareLadder`) and
  frozen while it steps, taking in the gallery's answer for the image in
  view when it comes. The answer: `GET /api/gallery?model=<id>&limit=200`
  200 ms after the image lands (`askLadder`, as `passesLine`), read on by
  the id cursor (`before=<last id>`, `fetchLadder`) while `has_more`, up to
  `LADDER_PAGES` (5) pages - one page let a seed kept across a few hundred
  prompts fill it with the others and hide an older neighbour - searching
  `seed:` (or `run:` where the seed is not a safe JavaScript integer, and
  nothing asked with neither), `size:`, `steps:`, `refine:` (`none`),
  `mu:` (`none`; left out for an exponent) and `lora:=` (where there is an
  adapter whose name has no space) or `lora:none` (where there is none),
  cached by query in `lbLadders` until `closeLightbox`, pruned by
  `forgetImages` (`forgetLadderImages`) and dropped for every query of a
  landed image's checkpoint, seed and size (`noteLadderImages`, from the
  stream and from `joinFinished`; every answer already on its way for one,
  whichever image asked it, is marked stale in `lbLadderFlights` and taken
  out, so it is not kept and the image in view asks again; one still
  waiting out its 200 ms reads after the save and stands). `has_more` past
  the last page, or a failed request, makes the note add "may be
  incomplete: more may be in the gallery", and the held reason says how far
  it looked (below). The box
  (`#lb-compare`) is on every image from 901 px: its note first (the column
  is pinned to the bottom, so a line written there grows the box upward and
  moves no button), Compare with source on a pass only, Compare CFG and
  Compare steps on one row (`.lb-kinds`, gone where both are hidden), Swipe
  and Side by side, ← Lower CFG / Higher CFG →, ← Fewer steps / More steps
  → and Against. Each CFG
  control is `aria-disabled` with the reason as its title - "Looking for
  other CFG values of #N…" while the gallery is asked and the page holds no
  other value, "No other CFG value of #N at this seed, size and settings"
  once it has answered, "No other CFG value of #N in the newest K at this
  seed and size: more may be in the gallery" where the answer was cut
  short, and "… in what the page holds: the gallery could not be asked"
  where it failed (`ladderHeldReason`). Swipe and Side by side, held with
  them where a press would turn this comparison on (or on a pass whose
  source is gone), take the same reason as their title, and their own
  title back when they can be pressed (Side by side's from
  `layoutLightbox`, `lbSidesTitle`, which way the halves go) - shown on
  every image now, a plain Generate's held mode buttons were untitled. A
  press, S or a mode press says it in
  `#lb-compare-note` (`lbCompareSaid`, kept for that image until another
  is drawn or a comparison is turned on or off). Under 901 px the box is a
  row under the stage that the picture and the facts' letterbox pay for: it
  is drawn on a pass, or where the page's own records give the image a CFG
  neighbour when the image is drawn (`ladderShown`, `lbCfgShown`), kept until
  another image is, and from a comparison turned on - never on the
  gallery's later answer; S works without it. S with no kind
  (`toggleCompare()`) turns a comparison on as the image's kind (source on
  a pass, else CFG) and off whichever kind is on; each kind's button turns
  its own on and off, or switches the one on to it; a mode turns on the
  default kind, or switches mode keeping the kind. The partner starts as
  the CFG value directly below, or above for the lowest; Against picks
  another (the comparison on, or turned on). Against is drawn as the small
  buttons above it (lightbox.css), and ← → on it are the select's own:
  app.js's keydown returns before walking when the target is a SELECT.
  The lower CFG is always on the left (or the upper half): where the image
  in view has the lower CFG (`lbSwapped`) `layoutLightbox` puts it in the
  first port and
  `layoutCompare` cuts the partner's layer to the right of the divider and
  places it in the second port, the tags following; the divider's
  `aria-label` is "Divider: CFG a left of it, CFG b right" and its
  `aria-valuetext` "CFG a (#i) on the left N%, CFG b (#j) on the rest". The
  tags read "CFG 2 · #41" (with "· True CFG x" where it is not an echo),
  the partner's alt "#41, CFG 2", the note "CFG 2, #41 · 2 of 4 CFG values"
  with "· identical pixels: CFG changed nothing here" on an equal
  `content_hash`. ← → while a CFG comparison is on step to the next value
  that way that `lbList` holds (`stepLadder`, `ladderReach`), against the value
  below it, mode, divider, zoom and focus kept; values only the gallery's
  answer brought are partners, never steps, and `lbList` is never replaced
  or added to. At an end the press stays and says "CFG 7 is the highest of
  these" (the ladder is ordered; it does not wrap as passes do), and where
  `lbList` holds no other value either way the arrows walk, ending it.
  Lower/Higher CFG take the same step, turning the comparison on where it
  was off, and are held with the reason at an end or where the next value
  is not in the strip. `lbSiblingPass` answers only for kind `'source'`,
  `stepLadder` only for `'cfg'` and `'steps'`. The keys line names S on every image and
  reads "← → CFG" while the arrows step along CFG; the divider's
  `aria-keyshortcuts` and title say so. A partner deleted by this page ends
  the comparison with "#N is no longer in the gallery"; a partner whose
  picture will not load ends it with "#N could not be loaded", asked for by
  id and then said as deleted (and dropped from every ladder) where the
  server no longer has it; `returnToLightbox` puts back a CFG comparison's
  frozen ladder less deleted ids, and not one whose partner went.
- The comparison across steps is the same code along another axis
  (`LADDER_AXES.steps`; `ladderAxis(kind)` answers the axis of `'cfg'` or
  `'steps'`, null for a pass or a picked pair): the ladder skips Steps
  alone (`sameExcept(a, b, ['steps'])`, True CFG compared as any axis, no
  echo filter), one image a step count, sorted ascending, from the same
  pool. Its gallery query is CFG's with `cfg:<the image's CFG>` (left out
  for an exponent) in place of `steps:` (`ladderQuery(rec, axis)`): both
  are asked together after the same 200 ms (`askLadder` over
  `ladderQueries`), each cached, in flight and dropped as above, and each
  axis's held reason and "may be incomplete" read its own answer
  (`ladderState(rec, axis)`). One query without either term served both
  for a while, and a steps sweep at one seed crowded an older CFG
  neighbour out of its five pages. Compare steps never holds S (S stays
  the pass's or CFG's) and is held with "Looking for other step counts of
  #N…", "No other step count of #N at this seed, size and settings" and
  the cut-short and failed forms; its title is "Compare 28 steps with 20
  steps, #41, at the same crop, the fewer steps on the left". ← Fewer steps
  / More steps → (`pressLadderStep('steps', ±1)`) step as Lower/Higher CFG
  do, held with "20 steps is the fewest of these" / "28 steps is the most
  of these". A step button of the other axis switches the comparison to
  its axis (`setCompare(null)`, then the step, then `compareLadder`), and
  Compare steps or Compare CFG switches the kind, keeping the mode. Against
  lists the ladder of the comparison on - "20 steps · #41" - and CFG's
  while neither is; so a step held because the next value is not in the
  strip says "… pick it under Against" for the axis Against lists and "…
  turn on Compare steps (or Compare CFG) to pick it under Against" for the
  other (`ladderEndReason`'s `against`). The tags read "20 steps · #41" ("1 step" for one), the
  divider "Divider: 20 steps left of it, 28 steps right", the note "20
  steps, #41 · 2 of 2 step counts" (with "· identical pixels: the step
  count changed nothing here"), Swipe's title "Fewer steps and more steps
  over each other, …", Side by side's "Fewer steps left of more steps, …",
  the keys line "← → steps" and the divider's title "… ← → step along
  steps". The fewer steps are on the left (`lbSwapped` by the axis's
  value). S with the divider focused hands focus to Compare steps. Under
  901 px each axis's part (its Compare button and its step row) is drawn
  where the page's own records give the image a neighbour along it, or
  once a comparison along it is on (`ladderShown` answers per axis), and
  Against with the part of the axis it lists - never for CFG's "none"
  beside steps' row alone.
  `factsList(rec, { useSeed })` (`dl.facts`, facts.js) is the one facts renderer, the
  viewer's column's too; the lightbox's adds Use seed to its Seed row
  (`useSeed`, said in `#lb-error` alone, never the panel's `#error`, and
  taken back once another image is drawn - `lbSeedNote`), the column's does
  not. The lightbox's (`search: true`) ends each row the search grammar can
  say (Model, Size, Sampling, LoRA without whitespace, Refined from, Seed,
  Created as `date:` the local day, Run; the Created row's peak,
  `peakFact`, "N.NN GiB peak" in GiB of the allocated bytes, "of its batch"
  when `batch > 1`, the reserved figure and the allocator caveat in its
  title, nothing at all without one) with a muted ⌕ (`.lb-search`,
  `aria-label` "Search the gallery for …", the terms in its title), and
  `promptSearchLine` puts "⌕ Search for this prompt" over the facts: the
  prompt's words, less any with no letter or digit or that would read as a
  filter. A press goes through `showGallerySearch` (the passes link's too):
  the lightbox closed, `mergeSearch` into the box - a filter replaces the
  box's terms with its key, a word already there (any case) is not added
  again, the rest is kept in order; the prompt's line goes through
  `mergePromptSearch` instead, the box's filters kept and its free text
  replaced by the prompt's words, so a second prompt is a change of prompt
  as a second seed is - the gallery asked again only where the
  box changed, the head brought into view and the box focused. Only keys
  `search.FILTERS` has; `SEARCH_KEYS` is held equal to it by
  `tests/test_storage.py`. A pass whose source has gone reads "Refined from #N
  (deleted) @ s". Its References row numbers the references as the panel's
  chips do. Use these references asks for every id again as Reuse does,
  leaves out and names the ones the gallery no longer has, and is
  aria-disabled with the reason where the checkpoint takes none
  (`REFERENCES_NONE`) or every one has gone; that state is re-synced on every
  render (`syncUseReferences`). "Refined from #N" opens [source, pass]; Run
  does what the card's ⊞ does. `settingsLine` stays the dialog's hidden
  description (`#lb-desc`) and the delete confirm's line, every image named
  `#N` ("refine of #N (deleted)", "ref #a+#b"). The counter reads
  "i/total · #id" with the gallery's server total. The star follows the
  toggle rule, and `setFavorite` re-renders it (`renderLightbox`) rather than
  writing it. The user's closes (Escape, ✕, the backdrop, a stage click) go
  through `dismissLightbox`, which focuses the image the walk ended on or,
  where no strip shows it (another checkpoint's scope, a delete elsewhere,
  the gallery's column hidden under the prompt check with Gallery beside),
  the grid's current card, else its first, else the search box - each only
  where shown (`isShown`) - else the viewer's first image, else Generate;
  its Escape is taken in app.js's keydown before the Examples and Tags
  menus' branches - a menu left open under it (the Tags helper, after focus
  left it by keyboard) had taken it, and the dialog's own cancel closed the
  lightbox without `dismissLightbox` - and the menu stays open under the
  page; every programmatic close is `closeLightbox` alone. The arrows are
  strips only at fit, with backdrop beside the image (`--lb-gutter`, from
  the circle's reach less 2 px - 48, 60 at the wide tier - to 160 px,
  20%-80% of the height); zoomed in, below fit, disabled or with no backdrop they are
  their circles. A strip is a hit area only, never painted: no hover tint,
  no ring round it - hover lights the circle and `:focus-visible` rings the
  circle (`.lb-nav span`), not the button. The stage has no `title` (its
  tooltip covered the picture); its mouse help is `#lb-zoom`'s title. The
  lightbox's facts (`megapixels: true`) follow Size with its megapixels to
  one decimal, "1536×1024 (1.6 MP)"; the viewer's column keeps the bare
  size. The wheel listener is `.lb-main`'s, so a strip zooms too. A
  stage press within 500 ms of an arrow press (`lbAfterArrow`) neither closes
  nor zooms: it is the rest of a double-click on the arrow. The arrows'
  clicks are never filtered by click count. The keys line (`#lb-keys`,
  `lbKeysLine`) sits outside `#lb-meta`, names S on every image, binds each
  key to what it does (no-break spaces, a word joiner in +/−) so it breaks
  only after a ·, and is hidden under 901 px. The no-break spaces and the
  word joiner are written as ` ` and `⁠` escapes, never the
  characters themselves: a literal one reads as a plain space, and an
  editor turning it back into one splits "+/" from "−" again. There `#lb-meta` is six lines
  plus `--lb-room`, the letterbox the fitted picture (or stacked halves)
  leaves the stage with the facts at six lines; `lbFactsRoom` sets it,
  after the split is decided and first in `layoutLightbox`, and writes it
  only when it changes; its prompt is four lines there (`#lb-prompt` keeps
  its line breaks, `pre-line`, at most `--prompt-cap` elsewhere).
  lightbox.js declares only; its
  handlers (`lbImageFailed`, `lbImageLoaded`, `lbWheel`,
  `lbArrowDown`, `lbPointerDown/Move/Up/Cancel`, `lbDoubleClick`) and the
  dialog's `close` → `closeLightbox`, only while the
  dialog is still closed (the event is queued; a close followed at once by
  an open must not shut the new lightbox), are wired in app.js; so are
  lightbox-compare.js's (`lbSourceFailed`, `lbSplitKey`, `nudgeSplit`,
  `stepPass`, `toggleCompare`) and lightbox-cfg.js's (`pressLadderStep`, `pickAgainst`, Compare CFG's
  `toggleCompare(null, 'cfg')` and Compare steps' `toggleCompare(null,
  'steps')`), which declares only too.
- The gallery's cards. Each card is `role=group` named "#N" (`card()`; a
  viewer `.result` too, `resultFigure`), so its star and tools are heard as
  that image's; `tombstone` drops both attributes. Never a list or grid
  role: the day lines and tombstones are the cards' siblings. The tools are
  ↻ ⤢ ❐ ⊞ ⌦. Each button's textContent is
  its glyph (the smokes find them by it), and its short name is in
  `data-tip`, which a `::after` shows on hover and keyboard focus at the tool
  row's left edge; Delete's sits at the row's right edge, over Delete. Above
  the row, on hover and focus-within, the row's `::before` draws "#N · W×H ·
  seed N" from its `data-n`/`data-size`/`data-seed` (never `data-id`, which
  `forgetImages` queries), with empty alt text because the image's label
  already says it; it is hidden while a tool's name shows, and dropped
  while a card note is up (`.card:has(> .card-note) .tools::before`), as
  the names are. The row's children stay its five buttons, which the
  smokes' `nth-child` selectors rely on. The
  tools shrink to fit (`flex: 0 1 auto; min-width: 0`), with Delete set apart
  (`.del`, `margin-left: auto`) and hovering in `--danger`. ⤢ (`.refine-tool`)
  and ❐ (`.ref-tool`) are aria-disabled with the reason (`REFINE_NONE`,
  `REFERENCES_NONE`, gallery.js) as title and label where the checkpoint
  lacks the capability, set in `card()` and again where status flips it
  (`syncCardTools` from status.js for refine, `syncReferenceMarks` for
  references); ❐ is also aria-disabled on non-reference cards at four. A
  press explains in `#gallery-error` and in an aria-hidden one-line
  `.card-note.refused` holding the gist (`refuseInCard`); the row's tips are
  hidden while a note is up. ❐ follows the toggle rule, and `setReferences`
  stays the one writer. `syncReferenceMarks`, called by `renderReferences`
  (and `markReference` in the sheet's cell builder), rings and numbers every
  card and sheet cell that is a reference (`.ref`, `data-ref-n`, a `.ref-n`
  span); a role=status `.card-note` says "Added as image N" or "Removed". The
  star is a `<button class="star fav-badge" aria-keyshortcuts="F">` in the
  card's corner: always shown on a starred card, otherwise on hover,
  focus-within or a coarse pointer. `syncStar` draws any star button (glyph,
  `aria-pressed`, the fixed "Star" name, `starTitle`); the title names (F)
  only on a button whose `aria-keyshortcuts` is F. `setFavorite` updates every
  `.card`, `.sheet-cell` and `.result` with the id, and the lightbox only
  when it shows that record. F passes `{ say: true }` (`galleryKey`,
  `sheetKey` through `starCell`, and the lightbox's keydown in app.js):
  where focus is not on a `.star` or `#lb-fav`, `sayStar` writes "Starred
  #N" / "Unstarred #N" into a visually hidden polite line - `#star-said`
  under the gallery's head, or `#lb-star-said` inside the lightbox while it
  is up, since the page behind a modal dialog is inert - emptied first,
  written on the next frame, and emptied again after 5 s. A star answered
  "unknown image" (deleted or binned in another tab; `set_favorite` refuses
  a binned image) does what ⌦ does with a gone id:
  `forgetImages`, then "#N was already deleted." in `#gallery-error`, or
  through `noteLightboxDelete` while the lightbox is open, and the Runs
  picker is asked again (`loadRuns`); its other failures go to
  `#gallery-error`. A pass is `.card.refined` with `data-strength`
  (the badge is the sheet's `::before` rule). Thumbnails are
  `object-fit: contain`, or `cover` from 1800 px where the head's Crop is
  chosen (`#gallery[data-fit="cover"]`); below 1800 px neither choice
  applies. `thumb(rec, list, src, open)` labels every image "Open #N:
  <prompt>" - a gallery card's becomes "Select #N: <prompt>" in
  `syncCardSelected`, its press being a pick; a sheet cell is labelled instead by what the sheet lays out
  ("Open #N · Steps 2 · Seed 7", a version by its name), set in
  `renderSheet`. `thumb` titles the image with `cardTitle(rec)` (#N · W×H ·
  seed N · a pass's source, "(deleted)" after a gone one, then the prompt)
  through `titleImage`, whose `aria-description` is the title's first line
  less any fact the name already says, so a sheet cell is described by its
  size and source alone; `renderSheet` calls `titleImage` again after
  naming a cell, and `forgetImages` retitles through it. This holds on the
  card, the viewer, the sheet and `#recent` alike; `card()` adds only
  `aria-keyshortcuts`, and `open` is what a press does (default: the
  lightbox on `list`). `card()` sets `data-made`, which `tombstone` keeps;
  `loadGallery` draws an `h3.gallery-break` (no data-id) where the local day
  changes from the card before (`cardDay`) and over the first card where its
  day is not today's (`isToday`), and it stays over a day of tombstones. The
  grid is one tab stop: `galleryCurrent` (an id, kept across a gallery reset
  or new query while the record is loaded, not across a page reload) holds
  the image, star and tool stops (`cardStops`). focusin moves it, and a page
  without it makes the first card current. F stars the card from anything
  inside it (`galleryKey`). On a card image, the arrows move by one and to
  the nearest card across in the next row (`rowNeighbour`, from layout, not
  the column count), Home/End jump, and Down or Right past the last loaded
  card pages in. Anything that appends or prepends cards makes them with
  `card()` and calls `keepCurrentCard()` (or `prependFinished`'s tab-stop
  rule) afterwards.
- A card's batch (`gallery-batch.js`) is presentation over records the
  page already holds: while the mouse (`pointerType`) rests on a card for
  `MATE_DELAY`, the gallery's other cards of the same `run_id` (`job_id`
  for a record with none) that agree with it on everything but the seed
  (`sameExcept(r, rec, ['seed'])`) get `.batch-mate`, a faint 1 px outline,
  so nothing moves - a Batch x Repeat and the More seeds that extended it
  are one set, a grid's other cells are not. No request, no panel; not in
  the bin or the sheet; leaving the card takes the marks down. A card
  alone in its run marks nothing.
- The gallery's selection (`gallery-select.js`) is the one set of picked
  images that actions on several images read (Swap tags, Mass refine, More
  seeds):
  `selectedIds()` (in the order picked), `selectedRecords()`, `isSelected`,
  `onSelectionChange(fn)` (called with the ids after every change, every
  toggle of Select, every page that lands and a page that fails) and
  `clearSelection()`; `galleryActionTarget(purpose, one)` is what such an
  action takes now - `{scope, n, words}` (the ids, else the filter under
  Re-render's conditions) or `{why}` - and `galleryActionScopeLine(target,
  lead)` its dialog's scope line, so the actions hold and word alike. It is ids, never the cards; `selectedRecords()` hands
  over the loaded page's copy of each record where `galleryRecords` has one,
  else the record as picked (a snapshot - actions send the ids and the
  server reads the rows). A search, a
  filter, a sheet or a page keeps it, and `#select-count` says "N selected ·
  M not shown" for the ids no live card shows. `forgetImages` takes deleted
  ids out of it (`forgetSelected`) and `tombstone` drops `.selected`; nothing
  else removes an id unasked. The recycle bin sets it aside while it is shown
  (`stashSelection` in `enterBin`, `unstashSelection` in `leaveBin`), the
  bin's own picks in its place; `forgetSelected` drops ids from the set-aside
  picks too, except the bin's cards leaving its view on a Restore
  (`dropFromBinView`, `stashed: false`). Select (`#btn-select`, `aria-pressed`, the
  footer row beside Bundles…, styled as it is so the line keeps its height)
  is the touch path. A card image's press is `cardPress`, which `card()`
  hands `thumb` as `open`: a plain click opens the lightbox without changing
  the picks or anchor. Ctrl/Cmd+click toggles a pick; Shift+click picks the
  range from the anchor alone, falling back to `selectOnly` without a loaded
  anchor, and Ctrl/Cmd+Shift+click adds it (`selectRange(div, {add})`). Only
  the first click of a multi-click opens. Enter opens; Space toggles and
  Shift+Space adds the range. In select mode every click toggles, quick ones
  too, Shift adds, and only Enter opens. The anchor is the last card toggled or picked (a pick refused at
  the cap and a range leave it); a range runs in DOM order. F and the tools
  are unchanged. The image's Shift mousedown, and a double click's, are kept
  from selecting text; a click between the cards (`galleryBlankClick`, on
  `#gallery`; never in select mode, with a modifier, or ending a text
  selection's drag) empties the selection. Escape on a card,
  Select or Clear (`selectKey`, on `#library`) empties the selection, then
  leaves select mode, and is otherwise left to the page. Clear hands its
  focus to Select as it hides. The cap is `sheetPage` (the server's
  `max_grid_images`): a pick past it is refused and a range picks what fits,
  said in `#gallery-error`. On a sheet Select is aria-disabled with
  `SELECT_ON_SHEET` as its title, reads `aria-pressed="false"` and says it
  when pressed, and the count offers no "Click cards" hint; the sheet's
  cells are not picked. `#select-count` (`countParts`: `.num`, `.word`,
  `.tail`, its text the words above) gives way before the buttons do and
  after the gallery's message; its number is never cut, " selected" goes
  out of sight for a ✓ (alt text empty) where the footer, a size container,
  is under 800 px, and only then does the "M not shown" tail end in an
  ellipsis. Before a shown Clear it keeps 4 px, Clear's own padding on that
  side halved; with Clear hidden, the row's gap before Select. A picked card is `.selected` (its group named "#N
  (selected)"); in select mode `#gallery.selecting` draws a ring at each
  card's top centre. The image is a toggle only in Select (`aria-pressed`,
  named "Select #N"); outside it is named "Open #N" without `aria-pressed`.
  Its `aria-keyshortcuts` includes Enter in both modes. `syncSelection` derives all of it from the state; call it
  after anything that adds cards other than through `card()` and
  `loadGallery`.
- The Picked menu (`#picked`: `#btn-picked`, `aria-haspopup="menu"`, and
  `#picked-menu`, `role="menu"`) stands between Clear and Select, always
  shown, and acts on the picks alone, never the filter: **Star picked** and
  **Unstar picked** (`starPicked`, one `POST /api/images/favorite` with
  `{ids, value}`, `storage.set_favorites`, answered per id `{changed,
  already, binned, missing}`; `markFavorite` marks each card, cell, result
  and the lightbox as a single star does; binned and missing ids go
  through `forgetImages`; the picks stay) and **Bin picked** (`binPicked`,
  `POST /api/images/bin` with the ids, `keepBinned`, `forgetImages`
  binned, and `noteBinned` with Restore for all of them; no question, as
  ⌦ asks none). Each outcome is said in `#gallery-error`, ids named up to
  six in id order, then counted. F stays the focused card's. The button is
  aria-disabled with `PICKED_NONE` while nothing is picked and `BIN_HELD`
  in the bin (`syncPickedButton`, from `syncSelection` and
  `syncBinControls`), a held press saying it; its title and its items'
  name the count. It keeps Examples' menu keyboard (`pickedKey`: ArrowDown
  on the button opens on the first item, the arrows, Home and End walk,
  Enter or Space chooses, Escape closes, focus back on the button; a
  keyboard click goes in on the first item), closes on a click elsewhere
  and on focus leaving `#picked`, and opens over the button (`.up`) where
  the window has no room under it. Escape on the closed button empties the
  selection, as on Select and Clear (`selectKey`). The menu hangs from the
  button's left edge, over the grid, and moves nothing. Under the footer's
  800 px container size its word and Clear's go out of sight (still their
  names) for ▾ and ✕, 3 px padding, 4 px from a shown Clear and from
  Select: the row has no room for more. There is no sheet of the picks
  (design-notes, The gallery selection).
- Swap tags (`POST /api/swap-tags/preview`, `POST /api/swap-tags`,
  `tag_swap.py`, `tag-swap.js`, `styles/swap.css`) is the fourth producer of
  cells. The server rewrites, the page shows: rows of `find` (comma items)
  and `replace` (empty removes), at most 16 rows and 20 finds a row, a find
  that two rows share refused (`parse_rows`, in the request's validator);
  `field` is `prompt` or `negative_prompt`, the latter 422 where the
  checkpoint has no negative prompt. Matching: whole comma items through
  `prompt_text._norm`; a weight wrapper (`(x:1.2)`, `((x))`, `[x]`) looked
  through only where it wraps the whole item and put back round the
  replacement, which takes the matched item's underscores and escaped
  parentheses; inside a `_is_prose` item a one-find row also matches as a
  phrase bounded by lookarounds on the find's own word edges, never inside
  `rendered_spans`, a weight typed round the find looked through as for
  items; a phrase match inside another row's phrase match is not applied
  (`_put_phrases` keeps the earlier, `_applied_phrases` says which), so a
  row left with none of its own is `missing`; a find with Han or kana is a
  plain substring; a
  many-to-one row that would need a phrase is `by_hand`; XML and JSON
  prompts are `structured`. All rows match the original at once; strict -
  an image missing any find of any row is `missing` and not queued. Many to
  one lands at the first matched item in the prompt's order, the others
  removed with the separator before each; a replacement already in the
  prompt is not written twice. A variable's value is rewritten where it is
  part of the swapped text, never `variant`. Per image, never a whole
  refusal: `queued`, `missing`, `unchanged`, `rendered` (the gallery holds
  the cell at its size), `invalid` (pydantic, the budget, the refine steps:
  `error_detail`), `source_deleted`, `lora_missing`, `duplicate` ("same as
  #N": cell keys as `applied()` runs them, plus the size), `by_hand`,
  `structured`, `deleted` (an id gone, or binned since it was picked:
  detail "in the recycle bin", `submission.gone_detail`), `foreign`. The preview lists the
  first `api_swap.SWAP_PREVIEW_ROWS` (200) of them - each with its text, the
  original where it is not changed, so a missing image can be read - and
  counts all, with the ids the
  selection resolved (`ids`), how many are queued and in how many jobs; it
  has no side effects and is not a model-dependent request (the page's
  `api` treats it as advisory). The queueing route is model-dependent
  (`hold_model_for_request`), recomputes everything, refuses 409 when
  `expected` is not the ids resolved now, leaves out every cell the gallery
  holds at queue time (`pruned`: the preview's already rendered ones and any
  since), answers 409 when everything is rendered and 422 when
  nothing else is queueable, and submits one job per distinct size
  (`jobs.submit_cells(..., swap=summary)`, run id per job as any run's),
  answering the first job at the top level and the rest in `more`; the
  snapshot's `swap` names the job (`describeJob`: "swap · …").
  `startJob` registers `more` as this page's own and queues them behind
  the first, recording all of them as the submission's `split`, so each
  one's finish line says "Job k of n of this swap, at WxH". The button is
  `#btn-swap`, in the footer row between Select and Bundles…, always
  shown and styled as they are: it acts on the
  selection when one exists, else on the filter under Re-render's
  conditions, and is `aria-disabled` with the reason otherwise
  (`swapTarget`, `syncSwapButton` from `syncRerender` and
  `onSelectionChange`), its title naming what it takes; a held press says
  why in `#gallery-error`. The dialog fixes its scope at open, checks
  400 ms after an edit (at once on open, on a row removed, on the field
  changed, from `forgetImages` when an id it listed went:
  `forgetSwapImages`, and, on a filter scope, when the gallery's `total`
  changes under it - a job landing matching images - with the answer
  dropped first: `swapGalleryChanged` from `syncSwapButton`), aborting a
  check still out; the scope line says the filter's count from the latest
  answer's `total` (`swapSayScope`); Queue is aria-disabled
  with "Add a tag to find", "Checking the swaps…" (the answer is not to the
  payload typed now), the server's refusal, "The checkpoint changed…" or
  "No image would change"; a press (Queue, Enter in a box, Ctrl+Enter)
  first waits for the check of what the rows say, as More seeds' and Mass
  refine's do (`queueSwap`; a second press while one waits does nothing -
  `swapWaiting`, `msWaiting`, `rbWaiting` - since two waits aborted each
  other's check in a loop that never yielded), then says a hold in
  `#swap-error`, and sends
  the answer's `ids` as `expected`; accepted, the dialog closes, refused,
  the old answer is dropped and it checks again, the server's reason kept in
  `#swap-error` (`swapRefused`) until the next edit (`swapEdited`) or
  press - a check that frees Queue does not clear it. Its marks are the
  server's `spans`, and under a changed text "Took out: …" lists the
  server's `removed` (`tag_swap.Outcome.removed`: every item dropped or
  written over and every phrase replaced, as the original wrote it, in its
  order, leaving out a tag the new text still has as an item - an identity
  or a reorder row) struck through (`swapTookOut`), since a removal leaves nothing to
  mark; the rows box and the results list have fixed heights
  and scroll inside the dialog, the count line keeps two lines and
  `#swap-error` one, so neither a row added nor a check moves anything
  below them. A removed row keeps its slot hidden (`.gone`, left out by
  `swapLiveRows`) until the pointer leaves the rows box
  (`collapseSwapRows`), so the next row's × never comes under the pointer;
  the Add row button is aria-disabled at 16 rows ("At most 16 rows in one
  swap"), and both boxes take at most 1000 characters (`maxLength`), the request's
  own limits. Enter in a box
  is Queue's press, as Ctrl+Enter is anywhere in the dialog; Escape, Cancel
  and a press outside close it.
- Mass refine (`POST /api/refine/batch/preview`, `POST /api/refine/batch`,
  `refine_batch.py`, `refine-batch.js`, `styles/refine-batch.css`) is the
  fifth producer of cells: a refine pass of each selected image, from its
  own pixels (`refine = record.id`), at one `strength` (None: the
  checkpoint's, `default_strength`) and one `scale` (1-2, the record's
  sides times it through `refine.scaled`), `steps` None keeping each
  record's own. `refine_batch.plan` decides per record from the record
  alone, in this order: `pass` (a record that is itself a pass, while
  `skip_passes`, the default; unticked, the pass is refined itself, its
  source the pass), `foreign` (another checkpoint's, unless
  `include_foreign`; then its prompt pair and seed with the running
  profile's steps, CFG and mu, no true CFG and no LoRA, each said in the
  row's `notes`), `too_big` (the scaled size past `refine_max_side` or
  `refine_max_pixels`, checked before the request is built), `invalid`
  (pydantic), `no_step` (`check_refine_steps`), else a cell through
  `grid.cell_from_record` with `reference=[]` and `label="#<id>"` - one
  record, one cell, in the resolver's order (oldest first), never a loop
  over axes, so `grid.expand()` stays the one cell loop. The route adds
  `file_missing` (`storage.missing_files`), `lora_missing` (a same
  checkpoint's adapter, `submission.unusable_input` once per name), `rendered`
  (`submission.not_yet_rendered` per size) and `deleted` (an id gone or binned). Only a
  checkpoint with no pass refuses the whole of it (422,
  `submission.refine_unavailable`, the single pass's words). The preview lists the
  first `api_refine_batch.REFINE_BATCH_PREVIEW_ROWS` (200) with each source's size and the
  pass's, counts all, and answers the resolved `ids`, the strength, the
  `foreign` count and `models`, `passes`, `largest` (the largest source the
  scale applies to, by pixels, `pass` and `foreign` left out), `seconds`
  (summed per size from `gallery_reports.refine_cost_ms`, None with
  `unpriced_sizes` where any size has no history) and
  `start_sigma_low`/`high` (`backend.start_sigma` over the distinct applied
  passes, None where any reads None or past `api_refine_batch.REFINE_BATCH_SIGMA_KEYS`, 8);
  it has no side effects, is not model-dependent and is advisory in `api`.
  The queueing route is model-dependent, refuses 409 when `expected` is
  not the ids resolved now, prunes what the gallery holds (`pruned`),
  answers 409 only where every image's status is `rendered` and
  otherwise 422 naming every count (the rendered among them, so the ones
  the user could fix are said), as the dialog's hold words it, and submits one job per distinct size
  (`jobs.submit_cells(..., refine_batch=scale)`), the rest in `more` as
  Swap tags'. The snapshot's `refine_batch` and `strength` name the job
  ("refine ×1.5 at strength 0.3", `describeJob`), its `refine` is None;
  the Cancel title is "Stop this batch refine" and a split job's finish
  line "Job k of n of this batch refine, at WxH". Cells run oldest first;
  a source deleted mid-run fails its own cell, named by its label. The
  button is `#btn-refine-batch` ("Refine…"), in the footer row after
  Swap tags… and before Bundles…, always shown: `refineBatchTarget` holds
  it with `REFINE_NONE` (settled) or the hedge (a guess) where the
  checkpoint has no pass, else with `galleryActionTarget`'s reason; synced
  from `syncRerender`, `onSelectionChange`, `syncCardTools` and every
  status (`syncRefineBatchButton`), a held press said in `#gallery-error`.
  The dialog opens only on the first preview's answer: the opening scale
  is the remembered one (`REFINE_KEY`'s `scale`, else 1.5×) capped by
  `openingRefineSize` over `largest`, asked again where capped, the passed
  over factor's button saying "(last run at 2×)" or "(the usual opening
  size)"; Strength opens at this checkpoint's remembered single strength,
  else its default, focused and selected; Steps blank; Skip refine passes
  ticked; Include other checkpoints hidden unless the answer's `foreign` is
  above 0 (or it is ticked). A list or range in Strength is refused on the
  page (`RB_ONE_STRENGTH`), as is Steps that is not a whole number 1-100
  (`RB_STEPS_PROBLEM`), without asking the server. Steps is a text box, as
  Strength: a number box reports text it cannot parse ("1e", "-") as an
  empty value, which here means each image's own. Checks run 250 ms after
  an edit (at once on a scale, a tick, from `forgetImages` when a listed id
  went: `forgetRefineBatchImages`, and, on a filter scope, when the
  gallery's `total` changes under the dialog: `refineBatchGalleryChanged`
  from `syncRefineBatchButton`, the answer dropped first so a press waits
  for the new one), aborting one still out. The scope line says the filter's
  count from the latest answer's `total` (`rbSayScope`). Refine is
  aria-disabled with `submitHold()` (re-synced from `syncSubmitButtons`, so
  "The server is not answering" holds it at once), no refine pass, the field's refusal,
  "Checking the images…", the server's refusal, "The checkpoint
  changed…", "Every image already has this pass at this strength and
  scale" or "No image can be refined at these settings; each one says
  why"; a press (Refine, Enter in a box, Ctrl+Enter) first waits for the
  check of what the dialog says, then says a hold in `#refine-batch-error`
  and marks Strength or Steps where it is about them. While the POST is out
  Refine reads "Sending…" and Escape, Cancel and the backdrop wait
  (`rbSending`); accepted, the dialog closes and the scale (unless it is
  the capped opening one, unchanged) and a single strength are remembered
  in `REFINE_KEY` (`rememberRefineBatch`); refused, it stays open with the
  reason and checks again. ↑/↓ step Strength by 0.05, 0.01 with Shift. The
  results list has a fixed height and scrolls, the count keeps two lines
  and `#refine-batch-error` one, so nothing below moves.
- Then refine (`docs/plan-then-refine.md`; `then_refine.py`,
  `JobManager._passes`, `then-refine.js`) is a submission field, never a
  request field: `GenerateSubmission.then_refine` (`/api/generate`) and
  `GridRequest`'s, `ThenRefine` (scale 1-2, strength None for the
  checkpoint's own, steps None for each cell's). The cells are built from
  the request without it (`GenerateSubmission.base()`; `grid.expand`
  ignores it), so a then-refine run's base rows, `CellKey`, run id and
  Resume are a plain run's. The route resolves the strength
  (`then_refine.resolved`) and refuses in words before queueing
  (`then_refine.refusal`: `refine_unavailable`, a base that is already a
  pass, a pass past `refine_batch.too_big`, a strength leaving no step of
  any cell's steps, checked on a stand-in pass per distinct steps); the
  worker checks a stand-in against the loaded pipeline with the cells.
  After each cell's saves, `_passes` builds one pass per record with
  `refine_batch.plan` (Mass refine's cell, so Mass refine reports it
  `rendered`) and runs them before the next cell, under
  `job.pass_run_id` (`then_refine.pass_run_id` of the base run and the
  spec: deterministic, so a Finish or Resume continues it; a run keeps one
  size). A cell and its passes are one unit: pause and the budget act
  before cells only, a cancel stops inside a pass with the base kept. A
  failed pass is in `failed_cells` ("pass of #N") and `failures`, never in
  `failed_indices`, and has its own streak (three passes in a row end the
  job; a base made between them does not reset it) beside the share of
  cells plus passes. Pending passes are in `planned_keys`. The snapshot
  carries `then_refine`, `pass_run_id`, `stage` ("pass") and
  `planned_images` doubled; the retained metadata `then_refine` with its
  `pass_run_id`, which Finish passes back, and a pass run's Finish names
  its base run (`storage.retained_run_metadata`). The grid preview gives
  `passes`, `pass_width`/`pass_height`, the passes' time at the pass size
  and disk, budget seeds pricing each cell with its passes, and the pass's
  start sigma. The page's opener, `#then-refine-open`, ends the Notify
  row under Generate at that row's own height (the hint before it
  ellipsizes), because the panel has no height to spare at 1400x900 - a
  fold of its own there pushed Run grid below the window and scrolled the
  stage; it opens `#then-refine-dialog`, whose tick is off on every load
  and held (`aria-disabled`, `REFINE_NONE`, a press says it) without a
  pass. The opener's words ("Refine 1.5× · 0.4"), Generate's and Run
  grid's words and titles say the setting while it is on; scale and
  strength are `REFINE_KEY`'s. The CLI's
  `--then-refine-*` run the same plan (`_then_refine_passes`).
- More seeds (`POST /api/more-seeds/preview`, `POST /api/more-seeds`,
  `more_seeds.py`, `more-seeds.js`, `styles/more-seeds.css`) is the sixth
  producer of cells: `count` more images of each batch the selection
  holds. A batch is `more_seeds.batch_key`: the lineage (`run_id`, else
  `job_id`, else the image alone), checkpoint, quantization, size and the
  resume key's elements but the seed (`settings_key`, as
  `completed_cells` reads a row) - one cell of a grid - so picked images
  of one batch are one batch, in the order of their oldest. The selection
  is `submission.resolve_selection`'s: picked `ids`, else the filter, or
  with `run` (refused beside `ids` or `search`, 422; `favorites` and
  `all_models` ignored) every live image of that run from
  `lineage_images([run], [])`, every checkpoint's, uncapped - 404 "run X
  is not in the gallery", or "every image of run X is in the recycle bin";
  its binned images are no batches but still hold their seeds, and a run
  scope's `expected` is not capped either. Per batch, in
  the resolver's order: `foreign` (another checkpoint's), else
  `more_seeds.plan` over every image the gallery holds of its lineage
  (`storage.lineage_images`, picked or not, binned ones included, so the
  seeds follow on past a binned seed and its Restore never duplicates one): the Batch is the largest
  `batch` recorded on them (a call that ends a batch holds fewer), else the
  request's `batch` (the panel's), noted, and held to `max_batch_pixels` at
  its size (`fitting`; a refine pass is not), noted; the seeds are
  `next_seeds` from one past its highest, stepping over every key the
  gallery holds at its size (`completed_cells`) or a live job will write
  (`jobs.planned_keys()`) as `applied()` runs them, the chosen ones added so
  a later batch of the same settings steps past them too; consecutive seeds
  are one cell of up to the Batch (`calls`), each `grid.cell_from_record`
  of its newest picked image with `seed` and `num_images` named and a
  `seeds a–b` label, `invalid` where that or `check_budget`/
  `check_refine_steps` refuses. The route then adds `source_deleted` and
  `lora_missing` (`submission.unusable_input` once per spec), a note where the batch
  was made under another quantizer, and `deleted` (an id gone or binned); nothing is
  ever `rendered`. Past `max_grid_images` images in all (batches ×
  `count`) is a 422 for the whole, by count. The preview lists the first
  `api_more_seeds.MORE_SEEDS_PREVIEW_ROWS` (200) by id - each batch's newest picked image,
  its picked `ids`, the prompt (300 characters), the images and seed range
  it holds, `first`/`last` new seed, `skipped`, `batch` and `recorded` -
  counts all, and answers the resolved `ids`, `queued` (images),
  `batches`, `jobs`, `unrecorded`, `seconds` (per size from
  `step_cost_ms`/`refine_cost_ms`, None with `unpriced_sizes` where any
  size has none), `count_default` (`more_seeds.count_default`: the Batch
  every batch that recorded one agrees on, else None - what a run surface
  opens the count at) and, for a `run` scope with something queued,
  `headline` (`more_seeds.run_headline`: "N more of each of C cells = T
  images · seeds a–b · Starred ignored", "of its one cell" for one, the
  Starred part only where the request ticked it); it has no gallery/queue side effects, is not model-dependent and is
  advisory in `api`. The queueing route is model-dependent, refuses 409
  when `expected` is not the ids resolved now, 422 naming every count when
  no batch is queued, and submits one job per lineage and size
  (`api_more_seeds._more_seeds_jobs`; `jobs.submit_cells(..., run_id=<the batch's run>,
  more_seeds=count)`), so the images join the run they extend and every
  job keeps one size; a batch with no run id gets a run of its own. The
  snapshot's `more_seeds` names the job ("more seeds · <first prompt>",
  `describeJob`); the Cancel title is "Stop these more seeds"; the finish
  line gives the seed span (min–max) as a plain job's does, without Use
  seed, and a split job's says "Job k of n of these more seeds, at WxH".
  The button is `#btn-more-seeds` ("More seeds…"), in the footer row
  after Refine… and before Bin and Bundles…, always shown, held with
  `galleryActionTarget`'s reason; synced from `syncRerender` and
  `onSelectionChange` (`syncMoreSeedsButton`), a held press said in
  `#gallery-error`. The dialog opens on the first answer, the count (from
  the footer) at the last one queued (`MORE_SEEDS_KEY`) else 4, focused and
  selected; a first
  check that fails is said in `#gallery-error` and opens nothing, except the
  422 by count ("N batches × C is…", which only the count box can lower):
  that opens the dialog with nothing listed, the refusal in the count line
  and `#more-seeds-error`, the box `aria-invalid` and Queue held with it
  (`msProblem`), until an edit is checked; it is a
  text box, refused on the page unless a whole number 1 to `sheetPage`
  (`msCountProblem`), ↑/↓ stepping it by one. Checks run 250 ms after an
  edit (at once from `forgetImages` when a listed id went:
  `forgetMoreSeedsImages`, and, on a filter scope, when the gallery's
  `total` changes under it: `moreSeedsGalleryChanged`), aborting one still
  out; the scope line ("From …") counts the answer's `ids`. Queue is
  aria-disabled with `submitHold()` (re-synced from `syncSubmitButtons`),
  the count's refusal, "Checking the settings groups…", the server's refusal,
  "The checkpoint changed…" or "No batch can have more seeds here; each
  one says why"; a press (Queue, Enter in the box, Ctrl+Enter) waits for
  the check of what the dialog says, then says a hold in
  `#more-seeds-error`, marking the box where it is about the count. While
  the POST is out Queue reads "Sending…" and Escape, Cancel and the
  backdrop wait (`msSending`); accepted, the dialog closes and the count is
  remembered; refused, it stays open with the reason and checks again. The
  results list has a fixed height per opening and scrolls, the count line
  keeps two lines and the feedback three, so nothing below moves; Queue and
  Cancel keep fixed action columns.
- The direct results action (`#viewer-more-seeds`) stays outside the strip,
  on the same element as images arrive. It takes only settings groups already
  represented by completed results, and says that scope. `#viewer-note-text`
  owns the former result-note text; `#viewer-note` also holds the action.
  The caption keeps its reserved line even when empty or hidden by the wide
  result column, preserving gallery geometry through arrival and docking.
  More seeds leads with additional images and unchanged settings, calls a
  ready preview Ready, and puts technical seed/Batch explanations in details.
  Each opening fixes its compact or multiple-group list height; fixed action
  columns preserve Queue and Cancel rectangles through asynchronous replies.
  An active job permits queuing; another POST still pending visibly holds it.
  Opening previews can be cancelled with Escape and cannot open after their
  originating result, lightbox, footer scope or modal context has gone.
- Browser More seeds submissions carry the preview's `intent` and resolved
  `expected` ids. `more_seeds_intents.IntentLedger` owns one admission task
  per token through disconnects, under the same rerun admission lock. It
  bounds itself to 512 entries, issues previews for 15 minutes and retains
  settled receipts for one hour. Unknown, expired or mismatched tokens refuse
  before allocation; repeated accepted tokens return the original job ids.
  A partial multi-job admission acknowledges the admitted jobs and counts
  the remainder in a terminal warning; replay never adds that remainder.
  Pending intents count as busy and drain before worker shutdown. This is
  scoped to the manager lifetime, never durable exactly-once execution.
  After a lost reply, the browser freezes the exact body/count/token through
  later errors, including generic 4xx middleware refusals; Retry same request
  does no preview. Receipt-specific 409s stay held and require reviewing the
  queue before a deliberate new request. Closing can abandon an uncertain
  reply, as the dialog says. Legacy callers without a token still request a
  deliberate new extension on every POST.
- More seeds where a run ends (`docs/plan-continue-run.md`, phase 1) is the
  same dialog on a ready target, `openMoreSeeds(target, {count, opener, host})`
  with `target` `{scope, line}`: an image's batch (`{ids: [id]}`,
  `moreSeedsImageTarget`, "From #N's batch.") from the lightbox's
  `#lb-more-seeds` (paired with Reuse in an `.lb-pair`, hidden under 900 px
  where the menu has it), the menu's More seeds of this batch… and the
  result column's More seeds…; a finished job's images (`{ids}` of those
  still live, `finishMoreSeeds`) on the finish line of a plain job
  (after its Use seed), a More seeds job and a single refine, inpaint or outpaint
  pass (not a grid's, a swap's or a batch refine's); and a whole run
  (`{run}`, `moreSeedsRunTarget`, `moreSeedsRunButton`) for a `done` grid
  after Show sheet, the banner and the sheet's run bar. The label names the
  scope (of this batch, of this run); nothing is a bare "Continue". The
  footer alone reads and writes `MORE_SEEDS_KEY` (`msFooter`); a ready
  target opens at the press's `count`, else the server's `count_default`
  (asked again at it where the first check was not), else 4, and its scope
  line is the target's own `line`. Every such button mirrors one hold
  (`moreSeedsToolHold`, the button's `msOf` kept on it, `aria-disabled`
  with the reason as its title, never hidden): every image of a job gone
  since, the recycle bin, another checkpoint ("Made on X; the server is
  running Y"), the server offline, the dialog open or opening; everything
  else - one batch among several from another checkpoint, a source or an
  adapter gone - is the server's, per batch. The banner's `#last-run-more`
  is held for no checkpoint: the job snapshot it reads names none, so a run
  made on another is the server's `foreign`, batch by batch, with Queue
  held where nothing is left. They carry `.more-seeds-tool`
  and are synced again together (`syncMoreSeedsTools`) from
  `syncReuseButtons`, `syncSubmitButtons`, `forgetMoreSeedsImages` and the
  dialog's open and close; the menu's item is held by the same function, so
  a new place for the button adds no hold of its own. A refusal while
  opening is said where the press was (`msOpenSlot`): `#lb-error` while the
  lightbox is up (the dialog opens over it, the lightbox stays, and its
  keys are ignored while the dialog is open, so Escape closes the dialog
  alone), `#error` for a press in the panel (the finish line, the banner),
  else `#gallery-error` - and, for a card or result tile (`host`: the
  menu's item, the result column's More seeds…), over it too, through
  `refuseInCard`. `aria-busy` is the pressed button's only (`opener`; the
  footer passes its own); a menu item, its menu gone, has none.
  `moreSeedsGalleryChanged` skips a run scope as it skips `ids`; the count
  technical details retain the server's `headline` where there is one. The sheet's
  run bar is More seeds of this run… after Resume this run, on
  `sheetShownRun` (one run opened on `run:<id>`, a refine sweep's and a
  comparison's too, held - not hidden - on another checkpoint), then its
  status line (`sheetRunStatus`, "Run X · N images · checkpoint quant",
  "starred images" under Starred) last, on a row of its own
  (`.sheet-run-status`), since its count changes while a job writes to the
  run and text before the buttons would move them under the pointer.
- The sheet's layout contract - axes, prompt labels, pins, variables - lives in
  the client, and the export is client-side, not a route. Export PNG draws
  the slice on screen. Export all, shown only where something is pinned,
  draws every pinned slice that has images (`sliceCells`): it stacks them in
  one PNG, each titled by its pins, at a cell width that keeps the stack
  within `EXPORT_MAX_EDGE` and never below 96 px, named
  `sheet-<run>-every-<axes>.png`; past that edge it saves one image per
  slice, named by its pins with -2, -3 where two come out alike, at most
  `EXPORT_FILES_MAX` (24), and says so in `#gallery-error`. A fitted sheet
  exports thumbnails, cells up to 384 px; a filled one exports at the larger
  of its cells' device pixels on screen (`exportLayout()`'s `shown` × `dpr`)
  and the thumbnail's width, so never smaller than the fitted export, from
  the PNG where that is wider than the thumbnail. Every export draws from
  `exportLayout()`'s snapshot (the layout, its axes and `sheetWords()`: the
  prompt labels, the adapter and the fixed Mu), never the live
  `sheetAxesList`/`promptShort`/`sheetAdapter`/`sheetMuStatic`, so a sheet redrawn while
  Export all writes its files changes none of them. Both export buttons are
  aria-disabled with the reason while one runs. A failure is said in
  `#gallery-error` ("The sheet could not be exported: …"), and the smokes
  stand in for `saveCanvas`. `sheetFileName` drops a hyphen that a value's
  closing mark ("none (1.79)", a clipped label's "…") left before the next
  pin or `.png` (`sheet-<run>-mu-none-1.79.png`). Swipe CFG
  (`sheetSwipeButton('cfg')`, sheet-render.js) is drawn after the exports only where
  CFG is a laid-out axis (Columns or Rows), never where it is pinned or
  does not vary; it is `aria-disabled` with "No two cells here differ in CFG
  alone" where no shown cell has a ladder of two (`syncSheetSwipes`: the
  cells grouped by `ladderGroupKey`, then the press's own `ladderOf` within
  each group with more than one CFG value - the key leaves True CFG out, and
  a CFG x True CFG grid with cells missing grouped two values no ladder
  held), its press saying so in the page's line, and saying the hold's
  reason, never the pressable title, where the cells changed under it. A press opens the lightbox on the
  cell the keyboard was last on in the sheet (`sheetSwipeFrom`, a focusin in
  app.js) where it has a CFG neighbour, else the first shown cell in
  reading order with a lower CFG beside it, walking that cell's own ladder
  from the shown cells (`cfgLadder(rec, sheetLayout.cells)`, ascending -
  its row or column, never the pinned-out cells), with the CFG comparison
  on in swipe (`toggleCompare('swipe', 'cfg')` after the open). Swipe
  steps (`sheetSwipeButton('steps')`, `SHEET_SWIPES`) is the same along
  Steps, drawn after Swipe CFG where Steps is laid out: held with "No two
  cells here differ in steps alone", the group key skipping Steps alone,
  opening the cell's own steps ladder from the shown cells with the steps
  comparison on (`toggleCompare('swipe', 'steps')`). The sheet asks for
  the whole run in one page. A variable axis, and the Prompt axis, list their
  values by first appearance, not alphabetically: the oldest image carrying
  each value. Under Starred or a search narrower than the run, that is the
  oldest in the whole run, which `learnSheetOrder` asks for once. A skipped
  first cell, a late resume or rows Delete unstarred removed can still move a
  value down. A varying `var:variant` axis takes the rows whenever anything
  else varies, however few of its images are loaded (placed from every
  varying axis, so the listlike rule drops neither it nor the axis across it;
  `VARIANT_AXIS`, held to `grid.VARIANT` by `tests/test_grid.py`), so a
  comparison reads Original, Reordered, Rewrite, Your version, top to bottom:
  display names (`VERSION_NAMES`, under the axis label Version) on the sheet
  and in its export, with the stored labels, the `var:variant` key and
  `stats --by variant` unchanged. A shared run
  or mixed search can give one version label different prompt texts: retain
  Prompt as an axis when the complete variable assignment is ambiguous.
  `stats --job <id> --by variant` reads one comparison; a broader grouping
  with conflicting texts for one assignment is refused with advice to group
  by prompt too. Ordinary multi-variable grids and explicit grouping fields
  still aggregate as requested. An empty sheet draws no grid, only its note.
  Laid out again (`renderSheet`: a page landing, a delete, a pin, an axis, a
  wrap that flips), the sheet keeps the keyboard where it was.
  `sheetFocusPlace` records the place before anything is removed
  (`forgetImages` takes it before it drops the cells), and
  `restoreSheetFocus` returns to it, with `preventScroll`, unless focus was
  kept or moved on purpose meanwhile: the same cell's image or star by id;
  else the image now at its place in reading order, else the last; a
  control by its name (a select by its label, a button by its words), else
  the control at its index; the search box where nothing is left. F on a
  cell another tab had deleted, and an ArrowDown on Columns, used to drop
  focus to `<body>`.
- Every sheet the page opens (a comparison, a refine sweep, Runs, a card's ⊞,
  both branches of Show run) goes through `showSheetScope`. It keeps the
  search it replaced (`sheetReturn`), which Sheet off restores while the box
  still holds what the page wrote. It asks `setSheet(true, { land, asker })`,
  which, once the sheet is drawn and only if it is still the one shown,
  scrolls the gallery head under the topbar (`landOnSheet`),
  reserving the room under the sheet its head needs to reach the topbar
  (`reserveSheetRoom`: a margin outside the box `fitSheet` measures, from
  where `main` ends - never the document's scroll height, which is at least
  the window - cleared by `setSheet`). Once that scroll settles
  (`scrollSettled`: three frames at one scrollY, 1.5 s at most),
  `focusSheet` moves focus with `preventScroll` to the sheet's first cell,
  else the search box, in two cases: focus is still on the control that
  asked and that control is not wholly in sight (`wholeInSight`: inside the
  window under the topbar and inside every ancestor that clips its
  overflow, such as the docked check's scroller at 2560), or focus has
  fallen out of the page (`<body>`, or hidden: a card's ⊞ under the sheet).
  `asker` is the element focused at the call; a job's sheet is given the
  element pressed at submission, kept in `submissions`. Where the page
  itself moves focus off a job's asker (the Refine dialog closing on its
  run, `refineClosed`, the strip a job's first frame empties), it does so
  through `handJobFocus`, which passes that submission's asker on to the new
  target. Focus anywhere else is never moved. Every sheet `setSheet(true)`
  opens is said in `#sheet-said` (polite, outside `#sheet`: "Sheet: 6
  images, Steps × Seed", via `saySheet`), written on the next frame after an
  empty line. A refine sweep's source cell is
  cropped to the passes' shape (`object-fit: cover`, `refine.fitted`'s
  centre crop), captioned "cropped as the passes were" where the shapes
  differ. The Sheet button does not scroll - except where the sheet takes
  the page out of Gallery beside, which `setSheet` lands on; over an empty
  search with several runs it opens the newest run, with a `sheetNotice`,
  and a typed search is never redirected. A matrix's cells are capped to
  the window's height (`--cell-max` from `fitSheet`'s measured
  `--sheet-chrome`) with a 96 px floor. With Fill width pressed they take
  the width instead: `.sheet.fill-width` sets the template's `--cell-w` to
  `1fr`, never through the class `.fill`, which is the progress bar's. The
  toggle keeps one name, with its state in `aria-pressed` and its title; it
  is remembered under `auradream.sheet.v1` through
  `remembered`/`remember`, is not forgotten by Reset, and is aria-disabled
  with its reason on a wrapped row. A matrix zooms (sheet.js:
  `sheetZoom` times `sheetZoomBase`, the cells' width at zoom 1 that
  `fitSheet` measures with the zoom off, set on the grid as `--cell-min`
  and `--cell-w`, the box's scroll restored), from 1 to `SHEET_ZOOM_MAX`
  screen pixels per image pixel (400%), by Ctrl+wheel about the pointer (anchored
  to the nearest cell), `+`/`−`/`0`/`1` on a focused cell, and the `−` `+`
  `1:1` buttons with a readout that goes back to zoom 1 when pressed; the
  buttons are held with a reason where they would do nothing, a wrapped row
  included. Zoomed, every cell's image is its PNG (`data-full`, the srcset
  dropped), never the thumbnail. A left-button drag (not a touch, a button
  or a select) pans `.sheet-scroll` wherever it overflows; under 4 px it
  is a click, and a pan's release never opens the lightbox. Every sheet
  opens at zoom 1; a redraw keeps the zoom and the box's pan
  (`sheetScrolled`, taken by drawSheet before it empties the grid), and a
  new base (Fill width, a resize) keeps the cells' drawn width; the zoom is said in
  `#sheet-zoom-said`, a wheel's once its burst ends; the export ignores
  it. Its heads stick inside `.sheet-scroll`, whose scroll
  padding is the measured head row and row-head column (`--sheet-head`,
  `--sheet-rowhead` on `.sheet`) so a keyboard-focused cell is never under
  them. `refitSheet` (debounced, on the window's resize) measures again and
  re-lays out when the wrap test (`sheetWraps`) flips. A row-less sheet wider
  than fits wraps with captions, and so does its export. Head labels are
  whole on screen (span.label, clamped, title = full value). A number of
  five digits or more goes to a line of its own whole (`numberLabel`: an
  inline-block `.number`); it breaks only where it alone is wider than its
  column, and then only between groups of four digits counted from its end
  (a `<wbr>` between them), in balanced lines (`text-wrap: balance`); the
  text is still the whole number. `clip` is for pins, the note and the
  canvas. A matrix is a role=table named by `sheetSummary`; its rows are
  `.sheet-row` with role=row and `display: contents`, so heads and cells
  stay the grid's items and the heads stick; its heads are columnheader and
  rowheader (the corner is a cell), and its cells role=cell (the source
  with `aria-rowspan`). A wrapped row is a role=list of listitems. Prompt
  labels that repeat or exceed
  `LABEL_CHARS` fall back to `promptEdit` against the oldest prompt, up to
  `EDIT_LABELS_MAX` prompts, memoised per prompt list. A version that only
  puts the original's top-level XML blocks in another order, with nothing
  in or between them changed (the loose text compared gap by gap), reads
  "same blocks, moved: `<general_tags>` first" (`blockMove`, over
  `xmlBlocks`); its `XML_BLOCK` is `critic_reorder._XML_BLOCK` with Python's Unicode
  `\w` written out, held equal by `tests/test_critic_reorder.py`. One block's move
  reads "first", "last", or "before"/"after" another; anything else reads
  "in the order …", cut between two tags within 80 characters (`tagLine`,
  `EDIT_CHARS`). The edit fallback labels such a prompt by its move too,
  and the oldest as "blocks `<a> <b> <c>`" where every other prompt is a
  move. A version that only reorders the original's words reads "same
  words, moved: …" (`versionEdit`). A refine sweep's shared source is a leading column
  fetched by id (`sheetSources`) and never a sheet record, axis value or
  count; `forgetImages` nulls it. Records that pass the pins but find their
  cell taken are counted in the note, repeats of the cell's image
  (`sameSettings`: every axis, size, checkpoint, quantization, refined
  source; `sameExcept` with no axes skipped) apart from images that differ in something the sheet does not lay
  out; the note reads "N of M loaded" only when none collide. Where the
  Prompt axis varies and every image has a seed of its own (`ownSeeds`),
  the note adds `OWN_SEEDS` ("each image has its own seed: Check's Compare
  on the same seeds puts versions side by side") ahead of a `sheetNotice`.
  The shared settings go in `#sheet-fixed`, never the note. The starred
  cell uses the card's own `.fav` and `.fav-badge` rules (`.card.fav,
  .sheet-cell.fav`, `.sheet-cell .fav-badge`). Every cell has the card's
  star button: tabindex -1, so no tab stop; shown on hover, focus-within
  and a coarse pointer, and always when starred; drawn by `syncStar`. That
  button and F on a focused cell both star through `starCell` →
  `setFavorite` (F said in `#star-said`). A `.sheet-cell` is `isolation:
  isolate`, so its badges stay under the stuck heads. "Resume this run" (one run of the running
  checkpoint opened on `run:<id>`, not a comparison or refine sweep) fills
  the run's seed and size and never persists Resume. `resumeRun(runId,
  known)` always reads the base seed from `/api/runs` (the run's oldest
  image); `known` gives only the size, for a page with no sheet. The
  last-run banner appends ", N since deleted" or ", all since deleted" from
  `deleted_count`, hiding Show run when all are gone, and offers Resume
  (`resumeLastRun` → `resumeRun(run_id, { width, height })`) for a grid that
  stopped short - cancelled, its budget, or `stopped_by` failures - with an
  image of its own left; never for a re-render, a refine sweep or a prompt
  comparison (the snapshot's `comparison`). For a `done` job with a run id
  and an image left it shows More seeds… (`#last-run-more`,
  `moreSeedsLastRun`) beside Show run, on the run scope. When Resume this
  run's preview renders no cell but skips or queues some, `resumeRun`
  appends "Nothing in run X is missing. More seeds of this run… adds
  seeds." to it. The snapshot's `seed` is a job's
  first remaining cell, past the base after a Resume, and is never used as
  the base.
- `generate`, `clear`, `stats` and `reindex` reconfigure stdout to UTF-8 at
  the top of `main()`: on Windows always, elsewhere when it is not a
  terminal, and never on a stream without `reconfigure` (a StringIO,
  pythonw's None). On Windows a redirect, a pipe or `> NUL` otherwise uses
  the locale's code page, and a CJK or emoji prompt is a UnicodeEncodeError;
  the console is UTF-8 already. A new CLI does the same; `tests/test_cli.py`
  holds them to one setup.
- `generate.py` duplicates `jobs._generate`'s loop deliberately. It applies
  `--model/--quant/--offload` to `os.environ` before importing
  `backend.config`, as `launch.configure()` does for the server; both take the
  flags from `launch.add_model_arguments`.
- `generate.py`'s header shows what `applied()` will run: a negative
  prompt, `--true-cfg` or Mu the checkpoint drops is printed with
  `[ignored: ...]` (a `--grid-mu` axis too), and the duplicate warning names
  a Mu where it takes none
  (`test_a_mu_the_checkpoint_cannot_take_is_reported_as_ignored`).
- The gallery is the running checkpoint's: `GET /api/gallery`, `/api/runs`,
  `GET`/`DELETE /api/gallery/unstarred`, `POST /api/rerender` and
  `POST /api/swap-tags[/preview]`, `POST /api/refine/batch[/preview]` and
  `POST /api/more-seeds[/preview]` without `ids` scope to
  `current_model_id(settings)` (`AppServices.model_scope`, an exact predicate, never the
  `model:` LIKE) unless `all_models=true`; `GET /api/gallery?model=<id>` is the
  same exact scope for any id, in place of both, and `model:=<id>` in the
  search (a leading `=` on any text key) its exact form; the page's All models tick is the
  one switch, unpersisted like Starred, and `GalleryPage.other_models` is
  what the same filter matches elsewhere. Delete unstarred takes the scope
  because its count must be the grid's. Re-render and Delete unstarred,
  after the search, Starred and All models in a row the free space keeps
  to the right, come and go through `showInHead`: while the pointer rests
  on the head's controls (`headPointer`, from the head's pointerover and
  leave, its footer row not counted; never `:hover`) one that goes keeps
  its box, hidden (`.kept`), and one that comes waits, both put right as
  the pointer leaves (`releaseHead`), so nothing before them slides under
  it, and All models' " (+N)" (`showOtherModels`, `headOthers`) waits with
  them; the bin's way in and out (`syncBinControls`, `now`) is at once.
  What is written there meanwhile does not break the row: on a head whose
  controls are one row, the search gives way (`.held`, its basis and floor
  gone) to the gallery's count growing, and Re-render's label, its
  confirmation, is written at once in the width its box had (inline, an
  ellipsis inside), both put right by `releaseHead`.
  Re-render's refusal of another
  checkpoint's images (`api_generate.foreign_refusal`) advises unticking All
  models; it offers `model:<the whole running id>` only where that id is
  contained in none of the foreign ids selected (`model:Anima` is
  Animagine too), and its verb agrees with the count. For picked cards,
  which no filter reaches, it names them to unpick ("Unpick #4 and #9", six
  by number, then "N more") - All models does not scope `ids`, so it is
  refused with or without the tick.
- Re-render takes the picked cards (`ids` on `RerenderRequest`, through
  `submission.resolve_selection`) where there are any, else the filter, by
  `galleryActionTarget` as the footer's actions do (`rerenderTarget`): it
  shows whenever there are picks, outside the bin, labelled "Re-render N
  selected at WxH" with a title naming them, and on the filter "Re-render N
  at WxH" with `RERENDER_FILTER_TITLE`. A picked id gone since (binned or
  deleted) is left out and answered in `RerenderAccepted.deleted` (`{id,
  detail}`, `submission.gone_detail`'s words); `startJob` hands
  `onAccepted` the whole answer, and `sayRerenderGone` says them in
  `#gallery-error` ("Re-rendering the rest: #7 is in the recycle bin.")
  and drops them through `forgetImages` (binned or not by the detail).
  Every pick gone is 404, naming each. A change of the picks keeps the
  head's rows (`syncHeadForPicks`, `onSelectionChange`): at 1400 Re-render
  and Delete unstarred side by side take two rows where either alone takes
  one, so after any change of the picks the head's height is measured and,
  where it changed, Delete unstarred gives Re-render its place, or comes
  back where that keeps it (`unstarredGivesWay`, false at a first pick, read
  by `syncDeleteUnstarred` only while something is picked, reset when
  nothing is; where neither way keeps the height it stays as it was).
  Emptying the picks is measured too: where Delete unstarred had given way
  and bringing it back would change the head's height (a filter typed
  meanwhile, whose own Re-render takes the row beside it) and the pointer
  is on the gallery (`libraryPointer`, #library's pointerover and leave,
  never `:hover`), it stays given way (`unstarredHeld`, the `viewFilter()`
  it was held on) until the pointer leaves #library
  (`libraryPointerLeft`) or the filter changes. A filter typed, a tick or
  a job landing lays the head out as before. The Runs picker asks
  `/api/runs?limit=200` and groups the result: "Grids and batches" (images > 1
  or jobs > 1, labelled with count, distinct, "N prompts" and what varied -
  `runAxes` over `RunSummary.axes` (each setting's count of distinct values
  where above one, from `gallery_reports._RUN_AXES`), `refine` and `strengths`:
  "steps×CFG 3×2", "steps ×8 · 5 seeds", "strength 0.05-0.15 of #37",
  "comparison · 3 versions"), then at most
  eight "Single images" by time to the second and seed. It names the run the
  search box shows (`syncRunList`, from `resetGallery` and `loadRuns`; its
  "Runs…" option is hidden and disabled), and picking one never resets it.
  A pointer pick, or Alt+Down then Enter, opens the run at once
  (`runListChange`). A change made by a browsing key on the closed picker
  is only noted: arrows, Home, End, Page Up/Down, or a printable type-ahead
  key, none with Alt, Ctrl or Meta (`runListKey` marks only the change that
  key's own default makes). The noted run (`runPending`, which
  `syncRunList` keeps through a rebuilt list and retitles from) opens on
  Enter or on the picker's blur (`commitRunList`); Escape puts back the run
  shown. Closed, it is `--runs-w` wide (190 px; `clamp(240px, 14vw, 360px)`
  from 1800 px), cut with an ellipsis, and `syncRunList` puts the picked
  run's whole label in its title ("Recent runs" otherwise). A label's
  prompt is `promptGist`'s, one line with an XML prompt's tags dropped,
  before `clip`.
- Search terms are `key:value` predicates plus FTS text; `date:` is a local
  day, `mu:none` is `IS NULL`, `size:1024x768` (or `×`) is the width and the
  height exactly, `id:` is the gallery id the page shows as #N
  (a range is a span), and `search.strict_scope` is the reading for
  `clear --where`, `keep_rates` and the gallery's Delete unstarred
  (`GET`/`DELETE /api/gallery/unstarred`): `clear --where <box>
  --keep-favorites` over HTTP, the search box strict or empty, gated on the
  count typed back and pinned to it as `expected`, which `delete_matching`
  checks under the write lock and answers 409 on a mismatch ("Nothing was
  moved; look again.") - adding that a
  job is still running and may add more where `jobs.busy()`. It moves the
  rows to the recycle bin (`to_bin=True`, files untouched; the answer's
  `binned`, with `deleted` kept as an alias and `files` 0) and counts live
  rows only (`live_only=True`), so a Restore undoes it; the dialog says
  "They move to the recycle bin; Empty recycle bin removes them from disk."
  and the outcome "Moved N images to the recycle bin, keeping K starred"
  (and the sources it keeps). The CLI's `--where` keeps
  the defaults: every row, the bin's included, deleted for good, and a
  binned star is not kept. Outside the sheet, the page first brings the grid
  up to what the sweep reaches with `prependFinished(0, { now: true })`
  (the same filters, without waiting for the pointer to leave the gallery's
  column), then asks the preview with `newer_than`, the grid's head before
  that. The preview's `newer` counts what the delete takes above that id,
  by the same `_UNKEPT` predicate in the same scope, and becomes the
  dialog's "N of them were made since this page last loaded; they are at
  the top of the grid now" (`freshLine`; "the newest are" where the reload
  from the top left more pages); the page never counts these itself,
  because it cannot see which ones a starred image keeps. The dialog's
  running-job line is the preview's `job_active` (`jobs.busy()`), never the
  page's `busy`, which missed another tab's job. A scope with no image left
  at all says "Nothing to delete: no image is left <scope>.", not that every
  image is starred, and where a reload hides the focused button - nothing
  left, a sweep that took every image, a 409, a failed preview - focus goes
  to the search box (`keepUnstarredFocus`), never `<body>`.
  `keep_favorites` keeps the live stars and what a live starred
  row was made from - its `refine` source and its `reference` ids - through
  one predicate (`gallery_delete._UNKEPT`) that `delete_matching`, `clear_history`
  and the counts share; `matching_preview` returns `sources_kept`, and the
  route's `unstarred` and both CLI counts are `records - favorites -
  sources_kept` (`favorites` the live stars), so the typed count is the
  delete's. A 409 reloads the
  gallery and the Runs picker. `forgetImages(ids, { binned })` is the one
  place the page drops deleted or binned ids - every list, the sheet, the
  panel's references and a
  Reuse's Undo snapshot (it answers the sentence the caller's note adds) -
  and marks at once the passes of what it took (records, card titles,
  every Reuse, the lightbox's facts and `lbSources`); it keeps every id it
  has seen deleted (`deletedIds`) and, apart, every id it has seen binned
  (`binnedIds`), and the stream marks a pass arriving
  after its source went (`markGoneSource`, in `show`: `refine_missing` for
  both sets, `refine_binned` too for a binned source). Its notes, the
  viewer's placeholder ("…moved to the recycle bin") and the
  reference-dropped sentence take the word. `unforgetImages(ids)`, its
  sibling, is the Restore: the ids leave both sets, the passes of a
  restored source are unmarked and retitled (`syncReuseButtons`,
  `viewerSourcesGone`, `sheetSources`/`lbSources` reset), and a tomb whose
  record the page held when it binned it (`binnedRecs`, captured before
  `forgetImages`) is replaced by a fresh `card(rec)` in its own box, the
  record spliced back into `galleryRecords` by id, the total and the head
  resynced; nothing scrolls. References are never re-added - only
  `setReferences` writes them. A single card or viewer
  tile keeps its slot as a tombstone (`tombstone`, "Deleted · #N", or "In
  the recycle bin · #N" for a binned one, which keeps `data-tomb-id`), which
  drops the card's fav, ref and refined marks and keeps its `data-made`;
  focus and the keyboard's current card move to the next live image. A
  viewer whose last image went with no job running gets its placeholder back
  in place of its tombstones, held at the strip's height ("#N was deleted.
  Generated images appear here."), and `appendResult` counts live tiles, not
  slots. A lightbox delete of the last loaded image pages the next one in
  (`pageInBehind`), never stepping back; one that takes the lightbox's only
  image closes it with focus where the walk came from
  (`closeLightboxAfterDelete`: the viewer's held line or tombstone, the
  grid's current card, else the search box), never `<body>`. The per-image
  delete (⌦, the result's Delete, the lightbox's Delete and its Del key)
  never confirms: it moves the image to the recycle bin
  (`POST /api/images/bin`), and the undo is the note it leaves - "Moved #12
  to the recycle bin." and a `ghost tiny` Restore (`noteBinned`, in
  `#gallery-error` as `noteOutsideView` writes, a run of bins gathered into
  one note, "Moved #12, #13 to the recycle bin.", until the next unrelated
  message; in the lightbox through `noteLightboxDelete`, whose
  `pointerdown`/`keydown` clearing skips its own Restore,
  `[data-bin-restore]`, and a press of Delete itself, so a run of presses
  gathers one note as Del does; its Restore says "Restored #N." and puts the
  record back only into a list it came from - the grid's records while the
  grid is still the query it was binned from, the viewer's strip, #recent's
  - with `lbIndex` moved so the image in view stays). The answer's `missing`
  reads "#N was already deleted", its `already` "#N was already in the
  recycle bin". A Restore answered `missing` (emptied elsewhere) forgets
  those ids for good (`forgetImages`), and one answered `not_binned`
  (restored elsewhere) unforgets them, so no pass keeps saying "(in the
  recycle bin)". A delete for good - Empty, Delete permanently - rewrites the
  "In the recycle bin" tombstones and the viewer's held line of those ids to
  "Deleted". The lightbox's Del ignores a held key's repeats (`!e.repeat`),
  and `#lb-delete` a held Enter's or Space's, so holding either bins one
  image, not a run. The questions - Delete unstarred's count,
  Empty recycle bin and Delete permanently - are the page's own
  `<dialog id="confirm-dialog">` (`openAsk`, its `okLabel` defaulting to
  "Delete", placed beside what was pressed
  by `placeAsk`, inside `documentElement`'s clientWidth/clientHeight -
  `innerWidth` counts the scrollbar), never `window.confirm` or `window.prompt`: Cancel focused,
  or the count's box; Escape and a press outside are Cancel; while it is up
  the document's keys do nothing (its Escape closed the lightbox under it).
  The smokes stand in for it by assigning `askConfirm`/`askCount` and put
  back `confirmDialog`/`countDialog`; `smoke-delete.json` drives the dialog
  itself. A delete under way ignores a second
  press of the same image (`deleting`); `gallery_delete.bin_images` looks up and
  writes under `storage._write_lock`, so a second bin answers `already`, and
  `gallery_delete.delete_image` does the same, so a second permanent delete is
  "unknown image".
- The right-click menu on an image (`context-menu.js`,
  `styles/context-menu.css`; `smoke-context-menu.json`): a live gallery
  card, a viewer result's tile and the lightbox's stage open it; the
  recycle bin's cards, tombstones and everything else keep the browser's
  menu, as do Shift with the right button and every text field. The Menu
  key or Shift+F10 opens it on the image focus is in (in the lightbox, from
  any control but a field), on its first item. Its items press the image's
  own buttons (`buttonItem`): the hold and the title are the button's as
  they are now, and a held item, chosen, says why as the button does - never
  a copy of a button's logic. Where the place has no such button (Outpaint…
  and ❐ on a card or a result, ❐ in the lightbox) the item is held by that
  button's sync function on a scratch button (`syncedItem`) and runs the
  same function. Beyond the buttons: Use seed, More seeds of this batch…
  (the lightbox's `#lb-more-seeds` there, elsewhere `syncedItem` over
  `syncMoreSeedsFor` running `moreSeedsOfImage`), Show run as sheet
  (`showRecordRun`, the card's ⊞), Copy prompt, Copy seed, Copy image (the
  full PNG through `ClipboardItem`, the fetch handed over as a promise) and
  Open PNG in new tab; the copy items are held where the page is no secure
  context. A picked card, in select mode or out of it, gets the picks' items instead (the
  Picked menu's three, Swap tags…, Refine…, More seeds…). One `ul.menu`,
  moved into the lightbox while that is up (a modal dialog leaves the page
  inert), fixed at the pointer and turned up or left at the window's edge.
  A press outside, a scroll, a resize or the window going closes it; a
  left press on the lightbox's stage that only closed it is the menu's
  (stopped in capture), so the lightbox stays open. Escape and Tab close it
  with focus back where it was, and its keys go no further. `cardNote` and
  `refuseInCard` take the lightbox's stage: the note goes in the lightbox's
  line.
- The recycle bin's view (`bin.js`, `styles/bin.css`; `smoke-bin.json`).
  `#btn-bin` ("Bin" and a `.count`, `aria-pressed`) sits in the footer row
  between More seeds… and Bundles…; its title, from
  `GET /api/gallery/bin/summary` (`syncBin`, after every bin, restore and
  Empty), reads "Recycle bin: N images (X MB). Restore them, or empty it to
  delete from disk". Its text stays short so the row does not wrap from
  600 px up; at 520 px and under (a phone) the footer wraps
  (`flex-wrap`, gallery.css) rather than run the page past the screen.
  `enterBin()` sets `binMode`, sets the gallery's selection aside
  (`stashSelection`; `leaveBin()` puts it back, less what was binned or
  deleted meanwhile - and less any its view listed, binned by another tab
  unheard of - and drops the bin's picks) and `resetGallery()`s;
  `loadGallery` then asks `/api/gallery/bin` (`galleryEndpoint()`), every
  checkpoint's binned images newest id first under the search box (Starred
  does not apply), the head reads "Recycle bin" (`#gallery-title`) with its
  `#count` "(N)", and an empty one "The
  recycle bin is empty.". In bin mode Delete unstarred is hidden
  (`syncDeleteUnstarred` adds `!binMode`) and **Empty recycle bin**
  (`#btn-empty-bin`, `.danger`) takes its slot, aria-disabled "The recycle
  bin is empty" at 0; Sheet, Starred, Runs, Gallery beside, Swap tags…,
  Refine… and More seeds… are held aria-disabled "Not in the recycle bin —
  leave it with Bin", never `disabled`, and Re-render is hidden, since its
  count would be the bin's; `galleryActionTarget` refuses; the Picked menu's
  button is held with the same reason (`syncPickedButton`, from
  `syncBinControls`), the bin's picks being Restore picked's;
  Select stays, with **Restore picked** before Empty, held while nothing is
  picked. A bin card's tools are ↶ Restore (`.restore`) and ⌦ Delete
  permanently (`.del`, confirmed, last); its picture opens the lightbox; no
  ⤢ (Refine everywhere else), star, ❐, Reuse or ⊞. Restore
  says "Restored #N." (its other clauses "… were not in the recycle bin."
  and "… already deleted.", each naming up to six ids and counting past
  that, as `noteBinned` does) and tombstones the card "Restored · #N" (out
  of the bin's records, not into
  `deletedIds`); Delete permanently (`DELETE /api/images/{id}`) tombstones
  it "Deleted · #N" through `forgetImages`. In the lightbox `#lb-delete`
  reads "⌦ Delete permanently" (title "Delete permanently (Del)"),
  confirmed and placed by `placeAsk`, `#lb-restore` ("↶ Restore", no key)
  is shown only there, and `lbKeysLine` says "Del delete permanently" (else
  "Del bin"); Del never bins a binned image again. `emptyBin()` asks the
  summary, then `askConfirm` with "Empty bin" as the OK label: "Permanently
  delete all N images in the recycle bin (X MB)?", then "That is every one
  of them, not only those this search shows." under a search, "K of them are the source
  or a reference of an image outside the bin; those can't be made again."
  for `sources_in_use`, "A job is running; a cell that uses one of them will
  fail." for `job_active`, and "This cannot be undone."; it sends
  `DELETE /api/gallery/bin?expected=N&pin=P` and shows a 409's detail, else
  `forgetImages`, `syncBin()` and `loadRuns()`. `leaveBin()` runs from Show
  all (`showAllImages`), the Models picker's See all, `showSheetScope` (a
  run opened from an image or a job: `showRun`, a run fact, ⊞),
  `showGallerySearch` (an image's search links) and a second press of Bin;
  the search box narrows the bin rather than leaving it, and the Runs
  picker is held; `prependFinished` returns
  at once in bin mode and `viewFilter()` counts it. Bin mode is not
  persisted. Every other gallery query keeps the live listing, which the
  server gives by default. A record from `GET /api/images/{id}` (and
  `/thumb`) of a binned image is a 404 "unknown image (#N is in the
  recycle bin)" unless `?binned=true`, so every reader of "unknown image"
  still reads it as gone. The bundles dialog's preview line adds "(N in the
  recycle bin)", and after an export-clear `syncBin()` follows
  `forgetImages`.
- The gallery pages by id, never by offset,
  and pages itself as "Load more" scrolls near the viewport, or near the
  foot of the gallery's own column with Gallery beside (`moreWatcher`, made
  by `armMore` with its root where the gallery scrolls - the window or
  `#library-scroll` - and made again when that changes; re-armed by
  `watchMore` after each page that lands, never after one that failed - the
  button is the retry - and `moreArmed` keeps a trigger made again
  meanwhile from fetching).

## Tests

`tests/test_instruction_edit.py` checks the standalone frozen Kontext recipe,
strict request/stored source types, full original-RGB provenance, unresolved
source refusal and exact-aspect preprocessing. Installed-pipeline assumptions
are read from source without importing Diffusers. These are pure CPU/source
checks; they do not qualify persistence, HTTP admission, bundles, a browser,
model loading, rendering or the card.

`tests/test_instruction_edit_identity.py`, `test_instruction_edit_storage.py`,
`test_instruction_edit_bundles.py`, `test_instruction_edit_reindex.py`,
`test_instruction_edit_admission.py` and `test_instruction_edit_lifetime.py`
check strict copied-request admission, bound source keys/legacy run ids,
SQL/PNG migration/carry/conflicts, v2/v1 bundle compatibility, full-RGB remapping,
unresolved recovery and cleanup/source availability on scratch galleries.
`test_instruction_edit_loader.py` checks known-editor and unreadable custom-id
class refusal before weights. `test_instruction_edit_ui.py` uses Node VM and
an inert DOM. It checks that Reuse on a resolved edit routes to `reuseEdit`
and leaves the composer untouched. It checks that unresolved, deleted and
binned sources are held with words, and it covers the facts and viewer
lineage branches.

Three files cover stage 2:

- `test_instruction_edit_gate.py` checks the gate. The base and real
  backends answer False even with every `AURADREAM_*` set, and a source scan
  holds every `instruction_edit_qualified` under `backend/` to a literal
  False. `qualified=1` stays closed. It also checks the wording constants,
  `_merged_defaults` and the config pin: synthetic configs fail
  `require_pinned`, pass with a monkeypatched pin, and fail again with one
  path changed.
- `test_instruction_edit_call.py` drives `_edit_images` against a
  Kontext-shaped stub whose call reports the installed signature. It checks
  the exact kwargs, the one seeded generator, grad and inference mode off,
  progress, the preview, cancel and failure cleanup (hooks freed and the
  source closed), and the refusals. It also checks that a plain backend and
  editor loads still refuse.
- `test_instruction_edit_queue.py` checks the route's admission order and
  refusals, a saved edit row and PNG, the per-cell recheck, cancel,
  `planned_keys`, and the held producers on both gate states.

None of these enables the real server's editor or qualifies a model or the card.

`tests/test_instruction_edit_preflight.py` checks the supplied-config refusal
boundary with synthetic JSON and the installed complete PIL-to-normalized-CPU-
tensor preprocessing path using source-extracted methods in the fresh
`tests/fixtures/instruction_edit_processor_cpu.py` child. The child forbids
pytest/conftest, model and application imports, poisons CUDA initialization
and verifies no initialization attempt before or after preprocessing. The six
pinned gated configs were fetched at the revision and qualified on 2026-10-27,
outside the repository and not committed; their hashes are
`QUALIFIED_CONFIG_HASHES`. A passing preflight never permits execution.

`tests/test_instruction_edit_loader.py` checks strict custom/local class
inspection, unreadable/malformed/editor aliases, case-insensitive named-editor
refusal and unchanged catalogue/Nunchaku paths. Synthetic backend loads prove
that refused aliases never reach weight overrides, `from_pretrained` or
placement; no Hub/network or actual weights are used.

`tests/test_backup_credentials.py` and `tests/test_backup_credential_integration.py`
check WinCred ABI declarations through API stubs, exact opaque references, bounds,
redaction/free/cleanup, inactive pending provisioning, no uncertain retry, hidden
input, independent-password recovery and worker-only lookup after due/nonempty
admission. A fake provider exports/restores an actual owned encrypted gallery
without another verifying pass. `tests/conftest.py` blocks the real credential
factory for every test; only explicit fake overrides may pass. No Windows vault
entry is read/written/listed/deleted and no OS credential qualification follows.

`smoke-credential.json` uses a fresh FLUX.1 Krea [dev] fake on 7929. Backup API
responses/saves are intercepted fixtures: explicit protection selection, absent/
pending/ready credential profiles, fresh encrypted/recovery acknowledgement,
generation/dirty-draft guards, plaintext transition, disabling and layout. It
enters no password and reads/writes no real Windows vault entry. OS-API evidence
comes from stub tests, not this browser fixture.

`tests/test_execution.py` checks frozen/strict named recipes, internal-copy
validation, canonical empty/key spelling, cold distribution facts and the
first regional candidate matrix. A matching candidate remains unavailable.
HTTP, queue and direct backend/fake checks refuse nonempty recipes before
mask stamping, retained work or model work; explicit run ids do not bypass
queue admission. Empty execution retains legacy keys/PNG metadata and normalizes only
through `applied()`, without revalidating a stamped outpaint. These are CPU
contract checks, not compiler or GPU qualification.

`tests/test_execution_resolver.py` checks composer inheritance versus complete
replay, explicit Off/channel conflicts, resolution before run/planned identities,
legacy retained runs, source PNG/CLI intent and stamped-mask-safe narrowing.
Direct submission subtypes must consume or refuse cache intent before device,
source, queue or backend effects. These synthetic checks grant no availability.

`tests/test_regional_diagnostics.py` checks immutable per-load observations,
single first-call tokens, terminal freezing, retirement, competing callers,
stale updates, model collection and import-cold readers. Pending means no forward
attempt; returned is not a warm-kernel, cache-hit or qualification result.

`tests/test_api_execution_diagnostics.py` checks fixed diagnostic serialization,
default/unpublished/retired absence, a fresh read after awaited descriptions,
catalogue selection changes and same-checkpoint reloads. Draft catalogue options
never label a per-model detail with the active load's observation; routes start
no model load.

`tests/test_job_execution_diagnostics.py` checks active first-call observations,
terminal snapshot freezing, reader retirement/replacement and weight-free
lifetimes. Diagnostic observations stay outside retained job definitions and
reproduction identities; optional readers never change generation outcomes.

`tests/test_execution_diagnostics_ui.py` exercises the page's fixed scalar
captions and load/job lifecycle using Node VM with inert DOM/EventSource stubs.
It checks progress counters, terminal observations and stale-detail clearing.
These source contracts do not boot a page or qualify browser layout/accessibility;
the actual browser denial and incomplete UI validation remain in the tooling plan.

`tests/test_execution_candidates.py` checks the initial cuDNN/FBC candidate
matrices, combinations, copied-spec validation, external override refusals and
FBC guidance/cache exclusions. An isolated subprocess forbids framework/config
imports and metadata/environment reads by the validators, proves the environment
unchanged and confirms matching recipes remain unavailable to execute.

`tests/test_execution_storage.py`, `tests/test_execution_bundles.py` and
`tests/test_execution_replay.py` cover the nullable recipe migration and
canonical codecs, legacy hash fixtures including mixed-size/crop/comparison
runs, distinct recipe keys, stored/PNG/bundle/reindex carry and unavailable
rerun/PNG/CLI/retained recovery. Known recorded recipes confer no permission
to execute; invalid or contradictory metadata is refused rather than dropped.
`tests/test_regional_compile.py` checks discovery, exact arguments, failed
candidate disposal, precompiled root/region refusal and idempotent reference
cleanup with stubs. Installed `ConfigModule.patch` over synthetic flags
checks nested/thread restoration. Real Flux/Dynamo imports, compiler calls,
fallback and device behavior remain unexecuted.
`tests/test_pipeline_ownership.py` exercises the actual eager loader over strict
synthetic runtime modules, unpublished override/pipeline failure at context entry
and exit, publication order, registration transfer and shared unload behavior.
Weak-reference assertions outside exception handling prove application containers
drop ownership before release; arbitrary library traceback references and actual
GPU reclamation are separate. An isolated subprocess forbids framework imports
while checking private loaded-recipe refusal before work.
`tests/test_regional_backend.py` exercises the actual private preparation and
invocation paths with synthetic runtime modules, pipeline and regions. It checks
cold preflight before weights/imports, registration/publication failure ownership,
strict-policy restoration for success/error/cancellation, fixed handle/facts and
unchanged eager direct/refine semantics. Public nonempty execution stays refused;
no real compiler, model, device or toolchain qualification follows.
`tests/test_compiler_cache.py` checks pure runtime/gallery namespacing, the owned
startup window, environment and path ownership, handle-owned scratch probes,
partial/uncertain initialization and later binding drift. It imports no real
compiler or model; directory usability is not toolchain or performance evidence.
`tests/test_model_switch_execution.py` checks typed switch recipes, direct/copied
internal bypasses and fixed refusal before Hub, save, unload, selection or preload
effects. Empty/legacy Eager switches and stale revision guards remain covered.
`tests/test_api_model_readiness.py` checks truthful ready/unloaded presentation,
read-only GET, explicit and same-checkpoint reload without a preceding poll,
concurrent load coalescing, held loading and preserved error/retry behavior with
fake pipelines. This does not establish public FBC reachability or card behavior.
`tests/test_execution_selection_admission.py` checks pure configuration limits,
invalid/selected startup intent, omitted versus explicit Eager requests and closed
queue/backend/HTTP admission before side effects. Portable normalization/codecs
remain independent of current selection. CLI cases additionally refuse unavailable
flags/environment before source or project reads and preserve explicit Eager intent.
`tests/test_model_selection_performance.py` and `tests/test_loader_startup.py`
check unavailable configuration saves, structural refusals, omission/clear,
execution-only revisions, exact invalid-file rollback, frozen descriptor snapshots,
v1/v2 startup compatibility/precedence and fail-closed invalid intent. None grants
runtime permission; saved preload cannot bypass the load guard.

`tests/test_browser_diagnostics.py` uses a synthetic no-pid child stream to verify
opt-in failure-only stderr diagnostics and a bounded tail. It opens no browser,
socket or native cleanup process. Real browser/profile checks are separate.

`smoke-execution-selector.json` uses a fresh FLUX.1 Krea [dev] fake on 7930.
It checks inspectable configuration, honest unavailable status and held Load/render
actions, structural conflicts, saved and unsaved recipes across polling/reload,
explicit Eager recovery and per-render cache persistence/Reuse/Reset/Undo. Nonempty
record Reuse stays refused. Only an ordinary Eager fixture may generate; no compiler,
model download, non-Eager execution or performance/quality acceptance is established.
`tests/test_attention_selection.py` checks coverage, transactional failures,
shared ownership, later replacements and idempotent restoration with stubs.
It also executes source-extracted installed public context/setter bodies.
Its focused run uses `--noconftest`, with no actual Diffusers import, tensor,
forward or device probe; real runtime/card qualification remains separate.
`tests/test_attention_environment.py` retains the cold warning/source checks:
exact startup snapshots, fixed redaction, CLI logging, unchanged selection and
status reads that do not reread environment or import a framework.
`tests/test_attention_policy.py` checks the experimental Native baseline using
synthetic modules: startup/current/frozen observations, exact registry function,
redaction, drift refusal and policy-independent restoration/cleanup. It proves
no Native/cuDNN kernel, device fit, pixels, speed or repeatability.
`tests/test_attention_backend.py` checks actual private preparation/publication/
unload/invocation with synthetic runtime modules, exact ownership and closed
public guards. It covers primary error/cancellation plus post-forward ownership
failure, cleanup outside locks, terminal weak exclusions, GC/identity-reuse
safety and fresh-root recovery. No real dispatcher or card behavior is established.
`tests/test_fbc_backend.py` checks private preparation, one-forward ownership and
deferred disposal with synthetic dependencies: hook-free load, effective Native,
partial cleanup/foreign hooks, primary errors/cancel, scope/lock order, pending
not-ready state, stale callbacks and public refusals. No real model/GPU runs.
`tests/test_fbc_request.py` checks request-driven On/Off/On over one synthetic
Eager load, fresh ownership, explicit Off versus legacy private omission, loader
equality, unsupported request/call refusals and deferred unsafe disposal. It
preserves generator/callback/sigma scope behavior and every public admission guard;
it does not qualify FBC arithmetic, model/device behavior, quality or performance.
File-based readers use `metadata.require_header_metadata` to refuse app-owned
text after PNG pixel data, which Pillow's header-only `info` would miss.
The scan skips payloads, reads at most 80 bytes per text keyword and restores
the stream position; it adds no pixel decode. Browser-provided text chunks
already include the complete parsed record and need no file-position scan.

`tests/test_backup_policy.py` checks pure default-off backup decisions with
explicit age, retention and protection. It covers UTC age boundaries and
repeated-hour ordering, unavailable or untrusted copies, exact scope,
conflicting identities, new-success retention and download leases. These
checks perform no archive I/O and confer no deletion or activation authority.
`tests/test_backup_preferences.py` checks bounded strict schema reads,
database isolation, exact integer-microsecond durations, fixed failure
warnings and atomic saves preserving prior bytes on write/fsync/replace
failure. The shared 10,000-copy validation ceiling is an engineering limit,
not a selected retention policy. Neither module is connected to startup.
`tests/test_backup_activation_settings.py` and `tests/test_api_automatic_backup.py`
cover explicit v2 activation, inactive legacy
policies, stable gallery/destination namespaces and scratch startup/configuration
integration. Status stays available during maintenance, configuration reserves
before writing, and preload waits for the admitted attempt. No OS credentials or
retention deletion are exercised.
`tests/test_backup_receipts.py` and `tests/test_bundles_managed.py` check
strict bounded ownership journals, portable names, exact outer-byte proofs,
plain/encrypted scratch restore and single verifying reads. They inject
publication/journal failures and partial replacements before verification
and after claim, preserving unowned files and prior successful copies.
No real gallery, credential store, startup or pruning is exercised.
`tests/test_bundle_leases.py` checks canonical path sharing, reader/exclusive
exclusion and complete FileResponse lifetime, including range/HEAD/error/
cancel/background behavior and disabled pathsend without scope mutation.

`tests/test_bundle_mode.py` and `tests/test_api_bundle_mode.py` cover explicit
destination-mode installation and discovery, default non-writing downloads,
installed-mode failures and ASGI response admission across local cutover. Native
checks use owned Windows scratch processes, never unknown server inventory.
`tools/install_bundle_mode.py` is the explicit offline installation entry point;
other-server quiescence is operator-attested, not machine-verified. Neither the
marker nor the tool enables retention or turns a v2 saved N into deletion consent.

`tests/test_bundle_consumers_leases.py` covers public encrypted-reader/header
lifetime, saved-import reservation order and complete merge lifetime, listings,
receipt hashes, wrong scopes and separate private upload/export partials. Owned
Windows child processes exercise native exclusion across databases; injected
uncertain file closes preserve primary failures and the original reader claim
even after pathname retargeting. These are scratch tests, not retention acceptance.

`tests/test_bundle_delete.py` exercises the Windows same-handle
deletion mechanism on owned scratch files, required installed-mode exclusion,
receipt scope/identity/raw-byte refusals and single-use mutation. Injected
mark/close failures check original-key quarantine and honest post-mark outcomes.
No test operates on a real gallery destination.

`tests/test_backup_retention.py` and `tests/test_backup_retention_journal.py`
exercise v3 consent, unchanged v2 behavior, pending/retired transitions, current-N
protection, leased excess, retired-operation retry and caps. Owned Windows scratch
lifecycle cases actually export, mark/close and retire copies with unchanged gallery
bytes. Injected failures check pending/unknown state, complete-close restoration,
no pruning after failed/empty/skipped/unmanaged exports and preserved success/time.

`smoke-execution-reuse.json` uses a fresh FLUX.1 Krea [dev] fake on 7926.
Browser fixtures add execution recipes to a synthetic eager record, checking
held Reuse buttons, status refresh, lightbox reason without dismissal,
direct panel/pass/inpaint/outpaint refusal before source reads and unchanged
composer/scroll. Legacy empty recipes retain Reuse and Undo. It runs no
non-default recipe and establishes no compiler or GPU behavior.

`smoke-automatic-backup.json` uses a fresh FLUX.1 Krea [dev] fake on 7927.
It checks explicit plaintext consent and settings validation, stale revisions,
dirty edits and reload, fixed startup-outcome presentation, manual-field
preservation, disabling and narrow layout. It configures scratch settings only:
no server restart, automatic export, real gallery or credential store is used.

`smoke-retention.json` uses a fresh FLUX.1 Krea [dev] fake on 7928. All backup
API responses/saves are intercepted synthetic data: inert old N, missing/stale
mode, fresh separate deletion consent, dirty drafts, marked/unknown wording,
pending while off and disabling without destination access. Desktop/narrow
screenshots check containment and stationary Save. No policy, shared-mode install,
restart, export or deletion is executed; native lifecycle evidence is CPU scratch.

`tests/test_classic_scripts.py` invokes `tools/check_classic_scripts.py`
and its Node helper to parse every local classic script in HTML order,
including `app.js`, and instantiate declarations without executing source
bodies. It covers syntax, cross-file lexical collisions and permitted
sharing, strict/sloppy block functions, unsupported script kinds, missing
files and missing Node. Node is required locally and verified by CPU CI;
this guard does not replace the independent browser smokes.

`tests/test_browser_capabilities.py` captures the actual harness/helper module
bodies before instrumentation and loads them as data URLs, so Node's own module
loader does not trip the application file-read guards. It runs the real harness
with absent or noncallable `fetch`/`WebSocket` globals. Fixed exit-2 diagnostics occur
before browser lookup, script reads, output/profile creation or any network
call; a callable fixture continues to the existing browser-discovery path.
Negative controls keep input/module reads, output writes, spawn and network APIs
forbidden after source capture.

`tests/test_fbc_runtime.py` exercises installed FBC hooks and arithmetic on tiny
CPU blocks in an isolated worker. Only the unrelated optional bitsandbytes backend
is modeled as absent; a discoverable poison package verifies that boundary.
Compiler, device and network counters stay zero across imports and all arithmetic,
context, freshness, cancellation and cleanup cases. This is no real-model or card
qualification.

`smoke-control-sizing.json` uses a fresh FLUX.1 Krea [dev] fake on 7923.
It measures prompt/footer targets and primary emphasis at 1400x900, 600x900,
1536x864 and 2048x1040 (the latter two at DPR 1.25), checks actual hit targets,
focus rings, held/selected geometry, fixed-seed/panel widths and a crowded
footer with picks and a nonempty bin. Every cold catalogue caption combination
is checked for fit, with no model load.

`smoke-numeric-inputs.json` uses a fresh FLUX.1 Krea [dev] fake on 7921.
Native typing checks incomplete Steps/CFG/Seed/size/True CFG before a POST,
button and keyboard paths, intentional blanks, queuing, independent More
seeds settings, grid Seeds/Budget and sampling overrides, and comparison
controls that replace unrelated panel values.

`smoke-queue-outcomes.json` uses a fresh FLUX.1 Krea [dev] fake on 7922.
Gated listings reproduce fast queued completion/failure, once-only outcomes
and Finish recovery, ordered rapid receipts, delayed POST/history snapshots,
transient history retries, cancellation/foreign-job exclusions and no
historical replay after reload.

`tests/test_prompt_parts.py` covers the reported mixed example, pure tags
and prose, a prose passage in a majority-tag prompt, known titles and
concrete validation findings, explicit prefixes and grid variables,
punctuation/UTF-16 spans, matching context and grid lookup tails, cap/offline
boundaries, hybrid capability metadata, unchanged raw/cached verdicts,
checkpoint switching, and Check/critic reorders that would split prose.
`smoke-mixed-prompts.json` uses a fresh Kohaku fake on 7920 for two actual
tag chips beside one exact comma-containing prose span, accessible counts,
pure-prose tray hiding and token-limit readout, strict misspellings, cached
edits, a majority-tag prompt, raw whitespace/non-BMP offsets, tag
edit/move/delete/insertion and undo around the prose, a gated suggestion
reply, Kohaku-to-Animagine-and-back reclassification, and matching Refine
warnings. The browser checks use synthetic images and no real model.

```powershell
uv run python tools/run_smoke.py smoke-mixed-prompts
```

`smoke-button-states.json` uses a fresh Animagine fake on 7918 to check
secondary/primary and selected text contrast, held hover and reason-on-press,
focus rings, unchanged state-transition geometry, and footer widths at
600/1400/2560/2048 pixels (the last at DPR 1.25).

`tests/test_more_seeds_intents.py` and `tests/test_api_more_seeds_intents.py`
cover receipt replay, conflicting choices, concurrent tokens and waiters,
cancelled HTTP waiters, bounded retention/expiry, shutdown and partial
multi-job admission. The extended `smoke-more-seeds.json` checks active
Batch x Repeat, delayed/repeated presses, two deliberate extensions, lost
replies through 503 and generic 409 responses, terminal receipts, partial
warnings, stale opening contexts and narrow/normal/wide result reachability.
`smoke-more-seeds-tabs.json` uses a fresh FLUX fake on 7919 with two actual
tabs, separate owned-job/viewer/composer state, simultaneous stale previews,
distinct seed allocations, lost-response replay and queue transitions.

`tests/test_size_crop.py` and `tests/test_crop_anchor_storage.py` cover paired
size parsing, limits, shape identity and legacy hashes, crop pixels and the
GAN path, actual-size Resume/Finish/estimates, persistence and source-loss
cleanup. `smoke-size-crop.json` uses a fresh FLUX.1 Krea [dev] fake on 7917
for sizes, Resume/sheet axes, crop frames, metadata, Reuse and source views.

`tests/test_presets.py`, `test_preset_validation.py` and `test_api_presets.py`
cover strict saved drafts, bounded atomic failure/corruption preservation,
thread/process concurrency, optimistic revisions, request/profile and live
`applied()` parity, grid grammar/budgets, null seeds and disposable random
examples, held execution, model-switch capture races and gallery-independent
CRUD/validation. These CPU/API checks do not qualify card behavior.

`smoke-generation-presets.json` uses a fresh FLUX.1 Krea [dev] fake on 7932.
It covers saving two raw Generate/Grid setups and reloading them, null profile
ownership and explicit zero, explicit Restore, whole-draft Undo and native
Ctrl+Z, cleared conditioning/Then refine and their Undo, Resume off, held
performance, stale item edits and draft/model validation replies, the foreign
checkpoint hold and Models opener, ordinary grid preview/submission, modal
keys/focus, page errors and geometry at 1400x900 and 400x650. It uses scratch
gallery data and synthetic conditioning; no real weights or card run is involved.

`smoke-setup-list-recovery.json` uses a fresh FLUX.1 Krea [dev] fake on
7938 for native saved-setup list recovery. The final regression covers held
collection reads, explicit sole-item revision adoption, view toggles,
validation selection cancellation and close/reopen late replies while keeping
the composer unchanged. The dated setup-list-recovery review distinguishes
the initial diagnostic baseline from actual final acceptance.

`tests/test_snippet_portability.py` checks the cold portable codec, exact
Unicode/multiline/order round trips, strict format/identity fields, duplicate
JSON keys, byte/depth bounds, collection/input digests, normalized-name
conflicts and explicit file-index choices. Pure plans preserve local positions,
require explicit replacement ids/revisions and bound final count/encoded bytes;
`tests/test_api_snippet_portability.py` adds scratch snapshot/atomic import,
whole-collection and target revision refusals, concurrent writers, complete
choices, independent databases, raw-body bounds and malformed local-file
preservation. Write/fsync/replace failures keep the previous sidecar, and
model/queue/vault work is blocked. Browser workflow qualification remains.

`smoke-snippet-portability.json` uses a fresh FLUX.1 Krea [dev] fake on 7934
for portable export/merge. It checks the actual export-response
bytes, file-index conflict choices, stale/uncertain commit recovery, invalid
files, dirty-editor preservation, modal keys/focus and contained stationary
controls at desktop/narrow/short sizes. The 183-step Windows workflow passes;
final screenshot reinspection remains blocked by an artifact-directory read
denial. Exact evidence and limits live in `docs/plan-snippet-portability.md`.

`smoke-library-write-recovery.json` uses a fresh FLUX.1 Krea [dev] fake on
7935. Its 221-step Windows pass exercises actual committed snippet/setup
writes with lost replies,
held retries, fresh-read/explicit-choice recovery, retained drafts, known
validation/stale refusals, New during delayed snippet reads, pending close,
modal keys and standard/narrow/short containment. The dated review owner
`docs/review-2026-10-28-library-recovery.md` records actual acceptance.

`smoke-saved-name-unicode.json` uses a fresh FLUX.1 Krea [dev] fake on 7936
for saved-name parity. Native text insertion creates/edits 80 astral-code-point
names in snippets, setups and saved searches; mixed 81-point names still fit
the UTF-16 capacity but must hold Save without writes. It also checks BMP
bounds, programmatic overlimits, reload/navigation persistence and unchanged
composer/gallery state. Evidence lives in the library-recovery review owner.

`smoke-picked-favorites.json` uses a fresh FLUX.1 Krea [dev] fake on 7937
for confirmed favorite propagation to hidden active/stashed picks. It seeds
owned images before a fresh navigation, excludes viewer/session/lightbox
aliases, and checks storage separately from retained snapshots and the actual
Compare picked star without record rereads. Unstar/already answers, a reply
arriving in Bin, independent bin picks, selection order and composer text are
covered. The dated picked-favorites review records reproduction and acceptance.

`tests/test_snippets.py` covers CRUD, exact text, bounds, malformed-file
preservation, independent locking, stale/concurrent edits and admission
during bundle maintenance. `smoke-snippets.json` uses a fresh FLUX.1 Krea
[dev] fake on 7915 for CRUD, server persistence, Append/Prepend with Undo,
held negative insertion, conflicts, modal keys and the narrow layout. It also
checks that the menu adds no caption height beside ordinary, Tags and Expand
controls at 1400x900 and 400x650, and insertion into an offscreen prompt keeps
page scroll. Native Undo in the browser harness carries the Z key code.

`tests/test_saved_searches.py` and `tests/test_api_saved_searches.py` cover
strict fields and file envelopes, Unicode/name/query limits and existing search
grammar, database isolation, bounded/cold reads, atomic failure preservation,
concurrent edits and optimistic revisions, plus model/maintenance-independent
CRUD. `smoke-saved-searches.json` uses a fresh FLUX.1 Krea [dev] fake on 7933
for the dialog, explicit filter application, saved scope, retained picks,
stale/late replies, modal keys/focus and responsive footer/dialog geometry.

`smoke-queue-controls.json` uses a fresh FLUX.1 Krea [dev] fake on 7914 with
0.3 s steps. It checks authoritative Move order and focus through reload,
delayed-response/new-submission races, a watched queued job's Move control,
one-call Skip and Finish recovery, and held paused/Then refine reasons.

`smoke-random-prompts.json` uses a fresh FLUX.1 Krea [dev] fake on 7916.
It checks random/cross selection, saved mode and Reset, deterministic choices
across numeric axes and per-image batch seeds, preview counts and unseeded
sample wording, saved choices, Resume, and unchanged Cartesian behavior.

`smoke-compact-setup.json` uses a fresh fake on 7925. It checks inactive and
active disclosures, value preservation and actual status-poll non-reopening,
restore/Reuse/Undo and pending LoRA catalogue behavior, numeric-focus reveal,
reference activation/order/capability state, native expansion, scoped `/` and M,
hidden-pick independence, no immediate queue, modal/editing/modifier/IME/repeat
protection and responsive screenshots at 390/600/1400/2048 px.

`smoke-shortcut-dispatch.json` uses a fresh fake on 7924. It checks modal and
menu ownership, exact modifiers, handled/composing/repeated events, native
button repeat, lightbox editing/select/browser navigation, ordinary Generate
and Grid, and delayed More seeds/help Escape precedence.

`smoke-shortcuts.json` uses a fresh fake on 7913. It checks the complete
shortcut list before any images exist, the top-bar and `?` openers, typing,
composition/modifier/repeat guards, help above a lightbox and a Refine draft,
no underlying submission or image action, native Escape, Close and actual
backdrop focus restoration, context ordering with the full inventory, menu-opener
fallbacks, accurate menu hints and a 390 px layout without horizontal overflow.

`uv run --group dev pytest`. They stub the pipeline (`tests/fakes.py`'s
`FakeBackend`, wired by the `fake` and `client` fixtures in `tests/conftest.py`
and driven through `tests/api_helpers.py`; the HTTP tests are `tests/test_api*.py`,
one file per area) and the scheduler (`tests/test_schedule.py`'s `StubPipeline`), so
nothing downloads weights or touches the GPU, and they run against a throwaway
database and outputs directory (`tests/conftest.py`). They run on `config.py`'s
defaults: `tests/conftest.py` drops every `AURADREAM_*` but the gallery
and adapter paths before anything imports `backend`, so a sourced
`wsl/env.sh` changes nothing. They cover the invariants
that are expensive to get wrong: the schedule the scheduler ends up with, seeds
surviving a JSON round trip, the search fallback keeping its filters, and the
HTTP guards. `tests/test_docs_maps.py` holds `CLAUDE.md` to the one
`@AGENTS.md` import; every backend module, `web/*.js` and `web/styles/*.css`
named in the Layout above (the stylesheets in `index.html`'s link order, and
its count of classic scripts); and every `tools/smoke*.json` in `AGENTS.md`'s
port table, on the row whose port its first `navigate` uses, and named
below - so a new file is added to the maps with it.

The recycle bin is covered where each read lives: `tests/test_storage.py`
(the migration, `list_images` live and `binned=True` with total, page and
`others` agreeing through the LIKE fallback, `runs`/`keep_rates` leaving the
bin out and `completed_cells`/`lineage_images` keeping it, a binned star
unkept, `reference_paths` refusing, `refine_binned`), `tests/test_gallery.py`
(a double bin answered once, restore, `empty_bin` with its files and its
`SelectionChanged`, the orphan sweep sparing binned PNGs),
`tests/test_api_gallery.py` (every bin route, the 404s without
`?binned=true`, the favourite's 404, the picked cards' stars answered per
id - a binned one `binned`, its flag kept - Delete unstarred binning, Empty
recycle bin's count and its pin),
`tests/test_api_jobs.py` (the replay and `deleted_count` through a bin and a
restore), the refine, grid, Swap, Mass refine and More seeds files (a binned
source refused, a binned cell not remade until Empty, a picked-then-binned
id reported "in the recycle bin", seeds past a binned one),
`tests/test_bundles_repeat.py` (the state through a round trip, a repeat
import keeping the local state), `tests/test_bundles_merge.py` (an old
bundle live), `tests/test_api_bundles.py` (`binned` in the preview),
`tests/test_clear.py` (`--bin` and its pin, the whole clear
and `--where` reaching the bin, `--where` pinned to its count) and `tests/test_reindex.py` (in place and
`--database` keeping the state, PNG-only bringing it back live).

The bundle tests share helpers from `tests/bundle_helpers.py`.
`tests/test_bundles.py` covers export and cleanup: archive/merge round trips,
preserved favourites and timings, remapped image links, WAL snapshots,
monotonic ids, changed source files (a same-size, same-mtime edit included,
so only the pre-cleanup hash catches it), partial cleanup (a database
replacement refused after files went still reports `deleted_ids`) and a
locked database stopping the clear before any file goes.
`tests/test_bundles_merge.py` covers corrupt or unsafe archives, conflicts,
rollback (including a database replacement refused after every file is
installed), the cross-volume copy and oversized or padded members.
`tests/test_bundles_repeat.py` covers restoring a lost file of a known row,
palette/mode identity and regenerated thumbnails.
`tests/test_bundles_staging.py` covers staging and free space per volume and
the leftover sweep, and `tests/test_gallery_lock.py` process/thread
reservations. `tests/test_bundles_directory.py` covers independent archive
storage, default fallback, containment and link refusals, and the fake
server overriding an inherited production archive path.
`tests/test_api_bundles.py` covers browser
export/download/upload/import, admission during jobs, the card's editor
work, a model switch or load lock and maintenance, another process's
reservation failing the operation readably, origins,
image-stream lifetime, cancellation of the supervising task, progress, and
password export/check/upload/import, a password with no UTF-8 form refused
with one 422 on every route, and an upload disconnected mid-body (driven as
raw ASGI, since TestClient sends whole bodies) failing readably with no staging
left. `tests/test_bundle_crypto.py` covers the
cipher: round trips and seeks, Unicode normalisation, wrong passwords,
flipped, truncated, extended and reordered chunks, a short header or no body,
unbounded parameters, the shipped scrypt parameters (17/8/1, 1 MiB chunks;
`conftest` otherwise writes the cheapest cost) staying within the reader's
bounds, and an unfinished writer. Password bundles, altered ciphertext (in a
later chunk too, after members were staged), a failed encrypted
read-back that clears nothing and one scrypt run per password export are in
`tests/test_bundles_password.py`. Re-encoded existing images on repeat
import are in `tests/test_bundles_repeat.py`, with the speed paths'
guarantees: `bundle_images._raw_digest` equal to `tobytes()` and a repeat
import that decodes and unpacks nothing present but still validates its
rows; `tests/test_bundles_merge.py` holds a declined member that vanished
and a failed parallel metadata rewrite.
`test_bundle_crypto.py` also covers writes of any size and the writer's key
opening only its own header; `test_api_bundles.py` an upload in many batches.

`tests/test_bundles_backup.py` covers the backup and the chosen bundle: a
backup leaving every file and row byte for byte and importing like any
bundle, no count gate but the read-back kept (a failed one publishes
nothing), only the chosen rows, PNGs and thumbnails archived with no other
prompt in the database bytes, `left_out` and `links_outside`, the import's
`unresolved_references`, a binned chosen row carried, a selection never a
clear, a missing file refusing only when chosen, and too little free space
(a chosen bundle's database counted with its log and twice)
refused before anything is written. `tests/test_api_bundles.py` covers the
routes: a backup keeping the gallery and its jobs' images, picked (a binned
and an unknown id left out) and starred (the bin's star not counted; none
starred a 404), the preview's `starred`, every out-of-shape request
refused before any work (a selection with a typed count included), and a
chosen bundle refused as busy before its selection is read.

`smoke-bundles.json` uses a fresh fake on 7897 and checks the browser dialog,
all-model scope, typed confirmation, a bundle name stamped in local time,
preserved favourites and metadata,
saved-bundle and uploaded-file imports, repeated import, damaged ZIP refusal,
password export (mismatch refused, fields cleared, Protected tag), wrong
password refused before upload (the unlock field kept), protected import
(the unlock field cleared), Cancel upload (refused with its reason while idle;
mid-send it aborts, with a stubbed upload fetch), progress counts outside
the live region, a reopened dialog's result labelled as past, narrow layout,
an export replayed to a reloaded page shown as past without forgetting
ids, a missing file's preview (stubbed with `MissingImage`'s 422) whose
inline Show #N finds the row in the gallery and, binned, in the bin, and
Back up: the entire gallery with no typed count and a result that is no
failure, Picked and Starred with their counts, Bundle & clear held for a
selection, the scope moving no control, the gallery's picks used while the
bin is shown with one binned from another tab left out, an import's
missing-source count said as a fact, a gallery read refused mid-backup
(stubbed 409) repaired when the backup ends with the typed count kept, the
holds with nothing picked or
starred, and the layout at 600x900 and 2560x1305.

Its sizing/Bundles review cases also cover explicit password association and
keyboard navigation, the clear safety explanation and enablement, blank/wrong
and leading-zero count entries, a synthetic preview whose record/file/bin
counts differ, unchanged creation/import password lifetimes, and 360/390 px
layouts plus forced colours. Scope changes compare dialog-content positions,
so explicit test scrolling does not masquerade as a moving control.

```powershell
uv run python tools/fake_server.py --port 7897 --data <new-scratch-directory>
node tools/browser.mjs tools/smoke-bundles.json
```

The refine pass is covered the same way: `tests/test_refine.py` (each
profile's `refine_supported` and each sibling's call held to the installed
diffusers, Anima's and NewBie's answer final before a load
(`_OWN_PIPELINES`), `CARRIED` held to `generate()`'s own kwargs, a stub
sibling built over a stub pipeline's components and called once per image
with that image's generator, the cancel paths, mu through the shared
scheduler, the row, the PNG, the key, the migration, the reindex and the
merge), `tests/test_refine_schedules.py` (each profile's start index
against its sibling's real `get_timesteps`, the duplicate advisory, and the
start sigma: each `START_RECIPES` entry, `SHIFT_CALL`, the default schedule
and the VAE scale held to the installed source, verified.md's FLUX.1
sigmas reproduced on a real `FlowMatchEulerDiscreteScheduler` with and
without Mu, a static shift with and without a requested mu, the loaded
scheduler left untouched, None where it would be a guess),
`tests/test_api_refine.py` (the status flag, whether it is final
(`refine_settled`) and whose the strength is (`refine_strength_profiled`),
the submit refusals, the strength axis with resume, the pass's own size
guard and its lack of a batch budget, `GET /api/images/{id}`, Re-render,
and the preview's `start_sigma`, asked with the applied pass, not for a
sweep or a direct render, end to end over a real FLUX.1 schedule) and the
CLI's flags in `tests/test_cli.py`. What a pass does to pixels, its time and its
memory are the card's to say, in `docs/verified.md`.

The inpaint is covered by `tests/test_inpaint.py` (the inpaint class per
catalogue pipeline against the installed diffusers; every profile's
`inpaint_supported` True an installed class outside `INPAINT_DEFERRED`,
every class left off in it, every catalogue repo one of the cases; `build`
over stand-in components for all five classes; the strength default read,
not copied; the composite keeping the source where the mask is 0 and taking
the result where it is 255; `pipeline_mask` repainting a thin stroke off
the cell corners and covering every pixel a feather weighs, the class handed
it and the composite the painted mask; `inpaint_family` and the strength an
inpaint without one starts at; the `applied()` rules and the hash pattern
refusing paths; one call per image, cancel, the size refusal, a deferred
family refused with its class installed, the loaded answer, unload; the
status fields and the fake; the live preview drawn off the inpaint class,
and Z-Image's class blending before its callback), `tests/test_refine_schedules.py`
(`start_index` and `steps_run` with `inpaint=True` against each enabled
class's `get_timesteps`, None for the deferred; every flow inpaint profile's
class in `START_RECIPES`, Z-Image's inpaint class starting where its
image-to-image one does, the backend asking the inpaint class and SDXL's
answering none), `tests/test_inpaint_storage.py`
(the hash over size and levels, `mask_path`'s two refusals, `_migrate` and
the `masks` table on a replaced file, the mask partial finished, one cut
short or of other pixels discarded, an upload into a directory a clear
removed meanwhile, the chunk
keys and the embed, an image saved though its mask cannot be embedded,
`index_file`, `completed_cells`, the cost split), `tests/test_api_inpaint.py`
(the upload's refusals and its hash, `GET`, the submit refusals, a strength
grid over one mask, the preview's `start_sigma` asked with the inpaint and
not for a sweep, a mask gone between cells, cancel mid-inpaint, the
duplicate count with two masks, More seeds and Swap tags over the same mask
and naming a missing one by its image, Mass refine dropping it, Re-render
naming a missing mask and an inpaint at another size, the snapshot naming
a source only where every cell shares it, `/keep`, an RGBA or 16-bit
upload, Load settings from an inpaint's PNG, search and stats),
`tests/test_inpaint_bundles.py` (export, selection, password and merge by
hash; a renamed, altered, resized or path-named mask refused; a lost mask
restored from its result's PNG, `masks_missing` where that has none too;
stray files in `masks/` left out of a whole-gallery export; text chunks
outside the identity; PNG-only reindex from
files and from the embedded copy; `--database`), `tests/test_inpaint_sweep.py`
(the sweep's three sparing conditions, a re-upload restarting the window, the
whole `clear`, Keep starred, the orphan sweep, `clear --where`, a delete
and a CLI `clear --bin` sparing the mask of a cell held before it saves,
holds dropped at a server's start, a mask swept between the check and the
hold refused), `tests/test_inpaint_ui.py` (`inpaint_deferred`
held to the list; the dialog's scripts and stylesheet in page order), the run
id pins in `tests/test_conditioning.py`, the inpaint classes' prompt
building held to their text-to-image classes' in
`tests/test_prompt_encoding.py`, and the `--mask` (an RGBA, 16-bit or CMYK
file; no strength) and `--from-image` (a source that is another picture
here) cases in `tests/test_cli.py`.

The outpaint is covered by `tests/test_outpaint.py` (both types' grid and
bound checks, the fill name, the key round trip, the validator's refusals,
`applied()` dropping it with its source; every preset's integer arithmetic,
the odd 16, each aspect held at its own tiers, within 16 px, past the guard
and off the grid; the mask's levels, corners and hard edge, its pipeline
mask; the kept rectangle and `band_too_wide`'s axis and sides; the fill
keeping every source pixel and stretching the edge; an unknown fill; the
fill's and the mask's pinned SHA-256; the composite over the padded
canvas), `tests/test_api_outpaint.py` (a run over the padded source and its
record; the stamp before the job, a worded 503, a lost mask file healed and
one gone mid-run; every submit refusal, the one-sided band among them; the
refine budget, a binned source, a checkpoint without the class, an off-grid
source; the presets route, `min_kept` and a binned source's 404; a strength
sweep resumed by its spec; the preview's `start_sigma` asked at the
canvas's size; Re-render carrying the outpaint, refusing another
size, checking each fill apart under one mask, refusing a row whose spec
does not build its mask (with Swap tags' and More seeds' `outpaint` status)
and answering a row no request could make with a worded 422; Swap tags and
More seeds checking it apart from a refine; Mass refine dropping it; Load
settings from its PNG; the status types), `tests/test_outpaint_storage.py`
(the `_migrate` column, the row, chunk and A1111 line; `index_file`'s
invariant; reindex from PNGs, an unknown fill kept, a malformed spec left off
and counted, a lost mask built again, a stripped spec recovered only where the
mask hashes; `wellformed()` and a merge leaving off a spec no request could
make; a database merge carrying and dropping it; bundles carrying it, a
broken row refused (a band past the bound among them), a lost mask built
again or counted missing, the key stored in its one spelling, a leading-zero
key keyed as its re-render), `tests/test_inpaint.py` (the real backend's
outpaint branch: the padded canvas at the outpaint's size, the built mask's
pipeline mask, the kept rectangle composited exactly; `outpaint_source`'s
size refusal; an unstamped outpaint refused by `check_refine_supported` in
the inpaint's words; an outpaint without a strength at the class's default),
the outpaint key and run-id cases in `tests/test_conditioning.py`, the
dialog's statuses in `tests/test_inpaint_ui.py`, and the `--outpaint`,
`--band` and `--from-image` cases in `tests/test_cli.py` (a `--band` of its
own beside a carried outpaint running, a rewritten mask hash refused).
`tests/test_reindex_provenance.py` covers foreign PNG conditioning ids, lost
databases, missing or mismatched source identities and safe in-place metadata
remapping. API tests hold a fake before its first callback to check initial
per-cell progress totals, and API/CLI tests cover axis replacements and the
capability refusal both before and after loading. The job snapshot's
`comparison` is asserted true for a prompt-variants grid (`test_api_grid`)
and false for plain and variable grids (`test_api_jobs`). Swap tags is
`tests/test_tag_swap.py` (the rewrite as text: matching, weights and
underscores, strictness, rows at once, phrases and lettering, Han, XML,
variables) and `tests/test_api_swap.py` (every per-image status, one job
per size keeping seeds, sizes and settings, `expected`, `pruned`, the
filter scope, the negative field, the revision header, the resolver, and
Re-render over mixed sizes through `cells_from_records`); Re-render's
picked cards are `tests/test_api_grid.py` (exactly the picks whatever the
filter, oldest first, a binned and a deleted pick named in `deleted`, all
gone a 404 naming each, `ids` validated, and another checkpoint's picks
refused with or without All models, named to unpick). Mass refine
is `tests/test_refine_batch.py` (the per-record plan's order, the cell it
builds, the profile a foreign record takes, `cell_from_record`'s named
changes) and `tests/test_api_refine_batch.py` (every per-image status, one
job per size keeping each image's prompt pair, seed, CFG, mu and variables,
references dropped, the PNG chunk's `refine`, Steps, Skip refine passes,
Include other checkpoints, the whole refusal without a pass, a live
capability drop failing the job before any image, a source deleted mid-run
failing its own cell only, `expected`, `pruned`, the filter scope, the
start-sigma range and its cap, the per-size price, and the revision
header). More seeds is `tests/test_more_seeds.py` (batches by run, job or
image, the Batch as the largest recorded, the budget's cap, seeds stepping
over and wrapping, calls of consecutive seeds) and
`tests/test_api_more_seeds.py` (a picked image taking its whole batch, two
of one batch as one, the batch's own Batch and run, a second press past
the first, seeds the gallery or a live job holds stepped over, the panel's
Batch for an unrecorded batch and the budget's cap, a grid's cells as
batches of one job, one job per run, every per-batch status, `expected`,
the whole refusals, the filter scope, `batch` through a reindex, and the
`run` scope: the run's live images whatever Starred, a binned cell no
batch and a binned top seed still held, its refusals beside `ids` or
`search`, 404 unknown or wholly binned, 409 `expected`, per-batch
`foreign`, `count_default` agreeing or not, the headline, and a run past
`max_grid_images` images queued in few batches).
Load settings from PNG is `tests/test_api_png.py` (a gallery PNG's whole
record back, an explicit null Mu, the legacy key, a reference carried only
where this gallery holds the same picture, one on a checkpoint that takes
none, a refine pass at the panel's size, an adapter the installation
cannot load, a negative `applied()` leaves out kept for the box, an A1111
block's negative judged at the panel's CFG rather than the profile's, an A1111
block's prompt, negative and size with its not-carried and not-read
notes, the record winning over its own 'parameters' line, sizes a direct
render does not take, every refusal's words, the 1 MiB and keyword caps,
a stale revision, and `parse_parameters` against A1111's reading) and
`tests/test_cli.py::test_from_image_refuses_a_record_that_is_not_one`
(`--from-image` through the shared `png_settings`). What holds the
card, and whose expansion it is, is
`tests/test_jobs.py::test_the_card_names_its_holder_and_an_editor_actions_ticket`
and, through the status,
`tests/test_api.py::test_status_names_what_holds_the_card_and_whose_expansion`.

The prompt analyser is tested the same way, with no network and no GPU:
`test_prompt_encoding*.py` (`test_prompt_encoding.py` the encoder table,
templates held to the installed diffusers source and the real tokenizers,
whose cases skip when nothing is cached; `_budgets` the budgets, cuts and
control tokens; `_syntax` the weighting syntax; `_loading` the load, cache
and lock lifecycle; their shared fakes in `prompt_encoding_helpers.py`),
`test_prompt_rules*.py` (the `analysis` data against the prose, the
authors' examples raising no warning, the chips' order slots against
`reorder_tags` and the order finding's step labels, and `reads_order`
wherever the slots are None), `test_critic*.py` (pins, install,
server command and stop, the server's count and the fit checks, answer
validation, grounding, and `web/sheet.js`'s copy of `_XML_BLOCK`) and
`test_api_analysis*.py` for the routes, the chips' order slots on
`/api/tags/validate` (offline, past the 64, a prompt past Check's limit, no
category lookup where no order is read, a fault in them leaving the
verdicts), an item longer than any tag answered unknown without asking
the index (`test_an_item_longer_than_any_tag_is_unknown_without_asking`),
the page's `PROMPT_MAX_CHARS` held to every request's `max_length`
(`test_the_pages_measured_length_is_every_requests`), the gate
(Check beside a running job, off the loop; the critique's re-check after
`prepare()`), the status flags (status never loads a tokenizer) and
shutdown (the lifespan's order, `main.Server.handle_exit`, which ends a job
waiting on a load at once). Where the point
is the real thing, `test_critic_process.py` and `test_critic_server.py`
start Python children and loopback servers: a wrapped stand-in llama-server's process tree, its listener and
its counts, and the proxy bypass. The job-object cases run on Windows only;
the wildcard-listener and half-exited-process cases run on Linux only. Its
real-tokenizer case in `test_critic_tokens.py`,
`test_measured_counts_and_pieces_are_the_tokenizers`, skips when klein's tokenizer is not cached. The `fake` fixture installs
`FakeCritic` as `main.prompt_critic` - its six methods held to
`PromptCritic`'s signatures by a test, its question sized by the real
`critic.check_fits` on the unscaled estimate, standing in for the server's
count (a prompt too long for the window is a 502 here too, though near the
window's edge the two can disagree) and
its answer raw JSON put through the real `critic.validate`, so nothing it
returns is something the real pipeline could not - and
`fakes.fake_tokenizer` as `prompt_encoding._load_tokenizer`: a token per word
or mark (on Qwen also per run of newlines, which its BPE merges into one
token), with real offsets, at each encoder's real budget.
`test_the_fake_tokenizers_budgets_are_the_real_ones` holds them to the
`REAL` table.

`uv run --group dev ruff check .` lints with `pyproject.toml`'s
`[tool.ruff]`: correctness rules only (`E4`, `E7`, `E9`, `F`, `B`), with
FastAPI's `Body()` defaults, unchecked `zip` lengths, the base backend's
optional no-op hooks and the fixtures `tests/` imports from helper modules
allowed; style is the code's own and no formatter runs.
`.github/workflows/ci.yml` always runs a Linux lint job (Python 3.13,
`uv sync --locked --only-group dev`, then `uv run --no-sync ruff check .`).
It installs only development tools, not torch or the application. For a
documentation-only push or PR, this is the only job that runs: no tokenizer
downloads or test matrix. `tools/ci_changes.py` compares the entire push's
before/after range or the PR's merge-base/head range with a local Git diff.
Root and `docs/` Markdown/reStructuredText, plus images/PDFs under `docs/`,
are documentation; executable files and JSON under `docs/` still get full
tests, as do workflows, dependencies and any other paths. Renames check
both the old and new paths. New branches without a previous SHA, missing
history, unknown/empty comparisons and manual dispatch keep the full gate.
A docs-only tip on a code-changing PR does not bypass its tests.

Other changes and manual runs get the CPU suite on Ubuntu
(`uv sync --locked --group dev`, the pinned type gate, `pytest -n auto`); on main (pushes and by
hand) it runs on Windows too, about five times slower. A separate
`browser` job (same condition, never holding up `test`) runs the whole smoke
table on the runner's Linux Chrome (`run_smoke.py --all --jobs 3`) and
uploads each smoke's result, browser log, screenshots and failed fake's log
as `smoke-artifacts` when it fails. Before it, `tools/smoke_fonts.py`
installs Selawik as Segoe UI with Windows line metrics, beside copies of
DejaVu Sans and Noto Color Emoji for the symbols Segoe UI lacks, hiding
Inter and the system's own copies of those two (`FONTCONFIG_FILE`); other
system fonts stay, since Google Chrome aborts on a character no font has. Every copy carries the advance
widths Windows measured (`tools/segoe-widths.json`), so text is laid out as
on Windows; `tools/font-widths.json` logs every page character's advance
width, the run that table came from. A character the page newly draws gets
a Windows run of it and a new table. Windows stays where the smokes are written. `tests/conftest.py` makes
`torch.cuda.is_current_stream_capturing` answer False on a driverless Windows
machine, where torch's native call crashes the worker. It runs the suite with
`pytest -n auto` (pytest-xdist, one worker per core). Before it,
`tools/fetch_tokenizers.py` caches the tokenizer files (never weights) that
the real-tokenizer tests in `test_prompt_encoding.py` and
`test_critic_tokens.py` read, so those tests run instead of skipping. The
`HF_TOKEN` repository secret covers the gated FLUX.1-Krea-dev and Krea 2
Turbo; without it, as on forks' pull requests, those two warn and skip.

`tests/test_ci_changes.py` exercises the path policy and real temporary
Git histories: full pushed ranges, PR merge bases, deletions/renames and
the full-test fallback when classification cannot be established. The
design notes' [CI range policy](design-notes.md#ci-compares-the-whole-change-range)
explains why the workflow itself is never path-filtered away.

`uv run --group dev python tools/typecheck.py` checks basic Pyright over
all Python in `backend/` and `tools/`, using `pyrightconfig.json` and the
locked `pyright==1.1.407` Python package. Node must already be installed.
The gate verifies the package and bundled JavaScript version, constrains
the wrapper to that bundle/global Node and disables its update/provision
paths. Missing tools, absent/empty includes, malformed output and zero
analyzed files fail; checker diagnostics and nonzero status are preserved.
`tests/test_typecheck.py` covers these refusal and success paths. The CPU
CI job runs this same command after installation, before tokenizer fetches;
documentation-only changes retain the lightweight lint-only path.

`uv run python tools/run_smoke.py <smoke> [<smoke> ...]` runs smokes without
a second terminal: it reads `AGENTS.md`'s port table (the port, the
`--model`, short names resolved through the line under the table, and
`--step`), starts the row's fake on a gallery of its own in a new temporary
directory, runs the named smokes on it in the row's order, stops it, and
keeps the fake's log (named on a failure) only when a smoke failed. So
every fake it starts is a fresh one, which meets every row's "fresh scratch
`--data`" and keeps `smoke.json`'s planted count its own. A smoke named
alone runs alone, on a fake of its own; `--all` runs every row, a row's
smokes on one fake in order, as the commands below run them by hand.
`--offset N` adds N to every port, the fake's and the smoke's (through
`AURADREAM_PORT_OFFSET`), and refuses a port already in use;
`--jobs N` runs N rows at once, each worker 100 ports further on, the
smokes' output kept until each finishes; `--list` prints the table as read.
It exits 0 when every smoke passed, 1 on a failure, 2 on a refusal.
Screenshot destinations (`shot`, `fullshot` and the second `elshot` value)
are portable relative filenames: forward slashes for subdirectories, no
absolute or drive-relative paths, `..`, backslashes or Windows-invalid
characters, device names or components ending in a dot or space.
`browser-output.mjs` applies one policy to all three forms and
creates their parent directories. `AURADREAM_SCREENSHOT_DIR` selects
the output root; a direct `browser.mjs` call without it creates and reports
a fresh system-temp directory. `run_smoke.py --artifacts PATH` creates a
unique run directory below PATH (system temp when omitted), then a unique
directory for each smoke and reports it. Screenshots survive successes and
failures, independently of the fake's disposable gallery; parallel runs
cannot overwrite one another's files. `tests/test_browser_output.py` checks
all repository screenshot fields and deliberately invalid destinations;
`tests/test_run_smoke.py` checks isolated output and failure retention.
Each smoke retains `result.json` and combined `browser.log`, with exact
exit status, phase and duration; failures copy `fake-server.log` after the
owned fake stops. A completed run writes `summary.json` at the run root.
Browser output is printed after each smoke finishes. An interrupted smoke
retains its full partial output and result, without a final run summary.
The runner grants the Node harness cleanup time before its owned-process
fallback; focused tests cover startup failures, interruption and cleanup.
`tests/test_then_refine.py` covers Then refine: each image's pass after
its cell (a batch's bases then their passes, runs alternating), the pass
equal to Mass refine's and reported `rendered`, the base rows a plain
Generate's, every refusal in words, a failed pass kept off
`failed_indices`, three in a row ending the job, a cancel inside a pass
keeping the base, a grid's passes and retained spec, a pass run's Finish
naming its base, Finish carrying the spec, and the preview's passes;
`tests/test_cli.py` the CLI's passes equal to the server's and its flag
refusals. `smoke-then-refine.json` uses a fresh FLUX.1 Krea [dev] fake on
7912 with `--step 0.3`: the opener ending the Notify row at its height,
off, the dialog and its focus, the tick, the opener's and both buttons'
words and titles, the hint's pass size, Generate and the grid fold not
moving, Escape closing it, a batch of two making its bases then their passes with the pass
in the cell line, the pass rows' size, strength, seeds and run, the grid
preview's passes, the tick off after a reload with the strength
remembered, and the tick held with `REFINE_NONE` on a status stubbed to no
pass:

```powershell
uv run python tools/fake_server.py --port 7912 --model black-forest-labs/FLUX.1-Krea-dev --step 0.3 --data <new-scratch-directory>
node tools/browser.mjs tools/smoke-then-refine.json
```

`tests/test_peak.py` covers the peak pair: the row and the chunk, an older
database, a PNG-only reindex and a malformed chunk value, no change to a
cell's key, run id or Resume, a batch sharing its call's peak and a raised
call leaving none, a bundle carrying it (and dropping a bad one) into
INTEGER columns, Load settings from PNG ignoring it, and the real
backend's reset/read order over a stubbed `torch.cuda`, a failed read and
no CUDA device; `tests/test_cli.py` the CLI saving it.
`tests/test_run_smoke.py` holds every smoke to one parsed row with a full
checkpoint id. Under load from parallel rows a timing-sensitive smoke can
run slower than alone; a failure seen only under `--jobs` is rerun alone
before it is believed.

There is no JavaScript test suite. `web/` is checked in a real browser
instead: `uv run python tools/fake_server.py` serves the app on 7861 on the
tests' fake backend (gradient images stamped with seed and steps, a throwaway
gallery under the system temp dir, `--step 0.3` to make pause and queueing
observable; a cell whose prompt holds `[fail]` fails,
`POST /api/fake/load` (`{seconds, fail: "gated"}`) makes the loads after it
slow or gated, `POST /api/fake/step` (`{seconds}`, answering the pace it
replaced) slows the jobs after it, for a step that needs a job to outlast its
own round trips - a cancel, a stream attaching - and `POST /api/fake/expand`
(`{seconds, fail, hold, fresh}`) does one of four things to ERNIE's Expand
on the server: slows it, holds it until the next call (30 s at most), makes
it fail with a RuntimeError of that message, or puts the rewriter and
translator back off disk so a first use shows again on a server already
used; `POST /api/fake/lora` (`{name, present}`) moves an adapter out of the
fake's directory and back, as a rename by hand would; `POST
/api/fake/bin-age` (`{ids, days}`) backdates binned rows' `binned_at` by
`days` for Empty's age scope, leaving live rows live; the fake's rewriter and translator count as downloaded from their
first use), and `node tools/browser.mjs tools/smoke.json` drives headless
Edge or Chrome over the DevTools protocol with Node's own WebSocket - no npm
install - through the lightbox zoom and the focus its close leaves on the
walked-to card, the sheet, the export, the sheet's declared variable order
(under Starred too), a sparse comparison's rows, the re-render button, the
panel's persistence (a write merged over another tab's, an empty store
taking the whole panel) and the duplicate/distinct counts, plain CFG labels
on Turbo, and Turbo's woken negative caption and its grid examples (6,8,10
/ 0,0.5,1, Mu none,1.15), exiting 1 on a failed `expect` - one that
throws, or whose promise rejects, included - or on a DevTools call the
browser leaves unanswered for five minutes, or a connection it closes with
one pending, printing the step it was on, and 2 on a script it refuses
before starting - among them an
`expect` joining an async helper (`__until`, or a `window.__x = async` the
file defines) to more with && or ||, which is never awaited: the whole
condition goes inside the predicate. Each run's browser profile, under the
system temp directory, is removed on every way out - the steps' end, a
failure, a throw, Ctrl-C or SIGTERM - once the browser's whole process tree
has stopped; one a killed run left behind is swept by a later run after six
hours (`tests/test_browser_profile.py`, skipped without Node and a browser).
A `hover`, `click` or `dblclick` step takes a
CSS selector in place of a point, aiming at the middle of that element's box
as it is then. It opens the page at 1400x900,
where the smokes are written: `AURADREAM_VIEWPORT=2560x1305` (a
maximized QHD browser) and `AURADREAM_DPR=1.25` (with 2048x1040, the
same screen at 125% scaling) move it, a `viewport` step resizes it mid-run,
a `media` step emulates media features (`{"media":
{"prefers-reduced-motion": "reduce"}}`; "" drops one's override), and
`AURADREAM_PORT_OFFSET=N` adds N to every `127.0.0.1:<port>` in
the steps, so a second checkout can run the same smokes against fake
servers of its own. `smoke-conditioning.json` runs
against a klein-claiming fake on 7864 for the negative field as its caption
alone, the viewer result's ❐ (live, pressed once its image is a reference),
the reference chips (numbered and
named by gallery id, reordered by Alt+Arrow and by a drag, dropped by Delete
with focus kept in the field, the last one's × clicked leaving the emptied
row focused and the tip closed, a deleted image's chip marked), a deleted
reference leaving the chips (and named in the note after the bin), Reuse dropping a
reference the server no longer has, a reference deleted while a Reuse can be
undone staying out after the Undo, a reference deleted or binned in another
tab and an adapter moved out of the directory (`/api/fake/lora`) each
taken out by the Generate they refuse, in the page's words, the next one
running without it, a fifth reference refused under the
gallery (the other cards' ❐ aria-disabled at four, the card note's gist
"Four is the most"), the card's ❐ as a toggle named "Reference image" with
its number in `aria-description` (the card ringed, numbered and pressed, its
"Added as image 1" / "Removed" status, a sheet cell marked as a reference),
the card's and the viewer result's ⤢ aria-disabled with the reason, their
refusal in `#gallery-error` and over the tile, nothing opened, the
lightbox's description reading `ref #N` and its References row (numbered as
the panel's chip, opening it with the result after it, and Use these
references, which leaves a binned reference out with a note - its slot
reading "in the bin" - and goes dead with the reason on a checkpoint without references), the adapter select and
its tip, a weight sweep (its preview naming the adapter), the prompt
check going stale when the references change (chips, a reorder, Reuse),
the references finding's "image N" links focusing and flashing that chip
("not attached" past the last, taken back by a press that finds its chip,
another sentence in the slot left alone), a gone chip kept while the server
does not answer and dropped on its "unknown image", a reference deleted in
another tab dropped by the Generate it refuses, and one remembered across a
reload, and the distilled klein's grid examples (steps 3,4,5, no CFG sweep);
`smoke-flux.json` runs against a second fake server
claiming FLUX.1 (`--port 7862 --model black-forest-labs/FLUX.1-Krea-dev`)
for the True CFG box, the "CFG (distilled)" labels, with the sampling row's
names whole, Mu's auto unclipped and each of the row's ? tips wrapping
inside its box and the window, the lightbox's Sampling ("True CFG 1.5 · no
Mu") and Negative rows, an Example's Undo note in the panel's flow while
the negative is greyed and over its caption while it is live, the size
line's exact 1.64 MP bound, a typed side off the 16 grid named there with
its neighbours and refused at Generate in the same words, the hello-world
Example's note, a Seed past 2^53 - 1 refused with no request sent, and a
whitespace-only negative under a live True CFG box sent as `" "` (the
request spied, the row's True CFG 1.5, the lightbox's Negative "a single
space", the caption "(~2× slower; one space counts)" on one line),
while an empty or greyed box sends none, and
`smoke-preview.json` against one slowed to `--step 0.3` on 7863 for the live
preview, the tab's title and the topbar line during the job, and the
finished title and the finish line's seed after it. `smoke-ernie.json` runs
against an ERNIE-claiming fake on 7865 for the explicit prompt expansion
flow: Expand aria-disabled with its reason and focus kept, "Expand…"
keeping the caption on one line, and a press while it runs saying what the
result will do (written in, held, dropped); the first-use download and the
model's load named under the prompt with a clock, and the page's own status
requests (250 ms after the press, and at the end) counted by a spy; the
note's Undo and Ctrl+Z, and an Example over typed text; an expansion typed
over held (Use it, Discard), with the running line and the held note in the
panel's flow above a live negative; a Reset and a Reuse during one keeping
their Undo; a failure said as Expand's with the English kept, and said
under the prompt beside a Reuse's Undo, an Example's Undo or a held
expansion the running line took offered again; a size, and a prompt past
4000 characters, refused at its box; a gated load's pointer to the card,
and its sentence once beside images; the hold during a job, and Generate's
and Run grid's titles while Expand holds them (the request held on a
promise the smoke releases); Expand held aria-disabled with the reason by
another tab's paused job, keeping focus, a press saying why until the job
ends; the English in the panel at 1400x900, at the head of the results with
its note and Generate in the window at 2048x1040@1.25, and beside the
docked check at 2560x1305; Baidu's seven sizes (the seven framings marked,
12:5 named, a typed 768x1376 recognised, 1008x1008 not one of the sizes,
and 1400x800 and 784x1360 read by the nearest framing, 7:4 and 4:7, the
1400 also named off the 16 grid); and
the check's character counts and tag-list note on a fullwidth-comma and a
one-per-line list; Expand pressed during this page's job under a Reuse's
Undo, said under the prompt with the Undo kept, and the critic's press then
a muted note gone at the job's end; and an Expand waiting behind another
tab's (a request held on the server, `card_holder` "expand") leading its
running line with the wait for the card. The fake takes the claimed profile's threshold, true-CFG
flag and `mu_supported` (a Mu dropped from the row where the schedule takes
none), its model id's prompt-expansion capability, and
`loading.lora_refusal` for the id (Anima Turbo/Aesthetic and NewBie refuse adapters,
422 at submit), so it behaves as the real one would.
`smoke-animagine.json` runs against an Animagine-claiming fake on 7866 for its
tag guide, deterministic autocomplete/validation, CFG 5 default, recommended
aspects, orientation switch and disabled non-flow Mu controls, the chips past
the CLIP cut, the token count and "cut unknown", the offline state, Tab under
three characters, keyboard focus after Alt+Arrow and Delete, the 64-item
lookup cap, and the prompt
check on a tag checkpoint: a tag out of the documented order, Use reordered,
the critic's reorder set aside where Check placed every tag and found
nothing to move, the critic's reorder kept, and titled, as Check's own order
where it is one (`reordered_kind` "check"), a 70-tag prompt's order shown as
partly checked (the 64-tag lookup cap) with the critic's reorder kept where
the order rule finds nothing wrong with it (`reordered_kind` "allowed") and
set aside, in the rule's words, where it flags one, a tag the fake index
does not know (`zzkx oc`) after a general tag and a prose prompt shown as
partly checked with the items named (the first with the critic's reorder
kept as "allowed"), the same tag ahead of the general tag, alone, and a
one-clause prose prompt shown as checked (the first with the critic's
reorder set aside), "Partly checked" following `order_detail` and never `categories_detail`
alone, Anima's artist-check notes (no tag checked, not every tag), the
chips' order steps matching Check's order finding and Use reordered, a
guide tag past the 64 keeping its step, the recommended-negative Apply going
dead once the negative box is edited, a negative-prompt finding's Show,
held with the reason while CFG greys the box, saying why when pressed, and
waking with it, the order finding's Show beside "Same as Use reordered
above." with no Apply, a typed 1232x832 not one of the recommended sizes,
the 0.98 MP floor as written, the live negative's caption, the dead grid Mu
axis quoting the Mu tip beside the live LoRA weight axis, and the grid
examples 21,28,35 / 4,5,6.
`smoke-tags.json` uses an Animagine-claiming fake on 7878 for the chips'
verdict cache, typing state and live summary, an accepted suggestion's undo,
a qualified suggestion written with bare parentheses (an escaped copy typed
by hand valid with no "canonical:" note, an underscored one noted and
marked repeated),
category edges, repeats and the focus ring on one, the shared tip and the
after-cutoff mark, + tag, gap hover (waited for, as a late tray render
redraws it) and click (none on the label or hint),
the Tags helper under the prompt and whole, its order line (the server's
steps, the cite as title), the caret line naming the steps that fit from
the chip slots and their marks, a lone tag's step (1girl clicked into an
empty prompt: "Person count through Quality"), the no-placement lines for
an unknown tag and a line-broken list, the group headings' steps, Escape closing the suggestions
before the helper, and the open prompt check after both, the Examples
menu's keyboard (ArrowDown, Enter), its Undo note (on its own line at
800 px), the needs-a-reference note, the file-drop guard, the singular
summary ("1 prompt tag", checked and offline), a chip editor's
aria-activedescendant and option ids, the suggestion lists' and the
Examples menu's names, a held panel not moving for chip edits at
1400x900, Escape in a lightbox opened from the keyboard over a Tags helper
left open going to the lightbox (focus on the walked-to card, the helper
kept until the next Escape), and each chip's order step from the server:
caption, title, label and the tray's legend; none for an unknown tag, an @
name or lettering; kept on its tag across a move while the answer is held;
none where a line break stands the order down; every chip one line at
1400x900; and a grid `{hair}` drawn as the grid's chip (no verdict, no
step, "1 grid variable" in the summary, not among its prompt tags), its
declared values asked beside the chips and named in its tip with their
verdicts, and the tip saying
nothing fills it once the fold is shut.
`smoke-illustrious.json` uses an Illustrious-claiming fake on 7867 for hybrid
prompting, the native 1-2.25 MP size range, 4 MP warning against the 2.25
MP bound and disabled Mu.
`smoke-z-anime.json` uses a Z-Anime-claiming fake on 7868 for Base defaults,
natural-language prompting, negative guidance, enabled Mu, the documented
4 MP ceiling, the complete scene example and two templates' fill-in notes and slot selection,
the missing-negative note and a whole-item slot reported once; the fixed
shift's 1.79 in the Mu box, the grid preview and a Mu sweep's sheet (none
sorted between 1 and 2.5, just ahead of an explicit 1.79, in the Fixed line
and the export's file name), the profile's grid examples, the negative's
live, greyed and grid captions each on one line, and the 0.26 MP floor;
and the Refine dialog: the author's 1024x1536 opening at 1x with why, the
0.1,0.15,0.25 sweep and no sideways scroll.
`smoke-paging.json` uses a fresh Krea-claiming fake on 7867 for the gallery
paging itself on scroll (it builds a 60-image grid, so it holds once per
gallery). Run it separately from the Illustrious smoke on the same port:

```powershell
uv run python tools/fake_server.py --port 7867 --fresh
node tools/browser.mjs tools/smoke-paging.json
```

`smoke-anima.json` checks Anima Turbo defaults, its documented size range,
disabled Mu, applied saved settings, the artist check on the card's
tags-then-sentences shape (Apply, and two artists named together; the fake
index knows kantoku and wlop as artists) and the rewrite's artist check, the
card's tag order (the finding in the card's steps, Use reordered moving
tags round a sentence that keeps its place, the @ name's Artist chip slot),
the Tags menu (its order line and sections, Start with written ahead of a
prose prompt, held once complete with its reason said on a press, undone
by Ctrl+Z, no `safe` beside `nsfw`, the empty prompt's caret line), a
FLUX.1 pass reused there (the final no-pass line, its strength said as
FLUX.1's and 0.4 as the general starting strength), the refused adapter
(greyed select with the reason, a 422 at submit, no LoRA on the row), the
grid's dead Mu and LoRA weight axes with their reasons and their syntax
back when they apply, a Reuse writing no Mu into the dead box nor sending
one, a Mu dropped from the row, the two-sentence Example passing Check, the
1, 1.5 and 2 MP tiers marked, the 0.26-2.36 MP bounds as written, and the
grid's 1,2,3 CFG example, on port 7869:

```powershell
uv run python tools/fake_server.py --port 7869 --model circlestone-labs/Anima
node tools/browser.mjs tools/smoke-anima.json
```

`smoke-anima-aesthetic.json` checks Aesthetic's label, 30-step/CFG-4 defaults,
its own prompt advice and Example, its Tags menu (the family's order, no
score tags, no Start with) and the card's order checked with a score tag
placed though not offered, active negative prompt, unsupported
controls, saved variant identity and final no-Refine answer on port 7895:

```powershell
uv run python tools/fake_server.py --port 7895 --model circlestone-labs/Anima@aesthetic-v1.1
node tools/browser.mjs tools/smoke-anima-aesthetic.json
```

`smoke-anima-preview.json` checks Preview v1's name and public identity,
30-step/CFG-4 defaults, picker row and source/download/size facts, preview
guide and size caveat, score-free Tags menu with inherited parent order as
information,
its two encoder budgets, unchecked tag order, negative-prompt activation,
unsupported-input refusals and applied saved settings on port 7911, using a
fresh scratch gallery:

```powershell
uv run python tools/fake_server.py --port 7911 --model Gazingstars123/Anima-2.9B --data <scratch-directory>
node tools/browser.mjs tools/smoke-anima-preview.json
```

`smoke-model-switch.json` runs a FLUX.1-claiming fake on port 7896, on a
fresh `--data` (its Tab from the chosen row's link to the startup checkbox
holds only while FLUX has made no images, so a gallery other smokes or an
earlier run wrote FLUX images into fails there, saying so), to check
the Models picker, Escape/focus and held-button reasons, FLUX to Animagine
and back without a server restart, preserved prompts, per-checkpoint sampling
through a reload, guide/tag/capability refresh, old analysis retirement,
gallery scope and saved model metadata. Delayed LoRA and run-list replies
cannot replace the new checkpoint's data; saved requests are held under a
different checkpoint and work again after switching back. It also checks a
switch held during a job, failed target loading and Retry load recovery, and
topbar/picker fit on narrow, laptop and QHD windows. It also checks cold
Apply & save, quantization/offload changes, the saved startup checkbox,
save errors and stable action placement across choices and short viewports.
For the note it checks the loaded line with nothing boilerplate on opening,
the checkbox in the footer above the actions, a refusal that goes when its
reason does, Apply & save's unload title, the unsaved line with Revert
(through a reopening), an Apply & save outcome kept through its revision,
load progress, the `startup_overrides` sentence, a failed load and a failed
switch in place of the outcome, a switch the Hub refuses (gated, then not
found) saying what to do in the note and the error line with the running
model still loaded and its revision unchanged, a failed list kept dimmed, and a refused
switch whose words leave the error line once a later press switches.
Through a switch's load Generate is not held and Check answers, and a
Generate pressed while the switch is being sent says so as a note that goes
once the change ends; another tab's switch names the unsaved choice it
dropped; Qwen-Image's row and note say its torchao builds are larger than
the card, and no other row does. It ends with Animagine loaded, as
smoke-model-startup expects.
For the checkpoint table it checks every catalogue model's row and facts
(style, refine with its note, size with its label, speed with its sample
and basis) without choosing any, focus on the chosen radio at the first
opening and through rebuilds, choosing by radio and by a fact cell, stars that
toggle in place and group at the next opening, persistent favourites
(corrupt data, blocked writes, other tabs) and speed references that follow
draft quantization/offload choices. It also checks the server row's
Loaded/Loading…/Load failed badge without a row resizing, each row's access
label and note, radio descriptions holding the notes, muted timings from
another setup, the Size header and its titles, the Models section name, the
new-tab link label, the Favourite prefix following the grouping, one row's
star and link as tab stops (Tab from the radio reaches its link, then the
startup checkbox) and the sticky heading row, with a focused row scrolled
in below it. With the bottom row's model loaded it checks that the first and a
later opening focus its checked radio inside the picker body, the page
unscrolled; that an unsubmitted offload edit survives another tab's
checkbox-only save, which still lands; and that a list lacking the chosen
checkpoint chooses none, then the same one, or its other build, again.
`smoke-model-gallery.json` uses a fake like it, but one of its own on a
fresh `--data`, run apart from the other three (smoke-model-switch
leaves FLUX and Animagine images behind, and it counts exactly: FLUX and
Animagine empty, the planted pair, then fourteen), for the strip
of the chosen checkpoint's images: shown while the weights load, FLUX's
empty line, the planted Krea 2 Turbo pair with count, alt text and
thumbnails on choosing its row, Animagine's empty line, one request for a
quick run of choices, a delayed reply that cannot overwrite a newer choice,
a refused read said in place and retried from a poll, the column beside
the 898 px table side at 1400x900 at the picker's height, as many of 14
images as fit whole (four rows of two 150 px pictures, and no fifth would)
newest first with the whole count, a press on the column's blank space
leaving the picker open with the focus on its footer, the footer's boxes
unchanged
through every state, Tab order (link, See all, one picture stop, checkbox)
and arrow keys, the lightbox over the picker from a key and a click with
focus back on the picture, Reuse closing both, a delete re-counted, See all
closing the picker into an All models `model:=` gallery, the strip under
the table at 390x600 (40 px), and at 2560x1305 the column three across at
the S cards' 6.5vw beside a 1,198 px table side in whole rows, no sideways
page; then, still open, a resize to 1400x900 dropping the oldest pictures
without asking and to 1100x800 asking again for the strip's row, and the
column at 1280x800 (gutter kept) but the strip at 1279. `test_api_gallery.py` covers `?model=` (exact,
during a switch, under a stale revision, refused during bundle maintenance)
and `model:=`.
`smoke-model-startup.json` uses the same fake to check one-time automatic opening
for an idle unloaded/error server, with the not-loaded or failed-load note
and every model's row and facts shown,
suppression when ready/loading/busy, no reopening after dismissal or later
Unload, and focus/caret preservation.
`smoke-model-hover.json` hovers a table row and checks that its hover, every
row's position and the focus survive status polls, a catalogue reply and
another tab's favourites, with only the star's glyph (and, for a server
change, the current row's badge word) changing. For the
base-select loader lists it checks that a revision or checkbox-only change
waits for an open list to close, that keys in an open list leave the prompt
check open and start no job, and that Escape closes only the list. It covers
arrow-key drafts, a loader change elsewhere while a radio has the focus, a
checkpoint changed elsewhere, recovery from an outdated catalogue reply and
an off-catalogue runtime id requested once with no row chosen. CPU tests in
`test_startup_preferences.py`, `test_launch.py`, `test_runtime_settings.py`
and `test_api_model_switch.py` cover persistence, explicit overrides and the
saved choice set aside for another model, per-process revisions, loader
snapshots, metadata, refusal before releasing working weights, a failed
teardown keeping the running model and saved file, and Switch & load retrying
the selected model; `test_hub_access.py` covers the access check's sources,
revisions, refusals and inconclusive answers and the route keeping the
running model; `POST /api/fake/model` sets the fake's `overrides`, a
`stuck_critic` and an `access` refusal ("gated"/"not_found") for page checks;
`test_model_benchmarks.py` covers speed categories, exact sample matching and
no category for an unmatched setup or a first run on a busy PC; `test_model_info.py` covers compact facts
for every catalogue entry, favourite ids shared by Lite/Nunchaku variants,
access, Qwen-Image's fit, size labels that never claim an unsourced maximum and the shared
loader descriptions:

```powershell
uv run python tools/fake_server.py --port 7896 --model black-forest-labs/FLUX.1-Krea-dev --data <new-scratch-directory>
node tools/browser.mjs tools/smoke-model-switch.json
node tools/browser.mjs tools/smoke-model-startup.json
node tools/browser.mjs tools/smoke-model-hover.json
# stop that fake; smoke-model-gallery needs one of its own on another new directory
uv run python tools/fake_server.py --port 7896 --model black-forest-labs/FLUX.1-Krea-dev --data <new-scratch-directory>
node tools/browser.mjs tools/smoke-model-gallery.json
```

`smoke-kohaku.json` checks Kohaku's hybrid tag guide, defaults, size guidance
(the exact 1.05 MP bound), saved settings, and a hybrid prompt's
213-character sentence displayed as prose beside the checked guide tags
(the tray never "Danbooru unreachable") and their order slots, on port 7870:

```powershell
uv run python tools/fake_server.py --port 7870 --model KBlueLeaf/Kohaku-XL-Zeta
node tools/browser.mjs tools/smoke-kohaku.json
```

`smoke-netayume.json` checks NetaYume v4 defaults, the qualified size guide,
author example, a retyped and a foreign system instruction cut by Apply,
enabled Mu, applied saved settings, the catalogue's name in the topbar, the
facts and the prompt check, the example-size wording (1.57 and 1.05 MP of
the authors' example sizes, in the line and the tier titles), Mu's 1.79,
the live negative's caption, the grid examples, and another checkpoint's
record reused, the note and the ⚠ title naming NetaYume Lumina v4, on port
7871:

```powershell
uv run python tools/fake_server.py --port 7871 --model duongve/NetaYume-Lumina-Image-2.0
node tools/browser.mjs tools/smoke-netayume.json
```

`smoke-newbie.json` checks NewBie defaults, the qualified size guide (the
author's example wording), XML example selection with its valid anonymous
character identifiers and no fill-in warning, the critic's XML reorders set
aside (words moved between blocks; `<general_tags>` ahead of a character)
and persistence, applied saved settings, the live negative's caption, the
grid examples, the refused adapter and its dead LoRA weight axis, Mu kept
live, the XML prompt's line breaks capped at twelve lines in the result
column (a Tab stop) and the lightbox (four under 901 px), and a
comparison's block reorder named by tag (loose text moved, a non-ASCII tag
and a long order cut between tags included). It checks the token line
under the prompt where there are no chips, from the authors' first example
at 1400x900: hidden under three quarters of Gemma's 511 tokens; the count
from there; amber past the limit, naming where Check says reading stops;
said once in the summary line and cleared under it; the Example's Undo note
hanging under it; "cut unknown" while the server does not answer, then
counted again without a keystroke once it does; hidden where the measure
says no encoder is known; and its change held while the mouse is on
Generate or the negative box, released by leaving it and by a key. It also
runs three XML jobs that differ in one attribute, two of them queued
(slowed by `POST /api/fake/step`; the first runs 40 steps), which must be
told apart by their words in the Now line, Cancel's title, the queue rows
and the Runs picker, the Now line's title keeping the markup. On port 7872:

```powershell
uv run python tools/fake_server.py --port 7872 --model NewBie-AI/NewBie-image-Exp0.1
node tools/browser.mjs tools/smoke-newbie.json
```

`smoke-wai.json` checks WAI v17 defaults, author tag recipes, the author's
reference sizes in the size line (1.38 and 1.05 MP) and in the tier titles,
disabled Mu and saved settings on port 7873:

```powershell
uv run python tools/fake_server.py --port 7873 --model LyliaEngine/waiIllustriousSDXL_v170
node tools/browser.mjs tools/smoke-wai.json
```

`smoke-rouwei.json` checks RouWei 0.8 epsilon's defaults, hybrid prompt
help, artist-last fallback and recommended size range on the fake backend:

```powershell
uv run python tools/fake_server.py --port 7890 --model Minthy/RouWei-0.8
node tools/browser.mjs tools/smoke-rouwei.json
```

`smoke-noobai.json` checks NoobAI V-pred's defaults, author prefix and negative,
tag helper (no checked order: the authors' caption template as Check shows
it, never an empty "Recommended order:", one Date and one Year heading,
Check's no-order note without "its guide states none"), CLIP budget, saved identity and SDXL Refine pass on a fake backend:

```powershell
uv run python tools/fake_server.py --port 7891 --model Laxhar/noobai-XL-Vpred-1.0
node tools/browser.mjs tools/smoke-noobai.json
```

`smoke-refine.json` checks the Refine dialog on a FLUX.1 Krea [dev] fake on
7875, whose refine cells are the source image under a stamp: no refine
control in the panel, the card's and the lightbox's Refine opening the
dialog with the image's prompt and seed and the panel untouched, the
titles in "#N" form, focus on Strength with its value selected, the
profile's strength, the shortcut hint by the shared rule, the anchored top
across a fold toggle at 1400x900 (scrolling short of the window's bottom),
the scale buttons and the size line - naming the measured run, FLUX.1's note
under Strength, the opening size capped below it with the skipped scale
named on its button and a run at the capped size not remembered as the
scale, the crop's kept percentage and Keep shape - Versions and their
seeds, the rows and the lightbox's settings line (`refine of #N`), the
facts' Refined from link and S comparing a pass with its source, a pass past
4 MP and one past the guard refused in place, Escape on a dialog opened from
the lightbox giving the lightbox back on the same image, a strength sweep's
preview, its sheet axis and `refine:` in the search box, the passes strip
opening the lightbox beside the source, compared, a sweep that leaves no step refused
in the preview and a refusal moved to the error line at Strength or W with
Refine not moving, a strength typed and entered at once refused in the
dialog, a strength past 1 refused as a fraction, Compare 3 (held with the
Steps that would give a sweep at 2 steps, writing nothing), Compare 2 at
12 steps, Mu's "auto", the preview's "starts at N% noise" from an injected
`start_sigma` (the fake reads none), none for a sweep, and the reading
taken away by an unload and put back by a load with the dialog open and no
box touched, Refine held while the server does not answer, arrow-key
stepping and the dice, Ctrl+Enter and Enter running the pass, the
remembered strength and Reset forgetting it, the unsent draft and Start
over, Reuse opening a pass (its title and reused line, and another
checkpoint's, with this one's sampling), a pass whose source has gone, after
a viewport step to 2560x1305 the two panes with the kept-region frame and a
top 8vh down that does not move as the fold opens, the source standing in
for a pass whose frames are dropped - full size, in the result's own box,
captioned in the tile's `.foot` "Refining #N → W×H · strength s · step 1/…" -
focus off `<body>` after that pass, opened from script, ran - a source
deleted while the dialog is open refused in the dialog's own line, in the
page's words with the source line marked, with the dialog kept open, and
one binned the same way in bin.js's words, the carried adapter moved out
of the directory (`/api/fake/lora`) taken out of the pass at a refused
Refine and at a preview a press waited on, said, with nothing run until the
next press, which runs without it, the
dialog with no sideways scroll at 1400x900 and 2560x1305, the scale
buttons' aria-pressed, a first open passing over the usual 1.5x (its title
and the size line's why), a foreign pass's reused line naming FLUX.1 Krea
[dev] by its catalogue name, and the dialog's CFG named "CFG (distilled)"
as the panel's, with the greyed negative's caption "(needs True CFG > 1 —
~2× slower)"; a pass whose source is deleted while it runs (`/api/fake/step`
slowing it) landing with "(deleted)" in its title and the note and its
Reuse saying so, `markGoneSource` marking an arriving pass, and a sweep
stopped by failures when its source went: Resume held, in words, with focus
kept and nothing sent, and the server's refusal of the resume in the words
the button reads.
`smoke-conditioning.json` has the other side
on its klein fake: the card's, the viewer's and the lightbox's Refine
answering with the reason and opening nothing.

```powershell
uv run python tools/fake_server.py --port 7875 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-refine.json
```

On the same FLUX fake at 7875, `smoke-refine-unavailable.json` checks an
unloaded off-catalogue capability guess: ⤢ opening nothing, a refined
record's Reuse keeping the pass in the dialog with Refine dead and nothing
submitted, the dialog waking when status says the pipeline has a pass,
Generate meanwhile rendering directly, a final answer's wording
(`refine_settled`) and the guess's back, and a foreign pass's general
starting strength (`refine_strength_profiled` false); and Mass refine's
Refine… held with the hedge, then `REFINE_NONE`, even under a search that
gives it images, a press opening nothing, and live once the status says
the pipeline has a pass. `browser.mjs` sends Escape with its
key code, so a modal dialog closes on it as on a real press. A click, right click, double click, hover or wheel on a
selector waits until the element's centre holds still across two reads
50 ms apart (2 s at most), so a press made during a smooth scroll lands
on its target: `smoke-bundles` missed the Bundles button now and then
after Show new gallery images scrolled to the gallery's head. Fake servers
share one throwaway gallery unless `--data` gives each its own, and
`smoke.json` counts the images planted under another checkpoint: run it on
a fake of its own while others are up.

`smoke-refine-sdxl.json` checks the Refine dialog on an Animagine fake on
7877: Animagine's measured tight line (2896x2896, 2x inside it unwarned) and,
on a status stubbed to an unmeasured profile, the generic one, SDXL's strength
sweep, the pass prompt's own check (the 77-token cut outside the fold, Show,
the checking line during a slow first use, Apply into the dialog's box with
Ctrl+Z, dead while stale and dimmed by the shared 0.8 / `--text-3` rule, the
negative-prompt Apply dead while CFG greys the negative box, the hidden line
that announces the check only when its text changes), the unknown-tag line
(a repeated tag named once) and its silence on a failed lookup, a capped
opening on the generic line, From panel (held with its reason, a press
saying so on the line under it in the dialog's view; the copy and its two
Ctrl+Z steps; a greyed negative left as it was while the prompt is copied
and undone, held while only that negative differs, and copied once CFG
wakes it; an empty panel prompt), the prompt's tag suggestions (nothing
selected under three characters with Tab moving on; the preselect from
three; Escape closing the list and not the dialog; Enter and Tab taking a
suggestion; Ctrl+Z; a list kept up and unselected while the next key asks;
a click in the listed item asking nothing; the list over the box at
1400x620 where there is no room below; the panel's list untouched), the
dialog's CFG named plain CFG and its negative caption in both states
("(needs CFG > 1 — ~2× slower)" greyed, "(runs at CFG > 1, no extra
cost)" live), and nothing reaching the panel:

```powershell
uv run python tools/fake_server.py --port 7877 --model cagliostrolab/animagine-xl-4.0
node tools/browser.mjs tools/smoke-refine-sdxl.json
```

`smoke-wide.json` checks the page at the sizes it is used at, switching
with `viewport` steps between 2560x1305, 2048x1040 at 125%, 1920x947 and 1400x900 on a
FLUX.1 Krea [dev] fake on 7876: the wide tier's scale (the panel's 461 px at
2560 and 400 px at 2048, a 15.5 px body and 13.5 px captions at 2560 and a
third of the way there at 2048, eight and seven gallery columns) and the
unchanged one at 1400x900 (330 px, 14 px, 12 px, five columns), Generate on
screen, the page's own focus ring on a checkbox and a button and the LoRA
select drawn dark at both wide sizes, the caption buttons 30 px at 13.5 px
and the tick boxes 18 px at 2560, a 16-digit seed fitting its box on the
Output row's one line at every size, its dice in the Seed name's line with
a target that stops at the box - and the Refine dialog's at 125%, its row
and the dialog with no sideways scroll - the Output caption styled as
Sampling's at 2560x1305 and in one column, and the group's hidden name,
with Generate whole at scroll 0, at 1920x947 and 2048x1040@1.25; at
1400x900 scroll 0 the caption the group's hidden name, the group its row's
height and Generate's top at least 16 px inside the window; a
size strip's one tab stop with aria-pressed and the arrows moving focus
without choosing, the greyed negative's Set True CFG button above the box
(never over its text, the box still under the pointer) focusing True CFG and
writing nothing, the prompt box growing to twelve lines (eight on the short
window) with a dragged height winning over the cap, and with the grid fold
open - by a click on its summary, which brings Run grid into view - Run grid
on screen after a wheel over the stage and at the end of a 160-image
gallery, the panel holding its bottom a gutter above the window's (the
gutter and the resting top read off the page; at 1400x900, where References
and LoRA run on under the fold, stuck with Run grid in view), the panel's
order - the run hint under `#progress`, the Step cache row after the grid
fold, References and LoRA last, after it - and a hand-drawn running
batch's Now line, both bars and elapsed line in the window at scroll 0 at
2048x1040@1.25, the panel coming back down
as far as a wheel scrolls up, the prompt and Examples focused from deep in
the gallery and by Shift+Tab (past the negative's button) brought into view
with the page left where it was, a finding clicked in the prompt check
selecting its words in view without the check moving, Generate pressed with
the panel held staying under the pointer through its run and the panel
holding its bottom again once the pointer leaves, Resume's tip opening
upwards inside the window, a fold restored at load scrolling nothing, the
panel static in one column below 901 px, and in one column Batch's tip
inside the window and the comparison's always-run original box undimmed:

```powershell
uv run python tools/fake_server.py --port 7876 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-wide.json
```

`smoke-narrow.json` checks one column and narrow windows on a FLUX.1 Krea
[dev] fake on 7889 at `--step 0.3`: Reset's note in the box it emptied at
1400x900 (greyed negative) and at 600x900, Generate unmoved as it and its
Undo's note come and go, and the presses on Generate below landing with
the note still up; a running job with a long prompt
keeping the page its width at 600 px, the Now line ellipsized and Cancel in
the window; at 600x900, 800x900 and 900x700, New image as a pill centred
under the topbar and clear of Generate, for a viewer below the window, and
for one above it (scrolled to the gallery's end during a load), its arrow
following a jump to the page's top and the page never scrolling by itself;
its press focusing the live preview's tile, handed on to the image that
replaces it; a real press on Generate's right end with the pill up starting
a job (fixed points, each guarded by an `elementFromPoint` check); a
job cancelled after its first frame withdrawing the pill, its focus going
to Generate; a press with nothing in the viewer focusing the viewer, which
lets its tabindex go on blur; after a job, focus arriving in the panel
keeping the pill, and a typed character or a real press on the prompt
withdrawing it; the topbar's mirror whole with its load clock at 600 px,
the status on one line and Unload at the bar's end, with the free space
back at 1400; the Search tip inside the window at 600 and 700 px (risen
by `placeTip` where it fits neither under nor over its badge) and
unshifted sideways at 1400; the grid's LoRA weight tip clear of the left edge at
1400 and 2560x1305; a fitting `.end` tip unshifted; and in two columns New
image offered for a viewer scrolled above the window, static in the panel
after the finish line and kept past the job's end; and, at scroll 0, Batch
4 and Repeat 3 leaving Generate where it was with the run hint under it,
then that batch running with its Now line, Pause, Cancel, both bars, the
counters and the elapsed line (one line) in the window and the run hint
under them, at 1400x900, 1920x947, 2048x1040@1.25 and 1536x864@1.25 (the
panel's gap 10 px there, 14 px back at 1400x900):

```powershell
uv run python tools/fake_server.py --port 7889 --model black-forest-labs/FLUX.1-Krea-dev --step 0.3
node tools/browser.mjs tools/smoke-narrow.json
```

`smoke-grid.json` checks the grid fold on a FLUX.1 Krea [dev] fake on 7879:
two direct renders back to back, each awaited to its end with one stream and
one tile; the resolved prompt and the one-value warning (commas named); Run grid as
wide as Generate; the totals at body size with factors of one trimmed; a
spaced range, `6..12` and an axis message naming its box, a value past the
schema's bounds included (CFG 25, Steps 150); Resume's refusal
without pydantic's prefix; Repeat above one; the budget line, including 0
seeds, the jobs ahead, and the load note following a real unload and load
with no box touched; the cells already in the gallery once a grid ends,
without Resume; Ctrl+Enter running the grid from a grid box, refusing from
the prompt while the grid holds its placeholders, and generating on a plain
prompt; FLUX.1's grid examples (21,28,35 / 2.5,3.5,4.5), no Mu example, Mu
"auto" and no fixed shift; and, at 2560x1305: Run grid in the window after
a real click on the fold's summary once the preview has landed; a 30-line
prompt whose variable is listed by its line, with the preview's top and Run
grid in the window together; the one-value case; a long one-line prompt's
listing capped at eight lines and scrolling inside itself (both previews
awaited, not debounced, and the box logged); and with the fold opened by
hand before the variables are typed, Run grid no further under the window
than the listing box and one line (the panel holds still under the
pointer):

```powershell
uv run python tools/fake_server.py --port 7879 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-grid.json
```

`smoke-sheet.json` checks the contact sheet at 2560x1305 on a FLUX.1 Krea
[dev] fake on 7880: the page landing on a sheet it opens (Runs, a
comparison, a sweep), a 4 x 3 on one screen with heads at the page's
`--fs-body-sm` and srcset sizes, heads sticking in the sheet's own scroll
box and a keyboard-focused cell scrolled clear of them, the fit and the wrap
following a resize, a twice-run grid's repeats in the note, a seeds-only run
wrapped (and its export), edit labels for prompt iterations, the restored
search on Sheet off, the Sheet button's newest run, pins taking the
commonest value, F starring a focused cell (said in `#star-said`), the lightbox opened from the
sheet starring the cell (its star "★ Star", `aria-label` Star, its title
and `aria-pressed`) and F on the cell it hands focus back to, the Fixed line
under Starred, the run bar (More seeds of this run… right after Resume
this run, the run's line "Run … · N images · checkpoint quantization" last,
on a row of its own under the buttons), Resume this run filling the seed and
size (its warning keeping the preview's `.totals`), the dead-end pointer
"Nothing in run … is missing. More seeds of this run… adds seeds." on a run
with every cell done and not on another grid's mismatch, its mismatch
warning and its absence on another checkpoint's run, whose More seeds of
this run… stays, held with "Made on …", and says it under the gallery's head, the version rows with their edits and titles (a
reorder as "same words, moved"), a strength sweep's source column and
lightbox walk, the refined badge, two passes alike on every axis but
their source counted in the note as sharing a cell, pins in the export's
file name, an
empty sheet drawing only its note; the cell star (hidden until hover or
focus, out of the tab order, a click starring without opening the lightbox,
under the stuck heads when the box scrolls); iterations on random seeds (a
diagonal, then the seed pinned) with the note's Compare pointer, and none on
one fixed seed; Fill width (aria-pressed, the cells filling the width at
2560, 2048x1040@1.25 and 1400 with `sizes` following, the export at the
on-screen width from the PNGs at 2560, at the cells' device pixels at 125%,
and no smaller than Fit's at 1400, Fit's export from thumbnails, held on a
wrapped row, remembered across a reload and working with storage
throwing); Export all (beside Export PNG only with a pin, both buttons held
while it runs, one stack named `-every-cfg`, five slices past the edge
saved as one image each with distinct names, the same five files written
when the sheet is redrawn on another run mid-export, and a failing export
said under the gallery head); the matrix as a table (rows `display:
contents`, header and cell roles) with cells named by their axis values and
described by the facts the name leaves out, its opening said, a wrapped
sheet as a list; a 2 × 8 comparison's 16-digit seed heads whole under Seed
at 1500x900, and at 1400x900 breaking between groups of four into balanced
lines; and at 800x900 the finish line's Show sheet and a strength sweep run
from the viewer's ⤢ each handing focus to the first cell, with Tab then not
moving the page; and at 1400x900 the sheet's focus kept through F on a
cell another tab deleted (onto the cell now in its place), a changed
Columns select, and another tab's job ending under it, the sheet left as
it was. `smoke-sheet-klein.json` uses a FLUX.2
[klein] fake on 7886 for the References axis, the Size row and its pin, and the
adapter named in LoRA heads, the Fixed line and the export:

```powershell
uv run python tools/fake_server.py --port 7880 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-sheet.json
uv run python tools/fake_server.py --port 7886 --model black-forest-labs/FLUX.2-klein-4B
node tools/browser.mjs tools/smoke-sheet-klein.json
```

`smoke-sheet-zoom.json` checks the sheet's zoom and pan at 1400x900 on a
FLUX.1 Krea [dev] fake on 7910 with a fresh scratch `--data` of its own: a
1024 px steps x CFG 4 x 3, the zoom group after Fill width (− + 1:1 and the
readout, − and the readout held at zoom 1 with the reason, a press saying it
under the gallery head), + growing the cells by `ZOOM_STEP` with the box
scrolling and the button not moving, every zoomed cell in the box's view
showing its full PNG (`src` the record's `url`, no `srcset`, the decoded
bitmap 1024 px - `naturalWidth` is no measure, Chrome keeping the density
of the srcset pick the swap replaced) and keeping it zoomed back out, the
heads stuck, Ctrl+wheel zooming about the pointer (the point under it kept
within 3 px) and a plain wheel not, a real mouse drag panning the box by
the drag with no lightbox while a click and a press moving under 4 px open
it, the keys 1 + - 0 on a focused cell and a held key saying why, the zoom
kept across `renderSheet()` and back to 1 on Sheet off and on, Fill width
as the zoom's base with the export at the unzoomed width, a wrapped
seeds-only row holding every zoom button with its reason; and at 2560x1305
1:1 drawing each image at its own pixels and the zoom stopping at 400%:

```powershell
uv run python tools/fake_server.py --port 7910 --model black-forest-labs/FLUX.1-Krea-dev --data <fresh scratch dir>
node tools/browser.mjs tools/smoke-sheet-zoom.json
```

`smoke-delete.json` checks deletes taken to completion on a FLUX.1 Krea
[dev] fake on 7881 (a checkpoint with a refine pass, so a starred pass can
keep its source): the per-image delete moving to the recycle bin with no
dialog, and its wording; the page's own dialog for the bin view's Delete
permanently, beside the card's ⌦ and over the lightbox's, Escape as Cancel,
in the danger colour, and its count box for
Delete unstarred, with its "They move to the recycle bin" line and outcome,
a lightbox of one image deleted (reopened at once after a
close, the check waiting for the delete itself and
`closeLightboxAfterDelete`'s focus), Delete unstarred keeping a starred
pass's source with the typed count, a landed image's 409; images another
tab made since the grid loaded, put on it before the count is asked and
counted in the dialog by the server's `newer`; the All models/running-job
dialog and the 409's running-job sentence for a job another tab started
(the server's `job_active`, not the page's `busy`); the dialog kept inside
the scrollbar, by real presses, for a row's last card and for the
lightbox's Delete permanently; a pass whose source has gone - marked at once on
the page that deleted it (record, card title, Reuse) - tombstones
("Deleted · #N"), focus and the emptied viewer's placeholder, an image already deleted,
a tile deleted elsewhere (once the viewer's PNGs have loaded: on Windows the
server keeps a file it is still streaming) and the job's replay, the
lightbox paging in behind a delete (its counter reading the gallery's total
and the id), a double-clicked delete binned once, and Reuse: Ctrl+Z and its Undo, the
Undo retired by a change (Ctrl+Z, a seed, an Example) and a Reuse taking an
Example's Undo, the page unmoved mid-gallery with the prompt in sight, the
page's end, one column at 800x900, the Undo in the window at 2048x1040
@1.25, every image in scope deleted by another tab (a real press says no
image is left, focus on the search box), and a real sweep that takes every
image handing focus to the search box, then a catch-up landing after that
sweep's own reset read past its images (`newestImageSeen` lowered, the
status received): no reset and the "Moved 2 images" line kept
(`prependFinished`'s `galleryReadHead` skip); and the head holding still
under real clicks: Starred ticked with Delete unstarred's box kept, a
point it was clicked at still Starred's, a second click unticking it
rather than focusing the search, a search typed with the pointer on the box
keeping Re-render (and All models' count) back, the pointer leaving putting
them right, and with both buttons shown another job's twelve images landing
under the pointer: the count and Re-render's label written, Starred, All
models and Delete unstarred unmoved, the search narrower (`.held`). Every wait in it is a condition
(`__until`, `__settled`), none a sleep:

```powershell
uv run python tools/fake_server.py --port 7881 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-delete.json
```

`smoke-jobs.json` runs against the default Turbo fake on 7882 at
`--step 0.3` (it passes at `--step 0.15` too): the tab title and topbar mirror; the one-line Now row, count
first, with Pause and Cancel and its title; the queue's named rows, a
per-row cancel and the single "Next:" row; Unload held with its reason,
and a press held by a load with no job said as a note that goes once the
load lands;
Pause answered at once with no aria-pressed and held on a one-image job,
its clock ("for 2 min" in the cell line, "⏸ Paused 2 min" in the title, its
`paused_at` set back past a minute, at the next poll),
its note gone with its job; the finish line's seed with Use seed and the
fixed-seed hint; a grid named by its own prompt, skipping a `[fail]` cell
live, with Show sheet and a Retry that sends its own payload with Resume and
the base seed, once, keeping focus; the failure rule's "Stopped at cell" and
Resume, kept as the previous job's while a queued job runs; a plain job
whose every image failed, said in images ("1 of 1 image was skipped",
"Stopped at image 3 of 3") and titled "Nothing made"; a cancelled
job's outcome led by "Previous job:" while the next runs and gone at its
end; focus after Cancel (Generate) and after Pause held to the end (Use
seed); a time budget's held ETA and count, and its finish line's Resume
the remaining N cells (N from the cells the budget reached, "under the same
0.06-minute budget" in its title) sending the grid once with Resume and its
budget, then "Resume sent"; the finish line's Use seed then More seeds…, and
a done grid's Show sheet then More seeds of this run…; the gallery kept where it was
scrolled with New image; the offline state through a stubbed fetch and a
restart through a changed start time; the load's ticker with its
aria-hidden clock and indeterminate bar, also shown in the empty viewer's
one line, whose own words come back after a cancel; a second job queued
during the load keeping the load's note under the bars through the queue's
redraw; a cancel during the load; a gated
load's detail said in the panel line after every Generate while the viewer
holds images; a gated load's card said once and not re-announced by the
banner, its Retry load handing focus to Generate; Load and Unload handing
focus to each other; the banner hidden by a new job and not repeated
after a watched one; a gated load's finish line "Not made: the model could
not load"; and a watched job whose server goes (a stand-in EventSource and
fetch): the job line in the topbar's phrase with the count, no steps, time
left or topbar mirror, the tab's title and the strip's note the same, then
`showLostRun`'s card, reworded once the server answers, its Run it again
running the request with focus going to Generate; each job watched once
while the page is between jobs - gated fetches hold each ordering open, with
its preconditions asserted: a Generate during watchNext's listing and
during resumeActiveJob's; a queued POST answered after the listing found
nothing; one answered after the listing had made its job the watched one;
and one whose job the listing watched to its end first. Each gives one
stream, one tile and no queue row; the first four also give a finish line
and `startJob` resolving at the job's end, and the last gives `startJob`
resolving on its POST's answer, after the job has ended, keeping nothing of
it; live regions (the topbar's name not rewritten by a poll nor by another
tab's job, a single image's step count shown and not said, each run
counter said once, `#job-said` at a job's end and for a queued job, nothing
there for a cancel or a load failure the panel line says); and a second
tab, an iframe of the page: its job (held paused) holding Unload and the
critic here with the reason and the topbar's aside, its images joining this
gallery at its end, and taken in after this page's own job, queued behind
it, is cancelled at its first step (that job of B's held paused until this
page's poll has seen it, which a two-image job at `--step 0.15` outran);
its queued job picked up here as
another tab's (viewer kept, this page's New image and its own finish line,
Use seed and all, kept through it, its end said on `#job-said` only, this
tab's own finished title given back until a key), the estimate leaving the
first step out once a second has landed (none at step 1, a one-step job
none, a sweep of one-step cells from its second),
a cancel and a queued row's cancel from that tab, a pause here with its job
behind ("waiting: job ahead paused" there, "has been paused for 3 min" once
its `ahead_paused_at` is set back past a minute, where Pause on that queued job
says it will hold before its first image and a second press withdraws it;
its row marked as another tab's here, Cancel queued's title, the count), Cancel queued here said there as a
cancel from elsewhere, a job that ended at the load reworded once that
tab's Load lands, with the panel line's reason taken back, and, on a page
opened afresh, one that ended at the load over the card said on
`#job-said`, reworded by the card's Retry load and again by an unload; the
scrolled gallery's step moves the pointer off the cards first, which would
otherwise hold the new card; and at 2560x1305, with the pointer resting on
the Grid search summary under this page's running job, another tab's
queued job waiting (`queueWaiting`, the summary unmoved, no row) until the
pointer leaves, then listed, and one queued and cancelled there while the
pointer rests dropped from `queueWaiting`, no row as it leaves:

```powershell
uv run python tools/fake_server.py --port 7882 --step 0.3
node tools/browser.mjs tools/smoke-jobs.json
```

`smoke-results.json` checks the results viewer on a FLUX.1 Krea [dev] fake
on 7883 (written at `--step 0.3`; it passes at the default step too): every
? badge named and described, and Resume's badge leaving Resume unticked; the
one-line idle viewer, the empty-install box and its Examples; the empty
gallery's line and its Examples; the idle viewer staying one line beside a
docked, content-sized check with the gallery beside it; a result's
figure (a group named #N), label, title, facts-only description and card
tools at 1400x900 (❐ hidden on FLUX), with `this-run`, the star and ⤢; the
note under it, "#N · W×H · seed N · Use seed", with the gallery head
unmoved as a single image lands at 1400, and hidden at QHD; at 2560x1305 the
column with Delete in words (Star, Refine…, Inpaint…, Outpaint…, Reuse,
Use seed, More seeds…, Delete; the row under the picture with a docked check
leaving More seeds… out) and
the row under the picture empty at `--h-tool`; the live preview
keeping the result's box; `#recent` level with the viewer, the wheel over
it with nothing to scroll scrolling the page, and staying
beside it with the check docked while the image and the gallery head stay
put, the column's actions and `.result-brief` then the row under the
picture; the fit at 2048@1.25; the lightbox walk from the single image; a batch
of four best-fit; a pass's note and lineage and its Compare with source
(the lightbox on the pass, compared; Escape back on the strip's image; no
Use seed in its facts), then its source deleted ("#N (deleted)", no
buttons, the title, a facts-only description, Compare with source
aria-disabled with the reason and refused over the tile); a portrait pass
with the check docked, its row under the picture whole at 2048x1040@1.25
and 2560x1305, Compare with source left to the column; a 5:12 direct image
and its pass at 2048x1040@1.25 with the check docked, the row under the
picture wider than it, centred, inside the strip, every word whole, the
picture and the gallery head unmoved; four landscape images two by two with
the gallery head unmoved from the first frame; a deleted result held at its
height with focus on its line (the column's Delete); a 28-seed run at
2048@1.25 ending scrolled to its newest row, the wheel over it scrolling
the page at its end (down at its end, up at its top) and the strip while it
can scroll; 28 at QHD with no padding-only
scrollbar and the gallery head unmoved from its first frame (`reserveStrip`),
the wheel over it scrolling the page; the banner's
"2 since deleted" / "all since deleted"; the banner's Resume after a
resumed-and-cancelled grid putting the run's base seed in Seed; no Resume
for a cancelled comparison; a batch of four at QHD followed from its first
frame to its end - the preview in the first image's column, no tile moving
within the strip, the tile size and the gallery head where the first frame
put them - and the strip fitted alike with the page's scrollbar forced on
and off; Reset with its Undo in the prompt box it
emptied - inside the window at 1400x900 and at 800x900 - Ctrl+Z after it
giving the prompt back, and the store kept forgotten past the prompt's
blur; Reset taking a Reuse's note from the panel line, with its Undo
on offer or already retired; and, in one column at 1400x900, another tab's
image held while the pointer rests on a card (the count new at once, the
card under the pointer and its box unmoved) and joined once it leaves:

```powershell
uv run python tools/fake_server.py --port 7883 --model black-forest-labs/FLUX.1-Krea-dev --step 0.3
node tools/browser.mjs tools/smoke-results.json
```

`smoke-lightbox.json` checks the lightbox on a FLUX.1 Krea [dev] fake on
7884: the facts list and its rows, the Created row's peak (made through
`POST /api/fake/peak`: its words, the reserved figure and caveat in its
title, "of its batch" on a batch of two, nothing without one, and the head
still fitting), the Seed row's Use seed (said in the
lightbox's line, the panel's kept, taken back by a step to another image);
the keys line, naming S on every image with each key bound to its action
("Del bin"),
and the toggle's S by Generate's hint rule; the star ("★ Star",
`aria-label` Star; F said in `#lb-star-said` away from Star, not on it); a
pass's Reuse pass… name; a pass compared with its source (S, the button, a
dragged and keyed divider, ← → swapping to the other pass of the source
with the comparison, divider and focus kept and wrapping round past the
source, Shift+arrows and PageDown moving the divider on it and Shift+arrows
off it, the keys line and the divider's `aria-keyshortcuts` saying so, a
pass with no other pass in the strip keeping the old keys and Shift+arrows
the same 2% on the divider and off it, S on the focused
divider handing focus to the toggle, swipe and side by side keeping 1:1 and fit, side by side at fit
clear of the arrows' circles, a pass reshaped to 3:2
compared with the part it kept, a deleted source's dead controls); the
source link with the view carried; the passes link; the facts' ⌕ search
buttons (which rows have one, their titles, a seed replacing the box's seed,
a repeated press adding nothing, size, the prompt's words, the sampling row
with True CFG and `mu:none`, model and the local day, each narrowing the
gallery as the server reads it); a pass whose source has
gone (its Reuse and its card's title); the arrow strips (a press 60 px off
the circle steps, the backdrop below still closes, the wheel over a strip
zooms, the first image's disabled arrow a circle with the backdrop beside it
closing, a double-click whose step takes the strip away stepping once,
circles again when zoomed); a close followed at once by an open keeping the
new lightbox; focus after Escape on the walked-to card, with the Examples
menu left open under it too (the menu kept, the next Escape closing it),
and Ctrl+Enter there generating and opening nothing; Escape on an image no strip
holds focusing the search box; Cancel on a Refine opened from it, a side by
side 1:1 included, and no reopening after a run; an Escape and a Cancel held
while the pass is being sent ("Sending…"), with the pass run and focus on
the source's card; F on an image another tab deleted (the note, the next
image, nothing in the panel's line); an image that will not load (deleted,
and with the server not answering: the offline sentence as an offline
message, taken back by the status poll that answers, which asks for the
picture again); at 2560x1305, 1:1 on a 1 MP image by key, button, wheel
and double-click, the 52 px circles clear of the picture and side by
side's halves inside them, circles below fit with the double-click beside
the picture giving fit back, and a double-click beside a below-fit image
giving fit back; at 800x900 no keys line; a wheel burst said once from the
hidden `#lb-zoom-said`, a keyed zoom at once, and F rewriting neither the
zoom nor the counter; at 600x900 the counter ending a gap before Close,
clear of its focus ring, and the facts taking a square picture's
letterbox (every row in view) while a portrait picture keeps the six-line
scroller, and is fitted again, whole, once Use seed's line takes a row
from the stage; Side by side titled stacked before the press; side by
side stacked at 600x900 for a square pass (a wheel
zooming about a point in the lower half) and a 7:4 pass (halves over 400
px with the facts' room, opened again stacked again), and left and right
for a portrait pass; and at 800x900 a square pass left and right with its
facts in view; Reuse and More seeds… sharing one `.lb-pair` row, a half each
and uncut, More seeds… titled with the image's batch on every image, and
neither button, Refine nor Delete moving from a pass (Reuse pass…) to its
source (Reuse settings) and an orphaned pass:

```powershell
uv run python tools/fake_server.py --port 7884 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-lightbox.json
```

`smoke-compare-cfg.json` checks the comparison across CFG on a FLUX.1 Krea
[dev] fake on 7899 with a fresh `--data` of its own (another smoke's images
at seed 78 would join its ladders): a CFG x steps grid, separate Generates
at one seed, a refine pass and a grid with no CFG axis. On a sweep's image
the box is there before the gallery answers, Compare CFG holding S, Compare
with source absent, S on the keys line, Against listing the other CFG
values at these steps only; S on against the value below (tags, the
divider's `aria-label` and `aria-valuetext`, the note, the partner's alt,
Compare CFG and Star where they were); → stepping with divider, mode and
1:1 kept; at the top the arrow and the held Higher CFG saying why; Against
swapping the partner, drawn at the small buttons' height, type and border,
and ← → on it changing the partner with the image and focus kept; Lower
CFG; side by side with the lower CFG left; Shift+→ on the divider; S on
the divider handing focus to Compare CFG; the
lowest value in view on the left, its partner cut to the right of the
divider and in the right half; two separate Generates pairing and a 20-step
one left out; an image with no other value held with the reason (S and a
mode press saying it, Swipe and Side by side titled with the reason, a
step taking it back), a late catch-up for images that ladder read already
(`newestImageSeen` lowered, the status received) keeping it, the swipe
still saying there is none rather than "Looking for…" (`noteUnseenImages`'s
`head >= newest`), and their own titles back on an image with a
neighbour; the held reason "in the newest 1" for an answer cut short; a job
landing a new CFG value for the image in view (the script's, at the fake's
own pace: it ends between two status polls, never in `active_jobs`, and
the status's `newest_image_id` moving takes it in), and again with Starred
on, the new image in no strip the view holds and the ladder asked again
all the same (`noteUnseenImages`); "may be incomplete" on a
full page; an image opened alone held while asked ("Looking for…"), the
request's terms (CFG's with `steps:`, steps' with `cfg:`, two requests),
then a partner the arrows do not step to, Fewer steps' held title saying to
turn on Compare steps; the CFG ladder read past a full page by `before=`
(one-row pages); an answer held in flight
while the walk moves on and a job lands a value it would have held, not
kept, and asked again on the way back; a pass's S still the source's with
Compare CFG held; a partner deleted by the page ending the comparison with
its note; at 600x900 no box for an image with no neighbour and one for a
sweep's image; and the sheet's Swipe CFG (laid-out CFG only, the first cell
with a lower value, the keyboard's cell and its row, held for two cells at
different True CFG and saying why on a press, gone when CFG is pinned or on
a run with none). Then across steps: Compare steps beside Compare CFG on
one row, its title and Fewer / More steps' titles and holds; turned on, the
tags, the divider's name and value, the note ("2 of 2 step counts"),
Against's step counts, the keys line "← → steps" and Swipe's words; ← to
the fewer steps, kept on the left at 1:1, the end saying "20 steps is the
fewest of these" on the arrow and on Fewer steps; Higher CFG from a steps
comparison turning it to CFG, and Compare steps turning it back in side by
side; More steps; S on the divider handing focus to Compare steps; More
steps turning it on from off; the fox (CFG and steps both differ) holding
Compare steps with its reason; at 600x900 only CFG's part for the fox,
only steps' for the teapot's other cell (Against hidden until Compare steps
is on, then listing step counts), and both for a grid image; and the sheet's Swipe steps (the keyboard's cell and
its column at one CFG, held and saying why for cells that differ in CFG
too, and on the teapot's steps-only run, where Swipe CFG is absent):

```powershell
uv run python tools/fake_server.py --port 7899 --model black-forest-labs/FLUX.1-Krea-dev --data <new-scratch-directory>
node tools/browser.mjs tools/smoke-compare-cfg.json
```

`smoke-gallery.json` checks the gallery on a default (Krea 2 Turbo) fake on
7885, a checkpoint with no pass and no references: the Runs picker's groups,
labels (what each run varied: "steps ×8 · 5 seeds", "2 seeds") and sync with
the search box (a pick, a card's ⊞, typing, Sheet off), the picker browsed
from the keyboard (no fetch while arrowing, Escape, Enter, leaving it),
at its own width with the search box whole and the picked run in its title;
`id:` and the search tip (with `mu:none` as the pipeline's own shift); the Starred tick and Re-render's label; each card
a group named #N; the five tools and their names on hover and focus inside
the card, with Delete's over Delete, and the image's facts-only
description; the card's #N · W×H · seed caption on hover and focus, hidden
under a tool's name and dropped under a card note; Refine and Reference
aria-disabled, their refusals in `#gallery-error` and a one-line card note,
with the card and grid unmoved, a second press landing on the same tool,
and the row's names held down while the note is up; the star button (named
Star in both states, hidden until hover, a click starring without opening
the lightbox, F from the star saying nothing more, F from a tool and from
the image said in `#star-said`, the lightbox's F reaching the card and said
in `#lb-star-said` - each read from `__said`, a record of the two lines'
writes - the lightbox star's title and `aria-pressed` matching the card's,
and a star on an image deleted elsewhere tombstoning it with "#N was
already deleted." in the gallery's line and the Runs picker asked again);
a refined record's badge and Reuse labels (planted into the page's gallery
replies; its name led by "Reuse pass"); the ⚠ carry-over warning,
said again when references flip and when the adapter list lands; the day
line across pages, a line over the first card when the newest images are
not today's, and a deleted head's tombstone across a day line drawing no
second line; the roving tab stop, arrows across a short row and the day
line, Home/End, Down paging in, F, a delete handing the stop on, the
lightbox's user close handing the stop to the card it ended on, and a job's
newest image taking the stop where none was chosen, prepended under an old
head made two days older with the day's line between them; the head's
S/M and Fit/Crop from 1800 px (eleven and nine a row at S with the tools
fitting, Crop's cover, one tab stop per pair, remembered across a reload,
hidden on a sheet, kept by Reset, not applied at 1400); S four across in
Gallery beside's column at QHD, M three; the tools fitting at
2560x1305 and 2048x1040@1.25; and a job a script sends at the fake's full
speed just after a status poll, ended before the next and never in
`active_jobs` (`elsewhereBusy` never set, the page watching none of it),
taken in by the status's `newest_image_id` (`followGalleryHead`) within a
poll or two - once, at the head, one more card and one more in the count,
and two more polls (counted, not waited out) adding nothing; another tab's
two jobs in a row, the first one's image in while the second still runs;
and sixty images between two polls (the page's statuses held meanwhile),
more than one page reaches back over, joining the head with the old head's
card the same node and the paged-in tail kept, no reset:

```powershell
uv run python tools/fake_server.py --port 7885
node tools/browser.mjs tools/smoke-gallery.json
```

`smoke-batch-mates.json` checks a card's batch outlined on hover on a
default fake on 7892 with a fresh `--data`: a Batch 3 x Repeat 2's other
five cards outlined (1 px, solid) and not the hovered one, no card moving;
a steps grid's cell at Batch 2 a set of two; a single image marking
nothing; a press keeping the marks and the pointer leaving taking them down:

```powershell
uv run python tools/fake_server.py --port 7892 --data <new-scratch-directory>
node tools/browser.mjs tools/smoke-batch-mates.json
```

`smoke-select.json` checks the gallery selection on a default fake on 7898
with a fresh `--data` (it builds a 60-image grid): Ctrl+A (Cmd+A) on a
focused card or on Select, with Select off as well as in it, adding just the
loaded cards (no gallery fetch, saying unloaded images were not picked),
with picks from another search kept and the cap applied, and leaving native
text selection in the search and the prompt; in the recycle bin adding only
the bin's cards, the gallery's picks set aside and back unchanged; Select in
the footer row before Swap tags…, Refine…, More seeds… and Bundles…, at
Bundles…'s height; with Select off, plain click opening a card while keeping
the selection (including hidden picks) and range anchor, Ctrl+click
(Cmd+click) toggling a pick and moving the anchor, Shift+click replacing the
picks with the range and Ctrl/Cmd+Shift+click adding it; duplicate click
details and non-left clicks opening or picking nothing; Space toggling and
Enter opening with picks left alone; image names Open outside Select and
Select with aria-pressed inside it, Enter first among the keys; a card's star starring it with the picks
kept; a click on blank grid space clearing the selection with Select off but
not in Select mode; Select pressed with neither it nor the grid moving as
the count appears, its hint the row's gap from Select and the count 4 px
from a shown Clear; in Select mode a quick double click toggling twice and
opening nothing, a click, Space, Shift+click and Shift+Space (a range from
the anchor out, backwards too) picking, a second click unpicking, the
image's `aria-pressed` and name and the group's "(selected)"; Enter still
opening the lightbox and F still starring; a search showing one of eight
counted as "7 not shown", and back; the next page's cards marked and a range
across the page boundary; a picked image deleted by its ⌦ leaving the
selection; the cap (a range picking what fits, a single pick refused, both
said); Escape clearing, then leaving select mode (the image becomes an
"Open #N" button), then left to the page; Select off keeping a selection that
a plain click does not change; Clear handing focus to
Select; on a sheet Select held with its reason, not pressed and with no
hint, and the selection kept and counted; a refused pick leaving the anchor;
`selectedRecords()` giving the loaded page's records after a search; the
gallery line as the footer's `1 1000 auto` item with no margin of its own,
the footer full width at -8 px (a layout pin: it passes on the CSS from
before the footer moved to gallery.css too); a rejected search counting
the whole selection as not shown; and a long
message in the gallery's line beside a long count staying one line - the
footer, Select and the grid unmoved, the text the title - at 1400, at 600
and beside the viewer at 2560, the count's number whole each time, its
words whole at 1400, " selected" a ✓ at 600 and beside the viewer, where
"✓ 2 · 1 not shown" fits whole, Picked a ▾ and Clear a ✕ there, 4 px apart
and the row inside the column. Then the Picked menu between Clear and
Select, held with its reason and said when pressed; a first pick with no
filter showing "Re-render 1 selected at WxH" while Delete unstarred gives
way, the head's height, the grid and Select unmoved, and Clear putting it
back; on a filter whose own Re-render shows, a pick keeping the head's
height and taking the label; picks made with no filter, then a filter
typed, then Clear pressed under the pointer keeping the head, the grid
and Select where they were with Delete unstarred still given way, and
back once the pointer is on the top bar; the menu opening over the grid without moving
it, inside the window, its Escape, ArrowDown, ArrowUp and Enter; Star picked
(both cards marked, the server's favourites, the picks kept), again
("already starred"), Unstar picked, and F starring only the focused card;
Re-render of two picks one of which another tab binned (said, dropped,
the other rendered again); Bin picked (tombstones, the note's Restore, the
bin's records, the head unmoved) and Restore; the menu opening over its
button low in the window; and Picked held with the bin's reason in it:

```powershell
uv run python tools/fake_server.py --port 7898 --data <new-scratch-directory>
node tools/browser.mjs tools/smoke-select.json
```

`smoke-swap.json` checks Swap tags on an Animagine XL 4.0 fake on 7900 with a
fresh `--data` (a `{hair}` grid at 832x1216 and one 1024x1024 image): the
button between Select and Refine… (then More seeds… and Bundles…) at their height, held with its reason
and saying it, titled by a search and then by picked cards, not moving as
cards are picked; the dialog's scope line, focus, both fields, Queue held
("Add a tag to find", "Checking the swaps…" while typing) and saying it;
the server's statuses, counts and marks (a weight and underscores kept, the
missing cells' text unmarked, what each queued image's swap took out
struck through as written), Queue and Cancel unmoved; a second row's
duplicate find refused in the count line; rows added moving neither + Add
row nor Queue, four scrolling in the rows box; two clicks on one row's ×
removing that row only, its slot kept until the pointer leaves; an image
deleted while open shown as deleted; one deleted on the server only, with
Queue's 409 kept in `#swap-error` after the check frees Queue; Queue
watching the first of two jobs (one per size) and listing the second as its
own, both named "swap · …", the last one's finish line "Job 2 of 2 of this
swap, at 1024×1024."; the new images' prompts, seeds, sizes and rewritten
variables; the same swap again all already rendered, held with "No image
would change"; the negative prompt field; + Add row held at sixteen rows;
Escape; the page never scrolled; then a search for one image, and a second
matching image landing while the dialog is open, checked again on its own
with Queue and the scope line saying two; and Enter pressed straight after
typing a row, before its check, queueing it without ever saying "Checking
the swaps…":

```powershell
uv run python tools/fake_server.py --port 7900 --model cagliostrolab/animagine-xl-4.0 --data <new-scratch-directory>
node tools/browser.mjs tools/smoke-swap.json
```

`smoke-refine-batch.json` checks Mass refine on a FLUX.1 Krea [dev] fake on
7901 with a fresh `--data` (four images, three starred, and the fake's two
of another checkpoint): Refine… after Swap tags… and before More seeds… at
their height, held with its reason and saying it, then titled by Starred,
nothing in the footer or the grid moving; the dialog opening on the
server's answer - its title's count, the scope line, Strength focused and
selected, 1.5× pressed and titled with the largest image's pass, every
image listed with the pass's size, the count line with the start-sigma
range (stood in for, the fake reads none) and "no estimate: 2 sizes…";
a list in Strength refused on the page and a press saying it at Strength,
Refine and Cancel unmoved; "Checking the images…" while typing, ↑ stepping
0.05; Steps 500 refused; 2× checked again; the POST held as "Sending…"
with Escape waiting; the Now line and the queued row named "refine ×1.5
at strength 0.3", Cancel's "Stop this batch refine", the last finish line
"Job 2 of 2 of this batch refine, at 768×384."; each pass of its own
image with its own seed and prompt; the scale and strength remembered;
the same again all already refined, Ctrl+Enter saying the hold; All
models offering Include other checkpoints and its row's note; picked
cards over the filter, the opening scale capped by a 1024x1024 image with
2× saying it was last run there; an image deleted while open listed as
deleted; Steps "1e" refused rather than read as blank; Refine held with
"The server is not answering" as soon as the page goes offline, and freed
when it answers; a starred image landing while open checked again on its
own, title, scope line and Refine saying four, then three; with the model
picker opened under the dialog, Escape closing the dialog and not the
picker; the page never scrolled:

```powershell
uv run python tools/fake_server.py --port 7901 --model black-forest-labs/FLUX.1-Krea-dev --data <new-scratch-directory>
node tools/browser.mjs tools/smoke-refine-batch.json
```

`smoke-more-seeds.json` checks More seeds on a FLUX.1 Krea [dev] fake on
7902 with a fresh `--data` (a Batch 2 x Repeat 2 at seed 100, a single
256x384 image at seed 7, a single one at seed 50): More seeds… after
Refine… and before Bundles… at their height, held with its reason and
saying it, titled by a search and then by picked cards, the footer and the
grid unmoved once the search is cleared; the dialog opening on the
server's answer - "8 more images", the scope line, the count at 4 focused
and selected, one picked image of the four taking all four ("4 images,
seeds 100–103 → seeds 104–107 · Batch 2") beside the single one, the
count line, Queue's label and title; a count of 0 refused on the page, a
press saying it and marking the box, Queue's right edge, Cancel and the
list unmoved; "Checking the settings groups…" while typing, ↑ and ↓ stepping it;
Ctrl+Enter queueing one job per run, the Now line "more seeds · …",
Cancel's "Stop these more seeds", the last finish line "seeds 8–10. Job 2
of 2 of these more seeds, at 256×384."; the new images in the run they
extend, each recording its call's Batch (2, 2, then 1); the four's run
seven images, seeds 100–106; asked again from an image of the
extension, the seven carried on at Batch 2 (seeds 107–109); the single
image deleted while listed shown as deleted, the count line and Queue
saying what is left; Escape closing it with focus on More seeds…; a
remembered count of 600 over two picked batches opening the dialog on the
server's refusal by count, said in the count line and `#more-seeds-error`
with the box marked and Queue held, and a count of 2 freeing it; then where
a run ends: a plain Batch 2 job's finish line with More seeds… after Use
seed, opening on "From the 2 images this job made." at their Batch 2 rather
than the remembered count, Escape giving focus back to it; More seeds of
this batch… after Use seed in an unpicked card's menu and a result tile's,
each opening on that batch alone at its own Batch; the lightbox's More
seeds… beside Reuse (one row, a half each, uncut, nothing moving as it
steps), the dialog over the lightbox and Escape closing the dialog alone
with focus back on the button; another checkpoint's image held with "Made
on …; the server is running …", a press saying it in `#lb-error` and
nothing in the gallery's line; the sheet's run bar under Starred (More seeds
of this run… after the buttons, the run's line "Run … · 1 starred image ·
…" last on a row of its own), its dialog on every image of the run with
the server's headline "N more of its one cell = … · Starred ignored", and
its Queue adding that many images to the run; and the banner's More seeds…
beside Show run, opening on the run:

```powershell
uv run python tools/fake_server.py --port 7902 --model black-forest-labs/FLUX.1-Krea-dev --data <new-scratch-directory>
node tools/browser.mjs tools/smoke-more-seeds.json
```

`smoke-bin.json` checks the recycle bin on a fake of the default checkpoint
on 7903 with a fresh `--data`: ⌦ binning with no dialog (nothing asked),
the tombstone "In the recycle bin · #N" and the note's Restore putting the
card back in its own box; the lightbox's Del binning one image, a held key
binning no more, and the note's Restore surviving its own press; its
Delete pressed twice with the mouse gathering one note, and a held Enter's
repeat on it prevented; the bin view - Bin's count and title, the head's
"Recycle bin" and its "(N)", the held
controls and their titles, a card's Restore, Delete permanently with its
confirm, and Empty recycle bin with the count, its "Empty bin" label, the
danger colour, Cancel, a 409 when the bin changed (its count, and the same
count of other images: the pin) and the empty state
"The recycle bin is empty."; leaving the bin by a second press of Bin, by
a search from an image (`showGallerySearch`) and by a run opened as a sheet
(`showRun`); three gallery picks set aside while the bin is shown (its own
pick in their place) and back on leaving it, less one of them binned
meanwhile and one binned behind the page's back that the bin's view listed,
the bin's pick not brought along; the bundles dialog's preview
line counting the bin; a Delete
and its Restore in a lightbox walking the viewer's strip keeping the image
in view; and a viewer tile's "In the recycle bin" tombstone becoming
"Deleted · #N" once the bin is emptied, from another tab (a Restore then
answering "already deleted") and from the page; Restore picked over more
than six images saying "Restored N images." and a Restore of seven no longer
in the bin "7 images were not in the recycle bin."; Delete picked
permanently; and Empty's age scope (`#bin-age`, rows backdated through
`POST /api/fake/bin-age`): All by default, Older than 30 days counting one
image with "N images binned since stay" and its OK "Delete older", an image
backdated past the fixed cutoff while asked refusing it (409), Older than 7
days deleting only the two old ones as "Deleted" tombstones, and nothing
old enough said without a question. Re-run
`smoke-select.json` at 600 and 1400 px wide after a footer change, so the
row with Bin does not wrap:

```powershell
uv run python tools/fake_server.py --port 7903 --data <new-scratch-directory>
node tools/browser.mjs tools/smoke-bin.json
```

`smoke-png-settings.json` checks Load settings from PNG on a fake of the
default checkpoint on 7904 with a fresh `--data`, with PNGs built in the
page - signature, IHDR, hand-written tEXt, zTXt and iTXt chunks with their
CRCs, one IDAT pixel, IEND - and one the fake rendered (Pillow writes its
CJK 'parameters' line as iTXt): that PNG dropped over a half-written prompt
filling every setting with Reuse's note, its Undo and Ctrl+Z, the page
unmoved, and the posted body only `{name, chunks, panel}`, the panel's
sampling; another checkpoint's
hand-written record bringing Reuse's other-checkpoint sentence; an A1111
block in tEXt filling the prompt, negative and size while steps, CFG and
the seed stay the panel's, its note naming what does not carry over and
what is not read and saying nothing of the negative at the panel's CFG
1.5, an XMP packet beside it sent by its name alone, and at the panel's
CFG 0 the negative greyed with the note saying so; the same read from
zTXt and a CJK iTXt; a JPEG, a PNG with no text, a ComfyUI workflow and
two files refused in words with the panel untouched; a gallery card's
image dragged with the mouse onto the prompt box bringing no outline, no
request and no words, and a file dropped after it read; Load settings
from PNG… after Generate and directly over `#error`, opening the picker,
a picked file loading and the same file picked again reloading; the drag
outline and its words, gone on leaving the window, and none and no drop
over a modal dialog; the gallery's count unchanged; the button with its
hint on one line in the panel below Generate, no sideways scroll, at
1400, 2560x1305 and 600 px; and at 2560x1305 a picked A1111 PNG's note
written under the button, which stays where it was:

```powershell
uv run python tools/fake_server.py --port 7904 --data <new-scratch-directory>
node tools/browser.mjs tools/smoke-png-settings.json
```

`smoke-inpaint.json` checks the Inpaint dialog (`inpaint.js`,
`inpaint-mask.js`, `styles/inpaint.css`) on an Animagine fake on 7905 with a
fresh `--data`: the card's ✎ live and last before Delete; opened from the
lightbox's Inpaint…, the lightbox closed, focus on the canvas named "Mask
over #N" with a bitmap of the source's size; Strength at the inpaint class's
own default, said as the library's number and as measured on Animagine XL
4.0 (`hints.inpaint_strength`); a stubbed `start_sigma` adding "starts at
51% noise" to the preview, the SDXL fake's none adding nothing; Inpaint
held with "Paint the part
of #N to repaint first" and a press saying it; a brush drag, the eraser (E),
[ and ], a plain wheel scrolling and Alt+wheel sizing, the feather giving
another hash and 0 the same one again (`created: false`), Invert (I), Ctrl+Z,
Ctrl+Y and Ctrl+Shift+Z, I in the Seed box painting nothing, the keyboard
brush (arrows and Space); the tools row's boxes and the dialog's scrollTop
unchanged through painting and a preview refusal, Inpaint not moving under
the pointer; the result an inpaint of #N @ 0.6 on the card, its title and
the settings line; the lightbox's Mask row, the outline drawn over the
source's thumbnail in an 82 px box (80 and its border), its tint's share
within 0.12 of the mask's, the hash's first 16 digits, a binned source's
thumbnail asked for with `?binned=true`, and a deleted source leaving the record's own picture under it; every pixel the mask
keeps the source's exactly; Reuse reopening on the source with the stored
mask, no upload of the picture and one `/keep` of the stored mask (the page's POSTs counted); a strength list
opening a sheet over one mask; a close that ran nothing giving the lightbox
back and its strokes on the next open, and Start over. `smoke-inpaint-
unavailable.json`, on the default Krea 2 Turbo fake on 7906: the card's, the
result's and the lightbox's entry points `aria-disabled` with "Krea 2 Turbo
has no inpaint pipeline in the installed diffusers", a press saying it and
opening nothing; a stubbed `inpaint_deferred` status saying "not enabled yet"
in the titles; Reuse of an inpaint opening the dialog held, saying why:

```powershell
uv run python tools/fake_server.py --port 7905 --model cagliostrolab/animagine-xl-4.0 --data <new-scratch-directory>
node tools/browser.mjs tools/smoke-inpaint.json
uv run python tools/fake_server.py --port 7906
node tools/browser.mjs tools/smoke-inpaint-unavailable.json
```

`smoke-inpaint-unavailable.json` also holds the lightbox's Outpaint… and
the result column's ⬚ Outpaint… with "..., which an outpaint runs through",
the deferred wording, Reuse of an outpaint opening the dialog held, and
`openOutpaint` refusing. `smoke-outpaint.json` checks the Outpaint dialog on
an Animagine fake on 7907 with a fresh `--data`: the lightbox's row of three
not wrapping or cutting a word; opened from it with Strength at the class's
default and the note naming Animagine XL 4.0 as measured, the band at 32
and its note saying 32 joined best of 0, 32, 64 and 128; a stubbed
`start_sigma` adding "starts at 60% noise" to the preview, and the SDXL
fake's none adding nothing; Outpaint held with "Extend at least one side" and a press saying
it; a typed 50 committed as 48 on Enter with "left 50 → 48, a multiple of
16"; the server's presets over 1024 x 1024 (+128 all round, +400/+0/+400/+0,
+0/+256/+0/+256), the To shape… list, held presets on the 16:9 preset image
and on 1376 x 768 (within 16 px), a stubbed off-grid answer; the frame's
source and band placement; holds past the side budget, a preview refusal, a
band leaving nothing kept, a one-sided band leaving 8 px (held at
`min_kept`, naming the left side alone, as the server's preview refusal
does) and a band past 256; To shape… moved by real arrow presses without
applying, Apply held and applying; a side typed after a preset clearing its
words; every box's rect and the
scrollTop unchanged; Escape closing only the dialog over the model picker;
Ctrl+Enter in a side box running the outpaint, not the panel; the result
an outpaint of #N +400/+0/+400/+0 whose kept pixels are the source's at
its offset and whose border is wholly masked; the card's badge, title and
Reuse outpaint… after `syncReuseButtons`; the facts rows, and the Mask
row's thumbnail at the source's offset in a box of the canvas's shape, its
line only on the band's kept side; Against source
at the offset; Reuse reopening with the sides, band, strength and seed; a
strength list opening a sheet; an unknown fill refused by name; Start
over; `outpaint:yes`; the source binned with the dialog open (Outpaint
held with the recycle-bin words, a press saying it) and restored; a submit
in flight holding Outpaint with its reason and a press saying it, and the
job line's "the outpaint of #N"; a late presets answer opening nothing after
the lightbox closed or over the Inpaint dialog; a presets 404 for a source
binned elsewhere saying recycle bin; the lightbox's Outpaint… kept, uncut,
at 800 px. `browser.mjs`'s `{"key": "ArrowDown", "native": true}` sends the
key code, so a focused select moves as on a real press:

```powershell
uv run python tools/fake_server.py --port 7907 --model cagliostrolab/animagine-xl-4.0 --data <new-scratch-directory>
node tools/browser.mjs tools/smoke-outpaint.json
```

`smoke-context-menu.json` checks the right-click menu on a FLUX.1 Krea
[dev] fake on 7908 with a fresh `--data`: a card's menu at the pointer, its
items and groups, Refine, Inpaint and Outpaint live and Use as reference
held with the card button's title. Edit by instruction… follows Outpaint…,
live on a server whose editor is held and titled with
`instruction_edit_reason`. Chosen, it opens its dialog with that reason and
Edit held in those words. F sits beside Star; ArrowDown and Escape with
focus back on the image; Star starring the card; Use seed; the copy items
and Open PNG (the clipboard and `window.open` stubbed); a held Use as
reference saying why; FLUX.1's inpaint (enabled 2026-10-19): Inpaint…
opening the dialog at the profile's 0.95 with its note in the server's
words, a stroke and a run giving an inpaint at 0.95, and Outpaint opening
at the same 0.95 and note; Shift and
the prompt box keeping the browser's menu; the Menu key on a focused card
opening on its first item; a picked card's picks menu starring both picks
and an unpicked card's own menu; a press outside closing it; a result
tile's menu; and in the lightbox the menu inside the dialog, no Open, Del
beside the bin, a press on the picture beside the menu (a quarter of the way
across, since the menu now reaches below the picture's middle) closing the
menu and not the lightbox,
Escape closing the menu only, Use seed saying so in the lightbox's line,
and Show run as sheet closing the lightbox for the sheet; More seeds of this
batch… after Use seed on a card (unheld, titled with its batch) and a
result tile, and in the lightbox mirroring its More seeds… button, chosen
there opening the dialog over the lightbox, which Escape leaves open. `browser.mjs`'s
`{"rightclick": ...}` presses the right button (`"shift": true` holds
Shift):

```powershell
uv run python tools/fake_server.py --port 7908 --model black-forest-labs/FLUX.1-Krea-dev --data <new-scratch-directory>
node tools/browser.mjs tools/smoke-context-menu.json
```

`smoke-instruction-edit.json` checks Edit by instruction… on a fake claiming
FLUX.1 Kontext [dev] on 7931 with a fresh `--data`. Only a Kontext fake
starts with its gate open, and `POST /api/fake/instruction-edit
{"available": bool}` flips it. The fake plants a 3:2 and a 2:1 source beside
the 1:1 pair. The smoke checks:

- **Generate on the editor:** held with `text_to_image_reason`; a press says
  it and sends nothing, and the composer is unchanged.
- **Gate closed:** the menu item stays live and opens the dialog, which shows
  the reason with Edit held in those words. A press writes it in place and
  queues nothing. Open Models… opens the picker and switches nothing.
- **Gate open:**
  - opened from the lightbox, which closes, with focus in the instruction;
  - the recipe's 512×512, 28 steps and guidance 2.5, with no strength;
  - the draft from the closed gate kept;
  - an empty instruction holding Edit with its words, and Ctrl+Z taking
    typing back;
  - Escape giving the lightbox back;
  - the 2:1 source's aspect refusal shown in place;
  - a stale revision getting 409;
  - a stale preview (the gate closed behind the page) asked again when Edit
    is pressed, its refusal said in place and no edit sent;
  - the dice clearing the seed with the kept draft's note still shown and
    the dice, instruction and Edit unmoved.
- **A run:** Ctrl+Enter queues it, and the progress line and queue row read
  "instruction edit of #1". A second edit queued behind it is cancelled from
  the queue. The saved record is an edit of #1 with its 64-hex proof, the
  Kontext id, 512×512, batch 1, and no refine, mask or negative. The viewer
  lineage, the "Edited from" fact and the `edited` badge show, and the
  pixels differ from the source. A cancel mid-run saves nothing.
- **Binning the source:** the edit's Reuse is held, and the facts say "in the
  recycle bin" with Restore. The source's own menu item is held with
  "restore it first", and Restore frees both.
- **Reuse:** reopens the dialog on the source with the instruction and seed,
  and the composer is untouched.
- **Layout:** the page neither scrolls nor moves. At 800x900, 1400x700 and
  1900x1000 the dialog fits with Edit in view. No page errors.

```powershell
uv run python tools/fake_server.py --port 7931 --model black-forest-labs/FLUX.1-Kontext-dev --data <new-scratch-directory>
node tools/browser.mjs tools/smoke-instruction-edit.json
```

`smoke-tag-finder.json` checks Find tag on an Animagine-claiming fake on
7909 (the fake's `describe` searches a three-page wiki's text and the name
typed, as the real one does; "fake offline" is the index unreachable): no
button and a one-line caption, no finder in the helper, the list's find
row only from three characters, last, unselected, and alone where no name
matches, ArrowDown and Enter on it opening the finder under the prompt
searched with the words in its box, `ahoge` for "stupid hair" with its
sentence and count, Enter on it (`"native": true`, which also runs a
form's own submit) replacing the description with focus kept and Enter
again taking it out, a later press inserting at the caret, a description
changed meanwhile left whole, an empty search's line, results by count and
a page with no sentence, the exact name first, nothing found and offline
each clearing old results, a click in the prompt keeping it open, Escape
with focus back in the prompt, the row's mouse click opening it, and the
helper, Examples and a click elsewhere closing it:

```powershell
uv run python tools/fake_server.py --port 7909 --model cagliostrolab/animagine-xl-4.0
node tools/browser.mjs tools/smoke-tag-finder.json
```

`smoke-analysis.json` checks the prompt check on a FLUX.1 Krea [dev] fake on
7874: the CLIP cut, the negation and quote findings with their sources, span
and cut selection (an emoji ahead of them included) with the selection
scrolled into its box, the stale rule for both prompt boxes with the status
slot undimmed while stale, Apply, the critic's first-use and timing lines -
built from `ModelStatus.prompt_critic`: its label, the downloads a first use
makes, neutral wording under an override - and its clock, its hold on
Generate, inline cites, a cite's whose-words note and the General label, Use
rewrite with undo and a reload and filling Your version, an edit while the
critic thinks, a failed critique's row in one slot (`#error` only after
Close), a two-version comparison opening its sheet with a blank Seed left
blank, a Generate under that sheet's `job:` scope counted in the gallery's
line with Show all opening the whole gallery, another opening all its
cells with Starred ticked (the tick's notice gone once the run is
reopened), the chips and the check moving from the wide host back under the
prompt with focus and selection kept, a prose prompt's order shown as not
checked, a one-clause prompt the critic cannot reorder, a comparison against
Your version, the cut excerpt in code points, focus after Check
(two-column), after Check again, after Close during the critic's check and
after the critic's answer, the critic's hold through its pre-check (a
replaced one saying so, a cleared one silent, a throwing one giving Check
back), the stale badge left alone while typing, the Seeds select read at run
time, by injecting the server's `reordered_kind`, the Use reordered title
following what the server says it kept, not the page's own last check, with
Compare offering the critic's reordering the server kept, and the docked
check at 2560x1305 and 2048x1040@1.25 (the image just made wholly in view
and unmoved after Check, the critic's answer and Ctrl+Enter; the column not
setting the rows' height; its head stuck after the answer, showing the stale
badge; Check again resetting its scroll; the gallery's controls clear after a
400 px page scroll; Close on the head's first line at 125%), the critic's two versions stacked there and side by
side full width at 1400 px, and the stacking at 1400x900; the 600 ms outline
on the box a finding selects in, and none under an emulated reduced motion,
Apply keeping the replaced prompt in Your version, the check capped at 55vh
above the results at 1400x900 with its head stuck, Escape closing the check
(focus to Check from inside it; the Examples menu, a search box with text
and the lightbox first; the caret kept from the prompt); the critic's
line's live half only, its ready line at once after an answer, and
Generate's, Run grid's and Run comparison's titles while the critic holds
them; the page's h1, the Settings region, the viewer as a region named
Results with no unseen heading in the stage, the greyed negative's
description, each finding's and critic issue's Show with no item a tab
stop, the always-run Original's description; and, docked at 2560x1305, Tab
from the panel to the image just made, its note (a stand-in stop),
`#recent` where it sits between, the check, then the gallery, with nothing
in the docked stage out of its flow; the comparison's Batch tip hung from
the window, whole and on top with its row under the sticky head at
1400x600, following its badge as the check scrolls, hidden with the badge
scrolled out, and docked at 2560x1305; the Refine dialog's Refine held,
and said in the dialog, while the critic thinks; the critic held offline
(its title, the cost line, and the press's offline words taken back by the
answer) and during a load (its note gone with the load), and a critique
the server never answered said as the server's (no "?." after its
question) and taken back; a 900-word prompt's line over 4000
characters, its cut and its summary; and, docked at QHD, the first use's
download sentence and the clock as one run of text, the clock on its last
line.
`smoke-animagine.json` covers the tag-order check, the critic's reorder
under full and partial tag placement end to end (set aside, kept as Check's
own, kept as allowed, flagged, each with the kind the server says it kept),
the partly-checked order (past 64 tags, without the tag index, a tag the
index does not know, a prose prompt) and the checked one where no missing
category decides it (that tag ahead of a general tag, alone, one prose
clause), and the negative prompt's Apply, dead while the box is greyed,
and its Show, held with the reason. The fake server
serves the tests' `FakeCritic` and word tokenizers: `--critic-delay` (seconds
a critique takes, default 0.4, so its in-flight state can be seen; a prompt
too long for the critic's window is refused at once, a 502, on the
unscaled estimate - the real critic refuses at once only what cannot fit
however it tokenizes, and the rest on its server's count) and
`--no-critic` (a platform with no build: the page hides the critic). The fake
critic counts as downloaded from its first critique on, so the first-use line
shows only on a newly started server; the smoke accepts either line:

```powershell
uv run python tools/fake_server.py --port 7874 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-analysis.json
```

`smoke-rewrite.json` uses the same fake on 7874 for the visible rewrite diff,
the change's grouped gist, escape-only edits, Unicode, markup rendered as
text, unchanged/absent rewrite states, and the rewrite findings' "also in
your prompt" / "new" / "No longer flagged" / "Not flagged … asks the tag
index nothing of its own" marks.
`smoke-comparison.json` uses the same FLUX-shaped fake on 7874 to run
two comparisons with original and yours reversed on fixed seeds. Each
opens its own job, while opening their shared run exposes Prompt as an
axis. Run comparison clicked at 800x900, and at 2560x1305 under the docked
check, hands focus to the sheet's first cell (named by seed and version)
without moving the page, and focus moved to the prompt while the
comparison renders stays there; and with the grid open declaring `{size}`, a
version holding it refused in the preview and at the press with nothing
sent, an undeclared name or the fold shut previewing as usual. `smoke-comparison-batch.json` uses the same
fake for all six Seeds/Batch
pairings, their preview and request counts, select values changed without
events, paired output seeds, sheets and stale-run disabling. API tests cover
paired saved seeds, batch grouping and preview counts for all six pairings.
The analysis and Animagine smokes cover CFG/True CFG activation
changes while a grid keeps the negative editor enabled, inactive raw
edits, and settings changes while a check or critique is in flight.

`smoke-analysis-tags.json` uses an Animagine-claiming fake on 7887 for the
prompt check on a tag checkpoint at QHD: a Series-before-Character order
finding the rewrite keeps, listed as possibly still applying ("whose check
asks the tag index nothing of its own") rather than fixed, and the docked
check with the chips across both columns at 2560x1305 and 2048x1040@1.25,
where the image does not move and stays in view (6, 32 and 40 tags at
each), and Tab docked with chips going chips, image, its note or
`#recent`, check; beside an idle viewer the check's bottom in the window
with the critic's answer in it; with Gallery beside, the check in the
gallery's column over an idle viewer with the critic's answer (scrolling,
head stuck, Close under the pointer, the chips still, the page unscrolled)
and beside the image just made (the image, the chips and Generate
unmoved); at 1400x900 with 40 tags and the answer the check capped under
the chips, its head stuck and the image starting in the window's top three
quarters; at 1280x720 with 40 tags and the answer the cap 55% of the
window less the chips, the check at least 8rem and the image starting in
the window's top three quarters; and at 1280x600 the floor itself, the
check 8rem tall with its head stuck and Close under the pointer, the image
in the window's top 85%:

```powershell
uv run python tools/fake_server.py --port 7887 --model cagliostrolab/animagine-xl-4.0
node tools/browser.mjs tools/smoke-analysis-tags.json
```

`smoke-stage.json` checks Gallery beside on a FLUX.1 Krea [dev] fake on
7888. It opens at 1400x900 (no tick, no column), then at 2560x1305 checks
the tick off by default; ticked, a fixed column level with the viewer and
inside the window, the page not scrolling, three cards across with their
tools whole, the search on its own row, the paging trigger rooted on the
column, the tick remembered; the column paging by the wheel with the window
still; the arrow keys walking its rows, End bringing the last card into the
column's view, and Right past the last card paging in; with nothing made
yet (the viewer idle), the check and the critic's answer in the column's
box, scrolling inside it with its head stuck and Close under the pointer,
the page still (its grid, posted by fetch and so another tab's job to the
page, waited out first, since that job holds the critic); a job landing
with the pointer resting on a card (hovered by selector) changing nothing
in the column, with the count already new, until the pointer leaves, and
then first at the column's top; the image just made at its own size beside
the column; a batch landing above a scrolled column keeping its view;
`#recent` as a row under the image; the prompt check taking the column's
box, with the column hidden and the image, its column of words and
`#recent` unmoved and nothing cut; Tab reaching the check in the column
after the image just made, its note and `#recent` (`reading-flow:
flex-visual`); the lightbox's Escape from an image only the hidden column
holds focusing the viewer's image; Close giving the column back at the same
scroll; a job landing under the check, not held; the Sheet button and a
card's ⊞ opening a sheet across the stage's width, landed on, with the
tick's title saying when it counts and the column back on Sheet off; Reset
leaving the tick, a reload restoring it, and the window crossing 2000 px
both ways; at 2048x1040 at 125%, two cards across beside the image in
view, the check and the critic's answer in the column's box over an idle
viewer after the reload, and the check taking the column with the image
unmoved and nothing cut; and, unticked, the gallery back under the viewer,
with focus on the tick and the key forgotten:

```powershell
uv run python tools/fake_server.py --port 7888 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-stage.json
```

GPU generation, repeat-seed pixels, cancellation and any preview evidence
are recorded per checkpoint in `docs/verified.md`, and the real critic's runs
under its "Prompt analyser" section; CPU and browser checks alone do not
establish runtime quality or performance. WAI's conditioning
regression is `tests/test_wai_conditioning.py`: both positive and negative
SDXL conditioning must remain finite with its unused final CLIP-L NaNs
under none, FP8 and int8 quantization.


Static files are served from disk, so a `web/` edit needs no server restart;
a `backend/` edit does. The app's top-level bindings (`els`, `lbView`,
`sheetRecords`) are reachable from a step's `eval` because the page's scripts
are classic scripts sharing one global scope - which is also why they stay
top-level. `browser.mjs` gives
each run its own profile directory: a shared one stays locked by an Edge a
crashed run left behind, and every later launch then hands its command line
to that instance and reports "the browser did not come up".
