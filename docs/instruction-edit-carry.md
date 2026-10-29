# Instruction editing: carry checklist

**Status, 2026-10-28:** the CPU persistence/admission slice and stage 2 are
built; the state column records remaining boundaries. The
[feature plan](plan-instruction-editing.md) owns release acceptance. The six
pinned configs were fetched at the revision and qualified, and their hashes
are pinned. The editing call, the queueing route, the per-cell recheck and the
dialog are [built on a stand-in](plan-instruction-editing.md#stage-2-backend-and-dialog-2026-10-27):
a Kontext-shaped stub, the fake server and a real-browser smoke. On the real
server the gate is open since stage 4's release (slice 7, 2026-10-30; local
until the card acceptance step): the measured envelope admits only the exact
Kontext id on fp8, model offload and Windows, with 1:1 sources at 512x512.
Stage 3's approved [first card trial and bounded pair](verified.md#instruction-edit)
ran on 2026-10-27 at 512x512, fp8 and model offload; it fits, replays, cancels
and unloads, with drift beyond the instruction. Other sizes and configurations
remain unqualified. The catalogue entry and the Re-render/More seeds execution
of edits are stage 4. No live gallery was used.

## Frozen first-slice boundary

`kontext-v1` means one live gallery source, the request's `prompt` as the
instruction, and one image per pipeline call. It selects public checkpoint
`black-forest-labs/FLUX.1-Kontext-dev` at revision
`24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d`. The six component configs at
that revision were qualified on 2026-10-27, and their canonical hashes are
`QUALIFIED_CONFIG_HASHES` (a config pin, not a weights pin). Ordinary
text-to-image,
references, Refine, masks, outpaint, upscalers, crop anchors, user LoRAs,
performance recipes, grids, Then refine and mass instruction editing are
outside this recipe. The first source may belong to another checkpoint;
the new child belongs to the editor. No mask composite promises unchanged
pixels outside the instruction's subject.

The server uses the source's **exact** aspect ratio. The permitted mapping
is frozen; a ratio outside this table is refused, without a nearest-ratio
tolerance or a hidden crop.

| Source ratio | Working and output size |
|---|---|
| 1:1 | 512x512 |
| 3:2 | 576x384 |
| 2:3 | 384x576 |
| 4:3 | 512x384 |
| 3:4 | 384x512 |
| 16:9 | 512x288 |
| 9:16 | 288x512 |

Convert the decoded source to RGB, then resize explicitly with Lanczos.
Do not EXIF-transpose, crop or reuse `refine.fitted`. Working size equals
output size, the maximum area is 262,144 pixels, and the editor call must
disable the library's automatic preferred-resolution resize. A future
change to this preprocessing or mapping requires another recipe identity.

## Request and stored source identity

A new request may carry the strict, versioned edit spec:

```json
{"edit": {"source": 17, "recipe": "kontext-v1"}}
```

`source` is a strict positive SQLite-range integer; booleans, strings, zero,
negative and overflowing ids refuse. Unknown fields/recipes refuse. The instruction remains
`prompt`, never a second text copy. Admission resolves and freezes the
source descriptor before retained work or rendering can claim that source.
The complete stored replay form below is also accepted for durable/internal
transport; its supplied proof is untrusted and must match current source pixels.

The nullable SQL `images.edit` value will store canonical JSON with the
descriptor inside it. PNG metadata carries the same stored edit object:

```json
{
  "source": 17,
  "recipe": "kontext-v1",
  "edit_source": {
    "filename": "source.png",
    "hash": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "width": 1024,
    "height": 1024
  }
}
```

The descriptor is mandatory and frozen. Its filename is a bare PNG
basename, never a path: no separators, traversal, drive/ADS spelling or
unsafe Windows basename. It locates an import candidate; it does not
authorize reading outside the gallery. Its hash is the **full 64-character
lowercase SHA-256** of the original decoded image converted to RGB, before
resizing, with the original dimensions alongside it. This is a new
edit-specific identity. Existing `storage.content_hash` and
`image_provenance` use a 16-character digest and cannot be directly compared
or silently reinterpreted as this descriptor; future admission/import must
compute the edit digest from the source pixels.

Only the stored type allows `source: null`: it records an imported or
reindexed result whose source could not be proven/remapped. Keep its
recipe, instruction and descriptor. Never keep a foreign positive id as
a usable destination link, clear the whole edit, or turn the instruction
into a direct render. Re-export must preserve unresolved identity from SQL;
it must not recompute provenance from an unrelated local row.

`edit_missing` and `edit_binned` are derived availability state, not
reproduction settings or key material. A null, absent, binned or
descriptor-mismatching source cannot replay. `cell_from_record` refuses
before constructing a strict request from an unresolved stored edit.
Restoration/remapping must prove the descriptor before assigning a usable
local id. A completed child remains inspectable after its source disappears.

The edit's canonical key uses the bound stored identity: source, recipe and
the full descriptor (filename, dimensions and RGB digest). Admission proves
and binds that descriptor **before** expansion/run ids, Resume/deduplication
and More seeds keys are computed. The raw request's id/recipe alone is not
a complete key. Replacing source pixels at the same id must produce a
different identity for new work; replay checks the old proof and refuses.

## Integration checklist

The file names below are implementation seams, not permission to skip
their area's rules/design notes. Bundle format and older-reader refusal
are owned by [the bundle checklist](instruction-edit-bundles.md).

| State | Carry site | Required implementation |
|---|---|---|
| BUILT | `schemas_generate.py`, `schemas.py` | Add the strict request field/public re-export and exclusive-field validation. Guard `applied()` and internal copied requests; an unsupported edit must refuse, never normalize into plain generation. Enforce one image per call and initial grid refusal. |
| BUILT | `launch.py`, `model_defaults.py`, `model_info.py`, `schemas_model.py`, `main.py`, `app_context.py` | Add explicit text-to-image and instruction-edit capabilities, an exact editor profile/recipe and cold held reasons. Preserve existing catalogue behavior; no capability follows merely from a callable's `image` parameter or a catch-all FLUX profile. Built: capability fields, the exact profile, the held reasons (`instruction_edit_reason`, `text_to_image_reason`) and the Models "Edit" note. Stage 4 slice 5: the exact profile written out key by key with its own sourced `KONTEXT_GUIDE`, the held reasons worded from a backend's live `edit_reason` (`defaults_for`, `details_for`, `main`'s listing), and the editor's compact facts, download reference, encoder rows and timing kept by its exact id. Stage 4 slice 6: the `launch.py` catalogue entry, last, its menu summary "edit only"; listed is not loadable (the gate and the envelope hold every load, and the switch refuses before the Hub, the save and the unload). Stage 4 slice 7: with the real gate open the row and the held reasons follow the envelope: inside it the row says Yes and the switch goes on to the Hub check and an ordinary load; outside it they say the first unmeasured choice. |
| BUILT | `grid.py: cell_key`, `run_id`, `cell_from_record` | Add the bound stored edit key, including complete descriptor/hash, immediately before execution. Keep refine's existing index, shape at `[-2]` and seed last. Explicitly remove an absent edit slot before existing execution/shape compaction, including mixed-size/noncenter runs. Carry the original edit source/instruction on replay; reject unresolved intent. |
| BUILT | `storage.py: _migrate`, `_COLUMNS`, `_row_to_record`, `save_image`, `index_file` | Add nullable `images.edit`, canonical stored JSON and strict readers. Update both explicit insert dictionaries and PNG/A1111 metadata handling. Preserve the nested source descriptor durably; absent edit stays SQL NULL and absent in old-style PNG metadata. Keep `.png.part`/commit/rename and gallery-lock ordering. |
| BUILT | `storage.py: completed_cells`; `more_seeds.py: settings_key`, `batch_key` | Read the same canonical bound edit key from rows as from admitted requests. Different source pixels/descriptor, instruction or recipe cannot count as direct/refine output or another edit. Keep existing no-edit keys and batch grouping unchanged. |
| BUILT | `submission.py`; `api_generate.py: expand_grid`, Generate, Re-render, Finish | Resolve/check source pixels, recipe, capability, exact geometry and model revision before admission/loading. Include edit in every distinct-input verdict cache, notably Finish's `spec` tuple and Re-render's grouped tuple. Refuse unsupported resized edits and never reuse another source's verdict. |
| BUILT | `api_more_seeds.py`, `more_seeds.py` | Carry the original edit through `cell_from_record`, include it in the cached input-verdict tuple, keep each call at one image and refuse unreplayable sources. Retained source/instruction must refer to the original input, not the edit result. |
| BUILT | `api_swap.py`, `schemas_rerun.py` | Refuse edit records with an explicit instruction-workflow reason before tag replacement or cell construction; include an edit-aware status where needed. |
| BUILT | `refine_batch.py`, `then_refine.py` | An explicitly requested ordinary Refine child of an edit result drops `edit` as it sets `refine=record.id`. Decide Skip passes wording explicitly. Then refine refuses edits, and its stand-in cannot inherit one. |
| BUILT | `png_settings.py`, `api_png.py` | Validate known edit intent without consulting the current selection. Initially refuse restoring an edit workflow from PNG settings with a route to the gallery image; do not populate a plain-render instruction that appears replayable. Malformed/unknown edit metadata does not fall back to A1111 settings. |
| BUILT | `reindex.py: merge_database`, `_database_meta`, PNG-only `reindex` | Validate SQL/PNG edit agreement before mutation and in existing-file fast paths. Prove/remap sources, preserve `source:null` plus its descriptor when unproven, rewrite changed ids even during in-place recovery, and report unresolved edits separately. Dry-run remains nonwriting. |
| BUILT | `generate.py` local and `--server` paths | Carry a verified edit or explicitly refuse it, including `--from-image`, overrides and dry-run. Never use `_check_carried_source`'s existing source-dropping fallback for edits. Keep CLI/API applied-cell equivalence and pre-config import ordering. Stage 4 slice 3: the local path refuses an edit-only checkpoint (`command_line_refusal`) before `_project()`. |
| BUILT | `jobs.py`, backend base/real/fake implementations, `api_instruction_edit.py` | Retain edit/source identity, refuse unsupported execution before model work and recheck the backend's gate plus the source on each cell (`_recheck_edit`). Name progress/snapshots honestly. Preserve callbacks, cancellation, cleanup in `finally`, terminal `_finish` and saving only completed results. Built against a Kontext-shaped stub and the fake (`test_instruction_edit_call.py`, `test_instruction_edit_queue.py`), with no card. The real backend's gate is a literal False, and Re-render/More seeds execution of edits is held until stage 4. Stage 4 slice 2: every queue answer also asks the measured envelope (`instruction_edit_refusal`: gate, exact id, `loader_refusal`; then `geometry_refusal`) at the preview, the submit, `submit_cells`, `_run`, `_recheck_edit`, `planned_keys` and `describe()`; nothing stored changes. Stage 4 slice 3: every load entry (`load`, `_load_prepared`, `_require_execution`, `ensure_loaded`, `/api/model/load`, the switch's `validate`) passes the gate and `instruction_edit_loader` to `require_model_available`, and `_require_execution` refuses a request without `edit` on an edit-only profile. Stage 4 slice 4: inside the envelope with a True stand-in, `_load_prepared` loads the pinned editor (`loading.resolve_model`'s configs pass and pin, then `FluxKontextPipeline.download` at the revision and the pin again; the explicit class; class, VAE scale factor and `_EDIT_PARAMS` required); the real gate is still False. Stage 4 slice 7: `DiffusersBackend.instruction_edit_qualified` returns the literal True, so the real backend loads the pinned editor and runs a 1:1 edit inside the envelope (`test_instruction_edit_release.py`); every other gate stays the literal False. |
| BUILT | `storage.py` source readers; `gallery_delete.py`; `jobs.py: forget_images`, `restore_images` | Derive missing/binned state on get/list/save and job replay. Include verified edit sources in starred-source preservation and bin sources-in-use counts. A source binned/deleted after Queue fails before editor work; restoration updates availability without altering intent. |
| PARTIAL | `gallery_reports.py`, `grid.py` estimation, runs/search/stats | Exclude edits from direct `refine IS NULL` timing samples. Use matching edit history or no estimate, and distinguish edit pass kind in summaries/filters/facts; ordinary render/refine histories cannot price an edit. |
| BUILT | `web/` dialog, actions, Models, facts, lineage and shared removal | Add the single-source workflow, held reasons, exact server geometry and source/result inspection. Reuse opens the edit dialog; preserve the composer draft and Ctrl+Z. Source removal follows `forgetImages`; no hidden switch/queue or page scroll. Update classic-script/CSS maps and smoke inventory. Built: `instruction-edit.js`, `styles/instruction-edit.css` and `smoke-instruction-edit.json`, checked against the Kontext fake in a real browser with no card. A wide-tier viewer Edit tool is a follow-up. |

## Acceptance groups

`BUILT` means checks in the `test_instruction_edit_*` groups, plus a
real-browser smoke for the backend/browser row. None of it qualifies a runtime
or the card. Stage 2 adds the editing call against a Kontext-shaped stub, the
dedicated queueing route, the fake server's edit and the dialog. On the real
server the editor is listed (stage 4 slice 6) and, since slice 7, the
backend's gate is the literal True: the measured envelope holds every other
load and source. `PARTIAL` rows keep the requirements in the table
above. Reports have edit filters and grouping and give no estimate rather
than a wrong one; facts now show an edit's lineage and recipe.

| State | Group | Required evidence and existing patterns |
|---|---|---|
| BUILT | Contract and normalization | Strict request/stored types; unknown/malformed JSON, recipes and fields; all exclusivities; exact-ratio table and RGB/Lanczos pixels; full source digest before resize; no EXIF transpose; unsupported capability/copied-request refusal. Pure contract tests establish none of the routes or storage below. |
| BUILT | Legacy identity and persistence | Pin existing direct/refine/inpaint/outpaint/upscaler/execution/mixed-size/noncenter run ids. Compare row-derived and request-derived keys, migration on replaced old DBs, SQL/PNG round trips and absent-field compaction. Patterns: `test_conditioning.py`, `test_size_crop.py`, `test_crop_anchor_storage.py`, `test_execution_storage.py`, `test_inpaint_storage.py`. |
| BUILT | Source portability and recovery | Proven remapped source; omitted, absent, binned, forged, colliding-id or descriptor-mismatching source; unresolved re-export; SQL/PNG disagreement; repeated import and in-place reindex; dry-run; old-reader refusal. Patterns: `test_reindex_provenance.py`, `test_reindex.py`, `test_execution_bundles.py`, `test_inpaint_bundles.py`; also the bundle checklist. |
| BUILT | Producers and admission | Re-render/More seeds preserve original source; cache verdicts cannot cross source/recipe; Swap refuses; Mass refine changes pass kind; Then refine/grids refuse; Finish retains complete intent; missing source before/after Queue and stale model revision fail before loading/rendering. Patterns: `test_execution_replay.py`, `test_api_more_seeds.py`, `test_refine_batch.py`, `test_then_refine.py`, `test_cli.py`, `test_api_png.py`, `test_jobs.py`. |
| BUILT | Source lifetime and reports | Starred edits protect proven sources; bin/purge counts include them; deletion/bin/restore updates records and job snapshots; completed children stay visible and unreplayable when needed; edits never contaminate direct/refine estimates. Patterns: `test_gallery.py`, `test_gallery_purge.py`, `test_api_gallery.py`, `test_jobs.py`. |
| BUILT | Backend and browser | Stub exact kwargs, one-image generators, callbacks/cancel and failure cleanup; fake edit dialog smoke with composer draft/Undo, stale preview/revision, focus and narrow/wide layouts. Ruff, pinned whole-source types and relevant CPU regressions follow source integration. Built 2026-10-27: `test_instruction_edit_call.py`, `test_instruction_edit_queue.py`, `test_instruction_edit_gate.py` and `smoke-instruction-edit.json` on the Kontext fake (port 7931). The smoke presses Edit on a preview made stale behind the page (it asks again and sends no edit), checks a raw stale-revision 409, the kept draft's note not moving the controls, and the dialog at 800x900, 1400x700 and 1900x1000. Stage 3 measured the first 512x512 fp8/model-offload slice; stage 4's release review, nonempty admission and Re-render/More seeds execution of edits remain. Stage 4 slice 2 moved the smoke's runs and stale preview to the 1:1 sources and holds the 3:2 one as unmeasured; the envelope tests are in `test_instruction_edit_queue.py`. |

Windows browser access was restored on 2026-10-27
([tooling](plan-tooling.md#browser-smokes-in-ci)). `smoke-instruction-edit.json`
passes in a real Edge against the Kontext fake, and `smoke-context-menu.json`
checks the live menu item and its held dialog on a Krea fake. This qualifies
the page and the fake's path only. It says nothing about the model, its
pixels or the card.
