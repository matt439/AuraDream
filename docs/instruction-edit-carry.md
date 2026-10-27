# Instruction editing: carry checklist

**Status, 2026-10-27:** stage 0's implementation inventory and acceptance
matrix for `INP#11`. The [feature plan](plan-instruction-editing.md) owns
the stages and release decision. Every carry/integration entry below is
**TODO**. The current bounded source slice is the pure contract; it does
not add a database migration, catalogue selection, route, loader branch or
dialog. Stage 1 is not complete, and real editor execution remains unavailable.

## Frozen first-slice boundary

`kontext-v1` means one live gallery source, the request's `prompt` as the
instruction, and one image per pipeline call. It selects public checkpoint
`black-forest-labs/FLUX.1-Kontext-dev` at revision
`24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d`. This is a metadata pin;
the gated checkpoint's actual component/config files have not been
qualified in this slice. Ordinary text-to-image,
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

A new request carries only the strict, versioned edit spec:

```json
{"edit": {"source": 17, "recipe": "kontext-v1"}}
```

`source` is a strict positive integer; booleans, strings, zero and negative
ids refuse. Unknown fields/recipes refuse. The instruction remains
`prompt`, never a second text copy. Admission resolves and freezes the
source descriptor before retained work or rendering can claim that source.

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
| TODO | `schemas_generate.py`, `schemas.py` | Add the strict request field/public re-export and exclusive-field validation. Guard `applied()` and internal copied requests; an unsupported edit must refuse, never normalize into plain generation. Enforce one image per call and initial grid refusal. |
| TODO | `launch.py`, `model_defaults.py`, `model_info.py`, `schemas_model.py`, `main.py`, `app_context.py` | Add explicit text-to-image and instruction-edit capabilities, an exact editor profile/recipe and cold held reasons. Preserve existing catalogue behavior; no capability follows merely from a callable's `image` parameter or a catch-all FLUX profile. |
| TODO | `grid.py: cell_key`, `run_id`, `cell_from_record` | Add the bound stored edit key, including complete descriptor/hash, immediately before execution. Keep refine's existing index, shape at `[-2]` and seed last. Explicitly remove an absent edit slot before existing execution/shape compaction, including mixed-size/noncenter runs. Carry the original edit source/instruction on replay; reject unresolved intent. |
| TODO | `storage.py: _migrate`, `_COLUMNS`, `_row_to_record`, `save_image`, `index_file` | Add nullable `images.edit`, canonical stored JSON and strict readers. Update both explicit insert dictionaries and PNG/A1111 metadata handling. Preserve the nested source descriptor durably; absent edit stays SQL NULL and absent in old-style PNG metadata. Keep `.png.part`/commit/rename and gallery-lock ordering. |
| TODO | `storage.py: completed_cells`; `more_seeds.py: settings_key`, `batch_key` | Read the same canonical bound edit key from rows as from admitted requests. Different source pixels/descriptor, instruction or recipe cannot count as direct/refine output or another edit. Keep existing no-edit keys and batch grouping unchanged. |
| TODO | `submission.py`; `api_generate.py: expand_grid`, Generate, Re-render, Finish | Resolve/check source pixels, recipe, capability, exact geometry and model revision before admission/loading. Include edit in every distinct-input verdict cache, notably Finish's `spec` tuple and Re-render's grouped tuple. Refuse unsupported resized edits and never reuse another source's verdict. |
| TODO | `api_more_seeds.py`, `more_seeds.py` | Carry the original edit through `cell_from_record`, include it in the cached input-verdict tuple, keep each call at one image and refuse unreplayable sources. Retained source/instruction must refer to the original input, not the edit result. |
| TODO | `api_swap.py`, `schemas_rerun.py` | Refuse edit records with an explicit instruction-workflow reason before tag replacement or cell construction; include an edit-aware status where needed. |
| TODO | `refine_batch.py`, `then_refine.py` | An explicitly requested ordinary Refine child of an edit result drops `edit` as it sets `refine=record.id`. Decide Skip passes wording explicitly. Then refine refuses edits, and its stand-in cannot inherit one. |
| TODO | `png_settings.py`, `api_png.py` | Validate known edit intent without consulting the current selection. Initially refuse restoring an edit workflow from PNG settings with a route to the gallery image; do not populate a plain-render instruction that appears replayable. Malformed/unknown edit metadata does not fall back to A1111 settings. |
| TODO | `reindex.py: merge_database`, `_database_meta`, PNG-only `reindex` | Validate SQL/PNG edit agreement before mutation and in existing-file fast paths. Prove/remap sources, preserve `source:null` plus its descriptor when unproven, rewrite changed ids even during in-place recovery, and report unresolved edits separately. Dry-run remains nonwriting. |
| TODO | `generate.py` local and `--server` paths | Carry a verified edit or explicitly refuse it, including `--from-image`, overrides and dry-run. Never use `_check_carried_source`'s existing source-dropping fallback for edits. Keep CLI/API applied-cell equivalence and pre-config import ordering. |
| TODO | `jobs.py`, backend base/real/fake implementations | Retain edit/source identity, refuse unsupported execution before model work and recheck actual loaded capability plus source on each cell. Name progress/snapshots honestly. Preserve callbacks, cancellation, cleanup, terminal `_finish` and saving only completed results. |
| TODO | `storage.py` source readers; `gallery_delete.py`; `jobs.py: forget_images`, `restore_images` | Derive missing/binned state on get/list/save and job replay. Include verified edit sources in starred-source preservation and bin sources-in-use counts. A source binned/deleted after Queue fails before editor work; restoration updates availability without altering intent. |
| TODO | `gallery_reports.py`, `grid.py` estimation, runs/search/stats | Exclude edits from direct `refine IS NULL` timing samples. Use matching edit history or no estimate, and distinguish edit pass kind in summaries/filters/facts; ordinary render/refine histories cannot price an edit. |
| TODO | `web/` dialog, actions, Models, facts, lineage and shared removal | Add the single-source workflow, held reasons, exact server geometry and source/result inspection. Reuse opens the edit dialog; preserve the composer draft and Ctrl+Z. Source removal follows `forgetImages`; no hidden switch/queue or page scroll. Update classic-script/CSS maps and smoke inventory. |

## Acceptance groups

These groups define evidence to collect as the corresponding integration
lands. Existing tests listed here are useful patterns, not a claim that
they already cover edits.

| State | Group | Required evidence and existing patterns |
|---|---|---|
| TODO | Contract and normalization | Strict request/stored types; unknown/malformed JSON, recipes and fields; all exclusivities; exact-ratio table and RGB/Lanczos pixels; full source digest before resize; no EXIF transpose; unsupported capability/copied-request refusal. Pure contract tests establish none of the routes or storage below. |
| TODO | Legacy identity and persistence | Pin existing direct/refine/inpaint/outpaint/upscaler/execution/mixed-size/noncenter run ids. Compare row-derived and request-derived keys, migration on replaced old DBs, SQL/PNG round trips and absent-field compaction. Patterns: `test_conditioning.py`, `test_size_crop.py`, `test_crop_anchor_storage.py`, `test_execution_storage.py`, `test_inpaint_storage.py`. |
| TODO | Source portability and recovery | Proven remapped source; omitted, absent, binned, forged, colliding-id or descriptor-mismatching source; unresolved re-export; SQL/PNG disagreement; repeated import and in-place reindex; dry-run; old-reader refusal. Patterns: `test_reindex_provenance.py`, `test_reindex.py`, `test_execution_bundles.py`, `test_inpaint_bundles.py`; also the bundle checklist. |
| TODO | Producers and admission | Re-render/More seeds preserve original source; cache verdicts cannot cross source/recipe; Swap refuses; Mass refine changes pass kind; Then refine/grids refuse; Finish retains complete intent; missing source before/after Queue and stale model revision fail before loading/rendering. Patterns: `test_execution_replay.py`, `test_api_more_seeds.py`, `test_refine_batch.py`, `test_then_refine.py`, `test_cli.py`, `test_api_png.py`, `test_jobs.py`. |
| TODO | Source lifetime and reports | Starred edits protect proven sources; bin/purge counts include them; deletion/bin/restore updates records and job snapshots; completed children stay visible and unreplayable when needed; edits never contaminate direct/refine estimates. Patterns: `test_gallery.py`, `test_gallery_purge.py`, `test_api_gallery.py`, `test_jobs.py`. |
| TODO | Backend and browser | Stub exact kwargs, one-image generators, callbacks/cancel and failure cleanup; fake edit dialog smoke with composer draft/Undo, stale preview/revision, focus and narrow/wide layouts. Ruff, pinned whole-source types and relevant CPU regressions follow source integration. Real model quality/fit/timing and nonempty release admission remain separate card gates. |

The prior Windows browser launch ended with `IAppCapability` access denied
and the Linux/WSL path was also denied; [tooling](plan-tooling.md#browser-smokes-in-ci)
owns that evidence. Browser qualification is **unverified** until actual
access is restored and the new smoke runs. Static/Node VM/fake unit checks
do not substitute for the real-browser pass or close stage 1/2 acceptance.
