# Instruction edits: portable identity and bundle compatibility

**CPU implementation, 2026-10-27 (`INP#11`).** This document freezes the
compatibility contract for the first `kontext-v1` instruction-edit recipe.
The [feature plan](plan-instruction-editing.md) owns implementation order.
The policies below are implemented with synthetic storage/bundle/reindex
fixtures. No live gallery, credential store, model or GPU was used.
Validation belongs to the feature plan.

**Released recipe.** Stage 4 opened editor execution inside the measured
envelope on October 30 (PR #17). Its [real-server and browser acceptance](plan-instruction-editing.md#card-acceptance-result-2026-11-18)
passed on November 18. Imported edits can replay through More seeds when the
source is proven, live and square, and the loader is inside the measured
envelope. The released recipe processes the source at 512x512. An edit with
`source: null` remains unreplayable. The archive gate, omission counts and
merge proofs below govern portability.

## Two source representations

The public request type is `InstructionEditSpec`:

```json
{"source": 123, "recipe": "kontext-v1"}
```

`source` is a strict positive integer naming this gallery's source image.
Unknown fields or recipes, booleans and coercible strings are refused. The
request's `prompt` is the instruction; there is no second instruction field.

The persisted type is `StoredInstructionEdit`:

```json
{
  "source": 123,
  "recipe": "kontext-v1",
  "edit_source": {
    "filename": "source.png",
    "hash": "0000000000000000000000000000000000000000000000000000000000000000",
    "width": 512,
    "height": 512
  }
}
```

Here `source` is a strict positive local id **or null**. The nested
`edit_source` descriptor is mandatory even when `source` is null. The
descriptor must be stored inside the nullable `images.edit` JSON value,
as well as the PNG's stored edit value. It cannot exist only in a PNG
chunk: import rewrites and a later database export must retain it.

The example illustrates the stored shape; its all-zero hash is a placeholder.
The filename is a safe PNG basename, dimensions are strict positive
integers, and `hash` is the complete SHA-256 of
`image.convert("RGB").tobytes()` for the decoded original source. The recipe
owns colour conversion and geometry; the descriptor identifies the source before its
working-size resize. Compute this hash from the actual source pixels.
Neither the file's compressed bytes nor an unverified database claim proves
the source identity.

Refuse path separators, traversal, Windows drive/alternate-stream syntax,
reserved device names and unsafe trailing characters in the filename. A
descriptor alone never authorizes opening a path; resolve a proven local
gallery id through storage instead.

Existing `image_provenance` and `content_hash` use a 16-hex prefix. They
remain unchanged for existing features. That prefix is not sufficient for
the edit descriptor: compare all 64 hexadecimal characters, filename and
dimensions, using the bytes the editor would read.

`source: null` means the completed result still records an instruction edit,
but this gallery has no proven usable source link. Keep its recipe,
instruction, original descriptor and output visible; refuse replay. Never
retain an unresolved foreign id as a usable local id, and never drop the
edit spec to turn the instruction into a plain render. Source-missing or
source-binned presentation is derived state, not a replacement recipe.

## Archive version gate

Keep the existing `format: "auradream-gallery"` identifier (a bundle
written before the rename says `local-diffusion-gallery` and imports
unchanged). The [gallery bundle rules](../rules.md#image-storage-and-gallery-bundles)
own version selection. Edit intent requires at least v2. An archive with both
an edit and a tracked Tiled Refine result uses v3 and carries both intents.
Column presence alone does not require v2. The same rule applies to
whole-gallery, selected, manual, managed and password-protected exports.

The version change protected against a legacy reader that accepted only v1
and ignored unfamiliar row/PNG fields. An optional capability field in a v1
manifest would let that reader strip edit intent. An unsupported manifest
version makes that reader refuse before staging the database or installing images.

The reader accepts the supported strict integer versions. It refuses the
following edit-specific violations.

- Any v1 archive carrying non-null edit intent in either its database rows
  or tracked PNG metadata, including a PNG whose database lacks the column.
- Unknown or malformed stored edit data, unsupported recipes and a missing
  or malformed mandatory descriptor.
- SQL/PNG edit disagreement. Missing SQL columns or missing edit metadata
  must not silently convert an edit into an ordinary result. Recovery or
  repair of contradictory representations is outside the first slice.

These checks apply during export's existing read-back and every import
inspection path, including equal pixel-deciding chunks and a known file
whose pixels would otherwise avoid decoding. Check a retained destination
PNG against its own row before claiming it is present. A downgrade to v1
must fail even when every member checksum was recomputed.

Version validation and edit checks preserve the existing one-read archive
verification. Do not add a second full PNG decode or full archive pass.
Manifest checksums, complete pixel identity for filename conflicts,
encryption authentication, staging and installation checks remain required.

## Selection, merge and unresolved sources

Selected exports contain only the chosen images. Do not automatically add
an edit source or ancestor; this keeps the existing selected-export privacy
contract. `links_outside` counts each edit source omitted from the selected
rows, alongside existing reference/refine links. An already unresolved edit
also counts as an unresolved dependency. Explain the omission before export.

Before making any live change, validate the stored edit, its PNG copy and
all included source images. Build the complete source-id mapping, then
resolve edit links:

1. A mapped source is usable only if its actual decoded RGB hash, filename
   and dimensions match the saved `edit_source` descriptor. Rewrite the
   stored local id and PNG together; preserve the proven descriptor.
2. An omitted, missing or unproven source produces `source: null`, retains
   the complete original descriptor and increments `unresolved_references`.
   A well-formed but mismatched descriptor does not prove a link. Keep the
   child inspectable and refuse replay; never bind merely because a numeric
   id exists in the destination.
3. A proven source arriving binned may retain its mapped id and lineage,
   while ordinary source admission refuses a new edit until it is live.
4. Existing destination rows retain their source-resolution state. Repeating
   an import must not silently relink an orphan, change its recipe or
   instruction, or overwrite the destination PNG. Explicit source repair
   is outside the first slice.

Known filenames must agree on instruction-edit identity as well as their
existing metadata and pixel checks. Compare recipe, instruction and frozen
source provenance across galleries; raw source ids can differ after a
legitimate remap. A known ordinary result and an incoming edit result cannot
be treated as equivalent merely because their pixels match.

Within a gallery, the edit CellKey uses its bound stored identity, including
the complete descriptor and full source digest. Bind/prove that identity
before run ids, Resume/deduplication and More seeds grouping. A new source
file at an old numeric id cannot share the old edit's key, while replay
must refuse pixels that no longer match its recorded descriptor.

The metadata rewrite must retain an unresolved descriptor rather than
recompute it from an absent local id. The missing-file restoration path
must preserve the destination row's edit identity. Retain existing full
decoded-image identity checks for general filename conflicts; the edit
source's RGB descriptor does not replace mode/palette/transparency checks.

Database merges and PNG-only reindex use the same proof and unresolved
representation. A PNG-only rebuild must not trust old ids. Cyclic links,
forged provenance and a source replaced outside the app cannot make a
child replayable. None of these portable operations grants editor runtime
availability.

## Finite acceptance matrix

All fixtures use owned scratch galleries and synthetic images. Implemented
cases live in `tests/test_instruction_edit_bundles.py`,
`tests/test_instruction_edit_storage.py` and `tests/test_instruction_edit_reindex.py`.
They cover the matrix below without granting editor/runtime qualification.

| Case | Required result |
|---|---|
| Exports without edit or Tiled Refine intent, plain and encrypted | Manifest remains v1. Existing imports and legacy run-id fixtures remain unchanged. |
| Whole-gallery and selected edit exports without Tiled Refine, plain and encrypted | Manifest is v2. A v1-only reader refuses before staging or installing members. |
| Managed edit export without Tiled Refine | V2 manifest is included in capacity estimation, receipt proof and the existing verifying read-back. |
| An archive with edit and Tiled Refine results | Manifest is v3. The archive carries both intents and verifies their source proofs (`test_tiled_refine_bundles.py`). |
| Full source and child round trip into a gallery with occupied ids | Both records and PNGs carry the correctly remapped source; the origin gallery stays unchanged and destination source pixels are unchanged by metadata rewriting. |
| Child-only selection or an already unresolved edit | Omission count is honest; source remains null with recipe/instruction/descriptor intact; replay refuses. |
| Destination has an unrelated row at the old source id | That row is never used as the source. |
| Included source has a valid descriptor but other pixels, dimensions or filename | Link remains unproven and the child remains unreplayable. Full-hash mismatch must be detected even where a short prefix matches. |
| Source arrives binned or is later binned/deleted | Lineage and edit intent remain inspectable; new submissions refuse the unusable source. |
| Unknown recipe, malformed edit/descriptor or contradictory SQL and PNG | Entire bundle import/export read-back refuses before any live change or source cleanup. |
| V2 archive rewritten to v1 with recalculated checksums | SQL-only, PNG-only and matching SQL/PNG edit intent each refuse, including repeat-import fast paths. |
| Same pixels with ordinary versus edit metadata, another instruction, recipe or source | Whole import refuses rather than claiming present or overwriting metadata. |
| Matching repeated import and missing-file restoration | Existing local edit identity/resolution is retained; valid restoration uses the same proven destination metadata. |
| Database merge and PNG-only reindex | Proven sources remap; missing/unproven sources preserve unresolved intent; foreign ids are never trusted. |
| Failure during verification, rewrite or installation | Existing staging/rollback and authentication guarantees hold; source gallery and prior good archive survive. |

## Implementation seams audited

The archive gate belongs in `bundle_archive._verify_archive`; the writer
chooses the version in `bundles.export_bundle`. `bundle_images` validates
row/PNG identity and rewrites complete metadata. `bundles._links_outside`
and the merge's complete id map handle selection counts and relinking.
`bundle_capacity.managed_capacity` measures the supplied manifest JSON for
the selected archive version.

Extend the existing execution-metadata presence/conflict cases, provenance
reindex cases, selected bundle cases, password/managed export cases and
repeat-import tests. Update the bundle rules and their owner notes when
those changes land. The CPU implementation now uses this archive gate;
editor execution (stage 4's envelope) and its card acceptance are the
feature plan's, separate from this contract.
