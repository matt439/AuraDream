# Instruction edits: portable identity and bundle compatibility

**CPU implementation, 2026-10-27 (`INP#11`).** This document freezes the
compatibility contract for the first `kontext-v1` instruction-edit recipe.
The [feature plan](plan-instruction-editing.md) owns implementation order.
The policies below are implemented with synthetic storage/bundle/reindex
fixtures. No live gallery, credential store, model or GPU was used.
Validation belongs to the feature plan.

**Stage 4, 2026-10-30.** Stage 4 changed no bundle, archive or reindex
code, and this contract is unchanged. Its release (local on
`feat/inp11-stage4`, until the plan's
[card acceptance step](plan-instruction-editing.md#card-acceptance-step)
passes) opens editor execution on the real server only inside the measured
envelope. An imported edit replays only through More seeds, and only once
its source is proven and live here, the source is 1:1 and the loader is
inside that envelope; a `source: null` edit stays unreplayable. The v1/v2
rule, the omission counts and the merge proofs below still decide
portability, and none of them grants runtime availability.

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
unchanged). An export
whose selected rows contain any non-null stored edit uses manifest
`version: 2`; an export without edit intent keeps `version: 1`. Column
presence alone does not require v2. Whole-gallery, selected, manual and
managed exports, including password-protected ones, obey the same rule.

This version change is required because the existing reader accepts only
v1 and ignores unfamiliar row/PNG fields. An optional capability field in
a v1 manifest would let that reader strip edit intent. V2 makes that reader
refuse the archive before staging its database or installing images.

The new reader accepts only strict integer versions 1 and 2. It refuses:

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
PNG against its own row before claiming it is present. A downgrade from
v2 to v1 must fail even when every member checksum was recomputed.

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
| Edit-free legacy and current exports, plain and encrypted | Manifest remains v1; existing imports and legacy run-id fixtures remain unchanged. |
| Whole-gallery and selected edit exports, plain and encrypted | Manifest is v2; a v1-only reader refuses before staging/installing members. |
| Managed edit export | V2 manifest is included in capacity estimation, receipt proof and the existing verifying read-back. |
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
`bundle_capacity.managed_capacity` already measures the supplied manifest
JSON; its coverage must include v2 rather than assume a fixed manifest.

Extend the existing execution-metadata presence/conflict cases, provenance
reindex cases, selected bundle cases, password/managed export cases and
repeat-import tests. Update the bundle rules and their owner notes when
those changes land. The CPU implementation now uses this archive gate;
editor execution (stage 4's envelope) and its card acceptance are the
feature plan's, separate from this contract.
