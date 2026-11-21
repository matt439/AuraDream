# Workstation features requested November 21, 2026

Matt requested these additions after reviewing gaps in his generate, compare,
refine and archive workflows. The [backlog](../../TODO.md#now) tracks their
status. Public tiled Refine already has a [release plan](plan-tiled-refine.md#4-public-flows-and-release).
Portrait, landscape and 1024-square Kontext edits already have a
[card trial](plan-instruction-editing.md#running-the-isolated-trial).

## Restore one archived image

`GAL#38`. After Bundle & clear, Browse can find a keeper in a saved archive.
Today Import merges the whole archive. Add Restore to the browsed image.
Preview which source images, references and masks are available, already
present or missing. Let Matt include the dependencies needed for replay.
[Current bundle workflow](../usage.md#gallery-bundles)

Complete when one keeper from a plain or password-protected bundle restores
with its chosen dependencies, unrelated rows stay out of the gallery, and a
repeat import recognizes the same images. Exercise a colliding gallery ID and
a missing source. Remap links by the existing import rules and tell the user
when an omitted or unavailable source prevents replay.

## Save a Generate and Then refine recipe

`GEN#75`. Save a named setup with Then refine enabled, including scale,
strength and inherited or explicit step count. Today Save asks the user to
turn Then refine off. Extend the [saved setup format](plan-generation-presets.md#saved-contract)
and its [portable form](plan-setup-portability.md).

Complete when save, export, import and restore reproduce the same two-stage
request on its supported checkpoint. Show both stages in the restore preview.
Restore must leave the recipe ready to inspect and run, and Undo must recover
the previous composer. Cover older setups, which restore with Then refine off.

## Offline tag autocomplete and checking

`PA#20`. Offer a downloadable, dated tag index with canonical names,
aliases and categories for supported tag-based checkpoints. Show which index
answered a lookup. Keep the checkpoint author's own vocabulary available.
[Current tag lookup](../../backend/tags.py)

Complete when a fresh server with the index installed and the network
disconnected can suggest a previously unqueried tag, resolve an alias and
check its category. An unknown name should be described as absent from that
snapshot. Cover initial installation and an interrupted refresh that leaves
the previous snapshot usable.

## Blind comparison of gallery results

`GAL#39`. Offer a blind review for two picked images and for two prompt
or Refine variants rendered on shared seeds. Adapt the A/B/tie and reveal
interaction from the [fixed performance review](blind-performance-review.md)
to ordinary [gallery comparisons](../usage.md#browse-compare-and-keep-results).

Complete when the reviewer can inspect the same detail on both sides, record
a preference or tie, then reveal the prompts, settings and totals by variant.
Hide identifying labels until reveal and retain each randomized side assignment
through the review. Pair results by seed and report unmatched seeds explicitly.

## Zoom and pan while painting a mask

`INP#16`. Let the user switch between fit and actual pixels and pan to an
eye, fingertip or narrow edge while [painting an inpaint mask](plan-inpainting.md#the-canvas).
Keep the source and mask aligned throughout zoom and pan.

Complete when a stroke, erasure and undo at 1:1 affect the intended source
pixels on a large image. Returning to fit should show the same mask.
Exercise mouse and touch navigation, including a pan that leaves no brush mark.

## Draw a missed detailer region

`INP#17`. The [Detailer](plan-detailer.md) lets users accept or drop
detected faces and hands. Its fixtures include illustrated hands that the
detector missed. Let the reviewer draw a box around a missed region or
correct a detected box before queueing.

Complete when both a zero-detection result and a mixed manual/detected
selection can produce the reviewed masks in order. Preserve the final bounds
and which boxes the user drew or changed in replay metadata. `INP#12`
continues to own the separate on-card quality comparison.

## Photo and sketch import

`GEN#8`. Matt reopened the earlier [personal reference import decision](../design-notes.md#reference-images-by-gallery-id).
Provide file pick, drop and paste for a photo or sketch. Create a gallery
source with its own ID and an imported-source marker. Load settings from PNG
keeps its separate metadata-only action.

Complete when an imported image can be selected on a reference-capable
checkpoint or used as a source for a supported Refine pass, including after
reindex or bundle restore. Verify that a later rerun resolves the same source
and that the gallery clearly identifies the original as imported.

## ControlNet pose and depth guidance

`GEN#76`. Let the user supply a pose map or depth map to guide composition.
Matt reopened these two controls from the [peer review](review-2026-11-10-peers.md#against-the-peers).
Start by identifying a compatible catalogue checkpoint and control weights
for each mode, then specify how to select, preview and record the guide.

Complete when each mode has a qualified checkpoint and a reproducible example.
Compare guided and unguided renders on the same seed for pose or depth
adherence and measure the card and host memory required. Preserve the guide
image identity, control weights and strength for replay, and show which modes
the selected checkpoint supports.

## Region-specific prompts

`GEN#77`. Let a user describe distinct parts of one composition, such as
“red coat” on the left and “blue coat” on the right, alongside a whole-image
prompt. Start with a compatible checkpoint and an editable spatial region
for each local prompt. Define how overlapping regions combine.

Complete when a fixed-seed comparison shows what changes when the two
local prompts swap places. Record region geometry, text and the method used
so a saved image can restore and replay the setup. Qualify the supported
checkpoint and size on the card before offering them in the app.
