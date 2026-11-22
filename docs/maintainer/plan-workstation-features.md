# Workstation features requested November 21, 2026

Matt requested these additions after reviewing gaps in his generate, compare,
refine and archive workflows. The [backlog](../../TODO.md#now) tracks their
status. Public tiled Refine already has a [release plan](plan-tiled-refine.md#4-public-flows-and-release).
Portrait, landscape and 1024-square Kontext edits already have a
[card trial](plan-instruction-editing.md#running-the-isolated-trial).

## Restore one archived image

**`GAL#38` built and verified, 2026-11-21.** Open an image in Browse and
choose **Restore**. The preview lists available, already present and missing
source images, references and masks. Available image dependencies can be
included individually. The background bundle operation reports the local ID,
the recycle-bin state and any replay hold.
[Gallery bundle workflow](../usage.md#gallery-bundles)

The browser smoke used a plain and a password-protected archive. It covered
cancel, restoring the chosen source, excluding unrelated rows, repeating
the restore, opening the exact restored row, and full import. Backend checks
covered colliding gallery IDs, missing sources, masks, closing the view
during a restore, and the version 3 tiled recipe's original and prepared
RGB proofs. Existing rows keep their local links on repeat restore. If one
of those rows still lacks a required link, the receipt says replay is held.

## Save a Generate and Then refine recipe

**`GEN#75` built 2026-11-22.** Generate setups retain scale, strength
and inherited or explicit steps in a version 2 recipe. The restore preview
shows both stages and Undo recovers the previous composer. Older setups
restore with Then refine off. The saved setups and portability smokes
exercised the two-stage request, exact export/import and Undo.
[Saved setup format](plan-generation-presets.md#saved-contract) ·
[Portable format](plan-setup-portability.md)

## Offline tag autocomplete and checking

**`PA#20` built 2026-11-22.** The Tags menu downloads a dated snapshot
of up to 10,000 popular Danbooru tags, their active aliases and categories.
Offline autocomplete and Check report the snapshot date. The checkpoint's
own vocabulary remains available, and users can switch back to live lookup.
A real download populated 10,000 tags. The API and browser checks covered
alias resolution, categories, a dated miss, operation after reopening the
index, and keeping the previous snapshot after an interrupted refresh.

## Blind comparison of gallery results

**`GAL#39` built 2026-11-22.** Picked and contact sheets open a blind
A/B review. Images share a zoom and pan position. The reviewer records a
preference or tie, then can reveal the prompts, settings and totals. Pairing
uses the saved seed as an exact decimal string and reports missing matches.
Repeated renders use the newest row for each variant and seed, and the
review retains each randomized assignment for the session. Browser checks
covered Picked, missing seeds, synchronized pan, saved votes while
navigating, and selecting two strengths from a three-variant Refine sweep.
[Gallery comparison workflow](../usage.md#browse-compare-and-keep-results)

## Zoom and pan while painting a mask

**`INP#16` built 2026-11-22.** Fit and actual pixels share a fixed
frame. Pan works with mouse, touch and keyboard. The source, mask and
crop preview use the same transform. The Inpaint smoke verified precise
painting, erasing and Undo on a 1536-square picture, a stable mask on
return to Fit, and an emulated phone pan without a brush mark.
[Mask workflow](../usage.md#inpaint-part-of-an-image)

## Draw a missed detailer region

**`INP#17` built 2026-11-22.** The review supports drawing missed faces
or hands and correcting a detected box before queueing. The server applies
the normal padding, ordering and masks to both a manual-only review and a
mixed review. Version 2 provenance retains final geometry and whether the
user drew or edited the box. The separate on-card quality comparison
remains in `INP#12`. [Detailer plan](plan-detailer.md)

## Photo and sketch import

**`GEN#8` built 2026-11-22.** Gallery Import accepts file pick, drop
and paste and stores a normalized RGB picture under a new gallery ID.
The source marker travels through the PNG, reindex and bundle restore.
Import checks exercised a reference and a Refine job, replay, a binned
source and selective archive restore with colliding IDs. The PNG settings
smoke covered the desktop and phone import flows alongside the existing
settings-only flow. [Gallery import workflow](../usage.md#use-references-and-adapters)

## ControlNet pose and depth guidance

**`GEN#76` built and qualified 2026-11-22.** The composer previews a
gallery guide by ID before Generate. The first recipe uses a prepared
1024-square pose or depth map with the pinned Animagine XL 4.0 FP8/model
loader. The result records the guide ID and full RGB proof, mode, weight
revision and strength. Bundle restore and reindex resolve a guide by
proof. If the guide is absent, replay holds with the original proof.

The fixed-seed card comparison followed the synthetic guide in both modes.
The pose changed from arms lowered to one raised and one extended. The
depth render followed three boxes receding across the image. Both guided
replays matched their RGB hashes. The pose-only module is discarded on
cleanup, and an unguided control after the sequence matched its earlier
output. [Card measurements](../verified.md#spatial-guidance-on-animagine-xl-40)

## Region-specific prompts

**`GEN#77` built and qualified 2026-11-22.** The composer accepts up to
three rectangular local prompts with editable percentage bounds.
`sdxl-attention-v1` joins the whole-image and local tokens for covered
queries and averages the attention output of overlapping regions. The
whole-image pooled text and negative branch stay global. Reuse restores
the recorded layout and text. The recipe travels in the row, PNG, run
identity, replay, reindex and bundle.

The guarded public API trial on the pinned Animagine XL 4.0 FP8/model
recipe showed red and blue coats swapping sides when their prompts
swapped at the same seed. Replaying the original regions matched the
RGB hash. A three-region render ran and the unguided control after the
region tests matched its earlier output.
[Card measurements](../verified.md#spatial-guidance-on-animagine-xl-40)
