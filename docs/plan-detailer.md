# Plan: face and hand detailer

**Status, 2026-10-27:** Matt requested a plan for `INP#12`. This document
commissions no implementation, dependency installation, detector download,
gallery scan or card run. Detector selection and qualification are open.
The proposal follows [R10](review-2026-10-25.md#r10-a-face-and-hand-detailer);
the existing machinery is described in [Inpainting](design-notes.md#inpainting)
and [Generate, then refine](plan-then-refine.md).

## Outcome and proposed first slice

Choose one gallery image, find candidate faces or hands, review the marked
regions, and queue an inpaint of each accepted region. Keep the source and
every completed pass. Detection locates a region; it neither establishes
that a face/hand is wrong nor guarantees an improvement.

The proposed first slice is a gallery action with a region review, not a
default generation step. Faces and hands are separate capabilities: qualify
and expose either only when its detector works on the intended images.
Review lets Matt deselect a false positive and see the number of passes
before Queue. This interaction is a proposal, not a recorded preference.

Start with one source and one supported SDXL checkpoint. Use its existing
inpaint, whole-canvas size and prompt/negative, with explicit strength,
steps and seed. A generated mask is a padded, feathered box, not semantic
segmentation. Choose padding/feather/threshold values through trials;
publish no unsourced claim that they repair anatomy or preserve identity.

Do not include outside-image import (`GEN#8`), training, a new image model,
automatic aesthetic scoring, automatic acceptance, generation-wide defaults,
or a permanent detector resident on the GPU. Picked-image batches and a
Then detail pass come after the single-image path has useful evidence.

## Detector feasibility comes first

Primary sources checked on **2026-10-27** establish candidates, not a winner:

| Candidate | What is established | What remains |
|---|---|---|
| MediaPipe Face Detector / BlazeFace | The task returns face boxes/keypoints. Its documented models target camera imagery. [Task and model descriptions](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector). | Anime, profiles, small faces and deliberately malformed faces need a representative detection trial. |
| MediaPipe Hand Landmarker | It uses a palm detector and a landmark model; output includes hand landmarks and handedness. Turning those into a full-hand box is app policy. [Task and models](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker). | Occluded, extra/missing-finger and stylized hands may be missed; a returned landmark set is not a correctness verdict. |
| MediaPipe runtime | The repository code carries Apache-2.0. [Code licence](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE). Python tasks load a separately obtained model asset. [Python setup](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector/python). | Record the exact asset's licence separately. Repository/code-sample licensing does not establish a downloaded weight file's licence. Check Windows/Python 3.13 packaging and dependency compatibility. |
| Usual Ultralytics-based detailer models | Ultralytics offers AGPL-3.0 and enterprise licensing. [Vendor licensing](https://www.ultralytics.com/license). | Do not add these under an assumption that exporting weights removes obligations. Exact code/weight provenance and terms would need a separate review; no paid licence or repository relicensing is proposed. |

This plan has **not verified the exact candidate weight licences**. Stage 1
must locate their authoritative terms and preserve the required notices;
missing or ambiguous terms stop that candidate. A model-card claim, runtime
licence, exporter licence and training-data description are separate facts.
An alternative detector is acceptable only with equivalent evidence.

The first research slice is bounded: inspect at most two face and two hand
assets, their original sources, exact versions, sizes, hashes where
published, code/weight terms and available Windows/Python 3.13 wheels.
Read metadata without installing or importing a detector. Do not alter
torch, diffusers or the pinned environment to make a candidate fit.
If none passes, report the missing evidence and stop; no unbounded search
or detector-training project follows automatically.

After a candidate passes that inspection, an explicitly commissioned CPU
trial should use a small, frozen, permission-cleared test set, separated
into photographic/anime faces, hands, occlusion, tiny regions and negatives.
Label intended boxes before tuning; retain a holdout. Record missed regions,
false positives, box coverage, latency and process RAM, including failures.
Agree usefulness targets before selecting thresholds; no score is claimed
here. A face-only result can justify a face-only slice, leaving hands out.

## Current implementation seams and the crop boundary

- `refine.inpaint_source` requires the source's original dimensions;
  `refine.pipeline_mask` widens the nonzero mask to latent cells and
  `refine.composite` restores every unmasked source pixel. Reuse those
  contracts, including the hash of the composited result.
- `storage.store_mask` names size plus luminance levels by SHA-256. The
  mask is not a gallery image; every request uses a gallery source id and
  a mask hash, never detector paths or an arbitrary input filename.
- `JobManager._passes` demonstrates serial work after saving a base;
  `refine_batch.plan` and `grid.cell_from_record` demonstrate turning a
  record into a normalized pass. They do not currently implement a detailer.
- Current inpaint runs the whole canvas. `INP#9` owns crop-at-working-size
  inpainting; Z-Image's installed inpaint class lacks `padding_mask_crop`.
  First qualification must test whether a whole-canvas SDXL pass helps
  small regions enough to be useful. If not, stop release and make the
  required crop subset of `INP#9` an explicit dependency before proceeding.
  Do not silently resize an existing mask/source or claim a crop gain.
- Detection and diffusion quality are different gates. A good box finder
  can still feed an inpaint that changes identity, adds fingers or harms
  the image. The card comparison decides that second question.

## Proposed frozen detection and render contract

Use a pure, versioned detailer planner plus a lazy detector adapter. The
detector receives decoded RGB pixels, not a path supplied by the browser.
The proposed preview runs CPU detection through admitted idle work using
`jobs.run_when_idle`, on a daemon worker; it must not create a second
generation worker or load diffusion weights. Status/capability reads stay
cold and list missing assets without fetching them.

1. Resolve the live gallery source and its pixel hash/dimensions. Freeze
   detector id, runtime/version, asset revision/SHA-256, preprocessing,
   orientation/colour handling, image-mode settings and inference provider.
   Explicitly use CPU initially; no implicit GPU delegate.
2. Map boxes/landmarks back to original pixel coordinates. Freeze rounding,
   clipping, minimum region size, padding, feather and mask algorithm name.
   Reject NaN, inverted, off-image or empty regions. Bound source pixels,
   candidate count, accepted regions, response bytes and cached previews.
3. Apply deterministic duplicate suppression before review. Record class,
   box, confidence and suppression/threshold parameters. Sort accepted
   regions by a documented total order (class, top, left, bottom, right,
   then a stable tie-break). Do not rely on detector iteration order.
4. Show any overlap in the review. The proposal preserves overlapping
   accepted regions and runs them in the displayed order; later passes
   see earlier edits. Do not merge or reorder boxes invisibly. Deselecting
   a box freezes a new accepted order and therefore a new plan identity.
5. On Queue, revalidate source identity, selected model revision, actual
   inpaint capability, image bounds and all accepted masks/settings.
   Bind a bounded preview receipt to that source and configuration; stale
   or reused admission must not duplicate a chain or silently redetect.
6. Draw and store all full-size masks before job admission, then reserve
   them for the entire job, including future dynamic passes. Every render
   request still passes the ordinary `GenerateRequest.applied()` path.

No detections, or deselecting every region, gives a clear empty result and
queues no GPU work or duplicate gallery image. A corrupt asset, detector
exception, timeout or invalid output is an error, never an empty success.
First-use download is explicit and bounded with pinned bytes/hash checks,
an atomic cache and cancellation; do not download during startup or polls.
The dependency and asset-download implementation belongs to a later stage.

## Serial pass chain, identity and recovery

The proposed storage model is a **chain of ordinary saved inpaints**. Region
1 uses the chosen source; region 2 uses region 1's saved output, and so on.
Boxes are detected once on the original source, in unchanged full-canvas
coordinates. Every pass keeps its direct source id, mask hash, prompt pair,
seed, strength, sampling settings, model id, quantization, execution recipe
and peak result just as an inpaint does. Save intermediate outputs visibly as part
of the detail operation; the last completed pass is not labelled final if
the chain stopped early. Do not silently replace the chosen source.

Each region gets its own generator and a seed derived from a frozen base
seed plus accepted ordinal with the existing `seeds.py` range/wrap rules.
Record both base and actual seeds; no detector output consumes diffusion
RNG state. Rerendering an individual saved pass uses its saved mask/source
and seed, not a rerun of detection. A deliberate new detection is a new
operation with a new preview, even if the image is a previous result.

Add optional versioned detailer provenance to the row and PNG: original
source descriptor, detector/asset facts, frozen candidate/accepted geometry,
mask algorithm, ordered masks, recipe, group identity and this pass index.
Define its stored schema separately from request limits so later settings
cannot make old rows unreadable. Add a `_migrate` entry for any new column.
Bound the manifest and do not store model weights in PNGs or bundles.

The orchestration/run identity must include the frozen accepted plan,
source provenance and resolved sampling recipe. If detailer intent extends
`CellKey`, include it only when present and pin all old keys/run ids. Actual
pass keys must agree between saved rows, rerender and completed-cell reads;
elapsed time, detector confidence summaries and GPU peaks are results, not
reproduction controls. Finalize this encoding in Stage 2 before implementation.

Carry the new provenance through storage/metadata, `ImageRecord`, reindex,
PNG replay, bundles, source-id remapping and every rerun action. Keep
content-hash/dimension proof before restoring source links. Bundle selection
must explain whether it contains only selected children or their source
chain; it must not invent missing ancestors or treat an unavailable chain
as reproducible. Retain and verify all referenced masks; extend mask sweep,
holds and embedded-mask handling for the manifest's mask references.

Re-render and More seeds of one saved pass can reuse its ordinary inpaint
inputs; changing seed creates new actual seed provenance. Swap tags changes
the explicit per-pass prompt while retaining its mask/source. Mass refine
starts a new ordinary pass of the selected result and drops the old
detailer recipe as execution intent. Reuse may offer a new detailer review
but must not execute a whole chain silently. Load settings from PNG stays
text/settings only and grants no source or mask link without existing
provenance proof. Specify these routes and refusals before enabling them.

## Jobs, cancellation and partial results

Use the existing single worker/gate and normal load lifecycle. Do not load
or switch an extra checkpoint for a detail pass. Capability is checked
both before queuing and after load. CPU detection/asset work must check
cancellation between bounded operations; diffusion continues to poll its
step callback. No blocking `Event.wait()` and no preview callback raises.

Report detection separately from rendering, then region `i/N` and the
ordinary per-pass step count. One source has at most the admitted region
count of calls; estimates use matched history or say unavailable. Cancel
keeps committed passes, drops the current incomplete pass, skips remaining
regions and releases every mask hold through `_finish`. Pause is at a
documented region boundary. Hold Skip during a dependent chain initially,
with a reason, until an explicit skip/recovery policy exists.

A failed region stops that source's dependent chain: do not proceed using
an older intermediate as if the failed pass ran. Keep prior saved outputs
and mark partial/error with the last completed region. A source deleted
or binned before a dependent pass is checked under existing source rules.
Retry/Finish of the chain needs a retained frozen plan plus the verified
last completed child; until durable orchestration recovery exists, offer
explicit restart/new review only, without promising automatic Finish.

## Stages and completion criteria

1. **Detector evidence, CPU only.** Complete the bounded source/licence and
   package inspection, then separately commission the small CPU trial.
   Exit: exact code/weight terms, versions, dependency delta, asset bytes,
   measured CPU/RAM/quality report and an explicit face/hand capability
   decision. No candidate passing means the feature remains deferred.
2. **Pure contract and persistence.** Specify/freeze boxes, masks, overlap,
   ordering, seeds, preview receipts, manifests and per-route replay policy.
   Implement only when commissioned, with fake detections and throwaway
   storage. Exit: deterministic plan fixtures; legacy keys unchanged;
   corrupt/missing masks, stale receipts, source remaps, migration, PNG,
   bundle/reindex round trips and deletion/holds tested.
3. **Single-source backend and dialog.** Build the admitted preview and
   serial inpaint chain with fake detector/backend tests. Add a real-browser
   smoke using isolated fake data: no detections, box selection/overlap,
   unchanged draft, held reasons, stale source/model, cancel/failure/partial
   status, focus return and narrow/standard/wide layout. Run focused tests,
   Ruff, pinned type gate and the affected inpaint/queue/bundle smokes.
4. **Separate card qualification.** After `GEN#22` where its loader applies,
   agree one checkpoint/loader, one existing measured size, fixed sources,
   settings, region bound and finite comparisons. Record load/render RAM,
   commit headroom, VRAM/shared GPU memory, latency, kept-pixel equality,
   seams, anatomy and identity preservation against the untouched source
   and a manually selected inpaint baseline. Stop before shared-memory
   spill or the agreed host-RAM bound. No proposed size is a fit claim.
5. **Optional batch/Then detail.** Only after the first path helps Matt,
   propose picked-image admission and/or a post-generation stage, extending
   retained run definitions, bounds, cancellation and partial-result
   accounting. Do not add it implicitly to Then refine or enable by default.

The review's `L+card` is an initial estimate, not measured effort. Detector
failure or a crop dependency may change it. The first useful decision after
this plan is whether to commission Stage 1; UI values and detector choice
remain proposed until that evidence exists.
