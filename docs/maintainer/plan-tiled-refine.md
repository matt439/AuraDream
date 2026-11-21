# Tiled Refine

**Status, 2026-11-21.** The exact Animagine recipe below is available as an
opt-in under Refine. The [public workflow qualification](#public-workflow-result-2026-11-21)
covers the browser, real API and fresh standalone CLI. The exploratory
SDXL seam experiment passed its planning gate on November 17. The originating
[R2](review-2026-11-10-peers.md#r2-tiled-refine) owns why the experiment came first.

The aim is to refine a larger image through bounded overlapping regions,
using the existing image-to-image and masked-area paths. Each region sees
the composite produced so far. This is serial denoising and pixel overlay,
not multi-diffusion or a latent blend at every step. It can redraw content
and can expose seams. VAE tiling alone does not implement it.

## Exploratory evidence

One offline Windows/RTX 5080 session used Animagine XL 4.0, fp8/model
offload, Eager/Native and diffusers 0.42.1. A new synthetic 1024-square
render was enlarged exactly 2x with CPU Lanczos and honestly stored as a
fixture, never represented as a 2048-square diffusion render. Four normal
inpaint jobs then ran top-left, top-right, bottom-left and bottom-right
over that 2048-square canvas. Strength was 0.35, 28 full steps, CFG 5 and
tile seeds 12346-12349. Each mask spans 1088 square, overlaps its neighbor
by 128 pixels, and feathers inward by 64 pixels at interior edges. The
existing `bbox16:32@1` rule makes a 1120-square crop, denoises at 1040
square and resizes it back before ordinary overlay.

The first tile cancelled without a decoded/saved result; recovery and an
exact repeat matched. Every completed tile ran nine callbacks. All five
completed cropped outputs reconstructed bit for bit from their actual raw
working tiles, source, installed Lanczos paste and alpha overlay. Every
pixel outside each mask stayed unchanged. One load and all source/mask,
API/job/SQL/PNG/allocator checks passed, within unchanged resource guards.
The [dated verified result](../verified.md#four-region-sdxl-tiled-refine-look-2026-11-17)
and [receipt](gen49-sdxl-four-region-2026-11-17.json) retain measurements,
source identities, raw evidence hashes and cleanup.

Root and independent agent reviews inspected the overview and native
horizontal, vertical and centre source/final crops. They saw no obvious
rectangular boundary, centre cross or abrupt colour step; roof, cabin and
dock/water lines continue. Wood bracing, roof detail, foliage and reflections
substantially redraw and often smooth. A thin upright behind the roof is
more prominent or possibly introduced. The source did not faithfully realize
the prompt's bridge/river, so it establishes no bridge-preservation claim.
This is enough to investigate implementation. It proves neither useful
detail gain nor general fidelity, seam freedom, a default, or product quality.

## Initial boundary

Keep the first prototype at the exact measured 2048-square RGB canvas and
four-region algorithm. Freeze mask levels, overlap, feather, traversal,
crop rule/tier, resizing and per-tile seed arithmetic under an explicit
experimental algorithm identity. A changed algorithm gets a new identity.
The measured integer ramp is `min(255, max(1, distance * 255 // 64))`,
with distance starting at one inside an interior edge and corners taking
the minimum of their two axes. Its first painted level is **3**, not 1;
the scratch runner's old docstring says 1, but its retained code, mask
hashes and measured pixels use this formula. Canvas edges stay fully painted.
No flexible tile grid, automatic choice, other size/aspect, model/loader,
external server, mask supplied by the user, outpaint, instruction edit,
references or LoRA support follows from this experiment.

The CPU prototype receives an already prepared owned source and an injected
tile renderer. It does not load a model or access the gallery. It reuses
`mask_crop.region`, `crop_inputs`, `paste_back` and `refine.composite`.
Its image seed is the first tile's seed; tile ordinal is added through
`seeds.wrap_seed`. Batch image seeds retain the existing outer seed loop.
Each tile renderer receives exactly one generator seed and must return one
image at the planned working size. This private contract does not itself
qualify the eventual Diffusers adapter or promise public seed reproduction.

## Stages and completion

### 1. Private CPU orchestration

Build an unregistered module with immutable plan/result facts and one
serial in-memory composite. It must not change request schemas, profiles,
capabilities, public APIs, settings, the resolver or existing keys. Validate
the whole input and recipe before invoking a renderer. Keep caller-owned
images unchanged. Close owned masks/crops/intermediates on success, error
and cancellation; only a complete final image transfers to the caller.

Test frozen masks/hashes/crop geometry against the measured experiment,
the installed crop/composite arithmetic, current-source carry between
tiles, wrapped seeds, exact output count/mode/size, progress, cancellation,
failure and image ownership. Tests use synthetic CPU renderers and no
weights. Run focused checks, Ruff, the pinned type gate and the full suite.
Record evidence below, with real pipeline/GPU integration still unclaimed.

### 2. Exact backend integration and durable intent

Connect a private adapter over the existing loaded inpaint sibling,
without duplicating the pipeline, quantization, offload or sampler code.
One ordinary job cell produces one final image; internal tiles are not
gallery images, nested jobs or independent user-visible results. Do not
expose temporary gallery ids as the composite's source.

Keep the original root source id and full RGB/dimension/provenance proof
distinct from transient in-memory composites. Prove the root before work
and at every tile boundary. The first integrated recipe starts from a
1024-square RGB gallery root and prepares the 2048-square canvas with the
same exact CPU Lanczos resize. The prototype's source hash proves that
prepared canvas, not the original gallery root; retain both proofs and the
versioned preparation method. Internal masks are derived from the frozen
algorithm and never become sources or references. Preserve prompt, negative,
strength, steps, CFG, selected checkpoint and outer image seed unchanged.
One cell's cancellation/error produces no final image; completed earlier
cells remain ordinary results. Resume/Finish reruns an unfinished composite
from its original root, rather than treating a partial tile as complete.
Pause/cancel use the existing cooperative callback and owner cleanup.

Design one explicit tiled-pass spec and a versioned reproduction key before
public admission. Do not overload an ordinary mask hash with a fabricated
value, repurpose an existing slot silently, or insert an unconditionally
present field that changes old `run_id`s. Keep actual crop boxes, timings
and peak memory as results. Audit `cell_from_record`, completed-cell lookup,
More seeds, Finish, CLI, PNG import/reindex and bundle validation together.
New storage columns require migration and portable source-proof rules.
Choose bundle versioning from the actual carry contract, preserving every
existing edit-free v1 and edit v2 fixture. A tiled-containing bundle needs
explicit feature/version handling so an older reader cannot silently drop
the recipe. Unresolved sources retain intent
and refuse replay. No release precedes those contracts and regression checks.

### 3. Guarded private card qualification

Freeze several contrasting owned scenes before rendering: architecture and
straight joins, organic texture, lettering/fine edges, and a subject crossing
region boundaries. Preserve baseline failures, all intermediates and native
seam crops. Compare fidelity and useful detail, not merely smooth joins.
Use the actual new adapter in one guarded load at the exact initial recipe;
record real per-tile callbacks, sigma/counts, seeds, source proofs, allocator
and host/card/shared peaks. Check repeat, batch versus solo, cancellation
inside a later tile, recovery, final residency and ordinary generation after.
No larger canvas, different recipe or faster/lower-memory claim is inferred.

Use current card guards and a checkpoint-specific admission budget. The
exploratory 24 GiB RAM/commit admission, 16 GiB RSS cap and 2500 MiB idle-card
allowance supported that attempt; live 15500 MiB card, 1024/512 MiB shared
rises and 4 GiB RAM/commit floors were unchanged. Recheck current headroom;
these descriptive results do not authorize relaxing a failed admission.

### 4. Public flows and release

Offer a deliberate opt-in Refine choice with server-owned eligibility,
geometry, progress/counts and refusal reasons. Existing whole-image Refine
remains its existing choice. No implicit replacement or automatic tile mode.
Use an `aria-disabled` hold with a press explanation where unsupported.
The initial release remains limited to its independently measured envelope.

Qualify single/batch Refine, applicable Mass/Then refine, More seeds,
rerun/recovery, CLI, PNG and bundle carry through ordinary paths. A public
flow that is deliberately unavailable must retain an explicit refusal and
its intent. Real-browser fake-server checks must cover controls, holds,
progress and source/spec carry; real card/API/standalone checks qualify their
separate scopes. Update rules, design, usage and generated capabilities only
when support actually opens. Record visual limitations without promising an
improvement or Matt's preference. Remove `GEN#49` only after that bounded
release and its dated verification are complete.

## Private CPU result

[`backend/tiled_refine.py`](../../backend/tiled_refine.py) implements the
unregistered first stage under the private provisional identity
`private-provisional-sdxl-four-region-2048-v1`. It validates exact recipe
field types, RGB mode, dimensions and storable integer seed before a renderer
runs. Immutable plans bind the prepared source's full RGB proof, four frozen
mask hashes, exact crop boxes/working sizes and wrapped per-tile seeds.
Each renderer receives borrowed private crop inputs and the fixed recipe,
returns one fresh 1040-square RGB image and reports nine ordered steps.

The wrapper composes through the installed helpers in memory. Facts retain
each transient source/result RGB proof and callback count. Cancel, callback
failure and malformed progress latch even when a renderer catches them;
success transfers only one complete image. Borrowed root/input aliases are
refused, malformed image containers release their transferred fresh images,
and owned crops/masks/raw images/intermediates close on every terminal path.
Planning masks use explicit `closing`: Pillow's image context exit alone
does not close an in-memory pixel core. No gallery rows/files, schema fields,
keys, profiles, resolver, route, settings or capability changes were made.
The sole rules change maps the new leaf module in Layout.

Final focused verification on the frozen source passed **110 tests**:
96 prototype cases, eight module-map and six rules-section checks, with one
existing Starlette warning, in 10.50 s (`CUDA_VISIBLE_DEVICES=-1`, OMP 1).
The tests pin the measured masks/crops, real helper arithmetic, serial
current-source carry, seed wrap, outside-mask identity, exact output and
progress contracts, later-tile cancellation/helper/progress errors, all
owned image lifetimes and a cold config-before-renderer/framework boundary.
Ruff passed; pinned Pyright checked **218 sources**, zero errors/warnings.
Root and independent source/test reviews found no remaining blocker.

The final integrated suite passed **11167 tests, 325 skipped**, 85 warnings,
exit 0 in 418.69 s, with `CUDA_VISIBLE_DEVICES=-1` and four workers. Its log
is `work/overnight-tiled-final-2026-11-17-cpu.log`. The source-bound focused/
static receipt is `work/gen49-private-cpu-validation-20261117.json`.
An independent CPU replay of the actual saved GPU tiles passed in a fresh
process. It injected the first-tile recovery and the remaining three raw
1040-square images, with 36 explicitly synthetic callbacks. All four full
intermediate RGB hashes and the final composite match the measured chain
exactly, including final `7a04e5835b31c39ed8947c92be67e4ead678f24ac0c775a3075f77767212d6ad`.
Input crops/masks/seeds match the original source/crop proofs. The unmodified
installed `VaeImageProcessor.apply_overlay` was observed four times; no
replacement overlay or new inference was used. Original raw files stayed
unchanged. The replay is CPU orchestration evidence, not new GPU, quality,
pipeline, durability or performance qualification.

The first two replay attempts stopped before a completed overlay because
their import-only guard rejected optional Triton availability and then
installed Inductor metadata imports. Those failed attempts remain preserved.
The fresh passing attempt allows CPU metadata registration and records
ordinary optional-package absence separately from execution. CUDA, actual
compile entry points, learned model/pipeline imports, subprocess and network
guards stayed enforced. Its raw receipt is
`work/gen49-private-raw-cpu-replay-20261117-a3/receipt.json`; the independently
audited [bounded derivative](tiled-refine-raw-cpu-replay-2026-11-17.json)
retains the helper/source proofs and separate failed-attempt disclosures.

## Private loaded adapter and durable intent, 2026-11-20

The candidate load shares the ordinary Diffusers loader transaction and reads
the pinned Animagine snapshot from the local cache. It requires Windows,
RTX 5080 CUDA 0, fp8, model offload, Eager/Native and Diffusers 0.42.1. A
normally loaded pipeline cannot stand in for the pinned candidate. The
candidate's SDXL inpaint sibling shares the loaded VAE, UNet, scheduler and
text encoders. The adapter checks the sibling's class, latent scale and call
parameters before the first tile. Each tile uses a fresh CPU generator, the
requested prompt and negative prompt and the frozen strength, steps and CFG.
It returns the loaded owner's offload hooks after each forward. Progress
counts all 36 denoising callbacks per composite. The card receipt takes
conditioning from the job request and geometry, seeds and proof from the
tile plan. The prototype's example prompt stays in its exploratory recipe.

The stored `tiled_refine` spec carries the algorithm, original gallery id,
original basename, size and full RGB digest, the exact 2x Lanczos preparation
and its full RGB digest. The adapter checks the prepared canvas before work
and the live original before each tile and after the final tile. Tile
images and masks stay inside the adapter. A private queue job uses one
ordinary image save per completed cell, and cancellation closes the partial
composite. Retained cells preserve their bound source and recipe for a
deliberate private retry from the original root.

The reproduction key adds a field only for a Tiled Refine result. The row,
PNG and version 3 bundle preserve the recipe and both proofs. PNG reindex
and database merge verify both proofs before resolving the root. Missing or
changed roots keep unresolved intent. Existing bundles still use versions
1 and 2 for their existing features. Public Generate, Grid, Re-render,
More seeds, Finish and CLI replay return a Tiled Refine hold.

CPU checks use a scratch gallery, a stubbed snapshot lookup, synthetic SDXL
owners and actual crop and overlay helpers. They check both proofs, the
pinned load transaction, generator wrap and batch-versus-solo
identity, shared ownership and cleanup, all 36 callbacks, cancellation and
discard, changed roots and a private JobManager cancellation and recovery
through SQLite and PNG metadata. The new adapter's image quality, resource
use and residency await stage 3.

### November 20 card protocol

The private trial uses the local pinned Animagine snapshot above. Four original
1024-square procedural illustrations were frozen before any load. Each cell uses
the fixture's own prompt, negative `low quality, blurry`, 28 steps, CFG 5,
strength 0.35 and the exact four-region preparation. The fixture manifest has
SHA-256 `931fd50cbfdb4a6af86cc2fa6aac675d46fbd727bce997be2a087b1d3eb362a2`.

| Source | Seed | Detail to compare with the 2× Lanczos preparation |
|---|---:|---|
| Truss bridge | 4101 | Rail continuity, repeated diagonal braces and four piers |
| Fern | 4201 | Alternating leaflets, serrations and pale diagonal stem |
| Tide gauge | 4301 | Exact lettering “TIDE 07”, ruler ticks and four screws |
| Koi | 4401 | Single outline across both joins, four red patches, two fins and forked tail |

A fifth control carries the actual November 17 generated cabin/shoreline
root, exact original conditioning and first tile seed 12346. It compares
this adapter to the old four-job calculation. The bridge also runs as a
repeat, in a two-image adapter call with seeds 4101/4102, and solo at 4102.
A private queue job cancels at step 12 of 36 inside tile two. Its retained
cell supplies the recovery request. An ordinary 1024-square generation
follows on the loaded owner before unload. Private scratch files retain raw
tiles, partial/completed composites and full-sized seam crops for inspection.

The November 17 session peaked at 5,951 MiB whole-card, 9.31 GiB process
tree RSS and 11.27 GiB process private. Cold admission requires at least 24
GiB each of free RAM and free commit and at most 2,500 MiB whole-card use.
Immediately before load, physical availability still needs 24 GiB. Its
24 GiB commit requirement credits only the newly spawned child's measured
private allocation. Live ceilings are 15,500 MiB whole-card, 1,024/512 MiB
adapter/process shared rise, and 16 GiB each of tree RSS and private memory.
Both free RAM and free commit must stay at or above 4 GiB. A Windows Job
Object holds the child and descendants under the external watchdog until
exit. Three distinct card samples at or below 2,500 MiB verify post-unload
residency. The supervisor refuses a missing or stale measurement.

The [trial sources](gen49-stage3/) include the Windows supervisor, observing
child, independent CPU auditor, guard tests, original drawing generator and
November 17 baseline. The [frozen fixture manifest](gen49-stage3/manifest.json)
records full RGB and PNG digests, prompts and visual criteria. To reproduce
the trial from an idle Windows checkout, use its Python environment with the
project dependencies installed. The generator compares every rebuilt fixture
with the frozen manifest. Its original byte-for-byte source is at `ad0912f`.

```powershell
$protocol = 'docs/maintainer/gen49-stage3'
python "$protocol/draw_sources.py" work/gen49-stage3-fixtures
python -m pytest "$protocol/test_gen49_stage3_guard.py"
python "$protocol/gen49-stage3-guard.py" work/gen49-stage3-new "$protocol/gen49-stage3-run.py"
python "$protocol/gen49-stage3-audit.py" work/gen49-stage3-new
```

The guard invocation is the real card run. It requires the pinned weights
already in the local Hub cache. Private JobManager recovery saves the only
durable tiled result inside the attempt's scratch gallery. Observer PNGs,
including any completed prefix of a cancelled job, live beside that gallery.

**Initial guarded attempts.** The first cold check saw 31.59 GiB available
RAM and 23.86 GiB free commit (2,027 MiB card), so it never spawned a
child. The second attempt admitted at 31.71/24.12 GiB and 1,967 MiB;
immediately before load it measured 31.24 GiB RAM and 23.31 GiB free commit
plus 0.88 GiB of the owned child's private bytes. The Hub's offline cache
checker then refused the unfiltered snapshot for missing root files,
including documentation and an alternate monolithic checkpoint. No weights
loaded. The private candidate now requests only the pinned Diffusers
manifest and its seven component folders. A real local cache lookup passed.
A scratch-cache regression also removes an encoder weight and verifies refusal
before model loading. The third cold check saw 29.40/21.53 GiB and 2,063 MiB
card use, with no child spawned. The [attempt receipts](gen49-stage3/attempts.json)
retain the supervisor, worker and raw counter hashes.

Serial CPU checks passed all 34 loader, adapter and supervisor cases. Ruff
passed. Regenerating the four fixtures from the checked-in source reproduced
their frozen manifest, PNG and prepared RGB digests. No stage-three image
or visual result follows from these three attempts.

## Private card result, 2026-11-20

The full session (A4) used one offline pinned Animagine load. The four
drawings and fifth historical control produced ten complete composites in
total across their original, repeat, batch, solo and recovery calls. Each
completed tile made nine real EulerDiscreteScheduler callbacks at full
schedule indices 19-27 (timesteps 281, 246, 211, 176, 141, 106, 71, 36, 1).
The first active sigma was 0.7789236903, the last 0.041314; its successor
was zero. The runner recorded per-callback actual sigmas, timestep and
next sigma, generator seeds, original/prepared proofs and 52 successful
boundary rechecks. Each successful composite ran 36 callbacks.

The independent CPU audit reconstructed all 41 decoded tiles' full
intermediates from their actual raw working images, source and frozen
masks. Every zero-mask pixel stayed identical. The architecture repeat, both
images in the two-image call and the recovery each matched the equivalent
solo image, including intermediate digests. The fifth control reproduced
all four raw RGB and intermediate digests from November 17; its final RGB
SHA-256 is `7a04e5835b31c39ed8947c92be67e4ead678f24ac0c775a3075f77767212d6ad`.
Private JobManager Cancel reached callback 3 in tile two. The first tile's
diagnostic raster was retained, the second never decoded, and the gallery
kept its original five roots with no final or partial row/PNG. The retained
cell restarted from its original root with the same run id, saved one
verified row and PNG, and matched the original direct result. Ordinary
1024-square generation then completed all 28 steps on the same owner.
[Full-run receipt and file hashes](gen49-stage3/trial.json) and
[Independent reconstruction](gen49-stage3/pixel-audit.json).

The 258-second supervised session retained 1,031 samples (longest gap
0.272 s). It admitted cold at 33.039 GiB available RAM, 25.441 GiB free
commit and 1,810 MiB whole-card, then at 32.481/24.560 GiB and 1,732 MiB
immediately before load. The 28.08-second load and every render stayed
inside the frozen limits. Whole-card peak was 5,860 MiB, adapter/process
shared rises 94.21/82 MiB, process tree peak RSS/private 9.46/14.42 GiB,
and free-RAM/commit minima 23.67/11.14 GiB. A solo or serial two-image
tiled call peaked at 3.29 GiB allocated and 3.66 GiB reserved. These
observed timings include saving each instrumented tile and composite.

The normal unload made the backend unready and released both pipeline
objects. Three post-unload samples read 2,081, 2,089 and 2,089 MiB on the
whole card. A4 then exited 1 on its original strict assertion: torch
still reported 64 MiB allocated and reserved. This exit is retained in
the [supervisor receipts](gen49-stage3/attempts.json).

Two separate guarded controls identify that allocation. A5 used only a
512-square matrix multiplication on the main and worker threads. It
observed 32 MiB, then 64 MiB, persisting after the worker exited.
A6 used a fresh real model load and reproduced A4's bridge through the
direct adapter and private JobManager with different native thread IDs.
After normal unload, the two surviving allocator blocks had the same
addresses as the main- and worker-thread 32 MiB blocks observed while
the model was loaded. Clearing **only cuBLAS workspaces** freed both
blocks (zero allocated, 64 MiB reserved); releasing the allocator cache
then brought reserved bytes to zero. Both controls exited 0 with the
unchanged resource guards. Production unload remains as measured. To
repeat a control, substitute `cublas_residency.py` or `loaded_residency.py`
for the child in the supervisor invocation above and use a fresh attempt
directory. [No-model control](gen49-stage3/control.json) and
[Loaded-route confirmation](gen49-stage3/loaded-control.json).

Native horizontal, vertical and center comparisons against the exact 2x
Lanczos source found continuous bridge girders and diagonal braces, but
subtle bands in the previously flat sky. The main fern stem and leaflets
survived while fine veins and serrations softened. **TIDE 07** stayed readable;
the D corners rounded and its flat field acquired banding. The koi retained
one outline, the fins, forked tail and four red markings, but lost two
fine scale arcs inside a red patch. The reviewers saw no conspicuous
central cross and no consistent gain in faithful detail across these four
drawings. Raw native crop pairs and per-scene images are in
`work/gen49-stage3-20261120-a4/audit/`. [Four-scene overview](gen49-stage3/overview.png).

The private result covers the measured source size, model, recipe and
loader above. Final CPU
verification passed 165 focused tests, including queue, storage, reindex,
bundle, rule and guard cases. Ruff passed; pinned Pyright checked 235
backend/tool sources with zero diagnostics using the shared project venv.

## Public workflow result (2026-11-21)

Refine offers **Tiled Refine** on the fixed source and loader envelope.
The normal load prefers the verified pinned files from its local cache
and resolves that revision when needed. If it cannot resolve the pin,
ordinary Animagine can load and the tiled choice gives the hold reason.
Public submit, Reuse, More seeds, Finish and the CLI keep separate tiled
intent and the two source proofs. Grid, Mass refine, Then refine, size
changes and browser PNG settings have explicit holds for the tiled recipe.

The browser smoke used two versions and checked eligibility, fixed settings,
progress, persistence, Reuse, More seeds and source bin and restore. The
regular Refine smoke also passed. Backend tests covered queue admission,
retained recovery and PNG and archive proof carry, including a selective
restore with colliding gallery IDs and a missing source.

On the card, the frozen external guard admitted one offline server load
through the ordinary loader. The queue cancelled the second image in tile
two and then finished just that image from the original root. Both RGB
digests matched the private trial. More seeds carried the same proofs to
seed 4103, and the browser PNG route returned its deliberate hold. A fresh
standalone CLI load then replayed seed 4101 from its PNG and matched it pixel
for pixel. The two scratch attempts retained API and CLI results, raw
readings and guard receipts in `work/gen49-stage4-api-20261121-a1` and
`work/gen49-stage4-cli-20261121-a1`. Both unloaded within the original
limits. [Measured peaks and sample coverage](../verified.md#public-tiled-refine-on-animagine-xl-40-2026-11-21).
