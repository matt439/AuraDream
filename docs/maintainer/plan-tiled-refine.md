# Tiled Refine

**Status, 2026-11-17:** `GEN#49`'s throwaway SDXL seam experiment passed
the narrow gate for this plan. Product implementation is not released.
The private CPU orchestration prototype is being built; its results belong
below. The originating [R2](review-2026-11-10-peers.md#r2-tiled-refine)
owns why the experiment came first.

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
and at every tile boundary. Internal masks are derived from the frozen
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

Pending implementation and validation. The exploratory receipt above remains
separate from the prototype and every later integrated result.
