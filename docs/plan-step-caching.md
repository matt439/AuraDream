# Plan: opt-in First Block Cache on FLUX.1

**Status, 2026-10-26: CPU contracts, persisted carry, private loader/call ownership
and cold request candidate checks are built.** The unavailable per-render selector,
browser persistence, holds and Reset/Reuse/Undo source are built; real request-driven
cache admission/dispatch and a separately
approved card comparison remain `PERF#4`. This is the
bounded next slice from [R4](review-2026-10-25.md#r4-step-caching-for-the-long-undistilled-schedules).
No package installation, runtime model import, weights, tensor allocation
or GPU generation was used for the source stages. Public caching stays unavailable
and defaults are unchanged.

The selector stage keeps step cache as composer intent, never a loader preference.
Off is the default; choosing FBC retains the requested recipe while showing its
unavailable reason and holding Generate/grid/comparison before a POST. Ordinary
Eager Reuse explicitly chooses Off and Undo restores the previous intent; known
nonempty stored records retain their existing held Reuse boundary. Unsupported
placement/loader combinations are described by the server rather than inferred
as supported in the page. The shared
[configuration stage](plan-regional-compilation.md#unavailable-performance-configuration)
records source contracts and the partial browser evidence. No actual request can
enable cached rendering, and no quality/time/memory claim follows from selection.

The shared `ExecutionSpec`, nullable SQL/PNG carry and refusal paths are
implemented with `PERF#1`. A recorded `fbc-005-v1` remains unavailable to
render or Reuse. The inactive helper has synthetic ownership tests and an
isolated installed-FBC CPU probe; real model arithmetic, quality, speed and
memory remain unqualified.

## Inactive ownership helper built

`backends/fbc_cache.py` validates the exact eager FLUX root and joint/single
block classes, installer order, configuration counts and registered output
metadata before setup. It owns a fresh explicit 0.05 configuration for one
call and tracks installed hooks even if installation fails before assigning
the library's cache configuration. State and current context reset before
and after each call. Cleanup removes only those hooks and restores prior
forward attribute ownership, foreign hooks and child-registry caches.
The installed registry caches its child list without invalidating it on
hook changes; the helper invalidates it during setup and restores it on exit.

One identity-based weak claim covers a denoiser shared by base and sibling
pipelines. Duplicate owners refuse. Cleanup failure marks that exact model
unsafe before invoking the caller's shared-pipeline disposal callback;
closing or losing the owner cannot make the same model safe again. A fresh
model identity is required. Valid close releases ownership and references.
Original setup/call failures retain cleanup diagnostics, with no fallback.

All 52 focused synthetic tests pass, including partial setup, cancellation,
reset/removal failure, forward restoration, shared identity races and weak
lifetime cleanup. Ruff and narrow pinned Pyright pass. Independent review
identified the duplicate-owner recovery gap; its regression covers a failed
reset that leaves no visible hooks yet still makes reuse unsafe. These 52 tests stub
the real FBC/class adapter; no real FBC import, tensor, forward or compiler
execution was used. The ordinary test conftest still performs its existing
CUDA-availability check. Loader/generation/UI callers remain absent and
every nonempty execution recipe remains refused.

## Private loader, invocation and deferred disposal

Private cache-only preparation cold-checks the matrix and retains its owner after
ordinary placement/checks, without installing hooks. Pipeline, recipe and owner
transfer together; closed/orphan/mismatched state refuses. Public `load()` still
supplies None and every public nonempty request/applied/generation guard stays
closed. One private base forward requires exact root/owner/layout, batch one,
ordinary guidance and no adapter/compile/attention combination.

Effective Native dispatch is checked read-only using the source-verified lazy
getter and complete processor coverage; empty app metadata alone is insufficient.
Nonnative/unknown state refuses without resetting globals/environment. Each call
owns fresh cache hooks/state. Post-call changes fail the result while preserving
primary errors/cancellations with diagnostics, never retrying.

The expected owner's failure callback immediately detaches unsafe base/sibling/
adapter/recipe state under the backend lock into explicit pending disposal.
It closes no owner/allocator there. Per-call ownership drains once after the
cache helper, sigma/`no_grad` scopes and application aliases unwind, outside locks;
direct private calls own a local boundary. Pending disposal holds load/forward,
and stale callbacks cannot detach replacements. No TLS/admission bypass exists.
Terminal discard weakly invalidates unsafe roots without removing foreign hooks/
cache; ordinary repair semantics remain.

All **49 new isolated cases** pass with `--noconftest`; the broader backend/helper/
execution/refine group passes **745 checks**. Both production modules pass pinned
types with zero errors/warnings, Ruff passes and independent review is clear.
Cases cover partial setup/cleanup/reset/foreign replacement, cancellation, scope/
lock order, stale callbacks, pending not-ready state, weak invalidation and Native
changes during successful/failed/cancelled forwards. An old kwargs-introspection
test now follows the extracted body, preserving exact `refine.CARRIED` equality
and exclusions. All author jobs ended. These are synthetic wiring/lifetime
checks, not real imported dispatcher/FBC/model/device/kernels, memory, pixels or
main-status/card qualification. The combined whole-source gate follows the
concurrent credential-source stage.

## Application readiness after private disposal

The app now derives a returned ready/unloaded state from actual backend readiness
without mutating the load supervisor's cached state. A discarded private pipeline
therefore reports not-ready; a GET starts no replacement load. Loading/error states
and their details remain authoritative. Explicit Load independently checks actual
readiness, including the unchanged-checkpoint Switch & load path, so recovery does
not depend on a preceding status poll. Existing `ensure_loaded` locking and the
generation gate still own the one ordinary reload attempt.

This closes latent app/private lifecycle groundwork in this owner. Today's public
load still supplies None and nonempty generation guards prevent entering private
FBC through a valid public request. Fake readiness tests do not qualify that path,
physical GPU release or real cache arithmetic. Public selection/admission and the
fixed-pair quality/time/memory qualification remain open.

All **175 focused API/model-switch/analysis checks** passed, including 13 new fake
readiness cases with event-held loads and a concurrent request barrier. They prove
returned status is observational, explicit reload needs no preceding GET, loading/
error details remain supervised and retries/coalescing keep selection/preferences
unchanged. The changed production file passes pinned types with zero diagnostics,
Ruff passes and independent review is clear. Every author validation job closed;
no actual private-FBC/public request, model, compiler or GPU run occurred.

The complete offline gate then passed **5,441 tests, eight platform skips and
75 existing warnings** in **235.87 seconds**, using two workers with file-based
distribution on Windows/Python 3.13.15. Whole-scope pinned Pyright checked
**128 production/tool sources with zero errors/warnings**, repository Ruff passed
and all validation jobs closed successfully. This establishes the bounded source/
fake app lifecycle behavior; public selection and card/quality acceptance remain.

## Shared Native attention invariant

The effective-Native implementation was subsequently moved intact to the shared
attention helper, with this module delegating. Regional preparation/calls now
enforce the same invariant instead of trusting a startup warning. The refactor/
regional fix passed **755 focused** and **154 isolated synthetic checks**, pinned
three-source types, Ruff and independent review, without changing cache policy,
environment, eager behavior or public nonempty admission.

## Installed FBC CPU probe

Four focused assertions share one fresh guarded worker using the installed
FBC hooks, CacheMixin, HookRegistry and StateManager on tiny registered CPU
blocks. Only defining model/block bindings and registry bootstrap are
synthetic; no Flux constructor or checkpoint is used. The first call computes
every block, the repeated residual skips later blocks, and a changed leader
residual recomputes them. Exact outputs and call counts verify both tuple
components. Cond/uncond state remains separate; cached/off/cached calls start
fresh, and a propagated cancellation resets context/state and restores ordinary
bound forwards before subsequent off and cached calls.
The context cases use identical numerical inputs: they prove state/miss
separation, not real conditional/unconditional model outputs.

All four checks passed in 5.76 seconds with `--noconftest`, Ruff clean and
narrow pinned Pyright reporting zero diagnostics. The worker forces CUDA
availability false before library decorators, keeps the native initializer
guard installed and reports zero native/lazy CUDA, compute, compile and
network guard calls, uninitialized CUDA and one CPU/inter-op thread. Owned
compiler/debug directories remain empty. The receipt is
`work/pytest-perf4-fbc-runtime-20261026/fbc-runtime0/receipt.json`.
This is installed-library CPU arithmetic/lifetime evidence; real Flux,
offload, GPU behavior, speed and visual quality remain unqualified. There
are still no loader, generation or UI callers, and render admission stays closed.
Independent source review accepted the guard, arithmetic and lifetime coverage
with the conditioning-output limitation above.

Final combined integration passed **4,720 tests, eight skips and 75 existing
warnings** in **212.03 seconds** with two workers on Windows/Python 3.13.15.
The full production-source type gate analyzed **116 files with zero errors or
warnings**, and repository Ruff passed. Ordinary suite fixtures retain their
established CUDA-availability check; none of these results qualifies a real
checkpoint, GPU performance, pixels, Linux or hosted CI.

## One algorithm and one undistilled family

Start with diffusers' **First Block Cache (FBC)** on the exact catalogue
id `black-forest-labs/FLUX.1-Krea-dev`, fp8, CUDA, model offload, native
attention and eager execution. This undistilled FLUX.1 integration has
registered block types and a cache-aware pipeline in the installed
diffusers 0.40.0. Its usual 28-step schedule is the first qualification
case. Krea 2 Raw, klein base and Z-Anime remain later family slices.

Use `FirstBlockCacheConfig(threshold=0.05)` explicitly. This is the
library's documented dataclass default in the
[versioned implementation](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/hooks/first_block_cache.py),
not a tuned Krea-dev recommendation or measured quality setting. Propose
the fixed recipe id `fbc-005-v1`; no arbitrary threshold box in the first
UI. Another threshold or changed algorithm becomes another recorded
recipe. Off remains the default.

The [original FBC description](https://github.com/chengzeyi/ParaAttention/blob/7a266123671b55e7e5a2fe9af3121f07a36afc78/README.md#first-block-cache-our-dynamic-caching)
uses first-block residual change to decide whether later blocks can reuse
previous residuals. Its benchmarks concern other hardware/settings. The
[Diffusers cache guide](https://huggingface.co/docs/diffusers/main/en/optimization/cache)
also describes approximate reuse. Neither establishes a local speedup or
acceptable Krea-dev quality. Do not adopt the guide's different-model
example threshold as though it were this recipe's measured default.

| Request | First implementation proposal |
|---|---|
| Cache off | Existing capabilities and reproduction behavior |
| Exact Krea dev, fp8/model offload, CUDA, one direct image, ordinary guidance | Candidate FBC recipe; qualify the default schedule first |
| Another family/quantizer/offload, Batch >1, true CFG >1 | Refuse this explicit cache recipe initially; do not drop it in `applied()` |
| LoRA, Refine, inpaint, outpaint, Then refine or Mass refine | Refuse cached passes initially; explain selecting cache off |
| Compilation, nonnative attention or another enabled cache | Separate combination qualification; no silent stacking |

This narrowing bounds the first experiment rather than asserting library
incompatibility. `GEN#22`'s cold fp8 RAM check remains a prerequisite to
the proposed card run. No model download is authorized by this plan.

## Source and reset findings

Inspected installed `diffusers/hooks/first_block_cache.py`, `hooks.py`,
`_helpers.py`, `models/cache_utils.py` and the FLUX pipeline sources:

- FBC always computes the first block, compares its normalized residual
  difference with the threshold, and may reuse cached residuals through
  later blocks. It does not remove scheduler timesteps. Progress,
  cancellation and recorded Steps must retain their existing meanings.
- The generic installer gathers recognized top-level `ModuleList`s in
  child order, selects a leader and tail, and resolves block metadata
  through `TransformerBlockRegistry`. Validate that list, its minimum
  length, block classes and output conventions before installing hooks;
  a matching family substring or an `enable_cache` method is insufficient.
- Both FLUX block types are registered. Flux2/Krea2 block types are not
  registered there; Krea2 and ZImage also lack `CacheMixin` in their
  declared model bases. Those review examples need their own compatibility
  work rather than being added to a support table from architecture names.
- `FluxPipeline` separates `cond` and `uncond` cache contexts. StateManager
  owns a separate state per context and removes states when reset.
  Normal `maybe_free_model_hooks()` resets stateful denoiser hooks, but a
  callback exception can bypass this normal pipeline tail. The installed
  `cache_context` generator does not protect context restoration with a
  `finally`. Application cleanup must cover exceptional exits explicitly.
- `enable_cache` assigns its `_cache_config` after hook installation.
  Partial installation can fail before that assignment, when ordinary
  `disable_cache()` sees no enabled config. The installer must be
  transactional from the app's perspective. `disable_cache` removes the
  FBC leader/block hooks recursively when configuration is present.

These source findings motivate CPU contract tests; they are not a runtime
qualification of FBC under torchao or offload.

## Per-call ownership and failure policy

`execution.cache_candidate_reason` now implements the initial cold request and
distribution matrix from supplied facts: exact model/loader/device/batch, no
adapter/edit pass, no stacked recipe, ordinary guidance, no existing cache and
no nonnative external attention override. It does not inspect a loaded model or
read the environment, import a framework or grant runtime permission. Loaded
APIs, ownership and arithmetic remain separate checks. The shared cuDNN/FBC
slice passed 122 focused tests with `--noconftest`, narrow pinned Pyright and Ruff;
an independent review accepted it. All nonempty recipes remain unavailable.

Keep weights loaded by `ensure_loaded` alone. A cold capability helper
answers whether the request matches the recipe without importing/loading
a checkpoint. After the ordinary load, validate actual cache APIs and
block coverage without installing hooks permanently at startup.
Resolve the effective attention backend, including external overrides;
an empty app setting alone does not prove that native attention will run.

One small backend context owns cache setup for **one pipeline call** after
adapter checks and before entering the denoising loop. It starts with no
app-owned FBC hooks/state, enables a fresh fixed configuration, and always
disables/removes its hooks in `finally`, including cancellation, preview or
pipeline failure. Keep `torch.no_grad`, sigma construction and one seeded
generator per image unchanged. Never reload weights from generation.

Before installation, reject a foreign existing cache or conflicting hook
names. Track exactly the app's added hooks so a partial setup can unwind
only those registrations through `HookRegistry.remove_hook`; protect this
version-sensitive adapter with installed-source tests. Do not rely on a
half-set `_cache_config` or patch library defaults. On cleanup failure,
fail closed: discard the unsafe base and all shared sibling references,
report not-ready, and let the next ordinary `ensure_loaded` perform any
reload. No later cell may render with uncertain hooks under an off recipe.
Preserve the original failure alongside any cleanup error in diagnostics.

**Next source integration, with admission closed.** The private implementation
currently creates its cache owner from a load-owned cache recipe and dispatches
from `_loaded_execution`. The composer choice now exists, but actual per-request
cache-on/off/on over one ordinary Eager load is not wired. Add a private request
dispatch seam with fresh call ownership and unchanged loaded recipe, using the
existing deferred disposal rather than keeping residuals or reloading for every
toggle. Synthetic tests must cover on/off/on, a new seed/cell, cancellation,
foreign state and cleanup failure followed by safe uncached recovery. Keep every
public availability guard closed; no real FBC/model/card run is needed for this
source stage. Card quality acceptance remains a separate later requirement.

Fresh ownership prevents state crossing image seeds, prompts, sizes, cells,
jobs, retries, Pause/resume or checkpoints. Do not cache tensors on disk or
serialize them with retained jobs. A cached-to-off-to-cached sequence must
behave as three independent calls; disabling must restore ordinary module
forwards and allow normal offload-hook cleanup.

Sibling pipelines share the denoiser object. Before any uncached sibling
call, prove no app cache remains on that shared object. Later cached-edit
support must set fresh contexts and ownership for each base/pass call,
including the dynamic Then refine sequence, rather than reusing the base
residuals. Qualify every edit schedule, mask/source and peak separately.
LoRA support similarly needs clean state around injection, removal and
reweighting; weight zero still denotes an attached adapter. Initial
refusals preserve these requests rather than approximate them silently.

## Per-render choice and reproduction

Propose nullable `step_cache="fbc-005-v1"` on `GenerateRequest`; absence
means off and leaves existing metadata and identities unchanged. The
single `applied()` validates capability and retains the effective recipe.
The selected model's cold status/profile supplies a capability and reason,
not an implicit opt-in. Add a small off-by-default choice beside Steps,
persist it with composer settings, and carry it in Reuse/Undo/Reset as
other generation settings do. Switching to an unsupported model must show
and resolve the incompatibility before submission.

Add a nullable migration and unguarded row read, both storage insert paths,
PNG chunk, retained jobs and reindex. Carry it through `grid.expand`,
`grid.cell_from_record`, CLI and its deliberate job-loop copies,
`--from-image`, PNG settings, Re-render, Swap tags, More seeds, comparisons,
Resume/completed/planned/duplicate checks and verified bundle export/import.
Mass refine cannot inherit a cached source into an unsupported cached pass;
its planner must refuse or apply an explicit user-requested off conversion.
Do not silently strip a reproduction field because the source is foreign.

Coordinate compilation/attention/cache identity work before implementation:
widen `CellKey` once with a canonical optional execution-recipe extension
whose named fields cannot collide. The empty extension must preserve every
legacy hash/run id; a nonempty cache recipe must distinguish cached and
uncached cells. Include that extension in storage lookups and run hashing,
not just the browser request. Timing estimates must separate recipes or
label an unmatched timing as a reference. Cache hits, measured savings and
RAM/VRAM peaks are results, never reproduction-key components.

## Implementation stages and finite acceptance

1. **CPU carry and refusal contracts.** Cover old rows/PNGs/jobs/preferences,
   nullable migration, legacy and distinct keys, Resume/queued deduplication,
   every rerun/import/export path, CLI/job equality and matrix refusals
   before submission. No weights or network in ordinary tests.
2. **Lifecycle tests and UI.** Small synthetic registered blocks test full
   computation versus reuse, fresh first calls, distinct contexts,
   output conventions, partial setup, cleanup failure, callback cancellation
   and uncached shared-sibling isolation. Fake backend tests cover recipes
   across cells/jobs and unchanged step callbacks; an isolated browser
   smoke covers opt-in, held reasons, switching and Reuse/Undo. Run focused
   CPU checks, docs maps and ruff before card work.
3. **Separately approved card pass.** Complete `GEN#22` first. Fix three
   representative prompts and two seeds each before running: ordinary
   scene, multiple interacting subjects, and fine texture/legible text.
   Use identical 1 MP shape, 28 steps, CFG/mu and eager/native fp8/model
   settings with cache off versus `fbc-005-v1`, one image at a time. Keep
   the first-call cost separate from repeats; do not change the threshold
   after seeing a bad pair without proposing a new recorded recipe.

| Proposed check | Evidence to retain |
|---|---|
| Six matched off/on pairs | PNGs/full metadata; image and step time; number of reused versus computed block passes; working-set/private-commit and allocated/reserved/whole-card/shared peaks |
| Repeated chosen pair, then a new process | Cached and uncached repeat hashes; no cross-call state; first versus repeated time and memory |
| Native/off baseline after a cached image | Same baseline pixels and ordinary forward/hook behavior restored |
| Cancel mid-call, subsequent off/on calls, unload | No cancelled save, no stale state and memory released at ordinary boundaries |
| Quality review of all fixed pairs | Prompt coverage, anatomy/interaction, fine detail and text, with visible pair comparison plus mean/max pixel difference; numerical closeness alone is not acceptance |

Stop at existing RAM/card guard limits or shared-memory spill. A skipped
block count does not prove a wall-time improvement, and a measured speed
gain does not establish acceptable quality. Matt must judge the fixed
pairs before the experiment is offered for routine use. Keep the feature
off by default; other models, schedules, batches and compositions are
separate slices. Leave `PERF#4` OPEN for implementation, then CARD while
the real comparison remains.
