# Plan: opt-in First Block Cache on FLUX.1

**Status, 2026-10-26: CPU-only planning slice complete.** Implementation
and a separately approved card comparison remain `PERF#4`. This is the
bounded next slice from [R4](review-2026-10-25.md#r4-step-caching-for-the-long-undistilled-schedules).
No package installation, runtime model import, weights, tensor allocation
or GPU operation was used. Nothing enables caching or changes a default.

The shared `ExecutionSpec`, nullable SQL/PNG carry and refusal paths are
implemented with `PERF#1`. A recorded `fbc-005-v1` remains unavailable to
render or Reuse; this establishes metadata transport, not hook installation
or quality, speed and memory qualification.

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
