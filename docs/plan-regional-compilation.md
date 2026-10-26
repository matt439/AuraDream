# Plan: regional compilation of FLUX.1

**Status, 2026-10-26: CPU contracts, persisted carry, private loader/call wiring
and backend publication/teardown ownership are implemented and checked. Public
selection/admission, compiler/cache setup and card qualification remain `PERF#1`.** This is the next slice from
[R1](review-2026-10-25.md#r1-compile-the-denoisers-repeated-blocks), not a
speed result or approval to load a model. No package, model or compiler was
installed. The isolated CPU policy probe and its earlier native-CUDA initialization
failure are recorded below; no model or GPU generation was run.

## CPU contracts, admission and persisted carry

`backend/execution.py` defines one frozen, extra-forbid `ExecutionSpec`
with the three named recipe fields, canonical empty/key helpers, installed
distribution facts and the pure first-regional-candidate matrix. Even a
matching candidate cannot run. `GenerateRequest` recognizes the shared
field and refuses every nonempty recipe; submission, queue admission,
`applied()` and direct real/fake backend boundaries also refuse internal
requests which bypass schema validation. Queue admission checks the whole
cell list before mask stamping/holds, retained work or loading, even with
an explicit run id. The real direct-generation guard precedes device/peak
queries and pipeline work. Unknown fields or recipe names are rejected,
including unvalidated copies of the execution specification itself.

Only `GenerateRequest.applied()` writes empty execution back to None. Raw
validated requests may retain an empty specification; its canonical key is
None. This avoids another whole-request validation of a stamped outpaint.
The nullable `images.execution` column migrates on every connection. Shared
codecs carry canonical named JSON through records, PNG chunks, verified
bundles and both reindex paths. Known recipes remain inspectable and
portable without becoming executable. Unknown names and explicit SQL/PNG
disagreements are refused; a field absent from an older schema may recover
the other source's recipe. Gallery ids may remap or sources disappear
without dropping the recipe. Same-filename imports cannot replace another
recipe's identity. File-based readers explicitly refuse app-owned text after
PNG image data instead of missing a late recipe through Pillow's header-only
metadata view. A bounded keyword/header scan restores the source position
and adds no pixel decoding or second full archive read.

CellKey has one shared execution element before canvas/crop, keeping shape
at `[-2]` and seed last. An absent recipe is explicitly removed from hashed
run definitions before legacy shape/extension compaction; trailing-None
trimming alone would rename mixed-size and noncenter runs. Six pre-change
layout hashes are pinned, including comparison grids. Empty execution
remains SQL NULL and absent from PNG chunks. Existing eager identities and
metadata bytes remain unchanged.

Re-render, Swap tags, Mass refine, More seeds, retained Finish, PNG settings
and CLI replay refuse unavailable recipes rather than render eagerly.
Browser Reuse is held with its reason before composer/Undo changes or
source lookup; a held lightbox press keeps the dialog open. No Models,
environment or CLI selection, compiler installation or runtime compilation
is included.

Checked on Windows/Python 3.13.15: 605 focused execution, queue, grid, API,
inpaint/outpaint and CLI tests pass. The six affected production modules
pass the pinned Pyright basic check; metadata/import checks load no torch,
diffusers or Hub module in a fresh subprocess. No model or GPU acceptance
follows from these results. The carry stage additionally passed 537 focused
foundation/regression checks and 30 replay checks. Edge passed the new
synthetic-record `smoke-execution-reuse` and existing `smoke-png-settings`;
the former is a browser fixture over an eager fake image, never a compiled
render. Runtime and separately approved card stages must land before a
nonempty recipe can become available.

Final integration evidence: the complete offline CPU suite passed **4,405
tests, eight platform skips**, in **211.01 seconds** with two workers on
Windows/Python 3.13.15. The pinned checker analyzed **108 Python sources
with zero errors/warnings**, and repository ruff passed. Independent review
fixed retained-destination SQL/PNG disagreement and a held Reuse press
removing an existing Undo; the expanded fixtures cover both. No Linux,
hosted CI, model load, compiler execution or GPU qualification is claimed.

## Inactive registration and policy helpers

`backend/backends/regional_compile.py` now validates the exact defining
FLUX model and both repeated-block classes, complete declarations, callable
APIs and fresh root/region compile state before mutation. It registers
`backend="inductor", fullgraph=True, dynamic=False, mode="default"` inside
the installed context-local strict policy. Setup failures, including
cancellation and policy exits, discard the unpublished candidate exactly
once while preserving the original exception. A handle records region
counts and drops only its own references on close; it cannot unwrap a
partially registered model or reset another caller's compiler state.

All **32 helper tests** passed, with Ruff and narrow pinned Pyright clean.
Tests use strict defining-class/registration stubs and the installed
`ConfigModule.patch` on a synthetic two-flag module to check nested/thread
restoration. A reviewer caught the precompiled-root case; its regression
proves no policy entry or model/block registration occurs before refusal.
The helper cases use no real Flux or Dynamo import graph, compiler, tensor,
model or cache operation and reject new device calls. The normal test
conftest still ran its established Windows CUDA-availability guard; that is
no model or performance qualification. Real lazy imports,
fallback/recompilation behavior, toolchain/cache setup, loading/call wiring
and card qualification remain. There are no production callers and every
nonempty execution recipe remains refused.

## Backend publication and teardown ownership

The real eager `DiffusersBackend.load` now uses an unpublished candidate for
returned component overrides and the complete pipeline. It owns those references
through shard-context entry/exit, placement, VAE options, callback validation,
progress/diagnostics and weights-context exit. Publication happens last, transferring
parameters, effective recipe and an optional owned regional registration together.
No loader calls the regional registration helper and no public selection is added.

A failed construction can already have loaded overrides, so their argument mapping
is cleared even if shard-context entry or `from_pretrained` fails. Failed-load
frames retain the candidate rather than a separate pipeline/VAE/bound-method alias.
Candidate cleanup drops its application references before the existing allocator
release outside the backend and weights locks. Cleanup failures attach diagnostics
without replacing the primary exception. Retained arbitrary library exception
frames may still own their arguments; this is not a guarantee of physical GPU
reclamation or permission to strip useful tracebacks.

Both unload forms use one detach path for base/sibling pipelines, adapters,
parameters, recipe and registration. Compatibility caches and console preference
remain; a contended `try_unload` mutates nothing. A privately supplied nonempty or
malformed loaded recipe cannot make an ordinary eager request reach device, peak
or pipeline work, including direct private generation. Ordinary request admission
and every nonempty-render refusal remain unchanged.

All **23 isolated synthetic checks** passed with `--noconftest`; the combined
ownership/runtime-settings/execution/profile/upscaler group passed **195 tests**. The changed
production source passed pinned Pyright and Ruff. Independent review fixed retained
load aliases, ownership of overrides before pipeline construction, context-entry
cleanup and a test assertion which cleanup-error preservation could otherwise
swallow. Weak-reference results are now asserted outside the exception context.
Tests use synthetic loaders/runtime modules and release hooks, never model weights,
real compilation or device execution. A fresh subprocess also forbids framework
imports while checking loaded-recipe refusals. Full regional setup, actual execution
under strict compiler policy, selection/loader controls and card qualification
remain separate stages.

The complete offline CPU gate then passed **4,920 tests, eight platform skips
and 75 existing warnings** in **225.12 seconds** with two workers. Pinned Pyright
checked **120 production/tool sources with zero errors/warnings**, and repository
Ruff passed. The first full run found an old upscaler fixture constructing the
backend with `__new__`, bypassing its initialized execution fields. It now uses
the ordinary cold constructor; its unsupported-upscaler and no-normalization
assertions are unchanged. No production guard was weakened.

## Private loader and invocation wiring

The public no-argument `load()` now delegates to `_load_prepared(None)`, so the
ordinary eager path uses the same transaction as private regional preparation.
An explicit private regional specification is canonically revalidated and checked
against the cold dependency/model/placement matrix before weights or framework
imports. A fixed nonnative external-attention startup warning also refuses that
private experiment without echoing or changing the environment. This does not
settle later environment mutation/effective-import precedence for public selection.

After normal placement, VAE options and mandatory callback/progress checks, the
candidate registers its transformer and validates the returned owned handle and
fixed discovery facts. Failed setup invalidates the candidate immediately;
allocator release remains with the outer transaction after backend/weights locks
exit. The registration and recipe transfer together only at final publication.

Both actual direct and refine pipeline calls now use `_invoke_pipeline`. Eager
forwards once without entering compiler policy; its existing `no_grad`, schedule,
refine generator and hook-cleanup scopes stay outside the wrapper. A private
regional call requires its exact live base root/registration/fixed facts, one
image and no adapter, and enters strict policy around one forward. Success,
compiler failure and cooperative cancellation restore policy without eager retry.
Every public nonempty request, applied and direct-generation refusal is unchanged.
This wiring adds no selector, dependency install, warm-up or permission to render.

All **29 new isolated tests** passed with `--noconftest`. The focused ownership,
execution/runtime-settings, regional, schedule, refine/inpaint/upscaler and module
map group passed **394 checks**. Tests use synthetic pipelines, blocks and runtime
modules; a fresh subprocess forbids framework imports during cold refusal. Both
changed production sources pass pinned types with zero errors/warnings, Ruff is
clean and independent review is clear. These checks establish wiring and failure
ownership, not real registration, kernels, memory, pixels or fallback-free card
execution. Public admission, optional compiler/cache setup and the card protocol
remain separate stages.

The complete offline suite then passed **5,060 tests, eight platform skips and
75 existing warnings** in **223.40 seconds** with two workers, including the
new Windows mutation mechanism. Pinned Pyright analyzed **123 production/tool
sources with zero errors/warnings**, and repository Ruff passed. All owned
validation processes ended successfully. No real model, compiler or GPU
generation was used for this private-wiring stage.

## Shared Native attention invariant

The later shared Native invariant fix adds read-only effective-dispatch and
complete processor checks before regional registration/publication and before/
after one strict forward. A startup warning alone cannot exclude import-frozen
or later nonnative overrides. The common guard lives in `attention_selection`,
with FBC delegation; it neither resets defaults nor chooses public override
precedence. Changed successful results fail, while primary errors/cancellations
retain identity plus diagnostic notes. Eager and all public guards are unchanged.
The fix passed **755 focused checks** and **154 isolated synthetic checks**,
three-source pinned types with zero diagnostics, Ruff and independent review.
All jobs ended. Broader fixtures retain their ordinary availability guard; actual
dispatcher/model/device/kernel behavior remains unqualified.

## Installed CPU compiler-policy probe

One isolated worker now exercises the actual installed Dynamo configuration
and `strict_compiler_policy` with tiny float32 CPU arithmetic modules and a
custom Python graph backend. Its diagnostic uses `fullgraph=False`, static
shapes and isolated recompilation state: fullgraph has a separate hard-error
path which would not isolate the policy flag. Under a subprocess-local limit
of one graph, the same shape runs twice through that graph, a new shape raises
`FailOnRecompileLimitHit` without another backend execution, and an intentionally
failing backend propagates `BackendCompilerFailed` with its original cause.
Nested contexts and both error paths restore the caller's actual flags.

The worker runs with one CPU thread, owned temporary/cache/debug paths and
inherited Dynamo/compiler repro controls removed. CUDA availability is
explicitly mocked false to model CPU-only CI. A native `_cuda_init` guard,
as well as Python aliases, must remain untouched; the final receipt records
no initialization attempts, an intact guard and uninitialized CUDA. No
checkpoint, Diffusers model, Inductor backend or toolchain runs.

The first trial's incomplete Python-only guard did reach native CUDA
initialization while Dynamo preserved RNG state, despite CPU inputs. That
child exited with an error before its graph backend; no GPU tensor or
generation was requested. Its stderr is retained under
`work/pytest-regional-cpu-policy-20261026/test_real_strict_policy_propag0/worker.stderr`.
The corrected final guarded run passed its one subprocess test in 3.84 seconds,
with Ruff clean and independent review accepted. Its receipt is under
`work/pytest-regional-cpu-policy-20261026-3/test_real_strict_policy_propag0/`.
This proves the bounded CPU failure policy, not regional kernels, native CUDA
behavior, model compatibility, fallback-free card execution or performance.
Runtime admission remains closed.

## Initial slice and refusal matrix

Propose an explicit, experimental loader choice: **Eager** (the unchanged
default) or **Regional v1**. Start with the exact catalogue id
`black-forest-labs/FLUX.1-Krea-dev`, ordinary torchao fp8, CUDA and model
offload. Qualify text-to-image at one image per call before adding another
checkpoint, quantizer, batch size or pass. The cold Windows fp8 load still
needs `GEN#22`'s separately approved RAM check before a compiled card run.

| Selection or operation | Regional v1 first implementation |
|---|---|
| Exact Krea dev id, fp8, model offload, CUDA, text-to-image, Batch 1 | Candidate; available only after dependency preflight |
| Eager | Existing behavior and capabilities |
| Sequential offload, no offload or CPU | Refuse before unload or download |
| Nunchaku, BF16, INT8, another checkpoint or an unknown local model | Refuse explicitly; do not infer support from a class name |
| LoRA, including weight zero | Refuse this combination before submission; keep the requested adapter intact |
| Refine, inpaint, outpaint, Then refine or Mass refine | Refuse while Regional v1 is selected; offer selecting Eager |
| Re-render, Swap tags, More seeds, comparisons, grids | Permit only cells meeting the same matrix; preserve their recipe |
| Missing/incompatible Triton or unsupported runtime | Refuse the selection with the exact missing requirement; keep working weights |

These restrictions bound the first implementation; they do not change
Eager's current LoRA or edit support. Cold capability descriptions, direct
API requests and the CLI must agree. Do not normalize an explicitly
requested unsupported recipe or pass into another operation.

## Source inspection and dependencies

Read from this Windows environment on 2026-10-26: Python 3.13.15,
torch 2.13.0+cu130, diffusers 0.40.0, torchao 0.18.0, accelerate 1.14.0 and
peft 0.20.0. Distribution metadata reports neither `triton` nor
`triton-windows` installed. These checks read metadata and source files;
they establish no compiler or device compatibility.

- Installed `diffusers/models/modeling_utils.py`,
  `ModelMixin.compile_repeated_blocks`, discovers modules by the names in
  `_repeated_blocks` and calls each module's in-place `compile` with the
  supplied arguments. Empty declarations or no matching module raise.
  `models/transformers/transformer_flux.py` names both
  `FluxTransformerBlock` and `FluxSingleTransformerBlock`. The library
  therefore provides the intended regional boundary; no local block-class
  patch or full-pipeline compilation is needed.
- Installed `torch/__init__.py`, `compile`, documents guard-based
  recompilation and eventual eager fallback at the per-code-object limit.
  `torch/_dynamo/config.py` has `fail_on_recompile_limit_hit` and
  `suppress_errors`; first implementation must test a scoped failure policy
  rather than label an unnoticed eager fallback compiled.
- Diffusers documents regional compilation through
  [`compile_repeated_blocks`](https://huggingface.co/docs/diffusers/optimization/fp16#regional-compilation).
  Its [quantization/offload example](https://huggingface.co/docs/diffusers/optimization/speed-memory-optims#quantization-torchcompile-and-offloading)
  installs model-offload hooks before compiling the transformer. This is
  API guidance, not evidence for this app's torchao recipe or RTX 5080.

Current primary-source wheel check, without downloading: the
[Triton Windows maintainer's matrix](https://github.com/triton-lang/triton-windows#3-pytorch)
maps PyTorch 2.13 to Triton 3.7 and lists consumer Blackwell sm120 as
supported. [PyPI's 3.7.1.post27 files](https://pypi.org/project/triton-windows/3.7.1.post27/#files)
include `triton_windows-3.7.1.post27-cp313-cp313-win_amd64.whl`, released
2026-06-21, SHA-256
`e8ed215c02afc85a81f0097196f8bebdf6f68085f1bc0fe8771b9a74783f15ab`.
That is a candidate pin for an optional Windows compiler extra, not proof
that this torch/torchao/CUDA pairing compiles. Do not use the newest 3.8
wheel merely because it is newer.

Before implementation exposes the option, settle the optional dependency
and lockfile treatment; check the candidate wheel's actual toolchain needs
and license notices. Before the card pass, install it only with the
appropriate authorization and run a tiny compiler smoke in that same
environment. Windows and WSL are separate qualifications. For WSL, record
the environment's actual torch/Triton versions and C compiler, keep the
compiler cache on ext4, and account for `wsl/env.sh`'s
`TORCH_DISABLE_NATIVE_JIT=1`. That eager-kernel workaround does not establish
that Inductor works without a C compiler. No WSL inspection was run here.

## Loader, first call and teardown

Keep a small compilation helper separate from the model-specific loaders.
Its cold validator reads the explicit selection and installed dependency
metadata without a pipeline load, model download or trial CUDA operation.
It is called by browser-switch validation and the CLI before weights move.
Recheck the loaded transformer type and repeated-block declarations after
the normal `from_pretrained`; fail through the normal load-failure path if
the component contradicts the advertised capability.

Keep loading exclusively in `main.ensure_loaded()` after
`jobs.wait_for_gate`, under the existing asyncio and backend locks.
`loading.checkpoint_files_released` still encloses the quantizing load and
`loading.place_pipeline` still selects the device. After that placement,
before publishing `_pipe`, configure the repeated blocks in place with an
explicit recipe, initially `backend="inductor"`, `fullgraph=True`,
`dynamic=False`, default optimization mode. Leave text encoders, VAE,
scheduler, prompt expansion and all surrounding Python eager. Preserve
`torch.no_grad()` and the current scheduler/sigma behavior.

The helper registers compilation at load; actual kernel compilation is
lazy on the first image call. Report this distinction in Models and job
progress. A successful Load is not a warm image. The single job worker and
its gate own the first call; do not run a synthetic warm-up generation at
startup or compile beside a critic/Expand. An immediate setup failure
goes through `main._load_failure`; a lazy compiler failure fails the cell
through the existing job transition, saves no image and explains how to
select Eager. Do not silently retry or save it as another recipe.

Cancel/Skip stay cooperative: check before entering the pipeline and at
every existing `callback_on_step_end`. The compiler may delay the first
callback; document that latency, and check cancellation again before
saving. Do not promise instant compiler interruption or kill the shared
server process. Pause remains a cell-boundary operation.

Changing compilation mode is a real loader change. Reserve idle state,
save the validated preference, unload and increment selection revision
using the existing switch lifecycle. Both `unload` and `try_unload` drop
the compiled pipeline, sibling references, adapter state and per-load
compile diagnostics. Re-selecting Eager constructs clean modules; do not
try to unwrap a partially compiled pipeline in place.

## Shapes, cache and LoRA lifecycle

Use static compilation in the first recipe so behavior is explicit. A new
resolution can compile again; Batch >1 is initially refused. Instrument
the qualification harness with compiler guard/recompile diagnostics and
compare an unchanged image, a second resolution and a return to the first.
Do not increase Dynamo's global cache limit to hide churn. Use a scoped
`fail_on_recompile_limit_hit=True` with error suppression off while the
worker runs compiled code, restoring prior values afterward; test this
against the installed version before relying on it. If that cannot safely
prevent a compiled record from falling back to eager, keep the feature
unavailable until an observed, honest effective-mode policy is implemented.

Keep runtime/compiler diagnostics separate from reproduction data. Proposed
disk-cache root: `local-diffusion.compile-cache` beside the database,
with a platform/runtime/recipe namespace and an explicit environment
override. Set compiler cache configuration before compiler imports and
never redirect it while the worker is live. Cache write failure refuses
Regional v1 with a repairable reason; do not change the gallery or remove
another cache. Retain generated cache across ordinary unload/restart, with
no cache export in gallery bundles or automatic deletion in this slice.
PyTorch documents version/GPU compatibility checks for its
[compiler caches](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html);
cache reuse and restart latency still require measurement here.

Current `_apply_lora` injects lazily in `generate`, reweights with
`set_adapters`, and unloads on an adapter change or removal. That changes
the module structure after compilation. Diffusers'
[compiled hotswap guidance](https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference#compiled-models)
requires enabling hotswap before the first adapter, then loading that
adapter before compiling; rank and target-layer coverage matter, and
text-encoder LoRAs are unsupported by that hotswap route. This is why the
first slice refuses LoRAs instead of guessing a maximum rank or promising
recompile-free changes. A later LoRA slice must design one persistent
adapter slot, metadata-only rank/target preflight, weight zero/removal,
failed-injection cleanup and sibling ownership, then measure them.

`refine.build` constructs siblings over the same component objects and
keeps their existing offload hooks. Compiled blocks would therefore be
shared too; building a sibling is not a separate eager escape. A later
pass slice must qualify each sibling's shapes and shared hook cleanup,
including base/refine alternation, before removing the initial refusals.

## Selection and reproduction carry

Use one nullable recipe id, proposed `compilation="regional-v1"`; absent
means Eager and preserves all existing ids. An immutable runtime selection
owns it alongside checkpoint/quantization/offload. Add it to `Settings`,
environment/CLI configuration, `model_selection` equality and validation,
Models request/status/catalogue, saved startup preferences and their
explicit-override rules. Read old preference schema as Eager and write the
new schema atomically. Never mutate frozen startup settings or choose an
incompatible mode when a saved model changes.

Assume different arithmetic until the card proves otherwise: carry the
recipe through the request's single `applied()` normalization, image row,
PNG chunk, `CellKey`, `run_id`, duplicate/Resume/planned-key lookup and
retained job recovery. Add a nullable migration and unguarded row read;
keep legacy keys/hashes unchanged when the recipe is absent. Include
`grid.cell_from_record`, CLI save/`--from-image`, PNG settings, Reuse and
Undo, Re-render, Swap tags, More seeds, reindex and verified bundle
export/import. A recorded recipe incompatible with the current loader is
an explicit refusal or a clearly requested conversion, never silently lost.
Make any later dynamic-shape or optimization-mode change a new recipe.

Coordinate this carry with [attention](plan-attention-backends.md) and
[step caching](plan-step-caching.md): use one canonical optional execution
extension with named recipe fields, widened once. The empty extension
preserves legacy identities; independent features must not claim the same
positional key slot.

For that later carry, inserting the shared slot before shape preserves the
current penultimate shape position, but is not sufficient to preserve
hashed definitions. Explicitly omit absent execution from canonical run
serialization even when nonempty shape follows it. Pin legacy uniform,
mixed-size, noncenter and comparison hashes before widening; test them
all. Metadata/record validation must preserve a known specification while
normalizing older fields without granting runtime permission. Do not add
a blanket availability bypass to load a record or rebuild a stamped pass.

For a new composer submission, `applied()` stamps the selected effective
recipe after ordinary validation; a replay carries its recorded recipe
and is checked for equality. Request construction must distinguish these
intents so an old Eager image cannot acquire a compiled identity merely
because the current server changed. The same admission check belongs to
browser/API and CLI paths. The selection revision guard continues to
reject stale tabs before submission and validation races during switches.
Record compiler/runtime versions with qualification evidence; do not put
first-call latency, RAM, VRAM peaks or cache hits in the recipe or cell key.

## Implementation and acceptance stages

1. **CPU contracts and carry.** Implement cold matrix/preflight and the
   nullable recipe end to end. Fake-backed tests cover every refusal before
   unload/download, old preferences/rows/PNGs, new migration, unchanged
   Eager hashes, distinct compiled keys, Resume/queued deduplication,
   reruns/PNG/CLI/reindex/bundle round trips and stale switch revisions.
   Follow the existing CLI-vs-job cell equality checks.
2. **Backend and browser wiring.** Stub repeated blocks to prove exactly
   the intended compile arguments, order after quantization and placement,
   setup/lazy failure cleanup, cancellation and teardown. Exercise a small
   CPU module with a non-Inductor test backend only where needed to prove
   scoped recompilation/failure semantics; this proves no CUDA speed.
   A scratch-fake browser smoke checks Models save/load, held reasons,
   unsupported LoRA/pass selection, old/new record reuse and another tab's
   revision. Run focused tests, docs maps and ruff; retain the ordinary CPU
   gate. Missing optional compilers must not weaken Eager tests.
3. **Separately approved card qualification.** Complete `GEN#22` first,
   then run one shape at a time with no concurrent project/GPU workload.
   Use identical checkpoint/quantizer/steps/CFG/mu/prompt/seed and native
   SDPA; change only compilation. Never pass the existing RAM/VRAM guard or
   accept shared-memory spill as a fit result. Populate the table below in
   `verified.md` with versions, toolchain, driver, cache state and date.
   Keep `PERF#1` OPEN for implementation, then CARD while these runs remain.

| Proposed run | Separate evidence required |
|---|---|
| Eager baseline, first and repeated 1 MP image | Load and image time; process working-set/private-commit peak; allocator allocated/reserved peak; whole-card/shared memory |
| Regional v1, empty task-owned compiler cache, same image | Setup time, first-image compile/wall time and guard/recompile log; RAM/commit and card peaks |
| Same regional image, at least three repeats | Per-image and per-step warm time; no unexpected recompiles; pixels repeat exactly across runs |
| Second 1 MP aspect, then return to first | New-shape first time, repeated time, cache reuse, recompiles and all peaks separately |
| New process, retained compiler cache | Load, first-image and warm times; cache validation/reuse rather than an assumed cache hit |
| Eager versus compiled at the same seed | Hash, mean/max channel difference and differing-pixel count; retain both PNGs and complete recipe metadata |
| Cancel during first call, subsequent job, unload/reload | Cooperative latency, no cancelled save, hooks/resources released and next result's recipe honest |

An observed warm speedup alone is insufficient if cold startup, compiler
RAM/commit or spill makes the workstation worse. Do not advertise a timing
or broaden the capability matrix before those results are recorded.

## Decisions left explicit

These are questions before exposing or promoting the opt-in, not a requirement
for another conversation approval before ordinary source implementation and
fake-based checks. Loader/selection/status lifecycle work may continue while
every nonempty-render admission boundary remains closed. Package/toolchain
installation and actual model/card execution remain separate actions.

The implementation proposal above keeps defaults Eager and excludes LoRAs
and sibling passes initially. Before shipping the opt-in, decide whether
Matt finds that restricted experiment useful and whether its observed
cold/warm trade-off justifies a Models choice. Also settle the optional
compiler dependency/pin, cache location/size policy and whether a later
recipe should use dynamic shapes. Supporting another quantizer/family,
LoRA hotswap or any edit pass is a bounded follow-up with its own evidence,
not implied acceptance from the first checkpoint's result.
