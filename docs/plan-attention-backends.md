# Plan: native and cuDNN attention on FLUX.1

**Status, 2026-10-26: shared carry, private processor/loader/call ownership and
cold external-override diagnostics, request candidate checks and unavailable
selector/persistence/status/UI source are implemented.** Effective runtime
environment policy, real nonempty admission and separately approved card
qualification remain `PERF#3`. This is the
bounded next slice from [R3](review-2026-10-25.md#r3-attention-backend).

The shared [unavailable configuration stage](plan-regional-compilation.md#unavailable-performance-configuration)
adds inspectable Native/cuDNN intent without changing global attention state.
Compatible intent can be saved, but Load/render always refuse it; omission,
explicit Eager replacement, revision, startup and rollback semantics are shared
with compilation. This does not claim effective imported environment precedence
or qualify any native/cuDNN runtime. Browser validation has the actual Windows
access limitation recorded in [tooling](plan-tooling.md#browser-smokes-in-ci).
No package installation, model load, tensor allocation or GPU operation
was used to prepare this plan. No speed default changes.

The shared `ExecutionSpec`, nullable SQL/PNG carry and refusal paths are
implemented with `PERF#1`. A recorded `cudnn-v1` remains unavailable to
render or Reuse; this establishes metadata transport, not runtime processor
or card qualification.

## Inactive processor ownership and restoration

`backend/backends/attention_selection.py` leaves None import-cold and
unchanged. Explicit `cudnn-v1` configuration lazily checks exact defining
FLUX model/block/attention/processor classes, declared block counts and
every joint/single processor path before mutation. It snapshots identities
and prior per-instance/class-backed values, deduplicating shared processors.
The public native backend context encloses the setter and verification,
restoring its incidental global registry change on success and exceptions.
Failed setup rolls every original processor back and preserves its cause;
the unpublished candidate must be discarded if cleanup cannot complete.

One handle owns the transformer and its processors for all shared siblings.
Another transformer cannot claim or graft owned processors. Verification
detects changed blocks, attention modules, processors or backend values.
Restoration attempts every original processor, preserves unrelated replacements,
is idempotent after success and retains ownership after a restoration failure
for repair/disposal. Successful release drops module/snapshot/traceback
references. The future caller must retain the handle and obey idle/load
ownership; no loader, request, catalogue or UI path calls it now.

All **43 focused tests** passed with `--noconftest`; Ruff and narrow pinned
Pyright pass. Tests use strict stubs and source-extracted installed public
context/setter functions. They make no real Diffusers import, tensor,
forward, model or device call. Import audit found `torch_utils` asks for a
device during real dispatcher/transformer imports, so that binding path is
explicitly unexecuted here. Independent review fixed ownership across
replacement modules and shared processors. Real imports, dispatcher/shape
support, repeatability, performance and fit remain unqualified. Environment
override policy and activation decisions below are still unresolved; every
nonempty execution recipe remains refused.

Combined Windows integration passed **4,542 CPU tests, eight platform
skips**, in **204.93 seconds** with two workers. This includes the new
regional helper and managed-backup proof/journal stages. The pinned checker
reports **111 sources, zero errors/warnings**; repository Ruff passed.
No model load, render, Linux/hosted run or card qualification follows.

## Private loader, call and terminal-discard ownership

`DiffusersBackend._load_prepared` now accepts a private attention-only candidate
after cold matrix and fixed external-warning checks, before weights. Public
`load()` still supplies None and every public nonempty admission remains closed.
After ordinary placement/checks, configuration belongs to the unpublished pipeline
transaction; the exact root/processor owner transfers with its recipe only at
final publication. Both unload paths detach it with base/sibling references and
restore/dispose outside backend/weights locks. Eager and regional paths retain
their previous behavior and cannot carry an orphaned attention override.

The actual invocation boundary verifies the exact retained root, owner registry
and complete layout before and after one private base forward. A successful
forward with changed ownership is refused; a failed/cancelled forward retains
the exact original exception and attaches any verification failure as a note.
There is no native retry, global dispatcher mutation or save of an uncertain
result. The first private matrix still excludes adapters, siblings and recipe
combinations. No selector, environment mutation, dependency or real device call
was added.

Terminal discard and failed setup now retain weak identity exclusions for the
root and original processors, even after snapshot rollback. Otherwise dropping
the last failed owner could allow a partially changed layout to be adopted as a
new baseline. Weak references are prepared before mutation, callbacks check their
own identity before removal, and no weights or error tracebacks are retained.
A fresh root/processors is required after terminal discard. A retained owner's
ordinary restoration-repair semantics and None's cold no-op remain unchanged.

The focused lifecycle/execution/refine regression group passed **644 checks**;
the final isolated synthetic helper/regional/attention group passed **113 checks**
with `--noconftest`, including the combined forward-failure/cancellation cases.
Both production sources pass pinned Pyright with zero errors/warnings, Ruff
passes and independent review is clear. All author jobs ended successfully.
The isolated checks use no real dispatcher/framework/model/device work; the
broader group uses the existing ordinary fixture availability guard. No kernel,
memory, pixels, repeatability or GPU qualification is claimed. The final combined
whole-source gate follows the concurrent managed-retention implementation.

## Shared Native attention invariant

The read-only effective-Native guard is now shared by private regional and FBC
preparation/calls. It checks the source-verified dispatcher getter and complete
processor overrides without mutation; a cold startup warning alone is not an
effective-runtime proof. Regional pre/post-forward changes preserve primary
errors/cancellation with notes and never retry. This invariant fix passed **755
focused checks**, **154 isolated synthetic checks**, three-source pinned types
and Ruff, with independent review clear. Actual imported dispatcher/device behavior
and public Eager/environment precedence remain separate qualification/policy work.

## Cold external-override diagnostics

`config.ATTENTION_ENVIRONMENT_WARNING` snapshots the installed Diffusers
verbatim `os.getenv("DIFFUSERS_ATTN_BACKEND", "native")` contract before
framework imports. Missing or exact `native` values are silent; empty,
case-changed, whitespace and other values receive one fixed warning through
the common config logger. The typed optional
`ModelStatus.attention_environment_warning` carries the same snapshot.
Neither channel echoes the arbitrary environment value. Server and generation
CLI selection flags remain unchanged; no frontend edit is included.

The warning observes an existing external setting. It does not clear or
rewrite it, choose override precedence/accepted backends, infer an effective
execution recipe, change admission or qualify the requested backend. Status
does not reread the environment after framework initialization.

All 90 focused cold-diagnostic, status, launch, saved-preference and runtime
settings checks pass. New subprocess checks forbid Torch/Diffusers imports,
compare the installed source without importing it, cover empty/absent/exact
native and hostile long values, and prove unchanged environment/selection
and bounded fixed output. Ruff passes and narrow pinned Pyright reports zero
errors/warnings across the three affected production modules. Runtime and
card gates remain unchanged.
Independent source review accepted the snapshot, safe output, CLI ordering
and status wiring without further findings.

The final combined offline CPU gate passed **4,732 tests, eight skips and 75
existing warnings** in **221.50 seconds** with two workers. The pinned whole-source
gate checked **116 files with zero errors/warnings**, and repository Ruff passed.
No checkpoint load, generation, Linux/hosted qualification or card acceptance
follows. The earlier compiler-fixture native CUDA initialization and corrected
guarded trial remain recorded in the [compilation owner](plan-regional-compilation.md#installed-cpu-compiler-policy-probe).

## First comparison and capability matrix

`execution.attention_candidate_reason` now checks this initial request and
distribution matrix from supplied facts only. Every nonnative external override
is conservatively refused with a fixed reason; this selects no precedence policy
or changes to the environment. Exact loaded processor/API coverage remains the
inactive helper's separate concern. A matching candidate remains unavailable at
every existing request/runtime admission boundary.

The shared cuDNN/FBC slice passed 122 focused tests in 0.44 seconds with
`--noconftest`, including an isolated cold-import/environment guard. Narrow pinned
Pyright reports zero errors/warnings for the changed production source, Ruff
passes, and independent source review accepted the checks. No framework, model,
device, compiler or kernel operation was used by these candidate checks.

Start with the exact catalogue id `black-forest-labs/FLUX.1-Krea-dev`,
torchao fp8, CUDA, model offload and one-image text-to-image calls. Compare
**Native SDPA (automatic)** with the experimental **Forced cuDNN** recipe.
Native is PyTorch's dispatcher, not a promise to use its math kernel;
it may already select cuDNN. Forcing another backend may be slower.

| Selection | Initial proposal |
|---|---|
| Existing native/default selection | Keep current behavior and legacy identities |
| Exact Krea dev, fp8, CUDA, model offload, Batch 1, direct generation | Candidate for `cudnn-v1` after cold preflight |
| Sequential/no offload, BF16, INT8 or another checkpoint | Later qualification; refuse an explicit cuDNN request initially |
| CPU, Nunchaku or a processor not using the dispatcher | Refuse before download/unload; never silently keep native |
| LoRA or a sibling edit pass with cuDNN | First slice refuses; retain the requested adapter/pass and explain selecting Native |
| Compilation or step caching with cuDNN | Outside this first comparison; each combination needs separate evidence |

The first Windows fp8 load still needs `GEN#22`'s RAM check before a cuDNN
card run. Sequential offload does not inherently remove an attention
choice: the setting belongs to the processor, and attention still runs
when that module is resident. Its transfer costs and peaks need a separate
measurement before widening this initial matrix. Windows and WSL are
distinct qualifications.

## What the installed source establishes

Inspected Python source from diffusers 0.40.0 and torch 2.13.0+cu130:

- `diffusers/models/transformers/transformer_flux.py`, `FluxAttnProcessor`,
  passes its `_attention_backend` to `dispatch_attention_fn` after Q/K
  normalization and rotary embedding. The dispatcher covers this path;
  do not replace the whole processor or quantized projection modules.
- `models/attention_dispatch.py` registers `native` and `_native_cudnn`.
  Native calls PyTorch SDPA. The cuDNN path makes permuted Q/K/V contiguous
  and runs SDPA inside `sdpa_kernel(SDPBackend.CUDNN_ATTENTION)`. This is a
  forced kernel family, including a possible copy cost, not an extra wheel.
- The cuDNN registration checks matching devices, fp16/bf16 Q/K/V and
  compatible dimensions when dispatcher checks are enabled. Those checks
  are normally off. Weight-only fp8 does not imply fp8 Q/K/V: verify the
  loaded path's actual activation dtype during qualification.
- `models/modeling_utils.py`, `set_attention_backend`, checks the backend
  name/requirements and updates recognized processors, skipping ones
  without the field. It also changes the process-global active registry.
  `reset_attention_backend` clears processor overrides but does **not**
  restore that global registry. A successful setter is therefore neither
  proof that every intended processor changed nor a safe teardown policy.
- `torch/nn/attention/__init__.py` restores SDPA flags when its context
  exits. The installed SDPA documentation warns that fused arithmetic can
  change outputs; deterministic behavior is a separate requirement.

The [Diffusers attention guide](https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends)
describes this experimental dispatcher. The current
[PyTorch SDPA reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
explains automatic kernel choice, numerical differences and possible cuDNN
nondeterminism. Those online pages describe newer releases; the installed
source above is the implementation contract here. Neither source proves
RTX 5080 shape support, speed, repeatability or memory fit.

## Configuration, failure and shared components

Add one cold validator/helper with an explicit mapping `cudnn-v1` to the
installed `_native_cudnn` name. Validate the exact matrix and required
Python API before switching or downloading. Check loaded processor types
and coverage before publishing the pipeline. Do not run trial CUDA tensors
from catalogue/status calls. Tensor-shape/cuDNN support remains a runtime
check; an unsupported shape fails the cell with the original cause and a
Native recovery action, saving no image. Never retry with native while
retaining cuDNN metadata.

Keep `ensure_loaded`, gate/lock order, checkpoint-file release,
quantization and `loading.place_pipeline` unchanged. Configure the loaded
transformer after placement and before assigning `_pipe`, with explicit
processor coverage verification. Wrap the library setter in its public
`attention_backend("native")` context so its incidental registry change
is restored on success or failure; test this behavior against the installed
library. Do not mutate private registry fields or process-wide CUDA flags
as an unbounded side effect. Account for `DIFFUSERS_ATTN_BACKEND` at startup:
an unsupported external override must be reported rather than undermine a
recorded Native/cuDNN choice. Set any app-owned environment policy before
diffusers imports, with explicit override precedence documented.

Treat a backend change as an idle loader switch with saved preference,
selection revision and clean reload. Failed setup uses `_load_failure`;
unload drops base/sibling references. A cuDNN-to-native-to-cuDNN switch
must leave unrelated models and prompt helpers at their prior default.
Keep `torch.no_grad`, scheduling, generators and step callbacks unchanged.
Cancellation remains cooperative; it must still prevent a cancelled save.

`refine.build` shares component objects and their offload hooks. A future
edit slice must assert that each sibling sees the same processor recipe,
test base/edit alternation and qualify its distinct shapes/peaks. LoRA
injection/removal can alter module ownership: revalidate processor coverage
after each supported adapter operation, and fail before saving if it no
longer holds. Do not assume a new sibling or weight-zero adapter is an
independent native escape. These are explicit follow-ups to the initial
refusals, not evidence that the library cannot support them.

## Selection and reproduction carry

Propose nullable `attention_backend="cudnn-v1"`; absence means the existing
native/default behavior. Keep old `CellKey`, run hashes and resume results
unchanged for that absent value. Treat cuDNN as pixel-changing until proven
otherwise, and retain its recipe even if one comparison happens to match.

Carry selection through `Settings`, environment/CLI flags, immutable
`model_selection`, cold catalogue/status and Models requests,
`startup_preferences` with old-schema compatibility and atomic writes.
The selected recipe becomes part of effective request admission and the
single `applied()` path. A replay carries its recorded recipe and refuses
an incompatible current selection; an old native image cannot silently
become cuDNN because the server changed.

Add a nullable storage migration and unguarded row read; carry the field
through both insert paths, PNG metadata, retained jobs, `CellKey`/`run_id`,
completed/planned keys and duplicate/Resume checks. Include
`grid.cell_from_record`, Reuse/Undo, Re-render, Swap tags, More seeds,
comparisons, CLI/`--from-image`, PNG settings, reindex and verified bundle
round trips. Reserve and test a distinct optional key extension alongside
the planned compilation/cache fields; do not independently reuse the same
position. Runtime versions, timing and memory results belong to evidence,
not the recipe hash. Any changed arithmetic/determinism policy gets a new
recipe id.

Use the canonical optional execution extension described in the
[cache plan](plan-step-caching.md#per-render-choice-and-reproduction),
coordinated with [compilation](plan-regional-compilation.md#selection-and-reproduction-carry),
so the key is widened once with named recipe fields.

## Bounded implementation and acceptance

1. **CPU carry and admission.** Fake-based checks cover every matrix
   refusal before unload/download, missing APIs, processor-coverage gaps,
   old/new preferences/rows/PNGs, legacy hashes, distinct cuDNN identities,
   rerun/CLI/reindex/bundle round trips and stale revisions. Keep the
   deliberate CLI/job cell loops equal.
2. **Lifecycle and page.** Stub processors/setters to prove configuration
   order, restoration of the global default after exceptions, switch back
   to native, unload and failed-call cleanup. Test request recipe equality,
   cancellation and no silent fallback. A scratch-fake browser smoke checks
   Models save/load, held reasons and record reuse. Run focused CPU tests,
   docs maps and ruff; the browser remains an independent check.
3. **Separately approved card comparison.** After `GEN#22`, run one shape
   at a time with identical model, quantizer, prompt/negative, seed, steps,
   CFG/mu and no LoRA/compilation/cache. Use the existing guards and stop on
   shared-memory spill. Record driver/CUDA/cuDNN/torch/diffusers versions,
   dispatcher checks and determinism flags with retained PNGs.

| Proposed run | Evidence required separately |
|---|---|
| Native and cuDNN, first image then three repeats at one 1 MP shape | Load/first/warm image and step times; working-set/private-commit and allocated/reserved/whole-card/shared peaks |
| Same seeds across processes, each recipe | Exact hashes and repeatability; native-versus-cuDNN mean/max channel difference and differing-pixel count |
| Second aspect and return to first | Shape support, timings, peaks and no hidden fallback |
| Cancel, next image, unload/reload and native switch | Callback/cancel behavior, released hooks and honest subsequent metadata |

If cuDNN is not repeatable under the proposed policy, do not ship that
recipe as reproducible. Evaluate scoped determinism controls separately
and measure their cost; a seed alone cannot repair nondeterminism.

Before exposing the opt-in, settle whether the restricted experiment is
useful and how a nondefault `DIFFUSERS_ATTN_BACKEND` is represented or
refused. Later external Flash/Sage kernels need their own primary-source
Windows/Python/torch/CUDA/sm120 compatibility and license checks; no wheel
claim or installation is part of this slice. Keep Native as the default
until measured evidence justifies any separately chosen change. Leave
`PERF#3` OPEN for implementation, then CARD for the remaining runs.

These exposure questions do not block ordinary source implementation and
fake-based checks with render admission closed. The cold candidate helper's
conservative override refusal changes no environment/selection precedence.
