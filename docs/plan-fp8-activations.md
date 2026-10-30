# Plan: fp8 weights with fp8 activations on FLUX.1

**Status, 2026-10-30: `PERF#5` done (stage 1, this plan); stages 2-3 are
`PERF#2`'s.** The CPU planning slice was split from `PERF#2` on
2026-10-29, and Matt answered its three questions the same day
([below](#decisions-for-matt)). On 2026-10-30 every installed-source line
below was checked again against torchao 0.18.0, diffusers 0.40.0 and torch
2.13.0, and still holds; the repository lines were refreshed to the current
code. Matt approved every item of the 2026-10-29 TODO audit. This
plan records what the installed torchao 0.18.0 source says about
`Float8DynamicActivationFloat8WeightConfig`. It also sets the quantization
name, how that name is carried, and the refusal matrix that a later source
slice implements. Nothing was installed, imported on a GPU, loaded or
rendered to write it. The card measurement stays **`PERF#2`, BLOCKED** on
`PERF#1`'s card qualification ([regional compilation](plan-regional-compilation.md)).
The idea comes from [10-25 R2](review-2026-10-25.md#r2-fp8-with-fp8-activations).
There is no selector UI in this slice. The 10-28 library-recovery audit
found that "another unavailable selector would not complete an
independent current slice" (`docs/review-2026-10-28-library-recovery.md:103`),
so this plan adds none.

## Why it is worth a plan

Today's `fp8` choice is weight-only. `quantization_config` builds
`Float8WeightOnlyConfig()` for every quantizable component
(`backend/backends/loading.py:460`, `:466-473`). In the installed torchao,
a weight-only `Float8Tensor` meeting an unquantized activation dequantizes
the weight and runs an ordinary bf16 `torch.matmul`
(`.venv/Lib/site-packages/torchao/quantization/quantize_/workflows/float8/float8_tensor.py:461-469`,
the `else` branch of `_float8_addmm_impl`). So weight-only fp8 saves memory,
but its matmuls never use the card's fp8 tensor cores. That confirms R2's
premise from source.

`Float8DynamicActivationFloat8WeightConfig` stores the same `Float8Tensor`
weight class, with `act_quant_kwargs` set
(`.venv/Lib/site-packages/torchao/quantization/quant_api.py:1243-1259`).
At each call the activation is quantized first (`float8_tensor.py:347-355`),
and the matmul goes through `torch._scaled_mm`
(`.venv/Lib/site-packages/torchao/float8/inference.py:86-122`, reached from
`float8_tensor.py:449-457`).

Preference evidence: K4 "Change quantization" is rated **Never** and K2
"Compare model guidance, timings or examples" **Often**
(`docs/design/preferences.md:189`, `:191`). A new quantization choice is
therefore only worth offering if the card shows a clear speed gain at
acceptable quality. Even then it is a Models setting that Matt would rarely
change, so it should never become a default without his decision.

## What the installed source establishes

Package: `torchao-0.18.0` in `.venv/Lib/site-packages` (torch
`2.13.0+cu130`, CUDA `13.0` per `torch/version.py`; diffusers 0.40.0).

| Question | Source answer | Where |
|---|---|---|
| Defaults | e4m3 activations and weights; `granularity=None` becomes `(PerTensor(), PerTensor())`; `packing_format=PLAIN`; `kernel_preference=AUTO`; `set_inductor_config=True`; `version=2` | `quant_api.py:1163-1174`; `float8/inference.py:245-247` |
| Fast accumulation | `mm_config` defaults to `Float8MMConfig(use_fast_accum=True)`. It is off only for the 1x128/128x128 block granularity or on XPU | `quant_api.py:1185-1198` |
| Granularity choices | PerTensor, PerRow, or the 1x128/128x128 block pair. Activation and weight must be the same kind | `float8/inference.py:231-268` |
| PerRow weight dtype | PerRow with PLAIN packing asserts the input weight is bfloat16 | `quant_api.py:1237-1242` |
| Hardware gate, at transform | If CUDA is available, asserts compute capability ≥ 8.9 (or MI300/MI350) | `quant_api.py:1333-1336`; `torchao/utils.py:1140-1145` |
| Hardware gate, per tensor | PerTensor/PerRow assert XPU, or CUDA ≥ 8.9, or MI300/MI350. With no CUDA device this assertion fails, so a CPU-only process cannot quantize with this config | `float8/inference.py:271-312` (`_check_hardware_support`) |
| sm_120 | `is_sm_at_least_89` compares `get_device_capability() >= (8, 9)`, so the RTX 5080's (12, 0) passes torchao's gate. Whether torch's `_scaled_mm` kernel runs on sm_120 for each granularity is a torch/cuBLASLt question: torchao does not check it, and the meta registration (`torch/_meta_registrations.py:7138-7151`, through `_check_scaled_mm_sizes` at `:6942`) checks dtypes, shapes and scale sizes, never the compute capability | as cited |
| Kernel choice | `AUTO` picks `mslk` only if the `mslk` package is installed and the GPU is SM ≥ 9.0. `mslk` is not in the venv, so `AUTO` resolves to `"torch"` (`torch._scaled_mm`) for both activation quantization and matmul | `float8_tensor.py:184-207`, `:364-380`; `torchao/utils.py:1225-1232` |
| CUDA 12.9 warning | It warns only on CUDA 12.9 with PerRow. This build is CUDA 13.0, which the warning text names as fixed | `quant_api.py:1285-1322` |
| Compile need | There is no compile requirement in the source path. Eager quantizes each activation with ordinary ops (`_choose_scale_float8`, `_quantize_affine_float8`) and then calls `_scaled_mm`. R2's "typically only fast under compile" is a hypothesis for the card, not a source fact | `float8_tensor.py:233-242` |
| Global side effect | `set_inductor_config=True` calls `recommended_inductor_config_setter()`. That sets five process-wide `torch._inductor.config` values (`coordinate_descent_tuning`, `coordinate_descent_check_all_directions`, `force_fuse_int_mm_with_mul`, `fx_graph_cache`, `triton.unique_kernel_names`) and never restores them. Today's `Float8WeightOnlyConfig` has the same default and the same call, so every fp8 load already does this | `quantization/utils.py:675-694`; `quant_api.py:1078`, `:1107-1108`, `:1337-1338` |
| Silent skips | A 2-D/3-D weight whose last two dimensions are not both multiples of 16 is returned **unquantized**, with only an info log. The source's own TODO says it should raise. A 4-D/5-D (conv) weight with channels not divisible by 16 is also returned as is | `quant_api.py:1220-1235`; `quantization/utils.py:769-791` |
| Diffusers acceptance | Diffusers' `TorchAoConfig` requires an `AOBaseConfig` instance and torchao ≥ 0.15. Its quantizer names `Float8DynamicActivationFloat8WeightConfig` in `_TRAINABLE_QUANTIZATION_CONFIGS` and reports `is_compileable = True` for every torchao config | `diffusers/quantizers/quantization_config.py:584-604`; `diffusers/quantizers/torchao/torchao_quantizer.py:461-475` |
| Today's weight-only granularity | `Float8WeightOnlyConfig()` defaults to **PerRow** weights, unlike the dynamic config's PerTensor default. So a PerTensor `fp8-dynamic` changes weight scaling as well as activations against today's `fp8` | `quant_api.py:1076-1083` |

**Not established by source, left for the card (`PERF#2`):** that
`_scaled_mm` runs on sm_120 under CUDA 13.0 for the chosen granularity;
eager speed against weight-only `fp8`; quality against bf16 and `fp8`; the
load's RAM peak; and VRAM at rest and at peak.

## Fit with `PipelineQuantizationConfig(quant_mapping=...)`

The AGENTS.md rule stands: one `TorchAoConfig` per component, from the
library that owns it, built from an `AOBaseConfig` instance, and never
`quant_backend="torchao"` (`docs/design-notes.md#quantization-goes-through-quant_mapping`).
The new choice only changes which instance goes to which component:

| Component (FLUX.1 Krea dev) | Owning library | Config |
|---|---|---|
| `transformer` | diffusers | `Float8DynamicActivationFloat8WeightConfig(<frozen recipe>)` |
| `text_encoder` (CLIP), `text_encoder_2` (T5) | transformers | `Float8WeightOnlyConfig()`, unchanged from `fp8`. R2: "text encoders may stay weight-only" |
| `vae`, scheduler, tokenizers | — | Not quantized; `QUANTIZABLE_PREFIXES` is transformer/unet/text_encoder (`loading.py:37`) |

`quantization_config` today applies one `make()` to every component
(`loading.py:460`, `:473`). The source slice replaces that with a
per-component choice keyed by the quantization name. `fp8`'s and `int8`'s
mappings must not change, so their pixels and identities stay as they are.
The quantizing `from_pretrained` stays inside
`loading.checkpoint_files_released` (AGENTS.md, Model and loader). Its RAM
peak (`GEN#22`, `docs/design-notes.md#the-quantizing-loads-ram-peak`) has to
be measured again for this config on the card.

**Frozen recipe.** The quantization name names exactly one config. The
proposal is the library defaults with one change:

```python
Float8DynamicActivationFloat8WeightConfig(set_inductor_config=False)
```

That means PerTensor/PerTensor, e4m3, fast accumulation, `AUTO` kernel
(resolving to `torch`), PLAIN packing, version 2. Passing
`set_inductor_config=False` keeps this load from adding Inductor globals
that `PERF#1`'s scoped compile policy and compiler-cache binding would
then inherit. It does not change eager arithmetic. Leave `fp8`'s
setting as it is: changing it is `PERF#1`'s business, not this slice's.
PerRow or a different accumulation is a different recipe with a different
name, like an outpaint `FILLS` entry (AGENTS.md: "a changed fill gets a new
name"). Matt chose PerTensor as the first granularity
([below](#decisions-for-matt)), so `fp8-dynamic` is exactly the call above.

**Inventory check.** Because of the silent skip, the loader counts
transformer `Linear` weights after the load: how many are `Float8Tensor`
with activation quantization, how many weight-only, how many still bf16.
It logs the counts at load and reports them in `describe()` beside `quantization` (`backend/backends/diffusers_backend.py:1917-1923`); no such count exists today. Skips are not
refused in this slice, because the rule against silent loss covers weights
and scales and a skipped layer keeps its bf16 weight. They are shown,
never hidden, and the card stage records them.

## Name, metadata and run-id carry

Working name: **`fp8-dynamic`** (browser label "FP8 + fp8 activations").
Quantization is already a free string through every reproduction path,
so a new value needs no migration, no new column and no `CellKey`
extension:

- `run_id`'s definition hashes `"quant": quantization` (`backend/grid.py:1045-1047`).
  A new name gives new run ids, and every existing `fp8`/`int8` id stays
  unchanged.
- Rows keep the `quantization` column (`backend/storage.py:162`, `:237`,
  read at `:424`). The PNG's `auradream` chunk and its A1111 line carry it
  (`storage.py:582`, `:639`). Jobs record it in their definition
  (`backend/jobs.py:953`).
- Resume/completed-cell lookups are already scoped by exact quantization
  (`storage.py:1179-1180`). Re-render/Resume says "made under …; every cell
  would render again under …" (`backend/api_generate.py:302`), and More
  seeds notes the mismatch (`backend/api_more_seeds.py:116-117`). These
  paths need tests, not new code.
- The `applied()` normalisation and `CellKey` do not change: quantization is
  a loader choice, outside the request. Peak memory stays a result, never a
  key (AGENTS.md, Reproduction and storage).

Places that list quantization values and must learn the name:
`model_defaults.QUANT_MODES` (`backend/model_defaults.py:142`), the
`schemas_model.py:143` `Literal`, `loader_choices.QUANTIZATION_CHOICES`
(`backend/loader_choices.py:13-18`; the menu and picker wording),
`loading.QUANT_BACKENDS` and the mapping builder (`loading.py:28`,
`:437-476`), and the vocabulary check in `config.py:206` (through
`QUANT_MODES`). `startup_preferences.py:48` and `model_selection.py:179`,
`:228` validate through `QUANT_MODES`. `model_benchmarks.py:26`, `:171`
default to `fp8`: a new name has no retained timing, so the picker labels
any speed as a reference from another quantization and never claims one.

**Search consequence.** `quant:` is a substring key
(`backend/search.py:38`, `:232-244`), so `quant:fp8` would also match
`fp8-dynamic` once such images exist, and so would saved searches using it.
`quant:=fp8` stays exact. Matt kept the name with that widening
([below](#decisions-for-matt)), so the source slice documents `quant:=fp8`
as the way to find weight-only fp8 alone.

## Refusal matrix

Every refusal is checked before saving, reserving, unloading or
downloading, with the switch's normal error wording, and keeps the user's
choice in the picker, as unsupported execution recipes already are
("refused with their chosen values intact",
[unavailable performance configuration](plan-regional-compilation.md#unavailable-performance-configuration)).
The `AURADREAM_QUANT=nunchaku` preflight is the model (AGENTS.md: it
"refuses other pairings before any download").

| Selection | First slice |
|---|---|
| Exact `black-forest-labs/FLUX.1-Krea-dev`, `fp8-dynamic`, CUDA, model offload, text-to-image | Candidate. It stays unavailable to load until `PERF#2`'s card run passes |
| Another checkpoint | Refused: "fp8 activations are qualified only on FLUX.1 Krea dev" |
| `device` not CUDA | Refused cold. torchao's own gate would fail at quantize time (`float8/inference.py:271-312`) |
| GPU capability below 8.9 | Refused in the load, through `main._load_failure`, before weights move. Reading the capability initializes CUDA, so this cannot be a cold check |
| Sequential or no offload | Refused at first; each needs its own placement measurement |
| Nunchaku / WSL pairing | Refused: Nunchaku is pre-quantized (`loading.py:446-448`) |
| Regional compile, cuDNN attention or FBC with `fp8-dynamic` | Refused. `execution._configuration_placement_reason` already requires `fp8` exactly (`backend/execution.py:159-160`), and so does `_flux_candidate_reason` (`:300`), which the cuDNN and FBC candidate checks share. The source slice must reword that reason so it does not tell an `fp8-dynamic` user to "require fp8"; the combination stays its own later qualification |
| LoRA, refine/inpaint/outpaint siblings | Refused at first, keeping the requested adapter or pass and saying to choose `fp8`. `refine.build` shares components, so a sibling would run dynamic activations unqualified |
| Batch > 1 | Allowed by source (eager has no shape specialization). The card stage measures Batch 1 first, and a refusal is added only if the card shows a reason |

## Non-goals

- No selector UI, no unavailable Models row and no browser persistence in
  this slice. The picker learns the name only when `PERF#2`'s card result
  admits it.
- No change to `fp8`, `int8`, `none` or Nunchaku arithmetic, mappings,
  defaults or identities. No default change.
- No `mslk` install, no compile, no `torch.set_float32_matmul_precision`
  change, no other checkpoint or family.
- No claim of speed, quality or memory from this document.

## Decisions for Matt

**Answered 2026-10-29 (Matt took the proposals):** the name is
`fp8-dynamic`; the first granularity is PerTensor; `PERF#2` compares only
fp8 against `fp8-dynamic` on the `PERF#6` fixed cases, not bf16.

1. **Name.** Keep `fp8-dynamic`, which widens `quant:fp8` searches as
   described above, or pick a name without the `fp8` substring.
2. **First granularity.** The library default PerTensor (proposed), or
   PerRow (bf16 weights required, `quant_api.py:1237-1242`), which keeps
   the PerRow weight scaling today's `fp8` already uses
   (`quant_api.py:1082-1083`). Each is a separate recipe name.
3. **Card scope.** Whether `PERF#2` measures only the `PERF#6` fixed cases
   ([step caching](plan-step-caching.md#fixed-comparison-cases)) at fp8 vs
   `fp8-dynamic` (proposed), or also against bf16, as R2 proposed ("measures quality against bf16 at
   shared seeds"). The BF16 choice's own wording
   says FLUX-size models "may not fit 16 GB" (`backend/loader_choices.py:16`).

## Stages and completion criteria

1. **This plan (`PERF#5`). Done 2026-10-30.** Done when this file exists
   with source lines checked against the installed package. No code, no
   GPU. Matt's three answers are recorded above; the installed-source lines
   were rechecked and the repository lines refreshed on 2026-10-30.
2. **Source slice, CPU only (`PERF#2`'s CPU part, after `PERF#1`
   qualification).** Add the name to every list above. Build the
   per-component mapping and the inventory count. Implement the refusal
   matrix and reword the execution refusal. Checks: pytest proves
   `quantization_config` gives `fp8`/`int8` the same instances as before,
   and `fp8-dynamic` gives the transformer the frozen dynamic config and the
   text encoders weight-only (configs constructed and stubbed, not
   quantized: a CPU-only process cannot pass torchao's hardware gate).
   Every refusal fires before unload/download (fake backend; `hub_access`
   never reached). `run_id` changes only for the new name, with legacy
   `fp8` ids pinned. Resume/More seeds mismatch notes, PNG chunk,
   `reindex` and bundle round trips carry `fp8-dynamic`. The `--quant`
   vocabulary for the CLI and the launch menu is covered. Docs maps and
   `ruff` pass. No browser change, so no smoke.
3. **Card (`PERF#2`, needs Matt's OK).** This waits on `PERF#1`'s card
   qualification and uses the existing RAM/VRAM guards: stop on spill or
   at the free-RAM floor. Run one cold load each of `fp8` and
   `fp8-dynamic`, then the `PERF#6` fixed cases at each, one image at a
   time. Record: load working set and private peak; allocated, reserved,
   whole-card and shared peaks; first and warm image and step times; the
   layer inventory; and hash, mean/max channel difference and
   differing-pixel count against `fp8` at the same seeds, with both PNGs
   kept. Matt judges the pairs visually. Results go in `docs/verified.md`.
   A speed gain alone does not admit the choice.
