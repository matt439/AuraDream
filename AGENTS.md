# Local Diffusion

Local text-to-image generator on an RTX 5080. FastAPI backend + vanilla JS
frontend, no build step. The default checkpoint is **FLUX.1 Krea [dev]**
(`black-forest-labs/FLUX.1-Krea-dev`), which is click-through gated; **Krea 2
Turbo** (`krea/Krea-2-Turbo`) needs manual approval on HuggingFace and is what
the sizing notes in `docs/design-notes.md` are written around. The catalogue
in `backend/launch.py` lists the rest; any of them runs unchanged - set
`LOCAL_DIFFUSION_MODEL_ID`, or answer the start-up menu.

## Run

```powershell
uv run python -m backend.main      # http://127.0.0.1:7860; bare on a terminal, asks
uv run python -m backend.main --defaults   # LOCAL_DIFFUSION_* as set, no questions
uv run --group dev pytest          # no GPU, no weights, about a minute
```

```bash
bash wsl/setup.sh                  # WSL 2, once: Linux venv + CUDA check
source wsl/env.sh && bash wsl/serve.sh   # 4-bit Nunchaku, same URL
```

The model is *not* loaded at startup. The first `POST /api/generate` (or
`POST /api/model/load`) triggers a load that takes a few minutes.

## Layout

- `backend/config.py` — settings, most overridable by `LOCAL_DIFFUSION_*` env vars
- `backend/launch.py` — the start-up menu and the `--model/--quant/--offload`
  flags, run before `config` is imported; the checkpoint catalogue lives here
- `backend/schemas.py` — pydantic request/response models
- `backend/model_defaults.py` — recommended steps/CFG/mu per checkpoint and
  the control tooltips; each profile hangs its prompting guide from
- `backend/guides.py` — the guides themselves, one dict per checkpoint: prompt
  and size prose, the documented megapixel range with its word, the authors'
  example prompts, and `analysis` - the prose's checkable statements as data
  for the rule checks, each citing the prose verbatim
- `backend/prompt_encoding.py` — the prompt analyser's encoder layer: each
  checkpoint's own tokenizers, on the CPU, over the exact string its pipeline
  builds - token counts, where reading stops, template-tail loss, literal
  weighting syntax, typed control tokens, T5's unknown characters
- `backend/prompt_rules.py` — the analyser's rule checks against the guide's
  `analysis`: style, documented order, `reorder_tags`, the items left
  unplaced where that decides the order (`unplaced_items`) and what a critic's reorder is held to
  (`places_every_item`, `order_items`, `order_finding`), negations, unquoted text to render, the tag checkpoints' specifics;
  `heuristic` ones labelled
- `backend/prompt_sources.py` — what the critic may cite, per guide constant:
  the guide's own sentences (G) and short verbatim author excerpts (A) with
  their URLs, notes and caveats
- `backend/critic.py` — the local prompt critic: pinned llama.cpp
  `llama-server` and Qwen3-4B GGUF, fetched and verified by `prepare()`, one
  server per `critique()`, the answer validated against its sources
- `backend/translation.py` — lazy CPU-only Marian translation of ERNIE's
  expanded Chinese prompt for the UI; the Chinese prompt remains canonical
- `backend/tags.py` — cached Danbooru tag suggestions, their order
  (`rank_suggestions`: exact match, from the name's start, a later word,
  alias; post count within each; a guide tag ahead once half of it is typed;
  a tag of the recommended negative never first unless typed whole) and exact
  validation with a reason `code` for tag-caption checkpoints; remote failure
  means unchecked, never invalid
- `backend/seeds.py` — the seed range and the wrap, shared by server and CLI
- `backend/loras.py` — the adapter directory: what is in it, and the file
  behind a name (the only place a request's name becomes a path)
- `backend/refine.py` — the refine pass: the image-to-image class diffusers
  ships beside a pipeline's own (`sibling`), that class built over the loaded
  components (`build`), the steps a strength leaves, and the source scaled to
  cover the requested size and cropped; a leaf module like `seeds`
- `backend/sizes.py` — the preset table; the CLI's `--aspect/--mp`, config's
  `max_side`/`max_pixels` and a refine pass's `refine_max_side`/
  `refine_max_pixels` (any 2 MP preset at 2x), `web/state.js`'s copy,
  `web/refine.js`'s two refine bounds, `index.html`'s `max=` and the README
  table are all held to it by `tests/test_profiles.py`
- `backend/backends/base.py` — the `GenerationBackend` ABC
- `backend/backends/diffusers_backend.py` — in-process diffusers implementation
- `backend/backends/loading.py` — everything decided before the weights
  move: the quantization config, the Nunchaku preflight and the
  prequantized refusal, the components left out of the download, the
  device placement; takes `settings` as a parameter
- `backend/backends/single_file.py` — pinned, CPU-only first-load conversion
  for Illustrious XL v2.0-STABLE and WAI v17; atomic cached trees feed the
  normal loader; `sweep_work`, which every converter calls first under its
  target's lock, removes interrupted conversions' work directories (exactly
  `<tree>-` plus tempfile's 8 characters, a real directory holding only
  `tree`), and each converter's work directory ignores its own cleanup
  errors so a failed conversion reports its own error
- `backend/backends/anima_loading.py` — pinned CPU conversion of full Anima
  Turbo v1.1 and its conditioner, with official shared components and a cached tree
- `backend/backends/anima_pipeline.py` — upstream Anima modular blocks behind
  the standard pipeline interface, with per-step progress and cancellation
- `backend/backends/netayume_loading.py` — pinned CPU conversion of NetaYume
  v4's split denoiser, with the official Lumina 2 parent's shared components
- `backend/backends/newbie_loading.py` — pinned CPU conversion of NewBie Exp0.1;
  `newbie_pipeline.py`, `newbie_transformer.py` and `newbie_text_encoder.py`
  provide its callback-capable pipeline, pooled-CLIP denoiser and native Jina tower
- `backend/backends/schedule.py` — how a requested mu becomes the whole
  shift: `forced_mu`, `shifted_sigmas`, the static-shift fallback
- `backend/grid.py` — axis specs (`6-12x2`), prompt variables and prompt
  variants, and the cell expansion, shared
- `backend/latent_preview.py` — the latent-to-RGB maps the live preview
  draws with, one per VAE, fitted on the card
- `backend/jobs.py` — single-worker queue, SSE progress
- `backend/storage.py` — SQLite index + PNGs with embedded metadata
- `backend/search.py` — the search box's grammar: `key:value` terms as SQL
  predicates, the LIKE escape, `strict_scope`
- `backend/main.py` — routes
- `backend/generate.py` — headless CLI; drives the backend directly, no server
- `backend/clear.py` — deletes rows, their PNGs and any orphans
- `backend/stats.py` — the starred *rate* per setting (`storage.keep_rates`),
  scoped to a run by default; `--by variant` gives a comparison's rate
  per version
- `backend/reindex.py` — rows rebuilt from the PNGs' `local-diffusion` chunk, in place or
  copied in from another gallery; idempotent on the UNIQUE filename
- `web/` — served as-is from `/static`; the page script is twelve classic
  scripts `index.html` loads in order (`state`, `sizes`, `status`,
  `generation`, `panel`, `gallery`, `sheet`, `lightbox`, `refine`, `tags`,
  `analysis`, `app`), sharing one global scope; only `app.js`, the last, runs
  anything at load. `panel.js` also holds the results viewer and Reset's
  Undo; `refine.js` is the Refine dialog; `lightbox.js` is the lightbox, its
  comparison of a pass with its source and the shared `factsList`;
  `analysis.js` is the prompt check, the critic and the comparison
- `tools/` — `fake_server.py` (the app on the tests' fake backend, port 7861)
  and `browser.mjs` + `smoke.json` and a `smoke-*.json` per checkpoint or
  feature (headless browser over DevTools): the UI's stand-in for a test
  suite, see Tests
- `wsl/` — `env.sh` (Linux-side paths and the Nunchaku defaults, to source),
  `setup.sh`, `serve.sh`: running the same checkout from WSL 2
- `docs/design-notes.md` — why each part is the way it is, one section per
  area, with the history behind every rule below; read the section for an
  area before changing it
- `docs/verified.md` — what has been measured on the card, per checkpoint
  and per feature, and when

## Rules

The reasoning behind each of these, and what went wrong before it existed, is
in `docs/design-notes.md` under a heading of the same subject; read that
section before changing the area. `docs/verified.md` lists what has been
measured on the card and when.

Model and loader:

- `KBlueLeaf/Kohaku-XL-Zeta` uses its ordinary Diffusers SDXL tree and
  supplied epsilon/Euler scheduler. Start at 28 steps/CFG 5 and disable Mu.
  Keep its hybrid prose/tag guide, 2024-07-10 vocabulary cutoff and
  nonexclusive rating buttons; its explicit caption uses both `nsfw` and
  `explicit`. The published tokenizers limit this pipeline to 77 tokens,
  despite the card's longer training context. The size guidance is around
  1 MP, not a promise of large-square output. No Nunchaku build is mapped.

- Illustrious XL v2.0-STABLE is single-file-only upstream. Resolve it through
  `single_file.resolve_model` before the normal component scan, keeping the
  official id in runtime settings and saved metadata. The converted tree has
  a fixed epsilon/Euler Ancestral scheduler; do not pass pipeline-level
  quantization into `from_single_file` or silently skip the CLIP encoders.

- `LyliaEngine/waiIllustriousSDXL_v170` selects WAI0731's v17 through a pinned
  community mirror, not an official repository or a LoRA. `single_file` checks
  the source SHA-256 against the author's archived release before converting
  on CPU, retaining the embedded VAE and both CLIP encoders. Use pinned SDXL
  descriptors only, epsilon/Euler Ancestral and normal FP8/int8/bfloat16 loading;
  preserve the public mirror id in settings and metadata. Keep the known NaNs
  in the unused final CLIP-L layer unchanged: SDXL consumes that tower's
  penultimate states and the other tower's pooled output. Preserve the regression
  checking the used conditioning. The exact-id profile starts at 28 steps/CFG 6,
  disables Mu and has no Nunchaku mapping. The author's reference sizes and
  separate hires pass are not a published maximum or direct-generation guarantee.

- Z-Anime Base has a pinned nested `diffusers/` tree. `loading.resolve_model`
  resolves configs before quantization checks and weights after them; both
  passes use the same revision and keep the official id in settings/metadata.
  Diffusers 0.40 does not support pipeline-level `subfolder`. Keep the supplied
  FlowMatch Euler scheduler (shift 6); the card's ComfyUI Euler Ancestral/beta
  recommendation is a different setup, not a verified equivalent.
  Its bundled encoder has scaled FP8 tensors without quantization metadata.
  Skip those weights: `loading.component_overrides` loads the standard Qwen3
  encoder from pinned `Tongyi-MAI/Z-Image`, using the same Transformers-owned
  config from `quant_mapping`. Never silently discard the bundled scales.

- `circlestone-labs/Anima` selects the pinned full Turbo v1.1 split file.
  `anima_loading` converts its denoiser and conditioner on CPU, uses the pinned
  official Base repository only for shared components/configs, and caches an
  atomic tree for normal FP8/int8/bfloat16 loading. Keep the official id in
  settings/metadata. `AnimaTurboPipeline` wraps upstream modular blocks with
  the required callback; preserve its fixed flow schedule and exact-id profile
  (8 steps, CFG 1), with Mu, references and user LoRAs disabled. No Nunchaku
  build is mapped to this entry.

- `duongve/NetaYume-Lumina-Image-2.0` selects the pinned v4 split denoiser.
  `netayume_loading` strictly converts it on CPU and uses the pinned official
  Lumina 2 parent for configs, Gemma 2, tokenizers and FLUX VAE, excluding the
  parent's denoiser weights. Keep its public id in settings/metadata. Normal
  per-component quantization loads the atomic cached tree. Its fixed anime
  system instruction is passed explicitly per generation; keep FlowMatch
  Euler shift 6 and the exact-id profile (40 steps, CFG 5). The author's
  v2 family guidance and ComfyUI samplers are not v4 runtime equivalence.
  No Nunchaku build is mapped to this entry.

- `NewBie-AI/NewBie-image-Exp0.1` selects pinned official components, strictly
  converted on CPU by `newbie_loading`. Keep its public id in status/metadata.
  Diffusers 0.40 has no native NewBie pipeline; use the local classes, Gemma 3
  penultimate hidden states and Jina's unnormalized masked mean. Keep both
  prompts free of system prefixes, FlowMatch Euler shift 6, normalized reverse
  transformer time, and whole-sample CFG norm limiting. The exact-id profile
  starts at the card's 28 steps and the author's sample CFG 4. XML is text,
  never comma chips; 1024x1024 is an example, not a published size ceiling.
  References, user LoRAs and Nunchaku are unavailable for this integration.


- App-owned names use Local Diffusion: `LOCAL_DIFFUSION_*`,
  `local-diffusion.db`, `~/local-diffusion`, `local-diffusion.*` browser keys
  and the `local-diffusion` PNG chunk. Keep Krea only in checkpoint/vendor
  names and explicit compatibility reads for pre-rename data.
- Quantize through `PipelineQuantizationConfig(quant_mapping=...)`, one
  `TorchAoConfig` per component from the library that owns it, each built from
  a torchao `AOBaseConfig` *instance*. The `quant_backend="torchao"` shorthand
  raises on a transformers text encoder.
- `torch.no_grad()`, never `torch.inference_mode()`, around the pipeline call:
  model offload moves components inside `forward()` and torchao's tensor
  subclasses trip torch 2.13's storage-aliasing guard under inference_mode.
- Loading happens only in `main.ensure_loaded()` under its asyncio lock; jobs
  await it, `generate()` raises on a missing pipeline, and
  `POST /api/model/unload` answers 409 while a job or a load is live; once
  `try_unload` succeeds, the route sets the state to `unloaded` before the
  translator, critic and tokenizers are let go. A load
  first waits out whatever holds the generation gate (`jobs.wait_for_gate`,
  inside the lock, without keeping the gate): a job queued during a critique
  must not move weights onto the card beside the critic's server. A failed
  load is mapped by `main._load_failure` to a sentence with the steps, with
  `ModelStatus.detail_kind` and `detail_url` beside `detail`, for
  huggingface_hub's `GatedRepoError`, `RepositoryNotFoundError`,
  `OfflineModeIsEnabled` and `LocalEntryNotFoundError` (by class name over
  the MRO and the `__cause__`/`__context__` chain; the gate's kind from
  `launch.CHECKPOINTS`); anything else stays `short_error`, and the log
  keeps the library's words. It is raised on as `jobs.LoadFailed(detail)`,
  so a job that ended at the load and Expand carry the status's sentence.
  `ModelStatus.started_at` is set once, at import.
- `load()` refuses a pipeline whose `__call__` lacks `generator` or
  `callback_on_step_end` (`_REQUIRED_PARAMS`): the seed and the cancel depend
  on them.
- `LOCAL_DIFFUSION_QUANT=nunchaku` means a pre-quantized lite-infer repo, Linux only
  (WSL 2 here), and no quantization config; `loading.nunchaku_preflight` and `loading.refuse_prequantized` refuse the other pairings before any download. Under
  WSL, `source wsl/env.sh` before `uv`.
- `config.py` sets `DIFFUSERS_TRUST_REMOTE_KERNELS` and must be imported
  before diffusers everywhere, the CLI included. The offload hooks are handed
  `LOCAL_DIFFUSION_DEVICE` explicitly; given nothing, diffusers takes GPU 0.
- `loading.UNUSED_COMPONENTS` names what ordinary image loading leaves out
  (ERNIE-Image's prompt rewriter). `load()` reads `model_index.json` ahead of
  the weights and hands them as None to the *concrete* class's
  `from_pretrained`, which keeps them out of the download and the load; the
  base class takes `pe=None` for a pipeline kwarg and pulls the 7.7 GB.
  The explicit Expand endpoint loads those components on first use, rebuilds
  the offload chain, and puts the result in the prompt editor. `generate()`
  still sets `use_pe=False` wherever the switch exists. The tests'
  conftest refuses `loading.hub_file` for every test; one that needs an index stubs it.

Schedule and guidance:

- A requested `mu` is the whole shift. `schedule.forced_mu` overrides the value
  on its way into a dynamically shifting scheduler; a static one gets
  `schedule.shifted_sigmas` pre-shifted by `mu - ln(shift)`. Assert on
  `scheduler.sigmas`, never on the list the backend computed. Mu requires a
  `FlowMatch*` scheduler first: SDXL's Euler scheduler accepts custom sigmas
  too, but is not flow matching. `mu_supported` is profiled before load and
  read from the scheduler after it, and disables both Mu controls when false.
- Where guidance switches on is the pipeline's: `cfg_threshold()` is 1.0 with
  `true_cfg_scale`, 0.0 for a `negative_prompt` pipeline in
  `_GUIDES_FROM_ZERO`, else 1.0. The profile guesses until a load;
  `/api/status` merges the live answer, as it does `mu_supported`,
  `negative_prompt_supported`, `true_cfg_supported`, `reference_supported`,
  `refine_supported` and `lora_supported`.
- `GenerateRequest.applied()` is the one normalisation - negative prompt, mu,
  references, `lora_scale`, the refine source and its `strength` - and
  `backend.applied()` answers it from the parameter list or the profile. Every save path, the resume prune and the
  duplicate count go through it, never through the raw request.
- `true_cfg` on a row means the scale the negative prompt ran at, or None, on
  every checkpoint.
- `model_defaults._PROFILES` is first-match on the id, so a distilled variant
  sits ahead of its family's token (klein and schnell before flux, z-image-turbo
  before z-image, ernie-image-turbo before ernie-image); a catalogue entry
  that reaches the fallback is a test failure. Steps, CFG and mu are the
  checkpoint's; `mu` is Turbo's alone. `krea-dev` sits ahead of `flux` for
  its guide alone and is the flux profile in every number and control hint.
  `_EXACT_PROFILES` takes precedence for pinned releases such as WAI v17
  and Anima Turbo; never match Anima with a broad `anima` token, which also
  occurs in Animagine.
- Every profile's `guide` (`backend/guides.py`) is sourced from the checkpoint's authors - the
  model card, README, prompting guide or report - never invented; the
  design notes keep the URLs. `pixels` is the range they put in writing and
  `pixels_word` what that writing was (documented, recommended, trained);
  the range must admit the 1 MP presets. Its prose rides in `hints` as
  `prompt` and `size`, its data in `ModelStatus.guide`.

Reproduction:

- Seeds are drawn from `2**53 - 1` and wrapped at `2**63 - 1`; both live in
  `seeds.py`. One generator per image, seeded `seed + i`; runs take base
  `seed + i * num_images`.
- `grid.expand()` is the one cell loop for the CLI and `/api/grid`;
  `jobs.submit()` keeps its inline copy of the seed loop deliberately, asserted
  equal to the CLI's in `tests/test_cli.py`.
- The reproduction set - prompt pair, size, steps, CFG, true CFG, mu, seed,
  checkpoint, quantization, references, LoRA and weight, refine source and
  strength, variables, hash - is on the row and in the PNG's
  `local-diffusion` chunk. `CellKey` holds the prompt pair, steps, CFG, mu,
  true CFG, references, LoRA and weight, refine source and strength, and the
  seed; size, checkpoint and quantization scope the resume lookup
  (`storage.completed_cells`) and, with the batch, `run_id`'s definition;
  variables and hash are the row's and the chunk's alone. `run_id` names a
  cell by the shortest key carrying its non-None optional elements, so every
  existing run id survives a widening. `/api/rerender` and `--from-image`
  carry a row's variables. The CLI carries them only while the prompt pair is
  the image's own and never beside a `--grid-prompt` axis; it rebuilds a
  comparison version through `expand(variants=)`, verbatim, and attaches any
  other assignment, including a `variant` value no `PromptVariant` label
  admits.
- Resume keys on settings (`storage.completed_cells`, six-decimal rounding,
  scoped by size, checkpoint and quantizer), requires an explicit seed unless
  the seed axis is a list, and answers 409 when nothing is left. Without
  Resume, `/api/grid/preview` asks the same lookup (`main._not_yet_rendered`)
  only for an explicit seed or a seed list, reports it as
  `GridPreview.in_gallery` and drops nothing.
- A new column needs a `_migrate` entry, not a `"x" in row.keys()` read guard:
  `_row_to_record` reads every column unguarded. `save_image` and
  `reindex.index_file` share `_insert_row` over `_COLUMNS`.
- Reference images are gallery ids, never paths. The page's chips are
  numbered in `referenceIds` order, since that order is what a prompt's
  "image 2" means; every reorder or removal (a drag, Alt+Left/Right or Delete
  on a chip's ×) builds a new array through `setReferences`, the one writer,
  and chip nodes are never moved. After a removal focus goes to the next ×,
  else the previous, else the emptied `#reference-list` (tabindex -1), never
  the field's badge, whose tip a focus opens. A LoRA is a bare name that
  only `loras.py` turns into a path, `.safetensors` only, and it is
  hot-swapped in `generate()`, never baked into `load()`. Whether it fits is
  `loras.mismatch` - the adapter's module names and widths against the
  denoiser's, using the pipeline's adapter reader and PEFT key conversion -
  from the checkpoint's configs on the meta device before a load
  (`loading.denoiser`) and the live pipeline
  after; a misfit is listed with its `detail`, hidden by the page and
  refused at submit and by the CLI, unreadable configs mean unchecked rather
  than refused, and a listing never converts a single-file checkpoint.
- A refine pass is `GenerateRequest.refine` - a gallery id, like a reference -
  and `strength`: the source resized to the request's size (`refine.fitted`:
  scaled to cover and centre-cropped, never stretched) and re-denoised from
  the point on the schedule the strength names. It runs the image-to-image
  class diffusers exports beside the loaded pipeline's own (`refine.sibling`:
  by name, only for the class diffusers itself exports, and only where its
  call takes `refine.REQUIRED` and every `refine.CARRIED` argument the
  text-to-image call takes; `CARRIED` is held to `generate()`'s own kwargs by
  `tests/test_refine.py`), built once over the pipeline's own components by
  `refine.build` - never `from_pipe`, which ends by casting the modules it
  was handed - and let go on unload. `refine_supported` is profiled before a
  load, each guess held to the installed diffusers, and the class's answer
  after it; unsupported, or a source that is gone, is refused at submit and
  by the CLI. Both recheck after loading and fail before rendering if the
  live capability would drop the requested source. `applied()` drops the source, puts `strength` back to
  the checkpoint's default with it, and drops references under a pass. That
  default is the profile's `refine_strength` (`schemas.default_strength`,
  `refine.DEFAULT_STRENGTH` where the profile names none), filled in for a
  request that gives no strength and returned by live status when an
  unprofiled pipeline supports the pass, and where the Refine dialog's
  Strength starts, because a strength is an index into the
  checkpoint's own schedule: 0.4 on SDXL, 0.1 on FLUX.1 [dev], where 0.4
  repaints a face. Each value must leave a step of its profile's own step
  count; `docs/design-notes.md` has each one's basis and `docs/verified.md`
  says which were measured. One call
  per image of a batch, each with that image's own generator
  (`DiffusersBackend._refine`): the VAE's encoding of the source is sampled,
  and `FluxImg2ImgPipeline` samples a batch's from the first generator alone.
  The loaded pipeline's `maybe_free_model_hooks()` runs in that method's
  `finally`: the sibling has no hook list of its own. `steps` on the row is
  the schedule's length, of which about `steps * strength` run
  (`refine.steps_run`, SDXL/flow rounding from `refine.start_index`; unknown
  profiles use the floor until the callback supplies the pipeline's count).
  A job's initial `total_steps` uses that count. `storage.step_cost_ms` prices
  direct rows only; `storage.refine_cost_ms` prices a pass from full per-image
  durations at the same size, checkpoint and denoised step count, with no
  estimate where matching history is absent. The duplicate advisory compares
  schedule start indices; reproduction keys retain the requested strength.
  `GenerateRequest.check_refine_steps()` refuses less than one step after
  the capability check and on each expanded grid cell, never the unused base.
  `refine_key`
  (`id@strength`) is `CellKey`'s fourth optional element;
  `GridRequest.strength_axis` is an axis only beside a source, as the LoRA
  weight is only beside an adapter. Re-render, `--from-image` and Reuse carry
  both; a database merge remaps the source id as it does a reference's.
  PNGs also carry `reference_sources`/`refine_source` descriptors (filename,
  pixel hash and dimensions): PNG-only recovery verifies these before
  assigning new ids, clears/counts unproven legacy or missing links, and
  rewrites changed ids in PNG metadata even during an in-place rebuild. The
  CLI's `--scale` and the dialog's scale buttons share one rounding
  (`refine.scaled`, half up). A pass has a size guard of its own,
  `refine_max_side`/`refine_max_pixels` (any 2 MP preset at 2x: 4448 and
  8,411,136): the schema's `width`/`height` bound is that,
  `_direct_renders_stay_in_the_presets` holds a request with no source to
  `max_side`, and `check_budget` holds a pass to `refine_max_pixels` and
  skips the batch budget, since a pass is one call per image. A typo guard,
  not a VRAM model: what fits is the checkpoint's (`docs/verified.md`). The
  dialog's tight line - where its size line warns and past which it does not
  open - is the profile's `refine_measured` (`model_defaults.Measured`: the
  checkpoint and quantization of the largest pass verified.md's "Past 4 MP"
  table ran without spilling, and its size; `tests/test_profiles.py` holds
  each to its row), reaching the page as `defaults.refine_tight_pixels` and,
  in words, `hints.refine_tight`; None, the generic 1600x1600 line, for
  every profile with no row. A shared profile shares its row (FLUX.1 dev and
  the lite repos quote Krea dev's fp8 run), so the size line names the run
  it quotes, never "this checkpoint". FLUX.1's `hints.refine_large` is shown
  under Strength past it (`applyHints` clears it where a status has none).
  The preview reports no live start sigma: for a dynamically shifting
  pipeline it would need `calculate_shift` re-derived outside the pipeline.
  `ImageRecord.refine_missing` is read, not stored (`storage._missing_sources`,
  from `list_images` and `get_image`), and is not in the reproduction set.
- Under Nunchaku (WSL) a seed does not reproduce bit-for-bit - Z-Image
  within noise, ERNIE visibly - so a duplicate count or a re-render's hash is
  the fp8 path's guarantee only; `docs/verified.md` has the numbers.
- `GridRequest.prompt_variants` (2-8 `{label, prompt}`) is the prompt axis in
  place of the variables, never beside them: each variant is taken verbatim
  (nothing substituted; a `{`, `|` or newline is text), seeds stay outermost
  with the variants next in the order given, and each cell records
  `variables={"variant": label}`, whose starred rate per label is
  `stats --by variant`. `CellKey`, `run_id`, resume and the duplicate count
  read the resolved prompt, not the label. A placeholder in the negative
  prompt is still refused.

Jobs:

- Cancel and pause are cooperative and polled: cancel from
  `callback_on_step_end` (raise `Cancelled`), pause at the cell boundary
  every quarter second. Never `Event.wait()`. The page answers a Pause press
  at once ("Pausing…"; a press on it withdraws the request, one on Resume
  reads Pause) and shows Resume only on the worker's `paused`, deriving the
  label from every status frame's `paused`/`pause_requested`; the label is
  the state, so it has no `aria-pressed` (the toggle rule, Frontend). A
  running job of one cell has Pause `aria-disabled` with the reason. A plain
  job's units are images, or batches where a repeat makes several
  (`unitWord`, from the snapshot's `grid` and `planned_images`), in the
  progress, the finish line and the last-run banner ("stopped at image 2 of
  3", "(2/3 batches)"); "cell" is a grid's word.
- A job is a list of cells; seeds are the outermost loop, prompts next. A
  cell's failure skips the cell; three in a row or more than a third of the
  grid ends the job. Every terminal transition goes through `_finish`, and a
  budget stop returns through the cancel path rather than a fourth status.
  Each skipped cell is kept in `Job.failed_cells` ({label, seed, error}),
  sent only with the records (the full snapshot). When the failure rule ends
  a job, `stopped_by` is "failures" (status still `error`, through
  `_finish`) and its error reads "Stopped at cell X of Y after N cells
  failed; K images kept. Last: <short_error>". The page's Retry ("Retry the
  N skipped cells") and Resume ("Resume the remaining N cells") send the
  job's own submitted payload (`submissions`) again with `resume: true` and
  `payload.seed ?? JobAccepted.seed` (none beside a seed list) - never the
  POST's seed over the payload's own, never the panel - each once ("Retry
  sent", "Resume sent", `aria-disabled`, focus kept).
- Each SSE stream has its own queue; `events()` opens with `done` for a job
  that has finished; the per-cell event and `/api/jobs` send
  `snapshot(images=False)`. Progress events carry `elapsed`; `waiting_on`
  says `load` or `queue`. A snapshot's `images` are the job's live records:
  the delete routes call `jobs.forget_images`, which takes deleted ones out
  of every job's replay and counts them in `deleted_count`; `image_count`
  stays what the job produced. Snapshots also carry `prompt` (its first 120
  characters), `refine`, `width`, `height`, `planned_images`, `grid`,
  `comparison` (`run_grid`'s `bool(prompt_variants)`) and `prompt_template`
  (a grid's prompt as submitted, its variables unfilled, from `/api/grid`;
  None for a plain request and a re-render), so a page that did not submit
  a job can name it - a grid by its own prompt, never its first cell's.
- `main.Server.handle_exit` cancels the active jobs and starts the critic's
  stop on the signal itself. The load, the critic's `prepare()`, Check's
  analysis and the critique's analysis and lint run on daemon threads
  (`_detached`), as does each tokenizer load, so interpreter exit joins none
  of them; a first use's tokenizer fetch holds a Check, and so a Ctrl-C's
  drain, for `prompt_encoding.LOAD_WAIT` at most. Lifespan stops the critic,
  then `jobs.shutdown()`, then calls `try_unload()`.
- `save_image` writes `.png.part` and renames after the row commits, retrying
  the rename; the orphan sweep leaves `.part` files younger than an hour.
- The preview callback (`_step_end` -> `latent_preview`) never raises - an
  exception there is how `Cancelled` ends a job - and `jobs.on_preview` checks
  for subscribers per frame. A preview needs a matrix for the VAE and either
  `_unpack_latents` or already-unpacked 4-D latents.
- Generation is serialized through one worker thread; the GPU cannot usefully
  run two. Interactive accelerator work - ERNIE's Expand, the prompt critic -
  goes through `jobs.run_when_idle`, so it neither cuts ahead of queued images
  nor overlaps a newly submitted job; it checks the queue again once it has
  the gate. The critic's download and its rewrite's lint run outside the
  gate. Nothing that holds the gate awaits the load lock, which is why
  `wait_for_gate` cannot deadlock.

Prompt analyser:

- Three layers over the body Generate sends (`PromptAnalysisRequest`): Check
  (`POST /api/prompt/analyze`), the critic (`POST /api/prompt/critique`) and
  the comparison grid (`prompt_variants`). Check works cold: CPU tokenizers
  and rules on a `_detached` daemon thread off the loop, never
  `ensure_loaded`, never the gate, `applied()` from the profile before a
  load. The negative prompt is measured and checked only as `applied()` runs
  it. `POST /api/prompt/measure` is its encoder layer alone
  (`prompt_encoding.measure`, no rules, no tag lookup) for the chip tray's
  token count and cut, on the same terms; an encoder left uncounted is "cut
  unknown" there, never "no cut". The Refine dialog checks its own request
  (never `collect()`) on Check's route (`POST /api/prompt/analyze`, the
  request without `strength`, `strength_axis` or `seeds`) under an epoch of
  its own
  (`checkRefinePrompt`): the encoders' non-pooled cuts with Show, and rule
  findings' whole-field Apply written through `replaceBoxText` into the
  dialog's boxes, never the panel's; stale while either box's text or the
  negative's activation differs from what it read, a negative-prompt Show or
  Apply dead (saying why) while the dialog's negative box is greyed; never a
  live region itself - a visually hidden polite line (`#refine-check-said`)
  carries its text, written only when that text changes; it never holds
  Refine and is silent on failure.
- Every finding says what it rests on: `source` (`guide`, `author`,
  `encoder`, `heuristic`), `cite` and `url`. Rule config lives in each guide's
  `analysis` key, never a profile key or `guide_for`; a `guide` cite is a
  verbatim substring of that guide's `prompt` prose and an `author` one of an
  example (`test_every_cite_is_the_guides_own_words`). A check no author
  sentence supports is `heuristic`, with no cite. `documented_order` is None
  where the guide states none; never invent one, and never say the authors
  give none - the page says the app checks none and shows the authors' own
  (`Excerpt.order` -> `author_orders`). `order_checked` is True only where a
  rule placed the prompt (a tag checkpoint's comma items, each stretch of
  lettering one item with its commas: not XML or JSON, at most 256 items, and
  not where a line break or a fullwidth comma splits an item outside its
  lettering - multi-line prose included; `_placeable` is the one predicate
  `order_checked`, `reorder_tags` and the order rule share); the page says a
  documented order it was not is "Not checked". A rule's `suggestion` is a
  whole replacement for its field, offered only where it is unambiguous
  (lettering whose end is a guess gets none); an encoder's is advice text.
  `prompt_rules.QUOTED` is the one reading of quoted text, for the rules, the
  encoder layer and the critic; its single-quote pair may touch Han, kana
  and Hangul but no other letter or digit (写着'营业中'的 is a quotation,
  animal's an apostrophe). `prompt_rules.rendered_targets` gives each place
  the text asks for lettering as (start, end, ends, head): `ends` is False
  only where the rules stopped at a guess and the lettering may run on - a
  joining or capitalized stop word, six words, the 12-character cap, or
  lettering going on past a space or an ASCII mark (写着OPEN NOW, 9:00, www.shop.com) - and only
  there may the critic read on past `end` in any words of its clause; past
  an end the rules are sure of, it reads on only in words that start with a
  capital or a digit. The rules stop a participle's lowercase words at a negation
  the negation rule reads an absence from there, other than one right after
  "or" or "than" (`_absence_at`, so "reading do not enter" and "reading now
  or never in bold type" are whole, and "reading love without limits"
  quotes "love" and warns on the rest); `rendered_targets` reads them on
  through it to where they stop without it, `ends` True ("love without
  limits", and "above the stage" after it is description), with `head` at
  the rules' stop (`end` elsewhere): the critic takes as lettering only
  words that start before `head`, so "without limits" alone is the
  absence, not a piece of the lettering. It is not the Apply's
  `_Target.exact`: a Chinese target stopped at 的/之, at 等 before 字样, or
  at a verb placing the sign (挂在, 位于) that no request or prohibition asks
  for gets no Apply, and neither do lowercase words after a participle, yet
  what follows them is description; nor do words not in capitals after a
  noun trigger ("with the words set in bold type"), whose `ends` stands.
  `rendered_spans` is the same list without the flag or `head`; the rules'
  own reading of lettering (`_read`, and `_tag_items` outside `check`) keeps
  the negation's stop. The
  rules read each stretch of lettering, commas and all, as one comma item
  (`_units`): it moves whole and is placed by its shape alone, is never
  matched or rewritten as a tag (`_Context.norms` is None for it), is never
  asked of the tag index, and a reorder that would change what `QUOTED`
  reads is not offered. English punctuation inside times, prices, URLs or
  capitalized lists makes the quote boundary uncertain too: no partial Apply.
  NetaYume's prefix rule searches `ctx.bare` so prefix-shaped lettering is
  preserved and does not hide a genuine duplicate later in the prompt.
- `prompt_encoding` measures over the exact string the pipeline builds, and
  every template string, length and `enable_thinking` switch it copies is held
  equal to the installed diffusers source (and the repo's Anima and NewBie
  pipelines) by `tests/test_prompt_encoding.py`. Its table is first-match
  like `_PROFILES`, exact ids first, reusing the loaders' pinned constants.
  Tokenizer files only. Each load runs on a daemon thread outside the cache's
  lock (`_start`/`_fetch`); concurrent checks share the load in flight until
  an `unload()` forgets it, and a load is kept only if it succeeds with no
  `unload()` since it began. A check waits `LOAD_WAIT` (30 s) for its loads
  at most, then reports the tokenizer as still loading while the load goes
  on; `unload()` never waits on a load. `measure` never raises, and
  `encoders_detail` says what went unchecked, each encoder under its own
  reason. Never from `/api/status`; `prompt_encoding.unload()` runs on
  Unload and shutdown. `\(`/`\)` escapes are not flagged.
- Tag categories for the order and artist checks come from the chips' own
  lookup (`_lookup_tags`), only on a Danbooru-vocabulary guide or one whose
  rules read artists (Anima, `wants_artist_categories`, no cutoff) and a tag
  or hybrid prompt, capped at the chips' 64 items; any failure there is
  `None` - order placed from the guide's own groups, never guessed - not an
  error. What that cost is said in `PromptAnalysis.categories_detail`: the
  index unreachable (a connection error), no usable reply from it (an HTTP
  error, or a cut or unreadable body), or items past the 64. Whether a
  placed order was only partly checked is one predicate, the placement
  itself: `order_checked` and `prompt_rules.places_every_item` false with
  the analysis's categories - some item the order rule reads kept every
  slot where that decides the order (`unplaced_items`: some slot puts it
  out of order beside another item - not its own copy, written the same and
  so in the same slot - and some does not; a lone item, or one whose every
  slot gives the same answer, is not partly checked): never
  looked up for any of those reasons, a tag the index answered for with no
  usable category (unknown, deprecated, past the cutoff), a prose prompt's
  clause that is not a whole sentence (prose is never looked up), lettering
  or an @ name. `PromptAnalysis.order_detail` is set exactly there, naming
  those items and why; the page then shows "Partly checked: <order_detail>" under the
  chain and "order partly checked" in the summary, never from
  `categories_detail`, and the critic's reorder other than Check's can stand
  exactly there (the critic's answer bullet below), since Check's own order
  is then one the documented order allows, not the only one. Where the
  guide states no order (Anima), `categories_detail` alone makes the page
  say not every tag was checked for an artist name. The reported `style` is
  `detect_style(prompt, categories)`, the one the rules applied.
- The critic is pinned: llama.cpp `b11206` archives (`critic.BUILDS`: Windows
  CUDA 13.4, and Ubuntu for WSL) and `Qwen/Qwen3-4B-GGUF` `Qwen3-4B-Q8_0.gguf`
  at a fixed revision, each held to its size and sha256 (`.part`, verify,
  extract beside, rename; traversal refused; a tree whose marker is missing
  or does not decode never used).
  `LOCAL_DIFFUSION_CRITIC` (default on), `LOCAL_DIFFUSION_CRITIC_SERVER` and
  `LOCAL_DIFFUSION_CRITIC_MODEL` (absolute, resolved only on the server's
  command line: a link keeps the name answers are labelled with; neither
  held to the pins). The server setting should name llama-server itself: a
  wrapper is stopped with its tree, but a server it moves out of that tree
  (on POSIX, into a session of its own) escapes the stop, and on Windows and
  Linux the listener check refuses its port.
  `supported()`, `ready()` and `describe()` are file checks (and
  `supported()`'s one libc call), fit for `/api/status`: without a server of
  one's own, `supported()` is False where the Linux build's glibc 2.38 is
  missing. `describe()` (`ModelStatus.prompt_critic`) gives the page the
  model's label - the answer footer's name - whether it is the pinned pair,
  and what a first use would download; the line under the button promises
  the measured time and the GPU for the pinned pair only. For a pinned build
  `prepare()` runs `critic.preflight` ahead of any download - glibc, then
  libgomp.so.1/libssl.so.3/libcrypto.so.3 asked of the dynamic loader in a
  child interpreter - and refuses (503) naming the apt packages.
  `install_server` sweeps, under its lock, the work directories interrupted
  installs left (`critic._is_work`): exactly `<target>-` plus tempfile's 8
  characters (`[a-z0-9_]`), never another build's own directory, a real
  directory (no link or junction) holding only `tree` and `archive-<n>`
  (whole names) - never a user's copy named like one; best-effort, it never
  fails an install. A cached GGUF off its pin is downloaded again once
  (snapshot entry removed, `force_download`). A re-download that fails is a
  `CriticError` (502) quoting the Hub's reason; one still off its pin names
  the file and says the pin or the download path is wrong, never
  "delete it and retry".
- The critique route: 422 unsupported (`critic.pinned_build_refusal()` names
  the glibc a published build needs, where that is why `supported()` is
  False), 409 while a job is queued or running or the image model loads
  (checked before `prepare()` and again after it),
  `prepare()` on a daemon thread outside the gate, then `critique()` inside
  `jobs.run_when_idle`. One `llama-server` per call on 127.0.0.1, a free port
  and a fresh API key passed as `LLAMA_API_KEY` in its environment (never on
  the command line, which other local users can read), refused when
  `/v1/models` answers without it or when the port's listener is not its own
  process tree (`_listener_is_ours`; Windows and Linux, unchecked elsewhere).
  Only then is it asked to count the question itself (`_counted`:
  `/apply-template`, `/tokenize`); `check_fits` refuses the question unasked
  on that count, and `max_tokens` is sized from it. `estimate_tokens` only
  refuses, scaled by `_ESTIMATE_FLOOR` before a server starts, what cannot
  fit however it tokenizes, and sizes the question for a server that will
  not count. The server is stopped (terminate, wait, kill) as a whole tree in
  `finally`, before the gate opens (`_ServerProcess`): on Windows a job
  object, joined while the child is suspended and waited for by each
  process's handle; on POSIX its own session, and on Linux every process in
  that session. A spawn failure and a kill not waited out are `CriticError`s.
  `_process` is cleared and the log removed regardless (retried, and only
  logged if still held).
  `prompt_critic.unload()` on Unload, in lifespan before `jobs.shutdown()` and
  on a thread of `handle_exit`'s own, each through `main._stop_critic`, off
  the event loop: a `CriticError` (a kill not waited out) is logged and the
  rest still runs - Unload still drops the tokenizers and answers with the
  status, whose "unloaded" is the image model's and whose VRAM is read off
  the image model's card (the server is pinned to no card, so only on a
  one-card machine does what it still holds show there).
  `CriticUnavailable` is 503, `CriticError` 502; `_prompt_failure` sends
  their messages without the class-name prefix (the log line keeps it).
- The critic's answer is checked before it is shown (`critic.validate`):
  `cite_id` is an enum of the numbered G and A sources in the answer's JSON
  schema and is resolved server-side (an unknown id drops the issue, counted
  in `dropped`); a quote is located in the prompt as whole words
  (`critic.locate`: never inside a longer word, except in Han, kana, Hangul,
  Thai, Lao, Myanmar or Khmer, with a tag's underscore as a space; curly and
  straight apostrophes, quotation marks and dashes folded one for one) or
  nulled; a part must be the prompt's own words and `text_to_render` its
  lettering (`prompt_rules.rendered_targets`, starting before a target's
  `head` and running on past an end the rules are sure of only in words
  that start with a capital or a digit);
  issues restating a rule finding reported on the prompt are left out - by
  wording, or by a quote that takes in a finding's words and runs at most 8
  characters past them (none past a single word), on guide, author,
  heuristic and weighting spans only, never a cut, template tail, unknown
  characters or control token - and one restating a negative-prompt finding
  only when it names the negative prompt (`reported_negative`); so are issues
  speaking to a length the rules measured in range (`measured_length`, not
  XML or JSON; `_says` reads a sentence at a time and skips lettering and a
  count worded as some part's - "at least 5 words for the lighting", "only
  3 words of the prompt are about the lighting", "describe the umbrella in
  under 5 words", "the lighting has only 3 words", "keep the lighting
  description under 5 words": `_SOME_PART`, `_MANNER`, `_PART_COUNT` -
  unless it names the prompt, its description (the word just ahead of
  "description" one of `_DETERMINERS` or `_WHOLE_WORDS`: "your
  description", "the revised description"; any other word there names a
  part, "your lighting description"), "it", a total, or the guide or its
  range as such, or fits them: "to reach the range", not "the dynamic
  range" or "to match the lighting"; a heuristic, every word of it pinned by a test row), an order issue the documented order reverses (a part first or a
  "put X before Y", `_reversed`; a negated instruction wants nothing), and
  order issues that cannot all hold.
  `reordered` survives only under a documented order and as a pure reorder
  of whole tags, clauses, sentences or XML blocks with quoted text unchanged
  (`is_reorder`, typed marks folded; every `rendered_targets` span is kept
  whole during piece splitting, even when tag order is unchecked, and initials
  or common abbreviation periods never split a name; for JSON, equal values
  with the keys sorted, each number as written - 0.50 is not 0.5 (`_json_forms`) - and no
  repeated key). What is kept has the prompt's own pieces, each as the
  prompt writes it, in the critic's order with the critic's space and
  punctuation between them (JSON keeps the critic's text, whose values are
  then the prompt's writing). Where Check placed the tags
  (`order_checked`) it stands only as the rules' `reorder_tags` text -
  unless the rules could not place some item the order rule reads where
  that decides the order (`prompt_rules.places_every_item`, placing with
  the analysis's own categories, `PromptAnalysis.categories`, excluded from
  the response: an item never looked up or one the tag index had no usable
  category for - decided from the placement, never from `categories_detail`, which says only that some item
  was never looked up; the same predicate sets `order_detail`, so the page's
  "Partly checked" and this exception cover the same prompts). There it
  stands where it keeps the items the order rule reads whole
  (`prompt_rules.order_items`: lettering one item, commas and all), the
  rule reads it, and the rule, placing with the same categories, flags
  nothing in it (`prompt_rules.order_finding`).
  `PromptCritique.reordered_kind` says which was kept (`check`, `allowed`,
  or `unchecked` where Check reads no order); the page's Use reordered title
  follows it, and the page has no reorder rule of its own. A rewrite wrapped
  in quotation marks (sentence punctuation after the closing mark included,
  spaced or not: a stop kept, a comma, semicolon or colon dropped) is
  unwrapped; retained whole tags keep their original parenthesis escaping
  before the unchanged check, so removing those escapes alone is no rewrite.
  The critic is told that character and series names are optional: an order
  template positions supplied parts and never requires inventing identity.
  One that is the prompt unchanged (whitespace, a closing stop and typed
  marks aside) is dropped; one quoting words the prompt lacks is dropped; one
  quoting its description is flagged. What was left out besides `dropped` -
  the issues above, named parts that failed, a rewrite over-length,
  unchanged or quoting words the prompt lacks (and a wrap taken off), a
  reordered version too long, changed, not Check's, regrouped, unread or
  flagged by the order rule - is said in `set_aside`. Typed Qwen3 control
  text (`<|...|>`, `<think>`, `<tool_call>`, `<tool_response>` and closers)
  reaches llama-server defused by a zero-width joiner after the `<`, which
  `validate` takes out of the answer. The rewrite is linted by layer 1 in
  the route after the gate opens (`_lint`, no tag lookup), since a lint can
  fetch a tokenizer. What survives is still the model's suggestion; the docs
  must not call it trusted. Reasoning stays on. A G sentence listed
  `unsourced` in `prompt_sources` - the app's own menus, buttons, sampler,
  what it does not implement, or its own judgement - is never citable and
  no rule entry cites it; how the pipeline reads the prompt (a token limit,
  held to `prompt_encoding`) stays citable. A G sentence carries one page's
  words and links the page of the `analysis` entry whose cite it carries,
  else its `sentence_urls` page, else the guide's `url`; the fallback's
  sentences carry the app's own note, are labelled General, not Guide, and
  are never called the authors'. A borrowed or translated excerpt says whose
  it is, to the critic and on the page (`Citation.note` ->
  `CritiqueIssue.note`, under the issue and in its tooltip).

Frontend and CLI:

- `web/` has no build step and no test suite; check it in a real browser with
  `tools/fake_server.py` and `tools/browser.mjs` (see Tests). The page script
  is classic scripts sharing one global scope, loaded in the order
  `index.html` lists them; their top-level bindings stay top-level, and only
  `app.js`, the last, runs anything at load.
- `web/style.css` takes its type, greys, spacing, control heights, radii and
  layout widths from the tokens on `:root`, whose table says what each is
  for; a size the page shares is a token, a one-off stays a literal, and a
  control's vertical padding is worked out from its height token
  (`--line-strong` is the controls' borders, `--topbar-h` the topbar's set
  height). The settings panel's sticky top is `panel.js`'s (`panelPos`,
  written inline): resting under the topbar while the panel fits, else
  between that and the top that holds its bottom a gutter above the window's,
  moved there by the user's own scrolling only (never by scroll anchoring or
  a page that shortens), placed on a change of its height or the window's
  but not while the pointer is over it (Generate must not move under the
  pointer that pressed it), and moved to bring a control focused out of
  sight into view, the browser's own scroll to it taken back. Never a
  window-height media query to unpin it or set its top, and never
  `overflow-y: auto` on the panel, which would clip the Examples menu, the
  tag menu and the tips; it stays `position: sticky` at z-index 10 above 900
  px and static at or below it, and a tip that would run past the window's
  bottom opens upwards (`placeTip`).
- A window 1800 px and wider overrides the tokens in one media query (the
  wide tier): the type ramps by `--ramp` from its 1800 px size to its 2560
  one (body 14 to 15.5 px, captions 12 to 13.5), the panel is
  `clamp(400px, 18vw, 480px)`, the cards `clamp(200px, 9vw, 260px)`, and the
  prompt box grows with its text from five lines to twelve. That growth rule
  is `#prompt:not([style*="height"])`, so a box dragged by its grip keeps the
  height it was dragged to; nothing may write an inline height on `#prompt`
  for any other reason. A second query, for wide windows under 1150 px tall,
  sets compact control heights of its own (a field 40 px, the primary 44, the
  small and tiny buttons 28 and 26), today's panel padding and topbar, a 12 px
  gap and an eight-line prompt cap. In the wide tier the tiny buttons are
  30 px at a caption's type, tick boxes 18 px, a tag chip's × and a
  reference chip's × 24 px targets, reference chips up to 96 px badged
  "image N", and a card's and a long run's tile tools and a result's row
  `--h-tool` (32 px). Nothing changes under 1800 px, where the
  smokes run. A component's wide-tier rule goes in its own section, inside an
  `@media (min-width: 1800px)` block, and new CSS uses the tokens. Contrast
  and focus are not scale and apply at every width: `--line-strong`
  (#666680), `--accent-strong` (#6a4cf0), placeholders in `--muted`, and the
  page's own `:focus-visible` ring (2 px of `--accent-2`), which ships with
  `color-scheme: dark` because that turns the UA ring near black; a control
  with a focus style of its own outranks the ring. A disabled box is dimmed
  in the panel's field rows only (`.panel .sub > input:disabled`, `select`):
  under 901 px the prompt check sits in the panel, and its comparison's
  "original" box is disabled because it always runs.
- A tip lets clicks through (`.tip { pointer-events: none }`) and keeps its
  line breaks only where the code writing it sets `.tip.pre` (`applyHints`,
  `refreshLoras`), never because it is `.tip.wide`. Every `.help` badge has a
  static `aria-label` naming what it explains ("Steps help", "Steps axis
  help", "Grid seeds help", the dialog's "Refine size help") and
  `aria-describedby` pointing at its own tip, keeps role=note and
  tabindex=0, and stays outside its field's `<label>`, where a press ticked
  the box (Resume's). Red on hover is the `.ghost.danger` class (Cancel, a
  queue row's ×, Cancel queued, Delete unstarred, Reset), never every small
  ghost button.
  A button whose words are its state (Pause / Pausing… / Resume) has no
  `aria-pressed`. A toggle with `aria-pressed` keeps one name in every
  state - "Star" for every star (the lightbox's words are "☆ Star" / "★ Star"
  with `.active`), "Reference image" for a card's ❐ - and its state is
  `aria-pressed`, the glyph and, in words, its title (`starTitle(on, button)`
  for every star; ❐'s "Reference image N - click to remove" or its reason,
  also its `aria-description`). A button held for a reason (Generate, Run
  grid, Unload, Pause, Refine, Ask the critic, a card's ⤢ and ❐) is
  `aria-disabled` with the reason as its title, never `disabled`, so it keeps
  focus and a press can say why. The prompt check's dead Use, Apply and Run
  comparison (stale, or fewer than two versions), the Refine dialog's check
  buttons (Show, Apply) and its scale buttons past the guard are `disabled`
  with the reason as their title - the documented exceptions (below): each
  dies with a state its own panel says in words beside it.
- Each of the panel's size strips (`#presets`, `#tiers`) is a `role="group"`
  with one tab stop (the chosen preset, else the first) and `aria-pressed`;
  Left and Right, Home and End move focus
  along `strip.querySelectorAll('button')` without choosing (`stepPreset`),
  and `syncSize` resets the stop. A strip whose every preset is in range
  carries no marks (`all-guided`). While the negative box is greyed and the
  checkpoint takes a negative, `#neg-enable` ("Set True CFG" or "Set CFG"),
  after the note in the field's caption, focuses and selects the deciding box
  and writes nothing (`focusNegativeKnob`); it never goes over the box, whose
  text is what a greyed negative shows, and the greyed box keeps the pointer,
  so a carried-over negative can still be scrolled. Where the checkpoint
  takes no negative prompt at all (klein) the field is its caption alone
  (`.unsupported` hides the box; its text is kept, and the note says so).
  `#negative-field` is a div with `<label for="negative">`: a button inside a
  label becomes the label's control.
- The size line compares the request against `guide.pixels` a tenth either
  way and quotes `pixels_word`; the Examples menu is built on open from the
  last status. A pick writes through `writeBox` (tags.js) and tells the
  preview and the panel the way `choose` does, and over a prompt of the
  user's own offers an Undo (`offerPromptUndo`): an overlay on two columns,
  in the panel's flow under 901 px and while the negative box is greyed. One
  Undo is on offer at a time: an Example's and a Reuse's retire each other.
  The menu is a role=menu whose items stay the clickable `<li>` children:
  ArrowDown, Enter or Space on the button opens it on its first item,
  arrows, Home and End walk it, Escape hands focus back to the button, and
  focus leaving closes it. Where references are supported and none is
  attached, an example matching `EDIT_PROMPT` (app.js, held equal to
  `prompt_rules.EDIT_PROMPT` by `tests/test_prompt_rules.py`) says it needs a
  reference image.
- Prompt expansion is ERNIE-only and explicit: its result replaces the editor
  only if the user did not type while the long request was in flight, then the
  usual persistence and grid preview paths are told about the programmatic edit.
  The critic holds what Expand holds (Generate and Run grid aria-disabled,
  Expand, Unload) from the click on, including the check it first runs on a
  changed prompt, and `startJob` refuses while it runs. Check and Check again
  are disabled during that check and re-enabled in a `finally`; if that check
  is replaced anyway the page says nothing was sent, if it is cleared (a
  Reset, another checkpoint) it says nothing, and Close pressed meanwhile
  focuses the prompt, not the disabled Check. Ask the critic is
  aria-disabled, never disabled, so it keeps focus. An answer takes focus
  (`#critic-result`, tabindex=-1) while focus is still on the button; so does
  a failed critique's row (`critiqueFailure`, never a `critiqueResult`), which
  reads "The critic did not answer: <the server's reason>." with nothing
  generic added; `#error` gets that sentence only where the critic's block is
  not shown. The panel's own messages (words moved, an answer set aside or
  never asked) go to `#analysis-status` through `analysisMessage`, which falls
  back to `showError` where the critic's section is hidden, and stays at full
  strength while the rest of a stale panel is dimmed; refusals stay in
  `#error`. Check pressed on a two-column page moves focus to the panel
  heading (`checkFromButton`), and Check again hands focus to the heading
  when it hides; under 901 px focus stays on Check.
- The prompt check describes one text: the prompt and negative prompt as
  `collect()` sent them, whether CFG or True CFG applies that negative, and
  the reference images it counted
  (`analysisResult.reference`; klein's references rule reads them). Any
  change to either box - typing, every programmatic write through
  `promptReplaced()`, `syncCfg()` greying the negative or changing its
  activation while a grid keeps the editor enabled - or to the references
  (`setReferences`, Reuse, a `reference_supported` flip) marks it stale:
  dimmed (opacity 0.8 with `--muted` raised to `--text-3`, one rule shared
  with the Refine dialog's `.rf-check`), every Use/Apply and Run comparison
  dead, a finding's words selected only while still where the check found
  them, and scrolled into their box (`scrollToSelection`); the badge says
  "references changed since" when only the references moved. A dead Use or
  Apply gives the reason in its title but stays `disabled`, not
  `aria-disabled` - the documented exception - since they die as a set while
  the stale badge and Check again, at full strength in the head, say why. A
  checkpoint change clears it. Raw negative text and effective activation
  are captured before Check and the critic await replies; inactive raw edits
  still make it stale. Apply is offered on rule findings only; an encoder
  suggestion is shown as text. Use and Apply write through `replaceBoxText`
  (so Ctrl+Z restores) and check again; every other whole-box write of the
  prompt or negative - the tag tools, Examples, Reuse's `fillPanel` and its
  Undo - goes through tags.js's `writeBox` (`replaceBoxText`, then no
  suggestion list or typing state over the new text), never `.value`, and
  Reuse's through `rewriteBox`, which focuses the box with `preventScroll`
  first. Use rewrite and Use reordered (the critic's and Check's own) first
  put the prompt they replace into an empty Your version (`useVersion`), with
  a note; the order finding whose suggestion equals `reordered` says "Same as
  Use reordered above." instead of Apply. Server offsets count code points;
  the page converts them (`units()`) before selecting. Rewrites show
  character counts and an expandable exact diff against the checked source
  (`promptChanges`), using text nodes with additions/removals marked, and,
  outside the closed diff, a gist of its first three changes grouped as a
  reader names them (`changeHunks`, `gistItems`: a word replaced is one
  change, edits only spaces and marks keep apart are one, words put elsewhere
  are a move); no rewrite says so, with the server's set-aside explanations.
  The rewrite's findings are marked "also in your prompt" or "new", followed
  by "No longer flagged: …"; a prompt finding that rests on the tag index
  (`readsTagIndex`: the order, the style, an artist's @), which the rewrite's
  lint never asks, is never called fixed but listed as possibly still
  applying. The comparison has its own Batch
  select (1, 2 or 4; default 1) and Seeds (4 or 8 total per version):
  `num_images` is its batch size and grid `seeds` is Seeds divided by Batch,
  keeping the image count versions times Seeds. Its preview states the total
  seeds per version and batch size. The main panel's Batch and Repeat do not
  apply; its Seed still does, and a blank one stays blank: the comparison is
  not the panel's grid (`startJob`'s `panel: false`), so its base seed is
  never written back for a Resume it does not take. The mechanics are a ? tip after the
  comparison's Batch select; the 4-vs-8 caveat stays visible. Preview and Run
  read both comparison selects directly (they and Your version are
  `autocomplete=off`). It takes Check's `reordered` as its Reordered row, else
  the critic's as the server kept it (`critic.validate` decides; the page
  adds no rule of its own), never writes `els.prompt` or `els.gridPrompt`,
  drops a version identical to another, and opens its submission
  (`job:<id>`) as the sheet when done - Starred unticked first, and the
  sheet's note says so ("Starred unticked: nothing in a new comparison is
  starred yet"; `showJob(jobId, { fresh: true })`) - with the page scrolled
  to it (`showSheetScope`); its Your version box (`yours`) is the one way to
  test order on prose. A negative-prompt Apply is dead while the negative box
  is greyed.
- Tag autocomplete and comma-item validation use the fixed Danbooru
  tag API through `backend/tags.py`, filtered when a checkpoint documents a
  knowledge cutoff. WAI v17 publishes none: keep its cutoff `None` and the
  live lookup advisory rather than inventing a date. Model-card
  special tags validate locally; lookup failure stays advisory and never
  blocks. The page selects the first suggestion only from three typed
  characters it starts with (Tab otherwise moves on). Validation asks about
  the first 64 items (`TAG_LOOKUP_CAP`, the request's ceiling) and marks the
  rest unchecked; verdicts are cached per checkpoint, and the tray's state is
  pending, checked or offline (amber, never red) in `data-state`. The Refine
  dialog lists the comma items `/api/tags/validate` answers `valid: false`
  for under its Prompt box, each once, only where `tagVocabularyEnabled`,
  silent when the lookup is unavailable; it has no chips and never calls
  `writePromptTags`. On a two-column page validation lives above the
  results, not in the panel; under 901 px the same node returns below Prompt.
  The prompt check's panel follows it, in that order (`placeTagValidation`
  moves both, keeping focus and the selection: `moveBefore` where the
  browser has it, restored by hand elsewhere). From 1800 px the stylesheet
  docks the open check beside the results instead: a `:has()` grid on
  `.stage` (`--dock-w`) puts the chips, the viewer and its note in the left
  column and the check in the right, CSS only - the DOM order and
  `placeTagValidation` are unchanged, and without `:has()` the check stays
  above the results. The check takes the left column's height rather than
  setting it (`contain: size`, stretched, at most the window under the
  topbar), and the viewer holding something is then at least `min(36rem,
  that window)`; idle, it stays its one line and the gallery comes up beside
  the content-sized check (the results viewer below). The check scrolls
  inside itself, its head sticky at the scroller's very top (`top: 0`, the
  column's top padding in the head's own, so nothing scrolls past above
  it) with Close absolute in its top right corner, never wrapped onto a line
  of its own, and `focusAnalysisHeading` resets that scroll. Never `position: sticky` or a
  min-height on the column itself. With no chips the image does not move; on
  a tag checkpoint it moves by exactly the lines the chips wrap onto in the
  narrower column (where `#recent` docks beside it they span both columns:
  the results viewer below).
- Validation chips edit the prompt: drag (or Alt+Left/Right) reorders, × or
  Delete removes, and a click or right-click in a gap inserts there (a
  chip's right-click or Menu key inserts after itself, + tag at the end) with
  autocomplete; a plain click and the hover marker count only within a row
  of chips (`tagInsertionIndexAtPoint(..., { gap: true })`, null on the label
  and the hint), a right-click takes the nearest place; the marker is drawn
  again at the end of every tray render while the pointer rests in a gap
  (`tagHoverPoint`, not under an open editor), since a lookup or a count
  landing rebuilt the chips without it. All mutations must go
  through `writePromptTags`, which writes through `writeBox` and hands focus
  back to the chip the edit leaves in place (`tagFocusIndex`); a chip's write
  visits the prompt in passing - focused with `preventScroll` and exempt from
  `revealInPanel` through `panelPos.press` - so neither the panel nor the
  page moves. The tray is one roving tab stop and no live region
  (`#tag-summary` is, not counting the item being typed); its children are
  only chips, the editor and + tag - the label, state and token count live in
  `data-*` attributes its `::before` reads, and an unknown chip's reason in
  the shared `#tag-tip`. A focused chip's ring is its own `:focus-visible`
  outline; no state rule may use the chip's outline (a repeat is marked on
  its text). The Tags helper drops under the prompt box (`#tags-menu` inside
  `.prompt-editor`, capped by the window alone), a click in the box moves its
  caret without closing it, and it names where the next tag goes; it inserts
  at the caret and never slots tags itself. An Escape the suggestion list or
  a chip editor took closes that alone (the document's Escape skips
  `defaultPrevented`); the next closes the helper.
- Ctrl+Enter is routed by focus (app.js's keydown): in the Refine dialog it
  runs the pass, and so does a plain Enter in the dialog's one-line boxes
  (`refineKeydown`); from a box of the open grid fold it clicks Run grid;
  anywhere else, a focused image included, it generates - `thumb()`'s Enter
  and Space open the lightbox only without Ctrl, Meta or Alt. Generate
  itself refuses in the page alone (never the server, where braces can be
  text or {tag} emphasis) while the fold is open, its variables box has text
  and either prompt holds a `{name}`. Generate, Run grid and Refine show the
  hint through one `::after` rule (empty alt text, smaller and lighter, never
  opacity, since `setBusy` rewrites the buttons' text) plus
  `aria-keyshortcuts`.
- The grid preview is the server's expansion; the page adds only
  presentation and advice. It trims factors of one other than the seeds
  from the totals and writes ×, since `grid.describe` keeps its ASCII x and
  its ones for the CLI's log. It lists the resolved prompt whenever the
  variables box has text, one included. Its client-side warning lines (a
  variable of one value, a budget that completes no seed, the panel's
  Repeat above one, cells already in the gallery) never refuse anything, and
  are appended to `#grid-preview` as nodes, never by rewriting `textContent`,
  since `.totals` is its first child. The budget line needs the preview's
  estimate, is left out under Resume, and names what else spends a budget
  that runs from submission: the load while the topbar's dot is not ready,
  the jobs ahead while the page watches one. The preview is asked again by
  `setBusy` (a job's start and end) and by `refreshStatus` when the model
  becomes ready or stops being so, not only by the boxes; and the first
  preview after the fold is opened by hand reveals Run grid again
  (`gridRevealPending`, set by the summary's click), since its reply lands
  under the pointer where the panel holds still. Axis messages lead with the
  box's name (`Steps`, `CFG`, `Mu`, `LoRA weight`, `Strength`, `Seed list`);
  `parse_axis` closes up a spaced range between two numbers, leaving a lone
  `-` as Mu's none, and refuses a stepless range that would stop at its
  first value (`0.8-1.2` needs a step).
- A refine pass is the Refine dialog's (`web/refine.js`), never the
  panel's: `collect()` sends `refine: null`, so Generate, Batch, Repeat, the
  grid and the comparison always render from the prompt, and the grid has no
  strength box. `openRefine(rec, pass)` - a card's ⤢, a viewer result's ⤢ and
  its column's Refine…, the lightbox's Refine (which closes the lightbox; a
  close that runs nothing gives it back on the same image of the same strip
  at the same view and comparison - `refineReturn` from `lightboxPlace()`,
  resolved by id in the current gallery or viewer strip, the comparison put
  back before the view, whose scale is relative to the fit of half the stage
  side by side - unless the passes strip has opened it on a pass already)
  and `reuse` on a refined record, which first asks `GET /api/images/{id}`
  for its source - fills the dialog from the record: prompt, negative, steps,
  CFG, Mu, True CFG and an adapter the list still offers, or this
  checkpoint's profile where the record is another checkpoint's (the fold
  opened, saying so); the seed; the remembered scale or 1.5x where that fits
  the pass's guard and is not past the tight line (`refineTight`: the
  profile's `refine_tight_pixels`, else `sizeIsTight`), else the largest
  factor that is, else 1x - a remembered scale passed over says so on its
  button, and a run at the size it opened at in place of the scale asked for
  leaves the remembered scale alone (`refineCapped`) - unless settings a
  close left unsent on the same image and checkpoint are restored
  (`unsentRefine`, page memory only, never `local-diffusion.refine.v1`),
  which it says, with Start over; a run, another image, Reset and a
  checkpoint change forget them. It opens focused on Strength, its value
  selected. `refineSource` is that record while the dialog is open and null
  once it closes. A card's or viewer result's ⤢ on a checkpoint without a
  pass is aria-disabled with the reason (`REFINE_NONE`, as the lightbox's
  Refine), and a press says so in `#gallery-error` and for two seconds over
  the card's tools (`refuseInCard`), opening nothing - decided in `card()`
  and `refineResult`, not in `openRefine`, whose own refusal (the lightbox's)
  is `showError(REFINE_NONE, true)`, mirrored into the lightbox; a refined
  record's Reuse opens it anyway, Refine aria-disabled with the reason, until
  status says the pipeline has one. A source gone at Reuse fills the panel
  instead, its size unchanged, with a note. Versions is `num_images` on
  consecutive seeds; a Strength holding a list or a range is a `/api/grid`
  strength axis with Versions as `seeds`, opened as its sheet. A pass starts
  through `startJob(path, body, { sheet, panel: false, slot: els.refineError,
  placeholder, onAccepted })`, which never writes the panel's Seed box; the
  dialog closes in `onAccepted`, so a refusal at submit (a source deleted
  elsewhere) stays in `#refine-error` with the dialog's edits, and the
  placeholder is the source's full image (`url`). While the POST is out the
  dialog holds (`refineSending`): Refine reads "Sending…", and `closeRefine`
  (Cancel, ✕), the dialog's prevented `cancel` (Escape) and the passes strip
  wait for the answer; a close the browser will not hold counts as a run in
  `refineClosed`, and a refusal after it is said in `#error`. A pass run from
  a dialog the lightbox opened puts focus on the source's gallery card, else
  Generate, with `preventScroll`; any other run whose close leaves focus on
  `<body>` (an open from script, Reuse of a pass) puts it on Generate, or New
  image ↑ where that is shown, on the next frame. Every change asks `/api/grid/preview` with
  the grid-shaped request; its refusal, a size past the guard or no
  capability makes Refine aria-disabled with the reason as its title, and a
  press says it in the dialog. Every way of running a pass first waits for
  the preview of what the boxes hold (`refinePreviewed`, the request the last
  preview answered), since the debounced preview is what refuses a strength
  past 1, one that leaves no step and a source deleted meanwhile; a refused
  press moves the preview's sentence to the error line in the preview's
  place and at its size (the preview hidden until the next change) and
  focuses the field it is about, and the dialog's column never shrinks its
  items. Its Strength help is the profile's: the placeholder, Compare 3
  (`compareStrengths`) and the tip's example (`model_defaults.strength_sweep`)
  are the same three values - half, the profile's strength, half again on
  the 0.05 grid, less any that leaves no step - one arithmetic in the same
  order in both, held by `tests/test_profiles.py`; Compare 3 only fills the
  box. Where Steps leaves fewer than two of them a step (a source made at 2
  steps), Compare 3 is aria-disabled, its title naming the Steps that would
  give a sweep (`compareStepsNeeded`), a press says so and writes nothing,
  the placeholder is the start alone and the sweep line names no values.
  From 1600 px the dialog is two panes (the source's PNG, the kept
  region of a reshaped size framed with `refine.fitted`'s arithmetic,
  `refineKept`), with the same DOM and tab order as the one column; Keep
  shape moves the other side on trusted input only. The dialog remembers the
  scale, the versions and, per checkpoint, a single strength the user ran
  (`local-diffusion.refine.v1`); `resetPanel` forgets it.
- `startJob(path, body, { sheet, panel, slot, placeholder, onAccepted })`:
  `slot` (default `els.error`) is where its refusals go; `onAccepted(jobId)`
  runs once the server has taken the job, before it is watched, and a
  refusal never calls it; `placeholder` ({src, width, height, label}) is
  keyed to the job (`jobPlaceholders`) and shown from its first progress
  event with no frame as the viewer's `figure.pending` (the source dimmed and
  blurred, in the result's own tile, its caption the tile's `p.foot`) until a
  frame or the result replaces it. A stream that gives up probes
  `GET /api/jobs/<id>`: alive, it streams again (three times at most); no
  answer, an offline message counting the saved images; a 404, "The server
  restarted, and this job ended with it after N of M images". The stale
  preview is dropped, and where nothing of the job reached the strip the
  viewer holds `showLostRun`'s card (the sentence, Run it again where this
  page sent it, `lostRunBack` rewording it once the server answers). While
  it reconnects the job line reads `LOST_PHRASE` (the topbar's "server not
  answering — retrying") with the count, no steps or time left, no topbar
  mirror (`streamLost`), the title and the strip's note the same phrase. The
  progress block is a one-line Now row
  (`#progress-now`, count · size · what - "Now: 3 images · 1024×1024 · a fox
  in the snow" (`describeJob`), the prompt last so the ellipsis takes it,
  whole in its title) with Pause and Cancel at its end, the bars at the
  panel's width, and the live cell line (`#cell-note`) inside `#progress`;
  queue rows read the same, and a single queued job is one "Next:" row. A
  finished job's outcome and actions (Use seed, Show sheet, Retry/Resume) are
  the finish line, `#finish-note`, outside `#progress` (`setFinish`): only
  the next job's end replaces it, never a queued job's start; it is led by
  "Previous job:" while another runs or after a lost stream (`finishStale`).
  A job's outcome in `#error` - cancelled, its budget, an error, skipped
  cells (`showOutcome`, `jobOutcome`) - lasts as the finish line does: led by
  "Previous job: " while another runs (`#error.previous`, a generated
  `::before`; `showError` drops the class) and cleared when the next job's
  end is drawn (`finishJob`). A press on a held Pause or Unload says why
  until the job ends (`showJobNote`, cleared by `setBusy(false)`). A job that
  ends with focus in `#progress` or `#queue-note` hands it to Generate and
  the finish line takes it on to its first button (`finishTakesFocus`);
  replaced with focus inside it, the finish line hands focus to its new first
  button, else Generate. A finished job's images are prepended to the
  gallery (`prependFinished`: same filters, `made` + PAGE, chained after
  `galleryPending`, dropped on an epoch change, reset where it does not join
  up or the gallery is empty, never in sheet mode), never `resetGallery`,
  drawing the day's lines as `loadGallery` does - over the first new card
  where its day is not today's, between two new cards, and between the last
  of them and the old head read off the grid's first card (`data-made`, a
  tombstone's too; none where the grid opens with a line) - and giving the
  grid's tab stop to the newest image where it was the default
  (`galleryChosen` false: set by focus arriving, cleared by `resetGallery`)
  or the old head with focus outside the grid, else `keepCurrentCard()`, as
  anything adding cards must. "New image ↑" (`#btn-new-image`) appears when a
  job's first image or frame lands while the viewer is out of sight; its
  press scrolls the viewer into view and focuses the newest image, and the
  page never scrolls by itself. `setBusy(true)` hides the last-run banner
  without `remember()`; the done branch remembers the job. The tab's title
  and the aria-hidden topbar line (`#job-mirror`, `#topbar-line`) mirror the
  job, and the finished title is held until the tab is looked at again.
- `api()` turns fetch's own TypeError (never an HTTP error or an abort) into
  an Error with `offline: true` and `OFFLINE_MESSAGE`; HTTP errors carry
  `.status`. `showError(message, note, { slot, mirror, offline })` records
  the slots holding such a message (`offline` defaults to whether the text
  contains `OFFLINE_MESSAGE`), and `clearOffline()` empties them on the first
  answered status poll. A poll that gets an offline error calls `goOffline`:
  dot error, "server not answering — retrying", Load hidden, Generate and Run
  grid aria-disabled with the reason (`syncSubmitButtons`, their one writer,
  which has the Refine dialog's Refine follow through `syncRefineRun`),
  Unload held. A changed `started_at` writes the restart note into an empty
  `#error`. The lightbox's picture is one such slot: `els.lbImg.onerror` asks
  for the record and, where the server is not answering, writes
  `OFFLINE_MESSAGE` into `#lb-error` as an offline message, which the poll
  that answers clears before asking for the picture again
  (`retryLightboxImage`); otherwise "#N is no longer in the gallery." (an
  unknown image), "The file of #N could not be loaded." (the record exists)
  or "This image could not be loaded." (any other refusal), cleared by the
  next picture that loads. `syncUnload` is Unload's one writer (busy, Expand,
  the critic, a load, offline): `aria-disabled` with the reason as its title,
  never `disabled`, and a press says why. Unload is hidden while Load is
  shown, and the press that hides one of the two hands focus to the other. A
  failed load is a card in the viewer's empty state (`syncLoadFailure`:
  heading, detail, model page link, a primary Retry load calling
  `loadModel`); a card that goes with focus in it hands focus to the new
  card's Retry load where one replaces it, else to Generate.
  Where the viewer holds images the detail goes to `#error`, once per
  detail, with `mirror: false`, and again after a job that ended at the load
  (`finishJob` forgets `shownLoadDetail`, since the press that queued it
  cleared the line); that job's finish line says "Not made: the model could
  not load" and where the reason is. The card is centred in the viewer. The topbar's Load reads Retry load meanwhile, and the
  status text keeps the checkpoint's short name in every state. While a job
  waits on the load, a client ticker writes "loading weights" with an
  `aria-hidden` clock span ("· m:ss") in `#progress-text` over an
  indeterminate bar; the stream's error handler stops it.
- `showError(message, note, { slot, mirror, offline })` says each message in
  one slot (both are role=alert). What a gallery control did - Delete
  unstarred, a failed gallery page, Re-render's submit errors, a card's ❐,
  ⊞ and ⤢ refusals, deletes, the sheet's Resume this run refusals and its
  export failure - is said in `#gallery-error` (`els.galleryError`), which
  `resetGallery` clears; Reuse's notes stay in the panel's `#error`.
  `#gallery-error` is the gallery head's last row (a full-width flex item
  inside `.gallery-head`), and its line is kept while it is empty
  (`#gallery-error[hidden]` is `visibility: hidden`, not `display: none`), so
  a message never moves the grid under the pointer; only one that wraps
  grows it.
- Profile defaults apply per control (`owned`); an empty Steps or CFG box is
  sent absent and 0 is a valid CFG. The panel persists to `localStorage`
  with every access in try/catch; `resetPanel` is the one thing that forgets
  it, since clearing the gallery never reaches the browser. Reset first keeps
  an in-memory snapshot read off the page (`panelState()`, the raw sampling
  values, `owned`, the size preset, `REFINE_KEY`'s value and
  `unsentRefine`) and says "Panel reset." with an Undo (`offerResetUndo`),
  retired by the first panel change (`syncResetUndo` from `persistPanel`) or
  after 20 s; the Undo writes the boxes through `rewriteBox`, restores the
  rest and runs the sync functions again. Resume is never restored, and
  nothing of it survives a reload. Before it fills, Reuse keeps
  `panelState()` (what `persistPanel` stores), the raw steps/CFG/mu/true-CFG
  values and `owned`. Its note in the panel's `#error` opens "Settings from
  #N loaded" (" - your prompt was replaced" where it was) with an Undo right
  after that sentence, which restores them; the rest follows: the checkpoint
  mismatch, references on a checkpoint that takes none, an adapter with
  `/api/loras`' reason. The Undo is offered only while the panel holds what
  the Reuse left (`reuseNote.after`; `syncReuseUndo` from `persistPanel` and
  `resetPanel`): the first change removes the button and keeps the text, and
  a reference dropped from what it left (deleted, or "unknown image") is no
  change. It drops a restored reference only on the server's "unknown
  image". Where the page goes is `revealReused`'s: the panel is placed and
  moved (`placePanel`, `revealInPanel`), and the page scrolls only where the
  panel cannot move (one column, the page's end), and only as far as the
  panel's resting place. A stuck panel is then lifted to bring the Undo into
  the window only as far as the prompt's field stays clear of the topbar
  (`showUndo`); at 1400x900 the prompt wins. The lightbox's Reuse closes the
  lightbox first. `syncReuseButton(button, rec, compact)` takes the record
  and keeps its checkpoint, refine source, references and adapter in data
  attributes for `syncReuseButtons`. A refined record's button reads "Reuse
  pass…" (the card keeps ↻/⚠), titled "Open the Refine dialog on #N with this
  pass's size, strength and seed"; a pass whose source is gone
  (`refine_missing`, kept in `data-reuse-pass-gone`) reads "Reuse settings",
  titled why the pass cannot be made again, since `reusePass` fills the panel
  naming the source "#N". A ⚠ on another checkpoint's record adds which
  references or adapter will not carry over (`notCarried`, `fillPanel`'s own
  test). That verdict reads the status, so status.js runs
  `syncReuseButtons` where the checkpoint changes, where
  `reference_supported` or `lora_supported` flips, and at the end of
  `refreshLoras`.
- The results viewer (panel.js). `clearResults(job)` runs at a job's first
  frame or image: it moves what the strip held into `sessionRecords` (newest
  first, not the same job's) and sets the tier from the job's
  `planned_images` (`viewerJob`), never the running count; focus inside the
  strip it empties goes to Generate. One image is shown
  at 68vh; two to four are best-fit (`fitViewer`: `--cols`, `--rows`,
  `--tile-h`, `.fitted`, `grid-template-rows` reserving every row the job
  will fill from its first frame), refit on the next frame from a
  ResizeObserver and on the window's resize; more than four are small tiles,
  whose tools are out of the tab order, and from 2000 px those are
  `auto-fill minmax(--strip-min, 1fr)` under `min(560px, 45vh)` with their
  own scroll and `scrollbar-gutter: stable both-edges`, no 380 px floor, and
  every row the job will fill reserved from its first tile (`reserveStrip`:
  an inline min-height from that tile and the strip's column count, up to
  the cap, never under the stylesheet's own minimum, recomputed when the
  strip's width moves, let go by `clearResults`). Every tile appended
  goes through `stripAppend`, the preview and the stand-in included, so a
  strip at its end stays there, and `settleStrip` lets the strip grow by an
  overflow no bigger than its bottom padding. Every tile is a `viewerTile`
  with `--w`/`--h`, `--tile-h` and a foot of `--h-tool` (`fitViewer` reads
  the same token): a result, the live
  preview `figure.live`, and a pass's stand-in `figure.pending`. A result is
  `figure.result[data-id]`, never `.card`, holding `thumb()`'s image and the
  card's tools: `reuseButton`, ⤢ (`refineResult`), ❐ (`toggleReference`,
  hidden where the checkpoint takes no references), a `.star` via
  `setFavorite`, and ⌦ via `deleteImage`; no ⊞. Alone in the strip, a
  result carries `.result-side` from 1800 px: the lineage chip
  (`passLineage`), Star / Refine… / Reuse / ❐ Reference / Use seed and
  Delete (`.ghost.danger`, at the row's end) in words, the prompt and
  `factsList(rec)`; its tile's tools row stays, empty, at its height. With
  the check docked the side is a row under the picture in the tools row's
  place, at `--h-tool`: its actions and `.result-brief` (seed, size, steps,
  CFG; `settingsLine` in its title) - one set of words for the image just
  made either way, and neither the picture nor the gallery moves as the
  check opens.
  `#viewer-note` names a pass's source ("Refined from #N at s · W×H",
  `sourceLink` → `openRefineSource`). `#recent` (from 2000 px, single tier,
  in the viewer's row only) shows up to six `sessionRecords` and opens the
  lightbox on [current, ...sessionRecords]; with the check docked, `fitRecent`
  keeps it between the image and the check in the width the image leaves at
  its own size (`.recent-docked`, `--recent-room`), hides it under
  `--recent-min`, and a tag checkpoint's chips then span both columns. The
  docked test is written twice - the CSS `:has()` and `fitRecent`'s
  `matchMedia('(min-width: 2000px)')` with the check open in
  `#tag-validation-wide` - and a change to the dock's selector changes both.
  A single viewer image opens the lightbox on `galleryRecords` from its copy
  there, else on [rec, ...sessionRecords]; `openLightbox(rec, list, origin)`'s
  `origin` of "viewer" or "recent" is where `dismissLightbox` returns focus.
  Idle, the viewer is one line, keyed on `:has(> .placeholder:not(.held))`,
  never on `viewerRecords`; `.empty-install` (`syncEmptyInstall`, from an
  unfiltered gallery page with total 0 and no `other_models`) keeps the tall
  box with an Examples button. Idle with the check docked, the viewer stays
  that line, the gallery comes up in the first column beside the check, and
  the check is as tall as what it holds up to the window (`contain: none`,
  spanning the rows down to the gallery's 1fr); the docked 36rem minimum is
  only for a viewer holding something. A gallery with nothing to show says
  so under its head (`syncGalleryEmpty`, `#gallery-empty`: what fills it and
  the authors' Examples, or that nothing matches the filter, and what All
  models would add), never over tombstones or in sheet mode. A finished
  job's images the gallery's filter leaves out - a sheet's `job:` scope, a
  search, Starred - are counted in the gallery's line with Show all
  (`noteOutsideView`, per filter; `showAllImages` empties the search,
  unticks Starred and closes the sheet). The strip's run is `.card.this-run` in the
  gallery. When ⌦ deletes the strip's last image, focus moves to the held
  placeholder, never `<body>`. `forgetImages` splices `sessionRecords` and
  re-renders `#recent`, retitles every `img[data-id]` of a pass whose source
  it deleted (`cardTitle`) and calls `viewerSourcesGone` (the note, the
  lineage and the facts read "#N (deleted)"); `recordsWithId` includes
  `sessionRecords`, and `syncCardTools` also syncs the viewer's
  `.refine-tool`s and `.ref-tool`s.
- The lightbox (`web/lightbox.js`). `lbView` is a scale relative to fit,
  never below `min(1, oneToOneScale)` nor above 8x device pixels
  (`clampScale`), carried across records whose aspect agrees within 1%
  (`lbView.aspect`, a tolerance, never a rounded key) and clamped to the new
  record's limits. The zoom line reads "1:1", "fit · N%" or "N%", tested in
  that order; a double-click returns to fit from any other scale, and a
  backdrop click below fit waits out a double-click (`lbCloseTimer`). F, S,
  Delete and the zoom keys skip a key with Ctrl, Meta or Alt held (the
  browser's find and zoom), and a key a lightbox control already handled
  (`defaultPrevented`: the swipe divider's arrows) is not walked. On a record
  with `refine`, S or Compare with source shows the source under the pass
  left of a divider (swipe), or side by side with each half fitted inside
  the arrows' circles (`lbPorts`, `LB_NAV_ROOM`), on one `lbView`;
  switching keeps the magnification. The
  source is fetched by id once a session (`lbSources`) and drawn with
  `refineKept` and a clip-path. A step to another record ends the
  comparison; a re-render of the same record does not. A deleted source
  leaves the controls aria-disabled with the reason ("source #N deleted"). S
  pressed on the focused divider hands the focus to the toggle.
  `factsList(rec)` (`dl.facts`) is the one facts renderer, the viewer's
  column's too; a pass whose source has gone reads "Refined from #N
  (deleted) @ s". Its References row numbers the references as the panel's
  chips do. Use these references asks for every id again as Reuse does,
  leaves out and names the ones the gallery no longer has, and is
  aria-disabled with the reason where the checkpoint takes none
  (`REFERENCES_NONE`) or every one has gone; that state is re-synced on every
  render (`syncUseReferences`). "Refined from #N" opens [source, pass]; Run
  does what the card's ⊞ does. `settingsLine` stays the dialog's hidden
  description (`#lb-desc`) and the delete confirm's line, every image named
  `#N` ("refine of #N (deleted)", "ref #a+#b"). The counter reads
  "i/total · #id" with the gallery's server total. The star follows the
  toggle rule, and `setFavorite` re-renders it (`renderLightbox`) rather than
  writing it. The user's closes (Escape, ✕, the backdrop, a stage click) go
  through `dismissLightbox`, which focuses the image the walk ended on or,
  where no strip holds it (another checkpoint's scope, a delete elsewhere),
  the grid's current card, else its first, else the search box; every
  programmatic close is `closeLightbox` alone. The arrows are strips only at
  fit, with backdrop beside the image (`--lb-gutter`, 48-160 px, 20%-80% of
  the height); zoomed in, below fit, disabled or with no backdrop they are
  their circles. The wheel listener is `.lb-main`'s, so a strip zooms too. A
  stage press within 500 ms of an arrow press (`lbAfterArrow`) neither closes
  nor zooms: it is the rest of a double-click on the arrow. The arrows'
  clicks are never filtered by click count. The keys line sits outside
  `#lb-meta` and is hidden under 901 px. lightbox.js declares only; its
  handlers (`lbImageFailed`, `lbImageLoaded`, `lbSourceFailed`, `lbWheel`,
  `lbArrowDown`, `lbPointerDown/Move/Up/Cancel`, `lbDoubleClick`,
  `lbSplitKey`) and the dialog's `close` → `closeLightbox` are wired in
  app.js.
- The gallery's cards. The tools are ↻ ⤢ ❐ ⊞ ⌦. Each button's textContent is
  its glyph (the smokes find them by it), and its short name is in
  `data-tip`, which a `::after` shows on hover and keyboard focus at the tool
  row's left edge; Delete's sits at the row's right edge, over Delete. The
  tools shrink to fit (`flex: 0 1 auto; min-width: 0`), with Delete set apart
  (`.del`, `margin-left: auto`) and hovering in `--danger`. ⤢ (`.refine-tool`)
  and ❐ (`.ref-tool`) are aria-disabled with the reason (`REFINE_NONE`,
  `REFERENCES_NONE`, gallery.js) as title and label where the checkpoint
  lacks the capability, set in `card()` and again where status flips it
  (`syncCardTools` from status.js for refine, `syncReferenceMarks` for
  references); ❐ is also aria-disabled on non-reference cards at four. A
  press explains in `#gallery-error` and in an aria-hidden one-line
  `.card-note.refused` holding the gist (`refuseInCard`); the row's tips are
  hidden while a note is up. ❐ follows the toggle rule, and `setReferences`
  stays the one writer. `syncReferenceMarks`, called by `renderReferences`
  (and `markReference` in the sheet's cell builder), rings and numbers every
  card and sheet cell that is a reference (`.ref`, `data-ref-n`, a `.ref-n`
  span); a role=status `.card-note` says "Added as image N" or "Removed". The
  star is a `<button class="star fav-badge" aria-keyshortcuts="F">` in the
  card's corner: always shown on a starred card, otherwise on hover,
  focus-within or a coarse pointer. `syncStar` draws any star button (glyph,
  `aria-pressed`, the fixed "Star" name, `starTitle`); the title names (F)
  only on a button whose `aria-keyshortcuts` is F. `setFavorite` updates every
  `.card`, `.sheet-cell` and `.result` with the id, and the lightbox only
  when it shows that record. A pass is `.card.refined` with `data-strength`
  (the badge is the sheet's `::before` rule). Thumbnails are
  `object-fit: contain`. `thumb(rec, list, src, open)` labels every image
  "Open #N: <prompt>" and titles it with `cardTitle(rec)` (#N · W×H · seed N
  · a pass's source, "(deleted)" after a gone one, then the prompt), on the
  card, the viewer, the sheet and `#recent` alike; `card()` adds only
  `aria-keyshortcuts`, and `open` is what a press does (default: the
  lightbox on `list`). `card()` sets `data-made`, which `tombstone` keeps;
  `loadGallery` draws an `h3.gallery-break` (no data-id) where the local day
  changes from the card before (`cardDay`) and over the first card where its
  day is not today's (`isToday`), and it stays over a day of tombstones. The
  grid is one tab stop: `galleryCurrent` (an id, kept across a gallery reset
  or new query while the record is loaded, not across a page reload) holds
  the image, star and tool stops (`cardStops`). focusin moves it, and a page
  without it makes the first card current. F stars the card from anything
  inside it (`galleryKey`). On a card image, the arrows move by one and to
  the nearest card across in the next row (`rowNeighbour`, from layout, not
  the column count), Home/End jump, and Down or Right past the last loaded
  card pages in. Anything that appends or prepends cards makes them with
  `card()` and calls `keepCurrentCard()` (or `prependFinished`'s tab-stop
  rule) afterwards.
- The sheet's layout contract - axes, prompt labels, pins, variables - lives in
  the client, and the export is client-side, not a route. The sheet asks for
  the whole run in one page. A variable axis, and the Prompt axis, list their
  values by first appearance, not alphabetically: the oldest image carrying
  each value. Under Starred or a search narrower than the run, that is the
  oldest in the whole run, which `learnSheetOrder` asks for once. A skipped
  first cell, a late resume or rows Delete unstarred removed can still move a
  value down. A varying `var:variant` axis takes the rows whenever anything
  else varies, however few of its images are loaded (placed from every
  varying axis, so the listlike rule drops neither it nor the axis across it;
  `VARIANT_AXIS`, held to `grid.VARIANT` by `tests/test_grid.py`), so a
  comparison reads Original, Reordered, Rewrite, Your version, top to bottom:
  display names (`VERSION_NAMES`, under the axis label Version) on the sheet
  and in its export, with the stored labels, the `var:variant` key and
  `stats --by variant` unchanged. A shared run
  or mixed search can give one version label different prompt texts: retain
  Prompt as an axis when the complete variable assignment is ambiguous.
  `stats --job <id> --by variant` reads one comparison; a broader grouping
  with conflicting texts for one assignment is refused with advice to group
  by prompt too. Ordinary multi-variable grids and explicit grouping fields
  still aggregate as requested. An empty sheet draws no grid, only its note.
- Every sheet the page opens (a comparison, a refine sweep, Runs, a card's ⊞,
  both branches of Show run) goes through `showSheetScope`. It keeps the
  search it replaced (`sheetReturn`), which Sheet off restores while the box
  still holds what the page wrote. Once `setSheet` resolves, and only if that
  sheet is still shown, it scrolls the gallery head under the topbar,
  reserving the room under the sheet its head needs to reach the topbar
  (`reserveSheetRoom`: a margin outside the box `fitSheet` measures, from
  where `main` ends - never the document's scroll height, which is at least
  the window - cleared by `setSheet`). A refine sweep's source cell is
  cropped to the passes' shape (`object-fit: cover`, `refine.fitted`'s
  centre crop), captioned "cropped as the passes were" where the shapes
  differ. The Sheet button does not scroll; over an empty search with
  several runs it opens the newest run, with a `sheetNotice`, and a typed
  search is never redirected. A matrix's cells are capped to the window's
  height (`--cell-max` from `fitSheet`'s measured `--sheet-chrome`) with a
  96 px floor, and its heads stick inside `.sheet-scroll`, whose scroll
  padding is the measured head row and row-head column (`--sheet-head`,
  `--sheet-rowhead` on `.sheet`) so a keyboard-focused cell is never under
  them. `refitSheet` (debounced, on the window's resize) measures again and
  re-lays out when the wrap test (`sheetWraps`) flips. A row-less sheet wider
  than fits wraps with captions, and so does its export. Head labels are
  whole on screen (span.label, clamped, title = full value); `clip` is for
  pins, the note and the canvas. Prompt labels that repeat or exceed
  `LABEL_CHARS` fall back to `promptEdit` against the oldest prompt, up to
  `EDIT_LABELS_MAX` prompts, memoised per prompt list; a version that only
  reorders the original's words reads "same words, moved: …"
  (`versionEdit`). A refine sweep's shared source is a leading column
  fetched by id (`sheetSources`) and never a sheet record, axis value or
  count; `forgetImages` nulls it. Records that pass the pins but find their
  cell taken are counted in the note, repeats of the cell's image
  (`sameSettings`: every axis, size, checkpoint, quantization, refined
  source) apart from images that differ in something the sheet does not lay
  out; the note reads "N of M loaded" only when none collide. The shared
  settings go in `#sheet-fixed`, never the note. The starred cell uses the
  card's own `.fav` rules (`.card.fav, .sheet-cell.fav`). F on a focused cell
  stars it through `setFavorite`. "Resume this run" (one run of the running
  checkpoint opened on `run:<id>`, not a comparison or refine sweep) fills
  the run's seed and size and never persists Resume. `resumeRun(runId,
  known)` always reads the base seed from `/api/runs` (the run's oldest
  image); `known` gives only the size, for a page with no sheet. The
  last-run banner appends ", N since deleted" or ", all since deleted" from
  `deleted_count`, hiding Show run when all are gone, and offers Resume
  (`resumeLastRun` → `resumeRun(run_id, { width, height })`) for a grid that
  stopped short - cancelled, its budget, or `stopped_by` failures - with an
  image of its own left; never for a re-render, a refine sweep or a prompt
  comparison (the snapshot's `comparison`). The snapshot's `seed` is a job's
  first remaining cell, past the base after a Resume, and is never used as
  the base.
- `generate`, `clear`, `stats` and `reindex` reconfigure stdout to UTF-8 at
  the top of `main()`: on Windows always, elsewhere when it is not a
  terminal, and never on a stream without `reconfigure` (a StringIO,
  pythonw's None). On Windows a redirect, a pipe or `> NUL` otherwise uses
  the locale's code page, and a CJK or emoji prompt is a UnicodeEncodeError;
  the console is UTF-8 already. A new CLI does the same; `tests/test_cli.py`
  holds them to one setup.
- `generate.py` duplicates `jobs._generate`'s loop deliberately. It applies
  `--model/--quant/--offload` to `os.environ` before importing
  `backend.config`, as `launch.configure()` does for the server; both take the
  flags from `launch.add_model_arguments`.
- The gallery is the running checkpoint's: `GET /api/gallery`, `/api/runs`,
  `GET`/`DELETE /api/gallery/unstarred` and `POST /api/rerender` scope to
  `settings.model_id` (`main._model_scope`, an exact predicate, never the
  `model:` LIKE) unless `all_models=true`; the page's All models tick is the
  one switch, unpersisted like Starred, and `GalleryPage.other_models` is
  what the same filter matches elsewhere. Delete unstarred takes the scope
  because its count must be the grid's. The Runs picker asks
  `/api/runs?limit=200` and groups the result: "Grids and batches" (images > 1
  or jobs > 1, labelled with count, distinct, "N prompts" and what varied -
  `runAxes` over `RunSummary.axes` (each setting's count of distinct values
  where above one, from `storage._RUN_AXES`), `refine` and `strengths`:
  "steps×CFG 3×2", "steps ×8 · 5 seeds", "strength 0.05-0.15 of #37",
  "comparison · 3 versions"), then at most
  eight "Single images" by time to the second and seed. It names the run the
  search box shows (`syncRunList`, from `resetGallery` and `loadRuns`; its
  "Runs…" option is hidden and disabled), and picking one never resets it.
  Closed, it is `--runs-w` wide (190 px; `clamp(240px, 14vw, 360px)` from
  1800 px), cut with an ellipsis, and `syncRunList` puts the picked run's
  whole label in its title ("Recent runs" otherwise).
- Search terms are `key:value` predicates plus FTS text; `date:` is a local
  day, `mu:none` is `IS NULL`, `id:` is the gallery id the page shows as #N
  (a range is a span), and `search.strict_scope` is the reading for
  `clear --where`, `keep_rates` and the gallery's Delete unstarred
  (`GET`/`DELETE /api/gallery/unstarred`): `clear --where <box>
  --keep-favorites` over HTTP, the search box strict or empty, gated on the
  count typed back and pinned to it as `expected`, which `delete_matching`
  checks under the write lock and answers 409 on a mismatch; rows and files
  only, no orphan sweep. `keep_favorites` keeps the stars and what a starred
  row was made from - its `refine` source and its `reference` ids - through
  one predicate (`storage._UNKEPT`) that `delete_matching`, `clear_history`
  and the counts share; `matching_preview` returns `sources_kept`, and the
  route's `unstarred` and both CLI counts are `records - favorites -
  sources_kept`, so the typed count is the delete's. A 409 reloads the
  gallery and the Runs picker. `forgetImages` is the one place the page
  drops deleted ids - every list, the sheet, the panel's references and a
  Reuse's Undo snapshot (it answers the sentence the caller's note adds) -
  and marks at once the passes of what it deleted (records, card titles,
  every Reuse, the lightbox's facts and `lbSources`). A single card or viewer
  tile keeps its slot as a tombstone (`tombstone`, "Deleted · #N"), which
  drops the card's fav, ref and refined marks and keeps its `data-made`;
  focus and the keyboard's current card move to the next live image. A
  viewer whose last image went with no job running gets its placeholder back
  in place of its tombstones, held at the strip's height ("#N was deleted.
  Generated images appear here."), and `appendResult` counts live tiles, not
  slots. A lightbox delete of the last loaded image pages the next one in
  (`pageInBehind`), never stepping back; one that takes the lightbox's only
  image closes it with focus where the walk came from
  (`closeLightboxAfterDelete`: the viewer's held line or tombstone, the
  grid's current card, else the search box), never `<body>`. Both questions
  - the per-image confirm and Delete unstarred's count - are the page's own
  `<dialog id="confirm-dialog">` (`openAsk`, placed beside what was pressed
  by `placeAsk`), never `window.confirm` or `window.prompt`: Cancel focused,
  or the count's box; Escape and a press outside are Cancel; while it is up
  the document's keys do nothing (its Escape closed the lightbox under it).
  The smokes stand in for it by assigning `askConfirm`/`askCount` and put
  back `confirmDialog`/`countDialog`; `smoke-delete.json` drives the dialog
  itself. A delete under way ignores a second
  press of the same image (`deleting`), and `storage.delete_image` looks up
  and deletes under `_write_lock`, so a second delete is "unknown image". The
  gallery pages by id, never by offset,
  and pages itself as "Load more" scrolls near the viewport (`moreWatcher`,
  re-armed by `watchMore` after each page that lands, never after one that
  failed - the button is the retry).

## Adding a backend

Implement `GenerationBackend` (`load`/`unload`/`is_ready`/`generate`/`describe`)
and register it in `backend/backends/__init__.py:get_backend`. `generate()` is
handed a `should_cancel` predicate and must poll it at every step boundary,
raising `Cancelled` when it answers True. Nothing in the HTTP layer or the UI
needs to change.

## Tests

`uv run --group dev pytest`. They stub the pipeline (`tests/fakes.py`'s
`FakeBackend`, wired by the `fake` and `client` fixtures in `tests/conftest.py`
and driven through `tests/api_helpers.py`; the HTTP tests are `tests/test_api*.py`,
one file per area) and the scheduler (`tests/test_schedule.py`'s `StubPipeline`), so
nothing downloads weights or touches the GPU, and they run against a throwaway
database and outputs directory (`tests/conftest.py`). They run on `config.py`'s
defaults: `tests/conftest.py` drops every `LOCAL_DIFFUSION_*` but the gallery
and adapter paths before anything imports `backend`, so a sourced
`wsl/env.sh` changes nothing. They cover the invariants
that are expensive to get wrong: the schedule the scheduler ends up with, seeds
surviving a JSON round trip, the search fallback keeping its filters, and the
HTTP guards.

The refine pass is covered the same way: `tests/test_refine.py` (each
profile's `refine_supported` and each sibling's call held to the installed
diffusers, `CARRIED` held to `generate()`'s own kwargs, a stub sibling built
over a stub pipeline's components and called once per image with that
image's generator, the cancel paths, mu through the shared scheduler, the
row, the PNG, the key, the migration, the reindex and the merge),
`tests/test_api_refine.py` (the status flag, the submit refusals, the
strength axis with resume, the pass's own size guard and its lack of a
batch budget, `GET /api/images/{id}`, Re-render) and the CLI's
flags in `tests/test_cli.py`. What a pass does to pixels, its time and its
memory are the card's to say, in `docs/verified.md`.
`tests/test_reindex_provenance.py` covers foreign PNG conditioning ids, lost
databases, missing or mismatched source identities and safe in-place metadata
remapping. API tests hold a fake before its first callback to check initial
per-cell progress totals, and API/CLI tests cover axis replacements and the
capability refusal both before and after loading. The job snapshot's
`comparison` is asserted true for a prompt-variants grid (`test_api_grid`)
and false for plain and variable grids (`test_api_jobs`).

The prompt analyser is tested the same way, with no network and no GPU:
`test_prompt_encoding.py` (budgets, cuts, templates held to the
installed diffusers source; its real-tokenizer cases skip when nothing is
cached), `test_prompt_rules.py` (the `analysis` data against the prose, the
authors' examples raising no warning), `test_critic.py` (pins, install,
server command and stop, the server's count and the fit checks, answer
validation, grounding) and `test_api_analysis.py` for the routes, the gate
(Check beside a running job, off the loop; the critique's re-check after
`prepare()`), the status flags (status never loads a tokenizer) and
shutdown (the lifespan's order, `main.Server.handle_exit`). Where the point
is the real thing, `test_critic.py` starts Python children and loopback
servers: a wrapped stand-in llama-server's process tree, its listener and
its counts, and the proxy bypass. The job-object cases run on Windows only;
the wildcard-listener and half-exited-process cases run on Linux only. Its
own real-tokenizer case, `test_measured_counts_and_pieces_are_the_tokenizers`,
skips when klein's tokenizer is not cached. The `fake` fixture installs
`FakeCritic` as `main.prompt_critic` - its six methods held to
`PromptCritic`'s signatures by a test, its question sized by the real
`critic.check_fits` on the unscaled estimate, standing in for the server's
count (a prompt too long for the window is a 502 here too, though near the
window's edge the two can disagree) and
its answer raw JSON put through the real `critic.validate`, so nothing it
returns is something the real pipeline could not - and
`fakes.fake_tokenizer` as `prompt_encoding._load_tokenizer`: a token per word
or mark (on Qwen also per run of newlines, which its BPE merges into one
token), with real offsets, at each encoder's real budget.
`test_the_fake_tokenizers_budgets_are_the_real_ones` holds them to the
`REAL` table.

There is no JavaScript test suite. `web/` is checked in a real browser
instead: `uv run python tools/fake_server.py` serves the app on 7861 on the
tests' fake backend (gradient images stamped with seed and steps, a throwaway
gallery under the system temp dir, `--step 0.3` to make pause and queueing
observable; a cell whose prompt holds `[fail]` fails,
`POST /api/fake/load` (`{seconds, fail: "gated"}`) makes the loads after it
slow or gated, and `POST /api/fake/step` (`{seconds}`, answering the pace it
replaced) slows the jobs after it, for a step that needs a job to outlast its
own round trips - a cancel, a stream attaching), and `node tools/browser.mjs tools/smoke.json` drives headless
Edge or Chrome over the DevTools protocol with Node's own WebSocket - no npm
install - through the lightbox zoom and the focus its close leaves on the
walked-to card, the sheet, the export, the sheet's declared variable order
(under Starred too), a sparse comparison's rows, the re-render button, the
panel's persistence and the duplicate/distinct counts, exiting 1 on a failed
`expect` and 2 on a script it refuses before starting - among them an
`expect` joining an async helper (`__until`, or a `window.__x = async` the
file defines) to more with && or ||, which is never awaited: the whole
condition goes inside the predicate. A `hover` or `click` step takes a
CSS selector in place of a point, aiming at the middle of that element's box
as it is then. It opens the page at 1400x900,
where the smokes are written: `LOCAL_DIFFUSION_VIEWPORT=2560x1305` (a
maximized QHD browser) and `LOCAL_DIFFUSION_DPR=1.25` (with 2048x1040, the
same screen at 125% scaling) move it, a `viewport` step resizes it mid-run,
and `LOCAL_DIFFUSION_PORT_OFFSET=N` adds N to every `127.0.0.1:<port>` in
the steps, so a second checkout can run the same smokes against fake
servers of its own. `smoke-conditioning.json` runs
against a klein-claiming fake on 7864 for the negative field as its caption
alone, the viewer result's ❐ (live, pressed once its image is a reference),
the reference chips (numbered and
named by gallery id, reordered by Alt+Arrow and by a drag, dropped by Delete
with focus kept in the field, the last one's × clicked leaving the emptied
row focused and the tip closed, a deleted image's chip marked), a deleted
reference leaving the chips (and named in the confirm), Reuse dropping a
reference the server no longer has, a reference deleted while a Reuse can be
undone staying out after the Undo, a fifth reference refused under the
gallery (the other cards' ❐ aria-disabled at four, the card note's gist
"Four is the most"), the card's ❐ as a toggle named "Reference image" with
its number in `aria-description` (the card ringed, numbered and pressed, its
"Added as image 1" / "Removed" status, a sheet cell marked as a reference),
the card's and the viewer result's ⤢ aria-disabled with the reason, their
refusal in `#gallery-error` and over the tile, nothing opened, the
lightbox's description reading `ref #N` and its References row (numbered as
the panel's chip, opening it with the result after it, and Use these
references, which leaves a deleted reference out with a note and goes dead
with the reason on a checkpoint without references), the adapter select and
its tip, a weight sweep (its preview naming the adapter) and the prompt
check going stale when the references change (chips, a reorder, Reuse);
`smoke-flux.json` runs against a second fake server
claiming FLUX.1 (`--port 7862 --model black-forest-labs/FLUX.1-Krea-dev`)
for the True CFG box, the lightbox's Sampling ("True CFG 1.5 · no Mu") and
Negative rows, and an Example's Undo note in the panel's flow while the
negative is greyed and over its caption while it is live, and
`smoke-preview.json` against one slowed to `--step 0.3` on 7863 for the live
preview, the tab's title and the topbar line during the job, and the
finished title and the finish line's seed after it. `smoke-ernie.json` runs
against an ERNIE-claiming fake on 7865 for the explicit prompt expansion
flow; the fake takes the claimed profile's threshold and true-CFG flag, and
its model id's prompt-expansion capability, so it behaves as the real one
would.
`smoke-animagine.json` runs against an Animagine-claiming fake on 7866 for its
tag guide, deterministic autocomplete/validation, CFG 5 default, recommended
aspects, orientation switch and disabled non-flow Mu controls, the chips past
the CLIP cut, the token count and "cut unknown", the offline state, Tab under
three characters, keyboard focus after Alt+Arrow and Delete, the 64-item
lookup cap, and the prompt
check on a tag checkpoint: a tag out of the documented order, Use reordered,
the critic's reorder set aside where Check placed every tag and found
nothing to move, the critic's reorder kept, and titled, as Check's own order
where it is one (`reordered_kind` "check"), a 70-tag prompt's order shown as
partly checked (the 64-tag lookup cap) with the critic's reorder kept where
the order rule finds nothing wrong with it (`reordered_kind` "allowed") and
set aside, in the rule's words, where it flags one, a tag the fake index
does not know (`zzkx oc`) after a general tag and a prose prompt shown as
partly checked with the items named (the first with the critic's reorder
kept as "allowed"), the same tag ahead of the general tag, alone, and a
one-clause prose prompt shown as checked (the first with the critic's
reorder set aside), "Partly checked" following `order_detail` and never `categories_detail`
alone, Anima's artist-check note, and the recommended-negative Apply going
dead once the negative box is edited.
`smoke-tags.json` uses an Animagine-claiming fake on 7878 for the chips'
verdict cache, typing state and live summary, an accepted suggestion's undo,
category edges, repeats and the focus ring on one, the shared tip and the
after-cutoff mark, + tag, gap hover (waited for, as a late tray render
redraws it) and click (none on the label or hint),
the Tags helper under the prompt and whole, Escape closing the suggestions
before the helper, the Examples menu's keyboard (ArrowDown, Enter), its Undo
note (on its own line at 800 px), the needs-a-reference note, the file-drop
guard, and a held panel not moving for chip edits at 1400x900.
`smoke-illustrious.json` uses an Illustrious-claiming fake on 7867 for hybrid
prompting, the native 1-2.25 MP size range, 4 MP warning and disabled Mu.
`smoke-z-anime.json` uses a Z-Anime-claiming fake on 7868 for Base defaults,
natural-language prompting, negative guidance, enabled Mu and the documented
4 MP ceiling.
`smoke-paging.json` uses a fresh Krea-claiming fake on 7867 for the gallery
paging itself on scroll (it builds a 60-image grid, so it holds once per
gallery). Run it separately from the Illustrious smoke on the same port:

```powershell
uv run python tools/fake_server.py --port 7867 --fresh
node tools/browser.mjs tools/smoke-paging.json
```

`smoke-anima.json` checks Anima Turbo defaults, its documented size range,
disabled Mu and applied saved settings on port 7869:

```powershell
uv run python tools/fake_server.py --port 7869 --model circlestone-labs/Anima
node tools/browser.mjs tools/smoke-anima.json
```

`smoke-kohaku.json` checks Kohaku's hybrid tag guide, defaults, size guidance
and saved settings on port 7870:

```powershell
uv run python tools/fake_server.py --port 7870 --model KBlueLeaf/Kohaku-XL-Zeta
node tools/browser.mjs tools/smoke-kohaku.json
```

`smoke-netayume.json` checks NetaYume v4 defaults, the qualified size guide,
author example, enabled Mu and applied saved settings on port 7871:

```powershell
uv run python tools/fake_server.py --port 7871 --model duongve/NetaYume-Lumina-Image-2.0
node tools/browser.mjs tools/smoke-netayume.json
```

`smoke-newbie.json` checks NewBie defaults, the qualified size guide, XML
example selection and persistence, and applied saved settings on port 7872:

```powershell
uv run python tools/fake_server.py --port 7872 --model NewBie-AI/NewBie-image-Exp0.1
node tools/browser.mjs tools/smoke-newbie.json
```

`smoke-wai.json` checks WAI v17 defaults, author tag recipes, the qualified
reference-size guide, disabled Mu and saved settings on port 7873:

```powershell
uv run python tools/fake_server.py --port 7873 --model LyliaEngine/waiIllustriousSDXL_v170
node tools/browser.mjs tools/smoke-wai.json
```

`smoke-refine.json` checks the Refine dialog on a FLUX.1 Krea [dev] fake on
7875, whose refine cells are the source image under a stamp: no refine
control in the panel, the card's and the lightbox's Refine opening the
dialog with the image's prompt and seed and the panel untouched, the
titles in "#N" form, focus on Strength with its value selected, the
profile's strength, the shortcut hint by the shared rule, the anchored top
across a fold toggle at 1400x900 (scrolling short of the window's bottom),
the scale buttons and the size line - naming the measured run, FLUX.1's note
under Strength, the opening size capped below it with the skipped scale
named on its button and a run at the capped size not remembered as the
scale, the crop's kept percentage and Keep shape - Versions and their
seeds, the rows and the lightbox's settings line (`refine of #N`), the
facts' Refined from link and S comparing a pass with its source, a pass past
4 MP and one past the guard refused in place, Escape on a dialog opened from
the lightbox giving the lightbox back on the same image, a strength sweep's
preview, its sheet axis and `refine:` in the search box, the passes strip
opening the lightbox beside the source, a sweep that leaves no step refused
in the preview and a refusal moved to the error line at Strength or W with
Refine not moving, a strength typed and entered at once refused in the
dialog, a strength past 1 refused as a fraction, Compare 3 (held with the
Steps that would give a sweep at 2 steps, writing nothing), Mu's "auto",
Refine held while the server does not answer, arrow-key
stepping and the dice, Ctrl+Enter and Enter running the pass, the
remembered strength and Reset forgetting it, the unsent draft and Start
over, Reuse opening a pass (its title and reused line, and another
checkpoint's, with this one's sampling), a pass whose source has gone, after
a viewport step to 2560x1305 the two panes with the kept-region frame and a
top 8vh down that does not move as the fold opens, the source standing in
for a pass whose frames are dropped - full size, in the result's own box,
captioned in the tile's `.foot` "Refining #N → W×H · strength s · step 1/…" -
focus off `<body>` after that pass, opened from script, ran - and a source deleted while the dialog is open refused in the dialog's own
line with the dialog kept open. `smoke-conditioning.json` has the other side
on its klein fake: the card's, the viewer's and the lightbox's Refine
answering with the reason and opening nothing.

```powershell
uv run python tools/fake_server.py --port 7875 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-refine.json
```

On the same FLUX fake at 7875, `smoke-refine-unavailable.json` checks an
unloaded off-catalogue capability guess: ⤢ opening nothing, a refined
record's Reuse keeping the pass in the dialog with Refine dead and nothing
submitted, the dialog waking when status says the pipeline has a pass, and
Generate meanwhile rendering directly. `browser.mjs` sends Escape with its
key code, so a modal dialog closes on it as on a real press. Fake servers
share one throwaway gallery unless `--data` gives each its own, and
`smoke.json` counts the images planted under another checkpoint: run it on
a fake of its own while others are up.

`smoke-refine-sdxl.json` checks the Refine dialog on an Animagine fake on
7877: the generic tight line where nothing was measured, SDXL's strength
sweep, the pass prompt's own check (the 77-token cut outside the fold, Show,
the checking line during a slow first use, Apply into the dialog's box with
Ctrl+Z, dead while stale and dimmed by the shared 0.8 / `--text-3` rule, the
negative-prompt Apply dead while CFG greys the negative box, the hidden line
that announces the check only when its text changes), the unknown-tag line
(a repeated tag named once) and its silence on a failed lookup, and nothing
reaching the panel:

```powershell
uv run python tools/fake_server.py --port 7877 --model cagliostrolab/animagine-xl-4.0
node tools/browser.mjs tools/smoke-refine-sdxl.json
```

`smoke-wide.json` checks the page at the sizes it is used at, switching
with `viewport` steps between 2560x1305, 2048x1040 at 125% and 1400x900 on a
FLUX.1 Krea [dev] fake on 7876: the wide tier's scale (the panel's 461 px at
2560 and 400 px at 2048, a 15.5 px body and 13.5 px captions at 2560 and a
third of the way there at 2048, eight and seven gallery columns) and the
unchanged one at 1400x900 (330 px, 14 px, 12 px, five columns), Generate on
screen, the page's own focus ring on a checkbox and a button and the LoRA
select drawn dark at both wide sizes, the caption buttons 30 px at 13.5 px
and the tick boxes 18 px at 2560, a 16-digit seed fitting its box - and the
Refine dialog's at 125% - a
size strip's one tab stop with aria-pressed and the arrows moving focus
without choosing, the greyed negative's Set True CFG button above the box
(never over its text, the box still under the pointer) focusing True CFG and
writing nothing, the prompt box growing to twelve lines (eight on the short
window) with a dragged height winning over the cap, and with the grid fold
open - by a click on its summary, which brings Run grid into view - Run grid
on screen after a wheel over the stage and at the end of a 160-image
gallery, the panel holding its bottom a gutter above the window's (the
gutter and the resting top read off the page), the panel coming back down
as far as a wheel scrolls up, the prompt and Examples focused from deep in
the gallery and by Shift+Tab (past the negative's button) brought into view
with the page left where it was, a finding clicked in the prompt check
selecting its words in view without the check moving, Generate pressed with
the panel held staying under the pointer through its run and the panel
holding its bottom again once the pointer leaves, Resume's tip opening
upwards inside the window, a fold restored at load scrolling nothing, the
panel static in one column below 901 px, and in one column Batch's tip
inside the window and the comparison's always-run original box undimmed:

```powershell
uv run python tools/fake_server.py --port 7876 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-wide.json
```

`smoke-grid.json` checks the grid fold on a FLUX.1 Krea [dev] fake on 7879:
the resolved prompt and the one-value warning (commas named); Run grid as
wide as Generate; the totals at body size with factors of one trimmed; a
spaced range, `6..12` and an axis message naming its box; Resume's refusal
without pydantic's prefix; Repeat above one; the budget line, including 0
seeds, the jobs ahead, and the load note following a real unload and load
with no box touched; the cells already in the gallery once a grid ends,
without Resume; Ctrl+Enter running the grid from a grid box, refusing from
the prompt while the grid holds its placeholders, and generating on a plain
prompt; and, at 2560x1305, Run grid in the window after a real click on the
fold's summary once the preview has landed:

```powershell
uv run python tools/fake_server.py --port 7879 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-grid.json
```

`smoke-sheet.json` checks the contact sheet at 2560x1305 on a FLUX.1 Krea
[dev] fake on 7880: the page landing on a sheet it opens (Runs, a
comparison, a sweep), a 4 x 3 on one screen with heads at the page's
`--fs-body-sm` and srcset sizes, heads sticking in the sheet's own scroll
box and a keyboard-focused cell scrolled clear of them, the fit and the wrap
following a resize, a twice-run grid's repeats in the note, a seeds-only run
wrapped (and its export), edit labels for prompt iterations, the restored
search on Sheet off, the Sheet button's newest run, pins taking the
commonest value, F starring a focused cell, the lightbox opened from the
sheet starring the cell (its star "★ Star", `aria-label` Star, its title
and `aria-pressed`) and F on the cell it hands focus back to, the Fixed line
under Starred, Resume this run filling the seed and size (its warning
keeping the preview's `.totals`), its mismatch warning and its absence on
another checkpoint's run, the version rows with their edits and titles (a
reorder as "same words, moved"), a strength sweep's source column and
lightbox walk, the refined badge, pins in the export's file name, and an
empty sheet drawing only its note. `smoke-sheet-klein.json` uses a FLUX.2
[klein] fake on 7886 for the References axis, the shared-cell count and the
adapter named in LoRA heads, the Fixed line and the export:

```powershell
uv run python tools/fake_server.py --port 7880 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-sheet.json
uv run python tools/fake_server.py --port 7886 --model black-forest-labs/FLUX.2-klein-4B
node tools/browser.mjs tools/smoke-sheet-klein.json
```

`smoke-delete.json` checks deletes taken to completion on a FLUX.1 Krea
[dev] fake on 7881 (a checkpoint with a refine pass, so a starred pass can
keep its source): the per-image confirm - the page's own dialog beside the
card's ⌦, Escape as Cancel, over the lightbox too - and its count box for
Delete unstarred, a lightbox of one image deleted, Delete unstarred keeping a starred
pass's source with the typed count, a landed image's 409, the All
models/running-job dialog, a pass whose source has gone - marked at once on
the page that deleted it (record, card title, Reuse) - tombstones
("Deleted · #N"), focus and the emptied viewer's placeholder, an image already deleted,
a tile deleted elsewhere and the job's replay, the lightbox paging in behind
a delete (its counter reading the gallery's total and the id), a
double-clicked delete, and Reuse: Ctrl+Z and its Undo, the Undo retired by a
change (Ctrl+Z, a seed, an Example) and a Reuse taking an Example's Undo, the
page unmoved mid-gallery with the prompt in sight, the page's end, one
column at 800x900, and the Undo in the window at 2048x1040 @1.25:

```powershell
uv run python tools/fake_server.py --port 7881 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-delete.json
```

`smoke-jobs.json` runs against the default Turbo fake on 7882 at
`--step 0.3`: the tab title and topbar mirror; the one-line Now row, count
first, with Pause and Cancel and its title; the queue's named rows, a
per-row cancel and the single "Next:" row; Unload held with its reason;
Pause answered at once with no aria-pressed and held on a one-image job,
its note gone with its job; the finish line's seed with Use seed and the
fixed-seed hint; a grid named by its own prompt, skipping a `[fail]` cell
live, with Show sheet and a Retry that sends its own payload with Resume and
the base seed, once, keeping focus; the failure rule's "Stopped at cell" and
Resume, kept as the previous job's while a queued job runs; a cancelled
job's outcome led by "Previous job:" while the next runs and gone at its
end; focus after Cancel (Generate) and after Pause held to the end (Use
seed); a time budget's held ETA and count; the gallery kept where it was
scrolled with New image; the offline state through a stubbed fetch and a
restart through a changed start time; the load's ticker with its
aria-hidden clock and indeterminate bar; a cancel during the load; a gated
load's detail said in the panel line after every Generate while the viewer
holds images; a gated load's card said once and not re-announced by the
banner, its Retry load handing focus to Generate; Load and Unload handing
focus to each other; the banner hidden by a new job and not repeated
after a watched one; a gated load's finish line "Not made: the model could
not load"; and a watched job whose server goes (a stand-in EventSource and
fetch): the job line in the topbar's phrase with the count, no steps, time
left or topbar mirror, the tab's title and the strip's note the same, then
`showLostRun`'s card, reworded once the server answers, its Run it again
running the request with focus going to Generate:

```powershell
uv run python tools/fake_server.py --port 7882 --step 0.3
node tools/browser.mjs tools/smoke-jobs.json
```

`smoke-results.json` checks the results viewer on a FLUX.1 Krea [dev] fake
on 7883 (written at `--step 0.3`; it passes at the default step too): every
? badge named and described, and Resume's badge leaving Resume unticked; the
one-line idle viewer, the empty-install box and its Examples; the empty
gallery's line and its Examples; the idle viewer staying one line beside a
docked, content-sized check with the gallery beside it; a result's
figure, label, title and card tools at 1400x900 (❐ hidden on FLUX), with
`this-run`, the star and ⤢; at 2560x1305 the column with Delete in words and
the row under the picture empty at `--h-tool`; the live preview
keeping the result's box; `#recent` level with the viewer, and staying
beside it with the check docked while the image and the gallery head stay
put, the column's actions and `.result-brief` then the row under the
picture; the fit at 2048@1.25; the lightbox walk from the single image; a batch
of four best-fit; a pass's note and lineage, then its source deleted ("#N
(deleted)", no buttons, the title); four landscape images two by two with
the gallery head unmoved from the first frame; a deleted result held at its
height with focus on its line (the column's Delete); a 28-seed run at
2048@1.25 ending scrolled to its newest row; 28 at QHD with no padding-only
scrollbar and the gallery head unmoved from its first frame (`reserveStrip`); the banner's
"2 since deleted" / "all since deleted"; the banner's Resume after a
resumed-and-cancelled grid putting the run's base seed in Seed; no Resume
for a cancelled comparison; and Reset with its Undo:

```powershell
uv run python tools/fake_server.py --port 7883 --model black-forest-labs/FLUX.1-Krea-dev --step 0.3
node tools/browser.mjs tools/smoke-results.json
```

`smoke-lightbox.json` checks the lightbox on a FLUX.1 Krea [dev] fake on
7884: the facts list and its rows; the keys line; the star ("★ Star",
`aria-label` Star); a pass compared with its source (S, the button, a
dragged and keyed divider, S on the focused divider handing focus to the
toggle, swipe and side by side keeping 1:1 and fit, side by side at fit
clear of the arrows' circles, a pass reshaped to 3:2
compared with the part it kept, a deleted source's dead controls); the
source link with the view carried; the passes link; a pass whose source has
gone (its Reuse and its card's title); the arrow strips (a press 60 px off
the circle steps, the backdrop below still closes, the wheel over a strip
zooms, the first image's disabled arrow a circle with the backdrop beside it
closing, a double-click whose step takes the strip away stepping once,
circles again when zoomed); focus after Escape on the walked-to card, and
Ctrl+Enter there generating and opening nothing; Escape on an image no strip
holds focusing the search box; Cancel on a Refine opened from it, a side by
side 1:1 included, and no reopening after a run; an Escape and a Cancel held
while the pass is being sent ("Sending…"), with the pass run and focus on
the source's card; an image that will not load (deleted, and with the
server not answering: the offline sentence as an offline message, taken
back by the status poll that answers, which asks for the picture again); at
2560x1305, 1:1 on a 1 MP image by key, button, wheel and double-click,
circles below fit with the double-click beside the picture giving fit back,
and a double-click beside a below-fit image giving fit back; and at 800x900
no keys line:

```powershell
uv run python tools/fake_server.py --port 7884 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-lightbox.json
```

`smoke-gallery.json` checks the gallery on a default (Krea 2 Turbo) fake on
7885, a checkpoint with no pass and no references: the Runs picker's groups,
labels (what each run varied: "steps ×8 · 5 seeds", "2 seeds") and sync with
the search box (a pick, a card's ⊞, typing, Sheet off),
at its own width with the search box whole and the picked run in its title;
`id:` and the search tip; the Starred tick and Re-render's label; the five
tools and their names on hover and focus inside the card, with Delete's over
Delete; Refine and Reference aria-disabled, their refusals in
`#gallery-error` and a one-line card note, with the card and grid unmoved, a
second press landing on the same tool, and the row's names held down while
the note is up; the star button (named Star in both states, hidden until
hover, a click starring without opening the lightbox, F from the star and
from a tool, the lightbox's F reaching the card, the lightbox star's title
and `aria-pressed` matching the card's); a refined record's badge and Reuse
labels (planted into the page's gallery replies); the ⚠ carry-over warning,
said again when references flip and when the adapter list lands; the day
line across pages, a line over the first card when the newest images are
not today's, and a deleted head's tombstone across a day line drawing no
second line; the roving tab stop, arrows across a short row and the day
line, Home/End, Down paging in, F, a delete handing the stop on, the
lightbox's user close handing the stop to the card it ended on, and a job's
newest image taking the stop where none was chosen, prepended under an old
head made two days older with the day's line between them; and the tools
fitting at 2560x1305 and 2048x1040@1.25:

```powershell
uv run python tools/fake_server.py --port 7885
node tools/browser.mjs tools/smoke-gallery.json
```

`smoke-analysis.json` checks the prompt check on a FLUX.1 Krea [dev] fake on
7874: the CLIP cut, the negation and quote findings with their sources, span
and cut selection (an emoji ahead of them included) with the selection
scrolled into its box, the stale rule for both prompt boxes with the status
slot undimmed while stale, Apply, the critic's first-use and timing lines -
built from `ModelStatus.prompt_critic`: its label, the downloads a first use
makes, neutral wording under an override - and its clock, its hold on
Generate, inline cites, a cite's whose-words note and the General label, Use
rewrite with undo and a reload and filling Your version, an edit while the
critic thinks, a failed critique's row in one slot (`#error` only after
Close), a two-version comparison opening its sheet with a blank Seed left
blank, a Generate under that sheet's `job:` scope counted in the gallery's
line with Show all opening the whole gallery, another opening all its
cells with Starred ticked (the tick's notice gone once the run is
reopened), the chips and the check moving from the wide host back under the
prompt with focus and selection kept, a prose prompt's order shown as not
checked, a one-clause prompt the critic cannot reorder, a comparison against
Your version, the cut excerpt in code points, focus after Check
(two-column), after Check again, after Close during the critic's check and
after the critic's answer, the critic's hold through its pre-check (a
replaced one saying so, a cleared one silent, a throwing one giving Check
back), the stale badge left alone while typing, the Seeds select read at run
time, by injecting the server's `reordered_kind`, the Use reordered title
following what the server says it kept, not the page's own last check, with
Compare offering the critic's reordering the server kept, and the docked
check at 2560x1305 and 2048x1040@1.25 (the image just made wholly in view
and unmoved after Check, the critic's answer and Ctrl+Enter; the column not
setting the rows' height; its head stuck after the answer, showing the stale
badge; Check again resetting its scroll; the gallery's controls clear after a
400 px page scroll; Close on the head's first line at 125%), the critic's two versions stacked there and side by
side full width at 1400 px, and the stacking at 1400x900.
`smoke-animagine.json` covers the tag-order check, the critic's reorder
under full and partial tag placement end to end (set aside, kept as Check's
own, kept as allowed, flagged, each with the kind the server says it kept),
the partly-checked order (past 64 tags, without the tag index, a tag the
index does not know, a prose prompt) and the checked one where no missing
category decides it (that tag ahead of a general tag, alone, one prose
clause), and the negative prompt's Apply,
dead while the box is greyed. The fake server
serves the tests' `FakeCritic` and word tokenizers: `--critic-delay` (seconds
a critique takes, default 0.4, so its in-flight state can be seen; a prompt
too long for the critic's window is refused at once, a 502, on the
unscaled estimate - the real critic refuses at once only what cannot fit
however it tokenizes, and the rest on its server's count) and
`--no-critic` (a platform with no build: the page hides the critic). The fake
critic counts as downloaded from its first critique on, so the first-use line
shows only on a newly started server; the smoke accepts either line:

```powershell
uv run python tools/fake_server.py --port 7874 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-analysis.json
```

`smoke-rewrite.json` uses the same fake on 7874 for the visible rewrite diff,
the change's grouped gist, escape-only edits, Unicode, markup rendered as
text, unchanged/absent rewrite states, and the rewrite findings' "also in
your prompt" / "new" / "No longer flagged" / "Not flagged … without the tag
index" marks.
`smoke-comparison.json` uses the same FLUX-shaped fake on 7874 to run
two comparisons with original and yours reversed on fixed seeds. Each
opens its own job, while opening their shared run exposes Prompt as an
axis. `smoke-comparison-batch.json` uses the same fake for all six Seeds/Batch
pairings, their preview and request counts, select values changed without
events, paired output seeds, sheets and stale-run disabling. API tests cover
paired saved seeds, batch grouping and preview counts for all six pairings.
The analysis and Animagine smokes cover CFG/True CFG activation
changes while a grid keeps the negative editor enabled, inactive raw
edits, and settings changes while a check or critique is in flight.

`smoke-analysis-tags.json` uses an Animagine-claiming fake on 7887 for the
prompt check on a tag checkpoint at QHD: a Series-before-Character order
finding the rewrite keeps, shown as not flagged without the tag index rather
than fixed, and the docked check with chips at 2560x1305 and
2048x1040@1.25, where the image moves by exactly the chips' extra lines and
stays in view (6 tags at 2560, 32 at 2048@1.25, 40 at 2560):

```powershell
uv run python tools/fake_server.py --port 7887 --model cagliostrolab/animagine-xl-4.0
node tools/browser.mjs tools/smoke-analysis-tags.json
```

GPU generation, repeat-seed pixels, cancellation and any preview evidence
are recorded per checkpoint in `docs/verified.md`, and the real critic's runs
under its "Prompt analyser" section; CPU and browser checks alone do not
establish runtime quality or performance. WAI's conditioning
regression is `tests/test_wai_conditioning.py`: both positive and negative
SDXL conditioning must remain finite with its unused final CLIP-L NaNs
under none, FP8 and int8 quantization.


Static files are served from disk, so a `web/` edit needs no server restart;
a `backend/` edit does. The app's top-level bindings (`els`, `lbView`,
`sheetRecords`) are reachable from a step's `eval` because the page's scripts
are classic scripts sharing one global scope - which is also why they stay
top-level. `browser.mjs` gives
each run its own profile directory: a shared one stays locked by an Edge a
crashed run left behind, and every later launch then hands its command line
to that instance and reports "the browser did not come up".

## Environment

Python 3.13 (not 3.14 — torchao/diffusers wheels lag there).
torch 2.13.0+cu130, diffusers 0.40.0, torchao 0.18.0.
