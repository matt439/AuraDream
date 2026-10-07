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
- `backend/prompt_text.py` — shared prompt parsing, prose/tag detection and lettering spans
- `backend/prompt_order.py` — documented tag placement, ordering and category coverage
- `backend/prompt_rules.py` — the analyser's rule checks against the guide's
  `analysis`: style, documented order, `reorder_tags`, the items left
  unplaced where that decides the order (`unplaced_items`), each comma
  item's step for the chips (`order_slots`, answering only where
  `reads_order`; these and the order rule itself place through one
  `_places`) and what a critic's reorder is held to
  (`places_every_item`, `order_items`, `order_finding`, `scene_ahead`),
  what a check asks the tag index about (`lookup_items`) and whether a
  rewrite's artists were all read (`artists_read`), negations, unquoted
  text to render, the tag checkpoints' specifics; `heuristic` ones labelled
- `backend/prompt_sources.py` — what the critic may cite, per guide constant:
  the guide's own sentences (G) and short verbatim author excerpts (A) with
  their URLs, notes and caveats
- `backend/critic.py` — the local prompt critic: pinned llama.cpp
  `llama-server` and Qwen3-4B GGUF, fetched and verified by `prepare()`, one
  server per `critique()`, the answer validated against its sources;
  `critic_runtime.py` owns pinned installation, `critic_server.py` the process
  tree and loopback requests, `critic_context.py` the question and token budget,
  `critic_reorder.py` pure prompt permutations, and `critic_validation.py`
  answer grounding and rewrite validation
- `backend/translation.py` — lazy CPU-only Marian translation of ERNIE's
  expanded Chinese prompt for the UI; the Chinese prompt remains canonical.
  `cached()` is a file check (a local tree or the Hub cache, no network) for
  the status's first-use line. The load sets `DISABLE_SAFETENSORS_CONVERSION`
  for itself alone (`_without_conversion`, the previous value put back),
  since Transformers would otherwise fetch the conversion bot's safetensors
  copy of the `.bin` it loaded, a second 312 MB
- `backend/tags.py` — cached Danbooru tag suggestions, their order
  (`rank_suggestions`: exact match, from the name's start, a later word,
  alias; post count within each; a guide tag ahead once half of it is typed;
  a tag of the recommended negative never first unless typed whole) and exact
  validation with a reason `code` for tag-caption checkpoints; remote failure
  means unchecked, never invalid - except an item longer than any tag
  (`TAG_NAME_MAX`, 200), answered unknown without the index, reachable or
  not
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
  device placement, and the guard that keeps the adapter fit's meta-device
  build apart from a load (`weights_loading`, `empty_weights`); takes
  `settings` as a parameter
- `backend/backends/single_file.py` — pinned, CPU-only first-load conversion
  for Illustrious XL v2.0-STABLE, WAI v17 and RouWei 0.8 epsilon; atomic cached
  trees feed the normal loader; `sweep_work`, which every converter calls first under its
  target's lock, removes interrupted conversions' work directories (exactly
  `<tree>-` plus tempfile's 8 characters, a real directory holding only
  `tree`), and each converter's work directory ignores its own cleanup
  errors so a failed conversion reports its own error
- `backend/backends/anima_loading.py` — pinned CPU conversion of full Anima
  Turbo and Aesthetic v1.1 and their conditioners, with official shared components
  and separate cached trees
- `backend/backends/anima_pipeline.py` — upstream Anima modular blocks behind
  the standard pipeline interface, with per-step progress and cancellation
- `backend/backends/netayume_loading.py` — pinned CPU conversion of NetaYume
  v4's split denoiser, with the official Lumina 2 parent's shared components
- `backend/backends/newbie_loading.py` — pinned CPU conversion of NewBie Exp0.1;
  `newbie_pipeline.py`, `newbie_transformer.py` and `newbie_text_encoder.py`
  provide its callback-capable pipeline, pooled-CLIP denoiser and native Jina tower
- `backend/backends/schedule.py` — how a requested mu becomes the whole
  shift: `forced_mu`, `shifted_sigmas`, the static-shift fallback, and
  `requested_shift`, which says which of the two a pipeline takes, for
  `generate()` and the Refine preview alike; and `start_sigma`, where a
  refine pass starts, read off a copy of the loaded scheduler, with
  `START_RECIPES` (each flow image-to-image class's sequence-length and
  default-schedule statements) and `SHIFT_CALL`, held to the installed
  diffusers source by `tests/test_refine_schedules.py`
- `backend/grid.py` — axis specs (`6-12x2`), prompt variables and prompt
  variants, and the cell expansion, shared
- `backend/latent_preview.py` — the latent-to-RGB maps the live preview
  draws with, one per VAE, fitted on the card
- `backend/jobs.py` — single-worker queue, SSE progress
- `backend/storage.py` — SQLite index + PNGs with embedded metadata
- `backend/search.py` — the search box's grammar: `key:value` terms as SQL
  predicates, the LIKE escape, `strict_scope`
- `backend/main.py` — app lifecycle, model/prompt routes and generation submission;
  `api_gallery.py` owns gallery browsing, runs and deletion, and `api_jobs.py`
  job inspection, cancellation, pause and SSE, with app-owned dependencies
  supplied by router factories
- `backend/generate.py` — headless CLI; drives the backend directly, no server
- `backend/clear.py` — deletes rows, their PNGs and any orphans
- `backend/stats.py` — the starred *rate* per setting (`storage.keep_rates`),
  scoped to a run by default; `--by variant` gives a comparison's rate
  per version
- `backend/reindex.py` — rows rebuilt from the PNGs' `local-diffusion` chunk, in place or
  copied in from another gallery; idempotent on the UNIQUE filename
- `web/` — served as-is from `/static`; `index.html` loads 26 classic
  scripts in order, sharing one global scope; only `app.js`, the last, runs
  the wiring and boot. `state.js` owns shared state; `generation.js`,
  `grid-preview.js`, `job-stream.js` and `job-results.js` handle submission,
  previews and progress; `panel.js`, `runs.js` and `viewer.js` the settings,
  run history and results; `gallery.js`, `gallery-actions.js` and `reuse.js`
  the cards and their actions; `sheet.js`, `sheet-render.js` and
  `sheet-export.js` the contact sheet; `lightbox.js` the lightbox and shared
  `factsList`; `refine.js` and `refine-prompt.js` the Refine dialog;
  `tags.js`, `tag-tokens.js` and `tag-chips.js` the prompt helper; `analysis.js`,
  `analysis-critic.js` and `comparison.js` the check and critic. `sizes.js`
  and `status.js` retain their control/profile roles. The stylesheet is
  `style.css` (tokens and base layout), followed by `styles/*.css` in the
  explicit link order; preserve that cascade. See `docs/source-audit.md`
  for the responsibility map and retained long files.
- `tools/` — `fake_server.py` (the app on the tests' fake backend, port 7861)
  and `browser.mjs` + `smoke.json` and a `smoke-*.json` per checkpoint or
  feature (headless browser over DevTools; among the features
  `smoke-stage.json`, Gallery beside, and `smoke-narrow.json`, one column
  and narrow windows): the UI's stand-in for a test suite, see Tests
- `tests/` — the pytest suite, no GPU and no weights: `conftest.py`,
  `fakes.py` and `api_helpers.py` (the fake backend and its fixtures),
  `test_api*.py` (the HTTP routes, one file per area), and a file per module
  or invariant beside them - `test_refine_schedules.py`, for one, holds a
  pass's start index and start sigma to the installed diffusers; see Tests
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

- `Laxhar/noobai-XL-Vpred-1.0` selects its pinned official Diffusers tree.
  `loading.resolve_model` reads configs before quantization checks and then
  only component weights/tokenizers, excluding the duplicate single file.
  Keep the public id in settings/metadata. The published scheduler incorrectly
  declares epsilon: `loading.component_overrides` installs Euler with
  `v_prediction`, zero-terminal-SNR rescaling and trailing timesteps (999
  first). Trailing is the app's Diffusers-backed integration choice; the
  author explicitly specifies the other two overrides. Use exact-id 28
  steps/CFG 5, disabled Mu and ordinary SDXL Refine at strength 0.4.
  No Nunchaku build is mapped. Preserve the author's quality/date vocabulary,
  approximate 2024-10-23 Danbooru cutoff, separate prefix and caption order,
  77-token CLIP limit and approximately 1 MP size advice.

- `Minthy/RouWei-0.8` selects the author's pinned epsilon single file through
  `single_file`, with its SHA-256 checked before CPU conversion. Preserve its
  VAE, both CLIP encoders and public id in settings/metadata. Use the exact-id
  28-step/CFG-6 profile, epsilon/Euler Ancestral, disabled Mu and the ordinary
  SDXL Refine pass (strength 0.4); no Nunchaku mapping. Its hybrid tag/prose
  guide uses the 2025-04-30 cutoff, `by ` artist prefixes and artist-last
  fallback. The author prefers separate CLIP chunks, but this app has only
  77 tokens per CLIP and no chunking or `BREAK` operator. The author's
  1-1.5 MP recommendation and separate upscale recipes are not a direct
  large-image guarantee. The same repository's V-pred file is a different
  checkpoint and must not replace this epsilon recipe.

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

- `circlestone-labs/Anima@aesthetic-v1.1` is the app's distinct selector for
  the official Aesthetic v1.1 split file in `circlestone-labs/Anima`.
  `anima_loading` pins its own revision and cache, sharing the strict CPU
  conversion and official shared components with Turbo. Keep the selector
  in settings and saved metadata, but use the actual repository for Hub
  downloads and links. Never change the bare id's Turbo meaning or cache.
  `AnimaAestheticPipeline` shares the inference blocks with 30-step/CFG-4
  defaults and fixed FlowMatch Euler shift 3. Mu, references, user LoRAs,
  Refine and Nunchaku are unavailable. Its own guide and critic grounding
  omit Turbo's CFG-1/distillation claims and the generic `score_7` prefix:
  the authors discourage `score_*` in both Aesthetic prompts.

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
  the MRO and the `__cause__`/`__context__` chain, the first found except
  that `OfflineModeIsEnabled` wins wherever it is - huggingface_hub raises
  `LocalEntryNotFoundError` from it under `HF_HUB_OFFLINE`; the gate's kind
  from `launch.CHECKPOINTS`); anything else stays `short_error`, and the log
  keeps the library's words. It is raised on as `jobs.LoadFailed(detail)`,
  so a job that ended at the load and Expand (a 503 from its route) carry
  the status's sentence.
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
  the offload chain, and puts the result in the prompt editor.
  `ModelStatus.prompt_expander` (`ExpanderStatus`: `loaded`, `fetch`
  [{name, bytes, step}], `fetch_bytes`; None without an expander) says,
  from file checks only (`backend.prompt_expander_status()`,
  `loading.prompt_expander_cached`, `MarianTranslator.cached()`), whether
  the rewriter is on the live pipeline and what a first use downloads, each
  at its step: the rewriter's `pe/model.safetensors`
  (`loading.PROMPT_EXPANDER_BYTES`, step `expand`) and the translator's
  `.bin` (`translation.MODEL_BYTES`, step `translate`). It is the
  expander's counterpart of the critic's `describe()`. The route answers a
  load that failed with 503 and the status's sentence; its other failures,
  and a translation's, come without the class name in front
  (`main._expand_detail`) - a message of one word keeps the class, and the
  log always does. `generate()`
  still sets `use_pe=False` wherever the switch exists. The tests'
  conftest refuses `loading.hub_file` for every test; one that needs an index stubs it.

Schedule and guidance:

- A requested `mu` is the whole shift. `schedule.forced_mu` overrides the value
  on its way into a dynamically shifting scheduler; a static one gets
  `schedule.shifted_sigmas` pre-shifted by `mu - ln(shift)`;
  `schedule.requested_shift` decides which of the two a pipeline takes, for
  `generate()` and the Refine preview's `start_sigma` alike. Assert on
  `scheduler.sigmas`, never on the list the backend computed. Mu requires a
  `FlowMatch*` scheduler first: SDXL's Euler scheduler accepts custom sigmas
  too, but is not flow matching. `mu_supported` is profiled before load and
  read from the scheduler after it, and disables both Mu controls when false.
  `mu_static` (in every status's defaults) is the shift a blank Mu leaves
  where the scheduler's shift is a fixed number: ln 6 on Z-Anime, NetaYume
  and NewBie, ln 3 on Z-Image and Z-Image Turbo (lite repos included), None
  elsewhere; it is set only where Mu applies and the profile sends no mu of
  its own (`test_a_fixed_shift_is_the_profiles_mu_static`). The panel's and
  the Refine dialog's Mu placeholder is that number (`muPlaceholder`: "1.79"
  where FLUX's says "auto", nothing on a dead box). The grid preview and the
  sheet read Mu's none as "none (1.79)" - on the sheet its heads, the Fixed
  line, pins and the export - and the sheet sorts it at the printed 1.79,
  just ahead of an explicit 1.79, only while every image on the sheet is the
  running checkpoint's. The stored mu stays null, `none` stays the grid's
  word, the lightbox's "no Mu" is unchanged, and the Mu tips format it from
  the profile.
- Where guidance switches on is the pipeline's: `cfg_threshold()` is 1.0 with
  `true_cfg_scale`, 0.0 for a `negative_prompt` pipeline in
  `_GUIDES_FROM_ZERO`, else 1.0. The profile guesses until a load;
  `/api/status` merges the live answer, as it does `mu_supported`,
  `negative_prompt_supported`, `true_cfg_supported`, `reference_supported`,
  `refine_supported` and `lora_supported`. It also says whether
  `refine_supported` is final (`refine_settled`: a live answer, or one of
  `model_defaults._OWN_PIPELINES` - Anima Turbo/Aesthetic and NewBie, whose pipelines
  are this repo's own classes, beside which diffusers exports no sibling),
  and whether `refine_strength` is the profile's own
  (`refine_strength_profiled`) or the general default filled in for an
  unprofiled pipeline with a pass.
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
  its guide alone and is the flux profile in every number and control hint,
  `cfg_label` included. That is the one hint that is a name, "CFG
  (distilled)": `applyHints` writes it on `#cfg-label`, `#grid-cfg-label`
  and the Refine dialog's `#refine-cfg-label` (plain "CFG" where a status
  has none), and `hints_for`
  drops it where a loaded pipeline's negative-prompt answer replaces the CFG
  tip. No other profile has one: schnell's box is ignored and klein's
  guidance is baked in. `cfg_ignored` (in every status's defaults) marks a
  profile whose CFG box never reaches the model - the distilled klein,
  schnell and Qwen-Image - exactly where its CFG tip opens "Ignored"
  (`test_cfg_ignored_is_exactly_where_the_cfg_tip_says_ignored`).
  `_EXACT_PROFILES` takes precedence for pinned releases such as WAI v17
  and Anima Turbo; never match Anima with a broad `anima` token, which also
  occurs in Animagine.
- Every profile's `guide` (`backend/guides.py`) is sourced from the checkpoint's authors - the
  model card, README, prompting guide or report - never invented; the
  design notes keep the URLs. `pixels` is the range they put in writing and
  `pixels_word` what that writing was (documented, recommended, trained),
  or `example` where the authors publish no range and `pixels` is only the
  span of their own example sizes (NetaYume, NewBie, WAI). `pixels_source`
  names whose sizes those are, in the size prose's words ("the authors'
  example sizes", "the author's example", "the author's reference sizes"),
  set exactly where the word is `example`
  (`test_the_example_word_is_exactly_where_the_prose_says_no_bound`, which
  also ties the word to the size tips that say no maximum). The range must
  admit the 1 MP presets. `aspects` and `resolutions` are a card's finite
  table of shapes: Animagine's, and ERNIE's seven "Recommended Parameters"
  sizes on both ERNIE guides, where the 9:16/16:9 preset (768x1360) stands
  for 768x1376. Every listed aspect is a preset name, and every listed size
  takes one of the guide's framings by the page's nearest-aspect rule
  (`test_a_guides_aspects_are_presets_and_its_sizes_their_framings`). Its
  prose rides in `hints` as
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
  existing run id survives a widening - except a grid whose `none` value
  sat beside doubled spaces, indentation or trailing spaces, or right after
  `(`, `<tag>` or a quote: `grid.substitute` closes only the hole
  (`_close_hole`), so its none cells resolve to another prompt, and a
  Resume of a run begun before that rendered them again under a new run id.
  `/api/rerender` and `--from-image`
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
  and chip nodes are never moved. A chip whose picture fails is marked
  `.gone` and asks `GET /api/images/{id}` (`dropGoneReferences`); only
  "unknown image" forgets it (`forgetImages`, with its note in `#error`). A
  submit refused with "no gallery image with id N" for one of the payload's
  references does the same (`dropRefusedReferences`) and says "Nothing was
  queued: reference #N has been deleted, so it was taken out of the
  references. Generate again to run without it." ("Run the grid again",
  "Run the comparison again" after those). After a removal focus goes to the next ×,
  else the previous, else the emptied `#reference-list` (tabindex -1), never
  the field's badge, whose tip a focus opens. A LoRA is a bare name that
  only `loras.py` turns into a path, `.safetensors` only, and it is
  hot-swapped in `generate()`, never baked into `load()`.
  `loading.lora_refusal(model_id)` is the one id-based refusal (Anima Turbo
  and NewBie, from the loaders' `MODEL_ID`s, case-insensitive):
  `DiffusersBackend.lora_unavailable_reason` asks it after the Nunchaku
  check and ahead of the peft and loader checks, and `tools/fake_server.py`
  sets the fake's `lora_reason` from it. Whether it fits is
  `loras.mismatch` - the adapter's module names and widths against the
  denoiser's, using the pipeline's adapter reader and PEFT key conversion -
  from the checkpoint's configs on the meta device before a load
  (`loading.denoiser`) and the live pipeline
  after. That build is accelerate's `init_empty_weights`, which patches
  `Module.register_parameter` for the whole process, so it goes through
  `loading.empty_weights` and never overlaps a load: `load()` holds
  `loading.weights_loading()` throughout (its conversions inside) and waits
  out a build in progress, a build asked for during a load raises
  `LoadInProgress` and the fit goes unchecked, and builds go one at a time.
  A misfit is listed with its `detail`, hidden by the page and
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
  section ran without spilling, and its size: fp8, or NVFP4 for Qwen-Image,
  which fits the card only under Nunchaku; `tests/test_profiles.py` holds
  each to its row), reaching the page as `defaults.refine_tight_pixels` and,
  in words, `hints.refine_tight`; None, the generic 1600x1600 line, for
  every profile with no row - since 2026-10-05 every catalogued profile
  with a pass has one, so only an unprofiled pipeline gets it. A shared
  profile shares its row (FLUX.1 dev and the lite repos quote Krea dev's
  fp8 run), so the size line names the run it quotes, never "this
  checkpoint". FLUX.1's `hints.refine_large` is shown
  under Strength past it (`applyHints` clears it where a status has none).
  The preview reports where a pass of one strength starts
  (`GridPreview.start_sigma`, the dialog's "starts at N% noise"), read by
  `backend.start_sigma` and never computed by the page.
  `schedule.start_sigma` builds a copy of the loaded scheduler from its
  config, with the loaded one's static shift, and hands it what the
  sibling's call would - its module's own `calculate_shift` over the
  sequence length that call measures, with the scheduler config's base and
  max shift; its default schedule; a requested mu through `requested_shift`
  and `forced_mu`, as the pass applies it - through the module's own
  `retrieve_timesteps`, and reads `sigmas[refine.start_index * order]`. The
  loaded scheduler is never touched. `START_RECIPES` copies
  `FluxImg2ImgPipeline`'s, `QwenImageImg2ImgPipeline`'s and
  `ZImageImg2ImgPipeline`'s statements; `tests/test_refine_schedules.py`
  holds them, `SHIFT_CALL`, the default schedule and the sibling's VAE scale
  to the installed source, and every flow refine profile must have a
  recipe. The reading is None before a load, for a scheduler that is not
  `FlowMatch*`, a class not in the table, an unprofiled start index, a
  sweep, or a reading that raises (it is advice, and a preview it broke
  would refuse the pass). An open dialog's preview is asked again when the
  model becomes ready or stops being so (status.js, beside the grid
  preview's re-ask), so a load that lands with the dialog open adds the
  reading and an unload takes it away. The fake backend reads none.
  `ImageRecord.refine_missing` is read, not stored (`storage._missing_sources`,
  from `list_images`, `get_image` and `save_image`, which asks under the write
  lock, so a pass whose source was deleted while it ran lands saying so;
  `jobs.forget_images` marks a job's replayed passes), and is not in the
  reproduction set.
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
  budget stop returns through the cancel path rather than a fourth status,
  as does a shutdown's (`JobManager.stop_all("shutdown")`, from
  `shutdown()` and `Server.handle_exit`, `stopped_by` "shutdown"), which the
  page says as the server's stop, never as a cancel.
  Each skipped cell is kept in `Job.failed_cells` ({label, seed, error}),
  sent only with the records (the full snapshot). When the failure rule ends
  a job, `stopped_by` is "failures" (status still `error`, through
  `_finish`) and its error reads "Stopped at cell X of Y after N cells
  failed; K images kept. Last: <short_error>". The page's Retry ("Retry the
  N skipped cells") and Resume ("Resume the remaining N cells") send the
  job's own submitted payload (`submissions`) again with `resume: true` and
  `payload.seed ?? JobAccepted.seed` (none beside a seed list) - never the
  POST's seed over the payload's own, never the panel - each once ("Retry
  sent", "Resume sent", `aria-disabled`, focus kept). A sweep whose refine
  source has gone cannot run again: the press holds the button
  (`aria-disabled`, the sentence as its title, a press saying it again) and
  says "#N, this sweep's source, was deleted: the run cannot resume."
  (Retry: "its skipped cells cannot run again") - before sending where the
  page saw the delete (`deletedIds`), else on the server's refusal of that
  id (`startJob`'s `onRefused`).
- Each SSE stream has its own queue; `events()` opens with `done` for a job
  that has finished; the per-cell event and `/api/jobs` send
  `snapshot(images=False)`. Progress events carry `elapsed`; the page's
  estimate of the time left runs from the end of the job's first progress
  event once a second has landed (`etaFrom`; none before it: the first step
  carries a pass's VAE encode and the denoiser's move onto the card); `waiting_on`
  says `load`, or what holds the generation gate (`JobManager._holder`: a
  job's id, or `run_when_idle(fn, holder=)`'s "critic" / "expand", passed by
  the two routes, with the ticket an editor action was sent with,
  `_ticket`; `JobManager.card()` answers ("job" | "critic" | "expand" |
  None, ticket) for the status's `card_holder`/`card_ticket`): `queue`
  behind a running job, `paused` behind a paused
  one, `critic` or `expand` behind an editor action, and None for a queued
  job with nothing ahead. Every job past its load and waiting at the gate
  (`_at_gate`) is relabelled, with a status frame, whenever the label would
  change: when a job takes the gate (the second of two jobs queued during a
  critique said "critic" for the whole of the first's run), and from
  `_hold` after a pause and after a resume (`_relabel_queued`, on the event
  loop, which owns the job table). What a queued job waits on, in the
  page's words (the job ahead paused, another tab's critic or Expand, the
  first load), rides in `cellArgs` (`showCell`'s `note`), so the queue's
  redraw (`setQueue`) and a Pause press say it again rather than wiping it
  under a progress line still saying "waiting". A
  snapshot's `images` are the job's live records:
  the delete routes call `jobs.forget_images`, which takes deleted ones out
  of every job's replay and counts them in `deleted_count`; `image_count`
  stays what the job produced. Snapshots also carry `prompt` (its first 120
  characters), `refine`, `width`, `height`, `planned_images`, `grid`,
  `comparison` (`run_grid`'s `bool(prompt_variants)`) and `prompt_template`
  (a grid's prompt as submitted, its variables unfilled, from `/api/grid`;
  None for a plain request and a re-render), so a page that did not submit
  a job can name it - a grid by its own prompt, never its first cell's.
- `ModelStatus.active_jobs` (the live ids, oldest first) feeds
  `followServerJobs` on every status poll. A page watching nothing that sees
  a job it neither submitted nor watched sets `elsewhereBusy`: Unload and
  Expand are held aria-disabled with "A job started in another tab or
  window is running; Unload|Expand once it ends", and Ask the critic with
  "A job started in another tab or window is running. The critic needs the
  card to itself, so it waits for that job to finish." A press says it as a
  sentence until that job ends (`elsewhereNote`): in `#error`, and for
  Expand through `expandSay`. The topbar shows "· busy in another tab"
  beside the name while the model is ready (`#status-aside`, aria-hidden
  and outside the name's live region, hidden at 600 px and under), Generate
  stays live (the server queues it), and when the flag clears the gallery
  and the Runs picker catch up (`prependFinished(0)`, `loadRuns`) - never
  under a sheet (a run's or a search's), which would be laid out again
  whole at a moment no one chose; Sheet off resets the gallery, which takes
  the images in then. Where a job of this page's own took over while
  another tab's ran (`setBusy(true)` clears `elsewhereBusy` quietly), the
  catch-up is owed (`elsewhereCatchUp`). That job's end takes it, whatever
  the job made, a cancel before its first image included; a stream that
  gives up takes it too, and the server answering again clears it. It is
  never attached
  to: watching it would take this page's viewer. A busy page sets its queue
  rows to the server's jobs behind the watched one (`resyncQueue`, keeping
  ids it listed after the poll was asked, `queuedAt`) through `setQueue`,
  the rows' one writer, which keeps focus as a row's × does; a row this page
  did not submit reads "Another tab's job: …" ("Next, another tab's job: …"
  alone), its × "Cancel another tab's queued job: …", and Cancel queued's
  title says it takes another tab's too; a paused job's cell line counts the
  jobs waiting behind it. A job `watchNext` picks up that this page did not
  submit, while its viewer holds this session's own results (`isForeign`),
  is shown in the progress row only - no `clearResults`, frame or stand-in -
  named "Another tab's job:" in the Now row, the mirrors and the title, and
  its end, "Another tab's job finished: N images.", is said once on
  `#job-said` alone, with no Use seed and no held title: the tab's own
  unseen outcome (`titleHeld`) gets the title back. The finish line stays
  this page's own last job, Use seed and all, with its lead as it was
  (`staleBefore`), and New image is left alone (a job withdraws it only
  once its first frame clears the strip, `landed`). A
  reload's attach, with nothing of this page's in the viewer, is shown as
  before. A cancel this page did not ask for (`cancelledHere`: Cancel, a
  row's ×, Cancel queued; a POST that got no answer is not one) reads
  "Cancelled from another tab or window — …" and holds the title "Cancelled
  elsewhere". `ownJobs` and `cancelledHere` are per-tab and kept across a
  reload in sessionStorage (`TabJobIds`: `local-diffusion.own-jobs`,
  `local-diffusion.cancelled-jobs`, read once at load, the newest 50, every
  access in try/catch).
- `main.Server.handle_exit` stops the active jobs (`jobs.stop_all("shutdown")`)
  and starts the critic's stop on the signal itself. The load, the critic's `prepare()`, Check's
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
  (`prompt_encoding.measure`, no rules, no tag lookup), on the same terms
  and the same 600 ms pause, on every checkpoint. It serves the chip tray's
  token count and cut and, where no chips show, the line under the prompt
  box (`#prompt-tokens`; `syncPromptTokens` is its one writer). That line is
  hidden under three quarters of the tightest encoder's budget; from there
  it gives the count and every encoder that is as tight, and past the limit
  it is amber and names where reading stops (`tagCut`, the excerpt Check's
  cut link gives). An encoder left uncounted (a tokenizer still loading or
  unfetchable, the server not answering) is "cut unknown" in either place,
  never "no cut". The page measures the first 4000 code points
  (`PROMPT_MAX_CHARS`, tag-tokens.js, held equal to `PromptMeasureRequest`'s and
  `GenerateRequest`'s `max_length` by `tests/test_api_analysis_tags.py`). Past
  them the line leads with "Over 4000 characters: Generate and Check refuse
  it", in amber, then the count of what was counted and its cut (the chip
  tray's label says "over 4000 characters", the count of the first 4000
  after it), and the summary line says it once; a refused measure had read
  "Tokens: cut unknown" and taken the warning back. Where the encoder table has no entry
  for the checkpoint (`PromptMeasure.encoders_known` false) nothing will
  ever count the prompt, so the line stays hidden; Check's Encoders section
  says why. A measure that got no answer is asked again for the same text;
  `refreshStatus` asks again when `loadedModel` changes and when the server
  answers after an offline spell. The line is no live region: a change to
  past the limit is said once, in `#tag-summary`, and cleared on the way
  back under. It is inside `.prompt-editor`, under the box at the field's
  6 px gap, so the Example's Undo note, the Tags helper and the suggestions
  hang under it; a click on it closes the suggestions. Whether it shows
  moves everything under it in the panel by a line, so while the mouse is on
  a control at or below the prompt box's bottom (`promptTokensAimed`, set
  from the panel's pointerover) that change waits (`promptTokensHeld`) until
  the mouse leaves that control or the panel, or a key is pressed; its text
  and colour change in place. The Refine dialog checks its own request
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
  `analysis` key, never a profile key or `guide_for` (which passes on only
  the placeholder rule's patterns, as `guide.slots`, for the Examples menu);
  a `guide` cite is a verbatim substring of that guide's `prompt` prose and
  an `author` one of an example (`test_every_cite_is_the_guides_own_words`).
  An `author.placeholder` finding quotes its own slot: the entry's cite
  where it shows the slot, else the shortest example line holding it
  (`_slot_cite`; `test_each_slot_cites_the_authors_own_words`), and says
  what to put there (the entry's `fill`). One span, one finding:
  `main._checked` drops an `encoder.weighting` finding on exactly an
  `author.placeholder` span. The page's summary says characters for Han and
  kana (`PromptAnalysis.characters`, of `words`; `wordsLine`). The style
  note counts the pieces the style is read from (commas, "，", line breaks;
  lettering one piece), three or more. A check no author
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
  preserved and does not hide a genuine duplicate later in the prompt. It
  matches the loader's instruction in any case and spacing, its full stop
  optional. Where no such instruction precedes the first `<Prompt Start>`,
  and the text before the marker is one sentence (`_MORE_SENTENCES`) with no
  lettering that opens as an instruction does (`_INSTRUCTION`), that text
  and the marker are one span, worded as another workflow's instruction. An
  instruction opens "You are a/an", up to three words, then assistant,
  model, AI, generator or bot ending the opener: a stop or the end after
  it, or designed, that, which, who, built, trained, made, specialised,
  capable, for or to. A scene that opens "You are" ("You are looking at…",
  "You are a model student…", "You are an AI artist…"), or more than one
  sentence before the marker, loses the marker alone.
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
  rules read artists (Anima, `wants_artist_categories`, no cutoff), and only
  for the items `prompt_rules.lookup_items` names: a tag or hybrid prompt's
  `category_items`, or - where the rules read artists - a prose prompt's
  tag-shaped items, never a clause (no guide that states an order reads
  artists, which a test holds), capped at the chips' 64 items; any failure there is
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
  say not every tag was checked for an artist name ("artists not all
  checked"); where the lookup gave nothing
  (`PromptAnalysis.categories_unread`: unreachable, no usable reply) it says
  no tag was ("artists not checked"), and the empty line above that note
  does not repeat its reason. The artist rule reads each comma item that is
  no sentence or lettering, whatever the whole prompt reads as, and names
  several with one verb, each once ("wlop and kantoku are artists"). The
  reported `style` is `detect_style(prompt, categories)`, the one the rules
  applied.
- The chips' order slots come from that same placement.
  `POST /api/tags/validate` takes an optional `prompt` and answers
  `order_steps` and `slots`; a blank prompt, or one longer than
  `TAG_SLOT_PROMPT_MAX` (Check's 4000), counts as no prompt, never a 422
  that would leave the chips unchecked. `slots` covers every comma item of
  the prompt, past the lookup's 64 too, each with the step
  `prompt_rules.order_slots` places it in - through `_places`, the one
  placement the order rule, `reorder_tags` and `unplaced_items` read, with
  `_tag_categories`' categories for that prompt. Those come from the cache
  the chips' lookup just filled, except names past the chips' 64 that
  Check's own 64 reach (guide tags are not asked about): the validate route
  fetches those once, as a Check would, and caches misses. Offline the
  categories are None and the index is not asked again. The route asks
  `prompt_rules.reads_order` first, so a prompt it is false for costs the
  index nothing; where it is false (`order_checked` false, or fewer than
  two items) `slots` is None for the whole prompt. An item's slot is None
  where the item holds lettering or keeps more than one slot: an @ name, an
  item never looked up, or one with no usable category. A failure placing
  the slots costs the slots, never the verdicts answered beside them. The
  page never places a chip itself.
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
  installs left (`critic_runtime._is_work`): exactly `<target>-` plus tempfile's 8
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
  repeated key). On XML it never puts or keeps the guide's scene block ahead
  of a `<character_N>` block (`prompt_rules.scene_ahead`), the order the guide
  states, though Check reads no XML order. What is kept has the prompt's own pieces, each as the
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
  reordered version too long, changed (an XML candidate with the prompt's
  own words says instead that it moved words inside or between blocks,
  changed only punctuation or spacing, or changed the text outside the
  blocks - `critic_reorder._not_a_reorder`), with `<general_tags>` ahead of a
  character block ("puts", or "keeps" where the prompt already had it
  there), not Check's, regrouped, unread or flagged by the order rule - is
  said in `set_aside`. Typed Qwen3 control
  text (`<|...|>`, `<think>`, `<tool_call>`, `<tool_response>` and closers)
  reaches llama-server defused by a zero-width joiner after the `<`, which
  `validate` takes out of the answer. The rewrite is linted by layer 1 in
  the route after the gate opens (`_lint`, no tag lookup of its own: where
  the rules read artists it gets the analysis's artist categories), since a
  lint can fetch a tokenizer. `PromptCritique.rewrite_artists_checked` says
  whether that check read every artist the rewrite may name
  (`prompt_rules.artists_read`): every item a Danbooru name could be
  (`category_items`, a prose rewrite's clauses included) was among the
  analysis's `categories_asked`, excluded from the response, and no artist
  that answer named stands in the rewrite other than as a whole item the
  rule reads or after its @ ("in the style of wlop" keeps the hedge). What
  survives is still the model's suggestion; the docs
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
  px and static at or below it, where the one column is `minmax(0, 1fr)`,
  never a bare `1fr`, whose least width is the panel's min-content (a
  running job's unbroken Now line widened the page under 680 px). A tip
  that would run past the window's bottom opens upwards; one that would run
  past either side moves in by as much, 8 px short of the edge and rounded
  away from it (`placeTip`: an inline `translate` cleared at each placement,
  never the markup's `.help.end`). A badge inside a box that clips its
  overflow (`tipClipper`: the prompt check's scroller, above the results
  and docked, and the Refine dialog) hangs its tip from the window instead
  (`floatTip`, `.tip.floating`, position fixed). The tip goes under the
  badge where it fits, else over it, moved in from the window's sides, and
  is placed again on any scroll while its badge holds the pointer or the
  focus (`followFloatingTip`). It is hidden while the badge is scrolled out
  of the box (`.help.clipped`). Placed inside the check, the comparison's
  Batch tip was cut off by the check's edge and lay under its sticky head.
- A window 1800 px and wider overrides the tokens in one media query (the
  wide tier): the type ramps by `--ramp` from its 1800 px size to its 2560
  one (body 14 to 15.5 px, captions 12 to 13.5), the panel is
  `clamp(400px, 18vw, 480px)`, the cards `clamp(200px, 9vw, 260px)` (M; the
  gallery head's S is `clamp(150px, 6.5vw, 180px)` on
  `#gallery[data-density="s"]`), and the
  prompt box grows with its text from five lines to twelve. That growth rule
  is `#prompt:not([style*="height"])`, so a box dragged by its grip keeps the
  height it was dragged to; nothing may write an inline height on `#prompt`
  for any other reason. A second query, for wide windows under 1150 px tall,
  sets compact control heights of its own (a field 40 px, the primary 44, the
  small and tiny buttons 28 and 26), today's panel padding and topbar, a 12 px
  gap and an eight-line prompt cap, and leaves the Output caption the
  group's name alone, as on a two-column page under 1800 px. In the wide
  tier the tiny buttons are 30 px at a caption's type (a finding's Show,
  inline in its sentence, is the line's height instead:
  `.analysis-message > .analysis-show`), tick boxes 18 px, a tag chip's ×
  and a reference chip's × 24 px targets, reference chips up to 96 px
  badged "image N", a card's (24 px wide at the gallery's S, 32 tall) and a
  long run's tile tools and a result's row `--h-tool` (32 px), and the
  lightbox's arrow circles `--lb-nav` (52 px; 40 under 1800 px). Nothing
  changes under 1800 px, where the
  smokes run. A component's wide-tier rule goes in its own section, inside an
  `@media (min-width: 1800px)` block, and new CSS uses the tokens. Contrast
  and focus are not scale and apply at every width: `--line-strong`
  (#666680), `--accent-strong` (#6a4cf0), `--danger-strong` (#b4233a, the
  delete dialog's Delete fill), placeholders in `--muted`, and the
  page's own `:focus-visible` ring (2 px of `--accent-2`), which ships with
  `color-scheme: dark` because that turns the UA ring near black; a control
  with a focus style of its own outranks the ring. A disabled box is dimmed
  in the panel's field rows only (`.panel .sub > input:disabled`, `select`):
  under 901 px the prompt check sits in the panel, and its comparison's
  "original" box is disabled because it always runs, which its
  `aria-description` says.
- A tip lets clicks through (`.tip { pointer-events: none }`), wraps by its
  own `white-space: normal` (never inheriting a caption's or a label line's
  nowrap, which ran the sampling row's tips out past the window), and keeps
  its line breaks only where the code writing it sets `.tip.pre` (`applyHints`,
  `refreshLoras`), never because it is `.tip.wide`. Every `.help` badge has a
  static `aria-label` naming what it explains ("Steps help", "Steps axis
  help", "Grid seeds help", the dialog's "Refine size help") and
  `aria-describedby` pointing at its own tip, keeps role=note and
  tabindex=0, and stays outside its field's `<label>`, where a press ticked
  the box (Resume's). Red on hover is the `.ghost.danger` class (Cancel, a
  queue row's ×, Cancel queued, Delete unstarred, Reset), never every small
  ghost button. The page's one h1 is the word mark (`.brand`), with a
  no-width `.brand-gap` space so it reads "Local Diffusion", not
  "LocalDiffusion". The settings panel is a `<section>` named by its first
  child, an `.sr-only` h2 "Settings" (`aria-labelledby="panel-head"`).
  `#viewer` is a region named Results (`role="region"`, `aria-label`), never
  with an unseen heading before it: the docked stage's reading flow read
  such a heading after the gallery. `.sr-only` is out of the flow (no flex
  gap, no grid cell), so it is how a heading joins the panel without moving
  it, and it is never a child of `.stage`.
  A button whose words are its state (Pause / Pausing… / Resume) has no
  `aria-pressed`. A toggle with `aria-pressed` keeps one name in every
  state - "Star" for every star (the lightbox's words are "☆ Star" / "★ Star"
  with `.active`), "Reference image" for a card's ❐, "Fill width" for the
  sheet's - and its state is
  `aria-pressed`, the glyph and, in words, its title (`starTitle(on, button)`
  for every star; ❐'s "Reference image N - click to remove" or its reason,
  also its `aria-description`). A button held
  for a reason (Generate, Run grid, and Run comparison under the same
  `submitHold`, Unload, Pause, Refine, the Refine dialog's Compare 3 and From panel, Ask
  the critic, Expand, a card's ⤢ and ❐, a pass's Compare with source in the
  result column, the sheet's Fill width on a wrapped row and its export
  buttons while one runs, a negative-prompt finding's Show) is
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
  and `syncSize` resets the stop. The gallery head's two card-view pairs
  (`#card-view`: S/M and Fit/Crop, shown from 1800 px only, hidden on a sheet
  by `setSheet`) follow the same pattern: one tab stop each (the chosen
  button), `aria-pressed`, and `stepPreset` arrows. The Refine dialog's
  scale buttons (`#refine-scales`, role=group) carry `aria-pressed` beside
  `.active`; past the guard they stay `disabled` (the documented exception),
  and they are not a roving tab stop. A strip whose every preset is in range
  carries no marks (`all-guided`). While the negative box is greyed and the
  checkpoint takes a negative, `#neg-enable` ("Set True CFG" or "Set CFG"),
  after the note in the field's caption, focuses and selects the deciding box
  and writes nothing (`focusNegativeKnob`); it never goes over the box, whose
  text is what a greyed negative shows, and the greyed box keeps the pointer,
  so a carried-over negative can still be scrolled. The negative's caption
  names a cost only where it is the field's own: greyed ("(needs CFG > N —
  ~2× slower)") and under True CFG. Live where CFG is the knob, the uncond
  pass runs above the threshold whatever the box holds, so the caption reads
  "(runs at CFG > N, no extra cost)", or "(for grid cells above CFG N)"
  where only an open grid's axis reaches past the threshold; each fits on
  one line beside the label at 1400x900 and at 2048x1040 at 125%, as the
  greyed one does. The Refine dialog's negative caption uses the same words
  for the same state: "(runs at CFG > t, no extra cost)" where CFG is the
  knob and the box is live, "(needs CFG|True CFG > t — ~2× slower)" where
  it is greyed or True CFG is the knob, and "(this checkpoint takes no
  negative prompt)" where it takes none. Where the checkpoint
  takes no negative prompt at all (klein) the field is its caption alone
  (`.unsupported` hides the box; its text is kept, and the note says so).
  `#negative-field` is a div with `<label for="negative">`: a button inside a
  label becomes the label's control. `#negative` is described by `#neg-note`
  (`aria-describedby`), so a greyed box says why.
- The sampling row (`.sampling-row`) keeps each name on one line, with
  `white-space: nowrap` on the `<label>` alone - never on the label line,
  which also holds the ? badge and its tip. No box is narrower than its name
  (`min-width: min-content`), and its boxes take the counters' padding.
  Seed, Batch and Repeat form the Output group (`#output-caption`, the
  Sampling caption's twin, role=group), one line at every width, with
  `#run-hint` under the row inside the group. The seed's dice (`#btn-dice`)
  sits in the Seed name's line, as the ? badges do, at that line's height,
  its target reaching past its frame (`::after`: 6 px up - to the caption
  where it shows - 4 px down to the box, 25 px or more in all); the seed
  box therefore has the row less the two counters (156 px at 1400, for a
  16-digit seed's 121 px). The Seed field is a div with
  `<label for="seed">`, because a button inside a label becomes the label's
  control. The caption shows in one column and in the wide tier on a window
  taller than 1150 px, where the Output group is exactly as tall as the
  Sampling one. Elsewhere it is the group's name alone, visually hidden
  (absolute, 1 px, clipped): a two-column page under 1800 px and the short
  wide windows the compact controls are for (`(900px < width < 1800px),
  (min-width: 1800px) and (max-height: 1150px)`, the complement of the
  one-column query and the wide tier, fractional widths included). There
  the group is its row's height, so Generate's top is in the window at
  scroll 0 (884 at 1400x900, 16 px of it showing); smoke-wide holds
  1400x900, 1920x947 and 2048x1040@1.25.
- The size line compares the request against `guide.pixels` a tenth either
  way and quotes `pixels_word`. An `example` span reads "above/below the N
  MP of <pixels_source>", never what the checkpoint is "for", and its
  in-range presets' titles read "within the span of <pixels_source>", or
  "about the size of <pixels_source>" where the span is one size. `fmtMp`
  quotes every bound to at most two decimals, trailing zeros dropped (0.98,
  1.05, 1.57, 0.26, 2.36, 2.25, 1.64, 1.77). Where the guide has `aspects`,
  a typed size's framing is the preset aspect nearest its ratio within 3%
  (`framingOf`: 1400x800 is 7:4, not the 16:9 1.6% away); a size in a
  guided framing that is not one of `resolutions` is "not one of the
  recommended sizes", and any other framing "not a recommended aspect". The
  Examples menu is built on open from the
  last status. A pick writes through `writeBox` (tags.js) and tells the
  preview and the panel the way `choose` does, and over a prompt of the
  user's own offers an Undo (`offerPromptUndo`) in the note under the prompt
  (`#prompt-note`, `showPromptNote`). The note is an overlay on two columns,
  and in the panel's flow under 901 px and while the negative box is
  greyed - except Reset's notes (`at: 'box'`, `data-at="box"`), which lie
  in the box itself at every width. It holds one thing at a time, by
  `promptUndo.kind`, which it also
  sets as `data-kind`: 'write' (an Example's or an expansion's Undo, gone
  once the box holds anything else), 'info' (gone the same way, with no
  button), 'reset' (Reset's Undo, under Reset's own rule, lying in the
  emptied box itself), and 'running' and
  'held' (Expand's running line and a held expansion: typing does not take
  them and they have no timer, so they always take a line of the panel's
  flow, never the overlay over a negative box that may be live). Each kind
  expires after its `ms` unless focus is on its buttons, and `onDrop` runs
  whatever takes it. A Reuse's Undo and this note retire each other. While
  shown, a note may sit at the end of another box (`host`): an expansion's
  note sits at the end of its English from 1800 px.
  The menu is a role=menu named by its button (`aria-labelledby="btn-examples"`)
  whose items stay the clickable `<li>` children:
  ArrowDown, Enter or Space on the button opens it on its first item,
  arrows, Home and End walk it, Escape hands focus back to the button, and
  focus leaving closes it. Where references are supported and none is
  attached, an example matching `EDIT_PROMPT` (app.js, held equal to
  `prompt_rules.EDIT_PROMPT` by `tests/test_prompt_rules_prose.py`) says it needs a
  reference image. An example holding one of the authors' template slots
  (Z-Anime's; NewBie's anonymous character identifiers can stay as written)
  says which slots are left to fill. The slots are `ModelStatus.guide.slots`,
  the placeholder rule's patterns, read case-insensitively and held by a
  test to syntax both engines read alike. The note reads "fill in
  [character]", names two slots where there are two, and gives the first and
  a count where there are more. A pick selects the first slot
  (`templateSlots`, `setSelectionRange`, `scrollToSelection`); selecting
  writes nothing. `guide.example_notes` maps prompt text to an editing note,
  displayed alongside an author scene excerpt and never inserted into the
  prompt. Every catalogue guide has a substantive scene example; quality
  recipes remain in Tags. Preserve author wording and label excerpts.
- Prompt expansion is ERNIE-only and explicit. `syncExpandButton` is
  Expand's one writer: it is held while it runs (reading "Expand…", never
  the wider "Expanding…"), during this page's job or another tab's
  (`EXPAND_ELSEWHERE`, after busy in its reasons), during a critique, or
  with the server offline - aria-disabled, with the reason as its title, so
  a keyboard user's focus stays; otherwise the title names a first use's
  cost from `ModelStatus.prompt_expander`. A press on it held says why,
  through `expandSay`: this page's job's reason is a job note gone at the
  job's end, another tab's is said until that job ends, and while a Reuse's
  Undo holds the panel's line it is said under the prompt, never over the
  Undo. While it runs a press says what the result will do as things stand
  (`expandOutcome`: written in, held, or dropped). The line under the
  prompt (`#prompt-note`, kind 'running') says what Expand waits on, in the
  server's order - first the card, while something else holds it (another
  tab's critic or Expand, or a job another tab queued during the load:
  `ModelStatus.card_holder`); then the model's load, the rewriter's first
  download or its load, the expansion, then the translator's first
  download - beside an aria-hidden clock. The press sends a `ticket` of its
  own (`PromptExpansionRequest.ticket`), which the status answers as
  `card_ticket` while that expansion holds the card, so the page tells its
  own press from another tab's ("Waiting for the card, which another tab's
  Expand has, then …"). `syncExpandRunning` follows the status, which is
  asked 250 ms after the press and again when Expand ends, and
  `refreshStatus` asks for the line again as the holder changes. Before sending, a
  prompt past `EXPAND_MAX_CHARS` (app.js, held equal to
  `PromptExpansionRequest`'s `max_length` by `tests/test_ernie.py`), or a W
  or H that is not whole pixels in range, is refused at its box in the
  panel's words. The result is written through `rewriteBox` (Ctrl+Z gives
  back what it replaced), `promptReplaced()` follows, its English is shown
  after the write (whose input event clears one), the caret and both boxes'
  scroll go to their start, and "Prompt expanded: …" offers an Undo back to
  the replaced prompt and its English; the usual persistence and grid
  preview paths are told about the programmatic edit. The result never
  replaces a box that changed while the request was in flight: typed over
  (no `replaceBoxText` write meanwhile: `replacingBox`), it is held under
  the prompt (Use it, Discard); replaced whole by a Reset, Reuse, Example or
  Use, it goes without a word. An Example's Undo or a held expansion that
  the running line took is offered again where the expansion fails.
  Failures go to `#error` as "Expand did not finish: <reason>. The prompt is
  unchanged."; a failed load (503) is a pointer to the viewer's card where
  the card is shown (`loadFailureInViewer()`), elsewhere the sentence, once,
  with its commands in quotation marks. While a Reuse's Undo holds the
  panel's line, these are said under the prompt instead, as a 20 s note
  (`expandSay`). The English is cleared only by another expansion or a
  prompt change; it rides in `panelState()`, so Reset's and Reuse's Undo and
  a reload bring it back while the prompt is the one it translates. It sits
  under the prompt below 1800 px; from 1800 px it heads the results
  (`placeTranslation`), at its own height up to 40vh, with the expansion's
  note at its end, and beside the docked check it takes the first row of
  the columns left of it (`#prompt-translation-field`, the no-chips dock
  only, whose rules repeat the dock's selector and read "no chips" from the
  chips' host as the dock's do). Its hint (`#prompt-translation-hint`)
  describes the box.
  The critic holds what Expand holds (Generate, Run grid, Run comparison and
  the Refine dialog's Refine aria-disabled, Expand, Unload) from the click
  on, including the check it first runs on a
  changed prompt, and `startJob` refuses while it runs. Check and Check again
  are disabled during that check and re-enabled in a `finally`; if that check
  is replaced anyway the page says nothing was sent, if it is cleared (a
  Reset, another checkpoint) it says nothing, and Close pressed meanwhile
  focuses the prompt, not the disabled Check. Ask the critic is
  aria-disabled, never disabled, so it keeps focus. It is held
  (`criticHold`) while the server does not answer and while the image
  model loads (the route's 409), then for a job (this page's or another
  tab's) and for Expand, with the reason as its title and on
  `#critic-cost`; `goOffline`, the poll that answers again and a change of
  the model's state each sync it. A press says why: offline, the offline
  words (`OFFLINE_MESSAGE`), which the poll that answers takes back; during
  the load, a note taken back when the load ends (`criticLoadNote`); during
  this page's job, a job note (`showJobNote`); during another tab's job,
  said until it ends (`elsewhereNote`). A critique the server never
  answered reads "The critic did not answer: the server did not answer."
  (`critiqueOffline`) and is taken back by the status poll that answers
  again (`takeBackCriticOffline`); `critiqueFailureText` drops any closing
  stop, question mark or exclamation mark of the reason's own ("…still
  running?." came of keeping it).
  Its line is two spans:
  `#critic-progress` (aria-live polite) says only what answers a press
  (Downloading…, Thinking…), and `#critic-cost` (not live) says what a press
  costs or what holds it (no server, the load, a job). The clock
  (`#critic-clock`, aria-hidden, outside the live region) follows them in
  one run of text (`.critic-words`), never a flex item of its own, kept whole
  and joined to the last word by a word joiner in its `::before`. A size is
  binary and labelled so (`humanBytes`: KiB, MiB, GiB), as every size the
  page shows is - the topbar's VRAM and free disk too - and the CLI's
  `storage.human_bytes`; any answer, shown or
  set aside, sets `promptCritiqueReady` at once. An answer takes focus
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
  when it hides; under 901 px focus stays on Check. Escape closes the check
  as Close does (`escapeAnalysis`, app.js's keydown): after the Examples
  menu, the Tags helper and anything that took the key
  (`defaultPrevented`), never under the lightbox or the Refine dialog,
  during a composition or in a search box with text; focus inside the check
  (or on `<body>`) goes to Check (to the prompt while Check is held), focus
  elsewhere stays.
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
  them, scrolled into their box (`scrollToSelection`) and the box outlined
  for 600 ms (`flashBox`, `.box-flash`; none under prefers-reduced-motion).
  A finding or critic issue with a span does this through its Show
  (`makeSelectable`): a `.analysis-show` button after its words, described
  by them through `aria-describedby`, and titled with the box it selects in
  (the prompt, the negative prompt, or the rewrite for the rewrite's own
  findings). The item itself is no tab stop, and a click elsewhere on it
  does the same for the pointer. Show stays live while the panel is stale
  and says so when the words have moved. A negative-prompt finding's Show is
  `aria-disabled` with the reason as its title while the negative box is
  greyed (`syncNegativeShow`, from `syncNegativeApply`), and any selection
  in a greyed box (Show, a click on the finding, an encoder's cut link) says
  why in `#analysis-status` (`selectIn`). The badge says
  "references changed since" when only the references moved. A dead Use or
  Apply gives the reason in its title but stays `disabled`, not
  `aria-disabled` - the documented exception - since they die as a set while
  the stale badge and Check again, at full strength in the head, say why. A
  checkpoint change clears it. Raw negative text and effective activation
  are captured before Check and the critic await replies; inactive raw edits
  still make it stale. Apply is offered on rule findings only; an encoder
  suggestion is shown as text. In klein's references finding each "image N"
  it names is a link (`findingText`, `referenceLink`; the words unchanged)
  that focuses chip N's × and flashes the chip, or, past the last chip, says
  in `#analysis-status` that image N is not attached now; it stays live
  while the check is stale, and a press that finds its chip takes back only
  that sentence of its own (`notAttachedSaid`), never another the slot
  holds. Use and Apply write through `replaceBoxText`
  (so Ctrl+Z restores) and check again; every other whole-box write of the
  prompt or negative - the tag tools, Examples, Reuse's `fillPanel` and its
  Undo, Expand and its Undo, Reset and its Undo - goes through tags.js's
  `writeBox` (`replaceBoxText`, then no suggestion list or typing state over
  the new text), never `.value`, and Reuse's, Expand's and Reset's through
  `rewriteBox`, which focuses the box with `preventScroll` first;
  `replaceBoxText` sets `replacingBox` while it writes. Use rewrite, Use
  reordered (the critic's and Check's own) and a rule finding's Apply on the
  prompt first put the prompt they replace into an empty Your version
  (`useVersion`), with a note (a negative-prompt Apply has no Your version:
  Ctrl+Z is its way back); the order finding whose suggestion equals `reordered` says "Same as
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
  lint never asks of its own, is never called fixed but listed as possibly
  still applying ("Not flagged on the rewrite, whose check asks the tag
  index nothing of its own - these may still apply") - except an artist's @
  where the server says `rewrite_artists_checked`. The comparison has its own Batch
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
  characters it starts with (Tab otherwise moves on). Every suggestion list
  (the prompt box's, a chip editor's, the Refine dialog's) is a listbox
  named "Tag suggestions"; its options have ids, and the box's
  `aria-activedescendant` names the selected one and is dropped when none
  is selected. Validation asks about
  the first 64 items (`TAG_LOOKUP_CAP`, the request's ceiling; the order
  slots' category lookup can ask about names past it, as a Check would) and
  marks the rest unchecked. An item longer than any tag
  (`tags.TAG_NAME_MAX`, 200 characters: a hybrid prompt's sentence) is
  answered unknown (`tags.too_long_result`: `code` not_found, "longer than
  any Danbooru tag") without asking the index, reachable or not (`main._lookup_tags`,
  `tags.unchecked_results`); only a blank item or one past 4000 characters
  is refused (`TagValidationRequest`). One 213-character sentence used to
  make the whole tray read "Danbooru unreachable" (a 422). Verdicts are
  cached per checkpoint, and the
  tray's state is pending, checked or offline (amber, never red) in
  `data-state`. Each chip that the server's last answer gives a step
  carries it as `data-slot`. That answer is `tagSlots`, sent with the
  verdict request's `prompt`. `.tag-check-label[data-slot]::before` draws
  the step as a lowercase caption at the chip's start, set off from the mark
  by a 1 px `--line-strong` rule, never as text in the chip. The step is
  also in the chip's title and aria-label, and the steps form the tray
  title's legend only while some chip has one. For a text not answered yet,
  a chip keeps the last answer's step for the same tag until its own answer
  lands (`chipSlots`). The page works out no slot itself. The Refine
  dialog lists the comma items `/api/tags/validate` answers `valid: false`
  for under its Prompt box, each once, only where `tagVocabularyEnabled`,
  silent when the lookup is unavailable; it has no chips and never calls
  `writePromptTags`. Its Prompt box suggests tags there too, from tags.js's
  editor list: `scheduleEditorSuggestions` and `renderEditorSuggestions`
  with `query` set to the comma item at the caret
  (`currentTagRange(els.refinePrompt)`), `pick` to `applyRefineSuggestion`,
  `place` to `placeRefineSuggestions` and `keep`. The list is
  `#refine-tag-suggestions`, under the box, or over it where the dialog
  shows no room below for the list as drawn. The same three-character
  preselect applies, and Enter or Tab takes the selected item as an insert,
  one Ctrl+Z from what was typed. A list already up stays while the next
  key asks, nothing in it selected until the answer
  (`deselectEditorSuggestions`), and a click or a caret key that stays in
  the item the list answers (`menu._query`) asks nothing. It never touches
  `els.prompt`, `#tag-suggestions`, the chips or `tagTyping`. The chip
  editor and the dialog share `tagEditorSuggestionTimer` and
  `tagEditorSuggestionEpoch`, which is safe only because the dialog is
  modal. On a two-column page validation lives above the
  results, not in the panel; under 901 px the same node returns below Prompt.
  The prompt check's panel follows it, in that order. On a two-column page
  each has a `display: contents` host of its own in `.stage`, the chips'
  first: `#tag-chips-wide`, then `#tag-validation-wide`, which keeps the id
  the check's selectors use; under 901 px both go in
  `#tag-validation-home`, chips first. `placeTagValidation` moves each to
  its host, keeping focus and the selection (`moveBefore` where the browser
  has it, restored by hand elsewhere). Above the results below 1800 px the
  check is capped at 55vh less the chips above it (`--chips-h`; never under
  `clamp(8rem, 15vh, 12rem)`) and scrolls inside itself, its head
  (`.tag-validation-wide .analysis-head`) sticky at the scroller's top as in
  the dock; in the panel under 901 px it is uncapped. From 1800 px the
  stylesheet docks the open check beside the results instead: a `:has()`
  grid on `.stage` (`--dock-w`) puts the chips across both columns in the
  first row, then the viewer and its note in the left column and the check
  in the right beside them. This is CSS but for the chips' height
  (`--chips-h`, written on `.stage` by tags.js's `chipsResized` from a
  ResizeObserver): the DOM order is unchanged, and without `:has()` the
  check stays above the results. CSS that reads "no chips" reads the chips'
  host, `.stage:has(> #tag-chips-wide > #tag-validation[hidden]):has(>
  #tag-validation-wide > #prompt-analysis:not([hidden]))`, never
  `#tag-validation[hidden] + #prompt-analysis`, which can no longer match.
  The docked grid reads in source order (`reading-flow: source-order`) with
  the check's host after the stage's own items (`reading-order: 1` on
  `#tag-validation-wide`, 2 on `.library`), so Tab and the accessibility
  tree go: the chips, the image just made and its tools, `#viewer-note`,
  `#recent` where it docks between, the check, the gallery. The chips and
  the check need a host each because a `display: contents` host is ordered
  as one item, placed where its first child is drawn. Nothing out of the
  flow (absolute, fixed, `.sr-only`) is a child of `.stage`: a reading-flow
  container reads such a child after all its items. `reading-flow` is
  Chromium's; elsewhere the DOM order stands. The check takes the left
  column's height rather than setting it (`contain: size`, stretched, at
  most the window under the topbar and the chips), and the viewer holding
  something is then at least `min(36rem, that window)`; idle, it stays its
  one line and the gallery comes up beside the content-sized check (the
  results viewer below). The check scrolls
  inside itself, its head sticky at the scroller's very top (`top: 0`, the
  column's top padding in the head's own, so nothing scrolls past above
  it) with Close absolute in its top right corner, never wrapped onto a line
  of its own, and `focusAnalysisHeading` resets that scroll. Never `position: sticky` or a
  min-height on the column itself. The image does not move as the check
  opens, chips or none (with `#recent` docked the chips span all three
  columns). With Gallery beside (below) the check is not docked in the
  stage: every dock rule that places or sizes the check carries
  `:not(.beside)` - its grid row, `contain`, `align-self` and cap, idle or
  not - and their negations are written `:is(.beside, :not(:has(...)))`.
  The open check takes the gallery's column instead: the gallery-beside
  section gives it the column's whole box (its insets, `overflow-y: auto`,
  `overscroll-behavior: contain`, `max-height: none`) whatever the viewer
  holds, nothing made yet included, with `#library` `visibility: hidden`
  under it until Close and its scroll kept, and `.stage.beside` takes
  `reading-flow: flex-visual` while the check is open, so its fixed column
  comes after the image, its tools, the note and `#recent`. Only the
  sections' single column (`.analysis-body`) and the sticky head
  (`.tag-validation-wide .analysis-head`) are the check's in both places.
  Nothing in the stage moves as it opens, the chips included.
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
  (`#tag-summary` is, not counting the item being typed, and counting with
  `plural`: "1 prompt tag"); its children are
  only chips, the editor and + tag - the label, state and token count live in
  `data-*` attributes its `::before` reads, and an unknown chip's reason in
  the shared `#tag-tip`. A focused chip's ring is its own `:focus-visible`
  outline; no state rule may use the chip's outline (a repeat is marked on
  its text). The Tags helper drops under the prompt box (`#tags-menu` inside
  `.prompt-editor`, capped by the window alone), a click in the box moves its
  caret without closing it, and it names where the next tag goes; it inserts
  at the caret and never slots tags itself. An Escape the suggestion list or
  a chip editor took closes that alone (the document's Escape skips
  `defaultPrevented`); the next closes the helper, and the one after that an
  open prompt check. In the Refine dialog the list's Escape is prevented on
  the box's keydown, which keeps the modal dialog's cancel from firing: it
  closes the list, and the next one closes the dialog.
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
  `aria-keyshortcuts`; the lightbox's Compare with source shares that rule,
  its content S (style.css's lightbox section).
- The grid preview is the server's expansion; the page adds only
  presentation and advice. It trims factors of one other than the seeds
  from the totals and writes ×, since `grid.describe` keeps its ASCII x and
  its ones for the CLI's log. It lists the resolved prompt whenever the
  variables box has text, one included, in a `.prompts` box of its own
  after the axes line, at most eight lines tall and scrolling inside itself
  (a nested scroller, never the panel's overflow). A prompt written over
  lines is listed by the lines a `{variable}` is on, whole and numbered
  (`promptListing`: "prompts, line 4 of 29:"), which rests on
  `grid.substitute` keeping every line in place. Its client-side warning lines (a
  variable of one value, a budget that completes no seed, the panel's
  Repeat above one, cells already in the gallery) never refuse anything, and
  are appended to `#grid-preview` as nodes, never by rewriting `textContent`,
  since `.totals` is its first child. The budget line's count is the
  server's (`GridPreview.budget_seeds`, `grid.seeds_within_budget`: the
  job's stop rule, a cell starting until the deadline has passed, so a seed
  whose last cell starts in time completes); it needs the preview's
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
  first value (`0.8-1.2` needs a step). The grid's Steps and CFG
  placeholders are the profile's (`gridExamples`, from `applyDefaults`):
  Steps round(.75s), s, round(1.25s), clamped to the Steps box; CFG c-1, c,
  c+1, or c, c+1, c+2 where the profile's CFG is at or below a threshold
  that is CFG's own, not True CFG's (guidance off: Anima Turbo and ERNIE
  Turbo give 1,2,3), 0,0.5,1 for a CFG-0 profile, and nothing where
  `cfg_ignored`; Mu `none,<profile mu>` where the profile has one. The tips
  stay syntax help. A grid axis dead for the checkpoint or the installation
  reads "unavailable", and its reason is its tip and its wrapper's title -
  for Mu `hints.mu` (`syncMuWords`), for the LoRA weight `/api/loras`'
  detail once that list answers (`loraRefusal`, `syncConditioning`); the
  syntax help is kept as nodes (`gridMuTipDefault`, `gridLoraTipDefault`)
  and comes back where the axis applies.
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
  opened, saying so, naming only the knobs this checkpoint has: mu only
  where Mu is supported, as Reuse's note does; a pass's strength is
  replaced too, and
  `#refine-reused` says whose the old strength was and what Strength starts
  at - the strength last run on this checkpoint, else its own where
  `defaults.refine_strength_profiled`, else "the general starting
  strength"); the seed; the remembered scale or 1.5x where that fits
  the pass's guard and is not past the tight line (`refineTight`: the
  profile's `refine_tight_pixels`, else `sizeIsTight`), else the largest
  factor that is, else 1x - a scale passed over says so on its button
  ("(last run at 2×)" for a remembered one, "(the usual opening size)" for
  the 1.5x default; `refineSkipped`), and while W and H are the size it
  opened at the size line names the next scale up and the line it is past
  (`refineCapNote`: "1.25× and larger go past the 1536×1536 (2.36 MP)
  measured on Z-Anime at fp8", or the generic "may run tight on 16GB
  VRAM"); a run at the size it opened at in place of the scale asked for
  leaves the remembered scale alone (`refineCapped`) - unless settings a
  close left unsent on the same image and checkpoint are restored
  (`unsentRefine`, page memory only, never `local-diffusion.refine.v1`),
  which it says, with Start over; a run, another image, Reset and a
  checkpoint change forget them. It opens focused on Strength, its value
  selected. The fold's Prompt field is a div with `<label for="refine-prompt">`.
  From panel, at its caption's end, puts the panel's prompt and negative
  into the dialog's boxes (the negative first, so the prompt's Ctrl+Z is on
  top) and never writes the panel; every whole-box write into the dialog's
  prompt or negative (Apply, From panel) goes through `writeRefineBox`
  (`replaceBoxText`, then no suggestion list over the new text), which
  leaves a greyed box alone, since Chromium keeps no undo step for an edit
  to a box greyed again straight after it. While the dialog's negative is greyed
  the press copies the prompt alone and says to set True CFG (or CFG) above
  the threshold and press again, as its title says beforehand
  (`refineFromPanelPlan`); on a checkpoint with no negative prompt the
  negative is not compared. It is aria-disabled with the reason as its
  title (`syncRefineFromPanel`, from `syncRefine`) while there is nothing it
  would write: the boxes hold the panel's texts already, only a greyed
  negative differs, or the panel's prompt is empty. What a press did not do
  is said on a status line of the caption's own under the button
  (`#refine-from-note`, `sayRefineFromPanel`): empty, it has no height; it
  is scrolled into the dialog's view and cleared by `refineChanged`, an open
  and a close; it is never `#refine-error`, which sits at the dialog's foot,
  out of view with the fold open at 1400x900. `refineSource` is that record
  while the dialog is open and null once it closes. A card's or viewer
  result's ⤢ on a checkpoint without a
  pass is aria-disabled with the reason (`REFINE_NONE`, as the lightbox's
  Refine), and a press says so in `#gallery-error` and for two seconds over
  the card's tools (`refuseInCard`), opening nothing - decided in `card()`
  and `refineResult`, not in `openRefine`, whose own refusal (the lightbox's)
  is `showError(REFINE_NONE, true)`, mirrored into the lightbox; a refined
  record's Reuse opens it anyway, Refine aria-disabled with the reason, until
  status says the pipeline has one; `#refine-unavailable` hedges ("A load
  answers for certain") only while `defaults.refine_settled` is false, and
  is `REFINE_NONE`'s sentence where the answer is final - a loaded
  pipeline's, or one of `model_defaults._OWN_PIPELINES` (Anima Turbo/Aesthetic,
  NewBie). A source gone at Reuse fills the panel
  instead, its size unchanged, with a note. Versions is `num_images` on
  consecutive seeds; a Strength holding a list or a range is a `/api/grid`
  strength axis with Versions as `seeds`, opened as its sheet. A pass starts
  through `startJob(path, body, { sheet, panel: false, slot: els.refineError,
  placeholder, onAccepted })`, which never writes the panel's Seed box; the
  dialog closes in `onAccepted`, so a refusal at submit (a source deleted
  elsewhere) stays in `#refine-error` with the dialog's edits - a gone
  source in the page's words, "#N has been deleted, so it cannot be
  refined." (`refineSourceGone`, the preview's refusal and Refine's title
  too), the source line ending "#N (deleted)" - and the
  placeholder is the source's full image (`url`). While the POST is out the
  dialog holds (`refineSending`): Refine reads "Sending…", and `closeRefine`
  (Cancel, ✕), the dialog's prevented `cancel` (Escape) and the passes strip
  wait for the answer; a close the browser will not hold counts as a run in
  `refineClosed`, and a refusal after it is said in `#error`. A pass run from
  a dialog the lightbox opened puts focus on the source's gallery card where
  it is shown (`isShown`), else Generate, with `preventScroll`, through
  `handJobFocus`; any other run whose close leaves focus on
  `<body>` (an open from script, Reuse of a pass) puts it on Generate, or New
  image where that is shown, on the next frame. Every change asks `/api/grid/preview` with
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
  the 0.05 grid, half rounding up on the quotient snapped to six places
  (Z-Anime's 0.15 is 0.1,0.15,0.25), less any that leaves no step - one
  arithmetic in the same order in both, held by `tests/test_profiles.py`;
  Compare 3 only fills the box, and where one value is left out it reads
  Compare 2, its title naming the values it writes and the one left out and
  why (`compareSweep`), the label written only when it changes. Where Steps
  leaves fewer than two of them a step (a source made at 2
  steps), Compare 3 is aria-disabled, its title naming the Steps that would
  give a sweep (`compareStepsNeeded`), a press says so and writes nothing,
  the placeholder is the start alone and the sweep line names no values.
  From 1600 px the dialog is two panes (the source's PNG, the kept
  region of a reshaped size framed with `refine.fitted`'s arithmetic,
  `refineKept`), with the same DOM and tab order as the one column; Keep
  shape moves the other side on trusted input only. The dialog remembers the
  scale, the versions and, per checkpoint, a single strength the user ran
  (`local-diffusion.refine.v1`); `resetPanel` forgets it.
- `startJob(path, body, { sheet, panel, slot, placeholder, onAccepted,
  onRefused })`: `slot` (default `els.error`) is where its refusals go.
  `onAccepted(jobId)` runs once the server has taken the job, before
  startJob watches it, though a listing between jobs (`jobListing`) may
  already be streaming it; a refusal never calls it, and calls
  `onRefused(err)` once said in `slot`, for the caller's own words. `placeholder` ({src, width, height, label}) is
  keyed to the job (`jobPlaceholders`); the stream reads it again at each
  progress event until it is there, and shows it from the first such event
  with no frame as the viewer's `figure.pending` (the source dimmed and
  blurred, in the result's own tile, its caption the tile's `p.foot`) until
  a frame or the result replaces it. startJob resolves on a refusal, once
  the job has ended, or, for a job queued behind the watched one, once the
  server has taken it. A submission queues, with no `setBusy`, while the
  page watches a job or while `jobListing` is out. `jobListing` is the
  request a page between jobs makes to pick its next job (`adoptListedJob`,
  behind `watchNext` and `resumeActiveJob`); it checks `busy` again once it
  answers and opens the adopted job's stream itself. After its POST, a
  queued submission waits for that listing, then: a job the listing made
  the watched one is awaited through its stream (`jobStreams`), or
  forgotten (`forgetJob`) if that stream has already ended; otherwise it
  takes the page if nothing is watched, and else queues an id that is not
  queued already. So no job is streamed twice (`watchedJobs`, every id a
  stream was opened for) or listed as its own Next: row. A stream that gives up probes
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
  whole in its title; an XML prompt is named by its words, its
  `<name>`/`</name>` tags dropped (`promptGist`), here, in the queue rows,
  in Cancel's title and in the Runs picker's labels, the title keeping it as
  written) with Pause and Cancel at its end, the bars at the
  panel's width, and the live cell line (`#cell-note`) inside `#progress`;
  queue rows read the same, and a single queued job is one "Next:" row. A
  finished job's outcome and actions (Use seed, Show sheet, Retry/Resume) are
  the finish line, `#finish-note`, outside `#progress` (`setFinish`): only
  the next job's end replaces it, never a queued job's start; it is led by
  "Previous job:" while another runs or after a lost stream (`finishStale`).
  Its sentence, without buttons, is said once on the visually hidden polite
  `#job-said` (`sayJob`) unless `#error` says the job's outcome (a cancel,
  an error, skipped cells, a load's detail where the viewer holds images),
  and so is "Queued: <job>" when a press queues one behind the watched job.
  A job's outcome in `#error` - cancelled, its budget, an error, skipped
  cells (`showOutcome`, `jobOutcome`) - lasts as the finish line does: led by
  "Previous job: " while another runs (`#error.previous`, a generated
  `::before`; `showError` drops the class) and cleared when the next job's
  end is drawn (`finishJob`). A press on a held Pause, Unload, Expand or Ask
  the critic says why until the job ends (`showJobNote`, cleared by
  `setBusy(false)`; Expand's goes through `expandSay`). Every held press
  says its reason as a sentence with a closing stop; Pause's, Unload's and
  Expand's titles are the same reason without it (the critic's titles are
  whole sentences already). A job that
  ends with focus in `#progress` or `#queue-note` hands it to Generate and
  the finish line takes it on to its first button (`finishTakesFocus`);
  replaced with focus inside it, the finish line hands focus to its new first
  button, else Generate. A finished job's images are prepended to the
  gallery (`prependFinished`: same filters, `made` + PAGE, chained after
  `galleryPending`, dropped on an epoch change, reset where it does not join
  up or the gallery is empty, never in sheet mode; the head's count,
  `other_models`, Re-render, Delete unstarred and a gallery lightbox's
  counter are drawn as the page lands, and `prependFinished` answers the
  page's ids, so `noteOutsideView(images, coming)` says its line at once.
  The cards go in through `joinFinished`: with Gallery beside, and only
  where some cards are new, it first waits until the pointer is off the
  gallery's scrolling column (`galleryPointerGone`: the `galleryPointer`
  flag from `#library-scroll`'s pointerenter/pointerleave, as
  `panelPos.pointer`, never `:hover`, which a tap leaves behind; the column
  means its cards, gaps, scrollbar and Load more), not while the column is
  hidden under the check, released by `galleryPointerLeft` or by
  `syncBeside` turning the column off, then checks again that the page
  joins up; a column scrolled down keeps its view (`holdGalleryView`: the
  first child in its view, measured across the insert); `now` skips the
  wait), never `resetGallery`,
  drawing the day's lines as `loadGallery` does - over the first new card
  where its day is not today's, between two new cards, and between the last
  of them and the old head read off the grid's first card (`data-made`, a
  tombstone's too; none where the grid opens with a line) - and giving the
  grid's tab stop to the newest image where it was the default
  (`galleryChosen` false: set by focus arriving, cleared by `resetGallery`)
  or the old head with focus outside the grid, else `keepCurrentCard()`, as
  anything adding cards must. "New image ↑" or "↓" (`#btn-new-image`)
  appears when a job's first image or frame lands while the viewer is out of
  sight, above or below the window: `viewerOutOfSight` answers above or
  below at any width, and `syncNewImage` writes the arrow and its
  aria-label and follows a scroll that jumps past the viewer. Under 901 px
  it is a pill fixed and centred under the sticky topbar, never in the
  window's foot, where it covered Generate's right end and took a press
  meant for Generate. Its press scrolls the viewer into view and focuses the
  newest image; before one has landed, the live preview's tile (or a pass's
  stand-in), which `streamJob` hands on to the image that replaces it, or to
  Generate where nothing does; with nothing in the viewer, the viewer itself
  (tabindex -1, let go on blur) or its held line. It goes
  (`withdrawNewImage`, with focus on it handed on, never to `<body>`) once
  the viewer is in view, when a job's first frame or stand-in clears the
  strip (`landed`, never at `setBusy(true)`: a job that never reaches the
  strip, such as another tab's job picked up over this page's results,
  leaves it pointing at this page's image), when a job whose first frame
  cleared the strip ends or gives up with nothing of it in the strip,
  and - the pill only, once the job has ended - at the
  user's own press in the panel or typing in one of its boxes
  (`newImageBackToPanel`), since it lies over whatever has scrolled under
  the topbar; never at a focus arriving, which the finish line makes
  itself. The page never scrolls by itself. `setBusy(true)` hides the
  last-run banner without `remember()`, and `announceLastRun` checks `busy`
  again after its requests, so the banner never comes up over a job
  started while they were out; the done branch remembers the job. The tab's title
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
  its reasons from `submitHold()` - no server, Expand, the critic; the
  POST's own moment keeps the plain title - which has the Refine dialog's
  Refine follow through `syncRefineRun`, ahead of its own reasons, a press
  saying it in the dialog as a sentence, and the comparison's Run through
  `syncCompareRun`), Unload, Expand and Ask the critic held. A changed `started_at` writes the restart note into an empty
  `#error`. The lightbox's picture is one such slot: `els.lbImg.onerror` asks
  for the record and, where the server is not answering, writes
  `OFFLINE_MESSAGE` into `#lb-error` as an offline message, which the poll
  that answers clears before asking for the picture again
  (`retryLightboxImage`); otherwise "#N is no longer in the gallery." (an
  unknown image), "The file of #N could not be loaded." (the record exists)
  or "This image could not be loaded." (any other refusal), cleared by the
  next picture that loads. `syncUnload` is Unload's one writer (busy,
  another tab's job (`elsewhereBusy`), Expand, the critic, a load, offline):
  `aria-disabled` with the reason as its title,
  never `disabled`, and a press says why. Unload is hidden while Load is
  shown, and the press that hides one of the two hands focus to the other. A
  failed load is a card in the viewer's empty state (`syncLoadFailure`:
  heading, detail, model page link, a primary Retry load calling
  `loadModel`; `loadFailureInViewer()` is whether it is shown). `loadModel`
  takes back only what was said about the load - in the panel's line its
  own refusal, Expand's pointer (`loadLine`) or the status's sentence
  (`shownLoadDetail`), and under the prompt Expand's words about the load
  (`aboutLoad`) - never the line's other messages; a card that goes with
  focus in it hands focus to the new card's Retry load where one replaces
  it, else to Generate.
  Where the viewer holds images the detail goes to `#error`, once per
  detail, with `mirror: false`, and again after a job that ended at the load
  (`finishJob` forgets `shownLoadDetail`, since the press that queued it
  cleared the line); that job's finish line says "Not made: the model could
  not load" and where the reason is, in a `.not-made` span (`notMadeText`),
  said once on `#job-said` where the viewer's card holds the reason (the
  card is no live region; where the viewer holds images `#error` says it).
  Once the state leaves error, in every tab, `syncLoadFailure` rewords it
  ("…could not load then; it is loading now" / "it has loaded since" / "it
  is not loaded now") and takes back the detail it posted to `#error`
  (`loadDetailInError`). The card is centred in the viewer. The topbar's
  Load reads Retry load meanwhile, and the status text keeps the
  checkpoint's name in every state. That name is the start-up catalogue's
  (`ModelStatus.model_label`, by exact id, through `checkpointName`, which
  the prompt check's head, the facts' Model row, the load card, a Reuse's
  note and its ⚠ title, Expand's running line and the Refine dialog's lines
  share), or the id's last part off the catalogue, with the id in their
  titles; another checkpoint's records are named by their id's last part. The
  status text is a live region, so it is written only when it changes. At
  700 px and below, while `#job-mirror` is shown, the free-space reading
  (`.vram`) is hidden so the mirror and its load clock fit; the mirror's
  `margin-right: auto` keeps Load and Unload at the bar's end; wider, the
  free-space reading stays. While a job waits on the load, a client ticker
  writes "loading weights" with an `aria-hidden` clock span ("· m:ss") in
  `#progress-text` over an indeterminate bar; while the strip is empty (its
  idle or held `.placeholder`), `viewerWaiting` (panel.js) also puts
  "Loading weights · m:ss - the first load of a session takes a few
  minutes." in that line - the clock aria-hidden, one line, `.waiting` -
  and the line gets its own nodes back when the ticker stops. The stream's
  error handler stops the ticker. `#progress-text` is otherwise written
  through `setProgressText`, only when its spoken text changes (`tick`
  writes the ticker's words and clock itself); a single image's step count
  is an aria-hidden shown half (`.seen`) beside a visually hidden
  "denoising…" (`.said`), and the mirrors and the title read
  `progressShown()`.
- `showError(message, note, { slot, mirror, offline })` says each message in
  one slot (both are role=alert). What a gallery control did - Delete
  unstarred, a failed gallery page, Re-render's submit errors, a card's ❐,
  ⊞ and ⤢ refusals and a pass's Compare with source refusal, deletes, a
  star's failure (a card's, a viewer result's, a sheet cell's or the
  lightbox's), the sheet's Resume this run refusals, Fill width pressed on
  a wrapped row, an export pressed while one runs, Export all's outcome and
  refusals, and its export failure - is said in `#gallery-error` (`els.galleryError`), which
  `resetGallery` clears; Reuse's notes stay in the panel's `#error`.
  `#gallery-error` is the gallery head's last row (a full-width flex item
  inside `.gallery-head`), and its line is kept while it is empty
  (`#gallery-error[hidden]` is `visibility: hidden`, not `display: none`), so
  a message never moves the grid under the pointer; only one that wraps
  grows it.
- Profile defaults apply per control (`owned`); an empty Steps or CFG box is
  sent absent and 0 is a valid CFG. The panel persists to `localStorage`
  with every access in try/catch. A write is a merge (`writePanel`): only
  the keys this page changed since it last read or wrote the store
  (`panelBase`, seeded from the store by `restorePanel`) go onto what the
  store holds now - each field, the fold, the references, the translation
  and each sampling value - and an empty store takes the whole panel.
  Nothing listens to `storage`. `resetPanel` is the one thing that forgets
  it, since clearing the gallery never reaches the browser. The cards' S/M
  and Fit/Crop (`cardView`, gallery.js) belong to the gallery: they are
  stored under `local-diffusion.gallery.v1`, every access in try/catch,
  read back at load (`restoreCardView`), and Reset leaves them. Reset first
  keeps an in-memory snapshot read off the page (`panelState()`, the raw
  sampling values, `owned`, the size preset, `REFINE_KEY`'s value and
  `unsentRefine`). It empties the prompt and negative through `rewriteBox`,
  so Ctrl+Z in the prompt brings the text back, forgets the store after
  those writes and keeps it forgotten until the panel changes (`resetLeft`:
  a write of the panel as Reset left it, such as the prompt's later blur,
  is skipped). It says "Panel reset." with an Undo in the top right of the
  prompt box it has just emptied, under Reset (`#prompt-note`, kind
  'reset', `offerResetUndo`, `showPromptNote`'s `at: 'box'`,
  `data-at="box"`). This holds at every width, never in the panel's flow:
  in the flow (one column, a greyed negative) it moved Generate down a
  line, and back up under the pointer when its 20 s ran out. The box is
  empty while the note shows, since the first panel change takes it, and
  its placeholder is transparent under it. It is retired by that first
  change (`syncResetUndo` from `persistPanel`) or after 20 s. The Undo
  writes the boxes through `rewriteBox`, restores the rest (the English
  included where the prompt is its source), runs the sync functions again
  and says "Reset undone: the panel is as it was." in the same place for
  6 s (`RESET_UNDONE_MS`): an 'info' note with no button and
  `pointer-events: none` over the text it put back, which typing takes.
  Nothing either note does moves Generate. Resume is never restored, and
  nothing of it survives a reload. Before it fills, Reuse keeps
  `panelState()` (what `persistPanel` stores), the raw steps/CFG/mu/true-CFG
  values and `owned`. `fillPanel` writes a record's Mu only where
  `muSupported`, as it writes True CFG only where that is a knob; the note
  for another checkpoint's record names "steps and CFG" alone where Mu is
  greyed, and `collect()` sends no mu from a greyed box, nor `gridPayload` a
  `mu_axis` from a greyed axis, as with the LoRA weight. Its note in the
  panel's `#error` opens "Settings from
  #N loaded" (" - your prompt was replaced" where it was) with an Undo right
  after that sentence, which restores them; the rest follows: the checkpoint
  mismatch, references on a checkpoint that takes none, an adapter with
  `/api/loras`' reason. The Undo is offered only while the panel holds what
  the Reuse left (`reuseNote.after`; `syncReuseUndo` from `persistPanel` and
  `resetPanel`): the first change removes the button and keeps the text, and
  a reference dropped from what it left (deleted, or "unknown image") is no
  change. A Reset takes the words too, while they are still what the line
  says (`clearReuseNote`, `reuseSaid`): "Settings from #N loaded - your
  prompt was replaced" had stood over the emptied panel. It drops a
  restored reference only on the server's "unknown
  image". Where the page goes is `revealReused`'s: the panel is placed and
  moved (`placePanel`, `revealInPanel`), and the page scrolls only where the
  panel cannot move (one column, the page's end), and only as far as the
  panel's resting place. A stuck panel is then lifted to bring the Undo into
  the window only as far as the prompt's field stays clear of the topbar
  (`showUndo`); at 1400x900 the prompt wins. The lightbox's Reuse closes the
  lightbox first. `syncReuseButton(button, rec, compact)` takes the record
  and keeps its checkpoint, refine source, references and adapter in data
  attributes for `syncReuseButtons`. A refined record's button reads "Reuse
  pass…" (the card keeps ↻/⚠), named "Reuse pass in the Refine dialog" (the
  visible words first, for voice control; a warning suffix after) and titled
  "Open the Refine dialog on #N with this pass's size, strength and seed"; a
  pass whose source is gone
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
  `--tile-h`, `--col-w`, `.fitted`, the grid's rows and columns reserving
  every row and column the job will fill from its first frame, so a lone
  preview or stand-in holds the first image's slot and nothing moves
  sideways as the images land; sized as if the page's scrollbar were there,
  `pageScrollbarWidth()` off the width while the page has none, since a
  gallery too short to scroll takes one with a job's progress block and
  gives it back at its end - never a stable gutter on the root, which in
  Chrome shrinks every vw size on every page), refit on the next frame from a
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
  `figure.result[data-id]`, never `.card`, a group named "#N" as a card is
  (`resultFigure`), holding `thumb()`'s image and the
  card's tools: `reuseButton`, ⤢ (`refineResult`), ❐ (`toggleReference`,
  hidden where the checkpoint takes no references), a `.star` via
  `setFavorite`, and ⌦ via `deleteImage`; no ⊞. Alone in the strip, a
  result carries `.result-side` from 1800 px: the lineage chip
  (`passLineage`), Star / Refine… / Reuse / ❐ Reference / on a pass Compare
  with source (`compareButton`: the lightbox through `openViewerImage`, then
  `toggleCompare`; aria-disabled with the reason once the source has gone,
  kept by `syncCompareResult` from `viewerSourcesGone`, a press refused
  through `refuseInCard`) / Use seed and Delete (`.ghost.danger`, at the
  row's end) in words, the prompt (its line breaks kept, `white-space:
  pre-line`, at most `--prompt-cap` - twelve lines - scrolling inside
  itself, as the lightbox's `#lb-prompt` does) and `factsList(rec)`
  (without the lightbox's Use seed: the row has one); its tile's tools row
  stays, empty, at its height. With
  the check docked the side is a row under the picture in the tools row's
  place, at `--h-tool`: its actions and `.result-brief` (seed, size, steps,
  CFG; `settingsLine` in its title) - one set of words for the image just
  made either way. Compare with source is left out of the row: in a row the
  picture's width it cut a portrait pass's every word to an ellipsis, and
  in a row its buttons' width it would take another 150 px from a docked
  `#recent`; the picture's lightbox and S compare. The row is the
  picture's width, or its buttons' `max-content` where that is more,
  centred under the picture and never wider than the strip, so neither the
  picture nor the gallery moves as the check opens and a narrow portrait's
  words stay whole. The brief is left out of that measure (`contain:
  inline-size`) and takes what is left: under a 5:12 picture that is
  nothing, and the row is its actions alone (the seed is in Use seed's
  title and the image's). With Gallery beside the check takes the gallery's
  column instead, and the side stays a column.
  `#viewer-note` names a pass's source ("Refined from #N at s · W×H",
  `sourceLink` → `openRefineSource`). It names a single direct result as
  the card's caption does, "#N · W×H · seed N · Use seed" (`.seed-line`,
  hidden from 1800 px, where the column or the docked row says both). It
  keeps one line's height while empty, and a delete from the strip writes
  it again (`forgetImages` → `noteViewer`). `#recent` (from 2000 px, single
  tier, in the viewer's row only - with Gallery beside, a row of its own
  under the viewer's note, check open or not, 88 px thumbnails beside its
  heading) shows up to six `sessionRecords` and opens the
  lightbox on [current, ...sessionRecords]; with the check docked, `fitRecent`
  keeps it between the image and the check in the width the image - or the
  row under it, where that is wider - leaves at its own size
  (`.recent-docked`, `--recent-room`), hides it under `--recent-min`; a tag
  checkpoint's chips span all three columns above. The docked test is
  written twice, and a change to the dock's selector changes both, and the
  translation's rows too (`#prompt-translation-field` beside the no-chips
  dock), which repeat its selector: the CSS `:has()`, on
  `.stage:not(.beside)` (or its negation `:is(.beside, :not(:has(...)))`)
  wherever a rule lays the stage out around the check, and `fitRecent`'s
  `matchMedia('(min-width: 2000px)')` with the check open in
  `#tag-validation-wide` and `!besideOn`.
  A single viewer image opens the lightbox on `galleryRecords` from its copy
  there, else on [rec, ...sessionRecords]; `openLightbox(rec, list, origin)`'s
  `origin` of "viewer" or "recent" is where `dismissLightbox` returns focus.
  Idle, the viewer is one line, keyed on `:has(> .placeholder:not(.held))`,
  never on `viewerRecords`; `.empty-install` (`syncEmptyInstall`, from an
  unfiltered gallery page with total 0 and no `other_models`) keeps the tall
  box with an Examples button. Idle with the check docked, the viewer stays
  that line, the gallery comes up in the first column beside the check, and
  the check is as tall as what it holds up to the window under the topbar
  and the chips (`contain: none`, spanning the rows down to the gallery's
  1fr); the docked 36rem minimum is
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
- Gallery beside (`syncBeside`, gallery.js; style.css's "gallery beside"
  section; `tools/smoke-stage.json`). The tick: from 2000 px, "Gallery
  beside" in the gallery head (`#gallery-beside` in
  `label.check#beside-label`, after Sheet and before Delete unstarred,
  hidden under 2000 px, no ? badge), off by default and remembered as
  `local-diffusion.gallery-beside` through `remember`/`forget`; not the
  panel's, so Reset leaves it. While a sheet is open it stays live, and its
  title says it takes effect when the sheet closes (`syncBesideTitle`). The
  gallery's blocks sit in `#library` (the head, then `#library-scroll`,
  which holds the empty line, the grid, the sheet and Load more); unticked,
  both are `display: contents` and pass their `grid-column` on
  (`grid-column: inherit`). It applies (`besideOn`, `.stage.beside`) only
  while the tick is on, the window is 2000 px or wider (`BESIDE_MEDIA`, with
  a `change` listener) and no sheet is open. `#library` is then
  `position: fixed` under the topbar at the window's right, `--library-w`
  wide, and the stage's `margin-right` is that width plus a gutter;
  `#library-scroll` scrolls by itself (`overscroll-behavior: contain`,
  `overflow-anchor: none`, scrollbar room kept), and the page keeps its
  scrollbar's room (`html:has(.stage.beside)`). Never sticky: a sticky
  column is held only within the stage's height. In the column cards are at
  least `--thumb-min-beside` wide (200 px, the five `--h-tool` tools' room,
  set as `--thumb-min` on `.stage.beside .library`), three across at QHD and
  two at 125% (M); at the head's S, 150 px
  (`.stage.beside #gallery[data-density="s"]`), four across at QHD and
  three at 125% - the S rule's own 166 px fitted three there, and S showed
  pressed with nothing changed. The search has its own row, its tip
  hanging left; a card
  brought into view there has no topbar margin. The stage keeps its own
  layouts in the width left: the result column and the strip, and
  `#recent` as a row under the note. The prompt check, open, takes the
  column's box whatever the viewer holds, and scrolls inside it; the column
  is `visibility: hidden` under it (scroll kept; no focus or pointer reaches
  it) until Close. `setSheet` calls `syncBeside` first, so a sheet always
  opens with the page in one column, across the stage's width, and is
  landed on; Sheet off puts the column back. `armMore` re-roots the paging
  trigger on every change. A job landing: only the cards wait (the
  finish-line bullet). The gallery's keyboard rules are unchanged:
  `moveCard`'s `scrollIntoView` and a focused card scroll the column, not
  the window, and `isShown` (state.js: `checkVisibility` with
  `visibilityProperty`) keeps focus out of the hidden column.
- The lightbox (`web/lightbox.js`). `lbView` is a scale relative to fit,
  never below `min(1, oneToOneScale)` nor above 8x device pixels
  (`clampScale`), carried across records whose aspect agrees within 1%
  (`lbView.aspect`, a tolerance, never a rounded key) and clamped to the new
  record's limits. The zoom line reads "1:1", "fit · N%" or "N%", tested in
  that order. It is written only when its words change (`writeZoom`) and is
  aria-hidden; what is said is its hidden copy `#lb-zoom-said` (polite),
  written 400 ms after the last change - a wheel's burst, a resize, a step -
  or at once for a zoom a key, a button or a double-click asked for
  (`sayZoom`). `#lb-count` is likewise written only when its words change,
  and `closeLightbox` empties all three. A double-click returns to fit from any other scale, and a
  backdrop click below fit waits out a double-click (`lbCloseTimer`). F, S,
  Delete and the zoom keys skip a key with Ctrl, Meta or Alt held (the
  browser's find and zoom), and a key a lightbox control already handled
  (`defaultPrevented`: the swipe divider's arrows) is not walked. On a record
  with `refine`, S or Compare with source shows the source under the pass
  left of a divider (swipe), or side by side with each half fitted inside
  the arrows' circles (`lbPorts`, `lbNavRoom`, measured off the circle as
  drawn - read once per `layoutLightbox`, before its writes), on one
  `lbView`; switching keeps the magnification. The halves go left and
  right, or the source above the pass: stacked only where the window is
  under 901 px or the stage is taller than wide, and only where stacked
  halves fit the picture at least an eighth larger (`lbStacked(rec)`,
  `LB_STACK_GAIN`), because stacking takes the letterbox the facts use; a
  landscape stage from 901 px is always left and right. The split is
  decided once per layout, on the stage with the facts at six lines
  (`lbBaseStage`), before `lbFactsRoom`, and both use that decision; every
  reader of the ports takes a port's y as well as its x. Side by side keeps
  its name, and its title says which way a press puts the halves, before
  the press as well as after it (`lbStacked`). The layout also runs again
  on the stage's own resize (a ResizeObserver in app.js), not only the
  window's: under 901 px a line in `#lb-error` (Use seed's, a delete's) is
  a row of its own that takes height from the stage, and the picture laid
  out before had kept its height, its foot cut off under a readout saying
  "fit". The Refine dialog's
  passes strip and a pass's Compare with source in the result column open
  the lightbox with it on (`toggleCompare` after the open). The
  source is fetched by id once a session (`lbSources`) and drawn with
  `refineKept` and a clip-path. A step to another record ends the
  comparison; a re-render of the same record does not. A deleted source
  leaves the controls aria-disabled with the reason ("source #N deleted"). S
  pressed on the focused divider hands the focus to the toggle.
  `factsList(rec, { useSeed })` (`dl.facts`) is the one facts renderer, the
  viewer's column's too; the lightbox's adds Use seed to its Seed row
  (`useSeed`, said in `#lb-error` alone, never the panel's `#error`, and
  taken back once another image is drawn - `lbSeedNote`), the column's does
  not; a pass whose source has gone reads "Refined from #N
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
  where no strip shows it (another checkpoint's scope, a delete elsewhere,
  the gallery's column hidden under the prompt check with Gallery beside),
  the grid's current card, else its first, else the search box - each only
  where shown (`isShown`) - else the viewer's first image, else Generate;
  its Escape is taken in app.js's keydown before the Examples and Tags
  menus' branches - a menu left open under it (the Tags helper, after focus
  left it by keyboard) had taken it, and the dialog's own cancel closed the
  lightbox without `dismissLightbox` - and the menu stays open under the
  page; every programmatic close is `closeLightbox` alone. The arrows are
  strips only at fit, with backdrop beside the image (`--lb-gutter`, from
  the circle's reach less 2 px - 48, 60 at the wide tier - to 160 px,
  20%-80% of the height); zoomed in, below fit, disabled or with no backdrop they are
  their circles. The wheel listener is `.lb-main`'s, so a strip zooms too. A
  stage press within 500 ms of an arrow press (`lbAfterArrow`) neither closes
  nor zooms: it is the rest of a double-click on the arrow. The arrows'
  clicks are never filtered by click count. The keys line (`#lb-keys`,
  `lbKeysLine`) sits outside `#lb-meta`, names S on a pass only, binds each
  key to what it does (no-break spaces, a word joiner in +/−) so it breaks
  only after a ·, and is hidden under 901 px. There `#lb-meta` is six lines
  plus `--lb-room`, the letterbox the fitted picture (or stacked halves)
  leaves the stage with the facts at six lines; `lbFactsRoom` sets it,
  after the split is decided and first in `layoutLightbox`, and writes it
  only when it changes; its prompt is four lines there (`#lb-prompt` keeps
  its line breaks, `pre-line`, at most `--prompt-cap` elsewhere).
  lightbox.js declares only; its
  handlers (`lbImageFailed`, `lbImageLoaded`, `lbSourceFailed`, `lbWheel`,
  `lbArrowDown`, `lbPointerDown/Move/Up/Cancel`, `lbDoubleClick`,
  `lbSplitKey`) and the dialog's `close` → `closeLightbox`, only while the
  dialog is still closed (the event is queued; a close followed at once by
  an open must not shut the new lightbox), are wired in app.js.
- The gallery's cards. Each card is `role=group` named "#N" (`card()`; a
  viewer `.result` too, `resultFigure`), so its star and tools are heard as
  that image's; `tombstone` drops both attributes. Never a list or grid
  role: the day lines and tombstones are the cards' siblings. The tools are
  ↻ ⤢ ❐ ⊞ ⌦. Each button's textContent is
  its glyph (the smokes find them by it), and its short name is in
  `data-tip`, which a `::after` shows on hover and keyboard focus at the tool
  row's left edge; Delete's sits at the row's right edge, over Delete. Above
  the row, on hover and focus-within, the row's `::before` draws "#N · W×H ·
  seed N" from its `data-n`/`data-size`/`data-seed` (never `data-id`, which
  `forgetImages` queries), with empty alt text because the image's label
  already says it; it is hidden while a tool's name shows, and dropped
  while a card note is up (`.card:has(> .card-note) .tools::before`), as
  the names are. The row's children stay its five buttons, which the
  smokes' `nth-child` selectors rely on. The
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
  when it shows that record. F passes `{ say: true }` (`galleryKey`,
  `sheetKey` through `starCell`, and the lightbox's keydown in app.js):
  where focus is not on a `.star` or `#lb-fav`, `sayStar` writes "Starred
  #N" / "Unstarred #N" into a visually hidden polite line - `#star-said`
  under the gallery's head, or `#lb-star-said` inside the lightbox while it
  is up, since the page behind a modal dialog is inert - emptied first,
  written on the next frame, and emptied again after 5 s. A star answered
  "unknown image" (deleted in another tab) does what ⌦ does:
  `forgetImages`, then "#N was already deleted." in `#gallery-error`, or
  through `noteLightboxDelete` while the lightbox is open, and the Runs
  picker is asked again (`loadRuns`); its other failures go to
  `#gallery-error`. A pass is `.card.refined` with `data-strength`
  (the badge is the sheet's `::before` rule). Thumbnails are
  `object-fit: contain`, or `cover` from 1800 px where the head's Crop is
  chosen (`#gallery[data-fit="cover"]`); below 1800 px neither choice
  applies. `thumb(rec, list, src, open)` labels every image "Open #N:
  <prompt>"; a sheet cell is labelled instead by what the sheet lays out
  ("Open #N · Steps 2 · Seed 7", a version by its name), set in
  `renderSheet`. `thumb` titles the image with `cardTitle(rec)` (#N · W×H ·
  seed N · a pass's source, "(deleted)" after a gone one, then the prompt)
  through `titleImage`, whose `aria-description` is the title's first line
  less any fact the name already says, so a sheet cell is described by its
  size and source alone; `renderSheet` calls `titleImage` again after
  naming a cell, and `forgetImages` retitles through it. This holds on the
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
  the client, and the export is client-side, not a route. Export PNG draws
  the slice on screen. Export all, shown only where something is pinned,
  draws every pinned slice that has images (`sliceCells`): it stacks them in
  one PNG, each titled by its pins, at a cell width that keeps the stack
  within `EXPORT_MAX_EDGE` and never below 96 px, named
  `sheet-<run>-every-<axes>.png`; past that edge it saves one image per
  slice, named by its pins with -2, -3 where two come out alike, at most
  `EXPORT_FILES_MAX` (24), and says so in `#gallery-error`. A fitted sheet
  exports thumbnails, cells up to 384 px; a filled one exports at the larger
  of its cells' device pixels on screen (`exportLayout()`'s `shown` × `dpr`)
  and the thumbnail's width, so never smaller than the fitted export, from
  the PNG where that is wider than the thumbnail. Every export draws from
  `exportLayout()`'s snapshot (the layout, its axes and `sheetWords()`: the
  prompt labels, the adapter and the fixed Mu), never the live
  `sheetAxesList`/`promptShort`/`sheetAdapter`/`sheetMuStatic`, so a sheet redrawn while
  Export all writes its files changes none of them. Both export buttons are
  aria-disabled with the reason while one runs. A failure is said in
  `#gallery-error` ("The sheet could not be exported: …"), and the smokes
  stand in for `saveCanvas`. `sheetFileName` drops a hyphen that a value's
  closing mark ("none (1.79)", a clipped label's "…") left before the next
  pin or `.png` (`sheet-<run>-mu-none-1.79.png`). The sheet asks for
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
  Laid out again (`renderSheet`: a page landing, a delete, a pin, an axis, a
  wrap that flips), the sheet keeps the keyboard where it was.
  `sheetFocusPlace` records the place before anything is removed
  (`forgetImages` takes it before it drops the cells), and
  `restoreSheetFocus` returns to it, with `preventScroll`, unless focus was
  kept or moved on purpose meanwhile: the same cell's image or star by id;
  else the image now at its place in reading order, else the last; a
  control by its name (a select by its label, a button by its words), else
  the control at its index; the search box where nothing is left. F on a
  cell another tab had deleted, and an ArrowDown on Columns, used to drop
  focus to `<body>`.
- Every sheet the page opens (a comparison, a refine sweep, Runs, a card's ⊞,
  both branches of Show run) goes through `showSheetScope`. It keeps the
  search it replaced (`sheetReturn`), which Sheet off restores while the box
  still holds what the page wrote. It asks `setSheet(true, { land, asker })`,
  which, once the sheet is drawn and only if it is still the one shown,
  scrolls the gallery head under the topbar (`landOnSheet`),
  reserving the room under the sheet its head needs to reach the topbar
  (`reserveSheetRoom`: a margin outside the box `fitSheet` measures, from
  where `main` ends - never the document's scroll height, which is at least
  the window - cleared by `setSheet`). Once that scroll settles
  (`scrollSettled`: three frames at one scrollY, 1.5 s at most),
  `focusSheet` moves focus with `preventScroll` to the sheet's first cell,
  else the search box, in two cases: focus is still on the control that
  asked and that control is not wholly in sight (`wholeInSight`: inside the
  window under the topbar and inside every ancestor that clips its
  overflow, such as the docked check's scroller at 2560), or focus has
  fallen out of the page (`<body>`, or hidden: a card's ⊞ under the sheet).
  `asker` is the element focused at the call; a job's sheet is given the
  element pressed at submission, kept in `submissions`. Where the page
  itself moves focus off a job's asker (the Refine dialog closing on its
  run, `refineClosed`, the strip a job's first frame empties), it does so
  through `handJobFocus`, which passes that submission's asker on to the new
  target. Focus anywhere else is never moved. Every sheet `setSheet(true)`
  opens is said in `#sheet-said` (polite, outside `#sheet`: "Sheet: 6
  images, Steps × Seed", via `saySheet`), written on the next frame after an
  empty line. A refine sweep's source cell is
  cropped to the passes' shape (`object-fit: cover`, `refine.fitted`'s
  centre crop), captioned "cropped as the passes were" where the shapes
  differ. The Sheet button does not scroll - except where the sheet takes
  the page out of Gallery beside, which `setSheet` lands on; over an empty
  search with several runs it opens the newest run, with a `sheetNotice`,
  and a typed search is never redirected. A matrix's cells are capped to
  the window's height (`--cell-max` from `fitSheet`'s measured
  `--sheet-chrome`) with a 96 px floor. With Fill width pressed they take
  the width instead: `.sheet.fill-width` sets the template's `--cell-w` to
  `1fr`, never through the class `.fill`, which is the progress bar's. The
  toggle keeps one name, with its state in `aria-pressed` and its title; it
  is remembered under `local-diffusion.sheet.v1` through
  `remembered`/`remember`, is not forgotten by Reset, and is aria-disabled
  with its reason on a wrapped row. Its heads stick inside `.sheet-scroll`, whose scroll
  padding is the measured head row and row-head column (`--sheet-head`,
  `--sheet-rowhead` on `.sheet`) so a keyboard-focused cell is never under
  them. `refitSheet` (debounced, on the window's resize) measures again and
  re-lays out when the wrap test (`sheetWraps`) flips. A row-less sheet wider
  than fits wraps with captions, and so does its export. Head labels are
  whole on screen (span.label, clamped, title = full value). A number of
  five digits or more goes to a line of its own whole (`numberLabel`: an
  inline-block `.number`); it breaks only where it alone is wider than its
  column, and then only between groups of four digits counted from its end
  (a `<wbr>` between them), in balanced lines (`text-wrap: balance`); the
  text is still the whole number. `clip` is for pins, the note and the
  canvas. A matrix is a role=table named by `sheetSummary`; its rows are
  `.sheet-row` with role=row and `display: contents`, so heads and cells
  stay the grid's items and the heads stick; its heads are columnheader and
  rowheader (the corner is a cell), and its cells role=cell (the source
  with `aria-rowspan`). A wrapped row is a role=list of listitems. Prompt
  labels that repeat or exceed
  `LABEL_CHARS` fall back to `promptEdit` against the oldest prompt, up to
  `EDIT_LABELS_MAX` prompts, memoised per prompt list. A version that only
  puts the original's top-level XML blocks in another order, with nothing
  in or between them changed (the loose text compared gap by gap), reads
  "same blocks, moved: `<general_tags>` first" (`blockMove`, over
  `xmlBlocks`); its `XML_BLOCK` is `critic_reorder._XML_BLOCK` with Python's Unicode
  `\w` written out, held equal by `tests/test_critic_reorder.py`. One block's move
  reads "first", "last", or "before"/"after" another; anything else reads
  "in the order …", cut between two tags within 80 characters (`tagLine`,
  `EDIT_CHARS`). The edit fallback labels such a prompt by its move too,
  and the oldest as "blocks `<a> <b> <c>`" where every other prompt is a
  move. A version that only reorders the original's words reads "same
  words, moved: …" (`versionEdit`). A refine sweep's shared source is a leading column
  fetched by id (`sheetSources`) and never a sheet record, axis value or
  count; `forgetImages` nulls it. Records that pass the pins but find their
  cell taken are counted in the note, repeats of the cell's image
  (`sameSettings`: every axis, size, checkpoint, quantization, refined
  source) apart from images that differ in something the sheet does not lay
  out; the note reads "N of M loaded" only when none collide. Where the
  Prompt axis varies and every image has a seed of its own (`ownSeeds`),
  the note adds `OWN_SEEDS` ("each image has its own seed: Check's Compare
  on the same seeds puts versions side by side") ahead of a `sheetNotice`.
  The shared settings go in `#sheet-fixed`, never the note. The starred
  cell uses the card's own `.fav` and `.fav-badge` rules (`.card.fav,
  .sheet-cell.fav`, `.sheet-cell .fav-badge`). Every cell has the card's
  star button: tabindex -1, so no tab stop; shown on hover, focus-within
  and a coarse pointer, and always when starred; drawn by `syncStar`. That
  button and F on a focused cell both star through `starCell` →
  `setFavorite` (F said in `#star-said`). A `.sheet-cell` is `isolation:
  isolate`, so its badges stay under the stuck heads. "Resume this run" (one run of the running
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
  because its count must be the grid's. Re-render's refusal of another
  checkpoint's images (`main._foreign_refusal`) advises unticking All
  models; it offers `model:<the whole running id>` only where that id is
  contained in none of the foreign ids selected (`model:Anima` is
  Animagine too), and its verb agrees with the count. The Runs picker asks
  `/api/runs?limit=200` and groups the result: "Grids and batches" (images > 1
  or jobs > 1, labelled with count, distinct, "N prompts" and what varied -
  `runAxes` over `RunSummary.axes` (each setting's count of distinct values
  where above one, from `storage._RUN_AXES`), `refine` and `strengths`:
  "steps×CFG 3×2", "steps ×8 · 5 seeds", "strength 0.05-0.15 of #37",
  "comparison · 3 versions"), then at most
  eight "Single images" by time to the second and seed. It names the run the
  search box shows (`syncRunList`, from `resetGallery` and `loadRuns`; its
  "Runs…" option is hidden and disabled), and picking one never resets it.
  A pointer pick, or Alt+Down then Enter, opens the run at once
  (`runListChange`). A change made by a browsing key on the closed picker
  is only noted: arrows, Home, End, Page Up/Down, or a printable type-ahead
  key, none with Alt, Ctrl or Meta (`runListKey` marks only the change that
  key's own default makes). The noted run (`runPending`, which
  `syncRunList` keeps through a rebuilt list and retitles from) opens on
  Enter or on the picker's blur (`commitRunList`); Escape puts back the run
  shown. Closed, it is `--runs-w` wide (190 px; `clamp(240px, 14vw, 360px)`
  from 1800 px), cut with an ellipsis, and `syncRunList` puts the picked
  run's whole label in its title ("Recent runs" otherwise). A label's
  prompt is `promptGist`'s, one line with an XML prompt's tags dropped,
  before `clip`.
- Search terms are `key:value` predicates plus FTS text; `date:` is a local
  day, `mu:none` is `IS NULL`, `id:` is the gallery id the page shows as #N
  (a range is a span), and `search.strict_scope` is the reading for
  `clear --where`, `keep_rates` and the gallery's Delete unstarred
  (`GET`/`DELETE /api/gallery/unstarred`): `clear --where <box>
  --keep-favorites` over HTTP, the search box strict or empty, gated on the
  count typed back and pinned to it as `expected`, which `delete_matching`
  checks under the write lock and answers 409 on a mismatch - adding that a
  job is still running and may add more where `jobs.busy()`; rows and files
  only, no orphan sweep. Outside the sheet, the page first brings the grid
  up to what the sweep reaches with `prependFinished(0, { now: true })`
  (the same filters, without waiting for the pointer to leave the gallery's
  column), then asks the preview with `newer_than`, the grid's head before
  that. The preview's `newer` counts what the delete takes above that id,
  by the same `_UNKEPT` predicate in the same scope, and becomes the
  dialog's "N of them were made since this page last loaded; they are at
  the top of the grid now" (`freshLine`; "the newest are" where the reload
  from the top left more pages); the page never counts these itself,
  because it cannot see which ones a starred image keeps. The dialog's
  running-job line is the preview's `job_active` (`jobs.busy()`), never the
  page's `busy`, which missed another tab's job. A scope with no image left
  at all says "Nothing to delete: no image is left <scope>.", not that every
  image is starred, and where a reload hides the focused button - nothing
  left, a sweep that took every image, a 409, a failed preview - focus goes
  to the search box (`keepUnstarredFocus`), never `<body>`.
  `keep_favorites` keeps the stars and what a starred
  row was made from - its `refine` source and its `reference` ids - through
  one predicate (`storage._UNKEPT`) that `delete_matching`, `clear_history`
  and the counts share; `matching_preview` returns `sources_kept`, and the
  route's `unstarred` and both CLI counts are `records - favorites -
  sources_kept`, so the typed count is the delete's. A 409 reloads the
  gallery and the Runs picker. `forgetImages` is the one place the page
  drops deleted ids - every list, the sheet, the panel's references and a
  Reuse's Undo snapshot (it answers the sentence the caller's note adds) -
  and marks at once the passes of what it deleted (records, card titles,
  every Reuse, the lightbox's facts and `lbSources`); it keeps every id it
  has seen deleted (`deletedIds`), and the stream marks a pass arriving
  after its source went (`markGoneSource`, in `show`). A single card or viewer
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
  by `placeAsk`, inside `documentElement`'s clientWidth/clientHeight -
  `innerWidth` counts the scrollbar), never `window.confirm` or `window.prompt`: Cancel focused,
  or the count's box; Escape and a press outside are Cancel; while it is up
  the document's keys do nothing (its Escape closed the lightbox under it).
  The smokes stand in for it by assigning `askConfirm`/`askCount` and put
  back `confirmDialog`/`countDialog`; `smoke-delete.json` drives the dialog
  itself. A delete under way ignores a second
  press of the same image (`deleting`), and `storage.delete_image` looks up
  and deletes under `_write_lock`, so a second delete is "unknown image". The
  gallery pages by id, never by offset,
  and pages itself as "Load more" scrolls near the viewport, or near the
  foot of the gallery's own column with Gallery beside (`moreWatcher`, made
  by `armMore` with its root where the gallery scrolls - the window or
  `#library-scroll` - and made again when that changes; re-armed by
  `watchMore` after each page that lands, never after one that failed - the
  button is the retry - and `moreArmed` keeps a trigger made again
  meanwhile from fetching).

## Adding a backend

Implement `GenerationBackend` (`load`/`unload`/`is_ready`/`generate`/`describe`)
and register it in `backend/backends/__init__.py:get_backend`. `generate()` is
handed a `should_cancel` predicate and must poll it at every step boundary,
raising `Cancelled` when it answers True. Optionally, `start_sigma(request)`
answers the Refine preview's start sigma for a pass as `applied()` runs it,
and `prompt_expander_status()` what an expansion would wait on and download
(file checks only, for the status poll); both default to None. Nothing in
the HTTP layer or the UI needs to change.

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
diffusers, Anima's and NewBie's answer final before a load
(`_OWN_PIPELINES`), `CARRIED` held to `generate()`'s own kwargs, a stub
sibling built over a stub pipeline's components and called once per image
with that image's generator, the cancel paths, mu through the shared
scheduler, the row, the PNG, the key, the migration, the reindex and the
merge), `tests/test_refine_schedules.py` (each profile's start index
against its sibling's real `get_timesteps`, the duplicate advisory, and the
start sigma: each `START_RECIPES` entry, `SHIFT_CALL`, the default schedule
and the VAE scale held to the installed source, verified.md's FLUX.1
sigmas reproduced on a real `FlowMatchEulerDiscreteScheduler` with and
without Mu, a static shift with and without a requested mu, the loaded
scheduler left untouched, None where it would be a guess),
`tests/test_api_refine.py` (the status flag, whether it is final
(`refine_settled`) and whose the strength is (`refine_strength_profiled`),
the submit refusals, the strength axis with resume, the pass's own size
guard and its lack of a batch budget, `GET /api/images/{id}`, Re-render,
and the preview's `start_sigma`, asked with the applied pass, not for a
sweep or a direct render, end to end over a real FLUX.1 schedule) and the
CLI's flags in `tests/test_cli.py`. What a pass does to pixels, its time and its
memory are the card's to say, in `docs/verified.md`.
`tests/test_reindex_provenance.py` covers foreign PNG conditioning ids, lost
databases, missing or mismatched source identities and safe in-place metadata
remapping. API tests hold a fake before its first callback to check initial
per-cell progress totals, and API/CLI tests cover axis replacements and the
capability refusal both before and after loading. The job snapshot's
`comparison` is asserted true for a prompt-variants grid (`test_api_grid`)
and false for plain and variable grids (`test_api_jobs`). What holds the
card, and whose expansion it is, is
`tests/test_jobs.py::test_the_card_names_its_holder_and_an_editor_actions_ticket`
and, through the status,
`tests/test_api.py::test_status_names_what_holds_the_card_and_whose_expansion`.

The prompt analyser is tested the same way, with no network and no GPU:
`test_prompt_encoding.py` (budgets, cuts, templates held to the
installed diffusers source; its real-tokenizer cases skip when nothing is
cached), `test_prompt_rules*.py` (the `analysis` data against the prose, the
authors' examples raising no warning, the chips' order slots against
`reorder_tags` and the order finding's step labels, and `reads_order`
wherever the slots are None), `test_critic*.py` (pins, install,
server command and stop, the server's count and the fit checks, answer
validation, grounding, and `web/sheet.js`'s copy of `_XML_BLOCK`) and
`test_api_analysis*.py` for the routes, the chips' order slots on
`/api/tags/validate` (offline, past the 64, a prompt past Check's limit, no
category lookup where no order is read, a fault in them leaving the
verdicts), an item longer than any tag answered unknown without asking
the index (`test_an_item_longer_than_any_tag_is_unknown_without_asking`),
the page's `PROMPT_MAX_CHARS` held to every request's `max_length`
(`test_the_pages_measured_length_is_every_requests`), the gate
(Check beside a running job, off the loop; the critique's re-check after
`prepare()`), the status flags (status never loads a tokenizer) and
shutdown (the lifespan's order, `main.Server.handle_exit`). Where the point
is the real thing, `test_critic_process.py` and `test_critic_server.py`
start Python children and loopback servers: a wrapped stand-in llama-server's process tree, its listener and
its counts, and the proxy bypass. The job-object cases run on Windows only;
the wildcard-listener and half-exited-process cases run on Linux only. Its
real-tokenizer case in `test_critic_tokens.py`,
`test_measured_counts_and_pieces_are_the_tokenizers`, skips when klein's tokenizer is not cached. The `fake` fixture installs
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
slow or gated, `POST /api/fake/step` (`{seconds}`, answering the pace it
replaced) slows the jobs after it, for a step that needs a job to outlast its
own round trips - a cancel, a stream attaching - and `POST /api/fake/expand`
(`{seconds, fail, hold, fresh}`) does one of four things to ERNIE's Expand
on the server: slows it, holds it until the next call (30 s at most), makes
it fail with a RuntimeError of that message, or puts the rewriter and
translator back off disk so a first use shows again on a server already
used; the fake's rewriter and translator count as downloaded from their
first use), and `node tools/browser.mjs tools/smoke.json` drives headless
Edge or Chrome over the DevTools protocol with Node's own WebSocket - no npm
install - through the lightbox zoom and the focus its close leaves on the
walked-to card, the sheet, the export, the sheet's declared variable order
(under Starred too), a sparse comparison's rows, the re-render button, the
panel's persistence (a write merged over another tab's, an empty store
taking the whole panel) and the duplicate/distinct counts, plain CFG labels
on Turbo, and Turbo's woken negative caption and its grid examples (6,8,10
/ 0,0.5,1, Mu none,1.15), exiting 1 on a failed `expect` - one that
throws, or whose promise rejects, included - or on a DevTools call the
browser leaves unanswered for five minutes, or a connection it closes with
one pending, printing the step it was on, and 2 on a script it refuses
before starting - among them an
`expect` joining an async helper (`__until`, or a `window.__x = async` the
file defines) to more with && or ||, which is never awaited: the whole
condition goes inside the predicate. A `hover` or `click` step takes a
CSS selector in place of a point, aiming at the middle of that element's box
as it is then. It opens the page at 1400x900,
where the smokes are written: `LOCAL_DIFFUSION_VIEWPORT=2560x1305` (a
maximized QHD browser) and `LOCAL_DIFFUSION_DPR=1.25` (with 2048x1040, the
same screen at 125% scaling) move it, a `viewport` step resizes it mid-run,
a `media` step emulates media features (`{"media":
{"prefers-reduced-motion": "reduce"}}`; "" drops one's override), and
`LOCAL_DIFFUSION_PORT_OFFSET=N` adds N to every `127.0.0.1:<port>` in
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
its tip, a weight sweep (its preview naming the adapter), the prompt
check going stale when the references change (chips, a reorder, Reuse),
the references finding's "image N" links focusing and flashing that chip
("not attached" past the last, taken back by a press that finds its chip,
another sentence in the slot left alone), a gone chip kept while the server
does not answer and dropped on its "unknown image", a reference deleted in
another tab dropped by the Generate it refuses, and one remembered across a
reload, and the distilled klein's grid examples (steps 3,4,5, no CFG sweep);
`smoke-flux.json` runs against a second fake server
claiming FLUX.1 (`--port 7862 --model black-forest-labs/FLUX.1-Krea-dev`)
for the True CFG box, the "CFG (distilled)" labels, with the sampling row's
names whole, Mu's auto unclipped and each of the row's ? tips wrapping
inside its box and the window, the lightbox's Sampling ("True CFG 1.5 · no
Mu") and Negative rows, an Example's Undo note in the panel's flow while
the negative is greyed and over its caption while it is live, and the size
line's exact 1.64 MP bound, and
`smoke-preview.json` against one slowed to `--step 0.3` on 7863 for the live
preview, the tab's title and the topbar line during the job, and the
finished title and the finish line's seed after it. `smoke-ernie.json` runs
against an ERNIE-claiming fake on 7865 for the explicit prompt expansion
flow: Expand aria-disabled with its reason and focus kept, "Expand…"
keeping the caption on one line, and a press while it runs saying what the
result will do (written in, held, dropped); the first-use download and the
model's load named under the prompt with a clock, and the page's own status
requests (250 ms after the press, and at the end) counted by a spy; the
note's Undo and Ctrl+Z, and an Example over typed text; an expansion typed
over held (Use it, Discard), with the running line and the held note in the
panel's flow above a live negative; a Reset and a Reuse during one keeping
their Undo; a failure said as Expand's with the English kept, and said
under the prompt beside a Reuse's Undo, an Example's Undo or a held
expansion the running line took offered again; a size, and a prompt past
4000 characters, refused at its box; a gated load's pointer to the card,
and its sentence once beside images; the hold during a job, and Generate's
and Run grid's titles while Expand holds them (the request held on a
promise the smoke releases); Expand held aria-disabled with the reason by
another tab's paused job, keeping focus, a press saying why until the job
ends; the English in the panel at 1400x900, at the head of the results with
its note and Generate in the window at 2048x1040@1.25, and beside the
docked check at 2560x1305; Baidu's seven sizes (the seven framings marked,
12:5 named, a typed 768x1376 recognised, 1008x1008 not one of the sizes,
and 1400x800 and 784x1360 read by the nearest framing, 7:4 and 4:7); and
the check's character counts and tag-list note on a fullwidth-comma and a
one-per-line list; Expand pressed during this page's job under a Reuse's
Undo, said under the prompt with the Undo kept, and the critic's press then
a muted note gone at the job's end; and an Expand waiting behind another
tab's (a request held on the server, `card_holder` "expand") leading its
running line with the wait for the card. The fake takes the claimed profile's threshold, true-CFG
flag and `mu_supported` (a Mu dropped from the row where the schedule takes
none), its model id's prompt-expansion capability, and
`loading.lora_refusal` for the id (Anima Turbo/Aesthetic and NewBie refuse adapters,
422 at submit), so it behaves as the real one would.
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
alone, Anima's artist-check notes (no tag checked, not every tag), the
chips' order steps matching Check's order finding and Use reordered, a
guide tag past the 64 keeping its step, the recommended-negative Apply going
dead once the negative box is edited, a negative-prompt finding's Show,
held with the reason while CFG greys the box, saying why when pressed, and
waking with it, the order finding's Show beside "Same as Use reordered
above." with no Apply, a typed 1232x832 not one of the recommended sizes,
the 0.98 MP floor as written, the live negative's caption, the dead grid Mu
axis quoting the Mu tip beside the live LoRA weight axis, and the grid
examples 21,28,35 / 4,5,6.
`smoke-tags.json` uses an Animagine-claiming fake on 7878 for the chips'
verdict cache, typing state and live summary, an accepted suggestion's undo,
category edges, repeats and the focus ring on one, the shared tip and the
after-cutoff mark, + tag, gap hover (waited for, as a late tray render
redraws it) and click (none on the label or hint),
the Tags helper under the prompt and whole, Escape closing the suggestions
before the helper, and the open prompt check after both, the Examples
menu's keyboard (ArrowDown, Enter), its Undo note (on its own line at
800 px), the needs-a-reference note, the file-drop guard, the singular
summary ("1 prompt tag", checked and offline), a chip editor's
aria-activedescendant and option ids, the suggestion lists' and the
Examples menu's names, a held panel not moving for chip edits at
1400x900, Escape in a lightbox opened from the keyboard over a Tags helper
left open going to the lightbox (focus on the walked-to card, the helper
kept until the next Escape), and each chip's order step from the server:
caption, title, label and the tray's legend; none for an unknown tag, an @
name or lettering; kept on its tag across a move while the answer is held;
none where a line break stands the order down; every chip one line at
1400x900.
`smoke-illustrious.json` uses an Illustrious-claiming fake on 7867 for hybrid
prompting, the native 1-2.25 MP size range, 4 MP warning against the 2.25
MP bound and disabled Mu.
`smoke-z-anime.json` uses a Z-Anime-claiming fake on 7868 for Base defaults,
natural-language prompting, negative guidance, enabled Mu, the documented
4 MP ceiling, the complete scene example and two templates' fill-in notes and slot selection,
the missing-negative note and a whole-item slot reported once; the fixed
shift's 1.79 in the Mu box, the grid preview and a Mu sweep's sheet (none
sorted between 1 and 2.5, just ahead of an explicit 1.79, in the Fixed line
and the export's file name), the profile's grid examples, the negative's
live, greyed and grid captions each on one line, and the 0.26 MP floor;
and the Refine dialog: the author's 1024x1536 opening at 1x with why, the
0.1,0.15,0.25 sweep and no sideways scroll.
`smoke-paging.json` uses a fresh Krea-claiming fake on 7867 for the gallery
paging itself on scroll (it builds a 60-image grid, so it holds once per
gallery). Run it separately from the Illustrious smoke on the same port:

```powershell
uv run python tools/fake_server.py --port 7867 --fresh
node tools/browser.mjs tools/smoke-paging.json
```

`smoke-anima.json` checks Anima Turbo defaults, its documented size range,
disabled Mu, applied saved settings, the artist check on the card's
tags-then-sentences shape (Apply, and two artists named together; the fake
index knows kantoku and wlop as artists) and the rewrite's artist check, a
FLUX.1 pass reused there (the final no-pass line, its strength said as
FLUX.1's and 0.4 as the general starting strength), the refused adapter
(greyed select with the reason, a 422 at submit, no LoRA on the row), the
grid's dead Mu and LoRA weight axes with their reasons and their syntax
back when they apply, a Reuse writing no Mu into the dead box nor sending
one, a Mu dropped from the row, the two-sentence Example passing Check, the
1, 1.5 and 2 MP tiers marked, the 0.26-2.36 MP bounds as written, and the
grid's 1,2,3 CFG example, on port 7869:

```powershell
uv run python tools/fake_server.py --port 7869 --model circlestone-labs/Anima
node tools/browser.mjs tools/smoke-anima.json
```

`smoke-anima-aesthetic.json` checks Aesthetic's label, 30-step/CFG-4 defaults,
its own prompt advice and Example, active negative prompt, unsupported
controls, saved variant identity and final no-Refine answer on port 7895:

```powershell
uv run python tools/fake_server.py --port 7895 --model circlestone-labs/Anima@aesthetic-v1.1
node tools/browser.mjs tools/smoke-anima-aesthetic.json
```

`smoke-kohaku.json` checks Kohaku's hybrid tag guide, defaults, size guidance
(the exact 1.05 MP bound), saved settings, and a hybrid prompt's
213-character sentence answered unknown without the index, the rest
checked (guide tags valid, the tray never "Danbooru unreachable") and the
order slots there, on port 7870:

```powershell
uv run python tools/fake_server.py --port 7870 --model KBlueLeaf/Kohaku-XL-Zeta
node tools/browser.mjs tools/smoke-kohaku.json
```

`smoke-netayume.json` checks NetaYume v4 defaults, the qualified size guide,
author example, a retyped and a foreign system instruction cut by Apply,
enabled Mu, applied saved settings, the catalogue's name in the topbar, the
facts and the prompt check, the example-size wording (1.57 and 1.05 MP of
the authors' example sizes, in the line and the tier titles), Mu's 1.79,
the live negative's caption, the grid examples, and another checkpoint's
record reused, the note and the ⚠ title naming NetaYume Lumina v4, on port
7871:

```powershell
uv run python tools/fake_server.py --port 7871 --model duongve/NetaYume-Lumina-Image-2.0
node tools/browser.mjs tools/smoke-netayume.json
```

`smoke-newbie.json` checks NewBie defaults, the qualified size guide (the
author's example wording), XML example selection with its valid anonymous
character identifiers and no fill-in warning, the critic's XML reorders set
aside (words moved between blocks; `<general_tags>` ahead of a character)
and persistence, applied saved settings, the live negative's caption, the
grid examples, the refused adapter and its dead LoRA weight axis, Mu kept
live, the XML prompt's line breaks capped at twelve lines in the result
column (a Tab stop) and the lightbox (four under 901 px), and a
comparison's block reorder named by tag (loose text moved, a non-ASCII tag
and a long order cut between tags included). It checks the token line
under the prompt where there are no chips, from the authors' first example
at 1400x900: hidden under three quarters of Gemma's 511 tokens; the count
from there; amber past the limit, naming where Check says reading stops;
said once in the summary line and cleared under it; the Example's Undo note
hanging under it; "cut unknown" while the server does not answer, then
counted again without a keystroke once it does; hidden where the measure
says no encoder is known; and its change held while the mouse is on
Generate or the negative box, released by leaving it and by a key. It also
runs three XML jobs that differ in one attribute, two of them queued
(slowed by `POST /api/fake/step`; the first runs 40 steps), which must be
told apart by their words in the Now line, Cancel's title, the queue rows
and the Runs picker, the Now line's title keeping the markup. On port 7872:

```powershell
uv run python tools/fake_server.py --port 7872 --model NewBie-AI/NewBie-image-Exp0.1
node tools/browser.mjs tools/smoke-newbie.json
```

`smoke-wai.json` checks WAI v17 defaults, author tag recipes, the author's
reference sizes in the size line (1.38 and 1.05 MP) and in the tier titles,
disabled Mu and saved settings on port 7873:

```powershell
uv run python tools/fake_server.py --port 7873 --model LyliaEngine/waiIllustriousSDXL_v170
node tools/browser.mjs tools/smoke-wai.json
```

`smoke-rouwei.json` checks RouWei 0.8 epsilon's defaults, hybrid prompt
help, artist-last fallback and recommended size range on the fake backend:

```powershell
uv run python tools/fake_server.py --port 7890 --model Minthy/RouWei-0.8
node tools/browser.mjs tools/smoke-rouwei.json
```

`smoke-noobai.json` checks NoobAI V-pred's defaults, author prefix and negative,
tag helper, CLIP budget, saved identity and SDXL Refine pass on a fake backend:

```powershell
uv run python tools/fake_server.py --port 7891 --model Laxhar/noobai-XL-Vpred-1.0
node tools/browser.mjs tools/smoke-noobai.json
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
opening the lightbox beside the source, compared, a sweep that leaves no step refused
in the preview and a refusal moved to the error line at Strength or W with
Refine not moving, a strength typed and entered at once refused in the
dialog, a strength past 1 refused as a fraction, Compare 3 (held with the
Steps that would give a sweep at 2 steps, writing nothing), Compare 2 at
12 steps, Mu's "auto", the preview's "starts at N% noise" from an injected
`start_sigma` (the fake reads none), none for a sweep, and the reading
taken away by an unload and put back by a load with the dialog open and no
box touched, Refine held while the server does not answer, arrow-key
stepping and the dice, Ctrl+Enter and Enter running the pass, the
remembered strength and Reset forgetting it, the unsent draft and Start
over, Reuse opening a pass (its title and reused line, and another
checkpoint's, with this one's sampling), a pass whose source has gone, after
a viewport step to 2560x1305 the two panes with the kept-region frame and a
top 8vh down that does not move as the fold opens, the source standing in
for a pass whose frames are dropped - full size, in the result's own box,
captioned in the tile's `.foot` "Refining #N → W×H · strength s · step 1/…" -
focus off `<body>` after that pass, opened from script, ran - a source
deleted while the dialog is open refused in the dialog's own line, in the
page's words with the source line marked, with the dialog kept open, the
dialog with no sideways scroll at 1400x900 and 2560x1305, the scale
buttons' aria-pressed, a first open passing over the usual 1.5x (its title
and the size line's why), a foreign pass's reused line naming FLUX.1 Krea
[dev] by its catalogue name, and the dialog's CFG named "CFG (distilled)"
as the panel's, with the greyed negative's caption "(needs True CFG > 1 —
~2× slower)"; a pass whose source is deleted while it runs (`/api/fake/step`
slowing it) landing with "(deleted)" in its title and the note and its
Reuse saying so, `markGoneSource` marking an arriving pass, and a sweep
stopped by failures when its source went: Resume held, in words, with focus
kept and nothing sent, and the server's refusal of the resume in the words
the button reads.
`smoke-conditioning.json` has the other side
on its klein fake: the card's, the viewer's and the lightbox's Refine
answering with the reason and opening nothing.

```powershell
uv run python tools/fake_server.py --port 7875 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-refine.json
```

On the same FLUX fake at 7875, `smoke-refine-unavailable.json` checks an
unloaded off-catalogue capability guess: ⤢ opening nothing, a refined
record's Reuse keeping the pass in the dialog with Refine dead and nothing
submitted, the dialog waking when status says the pipeline has a pass,
Generate meanwhile rendering directly, a final answer's wording
(`refine_settled`) and the guess's back, and a foreign pass's general
starting strength (`refine_strength_profiled` false). `browser.mjs` sends Escape with its
key code, so a modal dialog closes on it as on a real press. Fake servers
share one throwaway gallery unless `--data` gives each its own, and
`smoke.json` counts the images planted under another checkpoint: run it on
a fake of its own while others are up.

`smoke-refine-sdxl.json` checks the Refine dialog on an Animagine fake on
7877: Animagine's measured tight line (2896x2896, 2x inside it unwarned) and,
on a status stubbed to an unmeasured profile, the generic one, SDXL's strength
sweep, the pass prompt's own check (the 77-token cut outside the fold, Show,
the checking line during a slow first use, Apply into the dialog's box with
Ctrl+Z, dead while stale and dimmed by the shared 0.8 / `--text-3` rule, the
negative-prompt Apply dead while CFG greys the negative box, the hidden line
that announces the check only when its text changes), the unknown-tag line
(a repeated tag named once) and its silence on a failed lookup, a capped
opening on the generic line, From panel (held with its reason, a press
saying so on the line under it in the dialog's view; the copy and its two
Ctrl+Z steps; a greyed negative left as it was while the prompt is copied
and undone, held while only that negative differs, and copied once CFG
wakes it; an empty panel prompt), the prompt's tag suggestions (nothing
selected under three characters with Tab moving on; the preselect from
three; Escape closing the list and not the dialog; Enter and Tab taking a
suggestion; Ctrl+Z; a list kept up and unselected while the next key asks;
a click in the listed item asking nothing; the list over the box at
1400x620 where there is no room below; the panel's list untouched), the
dialog's CFG named plain CFG and its negative caption in both states
("(needs CFG > 1 — ~2× slower)" greyed, "(runs at CFG > 1, no extra
cost)" live), and nothing reaching the panel:

```powershell
uv run python tools/fake_server.py --port 7877 --model cagliostrolab/animagine-xl-4.0
node tools/browser.mjs tools/smoke-refine-sdxl.json
```

`smoke-wide.json` checks the page at the sizes it is used at, switching
with `viewport` steps between 2560x1305, 2048x1040 at 125%, 1920x947 and 1400x900 on a
FLUX.1 Krea [dev] fake on 7876: the wide tier's scale (the panel's 461 px at
2560 and 400 px at 2048, a 15.5 px body and 13.5 px captions at 2560 and a
third of the way there at 2048, eight and seven gallery columns) and the
unchanged one at 1400x900 (330 px, 14 px, 12 px, five columns), Generate on
screen, the page's own focus ring on a checkbox and a button and the LoRA
select drawn dark at both wide sizes, the caption buttons 30 px at 13.5 px
and the tick boxes 18 px at 2560, a 16-digit seed fitting its box on the
Output row's one line at every size, its dice in the Seed name's line with
a target that stops at the box - and the Refine dialog's at 125%, its row
and the dialog with no sideways scroll - the Output caption styled as
Sampling's at 2560x1305 and in one column, and the group's hidden name,
with Generate whole at scroll 0, at 1920x947 and 2048x1040@1.25; at
1400x900 scroll 0 the caption the group's hidden name, the group its row's
height and Generate's top at least 16 px inside the window; a
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

`smoke-narrow.json` checks one column and narrow windows on a FLUX.1 Krea
[dev] fake on 7889 at `--step 0.3`: Reset's note in the box it emptied at
1400x900 (greyed negative) and at 600x900, Generate unmoved as it and its
Undo's note come and go, and the presses on Generate below landing with
the note still up; a running job with a long prompt
keeping the page its width at 600 px, the Now line ellipsized and Cancel in
the window; at 600x900, 800x900 and 900x700, New image as a pill centred
under the topbar and clear of Generate, for a viewer below the window, and
for one above it (scrolled to the gallery's end during a load), its arrow
following a jump to the page's top and the page never scrolling by itself;
its press focusing the live preview's tile, handed on to the image that
replaces it; a real press on Generate's right end with the pill up starting
a job (fixed points, each guarded by an `elementFromPoint` check); a
job cancelled after its first frame withdrawing the pill, its focus going
to Generate; a press with nothing in the viewer focusing the viewer, which
lets its tabindex go on blur; after a job, focus arriving in the panel
keeping the pill, and a typed character or a real press on the prompt
withdrawing it; the topbar's mirror whole with its load clock at 600 px,
the status on one line and Unload at the bar's end, with the free space
back at 1400; the Search tip inside the window at 600 and 700 px and
unshifted at 1400; the grid's LoRA weight tip clear of the left edge at
1400 and 2560x1305; a fitting `.end` tip unshifted; and in two columns New
image offered for a viewer scrolled above the window, static in the panel
after the finish line and kept past the job's end:

```powershell
uv run python tools/fake_server.py --port 7889 --model black-forest-labs/FLUX.1-Krea-dev --step 0.3
node tools/browser.mjs tools/smoke-narrow.json
```

`smoke-grid.json` checks the grid fold on a FLUX.1 Krea [dev] fake on 7879:
two direct renders back to back, each awaited to its end with one stream and
one tile; the resolved prompt and the one-value warning (commas named); Run grid as
wide as Generate; the totals at body size with factors of one trimmed; a
spaced range, `6..12` and an axis message naming its box; Resume's refusal
without pydantic's prefix; Repeat above one; the budget line, including 0
seeds, the jobs ahead, and the load note following a real unload and load
with no box touched; the cells already in the gallery once a grid ends,
without Resume; Ctrl+Enter running the grid from a grid box, refusing from
the prompt while the grid holds its placeholders, and generating on a plain
prompt; FLUX.1's grid examples (21,28,35 / 2.5,3.5,4.5), no Mu example, Mu
"auto" and no fixed shift; and, at 2560x1305: Run grid in the window after
a real click on the fold's summary once the preview has landed; a 30-line
prompt whose variable is listed by its line, with the preview's top and Run
grid in the window together; the one-value case; a long one-line prompt's
listing capped at eight lines and scrolling inside itself (both previews
awaited, not debounced, and the box logged); and with the fold opened by
hand before the variables are typed, Run grid no further under the window
than the listing box and one line (the panel holds still under the
pointer):

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
commonest value, F starring a focused cell (said in `#star-said`), the lightbox opened from the
sheet starring the cell (its star "★ Star", `aria-label` Star, its title
and `aria-pressed`) and F on the cell it hands focus back to, the Fixed line
under Starred, Resume this run filling the seed and size (its warning
keeping the preview's `.totals`), its mismatch warning and its absence on
another checkpoint's run, the version rows with their edits and titles (a
reorder as "same words, moved"), a strength sweep's source column and
lightbox walk, the refined badge, pins in the export's file name, an
empty sheet drawing only its note; the cell star (hidden until hover or
focus, out of the tab order, a click starring without opening the lightbox,
under the stuck heads when the box scrolls); iterations on random seeds (a
diagonal, then the seed pinned) with the note's Compare pointer, and none on
one fixed seed; Fill width (aria-pressed, the cells filling the width at
2560, 2048x1040@1.25 and 1400 with `sizes` following, the export at the
on-screen width from the PNGs at 2560, at the cells' device pixels at 125%,
and no smaller than Fit's at 1400, Fit's export from thumbnails, held on a
wrapped row, remembered across a reload and working with storage
throwing); Export all (beside Export PNG only with a pin, both buttons held
while it runs, one stack named `-every-cfg`, five slices past the edge
saved as one image each with distinct names, the same five files written
when the sheet is redrawn on another run mid-export, and a failing export
said under the gallery head); the matrix as a table (rows `display:
contents`, header and cell roles) with cells named by their axis values and
described by the facts the name leaves out, its opening said, a wrapped
sheet as a list; a 2 × 8 comparison's 16-digit seed heads whole under Seed
at 1500x900, and at 1400x900 breaking between groups of four into balanced
lines; and at 800x900 the finish line's Show sheet and a strength sweep run
from the viewer's ⤢ each handing focus to the first cell, with Tab then not
moving the page; and at 1400x900 the sheet's focus kept through F on a
cell another tab deleted (onto the cell now in its place), a changed
Columns select, and another tab's job ending under it, the sheet left as
it was. `smoke-sheet-klein.json` uses a FLUX.2
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
Delete unstarred, a lightbox of one image deleted (reopened at once after a
close, the check waiting for the delete itself and
`closeLightboxAfterDelete`'s focus), Delete unstarred keeping a starred
pass's source with the typed count, a landed image's 409; images another
tab made since the grid loaded, put on it before the count is asked and
counted in the dialog by the server's `newer`; the All models/running-job
dialog and the 409's running-job sentence for a job another tab started
(the server's `job_active`, not the page's `busy`); the dialog kept inside
the scrollbar, by real presses, for a row's last card and for the
lightbox's Delete; a pass whose source has gone - marked at once on
the page that deleted it (record, card title, Reuse) - tombstones
("Deleted · #N"), focus and the emptied viewer's placeholder, an image already deleted,
a tile deleted elsewhere (once the viewer's PNGs have loaded: on Windows the
server keeps a file it is still streaming) and the job's replay, the
lightbox paging in behind a delete (its counter reading the gallery's total
and the id), a double-clicked delete, and Reuse: Ctrl+Z and its Undo, the
Undo retired by a change (Ctrl+Z, a seed, an Example) and a Reuse taking an
Example's Undo, the page unmoved mid-gallery with the prompt in sight, the
page's end, one column at 800x900, the Undo in the window at 2048x1040
@1.25, every image in scope deleted by another tab (a real press says no
image is left, focus on the search box), and a real sweep that takes every
image handing focus to the search box. Every wait in it is a condition
(`__until`, `__settled`), none a sleep:

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
aria-hidden clock and indeterminate bar, also shown in the empty viewer's
one line, whose own words come back after a cancel; a second job queued
during the load keeping the load's note under the bars through the queue's
redraw; a cancel during the load; a gated
load's detail said in the panel line after every Generate while the viewer
holds images; a gated load's card said once and not re-announced by the
banner, its Retry load handing focus to Generate; Load and Unload handing
focus to each other; the banner hidden by a new job and not repeated
after a watched one; a gated load's finish line "Not made: the model could
not load"; and a watched job whose server goes (a stand-in EventSource and
fetch): the job line in the topbar's phrase with the count, no steps, time
left or topbar mirror, the tab's title and the strip's note the same, then
`showLostRun`'s card, reworded once the server answers, its Run it again
running the request with focus going to Generate; each job watched once
while the page is between jobs - gated fetches hold each ordering open, with
its preconditions asserted: a Generate during watchNext's listing and
during resumeActiveJob's; a queued POST answered after the listing found
nothing; one answered after the listing had made its job the watched one;
and one whose job the listing watched to its end first. Each gives one
stream, one tile and no queue row; the first four also give a finish line
and `startJob` resolving at the job's end, and the last gives `startJob`
resolving on its POST's answer, after the job has ended, keeping nothing of
it; live regions (the topbar's name not rewritten by a poll nor by another
tab's job, a single image's step count shown and not said, each run
counter said once, `#job-said` at a job's end and for a queued job, nothing
there for a cancel or a load failure the panel line says); and a second
tab, an iframe of the page: its job (held paused) holding Unload and the
critic here with the reason and the topbar's aside, its images joining this
gallery at its end, and taken in after this page's own job, queued behind
it, is cancelled at its first step; its queued job picked up here as
another tab's (viewer kept, this page's New image and its own finish line,
Use seed and all, kept through it, its end said on `#job-said` only, this
tab's own finished title given back until a key), the estimate leaving the
first step out once a second has landed (none at step 1, a one-step job
none, a sweep of one-step cells from its second),
a cancel and a queued row's cancel from that tab, a pause here with its job
behind ("waiting: job ahead paused" there; its row marked as another tab's
here, Cancel queued's title, the count), Cancel queued here said there as a
cancel from elsewhere, a job that ended at the load reworded once that
tab's Load lands, with the panel line's reason taken back, and, on a page
opened afresh, one that ended at the load over the card said on
`#job-said`, reworded by the card's Retry load and again by an unload:

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
figure (a group named #N), label, title, facts-only description and card
tools at 1400x900 (❐ hidden on FLUX), with `this-run`, the star and ⤢; the
note under it, "#N · W×H · seed N · Use seed", with the gallery head
unmoved as a single image lands at 1400, and hidden at QHD; at 2560x1305 the
column with Delete in words and
the row under the picture empty at `--h-tool`; the live preview
keeping the result's box; `#recent` level with the viewer, and staying
beside it with the check docked while the image and the gallery head stay
put, the column's actions and `.result-brief` then the row under the
picture; the fit at 2048@1.25; the lightbox walk from the single image; a batch
of four best-fit; a pass's note and lineage and its Compare with source
(the lightbox on the pass, compared; Escape back on the strip's image; no
Use seed in its facts), then its source deleted ("#N (deleted)", no
buttons, the title, a facts-only description, Compare with source
aria-disabled with the reason and refused over the tile); a portrait pass
with the check docked, its row under the picture whole at 2048x1040@1.25
and 2560x1305, Compare with source left to the column; a 5:12 direct image
and its pass at 2048x1040@1.25 with the check docked, the row under the
picture wider than it, centred, inside the strip, every word whole, the
picture and the gallery head unmoved; four landscape images two by two with
the gallery head unmoved from the first frame; a deleted result held at its
height with focus on its line (the column's Delete); a 28-seed run at
2048@1.25 ending scrolled to its newest row; 28 at QHD with no padding-only
scrollbar and the gallery head unmoved from its first frame (`reserveStrip`); the banner's
"2 since deleted" / "all since deleted"; the banner's Resume after a
resumed-and-cancelled grid putting the run's base seed in Seed; no Resume
for a cancelled comparison; a batch of four at QHD followed from its first
frame to its end - the preview in the first image's column, no tile moving
within the strip, the tile size and the gallery head where the first frame
put them - and the strip fitted alike with the page's scrollbar forced on
and off; Reset with its Undo in the prompt box it
emptied - inside the window at 1400x900 and at 800x900 - Ctrl+Z after it
giving the prompt back, and the store kept forgotten past the prompt's
blur; and Reset taking a Reuse's note from the panel line, with its Undo
on offer or already retired:

```powershell
uv run python tools/fake_server.py --port 7883 --model black-forest-labs/FLUX.1-Krea-dev --step 0.3
node tools/browser.mjs tools/smoke-results.json
```

`smoke-lightbox.json` checks the lightbox on a FLUX.1 Krea [dev] fake on
7884: the facts list and its rows, the Seed row's Use seed (said in the
lightbox's line, the panel's kept, taken back by a step to another image);
the keys line, naming S on a pass only with each key bound to its action,
and the toggle's S by Generate's hint rule; the star ("★ Star",
`aria-label` Star; F said in `#lb-star-said` away from Star, not on it); a
pass's Reuse pass… name; a pass compared with its source (S, the button, a
dragged and keyed divider, S on the focused divider handing focus to the
toggle, swipe and side by side keeping 1:1 and fit, side by side at fit
clear of the arrows' circles, a pass reshaped to 3:2
compared with the part it kept, a deleted source's dead controls); the
source link with the view carried; the passes link; a pass whose source has
gone (its Reuse and its card's title); the arrow strips (a press 60 px off
the circle steps, the backdrop below still closes, the wheel over a strip
zooms, the first image's disabled arrow a circle with the backdrop beside it
closing, a double-click whose step takes the strip away stepping once,
circles again when zoomed); a close followed at once by an open keeping the
new lightbox; focus after Escape on the walked-to card, with the Examples
menu left open under it too (the menu kept, the next Escape closing it),
and Ctrl+Enter there generating and opening nothing; Escape on an image no strip
holds focusing the search box; Cancel on a Refine opened from it, a side by
side 1:1 included, and no reopening after a run; an Escape and a Cancel held
while the pass is being sent ("Sending…"), with the pass run and focus on
the source's card; F on an image another tab deleted (the note, the next
image, nothing in the panel's line); an image that will not load (deleted,
and with the server not answering: the offline sentence as an offline
message, taken back by the status poll that answers, which asks for the
picture again); at 2560x1305, 1:1 on a 1 MP image by key, button, wheel
and double-click, the 52 px circles clear of the picture and side by
side's halves inside them, circles below fit with the double-click beside
the picture giving fit back, and a double-click beside a below-fit image
giving fit back; at 800x900 no keys line; a wheel burst said once from the
hidden `#lb-zoom-said`, a keyed zoom at once, and F rewriting neither the
zoom nor the counter; at 600x900 the facts taking a square picture's
letterbox (every row in view) while a portrait picture keeps the six-line
scroller, and is fitted again, whole, once Use seed's line takes a row
from the stage; Side by side titled stacked before the press; side by
side stacked at 600x900 for a square pass (a wheel
zooming about a point in the lower half) and a 7:4 pass (halves over 400
px with the facts' room, opened again stacked again), and left and right
for a portrait pass; and at 800x900 a square pass left and right with its
facts in view:

```powershell
uv run python tools/fake_server.py --port 7884 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-lightbox.json
```

`smoke-gallery.json` checks the gallery on a default (Krea 2 Turbo) fake on
7885, a checkpoint with no pass and no references: the Runs picker's groups,
labels (what each run varied: "steps ×8 · 5 seeds", "2 seeds") and sync with
the search box (a pick, a card's ⊞, typing, Sheet off), the picker browsed
from the keyboard (no fetch while arrowing, Escape, Enter, leaving it),
at its own width with the search box whole and the picked run in its title;
`id:` and the search tip; the Starred tick and Re-render's label; each card
a group named #N; the five tools and their names on hover and focus inside
the card, with Delete's over Delete, and the image's facts-only
description; the card's #N · W×H · seed caption on hover and focus, hidden
under a tool's name and dropped under a card note; Refine and Reference
aria-disabled, their refusals in `#gallery-error` and a one-line card note,
with the card and grid unmoved, a second press landing on the same tool,
and the row's names held down while the note is up; the star button (named
Star in both states, hidden until hover, a click starring without opening
the lightbox, F from the star saying nothing more, F from a tool and from
the image said in `#star-said`, the lightbox's F reaching the card and said
in `#lb-star-said` - each read from `__said`, a record of the two lines'
writes - the lightbox star's title and `aria-pressed` matching the card's,
and a star on an image deleted elsewhere tombstoning it with "#N was
already deleted." in the gallery's line and the Runs picker asked again);
a refined record's badge and Reuse labels (planted into the page's gallery
replies; its name led by "Reuse pass"); the ⚠ carry-over warning,
said again when references flip and when the adapter list lands; the day
line across pages, a line over the first card when the newest images are
not today's, and a deleted head's tombstone across a day line drawing no
second line; the roving tab stop, arrows across a short row and the day
line, Home/End, Down paging in, F, a delete handing the stop on, the
lightbox's user close handing the stop to the card it ended on, and a job's
newest image taking the stop where none was chosen, prepended under an old
head made two days older with the day's line between them; the head's
S/M and Fit/Crop from 1800 px (eleven and nine a row at S with the tools
fitting, Crop's cover, one tab stop per pair, remembered across a reload,
hidden on a sheet, kept by Reset, not applied at 1400); S four across in
Gallery beside's column at QHD, M three; and the tools fitting at
2560x1305 and 2048x1040@1.25:

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
side full width at 1400 px, and the stacking at 1400x900; the 600 ms outline
on the box a finding selects in, and none under an emulated reduced motion,
Apply keeping the replaced prompt in Your version, the check capped at 55vh
above the results at 1400x900 with its head stuck, Escape closing the check
(focus to Check from inside it; the Examples menu, a search box with text
and the lightbox first; the caret kept from the prompt); the critic's
line's live half only, its ready line at once after an answer, and
Generate's, Run grid's and Run comparison's titles while the critic holds
them; the page's h1, the Settings region, the viewer as a region named
Results with no unseen heading in the stage, the greyed negative's
description, each finding's and critic issue's Show with no item a tab
stop, the always-run Original's description; and, docked at 2560x1305, Tab
from the panel to the image just made, its note (a stand-in stop),
`#recent` where it sits between, the check, then the gallery, with nothing
in the docked stage out of its flow; the comparison's Batch tip hung from
the window, whole and on top with its row under the sticky head at
1400x600, following its badge as the check scrolls, hidden with the badge
scrolled out, and docked at 2560x1305; the Refine dialog's Refine held,
and said in the dialog, while the critic thinks; the critic held offline
(its title, the cost line, and the press's offline words taken back by the
answer) and during a load (its note gone with the load), and a critique
the server never answered said as the server's (no "?." after its
question) and taken back; a 900-word prompt's line over 4000
characters, its cut and its summary; and, docked at QHD, the first use's
download sentence and the clock as one run of text, the clock on its last
line.
`smoke-animagine.json` covers the tag-order check, the critic's reorder
under full and partial tag placement end to end (set aside, kept as Check's
own, kept as allowed, flagged, each with the kind the server says it kept),
the partly-checked order (past 64 tags, without the tag index, a tag the
index does not know, a prose prompt) and the checked one where no missing
category decides it (that tag ahead of a general tag, alone, one prose
clause), and the negative prompt's Apply, dead while the box is greyed,
and its Show, held with the reason. The fake server
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
your prompt" / "new" / "No longer flagged" / "Not flagged … asks the tag
index nothing of its own" marks.
`smoke-comparison.json` uses the same FLUX-shaped fake on 7874 to run
two comparisons with original and yours reversed on fixed seeds. Each
opens its own job, while opening their shared run exposes Prompt as an
axis. Run comparison clicked at 800x900, and at 2560x1305 under the docked
check, hands focus to the sheet's first cell (named by seed and version)
without moving the page, and focus moved to the prompt while the
comparison renders stays there. `smoke-comparison-batch.json` uses the same
fake for all six Seeds/Batch
pairings, their preview and request counts, select values changed without
events, paired output seeds, sheets and stale-run disabling. API tests cover
paired saved seeds, batch grouping and preview counts for all six pairings.
The analysis and Animagine smokes cover CFG/True CFG activation
changes while a grid keeps the negative editor enabled, inactive raw
edits, and settings changes while a check or critique is in flight.

`smoke-analysis-tags.json` uses an Animagine-claiming fake on 7887 for the
prompt check on a tag checkpoint at QHD: a Series-before-Character order
finding the rewrite keeps, listed as possibly still applying ("whose check
asks the tag index nothing of its own") rather than fixed, and the docked
check with the chips across both columns at 2560x1305 and 2048x1040@1.25,
where the image does not move and stays in view (6, 32 and 40 tags at
each), and Tab docked with chips going chips, image, its note or
`#recent`, check; beside an idle viewer the check's bottom in the window
with the critic's answer in it; with Gallery beside, the check in the
gallery's column over an idle viewer with the critic's answer (scrolling,
head stuck, Close under the pointer, the chips still, the page unscrolled)
and beside the image just made (the image, the chips and Generate
unmoved); at 1400x900 with 40 tags and the answer the check capped under
the chips, its head stuck and the image starting in the window's top three
quarters; at 1280x720 with 40 tags and the answer the cap 55% of the
window less the chips, the check at least 8rem and the image starting in
the window's top three quarters; and at 1280x600 the floor itself, the
check 8rem tall with its head stuck and Close under the pointer, the image
in the window's top 85%:

```powershell
uv run python tools/fake_server.py --port 7887 --model cagliostrolab/animagine-xl-4.0
node tools/browser.mjs tools/smoke-analysis-tags.json
```

`smoke-stage.json` checks Gallery beside on a FLUX.1 Krea [dev] fake on
7888. It opens at 1400x900 (no tick, no column), then at 2560x1305 checks
the tick off by default; ticked, a fixed column level with the viewer and
inside the window, the page not scrolling, three cards across with their
tools whole, the search on its own row, the paging trigger rooted on the
column, the tick remembered; the column paging by the wheel with the window
still; the arrow keys walking its rows, End bringing the last card into the
column's view, and Right past the last card paging in; with nothing made
yet (the viewer idle), the check and the critic's answer in the column's
box, scrolling inside it with its head stuck and Close under the pointer,
the page still (its grid, posted by fetch and so another tab's job to the
page, waited out first, since that job holds the critic); a job landing
with the pointer resting on a card (hovered by selector) changing nothing
in the column, with the count already new, until the pointer leaves, and
then first at the column's top; the image just made at its own size beside
the column; a batch landing above a scrolled column keeping its view;
`#recent` as a row under the image; the prompt check taking the column's
box, with the column hidden and the image, its column of words and
`#recent` unmoved and nothing cut; Tab reaching the check in the column
after the image just made, its note and `#recent` (`reading-flow:
flex-visual`); the lightbox's Escape from an image only the hidden column
holds focusing the viewer's image; Close giving the column back at the same
scroll; a job landing under the check, not held; the Sheet button and a
card's ⊞ opening a sheet across the stage's width, landed on, with the
tick's title saying when it counts and the column back on Sheet off; Reset
leaving the tick, a reload restoring it, and the window crossing 2000 px
both ways; at 2048x1040 at 125%, two cards across beside the image in
view, the check and the critic's answer in the column's box over an idle
viewer after the reload, and the check taking the column with the image
unmoved and nothing cut; and, unticked, the gallery back under the viewer,
with focus on the tick and the key forgotten:

```powershell
uv run python tools/fake_server.py --port 7888 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-stage.json
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
