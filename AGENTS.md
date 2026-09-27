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
  `analysis`: style, documented order and `reorder_tags`, negations, unquoted
  text to render, the tag checkpoints' specifics; `heuristic` ones labelled
- `backend/prompt_sources.py` — what the critic may cite, per guide constant:
  the guide's own sentences (G) and short verbatim author excerpts (A) with
  their URLs, notes and caveats
- `backend/critic.py` — the local prompt critic: pinned llama.cpp
  `llama-server` and Qwen3-4B GGUF, fetched and verified by `prepare()`, one
  server per `critique()`, the answer validated against its sources
- `backend/translation.py` — lazy CPU-only Marian translation of ERNIE's
  expanded Chinese prompt for the UI; the Chinese prompt remains canonical
- `backend/tags.py` — cached Danbooru tag suggestions and exact validation for
  tag-caption checkpoints; remote failure means unchecked, never invalid
- `backend/seeds.py` — the seed range and the wrap, shared by server and CLI
- `backend/loras.py` — the adapter directory: what is in it, and the file
  behind a name (the only place a request's name becomes a path)
- `backend/sizes.py` — the preset table; the CLI's `--aspect/--mp`, config's
  `max_side`/`max_pixels`, `web/state.js`'s copy, `index.html`'s `max=` and the
  README table are all held to it by `tests/test_profiles.py`
- `backend/backends/base.py` — the `GenerationBackend` ABC
- `backend/backends/diffusers_backend.py` — in-process diffusers implementation
- `backend/backends/loading.py` — everything decided before the weights
  move: the quantization config, the Nunchaku preflight and the
  prequantized refusal, the components left out of the download, the
  device placement; takes `settings` as a parameter
- `backend/backends/single_file.py` — pinned, CPU-only first-load conversion
  for Illustrious XL v2.0-STABLE and WAI v17; atomic cached trees feed the normal loader
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
  scoped to a run by default
- `backend/reindex.py` — rows rebuilt from the PNGs' `local-diffusion` chunk, in place or
  copied in from another gallery; idempotent on the UNIQUE filename
- `web/` — served as-is from `/static`; the page script is eleven classic
  scripts `index.html` loads in order (`state`, `sizes`, `status`,
  `generation`, `panel`, `gallery`, `sheet`, `lightbox`, `tags`, `analysis`,
  `app`), sharing one global scope; only `app.js`, the last, runs anything at
  load. `analysis.js` is the prompt check, the critic and the comparison
- `tools/` — `fake_server.py` (the app on the tests' fake backend, port 7861)
  and `browser.mjs` + `smoke.json`, `smoke-flux.json`, `smoke-preview.json`,
  `smoke-ernie.json`, `smoke-animagine.json`, `smoke-analysis.json`
  (headless browser over DevTools): the UI's stand-in for a test suite, see
  Tests
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
  `POST /api/model/unload` answers 409 while a job or a load is live. A load
  first waits out whatever holds the generation gate (`jobs.wait_for_gate`,
  inside the lock, without keeping the gate): a job queued during a critique
  must not move weights onto the card beside the critic's server.
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
  `negative_prompt_supported`, `true_cfg_supported`, `reference_supported`
  and `lora_supported`.
- `GenerateRequest.applied()` is the one normalisation - negative prompt, mu,
  references, `lora_scale` - and `backend.applied()` answers it from the
  parameter list or the profile. Every save path, the resume prune and the
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
  checkpoint, quantization, references, LoRA and weight, variables, hash - is
  on the row, in the PNG's `local-diffusion` chunk and in `CellKey`. `run_id` names a
  cell by the shortest key carrying its non-None optional elements, so every
  existing run id survives a widening.
- Resume keys on settings (`storage.completed_cells`, six-decimal rounding,
  scoped by checkpoint and quantizer), requires an explicit seed unless the
  seed axis is a list, and answers 409 when nothing is left.
- A new column needs a `_migrate` entry, not a `"x" in row.keys()` read guard:
  `_row_to_record` reads every column unguarded. `save_image` and
  `reindex.index_file` share `_insert_row` over `_COLUMNS`.
- Reference images are gallery ids, never paths. A LoRA is a bare name that
  only `loras.py` turns into a path, `.safetensors` only, and it is
  hot-swapped in `generate()`, never baked into `load()`. Whether it fits is
  `loras.mismatch` - the adapter's module names and widths against the
  denoiser's, using the pipeline's adapter reader and PEFT key conversion -
  from the checkpoint's configs on the meta device before a load
  (`loading.denoiser`) and the live pipeline
  after; a misfit is listed with its `detail`, hidden by the page and
  refused at submit and by the CLI, unreadable configs mean unchecked rather
  than refused, and a listing never converts a single-file checkpoint.
- Under Nunchaku (WSL) a seed does not reproduce bit-for-bit - Z-Image
  within noise, ERNIE visibly - so a duplicate count or a re-render's hash is
  the fp8 path's guarantee only; `docs/verified.md` has the numbers.
- `GridRequest.prompt_variants` (2-8 `{label, prompt}`) is the prompt axis in
  place of the variables, never beside them: each variant is taken verbatim
  (nothing substituted; a `{`, `|` or newline is text), seeds stay outermost
  with the variants next in the order given, and each cell records
  `variables={"variant": label}`. `CellKey`, `run_id`, resume and the
  duplicate count read the resolved prompt, not the label. A placeholder in
  the negative prompt is still refused.

Jobs:

- Cancel and pause are cooperative and polled: cancel from
  `callback_on_step_end` (raise `Cancelled`), pause at the cell boundary
  every quarter second. Never `Event.wait()`.
- A job is a list of cells; seeds are the outermost loop, prompts next. A
  cell's failure skips the cell; three in a row or more than a third of the
  grid ends the job. Every terminal transition goes through `_finish`, and a
  budget stop returns through the cancel path rather than a fourth status.
- Each SSE stream has its own queue; `events()` opens with `done` for a job
  that has finished; the per-cell event and `/api/jobs` send
  `snapshot(images=False)`. Progress events carry `elapsed`; `waiting_on`
  says `load` or `queue`.
- `handle_exit` cancels the active jobs on the signal itself, the load runs
  on a daemon thread, and lifespan calls `try_unload()`.
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
  and rules off the loop, never `ensure_loaded`, never the gate, `applied()`
  from the profile before a load. The negative prompt is measured and checked
  only as `applied()` runs it.
- Every finding says what it rests on: `source` (`guide`, `author`,
  `encoder`, `heuristic`), `cite` and `url`. Rule config lives in each guide's
  `analysis` key, never a profile key or `guide_for`; a `guide` cite is a
  verbatim substring of that guide's `prompt` prose and an `author` one of an
  example (`test_every_cite_is_the_guides_own_words`). A check no author
  sentence supports is `heuristic`, with no cite. `documented_order` is None
  where the guide states none; never invent one, and never say the authors
  give none - the page says the app checks none and shows the authors' own
  (`Excerpt.order` -> `author_orders`). `order_checked` is True only where a
  rule placed the prompt (a tag checkpoint's comma items); the page says a
  documented order it was not is "Not checked". A rule's `suggestion` is a
  whole replacement for its field, offered only where it is unambiguous
  (lettering whose end is a guess gets none); an encoder's is advice text.
  `prompt_rules.QUOTED` is the one reading of quoted text, for the rules, the
  encoder layer and the critic.
- `prompt_encoding` measures over the exact string the pipeline builds, and
  every template string, length and `enable_thinking` switch it copies is held
  equal to the installed diffusers source (and the repo's Anima and NewBie
  pipelines) by `tests/test_prompt_encoding.py`. Its table is first-match
  like `_PROFILES`, exact ids first, reusing the loaders' pinned constants.
  Tokenizer files only, cached under a lock and kept only after a successful
  load; `measure` never raises and says what went unchecked in
  `encoders_detail`. Never from `/api/status`; `prompt_encoding.unload()` runs
  on Unload and shutdown. `\(`/`\)` escapes are not flagged.
- Tag categories for the order and artist checks come from the chips' own
  lookup (`_lookup_tags`), only on a Danbooru-vocabulary guide or one whose
  rules read artists (Anima, `wants_artist_categories`, no cutoff) and a tag
  or hybrid prompt, capped at the chips' 64 items; any failure there is
  `None` - order placed from the guide's own groups, never guessed - not an
  error. The reported `style` is `detect_style(prompt, categories)`, the one
  the rules applied.
- The critic is pinned: llama.cpp `b11206` archives (`critic.BUILDS`: Windows
  CUDA 13.4, and Ubuntu for WSL) and `Qwen/Qwen3-4B-GGUF` `Qwen3-4B-Q8_0.gguf`
  at a fixed revision, each held to its size and sha256 (`.part`, verify,
  extract beside, rename; traversal refused; an unmarked tree never used).
  `LOCAL_DIFFUSION_CRITIC` (default on), `LOCAL_DIFFUSION_CRITIC_SERVER` and
  `LOCAL_DIFFUSION_CRITIC_MODEL` (absolute; neither held to the pins).
  `supported()`/`ready()` are file checks only, fit for `/api/status`.
- The critique route: 422 unsupported, 409 while a job is queued or running or
  the image model loads (checked before `prepare()` and again after it),
  `prepare()` on a daemon thread outside the gate, then `critique()` inside
  `jobs.run_when_idle`. One `llama-server` per call on 127.0.0.1, a free port
  and a fresh API key passed as `LLAMA_API_KEY` in its environment (never on
  the command line, which other local users can read), refused when
  `/v1/models` answers without it, and stopped (terminate, wait, kill) in
  `finally` before the gate opens.
  `prompt_critic.unload()` on Unload, in lifespan before `jobs.shutdown()` and
  in `handle_exit`. `CriticUnavailable` is 503, `CriticError` 502.
- The critic's answer is checked before it is shown (`critic.validate`):
  `cite_id` is an enum of the numbered G and A sources in the answer's JSON
  schema and is resolved server-side (an unknown id drops the issue, counted
  in `dropped`); a quote is located in the prompt (`critic.locate`) or
  nulled; a part must be the prompt's own words and `text_to_render` its
  lettering (`prompt_rules.rendered_spans`); issues restating a reported rule
  finding, and order issues asking for different parts first, are left out;
  `reordered` survives only under a documented order and as a pure reorder
  of whole tags, clauses, sentences or XML blocks with quoted text unchanged
  (`is_reorder`); a rewrite wrapped in quotation marks is unwrapped, one
  quoting words the prompt lacks is dropped, one quoting its description is
  flagged. What was left out besides `dropped` is said in `set_aside`. The
  rewrite is linted by layer 1 in the route after the gate opens (`_lint`,
  no tag lookup), since a lint can fetch a tokenizer. What survives is still
  the model's suggestion; the docs must not call it trusted. Reasoning stays
  on. A G sentence listed `unsourced` in `prompt_sources` is never citable,
  and a borrowed excerpt says whose it is.

Frontend and CLI:

- `web/` has no build step and no test suite; check it in a real browser with
  `tools/fake_server.py` and `tools/browser.mjs` (see Tests). The page script
  is classic scripts sharing one global scope, loaded in the order
  `index.html` lists them; their top-level bindings stay top-level, and only
  `app.js`, the last, runs anything at load.
- The size line compares the request against `guide.pixels` a tenth either
  way and quotes `pixels_word`; the Examples menu is built on open from the
  last status, and a pick tells the preview and the panel the way `choose`
  does.
- Prompt expansion is ERNIE-only and explicit: its result replaces the editor
  only if the user did not type while the long request was in flight, then the
  usual persistence and grid preview paths are told about the programmatic edit.
  The critic holds what Expand holds (Generate and Run grid aria-disabled,
  Expand, Unload) and `startJob` refuses while it runs.
- The prompt check describes one text: the prompt and negative prompt as
  `collect()` sent them. Any change to either - typing, every programmatic
  write through `promptReplaced()`, `syncCfg()` greying the negative - marks
  it stale: dimmed, every Use/Apply and Run comparison dead, a finding's words
  selected only while still where the check found them. A checkpoint change
  clears it. Apply is offered on rule findings only; an encoder suggestion is
  shown as text. Use and Apply write through `replaceBoxText` (so Ctrl+Z
  restores) and check again. Server offsets count code points; the page
  converts them (`units()`) before selecting. The comparison sends
  `num_images: 1`, never writes `els.prompt` or `els.gridPrompt`, drops a
  version identical to another, and opens its run as the sheet when done;
  its Your version box (`yours`) is the one way to test order on prose. A
  negative-prompt Apply is dead while the negative box is greyed.
- Tag autocomplete and comma-item validation use the fixed Danbooru
  tag API through `backend/tags.py`, filtered when a checkpoint documents a
  knowledge cutoff. WAI v17 publishes none: keep its cutoff `None` and the
  live lookup advisory rather than inventing a date. Model-card
  special tags validate locally; lookup failure stays advisory and never blocks.
  On a two-column page validation lives above the results, not in the panel;
  under 901 px the same node returns below Prompt. The prompt check's panel
  follows it, in that order (`placeTagValidation` moves both).
- Validation chips edit the prompt: drag (or Alt+Left/Right) reorders, × or
  Delete removes, and right-clicking a gap inserts there (a chip inserts after
  itself) with autocomplete. All mutations must go through `writePromptTags`.
- Profile defaults apply per control (`owned`); an empty Steps or CFG box is
  sent absent and 0 is a valid CFG. The panel persists to `localStorage`
  with every access in try/catch; `resetPanel` is the one thing that forgets
  it, since clearing the gallery never reaches the browser.
- The sheet's layout contract - axes, prompt labels, pins, variables - lives in
  the client, and the export is client-side, not a route. The sheet asks for
  the whole run in one page. A variable axis lists its values by first
  appearance (the oldest image carrying each), not alphabetically, and a
  varying `var:variant` axis takes the rows whenever anything else varies
  (`VARIANT_AXIS`, held to `grid.VARIANT` by `tests/test_grid.py`), so a
  comparison reads original, reordered, rewrite, top to bottom.
- `generate.py` duplicates `jobs._generate`'s loop deliberately. It applies
  `--model/--quant/--offload` to `os.environ` before importing
  `backend.config`, as `launch.configure()` does for the server; both take the
  flags from `launch.add_model_arguments`.
- The gallery is the running checkpoint's: `GET /api/gallery`, `/api/runs`,
  `GET`/`DELETE /api/gallery/unstarred` and `POST /api/rerender` scope to
  `settings.model_id` (`main._model_scope`, an exact predicate, never the
  `model:` LIKE) unless `all_models=true`; the page's All models tick is the
  one switch, unpersisted like Favorites, and `GalleryPage.other_models` is
  what the same filter matches elsewhere. Delete unstarred takes the scope
  because its count must be the grid's.
- Search terms are `key:value` predicates plus FTS text; `date:` is a local
  day, `mu:none` is `IS NULL`, and `search.strict_scope` is the reading for
  `clear --where`, `keep_rates` and the gallery's Delete unstarred
  (`GET`/`DELETE /api/gallery/unstarred`): `clear --where <box>
  --keep-favorites` over HTTP, the search box strict or empty, gated on the
  count typed back and pinned to it as `expected`, which `delete_matching`
  checks under the write lock and answers 409 on a mismatch; rows and files
  only, no orphan sweep. `forgetImages` is the one place the page drops
  deleted ids. The gallery pages by id, never by offset,
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
database and outputs directory (`tests/conftest.py`). They cover the invariants
that are expensive to get wrong: the schedule the scheduler ends up with, seeds
surviving a JSON round trip, the search fallback keeping its filters, and the
HTTP guards.

The prompt analyser is tested the same way, with no network, no subprocess
and no GPU: `test_prompt_encoding.py` (budgets, cuts, templates held to the
installed diffusers source; its real-tokenizer cases skip when nothing is
cached), `test_prompt_rules.py` (the `analysis` data against the prose, the
authors' examples raising no warning), `test_critic.py` (pins, install,
server command and stop, answer validation, grounding) and
`test_api_analysis.py` for the routes, the gate and the status flags. The
`fake` fixture installs `FakeCritic` as `main.prompt_critic` - its five
methods held to `PromptCritic`'s signatures by a test, its answer raw JSON
put through the real `critic.validate`, so nothing it returns is something
the real pipeline could not - and
`fakes.fake_tokenizer` as `prompt_encoding._load_tokenizer`: a token per word
or mark with real offsets, at each encoder's real budget.

There is no JavaScript test suite. `web/` is checked in a real browser
instead: `uv run python tools/fake_server.py` serves the app on 7861 on the
tests' fake backend (gradient images stamped with seed and steps, a throwaway
gallery under the system temp dir, `--step 0.3` to make pause and queueing
observable), and `node tools/browser.mjs tools/smoke.json` drives headless
Edge or Chrome over the DevTools protocol with Node's own WebSocket - no npm
install - through the lightbox zoom, the sheet, the export, the re-render
button, the panel's persistence and the duplicate/distinct counts, exiting 1
on a failed `expect`. `smoke-conditioning.json` runs against a klein-claiming fake on 7864 for the
reference chips, the adapter select and a weight sweep;
`smoke-flux.json` runs against a second fake server
claiming FLUX.1 (`--port 7862 --model black-forest-labs/FLUX.1-Krea-dev`)
for the True CFG box, and `smoke-preview.json` against one slowed to
`--step 0.3` on 7863 for the live preview. `smoke-ernie.json` runs against an
ERNIE-claiming fake on 7865 for the explicit prompt expansion flow; the fake
takes the claimed profile's threshold and true-CFG flag, and its model id's
prompt-expansion capability, so it behaves as the real one would.
`smoke-animagine.json` runs against an Animagine-claiming fake on 7866 for its
tag guide, deterministic autocomplete/validation, CFG 5 default, recommended
aspects, orientation switch and disabled non-flow Mu controls, and the prompt
check on a tag checkpoint: a tag out of the documented order, Use reordered,
and the recommended-negative Apply going dead once the negative box is edited.
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

`smoke-analysis.json` checks the prompt check on a FLUX.1 Krea [dev] fake on
7874: the CLIP cut, the negation and quote findings with their sources, span
and cut selection (an emoji ahead of them included), the stale rule for both
prompt boxes, Apply, the critic's first-use and timing lines and its hold on
Generate, Use rewrite with undo and a reload, an edit while the critic
thinks, a two-version comparison opening its sheet, a prose prompt's order
shown as not checked, a one-clause prompt the critic cannot reorder, and a
comparison against Your version. `smoke-animagine.json` covers the
tag-order check and the negative prompt's Apply, dead while the box is
greyed. The fake server
serves the tests' `FakeCritic` and word tokenizers: `--critic-delay` (seconds
a critique takes, default 0.4, so its in-flight state can be seen) and
`--no-critic` (a platform with no build: the page hides the critic). The fake
critic counts as downloaded from its first critique on, so the first-use line
shows only on a newly started server; the smoke accepts either line:

```powershell
uv run python tools/fake_server.py --port 7874 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-analysis.json
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
