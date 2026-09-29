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
- `web/` — served as-is from `/static`; the page script is eleven classic
  scripts `index.html` loads in order (`state`, `sizes`, `status`,
  `generation`, `panel`, `gallery`, `sheet`, `lightbox`, `tags`, `analysis`,
  `app`), sharing one global scope; only `app.js`, the last, runs anything at
  load. `analysis.js` is the prompt check, the critic and the comparison
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
  on the row and in the PNG's `local-diffusion` chunk. `CellKey` holds the
  prompt pair, steps, CFG, mu, true CFG, references, LoRA and weight, and the
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
  the seed axis is a list, and answers 409 when nothing is left.
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
  `variables={"variant": label}`, whose starred rate per label is
  `stats --by variant`. `CellKey`, `run_id`, resume and the duplicate count
  read the resolved prompt, not the label. A placeholder in the negative
  prompt is still refused.

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
  it.
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
  joining word, six words, the 12-character cap, or Latin lettering going on
  past a space or an ASCII mark (写着OPEN NOW, 9:00, www.shop.com) - and only
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
  reads is not offered.
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
  `CriticUnavailable` is 503, `CriticError` 502.
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
  (`is_reorder`, typed marks folded; for JSON, equal values with the keys
  sorted, each number as written - 0.50 is not 0.5 (`_json_forms`) - and no
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
  unwrapped; one that is the prompt unchanged (whitespace, a closing stop and typed
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
- The size line compares the request against `guide.pixels` a tenth either
  way and quotes `pixels_word`; the Examples menu is built on open from the
  last status, and a pick tells the preview and the panel the way `choose`
  does.
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
  aria-disabled, never disabled, so it keeps focus; an answer takes focus
  (`#critic-result`, tabindex=-1) while focus is still on the button. Check
  pressed on a two-column page moves focus to the panel heading
  (`checkFromButton`), and Check again hands focus to the heading when it
  hides; under 901 px focus stays on Check.
- The prompt check describes one text: the prompt and negative prompt as
  `collect()` sent them, with the reference images it counted
  (`analysisResult.reference`; klein's references rule reads them). Any
  change to either box - typing, every programmatic write through
  `promptReplaced()`, `syncCfg()` greying the negative - or to the references
  (`setReferences`, Reuse, a `reference_supported` flip) marks it stale:
  dimmed, every Use/Apply and Run comparison dead, a finding's words selected
  only while still where the check found them; the badge says "references
  changed since" when only the references moved. A checkpoint change
  clears it. Apply is offered on rule findings only; an encoder suggestion is
  shown as text. Use and Apply write through `replaceBoxText` (so Ctrl+Z
  restores) and check again. Server offsets count code points; the page
  converts them (`units()`) before selecting. The comparison sends
  `num_images: 1`, reads its seed count from the Seeds select when it runs
  (it and Your version are `autocomplete=off`), takes Check's `reordered` as
  its Reordered row, else the critic's as the server kept it
  (`critic.validate` decides; the page adds no rule of its own), never writes
  `els.prompt` or `els.gridPrompt`, drops a version identical to another,
  and opens its run as the sheet when done - Favorites unticked first, and
  the sheet's note says so (`showRun(run, { fresh: true })`), since nothing
  in a new run is starred; its Your version box (`yours`) is the one way to
  test order on prose. A negative-prompt Apply is dead while the negative box
  is greyed.
- Tag autocomplete and comma-item validation use the fixed Danbooru
  tag API through `backend/tags.py`, filtered when a checkpoint documents a
  knowledge cutoff. WAI v17 publishes none: keep its cutoff `None` and the
  live lookup advisory rather than inventing a date. Model-card
  special tags validate locally; lookup failure stays advisory and never blocks.
  On a two-column page validation lives above the results, not in the panel;
  under 901 px the same node returns below Prompt. The prompt check's panel
  follows it, in that order (`placeTagValidation` moves both, keeping focus
  and the selection: `moveBefore` where the browser has it, restored by hand
  elsewhere).
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
  appearance, not alphabetically: the oldest image carrying each value.
  Under Favorites or a search narrower than the run, that is the oldest in
  the whole run, which `learnSheetOrder` asks for once. A skipped first
  cell, a late resume or rows Delete unstarred removed can still move a value
  down. A varying `var:variant` axis takes the rows whenever anything else
  varies, however few of its images are loaded (placed from every varying
  axis, so the listlike rule drops neither it nor the axis across it;
  `VARIANT_AXIS`, held to `grid.VARIANT` by `tests/test_grid.py`), so a
  comparison reads original, reordered, rewrite, top to bottom.
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
database and outputs directory (`tests/conftest.py`). They run on `config.py`'s
defaults: `tests/conftest.py` drops every `LOCAL_DIFFUSION_*` but the gallery
and adapter paths before anything imports `backend`, so a sourced
`wsl/env.sh` changes nothing. They cover the invariants
that are expensive to get wrong: the schedule the scheduler ends up with, seeds
surviving a JSON round trip, the search fallback keeping its filters, and the
HTTP guards.

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
observable), and `node tools/browser.mjs tools/smoke.json` drives headless
Edge or Chrome over the DevTools protocol with Node's own WebSocket - no npm
install - through the lightbox zoom, the sheet, the export, the sheet's
declared variable order (under Favorites too), a sparse comparison's rows,
the re-render button, the panel's persistence and the duplicate/distinct
counts, exiting 1 on a failed `expect`. `smoke-conditioning.json` runs
against a klein-claiming fake on 7864 for the reference chips, the adapter
select, a weight sweep and the prompt check going stale when the references
change (chips, Reuse); `smoke-flux.json` runs against a second fake server
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
prompt boxes, Apply, the critic's first-use and timing lines - built from
`ModelStatus.prompt_critic`: its label, the downloads a first use makes,
neutral wording under an override - and its hold on Generate, a cite's
whose-words note and the General label, Use rewrite with undo and a reload,
an edit while the critic thinks, a two-version comparison opening its sheet,
another opening all its cells with Favorites ticked (the tick's notice gone
once the run is reopened), the chips and the check moving from the wide
host back under the prompt with focus and selection kept, a prose prompt's
order shown as not checked, a one-clause prompt the critic cannot reorder,
a comparison against Your version, the cut excerpt in code points, focus
after Check (two-column), after Check again, after Close during the
critic's check and after the critic's answer, the critic's hold through its
pre-check (a replaced one saying so, a cleared one silent, a throwing one
giving Check back), the stale badge left alone while typing, the Seeds
select read at run time, and, by injecting the server's `reordered_kind`,
the Use reordered title following what the server says it kept, not the
page's own last check, with Compare offering the critic's reordering the
server kept.
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
