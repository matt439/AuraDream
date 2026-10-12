# Design notes

Why the code is the way it is, one section per area. `docs/rules.md` states the
rules in full (`CLAUDE.md` the ones that break things silently); this file keeps the reasoning behind each, the history of what went
wrong before the rule existed, and the evidence, in the notes kept while the
code was written. Read the section for an area before changing it. The
measurements quoted here are indexed by date and checkpoint in `verified.md`.

## The project-owned namespace follows Local Diffusion

**Only checkpoint and vendor names use Krea.** App-owned configuration uses
`LOCAL_DIFFUSION_*`; the default database is `local-diffusion.db`; WSL state
lives under `~/local-diffusion`; browser state uses `local-diffusion.*`; and
new PNGs carry their JSON in a `local-diffusion` text chunk. The PNG reader
still accepts the pre-rename chunk and the browser moves pre-rename state on
first read, so existing images and saved prompts survive without making the
old names the write path.

## Why float8 and model offload

Krea 2 is ~13B and needs **~25 GB in bf16**, which does not fit 16 GB. The
default path quantizes transformer + text encoder to float8 via torchao and
uses `enable_model_cpu_offload()`. Peak VRAM lands near 13 GB.

## Quantization goes through quant_mapping

Quantization must go through `PipelineQuantizationConfig(quant_mapping=...)`.
The shorthand `quant_backend="torchao"` form raises `ValueError` on this
pipeline: the text encoder is a *transformers* model (Qwen3-VL) and its
`TorchAoConfig.__init__` signature differs from the diffusers one. Both take a
torchao `AOBaseConfig` **instance** (`Float8WeightOnlyConfig()`), not a string.

## mu, sigmas and the shift that lands

**`sigmas` is an input to the shift, not the shift's result.** `__call__`
takes `sigmas` and no `mu`, but it derives a mu of its own - 1.15 hard-coded
for `is_distilled`, `calculate_shift(image_seq_len, ...)` otherwise - and
passes it to `set_timesteps`, which shifts whatever sigmas it was handed
whenever `use_dynamic_shifting` is set (it is, in both shipped
`scheduler_config.json`s). The shift multiplies the odds `s/(1-s)` by `e^mu`,
so two shifts compose into `mu_a + mu_b`. Pre-shifting the schedule therefore
*added* to the pipeline's mu rather than replacing it, and every Turbo image
made before that was fixed ran at 2.30 instead of 1.15. `DiffusersBackend`
now overrides the value on its way to the scheduler (`schedule.forced_mu`), and only
falls back to an explicit pre-shifted schedule (`schedule.shifted_sigmas`, the same
formula: `sigma' = e^mu / (e^mu + (1/sigma - 1))`) for a scheduler that does
not shift *dynamically*. **A non-dynamic scheduler still shifts, statically**:
`set_timesteps`'s else branch multiplies the odds by `self.shift` (1.0 on
the shipped checkpoints, 3.0 on SD3/3.5, 6.0 on Lumina 2) on top of whatever
sigmas it was handed, so the fallback pre-shifts by `mu - ln(shift)`
(`_static_shift`, read off the instance) and the total is still `e^mu`. At
`shift == 1.0` that is byte-identical, which is exactly why a test pinned to
1.0 hid it. `shift_terminal` and the karras/exponential/beta options
post-process in both branches and get a warning at load. Assert on
`scheduler.sigmas`, never on the list the backend computed - see
`tests/test_schedule.py`.

## Where guidance switches on, and applied()

Turbo is distilled: 8 steps, `guidance_scale=0.0`. **Where guidance switches
on is a property of the pipeline, not the constant 1.0**: `FluxPipeline` runs
an uncond pass only at `true_cfg_scale > 1`, while `Krea2Pipeline` guides from
0 up and substitutes `""` for a missing negative prompt - so the old `> 1.0`
gate silently guided Turbo against an empty prompt in the band `0 < g <= 1`.
The backend keys the threshold off `"true_cfg_scale" in self._call_params`,
and `/api/status` publishes it as `defaults.cfg_threshold` for the UI and the
CLI. Dropping the negative prompt below the threshold saves nothing - neither
pipeline encodes it outside its guidance branch - it exists to keep the
recorded settings honest about what ran, **and that only holds if the drop
reaches the row**: `GenerateRequest.applied()` is the one normalisation
(negative prompt gone where unsupported or at/below the threshold, mu gone
where the pipeline has no way to take one), `backend.applied()` answers it
from the loaded parameter list or, unloaded, from the profile, and both save
paths write `applied()` rather than the request. `api_generate.expand_grid` and the CLI
run the same normalisation over the cells before `prune_done`, so a resume
whose negative prompt the pipeline never saw still matches the rows it
wrote. The UI's `collect()` sends null for a greyed-out field, and the field
is gated on the highest CFG the open grid's axis reaches, not the panel's
single value - a grid decides guidance per cell.

## Nunchaku Lite: pre-quantized repos, Linux only

`LOCAL_DIFFUSION_QUANT=nunchaku` loads a checkpoint that is *already* quantized (Nunchaku
Lite, SVDQuant W4A4; `quant_method: nunchaku_lite` in the transformer's
`config.json`, bitsandbytes NF4 in the text encoder's) and passes diffusers no
quantization config at all - the repo's own configs name the quantizer, and
torchao on top would be wrong. The CUDA ops arrive through the `kernels`
package from `rootonchair/nunchaku-lite-kernels` (branch `v2`, pinned by
diffusers), which publishes x86_64 **Linux** builds only - torch 2.11-2.13 for
cu126/cu130/cu132 - so on this machine the mode means WSL 2. NVFP4 needs
Blackwell (sm_120 qualifies); the `int4` repos are for older cards.
`loading.nunchaku_preflight` refuses off Linux, without `kernels`/`bitsandbytes`, or
on a repo whose transformer is not pre-quantized, *before* any download: the
bf16 case otherwise loads fine and OOMs at the first image. diffusers reads
`DIFFUSERS_TRUST_REMOTE_KERNELS` once, at import, so `config.py` sets it
(`setdefault`) - it is imported before diffusers everywhere, the CLI included.
Under WSL, `wsl/env.sh` moves the venv (`UV_PROJECT_ENVIRONMENT`), the gallery
(`LOCAL_DIFFUSION_OUTPUTS_DIR`, `LOCAL_DIFFUSION_DB_PATH`) and the HF cache off `/mnt/c`: 9P is slow
and SQLite cannot lock over it. The original `nunchaku` engine is a different
package with its own transformer classes and no torch 2.13 wheel; not used.

The same file exports `LOCAL_DIFFUSION_QUANT=nunchaku` and a lite-infer model
id, and six tests written against the default checkpoint and fp8 failed under
WSL because of it: the suite read whatever the shell had. `tests/conftest.py`
now drops every `LOCAL_DIFFUSION_*` except `OUTPUTS_DIR`, `DB_PATH` and
`LORAS_DIR` before its first `backend` import - a fixture is too late, since
`settings` reads the environment once, on import - and asserts that nothing
imported `backend.config` ahead of it. The paths stay so that a test which
fails to redirect them writes to this environment's gallery rather than the
Windows one under `/mnt/c`.

## torch 2.13's Triton kernels under WSL, and the xet downloader

**torch 2.13 runs some CUDA ops as Triton kernels in eager mode** (`torch._native`
overrides `aten::bmm` and others), and Triton compiles a C helper on first use.
A fresh WSL Ubuntu has no C compiler, so the Qwen3-VL text encoder's rotary
embedding died on the first step with "Failed to find C compiler". `wsl/env.sh`
sets `TORCH_DISABLE_NATIVE_JIT=1`, which torch honours by declaring those
overrides unavailable and dispatching to ATen. Verified 2026-09-05: Krea 2
Turbo NVFP4 via Nunchaku on the 5080, 8 steps, model offload: 1 MP in 15 s at
8.1 GiB peak, 2 MP in 24 s at 8.6 GiB. `LOCAL_DIFFUSION_OFFLOAD=none` does **not** fit
this checkpoint: 14.7 GiB at 1 MP, 18.7 at 2 MP, and the WDDM driver spills
the excess into system RAM through WSL just as it does natively - the images
then took 26 s and 72 s. Offload's per-image weight transfer costs ~5 s and is
the cheaper of the two. The same file pins the xet downloader's
concurrency (`HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY=false`), because its
adaptive controller collapsed to one connection and stalled a 12 GB checkpoint
twice on this connection.

## Krea 2's components and offload order

Components are `text_encoder` (Qwen3-VL), `transformer` (Krea2Transformer2DModel),
`vae` (AutoencoderKLQwenImage). Offload order is `text_encoder->transformer->vae`.

## Loading happens in one place

Loading happens in exactly one place: `main.ensure_loaded()`, which holds an
asyncio lock across the whole load. Jobs `await` it rather than calling
`backend.load()` from the worker thread — doing both raced two concurrent
loads per request and produced duplicate tracebacks. `generate()` raises
rather than loading a missing pipeline, so that rule is enforced rather than
merely documented, and `POST /api/model/unload` answers 409 while any job is
live: unloading under a running job freed nothing (the worker holds its own
reference) and had the next run of a sweep reload from the gen thread.

**A load waits out the generation gate before it moves the weights.** A
queued job awaits `ensure_loaded()` *before* it takes the gate, and the load
itself never took it, so a job submitted while the prompt critic held the
gate (its `llama-server` has ~5 GB of the card) started loading 13 GB beside
it - `tests/test_api_analysis.py` recorded `load` before `critique done`
with the fix removed. `ensure_loaded()` now awaits `jobs.wait_for_gate()`
inside `_load_lock`, once it knows a load is needed: the gate is acquired and
released at once, which waits out whatever holds it without keeping it. That
covers every load - a job's, `POST /api/model/load`'s, Expand's - and, being
inside the lock, the critique route's "the image model is loading" refusal
covers the wait too. `asyncio.Lock` serves waiters in order, so a critique
that has passed its second busy check either holds the gate already or is
queued ahead of any later load. It cannot deadlock because nothing that holds
the gate awaits the load lock: `_run` loads before it enters, and
`run_when_idle`'s operations are synchronous work that never calls
`ensure_loaded`. (Expand's first use does register ERNIE's rewriter inside
the gate, but through the backend, not the load lock.) The cost: a job that
waits out a critique reports `waiting_on="load"` while the status does not
yet say `loading`, for the critique's fifteen seconds or so.

**A failed load says what to do, once, where the work is.** The default
checkpoints are gated, so a new user's first Generate most often ends in
huggingface_hub's `GatedRepoError`, whose text was a request id, a URL to
`model_index.json` and "Please log in" - six red lines under Generate,
re-posted on every 3 s poll over whatever else the line said, mirrored into
the lightbox, and repeated by the morning-after banner after a reload.
`ensure_loaded`'s except maps the exception by class name over its MRO, and
over its `__cause__`/`__context__` chain (a loader's OSError raised from the
Hub's error), as `prompt_encoding._reason` does for tokenizers:
`GatedRepoError`, `RepositoryNotFoundError`, `OfflineModeIsEnabled`,
`LocalEntryNotFoundError` (`main._load_failure`). Each becomes a sentence with
the steps - for a gate, the catalogue's own word for it (`launch.CHECKPOINTS`:
a click-through, or the authors' approval), `uv run hf auth login`, then Retry
load - and `ModelStatus.detail_kind` and `detail_url` (the model page) beside
`detail`; anything else is `short_error` as before. The log keeps the
library's words. It is raised on as `jobs.LoadFailed(detail)` from the
original, so the job that ended at the load carries the same sentence as the
status (`_run`) and Expand answers with it too. The page puts it in the
viewer's empty state - a card with the sentence, a link to the model page and
a primary Retry load, the topbar's Load handler, which reads Retry load
meanwhile - and says it in the panel's line only where the viewer holds images,
once per failure (a retry that fails again counts as a new one) and never in
the lightbox (`syncLoadFailure`). A job that ended at the load posts nothing
of its own, but it forgets the detail posted (`shownLoadDetail`): the press
that queued it cleared the panel's line, and a Generate on a gated
checkpoint with images in the viewer used to end in silence, the steps gone
from the page and never posted again. The banner skips such a job while the
status still says the same. A card that goes with focus in it - its Retry
load pressed, the load on its way - hands focus to Generate, or to the new
card's Retry load where the detail changed, not to the page's body.

**`HF_HUB_OFFLINE` is named wherever the Hub's miss came from it**
(2026-10-05). For an uncached file under `HF_HUB_OFFLINE`, huggingface_hub
1.29's request hook refuses the HEAD with `OfflineModeIsEnabled` and
`_raise_on_head_call_error` raises `LocalEntryNotFoundError` *from* it.
`_hub_failure` took the first class on the chain, so the card said "Hugging
Face cannot be reached: connect to the network" while the network was up
and the variable was the cause (`docs/verified.md`, Behaviour). It now
prefers `OfflineModeIsEnabled` wherever it is on the chain, and is
first-found otherwise. The test builds the chain with the library's own
`hf_hub_download` under its offline switch (`tests/test_api.py`), so a
release that chains it differently fails there rather than on the card.

**What became of the job, and where the load stands now** (2026-10-05). A
Generate that ended at the load sent the panel back to Generate with no word
of the image it had queued. Its finish line now says "Not made: the model
could not load - why, and Retry load, are in the viewer" (or "why is said
below" where the viewer holds images and the panel's line has the reason), a
`.not-made` span (`notMadeText`), said once on the hidden polite `#job-said`
where the card holds the reason, since the card is no live region. That
sentence went stale: retried here or in another tab, or unloaded since, the
card and the Retry load it pointed at were gone, and it went on saying so
over a ready topbar while the other tab kept the red detail in its panel's
line. So once the status leaves `error`, in every tab, `syncLoadFailure`
rewords it ("could not load then; it is loading now", "it has loaded since",
"it is not loaded now") and takes back the detail it posted to `#error`
(`loadDetailInError`, that one only). Expand answers a load that fails as a
503 carrying the status's sentence, which the page turns into a pointer to the
card where the card is shown (ERNIE-Image Turbo, below). The topbar's Load
and the card's Retry load take back only what was said about the load - the
press's own refusal, Expand's pointer (`loadLine`), the status's sentence
(`shownLoadDetail`) and Expand's words about the load under the prompt
(`aboutLoad`) - never an Undo on offer or a job's outcome in the same line.

## Switching checkpoints while the server stays open

The browser's **Models > Switch & load** selects another catalogue checkpoint
without restarting the process (2026-10-08). Startup `Settings` remains frozen;
`model_selection` owns one immutable `(model_id, revision, quantization, offload)` snapshot, and
`current_model_id(settings)` supplies the model to defaults, the loader, jobs,
saved metadata, resume lookups and gallery scope. `current_settings(settings)`
supplies a frozen copy with the effective loader options. Quantization and
offload can change through the same exclusive operation, even for the same
checkpoint; the revision advances so old requests are refused. Nunchaku offers
its nvfp4/int4 catalogue on Linux. An explicit startup id outside the catalogue
remains available under its startup quantization for switching back;
the HTTP API cannot introduce arbitrary repository ids or paths.

Switching requires no queued, running or paused jobs, no load or card holder,
and no model-dependent request in flight. That last condition starts at HTTP
middleware, before Pydantic fills checkpoint defaults: a Generate waiting on
adapter or source validation must not resume under a new model. It also covers
CPU prompt checks and scoped bulk deletion. Those requests in flight are
waited out for up to `SWITCH_DRAIN_SECONDS` (10 s), new ones refused
meanwhile, before the switch refuses: refusing at once failed the switch
whenever it was pressed just after a load landed, when the page's own adapter
list, token count and tag check are out (a first tokenizer fetch or a slow
Danbooru holds those for seconds on the real server), and the refusal then
stayed in the picker and the error line though nothing was busy any more
(2026-10-09). A refusal after the wait names the paths still counted, and a
later accepted switch clears the earlier refusal from the error line. The
browser sends its observed
revision so an old tab cannot submit old controls, or confirm a bulk deletion,
against the model another tab just selected. Status and model-list reads stay
available. The switch reserves its state before yielding, stops the prompt
helpers and then releases the old weights under the load lock, updates the
selection, then loads only through `ensure_loaded`. A background task keeps
that reservation if the browser disconnects. Load failures retain the selected
checkpoint and normal Retry load behavior. LoRA compatibility caches are
invalidated by checkpoint id.

A teardown failure once unloaded A, left it marked "error" with a bare
`RuntimeError`, and kept B in the saved file, so the next start auto-loaded
the choice that had failed (review, 2026-10-09). Helpers now stop first - a
critic that will not die aborts with A still loaded - and anything failing
before `select()` restores the previous preference file and auto-load flag and
reports `switch_error` ("Could not switch to B: ...; A is still selected")
instead of a load error. Revisions start from the process's start time, so
a long-hidden tab's revision 0 is no longer accepted by a restarted server.

The page follows both the model id and revision, including an A-to-B-to-A
change between polls. It retires old prompt answers and prepared Refine state,
refreshes guides, controls, adapters and gallery scope, and restores sampling
remembered for that checkpoint. Prompts and saved images remain. Old submissions
can retry only after switching back to their checkpoint. `api()` once kept its
own copy of the middleware's guarded routes, held equal by nothing, so a route
added on one side alone lost its stale-tab refusal (review, 2026-10-09). It now
sends the revision on every `/api/` request and the server decides; only the
advisory routes' after-the-answer check stays in the page.

The same picker now replaces the terminal questions (2026-10-08). **Switch &
load** and **Apply & save** atomically remember model, quantization, offload and
the startup-load checkbox in `local-diffusion.settings.json`, beside the
database. Save happens before unloading: a write failure leaves usable weights
in place and reports the error. Apply & save unloads changed settings without
starting a download; saving only the checkbox leaves an unchanged ready model
alone. Preload is false initially. An enabled checkbox loads through the same
background `ensure_loaded` path on the next launch, keeping the browser available
for progress and recovery. Prompts and sampling continue using their existing
browser persistence.

The first **About** disclosure repeated long guide paragraphs in small type and
pushed the load buttons down when opened. The picker now shows a fixed row of
compact facts instead: a short style label, Refine support, recorded speed and upper MP
guidance. The user preferred maximum recommendations to starting sizes, so
the size fact shows one upper figure. Its label distinguishes a recommendation,
documented/native maximum or reference/example size; the tooltip carries the
source and qualifications. An author's example does not become a guaranteed
maximum, and a documented range is not a measured quality ranking. A custom
launch id gets unknown/after-load facts rather than another model's guide.
Each model card stays one link away; full prompting guidance stays beside the
prompt controls.

The checkpoint is chosen from a table that shows every model's facts at once
(2026-10-08). With a dropdown and one facts block, comparing meant choosing
each model in turn. The picker is opened rarely, mostly at launch, so a large
panel (up to 900 px wide, its body scrolling under fixed actions) costs little.
Each row is a radio (Model, Style, Refine, Size with its label, Speed
with its sample and basis), its star and its model-page link; the label is
`display: contents` on a shared column template, so cells line up and a
press anywhere on the facts chooses the row. Below 700 px the facts take a
second line, and below 480 px two facts a line. Quantization and offload sit
side by side above the table, since they decide which checkpoints it offers
and which timings its Speed column shows; their explanations reserve their
height. A press on a row's gaps chooses the row, and the picker body takes the
focus for presses on its empty space, so the focusout check keeps it open.

A review of the table (2026-10-09) found it hard to read past the first
screen and hard to use without a mouse. The changes:
- The column names scrolled away after one wheel step, leaving 14 rows of
  bare values. The heading row is sticky in the picker body; its offset takes
  back the body's padding, and the body's `scroll-padding-top` keeps a row
  the arrow keys reach from scrolling in under it.
- Nothing marked the server's checkpoint, so after another row was chosen the
  loaded one looked like the rest. Its row now carries `data-state` and a dot
  and a word (Loaded, Loading…, Load failed, Current). They sit on the name
  cell's second line, which every row keeps for its access label, so the
  badge never shifts a name or a row. `syncModelRowStates` writes it in place
  outside the rebuild key, so a poll changes one word, never the rows.
- The header said Max size over values labelled Reference size or Largest
  example. It now says Size; the row labels keep the distinction, and the
  header's Refine, Size and Speed carry titles.
- The cells' notes (sources, caveats, no GPU memory guarantee) lived only in
  `title`s, which keyboard, touch and screen readers never reach. Each radio
  is now described by a hidden span joining them. A row under the pointer or
  focus dots the noted labels; on every row at once, 23 underlines were noise.
- Each row's star and link were tab stops: 44 between a radio and Switch &
  load. They are now one roving stop, the chosen row's (the first row's while
  none is), as the gallery grid has one stop. The row holding the focus keeps
  its stop until focus leaves, so a star pressed by keyboard stays reachable.
- A row drawn in Favourites has a hidden "Favourite, " before its name, from
  the opening's grouping, so a star toggled meanwhile does not rename the
  radio. Groups are not `role=group` wrappers, which would break the
  `display: contents` grid. The second group is Other models, since the
  gallery's All models tick means something else.
- The section is named Models after its opener. The Checkpoint heading names
  only the radiogroup. The model link says it opens a new tab.
- A Reference or Unmeasured speed reads muted; only a category timed with
  these loading options reads at full strength.
- The Nunchaku format list said which card each format is for only in its
  options' titles. A line under it now lists both ("NVFP4: RTX 50
  (Blackwell); INT4: RTX 20 to 40") and describes the select. It sits
  outside the select's label, which would otherwise add it to the name too.
- From 1800 px the picker grows with the wide tier's type, to 1200 px, so the
  Speed basis no longer wraps to three lines.
- The picker's rules moved to `styles/model-picker.css`, linked right after
  `style.css`, the position they held in it.

On a fresh browser visit, the first accepted status opens Models if the server
is idle and unloaded or has a load error. A ready model, an active load or work
elsewhere keeps it closed. This decision is made once per page visit: later
polls, manual Unload and a dismissed picker never reopen it. Automatic opening
does not take focus or move the prompt caret.

Hovering a row in the open checkpoint list made the highlight jump a few rows,
first up, then (after a fix that stopped polls reassigning unchanged values)
down. That fix had the wrong cause. Chromium's source shows a same-value
assignment never reaches an open native popup, and on any refresh its list
re-selects the row under the last pointer position. A still pointer can only
land on another row when the rows move: the popup is a separate window that
Chromium re-anchors (rewritten in M154), or rows are added or removed. Page
code cannot correct the window's placement, so the picker's selects now use
`appearance: base-select` (Chromium 135+) and draw their lists in the page,
where hover is plain `:hover`. Other browsers keep native lists. (The
checkpoint dropdown has since become the table above; the loader selects
keep this treatment.)

An in-page list moves under the pointer (and loses its focus) if its options
are rebuilt while open, so while any picker list matches `:open` catalogue
replies, other tabs' favourites and a status write to that list wait for it
to close, then land. The other controls take a server change at once, so a
checkpoint picked from the open list still sits on the new loading options.
Its keys now reach the page, which leaves them all to the list: Escape closes
the list rather than the picker or the prompt check, and Ctrl+Enter runs nothing. Unchanged catalogue
replies keep the existing option nodes. `modelPickerAppliedSettings` still
limits poll writes to fields that changed; a catalogue reply records what it
installed, or the id it asked for when the list lacks it, so an off-catalogue
runtime id is requested once instead of rebuilding the list on every poll.
Edits made while a poll's own catalogue refresh is in flight are merged, not
replaced, and a failed list retries every ten seconds while the picker is open.

The speed fact uses the retained RTX 5080 records in `verified.md`, including
the October 5 Animagine sample that the anime guide previously omitted. It
shows Fast (up to 15 s), Medium (over 15 through 45 s) or Slow (over 45 s), based
on the upper time of the recorded roughly 1 MP sample, plus that sample's
time and area. WAI retains its 1.38 MP portrait. Exact checkpoint variants and
quantization/format/platform/offload determine whether the sample matches.
Until 2026-10-09 an unmatched setup kept the sample's category and changed only
its basis ("NVFP4 reference"), so on Windows Krea 2 Turbo read Fast ~15 s for
FP8, and for BF16 with sequential offload too, from a 4-bit Linux sample. The
category is now only a matched sample's. Otherwise the word is Reference, or
Unmeasured for BF16 or sequential offload (too far from any sample to call),
with the sample's time kept and a plain basis naming what differs ("timed on
4-bit Linux build", "timed with FP8, model offload"). The closest sample is
picked by exact quantization/format, then its family (4-bit against torchao),
then platform, then offload, so INT8 on Linux borrows the Windows FP8 sample
rather than NVFP4. Busy-CPU first-image samples say First run, busy PC.
Missing timings say Unmeasured and show default steps, never an extrapolated
speed from those steps. The tooltip preserves settings, date and warm/cold
caveats, without file paths. These are single prior runs, not a controlled
comparison, live ETA or measurements at the maximum MP shown beside them.

The catalogue also reports Hugging Face access (2026-10-09). The `--menu`
printed each entry's gate, but the browser picker that replaced it did not, so
Krea 2 Turbo's manual approval surfaced only as a failed load, after the
switch had already released the working model. `details_for` maps the gate
word: click-through is Accept licence, manual approval is Needs approval
(which can take days), other entries and lite-infer re-uploads Open, a custom
id Unknown; the note says what to do on the model page. It is a catalogue
fact, not a Hub or cache check: a network check would hold up the cold
catalogue, and a pre-unload access check is left for later.

The quantization and offload explanations were vaguer in the browser than in
the menu, while the menu's figures were Krea 2's (~13 GB peak, ~25 GB in BF16)
and would be wrong for the SDXL rows. `loader_choices.py` now words both once:
FP8 and Model offload are recommended for a 16 GB card, BF16 and No offload
need the most VRAM (24 GB+ for FLUX-size models), with no per-model figures.
Changing draft offload settings refreshes these facts as changing quantization
does, without applying either setting or loading a model.

Each row's star saves a browser-local favourite immediately. The table groups
Favourites first and Other models after, without duplicating rows, but only when
it is drawn for an opening (or while the picker is closed): a star pressed in
an open picker changes in place, because regrouping then would move the row,
and the next one, under the pointer. The canonical catalogue id keys the star so
ordinary and Nunchaku formats share it; Anima Turbo and Aesthetic retain their
distinct identities. Starring preserves the choice and unsaved loader edits,
does not load weights, and keeps button focus. A table rebuilt for new loading
options returns focus to the same model's control, since losing it would close
the picker. Browser storage is guarded; when unavailable the session can keep
stars while the page reports that they could not be saved. Other tabs' storage
changes update the stars, and the grouping at the next opening.

These details require no weights or tokenizer load. Unsaved picker edits
survive status polling; a changed server revision retires them. Until
2026-10-09 one dirty flag froze every field, so after an offload edit another
tab's checkbox-only save (no revision change) never showed, and the next
Apply & save sent the old checkbox back. Edits are now per field, and the
others keep following the server. A choice lost to a catalogue for other
loading options also stayed lost: changing back left no row chosen. The last
deliberate choice (`modelChoiceIntent`) now returns by exact id, or by
canonical id to the same checkpoint's other build (Nunchaku formats).
Quantization catalogue requests use an epoch so delayed responses cannot
restore an obsolete list.

The first opening focused the body until the rows arrived, and the rebuild
put the focus back before checking the chosen row, so it landed on the
first row: one arrow key changed the checkpoint to one never looked at.
And `preventScroll` left a chosen row low in the table out of sight, with
nothing visible checked or focused. The rebuild now checks first, and
opening (or a rebuild's refocus, unless the pointer is on the body) scrolls
the picker's own body to the focused row below the sticky head. That is the
newly opened popover showing its current entry, not page scrolling: the
window never moves, and a hovered body is left alone.

The note under the actions said the wrong thing at the wrong time (review,
2026-10-09). "Settings saved" was wiped by the revision its own switch made,
a few seconds later, and named nothing; a failed Switch & load left it or the
generic first-load line, with the failure only in the viewer under the
picker; a refusal stayed after its reason had gone; a failed catalogue read
"still loading"; the live note repeated the first-load sentence on every
opening; and Apply & save unloaded a loaded model without a word. The note
now holds a request error, a refusal while its reason holds, the server's
`switch_error`, a failed load's detail, or an accepted press's outcome by
name with the `startup_overrides` that beat it, then Loading, the unsaved
edits or the server's state. Holds stay in titles. The revision reset keeps
the outcome only for this page's own switch; a failure replaces it. There is
no load timer: status has no load start, and a page clock would disagree
with another tab's. Apply & save's title and the table's help say what it
unloads. The startup checkbox moved from under 23 rows to the footer, and the
help went `.sr-only`, read with the table but not announced. Edits kept
across a closing looked like the running settings; the note now says
"Unsaved changes" with the server's settings and a Revert floated into its
first line, where its scrolling never hides it. `startup_overrides` reports
only which preferences an explicit model set aside, not whether it was the
saved model, so the sentence says both cases.
CPU and fake-browser checks establish these transitions, not real GPU load
times, memory release or image quality.

The picker shows the chosen row's own images (2026-10-09). The table's Style
column says "Photorealism" or "Anime" in a word; what a checkpoint actually
makes, at the settings this machine used, is best answered by the pictures it
made. They are in the gallery already, but the picker covers the gallery,
and finding them there meant closing the picker, ticking All models and
typing a search, then opening the picker again to choose. So the strip is
inside the picker, under the table and above the startup checkbox, or a
column on its left on a wide enough window (below): "Made with
<name> (<count>)", the newest pictures, and See all. It follows the
chosen row, not the server's checkpoint, and changes nothing until a press:
choosing a row only reads. See all is the explicit way out, closing the
picker and pointing the real gallery at that checkpoint.

Which images belong to a checkpoint is the server's: `/api/gallery?model=`
is the gallery's own exact scope (`model_id = ?`, Anima's are not
Anima@aesthetic-v1.1's), in place of the running checkpoint's and of
`all_models`. See all writes the same scope as one search word,
`model:=<id>`, since the box's `model:` is a substring and
`model:circlestone-labs/Anima` would also count the Aesthetic variant; the
grid's count is then the strip's. It is a gallery read, not model work, so it
is in no model-route list: it answers while the server is cold, loading or
switching, and under a stale revision. Bundle maintenance refuses it like
every gallery read; the strip says so in its own box and a status poll asks
again. The strip asks for only the pictures its grid holds whole (up to 24:
the grid's resolved columns times the rows its height takes, measured when
the request goes), after a 180 ms pause on a new choice, so
arrowing through the rows asks once; a newer choice aborts the older request,
and an epoch drops a reply that still arrives. The old pictures wait dimmed
and inert until then, rather than leaving a gap.

Under the table the strip has a fixed height (a head line and one row of
`--model-shot` pictures, smaller on a window under 640 px tall), filled,
empty, failed or waiting alike, so the checkbox and the actions never move
under the pointer. There it took about 95 px from the table at 1400x900,
and its 64 px squares (40 on a short window) were too small to tell one
picture from another; the gallery's own smallest cards are 150 px. The
gallery under the open picker is covered and cannot be used anyway, so on
a window with room (from 1280 px wide, 2026-10-09) the strip is a column
on the picker's left instead: the picker grows leftward by it, the pictures
are `--model-shot-side` (the gallery's S density: 150 px, 150-180 px from
1800), two across, three from 1800, in as many whole rows as the picker's
height holds - four rows of two (8 pictures) at 1400x900, six of three (18)
at 2560x1305. The table side keeps its 900 px (900-1200 px from 1800) and
its layout and gets the strip's height back: its body shows 653 px at
1400x900, nine rows from FLUX's. The column is out of the flow (absolute in
the picker's padding box, which `padding-left` reserves), so the table side
alone sets the picker's height and the column clips to it; nothing it holds
moves the table side. The 900 px table, the 341 px column and a gutter
each side make 1277 px, and `100vw` counts a classic 17 px scrollbar that
the right-aligned picker's containing block does not, so the picker's
width keeps 17 px back from it: from 1294 px all of it fits whole. The
first cut was 1300 px, which left a 1280 px window - a 1920 px laptop
screen at 150%, a common one - with the 64 px strip; from 1280 the table
side gives up to 14 px instead (886 px at 1280), which it already does on
any window under 936 px with the strip, and the left edge keeps its
gutter. Under 1280 the strip stays under the table. This is the picker's
own layout breakpoint, not the wide tier's scale, which is why it is not
at 1800. Between 1280 and about 1850 px the wider picker also covers part
of the settings panel, not only the gallery (from 465 px at 1400x900 its
left edge goes to 126, over the panel's last 222 px): the picker already
closes on a press outside it, and the panel cannot be used through it
either way, so nothing is lost that a press elsewhere did not already
give back. Three columns from 1800 make the picker 1399-1407 px of 1764
there and 1756 of 2524 at 2560; a fourth would leave about 200 px of the
window at 1800. The DOM, and so the tab order (link, See all, one picture
stop, checkbox), is the strip's in both places. Whole rows leave a band
under the last at most a picture high (about 145 px at 1400x900, under
four rows of two); a cut row would say there is more to see in a way the
count and See all already do.

The first column did not follow a resize while the picker was open: a
picker opened at 1200 px (twelve in one row) and widened to 1400 drew those
twelve into a two-across column holding eight, its fifth row cut through,
and one made shorter kept a bottom row clipped from view but still in the
arrow keys' walk. `refitModelShots`, on the window's resize, now drops the
oldest at once when fewer fit (newest first, so what stays is what a
request would answer) and, when more fit, asks again in place after the
resize rests, the pictures shown staying undimmed meanwhile.

The column also made a press on the picker's blank space a way to lose a
choice. A press on something unfocusable moves the focus to the nearest
focusable ancestor, or to the page; the picker's focusout close then saw
the focus leave and closed it, unsaved choice and all. `#model-picker-body`
was already `tabindex="-1"` for this; the strip's band under the table had
the same gap but was 64 px high, while the column's blank space under a
short set of pictures, or its empty or failed line, is most of the
picker's height. The footer, which holds the strip and so the column, is
`tabindex="-1"` too (no outline on it). The pictures
are one tab stop (arrow keys walk them), like the
table's stars and links, so Tab still reaches the actions in a few presses. A
picture opens in the lightbox, which is modal and above the picker; the
picker's outside-press and focus-out closes ignore it while it is open, the
user's own close focuses the picture again, and a close that hands the page to
something else (Reuse, Refine, a search) closes the picker too. A delete
anywhere (`forgetImages`) asks again for what is left.

## The loader is model-agnostic

The loader is model-agnostic: `DiffusionPipeline.from_pretrained` resolves the
class from `model_index.json`, and `_component_libraries()` reads the same file
to decide which components to quantize and with whose `TorchAoConfig`. FLUX
layouts therefore pick up `text_encoder_2` (T5) automatically.

## Anima Turbo: a standard interface over modular inference

**Anima Turbo v1.1** uses the official `circlestone-labs/Anima` selection.
`anima_loading.py` pins the full `anima-turbo-v1.1.safetensors` checkpoint
and its revision, and converts it once on CPU into an atomic cached component
tree. The official pinned `Anima-Base-v1.0-Diffusers` repository supplies
configs, tokenizers, Qwen3 encoder and Qwen-Image VAE only. Its Base denoiser
and text-conditioner weights are excluded: both learned components come from
the Turbo file. Turbo v1.1 uses the `model.diffusion_model.` prefix; split its
`llm_adapter` into `AnimaTextConditioner` and use Diffusers' Cosmos mapping
for the rest, with strict state-dictionary loading. Recipe changes invalidate
the cache. Settings and saved metadata retain the official repository id.
That id denotes this pinned Turbo recipe in this app; adding another Anima
variant must give it a distinct recorded identity.

Diffusers 0.40 exposes Anima through modular blocks, whose pipeline call
does not meet this app's generator/callback signature contract.
`AnimaTurboPipeline` registers the standard components on `DiffusionPipeline`
so the existing owned quantization configs and explicit device/offload path
still apply. It runs the upstream text encoding, conditioning, noise, schedule
and decode blocks. Its only denoising-loop extension invokes the app callback
after each completed scheduler step and lets cancellation propagate. Cleanup
offloads models in `finally`, including a cancelled call, and clears the
callback. CPU generator lists flow unchanged to the upstream noise sampler.
The offload chain is encoder, conditioner, transformer, VAE. The small text
conditioner and VAE remain bfloat16; the transformer and encoder use the
selected fp8/int8 configs. Nunchaku is refused before downloading.

Turbo defaults to 8 steps and CFG 1 (the author recommends 8-12 steps).
A fresh guider is built per call, enabled only above 1, matching recorded
negative-prompt normalization even at CFG 0. The scheduler is the official
Diffusers conversion's FlowMatch Euler with shift 3. Mu, reference images
and user LoRAs are unavailable in this integration. Five-dimensional image
latents are already in Qwen-VAE space; the preview adapter exposes them to
the existing fitted Qwen map. Author documentation covers 512 squared through
1536 squared pixels, tags and natural-language descriptions. ComfyUI sampler
alternatives and prompt-weight syntax are not promised by this adapter.

The Example is the card's dragon caption whole, both of its sentences
(checked 2026-10-05). The app had cut it to its first sentence, and Check
then answered the authors' own example with its "1 sentence; the guide asks
for at least 2". The size prose names the 1, 1.5 and 2 MP tiers the strip
marks in range, and the size line quotes the bounds as written, 0.26-2.36
MP (The prompting guide). The integration's two refusals are final before a
load: user adapters are refused by the id (LoRA adapters, below), and the
Refine dialog says Anima has no pass without hedging that a load will decide
(The refine pass, `_OWN_PIPELINES`).

Sources: [official model card and Turbo files](https://huggingface.co/circlestone-labs/Anima),
[official shared component tree](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers),
[Diffusers conversion](https://github.com/huggingface/diffusers/blob/main/scripts/convert_anima_to_diffusers.py),
and the installed Diffusers 0.40 Anima modular blocks. Runtime evidence belongs
in `verified.md`; CPU stub tests and fake-browser checks do not measure quality
or VRAM.

## Anima Aesthetic v1.1: a distinct selection from the same repository

Added 2026-10-07. `circlestone-labs/Anima@aesthetic-v1.1` is an app selector
for the official `split_files/diffusion_models/anima-aesthetic-v1.1.safetensors`,
pinned at `594c27fea35648b87c86a9b4d5436a6024c820b5`. It is not a separate
Hub repository. Its release SHA-256 is
`3c1868387a3a1ff504bbb87c33678321965ead381fcf87afbd0264daa600c082`.
The loader translates the selector to the real repository; tokenizer reads
use the pinned shared component repository, and load-error links use the
real model page. Settings, PNGs, gallery scope and Resume retain the selector
to distinguish it from Turbo. The bare Anima id and Turbo's conversion recipe
and cache path are unchanged.

Both v1.1 files have the same 685 tensor names, shapes and BF16 dtypes
(567 denoiser and 118 conditioner tensors), checked against their pinned
safetensors headers. The shared strict conversion therefore handles both;
Aesthetic's own conditioner comes from its file. Its recipe and cache are
separate. `AnimaAestheticPipeline` inherits the existing upstream-block
implementation, changing call defaults to 30 steps/CFG 4. The author gives
30-50 steps/CFG 4-5 and says Euler suits Aesthetic. The scheduler remains
the official Diffusers conversion's FlowMatch Euler, shift 3; this does not
implement the card's alternative ComfyUI samplers. Mu, references, user
LoRAs, Refine and Nunchaku remain unavailable, with the same before-load
refusals as Turbo.

The guide shares the family size, spelling, artist and prose guidance, but
replaces Turbo's distillation/CFG-1 text. The Aesthetic-specific advice says
quality tags are unnecessary, `masterpiece, best quality` may stay, and
`score_*` should be omitted from both prompts. Its critic grounding excludes
the generic Anima positive prefix containing `score_7` and includes those
Aesthetic exceptions. CPU tests cover both pipelines' guided arithmetic,
seeds and cancellation, cache isolation and loader selection; API tests
hold saved identity and Resume apart. `smoke-anima-aesthetic.json` checks the
profile, guide, control refusals and applied saved settings in the browser.

Sources: [official release](https://huggingface.co/circlestone-labs/Anima/commit/594c27fea35648b87c86a9b4d5436a6024c820b5),
[Aesthetic prompting](https://huggingface.co/circlestone-labs/Anima#aesthetic-version-prompting),
and the shared component/conversion sources above. GPU evidence is recorded
separately in `verified.md`.

## Profiles key off the checkpoint id

`model_defaults.py` keys recommended steps/CFG/mu off the checkpoint id, not the
pipeline signature: `Krea2Pipeline` reports 28 steps at CFG 4.5, which is the
family default and wrong for Turbo. `/api/status` serves the profile and the UI
seeds its controls from it once, on load.

## mu is Turbo-only

`mu` is Turbo-only. FLUX derives its own resolution-dependent shift, so forcing
Turbo's constant on it degrades output; the profile leaves mu null there. The
panel's Mu box says "auto" while blank, where Mu applies at all, rather than
sitting empty beside True CFG's "off"; the grid's Mu axis suggests
`none,<the profile's mu>` and nothing where the profile has none, instead of
Turbo's 1.15 on every checkpoint (U87, U106).

**A fixed shift has a number** (2026-10-05). Z-Anime's, NetaYume's and
NewBie's schedulers ship a static shift 6 (mu 1.79), Z-Image's and Z-Image
Turbo's a shift of 3 (1.10), lite repos included. There a blank Mu is that
number, and nothing about it is automatic, yet the box said "auto" and a
sheet of "Mu 1, none, 2.5" laid out "Mu none | Mu 1 | Mu 2.5", the 1.79
column left of 1 with nothing to say so. The profile's `mu_static`, in every
status's defaults and set only where Mu applies and the profile sends no mu
of its own (`test_a_fixed_shift_is_the_profiles_mu_static`), says it. The
panel's and the Refine dialog's Mu placeholder is the number (`muPlaceholder`:
"1.79" where FLUX's says "auto", nothing on a dead box). The grid preview and
the sheet read Mu's none as "none (1.79)" - on the sheet its heads, the Fixed
line, the pins and the export - and the sheet sorts it at the 1.79 its head
prints, so an explicit 1.79 typed from the placeholder sits just after it (at
ln 6's 1.7918 it sat before). That holds only while every image on the sheet
is the running checkpoint's, since a record carries no profile. The
placeholder is the number alone: "auto (1.79)" is 66 px in a box with 58 at
1400x900. The stored mu stays null, the grid's word stays `none`, the
lightbox's "no Mu" is unchanged, and the Mu tips quote the number from the
profile.

Where Mu cannot apply (SDXL, Anima Turbo), the grid's Mu axis had said
"none,1.15" and kept that syntax in its tip under a greyed box. It now reads
"unavailable", and its tip and its wrapper's title quote the panel Mu tip's
reason (`syncMuWords`); the syntax help is kept as nodes
(`gridMuTipDefault`) and comes back where Mu applies.

## True CFG is its own field

On FLUX, `guidance_scale` is a distilled conditioning value, **not** CFG - a
negative prompt is silently ignored unless `true_cfg_scale > 1.0`. **Real CFG
is its own request field there**, `GenerateRequest.true_cfg` (None = off):
the two used to be set from one number, so the only reachable states were
(3.5, off) and (g, g), and a negative prompt typed at 3.5 stacked guidance
twice. `describe()` publishes `true_cfg_supported` (`"true_cfg_scale" in
_call_params`; the profile guesses it from the id until then), `/api/status`
merges it like the threshold, and where it is True `applied()` compares the
threshold against `true_cfg`, not `guidance_scale`. **The recorded `true_cfg`
means one thing on every checkpoint: the scale the negative prompt was
applied at, or None** - on Krea 2/SD3/klein that is `guidance_scale` when
the negative prompt ran, so a column, a filter (`tcfg:`), a `_GROUPS` entry
and the resume key all read the same way, and a row from before the column
(NULL) is read through `schemas_generate.effective_true_cfg` / `search.TRUE_CFG_SQL`
as "guidance_scale if there was a negative prompt", which is what ran then.
`CellKey` gained the element before the seed; `grid.run_id` names a cell
by the old six-element key whenever `true_cfg` is None, so every existing
run id survives (pinned in `tests/test_grid.py`). The UI's True CFG box is
greyed - not hidden - where the knob does not exist, the negative prompt
follows whichever box decides, and Reuse fills it only on a checkpoint that
has it. Verified on the card 2026-09-08 (FLUX.1 Krea dev, fp8): a negative
prompt with no true CFG produces the byte-identical image to no negative
prompt; with true CFG 1.5 a different, reproducible one at ~1.65x the time.
On the default checkpoint the negative box is therefore greyed on every fresh
page, with the box that wakes it a field's height below. While it is greyed and
the checkpoint takes a negative at all, a "Set True CFG" (or "Set CFG") button
after the note that names that box focuses and selects it and writes nothing
(`focusNegativeKnob`): a value is the checkpoint's, about 4 on Qwen-Image,
none on schnell. The field is a `<div>` with a `<label for="negative">`, as
the Prompt field is: inside a `<label>` the button became the label's control
and the textarea lost its name (U100). The button first sat in the box's
corner, to cost no height, with `pointer-events: none` on the greyed box so a
press on it reached the field and did the same. Both went: the button lay over
the box's own text, a carried-over negative being what someone looking at a
greyed box is reading, and the greyed box no longer took the wheel, so such a
negative could be read only to its second line. In the caption it costs
nothing where the note already wraps (the 330 px panel) and a line on a wide
one.

**Two things the review offered for the greyed box were not built, by
decision.** A press on the box itself reaching True CFG or CFG (U100's
optional part) is the `pointer-events: none` above, and it went for the
reason given: a long carried-over negative could no longer be scrolled or
read. The review plan's step 13 collapsed the dead box to one line; with its
overflow hidden that would hide the same negative, which the box keeps
readable on purpose, and the caption's button already covers what the press
was for. Only a checkpoint that takes no negative at all (klein) is its
caption alone (`.unsupported`), since nothing can ever wake its box.

**The caption names a cost only where it is the field's own** (2026-10-05).
It said "(needs CFG > 1 — ~2× slower)" under a live box too, which was wrong
wherever CFG is the knob: above the threshold the uncond pass runs whatever
the box holds - against an empty prompt where it is blank (Lumina 2 and SDXL
above 1, Z-Image and Krea 2 from 0) - so a live box costs nothing more. Under
True CFG the text alone adds the uncond pass, and greyed, waking it means
raising CFG; those two keep the cost. Live where CFG is the knob it reads
"(runs at CFG > N, no extra cost)", and "(for grid cells above CFG N)" where
only an open grid's axis reaches past the threshold. Each is kept as short as
the greyed caption: "(runs while CFG > N; no extra cost at this CFG)" wrapped
beside the label at 1400x900 and 2048x1040 at 125% and moved the box, and
everything under it, down a line - and at the threshold the caption changed
height as CFG crossed it. `#negative` is described by that note
(`aria-describedby="neg-note"`), so a greyed box says why to a screen reader
too. The Refine dialog's negative caption uses the same words for the same
state (`syncRefine`): "(runs at CFG > N, no extra cost)" where CFG is the knob
and the box is live, and "(needs CFG > N — ~2× slower)", or True CFG, where it
is greyed or True CFG is the knob; it had read "(needs CFG > N)" under a live
box too (2026-10-05).

**FLUX.1's CFG box is named "CFG (distilled)".** The box is the distilled
guidance value, and the label said CFG, the one word every other checkpoint
uses for real guidance (U87's optional qualifier, built 2026-10-05). The
profile's `cfg_label` is the one hint that is a name: `applyHints` writes it
on `#cfg-label`, `#grid-cfg-label` and the Refine dialog's
`#refine-cfg-label` (which had said plain CFG beside the panel's distilled
one), plain "CFG" where a status has none,
and `hints_for` drops it where a loaded pipeline's negative-prompt answer
replaces the CFG tip; `krea-dev` carries it as the flux profile carries
everything. No other profile has one: schnell's box is ignored and klein's
guidance is baked in, which their tips say. The longer name had to fit the
sampling row, whose names now keep to one line by a `white-space: nowrap` on
the `<label>` alone, with `min-width: min-content` on each box and the
counters' padding (62/86/62/62 px at 1400, Mu's "auto" whole). The first
version put the nowrap on the whole label line, which holds each ? badge and
its tip: every one of the row's four tips then ran out of its 238 px box, on
every checkpoint - True CFG's to 1,563 px at 1400, with a sideways scrollbar.
`.tip` now sets `white-space: normal` itself, so no caption or label line
can hand it a nowrap.

## FLUX.2 klein: no negative prompt, and the threshold rule

**FLUX.2 [klein]** (`Flux2KleinPipeline`, in diffusers 0.40) loads through the
same loader, but its ids contain `flux`, so `model_defaults.py` has `klein-base`
and `klein` entries *ahead of* `flux`: distilled klein is 4 steps and ignores
guidance (the pipeline warns above 1), base is 50 steps at real CFG 4.0. Its
`__call__` has **no `negative_prompt`** - only embeds, and base's uncond pass
runs against `""` - so the profile carries `negative_prompt_supported: False`,
`describe()` reports `"negative_prompt" in _call_params` once loaded,
`/api/status` merges it the way it merges `cfg_threshold`, and the UI greys the
field for the checkpoint rather than for the value. `cfg_threshold()`:
`true_cfg_scale` present -> 1.0 (FLUX.1); else `negative_prompt` present
**and the pipeline class is in `_GUIDES_FROM_ZERO`** (`Krea2Pipeline`, the
`ZImage*` family - the only nine of 131 diffusers pipelines whose
`do_classifier_free_guidance` is `> 0`) -> 0.0; else 1.0 (klein's classic
`guidance_scale > 1` gate, and SD3/SDXL/Chroma/Lumina 2's). "Has
`negative_prompt`" alone used to read 0.0, which made loading SD3 make
`/api/status` *less* accurate than the profile's guess. `hints_for` takes
the live `negative_prompt_supported` as well as the threshold, so the CFG
tooltip cannot contradict the greyed-out field. `load()` refuses a pipeline
whose `__call__` lacks `generator` or `callback_on_step_end`
(`_REQUIRED_PARAMS`): dropped silently, the first meant unseeded noise under
a recorded seed and the second a Cancel that never lands. Verified 2026-09-05 via Nunchaku NVFP4 on the
5080: 1 MP in 4.0 s at 2.9 GiB peak, 2 MP in 4.6 s at 3.4 GiB. Its pipeline also lacks
`enable_vae_tiling`/`enable_vae_slicing`; the loader falls through to the same
switches on `pipe.vae`. Its mu is empirical (sequence length and step count),
so the profile leaves mu null and `schedule.forced_mu` stays out of the way. Text
encoder is a plain `Qwen3ForCausalLM` (transformers), so the fp8 path
quantizes it like any other; the 4B's is 8 GB in bf16, bigger than its
7.75 GB transformer.

## Reference images, by gallery id

**Reference images are klein's and are addressed by id.** `Flux2KleinPipeline.__call__`
takes `image` first, resizes each reference to <=1 MP, VAE-encodes it and
concatenates the tokens onto the latent sequence every step; nothing else in
the catalogue has the parameter, so `describe()` reports
`reference_supported` (`"image" in _call_params`), the profile guesses it
from the id until then, `/api/status` merges it like the threshold and the
UI greys the field for the checkpoint. `GenerateRequest.reference` is a list
of **gallery ids**, never paths - a path joined onto `outputs_dir` is an
arbitrary-file read reachable from `/api/generate` - and
`storage.reference_paths` resolves them, naming an id that is missing and a
file that is gone. `applied()` drops them where the pipeline has none, and
`submission.refuse_unusable_inputs` refuses at *submit* rather than at cell 1 of
64: unlike a negative prompt, a dropped reference renders a plain
text-to-image that looks like a success. The module imports `Image` as the
class for typing, so `_reference_images` opens through a local
`from PIL import Image as PILImage`.
**The 2-3x cost the review predicted did not appear.** Measured on the card
2026-09-08 (klein 4B, fp8, model offload, 1024x1024, 4 steps): 10.2 s with
no reference, 9.2 s and 8.2 s with one, 11.3 s with two. Each reference
roughly doubles the sequence and attention is quadratic in it, but under
model offload the per-image weight transfer dominates both, so the estimate
in the panel needs no reference-aware term. The conditioning itself is
unmistakable and reproducible: the same fox, cabin and treeline as the
referenced image, in the pose the prompt asked for, and the same hash twice.

**The chips are numbered, and their order is editable.** klein's guide, its
prompts and the check name references "image 1", "image 2", so the order is
what the prompt means. Each chip carries its number at bottom left, its image
is named "Reference image N (gallery #id)", and it moves the way the tag chips
do - dragged, or Alt+Left/Right on its × - with Delete on the × dropping it and
focus going to the next chip's ×, else the previous one's, else the emptied
row (`tabindex="-1"`, a group named "Reference images"), rather than to
`<body>`. Not the field's badge: a focused badge opens its tip, which then lay
over LoRA and Seed for someone who had only clicked the last × away. Every move builds a new array and goes through
`setReferences`, the one writer, so the panel is remembered, the preview asked
again and the prompt check marked stale ("references changed since"). A chip
whose thumbnail does not load - an image deleted since, or an id this browser
remembered - is marked deleted rather than showing a broken image. The note
under the caption counts "1 of 4" and asks for numbers in the prompt only from
two references on (one needs none, as BFL's own example shows); it no longer
says each lengthens the sequence, a cost measured above as absent. Where the
field is greyed its empty chip row is hidden, the caption, badge and note
staying; the badge says what references do where they are taken, rather than
the unsupported checkpoint's sentence (U36, U62, U77, U95).

**A deleted reference leaves the chips.** An id the gallery no longer holds
used to stay a chip - with the browser's cached thumbnail in the session that
deleted it, a broken image after a reload - until Generate was refused with
"no gallery image with id 4", an id the chips never showed. `forgetImages`,
the one place the page drops deleted ids, drops them from `referenceIds`
too (through `setReferences`, the one writer) and its caller says so in the
gallery's line; the per-image confirm said "It is reference image N in the
panel" before the delete (since 2026-10-11 there is no per-image confirm:
the image goes to the recycle bin and the note after it says so, and only
Delete permanently in the bin asks). Reuse restores a record's references
synchronously and then asks `GET /api/images/{id}` for each, dropping only
those answered "unknown image" (see "Per-control defaults and Reuse"). The
server still refuses a missing reference at submit (`storage.reference_paths`),
and a binned one too, "is in the recycle bin; restore it to use it".
And a starred image keeps its references through Delete unstarred: they are
part of what made it (see "Delete unstarred").

**A reference deleted elsewhere is asked about, not guessed at**
(2026-10-05). `forgetImages` covers a delete this page made. One made in
another tab, or before a reload, reached nothing here: the chip kept the
browser's cached thumbnail, and every Generate was refused with "no gallery
image with id 5" until the chip was removed by hand. Now a chip whose picture
fails is marked `.gone` and asks `GET /api/images/{id}`, and so does a submit
refused with that sentence for one of its payload's references
(`dropGoneReferences`); either drops it only on "unknown image", as Reuse
does, through `forgetImages`. A passing failure - no server - keeps a live
reference, marked as a failed picture marks it. The refusal then says what
happened in the page's words - "Nothing was queued: reference #5 has been
deleted, so it was taken out of the references. Generate again to run without
it." ("Run the grid again", "Run the comparison again" after those) - so one
more press is enough.

**Bringing in a picture of one's own is not built.** The review's U61 is
titled for it: klein's main use is editing an outside photo, and references
come only from the gallery, through a card's ❐. Its part A is in - the edit
Example says it needs a reference image, Check notes an edit prompt with none
attached, and a file dropped on the page no longer navigates away - and part
B, an upload route that makes a gallery row and hands back its id with Add
image, drop and paste on the field, was deferred by the user (2026-10-05). It
is an L-sized change, not a field: a row's prompt, steps, CFG, seed and model
are NOT NULL and the reindex drops a PNG whose chunk lacks them, so an import
needs a source marker in the row and the chunk, through `_migrate`,
`_COLUMNS`, `index_file` and the merge; the gallery scope, Re-render, Reuse,
the facts and the estimates would each have to know such a row is no render.
Generation itself would not change: `storage.reference_paths` resolves any
row by id and the pipeline opens each reference as a picture. The refine pass
would get outside sources from the same route.

## LoRA adapters are the environment's and are hot-swapped

**A LoRA is the environment's capability, not the checkpoint's.** All three
families mix a loader in, so what decides it is peft and the quantizer:
`lora_unavailable_reason()` refuses `LOCAL_DIFFUSION_QUANT=nunchaku` (already 4-bit
SVDQuant; the modules PEFT would wrap are not `nn.Linear`) and a missing
peft up front, the way `loading.nunchaku_preflight` refuses, and `describe()`
answers `lora_supported` with no pipeline loaded at all. Adapters are
**hot-swapped on the live pipeline** in `generate()`, never baked into
`load()`: diffusers strips and re-attaches the offload hooks around the
injection, so the "adapters must precede `enable_model_cpu_offload`" rule
does not hold, and a second load path driven from the job worker is the one
thing `main.ensure_loaded` exists to prevent. `_apply_lora` tracks
`(name, weight)` itself and a weight change is a `set_adapters`, which is
what makes `0-1x0.25` one load and four re-weightings; a failed injection
unloads and clears that state, because the alternative is every later cell
running under an adapter nothing recorded. The request carries a *name*
(`_bare_name` refuses separators) and `backend/loras.py` is the only place
one becomes a path, resolved under `loras_dir` and checked to still be
inside it. `.safetensors` only: a `.bin` LoRA is a pickle.
**torchao fp8 + PEFT was the unverified part, and it works**: measured on the
card 2026-09-08 (FLUX.1 Krea dev, fp8, model offload, a 172 MB FLUX.1-dev
LoRA) - the adapter changes the image and reproduces, `unload_lora_weights`
restores the base hash exactly, and **weight 0 is byte-identical to no
adapter**, which is what proves the weight reaches the maths rather than
merely being recorded. A three-weight sweep was 17.5/16.3/18.6 s against
15.4 s unadapted: the swap between cells costs nothing measurable. The row
keeps `lora` at weight 0 rather than nulling it - the adapter *was* attached,
the way CFG 0 is recorded as 0 - and `distinct_count` is what says the 0 cell
matched the baseline.

**An adapter for another checkpoint is not offered, and is refused by name.**
The directory is the directory, but a LoRA is a set of deltas addressed to
modules by name, and PEFT injects it by looking those names up on the
denoiser: the one FLUX.1 adapter in `loras/` (Frosting Lane, keys under
`transformer.single_transformer_blocks.*`) was listed on Z-Anime, chosen, and
failed at the first cell with a PEFT traceback naming neither model
(2026-09-20). `loras.mismatch` is that lookup run ahead of time: `targets()`
reads what the file addresses through the pipeline class's own
`lora_state_dict` - so a kohya or ComfyUI layout is seen the way the injection
sees it - and finishes the names with `convert_unet_state_dict_to_peft`
exactly as `load_lora_adapter` does (SDXL's kohya conversion stops at
`attn1.processor.to_q_lora.down.weight`), then compares module names and
widths against a `loras.Denoiser`. A family table was considered and
rejected on the cases it gets wrong: FLUX.1 and FLUX.2 [klein] share every
block name and differ in the count, FLUX.1 and Krea 2 share the names and
differ in the width (3072 against 6144), and only the denoiser's own module
list knows either. The `Denoiser` is the live pipeline's once there is one
(`transformer_name`/`unet_name` names the attribute; a PEFT-wrapped target
is read through its `base_layer`) and before that `loading.denoiser` builds
the class from `model_index.json` and the denoiser's `config.json` under
`accelerate.init_empty_weights` - every name and width, no weights, tens of
milliseconds - which is what lets the listing answer at page open. Illustrious
is read from its cached conversion (`single_file.converted_tree`) or not at
all: a listing must never be what starts the 7 GB conversion. Where no
config can be read the fit goes *unchecked*, not refused - the load that
follows either answers it or fails with the real reason - and the page
fetches the list once more when the status turns ready. The listing carries
each adapter's `detail`; the page hides an entry on it and names it in the
tooltip with the reason, so a file dropped into the directory that never
appears is explained rather than missing - after the tooltip's own sentence
(the directory it reads, what Weight does), not in its place, and without the
file's name twice, since a `detail` starts with it (U118); `/api/generate`, `/api/grid` and
`/api/rerender` answer 422 with the same reason and the CLI exits 2 with it.
Measured on the frosting-lane file against the cached checkpoints: fits
FLUX.1 Krea; 494 modules Z-Image has not got; 454 klein has not got; the
right names at the wrong width on Krea 2; no `transformer` on SDXL.

**The skeleton is never built beside a load** (2026-10-05).
`accelerate.init_empty_weights()` builds on the meta device by replacing
`torch.nn.Module.register_parameter` - for the whole process, every thread,
while it lasts - and puts back what it found on entry. The page asks
`/api/loras` on its first status, so on a cold process the listing built the
denoiser's skeleton while a Load pressed within seconds of opening the page
ran on another thread: the load's parameters went through the patch as
torchao quantized them, and WAI failed three times out of three with
`Float8Tensor.__new__() got multiple values for argument 'qdata'`
(`docs/verified.md`, Behaviour). Two builds at once - two listings - were a
second hazard: the one leaving last restores the other's patch, and every
parameter registered after that, the next load's, would go to meta for the
life of the process. `torch.device('meta')` as a context would patch nothing
shared - its TorchFunctionMode stack is per thread in torch 2.13, and the
one module global it sets, `torch.utils._device.CURRENT_DEVICE`, only
dynamo's guards read - but it puts every tensor a constructor makes on meta,
buffers and intermediates included, where `init_empty_weights` leaves buffers
on the CPU, and a constructor that reads a value it computed (`.item()`,
`.tolist()`) fails there; the catalogue's denoisers were not each checked
against it. So the two are kept apart instead: `loading.weights_loading()`
is held across the whole of `load()`, the single-file conversions that patch
the same way inside it included, and `loading.empty_weights()` wraps the
build. A load waits out a build in progress - tens of milliseconds; a build
asked for during a load raises `LoadInProgress` rather than wait the minutes
out, and the fit goes unchecked until the load, as for a config out of reach,
the listing after the status turns ready answering from the live denoiser;
builds go one at a time. Every other weight load in the process - Expand's
rewriter, the translator - runs with the pipeline loaded, when the fit reads
the live denoiser and builds nothing. `tests/test_conditioning.py` runs each
case on threads; on the card, three fresh WAI processes with Load pressed
0.87 s after the page opened each loaded, the listing logging the fit
unchecked during the load.

**Two integrations refuse adapters by the id, whatever the environment**
(2026-10-05): Anima Turbo and NewBie, whose pipelines are this repo's own
classes with no adapter path. `loading.lora_refusal(model_id)` is that one
refusal, from the loaders' `MODEL_ID` constants and case-insensitive, and
`DiffusersBackend.lora_unavailable_reason` asks it after the Nunchaku check
and ahead of the peft and loader checks. The fake server sets its own
refusal from the same function, so the browser checks see the greyed select
and the 422 at submit the real server gives; until then the fake offered
adapters on both. A grid's LoRA weight axis dead for that reason, or any
other, reads "unavailable" and quotes `/api/loras`' detail once that list has
answered (`loraRefusal`, `syncConditioning`), as the dead Mu axis quotes the
Mu tip, and its syntax help (`gridLoraTipDefault`) comes back where an
adapter can apply.

## The refine pass

**A refine pass keeps the picture, which a direct render at a larger size
cannot.** The size guides had been saying so in the authors' words and the
app had no way to do it: FLUX.1's "render at 1-2 MP and upscale", and WAI's
1024x1344 followed by a 1.5x hires pass at denoising strength 0.35-0.5. What
the app had was Re-render, whose own notes say what it delivers - the latents
are drawn at the requested shape, so the same seed at another size is a new
sample - and the measurements say where a direct large render goes wrong
(Illustrious at 2048 rendering the shelves and omitting the doctor its 1536
kept). A pass is `GenerateRequest.refine`, a gallery id, and `strength`: the
source is resized to the request's width and height, encoded by the VAE,
noised to the point on the schedule the strength names and denoised from
there under the request's own prompt, negative prompt, steps, CFG and seed.

**It runs the sibling pipeline over the components already loaded.**
diffusers ships an image-to-image class beside most text-to-image ones, by
name (`FluxImg2ImgPipeline` beside `FluxPipeline`) and over the same
components, so no extra weights are downloaded or allocated. With model
offload, the source's VAE encode keeps the VAE resident beside the denoiser
through the pass until the final cleanup; the measured WAI and FLUX.1 peaks
are about 0.16 GiB above a direct render at the same size. Sharing weights
does not imply the same peak residency.
`refine.sibling(type(pipe))` is the lookup, and the whole capability: the
name with `Img2Img` in it, only for the class diffusers itself exports under
the pipeline's name (a local class that merely shares one has components
nobody checked), and only where the sibling's `__call__` takes
`refine.REQUIRED` - `image`, `strength`, and the `generator` and
`callback_on_step_end` `load()` insists on - and every argument in
`refine.CARRIED` that the text-to-image call takes. `CARRIED` is the list
`generate()` passes besides the size, held to that function's source by
`tests/test_refine.py`: a sibling that lacked `negative_prompt` or `sigmas`
would have it dropped by the backend's drop loop, at DEBUG, and render a
refine the row then describes wrongly. In diffusers 0.40 that gives FLUX.1
(Krea dev, dev, schnell), SDXL (Animagine, Kohaku, Illustrious, WAI),
Z-Image (Turbo, Z-Anime) and Qwen-Image a pass, and Krea 2, FLUX.2 [klein],
ERNIE-Image and Lumina 2 (NetaYume) none; Anima and NewBie run this repo's
own pipeline classes, which have no sibling. A table of families was not
written: the loader is model-agnostic, and so is this. Each profile carries
`refine_supported` as its guess before a load, `describe()` answers from the
class after it, `/api/status` merges the two like `reference_supported`, and
`tests/test_refine.py` holds every guess to the installed diffusers.
If the live class supports the pass but the profile has no strength, status
reports `refine.DEFAULT_STRENGTH`, matching request defaults and the tooltip.

**`refine.build`, not `from_pipe`.** `DiffusionPipeline.from_pipe` is the
library's own way to make one pipeline from another "without reallocating
additional memory", and it ends in `new_pipeline.to(dtype=dtype)` - float32
unless told otherwise - over modules the first pipeline is still using,
torchao-quantized ones among them. The sibling is constructed directly
instead: each constructor parameter from `pipe.components`, or from
`pipe.config` for the plain values (SDXL's `force_zeros_for_empty_prompt`),
and an error naming a required one the pipeline has not got. A constructor
only registers what it is handed. The offload hooks are the modules' own, so
they come along; the scheduler is the same object, which is what lets
`schedule.forced_mu` - patched onto the loaded pipeline's scheduler - reach
a pass, and a static scheduler is handed the pre-shifted `sigmas` exactly as
a text-to-image call is. The adapter goes into the denoiser through the
loaded pipeline (`_apply_lora`) and the sibling shares the denoiser. Built
on the first refine, kept, and dropped on unload - it is a second set of
references to the same modules, and kept past that an unload would free
nothing.

**One call per image, because the VAE's encoding is sampled.** Every other
batch here is one pipeline call with a list of generators, one per image, so
that image *i* is what a solo run of its recorded seed makes. An
image-to-image call encodes the source before it draws the noise, and that
encoding is a *sample* from the VAE's posterior: `FluxImg2ImgPipeline` loops
over the images it was given - one - and draws that sample from the first
generator alone, so image 2 of a batch got image 1's encoding and its own
noise, and its recorded seed reproduced something else. SDXL's sibling
samples per generator and would have been fine. `DiffusersBackend._refine`
makes one call per image with that image's generator (a scalar: the sample,
then the noise, from one stream, on every pipeline), which costs a text
encode and a VAE encode per image instead of per batch and makes the
invariant true by construction rather than per pipeline. Measured on the
card (`verified.md`): image 2 of a batch of two is byte-identical to a solo
run of its seed, on SDXL, FLUX.1 and Z-Image alike. Progress counts through the images
(`batch["index"] * steps + step`), so a batch's bar does not start over.

**The sibling has no hook list, so the loaded pipeline frees the card.**
`enable_model_cpu_offload` was called on the loaded pipeline;
`maybe_free_model_hooks`, which every call ends with, returns at once on a
pipeline that never had it called (`_all_hooks` is absent). After a refine
the last component used - the VAE - would stay on the card until the next
call's first forward. `_refine` calls the *loaded* pipeline's
`maybe_free_model_hooks()` in a `finally`, which resets the chain the way a
text-to-image call's own ending does, after a cancel or a failure too.
Measured: the device reads the same 1.4 GiB after a pass as after a
text-to-image call, and the source's own request renders the same hash after
the passes, a batch and a cancelled pass as before any of them.

**`steps` is the schedule's length; the strength decides how many run.**
The pipeline builds the full `num_inference_steps` schedule and starts part
of the way down it, and how the count rounds is the pipeline's own: SDXL
runs `int(steps * strength)`, the flow pipelines `steps - int(steps -
steps * strength)`, so 28 steps at 0.4 is 11 on SDXL and 12 on FLUX.1. The
row records `steps` as asked - it is what reproduces the pass. Progress and
estimates account for the sliced schedule: the step callback reads
`pipe.num_timesteps` for progress events, while the initial job total uses
`refine.steps_run` with the checkpoint's integer rounding. `refine.start_index`
also makes the duplicate advisory count strengths that begin at the same
step; reproduction keys retain the requested strength. `gallery_reports.step_cost_ms`
uses direct rows only. `gallery_reports.refine_cost_ms` keeps full per-image refine
durations, including encoding and decoding, grouped by actual denoised step
count at the same size and checkpoint. A refine estimate needs matching
history for every count; it does not extrapolate from direct renders or a
different count. SDXL's call raises below one step, so
`GenerateRequest.check_refine_steps()` refuses `int(steps * strength) < 1`
after the capability check, and for a grid checks each expanded cell by name
instead of unused base values. The preview reads its sigma at
`scheduler.begin_index + step + 1`:
a pass starts at the index its pipeline set on the scheduler, and read from
the top the gain was that of a far noisier step than the one just taken.

**The source is covered and cropped, never stretched.** `refine.fitted`
scales the source until it covers the requested size and crops the centre,
Lanczos. The scale buttons multiply the source's size (`refine.scaled`, each
side rounded to 16, half up - written out because `round()` rounds half to
even and the page's `Math.round` does not, and the scale buttons and the
CLI's `--scale` must land on the same size), where the crop is a few pixels
at most; a size of another shape is a reframing, and the size line says
"cropped to this shape" and how much it keeps ("keeping the middle 56% ×
100%"), from `refine.fitted`'s own arithmetic - cover, each side rounded
half to even as Python's `round()` does, the centre cut out - which the
page repeats (`refineKept`) so the number and the frame drawn on the large
source are the pass's. Keep shape, on by default, moves the other side to
the source's shape when W or H is typed: typing one side is how "2000 wide"
is asked for, and alone it reframed the picture. It acts on typed input
only, so a script that sets both sides means the size it sets; off, a
reframing is one keystroke. The resize is in the backend rather than left to
the pipeline because SDXL's image-to-image call takes its size from the
image and has no `width` or `height` at all.

**It opens at a size the checkpoint holds.** The dialog opens at the scale
last run, 1.5x the first time, or the largest smaller factor that the pass
takes and that is not past the tight line (below), else at the source's
own size; a scale it passed over says so on its button, one click away -
"(last run at 2×)" for a remembered one, "(the usual opening size)" for the
1.5x a first open takes (`refineSkipped`) - and while W and H are the size it
opened at, the size line says why, naming the next scale up and the line it is
past (`refineCapNote`: "1.25× and larger go past the 1536×1536 (2.36 MP)
measured on Z-Anime at fp8", or the generic "may run tight on 16GB VRAM").
Pixels grow with the factor, so what holds for that scale holds for every
larger one. Only a remembered scale used to be named: every Z-Anime source
from about 1.5 MP, the author's own 1024x1536 included, and every such FLUX.1
source opened at 1x and said nothing, though a refine pass is expected to
upscale (2026-10-05). The dialog used
to open at 1.5x or the remembered
scale whatever that came to: a 1.5 MP FLUX.1 source opened at 3.5 MP, and
after one 2x pass every later source opened at 2x - past what the card
holds (it fills at 2.4 MP), and on FLUX.1 at a size where the profile's
own 0.1 starts near sigma 0.65-0.92 and repaints the subject. Open and
Enter, the fastest path, ran exactly that. The line is the profile's
measured one where `verified.md` has it, so WAI, which ran 8.6 MP, still
opens at 1.5x; an unmeasured SDXL checkpoint opens capped at the generic
line, one click from 1.5x or 2x. The guard is unchanged and is still the
only refusal. A run at the size the dialog opened at in place of the scale
it was asked for leaves the remembered scale alone (`refineCapped`): that
size was the dialog's choice, and remembered as the user's, one Enter on a
1.5 MP FLUX.1 source opened every later 1 MP source at 1x - the 2x trap
above, turned round. The scale buttons carry `aria-pressed` beside `.active`, as the
panel's size strips do; past the guard they stay `disabled` with the reason,
the documented exception, and they are no roving tab stop.

**Everything else is the reference image's design.** The source is an id,
never a path, resolved through `storage.reference_paths` under its own noun;
refused at submit (`_refuse_unusable_inputs`, and the CLI before its load)
where the checkpoint has no pass or the image is gone, because a night of
refinements that rendered as plain text-to-image at the larger size would be
a night of new pictures. Jobs and the CLI check again after loading, before
any cell renders, and fail if the live capability would drop the requested
source. `applied()` still normalises an unsupported pass away,
with the strength back at its default - a strength with no source is not a
setting, as a LoRA weight with no adapter is not - and with any references,
since the sibling's `image` is the source. `refine` and `strength` are
columns (`_migrate`), keys of the PNG's chunk, `Denoising strength` and
`Refine` in the A1111 block, `refine:` and `strength:` in the search
grammar, `strength` and `refine` in `stats --by`, and the fourth optional
element of `CellKey` (`grid.refine_key`, `id@strength`). Re-render,
`--from-image` and Reuse carry both; a database merge remaps the source id
with the reference ids and drops one whose image did not come across.
PNG-only recovery uses `reference_sources` and `refine_source` descriptors
(filename, decoded-pixel hash and dimensions), recorded by `save_image` and
database merges. It imports each proven source first, then records its new id.
Legacy ids without provenance, mismatches and missing sources are cleared
and counted as unresolved; an unproven id is never used as a destination id.
Changed links are written back into in-place PNG metadata too, without
recompressing pixels, so `--from-image` sees the same links as the rebuilt row.
`GridRequest.strength_axis` (`--grid-strength`) is an axis only beside a
source; on the page it is the Refine dialog's Strength box holding more
than one value. Reuse on a refined record asks `GET /api/images/{id}` for
its source, which the record holds by id alone, and opens the dialog on it.
A source deleted while the dialog is open is the server's to refuse: in the
dialog, where Refine waits for the preview of changed boxes (the preview runs
the submit's checks), and at submit otherwise. The page says it as it names
any image that has gone - "#28 has been deleted, so it cannot be refined."
(`refineSourceGone`: the preview's refusal, Refine's title and the submit's),
with the source line ending "#28 (deleted)" - since the server's "no gallery
image with id 28", under a dialog still titled Refine #28 over its picture,
read as a setting gone wrong.

**A source deleted while its pass runs** (2026-10-05). A cell reads its
source as it begins, so a delete during the cell does not stop it: the pass
lands, made from an image that has gone. `save_image` returned its record
without looking the source up - `refine_missing` is no column, and only
`list_images` and `get_image` asked (`_missing_sources`) - and the page's
`forgetImages` marked only the records it already held, so on the card the
late pass read "refine of #N @ 0.3" with no "(deleted)", and its Reuse
offered the Refine dialog on the source until a reload, beside a pass that
had landed before the delete and said the source had gone. `save_image` now
asks `_missing_sources` in the transaction it inserts in, under the write
lock a delete takes; `jobs.forget_images` marks a job's replayed records
whose source a delete took, as the gallery's are; and the page keeps every
id it has seen deleted (`deletedIds`, from `forgetImages`) and marks an
arriving pass made from one (`markGoneSource`, in the stream's `show`), since
a record saved just before the delete committed reaches the page after it.
A source moved to the recycle bin is missing the same way, with
`refine_binned` beside it (2026-10-11): the page says "(in the recycle
bin)" and offers a Restore, which clears both flags in the gallery and in a
job's replay (`jobs.restore_images`).

The sweep that delete stops cannot resume. On the card, the source deleted
during cell 2 failed cells 3-5 within milliseconds (`ValueError: no gallery
image with id 1`) and the failure rule ended the job; its finish line offered
"Resume the remaining 8 cells", which the server refused at submit as "no
gallery image with id 1", said under Generate as that and nothing more, and
the button stayed live, refused the same at every press. Retry and Resume now
say "#N, this sweep's source, was deleted: the run cannot resume." (Retry:
"its skipped cells cannot run again") and hold, `aria-disabled` with the
sentence as the title and a press saying it again: before sending where the
page saw the delete (`deletedIds`), else on the server's refusal of the
source by id (`startJob`'s `onRefused`).

**A guess says it is one** (2026-10-05). `refine_supported` is the profile's
guess before a load and the loaded class's answer after it, and the status
now says which (`refine_settled`). Anima Turbo and NewBie are settled before a
load too: their pipelines are this repo's own classes, for which
`refine.sibling` answers None loaded or not (`model_defaults._OWN_PIPELINES`).
The dialog that a refined record's Reuse opens on a checkpoint without a pass
says "a load answers for certain" only while the answer is a guess; a final
one reads as the card's refusal (`REFINE_NONE`). Anima with its model loaded
had still told the user a load would decide. Likewise `refine_strength_profiled`
says whether `refine_strength` is the profile's own or the general default an
unprofiled pipeline with a pass was given, so the page never calls the general
0.4 a checkpoint's.

**A strength is an index into the checkpoint's schedule, so where it starts
is the profile's.** The first version had one default, 0.4, inside the
0.35-0.5 WAI's author documents - and on the card that number was right for
WAI and wrong for FLUX.1 Krea dev, the default checkpoint: at 1.5x of a 1 MP
image the pass kept the composition and repainted the man's face, at 0.4 and
still at 0.2. The pipelines turn a strength into a *step index* and the
schedule's shift then decides the noise level at that index. SDXL's epsilon
schedule at 0.4 starts where the signal-to-noise ratio is that of a flow
sigma of about 0.50 (leading spacing, 28 steps). FLUX.1's shift is dynamic and grows with the size - mu
is 1.15 at 1 MP and 2.0 at the 2.4 MP a 1.5x pass renders - so its 28-step
schedule there spends nearly all its steps at high noise: strength 0.4
starts at sigma 0.85, 0.2 at 0.67, 0.15 at 0.62, 0.1 at 0.47 and 0.05 at
0.37. The 0.85 and 0.67 values are computed from the schedule; 0.62, 0.47 and
0.37 were read off `scheduler.sigmas` at `begin_index` in the logged sweep
and match the formula. The face held at 0.05 and 0.1 and began to change at 0.15.
So `refine_strength` is a profile key, where the Refine dialog's Strength
box starts on the checkpoint, filled in by the schema for a
request that names none (`schemas_generate.default_strength`), and what `applied()`
puts a strength back to when there is no source. Each value starts the pass
at about half noise for a 1.5x pass of a 1 MP image:

| Profile | Start | Basis |
| --- | --- | --- |
| Animagine, Kohaku, Illustrious, WAI (SDXL) | 0.4 | WAI's author's 0.35-0.5; measured on WAI |
| FLUX.1 Krea dev, FLUX.1 dev | 0.1 | measured on Krea dev: sigma 0.47, 3 of 28 steps |
| Z-Anime Base | 0.15 | static shift 6: sigma 0.51, 6 of 40 steps; measured |
| Z-Image, Z-Image Turbo | 0.25 | static shift 3: sigma 0.5; arithmetic only |
| FLUX.1 schnell | 0.5 | unshifted 4-step schedule: 2 steps from sigma 0.5; arithmetic only |
| Qwen-Image | 0.3 | its published dynamic shift (max 0.9 over 8192 tokens) puts 0.3 near 0.53; arithmetic only, and from the config as published - this machine has no copy to check it against |

The rows marked arithmetic have not been run here: they are the installed
scheduler's formula applied to each checkpoint's shift, not a measurement,
and `verified.md` says which are which. Every value leaves at least one step
of its profile's own step count (`tests/test_refine.py`), or the Refine
dialog on a 4-step checkpoint would open on a request the schema refuses. A
checkpoint the table does not know has no number of its own and runs at
`refine.DEFAULT_STRENGTH`. Two consequences are FLUX.1's alone, and the
dialog says both under Strength (the profile's `refine_large` hint) once a
pass is past the tight line, where they bite: 2x redraws more than 1.5x at
one strength (at 4 MP mu is 3.2, and even the schedule's last step starts
near sigma 0.5; measured 2026-10-04, 0.1 at 2x starts at 0.752 and repaints
the man), and a Mu set for the pass moves the index back - measured, Mu 1.15
at strength 0.2 starts at sigma 0.46 with 6 steps where the native
schedule's 0.1 starts at 0.47 with 3, and at 2x it starts at 0.463 and keeps
him. The strength axis exists to put the candidates on one sheet.

**The strength's help is the checkpoint's own numbers.** The Strength tip
is two sentences - what it is, where it starts on this checkpoint - and a
sweep example, all format strings over the profile: the example is
`model_defaults.strength_sweep`, half the profile's strength, it, and half as
much again on the 0.05 grid, less any value that leaves no step (0.05,0.1,0.15
on FLUX.1, 0.2,0.4,0.6 on SDXL). The fixed example it replaced, 0.3,0.4,0.5,
repaints the subject three times on FLUX.1, and so did the old 0.4
placeholder. The dialog's Compare 3 writes the same three values into
Strength - the page's `compareStrengths`, the same arithmetic in the same
order so the floats round alike, at the Steps box's count, half rounding up
on the quotient snapped to six places first - its placeholder is the start
and the sweep, and one visible line says how a sweep is written. The
rounding came from Z-Anime's 0.15, which puts both neighbours on a tie: in
floats 0.5 * 0.15 / 0.05 is 1.4999999999999998 and 1.5 * 0.15 / 0.05 is
4.499999999999999, which floored the sweep to 0.05,0.15,0.2 (-0.10/+0.05)
where 0.1,0.15,0.25 are the three `verified.md` measured. Rounding the
hundredths instead does not fix it, and Python's `round()` is half to even
where `Math.round` is not, so the two copies would have disagreed at 0.25.
The button only fills the box; a sweep is never remembered. The review plan
had strength chips under the box ("0.05 lighter · 0.1 default · 0.2
stronger"); they were not built: Compare 3, the arrow keys and the placeholder
cover what they offered, and on FLUX.1 the planned "stronger" chip was a
strength measured to repaint the face (above, 0.15 already changed it). On a
source made at few steps - a 6-12x2 steps grid makes them - a sweep around
the profile's value can leave no strength a step: at 2 steps the list was
empty, the placeholder read "0.1 or " and Compare 3 wrote an empty Strength
over the user's. Where fewer than two values are left the button is held
(aria-disabled, its title naming the Steps that would give two,
`compareStepsNeeded`), a press says so and writes nothing, the placeholder
is the start alone and the sweep line names no values. Where exactly one
value is left out - 0.05 at 12 steps - the button reads Compare 2, and its
title names the two values and the one left out and why ("0.05 leaves no
step at 12 steps", `compareSweep`): it used to promise "either side" over a
sweep with one side. The label is written only when it changes, since the
dialog syncs on every status poll. ArrowUp
and ArrowDown step a single strength by 0.05 (0.01 with Shift) - one fixed
step, since the profile's half would move SDXL's 0.4 past WAI's documented
0.35-0.5 in one press - and a list or a range is left alone. A strength out
of (0, 1] is refused in words that say what a strength is (`_a_fraction`, a
custom error, since pydantic's own bound read "Input should be less than or
equal to 1" in the dialog), and a single strength of 0.9 or more says in the
preview that the pass keeps (almost) nothing of the source: description, not
a refusal.

**Where a pass starts, read off the schedule** (2026-10-05). The preview
says how much noise a pass of one strength starts at ("· starts at 47%
noise", `GridPreview.start_sigma`), since on FLUX.1 one strength means
different things at different sizes: 0.1 starts at sigma 0.47 at 1.5x of 1 MP
and at 0.75 at 2x. It is read off the scheduler by the backend
(`backend.start_sigma`), never computed by a formula in the page.
`schedule.start_sigma` builds a copy of the loaded scheduler from its config,
with the loaded one's static shift, and hands it exactly what the
image-to-image class's call would hand it:

- the module's own `calculate_shift` over the sequence length that call
  measures, with the scheduler config's base and max shift (`SHIFT_CALL`);
- the call's default schedule;
- a requested mu, through `schedule.requested_shift` and `forced_mu`, which
  decide for `generate()` too which of the two ways a pipeline takes it.

That goes through the module's own `retrieve_timesteps`, and the sigma at
`refine.start_index` (times the scheduler's order) is the answer. The
statements the reading copies - each flow image-to-image class's
sequence-length and default-schedule lines in `START_RECIPES`, for
`FluxImg2ImgPipeline`, `QwenImageImg2ImgPipeline` and
`ZImageImg2ImgPipeline`, and `SHIFT_CALL` - are held to the installed
diffusers source the way the analyser's templates are, every flow refine
profile must have a recipe, and `tests/test_refine_schedules.py` reproduces
`verified.md`'s FLUX.1 sigmas on a real `FlowMatchEulerDiscreteScheduler`
with and without Mu. The loaded scheduler is never touched, since a job may be
running on it, and nothing is built over the loaded components outside a job:
the VAE scale is the text-to-image pipeline's, held equal to the sibling's.
There is no reading before a load, on SDXL (whose sigma is no noise
fraction), for a class the table does not know, an unprofiled start index, a
sweep, or a reading that raises - it is advice, and a preview it broke would
refuse the pass. The page asks for the preview again when the model becomes
ready or stops being so (status.js, beside the grid preview's re-ask), so a
load that lands with the dialog open adds the reading without a change to the
boxes, and an unload takes it away. The fake backend reads none.

This reverses the 2026-10-04 decision, which left it out: for FLUX.1 without a
Mu the shift comes from `calculate_shift` inside the pipeline's `__call__`,
and reading it meant re-deriving that outside the pipeline, which
`schedule.forced_mu` exists to avoid, or running the pipeline on the gate. The
objection was to a second formula that could drift from the pipeline's.
Calling the module's own functions over statements held to its source is no
second formula, and a test fails the day diffusers moves one. The measured
tight line and FLUX.1's `refine_large` note stay: they say what was measured,
the reading what this pass will do.

**Not the author's recipe.** WAI's hires pass upscales with R-ESRGAN 4x+
Anime6B and has its own step count; this pass resizes with Lanczos and runs
the part of Steps the strength leaves. An upscaling model would be another
pinned download and another thing to verify; it is not here.

**A dialog, not a mode of the panel.** The pass first shipped as a field
under Size: the card's Refine button rewrote the panel - prompt, seed,
steps, size - and from then on Generate, Batch, Repeat, the grid and even a
prompt comparison all ran passes until the source chip was removed. That
was confusing in use, and it needed guards of its own: a remembered source
retained across reloads, Generate and Run grid blocked while a retained
source could not run, a comparison clearing a remembered strength axis, a
seed the panel shared between direct renders and passes. Since 2026-10-04
the pass is a modal dialog (`web/refine.js`), opened by a card's ⤢, the
lightbox's Refine and Reuse on a refined record - and, since the results
viewer gave the image just made the card's tools (U28, U45), by a viewer
result's ⤢ and its column's Refine…, which share the card's handler
(`refineResult`) - and `collect()` never sends a source. The dialog starts from the record itself - its prompt, negative
prompt, steps, CFG, Mu, True CFG, adapter and seed; this checkpoint's
profile instead where the record is another checkpoint's, with the fold
that holds them open to say so. A pass's strength is that checkpoint's
schedule index, so on another checkpoint Strength starts at this one's own -
the profile's, or the general starting strength where the profile names none
or the pipeline is unprofiled (`refine_strength_profiled`) - and
`#refine-reused` says so ("its strength 0.4 was waiIllustriousSDXL_v170's;
Strength starts at FLUX.1 Krea [dev]'s 0.1"; on Anima "... Strength starts at
0.4, the general starting strength"), where it named the old strength beside a
box holding another, and the fold's note counts strength among what is this
checkpoint's. That note names only the knobs this checkpoint has, mu only
where Mu is supported, as Reuse's note does ("steps, CFG and strength are
Anima Turbo v1.1's"): it had named mu on checkpoints whose Mu box is
greyed. The dialog has its own size (scale buttons and
the two boxes), Strength, Versions and Seed. Versions is `num_images`, on
consecutive seeds from the Seed box: a pass is one pipeline call per image,
so they cost time, not VRAM. A Strength box holding a list or a range is
the grid's strength axis instead, with Versions as its seed count, and the
job opens as a sheet when done. The dialog closes once the server has taken
the pass (`startJob`'s `onAccepted`, U83), not on the press: closed first, a
refusal at submit - a source deleted from another tab - landed in the
panel's line and took the dialog's edits with it; it now stays in
`#refine-error` with them. While the POST is out the dialog holds, Refine
reading "Sending…" (the lightbox's section has why). Each change asks `/api/grid/preview` with
the grid-shaped request - the same validation the submit runs - so a size
past the guard or a strength that leaves no step is refused in the dialog,
in the server's words, and Refine is aria-disabled with the reason as its
title. A ⤢ press on a checkpoint with no pass says so and opens nothing
(a card's or a viewer result's ⤢ is aria-disabled there, as the lightbox's
Refine is, and says why over its tile and under the gallery's head);
Reuse on a refined record opens the dialog anyway, with Refine dead and the
reason shown, until a status says the loaded pipeline has one. The dialog
remembers per browser (`local-diffusion.refine.v1`) the scale and versions
last run and, per checkpoint, a single strength the user ran over the
profile's - never a sweep, or the next open would start one; Reset forgets
it with the panel. Ctrl+Enter inside the dialog runs the pass, not
Generate, and its button shows the shortcut (a pseudo-element: `setBusy`
rewrites the label).

**Opened on the picture, at the controls it is opened for.** From 1600 px
the dialog is two panes: the source's own PNG as large as the window allows
on the left - a strength and a scale are judgements about this picture, and
the 72 px cover-cropped thumbnail showed the middle 56% of a 16:9 source -
with the kept region of a reshaped size framed on it and the rest dimmed,
and the controls on the right at the width they have in one column. Under
1600 px it is the one column, the thumbnail showing the whole picture. The
two wrappers are `display: contents` there, so the tab order is the tree's
at every width; the figure spans the grid's rows with the last row the
flexible one, so a tall picture lengthens that row and never spreads the
controls. It is anchored at a fixed top and grows downward: centred, every
change of height - the fold, a sweep's second preview line, a refusal -
moved the controls, the fold's own summary 136 px from under the pointer.
The top margin comes out of `max-height`, so it still scrolls on a short
screen. Focus lands on Strength with its value selected, so a typed number
replaces it, where it used to land on Refine, the last control, and the
first Tab left the dialog; a plain Enter in a one-line box runs the pass,
as a form's submit would, so "open, Enter" still runs what it opened with.
Refine - the button, Enter or Ctrl+Enter - first waits for the preview of
what the boxes hold (`refinePreviewed`, the request the last preview
answered, against the one they make now): the preview is what refuses a
strength past 1 or one that leaves no step, and it is debounced, so
"1.5, Enter" ran ahead of it, closed the dialog, put the server's refusal
in the panel and lost the typed settings with it. A press on a refused
Refine says why once: the preview's own refusal moves to the error line,
in the preview's place and at its size, with Strength focused and marked
`aria-invalid`; a size past the guard keeps its size line and focuses W -
two copies of one sentence read as two problems, and a line of another
height moved Refine from under the pointer. The dialog's column never
shrinks its items (`flex-shrink: 0`): too tall for the window, it squeezed
each to its min-height, a hint's 15 px, and a two-line preview ran over the
line under it.

**What is around the source.** Opened from Reuse on a pass, the title and a
line of their own name the pass being run again ("Refine #4 as #5 was
made", "Reusing #5: 1344×1728 at strength 0.2, seed 9"), since the dialog
opens on the source. Images are named "#N" as the gallery's ids. The passes
already made from the source are a row of thumbnails, newest first ("Passes
of #4 (3)", the gallery's own `refine:` search on every checkpoint); one
opens in the lightbox with the source ahead of it, so ← and → step between
the pass and what it was made from at the same crop, already compared with
its source (`toggleCompare` after the open): judging the pass against the
source is what the row is opened for, and it had taken an S or another press
(2026-10-05). A close that did not run - Escape,
Cancel, the cross, a pass opened from that row - keeps what was typed for
the next open of the same image on the same checkpoint, in this page's
memory only, and the reopened dialog says so with a Start over; Reuse, any
other image, Reset or a run forgets it. It is never written to
`local-diffusion.refine.v1`, where a sweep would start on another image.

**The pass's own prompt is checked too.** The panel's Check reads the
panel's prompt (`collect()`), which a pass never sends, so the dialog asks
`/api/prompt/analyze` with its own request - the same cold, off-the-gate
route, the strength left to the server - on open and a moment after the
prompt, the negative or what switches it changes, under an epoch of its own.
A line above the preview, outside the fold, says where each encoder stops
reading ("CLIP-L/CLIP-G read 75 of 132 tokens (22 of 42 tags); the rest,
from "outstretched arm", is never read") with Show selecting the rest, and
each rule finding that rewrites its field whole has an Apply that writes
the dialog's box through `replaceBoxText`, never the panel's: on SDXL a tag
added at the end of a long prompt is past the 77-token cut and does nothing,
which a hires-style pass is exactly where people add them. A pooled-only cut
(FLUX.1's CLIP-L) and an encoder's advice are not shown. It never holds
Refine, says it is checking only while a first use fetches the tokenizers,
and is silent when it fails. Show and Apply go dead while the boxes hold
other text than it read or the negative's activation has moved, and one on
the negative prompt is dead, saying why, while the dialog's negative box is
greyed - written there, nothing would run - as the panel's is. The lines
are rebuilt on every reply, a moment after every pause in typing, so they
are no live region: a screen reader re-read all of them each time, mostly
unchanged. A visually hidden line beside them carries their text and is
written only when it changes. On a tag checkpoint the Prompt box grows with
what it holds, and a line under it names the comma items the tag index does
not know, each once, from `/api/tags/validate` - advisory, and silent when
the lookup is unavailable, as the chips are. The source's prompt above is captioned:
the box may hold another by now.

**The pass's prompt box suggests tags, and can take the panel's prompt**
(2026-10-05; the review's U66(d) and its set-aside G4.3). On a tag
checkpoint the box suggests tags for the comma item at the caret from the
chip editor's list (`scheduleEditorSuggestions`, `renderEditorSuggestions`
with `query` the item at the caret, `pick` `applyRefineSuggestion`, `place`
`placeRefineSuggestions`), `#refine-tag-suggestions` under the box, or over it
where the dialog shows no room below for the list as drawn. The panel's rules
hold: the first item is selected only from three typed characters it starts
with; Enter or Tab takes it as an insert, one Ctrl+Z from what was typed; an
Escape the list takes closes only the list - its keydown is prevented, which
keeps the modal dialog's cancel from firing - and the next closes the dialog.
A list already up stays while the next key asks, nothing in it selected until
the answer (`deselectEditorSuggestions`), so Enter cannot take a match for
what was typed before; emptied on every key, it blinked shut as the panel's
never did. A click, or a caret key that stays in the item the list answers,
asks nothing. None of the panel's list, chips or typing state is used; the
chip editor and the dialog share the editor list's timer and epoch, which is
safe only because the dialog is modal.

From panel, at the end of the Prompt caption in the fold, is the way from the
panel's Check, critic and comparison to a pass: a rewrite or a comparison's
winner is put to use in the panel, and a pass opens on its source's prompt.
It writes the panel's prompt and negative into the pass's boxes as typing
would (`writeRefineBox`, through `replaceBoxText`; the negative first, so the
prompt's Ctrl+Z is on top) and never writes the panel. It leaves a greyed
negative alone: a first version woke the box for the write and greyed it
again, and Chromium kept no undo step for an edit to a box greyed straight
after it, so the source's negative was lost for good. While the dialog's
negative is greyed the press copies the prompt alone and says to set True CFG
(or CFG) above the threshold and press again, and the title says so
beforehand (`refineFromPanelPlan`). It is held, aria-disabled with the reason
(`syncRefineFromPanel`), while there is nothing it would write: the boxes hold
the panel's texts already, only a greyed negative differs, or the panel's
prompt is empty. What a press did not do is said on a line of the caption's
own under the button (`#refine-from-note`, `sayRefineFromPanel`), scrolled
into the dialog's view and cleared by the next change, an open and a close:
the dialog's error line, at its foot, is out of the window with the fold open
at 1400x900, and a held press there seemed to do nothing. The Prompt field is
a div with a label for the box, as the panel's negative field is, since a
button inside a label becomes the label's control. Every whole-box write into
the dialog's two boxes - Apply, From panel - goes through `writeRefineBox`,
which leaves a greyed box alone and no suggestion list over the new text.

The Strength, Versions, Seed and dice row sizes Strength to its caption's one
line (`min-width: min-content`). `max-content` had taken the text box's own
20-character width too, which ran the row 15 px past the dialog at 1400x900
and at 125%; smoke-refine and smoke-wide hold the dialog to no sideways
scroll.

**A pass has a size guard of its own.** `max_side` and `max_pixels` are the
largest preset because a direct render past a checkpoint's range composes
badly; a pass takes its composition from the source, so that reason does
not hold for it, and a 4 MP image could not be refined to anything larger.
`refine_max_side` and `refine_max_pixels` are any 2 MP preset at 2x (4448,
and 1776x1184 x 4 = 8,411,136 pixels), read off `sizes.py` like the others;
the page derives the same two numbers from its copy of the table, held to
the server's by `tests/test_profiles.py`. The schema's `width`/`height`
bound is the larger, and `_direct_renders_stay_in_the_presets` holds a
request without a source to the presets' side; `check_budget` holds a pass
to its own pixels and skips the batch budget, which bounds images resident
in one call. It is a typo guard, not a VRAM model, as the preset one is:
measured on 2026-10-04 (`verified.md`), WAI ran 8.6 MP at 5.5 GiB with the
card two-thirds used, Z-Anime filled the card at 4.2 MP and took 715 s at
8.4 MP while spilling into system memory, and FLUX.1 fills it at 2.4 MP.
What fits is quoted, not modelled: a profile's `refine_measured` is the
largest pass `verified.md`'s "Past 4 MP" table ran without spilling - the
row's checkpoint, its quantization and the size: 1536x1536 on FLUX.1 Krea
dev and on Z-Anime, 2560x3360 on WAI, past the guard, so no size the dialog
offers is tight there, all at fp8 - and `tests/test_profiles.py` holds each
to its row. The page gets its pixels as `defaults.refine_tight_pixels` and
its words as `hints.refine_tight`, so past it the size line says "may run
tight on 16GB VRAM, past the 1536×1536 (2.36 MP) measured on FLUX.1 Krea
dev at fp8". The run is named because a profile is shared: FLUX.1 dev and
the lite repos are the flux profile in every number and were never
measured - the same transformer at fp8 on the one, 4-bit under Nunchaku on
the others - and "measured on this checkpoint" said otherwise; so did one
decimal, "past the 2.4 MP measured" on a request it called 2.4 MP. Every
checkpoint whose profile has no row is None and keeps the panel's generic
line past 1600x1600. Animagine, Kohaku and Illustrious run WAI's pipeline,
and until 2026-10-05 were not measured and did not borrow its number; that
day each was measured on its own, one size per run, and each runs the
guard's own top size, 2896x2896, at about 10.3 GB of the card, so that is
their row. The same day gave schnell 1536x1536 (within a gigabyte of the
top, as Krea dev's same transformer), Z-Image Turbo 2048x2048 (12.5 GB with
no CFG; 8.4 MP not run, its growth putting it past the card), the
undistilled Z-Image 1536x1536 (as Z-Anime) and Qwen-Image 1280x1280. Qwen-
Image fits this card only as Nunchaku's NVFP4 re-upload, so `Measured`
took its first non-fp8 quantization there, and the words say "at NVFP4":
the generic 1600x1600 line would have opened its dialog past the 1.6 MP
that ran within 1.4 GB of the top. After that day every catalogued profile
with a pass has a row, and the generic line is an unprofiled pipeline's
alone. The time estimate, from passes already made at that size, is still
the honest per-checkpoint answer.

## CellKey and run_id

**Three things widen `CellKey`, and `run_id` survives each.** The key is
(prompt, negative, steps, CFG, mu, true CFG, references, LoRA, refine,
seed), with `grid.reference_key` (ids in order), `grid.lora_key`
(`name@weight`) and `grid.refine_key` (`id@strength`)
spelled once and built from a request in `grid` and from columns in
`storage.completed_cells`. `run_id` names a cell by the **shortest** key
carrying all its non-None optional elements - it drops true CFG, references,
LoRA and refine from the right while they are None - so a grid using none of
them is still named by the six-element key its images have carried since
before any of them existed (pinned in `tests/test_conditioning.py`). A weight axis
with no adapter selected is not an axis: `applied()` puts `lora_scale` back
to 1.0 where `lora` is None, or two cells that differ on a number no pixel
depends on would split the resume key, and `_sampling_values` returns the
single value rather than the parsed list.

## Browser settings and the optional start-up menu

**A bare `python -m backend.main` starts without questions** and restores the
browser's saved loader settings. Explicit `--model`, `--quant` and `--offload`
flags win over explicit `LOCAL_DIFFUSION_*`, then saved settings, then defaults.
The saved values are one choice for one checkpoint: filled one by one, a small
model saved with BF16, no offload and auto-load turned `--model
krea/Krea-2-Turbo` into a ~25 GB unoffloaded auto-load. An explicit model other
than the saved one now sets all four aside, with a warning; explicit loader
flags still apply. The launcher records which values were explicit
(`LOCAL_DIFFUSION_STARTUP_OVERRIDES`, reported as `startup_overrides`) so the
page can say a saved choice will not apply to a start like this one.
`--defaults` skips the saved file; `--menu` opts into the legacy terminal picker.
A missing or malformed preference file does not block startup, and a saved
Nunchaku configuration is ignored on Windows so the browser can open and repair
it. Preference writes use a same-directory temporary file and atomic replace.
`settings` is frozen at
import, so `launch.configure()` runs from a `__name__ == "__main__"` guard
placed *before* `main.py`'s own package imports - the same reason
`generate.py` applies its flags before importing `backend.config`. Both entry
points take the flags from `launch.add_model_arguments`, so they cannot
drift. Imported `backend.main` (tests or a custom uvicorn entry point) still uses
its supplied environment directly; remembered launch behavior belongs to the
documented `python -m backend.main` entry point. The optional menu asks
quantization first because it decides the repo family:
`nunchaku` (offered on Linux only) lists the lite-infer re-uploads for the
families that have one and asks nvfp4 or int4; the environment's values are
the marked defaults, so `wsl/env.sh` still gets what it set on a bare Enter.
`wsl/serve.sh` forwards its arguments. `wsl/env.sh` marks loader values it supplied
as defaults, allowing saved browser settings to replace those unchanged defaults;
actual environment overrides keep precedence. The standalone CLI still receives
the WSL defaults and never reads browser preferences.

The Linux server launcher prepares remote-kernel trust before diffusers freezes
its import-time flag, because Nunchaku can now be chosen later in the browser.
An explicit false remains in effect and Nunchaku is refused before unloading.
This does not import diffusers or load kernels on startup; only an explicit
Nunchaku selection reaches that loader. The standalone CLI retains its original
startup-quantization behavior.

The page names the running checkpoint by this catalogue's label
(`ModelStatus.model_label`, by exact id, through the page's `checkpointName`,
which the topbar, the prompt check's head, the facts' Model row, the load
card, a Reuse's note and its ⚠ title, Expand's running line and the Refine
dialog's lines share), not the repository's last segment: NetaYume v4 read
"NetaYume-Lumina-Image-2.0", the parent's name, and Z-Anime Base and Anima
Turbo v1.1 lost their versions (2026-10-05); a Reuse's note and the dialog's
lines still did after the topbar had changed. Off the catalogue it is still
the id's last part, and another checkpoint's records are named by their id's
last part, since the status labels only the running one. The public id stays
in every title, row and PNG chunk.

## no_grad, never inference_mode

**Use `torch.no_grad()`, never `torch.inference_mode()`, around the pipeline
call.** Model offload moves components across devices inside `forward()`, and
torchao's quantized tensor subclasses trip torch 2.13's storage-aliasing guard
when that happens under inference_mode. The failure surfaces deep in
`torchao/utils.py -> return_and_correct_aliasing` as "Attempted to set the
storage of a tensor on device cuda:0 to a storage on different device cpu".
This is also what the `Unable to import torchao Tensor objects` startup warning
is pointing at.

## Seeds fit in a double

Seeds are drawn from `2**53 - 1` (minus the largest sweep), not `2**63 - 1`:
JSON numbers are doubles, so a 19-digit seed came back from `res.json()`
rounded and Reuse then generated a *different* image while claiming to
reproduce one. The wrap modulus stays at `2**63 - 1`, the seed column's
bound, so a hand-entered `2**60` is still itself. Both live in
`backend/seeds.py` because the draw, the per-image offset and the generator
must not drift apart.

## One generator per image

A batch gets **one generator per image**, seeded `seed + i`. diffusers draws
each sample from its own generator and collapses a one-element list back to
the scalar path, so image *i* is identical whether it came out of a batch or
a solo run of its recorded seed. One shared generator made the per-image
seeds in the gallery unreproducible.

## Per-control defaults and Reuse

`mu` is part of the reproduction set, so `ImageRecord` carries it and the UI's
Reuse restores it. `/api/status` defaults are applied **per control**: each
of steps/CFG/mu follows the profile until the user touches *that* box
(`owned` in `web/state.js`) — `describe()` reads CUDA memory and can answer
minutes late from behind a load, and one flag for all three, wired to Seed
as well, used to freeze the panel at the fallback's 28 / 3.5 on a Turbo
server because the user pasted a seed in the first two seconds. Reuse owns
all three. Its button warns before the click, and the panel keeps an explanatory
note after the click, when the record's checkpoint is not the one the server is
running: steps, CFG and mu are checkpoint-specific. An empty
Steps or CFG box is sent as *absent*, which the schema fills from the
profile; `+''` is 0, and 0 is a valid CFG that generates unguided. Reuse
does not carry a record's variables: it fills an editable panel, and what
Generate then makes is a new image from `/api/generate`, which records none.
A comparison version's name rides the two reproductions, Re-render and the
CLI's `--from-image`, not Reuse. A refine pass is the Refine dialog's, not
the panel's (see "The refine pass"): Reuse on a refined record asks
`GET /api/images/{id}` for its source and opens the dialog on it with the
pass as it ran, and only when the source has gone does it fill the panel -
its prompt and sampling, at the panel's own size, with a note saying the
pass cannot be made again. The dialog fills its own boxes; nothing of the
panel's `fillPanel` is shared with it.

Reuse of another checkpoint's record writes its Mu only where Mu applies, as
it writes True CFG only where that is a knob (2026-10-05). On Anima the
greyed Mu box had shown the record's 0.8, `collect()` sent it, and the panel
remembered it; a Mu on the row then claimed a shift the schedule never took.
Now neither `collect()` nor `gridPayload` sends a greyed Mu box or axis, as
with the LoRA weight, and the note asks to check "the steps and CFG boxes"
there, not a Mu that is not on offer.

**Reuse can be taken back.** It replaces the prompt, the negative, the seed,
the size, steps/CFG/mu and the references and adapter in one click, and it
used to do so over a half-written prompt with no word: the boxes were
assigned `.value`, which empties a textarea's undo history (and in Chromium
a later Ctrl+Z could splice the old text into the new, which `persistPanel`
then remembered). `fillPanel` now writes both boxes through `replaceBoxText`,
as Use and Apply do, so Ctrl+Z walks back through them, and still
synchronously - the references are in place when Reuse returns. Before it,
`reuseIntoPanel` keeps `panelState()` - the object the panel persists - with
the raw steps/CFG/mu/true-CFG values and the `owned` flags beside it, and its
note opens with "Settings from #N loaded - your prompt was replaced." (the
last clause only when there was other text) and an **Undo** right after that
first sentence, which puts all of that back, the boxes again through
`replaceBoxText`. The note stays in the panel's line, where Reuse sends the
user; whatever else it has to say - the checkpoint-mismatch sentence, kept,
and the rest below - follows the Undo, so the action is on the note's first
lines, the ones in sight where the window shows the prompt and only the
start of the note. The note also says what `fillPanel`
could not carry, since a dropped reference renders a plain text-to-image
that looks like a success: references on a checkpoint that takes none,
and - found by asking, after the synchronous fill - an adapter the LoRA box
does not offer (the reason is `/api/loras`' own: a misfit's `detail`, an
installation that cannot load adapters, or a file no longer in the
directory) and a restored reference the server answers "unknown image" for,
which is then dropped. Only that answer drops one: a passing failure must
not take a live reference out. A reference deleted from the gallery while
the Undo is still on offer leaves the Undo's copy of the panel too
(`forgetImages`), or the Undo would bring back a chip the submit refuses.
The Undo is on offer only while the panel holds what the Reuse left there
(`reuseNote.after`, compared by `syncReuseUndo` on every `persistPanel`):
the first change - typing, Ctrl+Z, a seed, a preset, an Example - takes the
button and leaves the note's sentences. It used to stay through anything, and
the integration review pressed it after typing, a new seed and an Example
pick: all three were thrown away, the seed beyond Ctrl+Z's reach, and the
note said the panel was as it was - while the Example's own Undo, which
goes once its box changes, showed beside it with the opposite meaning. So
the page has the Examples' rule for both and one Undo on offer at a time: an
Example retires a Reuse's, and a Reuse an Example's (`dropPromptUndo`). A
reference dropped from what the Reuse left - deleted, or answered "unknown
image" - is no change of the user's, and comes out of `after` as well. A
Reset takes the note's words too, Undo or none, while they are still what
the line says (`clearReuseNote`, from `resetPanel`; `reuseSaid` is the note's
first sentence): once Reset's own note moved to the prompt box, "Settings
from #N loaded - your prompt was replaced" had stood over the emptied panel
(2026-10-05).
Both boxes are written through tags.js's `writeBox`, the page's one
whole-box write (`rewriteBox` focuses the box without a scroll first).

**Where Reuse leaves the page** is `revealReused`'s: the prompt's field,
label and all, in sight under the topbar, and the user's place in the
gallery kept wherever that allows. It used to scroll to the top of the page,
which beside the gallery threw that place away for nothing; then to the
prompt with `scrollIntoView({ block: 'nearest' })` when the prompt was out
of sight, which left it under the sticky topbar on one column and, beside
the gallery, scrolled a page whose stuck panel does not follow a page
scroll - 600 to 750 px of gallery lost at 1400x900, the prompt still out of
sight. The browser's own scroll to the focused box was no better:
`replaceBoxText` focuses the box, and the browser scrolled the page as
though a stuck panel moved with it (half a screen at the page's end), or
not at all for a prompt focused already. So `fillPanel` and the Undo focus
each box with `preventScroll` first (`rewriteBox`), and once the note is
written - the panel at its final height, placed at once rather than after
the frame - the panel is moved, not the page (`revealInPanel`, see "The
settings panel's sticky top"). Only where it cannot be - a one-column page,
or a panel at the page's end, which a sticky top cannot push down - does the
page move, as far as puts the panel where it rests, smoothly unless reduced
motion is asked for. Then, if the Undo is below the window, a stuck panel
goes up to bring it into sight, but only as far as the prompt's field stays
clear of the topbar (`showUndo`, after the smooth scroll ends): at QHD and
at 125% both fit; at 1400x900, a four-line mismatch note included, they do
not, and the prompt is what Reuse is about - Ctrl+Z in it still gives the
text back. `smoke-delete.json` holds all four: the page unmoved
mid-gallery with the prompt focused already, the page's end, one column,
and the Undo in sight at 2048x1040 at 125%. The lightbox's Reuse closes the
lightbox first: a modal dialog keeps focus to itself, and `replaceBoxText`
needs the prompt focused.

## One worker thread

Generation is serialized through a one-thread pool — the GPU cannot usefully
run two at once.

## A job outlives the page; per-stream queues

A job outlives the page that started it. Each SSE stream gets **its own**
queue (`Job.subscribers`): one shared queue is destructive, so a reconnecting
client overlapping the stream it replaces would split the events and only one
reader would ever see `done`. The browser's automatic reconnect is left alone
for the first few failures, and `GET /api/jobs` lists what is still live, so a
reload reattaches instead of orphaning a sweep with no reachable id.

**A stream that gives up asks what became of its job.** It used to say "Lost
the generation stream. The run carries on - reload to pick it up.", which was
false whenever the server had died: a reload found nothing, and an overnight
grid's missing hours went unnoticed. After the five reconnects (or a reply that
was no stream - a restarted server's 404 closes EventSource at once) the page
probes `GET /api/jobs/<id>` (`gaveUp`): still there, it streams again (three
times at most, then says the job runs on but its progress keeps getting lost);
no answer, "Can't reach the server - it may have stopped", how many images
were saved, and the offline state takes over; a 404, "The server restarted,
and this job ended with it after N of M images" (M is the snapshot's
`planned_images`), Generate again or a grid's Resume for the rest, and the
gallery takes in what was saved. The live preview of the unfinished cell goes
with the stream, rather than sitting in the strip like a second result.
Where nothing of the job had reached the strip - its first frame had cleared
it - the viewer was then an empty box, still empty once the server was back,
the only words red in the panel; it holds a card now (`showLostRun`, styled
as a failed load's): what went, the same sentence, and Run it again where
this page sent the request, its refusal said in the card. An answering
server rewords it ("The server is back", or "restarted at 04:58",
`lostRunBack`), and the next job's first frame clears it. While the stream
reconnects, the job says it in the topbar's words, "server not answering -
retrying (2)", and nothing beside them that has stopped being true: no step
count, no time left, no mirror of it in the topbar, the tab's title the same
phrase and the strip's note too ("reconnecting... (2)" sat under the
topbar's phrase with a frozen "~19s left", and the tab said "Lost the job");
the next frame brings the counts back. A give-up with the server gone holds
"Server not answering" in the title.

**When the server does not answer.** The topbar used to stay green, with the
checkpoint's name, for as long as the server was dead, and every action said
"Failed to fetch". `api()` turns a fetch rejection - only that, never an HTTP
error or a slow reply, which a load's CUDA read can cause and `statusPending`
absorbs - into an Error with `offline: true` and "No answer from the Local
Diffusion server - is it still running?". A status poll that gets one turns
the dot red ("server not answering - retrying"), hides Load, and holds
Generate, Run grid, the Refine dialog's Refine and Unload (`aria-disabled`,
titled with the reason; a press says it and polls at once); Refine's preview
fails with the same words.
`showError` remembers which slots hold such a message (`offline`, set from the
text or by the caller), and the first answer clears them (`clearOffline`).
The lightbox's own picture is one of them: one that fails to load asks for its
record, and where the server is not answering (`serverOffline`, or the ask's
`offline` error) `#lb-error` says the same sentence as an offline message -
it had a second wording of its own - which the first answer takes back while
asking for the picture again (`retryLightboxImage`). Otherwise it says
"#N is no longer in the gallery." on an unknown image, "The file of #N could
not be loaded." where the record is still there, and "This image could not be
loaded." for any other refusal; the next picture that loads clears it.
`ModelStatus.started_at` is set once, at import: a changed value is a restart,
with or without a failed poll between, which a quick one leaves none of; the
page then says "The server restarted at HH:MM - the model is not loaded, and
any job that was running ended with it" where the panel's line holds nothing
more particular, forgets the queue, and takes in what the old process saved
(`prependFinished`). Load more stays the retry for a failed page and is not
re-armed.

## Atomic PNG writes

`save_image` writes `<name>.png.part` and renames only after the row commits.
`_write_lock` is per-process and `clear.py` is a separate one: mid-encode, a
plain `.png` is an untracked file for its orphan sweep to delete. The encode
sits *inside* the cleanup guard (a KeyboardInterrupt mid-encode is a
BaseException Pillow does not tidy), the cleanup only runs while the row has
**not** committed - past that the partial is the only copy of an image the
database already names - and the rename is retried over a few hundred
milliseconds, because on Windows a just-closed PNG is exactly what Defender
or an indexer holds open. The orphan sweep and `summarise()` glob the
directory *before* reading the rows (a file the scan saw has a row by then),
and sweep `.part` files only past `PARTIAL_SETTLE_SECONDS` (an hour): a
fresh one may be another process mid-encode. `_migrate` reads
`PRAGMA table_info` on every connection: its once-per-process flag, keyed on
the database path since 2026-09-07, still skipped an older backup restored
over the *same* path, and with `binned_at` in every live read the whole
gallery failed until a restart (review 2026-10-12, R36).

A stale partial is not always debris (2026-10-12). Past the commit the
`.png.part` is the image, and a crash or a failed rename between the row and
the rename leaves exactly that: `storage._settle` renames a stale
`<name>.png.part` whose row exists and whose `<name>.png` does not into
place and discards the rest. Its `thumbnails/<name>.webp.part` is not the
only copy of anything: the PNG is, and reindex's `thumbnail_for` encodes for
rows already tracked, so a kill mid-encode leaves a cut-short WebP that,
renamed into place, would stay the card's broken thumbnail for good (nothing
re-encodes one that exists). It is discarded, and the card shows the PNG. Bundle operations swept only their own leftovers, and the bundle
inventory refuses any `.part`, so one killed save refused the counts,
export and import for good, restarts included, with a message naming
neither the folder nor anything the page could do. `settle_partials` now
covers image and thumbnail partials at startup, at every bundle preview and
before each operation, and `clear_history` settles before its thumbnail
sweep (which would take the thumbnail of a PNG still named `.part`). A
`.part` the inventory still finds is a write in flight, and the refusal
says so with its folder.

A row whose PNG is gone refuses the bundle preview too, and its thumbnail
can outlive it, so the card looks whole. The refusal named only the file,
which no search key matches. `bundles.MissingImage` names `#N`, the folder
and the way out (the file back, or Delete permanently from the bin, which
is the only route that removes a row), and its 422 carries `image_id` and
`binned` beside `detail`, so the dialog's inline **Show #N** can search
`id:N` in the gallery or the bin. Archiving without such rows would be a
silent loss inside an operation that clears the gallery, so it is not done.

## Thumbnails have their own directory

WebP previews used to sit beside full-size PNGs, making a batch selected in
File Explorer include both. PNG filenames, metadata and gallery ids stay the
same; previews now live in `outputs/thumbnails/`, served by the existing
`/outputs` static mount. `init_db` moves only previews belonging to indexed
PNGs that still exist. `thumbnail_for` also moves them during recovery of a
lost index, before deciding whether a new preview needs encoding.

Migration preserves the thumbnail bytes, skips an existing destination and
leaves a locked file in place for a later startup. Record reads prefer the new
location and fall back to the old one, then to the PNG in the browser. Open
tabs should be refreshed after restarting. Both locations participate in
deletion and disk accounting; partial cleanup scans those two directories
with the same one-hour age guard. Directory creation belongs to writes, so a
clear or reindex dry run does not reorganize files.
The CLI's Resume lookup opens its schema-checking connection directly;
`init_db` stays after the generation dry-run guard for the same reason.

## Shutdown has to reach the sweep first

**Shutdown has to reach the sweep before uvicorn drains connections.**
uvicorn closes listeners, waits for open connections, and only *then* runs
the lifespan shutdown - and the browser's EventSource stays open until the
job emits `done`. So `jobs.shutdown()` (the cancel) sat behind the stream
that was waiting for it, and Ctrl-C printed "Waiting for connections to
close" for the rest of a grid. `main.Server`, the `uvicorn.Server` subclass
`main.run()` serves with, stops every active job in `handle_exit`, i.e. on
the signal itself (`jobs.stop_all("shutdown")`: `cancel_requested`, through
the cancel path as a budget stop is, with `stopped_by` "shutdown", which
`shutdown()` sets too), and starts the critic's stop on a thread of its own.
The `stopped_by` is for the page: a shutdown read as a plain cancel, which a
second tab then reported as one pressed somewhere else (Two tabs on one
server); it says the server's stop now. `run()` sets
`timeout_graceful_shutdown=30` for a client
that stops reading. `Server` sits at module level so
`tests/test_api_analysis.py` can call `handle_exit`; that file also pins the
lifespan's order: the critic's server, then `jobs.shutdown()`, then the
weights. Lifespan calls `backend.try_unload()` (non-blocking lock acquire)
off the loop rather than `unload()`, which blocks behind a load in progress
and then discards what it built; `POST /api/model/unload` answers 409 while
a load is in flight for the same reason.

The load runs on a **daemon thread** (`detached.detached`), not
`asyncio.to_thread`: that pool is joined at interpreter exit, so a Ctrl-C
four minutes into a 32 GB download waited the download out. The critic's
`prepare()` runs on one too (`critic-prepare`), for the same reason: a first
use downloads about 5 GB. Since 2026-09-28 Check's analysis and the
critique's analysis and lint do as well (`prompt-check`), and each tokenizer
load runs on a daemon thread of its own (`prompt_encoding._fetch`), because a
first use's Hub fetch has no HTTP timeout; uvicorn still answers an open Check
before it stops, which `LOAD_WAIT` bounds.
`test_prepare_and_the_load_run_on_daemon_threads` drives the critique and the
load through `httpx.ASGITransport` on a loop in pytest's main thread, as
uvicorn runs its loop, not through TestClient: TestClient runs the app's loop
on a daemon thread, every thread that loop starts inherits the flag, and an
`asyncio.to_thread` or Starlette threadpool worker would pass for a daemon
there. After the calls it checks that each ran on a daemon thread that is not
in `concurrent.futures.thread._threads_queues` - the interpreter joins every
thread in that registry at exit whatever its flag, and a worker is added
only once it has started.

Unload stops the critic's server too, and all three stops go through
`main._stop_critic`. Unload's runs off the event loop - a stop waits up to
twice `STOP_TIMEOUT`, which on the loop would stall every progress stream
(`test_unload_says_unloaded_before_the_analyser_and_critic_let_go` holds it).
`prompt_critic.unload()` raises `CriticError` when a kill is not waited out.
Raised through the lifespan, that skipped `jobs.shutdown()`, `try_unload()`
and the tokenizers' unload; on the signal's thread it was an unhandled
thread exception; through `POST /api/model/unload` it made the route a 500
over weights already gone and left the tokenizers cached. It is logged and
the rest goes on. Unload still answers with the status, which stays true:
"unloaded" is the image model's state, and `vram_used_mb` is read off the
image model's card (`torch.cuda.mem_get_info` on `settings.device`,
device-wide), so on this app's one card, the memory `llama-server` still
holds shows there rather than being claimed free. `llama-server` is pinned to
no card, so on a second GPU only the ERROR log line, with its pid, says so.
No field of its own was added for a case that needs a kill to fail.

Once shutdown is complete, uvicorn re-raises the signal it captured, so a
Ctrl-C reaches `main.run()` as KeyboardInterrupt. `uvicorn.run()` swallows
it, but `run()` calls `Server(config).run()` directly for `handle_exit`, and
every Ctrl-C ended in a traceback after "Application shutdown complete".
`run()` exits 130 instead, as the start-up menu does on Ctrl-C;
`test_ctrl_c_ends_the_server_with_130_and_no_traceback` sends a real SIGINT
to a started server. Launched from a script without job control, a
background server has SIGINT ignored, and there the re-raise is a no-op that
exits 0.

## A cell's failure is the cell's

**A cell's failure is the cell's, not the night's.** `_generate` catches
`Exception` after `Cancelled`, records it on the job (`failures`, `error`),
skips the cell and continues; three failures in a row, or more than a third
of the grid, re-raise and end the job. A finished job with failures reports
`done` with an `error` line the UI shows. Every job logs a line when queued,
one per cell and one at the end, and carries `created_at`/`started_at`/
`finished_at`/`last_progress_at` as float seconds in its snapshot
(json-native; a datetime would raise mid-job on the SSE path). `_generate`
returns whether it left the loop early, and the terminal status comes from
that rather than from the
cancel flag - a cancel that lands after the last step used to relabel a
complete run "cancelled". Every terminal transition goes through `_finish`,
which owns the `done_sent` guard for state *and* event: a queued job the
endpoint had cancelled used to be rewritten to `error` when its load then
failed. `generate.py` mirrors the per-cell rule and exits 1 when any cell
was skipped. `Job.snapshot(images=False)` is what the per-cell status event
and `/api/jobs` send: the per-cell event used to carry every record so far,
258 MiB over a 1000-cell grid.

**The page names the cells it skipped and offers them again.** A skipped cell
used to show only as the cell line jumping a number and, at the end, "1
cell(s) failed" - which cell, no one said - and the obvious retry, Run grid,
rendered the whole grid again. `Job.failed_cells` keeps each one's label, seed
and error, sent with the records (the full snapshot: attach and `done`), never
on the per-cell events; the cell line counts "· N skipped" from the status
frames' `failures` as they happen, and the done line names them. A grid's
done line then offers "Retry the N skipped cells", which sends the payload the
page submitted (`submissions`, by job id) again with `resume: true` and the
base seed its POST returned - never the panel, which may have changed since -
so only the missing cells run. When the failure rule itself ends the job,
`stopped_by` is `"failures"` (still status `error`, through `_finish`) and the
error leads with where and why - "Stopped at cell 4 of 5 after 3 cells failed;
1 image kept. Last: <the exception>" - since three in a row or a third of the
grid read as one bad cell when only the last exception was shown; the page
offers "Resume the remaining N cells" (the total less the cells that rendered).
Both live in the finish line, `#finish-note`, apart from the cell line: when
they shared `#cell-note`, the next queued job's start cleared it the moment the
grid ended, and the Retry with it - in the overnight queue the button exists
for. The finish line stays until the next job's end replaces it, led by
"Previous job:" while another runs (or after one whose stream gave up). A
Retry or Resume is sent once: accepted, it reads "Retry sent", `aria-disabled`,
and keeps focus where the press left it; when the line is replaced with focus
in it, focus moves to the new line's first button, else Generate. A job that
ends with focus on Pause or Cancel (or a queued row's ×), which hide with
`#progress`, hands it to Generate (`setBusy(false)`), and the finish line
then takes it on to its first button (`finishTakesFocus`): a keyboard user
who pressed Cancel and let the queue run out was left on `<body>`.

What the job's end says in the panel's line, `#error` - cancelled, a time
budget, an error, the skipped cells, the failure rule's stop - is that job's
outcome (`showOutcome`, `jobOutcome`) and lasts as the finish line does: led
by "Previous job: " while another job runs (`#error.previous`, a generated
`::before`, so the alert is not read out again) and taken away when the next
job's end is drawn (`finishJob`). It used to stay until the next press of
Generate, and with a queue "Cancelled before anything finished." sat under
the next job's own "Finished 1 image" as though it were its outcome. A press
on a held Pause, Unload, Expand or Ask the critic says why in the same line
(`showJobNote`; Expand's through `expandSay`, which puts it under the prompt
while a Reuse's Undo holds the line), which goes when the job ends: "A single
batch has no pause point" outlived the job and the button it explained. Every
held press says its reason as a sentence with a closing stop (2026-10-05);
Pause's, Unload's and Expand's titles are the same reason without it, and the
critic's titles are whole sentences already. The critic's press during a job
had been a red error that outlived the job, and Pause's and Unload's notes
their bare titles, with no stop, beside Expand's sentence.

A screen reader hears the job's end once (2026-10-05). `#finish-note` is no
live region - its "Previous job:" lead is rewritten at every job's start - so
the finish line's sentence, without its buttons, is written on a visually
hidden polite line (`#job-said`, `sayJob`) unless `#error`, an alert, says the
outcome already: a cancel, an error, skipped cells, or a load's detail where
the viewer holds images. So is "Queued: <job>" when a press queues one behind
the watched job, since the queue's rows are no live region either. A shutdown
reads "Stopped: the server is shutting down", never a cancel.

## Cooperative cancellation

Cancellation is **cooperative**. A diffusion step cannot be interrupted from
outside, so `POST /api/jobs/{id}/cancel` only sets `Job.cancel_requested`; the
backend polls it from `callback_on_step_end` and raises `Cancelled`, which
unwinds out of the denoising loop. A cancel therefore lands within one step,
and everything already written to disk is kept and reported. A job still in
the queue is finished off by the endpoint itself rather than waiting for the
job ahead of it, so `_finish()` guards the terminal SSE event against being
emitted twice.

## Runs, not batches, for many 2K images

`request.runs` loops `jobs._generate` over N sequential pipeline calls with
base seed `seed + i * num_images`. That is the *only* way to get many 2K
images out of 16 GB: `num_images` holds the whole batch in VRAM at once, runs
hold one. Each image is emitted as its own `image` SSE event as it lands, so a
64-run sweep fills the page as it goes.

The batch-wide pixel ceiling is a typo guard, not a VRAM formula: peak memory
varies substantially by checkpoint, quantizer and conditioning. Its default is
the larger of two maximum-size 4 MP presets and a full batch (`MAX_BATCH` in
`seeds.py`, the one number the schema, the Batch box and the budget read) at
the largest 1 MP preset aspect. The original two-at-4-MP ceiling was 8,411,136
pixels, which accidentally rejected eight 3:4 or 4:3 images from the nominal
1 MP row; a measured batch of seven used only 10.7 GB of the 16 GB card. The
cap itself was 8 until 2026-09-12, when eight 1 MP images on Animagine XL 4.0
read 9.6 GB of the 16 as read off the card; it is 12 now, which the same
reading puts at roughly 13 GB on that checkpoint. A heavier checkpoint at
Batch 12 is the user's call, as it was at 8: the ceiling was never a VRAM
model. Large batches are still refused, and a machine with different headroom
can set `LOCAL_DIFFUSION_MAX_BATCH_PIXELS` explicitly. A refine pass is
outside it: it makes one pipeline call per image, so its batch never holds
more than one image at once ("A pass has a size guard of its own").

The panel labels the count **Repeat**; `request.runs`, the `runs` box id, its
remembered key and the CLI's `--runs` keep the name. "Run" is the
submission everywhere else on the page - the gallery's Runs picker lists one
Repeat-3 Generate as one run, `run:` searches for it, the last-run banner
reports it - and a panel box of the same name counted something inside it
(U88). Batch has a badge saying why Repeat exists, and the line under the
boxes says batches, not passes, which is the Refine dialog's word. In one
column (under 901 px) the badge's tip opens leftwards: there the badge is
600 px along an 800 px row, and a tip opening rightwards ran past the window
and widened the page.

## A job is a list of cells

A job runs a **list of cells**, not a request and a run count. A plain request
expands to `runs` cells of identical settings; a grid (`POST /api/grid`) to
the cartesian product of its axes. `_generate` cannot tell them apart, so
cancellation, the progress events and the per-image seed maths are written
once. `Job.total_steps` is the *cell's*, since a grid varies it between them.

## Seeds are the outermost loop

In a grid, **seeds are the outermost loop**. A night that ends early - a
cancel, a crash, a power cut - has then finished every parameter combination
for the seeds it reached, which is the comparison that was wanted; the other
order leaves every seed part-done and nothing comparable. Cells are built as
real `GenerateRequest`s, so an axis value outside the schema fails at submit
with the schema's own message rather than sixty images in.

## The prompt is an axis: variables

**The prompt is an axis too**, through `{name}` placeholders and a block of
`size = big | small` lines (`grid.parse_variables`). Values split on `|`, not
on the comma the numeric axes use: `a big, fluffy dog` is one prompt fragment,
and splitting it the other way turns a value into two cells of nonsense. The
names cross with each other and with steps/CFG/mu, so two variables of two
values is four prompts. Both halves of a mismatch are refused at submit rather
than found in the morning: a variable no placeholder mentions multiplies the
night by a value that changes nothing, and a placeholder with no variable
renders the literal `{ground}` into every image of the run - **including
when the variables box is empty**, which used to short-circuit past both
checks (a plain prompt with no placeholders still expands to itself). The
product of the variables is **counted before it is built**
(`math.prod` against `max_grid_images`): `MAX_AXIS_VALUES` caps values per
variable, nothing capped the variables, and eighteen binary lines built
262,144 dicts on the event loop per preview keystroke. `none` is a
value - the word is simply not said - and `substitute()` closes the gap it leaves,
so `{size} dog` is `dog` rather than ` dog`, and nothing else. The prompt loop
sits just inside the seed loop, for the reason seeds are outermost: a night
that ends early has swept every setting for the prompts it reached, which is a
comparison.

**`none` closes only its own hole** (2026-10-05). One space goes with the
word (`grid._close_hole`): the one after it where the word followed a space or
opened a line, a bracket or an XML element (`<appearance>{hair} long_hair`),
else, at a line's end, the one before; a placeholder alone on its line empties
that line and keeps the line count. It used to collapse every doubled space in
the prompt and strip every line whenever any value was empty. On an indented
XML prompt - NewBie's, pasted from its card - the none cell then lost every
line's indentation, 21 of the 25 Gemma 3 tokens it differed by, and the
comparison was no longer the word alone. `red_eyes, {hair}long_hair` keeps its
one space, and punctuation that belongs to the value still belongs inside it
(`mood = , at dusk | none`). A grid started before this change whose none cell
read differently - its `none` beside doubled spaces, indentation or trailing
spaces, or right after `(`, `<tag>` or a quote - gets a new run id on Resume,
and its none cells render again: the old cell was not the comparison it
claimed.

The one mistake neither check catches is a variable that does not vary.
`size = big, small` is legal - one value, `big, small` - and the night runs
`a big, small fox` on every cell. The preview listed the resolved prompts
only from two up, so it went quiet exactly when the substitution failed. It
now lists the prompt whenever the variables box has text (one as `prompt:
...`), and the page marks any line of one value in the warning colour,
naming the comma when that is why. That warning is the page's alone and
advisory: the `|` split, and the parser that decides, stay the server's.

The placeholders are also why **Ctrl+Enter goes where the focus is**. It
used to generate from anywhere, so pressed in a grid box it rendered the
literal `{size}` as one image, and the night's grid never started. From a
box of the open fold it now presses Run grid (its click, so the guards are
the button's), in the Refine dialog it runs the pass, and anywhere else it
generates. Generate itself refuses, in the page only, while the fold is
open, its variables box has text and either prompt holds a `{name}`: the
server never refuses braces, which in a plain prompt can be text or pasted
`{tag}` emphasis. Both buttons show the shortcut as generated content with
empty alt text, since `setBusy` rewrites their labels as text.

## Prompt labels on the sheet

The contact sheet's axes gained **Prompt**, whose labels are only the words
that vary across the run (`promptLabels` in `web/sheet.js`): a column header
repeating the forty words every cell shares is unreadable. Equal word counts
let the differing positions be picked out exactly; a `none` makes one prompt
shorter than its neighbours, and then the shared head and tail come off
instead. Pins address their values by position, because a shortened label is
not something that round-trips back to the prompt it came from.

Successive edits of one prompt defeat both: different word counts, and a word
added at the front keeps every head and tail, so six iterations of a portrait
labelled themselves with three clipped openings, five of six columns alike.
Where the two paths leave labels repeated or longer than `LABEL_CHARS`,
each prompt is labelled by its edit against the oldest (`promptEdit`, over
`promptChanges`, the critic's exact diff): `+, film grain`, `−golden +blue`,
runs of edits joined by `…`; the oldest keeps its own words. That needs the
prompt axis in the order the prompts were written, so `sheetValues` orders it
by oldest image as it does a variable (alphabetical put `35mm photo, …`, the
newest, first). On screen a label is whole - a heading wraps to three lines
and its title is the full prompt - where a 540 px column used to show 44
characters; `clip` stays for a line of fixed width: a pin's option, the note,
the export's canvas. Each edit is a diff, quadratic in the prompt's length, so
the edits stop at `EDIT_LABELS_MAX` (64) prompts - past that the labels are the
varying words, as before - and the labels of the last prompt list are kept
(`promptLabelMemo`) for the redraws a pin, an axis, a star or a delete asks
for: a typed search over a thousand 250-word prompts took 600 ms a redraw.

A comparison's `var:variant` axis reads as the prompt check names its
versions (`VERSION_NAMES`: Original, Reordered, Rewrite, Your version) under
the axis label Version, on the sheet and in its export, where it read `variant
yours`. The stored labels are untouched - they are what `stats --by variant`
groups by - and a label the map does not know reads as written. Where each
label names one prompt (`versionPrompts`; where one names several, Prompt is
kept as an axis instead), a version's row head carries the whole prompt as its
title and, under the name, its edit against the original, up to 80
characters: four 400-character versions open with the same words. A version
that keeps the original's words and only moves some - the critic's Reordered -
says `same words, moved: soft light` (`versionEdit`), each moved run once from
the added side: as an edit it read `+soft light, … −, soft light`, every moved
run twice.

**An XML version is read by its blocks** (2026-10-05). A NewBie version that
moved a whole block read as the words of every tag in it ("same words, moved:
character_2 … kasane_teto … pink_hair …"), and as an edit label the oldest
prompt was the whole XML clamped to its opening tags. `blockMove` finds each
prompt's top-level blocks as the critic does (`xmlBlocks`); its `XML_BLOCK` is
`critic_reorder._XML_BLOCK` with Python's Unicode `\w` written out as `[\p{L}\p{N}_]`
under the `u` flag, since JavaScript's `\w` is ASCII alone, and
`tests/test_critic_reorder.py` reads sheet.js and holds the two equal. Where a version
is the original's blocks in another order, nothing in them changed and the text
between them the same gap by gap, it names the move by tag - concatenating
that text had let loose text moved past a block pass as a block move. One
block taken out and put back elsewhere reads "`<general_tags>` first",
"`<character_1>` last" or "`<character_2>` before `<character_1>`", naming the
block that went earlier where either of two could be named; anything else
reads as the new order, cut between two tags within the row head's 80
characters (`tagLine`), where four blocks' order had been cut at its 80th
character, inside a tag. `versionEdit` says "same blocks, moved: …" (or "same
blocks, in the order …"), `promptLabels`' edit fallback labels such a prompt by
its move, and the oldest by its blocks' order where every other prompt is a
move. Any other edit reads as words, as before.

**Iterations on random seeds share no seed**, so the sheet can show them only
as a diagonal, or with the seed pinned to one image ("1 of 6 loaded"). Where
the Prompt axis varies and every image has a seed of its own (`ownSeeds`), the
note says so and points to Check's Compare on the same seeds (`OWN_SEEDS`,
ahead of a `sheetNotice`). Variable and variant grids share seeds by
construction and never get the clause (U72's recommendation item 5, which the
2026-10-04 record had called out of scope; built 2026-10-05).

## Grid preview and the per-step estimate

`POST /api/grid/preview` runs the same expansion without queueing it, so the
panel's count, its resolved axes and its error text are the run's own. Its
direct-render duration estimate is per *step* (`gallery_reports.step_cost_ms`), measured
from recent direct gallery rows at the same size **on the same checkpoint** (klein's step is half
of Turbo's at 1 MP): a grid's cells disagree about step count, so an average
image duration would misstate most of them. Refine estimates instead use full
per-image durations for the matching denoised step count, as described in
the refine section. No matching history, no estimate.

`grid.describe` names every axis, ones included, with an ASCII `x`: it is
the CLI's log line as well, and its tests hold it to that. The page trims
the factors of one but the seeds from the preview's first line and writes
the sign as `×`, since the axes line under it lists every value anyway. That
first line - the count, the time, the disk - is what decides whether to
start a night, so it reads at body size; the axes and notes stay hint text.
Under the prompts come the notes: cells already in the gallery (below), a
variable that never varies, Repeat above one (a grid steps its own seeds,
so the panel's `runs` is sent as 1 and the preview says it does not apply),
and a time budget. The budget line says how many seeds the minutes complete,
warning at none, as the server counts them (`GridPreview.budget_seeds`,
`grid.seeds_within_budget`) by the job's own stop rule: each cell priced as
the estimate prices it, in the order the job runs them, a cell starting while
the deadline has not passed and running to its end. It had divided the
budget by a seed's share of the estimate, which counts only the seeds that
end in time; the seed whose last cell starts before the deadline completes
past it, and on the card the line said 4 of 6 for a grid that made 5, its
fifth seed's last cell begun at 110 s of 120 (2026-10-05). It needs the
estimate, and is left out under Resume, whose seeds are no longer alike. The budget runs from submission (`jobs.submit_cells` sets the
deadline), so the line says the load counts too while the model is not
loaded, and the jobs ahead while the page is watching one: a Run grid
pressed then queues, and its minutes go on the wait as well.

The preview is asked again whenever something it reads moves, and not all
of that is a box: `setBusy` asks on a job's start and end (the gallery
holds a finished job's cells, and the queue ahead has emptied), and
`refreshStatus` when the model becomes ready or stops being so. Before,
only the boxes asked, and the two lines added for after a night - the
cells already in the gallery, the load - stayed as they were before it.
The first reply after the fold is opened by hand reveals Run grid again
(`gridRevealPending`): the reply lands under the pointer that opened the
fold, where the panel holds still (`panelResized`), and its lines pushed
Run grid back below the window the opening had just brought it into.

**The resolved prompts are a box of their own** (2026-10-05), between the axes
line and the notes (`promptListing`, `#grid-preview .prompts`), at most eight
lines tall and scrolling inside itself: a nested scroller, never overflow on
the panel, which would clip the Examples menu, the tag menu and the tips.
Printed whole, three resolved copies of NewBie's 29-line XML grew the preview
to about 2,260 px at QHD and put Run grid 2,200 px under the window. A prompt
written over lines is listed by the lines a `{variable}` is on, whole - a value
is read in its line, which is what the listing is for - and numbered
("prompts, line 4 of 29:"), so the rest is known to be there. That rests on
`grid.substitute` keeping every line in place: values are one line, and a
dropped one empties its line rather than removing it. A prompt that does not
line up, or whose variables are the negative's alone, is listed whole. The box
is an inline-block, so the preview's own line breaks (`pre-line`) around it
hold; `.totals` stays the first child, and the notes and the line `sheet.js`
appends still follow as text. Typing the variables after opening the fold by
hand still grows the preview under the held panel, by up to the box's eight
lines - the panel holding still under the pointer.

**The examples are the checkpoint's** (2026-10-05). The grid's Steps and CFG
boxes had shown Krea 2 Turbo's 6-12x2 and 0,0.5,1 on every checkpoint - far
outside Z-Anime's 28-50 steps and CFG 3-5, where CFG 0 turns guidance and the
negative off, and outside schnell's 4 steps. `gridExamples` (from
`applyDefaults`) gives a quarter of the profile's steps either side, and a
step of CFG either side of its CFG, clamped to the boxes:

- A CFG at or below a threshold that is CFG's own (Anima Turbo and ERNIE
  Turbo at 1, guidance off) sweeps upward, 1,2,3: 0,1,2 spent a cell on a copy
  of the profile's own image.
- A CFG-0 profile keeps 0,0.5,1, since one below is negative.
- Where the CFG box never reaches the model (`cfg_ignored`: the distilled
  klein, schnell, Qwen-Image) there is no example: 1,2,3 there was three
  copies of one image, and on klein a walk into the pipeline's warning above
  CFG 1. The profile says `cfg_ignored` exactly where its CFG tip opens
  "Ignored" (`test_cfg_ignored_is_exactly_where_the_cfg_tip_says_ignored`).

Mu's is `none,<profile mu>` where the profile has one, as before (mu is
Turbo-only). The tips stay syntax help.

Axis messages lead with the name of the box they came from - `Steps`,
`CFG`, `Mu`, `LoRA weight`, the Refine dialog's `Strength`, `Seed list` -
since the preview prints them under the whole fold. A range may be typed
with spaces (`6 - 12 x 2`): `parse_axis` closes up a `-` or an `x` between
two numbers before it splits on whitespace, which had cut `2 - 5` into a
`-` refused as a none token everywhere but Mu, where `0.8 - 1.2` silently
became three cells. Closed up, `0.8-1.2` is a range with no step, counted in
ones - the one value 0.8, which the trimmed totals would not show either -
so a range its ones never sweep is asked for its step (`needs a step, e.g.
0.8-1.2x0.1`), typed with the spaces or without. A `-` with no number on
both sides is still Mu's none, and `6..12` or `6 to 12` is answered with the
syntax this one uses.

## Resume keys on settings

**Resume keys on settings, not on the job.** `storage.completed_cells()` reads
back (prompt, negative prompt, steps, CFG, mu, seed) for the *distinct
prompts* a grid resolved to, at a size on a checkpoint **under a quantizer**
- a list, because a prompt grid renders several - and `grid.prune_done()`
subtracts them, so a night that died is continued by a *new* job and a cell
rendered by hand earlier counts too. A cell with `num_images > 1` is done
only when **every** per-image seed is present (`grid.cell_keys`): the rows
are per image, and one committed image of four used to mark the cell done.
Quantization is part of the reproduction set (same seed, different quantizer,
different pixels), so it is recorded on the row and in the PNG and scopes the
lookup; rows from before the column match any quantizer. Both sides round to
the six decimals `parse_axis` produces, so a float that has been through
SQLite's REAL still matches. Resume **requires an explicit seed** — a random
base would match none of the run it means to continue and would silently
render the whole grid again — and an entirely-finished grid answers 409
rather than queueing a job that reports "done" having drawn nothing. The seed
is the one setting the client is never given back by the images: every row
carries a *derived* seed (`base + seed_index * num_images`, plus the batch
offset), so `JobAccepted` and `Job.snapshot()` publish the base, the UI
writes it into a blank Seed box after `Run grid`, and the recovery
instruction everywhere says the *oldest* image of the run, never "any".

Without Resume, the preview still asks the same lookup and reports what it
found as `GridPreview.in_gallery`, dropping nothing: Run grid again after a
night with a failed cell rendered the whole night again, with nothing
before the press saying that Resume would not. Only where the seeds are
fixed - an explicit base or a list - since a random base matches nothing
and a lookup for it would be a scan per keystroke for a zero.

## job_id, run_id and _migrate

Every image records the `job_id` that made it - a browser job's id, or one
minted per CLI invocation - **and the `run_id` of the grid it belongs to**
(`grid.run_id`: a hash of the checkpoint, quantization, size, batch and every
cell key of the *full* grid, computed in `api_generate.expand_grid` and `generate.py`
before resume prunes anything). A resumed night therefore lands under a second
job id but the same run id, and the gallery's `run:` filter - which is what
the card's ⊞ button sets - shows the whole matrix where `job:` shows one
submission. Rows written before either column existed have NULL, which the
UI tolerates - and `_migrate()` adds the columns to an older database,
because `CREATE TABLE IF NOT EXISTS` leaves an existing table exactly as it
was. `_row_to_record` reads every column **unguarded**: rows only ever come
out of a connection `_connect` opened, and that ran `_migrate` first. A new
column needs a `_migrate` entry, not a `"x" in row.keys()` read guard - the
guard would let the SELECT succeed and every INSERT after it fail.

## The lightbox zooms

**The lightbox zooms, and the view carries across cells.** `lbView` in
`web/lightbox.js` is a scale relative to fit plus a pan, never below fit or
1:1, whichever is smaller, kept across records whose shape agrees, so the
arrows compare the identical crop at the identical magnification down a
steps or mu axis - and, since 2026-10-04, between a pass and the source it
was scaled up from, which share a shape and not a size. "Agrees" is a ratio
within 1% (`lbView.aspect`), not a key: the 16-pixel grid puts 2048x1152
and 1360x768 apart in the third decimal, where both rounded keys tried for
it broke; a carried scale is clamped to the new record's own floor and
ceiling. The floor used to be fit: on a 2560x1305 screen every preset up to
about 2 MP fits larger than 1:1 (a 1 MP square at 124%), so the 1 key, the
1:1 button and the double-click all did nothing on a default-size image,
and a sharpening sweep was judged through a 1.1-1.65x upscale. The zoom
line tests 1:1 first, then fit, then a plain percentage - a wheel stop
between the two is neither - and a double-click goes back to fit from any
other scale, a 1:1 below fit included. The image is laid out with explicit
width/height/left/top inside a `flex: 1 1 auto; min-height: 0` stage - not a
CSS transform, which can composite a fit-size texture and scale it - and
1:1 divides by `devicePixelRatio`, or Windows scaling makes it a 1.25x
resample, exactly the smoothing a sharpening sweep is judging. The wheel
listener is non-passive; the zoom keys sit below the `input, textarea`
guard, and below one for modifiers, so Ctrl+F and Ctrl+plus stay the
browser's. A click on the stage beside the image closes the lightbox only
at or below fit: zoomed, the stage is a pan surface and only Escape or the
cross close it. Below fit that close waits out a double-click
(`lbCloseTimer`): the picture is smaller there, and the double-click that
gives fit back can start on the backdrop. Where a press landed is read on
`pointerdown` and kept in `lbDrag`, and the double-click handler is the
stage's and hit-tests the picture (`elementFromPoint`, which honours the
comparison's clip-paths): the stage takes pointer capture on press, and from
then on every event of that pointer - `pointerup`, the `click` and the
`dblclick` made from it - targets the stage whether the mouse is over the
image or not. Testing `e.target` on the way up closed the lightbox on a
single click of the image, and a `dblclick` listener on the image never
heard one; `tools/smoke.json` clicks and double-clicks the image for both.

**A pass is compared with its source in place.** The only question a pass
asks is whether it improved its source, on detail, at a strength of 0.1,
and no route showed the two at one crop: the source was seldom next to its
pass in a strip, a step between them reset the zoom, and the viewer had
replaced it with the pass. On a record with `refine`, S (or Compare with
source, with its two mode buttons) lays the source under the pass and shows
it left of a divider (swipe), or beside it (side by side: each half fitted,
showing the same crop, the halves inside the arrows - `lbNavRoom`, measured
off the circle as drawn, once per `layoutLightbox` and before its writes,
since the wide tier's circle is 52 px and a constant written for the 40 px one
put the halves under it - whose circles had sat over the source's left edge
and the pass's right one), on the one `lbView`, through the same explicit
geometry. The source is fetched by id once a session (`lbSources`; a strip
that holds it already is looked in first) and drawn as `refine.fitted`
cropped it - `refineKept`'s arithmetic places its whole picture so the kept
part lands on the pass's box, and a clip-path cuts it to that box, its half
and the divider - so a reshaped pass is compared with what it kept.
Switching between the one view and the halves keeps the magnification and
the centre (fit stays fit), so a 1:1 stays a 1:1. A step to another image
ends it, its mode kept; a re-render of the same image (a star, a status)
does not. Where the source has gone (`refine_missing`, or the fetch's
"unknown image") the controls stay, `aria-disabled`, saying so. The divider
is dragged (a press on it is not a pan) or moved by its arrow keys, which it
takes before the lightbox's walk does; hidden with the focus in it - S
pressed there - it hands the focus to the toggle, not to the top of the
dialog.

**Comparing several passes, the arrows swap passes.** A Compare 3 run makes
three passes of one source, and the question is which of them improved it
most. The arrows moved the divider (the drag had left the focus on it) or
walked to the next image and ended the comparison, so each pass took a walk,
an S and the divider set again. Now, comparing, ← → go to the next pass of
the same source in the strip (`stepPass`), wrapping round and skipping the
source, with the comparison, its mode, the divider's place and the focus
carried, so the three are flipped at one crop. The divider keeps keys of its
own: Shift+← → from anywhere (`nudgeSplit`), and on the focused slider
Shift+← →, PageUp/PageDown, Home and End, its `aria-keyshortcuts` and title
naming them, since a screen reader user on a `role=slider` expects the
arrows there and is told otherwise. With one pass in the strip nothing
changed: the arrows on the divider move it and elsewhere walk. Shift+← → is
2% wherever it is pressed: it was first 10% on the focused divider with one
pass and 2% off it, so the same keys moved it by five times as much after a
drag as after a click elsewhere. PageUp/PageDown is the bigger step.

The keys line named S on a pass, and only there (`lbKeysLine`; on every
image since the comparison across CFG, below), and the toggle
carries S by the shortcut hint's own rule (the `::after` Generate's
Ctrl+Enter uses): with S only in its title and `aria-keyshortcuts`, and the
always-visible keys line not mentioning it, it was found only by hovering
(U3's letter). Each key is bound to its action - no-break spaces, a word
joiner in +/− - so the line breaks only after a ·: the pass's longer line had
split "+/" from "−" at 1400 px. The Refine dialog's passes strip and a pass's
Compare with source in the result column (The results viewer) open the
lightbox with the comparison on (`toggleCompare` after the open), since
judging the pass against its source is what both are opened for.

**Any image is compared across CFG** (2026-10-10). The swipe was built for
one question, whether a pass improved its source, and only that: the
controls existed on a pass alone, the partner was always `rec.refine`, the
labels said source and pass, and a step to any other image ended it. A CFG
sweep asks the other question the swipe is good at - what did CFG change,
at one crop - and its cells were walked one image at a time, the zoom kept
but the eye left to remember the last picture. Now `lbCompare.kind` is
`'source'` or `'cfg'`, and an image is compared with another that differs
from it in CFG alone (`web/lightbox-cfg.js`): the sheet's own test of "same
settings" with CFG and True CFG skipped (`sameExcept`, so the two cannot
drift), split by True CFG where it is a number of its own, one image a CFG
value, sorted. That ladder is presentation over records the server already
decided, so there is no route: the page's own strips first, then a
gallery search 200 ms after the image lands, as the passes' count is asked,
read on by its id cursor for up to five pages.
The search is narrowed to the seed, size, steps, refine source, mu and
LoRA, in the checkpoint's exact scope: a seed reused for hundreds of
prompts would otherwise fill the page with images the filter throws away,
and a full page says the ladder "may be incomplete". A seed past 2**53 (a
grid's `seed + i` at the top of the range) cannot be written exactly from a
JavaScript number, so the run is asked for instead; separate Generates are
reached through the seed, which is why the run is not asked for always.

The lower CFG is always on the left. The comparison first kept the image in
view under the partner, and stepping up the ladder put the higher value on
the right while the lowest image, its partner the one above, put the
higher value on the left: a reader had to check the tags to know which
side was which. The layers trade places instead (`lbSwapped`): the image in
view goes in the first port, and the partner's layer is cut to the right of
the divider. The tags, the divider's name and its value say the CFG.

The arrows step only within what they walk. Images the gallery's answer
brought are partners - Against offers them, the default neighbour may be
one - but never a step, and `lbList` is never replaced or added to: a
splice into the strip, or a copy of it, changed what ← → walked after the
comparison ended and what the counter said. At an end the press stays and
says why rather than walking off and ending it: the ladder is ordered, and
passes wrap only because they are not. Where the strip holds no other value
either way the arrows walk, as with a lone pass. The sheet's Swipe CFG
therefore opens the lightbox on the cell's own ladder as its strip, so its
whole row (or column) is a step away; it is drawn only where CFG is laid
out, as the axis pickers list only what varies.

The box is on every image from 901 px, its controls held with the reason
while the gallery is asked ("Looking for other CFG values of #N…") and
where there is nothing ("No other CFG value of #N at this seed, size and
settings"), and S is on the keys line everywhere, since it does something
everywhere - compares, or says why not. A box that appeared when the answer
came would have grown the column under the pointer; with the box always
there, its note first and the column pinned to the bottom, an answer, a
note or a press moves no button. Under 901 px the box is a row under the
stage, paid for by the picture and the facts' letterbox: a square picture
at 600x900 kept 77 px of letterbox where it had 126, and lost its last
facts. There the box is drawn only where the page's own records give the
image a neighbour, decided when the image is drawn and kept until another
is, never on the answer.

A review after it landed found four things the first version said or kept
wrongly (2026-10-10). Swipe and Side by side, now shown on every image, were
held on a plain Generate with no CFG neighbour with no reason in their
titles - Swipe had none, Side by side kept a description of a comparison
that could not start - so they take the held reason as their title and get
their own back when pressable. The ladder was one page of 200, and
"No other CFG value" was said outright where that page was full: a seed
kept fixed across a few hundred prompts at one size filled it with the
other prompts, and an image whose CFG neighbour was older was told it had
none. The page now reads on by the id cursor for up to five pages, and
where even that is cut short the held reason says how far it looked ("in
the newest K at this seed and size: more may be in the gallery"); the
search was not narrowed by prompt words, which the grammar matches by FTS
and would not match exactly. An answer on its way was dropped when a job
landed a value it would have held only if it was still the latest asked;
walking on to another seed let it be cached for the session, so every
query on its way is tracked and marked stale (`lbLadderFlights`). And the
sheet's Swipe CFG was held by a grouping that left True CFG out where the
press's ladder keeps it: a CFG x True CFG grid with cells missing showed a
pressable button whose press said its own pressable title as the error.
The hold now asks the press's question within each group.

The plan had a second group, images with the same seed whose CFG differs
and which also differ in steps or mu, offered under Against. It was left
out: the narrowed search does not bring them, so the group would have
depended on what the page happened to hold, and a pair that differs in two
settings is not a CFG comparison.

**The zoom is said once, the readouts only when they change** (2026-10-05).
`#lb-zoom` and `#lb-count` were polite live regions rewritten on every layout
and render: ten wheel notches 60-120 ms apart queued ten announcements
("151%" … "800%"), and F rewrote "fit · 124%" and "10/19 · #12" with
themselves, which a screen reader says again. Both are written only when
their words change (`writeZoom`). The readout on screen is `aria-hidden`;
what is said is its visually hidden copy `#lb-zoom-said`, written 400 ms after
the last change - a wheel's burst, a resize, a step to an image of another
size - so a burst ends in one announcement, or at once for a zoom asked for
by a key, the Fit or 1:1 button or a double-click (`sayZoom`), one per press.
`closeLightbox` empties all three, so a fresh open still says its position
and zoom.

**Side by side stacks where that shows the picture clearly larger.** On a
portrait window under 901 px, left and right halves gave a 1024x1024 pass and
its source 213 px each of a 549x626 stage ("fit · 21%"), and a 1344x768 pass
213x121; one above the other gives them 309 and 433x247. A first version
stacked wherever the stage was taller than wide, which shrank a portrait pass
from 213x372 to 177x309, and decided on the stage the facts had left, so a
landscape pass's facts were given the room of stacked halves, the stage left
was wider than tall, and the halves were drawn left and right at 213 px over a
blank band. `lbStacked(rec)` decides once per layout, on the stage with the
facts at six lines (`lbBaseStage`), before `lbFactsRoom` takes the letterbox,
and both use the answer. It stacks only where the window is under 901 px or
the stage is taller than wide, and only where stacked halves fit the picture
at least an eighth larger than left and right (`LB_STACK_GAIN`). The gain is
asked because stacked halves take the stage's height, which is the facts'
letterbox: with "larger wins", a square pass at 800x900 stacked to gain 15 px
a side (313 to 328) and lost every fact under Size (room 351 to 0), and a 7:4
pass at 900x700 gained 36 px for the same loss. A landscape stage from 901 px
is always left and right: a 16:9 pass at 1400x900 would fit 749x428 stacked
against 464x265, but that would change the QHD layout side by side was
designed for. The split can therefore differ between the records of one walk
(at 600x900 a square or landscape pass stacks, a portrait one does not). Each
half is the stage's width less the arrows' room and half its height less the
gap, and every reader of the ports takes a port's `y` as well as its `x`: the
picture's box, the clip to its half, the names (the pass's at its own half's
top), and the wheel's point (`stagePoint` takes the half whose centre is
nearer). The mode keeps its name, and its title says which way a press puts
the halves, before the press as well as after it (`lbStacked`, decided for
any pass, compared or not): before the press it said "left of the pass"
where the press stacked them (2026-10-05).

The layout also runs again on the stage's own resize (a ResizeObserver in
app.js), not only the window's. Under 901 px a line in `#lb-error` (Use
seed's, a delete's) is a row of its own that took 30 px from the stage, and
the picture laid out before kept its height, its foot cut off under a readout
still saying "fit". The observer cannot loop: the layout writes the facts'
room only when it changes (`lbFactsRoom`), from the stage as it would be with
the facts at six lines.

**The controls are beside the stage, not under it.** The dialog is a row:
`.lb-main` (the stage and its arrows) and a `.lb-side` column (280px;
340-400px from 1800 px, `--lb-side-w` with the wide tier's other widths) with
the cross and the counter, the prompt and the facts, the error slot, the
comparison, the keys line, the zoom row and the buttons. They used to sit
under the stage as a meta line and a row of buttons, which cost it around
150px of height on a 900px screen - and the screen is wider than it is
tall while most images are not, so height was the axis nearly every fit was
bound by; a square or portrait picture is still bound by the height, so the
wider column costs it nothing, and a 16:9 one about 3%. In the column the
prompt and facts scroll (`flex: 1 1 auto; min-height: 0`) and the controls
are pinned to the bottom, so Star and Delete are where the mouse left them
from one image to the next however long the prompt is; `#lb-close` carries
`autofocus`, or `showModal()` would focus the first control in the tree,
which is now an arrow. The facts are a list (`factsList`, a `dl.facts` any
view of one image can use): a row a fact, where a 12 px line of commas
buried the seed and left out the negative prompt its True CFG applied to,
the date and the run. The source, each reference (numbered as the panel's
chips are, with "Use these references") and the run are buttons: the first
two open that image with this one after it, so → comes back; the run opens
as the card's ⊞ does. In the lightbox the Seed row also has Use seed
(`factsList(rec, { useSeed: true })`, 2026-10-05): the lightbox is where
older images' seeds are browsed, and keeping one had meant a full Reuse or a
copy by hand. It is said in the lightbox's own line, not the panel's `#error`,
which is under the dialog and may hold a job's outcome, and taken back once
another image is drawn (`lbSeedNote`): stepping on had left it naming a seed
the picture did not have. The result column's facts leave it out, its row
having one. Use these references asks for every id again, as
Reuse does, and leaves out and names any the gallery no longer has: put in
the panel, a deleted one was an empty chip and a refused Generate. It is
`aria-disabled` with the reason where the checkpoint takes no references or
every one has gone, and a render of the same facts brings that up to date
(`syncUseReferences`), since a status after a load can change the first. A source's passes are linked as the gallery's
`refine:` search once the gallery says there are any. Each fact the search
grammar can say ends in a ⌕ that adds it to the gallery search, and the
prompt has "Search for this prompt" (2026-10-08): finding the rest of a seed
or a prompt had been a copy into the box by hand. They add to the box rather
than replace it, so prompt then seed narrows twice; a second value for a key
replaces the first, since two seeds AND to nothing. The prompt is one value
the same way: its press replaces the box's free text and keeps the filters,
since adding a second prompt's words to the first's ANDed both prompts and
found nothing (review, 2026-10-08). The prompt goes in as its
words - the grammar has no phrase, and FTS matches every word in any order,
which for a whole prompt is near enough to the same prompt - leaving out a
word that would read as a filter. The negative prompt, references and
variables have no key and no button. `size:` was added to the grammar for
the Size row. The press closes the lightbox, as the passes link does: the
results are the gallery's. The one-line
`settingsLine` stays as the dialog's description (`#lb-desc`, hidden) and
the delete confirm's. The counter names the image ("3/144 · #57") and
counts the gallery by the server's total, not the pages loaded. A picture
that will not load says why in the lightbox's own line, the record asked
for to tell an outage from a deletion.

**Under 901 px the facts take the letterbox.** The column drops under the
stage as a wrapping row, and `#lb-meta` was a six-line scroller (84 px): it
showed Model and Size and hid the seed, the date and the run, while a square
picture at 600 px wide left 126 px of letterbox above and below it. Fixed
sizes (`min-height: 50vh` on the stage, a `100vh - 100vw` cap on the facts)
changed nothing at 800x900 or 900x700 and too little at 600. The letterbox is
what is free, so it is measured: `lbFactsRoom`, first in `layoutLightbox`
after the split is decided, sets `--lb-room` to the stage's height less the
height the picture (or the stacked halves and their gap) is fitted to, both as
they would be with the facts at six lines. A tall picture therefore keeps the
whole stage, and nothing carries over from the image before. `.lb-meta`'s
`max-height` is `calc(6em + var(--lb-room))`, and `lbFactsRoom` writes only
when the room changes, so a wheel step still lays the page out once. The
prompt there is four lines, inside a box at least six tall, so the facts follow
it in view.

**The arrows are strips, and a close returns to the walk.** The arrows are
positioned in `.lb-main`, over the stage's edges but *not* inside the
stage: the stage takes pointer capture on press, so a press on a child of
it would have been a pan, and at fit a close, rather than the button's
click. A 40 px circle there left 560 px of backdrop each side of a portrait
picture at QHD, and a press 10 px off it closed the lightbox; so at fit,
with backdrop beside the picture, each arrow is a strip as wide as that
backdrop less a margin (`--lb-gutter`: from the circle's reach less 2 px - 48
for the 40 px circle, 60 for the wide tier's 52 - to 160 px, so a strip's
circle never reaches onto the picture) over the middle 60% of the height, its
circle where it always was. The circle is `--lb-nav`, 40 px, and 52 from
1800 px (the review plan's step 14, built 2026-10-05). The corners stay backdrop,
which `tools/smoke.json`'s close at [30, 30] relies on. Anywhere else the
arrows are their circles again: zoomed in, a strip would take the pan
surface's drags, and below fit the double-click that gives fit back - a
strip there stepped on its first click instead. So is a disabled arrow (the
first image's ‹, the last's ›): a disabled button takes the press, and as a
strip it left that side's backdrop closing nothing. The wheel is the
stage's parent's, so it zooms over a strip as over the stage rather than
scrolling the page behind the dialog. A press on the stage within half a
second of one on an arrow is the second half of a double-click on the
arrow (`lbAfterArrow`), neither a close nor a 1:1: the step can take the
strip from under the pointer when the next picture fills the width. The
arrows' own clicks are not filtered by click count: the guard a later round
of the 2026-10-04 review suggested (U108, ignoring `e.detail > 1`) was not
built, since it would have cost a quick walk every other press, and
`lbAfterArrow` already handles the double-click it was for. In use
(2026-10-10 testing notes) the strips themselves showed: a 5% white tint on
hover and the focus ring drawn round the whole strip painted the hit area
as a box over the backdrop beside the picture. Now nothing of a strip is
painted; hover lights the circle and keyboard focus rings the circle. The
stage's own title ("Scroll to zoom, drag to pan, double-click for 1:1")
went the same way - a tooltip over the picture being judged - and is the
zoom readout's title instead; the keys line already names the keys. The
same notes asked for megapixels after the size, so the lightbox's Size
row reads "1536×1024 (1.6 MP)", spaced as the size hints write MP; the
viewer's column, which shares `factsList`, keeps the bare size (`megapixels` is the lightbox's
option alone). The backdrop shows the default cursor, the
picture the zoom one. A close of
the user's own - Escape, the cross, a click on the backdrop or beside the
picture (`dismissLightbox`) - focuses the image the walk ended on, its card,
tile or sheet cell, which scrolls it into view - and where no strip holds it
any more (the server came back under another checkpoint and the gallery is
another scope, or it was deleted elsewhere, and the opener went with it),
the grid's current card, else its first, else the search box - each only
where it is shown (`isShown`, state.js: `checkVisibility` with
`visibilityProperty`, since a column hidden by `visibility` still reports its
cards as shown without it; The gallery beside the viewer) - else the viewer's
first image, else Generate, never `<body>`. Its Escape is the lightbox's
before the Examples and Tags menus' (app.js's keydown): the Tags helper stays
open when focus leaves it by keyboard, and its branch took the Escape of a
lightbox opened over it - the dialog's own cancel closed the lightbox without
`dismissLightbox`, and focus went back to the opener. The menu stays open
under the page until the next Escape. The dialog's `close` event is queued,
not fired by `close()`, so its listener resets only a lightbox still closed:
in smoke-delete's lightbox of one, reopened to delete from, a close followed
at once by an open had the late event shut the new lightbox under the delete
in flight, and the smoke read its count before the DELETE answered - its
intermittent pair of total checks. A close that hands the page to something else - Refine's dialog,
Reuse's panel, a delete, Show this run - is `closeLightbox` alone, since a
focused card would scroll the page back from where those took it. The Refine dialog closes the lightbox for itself;
closed without running, it gives the lightbox back on the same image of the
same strip, at the same view and comparison (`refineReturn`, resolved by
id, since a job landing meanwhile reloads the gallery into a new array; the
comparison put back before the view, whose scale is relative to the fit of
half the stage side by side - on the whole of it a 1:1 came back at 166%),
unless its passes row has opened the lightbox on a pass already. Run, it
puts the focus on the source's card where that is shown (`isShown`), else
Generate, without moving the page
(`refineClosed`): the dialog hands focus back to the lightbox's Refine,
closed with the lightbox, which left it on `<body>`. A run from a dialog a
script opened - Reuse of a pass - had nothing to hand focus back to, nor
one whose opener the pass took away: on the next frame, focus still on
`<body>` goes to Generate, or to New image where the strip is out of
sight. And a job's first frame, clearing the strip, puts focus that was in
it - a result's Refine… the dialog handed it back to - on Generate
(`clearResults`). Where the page itself moves focus off the control that
started a job this way - the dialog closing on its run, the strip a first
frame empties - it hands that control on (`handJobFocus`), so the sweep's
sheet can still land focus where the press was (The contact sheet). While the pass's POST
is out the dialog holds (`refineSending`): its button reads "Sending…", and
Escape (the dialog's `cancel`, prevented), Cancel and the cross wait for the
answer. Since the dialog closes only once the server has the pass (Refine
closes on acceptance), an Escape in that window had closed it, given the
lightbox back as though nothing ran, and saved the boxes as unsent, while
the pass rendered behind it. A close the browser will not hold - a second
Escape with no other press between - counts as the run it most likely is,
and a refusal arriving after it is said in the panel's line. The lightbox's
event wiring is app.js's, as every file's is: lightbox.js holds the named
handlers (`lbImageFailed`, `lbWheel`, `lbPointerDown`...; `lbSplitKey` is
lightbox-compare.js's since 2026-10-11),
and nothing in it runs at load. Below 900px wide the column
drops under the stage as a wrapping row, without the keys line. The
`tools/smoke.json` grid is 1152x896 for the same reason the layout changed:
at the driver's 1400x900 a 1024x768 image fits at 104%, and its 1:1 is
below fit; `tools/smoke-lightbox.json` has the rest.

## The sheet exports client-side

**The sheet exports client-side.** `renderSheet` leaves its matrix in
`sheetLayout`; `buildSheetCanvas` draws the same axes, pins and labels onto
a canvas from the *thumbnails* on a fitted sheet (cells shrink so the longer
edge stays under 8000 px) - a filled one (Fill width, The contact sheet) at
the larger of its cells' device pixels on screen and the thumbnail's width,
from the PNG where that is wider than the thumbnail, as the sheet's `srcset`
does (at its CSS width a 1400 px window's filled export came out at 1092 px
against the fitted 1716, and at 125% it lost a fifth of what the screen
showed) - with a caption naming the search, the pins, the checkpoint, the
run and the time, and `exportSheet` downloads it. Not a server route: that
would duplicate a layout contract the client already holds, the
non-round-trippable prompt labels especially.

The caption is what the image has to say on its own later: the prompt (the
run's one prompt, else the first image's, with a variable run's values as
declared - `size = big | small; ground = grass | dirt` - since a template is
not a prompt any cell ran), the Fixed line, and the time the run was made,
the oldest image's, where it used to print the moment of the export. The file
is named by the run and the pins (`sheet-<run>-cfg-2_seed-42.png`), since
every slice of a run used to download under one name; a pin's value ending in
a mark ("none (1.79)", a clipped label's "…") had left a hyphen before the next
pin or the extension, which `sheetFileName` drops
(`sheet-<run>-mu-none-1.79.png`).

Export PNG is the slice on screen. **Export all**, beside it wherever
something is pinned (2026-10-05; the review's U102, called optional and left
out the day before), takes every pinned slice that has images (`sliceCells`,
the one cell loop `renderSheet` uses too) and stacks them in one image, each
under a title naming its pins, at the cell width one slice would take or less,
down to the 96 px floor, so the stack stays within 8,000 px (`stackedCellWidth`,
`EXPORT_MAX_EDGE`); named `sheet-<run>-every-<axes>.png`. Past that it saves one
image per slice, each named as Export PNG names it, with -2 where sanitising
made two names alike ('red sky' and 'red-sky'); that fallback stops at 24
downloads (`EXPORT_FILES_MAX`), since the browser asks before several, and the
line under the gallery head says what happened. Both buttons are held,
aria-disabled with the reason, while an export draws: the slices are drawn one
after another, and a second press drew them again. Every export reads a
snapshot (`exportLayout`): the layout, its axes and the words its labels are
made of (`sheetWords`: the prompts' short labels and the adapter). Read live,
a sheet redrawn on another run halfway through the per-slice files threw on
the next slice's caption, unsaid. A failed export is now said under the
gallery head ("The sheet could not be exported: …"); the smokes stand in for
`saveCanvas`. A
wrapped single row (see "The contact sheet") wraps here too, `ceil(√n)`
across with each cell captioned under it - drawn as one row, the 96 px floor
beat the edge limit and a 120-seed run came out 12,254 px wide - and a refine
sweep's source is drawn as the first column, headed `Source #<id>`.

## Pause

**Pause is cooperative, like cancel, and lands between cells.**
`POST /api/jobs/{id}/pause?value=` sets `pause_requested`; `_generate`
answers at the top of the next cell by setting `paused` (a worker-written
acknowledgement - the UI flips its label on *that*, never on the click,
because a denoise may still be running), calls `backend.free_cache()`
(torch's `empty_cache`: under model offload nearly all the card holds
between cells is allocator reserve) and polls `pause_requested` every
quarter second. Polled, never `Event.wait()`: a bare wait deadlocks
`shutdown()`, which cancels and then joins the thread, and makes `cancel()`
a no-op on a paused job. `busy()` is unchanged, so unload still 409s - the
weights are needed for the resume - which is the cheap half of the design;
a full unload-while-paused only matters for `LOCAL_DIFFUSION_OFFLOAD=none`.

The click is answered at once all the same: on the card a cell can take tens
of seconds, and a button still reading Pause - its "pausing after this cell…"
overwritten by the next step - invited a second press or Cancel. It reads
"Pausing…" until the acknowledgement, which "claims nothing"; a press on it
withdraws the request, and each status frame's `pause_requested` sets it
again, so a reattached page shows it - except a frame that contradicts the
page's own press while its POST is still out (`pauseAsking`): that frame
predates the request. A press at a job's start lands before the stream's
opening snapshot, which read Pause again over the pause on its way; the
next press then asked for the pause a second time rather than withdrawing
it, and the job stayed paused (smoke-jobs' tab B, intermittently). The
label is the state, with no `aria-pressed` beside it: a toggle's label
must not change, and the first cut, pressed on both "Pausing…" and "Resume", was read as "Resume, pressed" - as if
resumed. That is the page's one rule for buttons with a state: a button whose
words say the state (Pause / Pausing… / Resume) has no `aria-pressed`, and a
toggle with `aria-pressed` (a star, a card's ❐) keeps one name in every state,
the state carried by `aria-pressed` and its glyph, and in words by its title
(The star is a button). The cell line says "pausing
after this image…", and once paused "Paused after image 1/3 · GPU handed
back", while the counter beside the bar stays the counter - written there, the
long text squeezed the bar to 5 px. A running job of one cell has no boundary
to stop at (a queued one does: `_generate` checks before cell 0), so its Pause
is `aria-disabled` and a press says why. A plain job's units are images, or
batches where a repeat makes several (`unitWord`, from the snapshot's `grid`
and `planned_images`): "cell" is a grid's word, and "run 2/3" was a fourth
meaning of run.

## The browser's queue

**The browser can hold a queue.** `startJob` no longer refuses while busy:
a submission during a job is POSTed and its id kept in `queuedIds`, the
buttons are held (`aria-disabled`) only for the POST itself as the
double-submit guard, and `watchNext()` attaches to the next job when the
watched stream ends. `resumeActiveJob` attaches to the running job, else
the *oldest* queued (`active[0]` - `active()` is newest last), never the
newest. Cancel acts on the watched job; "Cancel queued" on the rest.

**One watch per job** (2026-10-05). When a job ends the page asks
`/api/jobs` which job to watch next. `watchNext` checked `busy` before that
request and never after it, so a Generate pressed while the request was out
took the page and streamed its own job, and the listing then found the same
job and streamed it again: two tiles of one image, a grid's Retry or Resume
lost to the second finish line, and the next job watched twice too. A checker
reproduced it in 8 of 60 back-to-back submissions, and `smoke-grid` had spaced
its two renders 300 ms apart to step around it (the 2026-10-04 record left it
as it was). `resumeActiveJob` had the same shape at load. `startJob` also
decided whether to queue before its POST, so when the watched job ended while
the POST was out, either the new job ran unwatched - no progress, no finish
line, its id later listed as "Next:" behind another job - or the listing
adopted it and it was also its own "Next:" row, whose × cancelled the running
job (a race the record had missed).

Both listings now go through `adoptListedJob`. While it is out it is
published as `jobListing`; it checks `busy` again once it answers, and opens
the adopted job's stream itself, so anything waiting on it finds the job
watched whatever order their continuations run in, and it catches what it
throws, as `resumeActiveJob`'s own catch did at load. A submission made
meanwhile queues instead of taking the page, and after its POST waits for any
listing still out. A job the listing made the watched one is awaited through
its stream (`jobStreams`), so `await startJob(...)` still resolves when the
job ends: a first fix returned at the POST's answer, and 22 of 30
back-to-back awaited pairs came back with the job still running. A job
watched to its end before its answer landed is forgotten (`forgetJob`).
Otherwise `startJob` takes the page if nothing is watched, and else queues an
id that is not queued already. No id is streamed twice: `watchedJobs` holds
every id a stream was opened for. The listing decides, rather than a bare
`busy` re-check: with a re-check alone, a Generate during the listing would
take the page while a job this page had queued earlier ran, and that job
would read "Next:". The stream reads a pass's stand-in again at each progress
event until it is there, because the listing can adopt a submission before its
POST's answer registers it (The live preview). `announceLastRun` returns while
`busy`, so its banner does not come up over a job started while its requests
were out. `smoke-jobs` holds each ordering open with gated fetches and asserts
that the race window was actually reached; `smoke-grid` no longer spaces its
renders.

**Every job is named.** With a queue the prompt box held another job's text,
"1 more job queued behind this one" named none, and Cancel stopped the job the
page did not show. The snapshot carries what the job is - `prompt` (its first
120 characters), `refine`, `width`, `height`, `planned_images`, `grid`, and a
grid's own prompt, its variables unfilled (`prompt_template`, from
`/api/grid`) - so a page that did not submit it can name it too. A grid is
named by that: by its first cell's prompt, the line named a cell that had
long finished for the rest of the night ("grid · a fox, {mood}", Cancel's
title "Stop this grid"; a re-render, whose cells are its records', is "re-
render"). A "Now:" line over the bars and Cancel's title name the watched one,
as "3 images · 1024×1024 · a fox in the snow": the count and the size first,
since they tell two jobs apart and the line is cut at its end - put last, the
count was the first thing cut, even at 2560 - while the prompt is whole in the
title. The line is one line, cut to the panel's width, and Pause and Cancel sit at its end - wrapped to two lines, with the
buttons on a row of their own, it pushed Cancel and the queue below a 1040 px
window. Under the bars a row per queued job (a
submission's own words until `nameJob` has the server's: "1 image ·
1536×1536 · refine of #10", in the Now line's order) carries an × that cancels that job alone, `aria-disabled`
while its POST runs, focus moving to the next row's; one waiting is a single
"Next:" row with no heading, and Cancel queued stays for two or more.
An XML prompt opens with the same tags in every job (`<character_1> <n>…</n>
<gender>…`), so jobs and runs that differed in an attribute read alike in the
Now line, the queue rows, Cancel's title and the Runs picker, and a queued
row's × did not say which job it cancelled. They are named by their words
instead (2026-10-05): `promptGist` drops `<name>` and `</name>` tags before
`snippet` cuts, and the Runs label reads one line before `clip`. It drops
nothing else, so prose, Qwen's `<|…|>` control text and `<lora:…>` forms stay,
and a prompt of nothing but tags keeps them. The Now line's title keeps the
prompt as written, markup and all, as far as the snapshot's 120 characters go.
`startJob`'s `onAccepted(jobId)` runs once the server has taken a
job, before `startJob` watches it (a listing between jobs may have adopted
it already: One watch per job) - the Refine dialog closes there, so a refusal at
submit leaves it open with the reason in its own line - and its `placeholder`
is what the viewer shows of that job until a frame arrives (The live preview),
keyed to the job so a pass queued behind another cannot paint over that one.

**The job is in view wherever the eye is.** The tab's title follows it ("54%
· image 2/4", "⏸ Paused", "✓ Finished 4 images" until the tab is looked at
again - focus, a press or a key), and the topbar carries the same counters in
one `aria-hidden` line, with a 2 px line along its foot: the panel's bar was
113 px of 11 px text a column from the image, and at 125% below the fold.
Pause and Cancel sit on the Now line, so the bars take the panel's width. The
cell line is inside the progress block and the finish line is its own line
after it (A cell's failure is the cell's); either takes no room while empty,
where an empty one was a 47 px hole between the progress and the queue.
A job of this page's hides the morning-after banner without dismissing it
(The morning after). Under 700 px a load's state did not fit the bar: the
status and the free space wrapped to two lines each, and the mirror was cut to
"loading weights · …" - the part cut being the clock, which tells a slow load
from a hang, while in one column the mirror is the job's only readout in view.
At 700 px and below the free space is hidden while a job is mirrored
(`.topbar:has(#job-mirror:not([hidden])) .vram`), the mirror's
`margin-right: auto` keeping Load and Unload at the bar's end; wider,
everything fits and it stays.

**A finished job joins the gallery at its head.** It used to reset the
gallery: a user browsing old images during the wait was scrolled to the top,
the pages scrolled in dropped, at every job's end. `prependFinished(made)` asks
for the newest page under the same search, Starred tick and scope - `made` plus a
page, the sheet's ceiling at most - once any page in flight has landed, drops
the answer if the filters changed meanwhile, and puts the records newer than
the head in front, in place (an open lightbox on the gallery keeps its image);
the paging anchor, the last record, is untouched, and scroll anchoring keeps
the rows in view. The new cards get the day's lines as `loadGallery` draws
them, and the old head's day is read off the grid's first card itself
(`data-made`, which a tombstone keeps): worked out from `galleryRecords[0]`,
it took the day of the first live record while the first card was a deleted
head's tombstone, and drew a second line over it. The grid's one tab stop
moves to the newest image where the stop was the grid's default - no card
chosen since the last query (`galleryChosen`) - or the old head with focus
elsewhere: kept on the old head, Tab into the grid skipped everything the job
had just made. A card the user chose further down keeps it. Where the answer does not join up with what the page holds,
or the gallery is empty, it resets as before; never in sheet mode, whose sheet
lays itself out from the whole run. The head's numbers - its count,
`other_models`, Re-render, Delete unstarred and a gallery lightbox's counter -
are drawn as the page lands, and `prependFinished` answers the page's ids, so
`noteOutsideView` says its line at once; the cards go in through
`joinFinished`, which with Gallery beside may wait for the pointer (The gallery
beside the viewer). The viewer may now be a screen above: a
job's first image or frame landing out of sight shows New image (↑ or ↓; a
pill under the topbar in one column) under the progress, whose press scrolls
the viewer into view (reduced motion honoured) and focuses the newest image,
or the live tile that hands focus on to it - never a scroll of the page's
own - and which goes once the viewer is in view. The finish line offers what comes next: a plain
job's seed with Use seed (the Seed box alone - Reuse rewrites the prompt and
size too - after which, as after a grid's write-back, the run hint says the
seed is fixed and the dice frees it), Show sheet for a grid that does not open
its own (grids are never opened for the user), the deleted count beside what
the job made, and the Retry above.

**New image in one column** (2026-10-05). There the viewer sits under the
whole static panel, so it is as often a screen below as above: a Generate
pressed at the page's top put the image 150-800 px under the window's foot,
and New image, in the panel's flow, was never offered for it; for a viewer
above, it sat in the panel as far out of view. `viewerOutOfSight` now answers
above or below at any width, and `syncNewImage` writes the arrow and its
aria-label to match and follows a scroll that jumps past the viewer without
showing it (End, Home). Its press is still the page's only scroll. Under 901 px
it is a pill centred under the sticky topbar. It first sat in the window's
corner, over the right end of Generate, which is the panel's full width and
rests at the window's foot at the page's top: a press there scrolled to the
viewer and queued nothing, after the job as well as during it. Under the
topbar it lies over whatever has scrolled under it once the page moves - the
prompt's first line, Check - so once the job has ended, the user's own press
in the panel or typing in one of its boxes withdraws it
(`newImageBackToPanel`), the finish line still naming the result. Focus
arriving does not: the finish line takes focus itself, and a Tab towards the
pill passes through the panel. While the job runs it stays, since the image is
still coming. Before any image has landed the press focuses the live preview's
tile (or a pass's stand-in), which `streamJob` hands on to the image that
replaces it, or to Generate where nothing does; with nothing in the viewer it
focuses the viewer itself (tabindex -1, let go on blur) or its held line.
Focusing only an `img[data-id]` had left focus on the button as it hid, and so
on `<body>`. It goes (`withdrawNewImage`, focus on it handed on, never to
`<body>`) once the viewer is in view, when a job's first frame or stand-in
clears the strip (`landed`), and when a job whose first frame cleared the strip
ends or gives up with nothing of it in the strip - cancelled or failed after
its first frame, before any image, there is no image to go to. Not at
`setBusy(true)`, where it used to go: a job that never reaches the strip, such
as another tab's job picked up over this page's results, leaves it pointing at
this page's image, which is still there (2026-10-05). A refusal from Generate in one
column is still said in the panel's line under it, without a scroll: scrolling
to it would move the prompt being typed out of view.

## Two tabs on one server

**A second tab sees the server's jobs, and does not take them** (2026-10-05
area review). Every hold and every job word was keyed on the page's own
`busy`, and the page attached to a job only at load or after a job of its
own. A second tab said nothing was running, offered Unload, the critic and
Expand - each refused by the server in words naming a job it did not show -
and its gallery missed the other tab's images until a reload. The status now
carries the live jobs' ids, oldest first (`ModelStatus.active_jobs`, cheap on
the 3 s poll), which `followServerJobs` reads on every poll. A page watching
none of them sets `elsewhereBusy` and holds what the server would refuse, with
that reason: Unload, Ask the critic and Expand, aria-disabled - Expand too,
since a hold that arrives with a poll, at a moment no one chose, must not take
a keyboard user's focus. Unload and Expand are titled "A job started in
another tab or window is running; Unload|Expand once it ends", and Ask the
critic "A job started in another tab or window is running. The critic needs
the card to itself, so it waits for that job to finish." Each press says it as
a sentence until that job ends (`elsewhereNote`): in `#error`, and Expand's
through `expandSay`. Generate stays live: the server queues it. The topbar says
"· busy in another tab" beside the checkpoint's name while the model is ready
(`#status-aside`), aria-hidden and outside the name's live region, since it
comes and goes with jobs nobody in this tab started, and hidden at 600 px and
under, where the name would wrap. When the flag clears, the gallery and the
Runs picker catch up (`prependFinished(0)`, `loadRuns`) - never under a sheet
(a run's or a search's), which would be laid out again whole, every cell and
control rebuilt, at a moment no one chose; Sheet off resets the gallery, which
takes the images in then. Where a job of this page's own took over while
another tab's ran (`setBusy(true)` clears `elsewhereBusy` quietly), the
catch-up is owed (`elsewhereCatchUp`). That job's end takes it, whatever the
job made, a cancel before its first image included - such a job left the
other tab's images out until a reload. A stream that gives up takes it too,
and the server answering again clears it, since that answer takes everything
in. The page does not attach to such a job: watching it would take its viewer.

**A job that ends between two polls is seen by the gallery's newest id**
(2026-10-12). `active_jobs` only says what is live at the moment of the poll.
A job queued, run and ended inside the 3 s between two - a turbo checkpoint's
single image on the card, a scripted fetch, the fake at full speed - never
set `elsewhereBusy`, so its end never caught up, and its images stayed out
of this tab's gallery and CFG ladders until something else reset them. The
CFG smoke had been paced with `/api/fake/step` to outlast a poll, and
smoke-jobs' second-tab section failed at `--step 0.15`, for this reason. The
status now also carries `newest_image_id`, the gallery's highest id, and a
page catches up (`followGalleryHead`) when it moves past the last one said
and the page's own watched jobs do not explain it - the same
`prependFinished(0)` as another tab's job's end, with its rules for a
sheet, the bin, a filter and the pointer, and no duplicates. The server
decides what is new; the page only compares two numbers.

It counts every row, the bin's included, because the question is "was
anything inserted": ids are AUTOINCREMENT and stay monotonic across a clear
and a replaced database, so every new image lands above the value, and
nothing else raises it. A live-only maximum would fall when the newest is
binned and rise again at its restore, a catch-up for an image the page
already knew; a permanent delete of the newest lowers it either way, which
the page only remembers, since the next insert lands above the old value.
The page keeps the last value said rather than the highest seen, so a
gallery that starts again from 1 (a database deleted by hand) still moves
it. `MAX(id)` on the rowid is one b-tree descent, cheap enough for the poll;
the read takes the gallery reservation as every read does but waits only
0.2 s for it and answers None, "unknown", rather than stall the status
(exempt from maintenance admission) behind a bundle operation or a clear.
The route lists the live jobs before it reads the id, so a job absent from
`active_jobs` has all its rows counted: a job's saves commit before it ends.

What the page's own jobs made it knows by id (`ownImageIds`, from the
stream's images and the finished snapshot), and a rise that is all its own
is only remembered: those images land through `finishJob`. A rise while one
of its own jobs runs is owed to that job's end (`elsewhereCatchUp`), as an
ended other tab's job is; while it submits or lists, or another tab's job is
live, the rise waits for the poll after, since each of those ends with a
catch-up of its own. The first status, a restart, the server answering
again, a checkpoint change and another tab's job ending in the same answer
only set the baseline: each of those has just taken everything in.

Review of that change (2026-10-12) moved five things. The baseline came
from the first status, a read separate from the first gallery page, so an
image saved between the two was counted as seen and never shown; the
gallery page now carries the newest id read under its own reservation
(`list_images_at_head`), and that sets it. A rise during another tab's job
waited for that job's end, which for a chain of its jobs (a queue of
Generates, Mass refine split into several) was the end of the last one,
hours for a grid, so rises are now taken in while another tab's job runs
too, and its images come in job by job. The catch-up's page was the newest
48, so an import of a hundred rows in another tab did not join up and every
idle tab reset its gallery, emptying the grid under a pointer; the rise is
now passed as `made`, and a page that still does not join up resets only
once the pointer has left the column. The ladders learnt of new images only
through `fresh`, which a sheet, the bin, Starred or a search leave empty,
so every catch-up now asks every held ladder again (`noteUnseenImages`).
And the status poll's file lock, a millisecond or two every 3 s per tab,
could refuse a CLI `clear` or `reindex` whose single attempt landed inside
it; maintenance now polls the file lock for `MAINTENANCE_GRACE` (0.25 s)
before refusing, which rides out any read and still refuses a real holder
at once to a person.

Under load (eight fake servers and their browsers at once) that change
broke three smokes, two of them by a catch-up landing late. The setup of
smoke-compare-cfg and smoke-delete is a script's jobs, which the page never
watched, then a reset of the gallery; the status that saw the newest id
move could arrive seconds later, under the lightbox or after a delete. It
asked every ladder again, so a held CFG comparison said "Looking for other
CFG values…" for a moment in place of its "No other CFG value" note; and
its `prependFinished` on a grid emptied by Delete unstarred - or with that
delete's reset still out - reset the gallery again, which clears the
gallery's line, taking away "Moved 2 images" or "The selection changed".
Both reads had the images already. A ladder answer now keeps the newest id
its first page read (`head`), and `noteUnseenImages(newest)` drops only
those read before; the grid keeps the newest id its own page read
(`galleryReadHead`), and a catch-up up to it does nothing, while one that
meets a reset's page in flight waits for it before deciding. The third,
smoke-model-switch, failed every time and was the server's: the status
route read the newest id with an await after its revision/`switching`
retry loop, so a switch could run to its end in between, and Apply & save's
own answer came back as the old checkpoint and revision with `switching`
false. The page took that for a switch with no background revision, forgot
it was its own, and the revision that followed cleared the outcome ("Settings
saved; … was unloaded", "Switched to …"). The id is now read in the same
thread as the description, inside the loop.

A page that picks up another tab's job when its own ends (`watchNext`), while
its viewer holds this session's own results (`isForeign`), shows it in the
progress row only - no `clearResults`, frame or stand-in - names it "Another
tab's job:" in the Now line, the mirrors and the title, and says its end,
"Another tab's job finished: N images.", once on `#job-said` alone, with no
Use seed and no held title. Its first frame used to clear the image just made,
its finish line offered the other tab's seed as this page's, and its progress
wrote over this tab's own unseen outcome in the title, which its end now gives
back (`titleHeld`). It leaves this page's own finish line alone - its last job,
Use seed and all, with its lead as it was (`staleBefore`, saved as the stream
starts) - and New image too, which only a job whose first frame clears the
strip withdraws (`landed`): its finish line had replaced this page's, and its
start had taken New image away from an image still there. A
reload's attach, with nothing of this page's in the viewer, is shown as
before. A page busy with its own job takes its queue rows from the server
(`resyncQueue` through `setQueue`, the rows' one writer, which keeps focus as
a row's × does, and keeps the ids it listed after the poll was asked,
`queuedAt`): a row cancelled elsewhere goes, and a job queued elsewhere is
listed as "Another tab's job: …" ("Next, another tab's job: …" alone), its ×
"Cancel another tab's queued job: …", Cancel queued's title saying it takes
another tab's too - since both now reach another tab's work. A cancel this
page did not ask for (`cancelledHere`: Cancel, a row's ×, Cancel queued; a POST
that got no answer is not one) reads "Cancelled from another tab or window —
…" and holds the title "Cancelled elsewhere", and a shutdown says it is the
server's stop (`stopped_by` "shutdown"). Which jobs and cancels are this tab's
is kept in sessionStorage (`TabJobIds`: `local-diffusion.own-jobs`,
`local-diffusion.cancelled-jobs`, read once at load, the newest 50, every
access in try/catch), which a reload keeps and another tab does not share:
without it a reload's listing called the tab's own queued jobs another tab's,
and a Cancel pressed just before a reload was reported as another tab's.

Three more things crossed tabs. A reference deleted elsewhere is asked about
(Reference images, by gallery id). A failed load retried in one tab is
reworded in every tab's finish line, as is one unloaded since, and its detail
leaves the panel's line (Loading happens in one place). And the panel's store
is shared: a write merges (Panel persistence). The pages are checked in one
browser by an iframe of the page in `smoke-jobs`, whose lexical globals a step
reaches through `contentWindow.eval`; it shares the top page's sessionStorage,
which each page reads only at load.

## Time budget

**A grid can have a time budget.** `GridRequest.budget_minutes` reaches the
job through `submit_cells(budget_minutes=)` - GridRequest-only fields are
gone by the time cells are built - as an absolute `Job.deadline` from
submission (the load counts). `_generate` checks it at the top of every
cell ("start no cell after"), sets `stopped_by = "budget"` and returns
early through the *cancel* path: a fourth terminal status would have left
`active()`/`busy()` holding on and unload answering 409 for the session.
The UI reads `stopped_by` to say the plan worked and a Resume finishes the
rest; the CLI's `--budget` mirrors it and exits 130 like any other stop
short of the full count. Its count is `run`, which at a budget stop is already
the cells finished (the deadline is checked before `run` moves on) - the page's
`run - 1` said 34 beside 35 images - and the banner says "stopped at its time
budget" rather than "was cancelled".

## The variable assignment is recorded

**The variable assignment is recorded.** `Cell.variables` reaches
`save_image(variables=)` from both writers and lands in a `variables TEXT`
column (JSON; `_migrate`), the `local-diffusion` chunk and `ImageRecord.variables`.
Both reproductions - `/api/rerender`, and the CLI's `--from-image` while the
prompt pair is unchanged - hand a row's variables to the cells they build.
The sheet builds one axis per variable name seen (`axesFor` in `sheet.js`)
and **suppresses** the flat prompt and negative axes when any record
carries variables - kept as fallbacks, a 4-value prompt axis would beat a
2-value variable axis in `chooseAxes` and turn a single-variable run into a
diagonal. `""` (the `none` value) reads as `none`; a record without
variables in a mixed search reads as null. The axis list is rebuilt per
render and the names are part of the layout signature. `completed_cells`
and `CellKey` are untouched: the resolved prompt is what ran.

**A prompt variant records its label as the variable `variant`.**
`GridRequest.prompt_variants` - the prompt analyser's original, reordered and
rewritten versions of one idea - is a list of whole prompts, and each cell
records `variables={"variant": label}`, so a label rides the column, the chunk
and the sheet with no new field. That stretches `variables` from "what the
placeholders were set to" to "which version of the prompt this is"; a label is
not substituted anywhere. Variables could not carry the versions themselves:
`prompt_axis` stops at 2,000 characters where one prompt may be 4,000 (three
long Krea 2 prompts overrun it), `|` splits a value and parsing is a line per
variable, so NewBie's XML - all newlines - cannot be one value, and `none`
and its siblings become empty. A variant is therefore taken verbatim: nothing
is substituted into it, and a `{` in pasted JSON is text, not a placeholder
with no values. The request's own prompt is not rendered; its negative prompt
runs beside every variant, and a placeholder left in *that* is still refused,
since nothing would fill it. Labels are stored trimmed and a label of only
spaces is refused. Identical variant texts are not refused - counted as
duplicates in the preview, as `_expand` treats duplicates elsewhere - and the
page drops them before submitting. `CellKey`, `run_id`, resume and the
duplicate count read the variant's text, so renaming the variants changes
nothing about the run. In particular, swapping original and yours on the
same seeds shares a run, although the labels now mean different prompts.
The comparison therefore opens `job:<id>`, its own submission, and its
statistics recipe uses `--job`. This keeps existing run ids and partially
completed runs resumable. When a sheet reads several such submissions, the
flat Prompt axis remains available if a complete variable assignment names
more than one prompt: otherwise its (variant, seed) cell would silently keep
only the newest. Ordinary variable grids still suppress that redundant axis.

**The sheet lists a variable's values in the order the grid wrote them.**
`sheetValues` used to sort a `var:` axis with `localeCompare`, which read a
`size = small | big` grid as big, small - against the declared order
`parse_variables` promises. (The comparison's own labels - original,
reordered, rewrite, yours - happen to sort alphabetically in the order the
page sends them, so they never showed it; an earlier version of this note
said alphabetical put `original` below `reordered`, which it does not.) It
orders by the oldest image carrying each value: the grid renders values as
declared and variants in the order given, seeds outermost, so the first
image of each is in written order. Numeric and flat-prompt axes are
unchanged. The oldest *loaded* image is only the oldest a filter kept: star
the winners, tick Starred, and the order was whichever survived (a
`size = small | large` grid over seeds 20 and 21, with s20-large and
s21-small starred, read large, small). So when Starred is ticked, or the
search is anything but `run:<id>`, a sheet of one run with variables asks
once more for that run unfiltered before it renders (`learnSheetOrder`, from
`loadGallery`: the one extra request) and orders by the whole run's oldest
ids; if that request fails, the loaded images order themselves.
`chooseAxes` gives a varying `var:variant` axis the rows whenever anything
else varies: widest-first alone made the variants rows only while the seeds
outnumbered them (3 x 3, 3 x 2 and 8 x 4 all turned them into columns, the
tie going to the `var:` axes because `axesFor` lists them first). The
variants and the axis across them are placed from every varying axis, not
from what the listlike rule left. With one image per version loaded (a star
each under Starred, or what Delete unstarred kept), that rule dropped the
variants when each version had one image, or the seeds when each image had
its own seed, and the pin that took their place showed one image of the
comparison. Alone, the variants stay a strip in the columns, because rows
with nothing across are one column of full-width images. A variable a user
names `variant` by hand gets the rows too - the one name the sheet
recognises, held equal to `grid.VARIANT` by `tests/test_grid.py`. Known gap:
ordering by oldest id still moves a value down for the run when its first
cell failed (a job skips a failed cell), when a resume rendered it later, or
when Delete unstarred removed the rows that carried it first, since the
server no longer has them to ask for. Recording each cell's declared index
would close all three; it would need a new column and is not done.

## Two writers, one INSERT; strict scope; disk estimates

**Two writers, one INSERT.** `save_image` and `storage.index_file` (which `reindex` calls) both go
through `storage._insert_row` over `_COLUMNS`, so the column list exists
once and the FTS trigger fires for both. The reindex reads
the Local Diffusion PNG metadata, parses `created_at` from the filename prefix
as UTC (mtime failing that), copies a foreign file in `.part`-then-rename
and never over a different file of the same name, records `duration_ms=0`
(so `step_cost_ms`'s `> 0` ignores it) and loses `favorite`, which lives in
SQLite alone, and `binned_at` with it: a binned image comes back live. `search.strict_scope` is the `key:value`-only reading of the search
grammar for the paths where the LIKE fallback is a hazard: `clear --where`
(`delete_matching`, `DELETE ... RETURNING` in one statement, no orphan
sweep on that path), the gallery's Delete unstarred (the same function,
see its section) and `keep_rates`. `/api/status` carries
`disk_free_mb` (one `shutil.disk_usage` syscall; the directory walk in
`summarise()` deliberately does not ride the 3 s poll) and the preview
carries `bytes` (`image_bytes`: median PNG size at this size on this
checkpoint, times the images) under the same no-rows-no-estimate rule.

**A database-backed import keeps the database-only state.** `reindex --from
<outputs> --database <local-diffusion.db>` reads the source database in read-only mode,
copies only readable PNGs, and inserts through the same `_insert_row` seam.
Unlike PNG-only recovery it preserves `favorite`, `binned_at`, `duration_ms`, and every
newer reproduction column the source has. SQLite ids cannot be copied into a
non-empty gallery: source rows are therefore visited in ascending id order and
reference-image ids are translated through the source-to-destination map.
References necessarily point to images which already existed when the
conditioned image was made; a missing or conflicting predecessor is dropped
and counted rather than left pointing at an unrelated destination row. The
copied PNG's Local Diffusion text chunk is replaced with the remapped metadata by
rewriting PNG chunks, not pixels. Filename uniqueness remains the idempotence
key, and an unindexed destination file of the same name remains a conflict,
never an overwrite.

## Gallery bundles: verify before clearing, stage before merging

Copying PNGs without SQLite loses favourites and timings. **Gallery bundles**
packages both in one portable ZIP, with thumbnails and unindexed PNGs, then
clears the server's gallery. Its scope is every checkpoint and favourite;
the record count typed into the dialog is checked again under the reservation.
SQLite's backup API captures committed WAL contents. A versioned manifest
records each member's size and SHA-256; the archive is closed, flushed and
read back before publication. Tracked PNGs are decoded from the ZIP and their
dimensions and database fields pass the same validation as import; corrupt
tracked images therefore prevent cleanup. Unindexed image files are preserved
as bytes. A second source inventory and hash pass catches
files changed outside the app before cleanup. A published bundle is retained
even if cleanup cannot finish, and the result names the files that remain.

The server names the bundle `local-diffusion-YYYYMMDD-HHMMSS-<8 hex>` from its
own local wall clock (`datetime.now().astimezone()`), since it runs on the
user's machine; the name once used UTC and read hours off. Nothing parses a
time back out of the name: rows keep their own `created_at`, and the list's
newest-first order by name can misplace a bundle only across a clock change
or against older UTC-named files.

Deletion can fail on Windows when another program holds an image open. Rows
for retained PNGs survive; successful deletions retire their job and browser
records. The cleared database is replaced with a compact copy that preserves
its AUTOINCREMENT sequence. Simply deleting the database would reuse ids,
allowing old reference chips and job snapshots to name unrelated new images.
The running app therefore always has an empty usable database after a full
clear, even though the old database file has been replaced.

Cleanup first removed every file and only then deleted the rows, and it
reported `deleted_ids` before that DELETE ran. A database held by another
program ("database is locked" after storage's 15 s wait) or a full disk at
that point left rows pointing at missing PNGs, and the page forgot ids whose
rows survived. Cleanup now takes SQLite's write lock (`BEGIN IMMEDIATE`)
before it removes anything, deletes rows only for files that went, and
reports ids once the DELETE commits. Files still go first, so a retained
file keeps its row. A crash between the unlink and the commit can still
leave rows without files; re-importing the bundle repairs that (below).

Import treats ZIP contents as untrusted. It accepts only the versioned
manifest, database and known image locations; traversal, special files,
duplicate/case-colliding names, checksum failures and conflicting destination
files are refused. A private database copy receives the merge first. New ids,
reference/refine links and the PNG metadata agree before installation. Existing
rows and images win on repeated import; favourites changed since an earlier
import are not reset. Only new image files are installed, with exclusive
creation; a failed installation rolls those files back before the live
database is replaced. Missing source links are counted and cleared rather
than pointed at unrelated destination ids.

What counts as "present" was looser than that rule needs. The decode
fallback compared `content_hash`, the SHA-256 of `tobytes()`, which for a
palette image holds the indices only: a red and a blue palette image, or an
`L` image with the same bytes, passed as the same picture and the bundle's
was dropped. The fallback now compares the full decoded identity (mode,
size, palette, transparency and the untruncated digest); `content_hash` itself
is unchanged, since `hash:` search depends on it. A present image's
thumbnail was held to byte equality, so a cache regenerated by another
Pillow/libwebp build refused the whole re-import; it is now kept as its image
is. And a known row whose PNG was missing (an interrupted clear, a file
deleted by hand) was reported as "another image already uses the filename"
and refused the merge. Re-import now restores that file when the bundle
copy is plain RGB at the row's own size, its pixel hash equals the row's
`content_hash` (raw bytes only: a palette copy with the same index bytes
would match) and the identity metadata matches, rewriting its PNG chunk from the unchanged row; anything else is
refused with an accurate message.

A killed process (power cut, second Ctrl-C) skips every `finally`, and
bundle operations left gallery-sized debris nothing swept: `bundles/*.part`,
export's snapshot directory, import's unpacked copy (decrypted, for a
password bundle) and uploads. These follow the `.png.part` rule: once older
than `PARTIAL_SETTLE_SECONDS` they are removed at startup, before every
bundle operation and by `clear_history`, all under the reservation, which
every bundle operation holds throughout; an upload still arriving keeps its
file fresh. `summarise` counts them.

Staging used to sit beside the database. With `LOCAL_DIFFUSION_OUTPUTS_DIR`
on another drive every new image was staged on the database's drive, then
copied again by the install's copy fallback, and the free-space check looked
at the wrong drive. Imports now unpack, and uploads land, in
`.local-diffusion-staging` beside the images directory (inside it when it is
a mount point of its own, where the inventory skips directories), so
installation is a rename. The merged database is still built beside the
live one, which it replaces, and free space is checked for each volume. The
copy fallback, still used without hard links, writes `<name>.part` beside its
target and claims the final name only when complete: a crash mid-copy had
left a truncated PNG under the final name, refused as a conflict by every
later import. A tracked PNG is held in memory to be decoded, so one over
256 MiB (far above any generated image) is refused before it is read: a few
megabytes of deflated zeros could otherwise declare gigabytes. `file_size`
alone is not the bound: `ZipExtFile.read()` fetches the whole
`compress_size` in one call, so a small member padded after its deflate
stream allocated gigabytes before the CRC passed. A member stored larger than
deflate could make it (`file_size` + 0.1% + 1 KiB) is refused up front.

`api_bundles.BundleManager` reserves the idle server synchronously. ASGI
admission counts existing gallery requests through the end of image streams,
and blocks new generation and gallery operations while maintenance runs.
Normal status, job inspection and bundle download continue to answer. The
worker is independent of its initiating request; disconnecting cannot release
its reservation or remove an upload it is still reading. Graceful shutdown
waits for completion. `gallery_lock` extends the reservation to full storage
and reindex operations in other cooperating processes. It first failed every
contended session at once, reads included: a `stats` query or a CLI save in
another process turned the page's gallery requests into 409s and discarded a
job's finished render, where SQLite's busy timeout had absorbed the overlap.
Ordinary sessions and single-record writers now poll for up to
`SESSION_WAIT` (30 s; Windows' blocking `LK_LOCK` gives up after about ten of
its own); bundles, clear, vacuum and reindex still refuse at once rather than
queue behind a long operation. The wait is one deadline over both the
in-process lock and the file lock: a thread polling for another process holds
the in-process lock, and four threads queued behind it without a limit were
refused at 1, 2, 3 and 4 x `wait`, tying up the route thread pool for minutes.
A job's `save_image` that is still refused after one `SESSION_WAIT` is retried
by `jobs._save` until the job is cancelled: the render is in memory and an
operation in another process can outlast 30 s. `generate.py` retries the
same way, saying once that it is waiting, until a second Ctrl-C: its save
sat outside the per-cell `try`, so a `GalleryBusy` reached the outer
handler, discarded the image and ended the night with exit 1 (review
2026-10-12, R35). Its `--resume` lookup, before anything renders, refuses
in one line instead of a traceback. The reservation is also a per-process RLock
that every read takes, so `save_image` encodes its PNG and WEBP to `.part`
files before reserving: held through a 2048x2048 encode, a 3 ms `get_image`
on another thread took 330 ms. The sweeps do glob `*.part`; only their
`PARTIAL_SETTLE_SECONDS` age cutoff keeps them off an encode still in flight
outside the reservation, so do not shorten it to minutes. An export whose
post-verification inventory finds a `.part` clears nothing. Raw external programs
do not take that lock, so the final inventory/hash check and SQLite checkpoint
refusals remain necessary. No archive operation is run against the user's
gallery as part of tests: core/API tests and browser smokes use scratch data.

**The recycle bin travels with the gallery** (2026-10-11). An export is a
snapshot of what is on disk, and binned images are on disk: they are in the
archive, cleared from the source with the rest (in `deleted_ids`), and
the preview's `records` stays every row, the number typed back, with
`binned` beside it for the dialog's "(N in the recycle bin)". An imported
new row keeps the `binned_at` it left with; a row this gallery already
holds keeps its own state, as it keeps its star, and `binned_at` is no part
of the identity comparison, so binning or restoring an image never turns a
repeat import into a conflict. An old bundle has no column and imports
live; a newer bundle into an older build does the same, which is accepted
(see "Recycle bin: why the per-image confirm went"). A bin or restore in
another process mid-export changes a row, and the row-equality check
before cleanup refuses it, which is the right answer.

### Password bundles and the one-pass read-back

**Why not an encrypted ZIP.** Python's `zipfile` cannot write encrypted
archives, and the only scheme it reads, ZipCrypto, is broken. AES ZIP
(WinZip AE-2, via `pyzipper`) would open in 7-Zip, but it leaves member names
readable and relies on an old fork of `zipfile`. A password bundle is instead
the ordinary bundle ZIP inside one authenticated layer (`bundle_crypto.py`).
The header is magic, version, scrypt parameters, salt, chunk size, nonce
prefix and a password check. It is followed by 1 MiB AES-256-GCM chunks.
Each chunk's associated data binds the header digest, its index and a final
flag, so a flipped bit, a reordered or dropped chunk, truncation and
extension all fail authentication. The final chunk is authenticated before
zipfile ever seeks. scrypt runs at n=2^17, r=8 (128 MiB, about 0.2 s). Header
parameters come from an untrusted file, so they are bounded before any key
is derived. The check value is a hash of the half of the scrypt output that
the key does not use. A wrong password is reported as such, and the browser
can test it against the file's first 64 bytes (`POST .../password`) before
uploading gigabytes. Plain bundles stay plain ZIPs.

A password is checked once for a UTF-8 form in `api_bundles._password`.
JSON can carry a lone surrogate (a broken emoji pasted into the field); it
used to reach `_secret`'s `.encode("utf-8")` and answer the password check
with a 500, or put the codec's message, naming the character's position in
the password, into the bundle status. It is now one fixed 422 on every route.

An interrupted upload used to fail as `ClientDisconnect`: the starlette
exception has no message, and `fail` fell back to the class name, which the
dialog showed in red on every later visit. It now fails with a sentence that
says nothing was imported, and the staged `.part` is removed as before. The
dialog's Cancel upload aborts the request's `AbortController` while its body
(or the password check) is still being sent; it is always present and
`aria-disabled` otherwise, so nothing moves under the pointer when an upload
starts or ends.

The server keeps its last result until the next operation, and every new
page's first poll took it for news: a reload hours after an export said
"The gallery is empty" above a preview counting new records, and forgot ids
and reloaded the gallery a second time during start-up. The page now acts on
a result only when it started the operation or saw it running; one that
ended before it loaded is shown as "Last bundle operation:" and nothing else,
and so is a result already read when the dialog is reopened. The progress
counts were part of the polite live region's sentence, so a screen reader
re-read all of it at every 750 ms poll for a whole upload; they now sit in a
non-live line beside it. The unlock password stayed in its field after an
import, unlike export's, which empties at the press; it is now cleared once
an import is done and kept after a failure, so a mistyped one can be fixed.

**No extra pass.** Export points `zipfile` at `EncryptingWriter`. Its seek
fails, so zipfile writes data descriptors (flag bit 3, which `bundle_archive._verify_archive`
accepts). The archive is therefore written once, already encrypted.
`_verify_archive` reads through the seekable `DecryptingReader`. The single
existing read-back authenticates every chunk with the user's password
*before* anything is cleared, and no plaintext copy ever touches disk. Import
reads the uploaded ciphertext the same way. Opening it authenticates only the
header, the length and the final chunk, so a wrong password, truncation or
extension is refused before anything is staged; any other altered chunk is
refused when unpacking reaches it, after earlier members were staged but
before anything is installed, and the staging directory goes with it.
Authenticating every chunk up front would be the second pass this section
avoids. A writer closed without
`finish()` (an exception mid-archive) never yields a validly terminated file.

**Where export and import time went.** Profiling a 208-image, 495 MB scratch
gallery (synthetic noise PNGs, warm cache, Defender on) gave: export 5.65 s,
import 6.98 s, re-import 7.48 s, and HTTP import 8.93 s, all linear in image
count. About half of export and import, and 81% of re-import, was decoding
every PNG with PIL on a single thread to check its pixels. Import's work-file
copies plus one fsync per installed file took about 30%. The upload wrote
each ASGI chunk with its own thread hop into `%TEMP%`. The fixes keep every
guarantee above:

- Tracked PNGs are decoded from the bytes `_verify_archive` just read, on a
  pool of up to 8 threads. Export no longer reopens the ZIP, and import no
  longer reopens extracted files. Admission is bounded by the decoded size in
  each PNG header, and errors surface in row order.
- On repeat import the bundle copy of an image already present is never
  unpacked. The existing image is compared by its pixel-deciding chunks
  (IHDR, PLTE, tRNS, IDAT) and decoded only when those differ.
- Staged files are fsynced and moved into place: rename on Windows (which
  refuses an existing target), link on POSIX, and a copy across volumes.
  Uploads are staged in 8 MiB writes (now on the images' volume, above).
- The pre-cleanup inventory hash runs on the pool, after a size check.
- The status reports phase and done/total; the dialog shows it.

The same gallery then took 3.19 s to export, 3.63 s to import and 1.67 s to
re-import; with a password, 3.93, 4.23 and 2.35 s. These are CPU and disk
figures from the desktop, not card measurements.

A second round, profiled on 300 noise PNGs at 1024x1024 (791 MB bundle),
found the time in per-file latency on the threads that read and write the
archive. Opening each source file waits for on-access scanning, and so do
creating each staged file and each fsync. Moving that work off those threads
took the medians from 5.2 s to 2.7 s for export, 6.2 to 3.7 with a password,
5.3 to 2.2 for import, 5.6 to 2.4 with a password, and 3.1 to 0.6 for
re-import:

- Export reads its sources whole (up to 16 MiB each, `_ReadAhead`) and
  hashes them on a pool, a few files ahead of the writer. The writer now only
  does the CRC, the write and the encryption.
- Import writes, fsyncs and closes staged files on `bundle_archive._Stager` (16 threads,
  bounded by count and by 256 MiB of waiting bytes), not on the reader. Each
  tracked PNG is read whole and SHA-256 hashed on the decode pool before it
  is inspected. The metadata rewrites, each a full copy plus an fsync, run on
  a pool. They still all finish before the database commit and before
  installation, and on a failure before the staging directory is removed.
- A password export's read-back reuses the writer's key for its
  byte-identical header and skips the second scrypt run. Every chunk is still
  authenticated, and the header digest is still in every chunk's AAD.
  Import, upload and the password check always derive from the password
  they are given.
- On re-import, a present image whose pixel-deciding chunks match is not
  decoded at all. Its bundle row is still validated against the copy's IHDR
  size. A non-PNG member whose destination exists, such as a known image's
  thumbnail, is no longer unpacked; it is compared in place with its verified
  manifest entry. If such a file disappears before installation, the import
  is refused.
- `bundle_images._raw_digest` hashes the same bytes as `tobytes()`, block by block. The
  whole-image copy and join had held the GIL, and with it the decode pool
  managed only about half the throughput it now gets.
- `EncryptingWriter` encrypts whole chunks straight from the caller's buffer,
  and `DecryptingReader.read` joins block slices once rather than going
  through `readinto`.
- The upload keeps one 8 MiB write in flight while the next batch arrives.
  That write is shielded from request cancellation and always drained
  before the file is closed or removed.

What remains is the work the guarantees require: writing the archive with its
CRC and encryption; reading it back with CRC, SHA-256, authentication and a
full decode of each new image; the pre-cleanup hash; one fsync per installed
file; and the metadata rewrite. More threads did not help. The decode pool
and the metadata rewrite gained nothing past 8 threads, and the stager gained
nothing past 16.

Rejected because each would weaken a check: a zlib-only validation instead
of a full decode; size and mtime instead of the pre-cleanup hash (external
tools do not take `gallery_lock`); and trusting the live `content_hash`
instead of the existing file. A whole-file hash fast path for present images
never fires anyway, since import rewrites their metadata chunk.

## The CLI reads PNGs and probes the server

**The CLI reads PNGs and probes the server.** `--from-image` reads
the app's PNG metadata from `.info` - not `.text`, which decodes every pixel - and
fills what the flags left unset *before* `apply_env`, so the record's
checkpoint reaches `LOCAL_DIFFUSION_MODEL_ID`; quantization is deliberately not
carried (a Nunchaku PNG on Windows would ask for a mode the machine cannot
run) and job/run ids are displayed, never reused. `_server_holds_gpu`
probes `/api/status` after the `--dry-run` guard: `ready`/`loading` or a
timeout refuses with exit 4, connection refused is the ordinary no-server
case, `--ignore-server` overrides; the test fixture stubs it, or the suite
would refuse to run beside the developer's own server. `keep_rates` groups
on the same six-decimal rounding `completed_cells` uses and always includes
size; free text in its scope is refused rather than sent through the
fuzzy LIKE fallback.

**`--from-image` carries the variables** (`_image_variables`), as
`/api/rerender` does, because they are the only place a comparison
version's name lives. They carry only while the prompt pair is the image's
own, compared as text after the fill so a retyped prompt keeps them, and
never beside a `--grid-prompt` that holds a variable (a blank one is no
axis): a variable may have filled either half, and a label on another text
names a version it is not. A comparison version (`{"variant": label}`) goes
back through `expand(variants=[...])` as a one-variant prompt axis, so its
prompt is taken verbatim and a `{word}` in it is text, not a placeholder
with no values. A `{var}` assignment is already substituted into the prompt
and is attached to each cell as it is, leading the cell's log label as a
grid's own does. So is a `variant` value that is no label a comparison
could have recorded, because a user's own grid may name its variable
`variant`: its `none` value is empty and a long one is past
`PromptVariant`'s 40 characters, and the schema refusing it is what picks
the attach route. Known gap, older than that carry: every CLI run goes
through `expand()`, whose placeholder check `/api/generate` skips (see "The
CLI's loop, and jobs.submit's"), so a plain prompt holding a literal
`{word}` - typed, or read back by `--from-image` from an image Generate
made - exits 2 with `prompt: {word} has no values`; only a carried
comparison version is taken verbatim.

**`stats --by variant`.** It groups on `json_extract(variables, '$.variant')`
(`grid.VARIANT`), a comparison's version by its label. `--by prompt`, the
recipe first written for a comparison, printed one identical row per
version: a cell holds 37 characters plus `...`, and reorders share their
opening words. An XML prompt's newlines also broke the table; `_cell` folds
whitespace to single spaces. Rows rank by rate, then count, then first
appearance (`MIN(id)`), so a tie - an untouched comparison at 0% everywhere -
reads in written order, not in the label-alphabetical order of the GROUP BY.
The ranking itself is the table's contract, so versions are not forced into
written order. A comparison is read with `--job`, since run identity ignores
labels. A broader scope that gives a complete variable assignment two prompt
texts is refused with advice to use `--by variant,prompt` or a narrower job.
The check canonicalizes variable key order and stays within each emitted
stats group, so an ordinary `variant` by `colour` grid can aggregate colours,
and an explicit seed or steps grouping can separate different meanings.

**The CLIs write UTF-8 when redirected.** `generate`, `clear`, `stats` and
`reindex` reconfigure stdout to UTF-8 (`errors="backslashreplace"`) at the
top of `main()`: on Windows always, elsewhere when stdout is not a terminal.
Redirected or piped on Windows, stdout encodes with the locale's code page
(cp1252); a CJK or emoji prompt, or a CJK `--from` folder name, was then a
UnicodeEncodeError partway through the output, and `stats ... > out.txt`
exited 1 with only its header in the file. `isatty()` cannot be the Windows
test: `> NUL` is a character device, answers True and still encodes with
the code page, while the console itself is UTF-8 already
(`_WindowsConsoleIO`), so reconfiguring it changes nothing. A stream with no
`reconfigure` is left alone - a StringIO, or pythonw's `None`, where a
scheduled `clear --yes` must still run; the first version called `isatty()`
before that check, which made it an AttributeError. `reindex`'s per-file
lines go to `out`, bound to the same stream when the module is imported, and
reconfigure changes that stream in place, so they are covered too.
`tests/test_cli.py` holds the four copies to one setup. Windows PowerShell's
own `>` is beyond it: that re-decodes a program's bytes with
`[Console]::OutputEncoding` before writing the file, and the README says how
to set that to UTF-8.

## Panel persistence

**The panel persists to `localStorage`** (`local-diffusion.panel.v1`) in two phases:
the model-independent fields synchronously at start-up, *before* the sync
functions that light the presets and hints up from them, and steps/CFG/mu
from `refreshStatus` once the checkpoint is known - only the ones the user
owned, and only when the stored `model` is the loaded one. Writes come from
a delegated `input`/`change` listener on the panel plus explicit calls
after the programmatic assignments that fire no event (`choose`, `reuse`,
the dice, the seed written back after Run grid). Resume is never stored.
Every access is in try/catch; a blocked store is a page that works without
memory. **Reset** (`resetPanel`, the button at the end of the Prompt caption)
is the one thing that empties it: the store is removed, every `owned` flag
dropped, and the boxes put back to what a first visit shows - the last
status's profile for steps/CFG/mu, the 1 MP square, a random seed, no
conditioning, an empty folded grid. It exists because `backend.clear` is a
server-side deletion that never reaches `localStorage`, so after clearing a
gallery the old prompt came back on every open, and the only other way out
was wiping the site's data. It sits away from Generate on purpose: it costs
the prompt, and a slip next to the button pressed all day would cost it
unasked. It is red under the pointer (`.ghost.danger`), as the page's other
destructive buttons are, where it had looked like Check beside it. And one
press took the prompt and a grid spec that must be retyped exactly for
Resume to match, with nothing to bring them back - Ctrl+Z restored nothing,
and a reload kept the reset - so the panel as Reset found it is kept in
memory (`panelState()`, the raw sampling values and `owned`, the size's
preset, the Refine dialog's remembered settings and its unsent draft), read
off the page rather than the store, which is 300 ms behind and absent in a
private window. "Panel reset.", in the top right of the prompt box Reset has
just emptied, carries an Undo while the panel is as the Reset left it - the
first change takes the button, as a Reuse's does (`syncResetUndo` from
`persistPanel`) - and for 20 seconds at most. The Undo writes the boxes
through `rewriteBox`, puts the rest back (the English translation too, where
the prompt is the one it translates), runs the sync functions, remembers the
panel again and says "Reset undone: the panel is as it was." in the same
place for 6 seconds (`RESET_UNDONE_MS`); Resume is never put back ticked, and
nothing of it survives a reload.

**Reset's Undo is in the prompt box** (2026-10-05; `#prompt-note`, kind
`reset`), where Reset was pressed. In the panel's line under Generate it was
below the window at 1400x900 and in one column, and was retired unseen after
20 s. Reset empties the boxes through `rewriteBox`, so Ctrl+Z in the prompt,
which it focuses, gives the text back after the Undo has gone too: `.value =
''` had emptied the box's undo history. The store is forgotten after those
writes, whose `input` and `change` events would otherwise persist it 300 ms
on, and stays forgotten until the panel changes (`resetLeft`): the prompt's
later blur fires `change`, and a write of the panel as Reset left it is
skipped, so another tab's next write still takes the whole panel.

In the panel's flow (one column, or a greyed negative: FLUX.1, ERNIE and Turbo
by default) the note moved Generate down a line, 31 px, and when its 20 s ran
out, Generate moved back up under whatever pointer rested there. smoke-narrow
had to take the note back by hand so its fixed press on Generate landed
(66bcf14). Over the next field's caption, as an Example's note lies on two
columns, it would have covered the greyed negative's Set CFG button. So it
lies in the top right of the prompt box it has just emptied, under Reset, at
every width (`showPromptNote`'s `at: 'box'`, `data-at="box"`, which the two
flow rules leave out). The box stays empty for as long as the note shows,
since the panel's first change takes it. The placeholder, whose first line ran
under the note at 1400 px, is transparent meanwhile. "Reset undone", which the
Undo leaves over the text it put back, lies there too for 6 s, an 'info' note
that holds no button, lets the pointer through (`pointer-events: none`) and
goes as soon as the box is typed in. Nothing either note does moves Generate.

**Two tabs, one store** (2026-10-05). Each tab wrote its whole panel on every
change, so a second tab opened earlier that changed only its Seed wrote its
own empty prompt, grid spec and steps over the first tab's, and they were gone
at that tab's next load. A write now merges (`writePanel`): only the keys this
page changed since it last read or wrote the store go onto what the store
holds now - each field, the fold, the references, the translation, and each
sampling value on its own. The comparison base is `panelBase`: the stored
object `restorePanel` read, or the page as it opened where nothing was stored.
It is never the page at start-up where something was stored, because the
sampling half is applied only once the checkpoint is known and would have
looked changed. An empty store - a first visit, another tab's Reset - takes
the whole panel: written key by key into nothing, the sampling values lost
the checkpoint they belong to. Nothing listens to `storage`, so what another
tab wrote stays the store's until this page changes the same key.

What is not the panel's is not Reset's either: the gallery's card size and
fit (`local-diffusion.gallery.v1`), Gallery beside
(`local-diffusion.gallery-beside`) and the sheet's Fill width
(`local-diffusion.sheet.v1`) are view settings, like Starred, kept per browser
through the same try/catch helpers, and Reset leaves them.

The contact sheet's cells take the commonest aspect ratio on the
sheet (`--cell-aspect`, `object-fit: contain`) and carry `.fav`, toggled by
class from `setFavorite` - never by re-rendering, which rebuilds up to a
thousand `<img>` nodes per keypress.

## The seed axis is a count or a list

**The seed axis is a count or a list.** `GridRequest.seed_axis` (the UI's
Seed list box, the CLI's `--seed-list`, exclusive with `--grid-seeds`) goes
through `grid.parse_seeds` - `parse_axis` with an int cast - and `expand()`
takes `seed_list`, which wins over `seeds` and the base seed. The ceiling
counts whichever applies. Listed values are **not** wrapped: the schema's
`ge=0, le=MAX_SEED` on the cell is what turns a typo'd `-5` into the
schema's own error rather than a valid 9.2-quintillion seed. A listed grid
is determinate on its own, so `_resume_needs_a_fixed_seed` waives the base
seed for it, and the UI does not write the first listed seed back into the
Seed box. The preview's `values.seed` is the list, or `[]` for a count.

## waiting_on and the ETA

**A queued job says what it waits on.** `Job.waiting_on` is `"load"` while
the weights are on their way and `"queue"` behind another job - a field
beside `status`, not a fourth state, because `cancel()` keys its instant
path on `queued` and a new state would have turned Cancel-during-load into
a multi-minute no-op. The label is set only when a load is actually
needed. Progress events carry `elapsed` (server clock, since the job
started running); the UI's ETA is the job's own pace over the fraction of
cells done, rather than the gallery's per-step median, so a run that has
slowed shows it. It runs from the end of the job's first progress event
(`etaFrom`: `(elapsed - e1) * (1 - f) / (f - f1)`), once a second has
landed, and there is none before that (2026-10-05): the first step carries
what no later one does - a pass's VAE encode of its source, the denoiser's
move onto the card under offload - and extrapolated from the job's start a
WAI pass said 28 s at step 1 for 9.5 s, FLUX.1's first image ~13 min. A
one-step cell's next event is the next cell's, whose own encode every later
cell carries too; a one-step job has nothing to estimate. The `done` summary goes into the
finish line, `#finish-note`, which sits outside the element `setBusy` hides
and which the next job's start does not clear.

The load's wait is on the page's clock instead, from the submission, since the
server's `elapsed` starts only when the job runs: "loading weights · 0:42",
ticking each second, "The first load of a session takes a few minutes." under
it, the bar indeterminate (a sliding stretch; still under reduced motion, no
`aria-valuenow`), and the topbar "<checkpoint> · loading…" - asked for as soon
as the POST returns - where it used to drop the name. The clock is an
`aria-hidden` span inside `#progress-text`: that is a status, and the whole
text rewritten each second was read out each second for minutes; "loading
weights" is said once, and the topbar and the title carry the clock. The
stream's error handler stops the ticker - it wrote "loading weights" over
"reconnecting…" every second while the topbar said the server was not
answering - and the snapshot a reconnect opens with starts it again. A cancel
during the load says the weights keep loading, which they do. Unload has one
writer, `syncUnload` (busy, another tab's job, Expand, the critic, a load, no
server), held with
`aria-disabled` and the reason as its title, which a press says (as Generate's
does: disabled, the reason was out of a keyboard's reach), and hidden while
Load is shown; an Unload refused by a load goes when the state turns ready.
The two take each other's place, so the press that hides one hands focus to
the other rather than to the page's body.
Under a budget the estimate is held to it, on the server's clock alone:
`deadline - started_at - elapsed`, "~1 min left (budget)" where that is the
sooner - the budget means "start no cell after", so the end is up to a cell
later.

While the strip is empty - its idle or held line - the viewer's line mirrors
the clock too (`viewerWaiting`, 2026-10-05; the review's U27 option): "Loading
weights · m:ss - the first load of a session takes a few minutes.", the clock
aria-hidden, one line. After Generate the eye is on the viewer, where "Nothing
generated this session" had stood for the minutes a first load takes. The
line's own nodes are kept and put back when the ticker stops - at the first
progress event after the load, not at the first frame: SDXL sends no frames,
and the clock would have frozen there.

**What a queued job is behind** (2026-10-05). A paused job keeps the gate, so
a job queued behind it from another tab read "waiting for the job ahead…" for
as long as the pause lasted, and the tab that could press Resume showed
nothing waiting; a job submitted during another tab's critique said the same,
with no job ahead at all. The manager now keeps who holds the gate
(`JobManager._holder`: a job's id, or `run_when_idle`'s `holder`, "critic" or
"expand", which the two routes pass, with the ticket an editor action was sent
with, `_ticket`; `JobManager.card()` answers ("job" | "critic" | "expand" |
None, ticket) for the status's `card_holder` and `card_ticket`, which Expand's
running line reads), and a queued job's `waiting_on` is read
off it: `paused`, `critic`, `expand`, or `queue` behind a running job, and
None for a lone job's first frame on an idle server. The label must follow
the holder, not only the moment a job arrives: the second of two jobs queued
during a critique went on saying "critic" for the whole of the first job's
run. So every job past its load and waiting at the gate (`_at_gate`) is
relabelled, with a status frame, when a job takes the gate - which also
catches a job that looked at the gate between its release and the next
holder's start and saw nothing ahead - and, by `_hold`, after a pause and after
a resume. Both run on the event loop, which owns the job table the worker must
not walk. The page says "waiting: job ahead paused" with a line on where to
resume it, and "waiting for the critic…" or "waiting for Expand…" with whose
they are; the paused tab lists the job behind it, from the server, and its
cell line counts what waits. What a queued job waits on, in words (the job
ahead paused, another tab's critic or Expand, the first load), rides in
`cellArgs` (`showCell`'s `note`), so the queue's redraw (`setQueue`) and a
Pause press say it again: they had wiped it under a progress line still
saying "waiting" (2026-10-05).

**The live regions say only what changes.** `#progress-text` keeps its coarse
state and is written only when its spoken text changes (`setProgressText`;
the load's `tick` writes "loading weights" and its aria-hidden clock itself),
and a single image's step count is an aria-hidden shown half (`.seen`) beside
a visually hidden "denoising…" (`.said`), as the load's clock is; the mirrors
and the title read `progressShown()`. The topbar's name was rewritten with the
same words at every 3 s poll, a new node in a live region all session, and is
now written only when it changes. A job's end and a job joining the queue are
said once on `#job-said` (A cell's failure is the cell's).

## date: is a local day

`date:` in the search box is a **local** day (`today`, `yesterday`,
`YYYY-MM-DD`) as a half-open range on `created_at`, converted to UTC and
formatted the way the rows are so the string comparison is the time
comparison. Not a LIKE prefix: that cannot use the index and would need the
UTC date, which is the wrong one for anything made in the evening.

## Re-render

**`POST /api/rerender` is the third producer of cells**, beside
`/api/generate` and `/api/grid`: the gallery's own filter (search, the
Starred tick and the checkpoint scope, taken server-side because the
browser holds one page) at another size, one cell per record with the
record's prompt, negative prompt, steps, CFG, true CFG, mu, references,
adapter and weight, refine source and strength, seed and variables, oldest
first. It calls
`check_budget()`, refuses more than `max_grid_images` matches by count
rather than truncating, refuses records from another checkpoint (steps/CFG/mu
are the checkpoint's - only `all_models` lets one into the selection),
prunes with `completed_cells` like a resume and 409s when nothing is left.
Same seed at another shape is a *new sample* - the latents are drawn at the
requested size - which the README says out loud; getting the *same picture*
larger is what the refine pass is for. A refined record is the exception
that proves it: it re-renders as a refine pass of its own source at the new
size, which is the same picture again. Until 2026-09-28 the cells
were built without `variables`: a re-rendered comparison lost its version
labels, which live nowhere else, and its sheet fell back to a flat,
alphabetical Prompt axis; a `{var}` grid lost its axes the same way. A
selection mixing records with variables and records without them lays out
like the same mixed filter over the originals - usually Starred over a
`{var}` grid plus some one-off prompts: the variable axis replaces the Prompt
axis, the plain records sit under `none`, and where two land in one cell the
sheet shows the newest (`axesFor` and `renderSheet` in `web/sheet.js`).
Keeping the Prompt axis for such a mix is a sheet change, not a re-render
one.

**Its refusal offers only a search that works** (2026-10-05). Refusing
another checkpoint's records, it said "add model:Anima to the search", the
id's tail. `model:` is a substring LIKE, so Animagine stayed in the selection
and following the advice gave the same refusal - the trap the profiles avoid
with exact ids. `api_generate.foreign_refusal` offers `model:<the whole running id>`
only where that id, lowercased, is contained in none of the foreign ids the
selection reached and has no whitespace, which would split the term; then
adding it leaves every one of them out. Otherwise it says only "Untick All
models", which is the exact scope (`_model_scope`). The whole id is not safe
everywhere - `Qwen/Qwen-Image` is contained in `Qwen/Qwen-Image-Edit` - which is
why the check runs against the ids actually selected rather than a rule about
ids. The count agrees with its verb ("1 of the selected images was made").

## Swap tags

**"Redo with tags swapped"** (2026-10-10): a night's keepers again with
`red hair, long hair` as `silver hair`, each with its own seed, size and
settings. Reuse, edit, Generate was the way, once per image, and a typo in
the twelfth prompt was found in the morning. `POST /api/swap-tags` is the
fourth producer of cells, beside Generate, Run grid and Re-render.

**What it acts on.** The picked cards where there are any (the gallery
selection, `ids`), else the filter as Re-render takes it, and the button's
title says which. Re-render, Swap tags and Mass refine read
their images through one resolver (`submission.resolve_selection`) and build
their cells through one helper (`grid.cell_from_record`), so what "the
selection" means and what a record carries into its new cell cannot drift
between them; Re-render's behaviour, refusals and tests are unchanged by
the extraction. Picked ids that have gone since are reported per image
rather than refused: the user picked forty, one was deleted in another
tab, and the other thirty-nine are still wanted.

**The server rewrites, the page shows.** The page could split commas, but
the rules - what counts as the same tag, a weight wrapper, prose, lettering,
an XML prompt - are `prompt_text`'s, already written and tested on the
server, and the result is what runs: the preview is the server's own
answer per image (`tag_swap.swap`), with the offsets of what it wrote in
for the page to mark, and the queue recomputes it. Every row matches the
original prompt at the same time, so two rows can swap two tags with each
other and a replacement is never matched again.

**Strict, no partial.** An image that lacks any find of any row is left out
as `missing`, listed with what it lacks. A "queue the ones that have some"
tick was planned and dropped (the coordinator's decision, after a
critique): a row applied to half an image is a different swap from the one
typed, and the preview already names the images to pick out.

**Whole items, then phrases.** Tags are compared whole (`_norm`: case,
spacing, `_`, `\(`), never by substring - `hair` is not in `red hair`. A
weight wrapper is looked through only where it wraps the whole item, so
`(red hair) and more` is not `red hair`; the replacement goes back inside
it and takes the item's underscores and escapes, since `silver hair` typed
into a list of `long_hair` tags should read like them. In prose
(`_is_prose`) a one-tag row also matches as a phrase, bounded only on an
edge of the find that is a word character (so `hair` does not hit
`hairstyle`, and a find ending in `)` still matches before a comma); never
inside lettering, which is text to render. Han and kana match as plain
substrings: there are no spaces to bound on. A many-to-one row in prose is
`by_hand` - which phrase keeps the replacement and what becomes of the
other's words is an edit, not a rule. Two one-tag rows whose phrases
overlap (`red hair` and `hair` in "a girl with red hair") cannot both be
written; the earlier phrase wins, and the row that lost every match is
`missing`, since queueing it with that row silently unapplied is the
partial swap strictness rules out. A weight typed round a find (`(red
hair:1.2)`) is looked through in prose as it is for items: it cannot be
the way to aim at weighted occurrences only, and the hint says it is
ignored. XML and JSON prompts are not
rewritten at all. None of this is an analyser rule: it cites nothing and
claims nothing about a checkpoint, and the preview is its safeguard.

**Many to one.** The replacement lands where the first matched item sits
in the prompt (not the first find of the row), and the others go with the
separator before each, so a `,\n` layout keeps its shape. A replacement
already present as an untouched item is not written a second time.

**Per image, never the whole job.** One image whose prompt the swap empties
or pushes past 4000 characters, whose refine source or adapter has gone, or
that another checkpoint made (through All models) is a status of its own;
the rest are queued. Two images that become the same cell at the same size
(`applied()`'s keys) are queued once, "same as #N". A cell the gallery
already holds is "already rendered", which makes the same swap asked twice
a no-op and a cancelled one resumable by asking again - there is no Resume
for it, since the dialog is the definition.

**Pinned to the ids, not the derived set.** The queueing request sends the
ids the preview resolved as `expected`; a selection that changed (a delete,
a filter that now matches more) is a 409 and the dialog checks again. A
cell the gallery holds at the press - the preview's "already rendered" ones
and any rendered since - is left out silently and counted (`pruned`): the
outcome the user agreed to, minus work already done. A refusal stays said
in the dialog's error line until the next edit or press: the check it asks
again for usually frees Queue at once, and clearing the reason then (as a
hold-reason is cleared once the hold ends) left a press that did nothing
visible but change the count (review, 2026-10-10).

**Every listed image carries its text.** The preview sends the swapped
text for an image that changes and the original for one that does not -
"missing: long hair" is read against the prompt that lacks it. The cap on
listed rows (200) is what keeps the answer small, not leaving texts out.

**One job per size.** Every job had one size until now - the snapshot, the
log line, the ETA and `run_id`'s definition read the first request's. A
mixed-size job would have needed each of them taught; one job per distinct
size keeps them all true, at the cost of more than one row in the queue.
The route answers the first job at the top level, as every submission
does, and the rest in `more`, which `startJob` registers as the page's own
and queues behind it. The snapshot's `swap` (the rows, as the server reads
them) names each job, "swap · red hair, long hair → silver hair". Each
job's finish line counts only that job's images, so it says which part of
the press it was ("Job 2 of 2 of this swap, at 1024×1024."): "Finished 1
image" after a press that queued six read as five lost.

**Where the button sits.** The head's control row is full at 1400 px, and a
button that came and went beside Select would move it. Swap tags… sits in
the footer row between Select and Bundles…, styled as they are, and is
always there: held, with the reason as its title and said on a press, when
nothing is picked and no filter narrows the gallery. The dialog's results
list has a fixed height and scrolls inside it, the count keeps two lines
and the error line one (`showError` hides an empty slot; `#swap-error`
keeps its line as `#gallery-error` does), so a check that lists more never
moves Queue under the pointer. The rows box has a fixed height (three
rows) and scrolls too: a row added used to land where + Add row was and
push it, the options and Queue down a row's height. A removed row keeps
its slot, hidden, until the pointer leaves the rows box - as a browser's
tab strip keeps its tab widths while the pointer is on it - so a double
click on a × removes one row, not that row and the next one that slid
under the pointer; the box's fixed height means the slots going moves
nothing outside it. The request's own limits (16 rows, 1000 characters a
box) are held on the page - + Add row aria-disabled with the reason,
`maxLength` on the boxes - rather than met as pydantic's text.

## Mass refine

**A pass of every keeper** (2026-10-10): the night's starred images at
1.5x, each refined from its own pixels, was a card's ⤢, Refine, Enter,
twenty times over. `POST /api/refine/batch` is the fifth producer of
cells: one cell per image, `refine = record.id`, one strength and one
scale for all of them, every other setting the image's own.

**Records to cells, not a loop.** Each record becomes one cell through
`grid.cell_from_record`, the helper Re-render and Swap tags build theirs
with, now taking any other request field by name (`**changes`, validated
with the rest, an unknown name a TypeError rather than a silent drop) and
a `label`, so what carries over cannot drift from theirs. A pass of the
record itself, not a re-run of the record's own pass: Re-render runs a
refined record again as a pass of the same source, which is reproduction;
here a refined record in the selection would be refined again, which is
seldom wanted, so **Skip refine passes** is on by default and an image
that is a pass is listed and left out until it is unticked. One record is
one cell in the resolver's order, oldest first; nothing loops over axes,
so `grid.expand()` stays the one cell loop. References never carry: a
pass's image is its source, and `_refuse_unusable_inputs` would otherwise
refuse an image with references on a checkpoint that takes none.

**Per image, never the whole job** (the coordinator's decision, after a
critique). Re-render refuses a selection over one bad record; a batch of
forty passes left nothing for one image whose pass at 2x is past the
pixel guard, whose few steps the strength leaves no step of, whose file or
adapter has gone, or that is already refined. Each is a status of its
own, decided before any request is built - the scaled size is checked
against `refine_max_side`/`refine_max_pixels` first, since pydantic's
"less than or equal to 4096" names no image - and the rest are queued.
Only a checkpoint with no refine pass refuses the whole of it, as it
refuses a single pass, in the same words.

**Another checkpoint's images only when asked.** Re-render refuses them:
their steps, CFG and mu are another schedule's. A pass starts from pixels,
so the single Refine dialog takes them, substituting this checkpoint's
profile, and Mass refine can too - but a starred Krea 2 image refined on
FLUX.1 under All models is easy to queue without noticing, so they are
listed `foreign` and left out unless **Include other checkpoints** is
ticked, a tick offered only where the answer counts any. Ticked, each
keeps its prompt pair and seed and takes the running profile's steps, CFG
and mu, no true CFG and no LoRA (the adapter was the other checkpoint's),
and its row says so. A same checkpoint's LoRA that has gone is
`lora_missing`, never silently dropped.

**The scale is capped by the largest image.** The single dialog opens
below the remembered scale where the source's pass would pass the guard or
the measured tight line (`openingRefineSize`). A batch has many sources,
and the largest decides: one 1.5 MP image in the selection at 2x spills
the card for every image after it. The server answers `largest` (the page
may not hold every selected record), the dialog opens only once the first
answer is in, at the capped scale, asked again there, and the passed-over
factor's button says why. A smaller factor never refuses anything; a
larger one pressed is the user's, and images it takes past the guard are
`too_big`, listed.

**One strength, refused as a list on the page.** A sweep of strengths is
a contact sheet of one image (the single dialog's Compare 3); over forty
images it is forty sheets. "0.1,0.2" is refused in the dialog without
asking the server ("One strength for a batch. A sweep is one image's
Refine."). The strength and the scale are remembered in the single
dialog's `REFINE_KEY` - what was last run on this checkpoint is the same
number whichever dialog ran it - and a capped opening scale run as it
opened is not remembered as the user's choice, as `rememberRefine` has it.

**The start sigma is a range.** On a shifted flow schedule one strength
starts a larger pass at more noise (FLUX.1: 0.1 is 47% at 1.5x of 1 MP and
75% at 2x), and a selection of several sizes has several. The preview
reads `start_sigma` once per distinct pass as `applied()` runs it - size,
steps, mu, strength - and says the lowest and highest; any reading None
(before a load, SDXL) or more than eight distinct passes (each reading
builds a scheduler copy) says nothing. Advisory, never a refusal.

**Time, honestly unknown.** The price is summed per size from that size's
refine history (`refine_cost_ms`), and one size without history leaves the
whole estimate out, said as "no estimate: N sizes have no refine history"
rather than a total that silently prices part of the batch.

**One job per size, pinned to the ids** - as Swap tags, for the same
reasons: every reader of a job (snapshot, log, ETA, `run_id`) assumes one
size, and a selection that changed between the check and the press is a
409 rather than a batch queued unseen. The snapshot's `refine_batch` (the
scale) and `strength` name each job, "refine ×1.5 at strength 0.3"; its
`refine` is None, since "refine of #12" named forty passes by the first.
Each cell's label is its source's `#id`, which the progress line and a
failed cell's report show: a source deleted mid-run fails only its own
cell, named.

**Where the button sits.** Beside Swap tags… in the footer row, always
there, and held with the reason where nothing is picked, no filter
narrows the gallery or the checkpoint has no pass - the hedge until a load
answers, `REFINE_NONE` once it has, refreshed with every status. Its
dialog is Swap tags' shape: fixed-height results, a two-line count and a
one-line error, so a check never moves Refine; "Sending…" holds Escape and
Cancel while the POST is out, as the single dialog does.

**A filter that grows under the dialog** (2026-10-10, review). Pinning
`expected` to the answer's ids has a cost on a filter scope: a job that
keeps landing images the filter matches made every press a 409 ("the
selection changed"), since each refusal's fresh check was stale again by
the next press, and the scope line still said the count it opened with.
The modal dialog cannot change the filter, so the gallery's own `total` is
the filter's count; when it changes under an open dialog (the job-landing
`prependFinished`, a reload) the answer is dropped - Refine reads
"Checking the images…" and a press waits - and asked again, and the scope
line takes the latest answer's `total`. A press can still race an image
that lands between the poll and the click, refused once with the reason,
never repeatedly. Swap tags pins the same way and takes the same fix.
Pinning a filter's ids at open instead was rejected: the dialog says "the
images matching …", and an image that matches when Refine is pressed
belongs to it.

**Small holds that went quiet.** Steps is a text box: a number box reports
text it cannot parse ("1e", "-") as an empty value, which here means
"each image's own", so the pass ran with settings the box did not show;
the `^\d+$` test now sees what was typed. `syncSubmitButtons` re-syncs
this dialog's Refine as it does the single dialog's, so a server that
stops answering holds it at once rather than at the press. The 409 that
says every image already has the pass is kept for the case where that is
every image's status; mixed with images left out for a fixable reason
(too big, a LoRA gone) the refusal is the 422 that counts them all, as the
dialog's hold words it.

## More seeds

**"Do four more of these"** (2026-10-10): a Batch 4 that turned out well
wanted four more like it, which meant Reuse, add 4 to the seed by hand,
Generate - and the seed added wrong rendered an image the gallery already
had, or skipped four. `POST /api/more-seeds` is the sixth producer of
cells: N more images of each selected batch, its seeds following on, every
other setting the batch's own.

**A batch, not an image.** The unit is what a card's hover shows: one run's
images that agree on everything but the seed. Picking one image of a
Batch 4 takes all four, and picking two of them asks for one set more, not
two - the user picks "a grouping", and the grouping is the batch whichever
of its images was clicked. A selection over several batches (a grid's
cells, a few separate Generates) gets N more of each. The key is the
lineage (run, else job for a row from before run ids, else the image
alone), the checkpoint, the quantizer, the size and the resume key's
elements, read off the record as `completed_cells` reads a row, so no
request has to be built to group a thousand records.

**N counts images per batch** (the user's choice, 2026-10-10, over N more
Repeats of the original): "4 more" is four images whatever the batch was
made at. The alternative made "1 more" of a Batch 4 x Repeat 2 eight
images, which is not what the number says.

**Seeds after the batch's highest, never one already made.** Adjacent seeds
are no more alike than random ones; following on is what keeps the batch
readable as one range ("seeds 100–107") and what the user asked for. The
start is the highest seed the gallery holds of the whole batch - picked or
not, earlier extensions included, since they join its run - and every seed
the gallery holds at the batch's settings and size, under any run, is
stepped over: another run that used seed 104 by hand would otherwise be
rendered twice. So are the keys the live jobs will write
(`jobs.planned_keys()`): a second press while the first is still queued
used to compute the same seeds and render the same four images again. The
seeds one batch takes are added to what the next batch of the same settings
steps over, for two runs of one prompt picked together. A deleted image's
seed is not known and can come back if it was the batch's highest; one in
the middle never can.

**The batch's own Batch, recorded.** One generator per image makes the
Batch irrelevant to the pixels but not to the card: the user's Batch 4 at
1 MP is a choice about memory and speed that "keep batch settings from the
original run" asked to keep. Rows never recorded it - a Batch 2 x Repeat 2
and a Batch 4 are the same four rows - so `batch` is now a column and a
chunk key, the image count of the call that made the image. Rows from
before it run at the panel's Batch (the user's choice over Batch 1 or a
guess from equal durations), said per batch; either is held to
`max_batch_pixels` at the batch's size, said too. The batch's Batch is the
largest recorded on its images, not the newest's: each image records its
own call, and 3 more at Batch 2 is a call of two and a call of one, so the
next press read Batch 1 off the image of the second call (smoke, before
release). A run has one Batch - `run_id` hashes it - so the largest is it.

**Joining the run** (the user's choice, over a run of its own). Each job is
stamped with the run its batches extend, as a resumed grid's is, so the
new images and the old are one set to the hover, the Runs picker, `run:`
and Show sheet. The hover groups by run for it (by job before): a resumed
grid's cell reads as one set now too. A run_id is a definition's hash, and
an extension is not in the definition; nothing reads it as one - resume
keys on settings and the Runs line counts what carries the id. One job per
run and size, not per press: a job has one run id (and one size), and
per-cell run ids would have needed `_save`, the snapshot and the finish
line's Show sheet taught that a job spans runs. A press over ten separate
single images is ten jobs, each named and each a part of the press's
finish line ("Job 2 of 10 of these more seeds").

**Another quantizer is said, not refused.** A batch made under fp8 extended
under int8 is new seeds either way - new pictures - but its images were
all made one way, so the row notes it. Another checkpoint's batch is
refused, as Re-render refuses it: its steps, CFG and mu are another
schedule's.

**The button and the dialog** are Mass refine's shape: More seeds… after
Refine… in the footer row, always there and held with the reason; the
dialog opens on the first answer, its list and lines of fixed height, the
count a text box refused in words on the page, Queue held until the
answer is to what the box says, the ids pinned by `expected`. The count
opens at the number last queued, else 4. Not "More…": in a row of
buttons that reads as an overflow menu.

## Delete unstarred

**The gallery's one bulk delete is the inverse of the card's.** A night
makes hundreds of images and a few of them are worth keeping, and deleting
the rest one card at a time - each with its own confirm - is a chore that
was being skipped, leaving galleries of hundreds. The star already existed
as the gesture made while looking (`F` in the lightbox), it is
non-destructive, and it scales with what is kept rather than what was made;
so the page offers `GET`/`DELETE /api/gallery/unstarred?search=` - the
search box read through `strict_scope`, plus `favorite = 0` less what a
starred image was made from, plus the gallery's checkpoint scope unless
`all_models=true` - which is
`clear --where ... --keep-favorites` over HTTP, and shares `delete_matching`
with it. Rows and their files only: no orphan sweep and no vacuum, for the
reason the CLI's `--where` path has none. Since 2026-10-11 it moves the
rows to the recycle bin instead, files untouched, and counts live rows
only; see "Recycle bin: why the per-image confirm went".

**The hazard is inverted with the gesture, and the gate is built for the
inverted one.** A per-card delete goes wrong by taking the wrong image; a
sweep goes wrong by taking the wrong *scope* - a run not yet triaged, or the
whole gallery when one run was meant. So the confirmation is not a yes/no:
`GET` returns the counts and the CLI preview's sample of prompts and jobs,
the dialog shows them, and asks for the number of images to be typed back.
Typing "405" when "12" was meant is the moment the wrong scope is noticed.
The `DELETE` then carries that number as `expected`, and
`gallery_delete.delete_matching` compares it against a `COUNT(*)` taken under the
write lock before the `DELETE ... RETURNING`: a sweep that lands more
unstarred images while the dialog is open (the queue keeps running while
one triages) is answered 409 with both numbers rather than swept up on a
count confirmed for fewer. `expected` is required, not optional - the gate
is the route's contract, not the page's manners. `clear --where` pins the
count its prompt showed the same way (net of what `--keep-favorites`
keeps); it called `delete_matching` without it, and a run still being
written while the prompt waited lost its newest images too (review
2026-10-12, R34).

**The scope is strict, judged at the click.** The button shows whenever
the filter matches something and the Starred tick is off (with it on,
every match is a keeper). Whether the box is a strict scope is the
server's call: free text is refused with a 422 that says what to do,
rather than the page guessing at the grammar with a second copy of the
key list. An empty box is the whole gallery, which is why
`matching_preview` and `delete_matching(keep_favorites=True)` now accept
an empty scope and `clear.py` refuses a blank `--where` itself - the
"whole gallery is `clear` without --where" message is the CLI's manners,
not a storage invariant; the invariant that stayed is that a delete with
*no* predicate at all still raises.

**The grid is brought up to what the sweep reaches** (2026-10-05). A tab that
had not shown another tab's newest images counted and deleted them: the sweep
is the server's whole scope, while the page holds what it last loaded, and the
images it missed were the newest and unstarred - the ones the other tab was
likeliest to star next - so the number typed back was not the grid's. Outside
the sheet the press first awaits `prependFinished(0, { now: true })`, the
finished-job path under the same filters; `now` skips its wait for the pointer
to leave the gallery's column, since the press was on the head's own button or
a key and the modal question follows within one round trip, so no card moves
under a pointer about to press it. A page that does not join up is loaded again
from the top, as before. The preview is then asked with `newer_than`, the
grid's head before the update, and answers `newer`: how many of the images the
delete takes have a higher id, by the delete's own `_UNKEPT` predicate in the
same scope. The dialog says "2 of them were made since this page last loaded;
they are at the top of the grid now" - at the top, because a page scrolled down
keeps its view - or "the newest are at the top of the grid now" where the
reloaded page is all newer and more pages follow (`freshLine`). The page used
to count them itself, leaving out only the six oldest kept sources the preview
names, so a new image that a starred pass was made from was counted as one the
sweep takes; the page never counts these now, since it cannot see which ones a
starred image keeps. A scope with nothing left at all - another tab deleted
it - says "Nothing to delete: no image is left <scope>.", not that every image
is starred, and wherever a reload hides the focused button (that case, a sweep
that took every image, a refusal, a failed preview) the search box takes focus
(`keepUnstarredFocus`) rather than `<body>`. The sheet is left alone, since it
lays one run out whole, and the `expected` pin is unchanged: an image that
lands while the dialog is open is still refused, not swept.

**A star keeps what its image was made from.** Refine a source, star the
pass, sweep the rest: the sweep used to delete the source's PNG, the kept
pass then read "refine of #63 @ 0.1" for good and could never be made again,
and nothing before the click said so. The source is part of the pass's
reproduction set, and a klein render's references are part of its, so
`keep_favorites` keeps them with the stars: one predicate in `gallery_delete`
(`_UNKEPT`: `favorite = 0` and not the `refine` or a `json_each(reference)`
value of any starred row, one level deep, neither arm able to yield the NULL
that would turn `NOT IN` into NULL), shared by `delete_matching`,
`clear_history` and the counts. The count gate moved with it, or every
sweep would have answered 409: `matching_preview` returns `sources_kept`
(and the first few as `sources`), the route's `unstarred` is `records -
favorites - sources_kept`, and the CLI's two "Would delete" lines and
`summarise()` take the same field. Sources stay ids, and the dialog names
them: "keeping 4 starred and #63, which a starred image was made from".
A per-image delete of a source is still the user's to make, but its confirm
first asks `refine:<id>` across every checkpoint and says which pass could
not be made again; a record whose source has gone carries `refine_missing`
(looked up by `list_images` and `GET /api/images/{id}`, not stored), and the
lightbox's facts read "Refined from #63 (deleted) @ 0.1", its comparison
dead, saying "source #63 deleted".

**The dialog describes the scope, not three prompts.** The sample prompts
are the first three alphabetically, and say nothing of how many runs are
about to go or from when, so `matching_preview` also counts the runs the
delete reaches into and the first and last creation times among what it
takes ("From 23 runs, made Oct 4, 04:32 PM to Oct 4, 04:40 PM"). With All
models ticked the scope names every checkpoint it reaches from the server's
own `models` (six at most, "6 or more" at the cap), as the narrow scope
names the one. While a job is queued or running on the server - this page's
or another tab's: the preview's `job_active`, `jobs.busy()` - the dialog says
a landed image refuses the sweep. When one does, the 409 gives both numbers
and, where `jobs.busy()`, adds that a job is still running and may add more:
press again or let it finish. The page then reloads the gallery and the Runs
picker, so the count beside the message is the server's new one. Both used to
follow this page's own `busy`, and so said nothing exactly when the job was
another tab's.

**Everything it says, it says by the gallery.** The outcomes - "Deleted 12
images, keeping 3 starred.", the wrong number, the free-text refusal, the
409 - go to `#gallery-error` under the gallery's head (`showError`'s `slot`),
as do a failed gallery page, Re-render's submit errors and a card's
refusals. At QHD they used to land in the panel's line, 2,400 px from the
button; a sighted user saw a click do nothing. One slot per message, since
both are role=alert; the lightbox mirror is the same. `resetGallery` clears
the slot, so the sweep writes its outcome after the reset it starts.

**The page forgets by id.** The `DELETE` returns the ids it removed and
`forgetImages` - shared by the card's delete, the lightbox's, the sweep and
a viewer tile whose image would not load - drops them from every list and
element on the page in one pass, the panel's references included, then the
sweep's `resetGallery` reloads the (now mostly empty) page from the top for
the server's count and `loadRuns` the picker's. A single delete keeps its
place: the card becomes a tombstone ("Deleted · #N", dashed, no
pointer events, no `data-id`; by id, as the page names images, since the
cells of a sweep share a seed) and a viewer tile one of the tile's frozen
size - removed, the next card slid under the pointer that had pressed ⌦,
its own ⌦ lit, and an emptied viewer shrank the page by 537 px - and focus
on what went moves to the next image in the strip, else the one before.
A viewer whose last image went, with no job to fill it, says what it is for
again ("#9 was deleted. Generated images appear here.") in place of its
tombstones - a frozen box with the placeholder beside it read as two
leftovers - held at the strip's height so the gallery still does not move,
until the next run's first image clears it; the results viewer counts its
live tiles, not its slots, for the 48-tile cap and the "and N earlier" note.
The per-image confirm names
the image by id with the lightbox's settings line (`settingsLine`), since
a sweep's cells share a prompt and a seed. Asking `refine:<id>` for the
passes made the confirm wait on a round trip, and a double-click asked
twice and sent two deletes, both answered 200 since `delete_image` looked
the row up outside the write lock; a delete under way now ignores a second
press of the same image (`deleting`), and `delete_image` looks up and
deletes under the one lock, so a second delete is "unknown image". In the lightbox a delete says
which image went in its line until the next key or press - one listener
pair per note, both taken off by whichever comes first and by the next
note (an AbortController: `{ once: true }` took off only the one that
fired, and a run of deletes left the other behind each time) - and a delete
of the lightbox's only image closes it with focus where the walk came from
(`closeLightboxAfterDelete`: the viewer's held line or tombstone, the
grid's current card, else the search box), where its focused Delete went
with the dialog and left focus on `<body>`. A delete of
the last image loaded, with more on the server, pages the next one in
(`pageInBehind`, "Loading next…" meanwhile) rather than stepping back onto
an image already judged. A delete answered "unknown image" - another tab
took it, or a reattached job had replayed it - forgets it with a note. The
server does its part: the delete routes call `jobs.forget_images`, and a
job's snapshot replays only its live records, with `deleted_count` beside
an unchanged `image_count` (what the night produced), so a reload during
a run no longer brings a deleted image back as a broken tile. Both questions are
the page's own dialog (`#confirm-dialog`, `openAsk`), beside what was pressed
(`placeAsk`) and inside the window's client box -
`documentElement.clientWidth`/`clientHeight`, since `innerWidth` counts the
scrollbar, and a dialog placed for a card in a row's last column, or for the
lightbox's Delete, ran 7 px under it: the browser's confirm and prompt were a light box at the top
of the window, far from the card whose ⌦ asked, and held the whole page -
a headless run waited on one for good. Cancel has focus, or the count's
box, where Enter answers; Escape and a press outside are Cancel, and the
document's own keys are off while it is up, since its Escape also closed
the lightbox under it. The smokes stand in for it through `askConfirm` and
`askCount` (put back as `confirmDialog` and `countDialog`) to drive Delete
unstarred: a wrong number first, then the right one, then the free-text
refusal, checking that nothing went until the right number was typed - and
drive the dialog itself, Escape and Delete, over the lightbox and with a
typed count; `smoke-delete.json` takes the rest to the end,
a starred pass's source and a landed image's 409 included.

**The per-image confirm has gone; the bin is the undo** (2026-10-11). See
"Recycle bin: why the per-image confirm went". What is said above about the
confirm's settings line, its passes and references now holds for the bin
view's Delete permanently, and Delete unstarred moves what it takes to the
recycle bin, keeping its typed count.

## Recycle bin: why the per-image confirm went

**The confirm guarded a step that could not be taken back, and so was
asked every time.** Triage is a long run of ⌦ presses - most of a night's
images go - and a question per press is answered without being read: the
dialog that was meant to catch the wrong image trained the hand to press
Delete twice. An undo catches the wrong image after the fact, when it is
seen, at no cost to the presses that were right. So ⌦, the result's
Delete and the lightbox's Delete (and its Del key) move the image to a
recycle bin with no question, and the note they leave - "Moved #12 to the
recycle bin." - carries a Restore; a run of them gathers into one note. The
questions that remain are the ones whose step cannot be undone: Empty
recycle bin, the bin view's Delete permanently, and Delete unstarred's
count. The lightbox's Del, unconfirmed, ignores a held key's repeats, or
holding it would bin a run of images.

**Binned is a column, and files stay where they are.** `binned_at` is ISO
UTC text like `created_at`, NULL while live. Text rather than a 0/1 flag,
so a bundle merged into an older table gains it through the same ALTER
fallback as any unlisted column (anything unlisted becomes TEXT), and so
the time can be shown or one day drive a purge; never NOT NULL, so an
older database or bundle reads None and is live rather than refused. The
PNG and its thumbnail do not move to a bin folder: a bundle's inventory
takes only `images/*` and `images/thumbnails/*`, the orphan sweep and the
reindex would have to learn a second place, and a Restore is then one
UPDATE. Bin state is SQLite's alone, as the star is - never in the PNG
chunk - so a PNG-only reindex, or an older build importing a newer bundle,
brings a binned image back live, as a PNG-only reindex loses a star. That
is accepted: the image is not lost, only out of the bin. A repeat bundle
import keeps this gallery's bin state for a row it already holds, as it
keeps its star.

**Gone for every live read, present for every read about what exists or
was made.** One rule decides each query. Listings, selections, references,
refine sources, stars, runs and stats leave the bin out (`storage._LIVE`);
the orphan sweep, reindex, bundles, Resume's and More seeds' held keys,
lineage, provenance and the cost and size samples read every row. Filtered
to live rows, the orphan sweep would delete binned PNGs as untracked and a
reindex would add them back as new rows; a binned cell left out of
`completed_cells` would be made again by the next Resume, which is the
render the user just rejected. Emptying the bin makes the cell eligible
again, as a delete always has. More seeds follows on past a binned seed,
so a Restore never brings back a duplicate. A binned image is refused as a
reference or a refine source until it is restored (`reference_paths`,
"is in the recycle bin; restore it to use it"), which reaches a
submission, a queued cell and `generate.py` with no other change; a job
queued before the bin fails that cell, as it did when the image was
deleted. A pass whose source is binned reads `refine_missing` as before,
and `refine_binned` beside it says which, so the page can say "(in the
recycle bin)" and offer the Restore rather than "(deleted)". A binned star
keeps `favorite = 1`, so a Restore brings the star back, but while binned
it is unkept and protects nothing: the star is a promise to keep, and the
bin is the user taking it back. It cannot be starred again until it is
restored (404), which also keeps a stale tab from re-starring it there.

**`DELETE /api/images/{id}` stays permanent.** The bin is new routes
(`POST /api/images/bin`, `/restore`, `GET /api/gallery/bin`, its
`/summary`, `DELETE /api/gallery/bin`), so the tests and smokes that delete
an image by a raw `DELETE` keep meaning what they say, and the bin view's
Delete permanently is the old route. `GET /api/images/{id}` and `/thumb`
answer a binned image 404 "unknown image (#N is in the recycle bin)" unless
`?binned=true`: the page's many readers of "unknown image" mean "gone from
the gallery" and stay right, and one that wants to offer a Restore reads
the reason after it.

**No `bin:` search key.** A key would reach `strict_scope`, and through it
Delete unstarred, `clear --where` and `keep_rates`; it would make
`split_search` carry a flag, and the page's key list would have to learn
it. The bin's view is a route of its own, every checkpoint's at once (the
bin is one place, and Empty empties all of it), newest id first, narrowed
by the search box as the gallery is; Starred does not apply. Picked images
are a different matter: an id binned after it was picked is reported per
image by Swap tags, Mass refine and More seeds as `deleted`, "in the
recycle bin", never refused as a whole.

**Delete unstarred bins, and keeps its count.** Every page-side delete is
then reversible. The typed count stays: its preview of what is kept still
catches the wrong scope, which a bin does not make harmless - a whole
gallery restored by hand is not an undo. It counts live rows only, so what
is already in the bin is neither counted nor binned twice; the CLI's
`clear --where` keeps reaching the bin and deleting for good, and a binned
star is not kept there either.

**Empty is fail-fast and pinned to its images.** `empty_bin` is whole
maintenance, refused at once on a busy gallery as `delete_matching` is,
and it compares the count the confirmation showed under its lock, so an
image binned in another tab while the dialog was open is answered 409
rather than deleted on a number confirmed for fewer. The count alone was
not the set: a Restore and a bin of a different image in another tab kept
it equal, and the newly binned image was deleted for good under a
confirmation that never covered it (review 2026-10-12, R6). The summary
also answers `pin`, 16 hex digits of a sha256 over the sorted binned ids,
which the page and `clear --bin` send back and `empty_bin` recomputes under
the same lock: a digest, not the ids, because a bin can hold thousands. Bin and restore are
per-record writes and wait, as a star does. The confirmation says what
cannot be made again: the binned images a live one names as its source or
a reference (`sources_in_use`), and that a running job's cell using one
will fail. It still deletes them - a warning, not a guard - and there is no
automatic purge: nothing leaves the disk unless asked. `clear --bin` is the
same call from the terminal, previewed with the same summary.

## The morning after: runs and finished jobs

**The morning after has two sources.** `GET /api/runs` groups the gallery
by `run_id` (`gallery_reports.runs`: images, favourites, distinct prompts, jobs,
first/last timestamp, and the *oldest* image's prompt and seed - the base
seed a resume needs); it outlives the process and answers "how many and
when". `GET /api/jobs?include_finished=true` adds the terminal jobs from
memory (`jobs.finished()`, newest-finished first, `image_count` in every
snapshot) and is the only thing that answers "why" - a cancel, an OOM, the
cell it stopped at - for as long as the server is up. The UI reads the
first into the Runs picker over the gallery (pick one: `run:<id>` search
plus the sheet) and the second into a banner under the panel when nothing
is live, gated to a job finished within 24 h and announced once per job
per browser (`localStorage`, every access in try/catch). A job this page
watched to its end is remembered as announced in the `done` branch - its
finish line said it - and never in `onerror`, so a lost stream is still
announced after a reload; `setBusy(true)` hides the banner without
remembering it, since it described an older job beside a newer one's finish
line and its Show run opened that older run. A stopped job says where it
stopped - "stopped at cell 6 of 12", or "never started" - where "(6/12
cells)" read as six done; a failure stop's error says it already. In the
job's own units (`unitWord`): a plain job stopped "at image 2 of 3", or "(2/3
batches)" - "cell" is a grid's word. A job that
ended at the load is skipped while the status still gives the same detail.
It says what is left of the job, from the snapshot's `deleted_count`: "6
images (6/6 cells), 4 since deleted", or "all since deleted" with Show run
hidden - it had counted and offered deleted images, and Show run opened a
sheet of one. A grid the night did not finish (cancelled, out of time,
stopped by its failures) offers Resume: the sheet's Resume this run
(`resumeRun`) with the job's size, the panel's grid made ready, Resume ticked
for this session, and the preview saying what is left - or, where it would
skip nothing, that the panel holds another grid. The base seed is the run's,
from /api/runs (its oldest image's), as the sheet's is: the snapshot's
`seed` is the first cell the job had left, and a job that was itself a
Resume, its first seeds done, carried one past the base - the panel then held
another grid, with another run id, and Run grid rendered cells the night
never had. The finish line's Retry and Resume send their payload's own seed
for the same reason, and the POST's answer only where the payload gave none.
Not for a re-render, a refine sweep or a prompt comparison (the snapshot's
`comparison`, set from `prompt_variants`), none of which the panel's grid
makes - the sheet's Resume this run refuses a comparison too - and not for a
job with no image of its own left, which leaves the preview nothing to skip
and so saying the panel's grid was another.

**The picker is grouped, and follows the search box.** Every plain Generate
is a run of its own (`run_id` comes from the cell keys), so in an afternoon
of single images the picker became a click log: twenty lines that differed
in the minute, and the night's grid pushed off a list of twenty. `loadRuns`
asks for the server's 200 and lays them out in two `<optgroup>`s: "Grids and
batches" - more than one image, or more than one job - labelled with the
image count, the distinct count where it is lower and "N prompts" where
there are several; then "Single images", the newest eight, each by the
second it was made and its seed, which is what tells two of them apart where
the clipped prompt did not. Nothing new is asked of the server: `/api/runs`
returns images, prompts and jobs per run already. The select used to go
back to "Runs…" the moment a run was picked, leaving `run:949bbfa5c2c0` in
the box as the only label of what was on screen; it now names the run the
search box shows (`syncRunList`, from `resetGallery` and the end of
`loadRuns`), however the box got it - a pick, a card's ⊞, Show run, the
sheet's return, typing - and "Runs…", a hidden and disabled first option,
for anything else or a run it does not list. Every line of a session still
read alike - "6 img · A weathered lighthouse…" for a steps × CFG grid, a
batch and a refine sweep - so `/api/runs` now says what varied: each
setting's count of distinct values over the run where it is above one
(`RunSummary.axes`, from `gallery_reports._RUN_AXES`: steps, cfg, mu, true_cfg, the
adapter and its weight, strength, reference, seed and a comparison's
variant label), and the source every image refines with the strengths it
ran at (`refine`, `strengths`). The picker names the run by them ahead of
its prompt (`runAxes`): "steps×CFG 3×2", "steps ×8 · 5 seeds", "strength
0.05-0.15 of #37", "comparison · 3 versions", "4 seeds".

Closed, the picker has a width of its own, `--runs-w` (190 px; 240-360 px
from 1800 px), cut with an ellipsis, and the picked run's whole label is the
select's title. A select is as wide as its longest option, and with these
labels every one ran to the old 300 px cap: at 1400 the search box was
squeezed to 175 px and its placeholder cut, and at 2560 a picked run's own
label was cut mid-word. The open list is as wide as its labels need. A
label's prompt is `promptGist`'s, one line with an XML prompt's tags dropped,
before `clip` (The browser's queue).

**Browsing the closed picker opens nothing** (2026-10-05). On Windows the
arrows change a closed select at each press, and each change opened that
run's sheet - a 1024-image gallery fetch and a scroll of the page - so reaching
the tenth run opened nine (WCAG 3.2.2). `runListKey` marks a change made by a
browsing key's own default - the arrows, Home and End, the Page keys, or a
printable type-ahead key, none with Alt, Ctrl or Meta - and the mark is gone by
the next task, so only that key's change is held. The change is noted
(`runPending`, which `syncRunList` keeps across a rebuilt list, the title
following it) and opened on Enter or as the picker loses focus
(`commitRunList`); Escape puts back the run shown. A pointer pick, or Alt+Down
and Enter in the open list, comes with no such key and opens at once
(`runListChange`).

## The gallery is the running checkpoint's

**`GET /api/gallery` is scoped to `current_model_id(settings)` unless `all_models=true`**,
and so are `GET /api/runs`, `GET`/`DELETE /api/gallery/unstarred` and
`POST /api/rerender` - every route that reads "the gallery's filter". The
page's All models tick, beside Starred and unpersisted like it, is the one
switch. Steps, CFG and mu are the checkpoint's, so under a gallery that has
seen three checkpoints the grid was mostly settings that could not be reused,
each under a Reuse button that warned, and the working set - this checkpoint's
images - had to be found with a `model:` term remembered every visit.

The default is the *server's* rather than a `model:` term the page adds for
two reasons. The page learns the checkpoint from `/api/status`, and
`describe()` reads CUDA memory, which behind a cold load answers minutes
late - a client-side scope would paint every checkpoint's images first and
hide them later, or show nothing until the load ended. And `model:` is a
LIKE substring, on purpose (`model:flux` for the family), where the scope
has to be exact: a full id matches every longer id that extends it, and
`krea/Krea-2-Turbo` is one suffix away from a catalogue entry that a
substring scope would show under it. So `storage.list_images` takes
`model_id` as a predicate of its own, kept out of the `where` list the FTS
fallback rewrites, and `AppServices.model_scope` turns the flag into
`current_model_id(settings)` or None.

**`other_models` is on every page.** `list_images` answers a third number:
how many rows the *rest* of the filter matches on other checkpoints, one
more `COUNT(*)` over the same predicates with the scope inverted, taken
after the FTS fallback has settled so it counts under the same text
predicate the page did. The tick reads `All models (+40)` from it, and it
is what tells "this run made nothing" from "this run was made on the other
model": a `run:` typed or picked for a run from another checkpoint shows an
empty grid under the scope, and the tick beside it saying `+96` is the whole
explanation. 0 once the tick is on.

**Delete unstarred takes the scope because the grid does.** Its dialog
counts server-side, and with the scope on the grid and off the count, an
empty box would have confirmed "delete 405 unstarred images in the gallery"
against a screen showing 40 - the wrong-scope hazard the typed-back number
exists to catch, built into the number itself. `matching_preview` and
`delete_matching` take `model_id` as one more exact predicate, and the
dialog names the scope from the server's own `models` list
(`... in the gallery (Krea-2-Turbo only)`). Re-render's refusal of another
checkpoint's records stays, now reachable only with All models on; the
Runs picker follows the tick because a run picked from it opens as a sheet,
and a run the grid would not show is an empty one.

`status.js` re-asks for the gallery and the runs when the selected model or
its revision changes: the scope is server-side, so a runtime switch or a
server restarted under another checkpoint while the page stayed open would
otherwise keep showing the old one's images. The revision also catches
A-to-B-to-A changes between polls. The
`tools/fake_server.py` gallery plants two images under a checkpoint it is
not claiming, one starred, so `tools/smoke.json` can watch the tick reveal
them, the badge count them, the Runs picker follow and the dialog name the
scope; `.gallery-head`'s search box has a 140px flex basis for the same
change, because a row break is decided from each item's basis before
anything shrinks and the tick otherwise pushed Delete unstarred onto a
second row at laptop width.

## Paging by id

The gallery pages **by id, not by offset**: `GET /api/gallery?before=<id>`
returns rows older than that id and `has_more`, which the server knows
exactly by fetching one row past the page. OFFSET over a table a sweep is
inserting into repeated the tail of one page at the head of the next and
skipped as many older rows for good; `offset` stays for scripts. The
browser's anchor is the last record it holds, so a delete moves nothing.

The grid pages itself: an `IntersectionObserver` on the "Load more" button
(`moreWatcher`, made by `armMore` from `app.js`, and rooted on the gallery's
own column with Gallery beside - The gallery beside the viewer) calls
`loadGallery` as the button comes within 400 px of the viewport, or of that
column, so scrolling to the end keeps going and the
page is usually there before the button is. The observer reports a *change*
in visibility, and a page that does not push the button past the fold - a
wide grid, the tail of a short run - would leave it in view with nothing left
to fire, so `watchMore` re-observes the button after every page that lands:
observing afresh always reports the current state once. It is not re-armed
after a page that failed - re-arming there would fetch again from the server
that just refused - which is why the button stays: it is the retry, and the
visible sign that older rows exist. Hidden in the sheet and once `has_more`
is false, it never intersects. `armMore` makes the observer again whenever the
gallery's scroller changes (rooted on the window, a button inside the beside
column's box counted only once the box showed it, and the 400 px of warning
were gone); observing afresh reports once, so a button already near the new
root's edge fetches, unless the last page failed (`moreArmed`), as in
`watchMore`.

Where the local day changes between one record and the next, `loadGallery`
puts a line across the grid (`dayBreak`, an `h3` spanning `1 / -1`), judged
from the card already last in the grid - its day is the card's own
`data-made`, which a tombstone keeps - so a page boundary is no day boundary.
It puts one over the first card too where that card's day is not today's:
drawn only between days, a gallery whose newest images were yesterday's had
no heading over them, read as today's, and the first line on the page named
the older day under them. `prependFinished` draws it the same way at the
head - between two new cards, or between the last of them and the old head,
read off the grid's first card, tombstones included (A finished job joins
the gallery at its head) - since a job that ends after midnight under a page
loaded the day before ran the two days together. It carries no `data-id`:
paging, the deletes, the lightbox and `.card` counts never see it. A day whose cards are all deleted keeps its line over
their tombstones, which hold their slots so that nothing moves under the
pointer that pressed ⌦ - taking the line out would move every card after it.

**A gallery with nothing to show says so, and so do new images it hides.**
On a first visit the gallery was a blank screen under a disabled "No runs
yet": nothing said it fills as images are made. A page with total 0 now says,
under the head, what fills it and offers the authors' Examples (the first
run's own button's handler), or under a filter that nothing matches it, and
either way what the All models tick would add (`syncGalleryEmpty`; never
over tombstones, which say it, nor in sheet mode). And a comparison's or a
sweep's sheet keeps the gallery scoped to its `job:` after it opens: a plain
Generate afterwards saved its image where the sheet could not show it, the
count unchanged and the search box still reading `job:…`, as though nothing
had been saved. Where a finished job's images fall outside the filter - that
scope, a search, Starred - the gallery's line counts them, across jobs while
the filter stays the same ("5 new images outside this view"), with Show all,
which empties the search, unticks Starred and closes the sheet
(`noteOutsideView`, `showAllImages`).

## The gallery's cards

**Each tool names itself, and the row fits.** The tools were six glyphs named
only by a native title, and at QHD the row ran 182 px in a 171 px card, so
Delete was clipped. They are now five - the star left the row (below) - and
shrink before they overflow (`flex: 0 1 auto; min-width: 0`), with Delete
apart at the far end (`margin-left: auto`). Each carries a short name in
`data-tip` that a `::after` shows at once on hover and on keyboard focus,
anchored at the row's left edge: the buttons are not positioned, so the
label's containing block is the row, and the card's `overflow: hidden` never
clips it as it would one centred over Delete. Delete's is anchored at the
right edge instead, over Delete: at the left it put the destructive tool's
name over Reuse, some 200 px away. The glyphs stay the buttons' text - the
smokes find them by it - and the title stays the long name. Delete lights as
the page's other destructive buttons do (`#lb-delete`, `.ghost.danger`): its
glyph and an edge in the danger red on the dark fill, not the accent the
harmless tools take on hover.

**What a card is, on hover** (2026-10-05; the review plan's hover caption,
U79's graft). The title alone came a second late and never on keyboard focus.
On hover and focus-within the tools row's `::before` draws "#N · W×H · seed N"
over the row from the card's `data-n`/`data-size`/`data-seed` (never
`data-id`, which `forgetImages` queries), with empty alt text since the
image's label already says it; no-break spaces break a narrow card's line only
before "seed". A real `aria-hidden` span inside the row broke the smokes'
`nth-child`/`last-child` tool selectors, and one outside the row could not know
the row's height, so the row's children stay its five buttons. The caption is
hidden while a tool's name shows in the same place, and dropped while a card
note is up (`.card:has(> .card-note) .tools::before`), as the names are:
before that rule a pointer moved from a refused tool to the image drew it over
the note.

**Size and fit, from 1800 px** (2026-10-05). The review's U96 asked for an
S/M density and was left out the day before ("8 cards a row at QHD is already
judgeable"), and the recommended direction's own risk list named a way back to
`cover` should letterboxed cards prove a change of taste. Both are now built,
opt-in, so the default and every smoke written against it are unchanged: the
gallery head's two pairs (`#card-view`: S/M and Fit/Crop, shown from 1800 px
only, hidden on a sheet by `setSheet`), each a group with one tab stop, the
chosen button, `aria-pressed` and the size strips' arrows (`stepPreset`). S is
`clamp(150px, 6.5vw, 180px)` on `#gallery[data-density="s"]` - eleven a row at
QHD and nine at 125%, against eight and seven, and four in Gallery beside's
column at QHD, where M is three (`.stage.beside #gallery[data-density="s"]`,
150 px: the S rule's own 166 px fitted three there, as M did, and S showed
pressed with nothing changed) - and trades U13's 32 px-wide
tool targets for a 24 px minimum width (32 tall), the least a target should
be, Delete still apart. Crop is `#gallery[data-fit="cover"]`. Both belong to
the gallery, not the panel (`cardView`, stored under
`local-diffusion.gallery.v1`, read back at load by `restoreCardView`), so Reset
leaves them. Below 1800 px neither shows nor applies: the head is full at
1400, and a Crop chosen wide must not crop cards where it cannot be undone. A
sheet's cells are its own.

**Dead tools are marked, as the lightbox's Refine is.** ⤢ where the
checkpoint has no pass and ❐ where it takes no references are aria-disabled
with the reason as their title (`syncRefineTool`, `syncReferenceTool`),
never disabled, so each keeps its tab stop and a press still says why: in
`#gallery-error`, which stays and is announced, and for two seconds in an
`aria-hidden` note over the row (`refuseInCard`, `cardNote`). The card
decides the ⤢ refusal itself rather than leaving it to `openRefine`, so it
can say it there too; `openRefine` says it only for the lightbox's Refine,
in the panel's line that the lightbox mirrors. Each refusal is one sentence
in one place (`REFINE_NONE`, `REFERENCES_NONE` in gallery.js), which the
card, the lightbox's Refine and Use these references all say. Cards can render before the first status, so the
state is set in `card()` and again where the status flips a capability:
`syncCardTools` from status.js for refine, `syncReferenceMarks` (through
`syncConditioning` and `renderReferences`) for references.

The press must leave the tool where it was. `#gallery-error` sat between
the head and the grid, so its first message pushed the grid down 38 px and
the repeat press a click that did nothing invites landed on the image
below, opening the lightbox. It is now a row of the head's own whose line
is kept while it is empty (`[hidden]` there is `visibility: hidden`). A
message that wrapped still moved the grid, by its second line - rare while
only Bundles… shared the row, common once Select, its count and Clear did:
"This checkpoint can't use reference images; FLUX.2 [klein] can." wrapped
beside "2 selected · 2 not shown" beside the viewer at 2560 and at 600 px,
and the grid dropped 20 px under the pressed ❐. The line is now one line
whatever it says, a long message ending in an ellipsis with its whole text
as the title; the alert still reads all of it. The note
over the row is a gist on one line - "Four is the most", with the sentence
in the gallery's line - and the row's names stay down while it is up: the
two-line note grew up into the pressed tool's name and the two were drawn
over each other.

**❐ is a toggle, and references are marked where they are picked.** A click
on an image already among the references used to re-send the same list and
show nothing, and the only sign of a reference was a chip up to 1,600 px
away. `syncReferenceMarks`, which `renderReferences` calls, rings every card
and sheet cell that is a reference in the accent (an outline: the border
and its shadow are the star's) and numbers it top left as its chip is (a
`.ref-n` span; a pass's badge moves over for it). Its ❐ is pressed and
removes it; at four the other cards' ❐ are aria-disabled. Named "Reference
image" in every state, as the star is "Star" (below): which reference it is
("Reference image 2 - click to remove") or why it is dead is its title and
`aria-description`. A `role=status` note over the row says "Added as
image N" or "Removed" for two seconds, written once it is in the page so it
is announced. `setReferences` stays the one writer.

**The star is a button.** A starred card's amber ring and dark 28 px badge
are what a page of thumbnails is scanned for, but the badge was a
pseudo-element a click went through to the lightbox, while the real toggle
sat in the hover row 4 px from Delete. It is now a `<button class="star
fav-badge">` in the same place, before the tools: always shown on a starred
card, and ☆ on another's hover, focus-within or a coarse pointer, as the
tools are. It took the row's star's place, so a card's tab stops did not
grow. `syncStar` draws any star button from the state - glyph,
`aria-pressed` and title together, where the title followed only an in-page
toggle. Its name is "Star" in both states: it was "Unstar" when pressed,
heard as "Unstar, toggle button, pressed" - the misreading Pause's note
describes - while the lightbox's star named the same state "★ Starred". The title names F only on a button
that declares it in `aria-keyshortcuts`, as the card's does, and on a card
F works from the star and the tools as from the image (`galleryKey`): it
was the image's alone, under a star whose title promised it. `setFavorite`
updates every element showing the record by class and its star button:
cards, sheet cells, a `.result` beside the viewer, and the lightbox when it
shows that record. The lightbox's button is named Star too (its
`aria-label`, and its words, ☆ Star / ★ Star, with the amber of `.active`):
its words had said "★ Starred" when pressed, which a fixed name could not
contain. It takes the card star's title (`starTitle`, the one copy of
those words) and `aria-pressed` in `renderLightbox`, so the two say the same
after a toggle from either, or from the sheet. A sheet cell has the card's
button too (`.sheet-cell .fav-badge`, tabindex -1; The contact sheet), and F
stars a focused cell.

**A card is a group named by its id** (2026-10-05 area review). The card was
an unnamed div, so of its seven controls only the image ("Open #12: …") and
Delete named the image: the star and the Reuse, Refine, Reference and Sheet
tools were heard alone, and focus handed back to ⤢ from the Refine dialog said
which tool but not whose. Each card is `role=group` with `aria-label` "#N",
which a screen reader says as focus enters it, and so is a result beside the
viewer (`resultFigure`). The toggles keep their one name ("Star", "Reference
image"), as the toggle rule asks. Not a list or a grid: the day's lines and the
tombstones are the cards' siblings, and a grid role would want rows and cells
and put a screen reader in focus mode. A tombstone drops the role and the label
with the tools; its text says "Deleted · #N".

**F says what it did.** F stars from the card's image or any of its tools,
from a sheet cell's image, and from anywhere in the lightbox. The one control
whose state changed - the star - is not where focus is, so nothing was heard,
and in the lightbox the polite counter and zoom line were rewritten instead.
The three handlers pass `setFavorite(rec, value, { say: true })`, and where
focus is not on a `.star` or `#lb-fav` (a focused toggle says its own new
state) `sayStar` writes "Starred #N" or "Unstarred #N" into a visually hidden
polite line: `#star-said` in the gallery's head, or `#lb-star-said` inside the
lightbox while it is up, since the page behind a modal dialog is inert, live
regions and all. The line is emptied first and written on the next frame, so
the same words a second time are heard again, and emptied after five seconds,
so browse mode does not come on a stale one. The smokes read what was said
from a record of the two lines' non-empty writes (`__said`, a MutationObserver
installed at setup and cleared before each press), not from the lines, which
empty themselves; a check that nothing was said waits two frames after the
star changed. `aria-keyshortcuts="F"` stays on the images: on the image's stop
it is what says that F does something there.

**A star on an image deleted elsewhere is a delete's note.** Starring an image
another tab had deleted answered "unknown image" in the panel's line, which the
lightbox mirrored and kept under the next image, and the image stayed live
everywhere, where ⌦ in the same case said "#N was already deleted." and
tombstoned it. `setFavorite` now does what `deleteOne` does - `forgetImages`,
then the same sentence through `noteLightboxDelete` while the lightbox is open
(which moves on to the next image), or in `#gallery-error` as a note - and asks
the Runs picker again, which counts images. Its other failures go to
`#gallery-error` too: a star is a gallery control, wherever it is pressed.

**The whole frame, and what it is.** Cards are `object-fit: contain` on
their own grey, the trade the sheet's cells made: `cover` showed 57% of a
16:9 image and could not tell landscape from square. From 1800 px the head's
Crop is the optional way back to `cover` (above). A refine pass is
marked top left with its strength (`.card.refined::before`, the sheet's own
rule), since it keeps its source's picture by design. The image's label and
title name it as the page names images, "#N", with its size, seed and the
source of a pass - "(deleted)" after it where the source has gone, as the
lightbox's facts say it. Reuse on a pass says what it does: "Reuse pass…" in
the lightbox, named "Reuse pass in the Refine dialog" - led by the words shown,
for voice control, which "Reuse this pass in the Refine dialog" did not
contain - and on both the title "Open the Refine dialog on #4 with this
pass's size, strength and seed" (`syncReuseButton` keeps the source in
`data-reuse-pass` for `syncReuseButtons`); where the source has been deleted
(`refine_missing`, kept in `data-reuse-pass-gone`) it is "Reuse settings",
titled with why the pass cannot be made again, since the press fills the
panel instead - it had promised the dialog on an image that was gone. A
delete on this page marks the passes made from what it took the same way
(`forgetImages`: the records' `refine_missing`, the cards' titles, every
Reuse, the lightbox's facts), as a reload's records would be; a ⚠ on another checkpoint's
record adds what the Reuse will leave behind - references the checkpoint
takes none of, an adapter the LoRA box does not offer. Those are the
status's to say, and the card can be older than it, so `syncReuseButtons`
runs again where references or adapters flip and where the adapter list
lands (`refreshLoras`), not only where the checkpoint changes: a card kept
under All models across a restart named the old checkpoint's verdict.

**One tab stop for the grid.** Seven stops a card put the 30th image some
270 Tab presses down the page. One card is current (`galleryCurrent`, an
id, so a gallery reset or queried again keeps it where the record is still
loaded): its image, star and tools are tab stops and every other card's
are -1. Focus arriving any way - a click, the lightbox handing it back -
makes that card current; a page that lands without it makes the first
card current; a delete moves it to the next card along (`forgetImages`). F
stars the card from its image, star or tools; a job's images prepended at
the head take the stop where it was the grid's default (`galleryChosen`,
set by focus arriving and cleared by `resetGallery`) or the old head with
focus elsewhere; on the image the arrows walk the grid, and Home and End go
to the first and last card loaded; Down from
the last row loaded fetches the next page and goes on into it. Up and Down take the nearest card across in the next row, found
from where the cards are rather than from the grid's column count: a day's
line starts a row of its own, so a short row before it put "a row on" by
the count that many cards short. Focus moves with `preventScroll` and the
card is scrolled `nearest`, clear of the topbar by its `scroll-margin-top`.
The image's label ("Open #N: <prompt>"), title (`cardTitle`) and description
are `thumb()`'s, shared by the card, the viewer, the sheet and #recent. The
title had been the image's accessible description as well, so an 81-word
prompt was read twice per image, a 463-character name and a 481-character
description. `titleImage` sets `aria-description` to the title's first line
("#12 · 1024×1024 · seed 7775" and a pass's source) less any fact the name
already says, taking the title's place as the description while the pointer's
title keeps the prompt. The name keeps the prompt because it is what tells two
gallery images apart; a sheet cell is named by its axis values instead, and so
described by its size and source alone (The contact sheet). `forgetImages`
retitles through `titleImage` too. The image's Enter and Space, `thumb()`'s as
well, open the lightbox only without Ctrl, Meta or
Alt: Ctrl+Enter is Generate's from anywhere, and on a focused image - where
the lightbox's close leaves focus - it opened the lightbox and queued a job
behind it. The rest here is `card()`'s.

**A card in the recycle bin has the tools of an image on its way out**
(2026-10-11). In the bin's view a card has two tools: ↶ Restore, and its ⌦
become Delete permanently, still last and still apart, now asking first;
its picture opens the lightbox. ⤢ goes - it means Refine everywhere else -
and so do the star, ❐, Reuse and ⊞, since a binned image can be none of
starred, a reference, a source or a sweep's base until it is back. A restored card
tombstones "Restored · #N" in the bin and returns into its own box in the
gallery, so nothing moves under the pointer either way; a binned one, in
the gallery, reads "In the recycle bin · #N" where a deleted one reads
"Deleted · #N". The head's controls that act on live images are held with
"Not in the recycle bin — leave it with Bin" rather than hidden, so the
head keeps its shape - except Re-render, which is hidden, since the count it
would show is the bin's and no render's - and Empty recycle bin takes Delete
unstarred's slot.

## The gallery selection

**Picked cards, for actions on several images** (2026-10-10). Swap tags and
Mass refine act on many images at once, and their only scope was the filter,
as Re-render's and Delete unstarred's is: a search that matched exactly the
images wanted was often not to be had - three cells of a sweep, a few keepers
across two nights. The user chose click-to-select cards over a tick on each
card or a lasso. `gallery-select.js` keeps one selection for every such
action, and the actions read it (`selectedIds()`, `onSelectionChange`) and
say whether they act on it or on the filter.

**A mode, not a modifier.** Select in the head turns a card image's press
into a pick; out of it the card does what it always did. Ctrl+click on the
image was the other way, but the card already has a press of its own (the
lightbox), a click's modifiers are invisible to a touch screen and to a
screen reader, and a pick made by accident opens nothing to show it. In the
mode Enter still opens the lightbox - the keyboard can look before it picks
- and F and the tools stay the card's. Shift picks a range, from the last
card toggled to this one, in the loaded order (the grid's DOM order), as a
file manager's does: a night's sweep is a run of neighbours.

**Ids that outlive the cards.** The selection is ids, so a new search, a
filter, a sheet or another page keeps it: picking across two searches is the
point. The count beside Select says how many of it the view does not show
("8 selected · 7 not shown"), since an action would reach images the user
cannot see. A deleted image leaves it in `forgetImages`, which stays the one
place the page drops deleted ids; its tombstone drops the mark. Leaving
select mode keeps the selection, marked and counted - turning a mode off is
no request to throw away forty picks - and Clear or Escape empties it; a
second Escape leaves the mode. The cap is the grid ceiling the server holds
the actions' `ids` to (`max_grid_images`, the sheet's page), said when a
pick would pass it rather than at submit. The record kept with each id is
only a fallback: `selectedRecords()` reads the loaded page's copy where
there is one, since a search replaces `galleryRecords` with new objects and
a star made since lands on those; an action sends ids and the server reads
the rows. A page that fails to load re-counts too, as none of the selection
is shown then. A pick refused at the cap leaves the Shift anchor where the
last real toggle put it, so a range never starts from an unpicked card. On a
sheet Select is held, so it does not read as pressed and the count does not
invite picks.

**Where it sits, and what it must not move.** The head's control rows are
full at 1400 px, so Select, its count and Clear share the footer row with
Bundles…, styled as Bundles… is so the row keeps its 18 px line (a small
button there grew it and moved the grid, as Bundles…'s note says). They are
right-aligned with the count to Select's left, so the count coming and going
never moves Select under the pointer, and Clear hands its focus to Select as
it hides. On a sheet the cells are not picked: Select is held there
(aria-disabled with the reason) rather than hidden, and a selection made
before is kept and counted.

**Marks.** The top-left corner holds a pass's strength and a reference's
number and the top-right the star, so the pick's mark is a ring at the top
centre: empty on every card in select mode, a filled check on a picked card,
which also gets an accent ring round the card over a star's or this run's.
The image is a toggle in the mode (`aria-pressed`, "Select #N: …"); out of
it, a picked card's group is named "#N (selected)", since `aria-pressed` on
a press that opens the lightbox would lie. The grid has `user-select: none`
in the mode, so Shift+click picks a range rather than the page's text.

## LOCAL_DIFFUSION_DEVICE reaches the offload hooks

The offload hooks are handed `LOCAL_DIFFUSION_DEVICE` explicitly (when it is a CUDA
device) and `describe()` reads that card's name and memory: given no device,
diffusers resolves its own accelerator, which is always GPU 0, so `cuda:1`
used to load onto cuda:0 while the status line described cuda:0 as if that
were the setting.

## The search grammar

The gallery search box is `key:value` terms plus free text: the terms become
SQL predicates (`seed:42`, `cfg:<=0.5`, `steps:8-12`, `mu:none`, `model:flux`,
`run:ab12cd`, `job:ab12cd`, `id:12`) and only what is left goes to the FTS index, so a prompt that
happens to contain a colon still searches for itself. `mu:none` is `IS NULL` -
no shift at all is a different image from `mu:0`, and SQL matches NULL with
neither. `model:` is a substring by design, for the family; the running
checkpoint's scope over the gallery is a separate, exact predicate (above),
not a term the page writes into the box. `id:` is the image the page names
"#12" - in "refine of #4", the Refine dialog's title, a card's label - which
no search could find before; a range is a span of them, and like any other
term it is a strict scope for Delete unstarred and `clear --where`.

## The contact sheet

The contact sheet (`web/sheet.js`, `renderSheet`) lays one run out as the grid
that made it: the two widest-varying axes become columns and rows, any third
becomes a pinned selector rather than a third dimension nobody can read. The
exception is a comparison's `var:variant` axis, which takes the rows whenever
another axis varies, however few of its images are loaded (a star each under
Starred) - including a variable a user's own prompt grid happens to name
`variant` (see "The variable assignment is recorded"). It asks for the whole
run in one page (`sheetPage`, the status's `max_grid_images`) because a
matrix with a page boundary through it is a puzzle.

**Every sheet the page opens comes through `showSheetScope`** - a comparison,
a refine sweep, the Runs picker, a card's ⊞, both branches of Show run - which
asks `setSheet(true, { land })`: once the sheet is drawn, and only if it is
still the one shown, the page scrolls the gallery head under the topbar
(`landOnSheet`; its `scroll-margin-top`; smooth unless reduced motion is asked
for). A finished
comparison's sheet used to open at y = 1650 on a 1305 px screen, under the
prompt check and the viewer, with nothing in view changed but a line in the
panel. The landing is the gallery head rather than the sheet so the search,
Starred and Sheet - the sheet's own filters - stay in view; the viewer is
not hidden (a running job lands there) nor the check folded, both a scroll
up. The Sheet button does not scroll: it is pressed at the gallery head -
except where the sheet takes the page out of Gallery beside, whose column is
no width for a sheet; the page is then one column again and `setSheet` lands
on the sheet, which would otherwise open out of sight of the press (The
gallery beside the viewer). A short sheet under a tall viewer - a refine
sweep's three passes at QHD - ended the page before the head could reach
the topbar, and the scroll stopped 43 px down with the sheet's cells at the
window's foot: the scroll first reserves the room it needs under the sheet
(`reserveSheetRoom`, a margin outside the sheet's box, which `fitSheet`
measures; `setSheet` takes it off). The room is counted from where `main`
ends: counted from the document's scroll height, which is never less than
the window, a page shorter than the window got a margin short by the
difference, and the scroll stopped 85 px down with the viewer's tools row
cut in half under the topbar above the head.

**The landing takes focus along where it left it behind** (2026-10-05).
Scrolling moved no focus, a choice whose keyboard cost had not been weighed:
Run comparison, and the finish line's Show sheet in one column, were left a
screen above the sheet they opened (at 800x900 Run comparison sat at top
-1380), and the next Tab scrolled the page back to the control after them.
`showSheetScope` now takes the control that asked (`asker`, the focused element
at the call); a comparison's or a refine sweep's sheet opens when its job
ends, so `startJob` keeps the element pressed in `submissions` and `finishJob`
hands it on. Once the landing's scroll has settled (`scrollSettled`: three
frames at one scrollY, 1.5 s at most), `focusSheet` moves focus with
`preventScroll` to the sheet's first cell, or to the search box over an empty
sheet, in two cases. One is where focus is still on that control and the
control is not wholly in sight - inside the window under the topbar and inside
every ancestor that clips its overflow (`wholeInSight`). The window alone was
not enough: at 2560x1305 the docked check scrolls inside itself, and after the
landing Run comparison lay under the topbar but below the check's own box,
unseen, and Tab scrolled the page 640 px back to it. The other is where focus
has fallen out of the page: `<body>`, or an element the sheet hid (a card's ⊞,
or a card the lightbox handed focus back to). Where the page itself moves focus
off a job's asker - the Refine dialog closing on its run, the strip a job's
first frame empties - it hands the asker on (`handJobFocus`); without that a
sweep run from the viewer's ⤢ left focus on Generate 566 px above its sheet.
Focus anywhere else is never moved, because a comparison's job can end while
the prompt is being typed in; the Runs picker, at the gallery head, stays in
view and keeps its focus.

**Laid out again, the sheet keeps the keyboard where it was** (2026-10-05).
`renderSheet` - a gallery page landing, a delete, a pin or an axis picked, a
resize whose wrap flips - rebuilds every cell and control, and a keyboard's
focus on one fell to `<body>`: F on a cell another tab had deleted (the
star's refusal forgets the image, and its cell goes), an ArrowDown on the
Columns select, the end of
another tab's job. `sheetFocusPlace` records the place before anything is
removed - `forgetImages` takes it before it drops the cells - and
`restoreSheetFocus` returns to it with `preventScroll`, unless focus was kept
or moved on purpose meanwhile: the same cell's image, or its star, by id; else
the image now at its place in reading order, else the last; a control by its
name (a select by its label's words, a button by its own), else the control
at its index; the search box where no cell is left. Another tab's job ending
no longer reaches an open sheet at all (Two tabs on one server).

**A table to a screen reader, and said as it opens.** The sheet exposed
nothing of its structure: heads and cells were generic, and every cell was
named "Open #16: <the prompt>", the one thing a run's cells share, with the
steps it was there to show nowhere in it. A matrix is now `role="table"`, named
by `sheetSummary` ("Sheet: Steps × Seed"). Each row is a `.sheet-row` with
`role="row"` and `display: contents`, so the heads and cells are still the CSS
grid's items, the source still spans the rows (with `aria-rowspan` too) and the
heads still stick to the scroll box - measured layout-neutral, every cell's
rect unchanged. The heads are `columnheader` and `rowheader`, the corner a
`cell`, the cells `cell`; a wrapped row has no heads and is a `list` of
`listitem`s. Each cell's image is named by what the sheet lays out ("Open #16 ·
Steps 2 · Seed 7601…": the column's head, then the row's, a version by its name,
the source "· Source"), and described only by what the name leaves out, its
size and a pass's source (`titleImage`): "#16 · 512×512 · seed 7601…" read the
id and the seed a second time. A pin's value is every cell's, and the controls
say it. Every sheet `setSheet(true)` opens, from the page or the Sheet button,
is said in `#sheet-said`, a polite line outside `#sheet` (which is hidden until
then) - "Sheet: 6 images, Steps × Seed", "Sheet: 28 images, by Seed", or the
note where nothing is laid out (`saySheet`) - written a frame after an empty
line, so the same sheet opened again is said again.

**The cells fit the screen's height as well as its width.** Columns are
`minmax(96px, var(--cell-w))`, `--cell-w` being `--cell-max` (`1fr` under
Fill width, below), and `--cell-max` is the window less what
sits round the grid at that landing - topbar, gallery head, controls, the
Fixed line, the note - less the column heads, shared between the rows and
times the aspect. `fitSheet` measures that chrome rather than guessing it, so
a tier that scales the controls needs nothing, and `100vh` follows the window
from there. A 4 x 3 at QHD went from 524 px cells in a 1,607 px sheet to
339 px cells on one screen. The grid scrolls in a box of its own, no taller
than the same space, and the heads stick to its top and left edges on the
page's background, over their whole row or column, so a matrix still past
the 96 px floor keeps its labels beside the rows being read; the page itself
never scrolls sideways. Heads are at the body-small size in `--text` - 11 px
grey over 500 px cells read as an afterthought - each label clamped to three
lines with the whole value as the heading's title. A long number in a label
goes to a line of its own whole and breaks only where it alone is too wide
(2026-10-05). With `overflow-wrap: break-word` alone, a 16-digit seed in a
column under about 112 px broke anywhere: an 8-seed comparison at 1400x900
(111 px cells) read "Seed 760152042044950" over "9", one number taken for
another when read off to type back. A `<wbr>` between groups of four digits
alone moved the break between groups, but the line-breaker then filled the
first line after "Seed", so in 122-133 px columns (1500 and 1600 px windows)
seeds that had stood whole on a line of their own became "Seed 760152042044"
over "9509". `numberLabel` now wraps a number of five digits or more in an
inline-block `.number` with the `<wbr>`s inside it, counted from the number's
end so a tail is never one to three digits, and `text-wrap: balance` evens the
broken lines: one line from 144 px; "Seed" over the whole number at 122-133 px;
"Seed" / "76015204" / "20449509" at 96-111 px, within the three-line clamp. The
text is still the whole number for a copy. `overflow-wrap: normal` alone kept
one line but was 112 px in a 111 px label and clipped, and type sized by
container width rendered 7.8 px text. The heads' row and the row
heads' column, plus the gap, are the box's `scroll-padding` (`fitSheet`
measures both into `--sheet-head` and `--sheet-rowhead` on `.sheet`): a cell
the keyboard focused while the box was scrolled was lined up with the box's
edge, under the stuck heads, its top and its focus ring hidden. A cell drawn
wider than its 384 px thumbnail asks for the PNG (`srcset`, with `sizes` the
measured cell width; lazy loading still fetches only what is on screen): the
upscaled WEBP was soft exactly where a CFG or steps sweep differs. All of it
is measured, so it is measured again when the window is resized
(`refitSheet`, debounced, from app.js), and a row that now fits or no longer
does is laid out again: once per render, a QHD sheet resized to 1400 px kept
the `sizes` of 339 px cells drawn at 204 and its note under the window's edge.

**Fill width trades the height for the width** (2026-10-05; the review's U16
toggle, left out the day before). The template's `--cell-w` is `--cell-max`
fitted and `1fr` filled (`.sheet.fill-width`; `.fill` is the progress bar's
class, and reusing it zeroed the sheet). The box keeps its window-high cap, so
a filled matrix scrolls inside it under its heads, and `sizes` follows the
cells. The toggle keeps one name, its state in `aria-pressed` and its title, and
is aria-disabled with its reason on a wrapped row, which fills the width
already. It is a per-browser preference (`local-diffusion.sheet.v1`, through
`remembered`/`remember`, working with storage throwing): a view setting like
Starred, not panel state, so Reset leaves it. Its export is the filled sheet's
(The sheet exports client-side).

**A long single row wraps.** With no row axis and more columns than fit at
96 px - a seeds-only run, which every Batch or Repeat over 1 is - the cells lay
out like the gallery (`minmax(var(--thumb-min), 1fr)`), each captioned with
its value, rather than a ribbon several screens wide; one that fits stays a
row under its heads. Only a sheet with no row axis wraps, so a matrix and the
variant rows are never reshaped.

**Around the matrix.** A Fixed line names the settings every image shares -
under Starred, what the keepers have in common, which the tick otherwise
takes off every axis - in an element of its own, since the note is read back
exactly. A starred cell carries the card's ring and 28 px badge (a 12 px
glyph in a 690 px cell was not seen) - the card's own rules, which name
`.sheet-cell.fav` and `.sheet-cell .fav-badge` beside the card's so the two
cannot drift - and F on a focused cell stars it, one listener on the grid.
Every cell also has the card's star button (2026-10-05; U48's optional hover
star), shown where the pointer or the focus is, and always when starred, out
of the tab order (tabindex -1), so the pointer stars a keeper without a trip to
the lightbox and the tab stops stay one per image; the button and F both star
through `starCell` → `setFavorite`. Cells are isolated stacking contexts
(`isolation: isolate`): a badge sharing the heads' z-index drew over the stuck
heads once the box scrolled. The lightbox still walks the sheet in reading
order. A refine pass says so (`⤢ 0.1`, top left) unless
the strength is already an axis or a pin. Where every image is a pass of one
source, the source leads as a column of its own, headed `Source #<id>`,
spanning the rows and first in what the lightbox walks, framed as the
passes are - their cells' shape, centre-cropped as `refine.fitted` kept it,
captioned "cropped as the passes were" where the shapes differ (a whole 1:1
source letterboxed beside 3:2 passes cut from it set two framings side by
side, which the eye compared as one; the lightbox it opens has the whole
picture): it is fetched by id
(`sheetSources`, alongside `learnSheetOrder`) and never a sheet record - no
axis value, no cell of `cellOf`, no count - so a strength of null still means
a direct render; a deleted source is said in the note. The axes gained
References (`#8 + #10`) and Adapter, null where none applied as the LoRA
weight is, so with and without a reference at one seed is three cells, not
one; with one adapter the LoRA heads name it. Images that pass the pins but
find their cell taken are counted, where "1 of 3 loaded" read as images still
on their way, and in two kinds. One that has the cell's image's settings
throughout (`sameSettings`: every axis there is, the size, the checkpoint and
its quantization, the refined source) is a repeat - "2 of 4 shown · 2 repeat a
shown image's settings": the same grid run again without Resume keeps its run
id, and a repeat-seed check is exactly what a sheet is read for. The rest
"share a cell with another image and differ in something the sheet does not
lay out" - the size, mostly. Size stays off the axes.

**The way back, and on.** Sheet off restores the search a page-opened sheet
replaced (`sheetReturn`) while the box still holds what the page wrote, so a
sweep's three images do not look like a gallery whose rest had gone; back to
back sheets keep the search from before the first, and a search typed since
stays. Over an empty search the Sheet button opens the newest run, saying so
in the note, where a gallery of several runs gave a wall of `none` pins; a
typed search is laid out as typed. A new pin takes the value most images
carry under the pins already held (`reconcileAxes`): the first value is
`none` for every variable of a mixed gallery and no image was `none` on all
of them, so not one cell filled. Over a full run the counts tie and the first
value still wins. A sheet of one run opened on `run:<id>` offers Resume this
run: the run's base seed from `/api/runs` into the Seed box and its size into
Width and Height, Resume ticked for this session only, the grid fold opened
and Run grid brought into view. Resume is scoped by the size, the checkpoint
and the quantization: filling only the seed, the panel's size after any other
render made the preview skip nothing and blame the grid, so the size is
filled too, and a run of another checkpoint (All models) offers no button.
The axes are not stored per run, so whether the panel still holds that grid
is the preview's to say, and a preview that would skip nothing says the grid
- its prompt, axes or other settings, or the server's quantization - is a
different one. Not on a comparison or a refine sweep, which the panel's grid
did not make.

## The CLI's loop, and jobs.submit's

`generate.py` duplicates the sweep loop from `jobs._generate` rather than
importing it: that method is welded to a `Job`, an event loop and an SSE
queue, none of which exist in a CLI. What it no longer duplicates is *which
images to make*: the CLI and `/api/grid` both call `grid.expand()`, which
with no axes set yields exactly the old runs loop (`seed + index *
num_images`, same wrap helper), so a grid cell in the browser and one in the
terminal cannot drift apart. **`jobs.submit()` - the plain `/api/generate`
path - is the one remaining copy of that seed loop**, kept inline
deliberately: routing it through `expand()` would apply the grid's
placeholder check to an ordinary prompt with braces in it. Its arithmetic is
asserted equal to the CLI's in `tests/test_cli.py`. The CLI still owns the
per-image `seed + offset` and the `save_image` call, and shares
`grid.estimate_seconds` with the preview. Nor does it duplicate settings:
`--model/--quant/--offload` are written into `os.environ` *before* the first
import of `backend.config`, because `settings` is a frozen dataclass whose
fields are evaluated at import time.

## events() opens with done

`events()` opens with a `done` event, not `status`, for a job that has already
finished — otherwise a stream attached after the fact (a reload, or a queued
job the cancel endpoint ended) leaves the client waiting on an event that will
never come.

## Content hashes and duplicate counts

**Every image records a hash of its pixels** (`storage.content_hash`: 16
hex of the sha256 of `image.tobytes()`; column, metadata key `hash`,
`ImageRecord.content_hash`, `hash:` filter, and the reindex recomputes it
from the decoded file). It is what lets "did this axis do anything" be a
count: `Job.snapshot()` carries `distinct_count`, `RunSummary.distinct` is
`COUNT(DISTINCT content_hash)` only when every image of the run is hashed,
and the UI and the CLI say "only N distinct" when it is fewer than made.
The premise - bit-determinism through torchao's fp8 kernels with model
offload moving components mid-forward - was measured on 2026-09-08 rather
than assumed: the same request twice on FLUX.1 Krea dev is the same hash,
as is a negative prompt the pipeline dropped against none. The companion is
`grid.duplicate_count`: `GridPreview.duplicates` and the CLI's dry run
count the cells whose *applied* key repeats an earlier one (a
negative-prompt-only variable on klein, a CFG axis under the threshold),
compared through `backend.applied()` so the capability flags decide. Refine
cells compare their integer schedule start, not the raw strength, only in
this advisory; resume and run ids keep the requested strength. The
cells are still rendered as spelled - dropping them would leave a `{avoid}`
axis with one cell of three and a sheet unable to say why.

## The results viewer

**One job at a time, laid out for the whole job.** The strip (`#viewer`,
panel.js) holds the newest job's images and is cleared when its first frame
or image lands (`clearResults(job)`), so the previous result stays up through
the queue wait and the load. Its tier comes from the job's own count -
`planned_images`, from its status frame - not from the images so far: one
image at 68vh, two to four laid out to fit, more as small tiles. By the
running count a sweep was laid out three times in its first five images, and
the gallery under it jumped each time. Every tile - a result, the live
preview, a pass's stand-in - is the one box (`viewerTile`: the picture at
`--w`/`--h` and the tier's `--tile-h`, no wider than the strip, and under it
the tools row or a line of the same height), so whichever replaces another
leaves the gallery where it was. The 2026-10-04 review measured the preview
at a fixed 320 px after a 917 px result: the gallery head went from 1,046 to
509 and back at every Generate, about 540 px twice. It now stays put from the
first frame to the image (smoke-results).

**Two to four, best fit.** `fitViewer` tries each column count and keeps the
one with the largest tile within the strip's width and the single image's
height - 68vh and a tools row - the rows' gaps and tools rows counted, and
writes `--cols` and `--tile-h`: two squares side by side at 887 px at QHD,
four in a row at 491, four 16:9 images two by two. A fixed 34vh had used 42%
of the strip for a batch of two. The review plan's equal 2x2 (552 px cells at
QHD, from its Studio direction) was not built: best fit gives larger tiles
than that in every case U47 measured, and lays two to four out as their shape
asks. It writes `--rows` too, and every row the
job will fill is a track from the first frame (a tile's picture, its gap and
its row): the strip had only the rows that were filled, so a 2x2 was one row
while the preview denoised and grew by the second as the images landed - the
gallery head went from 484 to 950 at QHD at the end of every landscape batch
of three or four. So is every column (2026-10-05): the columns were `auto`,
the empty ones no width, so a lone preview - or a pass's stand-in - sat
centred, the first image landed there, and the row moved left by half a tile
as each of the others landed (1248 to 512 px at QHD within 0.25 s, on Anima
Turbo). `fitViewer` writes the tile's width as `--col-w` and the grid
reserves every column at it, so the preview holds the first image's slot.
The strip is also fitted as if the page's scrollbar were there, whether
or not it is (`pageScrollbarWidth()`, measured once on a probe, 0 for
overlay scrollbars, taken off the width while the page has none): a gallery
too short to scroll the page took a scrollbar while the progress block was
up and gave it back as the job ended, widening the strip by it, and the
tiles refitted 491 to 494 px and moved the gallery head 3 px at the end of
a batch of four. A stable gutter on the root was tried first and reverted:
in Chrome it shrinks the viewport units by its width, so every vw size on
every page narrowed - the panel's 18vw read 458.1 px for 460.8 at QHD, the
type ramp 15.47 px for 15.5, the lightbox's fit and strips moved - for a
3 px refit in one case. What is left is the strip's content moving sideways
by half a scrollbar as it goes, with the rest of the page. smoke-results
follows a batch of four from its first frame to its end and fits the strip
with the page's scrollbar forced on and off. A ResizeObserver refits on the next frame
when the strip's width moves (refitted in the observer's own callback, the
new height was a resize the frame could not deliver), and the window's resize
for its height. A strip whose images are not all the job's shape wraps as
before; no job here makes one.

**The result's tools.** A result is a `figure.result[data-id]` - never a
`.card`, whose counts and keyboard are the gallery's - holding the `thumb()`
image and under it a row of the card's own tools: Reuse (`reuseButton`),
Refine (the card's handler: on a checkpoint without a pass it says so in
`#gallery-error` and over the tile, and opens nothing), Star (`setFavorite`,
which updates `.result[data-id]` and its stars), ❐ where the checkpoint
takes references (`toggleReference`; on klein a reference had to come from a
card, the image just made a screen away from its own) and Delete
(`deleteImage`; `forgetImages` leaves a tombstone). Not ⊞, which is for
browsing. From 1800 px the row is `--h-tool`, 32 px - its buttons were 29 x
30 at QHD, a card's 25 x 26 - as are a card's tools and a long run's tiles'
there (a token rather than `--h-sm`, which every small button shares). The
row is always shown under one to four images and over the picture on hover
or focus in a long run's small tiles, as a card's - where they are no tab
stops, only the pictures are, as before: four more a tile put a 48-image
strip 240 presses deep, and the lightbox a picture opens has every action.
The image just made had none: starring or refining it took the lightbox, or
the same image found among the cards.

**The column beside one image (from 1800 px).** While the strip holds one
result its figure carries a column (`resultSide`): which image a pass refines
("⤢ Refine pass of #5 at strength 0.1", the source a press away), Star,
Refine…, Reuse, ❐ Reference (where the checkpoint takes references), on a
pass Compare with source (`compareButton`: the lightbox on the pass, as its
picture opens it, with the comparison already on - the question a pass that
has just landed asks, a press and an S away before; aria-disabled, saying why,
once the source has gone, `syncCompareResult`, and refused over the tile
through `refuseInCard`), Use seed and Delete in words - Delete red under the
pointer at the row's end, where it had been a lone ⌦ stranded under the
picture's corner, the one tool the column did not have - the prompt, and the
lightbox's own `factsList` - one facts renderer for both, without the
lightbox's Use seed, since the row has one. A square result at QHD had left
1,270 px of the strip empty. The column is as tall as the picture and its
row (`contain: size`, stretched) and scrolls inside itself, so it never sets
the strip's height; the live preview and the stand-in keep its room
(`figure:not(.result)::after`), so the picture does not move when it comes,
and the row under the picture stays, empty, at its height. The star's word
is drawn after its glyph (`data-word`), since `syncStar` writes the glyph.
Under 1800 px there is no column. While the prompt check is docked beside
the strip the check takes that width - with the column there the picture
shrank as the check opened and the gallery moved under it - so the column's
actions become the row under the picture, in the tools row's place and at
its height, with a line of the facts (`.result-brief`: seed, size, steps,
CFG, the whole settings line in its title) before Delete. The image just
made had fallen back to 26 px glyphs there, its words and facts gone while
"Earlier this session" stayed beside it; now it has one set of words for
its tools whether or not the check is open. With Gallery beside the check
takes the gallery's column instead, so the column stays (The gallery beside
the viewer).

The row leaves out a pass's Compare with source: as wide as the picture, a
portrait pass's row (405 px at 2048x1040 at 125%) had every word in it cut to
an ellipsis, and in a row as wide as its buttons the button would take another
150 px from a docked `#recent`; the picture opens the lightbox, where S
compares, and the column has the button again once the check closes. The row
is the picture's width, or its buttons' `max-content` where that is more,
centred under the picture and never wider than the strip (2026-10-05). When it
was only as wide as the picture, a 5:12 image at 2048x1040 at 125% (295 px)
had every word cut to an ellipsis, a direct image and a pass alike; 4:7's
405 px was the narrowest that held them. The 5:12 row is now 385.8 px at
2560x1305, 364.8 at 2048x1040 at 125%, 359.8 at 1920x947 and 355.3 at
1800x900, every word whole, and the 4:7 row at 1800x900, cut before, is whole
too. The brief is left out of the measure (`contain: inline-size`) and takes
what is left: under a 5:12 picture that is 0 px at all four sizes, so the row
is its actions alone, the seed in Use seed's title and the image's; at some
sizes between it is a few pixels, 8 px under a 4:7 picture at 1920x947, a lone
ellipsis as before, with the settings line in its title. Wrapping the row onto
two lines would have moved the gallery as the check opened, and glyphs alone
would have taken back the words the row exists to keep. `fitRecent` measures
the row with the picture, so a docked `#recent` leaves it its room.

**A prompt keeps its lines, and stops at twelve** (2026-10-05). The result
column (`.result-prompt`) and the lightbox (`.lb-prompt`) set the prompt in
`white-space: normal`, so a NewBie XML prompt's 29 lines ran into one paragraph
(`<character_1> <n>hatsune_miku</n> <gender>1girl</gender>…`): which
attributes were whose could not be read, and at 2048x1040 at 125% it was 800 px
that put the facts below the column's fold (the negative's `dd.text` was
already `pre-wrap`). Both are now `pre-line` and capped at `--prompt-cap`
(`12lh`), scrolling inside themselves past it; `pre-line` alone made the
lightbox's prompt 25% taller, since every line takes at least a line. An
85-word prose prompt is 8-11 lines there and is not cut, where the 8 lines
first tried cut it at every width. A scroller with nothing focusable in it is a
Tab stop of its own in Chromium, so the rest of a long prompt is reachable from
the keyboard. The column's prompt is `flex: none`, so the column's flex layout
cannot shrink it below the cap. The Refine dialog's three-line `.rf-prompt`
reminder is left as it was; under 901 px the lightbox's prompt is four lines
(The lightbox zooms).

**Under the strip.** `#viewer-note` names a pass's source - "Refined from #10
at 0.5 · 1344×1728", the source opening in the lightbox with the pass after
it - except where the column says it already; it was empty, and the image
said nothing of where it came from. For a long run it says how many images
there are and that the strip scrolls, or how many it no longer holds. A
source deleted while its pass is in the strip (`forgetImages`, then
`viewerSourcesGone`) turns the note's and the column's "#N" into "#N
(deleted)" and redraws the column's facts; every image of the pass on the
page - card, strip, #recent, sheet - is retitled with it (`cardTitle`). The
note and the column had kept a live "#N" whose press said only, in the
panel's line a screen away, that #N had gone.

Below 1800 px, where there is no column, the note names a single direct
result as the card's caption does - "#N · W×H · seed N · Use seed"
(`.seed-line`, 2026-10-05; U45's third bullet) - so the seed of the image just
made is a press from the strip; from 1800 px the column or the docked row says
both, and the line is hidden. The note keeps one line's height while empty,
and a delete from the strip writes it again (`forgetImages` → `noteViewer`): a
hint's 15 px min-height against a 16.5 px line had moved the gallery 1.5 px
when a delete swapped the result for the held placeholder, which
smoke-delete caught.

**Earlier results, from 2000 px.** Beside a single image the stage takes a
second column, `#recent`, with up to six of this page's earlier results:
`sessionRecords`, a list of its own that `clearResults` fills from what the
strip held, newest first - never `viewerRecords` or `galleryRecords`, so the
strip's records and its `#viewer img` are the newest job's as before. Each
opens the lightbox on the image in the strip and those. Iterating on a prompt
showed one picture at a time in a strip 2,000 px wide, the earlier ones a
screen down among the cards and below the fold whenever the prompt check was
open. It is a grid on `.stage` (`grid-auto-flow: dense`, in the viewer's row
alone, `contain: size`), hidden in the batch tiers. It spanned the note's
row as well, and ended 35 px below the viewer beside it. With Gallery beside
it is a row under the viewer's note instead, check open or not, its 88 px
thumbnails beside its heading, since beside both it and the gallery's column
the image was 440 px at QHD (The gallery beside the viewer).

With the prompt check docked it stays, between the image and the check: the
check is part of iterating on a prompt, and with it open the gallery's first
row - the only other place the earlier results were - is under the fold.
It takes only the width the image leaves at its own size (`fitRecent`,
measured with #recent out of the docked grid: the viewer's content width
less the tile's, into `--recent-room` and `.recent-docked`), and none where
that is under `--recent-min` (280 px), so neither the image nor the gallery
under it moves as the check opens. At QHD a square result leaves 316 px; at
125% it leaves 192 and #recent goes, while a portrait one leaves room for it
there. A landscape image, already narrowed by the dock, leaves none; the row
under the picture counts where it is wider than the picture. A tag
checkpoint's chips above keep the width they had, across all three columns,
as in the dock's own rules (The prompt analyser). It is
measured again on the strip's ResizeObserver - the check opening or closing,
the window - and at each tile, and its own coming leaves the image as it was,
so the refit it causes ends there.

The strip's image, alone, opens the lightbox on the gallery from it -
else on this page's earlier results - so → walks back through what came
before: it had opened a lightbox of one, both arrows dead. A close gives the
focus back to the image or the earlier result it was opened from
(`openLightbox`'s `origin`), not to its card a screen down. A delete from the
strip's own ⌦ that leaves nothing beside it gives the focus to the line that
takes the strip's place (`forgetImages`), not to `<body>`.

**Idle is one line.** Before the session's first image the viewer is one line
naming Ctrl+Enter and the gallery: a 380 px box saying "Generated images
appear here" was 29% of a QHD window, 37% at 125%, above the gallery after
every reload. The state is keyed on the placeholder (`:has(> .placeholder)`),
never on the records, since a frame clears the strip before any record has
landed; the delete's held placeholder keeps the height it holds. A gallery
with nothing in it at all - unfiltered, no image on any checkpoint
(`syncEmptyInstall`, from each unfiltered page) - keeps the tall box, with the
checkpoint's examples a press away. A failed load's card sits in the idle
viewer with room of its own. The gallery's cards of the strip's run are
edged in teal (`.card.this-run`, set in `card()` and `appendResult`, cleared
by `clearResults`), so its first rows read as the run above them rather than
a second copy.

## The running batch on wide screens

At 2000 CSS pixels and wider, the results viewer's `many` tier (more than
four images) is as many tiles across as fit - `auto-fill` at
`--strip-min`, 160 px: eleven at QHD, eight at 125% - under a height of its
own, `min(560px, 45vh)`, that scrolls; a strip at its end stays at its end as
tiles come, and the note under it says that it scrolls. Eight equal columns
had kept QHD at more than seven a row, where sizing by `20vh` fit seven, but a
28-image run was four rows, 1,090 px, and 48 six rows, 1,625 px, with the
gallery under it a screen down repeating the same images newest first. The
live preview takes a cell like any tile. Smaller windows keep the 20vh
wrapping row; the single and two-to-four tiers are "The results viewer"'s.

Three details keep the strip honest. Its scrollbar's room is kept whether it
scrolls or not (`scrollbar-gutter: stable both-edges`): taken only once the
tiles overflowed, the scrollbar made each row one tile shorter, so 18 tiles
at 125% were two rows that fitted or three that scrolled depending on what
had come before - and once the live preview's tile had tipped it into three,
dropping the tile left it there, scrolled to its top, and every image after
found the strip "not at its end" and left the newest row out of view for the
rest of the run. Every tile appended asks whether the strip is at its end,
the preview's and a pass's stand-in's too (`stripAppend`), so the preview
starting a row is followed into view. And a strip that overflows by no more
than its own bottom padding is let grow by that rather than scroll
(`settleStrip`): 28 images at QHD came to 3 px past the cap, drawing a
scrollbar for those 3 px while the note said to scroll the strip.

**The strip's height is the run's from its first frame.** The tier and the
tile size were the job's from the start, but the height was not: in the
strip's cap of its own the rows still came one at a time as the images
landed, and the gallery head moved mid-job (691 px from 512 for a batch of 4
x Repeat 7 at QHD; 509 to 530 at 125%). At the first tile `reserveStrip`
gives the strip a min-height of every row the planned count fills at the
column count the strip has - measured on that tile - up to the cap, never
under the least the stylesheet gives it (beside a docked check that is the
check's room), again when the strip's width moves, and `clearResults` lets it
go; the strip has no 380 px floor there either, which left a 6-image run's
one row of tiles at the top of an empty band. The live preview stays the
size of the cell its image will take: the finished layout of five or more
is a strip of small tiles, and a larger preview cell would move the tiles
as each image replaced it, which is what this layout exists to stop. That is
why U12's larger live preview for a run of five or more was not built: it
would bring back the reflow U26 removed.

**The wheel over the strip reaches the page.** The strip had
`overscroll-behavior: contain`, as the scrolling columns do, and with the
pointer over it the page could not be scrolled at all: a strip at its end,
which a running batch always is, took the wheel and moved nothing, and at
QHD a 28-image strip with nothing to scroll did the same (testing notes,
2026-10-10). The strip is part of the page's flow, not a column of its own,
so it keeps the browser's own chaining: the wheel scrolls the strip while
it can go that way and the page once it is at its end. Chrome latches a
gesture to the box it began in, so a flick that runs the strip to its end
does not carry on into the page; the next one does. The strip scrolls only
vertically; no wheel is translated sideways.

`#recent`, the earlier results docked beside a single image from 2000 px,
had kept `contain` as a column that scrolls on its own, and swallowed the
wheel the same way: at QHD, after four Generates, it had nothing to scroll
(951 px of content in 951) and a wheel over it moved neither it nor the
page. It sits in the page's flow level with the viewer, as the strip does,
so it chains as well.

## The gallery beside the viewer

**An opt-in second column, from 2000 px** (2026-10-05). Stacked under the
viewer, the gallery was one row of cards or none in view beside the image just
made: the 2026-10-04 review measured twelve cards in view beside one 1024²
image at QHD, and none beside a 28-image run (U12). Its optional part 4, a
two-column stage, was left out then - the result column (U28) and `#recent`
(U34) were to use the width - and is built now as an opt-in, so no earlier
layout and none of the smokes written against them changes: a "Gallery beside"
tick in the gallery head (`#gallery-beside` in `label.check#beside-label`,
after Sheet - placed before it, it broke smoke-wide's Tab sequence - and before
Delete unstarred, which stays last; hidden under 2000 px, with no ? badge).
It is off by default and remembered per browser
(`local-diffusion.gallery-beside`, through `remember`/`forget`), not the
panel's, so Reset leaves it. It applies (`besideOn`, `.stage.beside`,
`syncBeside`) only while the tick is on, the window is 2000 px or wider
(`BESIDE_MEDIA`, with a `change` listener) and no sheet is open; while a sheet
is open the tick stays live, and its title says it takes effect when the sheet
closes (`syncBesideTitle`).

The gallery's blocks sit in `#library` - the head, then `#library-scroll`,
which holds the empty line, the grid, the sheet and Load more. Unticked, both
are `display: contents` and pass their `grid-column` on (`grid-column:
inherit`), so the docked check's and `#recent`'s grids reach the blocks as
before. Ticked, `#library` is `position: fixed` under the topbar at the
window's right, `--library-w` wide (`clamp(440px, 27vw, 720px)`: 691 px at QHD,
553 at 125%), and the stage's `margin-right` is that width plus a gutter. It is
fixed, not sticky: a sticky column is held only within the stage's height,
and the page scrolls by the panel's excess (the grid fold open, QHD at 125%),
which took the column's head under the topbar. `#library-scroll` scrolls by
itself (`overscroll-behavior: contain`, `overflow-anchor: none`, its
scrollbar's room kept), and the page keeps its own scrollbar's room
(`html:has(.stage.beside)`), so the panel growing past the window does not
narrow both columns under the pointer. In the column the cards are at least
`--thumb-min-beside` wide (200 px, the room their five `--h-tool` tools take),
three across at QHD and two at 125% - the wide tier's own 230 px made two
across at QHD. That is M; at the head's S they are at least 150 px
(`.stage.beside #gallery[data-density="s"]`), four across at QHD and three at
125%. The S rule's own 166 px fitted three in the column, as M does, and S
showed pressed with nothing changed (2026-10-05). The search has a row of its own, since after the heading and
Runs it had 146 px and its placeholder cut at "Search prompts, or", and its tip
hangs left, over the stage, since a fixed column does not scroll to a tip past
the window's edge. A card brought into view there has no topbar margin.

The stage keeps its own layouts in the width left of the column - the result
column, the strip - and `#recent` is a row under the viewer's note (The results
viewer), since beside both it and the column the image was 440 px at QHD. The
long-run strip keeps its `min(560px, 45vh)` cap on purpose: a cap at the
window's height would put the strip's bottom under the fold wherever chips sit
above it, where a wheel over the strip at its end goes on to the page
(The running batch on wide screens).

**The prompt check takes the column's place.** Docked in the stage's remaining
width, as the stage docks it without the tick, it took that width from the
image just made: 862 to 520 px at QHD and 620 to 354 at 125%, its top moved from
91 to 169 and its actions cut to "St…", "Refin…" and so on. In the column's box
instead, the image, the result column, `#recent` and the chips keep identical
boxes as it opens and closes (509,94,862,862 at QHD and 451,91,620,620 at 125%,
before and after). Leaving beside mode for the check, as a sheet does, was
rejected: the picture and the gallery would both move. The column is
`visibility: hidden` under it until Close, which keeps its scroll and lets
nothing in it take focus or the pointer. `checkVisibility()` without
`visibilityProperty` still reports such elements as shown, hence `isShown`
(state.js) and the new last resorts of `dismissLightbox` and the Refine pass's
focus. What it costs is that the gallery and its head - the search, the tick,
`#gallery-error` - are out of sight until Close. Its box is the column's
whatever the viewer holds: at first two of the dock's rules had no
`:not(.beside)` and outranked the beside rule - the idle rule (`align-self:
start; contain: none`, there so the check is as tall as what it holds beside a
one-line viewer) and its no-chips cap - so on a page with nothing made yet the
fixed box took its content's height, and with the critic's answer ran past the
window, its head and Close under the topbar (1839.9,0,687.1,1627.1 at QHD;
1466.1,0,548.9,1604.4 at 125%). Every dock rule that places or sizes the check -
its grid row, `contain`, `align-self` and cap, idle or not - now carries
`:not(.beside)` (its negations written `:is(.beside, :not(:has(...)))`), and
the beside section gives the check its whole box: the column's insets,
`overflow-y: auto`, `overscroll-behavior: contain`, `max-height: none`. Only the
sections' single column (`.analysis-body`) and the sticky head
(`.tag-validation-wide .analysis-head`) are the check's in both places. It was
audited with and without chips, idle, with one result and its column,
`#recent`, a pass, four images and a 5:12 image, at 2560x1305 and
2048x1040@1.25; `fitRecent`'s `!besideOn` already matched. Tab reaches the
check after the image, its tools, the note and `#recent`, as it is drawn
(`reading-flow: flex-visual` on the stage while it is open: a fixed box comes
after the flow's items).

**A job's cards wait for the pointer.** A prepend moves every card a place
along the rows, and a card about to be pressed was another by the press. So
`joinFinished`, and only where some cards are new, first waits until the pointer
is off the gallery's scrolling column - its cards, gaps, scrollbar and Load
more (`galleryPointerGone`) - released by `galleryPointerLeft` or by
`syncBeside` turning the column off, then checks again that the page joins up.
It does not wait while the column is hidden under the check. Where the pointer
is comes from `#library-scroll`'s `pointerenter`/`pointerleave`
(`galleryPointer`), as the panel's does (`panelPos.pointer`), never from
`:hover`: after a tap Chrome keeps `:hover` while the touch pointer's
`pointerleave` has fired, so a job ending after a tap waited for the next tap in
the column. Only the cards wait: the head's numbers and the outside-view line
were held too, and a keyboard user with the mouse parked on the column saw
nothing change. A column scrolled down keeps its view (`holdGalleryView`: the
first child in its view, measured across the insert), as the page's anchoring
keeps a gallery it is scrolled into; at its top the new cards come in at the top,
in view.

A sheet always opens with the page in one column, across the stage's width:
`setSheet` calls `syncBeside` first, and lands on the sheet even from the Sheet
button (The contact sheet); Sheet off puts the column back. A sheet in the
column, or in the viewer's place, was not built, because `fitSheet` measures the
sheet's chrome from the head above it. Paging is rooted on the column
(`armMore`, Paging by id). The gallery's keyboard is unchanged: `moveCard`'s
`scrollIntoView` and a focused card scroll the column, not the window, and
`isShown` keeps focus out of the hidden column. Measured: the result column's
Delete wraps to a second row at 2048x1040 at 125% with or without Gallery beside
(348 against 346 px of column) - the column's own, not this layout's.
`smoke-stage.json` checks all of it at 2560x1305 and 2048x1040 at 125%.

## The ? badges are named

Every `?` badge (`.help`, role `note`, a tab stop) is named for what it
explains - "Steps help", the grid's "Steps axis help", the Refine dialog's
"Refine size help" - and described by its own tip (`aria-describedby`), in
`index.html`. All 22 were named "Help": seven "Help, note" between the prompt
and Generate, and nothing said which. The badge stays outside its label, as
the comment over the sampling row says, and Resume's now is too: inside its
`<label>`, a press on the badge ticked Resume and took the focus, closing the
tip it had opened.

## Headings and landmarks

An idle page's only heading was "Gallery (N)", and its landmarks were an
unnamed banner, main and complementary. The word mark was a div and the
settings panel an `<aside>` with no heading, so heading and landmark navigation
could reach neither the settings that drive every job nor the viewer
(2026-10-05 area review). The word mark is now the page's one h1, drawn as
before (`.brand`, margin 0). Its two words meet with no visible space, and a
space of no width between them (`.brand-gap`, font-size 0) makes it read "Local
Diffusion", not "LocalDiffusion". The panel is a `<section>` named by its first
child, an h2 "Settings" that is read and not seen (`.sr-only`,
`aria-labelledby="panel-head"`), so it is a region landmark; as complementary
it was announced as secondary to the controls the page is for. The results
viewer is a region named Results (`role="region"`, `aria-label`), not a heading
before it. Idle, the viewer is one line, which a seen heading would double. An
unseen heading is out of the flow, and the docked stage's reading flow read it
after the gallery (The prompt analyser); a heading inside `#viewer` would not do
either, since panel.js removes the strip's first children past `VIEWER_MAX`.
`.sr-only` is out of the flow - an absolutely positioned child is no flex item
and takes no grid cell - so it is how a heading joins the panel without moving
it, and it is never a child of `.stage`. The cards a failed load or a lost
stream put in the viewer keep their own h2s. Every suggestion list - the prompt
box's, a chip editor's, the Refine dialog's - is a listbox named "Tag
suggestions" whose options have ids, the box's `aria-activedescendant` naming
the selected one and dropped when none is; the Examples menu is named by its
button (`aria-labelledby`); and the chips' spoken summary counts with `plural`
("1 prompt tag", never "1 prompt tags").

## The settings panel's sticky top

The settings panel is `position: sticky`, and its top is kept by
`web/panel.js` (`panelPos`) and written inline over the stylesheet's
`--sticky-top`. While the whole panel fits the window it rests under the
topbar at `--sticky-top` (68 px), as it always has. While it does not - the
grid fold open, the last-run banner, a short window, QHD at 125% scaling
(2048 x 1040 CSS px, where even the idle panel runs past the window) - its
top lies anywhere from resting down to the window's height less the panel's
and a gutter, a negative number at which the panel's bottom holds a gutter
above the window's (the gutter is read off `main`'s padding, the one the page
ends with, and the resting top off the stylesheet with the inline top
cleared, so a tier that changes either needs nothing in the script). Within
that range it moves four ways.

- **With the user's scrolling** (`followScroll`), by as much as the page
  moves: down the gallery it goes up until its bottom - Generate, Pause and
  Cancel, the error line, Run grid - is in view and holds there; back up, it
  comes down until its top, the prompt, is in view. The first version held the
  bottom and nothing else (4cb5a76), and the prompt came back only within
  about 180 px of the top of the page (580 px with the fold open), however
  far up the user scrolled; the review of that commit found it, along with
  the hidden focus and the moving Generate below. Only a scroll of the
  user's moves the panel: the wheel, a touch, the scroll keys from outside any
  control, a press on the page's scrollbar (Chrome dispatches it to the root
  element) or the middle button open a window that each scroll event of the
  movement keeps open and `scrollend` or a quiet 400 ms closes. The browser
  scrolls on its own too, and following those slid the panel for nothing the
  user did: scroll anchoring holds the gallery still when the viewer above it
  empties at a run's start or fills as images land, so a lifted panel crept
  up the page as images arrived and slid down at a run's start - Generate
  from under the pointer that pressed it - and the page shortening under a
  reloaded gallery did the same. Space on a button starts a run rather than
  scrolling, which is why the keys count only from outside a control. The
  panel moves a frame behind the page, since it follows scroll events on the
  main thread; that was preferred to a lag-free scheme, which a sticky box
  cannot give: holding its top at rest while the page scrolls up needs room
  above its place in its containing block, and a margin there is taken out
  of that room.
- **With its height or the window's** (`placePanel`, from a ResizeObserver
  and `resize`), keeping the bottom where it was while the bottom was in view
  (held, or resting with room to spare), else the top. While the pointer is
  over the panel a change of its height moves nothing (`panelResized`) and
  the placement waits for the pointer to leave. A held panel otherwise grows
  upwards: the progress row, the queue note and the error line appear below
  Generate, Generate rose 40 px under the pointer that pressed it, and Pause
  and Cancel took its place - with Generate still live as "Generate (queue)",
  a second press meant to queue a run paused or cancelled the first. While
  the placement waits, the panel grows downwards instead: what appears under
  Generate is in view (Pause and Cancel at 2048 x 1040), and what goes below
  the window is the fold's summary - with the fold open, Run grid - until the
  pointer leaves or the user scrolls.
- **With focus** (`revealInPanel`, on `focusin`): a control focused from the
  keyboard or by the page - Shift+Tab, a finding clicked in the prompt check,
  Use and Apply, an example picked - out of sight above the window, under the
  topbar or below the window, has its block (its field, row or button)
  brought between where the first block sits while the panel rests and where
  the last sits while it is held, by moving the panel and not the page. The
  browser scrolls the page to a focused control before it dispatches the
  focus events, measuring a stuck panel as if it moved with the page, so that
  scroll moved the page and left the control hidden (WCAG 2.4.11). Its scroll
  event comes at the next frame, so the last position the panel saw is the
  one before the focus: the scroll is taken back, and made again only if
  moving the panel could not show the control - a panel still in the page's
  flow near its top, or pinned by its containing block at the end of the
  page. A press in the panel moves nothing, through to the click, where a
  label hands its control focus: what was pointed at is in view already, and
  moving the panel between press and release would move it from under the
  pointer.
- **With the grid fold opened by hand** (`revealGridRun`, below).

A help's tip opens under its badge, and over it (`placeTip`, `.help.up`)
where under would run past the window's bottom and over has more room: the
fold's last rows - Resume, Budget, the seed list - sit at the window's
bottom while the panel is held there, and their tips ran off the window,
which scrolling could not fix since the held panel does not move with it.
Sideways, a tip moves in by as much as it would run past either edge of the
window, 8 px short of it (`placeTip`: an inline `translate` cleared at each
placement, 2026-10-05). The shift is rounded away from the edge it clears:
rounded to nearest, the LoRA weight tip's 17.4 px shift at 2560 left it 7.6 px
from the edge. The Search tip had opened rightward past a 600 px window and
widened the page, and the grid fold's LoRA weight tip, hung leftward by the
markup's `.help.end`, ran 9 px past the left edge at 2560. It is a shift of its
own, not the `end` class toggled, since badges set that class statically to
hang left inside the panel; a tip that fits is not moved.

A badge inside a box that clips its overflow (`tipClipper`, the nearest
ancestor whose overflow is not visible: the prompt check's scroller, above the
results and docked, and the Refine dialog) hangs its tip from the window
instead (`floatTip`, `.tip.floating`, position fixed; 2026-10-05). Placed
inside the check, the comparison's Batch tip was cut off by the check's edge
and lay under its sticky head; the `.compare-seeds .tip` rule that opened it
upwards is gone. The tip goes under the badge where it fits, else over it -
neither past the topbar nor the window's foot - moved in from the window's
sides, and is placed again on any scroll, the box's own included, while its
badge holds the pointer or the focus (`followFloatingTip`, a capturing scroll
listener in app.js). It is hidden while the badge is scrolled out of the box
(`.help.clipped`).

In one column `main` is `minmax(0, 1fr)`. A bare `1fr` is `minmax(auto, 1fr)`,
whose least width is the panel's min-content, and a running job's Now line -
one unbroken line - set it, so under about 680 px the page scrolled sideways
for as long as the job ran, with Cancel past the window, rather than letting
the line's ellipsis take the prompt (2026-10-05 area review).

It replaces `@media (max-height: 900px) { .panel { position: static } }`
(496bfb1), written because everything past the fold - Generate, Cancel, the
error line, Run grid - was unreachable until the very end of the gallery
released the panel. That rule never applied: it and the `max-width: 900px`
query sat above the equally specific `.panel` rule, which came later and so
won, and the panel stayed sticky at 1400 x 900 and in one column below 901
px alike. 900 px was a guess besides: the 2026-10-04 UI review (U1) measured
the panel at 1489 px with the grid open in a 1305 px QHD window and at 1089
px idle in 1040 px, both past it, while the gallery's paging moves the end of
the page away as it is reached. The `max-width` query now sits below `.panel`
and really unpins the one-column page. No height query is kept as a fallback
for a page without scripts, which cannot generate anyway, and one below the
rule would override the measurement on every window under its height.
`overflow-y: auto` on the panel was rejected: the Examples menu, Animagine's
tag menu and the right-hand tips hang out past the panel's edge, and a scroll
box clips them. The panel stays sticky at z-index 10, so its tips still paint
over the cards.

Opening the grid fold by hand - a click on its summary, as Enter and Space
on it are - brings Run grid into view (`revealGridRun`): the panel is placed
for its new height first, since the ResizeObserver reports only after the
frame (and then under the pointer that opened it), Run grid is revealed as a
focused control would be, and where the panel is still in the page's flow
the page scrolls to it (`block: 'nearest'`, smooth unless reduced motion is
asked for). It listens for the click rather than the toggle event, which
`restorePanel` and `resetPanel` set off as well: a fold restored at load
scrolls nothing. `#btn-grid`'s `scroll-margin-bottom` is the gutter, the
panel's bottom padding and its border, so the page's scroll leaves Run grid
where a panel held by its bottom keeps it, with room for the preview's reply
to add a line. `tools/smoke-wide.json` checks all of it at 2560 x 1305,
2048 x 1040 at 125% and 1400 x 900, and the page's scale at each (the wide
tier's, below). The wide tier makes the panel taller and lets the prompt box
grow with its text, by as much as seven lines: that is a change of the
panel's height like any other, placed as above. On the FLUX fake with a
one-image last-run banner, at 2560 x 1305 the idle panel is 1,132 px of the
1,216 it can rest in, and still rests; at 2048 x 1040 it was already held by
its bottom (1,071 px), and the short-window tier keeps it a little shorter
than it was (1,034 px) while 70 px wider.

## Design tokens

`web/style.css` opens with tokens on `:root` - seven type sizes, the four
greys between `--text` and `--muted`, the spacing ladder, control heights,
radii and the layout's widths - under a table saying what each is for. They
came in on 2026-10-04 (step 5 of that day's UI review plan, U92) at the
values the page already had, checked pixel for pixel against the commit
before them at 1400 x 900, 2560 x 1305, 2048 x 1040 at 125% and 800 x 900,
and by every element's computed style in thirteen states. Those shots are
of freshly loaded pages, compared viewport by viewport. Rows of the panel
sit on half pixels (the size presets are 26.5 px tall), so after a scroll or
a change of the panel's top the browser can raster them a device pixel off
from a fresh page's, with every box where it was; and a full-page or element
capture lays the page out at a passing 1 x 1 viewport, where the one-column
rule now applies and the panel goes static, so such captures of the panel
are not comparable across the move of that rule below `.panel`. The point is
the steps after it: a wide tier or a short-window tier is one media query
overriding `:root`, not a restyle of every rule (the panel stays unpinned by
measurement, never by a window-height query; see the sticky top above). So
the type scale keeps the page's seven steps (10 to 16 px) rather than the
review prototype's five: the prototype's size names where a size has one use
(`--fs-xs` to `--fs-xl`), and role names, `--fs-label` and `--fs-body-sm`,
for the two steps it does not have. A value one rule alone uses - a hairline, a
pill's padding, a chip's state colours - stays a literal, and a glyph sized
to a box of its own (the ? badge, a finding's mark, the stars, the close and
arrow buttons) keeps its size, since scaling the type should not grow an
icon inside a fixed circle. A control's height is its token: the vertical
padding is worked out from it, `(height - line - border) / 2`, which is
exactly the 7 px a ghost button always had, so a tier that raises a height
or a type token gets that height without restating a padding. `--warn` and
`--star` are the same amber with different meanings; `--accent-strong`, the
primary buttons' fill, came in as the accent itself and `--line-strong`, the
controls' borders (inputs, selects, ghost buttons, presets, the ? badges), as
`--line`, so that the step after could change them alone for contrast (below),
leaving `--line` on the panel, cards, viewer and folds. `--topbar-h` is the topbar's
set height rather than a floor under its content, so the sticky top and the
check heading's scroll margin, which read it, read the real height whatever
a tier does to the buttons in it.

**The wide tier** (step 6 of the plan; U14, U15, U84, U96, U103, U90) is
`@media (min-width: 1800px)` overriding `:root`, and a second query for a
window under 1150 px tall inside it. At 2560 x 1305 the page had been the
1400 px page stretched: a 330 px panel (12.9% of the width) of 10-12 px text
and 19 px caption buttons, a five-line prompt box showing half of a
guide-length prompt, twelve 171 px cards a row - smaller than at 1400. The
type grows with the window: `--ramp` is `(100vw - 1800px) / 760`, a length
from 0 at 1800 px to 1 px at 2560, and each size is today's plus its steps
times it, capped at 2560's - body 14 to 15.5 px, captions 12 to 13.5, sub-labels,
hints and tips 11 to 13, badges 10 to 12. So nothing jumps at 1800, and QHD at
125% (2048 CSS px) gets a third of the increase: Windows already draws its
11 px as 13.75 device px, and the review measured that panel past its window
before any change (U15's verifier would have gated the type at 2200 px for
that reason; the ramp is the plan's answer instead). The layout steps at
1800: the panel `clamp(400px, 18vw, 480px)` (461 px at 2560, 400 at 2048),
the cards `clamp(200px, 9vw, 260px)` (eight a row at 2560, seven at 2048) or,
with the gallery head's S, `clamp(150px, 6.5vw, 180px)` (eleven and nine),
the search box to 480 px, tips to 320/380 px, small counters to 72 px, the
controls taller (fields 44 px, the primary 48) and the topbar with them. In
the wide tier the prompt box grows with what is typed, `field-sizing:
content` from five lines to twelve (eight under 1150 px), capped in lines
rather than in the window's height: a box 40% of the window tall would add
400 px to a panel already past the fold at 125% (U14's rule tension). The cap
is the growth's, not the user's: dragging the grip writes the box's height
inline, and the rule applies only to a box without one
(`#prompt:not([style*="height"])`), so a dragged box keeps the height it was
dragged to, as under 1800 px. The drag was the way round a long prompt before
the tier, and a grip that stopped at eight lines took it away at 125%.
`field-sizing: content` is Chromium-only - a risk the review's recommended
direction named - so in Firefox the box stays at five lines and scrolls; the
page is driven in Edge and Chrome, and the drag still works there. The
short-window tier keeps the type and the widths with compact controls of its
own - about today's heights rather than the wide tier's (a field 40 px, the
primary 44, the small buttons 28, two pixels over today's for the larger
type in them, and the tiny ones 26), today's panel padding and topbar, and a
12 px gap where today's is 14 - so at 2048 x 1040 the panel is about as tall
as before while 70 px wider. The secondary controls followed the fields in
a later pass: the tiny buttons - the Prompt caption's Examples, Check and
Reset, the check's Apply and Use rewrite - were 24 px tall with 12 px type
at QHD, about 2 mm on a 27-inch screen, and are 30 px at a caption's type -
except a finding's Show, which sits inline in the finding's sentence: at 30 px
it stretched its line by 8 to 10 px, so there it is the line's height (no
block padding, a 1.2 line height), an inline target (2026-10-05); tick boxes
18 px; a tag chip's × a 24 px target taken from the chip's
padding, so the chip keeps its size; reference chips up to 96 px, four to a
row, badged "image N" as the prompt names them, their × 24 px; and the tools
a card, a long run's tile or a result's row show `--h-tool`, 32 px. Under 1800 px every token is today's, which is where the
smokes run and why their computed-style checks hold. The widths the review's
later work added are tokens on the same table, each with its wide-tier value
beside it: the Runs picker's `--runs-w`, the lightbox's column `--lb-side-w`,
the docked prompt check's `--dock-w`, the result's column `--result-w`,
`#recent`'s `--recent-w` and `--recent-min`, a long run's least tile
`--strip-min`, the lightbox's arrow circles `--lb-nav` (40 px; 52 from 1800 px),
the most of a prompt shown above its facts, `--prompt-cap` (twelve lines,
`12lh`, at every width), and Gallery beside's column `--library-w`
(`clamp(440px, 27vw, 720px)`) with its least card `--thumb-min-beside` (200 px).

Some of the same change is not scale but contrast and focus, and applies at
every width: `--line-strong` is `#666680`, 3.33:1 on the
panel and 3.11:1 on the fields where `--line` was 1.24 and 1.15 (U24; the
review plan's `#62627a` is 2.92:1 on a field), `--accent-strong` is `#6a4cf0`,
white on it 5.4:1 where the accent gave 4.35 (U89, U92), `--danger-strong`,
the delete dialog's Delete fill, is `#b4233a`, white on it 6.5:1 where
`--danger` - the red of text and edges - gives 3.0, placeholders are
`--muted` (5.16:1, U89), and a disabled box in one of the panel's field rows
(`.panel .sub > input`, `select`) is dimmed like a greyed one (U94) - not
every disabled input in the panel, since under 901 px the prompt check sits
there and its comparison's "original" box is disabled because it always
runs. `color-scheme: dark` on `:root` turns the scrollbars,
checkboxes, spinners and select lists dark and `accent-color` ticks the boxes
in the accent (U21); it also turns Chromium's `outline: auto` focus ring near
black on this background, so the same change gives every control the page's
own ring, `:focus-visible` at the lowest specificity (2 px of `--accent-2`,
10.4:1), which any control with a focus style of its own outranks. Checkboxes
had none at all: the text boxes' `outline: none` caught them and their border
does not render (U54). A text box's focus is two pixels of the accent, since
one pixel of it beside the stronger border was a 1.3:1 change. The LoRA select
is drawn as the text boxes are, by its id alone: a rule for every select would
restyle the comparison's selects in the panel under 901 px (U21). Danger red
on hover is a `.danger` class on Cancel, a queue row's ×, Cancel queued and
Delete unstarred, not every small ghost button (U91) - and on Reset since it took the panel
with no way back (U68; "Panel persistence") - and a tip lets clicks through
(`pointer-events: none`, U139): one pinned open by a click lay over the Refine
dialog's scale buttons and swallowed the clicks meant for them. A tip's line
breaks are kept only where its text has them (`.tip.pre`, set by the code
that writes it), not on every wide tip, which broke a tip written in the
markup wherever its source lines fell.

## The panel's Sampling and Output rows

The review plan's step 13 gave the panel Sampling and Output headings, a
Conditioning fold for the LoRA and references, the dead negative box collapsed
to one line, and neutral size presets with an underline for the recommended
ones. Of those, the Sampling caption came in with U92 (it groups the sampler
boxes), and the rest was superseded by fixes with a measured reason behind
them. The Conditioning fold was not built: U95's own recommendation replaced
it with two CSS cuts - a greyed reference field's empty chips row hidden, and
an empty `#run-hint` taking no line - which took 57 px off the panel, and the
review's risk list said the fold would hide the LoRA select from anyone who
uses adapters every time. The negative's collapse went for the readability of
a carried-over negative (True CFG is its own field), and the presets keep
U81's mark rather than an underline (The prompting guide: the selection reads
first). The Output caption, which U95 had called optional, came with U45.

**The Output group** (2026-10-05). U45 found the seed box 107 px wide at 1400,
showing eleven of a seed's sixteen digits: the 41 px dice square and its gap
in the row took the room of six. Its advice was not to add to the 296 px row,
and this takes from it. Seed, Batch and Repeat form a group (`#output-caption`,
the Sampling caption's twin, role=group), one line at every width, with
`#run-hint` under the row inside the group. The dice (`#btn-dice`) sits in the
Seed name's line, as the ? badges do, the line's height, its target reaching
past its frame (`::after`: 6 px up - to the caption where it shows - and 4 px
down to the box, 25 px or more in all), so the seed box has the row less the
two counters: 156 px at 1400, for a 16-digit seed's 121 px. The Seed field is a
div with `<label for="seed">`, since a button inside a label becomes the
label's control; the tab order is the dice, Seed, Batch's ?, Batch, Repeat. The
first fix wrapped Batch and Repeat under the seed, which with the caption put
Generate wholly below the 900 px window at scroll 0 (top 977.5 against 884), and
the review sent it back. With the row one line, the caption's line and gap
(24 px) were the only height the group added, and that still put Generate below
the window at 1400x900 (top 908, against 884 with 16 px showing before), so the
recheck sent that back too. Shown from 1800 px whatever the window's height, it
did the same on short wide windows: Generate's top at 901.5 at 1800x900, and
5 px cut off its bottom (951.9) on a maximized 1080p browser at 1920x947. The
caption now shows only in one column and in the wide tier on a window taller
than 1150 px, where the Output group is exactly as tall as the Sampling one
(Generate 1000-1048 of 1305 at 2560). Elsewhere it is the group's name alone,
visually hidden (absolute, 1 px, clipped) - the row's own names say what it is,
as LoRA's and Weight's do - under `(900px < width < 1800px),
(min-width: 1800px) and (max-height: 1150px)`, the complement of the one-column
query and the wide tier, fractional widths included. There the group is its
row's height, and Generate's top at scroll 0 is 884 at 1400x900, 877.5 at
1800x900, 883.6 at 1920x947 and 890.9 at 2048x1040 at 125%. It is hidden at
2048x1040 at 125% too, where with it Generate fit (915.6-959.6): the caption
follows the compact controls' one breakpoint rather than a height bound of its
own measured against today's panel. Tightening the panel's gaps on short
windows was the alternative; it would have moved every field under 1800 px,
where the smokes are written, for 24 px. smoke-wide holds 1400x900, 1920x947 and
2048x1040 at 125%.

**References and LoRA under the grid fold** (2026-10-10). Testing found the
panel short of height: on QHD at 125% (2048x1040) a running batch's bar sat
at the window's foot (1038 of 1040 on the FLUX.1 fake, a two-line run hint)
and the elapsed line under it (`#cell-note`, 1098) past it. At scroll 0 the
panel is in the page's flow, and a sticky top - however negative - never
lifts a box above its place in the flow, so nothing in `panelPos` could show
them without the page scrolling, which it never does by itself. What sat
between Sampling and Output was the reference field and the LoRA row, both
rarely used and greyed on most checkpoints; they now sit last in the panel,
under the grid fold's summary, and everything from Output down moved up by
their 105-107 px: Generate's top at scroll 0 is 776.5 at 1400x900, 775 at
1800x900, 780.3 at 1920x947, 786.4 at 2048x1040 at 125% and 884.3 at
2560x1305, and a batch's bar and elapsed line end at 934 and 993.5 at
2048x1040 (1044.6 and 1108 of 1305 at 2560). At 1400x900 and 1920x947 the
elapsed line still runs past the window at scroll 0 (978.5, 985.3; the bar
920.8 and 926.6) and comes into view as the page is scrolled, as before; the
topbar's mirror shows the job at every size. No gap or control height was
tightened for it, which would have moved every field under 1800 px. The
grid's LoRA weight axis, which used to point up at the adapter, says LoRA
below; opening the fold at 1400x900 now brings Run grid into view with
References and LoRA still under the window rather than holding the panel by
its bottom, Run grid no longer being its last block. Tab from the panel's
last stop reaches the stage as before; that stop is now the LoRA row's.

**The run hint under Generate, the counters on the time's line**
(2026-10-10). The review of the move above sent it back: at 1400x900, the
size the smokes are written at, a running batch still showed only its Now
line, and at 1920x947 the step bar and the time ran past the foot. Two things
stood between Generate and the time that a job's own lines could do without.
The run hint ("12 images: 3 batches of 4, one after another, seed +4 each,
from a random start.") sat in the Output group above Generate, two lines
that came the moment Batch or Repeat passed 1 - exactly when a batch runs -
and pushed Generate down 41 px as they did. It now sits under Generate,
after `#progress`: Generate no longer moves as Batch and Repeat change, and
while a job runs its progress comes between, so the hint, not a control,
moves (Batch and Repeat are described by it, since it left their group).
And the progress block gave each bar a 20 px row for its counter beside it
and the time a line of its own: 78 px under the Now line. The bars are now
two full-width lines and the counters share one line under them with the
time ("batch 1/3 · 13/20  4s elapsed · ~14s left"), 42 px: the rows are
`display: contents` in a wrapping flex line, so a long grid cell line wraps
whole to a line of its own under the counters rather than in a column. On
the FLUX.1 fake (a two-line negative caption) a 4 x 3 batch's progress ends
at 895.1 at 1400x900, 901.2 at 1920x947 and 906.6 at 2048x1040 at 125%, all
at scroll 0; before, 978.5, 985.3 and 993.5. At 1536x864 (1080p at 125%
with no browser frame) the Now line, Pause and Cancel are in view and the
bars (864.5-875.5) and time (895.1) are not; fitting them too would mean
tightening the gaps or the box heights of every field under 1800 px, and a
real browser at that scaling is shorter still. The alternatives looked at:
a sticky-bottom `#progress` would sit over Generate on any window shorter
than the panel down to it, putting Cancel under the pointer that pressed
Generate; hiding the run hint during a job would lift Generate under it.
smoke-narrow runs that batch at all three sizes.

## The live preview

**The live preview is a linear map from the latents, keyed on the VAE.**
`_step_end` hands `cb_kwargs["latents"]` to `latent_preview`: unpacked
through the pipeline's own `_unpack_latents` (FLUX's 4-arg staticmethod,
Krea 2's 3-arg method returning 5-D; klein has none and gets no preview),
multiplied by a 16x3 matrix fitted by least squares on the card from
pipeline-normalised latents (`(z - shift) * scale` for FLUX's
`AutoencoderKL`, `(z - mean) / std` for Krea 2's `AutoencoderKLQwenImage`;
R² 0.96 / 0.95 on real images), with a gain of `1/(1 - sigma)` because the
callback's latents are x_t not x_0, a 3-tap blur against the leftover
noise, nothing at all above sigma 0.9, and never on the last step. At most
one frame per 0.4 s, a ~5 KB JPEG at twice latent resolution as a data URL
in a `preview` SSE event. The whole thing sits in a try/except that logs
once and switches itself off: an exception out of that callback is how
`Cancelled` ends a job. `jobs.on_preview` checks `job.subscribers` **per
frame**, not once per cell - the browser opens its stream after the POST
returns, by which time an idle server is inside the first cell, and the
once-per-cell check skipped every first cell (found on the card, fixed).
The matrices cannot be validated in CI; the fit script and its date are
in the module docstring, and the frames were looked at.

A refine pass has a stand-in until its first frame: SDXL sends none, and the
viewer showed whatever image was there before for the whole pass, with nothing
naming the source. `runRefine` hands `startJob` a `placeholder` (the source's
full image - the thumbnail was a fourfold upscale at QHD - the result's size,
"Refining #N → W×H · strength s"), and at the job's first progress event with
no frame (for a pass a listing between jobs adopted before its POST answered,
the first after that answer: One watch per job) `streamJob` clears the strip
and shows the source dimmed and softened
at the result's aspect, captioned with the step and the time left. It is sized
as the result will be - a tile of the strip (`viewerTile`, "The results
viewer"), the picture at the result's aspect and its tier's height, the caption
where the result's tools will be - where a fixed 68vh height with the width
clamped cropped the source and stood taller than the result, which jumped the
viewer when it landed. A frame or the result replaces it, as either replaces
the other; it is never a record. The live preview is such a tile too
(`previewTile`), its frames in place of the picture and the step under it,
where it was a fixed 320 px image a third the size of the result it gave way
to. Both go in at the strip's end through `stripAppend`, which a long run's
scrolling strip follows ("The running batch on wide screens").

## FLUX.1 dev and schnell, Z-Image Turbo, Qwen-Image

**FLUX.1 [dev], FLUX.1 [schnell] and Z-Image Turbo are in the catalogue**
(`launch.py`), each with a lite-infer re-upload so the nunchaku menu offers
them. dev is the `flux` profile as it stands. schnell has its own entry
*ahead of* `flux`: 4 steps, guidance 0 - its transformer has
`guidance_embeds: false`, so the CFG box never reaches the model and only
True CFG does - and a scheduler shipped `use_dynamic_shifting: false,
shift: 1.0`, i.e. unshifted. Z-Image Turbo is `ZImagePipeline` (6B, Qwen3
text encoder): 8 steps, CFG 0, and it guides from zero like Krea 2, which
`_GUIDES_FROM_ZERO` already covered; `z-image-turbo` sits ahead of a
`z-image` entry carrying the pipeline's own 50 / 5.0. It ships the FLUX
`AutoencoderKL` and normalises latents the FLUX way, so the fitted preview
matrix applies - but its latents are never packed and it has no
`_unpack_latents`, so `latent_preview.unpack` passes a 4-D latent through
and still answers None for a 3-D packed sequence with no hook (klein). Its
scheduler is `use_dynamic_shifting: false, shift: 3.0`: the pipeline
computes a resolution-dependent mu and the scheduler ignores it, so a blank
mu is a fixed shift of 3 (mu = ln 3) and a typed one goes through the
static-shift fallback and lands exactly. `test_launch.py` derives the
"another repo id" option number from the catalogue's length. Measured
2026-09-08 under Nunchaku NVFP4 on the 5080, model offload, the CLI
in-process with torch's peak allocated (`~/local-diffusion/measure.py` in WSL):
Z-Image Turbo 1 MP in 8 s at 4.1 GiB, 2 MP in 14.5 s at 4.6 GiB, an 11 s
load, and preview frames at steps 2/4/6 of 8 in the right colours; FLUX.1
dev 1 MP in 30 s and 2 MP in 48 s at 7.4 / 7.8 GiB; schnell 1 MP in
16-23 s and 2 MP in 25 s at the same peaks - most of a schnell image is the
per-image cost that does not scale with steps (T5 under bitsandbytes NF4,
the weight transfer), so it is not the speed win Z-Image is. The desktop
holds about 5,200-6,100 MiB of the card while Unreal Editor and the browsers
are open, and a run that lands within a gigabyte of the top spills into system
RAM (schnell's 2 MP read 35-45 s that way and 25 s after the baseline fell to
1,270 MiB). **Qwen-Image is in the catalogue too** (`QwenImagePipeline`, Krea
2's parent: same VAE, same unpack hook, FLUX-style `true_cfg_scale`), with
a `qwen-image` profile - 50 steps, guidance 1.0 which the transformer
ignores, true CFG separate at threshold 1.0 - and **the pipeline runs its
uncond pass only when a negative prompt is present**, the same gate FLUX.1
has; the True CFG hint says to type one. Measured the same way with true
CFG 4 and a negative prompt: 11.75 GiB peak at 1 MP and 12.1 GiB at 2 MP,
78 s per 1 MP image and 152 s at 2 MP once warm (210 s for the first of a
session), a 22 s load, and the rows record true CFG 4 and the negative
prompt. Its lite-infer re-upload is 12 GB of transformer at NVFP4.

## ERNIE-Image Turbo: the prompt rewriter, and a mu with no way in

**ERNIE-Image Turbo is in the catalogue** (`baidu/ERNIE-Image-Turbo`,
`ErnieImagePipeline` in diffusers 0.40; Baidu's 8B single-stream DiT, Apache
2.0), with lite-infer's re-upload - named without the `_r32` the other lite
ids carry, so the entry spells its id out instead of using `_LITE`. The
profile is `ernie-image-turbo` ahead of an `ernie-image` family entry at the
pipeline's own 50 / 4.0: 8 steps at guidance 1.0. Its
`do_classifier_free_guidance` is the classic `> 1.0`, so `cfg_threshold()`'s
else-branch answers 1.0 with no change to `_GUIDES_FROM_ZERO`; it takes a
`negative_prompt` and no `true_cfg_scale`, so to the controls it is klein
with the negative prompt back. **Its repo carries a prompt rewriter**: `pe`,
a 7.7 GB bf16 Ministral 3B with its own tokenizer, in the class's
`_optional_components`, and `__call__` defaults `use_pe=True`, under which
every prompt is rewritten by `pe.generate(do_sample=True, temperature=0.6)`
before it is encoded. A hidden rewrite would make the recorded prompt differ
from what the model saw and the image seed would not reproduce the whole
operation. `loading.UNUSED_COMPONENTS` therefore still maps the class name to the
two components and ordinary `load()` reads `model_index.json` ahead of the
weights (`_unused_components`) to pass `pe=None, pe_tokenizer=None` - **to the
concrete class named by the index**, not to `DiffusionPipeline`. The base
class resolves the same class from the same file, but `_get_signature_keys`
is a classmethod that reads `cls._optional_components`, which on the base is
empty: through it `pe` is not an expected module, `pe=None` becomes a pipeline
kwarg, `download()` never filters `pe/` out and the loop loads the model - seen
on the card (7.7 GB pulled at load, 11 GB of RAM taken, never called) before
the switch. Through `ErnieImagePipeline.from_pretrained` the two are passed
components: `download()` drops their folders from `allow_patterns` and
`load_module` skips them.

The explicit **Expand** editor action is the exception. `POST
/api/prompt/expand` waits for the normal model load, refuses to cut ahead of
an image job, then loads `pe` and `pe_tokenizer` from the component classes in
the same manifest on first use. They are registered on the pipeline and its
configured offload hooks are rebuilt so the large rewriter moves to the chosen
card only for its call. The selected width and height reach ERNIE's JSON user
message; the sampled result replaces the textarea only if the user has not
edited it while the load was in flight (held under it where the box was typed
over: Expand on the page, below). Later expansions reuse the component
until Unload. `generate()` still sets `use_pe=False` wherever the switch
exists: the expanded text is visible, editable, and is exactly what the row and
PNG record. The rewriter's bundled chat template contains its Chinese system
instruction and does not render an added system-role message, so asking the
same model for English through `_enhance_prompt_with_pe(system_prompt=...)`
does nothing. After expansion, a separate
[`Helsinki-NLP/opus-mt-zh-en`](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)
Marian model therefore translates the result. It is a 312 MB, CC-BY-4.0,
Chinese-to-English checkpoint loaded lazily through the Transformers and
SentencePiece packages already in the app, kept on CPU so it never disturbs
the image pipeline's offload chain, and dropped by Unload. The Chinese text
remains in the editor and is the only prompt sent and recorded; the English is
a read-only aid, cleared by the first edit of the prompt it translates. Translation failure is
returned separately so a useful multi-GB expansion is not thrown away. The
index read is a try/except that yields nothing on failure - an optimisation,
not a gate - and the tests' conftest now refuses `loading.hub_file` for every test,
since a stubbed `from_pretrained` had been the only thing between a loader
test and the Hub. **mu has no way in**: the pipeline builds `linspace(1, 0)`
itself and hands it to a scheduler shipped `use_dynamic_shifting: false,
shift: 4.0`, and `__call__` takes no `sigmas` and no `mu`, so `_mu_supported`
answers False, `applied()` drops the value and the row records none; the
shift that lands is the constant 4 (mu = ln 4), which the hint says. Sides
are multiples of 16 (`AutoencoderKLFlux2`, 32 latent channels, 2x2-patchified
into the 128 channels the callback carries as a 4-D tensor), which every
preset already is; `factors_for` has no map for that VAE, so the preview
switches itself off once at DEBUG and the cell pays nothing for it - a fit
would need the card and the pipeline's `bn_mean`/`bn_std` normalisation.
Baidu lists 1 MP shapes only; 2 MP and 4 MP render. **Measured 2026-09-08
in the evening**, Nunchaku NVFP4 on the 5080 with model offload, the CLI
in-process (`~/local-diffusion/measure.py`), Z-Image Turbo re-run the same
evening for a like-for-like: ERNIE 1 MP in 9.6-10.2 s once warm (17-18 s
for the first image of a process, 22 s load) at 5.2 GiB, 2 MP in 24.3 s at
5.9 GiB, a 4 MP pair in one call in 48.6 s at 10.3 GiB allocated / 12.0 GiB
reserved; Z-Image 10.6 s, 12.8 s and 23.5 s at 4.1 / 4.6 / 7.4 GiB (11.8
reserved) - level at 1 MP, twice the time above it. **Under Nunchaku a seed
does not reproduce bit-for-bit**: the same request in two processes gave
two hashes on both checkpoints, Z-Image's pair 4.5/255 apart on average
with 17% of pixels off by more than 8 (a different seed: 36/255, 85%),
ERNIE's 14-24/255 and 58-75% (a different seed: 41/255, 94%) - visibly
different details on the 8-step model, not rounding. The fp8 path on
Windows reproduces (see `verified.md`), so a duplicate count or a re-render
hash is that path's guarantee only. One measurement trap: right after WSL
reclaimed memory (`Cached: 0`) Z-Image read 17-19 s per image, 10.6 s on the
next run; the card was idle and boosting fine. And the 7.7 GB the base-class
load had pulled was deleted from the cache before the concrete-class load
was verified, so that fetching nothing was observable.

Baidu's card lists its seven 1 MP sizes under "Recommended Parameters"
(checked 2026-10-05). Both ERNIE guides now carry them as `resolutions` and
their framings as `aspects`: 1:1, 3:4/4:3 and 2:3/3:2 are those sizes exactly at
1 MP, and the 9:16/16:9 preset (768x1360) stands for 768x1376, which the size tip
says. Before this every 1 MP preset counted as in the guide, 12:5 included, so
the strip carried no marks, and Baidu's own 768x1376 typed in was not recognised
(The prompting guide: a typed size takes the nearest framing).

**Expand on the page** (2026-10-05 area review; the 2026-10-04 review never
covered Expand). Expand is held as every other held button is: aria-disabled
with the reason as its title (`syncExpandButton`, its one writer: while it
runs, during this page's job or another tab's, during a critique, or with the
server offline), never `disabled`. A first use runs for minutes, and the
disabled button had dropped a keyboard user's focus to `<body>` for all of them
and after every failure, while a press during a job or a critique did nothing
and said nothing. A press on it held now says why, as a sentence, through
`expandSay`: this page's job's reason is a job note gone at the job's end
(`jobNote`), another tab's is said until that job ends (`elsewhereNote`), and
while a Reuse's Undo holds the panel's line it is said under the prompt, never
over the Undo - written into the panel's line, it had taken the Undo and the
panel it would have put back. A press while it runs says what the result will do as things
stand (`expandOutcome`, the test its end applies): written in, held, or dropped.
Running, it reads "Expand…": "Expanding…" was 23 px wider, and at 1400 px
wrapped Reset onto a second caption line, moving the prompt box down for the
length of the request; reserving that width would have made the wrap
permanent.

What a first use costs is the status's to say, from file checks only, as the
critic's `describe()` is (`ModelStatus.prompt_expander`, `ExpanderStatus`:
`loaded`, `fetch` [{name, bytes, step}], `fetch_bytes`; from
`backend.prompt_expander_status()`, `loading.prompt_expander_cached` and
`MarianTranslator.cached()`): the rewriter's `pe/model.safetensors`
(`loading.PROMPT_EXPANDER_BYTES`, step `expand`) and the translator's `.bin`
(`translation.MODEL_BYTES`, step `translate`), which the page prints as about
7.1 GiB and about 297.6 MiB (`humanBytes`, binary units, as the critic's line). It
says so in Expand's title before the press and, each at its own step, in the
line under the prompt while it runs. The translator's load turns off
Transformers' safetensors conversion for itself (`DISABLE_SAFETENSORS_CONVERSION`
inside `_without_conversion`, the previous value put back after the load):
having loaded the `.bin`, Transformers had started a thread that fetched the
conversion bot's safetensors copy from an open pull request on the repository -
a second 312 MB, in the user's cache two minutes after the first, never read and
never named. The running line says what the expansion waits on, in the server's
order - first the card, while something else holds it (another tab's critic or
Expand, or a job another tab queued during the load: `ModelStatus.card_holder`);
then the model's load, the rewriter's first download or its load, the
expansion, then the translator's first download - beside an aria-hidden clock,
following the status (`syncExpandRunning`). Another tab's critique or Expand
holds the card through `run_when_idle`, and the load waits for it too: the line
had said "Expanding…" for the whole of their run (2026-10-05). The press sends
a `ticket` of its own (`PromptExpansionRequest.ticket`, a random id), which the
status answers as `card_ticket` while that expansion holds the card, so the
page tells its own press from another tab's ("Waiting for the card, which
another tab's Expand has, then …"); `refreshStatus` asks for the line again as
the holder or the ticket changes. The status is asked a quarter second
after the press, so the topbar says loading at once, and again as Expand ends,
so the title stops naming a download that is done. The line takes a line of the
panel's flow, as a held expansion does (`data-kind`): as the overlay a 15 s
Example note is, over a live negative box (ERNIE at CFG above 1), it hid the
box's caption and first lines for minutes.

Everything Expand says about the box goes in the note under the prompt
(`#prompt-note`, `showPromptNote`), where it was pressed. In the panel's line
under Generate it was out of sight at 1400x900, and its last word there wrote
over a Reset's or a Reuse's Undo still on offer, taking the button and the
panel it would have put back. The note holds one thing at a time, by
`promptUndo.kind`: 'write' (an Example's or an expansion's Undo, gone once the
box holds anything else), 'info' (gone the same way, with no button), 'reset'
(Reset's Undo, under Reset's own rule, lying in the emptied box itself,
`data-at="box"`, at every width and never in the flow), and 'running' and 'held' (Expand's
running line and a held expansion), which typing does not take, have no timer,
and so always take a line of the flow. Each expires after its time unless focus
is on its buttons, and `onDrop` runs whatever takes it; a Reuse's Undo and this
note retire each other. Its failures go to the panel's line, said as Expand's
("Expand did not finish: <reason>. The prompt is unchanged.", the class name
dropped server-side by `api_prompt.expand_detail`, which keeps it for a message of
one word, the log always), and while a Reuse's Undo holds that line they go
under the prompt instead, as a 20 s note (`expandSay`). A load that failed is a
503, which the page answers with a pointer to the viewer's card: repeated in red
with its backticks, the sentence was said twice. Beside images, where there is
no card, the sentence is said once with its commands in quotation marks. What
the server would refuse in pydantic's words - a prompt past 4,000 characters
(`EXPAND_MAX_CHARS`, held equal to `PromptExpansionRequest`'s `max_length` by
`tests/test_ernie.py`), a W or H that is not whole pixels in range - is refused
at its box, in the panel's words.

A result whose box changed while it ran is still never written over the change,
but it is no longer thrown away when the change was typing: it is sampled and
cannot be made again exactly, so it is held under the prompt (Use it, Discard).
Where the box was rewritten whole - a Reset, a Reuse, an Example, a Use
(`replaceBoxText`'s `replacingBox`) - the user replaced it on purpose, and the
result goes without a word. An Example's Undo or a held expansion that the
running line took is offered again where the expansion fails. The write is
`rewriteBox`'s, with an Undo of its own - "Prompt expanded: …" - that brings
back the replaced prompt's English too, and Ctrl+Z gives the replaced text back:
assigning `.value` had emptied the undo history, so an expansion, a long rewrite
that is hard to get again, could not be undone. The caret and both boxes' scroll
start at the top: the caret at the end had scrolled a long expansion's opening
lines out of the box. The translation is shown after the write, whose input
event clears one.

The English is no longer cleared at the click, so a failed second Expand leaves
the first one's; only another expansion or a change of the prompt clears it. It
rides in `panelState()`, so Reset's and Reuse's Undo and a reload bring it back
while the prompt is the one it translates. From 1800 px it heads the results
(`placeTranslation`) as a box of its own, as tall as its text up to 40vh, with the
expansion's note at its end: in the panel its four rows showed a fifth of a long
expansion, and the field and a note in the panel's flow (ERNIE's negative is
greyed at CFG 1, which keeps the note in the flow) pushed Generate below a
2048x1040 window at 125% while the results column stood idle. The viewer and the
gallery move down by the box's height when an expansion lands; that is the
user's own press. Beside the docked check it takes the first row of the columns
left of it, with the viewer and `#recent` a row down (CSS, the no-chips variant
only, since ERNIE's prose never shows chips; its rows repeat the dock's
selector, so a change to the dock changes them too). Made a div so the note could
sit in it, its label names only "English translation"; the hint
(`#prompt-translation-hint`) is the box's description (`aria-describedby`) and no
longer says the Chinese prompt is "above", which it is not at the head of the
results. There is no Cancel: the server holds the generation gate until the
expansion ends, so releasing the page would only queue a Generate behind it.

## Animagine XL 4.0: an SDXL schedule is not a Mu

**Animagine XL 4.0 Opt is in the catalogue**
(`cagliostrolab/animagine-xl-4.0`, CreativeML Open RAIL++-M and open). The
repository publishes a normal `StableDiffusionXLPipeline`: two CLIP encoders,
an SDXL UNet, `AutoencoderKL` and an Euler scheduler. Its Diffusers component
weights total about 6.5 GiB. The 2025-02-13 Opt commit changed the component
tree's UNet as well as adding `animagine-xl-4.0-opt.safetensors` at the root,
so `DiffusionPipeline.from_pretrained` gets Opt without a single-file loader.
That commit also changed the manifest from `EulerAncestralDiscreteScheduler`
to `EulerDiscreteScheduler`. The model card still calls Euler Ancestral the
preferred sampler, but the detailed example does not replace the manifest's
scheduler. Follow the current checkpoint rather than overriding it from stale
prose; a sampler control would need to become part of `CellKey`, the PNG and
the database before it could be changed reproducibly.

**A `sigmas` parameter is not proof of flow matching.** Current
`StableDiffusionXLPipeline.__call__` takes custom sigmas, so the old generic
`_mu_supported` test would have treated Animagine's epsilon-prediction Euler
scheduler as flow matching and replaced its schedule with `schedule.shifted_sigmas`.
Mu now requires a scheduler class from the `FlowMatch` family first. The
capability is part of the checkpoint profile before load and `describe()`
after load, so the main Mu box and its grid axis are disabled as well as a
stray request being dropped from the record. This also gives ERNIE's known
no-Mu pipeline the disabled control its tooltip had already promised.

**Use the checkpoint's tag vocabulary.** The model card says natural-language
input may not be effective. Its order is person count, character, series,
rating, the remaining subject/pose/scene tags, and quality tags last; both CLIP
tokenizers stop at 77 tokens. It recommends 25-28 steps (28), CFG 4-7 (5), a
specific negative prompt, and nine roughly-1-MP shapes. The six previously
missing aspects - 9:7, 7:4 and 12:5 plus their inverses - now use the card's
exact 1 MP dimensions. The existing 3:2 pair keeps ERNIE-Image's nearby sizes,
and every Animagine aspect is available across all four tiers. Its guide also
names those nine ratios in `aspects`: recommendation highlighting requires both
the named framing and the pixel range, so unrelated 1 MP presets such as 16:9
do not turn green merely because their area is close. `resolutions` carries the
card's exact nine dimension pairs for free-form W/H input; exact matching admits
the rounded 1216x832 "3:2" without loosely aliasing 16:9 to nearby 7:4.

**The Tags picker is guide data, not a hard-coded Animagine branch.**
`PromptGuide.tag_helper` carries the card's person-count, rating, quality and
score vocabularies, its `year {n}` form, recommended quality suffix and negative
prompt. The button appears only when that object is present. Each tag is an
explicit editor action at the current caret; active tags toggle off, exclusive
person/rating choices replace one another, and the suffix action moves that
known set to the end. Character, series, subject, pose and scene stay free text
because the card describes those slots but publishes no finite vocabulary for
them. The helper drops under the prompt box, the box's width, over the
negative prompt and the sizes: hung from its button it covered 209 of the
box's 296 px, caret included, so its own "insert at the prompt caret" pointed
at a caret nobody could see, and a click in the box to move it closed the
helper. A click in the box now moves the caret and leaves the helper open,
and its second line names the place - "A tag goes after "absurdres", at the
end" - recomputed on every insert and caret move (since 2026-10-08 it names
the steps that fit there too: "One order, three places" below). It does not slot tags
itself: the documented order differs per checkpoint (Kohaku's rating is
last, Illustrious's third, WAI's folded into subject and scene), the page
has no order of its own, and Check's Use reordered places them. The text is
the chips' 12 px, and the window is its only height cap: at that size
Animagine's helper is 573 px and outgrew the 520 px it had, its closing note
behind a scrollbar. Opened from the keyboard it takes focus; Escape gives focus
back to its button. Typing in the box with the helper open brings the
suggestion list over it, and an Escape there closes that list alone (the
document's Escape skips one already taken); the next closes the helper, and
the one after that an open prompt check (The prompt analyser). The
prompt remains visible, editable, persisted and recorded exactly as
before; the helper never rewrites on generation. Its documented limitations
are text, difficult anatomy, multiple characters, post-cutoff
characters and degradation at high resolutions such as 1536x1536. Sources,
checked 2026-09-10: the model card and repository
(<https://huggingface.co/cagliostrolab/animagine-xl-4.0>) and the Opt commit
(<https://huggingface.co/cagliostrolab/animagine-xl-4.0/commit/4688bebb86806957f6f83c39f2573161630e22db>).
The class signature, profile, guide and no-Mu application have automated
coverage. The user then confirmed on 2026-09-10 that the weights loaded and
the output looked good on this card; the exact quantizer and offload mode were
not recorded. Speed, VRAM, LoRA behavior, live-preview fit and torchao
quantization therefore remain unmeasured.

**The complete vocabulary is Danbooru's, not a list in the checkpoint.** The
maintainer explicitly says XL 4.0 ships no wildcard or character list because
it trained from nearly all Danbooru image data, and directs users to Danbooru's
counts. Danbooru's official `/tags.json` API supplies canonical names,
categories, post counts, normalized multi-name lookup and alias-aware prefix
search. `backend.tags` proxies that fixed HTTPS origin off the event loop and
caches prefixes, exact hits and misses; the browser sends only the active tag
prefix, or the comma-separated tag names with the prompt they were split from
(which stays on the local server, for the order slots below), never an image
or another generation setting; only tag names reach Danbooru.
Suggestions exclude empty/deprecated tags and anything created after the
checkpoint's 2025-01-07 knowledge cutoff. Model-card-only controls such as
`great score` and `year 2023` validate locally; finite special tags are also
merged into autocomplete and remain available offline, where they are the
whole list. They used to be merged ahead of every remote result, which made
"gre" offer `great score` before `green hair` and "lo" offer `low quality`
first - the item Enter and Tab take. `tags.rank_suggestions` orders the
list: an exact match, then names matching from their start, then at a later
word, then alias matches; within that the index's post-count order, a guide
tag going ahead of it only once the query covers half of it ("high s" is
heading for `high score`); and a tag of the guide's recommended negative
prompt is never first unless typed whole. The page selects the first item
only from three typed characters and when it starts with them, in the box,
in the chip editor and in the Refine dialog's prompt alike, so Tab after "lo"
moves on and Enter is a line break; ArrowDown still selects. The list calls the guide's own tags
`guide`, as the chips do.
A green chip means a model tag
or a nonempty Danbooru tag at the cutoff; red means unknown, deprecated,
empty or too new, with the reason as a `code` beside the sentence
(`not_found`, `deprecated`, `empty`, `no_date`, `after_cutoff`): a tag too
new for the checkpoint shows ⏲ where a typo shows ?, red either way, since
the fixes differ, and matching the sentence would break with its wording.
An unavailable API makes ordinary tags unchecked, not invalid, and never
blocks generation: presence in the index is useful
evidence, not proof that a particular caption reached training. The one
exception needs no index: an item longer than any tag (`tags.TAG_NAME_MAX`,
200 characters, such as a hybrid prompt's sentence) is answered unknown
(`tags.too_long_result`: `code` not_found, "longer than any Danbooru tag")
without asking it, reachable or not (`tag_lookup.lookup_tags`,
`tags.unchecked_results`), and only a blank item or one past 4000 characters
is refused (`TagValidationRequest`). Every item had been bounded at 200, so
one 213-character sentence on Kohaku refused the whole request (a 422), and
the tray read "Danbooru unreachable" (2026-10-05). The tray
says which of three states it is in (`data-state`): checking (… on the
chips waiting, title "Checking Danbooru…"), checked, or unreachable - the
label amber, "Danbooru unreachable, not checked", never red, guide tags
still green and the rest marked –. Checking and unreachable used to look
the same, and the checking title claimed an outage. The chips ask about the
first 64 items, the request's own ceiling, and mark the rest unchecked
"past the 64-item lookup cap", the label saying "first 64 of N checked": a
65-item prompt was one refused request that left every chip, guide tags
included, waiting for good. What the index said is kept per checkpoint (a
cutoff and guide tags change the answers), so a pause in typing renders
every known chip at once and only a new item waits: every chip used to go
grey and dashed at each pause and back to green a moment later. The item
being typed - a keystroke or a deletion in it, not a whole-box write - has
no verdict while it is unknown (`.typing`, no mark) until a comma, the caret
leaving it, a blur or 1.5 s: the one red chip was the word still being
written. A known one shows at once. Sources,
checked 2026-09-10: the maintainer's answer
(<https://huggingface.co/spaces/Asahina2K/animagine-xl-4.0/discussions/4>) and
Danbooru's tag API documentation
(<https://safebooru.donmai.us/wiki_pages/api:tags>).
On a two-column page the single validation node is moved into a panel above the
wide results area, where every chip wraps without a nested scrollbar and its
height cannot move Generate in the settings column. At 900 px and below it
returns under the prompt in normal document flow, where the whole page is
already a single scrolling column. The chips and the prompt check each have a
`display: contents` host of their own there - `#tag-chips-wide`, then
`#tag-validation-wide` - and both go back into `#tag-validation-home` under
901 px, chips first; one shared host had kept Tab from reaching the image just
made before the docked check (The prompt analyser).

**Those chips are a round-trip editor, not a second prompt.** HTML drag/drop
moves a comma item and immediately serialises the whole order back through
`writePromptTags`; Alt+Left/Right is the keyboard equivalent. Every chip has an
× button and accepts Delete/Backspace while focused. A context click in any
flex gap finds the nearest wrapped row and the boundary between the adjacent
chip centres; a context click on a chip inserts after it, as the Menu key on
a focused chip does. Its
visible listbox reuses the same cached tag-suggestion endpoint; a native
`datalist` did not reliably reopen when asynchronous results arrived. Click,
Enter or Tab accepts a suggestion (the first is selected as the box's is, from
three characters it starts with), while + commits literal text and Escape or ×
cancels, focus going back to the chip or beside the gap. A plain click in a
gap inserts there too, the pointer over a gap shows where with the drop's
accent bar, and a "+ tag" button after the last chip inserts at the end -
the visible way in for a pointer that does not right-click and for the
keyboard, where right-click had only a 10 px hint that moved with the last
chip. The hint is pinned to the tray's foot above the results. A gap for a
plain click and the marker is within a row of chips, from its first chip to
the last row's + tag, and within half the row gap above or below it: the
label before the chips and the hint under them are text, and a click on
"click a gap or + tag to insert" opened an editor in the middle of the
prompt. A right-click still takes the nearest place. A double click on a chip (Enter or F2 while it has focus) opens the
same editor in the chip's place, holding its text, so a typo is fixed where it
sits rather than by deleting the chip and inserting a new one; suggestions
start with the first keystroke, not with the opening, so Enter on an untouched
edit keeps the tag instead of taking the top match for it, and an unchanged
commit only closes. One editor is open at a time, and any prompt change from
outside it - the textarea, a drop, a restored panel - closes it, because its
index no longer names the item it opened on. Every edit goes through the same
preview, validation and panel-persistence path, so the visible textarea,
generated prompt and chips cannot become separate sources of truth.
`writePromptTags` writes the box through `writeBox` - analysis.js's
`replaceBoxText`, then no suggestion list or typing state over the new text -
so Ctrl+Z restores what the chips changed: assigning `.value` emptied the
undo history, and in Chromium a later Ctrl+Z could splice two prompts into
one, which the panel then saved. An accepted suggestion replaces the word
at the caret with an insert, one undo step from what was typed, and the
recommended negative prompt is written the same way. The write takes focus
to the prompt; it goes back to the chip the edit leaves in place (the moved
one, the one after a deleted one, the edited or inserted one) or to the
helper's button that made it, and every rebuild of the tray - a pending
render, then the lookup's answer - puts focus back on its chip until focus
leaves the tray, taking it only from `<body>`, where the rebuild left it:
one Alt+Arrow used to drop it there for good. A chip's visit to the prompt
is only in passing, so it moves nothing: the focus scrolls nothing, and
the panel's `revealInPanel` skips it as it skips a press in the panel - a
panel held by its bottom at 1400x900 jumped 62 px at each Alt+Arrow in the
other column. The helper's writes, whose focus does come back to the
panel, still reveal the box. The tray is one tab stop: the
chip last focused (or the first) takes Tab, arrows, Home and End walk the
chips, × is out of the tab order (Delete does the same), and "+ tag" is the
second stop. A chip's label says what it is and how to act on it; the tray
is a labelled group, not a live region - it holds every chip and is rebuilt
at each pause in typing - and a hidden polite summary says "N prompt tags:
K known, U unknown" once a lookup answers, not counting the item still being
typed, as the label does not. A focused chip draws a 2 px teal ring of its
own after every state rule, at full opacity past the cut.
Chip and suggestion text is 12 px. A valid chip uses high-contrast light text
on neutral charcoal, with green confined to its checkmark and outline; unknown
chips keep the corresponding red mark/outline instead of colour-on-colour text.
A valid chip's left edge carries the category the lookup returned, in
Danbooru's colours (general blue, character green, copyright purple, artist
red, meta yellow), and teal for the guide's own tags: what a tag is.

**Where a tag goes is the server's to say** (2026-10-05; the review's U85
option, left out the day before). A quiet caption at the chip's start, set off
from the mark by a 1 px `--line-strong` rule (`--line` did not read on the
valid chip's fill), names the step of the guide's documented order that Check's
order rule places the tag in. It arrives with the chips' verdicts:
`POST /api/tags/validate` takes an optional `prompt` and answers `order_steps`
and `slots` (`prompt_rules.order_slots`), placing with `_tag_categories`'
categories for that prompt through `_places`, the one placement the order rule,
`reorder_tags` and `unplaced_items` read. The review had rejected a slot map in
the page - Kohaku puts the rating last, Illustrious third, and WAI folds it into
subject and scene - and the left-out option was waiting for per-guide order data
the server already had. The route asks `prompt_rules.places_items` first, so a
prompt with nothing to place costs the index nothing, and `slots` is None for
the whole prompt there (`order_checked` false, or no items). It once asked
`reads_order`, whose two-item gate belongs to the order finding: a lone tag
got no slot, and the Tags helper's caret line, after 1girl was clicked into an
empty prompt - the moment it is most used - said it could not tell where the
next tag went; a lone item is placed now, as it would be beside any other. An item
gets no slot where it holds lettering or keeps more than one possible slot - an
@ name, a tag never looked up, or one with no usable category - so a chip never
shows a guess. `slots` covers every comma item, past the lookup's 64 too; a
blank prompt, or one longer than `TAG_SLOT_PROMPT_MAX` (Check's 4000), counts as
no prompt rather than a 422 that would leave the chips unchecked; and a failure
placing the slots costs the slots, never the verdicts answered beside them. The
page works out no slot itself. The caption is a `::before`
(`.tag-check-label[data-slot]`, lowercase), so the chip's text stays the tag's
own; the step is also in the chip's title and aria-label, and the steps form the
tray title's legend only while some chip has one. Between a change and its
answer a chip keeps the last answer's step for the same tag (`chipSlots`),
because every typing pause had blanked the marks and redrawn them. Checked at
one line per chip at 1400x900, at QHD and at 800 px. A
later copy of a tag (the same canonical name) has its text underlined in
amber dots, "repeated", as Check's heuristic says - on the text, not the
outline it first had, which overrode the focus ring, so the arrows landing
on a repeat showed nothing. An unknown chip's reason, with
"generation is still allowed", is in one shared tip outside the tray,
shown under the chip as it is hovered or focused and named by its
`aria-describedby`, so the chip's text stays the tag's own.

**One order, three places** (2026-10-08, user feedback). The Tags helper's
caret line - "A tag goes after "x", at the end - click in the prompt to move
its caret" - named a place but no order, and the user found it useless; and
the Prompt ? text, the Tags menu and Check disagreed. What was found: on
NoobAI and RouWei the menu read an empty "Recommended order:" (their
`tag_helper.order` is `[]`, deliberately) and still pointed at Check's Use
reordered, which they do not have, while their ? text quoted the authors'
orders (NoobAI's caption template, RouWei's artists at the very end) and
Check said "This app checks no word order for this checkpoint: its guide
states none" above those same orders; Animagine's menu and Check name the
steps Person count, Character, Series, Rating, Details and Quality, and its
? text named neither Details nor where year, artist and meta tags go (Check
places them in Details), nor that the menu's Score and Year groups go in
Quality and Details; Illustrious's ? text said "keep it short or put the
quality tags early" beside an order with quality sixth of seven, which
Check flags; Kohaku's said the menu carries quality, rating and date tags,
leaving out its person count; NoobAI's menu had two "Year" headings, one
for the date tags its ? text calls date tags; WAI's menu put Rating under
no step though Check places it in Subject and scene.

All three now read one source. The menu's order is served with it
(`_served_tag_helper`): `documented_order`'s steps and cite, Check's own,
or where none is checked the `author_orders` Check lists, from one
function. Each group says its step where that is not its name. The caret
line says which steps fit at the caret, read off the chips' server slots
for the nearest placed tags either side - the page places nothing; an
unplaced tag is passed over as the order rule passes it over - and marks
them in the order line and on the groups. Where no tag around the caret
is placed, the line first said Check gave them "no step yet", which read as
loading though often no answer was coming (the order not read for the
prompt, or a failed request); it now says "still being looked up" only while
this prompt's answer is pending, and otherwise names the whole order and
why the caret cannot narrow it. The ? prose was corrected where
it disagreed (Animagine names the steps, in a sentence marked as the
app's in `prompt_sources`; Illustrious's token advice no longer moves
quality forward; Kohaku lists person count; NoobAI's group is Date), and
Check's no-order note drops "its guide states none" where author orders
follow. `tests/test_tag_order_agreement.py` is the guard: served order and
author orders equal to Check's answer over HTTP for every tag checkpoint,
every menu group placed in a step, distinct headings, every step named in
the ? prose, and no "quality tags early" under an order.

**The chips show where the encoders stop reading.** On these SDXL
checkpoints both CLIP tokenizers stop at 77 tokens, and the guide's own
quality suffix, written last, is the first thing a long list loses - while
it showed four green ticks. `POST /api/prompt/measure` is Check's encoder
layer alone (`prompt_encoding.measure`, no rules, no tag lookup: a Check's
64 category lookups at every pause would be a second lookup beside the
chips'), on a daemon thread like Check, never `ensure_loaded` or the gate.
(The verdict request does run Check's category lookup for the order slots,
above, but only where the order rule reads an order in the prompt, and the
cache the chips' own lookup has just filled answers it: the index is asked only
about a name beyond the chips' 64 that Check's 64 reach - guide tags are
skipped - once per name, misses cached, and never again when the chips' lookup
could not reach it.)
The tray asks it 600 ms after typing stops, with its own epoch. A chip past
the earliest cut of an encoder that reads the words (not FLUX.1's pooled
CLIP-L) is dimmed and dashed (`.past-cut`), an amber bar where reading
stops; the one the cut splits keeps a dashed far edge, "partly read". The
count is in the tray's label - "101/75 tokens", amber when over - with no
element of its own, so every child of the tray is a chip, the editor or
"+ tag". An encoder left uncounted (a tokenizer still loading past
`LOAD_WAIT`, no network for a first fetch) is "cut unknown", never "no
cut". A Check of the same text serves until the count lands, and the last
count serves while the text up to its cut is unchanged, so a tag typed at
the end of a long list is past the cut at once. A first measure on a cold
server may fetch the checkpoint's tokenizer files before anyone presses
Check; that is the tray's to do, not `/api/status`'s, which still loads
none.

**A line counts where no chips do** (2026-10-05 area review). The count lived
only in the tray's label, and `scheduleTagValidation` returned before asking
`/api/prompt/measure` whenever there were no chips, so a checkpoint without
them - NewBie's XML, FLUX.1's T5, the Qwen and Gemma encoders, a hybrid
checkpoint's sentence - had no count between Checks. Yet NewBie's own two-character XML
example is 485 of Gemma 3's 511 tokens (488 with names filled, measured with
the real tokenizers), and a three-character version is 698, cut at code point
1464 inside the third character's `<clothing>`, so `<general_tags>` (count,
style, background, quality) is never read. The measure now runs on the same
pause with or without chips. Where none show, `#prompt-tokens`, a line under
the prompt box (`syncPromptTokens`, its one writer), appears once the prompt
has used three quarters of the tightest encoder's budget. It gives that count
and every encoder as tight (SDXL's two CLIP towers), each encoder's count in
its title; past the limit it is amber and names where reading stops (`tagCut`,
the excerpt Check's cut link gives). Under three quarters it is hidden: shown
always, its line took Generate past the edge of a 1400x900 window and said only
that there was room. An encoder left uncounted (a tokenizer still loading or
one that cannot be fetched, the server not answering) is "Tokens: cut unknown",
never "no cut", since there is no count to hold to the three quarters. The
page measures the first 4000 code points (`PROMPT_MAX_CHARS`, tag-tokens.js, held
equal to `PromptMeasureRequest`'s and `GenerateRequest`'s `max_length` by
`tests/test_api_analysis_tags.py`). Past them the line leads with "Over 4000
characters: Generate and Check refuse it", in amber, then the count of what
was counted and its cut, and the summary line says it once (the chip tray's
label says "over 4000 characters" the same way). Before, the measure refused
the whole prompt, and the line read "Tokens: cut unknown" in its quiet
colour, taking back the warning the prompt had already earned (2026-10-05).
The exception to "cut unknown" is a checkpoint the encoder table has no entry for (an
off-catalogue `LOCAL_DIFFUSION_MODEL_ID`): nothing will ever count that prompt,
and "cut unknown" stood under every prompt, two words included, moving
Generate's top from 888 to 911 px at 1400x900, so the measure says so
(`PromptMeasure.encoders_known`, False only there) and the line stays hidden;
Check's Encoders section gives the reason, and hidden is not "no cut". A
measure that got no answer is marked failed and asked again for the same text.
`refreshStatus` asks again when `loadedModel` changes - on a tag checkpoint the
helper's change did that, on one with no helper nothing did, and a prompt
restored at load had no count until it was typed in - and when the server
answers after an offline spell, where the line had kept "cut unknown", titled
with the offline sentence, until the next pause in typing. The line is no live
region, being rewritten at every pause: the change to past the limit is said
once in `#tag-summary`, which no chips are using then, and cleared on the way
under so the next crossing is said again. It sits inside `.prompt-editor`,
under the box at the field's 6 px gap, so the Example's Undo note, the Tags
helper and the suggestions hang under it - in the field's flow, the note lay
over it (the encoder's name and where reading stops) on NewBie, whose own
examples reach the line - and a click on it closes the suggestions. Whether the
line shows moves everything under it in the panel by a line, 23 px at 1400x900,
and it lands a pause and a round trip after the typing stopped, when the mouse
may be arriving on Generate. So while the mouse is on a control at or below the
prompt box's bottom (`promptTokensAimed`, set by the panel's pointerover), that
change waits (`promptTokensHeld`) until the mouse leaves that control or the
panel, or a key is pressed, which no mouse press is; its text and colour change
in place. The hold applies only on such a control: held over the whole lower
panel, a prompt past the limit went unsaid until the mouse left the panel. The
route stays the encoder layer alone, cold and on the CPU, and XML never becomes
chips. A first typing pause on a cold server may now fetch any checkpoint's
tokenizer files, not only a tag checkpoint's; `/api/status` still loads none.

## Illustrious XL v2.0-STABLE: a single file converted once

The official `OnomaAIResearch/Illustrious-XL-v2.0` repo has only the 6.94 GB
`Illustrious-XL-v2.0.safetensors`, not a `model_index.json`. Merely adding it
to the catalogue would fail before loading under the default FP8 component
scan. Diffusers 0.40's pipeline `from_single_file` also does not distribute a
`PipelineQuantizationConfig.quant_mapping` to its CLIP loaders. Rather than
invent a second quantization path, `backends/single_file.py` converts the
official checkpoint to an unquantized bfloat16 component tree once on CPU.
The ordinary loader then scans and quantizes that tree with the same
per-library TorchAoConfig instances as every other model.

Only this exact catalogue repo id takes that branch. The source is pinned to
`69459c1fe6f46db41ab31e6114f05acc0e06bcaa` (STABLE), and the SDXL base
descriptors to `462165984030d82259a11f4367a4eed129e94a7b`; the config download
allows JSON and tokenizer text only, never SDXL base model weights. The
conversion fixes epsilon prediction and Euler Ancestral, the sampler used
in the authors' report, and saves that scheduler in the tree. This is a
fixed model configuration, not a per-request sampler setting.

The cache is `$HF_HOME/local-diffusion/converted`, separate from the Hub's
managed snapshots. Its key hashes the source, config and versioned recipe.
A process lock covers conversion, and a temporary directory is renamed only
after `save_pretrained` and a recipe marker complete. Failure cannot publish
a partial tree. That temporary directory removes itself on the way out, but
not after a hard kill, nor on Windows whenever its cleanup meets a file still
open or still being deleted. Each such attempt could leave a partial tree as
large as the converted one (about 5 GB for Anima to 15 GB for NewBie), and
the cleanup's error was raised over the conversion's own, so a failed
conversion reported WinError 32 instead of why it failed. The work directory
is created with `ignore_cleanup_errors=True`, so the caller hears the
conversion's own error, and `single_file.sweep_work`, which all four
converters call first under the same lock, removes what is left on the next
load of that checkpoint, converted or not. A conversion holds that lock
until it is done with its work directory, so anything the sweep finds is
stale. It takes only the exact tree name, a hyphen and `tempfile`'s eight
characters (`[a-z0-9_]`), and only a real directory (not a link or junction)
holding nothing but `tree`: another checkpoint's work, a published tree and
a user's copy named like one are left alone. Removal is best-effort
(`ignore_errors`, no network); a directory the sweep cannot list is kept,
one it cannot remove yet is logged as such and tried again on the next load,
and nothing in the sweep fails a load (2026-09-28 review, F39). The same tree
supports FP8, int8 and none; Nunchaku is refused before downloading. Both
original and converted weights remain on disk, about 14 GB total. Only the
loader's local settings copy sees the cache path: status, profiles, cell keys
and PNGs retain the official repo id.

The app's starting values are 28 steps and CFG 5. The report gives >20 steps
and CFG 5-7.5, not a unique default; the authors' v2.0 release example runs
28 steps at CFG 6.5 (written as 2.5 under DPM++ 2S Ancestral CFG++, with
their own note that it "should be 6.5 in recent implementations"), and the
control hints quote both. Its v1.0+ native range is 1-2.25 MP, and
up to 4 MP carries some loss. Figure 19's direct 2048x2048 result is explicitly
v1.0; it is family context, not evidence that STABLE was measured here. The
release post adds that v2.0 "supports 512~1536 resolutions", trained to 1:10
in multiples of 32, and then renders its own example at 1824x1248; the size
hint reports both as written.

The prompt hint opens with what the authors say the checkpoint is, because
that is why a bare prompt disappoints next to Animagine: the v0.1 card calls
the family "untuned BASE model ... not tuned for aesthetics / preferences"
and notes "The model does not have any default style. This is intended
behavior for the base model"; the v2.0 release post repeats it - "not
trained with aesthetic set", released "'untuned'" as a "merging / training"
base - and the report says the non-fine-tuned release is deliberate, with
styles associated with artist names. Animagine 4.0 is an aesthetic-tuned
finetune whose score tags pull on training it received; Illustrious has
nothing for such a tail to pull on, so the hint sends the style to an
artist or style tag (the Danbooru lookup already labels the artist category)
and passes on the v0.1 card's composition advice: one of `upper body`,
`cowboy shot`, `portrait` or `full body`, and not several conflicting ones.
The two are separate sentences in the prose: the style one is the app's
reading of "no default style" and is `unsourced` for the critic
(2026-09-28), while the composition one is the v0.1 card's ("We suggest
using suitable composition tags like...") and links it.
The v2.0 examples demonstrate natural language, but the report also describes
every Illustrious version as tag-based and gives its caption order: person
count, character, rating, general tags, artist, quality and year. The shared
Danbooru autocomplete and chip editor therefore apply to Illustrious too, with
the archived v2.0 card's `2024-08` dataset update represented by an inclusive
`2024-08-31` vocabulary cutoff. The picker exposes the terms written in the
report - `1girl`/`1boy` from its example prompts, the six quality tiers in table
3 and the five year modifiers in table 4 - plus a Rating group holding
Danbooru's four rating names, `general`, `sensitive`, `questionable` and
`explicit`. That group exists because the rating slot cannot be filled any
other way: a Danbooru rating is a post field, not a tag, so the index carries
those words only as deprecated, empty placeholders that autocomplete drops and
validation refuses, and a local group is the one path that accepts them. The
report itself writes only `general` (its example prompts, and its note that
`general` in the prompt is a simple safety control); the other three are the
dataset's vocabulary, not the authors', and the Prompt hint says so (the
release post also lists `sfw`, `nsfw`, `r-18` and similar as working
alternatives; the hint does not repeat them). Animagine's `safe`/`nsfw`
spelling, quality suffix and numeric `year {n}` form are not copied over.
Because Illustrious also supports natural language, spellcheck remains
enabled and a single sentence is not rendered as one unknown tag chip;
comma-separated tag prompts get the complete editor. At the user's request,
the Prompt hint also turns the documented quality-then-year order into a
clearly qualified practical starting tail, `masterpiece, newest`; it stays
out of a Finish-with button because no author prompt ends that way (theirs
end in `masterpiece` alone, or put it mid-prompt).

The negative prompt was first left empty, on the reading that the authors had
published none for v2.0, with an app-invented baseline offered as
"experimental". That reading was wrong, and it hurt: the v0.1 model card runs
both of its examples with one negative prompt, verbatim `worst quality,
comic, multiple views, bad quality, low quality, lowres, displeasing, very
displeasing, bad anatomy, bad hands, scan artifacts, monochrome, greyscale,
signature, twitter username, jpeg artifacts, 2koma, 4koma, guro, extra
digits, fewer digits` (the second example appends `jaggy lines, unclear`),
and the v2.0 release example runs `worst quality, bad quality`. Since
2026-09-19 the helper's `negative_prompt` is the v0.1 card's list, so the
Tags menu offers the same "Use the recommended negative prompt" button
Animagine has, and the hint names both forms and their source rather than
an app recipe. The `displeasing`/`very displeasing` pair in it is the only
trace of aesthetic modifiers in any author text; the report says such
modifiers were used in later epochs and defers details, so no positive
aesthetic tag is suggested. The example menu now holds three author prompts
that fit in 77 tokens: the report's figure 3 prompt, the v0.1 card's
silhouette prompt and the release post's forest scenery; the post's
natural-language example is ~200 tokens and would be truncated silently, so
the hint quotes only how it ends (`persona 3, masterpiece`, tags after prose).
Community guides read for this - SeaArt's release-day user guide, which
fronts `masterpiece, best quality, amazing quality`, adds `very aesthetic,
newest`, puts CFG's sweet spot at 4.5-5 and says the model follows negatives
as closely as positives - are not authors' text and contribute nothing to
the hint. Their "Clip Skip 2" is diffusers' SDXL default already
(`hidden_states[-2]` in `encode_prompt`), so nothing needed changing there.

Sources checked 2026-09-17 and 2026-09-19:
- Official release/card and files: <https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0>.
  Its "ILXL Image Generation User Guide" link (`illustrious-xl.ai/updates/21`)
  and the site's blog and updates pages answered 404 on 2026-09-19.
- Archived v2.0 card with the dataset-through-2024-08 statement:
  <https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0/blob/f9885efc7cbd20d0291dbbb0d5fe810ea143911f/README.md>.
- The authors' v2.0 release post on Civitai - untuned base, natural-language
  example, 512-1536 and 1:10 in multiples of 32, the 1824x1248 example with
  its negative prompt and settings, the rating tokens - read through
  CivArchive because Civitai itself refuses this region:
  <https://civarchive.com/models/1369089?modelVersionId=1546777>.
- The v0.1 card: sampler, steps and CFG, the quality tiers, the composition
  advice, "no default style", and both example prompt/negative pairs:
  <https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0>.
- Authors' report, B.2 settings, B.1.4 language, figure 19 size, figure 3 prompt:
  <https://arxiv.org/html/2409.19946v1>.
- Diffusers single-file API: <https://huggingface.co/docs/diffusers/api/loaders/single_file>;
  the installed 0.40 `load_single_file_sub_model` confirms the CLIP quantization boundary.

Validation on 2026-09-17: 333 Python tests and the 15-check Illustrious browser
smoke passed. The 6,938,040,674-byte official checkpoint was verified against
SHA-256 `c2a1a3eaa13d4c107dc7e00c3fe830cab427aa026362740ea094745b3422a331`.
Conversion succeeded with the installed Diffusers 0.40 / Transformers stack,
including the checkpoint's extra CLIP `position_ids` buffer (the loader skips
non-state keys). The cached tree is 6,945,302,591 bytes. A second process reused
it without conversion and loaded the FP8 pipeline in 7 seconds. All three test
PNGs retained the official model id and requested settings, not the cache path.
GPU samples and their quality limits are recorded in `verified.md`. The
2026-09-19 prompting rewrite passed 335 Python tests and the 19-check
Illustrious smoke on the fake server; no GPU sample was taken for it, since
it changes text and one button, not the pipeline.

## NoobAI-XL V-pred 1.0: pinned SDXL with an explicit V-prediction schedule

`Laxhar/noobai-XL-Vpred-1.0` selects the official Diffusers tree at revision
`66aa55e3469c27c29a89813cd35dd95fb7485fa1`. It needs no single-file conversion.
The resolver fetches configs before quantization checks, then only the component
weights and tokenizers; the duplicate single-file checkpoint is excluded.
Settings, rows and PNG metadata retain the public repository id.

The published scheduler config incorrectly declares epsilon prediction and
leaves zero-terminal-SNR rescaling disabled. The author's Diffusers example
explicitly replaces it with Euler, `prediction_type="v_prediction"` and
`rescale_betas_zero_snr=True`. The loader applies those settings before the
pipeline can generate. It also uses `timestep_spacing="trailing"`, following
Diffusers' guidance to start from the final training timestep when rescaling
the noise schedule. This is an integration choice beyond the author's two
explicit overrides: at 28 steps the supplied leading schedule starts at 946,
while trailing starts at 999. CPU tests inspect the real scheduler's timesteps
and sigmas and its use by the ordinary SDXL Refine sibling.

The exact-id profile starts at the author's example values, 28 steps and CFG 5.
Mu is unavailable because Euler's noise schedule is not flow matching. Normal
FP8, int8 and bfloat16 component loading is supported; no Nunchaku build is
mapped. Refine starts at strength 0.4 and shares the corrected scheduler.
There is no four-channel SDXL latent-preview map, and the Refine preview does
not invent a flow start-sigma value for this scheduler.

The guide keeps NoobAI's quality/date vocabulary and author caption-order
excerpt separate from other SDXL models. The card gives both a quality prefix
and a caption order without defining their combined placement; the analyser
does not impose an invented total order. Examples inserts the card's complete
Diffusers prompt (76 tokens including special tokens in both published CLIPs).
The earlier prefix-only menu incorrectly assumed that prompt was too long.
Both CLIP encoders have a 77-token sequence,
about 75 content tokens. Danbooru suggestions use the stated approximate
2024-10-23 cutoff; the app has no e621 vocabulary lookup. A tag-index match
does not prove that this checkpoint learned a tag.

Size help recommends about 1 MP and retains the author's seven exact reference
dimensions. Shared presets select nearby framings and do not exactly reproduce
every listed size. Larger tiers remain available without a claim of author
support or local quality. GPU evidence belongs in `verified.md`; fake-browser
coverage is `tools/smoke-noobai.json`, port 7891.

Sources checked 2026-10-07:

- [Pinned author card](https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0/blob/66aa55e3469c27c29a89813cd35dd95fb7485fa1/README.md).
- [Published scheduler config](https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0/blob/66aa55e3469c27c29a89813cd35dd95fb7485fa1/scheduler/scheduler_config.json).
- [Diffusers noise-schedule guidance](https://huggingface.co/docs/diffusers/using-diffusers/scheduler_features#rescale-noise-schedule).

## RouWei 0.8 epsilon: an author checkpoint with hybrid prompting

`Minthy/RouWei-0.8` selects the author's
`rouwei_080_epsilon_fp16.safetensors`, pinned to revision
`95b524750edf2dcbc84967ded3d1bc99deed3b53`: 6,938,040,808 bytes, SHA-256
`1aba15decd15da1810054ed4c58984610dfafc728bef2fb3f3ebe846b2e287a4`.
The repository also contains a V-pred release, base weights and FP32 variants.
They are separate choices; this integration keeps the epsilon checkpoint's
identity and schedule. The Hub's generated Diffusers snippet does not turn
the single file into a loadable component tree.

`single_file.ROUWEI_RECIPE` uses the existing CPU conversion and pinned
SDXL descriptors, verifies the source hash, and retains the source VAE and
both CLIP encoders. The converted tree has epsilon-prediction Euler Ancestral
and uses its own recipe-derived cache key. The source and converted weights
take about 14 GB together. Normal component loading then applies FP8, int8
or bfloat16; Nunchaku is refused before conversion. Settings, metadata and
resume keys retain the public model id.

The exact-id profile starts at 28 steps and CFG 6, within the author's
20-28 and 4-8 ranges for epsilon. Mu is disabled because this is SDXL's
noise schedule. Its ordinary SDXL Refine sibling starts at strength 0.4.
That pass resizes and denoises an image; it does not reproduce the author's
latent or GAN upscaler. Direct-generation guidance is about 1-1.5 MP, with
dimensions divisible by 64. The app's preset grid is divisible by 16, so
the size hint retains the author's narrower recommendation without
claiming every preset follows it.

The guide supports tags, prose and mixtures, with a 2025-04-30 vocabulary
cutoff. Tags use spaces, artists use `by `, and the recommended negative is
`worst quality, low quality, watermark`. The author prefers artist styles
in a separate CLIP chunk, with putting them last as a fallback. This app
uses that fallback: ordinary SDXL reads 77 tokens per CLIP, about 75 of
content, and neither prompt chunking nor a `BREAK` operator is implemented.
An artist at the end still has to fit inside that limit. Keep this explicit
in the prompt help; the checkpoint's training on long captions cannot
extend the inference path's text budget.

Source facts and implementation choices are distinct from local quality
or performance measurements. The fake-browser recipe is
`tools/smoke-rouwei.json`, port 7890; actual card results belong in
`docs/verified.md`.

Sources checked 2026-10-07:

- [Author's model card](https://huggingface.co/Minthy/RouWei-0.8/blob/95b524750edf2dcbc84967ded3d1bc99deed3b53/README.md).
- [Pinned epsilon checkpoint and hash](https://huggingface.co/Minthy/RouWei-0.8/blob/95b524750edf2dcbc84967ded3d1bc99deed3b53/rouwei_080_epsilon_fp16.safetensors).
- [Pinned SDXL descriptors](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b).
  The author's licence section points to the original
  [Illustrious release](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0).

## WAI Illustrious v17: an author checkpoint through a pinned mirror

`LyliaEngine/waiIllustriousSDXL_v170` selects WAI0731's v17 full SDXL
checkpoint through a community transport mirror. The mirror is neither the
author's repository nor a LoRA, despite misleading Hub metadata. Its card
is not authoritative for the loading recipe or permissions. The primary
author source is the Tensor.Art v17 release page. The file's SHA-256 matches
the author's Civitai version 2883731 as recorded by CivArchive; direct
Civitai access returned HTTP 451 during integration.

The source is pinned to revision
`5ef4e2da7173a160ad04aebcaa2fdcd6d20ed792`, filename
`waiIllustriousSDXL_v170.safetensors`, 6,938,040,682 bytes, SHA-256
`f116b0c78ff441467b0cdc8f1936e1ed18ea31e9997c7b132b1b8db533f0bd04`.
`single_file.py` verifies that digest before converting. It shares the
Illustrious CPU conversion path and pinned SDXL base descriptors at
`462165984030d82259a11f4367a4eed129e94a7b`: only JSON and tokenizer text are
downloaded from the base, never replacement weights. The checkpoint's VAE
and both CLIP encoders are retained. The saved scheduler is epsilon-prediction
Euler Ancestral, matching the author's Euler a recommendation.

The existing lock, temporary tree, recipe marker, atomic rename and the
sweep of an interrupted conversion's work directory protect the conversion
cache. WAI's distinct source and digest produce a separate cache key; the
existing Illustrious recipe remains unchanged. The cached tree feeds normal
per-component FP8, int8 or bfloat16 loading, and Nunchaku is refused before
conversion. The original and converted weights occupy about 14 GB together.
Only the loader sees the cache path: status, saved metadata and resume
identity retain `LyliaEngine/waiIllustriousSDXL_v170`.

The source contains NaNs in CLIP-L's final transformer layer (layer 11).
They are preserved as source weights, not silently replaced or sanitized.
Standard SDXL prompt conditioning selects this tower's penultimate hidden
states (`hidden_states[-2]`) and takes pooled conditioning from the second
text tower. Consequently the contaminated final CLIP-L output is not a
conditioning input. A focused regression preserves that distinction by
checking the consumed conditioning rather than rejecting an unused output.
Changes to hidden-state selection must revisit this assumption.

The exact-id profile starts at 28 steps and CFG 6, within the author's
15-30 and 5-7 ranges. These are starting settings, not a measured optimum.
Mu is unavailable because the scheduler is not flow matching; negative
prompts use ordinary CFG above 1. The guide does not inherit Illustrious
v2.0's untuned-base description, native size range, ratings or negative
prompt. WAI has its own author-supplied positive quality recipe and negative
prompt. The quality group is nonexclusive because all three positive tags
are intended together, while the four rating buttons are exclusive.
Examples inserts the author's verbatim quality recipe rather than an
invented scene, and the hint explains that the subject still needs adding.
The family release notes describe natural-language support, so the helper
keeps spellcheck and single-sentence prose behavior. Standard SDXL CLIP's
77-token context applies; the author's weighted-prompt syntax is not parsed.

WAI publishes no dataset cutoff. Its Danbooru helper therefore passes
`None` to the tag client, which omits only the creation-date filter and
still excludes deprecated and empty tags. Existing dated profiles keep
their limits, including when they share cached tag records with WAI.
WAI's quality and rating tokens validate locally; remote matches remain
advisory vocabulary membership, not evidence of checkpoint training.

The author recommends original dimensions above 1024x1024 and describes
1024x1344 example inputs followed by a separate 1.5x hires refinement pass:
20 hires steps, R-ESRGAN 4x+ Anime6B, denoising strength 0.35-0.5. No maximum
is published. The existing guide contract uses the documented reference
span, 1.048576-1.376256 MP, with `pixels_word="example"` and
`pixels_source="the author's reference sizes"` since 2026-10-05
(`"documented"` before, which the size line quoted as what the checkpoint was
documented for: The prompting guide); the tooltip explains that this is
neither a maximum nor an equivalent to the refined examples. Generate renders directly at the requested size. Since 2026-10-03
the Refine pass (its own section above) is the nearest thing here to that
hires pass - 1.5x and a strength in the author's range, measured on this
checkpoint - with a Lanczos resize where the author uses the upscaling
model, which the size tooltip says. The fake-server smoke is `tools/smoke-wai.json`
on port 7873; actual card measurements belong in `verified.md`.

Sources checked 2026-09-26:
- [Author's v17 release card and permissions](https://tensor.art/models/990778216270553015/WAI-illustrious-v17).
- [Pinned community transport repository](https://huggingface.co/LyliaEngine/waiIllustriousSDXL_v170/tree/5ef4e2da7173a160ad04aebcaa2fdcd6d20ed792).
- [CivArchive's author-model record, including version 2883731](https://civarchive.com/models/827184/wai-nsfw-illustrious-sdxl).

## Z-Anime Base: a nested Diffusers tree

`SeeSee21/Z-Anime` is the next anime candidate after Illustrious. The author
publishes Base, distilled, GGUF and AIO weights in one repository, with the
Base component tree under `diffusers/`. The card's suggested pipeline-level
`subfolder="diffusers"` is not implemented by the installed Diffusers 0.40
`DiffusionPipeline.from_pretrained`/`download`: they still seek a root
`model_index.json`. Passing the nested local directory works through the
existing loader and its component-specific quantization path.

`loading.resolve_model` pins revision
`0f5fb51464638a2f7328a1d74590281e63e1fde2`, downloads only the manifest and
component/scheduler configs first, then lets the usual quantizer preflight
run. Only after that succeeds does it materialize the `diffusers/`
subtree, excluding its incorrectly packaged text-encoder weights (see below).
Both passes share the revision, so the preflight cannot inspect one
version and load another. Runtime settings, status, run keys and saved
metadata retain `SeeSee21/Z-Anime`; ordinary ids and local directories pass
through unchanged. Nunchaku is refused before weights because the denoiser is
unquantized, with no catalogue Lite counterpart.

**The supplied encoder is not loadable as declared.** The real first load
reported 253 unexpected `*.scale` tensors. Inspection of the cached
`text_encoder/model.safetensors` found 253 FP8 tensors and scalar BF16 scales
(for example, layer 0's Q projection scale is 0.0013275146484375), but its
config has no quantization declaration. Transformers drops those scales;
casting the raw FP8 values to BF16 or applying torchao cannot restore them.
The initial render run was stopped and is not verification evidence.

The card identifies the default text encoder as standard Z-Image Qwen3-4B,
the one the model was trained against. `loading.component_overrides` therefore
loads `Qwen3ForCausalLM` from the official `Tongyi-MAI/Z-Image` parent's
`text_encoder/`, pinned at `04cc4abb7c5069926f75c9bfde9ef43d49423021`, and
passes it into the pipeline. It receives the existing Transformers-owned
`quant_mapping["text_encoder"]`, or BF16 for `none`; the denoiser still uses
the normal Diffusers-owned config. This does not select the separately
advertised Engineer V4 alternative. The source encoder weights are excluded
from new downloads. The corrected combination totals about 20.7 GB (19.3 GiB)
in the HF cache; a machine that previously downloaded the malformed encoder
also retains that 4.8 GB until the owner clears it.

The profile uses the author's Diffusers example: 40 steps, CFG 4, Mu blank.
The real `ZImagePipeline` exposes generator and step callbacks, guides above
zero, and accepts negative prompts. The guide favours descriptive sentences
and carries the author's portrait prompt template in Examples, and since
2026-10-05 the card's "Backgrounds & Landscapes" template too; a pick says
which slots it leaves to fill and selects the first (The prompting guide), and
Check notes a missing negative prompt, which the card highly recommends. There is no
Danbooru tag helper. The card documents approximately 512x512 to 2048x2048,
so the guide says **documented** 0.26-4.2 MP rather than claiming a measured
training range. Its 1024x1536 and 1920x1088 suggestions are explained in the
size tooltip; the latter is already the 16:9, 2 MP preset.

The pinned scheduler is `FlowMatchEulerDiscreteScheduler`, static shift 6,
with beta sigmas and stochastic sampling disabled. Preserve it. The card's
ComfyUI recommendation (Euler Ancestral/beta) is a different sampling setup;
no equivalence or quality superiority is claimed. In particular, do not copy
Z-Image Turbo's shift-3 Mu hint or its 8-step/CFG-0 defaults. Mu remains
available as an explicit override under the existing flow-schedule rules; a
blank Mu is the fixed shift's 1.79, which the box and the sheet now show (mu is
Turbo-only). The Refine dialog opens the author's 1024x1536 at 1x and says
why: 1.25x is past the 1536x1536 measured here (The refine pass).

Sources checked 2026-09-19: [author's model card](https://huggingface.co/SeeSee21/Z-Anime),
[pinned manifest](https://huggingface.co/SeeSee21/Z-Anime/blob/0f5fb51464638a2f7328a1d74590281e63e1fde2/diffusers/model_index.json),
[pinned scheduler](https://huggingface.co/SeeSee21/Z-Anime/blob/0f5fb51464638a2f7328a1d74590281e63e1fde2/diffusers/scheduler/scheduler_config.json),
[standard encoder config](https://huggingface.co/Tongyi-MAI/Z-Image/blob/04cc4abb7c5069926f75c9bfde9ef43d49423021/text_encoder/config.json),
and the installed Diffusers pipeline loader and `ZImagePipeline` source.
`tests/test_nested_models.py` checks preflight ordering, quantization, tree
selection, the pinned encoder override and the retained public id; `smoke-z-anime.json` checks the UI on
port 7868. Card measurements belong in `verified.md`.

## Kohaku XL Zeta: ordinary SDXL with its own hybrid prompt guide

`KBlueLeaf/Kohaku-XL-Zeta` publishes a complete Diffusers component tree, so
the catalogue entry uses the normal loader. Keep its supplied epsilon-prediction
`EulerDiscreteScheduler`; its accepted custom sigmas do not make it a flow
scheduler. The profile disables Mu before load, as the live scheduler does
afterwards. The app's 28 steps and CFG 5 are starting points within the author's
12-50 and 3.5-6.5 ranges, not a separately measured optimum. No Nunchaku
repository is mapped to this entry.

`KOHAKU_GUIDE` uses Zeta's documented tag vocabulary and the Epsilon-rev2
caption order that Zeta refers to. The general-tag slot can hold prose or a
mixture of prose and tags. `hybrid_prompt` preserves spellcheck and avoids
validating a single sentence as one unknown Danbooru tag. The shared tag
service uses the model's 2024-07-10 Danbooru cutoff; no frontend checkpoint
branch is required. Quality and date choices are exclusive. Rating choices
remain individual toggles because the author's explicit-rating caption
contains both `nsfw` and `explicit`. There is no fixed suffix or recommended
negative-prompt button: the Zeta card supplies neither recipe.

The author's size recommendation is 1024x1024 or a comparable area, so the
guide publishes one recommended area in MP and lets the common tolerance
admit nearby 1 MP presets. It does not turn the training bucket's maximum
side length into a promise of large square outputs. The card also describes
300-token training context, but both published tokenizer configs declare
`model_max_length: 77`. The normal SDXL pipeline truncates to those limits;
no long-prompt embedding or chunking path has been added. The prompt hint
states that distinction.

The Examples menu takes `DEFAULT_TAGS` verbatim from the author's TIPO demo,
whose `diff.py` selects Kohaku XL Zeta by default. The twelve Zeta model-card
JPEGs contain no prompt metadata. Hugging Face's auto-generated library
snippet is not an author example. The demo's prompt is a useful author input,
not a claim that this app reproduces its separate K-diffusion sampling and
prompt-expansion setup. Hardware checks and their exact settings belong in
`docs/verified.md`.

Sources checked 2026-09-26:

- [Zeta model card](https://huggingface.co/KBlueLeaf/Kohaku-XL-Zeta/blob/c14affd9483ac7603cfff3f06b05da22a06deaa6/README.md).
- [Epsilon-rev2 caption order](https://huggingface.co/KBlueLeaf/Kohaku-XL-Epsilon-rev2#prompt-format)
  and [Delta's tag spelling](https://huggingface.co/KBlueLeaf/Kohaku-XL-Delta#tags).
- [Supplied scheduler](https://huggingface.co/KBlueLeaf/Kohaku-XL-Zeta/blob/c14affd9483ac7603cfff3f06b05da22a06deaa6/scheduler/scheduler_config.json),
  [first tokenizer](https://huggingface.co/KBlueLeaf/Kohaku-XL-Zeta/blob/c14affd9483ac7603cfff3f06b05da22a06deaa6/tokenizer/tokenizer_config.json),
  and [second tokenizer](https://huggingface.co/KBlueLeaf/Kohaku-XL-Zeta/blob/c14affd9483ac7603cfff3f06b05da22a06deaa6/tokenizer_2/tokenizer_config.json).
- Author's [TIPO demo input](https://huggingface.co/spaces/KBlueLeaf/TIPO-DEMO/blob/1ae60d3cbec4f5274ff349321b71817cfde60175/app.py)
  and [image backend](https://huggingface.co/spaces/KBlueLeaf/TIPO-DEMO/blob/1ae60d3cbec4f5274ff349321b71817cfde60175/diff.py).

## NetaYume Lumina v4: pinned denoiser and standard Lumina 2 components

`duongve/NetaYume-Lumina-Image-2.0` selects the author's full-precision
`Unet/v4/NetaYumev4_unet.safetensors`, pinned to
`69c29c441119d43405d8373c654b16827d3cf46e`. The repository also carries v2,
v2-plus, v3, v3.5-pretrained and all-in-one files; the menu explicitly names
v4 so the source recipe is unambiguous. Another release must have a distinct
recorded identity rather than changing this recipe's weights silently.

`netayume_loading.py` converts the 5.22 GB BF16 split checkpoint on CPU.
Diffusers' `convert_lumina2_to_diffusers` maps the original names and splits
QKV; it removes the original unused `norm_final.weight`. The converted state
must load strictly into the pinned parent transformer config. Header-only
validation of the published file found 400 source tensors and 459 converted
tensors, matching every expected key and shape. Missing, extra or wrongly
shaped weights fail instead of leaving random parameters.

The author identifies the encoder as pretrained Gemma-2-2b and the VAE as
pretrained FLUX.1-dev. Their descriptors, tokenizer and weights come from
the official `Alpha-VLLM/Lumina-Image-2.0` revision
`53504abd8178b30685b6c4c7a4cd181ff78b73e9`, together with the transformer
config and scheduler. The allowlist excludes all parent denoiser weights.
No community conversion or remote Python code is needed. A recipe hash,
process lock, temporary directory, final rename and the sweep of an
interrupted conversion's work directory (`single_file.sweep_work`) provide
the same atomic cache contract as Illustrious and Anima. FP8/int8
quantization applies only after conversion through the normal per-library
`quant_mapping`; full bfloat16 uses the same tree. Nunchaku is refused
before downloads.

The resulting pipeline is standard `Lumina2Pipeline`, which already takes
per-image generators and `callback_on_step_end`. The supplied scheduler is
FlowMatch Euler, static shift 6, with no terminal or beta post-processing.
Mu uses the existing whole-shift override. Negative prompting applies above
CFG 1, and saved True CFG is that effective scale. The pipeline retains
its default normalized CFG, full-step guidance and 256-token sequence.
The author's anime system instruction is passed explicitly on every call;
Lumina adds its `<Prompt Start>` separator. The user prompt remains the
stored image prompt. The fixed system instruction is part of the conversion
recipe, and is not duplicated in Examples or the editor. Negative text is
encoded as the user supplied it. These semantics and standard Euler are
not the author's ComfyUI Euler Ancestral/normal or
res_multistep/linear_quadratic workflow. The existing FLUX VAE preview map
applies to Lumina's unpacked 4-D latents. User LoRAs use the ordinary
compatibility check and hot-swap path; references are unavailable.

The exact-id profile starts at 40 steps and CFG 5. The author card recommends
40-50 and 4-7, but still describes v2, so the guide labels that family
guidance rather than inventing v4 tuning evidence. The only documented
sizes used by this guide are Neta Lumina's 1024-square example and the
author's v2 1024x1536 workflow. Their 1.05-1.57 MP span drives the advisory
highlighting; its prose explicitly says it is an example span, not a
supported maximum, and since 2026-10-05 so does the size line, which quotes it
as "the authors' example sizes" (`pixels_word` `example`, The prompting guide).
The page names the checkpoint "NetaYume Lumina v4", the catalogue's label,
where it had shown the repository's last part, the parent's name (The start-up
menu). The example prompt is the author's workflow image
prompt with only the system prefix and separator removed. No vocabulary
cutoff is published for v4, so no tag validator is guessed.

Sources:
- [Pinned author card](https://huggingface.co/duongve/NetaYume-Lumina-Image-2.0/blob/69c29c441119d43405d8373c654b16827d3cf46e/README.md).
- [Pinned v4 split file](https://huggingface.co/duongve/NetaYume-Lumina-Image-2.0/blob/69c29c441119d43405d8373c654b16827d3cf46e/Unet/v4/NetaYumev4_unet.safetensors).
- [Author v2 workflow](https://huggingface.co/duongve/NetaYume-Lumina-Image-2.0/blob/69c29c441119d43405d8373c654b16827d3cf46e/Lumina_image_v2_tensorart_workflow.json).
- [Pinned official Lumina 2 components](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0/tree/53504abd8178b30685b6c4c7a4cd181ff78b73e9).
- [Neta Lumina base guidance](https://huggingface.co/neta-art/Neta-Lumina/blob/65c078e75781946e9f3423fe41457adae3eba3cd/README.md).
- Installed Diffusers 0.40 `pipeline_lumina2.py` and `convert_lumina2_to_diffusers`.

Runtime measurements belong in `verified.md`; conversion tests and browser
smoke checks alone do not measure output quality, speed or VRAM.

## NewBie Exp0.1: pinned components and an explicit sampling recipe

`NewBie-AI/NewBie-image-Exp0.1` selects the official release at
`4f7825a1a7e1ed5c1882a52e3c51d1119a43e298`. Diffusers 0.40 has no
`NewbiePipeline`, despite the card's quickstart, and the published tensors
need explicit mappings. `newbie_loading` strictly converts them on CPU into
an atomic cache, preserving the official id in status and saved metadata.
The recipe hash includes the source revision, scheduler, prompt prefix and
CFG normalization so changes to those cannot silently reuse an older tree.

`newbie_transformer` extends the installed Lumina 2 blocks with the release's
pooled-CLIP conditioning. Gemma 3 supplies penultimate hidden states. The
separate Jina text tower uses native Transformers layers and the author's
unnormalized masked mean; its merged `jina-clip-v2.safetensors` is distinct
from the larger file containing a vision tower and adapter parametrizations.
No remote Python or unused vision weights are needed. Both encoders use
normal forward calls so model-offload hooks work. The native Jina classes
require Transformers 5.16.1 or newer. Quantization is applied after conversion
through the ordinary per-component loader.

The exact-id profile starts at **28 steps and CFG 4**: 28 is the model-card
example, while the author's sampling script uses CFG 4 with midpoint and
18 steps. That script is evidence for a starting guidance value, not proof
that this app's sampler reproduces its output. The author-proposed Diffusers
pipeline instead defaults to CFG 5. These author sources disagree; none is
presented as a measured optimum here.

This app fixes one recipe: FlowMatch Euler with static shift 6, an empty
system prefix for both encoders, CFG throughout sampling above 1, and
whole-sample norm limiting at the conditional prediction's norm. The empty
prefix follows the author's Diffusers proposal. Their training/demo code
also supports system-prefixed Gemma captions but keeps Jina captions raw;
the community proposal's generic Lumina prefix on both encoders is not used.
The norm cap follows the original model, rather than the community
proposal's unconditional rescaling along only the final spatial dimension.
Transformer time is `1 - scheduler_t / num_train_timesteps`, with model
velocity negated for the scheduler, as in the author's Comfy adapter. The
author Diffusers proposal's unscaled timestep is not copied. Blank Mu keeps
shift 6; an explicit Mu goes through the existing whole-shift override.

`NewbiePipeline` exposes one generator per image and the step callback used
for progress, preview and cancellation, with offload cleanup on exit. Its
Gemma input has a 512-token limit and Jina has 2048, following the author's
encoder path and keeping the supplied XML example intact. References
and user LoRAs are disabled; there is no Nunchaku mapping. The author XML
example remains editable text, without comma-tag chip parsing. The size
guide highlights the card's 1024-square example area and explicitly says
it is not a minimum, maximum or quality limit.

**Anonymous character labels are usable as written** (2026-10-07). The
author's XML example uses `$character_1$` and `$character_2$` literally in
the generation metadata of its [published example image](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1/resolve/4f7825a1a7e1ed5c1882a52e3c51d1119a43e298/image/XML_prompt_image.png).
The [pinned captioning recipe](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1/blob/4f7825a1a7e1ed5c1882a52e3c51d1119a43e298/XML%20Prompt.json)
specifies those labels when character names are unknown. The guide now
explains that they may stay, and neither Examples nor Check treats them as
slots that require names. The critic receives the same author excerpt and
caveat. The one-word `1girl` quickstart stub is removed from Examples; the
complete two-character XML remains verbatim.

Sources, checked 2026-09-26:

- [Pinned card and official files](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1/tree/4f7825a1a7e1ed5c1882a52e3c51d1119a43e298).
- [Author sampling settings](https://github.com/NewBieAI-Lab/NewBie-image-Exp0.1/blob/709e9a482fb2b861ad6aa50dcafb0198cad66c1d/scripts/sample.sh),
  [original CFG implementation](https://github.com/NewBieAI-Lab/NewBie-image-Exp0.1/blob/709e9a482fb2b861ad6aa50dcafb0198cad66c1d/models/model.py),
  and [training caption paths](https://github.com/NewBieAI-Lab/NewBie-image-Exp0.1/blob/709e9a482fb2b861ad6aa50dcafb0198cad66c1d/finetune.py).
- [Author Diffusers proposal, PR 12789](https://github.com/E-Anlia/diffusers/blob/694e38842aed2b53cb6a0b248e0799aceaf89101/src/diffusers/pipelines/newbie/pipeline_newbie.py).
- [Author Comfy timestep adapter](https://github.com/E-Anlia/ComfyUI-NewBie/blob/29655ed6fa86551abbf1835511764ff151dbbbc5/comfy/model_base.py)
  and [shift configuration](https://github.com/E-Anlia/ComfyUI-NewBie/blob/29655ed6fa86551abbf1835511764ff151dbbbc5/comfy/supported_models.py).
- [Community Diffusers proposal, PR 12803](https://github.com/huggingface/diffusers/pull/12803),
  implementation reference pinned to `e061f982c6c1e162309f9df919662e6fa61e60ea`.
- [Native Jina architecture reference](https://huggingface.co/jinaai/jina-embeddings-v3/blob/343dbf534c76fe845f304fa5c2d1fd87e1e78918/config.json).

`tools/smoke-newbie.json` checks the profile, XML example selection and
persistence, size warning and saved settings using the fake server on 7872,
and since 2026-10-05 what the area review found an XML prompt needs elsewhere
on the page: the token line under the prompt where no chips count (Animagine
XL 4.0), its anonymous labels accepted unchanged and the critic's XML reorders (The
prompt analyser), jobs and runs named by their words (The browser's queue), a
grid prompt listed by its variable lines and a none that closes only its hole
(Grid preview, The prompt is an axis), the prompt's lines kept beside its facts
(The results viewer) and a comparison's block moves named by tag (Prompt labels
on the sheet). The author's guide has no size range, so the size line quotes
"the author's example" (The prompting guide).
CPU and browser checks do not establish GPU performance or image quality;
hardware evidence belongs in `docs/verified.md`.

## The prompting guide

**Every profile carries a `guide`**: `prompt` and `size` prose for the ?
badges on the Prompt box and the Size presets, a `pixels` range in
megapixels with a `pixels_word`, and the authors' `examples`. The prose
rides in `/api/status`'s `hints` under `prompt` and `size`, beside the
control tooltips it is shaped like; the data is `ModelStatus.guide`
(`schemas_model.PromptGuide`), because the size line, Examples menu and optional tag
helper need numbers and lists, not a sentence. `hints_for` passes the guide's
prose through the same `.format` as the control hints, so the paragraphs
hold no braces. The checkpoints here disagree about nearly everything a
prompt can be - FLUX.1's CLIP reads 77 tokens and its T5 512 (256 on
schnell in BFL's reference, not in this app; schnell's guide says both),
Qwen-Image's encoder is prompted to *describe the image by detailing the
color, shape, size, texture, quantity, text, spatial relationships*, ERNIE
ships a rewriter this backend leaves out and so expects the expanded form
typed in - and about sizes: FLUX.1 is documented
for 0.1-2 MP, Krea 2 Turbo for 1k-2k, Krea 2 Raw *trained to 1k*, Baidu
lists 1 MP shapes only. A user who has read that "some are only validated
at 1 MP" was right to ask which.

**The word matters as much as the range.** A model card's "supports 0.1 to
2 MP", a README's "recommended 1024-1280" and a report's "trained to
1328x1328" are three different promises, so the size line quotes the
profile's word - `above the 1.64 MP this checkpoint is recommended for`,
`below the 1 MP this checkpoint is documented for` - rather than one
verb for all. The line warns a tenth outside the range: the 1 MP presets
land at 1.04-1.08 MP and every profile's range has to admit them
(`test_every_guide_admits_the_1_mp_presets`), or a fresh page would open
on a warning. The preset buttons use that same tolerance and are marked
(`.recommended`) when the full size they would choose is inside the range and,
when a guide publishes a finite `aspects` list, uses one of those framings:
aspect buttons are judged at the current megapixel tier, and tier buttons at
the current aspect. The VRAM warning stays and stacks after it.

**An example is not a range** (2026-10-05). NetaYume, NewBie and WAI publish no
size range at all; their `pixels` is the span of the sizes their authors' own
examples use - NetaYume's 1024² and 1024x1536, NewBie's 1024², WAI's 1024² floor
and 1024x1344 reference. That span still drives the presets' marks, but the
line said "above the 1.6 MP this checkpoint is documented for" under a tip
saying 1.57 MP was no supported maximum. Their `pixels_word` is `example`, and
`pixels_source` says whose sizes they are, in the size prose's words: NetaYume
"above the 1.57 MP of the authors' example sizes" (two guides' sizes, so
authors), NewBie "of the author's example" (one size, one author), WAI "of the
author's reference sizes", since its floor is the size its author says to
exceed rather than an example; the presets' titles read "within the span of
…", or "about the size of …" for one size, never what the checkpoint is "for".
The word decides this, not equal bounds: ERNIE's (1, 1) is documented and
Kohaku's 1024² recommended. A test ties the word to the size tips that say no
maximum and sets `pixels_source` exactly there
(`test_the_example_word_is_exactly_where_the_prose_says_no_bound`). `fmtMp`
quotes every bound to at most two decimals, trailing zeros dropped: one decimal
had misquoted 1024²'s 1.05 MP as 1.0, Anima's 512²-1536² (0.26-2.36) as 0.3-2.4,
Illustrious's 2.25 as 2.3, Animagine's 0.98 as 1.0, Krea's 1.64 as 1.6 and
Qwen-Image's 1.77 as 1.8.

**A typed size takes the nearest framing.** Where a guide has `aspects`, a size
typed in is judged by the preset aspect nearest its ratio within 3%
(`framingOf`), not by any guided aspect within 3%: 7:4 and 16:9 are 1.6% apart,
and 1400x800, exactly 7:4 (which ERNIE's table has not got), had read as 16:9.
A size in a guided framing that is not one of `resolutions` (1008x1008 on ERNIE,
1232x832 on Animagine) is "not one of the recommended sizes"; any other framing
is "not a recommended aspect". The test holds every listed size to the same rule
(`test_a_guides_aspects_are_presets_and_its_sizes_their_framings`): Animagine's
1216x832 is its 3:2 at 2.6%, ERNIE's 768x1376 its 9:16 at 0.8%.

**The selection reads first; the range is a mark.** A marked preset used to be
filled green, and on Turbo all 17 buttons were, on FLUX 15 - the largest
saturated block in the panel - while the chosen one differed from its
neighbours by a 1.4:1 fill (the 2026-10-04 UI review, U81). Now the chosen
preset is the accent's fill with white text and `aria-pressed="true"`, and a
marked one has brighter text, a green edge and a dot in its corner, absolutely
placed so it widens nothing. A strip where every button is marked
(`all-guided`: FLUX's aspects at 1 MP, everything on Turbo) drops the marks,
which say nothing there; the size line still quotes the range. The classes and
`presetIsInGuide` are unchanged. Each strip is a `role="group"` with one tab
stop, the chosen preset (or the first where the size is none of them), and
the arrow keys, Home and End move focus along `strip.querySelectorAll('button')`
- the pairs are nested - without choosing: a choice persists the panel and asks
for a preview, so it waits for Enter, Space or a click. The 17 size stops had
put Generate 36 Tab presses from the prompt (U105).

**A tag checkpoint's empty prompt shows tags.** The static placeholder is
prose, which failed Animagine's own style check. `applyGuide` writes the start
of the authors' first example, cut at the last comma within 60 characters and
ended with ", …", where the guide's `tag_helper` is Danbooru vocabulary and
not `hybrid_prompt` (Kohaku, Illustrious and WAI document prose as well), and
puts the markup's placeholder back elsewhere (U133).

**A pair of inverses is one box on the strip.** The thirteen aspects used to
sit in the order they were added - landscape/portrait pairs, but 16:9 ahead
of 3:2 and 9:7 behind both - and the strip, `Object.keys` of the table, showed
that. The table now lists 1:1 and then each portrait framing followed by its
landscape inverse, pairs from nearest-square to widest; `buildPresets` wraps
two adjacent inverses in a `.pair` container drawn as a segmented box, so
the pairing is the table's order and nothing else, and `syncSize` walks the
strip's buttons rather than its children. `test_the_aspects_are_in_strip_order`
holds the order, and the browser and README copy tests compare key order along
with the values. Portrait first in a pair is how the request was phrased.

**Orientation is the inverse preset, not a width/height swap.** The button
derives `b:a` from the selected `a:b`, looks it up in the same `SIZES` table and
routes the result through `choose()`, so dimensions, active/recommended states,
grid preview and panel persistence all move together at the current tier. Every
non-square entry is asserted to have an inverse with transposed dimensions.
For 1:1 there is no distinct inverse, so the button is disabled and greyed.

**FLUX.1 Krea [dev] has its own profile entry for the guide alone.**
`krea-dev` sits ahead of `flux` (its id carries that token) as a copy of
the flux profile with Krea's guide: Krea's README recommends 1024-1280 px
sides and 28-32 steps at CFG 3.5-5, which FLUX.1 [dev]'s card does not,
and the checkpoint's whole point is photographs without the oversaturated
look. `test_krea_dev_is_flux_to_the_controls_and_its_own_to_the_guide`
holds the numbers and the control hints identical so nothing about a
request changed. `test_every_catalogue_entry_has_a_prompting_guide`
refuses the fallback's "Unrecognised checkpoint" text under a checkpoint
the menu offers by name, on the official id and the lite-infer one.

**The Examples menu builds on open** from the list the last status
carried, so a status that lands minutes late behind a load is not stale by
the first click. A pick writes the box through `writeBox`, as typing would,
so Ctrl+Z brings back what it held, and tells the preview, the chips and the
remembered panel the way `choose` does; over a prompt of the user's own it
also shows "An example replaced your prompt." with an Undo, until the box
holds anything else or 15 s pass. On a two-column page it lies over the next
field's caption, out of the sticky panel's flow, so nothing moves; on one
column the chips come straight under the box, and lying over their first
row it took their clicks, so there it takes a line of its own - and so it
does while the negative box is greyed, when that caption says why and holds
the button that wakes it: lying over it, the note left "Negative prompt
(ne" of the sentence. A Reuse takes it, and it takes a Reuse's Undo: one
Undo on offer at a time ("Per-control defaults and Reuse"). Assigning
`.value` had emptied the undo history: the menu opens over
the prompt, and one stray click replaced minutes of work for good, the
panel saving it at once. The note is the prompt's one note (`#prompt-note`,
which Expand's and Reset's words share since 2026-10-05, Reset's lying in
the box it emptied rather than under it: ERNIE-Image Turbo, Panel
persistence), and hangs under the token line where that shows (Animagine
XL 4.0). It is a `role=menu`, named by its button (`aria-labelledby`), and keeps that promise: ArrowDown,
Enter or Space on the button opens it on its first item (its items are no
Tab stops, so a Tab from the button went on to Tags and closed it unread),
the arrows walk it round the ends,
Home and End jump, Escape hands focus back to the button, and focus leaving
it closes it - it stayed open over the prompt after a Tab. Where the
checkpoint takes reference images and none is attached, an example that
edits a picture "this" prompt takes to be there (`prompt_rules.EDIT_PROMPT`,
copied as `EDIT_PROMPT` in prompt-examples.js and held equal by a test) says "needs a
reference image" beside the authors' words, which stay as written. An
example that is a template - Z-Anime's portrait and landscape templates -
says beside it which slots it leaves to fill ("fill in
[character]"; two by name, more as the first and a count), and a pick selects
the first (`templateSlots`, `setSelectionRange`, `scrollToSelection`), so what
is typed next replaces it; selecting writes nothing. When the pick left the
caret at the end, the obvious first image on Z-Anime encoded "[character]" as
words, and only a manual Check said so (2026-10-05 area review). The slots are
the placeholder rule's own patterns, sent as `guide.slots` (`guide_for` passes
them on, and nothing else of `analysis`), read case-insensitively and held by
a test to syntax both engines read alike: data, like the examples, not the
rule. The
examples are verbatim from the authors - a model card's demo
prompt, a prompting guide's, a blog caption - because a checkpoint's own
examples are the shortest way to learn its voice, and because an invented
one would be a claim about the model with nothing behind it.

**Examples describe a scene** (2026-10-07). The catalogue audit found three
quality-only stubs: WAI, NoobAI and RouWei. NoobAI now uses its complete
Diffusers example; Z-Anime puts its complete "Good" scene before its two
editable templates. WAI and RouWei use explicitly labelled, contiguous
author scene excerpts: WAI's full gallery prompts exceed the 77-token CLIP
window, while RouWei separates style/quality from its scenes with `BREAK`,
which this app does not implement. The words inside each excerpt are unchanged
(including RouWei's original `kimonostand` typo). Quality actions remain in
Tags; an excerpt is not the author's full sampling recipe or a claim to
reproduce their image.

`guide.example_notes` maps the exact prompt to its editing note; `guide_for`
and `PromptGuide` carry it to `applyGuide`, which resets the map on every
status. The Examples menu displays it beside the text and never inserts the
note into the editor. Keying by text keeps a note attached when examples are
reordered. The catalogue regression rejects quality-only examples and a menu
containing only fill-in templates. Existing author-rule checks still apply.

Sources checked 2026-10-07; token counts include special tokens in **both**
of each SDXL checkpoint's actual CLIP tokenizers, loaded on CPU:

- WAI: [Miku underwater](https://tensor.art/images/990778028365743217?model_id=990778216270553015)
  and [oni in kimono](https://tensor.art/images/990778122855024231?model_id=990778216270553015),
  WAI0731's v17 gallery; prefix excerpts, 44 and 40 tokens.
- NoobAI: [pinned author card](https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0/blob/66aa55e3469c27c29a89813cd35dd95fb7485fa1/README.md),
  complete Arlecchino/graphite prompt, 76 tokens.
- RouWei: the PNG parameter metadata in [epsilon example 28](https://huggingface.co/Minthy/RouWei-0.8/blob/95b524750edf2dcbc84967ded3d1bc99deed3b53/examples_epsilon/28.png)
  and [epsilon example 33](https://huggingface.co/Minthy/RouWei-0.8/blob/95b524750edf2dcbc84967ded3d1bc99deed3b53/examples_epsilon/33.png),
  complete scene chunks, 71 and 72 tokens. Both identify epsilon hash
  `1aba15decd`; their omitted recipe includes hires/ADetailer.
- Z-Anime: [pinned card's Prompting Guide](https://huggingface.co/SeeSee21/Z-Anime/blob/0f5fb51464638a2f7328a1d74590281e63e1fde2/README.md),
  complete shrine-maiden scene; the template examples remain available.

All other catalogue menus already had substantive author examples. The
unknown-checkpoint fallback stays empty: there is no known author guide for
an arbitrary model id. These are source and text-budget checks, not new GPU
image-quality measurements.

Sources, checked 2026-09-10. FLUX.1: BFL's prompting basics
(<https://docs.bfl.ai/guides/prompting_unified_basics>), the launch post's
"0.1 and 2.0 megapixels" (<https://bfl.ai/blog/24-08-01-bfl>), the model
cards, the reference CLI's 1360x768 and its multiple-of-16 rounding
(<https://github.com/black-forest-labs/flux>), diffusers' 77/512/256
limits. Above 2 MP degrading is community evidence
(<https://huggingface.co/black-forest-labs/FLUX.1-dev/discussions/32>), and
the guide says "users report". Krea [dev]: <https://github.com/krea-ai/flux-krea>
(1024-1280, 28-32 steps, CFG 3.5-5), <https://bfl.ai/blog/flux-1-krea-dev>
("overcomes the oversaturated 'AI look'") and
<https://www.krea.ai/blog/flux-krea-open-source-release> ("opinionated",
the "AI look"). FLUX.2 [klein]: <https://docs.bfl.ai/guides/prompting_guide_flux2>
(subject-action-style-context, 30-80 words, quotes, no negative prompt,
JSON prompts, 64x64 to 4 MP with 2 MP recommended) - written for [pro]
and [max], applied to klein by family; the klein cards say only "prompt
following is heavily influenced by the prompting style", and every example
is 1024x1024. Krea 2: <https://github.com/krea-ai/krea-2> `docs/prompting.md`
("natural language", "long detailed prompts yield best results", quotes
for text, "up to 2k"), its README ("1k ~ 2k", Raw "trained to generate
upto 1k", "TRAIN on Raw and RUN on Turbo"), the technical report
(<https://www.krea.ai/blog/krea-2-technical-report>: 256/512/1024 px
stages, a text-rendering reward, an encoder chosen for multilingual
generalisation) and "rich, carefully constructed captions that describe
images with dense visual detail". The "validated at 1 MP only" claim was
**not** found for Turbo in any official source; the guide says 1k-2k
documented and 1024 px the top pretraining stage. Qwen-Image: the README's
positive magic and resolution table (<https://github.com/QwenLM/Qwen-Image>;
the HF copy's 1472x1140 is not a multiple of 16 and GitHub's 1104 is used),
the report's 1328x1328 final stage (<https://arxiv.org/abs/2508.02324>). Z-Image:
<https://huggingface.co/Tongyi-MAI/Z-Image-Turbo> ("512x512 to 2048x2048
total pixel area", bilingual text), the maintainers' "works best with long
and detailed prompts" and "does not use negative prompts at all"
(<https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/8>), the
report's "up to the 1k-1.5k range" (<https://arxiv.org/abs/2511.22699>).
The undistilled Z-Image: <https://huggingface.co/Tongyi-MAI/Z-Image>
(guidance 3.0-5.0, 28-50 steps, and a negative prompt "Optional, but would
be powerful when you want to remove some unwanted content"; checked
2026-09-28). ERNIE-Image: the card's seven 1 MP shapes and 8 steps at
guidance 1 (<https://huggingface.co/baidu/ERNIE-Image-Turbo>), the rewriter's
system prompt in `pe/chat_template.jinja` (expand a brief description into a
"content-rich, detail-sufficient visual description"), the report's
Ministral-3 encoder and 256/512/1024 stages (<https://arxiv.org/abs/2605.25347>);
its tokenizer's 2048 is the config's, not a card statement
(`model_max_length` in each repo's `tokenizer/tokenizer_config.json`, and the
report's own words are "we adopt Ministral-3(3B) ... as the text encoder";
checked 2026-09-28). The undistilled ERNIE-Image:
<https://huggingface.co/baidu/ERNIE-Image> (guidance 4.0, 50 steps; checked
2026-09-28).

**The guide is also data: `analysis`.** Each guide dict carries its
checkable statements as an `analysis` entry for the prompt analyser's rule
checks (`prompt_rules`) - a documented order's steps, a word range, "quote
the text", "describe what you want" - and every entry holds the words it
rests on as `cite`, a verbatim piece of that guide's own `prompt` prose (or,
for `author.placeholder`, of one of its examples), with a URL of its own or
the guide's. The prose stays the claim and the entry only points at it; the
next section says why that is enforced by a test. `analysis` is a key inside
the guide, not a profile key - a top-level one would reach
`ModelStatus.defaults` through `defaults_for` and fail its validation - and
is left out of `guide_for`, because the page shows findings, not the rules
behind them; the one exception is the placeholder rule's patterns, passed on as
`guide.slots` for the Examples menu (above). `FALLBACK_GUIDE`'s is empty. Sources the entries cite beyond
those above, checked 2026-09-27: BFL's unified guide covers the whole FLUX
family (FLUX.1, Kontext, FLUX.2; <https://docs.bfl.ai/guides/prompting_summary>),
and its technical page (<https://docs.bfl.ai/guides/prompting_unified_technical>)
is where "working without negative prompts" lives, with a table of
replacements ("no people" to "empty", "deserted", "solitary") and "Front-load
the positive description — word order signals priority"; its building page
(<https://docs.bfl.ai/guides/prompting_unified_building>) carries the 30-80
word table and "Start with a clear subject". Krea 2's
<https://github.com/krea-ai/krea-2/blob/main/docs/prompting.md> is the
natural-language, long-detailed and quotes guidance, and its companion
<https://github.com/krea-ai/krea-2/blob/main/docs/expansion.txt> - the system
prompt Krea publishes for an LLM of your choice to expand prompts with - is
where "one cohesive paragraph" comes from ("Write one cohesive paragraph
after the thinking block. No bullets, JSON, or markdown").

## The prompt analyser

The request was a tool that says whether a natural-language prompt is
structured right and in the right order, and how to improve it. The research
behind it (2026-09-27) found that "the right order" has an author's answer
on only some checkpoints: BFL's family guide puts the subject first for FLUX.1,
its FLUX.2 guide gives subject, action, style, context for klein, the tag
checkpoints' cards and Illustrious's report give tag orders, and NewBie's
captioning instruction gives XML field orders - while Krea 2's guide gives no
order, Qwen-Image's only in the 2512 rewriter's portrait flow, Z-Image's
rewriter a processing order rather than an output one, ERNIE's only in a
rewriter prompt its README does not link, Z-Anime's only for its 4-step
variant, NetaYume's only from its parent, and WAI's only its quality recipe
at the end (checked as a two-step order: subject and scene, then quality).
So there are three layers, each answering what the one before cannot, and
each held to the one before:

1. **Check** (`POST /api/prompt/analyze`) - what the checkpoint's own
   tokenizers do to the text (`prompt_encoding`) and what the guide says that
   a rule can check (`prompt_rules`). Deterministic, instant, on the CPU, and
   sourced by construction.
2. **The critic** (`POST /api/prompt/critique`) - what only reading can
   judge: whether the subject really comes first in a sentence, whether "shot
   on film" is a camera, what a better sentence would be. A local model held
   to numbered author sources, told the rule findings, and its rewrite put
   back through layer 1.
3. **Compare** (`GridRequest.prompt_variants`) - where the authors wrote no
   order, or an order is only a claim, the user's own seeds on this checkpoint
   are the evidence: original, reordered, rewrite and the user's own version
   as one grid, one sheet row each. A "reordered" row is a pure permutation
   of the original, so a difference between the two rows is the order and
   nothing else - which is why the user's own version exists: on a sentence
   prompt Check has no reordering and the critic's rarely survives
   `is_reorder` (below), so the reorder the user writes is the only
   one there is.

**Check works cold.** It takes the body Generate would send
(`PromptAnalysisRequest`), because half of what a checker says depends on
the rest of it: a negative prompt at CFG 0 on Krea 2 Turbo, or without True
CFG on FLUX.1, never runs, and that is `applied()`'s answer, which
`backend.applied()` gives from the profile before a load. It runs off the
loop, on a daemon thread of its own as the model load does (`_detached`),
never calls `ensure_loaded` and never takes the gate, so it answers in
a moment on a cold server and beside a running night - a first use's tokenizer
download holds it `LOAD_WAIT` at most (below). The negative prompt is
measured and checked only as `applied()` runs it: a budget for a negative the
pipeline drops is a number about nothing, and a negation in the prompt is a
warning only where no negative prompt runs. `POST /api/prompt/measure` is its
encoder layer alone, for the tag chips' count and cut and the token line under
the prompt where no chips show (both under "Animagine XL 4.0"): the same
`_detached` thread, no rules and no tag lookup.

**The encoder layer measures the string the pipeline builds.** No pipeline
here parses A1111 or ComfyUI weighting - diffusers has no prompt weighting,
and compel and sd_embed are not installed - and every overflow is cut silently
from the right: the SDXL negative prompt, every Qwen-encoded pipeline, ERNIE
and Anima never say so, and the others' warnings go to the server console.
`prompt_encoding` loads each checkpoint's own tokenizers on the CPU and counts
over the exact string: Krea 2 and Qwen-Image's 34-token system turn, Z-Image's
and klein's chat template with the pipeline's `enable_thinking` (on and off
respectively), NetaYume's system instruction and `<Prompt Start>` (the
negative gets none), and a boundary merge counted the way the tokenizer counts
it. The budgets that come out, which builder A matched against each pipeline's
own truncating tokenizer call: FLUX.1 CLIP 75 (one pooled vector - T5 reads
on, so its cut is info) and T5 511; Krea 2 507, its suffix appended after
truncation and never cut; Qwen-Image 512 after its 34 system tokens, the
template tail lost past 507; Z-Image and Z-Anime 509 (504 before the assistant
turn goes); klein 509 (500); NetaYume about 237 (255 for the negative, which
is sent bare, so its row carries no `role`; until the 2026-09-28 review it
said the negative shared the instruction's budget); NewBie Gemma 3 511 and
Jina 2046; Anima Qwen 512 and T5 ids 511; ERNIE 2047; SDXL 75 per CLIP. Past
the clean budget Z-Image, klein and Qwen-Image encode an unterminated user
turn (`encoder.template_tail`). The cut finding quotes the words from the cut
on, starting at the beginning of the word the cut splits. Chinese and Japanese
put no spaces between words, and until the 2026-09-28 review that walk back
ran to the prompt's first character, so the message named the opening as where
reading stops. Now Han and kana, and their 、，。, stand alone; Hangul does
not, since Korean spaces its words. A walk back longer than half the
40-character snippet (a URL, Thai, an unspaced Hangul run) quotes from the cut
instead. Every template string, length and thinking switch copied into
`encoder_table` is held equal to the installed diffusers source - and to this
repo's Anima and NewBie pipelines - by reading those files in
`tests/test_prompt_encoding.py`, so a diffusers upgrade that moves one fails a
test rather than the counts. The table is first-match on the id like
`_PROFILES`, exact ids first, reusing the loaders' pinned constants; its
tokens are narrower than the profiles' (`flux.1`, so FLUX.2 klein never
reaches FLUX.1's CLIP and T5). Control tokens are found by tokenizing, from
each tokenizer's own special tokens and the markers its chat template uses,
not from a list: Gemma 2's and 3's `<s>` and `</s>` turned out to be ordinary
HTML-strikethrough tokens beside `<b>` and `<table>`, which the research's
table had wrong, and `<think>` counts only where the template uses it (klein,
Z-Image, Z-Anime). A typed control token is compared with the stored one
through the tokenizer's own normalizer. CLIP lowercases before it matches its
added tokens, so `<|EndOfText|>` is its EOS; until the 2026-09-28 review that
typing read as an unknown character, because EOS is also CLIP's unk id, and
`<|STARTOFTEXT|>` went unflagged. FLUX.1-Krea-dev's CLIP-L stores its EOS with
`normalized=False`, so there the same typing is text and is not flagged.

**A first load is a download, and nothing waits on it for long.** Tokenizers
load from a directory - converted tree, then the Hub cache, then the Hub - not
a Hub id, because transformers 5.16 loading a fully cached pinned tokenizer by
id with `local_files_only` failed; the download patterns are tokenizer files
only (`*.json`, `*.txt`, `*.model`, `*.jinja`), since NewBie's tokenizer
folder sits beside 8 GB of weights. A load runs on a daemon thread of its own
and is kept only after it succeeds, so a gated repo or a first use offline is
retried next time. Until the 2026-09-28 review a load was cached under the
lock `unload()` takes (`translation.py`'s pattern), and the Hub fetch inside
it held that lock: `snapshot_download`'s `repo_info` and `list_repo_tree`, on
huggingface_hub's shared httpx client with `timeout=None`. So Unload and
shutdown waited out a first Check's download, forever on a stalled connection,
while `/api/status` said ready over no weights. Now the lock guards only the
cache and the loads in flight (`_loading`), and concurrent checks share one
load. `measure` starts every encoder's load before waiting on any, and waits
`LOAD_WAIT` (30 s) for all of them at most; past that the encoder is reported
as still loading while the load goes on, and the next check finds it cached.
`unload()` forgets the loads in flight: one that lands afterwards is handed to
the check waiting on it but not cached, and the next check starts a fresh
load, so Unload is also how a stalled fetch gets retried. The bound is a wait,
not an HTTP timeout: the only way to give that client a timeout is
`set_client_factory`, which is process-wide and would bound the weight
downloads too, so a fetch stalled for good keeps its thread until the process
exits. The Unload route says "unloaded" as soon as `try_unload` succeeds,
before the translator, critic and tokenizers let go. `measure` never raises
and says what it could not check in `encoders_detail`, each unchecked encoder
under its own reason, so "check again in a moment" never speaks for a gated or
broken tokenizer beside it. It did raise once: `json.loads` on 3,000 nested
`[` throws `RecursionError`, well inside the 4,000-character limit -
`_is_json`, `detect_style` and `is_reorder` now catch it. No status route
loads a tokenizer, because a first load can be a download; Unload and shutdown
drop them with the pipeline.

**The rule layer makes "never invented" mechanical.** Each guide's
`analysis` entry (previous section) holds the words a rule rests on as a
verbatim piece of the prose the ? badge shows, and
`test_every_cite_is_the_guides_own_words` fails an entry the prose does not
contain - so a rule can only be a statement the page already makes, and a
`guide` finding carries that cite and its URL to the page. What rests on an
author's example rather than the prose is `author` (Z-Anime's `[character]`
and `[location]` slots). Pattern matching with no author sentence
behind it - quality boilerplate on a prose checkpoint, a repeated tag - is
`heuristic`, carries no cite, and the page labels it Heuristic. The authors'
own example prompts raise no warning against their own guide (a test), apart
from those template slots; some raise info - Kohaku's own example departs
from its documented Epsilon order, two of Illustrious's put `masterpiece`
mid-prompt, FLUX.1's and klein's are under 30 words - which is why order and
length are advice, not warnings. A tag's Danbooru category (character,
series, artist) comes from the chips' own lookup (`_lookup_tags`, so the two
cannot disagree), only on a Danbooru-vocabulary guide - or one whose rules
read artists, Anima's - and a tag or hybrid prompt, or, for the artist rule
alone, a prose prompt's tag-shaped items (`prompt_rules.lookup_items`, below);
without it the order check places only what the guide's own groups
know and never guesses a slot, and any failure there - a truncated reply
raised `IncompleteRead`, which `tags._request_json` now also treats as
unavailable - costs the order check, never the answer, and says so
(`categories_detail` and `order_detail`, below). `reorder_tags` moves
the items and keeps their spelling and the text between them, a pure
permutation the tests assert, which is what the comparison's "reordered" row
needs. A rule's `suggestion` is always the whole field rewritten, so the page
can apply it in one click; advice that cannot be applied that way ("no
people" - BFL's table suggests "empty", "deserted", "solitary") stays in the
message. The review of the first build found the warnings that would have
cost trust: unquoted sign text ("reading NO PARKING") and romanised titles
("Sousou no Frieren") read as negations, and "A mother says goodbye" read as
text to render; negations now skip text-to-render spans and a lowercase
"no" between capitalised words, and "says", "labeled" and "spells" need
something that carries text earlier in the sentence. The final review found
the quote rule's one-click fix quoting half a sign: the text to render ended
at the first "and", "to", "with" or "for", so "says Fish and Chips" became
`"Fish" and Chips` and "reads Welcome to Paradise" `"Welcome" to Paradise`,
and the re-check then found nothing, so the user believed it fixed. A joining
word now carries the lettering on when a capital follows it (a name or a
title), a run of capitals before a lowercase one is the whole of it ("says
OPEN and a cat"), and where the end is a guess - "says fish and chips for
sale", or more than six words - the finding asks for the exact words in
quotation marks and offers no Apply. It also found the reported style
worked out without the tag categories while the rules used them, so the
summary read "tags and prose" beside an order finding only a tag list gets;
`detect_style` now takes the categories and `PromptAnalysis.style` is the
rules' own. "images 1 and 2", "images 1-3" and "images 1, 2 & 3" name every
image in them for the references rule. And `guide.artist_prefix` (Anima's
"put @ before an artist name") never fired: it needs an artist's category,
and Anima's guide has no tag helper, so the route never asked. The route now
asks the tag index for any guide whose rules read artists
(`wants_artist_categories`), with no cutoff and no special tags of its own;
Anima's card says it was trained on Danbooru-style tags.

**A one-click quote only where the lettering's end is sure.** The 2026-09-28
review found the quote rule's one-click fix quoting description as lettering,
after which the re-check came back clean: "with the words printed across the
chest" quoted "printed", "a book of spells glowing" quoted "glowing", and "a
newspaper reading without glasses by the fire" hid its negation as lettering.
The first word after a trigger now cannot be one that describes the text
(`_DESCRIBES`): a participle such as printed or carved, a preposition such as
into, about, by or without, an article or a possessive, or an -ly adverb
leading into one - in lowercase only, so "reads Into the Woods" is still a
title. A trigger that can be a noun (an -ing form, or "spells") is one after
"of", "with", an article or a possessive, unless a capital follows
(`_NOUN_USE`: "a book of spells" and "with reading glasses" are nouns, "a sign
above her reading OPEN" is lettering); a finite reads, says or said never is a
noun, because "her" is also an object: "a sign above her reads NO ENTRY" is
lettering, not an absence. Lowercase words after a participle trigger (reading,
saying, spelling) are not exact, because the participle may belong to any
earlier noun ("a boy on a box reading comics"): the finding stays, the Apply
goes. Lowercase lettering after a finite verb keeps its one-click quote where a
stop word ends it ("a wooden sign that says open above the bakery door", which
the analysis smoke applies). Making every lowercase target inexact, as the
review first suggested, would have taken that away, and requiring the bearer
directly before the verb would have lost "a shop sign in the window says
CLOSED". The price is accepted: lowercase lettering that opens with a
describing word - "a sign that says by appointment only", "says the end", a
clause-final "says daily." - now gets no finding at all, where it was quoted
rightly before, and "says without limits" reads as an absence instead.
Capitalised ("says By Appointment Only") it is
still lettering. Phase-4 re-verification found the noun triggers still
quoting description in one click, since the describing words are a closed
list: "with the words shimmering across the chest" quoted "shimmering", and
blinking, hanging, dripping, scratched, piped and "set in bold type" went the
same way. The reason for sparing lowercase targets was "says open", a finite
verb, and it does not reach "with the words". After a noun trigger ("with the
words", "the text", "the slogan", "the headline" and the rest) only words in
capitals now get the Apply ("with the words HAPPY BIRTHDAY", "2024", "FISH and
CHIPS"). Anything else keeps the finding and loses the one click, as the
review asked: "Happy Birthday" and "coming soon" there are likely lettering,
but "set in bold type" has the same shape. A lowercase -ing or -ed word with a
preposition of place after it ("shimmering across", "piped in", "dripping
down", "spelled out") places the words rather than being them, so it is no
lettering at all, and the critic flags a rewrite that quotes it. The price,
like "says by appointment only", is that lowercase lettering of that shape
after a noun trigger ("with the words dancing in the rain") gets no finding
unless it is quoted or in capitals. The same re-verification found a negation
hidden further into a participle's lowercase words: "a boy on a box reading
comics without pictures" read "comics without pictures" as lettering. Those
words now stop at without, never, no or not, and the negation rule has the
rest. A review of that stop found it cutting lettering that opens with a
negation or holds "do not": "a red sign reading do not enter" became "do",
"please do not touch" "please do" and "no pain no gain" "no pain", and nothing
warned, since the negation began inside the words it had stopped. So the words
stop only where the negation rule's own reading finds an absence starting at
that word (`_absence_at`) - not inside a negation that starts earlier, nor at
one with nothing after it ("now or never") - and those are lettering whole
again. The cost is lettering that holds a negation further in ("a banner
reading love without limits", "a sign reading dogs not allowed"), which warns
as an absence until it is quoted or in capitals. That stop is the rules' own:
the lettering may hold the negation, so the critic should still take "love
without limits" whole as text to render and not flag a rewrite quoting it,
which is the fix the quote rule's finding asks for. A first fix left the stop
in `rendered_targets` with `ends` False, a guess the lettering may run on
past, and the review's final completeness check found the critic then
reading on to the clause's end: "a banner reading love without limits above
the stage" kept "love without limits above the stage" as text to render and
let a rewrite quote it, description passing as lettering. `rendered_targets`
now reads the words on through the negation to where they would stop
without it - "love without limits", as "love forever" stops at "above" - with
`ends` True, so what follows is description, as after any participle's
words. The quote and negation rules keep the stop at the negation. The
review of that fix found the critic then taking the negation's own words for
lettering: a rewrite quoting only "without pictures" in "a boy on a box
reading comics without pictures" - the absence the negation rule warns on -
passed as a piece of "comics without pictures", where it had been flagged
while the lettering stopped at the negation. Each target now carries
`head`, the rules' stop (its `end` wherever there is none), and the critic
takes as lettering only words that start before it, as its run-on past a
guess always did: "comics", "comics without" and "comics without pictures"
pass, and "without pictures" alone is flagged. That review also found the
stop still cutting "now or never" wherever description followed it: "a
poster reading now or never in bold type" quoted "now or" and warned on
"never in bold type" as an absence, and "better late than never on the
door" went the same way. A negation right after "or" or "than" closes the
phrase ("now or never", "better late than never", "ready or not", "yes or
no"), so it stops nothing either; a place word after the negation would not
have told them apart, since "comics not in color" are black and white. The
2026-09-28 review also found the negation rule warning on "a cat trying to
avoid the rain" and "the film Dr. No". "avoiding" is no longer a head, and
"avoid" or a capital "No" asks for nothing mid-clause: right after a word
("trying to avoid", "the film No") or an abbreviation that goes before a
name (Dr., Mr., Mrs., Ms., Mt., Prof.), spaced or not ("Dr. No", "Dr.No").
One that comes after a name ends a sentence as often: with St., Jr. and Sr.
on the list too, the final completeness check found "a photo taken on Main
St. No cars." and "Martin Luther King Jr. No crowds." silent, so those now
open a clause, like any other full stop.
Anything else opens a clause - the text's start, a line break, a full stop
(after a year, a short word or "a.m." too), a spaced dash, a bullet, a bar or
a slash - and so do "please" and "also". A first fix also read a capital "No"
before any title-case word as a name. Phase-4 re-verification found that
silencing real absences: "Kyoto. No English text.", "No Chinese characters",
"No Apple logo", "No People on the beach". A clause's own "No" is capitalised
by grammar whatever follows it, so that reading is gone. A capital "No" is a
name only mid-clause ("a poster for No Country for Old Men", "a No Smoking
sign"). The accepted cost is a title that opens a clause: "No Country for Old
Men poster on a wall" warns as an absence, its span taking in the whole name.
And a negated phrase is no longer cut inside a run of capitalised words
("without ropes on El Capitan", not "... on El").

English lettering can also continue through a word normally read as a
location or through punctuation. A capitalized stop word in `MADE IN JAPAN`
or `OPEN AT NOON`, marks within `9:00`, `$9.99` or `www.example.com`, and a
comma before another capital in `HELLO, WORLD` now make the end uncertain.
Check asks for explicit quotes and offers no Apply, instead of quoting only
`MADE`, `9` or `$9`. `rendered_targets` carries that uncertainty to the critic.
Ordinary lowercase location descriptions and terminal sentence punctuation
still give the existing one-click quote.

**Chinese lettering has no spaces to end it.** The same review found the
Chinese triggers (写着, 印有 and the rest) taking up to twelve characters of
whatever followed as exact: 一块写着欢迎光临的木质招牌挂在门口 quoted the sign
and the door along with the words, and split 门口. A Chinese target is now
exact only where whitespace, punctuation, a quotation mark, a bracket or the
end of the text ends it - not an ASCII mark with a letter or digit on both
sides (营业时间9:00-21:00, 3.14, www.shop.com), nor a space inside Latin
lettering (写着OPEN NOW的). A target stopped at 的 or 之 is a guess and gets
no Apply, unless it is all Latin letters and digits, where the change of
script ends it (写着OPEN的招牌); so is one stopped by the twelve-character cap
with more text after. A title in 《》 is the lettering, marks and all. With no
word segmentation, a run that punctuation ends could still hold description:
写着营业中挂在门口， quoted 营业中挂在门口 in one click, and 写有欢迎字样，
quoted 字样 along with 欢迎. 字样, or a count of the characters (两个大字),
now ends the lettering for certain, so 写有欢迎字样 becomes `写有"欢迎"字样`
in one click; a 等 before either is a guess, "and so on" in 欢迎光临等字样 but
the end of the word in 平等字样, where a first draft quoted "平". A verb that
puts the sign somewhere, followed by 在 or 于 (挂在, 贴在, 悬挂在, 站在, 位于
and the like), ends it as a guess, the way 的 does: no Apply, and nothing past
it is read as lettering. The exception is a verb the sign itself asks for,
with a request or a prohibition right before it - an adverb such as 随意 may
stand between them, or 将 or 把 with one or two characters (请站在黄线外,
禁止停放在此, 请勿随意停放在此, 请将车辆停放在指定区域). A first draft
exempted only a prohibition up to two characters back, and cut 请站在黄线外 to
请; a request anywhere earlier in the run would have quoted 请勿吸烟挂在墙上
and 请把门关好挂在门上 whole. 在 alone never ends it, so 正在营业, 现在 and
在线 keep their Apply. Two gaps are left: a description that starts with a
bare 在 (写着营业中在门口) keeps a wrong Apply, and a slogan with a placing
verb of its own and no request before it (我们站在一起) loses its Apply. Every
target, English or Chinese, carries two flags: `exact` says whether the Apply
is offered, and `ends` whether what follows is description (False where the
lettering may run on past it), which `rendered_targets` hands the critic. Past
的 or a placing verb, or after a participle, the Apply is withheld but what
follows is still description, so the critic's run-on fallback does not take
it. Where a negation stopped a participle's words, `rendered_targets` reads
them on through it to their ordinary stop first, and what follows that is
description too; its `head` keeps the stop, before which lettering starts.
Words not in capitals after a noun trigger lose the Apply and keep the
`ends` their stop gave them.

**A stretch of lettering is one item, and no rule rewrites it.** The
2026-09-28 review found three rules rewriting lettering in one click -
repeated items, Anima's lowercase and the underscore rule - since each rewrote
a comma item whole, lettering included, and a comma inside quoted lettering
splits it into items: a quoted `"Ho, ho, ho, Merry Christmas"` lost a "ho",
`"HELLO"` became `"hello"` and `"user_name"` became `"user name"`. Its
follow-up found four more failures of the chips' split, which cuts
`"hello, world"` in two at its comma. The order
rule placed the halves apart (`1girl, world", holding sign, "hello, smile`); a
rating or artist rule rewrote a piece of a quotation that happened to be a tag
(`"hi, safe, bye"` became `"hi, general, bye"`, or `"@nnn yryr"`); a tag
inside lettering counted as present (a quality tier, the recommended finish,
NewBie's `2girls`); and an inserted or dropped item shifted `, ` into a
quotation's `"hi,there"`. A stretch of lettering - quoted, or unquoted text to
render such as a 《title》 - is now one item with its commas (`_units`), to
every rule that reads the prompt's items and to `reorder_tags`. It is matched
as no tag (`_Context.norms` is None for it), the tag index is never asked
about it, and it is placed by its shape alone: prose-shaped, it goes in the
general slot whether or not the index was asked about anything else. The
lowercase and underscore rules leave the lettering inside an item as written
(`_Context.segments`), and its words do not make a tag beside it prose for the
underscore rule. A "，" inside the lettering splits nothing for the order
check; one between it and a tag (`"hi"，1girl`) still stands the order down,
since that tag would sit inside the lettering's item where nothing places it
(the follow-up's first draft blanked the lettering to spaces and read such a
prompt as placed). Standing the order down wherever a comma fell inside
lettering was the alternative, and would have left a tag prompt with a quoted
sign unchecked; as one item the prompt is placed, and `order_checked` stays
true to what the rule did. A move that would still change what `QUOTED` reads
is not offered - `reorder_tags` is None and the order finding has no Apply: a
stray inch mark (`12" figure`) moved ahead of a quotation would pair with its
opening mark. Two such marks that `QUOTED` reads as one quotation
(`5" doll, 1girl, 7" doll`) are lettering to every rule, 1girl included -
`QUOTED`'s reading, shared with the critic and the encoder layer. And the
prose guides' style rule, which wants three pieces before it calls a prompt a
tag list, counts `1girl, "hi, there"` as two. The review also found NetaYume's
system-prefix rule locating the instruction in `text.lower()`, whose offsets
are not the text's: a leading "İ" lowers to two characters, and the Apply cut
into the tags. The rule now searches the same-length `ctx.bare`, with
lettering blanked, case-insensitively. A sign that explicitly says
`"<Prompt Start>"`, or quotes the system instruction itself, therefore keeps
its inscription; a real duplicate after it is still found. Apply tidies only
the cut's seam - one space, and one comma where there was one on either side -
instead of collapsing every run of whitespace in the prompt.

**The order is shown as checked only where it was.** Only a tag
checkpoint's order has slots a comma item can take (`_OrderSpec`); FLUX.1's
"subject first", klein's four steps and NewBie's blocks are documented and
shown, but nothing reads a sentence's order. The final review found the
panel reading as a pass on the default checkpoint - a prose prompt with its
subject last and one with it first got identical panels, "nothing to flag"
beside the chain. `PromptAnalysis.order_checked` says whether the rules held
the text to the order at all, and the page says "Not checked" under the
chain and, where nothing is flagged, "order not checked" in the summary. A
pattern check for a prose
prompt whose first clause is only light, camera or style was considered and
left out: BFL's own steps start with the image type, Krea's own example
opens "Moody night photography", and a rule that flags the authors' own
examples fails `test_the_authors_examples_raise_no_warning` for good
reason. Where a guide states no order, the page no longer says "the authors
document no word order": Anima's card, Neta Lumina's prompt book and the
ERNIE and Qwen rewriter instructions do give one, and the critic cites them.
It says the app checks none, and shows those excerpts (`Excerpt.order`,
`PromptAnalysis.author_orders`) as the authors', with their links, as
information.

**Where the rules placed nothing, or not every tag, the page says so.** The
2026-09-28 review found a list written one tag per line, or with fullwidth
commas, reading as tags while nothing in it was placed: `detect_style` splits
at commas, line breaks and "，", the chips and the order rule (`comma_items`)
at commas alone. The order still showed as checked, and the quality rules
called a present `masterpiece` or `best quality` missing (Animagine's suffix
rule appended it again). `_placeable` is now the one answer `order_checked`,
`reorder_tags` and the order rule share: a guide with slots, not XML or JSON,
at most 256 comma items, and no comma item that a line break or a fullwidth
comma splits (a stretch of lettering is one item with its commas, and one
piece of it). Prose written over several lines on a tag checkpoint is
therefore "Not checked" too, and `test_an_order_not_checked_is_never_flagged`
holds, on every catalogue checkpoint, that an order not checked is never
flagged. `_suffix` and `_quality_tier` count a tag on its own line, or after
"，", as present. The review also found the panel reading as a pass offline.
`_tag_categories` returned None for a failed lookup just as for nothing to
ask, and never asked about items past the chips' 64, so an out-of-order tag
prompt and an in-order one got identical answers with `order_checked` true.
The rules still place only what the guide's groups know, and a category is
never guessed; what changed is that `PromptAnalysis.categories_detail` says
why categories are missing or partial: the index unreachable (a connection
error, the cause under `TagLookupUnavailable`), no usable reply from it (an
HTTP error status, a cut or unreadable body), or the cap. The page then said
"Partly checked:" and that reason under the chain, and "order partly checked"
in the summary. Nor is a partly placed prompt's "nothing to move" a verdict,
so the server keeps a critic's reordering there that the order rule, placing
with the same categories, finds nothing wrong with (the critic's paragraph
below). The server reads "partly placed" off the placement itself
(`places_every_item`), so it also covers a tag the index answered for and did
not know, and a clause of a prose prompt a name could still be, which is never
looked up. Phase-4 re-verification of the review's fixes found the page
still keyed on `categories_detail`: for an OC's name the index did not know
(`zzkx oc, green hair` on Animagine) or a prose prompt's clauses, the Order
section drew the order as fully checked while the server, and the critic
block's "though it could not place every tag", treated it as partly placed,
and an out-of-order prompt and an in-order one again got the same "checked"
answer - F12's class. There is now one rule. `_analyse` computes the
placement once, `prompt_rules.unplaced_items` with the categories the rules
placed with (`places_every_item` is its emptiness, the critic's
`placed_partly`), and `PromptAnalysis.order_detail` is set exactly where
`order_checked` and some item went unplaced, naming the items (three, then
a count) under the reason each has no category: the lookup's own reason
(`categories_detail`, kept word for word), a tag the index does not know, a
prose prompt's clauses that are not whole sentences (prose is never looked
up, and a whole sentence sits in the general slot), and an item never asked
about (lettering, an @ name). The page shows "Partly checked:
<order_detail>" and the summary's "order partly checked" from that field
alone, wording it as a combination with `order_checked` rather than a second
copy of its conditions; `categories_detail` alone - a cap past which every
item was placed anyway, as prose - no longer makes an order "partly
checked". The note names no slot, because which slots an unplaced item keeps
is each guide's own. Phase-4 review of that fix found it said "Partly
checked" too widely, and one reason too narrowly. Every unplaced item
counted, so a one-tag or one-clause prompt, which no order can be wrong
about (the order rule and `reorder_tags` read fewer than two items not at
all), and an unknown tag every category would leave in order
(`1girl, zzkx oc, masterpiece`) or out of it (`zzkx oc, 1girl`, which is
flagged) read as partly checked, where before they had rightly shown as
checked - and the note told the user such a tag "may sit out of order
unflagged". And a tag the index knew but withheld a category for,
deprecated or newer than the checkpoint's cutoff, was said to be one the
index "does not know", beside chips giving the real reason. `unplaced_items`
now names an unplaced item only where its slot decides the order: some slot
it could take puts it out of order beside another item and another slot
does not (`_undecided`), a pair `_violation`, flagging only what is out of
order beyond doubt, passes. Where no item is, the rule's answer and
`reorder_tags`' order are the same whatever category each item has, and
`test_an_item_is_left_unplaced_exactly_where_its_category_decides` holds
that by enumerating the categories. So on Animagine `zzkx oc, green hair` is
checked (a character, a series or a general tag, it is in order ahead of
`green hair`), while `green hair, zzkx oc` is partly checked; on Kohaku,
whose meta tags follow quality, `1girl, zzkx oc, masterpiece` is partly
checked too. The review's final completeness check found a tag written
twice still partly checked: `1girl, zzkx oc, zzkx oc, masterpiece` named
both copies, since `_undecided` read each against the other as though their
slots were independent. Two items written the same, and both lettering or
neither, are placed alike under any categories (`_place` reads nothing
else), so they take one slot whatever it is and
are never out of order beside each other; that pair no longer counts, and
the prompt is checked. Beside another item a copy still counts (`green hair,
zzkx oc, zzkx oc` is partly checked, the tag named once). The critic's
`placed_partly` reads the same predicate, so a
reorder of the critic's where no missing category decides anything is held
to Check's own order, which is then the documented one (the critic's
paragraph below). The unknown tag's reason now reads "the tag index has no
usable Danbooru category for", which is true of all of them; the chips say
which. Where the guide states no order, the one rule that
reads a category is Anima's artist prefix (`lookup_items`). It reads none when
the lookup fails, and the page says no tag was checked for an artist name
(`categories_unread`); past the cap it reads not all, and the page says not
every tag was.

**Anima's artists are read item by item** (2026-10-05 area review). The review
found Check calling Anima's own recommended shape clean over an artist written
without its @. The card says "Use tags, natural language, or both", and its
positive prefix and tags followed by two sentences read as prose, so the tag
index was never asked, the @ rule (gated on a tag or hybrid prompt) never ran,
and nothing said it had not. On a guide whose rules read artists, a prose
prompt's tag-shaped items are now looked up too (`prompt_rules.lookup_items`),
never its clauses: the rule reads a whole comma item, a clause is no artist's
name, and asking about every clause would cost the index a request each Check.
The rule reads each item that is no sentence and no lettering, whatever the
whole prompt reads as, and names several artists with one verb, each once
("wlop and kantoku are artists"). The order rules keep their gate, since their
"partly checked" names a prose prompt's clauses as never asked; no guide that
states an order reads artists, which a test holds. An outage - no category at
all, `PromptAnalysis.categories_unread` - now says no tag was checked ("artists
not checked"), where "not every tag" understated it, "artists not all checked"
being kept for the cap, and the empty line above the note no longer says its
reason twice. The critic's rewrite is checked for artists with the analysis's
own answer, with no lookup of its own, and the critique says whether that answer
reads every artist the rewrite may name (`rewrite_artists_checked`,
`prompt_rules.artists_read`): every item a Danbooru name could be, a prose
rewrite's clauses included, must have been asked (`categories_asked`, kept off
the response), and no artist the answer named may stand in the rewrite other
than as a whole item the rule reads or after its @. A rewrite that adds the @ is
then fixed, not "may still apply" (the page, below). One naming an artist the
analysis never asked about, or keeping a named artist inside a clause or a
sentence ("in the style of wlop", which the rule never reads), keeps the hedge;
a first version asked only about the rewrite's tag-shaped items and called the
prompt's finding fixed over such a clause. The order and style rules' lint is
unchanged: the rewrite's new tags have no category.

**A slot is one finding, quoting its own line.** The same review found a
template slot standing as a whole comma item (`[character], soft rim lighting`)
reported twice on one span, as the author's slot and as A1111 de-emphasis. The
assembled check (`prompt_check.checked`) now drops the encoder layer's weighting
finding on exactly an author slot's span, that finding's own message having
already allowed it might be a slot. The card's "Backgrounds & Landscapes"
template is a second Z-Anime Example, and its four slots are the author's.
NewBie's second label was then treated as a slot and quoted the first one's
line (`<n>$character_1$</n>`); the labels are accepted unchanged since the
2026-10-07 source review above. An
`author.placeholder` finding now quotes the entry's cite where it shows that
slot, else the shortest line of the examples that does (`_slot_cite`,
`test_each_slot_cites_the_authors_own_words`), and says what to put there (the
entry's `fill`). Z-Anime's guide says to use a negative prompt, so its analysis
notes a missing one, without a suggestion, since the card names no text.

**A list is a tag list however it is parted, and a Chinese count is
characters.** The style note counted the chips' comma items, while the style
was read from pieces split at commas, fullwidth commas and line breaks, so on
ERNIE a Chinese list read "a tag list · nothing to flag" with the note silent. It
now counts those pieces, lettering one piece, three or more, and the note says
"a tag list", not "comma-separated". `word_count` counts a Han or kana
character as a word, and the rules keep that; the analysis sends how many of its
words are such characters (`PromptAnalysis.characters`), and the summary says
"12 characters", or both counts for mixed text (`wordsLine`).

**NetaYume's instruction as it is retyped.** The system-prefix rule matched the
loader's string exactly: without its full stop, or with diffusers' own Lumina 2
instruction ahead of the marker, Apply cut the marker alone and the check of the
result read clean. The loader's instruction is now matched in any case and
spacing, its stop optional. Where no such instruction precedes the first
`<Prompt Start>` and the text before the marker opens as an instruction, it and
the marker are cut as one span, worded as another workflow's instruction - but
only where that text has no lettering and is one sentence (`_MORE_SENTENCES`).
An instruction opens "You are a/an", up to three words, then assistant, model,
AI, generator or bot, the noun ending the opener: a stop or the end after it,
or a word an instruction goes on with (designed, that, which, who, built,
trained, made, specialised, capable, for, to; `_INSTRUCTION`). Each limit came
from a scene Apply deleted. A first pass took any "You are", and cut "You are
looking at a girl in the rain" whole. A second took any of those nouns, and cut
"You are a model student walking in the rain", "You are an AI artist painting a
girl" and, past a sentence break, "You are an AI. A girl stands in the rain."
Past a sentence nothing says how much is instruction, so the marker alone goes.
An ambiguous follower ("a model for a fashion shoot") can still read as one; the
span is quoted, and Ctrl+Z undoes the Apply.

**The critic's XML reorders.** Check reads no XML order, but NewBie's guide
states one and the critic is told to move pieces into it. A reorder with
`<general_tags>` ahead of a character block is now set aside against the
documented order (`prompt_rules.scene_ahead`), saying the critic "puts" it
there, or "keeps" it where the prompt already had it so; swapping the
characters stays an unchecked reorder. An XML candidate with the prompt's own
words that is not a pure reorder of whole pieces is still refused, since blocks
move whole so no attribute changes owner, but the set-aside sentence now says
what it did, where it used to say it changed words: it moved words inside an
XML block or between blocks, it changed only punctuation or spacing ("happy,
smile" to "happy smile", `<n> miku </n>`), or it changed the text outside the
blocks (`critic_reorder._not_a_reorder`). The area review's optional `_xml` info
finding for a scene block ahead of a character (newbie-5) was not built:
Check's "order not checked" for XML is deliberate, and the note under the chain
would contradict it.

**Deliberately not checked**, each for a reason: schnell's "about 180 words",
BFL's reference 256-token limit, which this app does not apply (below; the
prose says T5 reads 512 here and gives the 256 as BFL's, in a sentence the
critic is not given); a negative prompt on the undistilled Z-Image, which its
card calls "Optional, but would be powerful when you want to remove some
unwanted content" (the prose said Tongyi "strongly recommends" one until the
2026-09-28 review) and which is not in the catalogue; the order of a sentence,
an XML prompt's blocks or a JSON prompt's keys (above: shown as not checked);
Anima's card tag order and prefix, which are not in the guide prose - the
critic has them as excerpts and the page shows the order as the authors'; an
Illustrious prompt "missing" a composition or artist tag, which the authors'
own examples lack (the prose's "Give it a style" is the app's reading and
`unsourced`, so the critic cannot cite it to ask for an artist tag either; the
composition sentence is the v0.1 card's and stays citable); quality tags
placed past CLIP's 77 tokens, which the encoder cut already reports;
Illustrious's `sfw`, `nsfw` and `r-18`, which its release post accepts (only
`safe` is corrected, to `general`); a single klein reference with no "image 1"
(BFL's own "Turn this cat into a dog" names none - two or more are checked;
with none attached, an edit of "this" picture - `EDIT_PROMPT` - gets the
app's own note, `heuristic.no_reference`, since it renders from the words
alone, a plain text-to-image that looks like a success);
and `\(`/`\)` escapes, which the app's own autocomplete writes (below).
FLUX.1's "English is the most precise language" was on this list until the
2026-09-28 review took it out of the prose: BFL's current basics page says
other languages work "with the same level of quality", and while unchecked it
was still citable by the critic.

**The critic runs in llama.cpp, not in-process.** Measured on this card
(RTX 5080) with Qwen3-4B: llama.cpp b11206's CUDA build decodes Q8_0 at 170
tokens/s (`llama-bench`), about 151 end to end under a JSON grammar with
reasoning on - 8.8-9.0 s for 1,320-1,348 tokens at temperature 0, 2.8 s with
reasoning off - at about 5,160 MiB of VRAM over the desktop's (7,260 against
2,099 by `nvidia-smi`) at an 8k context, and the server loads in 1.7-1.9 s.
That was the research's greedy decoding; with the shipped sampling
`verified.md` records 8.1-16.0 s per critique (median 10.5), 19.3 s
for a tag prompt, and 5,581-5,611 MiB (about 5.5 GiB) over the
baseline. In-process transformers decodes the same model in bf16 at 33
tokens/s, so a reasoning-length answer takes about 40 s;
the image pipeline's own torchao fp8 encoder decodes at 9.4 (reusing it is
out); on the CPU, bf16 manages about 6. Constrained decoding in-process would
also need a new dependency. llama-cpp-python's CUDA wheels are built for
sm_75 to sm_90 only (Blackwell would JIT the PTX, cost unmeasured) with an
untuned CPU path, and a CUDA abort in-process takes the FastAPI server with
it; Ollama is a separate install with its own model store and a GPU scheduler
that knows nothing of the image jobs. The official release archives run as a
child process, so a crash is the child's. The model is Qwen's own Q8_0 GGUF
of Qwen3-4B (Apache-2.0) at a pinned revision - the research measured the
same weights converted from klein's text encoder, which is unmodified
Qwen3-4B, but a conversion needs tooling the app does not install and klein's
files exist only where klein does. Qwen3.5-9B had the strongest published
instruction-following scores the research found among Apache-licensed models
of 9B or less, but its speed on this build was not measured, so the pin is
the model that was.

**Reasoning stays on.** Without it the 4B models tried were unreliable in
ways that do harm: one called the camera missing when the prompt said "shot
on film", one applied the reference-image sentence to sign text, one advised
putting the negative phrase "don't make it look like a painting" in quotation
marks - which asks the model to letter it into the image - and rewrites kept
"no blur, no extra fingers" against the guide's negation rule. Krea 2's
Qwen3-VL and NewBie's Gemma 3, tried as critics, missed unquoted text and
negations too. With reasoning on, Qwen3-4B caught subject order, unquoted
text and negative phrasing and wrote a clean rewrite - and on one prompt
still kept the negatives in its rewrite, which is why the rewrite goes back
through the rules. The request uses Qwen3's recommended thinking sampling
(temperature 0.6, top-p 0.95, top-k 20; its card warns greedy decoding loops
in thinking mode), `--reasoning-budget 2048` so a long thought cannot run the
JSON off the end, and a `max_tokens` sized per request: the budget plus the
prompt's tokens as the running server counts them times the copies the answer
repeats (parts, the rewrite and, under a documented order, the reordered
version) plus 800, between 4,096 and 8,192. A fixed 4,096 failed review: a
4,000-character English prompt is 966 real tokens, and three copies after
2,048 tokens of thought do not fit.

**What cannot fit is refused before it is asked.** A question whose answer
cannot fit the 8k window even with no thought is refused with a `CriticError`
(`check_fits`), in two steps. Before a server starts, `critic.estimate_tokens`
scaled by 0.6 (`_ESTIMATE_FLOOR`) refuses only what cannot fit however the
text tokenizes - 4,000 characters of Chinese with as many again in a negative
prompt that runs, say. The estimate splits the text as Qwen2's pre-tokenizer
does after its NFC (Qwen3 keeps both), counts each piece a token at least, and
charges letters past ASCII at a per-script rate measured against Qwen3-4B's
own tokenizer: FLUX.2 [klein]'s `tokenizer.json`, whose ids equal b11206's
`llama-tokenize` on the pinned GGUF. It is a bound in neither direction. It
runs long on ordinary text, by a quarter to two fifths on English prose and at
most three fifths (Vietnamese), which the 0.6 covers - a bare list of
camel-case brands, two and a half times over, is the exception - and short on
vocabulary it cannot see: Turkish or Icelandic words and romanised names by a
fifth, Cantonese by two fifths, rare Han characters by half. So the server
counts too: once it is up and its listener checked, `_counted` asks its
`/apply-template` for the messages as it will read them and its `/tokenize`
for their count and the prompt's. A question that does not fit is refused
there, unasked - what it admits fits, unless the answer outgrows its copies of
the prompt and 800 tokens - and `max_tokens` is sized from the same count; on
the real server on 2026-09-28 the count of the messages equalled the prompt
tokens it then reported (1,072 of 1,072). A server that does not answer those
endpoints, such as one of the user's own, is sized by the estimate, with a
warning in the log. A refusal on the count costs a server start, 3 s on the
card and 28 s under WSL with the GGUF on /mnt/c; under WSL with it in the
Hugging Face cache on ext4, the default, a start and stop took about 3 s
while the file was in the page cache and 8-9 s after a drop (`verified.md`,
2026-09-29). Over 183 cases counted with
the real tokenizer (the 2026-09-28 review's 160, ten kinds of text from
Turkish to Cantonese and romanised names at 500-4,000 characters on klein and
Krea [dev]; English, brand-heavy, English under a 4,000-character negative and
Greek prompts on klein, Krea [dev] and Qwen-Image; Chinese on ERNIE), none
that fits was refused before a server started, none admitted failed to fit,
and no thought was cut while the window had room; 8 were refused before a
server started and 46 on the count. That review found the first estimate - a
token per digit and a third of one per other ASCII character or UTF-8 byte -
short by up to 45% on Greek, base64, Hindi and joined emoji: scaled by 0.6 it
admitted prompts needing 1.4 to 2.4 times the window, which ran to its end and
answered 502, and its `max_tokens` cut the thought short while the window had
room (F37). A second, per-script and unscaled, was still short on the
languages above and, long on the rest, refused prompts that fit.

**The page's "about 15 seconds" is the pinned pair's.** `verified.md` ("Prompt
analyser") has the times with the pinned GGUF and b11206: the 8.1-16.0 s above
over 19 FLUX.1 Krea [dev] critiques (median 10.5 s), each back at the baseline
VRAM a second later with no server left, and the Animagine tag prompt's
19.3 s, before the final review's fixes; 7.4-22.3 s over 20 after them (median
13.2 s), the card at its baseline and no server left after each run of ten.
The times include the server's start and stop; the page's figure lies within
both ranges and above both medians. The critiques generated 705-2,568 tokens,
thought included, which is what `MIN_TOKENS` (4,096) has to cover. An earlier
smoke by builder C, 12.7-18.6 s, once cited here as the basis, is history: it
ran the research's own `llama-quantize` conversion of klein's text encoder
(4,280,404,896 bytes against the pinned 4,280,404,704) with an earlier system
message, and is not in `verified.md` (review 2026-09-28, F79). Only the pinned
pair is promised it: `PromptCritic.describe()` (`ModelStatus.prompt_critic`)
gives the page the model's label - the name the answer's footer uses - and
what a first use would fetch, from the settings and file checks, and a server
or GGUF of the user's own gets neutral wording and no time. The review found
the fixed line naming Qwen3-4B and its time beside a footer naming another
GGUF, and promising a llama.cpp download that a configured server never makes
(F45). Card measurements belong in `verified.md` under "Prompt analyser".

**One pinned server per critique, keyed, and stopped before the gate opens.**
`critic.BUILDS` pins release b11206's Windows CUDA 13.4 archives and the
Ubuntu ones for WSL (the default architecture list includes sm_120) by size
and the sha256 the GitHub release API publishes, and `MODEL` pins
`Qwen/Qwen3-4B-GGUF` `Qwen3-4B-Q8_0.gguf` by revision, size and sha256. The
binaries are a download like the weights and handled like them: each archive
to `.part`, verified, extracted into a sibling with zip members that escape
the target refused and tarballs through `filter="data"`, the marker written
last and the tree renamed into place under `HF_HOME/local-diffusion/llama.cpp`;
a tree without the marker is never used, nor is one whose marker does not
decode (re-saved as UTF-16, say): that counts as none, so `ready()` answers
False rather than failing every `/api/status` poll, and the install replaces
the tree (review 2026-09-28, F18). The sibling is a
`tempfile.TemporaryDirectory`, which removes itself on the way out - but not
after a hard kill, nor on Windows when the interpreter exits with a file in it
still open on the prepare thread, and each such attempt left up to 744 MB
(F39). The lock holds through the extraction, so any such directory found
under it is stale, and `_sweep_work` removes them on every install call,
installed or not. The review's first fix took every `<target>-*` directory
that was not another build's target, which would have deleted a user's
`b11206-win-cuda-13.4-x64-backup` beside the install. The sweep takes only
what `tempfile` makes there (`_is_work`): the target's name, a hyphen and
eight characters of `tempfile`'s alphabet (`[a-z0-9_]`), as a real directory -
not a link or junction - and never another build's own (a key of this one's
and eight more characters would be named like work, and empty it would pass
the contents check), holding nothing but the `tree` and the `archive-<n>`
scratch directories an install extracts into. The contents check is what
spares a copy of an install named `...-backup01`, eight characters `tempfile`
could have written, since a copy holds the server and its marker. Removal is
best-effort and never fails an install: a file another process holds, a work
directory or an install root that cannot be listed are logged or passed over,
and what cannot go yet is tried again by the next
(`test_the_sweep_never_fails_an_install`, the converters'
`test_the_sweep_never_fails_a_load`). The model converters save into work
directories the same way and are swept by the same kind of check
(`single_file.sweep_work`). The GGUF is hashed
once, and a stamp of its path, size and mtime spares every later critique
hashing 4 GB again. A cached GGUF off its pin is downloaded again, once per
call, rather than left to "delete it and retry" (F40): in a Hub cache with
symlinks, deleting the snapshot entry leaves the blob, which `hf_hub_download`
relinks without downloading, and in one without them (Windows without
Developer Mode) the entry is the file itself, and `force_download` alone
replaces only the blob beside it - so the entry is removed and the download
forced. If the file is still off its pin, the error names the resolved file
and says it was freshly downloaded, so the pin or the download path is wrong,
rather than advising a retry that has just been made; the pin is the LFS
object's own sha256, so such bytes mean a wrong pin or corruption on the way,
a later click may legitimately succeed, and nothing is remembered. A forced
download that fails (the Hub unreachable, `HF_HUB_OFFLINE=1`) is a
`CriticError`, the route's 502, quoting the cause of `hf_hub_download`'s
error, whose own words for an unreachable Hub are only "the above error"; the
entry is gone by then, which is right, since it was off the pin. A first use
is about 5 GB (576 MB of binaries and runtime on Windows, 4.3 GB of model) -
the page sums the pins' sizes for only what is still missing (`describe()`)
and prints them in binary units, labelled so: "about 4.5 GiB" for both and
"about 4.0 GiB" for the model alone. Every size the page shows - the card's
memory, the free disk, a download, a grid's PNGs - was computed in 1024s and
labelled GB and MB, which made the critic's 4,856,283,677 bytes "about
4.5 GB" where the Hub lists 4.86 GB; they are labelled as computed now
(2026-10-05), GiB, MiB and KiB, as nvidia-smi reads the card, rather than
recomputed to decimal, in which a 16 GiB card would read 17.1 GB. The CLI's
`gallery_reports.human_bytes` says the same. So `prepare()` runs before the gate, on a daemon thread
as the model load does (a Ctrl-C must not wait out a download), and the route
checks the queue and the load again afterwards, since minutes may have passed.
A cached `prepare()` is not quite offline: `hf_hub_download` builds
huggingface_hub's user agent before it looks in the cache, and that fetches
the Hub's `/api/agent-harnesses` list, best-effort and at most once a day,
unless `HF_HUB_OFFLINE` or `HF_HUB_DISABLE_TELEMETRY` is set - the library's
request, left as it is. Only `critique()` runs inside `jobs.run_when_idle`:
`llama-server` on 127.0.0.1 and a free port, `--offline`, `--no-ui`,
`-c 8192`, `-np 1`, `--jinja`, `--fit on` (b11206 adapts only unset arguments,
so layer placement follows the VRAM actually free while the context stays
pinned by `-c`; hence the 180 s health and 600 s request timeouts, for layers
that land on the CPU), then, after the listener check below, the question
counted by the server itself and one chat completion, then terminate, wait,
kill in `finally`, applied to the whole process tree, so the card is free
before an image job is let in. A resident server would hold ~5 GB against the
pipeline for nothing when a start costs two seconds;
`--sleep-idle-seconds 120` is only the backstop for a server this process
could not stop.

**The stop takes the server's whole process tree.**
`LOCAL_DIFFUSION_CRITIC_SERVER` may name a wrapper - a .bat, a venv launcher,
a script that does not `exec` - whose server is a grandchild. The 2026-09-28
review found that stopping the child alone left such a server listening, and
holding the card, after the gate opened, and that the log it still held turned
a finished critique into a 500 (F19). So `critic_server._ServerProcess` stops a tree,
even when the child itself has already exited, since a wrapper's server
outlives it. On Windows the tree is a job object with kill-on-close. The child
is created suspended and resumed only once it is in the job (by ntdll's
`NtResumeProcess`, since Popen keeps only the process handle), so nothing it
starts escapes; where Windows refuses a job, the child runs alone and only it
is stopped, with a warning in the log. The stop waits until each process's own
handle is signalled, not merely until the job's process list is empty: a
process leaves that list before its handles close, and when only the list was
polled, 17 of 30 wrapped critiques under full CPU load returned with the log
still held and the gate opening mid-teardown, against 0 of 30 with the handles
waited on. On POSIX the tree is a session of the child's own. On Linux every
live process in it is signalled and waited for, whichever process group it has
moved to (`set -m`) - the same set the listener check accepts - and a process
counts as gone only when its last thread has exited, since /proc shows a
multithreaded process as a zombie while its other threads still hold its
sockets; elsewhere on POSIX the child's process group is signalled. A kill
that is still not waited out is a `CriticError`, since the card may still be
taken, and the critic forgets the process either way. The log's removal is
retried for a few hundred milliseconds, as `save_image`'s rename is, and a log
still held after that is a warning, not a lost answer. Open, from review, on
POSIX only: a server can be orphaned by a hard kill of the app, or by closing
the terminal the app runs in. It frees its VRAM after the idle timeout but
stays alive and listening, since its session outlives the app: the terminal's
hang-up no longer reaches it, and the app does not take the hang-up as a
shutdown, because uvicorn handles only SIGINT and SIGTERM. Before the process
tree, the hang-up did reach a server in the app's own process group
(2026-09-28). Treating SIGHUP like SIGTERM in `main.run()` would close the
second case; `PR_SET_PDEATHSIG` was left out, since `preexec_fn` is unsafe in
a threaded process and would reach the child alone. On Windows the job's
kill-on-close takes the server down with an app killed outright (checked with
a stand-in server on 2026-09-28, not with llama-server).

**The key, and who listens on the port.** Its CORS allows every origin, so a
web page in this browser could reach a keyless server: each start gets a fresh
random key, and before the prompt is sent a keyless `GET /v1/models` must
answer 401 (b11206 leaves only its health endpoints public), or the server is
refused. The key goes to the server as `LLAMA_API_KEY` in its environment,
which b11206 reads in place of `--api-key` (its `--help` says so): on the
command line, as first built, any local user could read it
(`/proc/<pid>/cmdline` under WSL). The child's environment otherwise drops
`LLAMA_*` - `LLAMA_ARG_TOOLS` would switch on its shell and file tools - and
loopback requests bypass any proxy. The 2026-09-28 review found the 401 probe
proving less than it was taken to: a process that took the port first - it is
on the command line for any local user to see - and refused keyless requests
would have been sent the prompt and the key, and could have answered in the
server's place (F43). Neither the bind failing (b11206 sets `SO_REUSEADDR`)
nor the 401 probe (a stranger can answer 401 too) guards the gap between
`_free_port()` and the server's bind. So, last before the key goes out (the
count and the question both carry it), the port's listener must be the
server's own process tree (`critic_server._listener_is_ours`, as
`_wait_until_healthy`'s comment says): on Windows the TCP tables' owning pid
is checked against the job's processes, and on Linux, WSL included, the
listening socket's inode in /proc/net/tcp and tcp6 against the descriptors of
the session's processes, a listener on 127.0.0.1, on every address or
dual-stack counting alike. A listener that belongs to anyone else, or no
visible listener at all, is refused with a `CriticError`, and neither the
prompt nor the key is sent; the real b11206 server passed the check on Windows
and under WSL 2 on 2026-09-28. Where /proc/net/tcp is empty or incomplete
(WSL 1, gVisor) that refuses every critique, which is accepted: the app runs
on Windows and WSL 2. On other platforms, or where the tables cannot be read,
nothing reads the owner, and the key and the 401 probe are the only guard. The
check reads the owner once, just before the first keyed request; a listener
bound beside the server after that moment is not seen. A Unix socket in a
private directory, the review's other suggestion, was left out: b11206's
`--host` takes a `.sock` path, but urllib has no transport for one.

**A server or a model of one's own is the user's choice.**
`LOCAL_DIFFUSION_CRITIC=false` switches the critic off;
`LOCAL_DIFFUSION_CRITIC_SERVER` and `LOCAL_DIFFUSION_CRITIC_MODEL` name files
already on disk in place of either download (made absolute, because the server
starts in its own directory, but not resolved: a Hub snapshot's entry is a
link to a blob named by its hash, and the critique's footer shows the
configured file's name; `server_command` still resolves both for the command
line), and naming one is choosing it: neither is held to the pins, but a
server of one's own must take b11206's flags and its key from the environment,
and should answer `/apply-template` and `/tokenize` with the key - without
them the question is sized by the estimate. The server setting should name the
llama-server binary itself, not a script or launcher that starts it: a wrapper
is stopped with everything it starts - its job on Windows, its session on
Linux whichever process group it moves to, its process group elsewhere - but a
server that leaves the tree (on POSIX, into a session of its own: `setsid`, a
daemon's double fork) escapes the stop, and where the platform can tell, the
listener check refuses its port.

**The Linux build needs what its tree does not carry.** The WSL tarballs'
layout - one top-level directory each, flattened on install so the CUDA
runtime sits beside the binary - was read from b11206's `release.yml` when
this was built and verified on the card on 2026-09-27 (`ldd` resolves every
library inside the tree; `verified.md`). What the tree does not carry was read
off its ELF headers under WSL 2 on 2026-09-28: symbols up to GLIBC_2.38 and
GLIBCXX_3.4.32, libgomp.so.1 for the CPU backends, and OpenSSL 3's libssl.so.3
and libcrypto.so.3 (`Build.glibc`, `Build.libraries`). The 2026-09-27 `ldd`
ran with libgomp1 already installed (it came with gcc); a fresh WSL Ubuntu has
none, and the 2026-09-28 review found its first critique downloading about
4.9 GB to end in exit 127, with the button still offered (F44). `supported()`
is False below glibc 2.38, as on Ubuntu 22.04 and Debian 12, unless a server
of one's own is named; the version is `os.confstr`'s, a libc call cheap enough
for the status poll. A direct call to the critique route then gets a 422
saying so (`critic.pinned_build_refusal`, read by `api_prompt.critic_unsupported`),
not that no build is published for the platform. `critic.preflight`, called
from `prepare()` ahead of both downloads, asks the dynamic loader for the
three libraries in a child interpreter (`ctypes.CDLL` there, so no second
OpenMP runtime is loaded beside torch's) and raises `CriticUnavailable`, the
route's 503, naming the apt packages (libgomp1, libssl3; libssl3t64 provides
the latter on 24.04 and later). A missing library leaves the critic offered:
the fix is one `apt install`, and the 503 says which. A probe that cannot run
checks nothing, GLIBCXX is not checked separately, and a server that still
exits 127 on a missing library names its package (`_exited`). `wsl/setup.sh`
checks the same with `getconf` and `ldconfig -p` and prints the apt command
rather than running sudo; it installs none of them.

**The critic cites only what it is given, and what it says is checked
before it is shown.** `prompt_sources` gives it, per guide constant, the
guide's own `prompt` prose split into sentences (G1..Gn - the text the ?
badge shows) and short verbatim excerpts the checkpoint's authors put in
writing (A1..An); the answer's JSON schema makes `cite_id` an enum of
exactly those ids, so a citation cannot be invented, only misapplied, and an
id that names nothing drops its issue (`dropped`). A G sentence links to the
page of the `analysis` entry whose cite it carries - the page that entry's
rule finding links - so a rule finding and a critic cite of the same words
open the same page, and the prose gives each page's words a sentence of
their own: `test_a_sentence_links_the_page_its_rule_findings_do` fails on a
sentence carrying two pages' words. Until the 2026-09-28 review every
sentence without a `sentence_urls` entry took the guide's `analysis["url"]`,
and 15 rule findings in 7 guides (13 sentences) linked one page from the
rule and another from a critic cite; the opening summaries of FLUX.1's,
schnell's, Krea [dev]'s and Krea 2's prose, which folded several of BFL's or
Krea's pages into one sentence, were split one page to a sentence. A
sentence no entry cites links to the page `Grounding.sentence_urls` names
for it, else to the guide's `analysis["url"]`: Krea [dev]'s "AI look" to
BFL's launch post (<https://bfl.ai/blog/flux-1-krea-dev>), its settings
sentence to Krea's README and its camera sentence to BFL's building page;
Krea 2's report claims (dense captions, the text-rendering reward, the
encoder) to the technical report and its CFG and Raw sentences to the
README (<https://github.com/krea-ai/krea-2>); Illustrious's untuned-base,
natural-language-example and v2.0-negative sentences to the v2.0 release
post through CivArchive
(<https://civarchive.com/models/1369089?modelVersionId=1546777>) and its
v0.1 negative prompt to the v0.1 card; ERNIE's encoder to the report
(<https://arxiv.org/abs/2605.25347>) and its 2048 tokens to each repo's
`tokenizer/tokenizer_config.json`; NewBie's 512/2048 to the author's
`finetune.py`; the undistilled ERNIE-Image's and Z-Image's settings to their
own cards. `test_each_sentence_links_the_page_that_says_it` pins each
of those guides but Z-Image sentence by sentence, and FLUX.1's and
schnell's; Z-Image's one `sentence_urls` entry is held only by the
check that every entry is followed, not by which page it names.
Z-Image's "Long, detailed natural language" rule entry cites the
maintainers' discussion 8 in the Turbo repo, where that advice is;
other sentences stating how this app's pipeline reads the prompt
(a CLIP or T5 limit) still link the guide's page. A sentence the prose
carries with no author behind it - the app describing its own menus,
autocomplete and chips, the Expand button, the sampler it runs, what it does
not implement, or a starting point or reading of its own - is listed
`unsourced` and never citable, and no rule entry may cite one: the
2026-09-28 review listed nineteen across ten guides, among them
Illustrious's "masterpiece, newest is a practical starting tail" and "Give
it a style", Kohaku's "Keep prompts concise" (its card's context is 300
tokens) and schnell's BFL-reference 256-token sentence, which does not hold
here. How the pipeline reads the prompt - a token limit, where guidance
brings the negative prompt in, the system instruction NetaYume's loader
adds - stays citable, because it is what runs:
`test_no_citable_sentence_misstates_a_token_limit` holds every stated limit
to `encoder_table`'s table and
`test_no_citable_sentence_is_the_app_talking` keeps the UI markers out; a
test fails if a listed sentence stops matching the prose. Krea [dev]'s
"photo language" and "give it room" were the first two such sentences,
until the final review found the critic, grounded in BFL's pages, telling
users to remove the camera details the badge beside it asked for - they
were taken out of the prose and replaced by what BFL's building page does
say (camera settings "when framing matters"). The fallback guide's
sentences are the app's general advice: its system message names no
authors (`build_context` drops "from the checkpoint's authors" when
`Sources.url` is None), its citations carry no url, and its "Unrecognised
checkpoint." label is unsourced.

Excerpts, checked against copies fetched 2026-09-27 - verbatim but for
markdown emphasis, elisions marked "…" and a few labelled reformattings
(three rows of BFL's replacement table, Neta's vague/specific pair, orders
flattened from lists): BFL's family pages for FLUX.1, schnell and Krea [dev]
(<https://docs.bfl.ai/guides/prompting_unified_basics>, `_building`,
`_technical`); for klein,
<https://docs.bfl.ai/guides/prompting_guide_flux2> (written for [pro] and
[max], klein by family) and the T2I upsampling instruction in
<https://github.com/black-forest-labs/flux2/blob/main/src/flux2/system_messages.py>,
which BFL's own CLI also runs for klein; Krea 2's `docs/prompting.md` and
`docs/expansion.txt`; Qwen's
<https://github.com/QwenLM/Qwen-Image/blob/main/src/examples/tools/prompt_utils.py>
(and the portrait flow from `prompt_utils_2512.py`, labelled same-family);
the Z-Image Turbo Space's rewriter
(<https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/pe.py>)
and Baidu's <https://github.com/baidu/ERNIE-Image/blob/main/src/pe_prompt.txt>,
both Chinese, so `text` is this app's translation, labelled, with the
original kept; the Neta Lumina prompt book
(<https://neta.art/blog/neta_lumina_prompt_book/>), labelled inherited from
NetaYume's parent; NewBie's card and the XML field rules of its captioning
instruction (`XML Prompt.json`), nothing of its adult-content captioning;
the Anima, Animagine, Z-Anime, Kohaku Zeta and Epsilon-rev2 cards; the
Illustrious report (<https://arxiv.org/html/2409.19946v1>); and WAI's v17
description through its archive
(<https://civarchive.com/models/827184/wai-nsfw-illustrious-sdxl?modelVersionId=2883731>).
A per-guide `caveat` says how far the sources reach: Krea published no
prompting rules for Krea [dev], so both lists are BFL's; on every FLUX.1
checkpoint, that BFL's steps start with the image type and subject - so
"Photo of" or "Moody night photography" still puts the subject first - and
that camera settings are part of BFL's template, not a fault (the excerpts
now carry "Start with the image type and subject" and the template with
"not a strict formula"; a critic held to a bare "subject first" called
Krea's own example out of order and told it to put its style first in the
same answer); NetaYume's excerpts are its parent's and the app adds the
system prefix itself; Z-Anime's "details early" is its 4-step variant's.
An excerpt's note travels with its citation to the page too
(`Citation.note` -> `CritiqueIssue.note`), shown under the issue and in the
cite's tooltip, as Order shows the same notes; until the 2026-09-28 review
it reached only the system message, so a cite of ERNIE's translated
rewriter instruction read as Baidu's words and NetaYume's inherited prompt
book as its own authors'. The fallback's cites carry this app's note
instead ("this app's general advice: no author sources are known for this
checkpoint"), and the page labels them General, not Guide.
The critic is also told that order templates position the parts actually
supplied, not require every slot. Generic and original subjects are valid;
character and series names are optional and must not be invented or flagged
as absent. The deterministic tag rules already work this way, and
Animagine's own examples include generic subjects without either name.

The system message also says the rewrite adds no object, scenery, mood or
text and never pads toward a word range - a range is the guide's band for
most images, not a minimum (almost every rewrite on the card padded a short
prompt, against the sources' own "Start short") - that quotation marks go
only around lettering, never the prompt, and that the authors' example
prompts lend their form, not their subjects or text. It stays under about
1.5k tokens (the phase-1 review measured 500-880 with the real Qwen3
tokenizer before findings; the budget test holds every checkpoint's under
6,000 characters).

Then every answer is checked (`validate`). A quote is looked for in the
prompt as whole words - "man" is not the end of "woman", "pen" not the
middle of "OPEN", "OPEN" not the start of "OPENING" - though Han, kana, Thai,
Lao, Myanmar and Khmer, written without spaces, match anywhere, as does Hangul
(Korean spaces its words, but a particle joins the word before it; `_SPACED`
builds on `QUOTED`'s `_WORD_NOT_CJK`), and a tag's underscore counts as a space
("hair" in "long_hair"). Case, runs of whitespace, and curly against straight
apostrophes, quotation marks and dashes are set aside (folded one character for
one, Word's non-breaking hyphen included, so the span is still the prompt's
own); the quote is nulled when absent, the span located on the server so the
page selects the prompt's own characters. A part is kept only if each of its
comma pieces is in the prompt, and "none" or "n/a" is no part, and
`text_to_render` only if it is lettering the prompt asks for
(`prompt_rules.rendered_targets`: its quoted stretches and unquoted sign
text) - on the card it held the whole prompt in most runs. Words may run on
within their clause past an end the rules had to guess; past one they are
sure of ("says OPEN in red letters") only words starting with a capital or
a digit run on - "OPEN IN 5 MINUTES", or "Cafe On Main", a name the rules
stop at its "On" - so lowercase description never passes as lettering.
Either way they start before the target's `head`: where lettering may hold
a negation the rules stop at ("reading comics without pictures"), the
negation's words alone are the absence the rules read, not lettering.

An issue that repeats a rule finding reported on the prompt (a negation,
quotation marks, tag order, weighting, quality boilerplate) is left out,
whether it repeats it in its wording or by a quote that takes in the words
the finding points at, since the model did so in nearly every answer
however it was told not to; a finding on the negative prompt counts only
for an issue that says it is about the negative prompt. Left out too is one
that speaks to the prompt's own length wherever the guide has a range and the
rules counted it (not XML or JSON), reported or not, since the model
miscounted words beside the rules' correct count; in range, `set_aside`
says the check measured the length (N words, inside the range) rather than
naming a finding that is not there. The 2026-09-28 review narrowed all
three checks. The length pattern had caught "a focal length of 85mm", "the
dress length is" and "add 3 words on its glow"; it now reads a sentence at
a time, and only claims about the prompt's length (its count, the guide's
band, a bound, "Too short"), not a number of words about something ("add at
least 5 words about the lighting") or lettering ("keep the sign text under
5 words"). Phase-4 re-verification found that narrowing held only for a
count followed by "on", "about", "of" or "describing": "add at least 5
words for the lighting", "add more than 3 words to the lighting", "only 2
words describe the setting", "describe the umbrella in more than 2 words",
"give the lighting at least 5 words", "the lighting gets only 3 words" and
"the ideal length of the raincoat" still went, under a note saying the
check had measured the prompt's length. Its fix made a count some part's
where "for", "to", "in", "covering" or "describe" followed it
(`_SOME_PART`), "in" came ahead of it, or a part was given it ahead ("give
the lighting", "the lighting gets"; `_PART_COUNT`), unless what followed
named the prompt, its guide or range, "it", a total or all, or a verb of
fit - and review of that fix found each exception wider than its purpose.
"Only 3 words of the prompt are about the lighting", which the first
narrowing had kept, went again, since "of the prompt" named the prompt;
"to match the lighting", "for all three figures", "the dynamic range" and
"the style guide" were the prompt's, since a fit verb, "all", or any range
or guide was; "the lighting has only 3 words" was still the prompt's
length; and the other way, "keep it in under 80 words", "the recommended
length of 30 words" and "too long a description" were no longer length
claims. Now a count of the prompt's words is a part's where the sentence
goes on to give them to one ("are about", "cover", "describe", "go to"); a
fit is the prompt's only where it fits the guide, its range, a limit or a
number ("to reach the range", "to stay under 80"); a range or guide only
bare or as the guide's, the model's, the recommended or suggested one or a
span of numbers; "all" only as "in all". A part named ahead takes its
count by give or devote, by has, is, are, gets, receives or uses, or by
cut, trim, shorten, expand, keep or limit to or under a bound
(`_PART_COUNT`); "in" or "with" ahead of a bound is how a part is to be
described unless "it", "this", the prompt or its description comes before
it (`_MANNER`); and a recommended length of so many words, of prompts or
of the description is the prompt's. Each of those words has a test row of
its own. It is still a reading of the critic's wording, so a phrasing
unlike these can go either way. "The prompt runs long" (reads, seems, a
little, slightly) counts as a length claim; "too long a shadow" does not,
"too long a description" or "a list" does. The last fix round found "keep
the lighting description under 5 words" still left out under the note that
the check had measured the prompt's length, since any description was the
prompt's: now the word just ahead of "description" decides. A determiner
("the description", "your description") or a word for the whole ("the
full description", "the image description"; `_WHOLE_WORDS`) makes it the
prompt's; any other word names a part ("the lighting description", "the
background description"), as an object other than the image or picture
does in "the description of the lighting" - in `_PART_COUNT` and in
`_MANNER` alike. It also found "the prompt reads as long" kept: a claim
about the prompt's length, like "reads long", it is now left out with the
measured note where the count was in range, and as repeating the finding
where one was reported. Review of that fix found any word the list did not
name taken for a part: "keep the revised description under 80 words", "the
new description has only 12 words" and the like, the critic's words for its
own version or for the whole, were kept where they had been left out. The
other way, "your lighting description", "a lighting description" and "the
soft warm lighting description" were still the prompt's in a count, whose
determiners lacked "your", "my", "our", "a" and "an" and which named a
part in two words at most, while a manner read them as parts; plural
descriptions were the prompt's in a count and a part's in a manner; and
"as" after any verb made "the prompt is as long as it needs to be, but the
lighting is vague" a length claim. Now one list of determiners
(`_DETERMINERS`) and one of words for the whole (`_WHOLE_WORDS`, the
critic's "revised", "new", "final", "short" and "detailed" among them)
decide both readings, singular or plural, a noun for the whole as a
possessive too ("the image's description"); a count takes a part named in
up to four words; "a" or "an" names a part only with its description, so
"keep a single paragraph under 80 words" is still the prompt's; and only
"reads as" is a length claim, never "as long as".
`CriticContext.reported` counted negative-prompt findings, so weighting in Animagine's negative prompt left
out "give the expression more emphasis".
A quote within 8 characters of any finding's span counted, so "a shop" and
"rain" beside "no people" went as restating the negation, and on FLUX.1
T5's `unknown_chars` spans, which cover nearly all of a Chinese prompt,
took every quoted issue. A quote now counts only if it takes in some of the
finding's words and runs at most 8 characters past them (none past a
single word, surrounding punctuation aside), and only the spans of
findings about meaning count: guide, author and heuristic rules and
weighting syntax, not a cut, a template tail, unknown characters or a
control token. Order issues are read as what each asks for: a part first,
or one part before another ("put the camera settings before the
lighting"); an instruction in the negative ("avoid placing the style before
the subject", "don't start with the style") asks for nothing it names. One
the documented order reverses is left out - a part first that the order
puts another part ahead of, or a pair it puts the other way: on FLUX.1
[dev], "lighting first" or lighting before camera; on Krea [dev], whose one
step is "subject first", anything but the subject asked for first or put
ahead of it. A step names every part in it, so WAI's "Subject and scene"
puts the scene first too. Then the order issues whose wants cannot all
hold together are all left out. The review found "Start with the subject"
and "Put the camera settings before the lighting" both left out as a
conflict, because the pair had been read as "camera first"; with the
documented order applied first, "Move subject to the beginning" beside
"Place style description first" on klein keeps the former, where both used
to go.

`reordered` survives only where the app has a documented order and only as
a pure reorder - the same whole tags, clauses, sentences or XML blocks (for
JSON, the same values written back with the keys sorted, each number as the
text writes it (`_json_forms`), with no key repeated, in another key
order), every quoted span unchanged, typed marks folded
(`is_reorder`). The movable pieces also protect every `rendered_targets`
span, including unquoted lettering such as `《你好, 世界》`; this protection
applies before either tag-order validation or the unchecked prose path.
Initials and common abbreviation periods stay inside their piece, so
`Mr. Green` cannot trade names with `Ms. Blue`, and `J. M. W. Turner`
cannot become `J. W. M. Turner`. An ambiguous abbreviation at a sentence end
conservatively prevents that split. Here "quoted" is `prompt_rules.QUOTED`,
the one pattern the rules, the encoder layer and the critic share (the
critic's own narrower copy let a «…» or „…“ quotation split across two pieces), whose
single-quote pair may touch Han, kana and Hangul but no other letter or
digit: Chinese writes 写着'营业中'的 with no space, and ERNIE's own
rewriter asks for single quotes. Until the 2026-09-28 review the pattern
refused that pair, so such lettering was nulled from `text_to_render` and a
rewrite that changed it was kept. Word multisets were the first reorder
check, and review broke it: "a woman in a red dress and a blue hat" came
back with the colours swapped and the sign text's quotes gone, and passed.
The 2026-09-28 review found the JSON check comparing values with Python's
1 == 1.0 == True, so a literal retyped as another type passed; its fix
compared `json.dumps(sort_keys=True)`, and phase-4 re-verification found
that still a comparison of values: 0.50 written 0.5 or 5e-1, 1.0 written
1.00 or 1e0, -0 written 0, passed beside a key moved, and the kept version,
the critic's text, gave the encoder other text than the user's. Numbers are
now read as written. What is kept (JSON aside, which keeps the critic's
text: its values are the prompt's writing, only the space between them the
critic's) has the prompt's own pieces, each as the prompt writes it, in
the critic's order, with the critic's space and punctuation between them: a
case-folded comparison once let an upper-cased reorder through, and T5 and
Qwen3 read case, while separators left where they lay turned "says EAT!"
into "says EAT." once the sentences moved. Where Check placed the tags
itself (`order_checked`), the critic's version stands only if its pieces
are the rules' `reorder_tags` result, and the text kept is that result: any
other permutation is not the documented order. That holds where every item
was placed. Where the rules could not place some item the order rule reads
(`prompt_rules.places_every_item` false, placing with the categories the
analysis used, `PromptAnalysis.categories`, kept off the response), that
item keeps every slot a category could give it, and Check's order - or its
"nothing to move" - is one the documented order allows, not the only one.
That covers an item never looked up (offline, an unusable reply, past the
64-tag cap) and one the index was asked about and had no usable category
for - an OC's name, a trigger word, a typo, a tag past the cutoff - where
that item's slot decides the order (below). There the critic's
version stands where the order rule, run again on it with the same
categories (`prompt_rules.order_finding`), flags nothing, and is set aside
in the rule's own words where it flags something. It must also keep the
items the order rule reads whole (`prompt_rules.order_items`) and be text
the rule reads at all: outside protected lettering, the critic's pieces
break at "、", ";" and line breaks too, so four tags joined by "、" are one
item to the rule, which no order can be wrong about. Lettering such as
"《你好, 世界》" is now protected directly during piece splitting, including
on prose checkpoints whose order the rules do not check. The first fix of
the 2026-09-28 review put this exception on the page: Compare offered the
critic's reordering where the lookup was partial (`categories_detail`), while the server still dropped
every such reorder, so the page's branch was reachable only by injection.
The follow-up moved the one rule to the server, first keyed on
`categories_detail`; its review found that key wrong - 64 tags the index
did not know were set aside with a note saying Check had placed them,
while 65 were kept - and the raw comma items it compared letting a critic
part 《…》 lettering at its comma. Both were fixed as described above.
Phase-4 re-verification found the page's "Partly checked" still on the old
key, so the critic's "allowed" reordering could sit under an order drawn as
fully checked; the page now reads `PromptAnalysis.order_detail`, set from
the same placement (`prompt_rules.unplaced_items`), and
`test_the_page_and_the_critic_read_partial_placement_alike` holds the two
to the same prompts. The review of that fix narrowed the predicate for both:
an unplaced item counts only where its slot decides the order. Where none
does, `reorder_tags`' order is in the documented order whatever the missing
categories are - a pair it swaps was out of order beyond doubt, and one it
keeps was in order beyond doubt, or is a tag and its copy, in order
whatever that tag is - so it is the documented
order itself, and a critic's other permutation is set aside as before. With
`green hair` known and `firefly` not, `1girl, firefly, green hair,
masterpiece` is in order whatever `firefly` is, and the critic's swap
putting `firefly` after `green hair` - out of order were it a character -
is now set aside instead of being kept as "allowed"; with the prompt the
other way round, `firefly`'s slot decides the order, and the swap putting it
first is kept.

On the real server the model's `reordered` added "great score, absurdres"
and once copied a character tag from the example prompts; on FLUX prose it
survived in 1 of 10 runs, and in 0 of 20 after the final fixes, being the
rewrite again 17 times (`verified.md`). So on tag checkpoints the rules'
`reorder_tags` is the dependable reordered version, and the page prefers
it; on prose the user writes one (Compare). The comparison takes the rules'
reordering, else the critic's as the server kept it, and adds no rule of
its own. The critic's Use reordered says which it is as the server says
(`PromptCritique.reordered_kind`: `check`, `allowed`, `unchecked`), not as
the page reads its own last check: Check's own order, one Check's rule
finds nothing wrong with though it could not place every tag, or - where
Check read no order - the critic's reading of the order, not one the app
checked. The critique route analyses the prompt again, and a lookup that
failed is not cached, so the page's last check may have placed the tags
with other categories than the ones the server decided with. On a tag or
hybrid checkpoint Check places a prose prompt too: a clause that
`_is_sentence` reads as a sentence takes the general slot, so there a
sentence order is tested under Your version, while a clause that a Danbooru
name could still be keeps every slot, since prose is never looked up, and a
critic's reorder of such clauses can stand as `allowed`. Where every clause
was placed, a critic's reorder of its sentences is set aside like any other
order but Check's; the note said Check placed "these tags" until phase-4
re-verification found it said so of sentences, and it now names tags only
on a prompt read as tags, and "this prompt" elsewhere. The last fix round
found its sibling, for a reorder that does not keep the comma items whole,
still saying Check would read "other tags" on a prose prompt ("a girl
standing in the rain. smiling softly, holding a red umbrella", whose items
`order_detail` calls clauses) and a hybrid one; it follows the same rule
now: "other tags" on a prompt read as tags, "other clauses" on prose and
"other tags and clauses" on a hybrid prompt.

Retained tag items keep their original parenthesis escaping. The critic had
returned tag prompts with only the backslashes in series names removed,
which looked like an improvement and became a redundant comparison row.
Validation restores the original spelling where a whole tag differs only
in parenthesis escaping, before the unchanged check; it does not normalize
prose, lettering, or other changes to a tag. The critic is instructed to
keep escapes and return the original when no useful edit is needed.

A rewrite that is the prompt as it stands (whitespace, a closing stop and
how its marks are typed aside) is dropped and said so, since the schema
requires a rewrite and an echo is the critic proposing none. One wrapped
whole in quotation marks, as four of the 19 FLUX rewrites on the card were
(`verified.md`), is unwrapped, sentence punctuation after the closing mark
included: a stop, spaced or not, stays on the text, closed up, and a comma,
semicolon or colon - the critic's own sentence going on - goes. The
2026-09-28 review found '"…sofa".' dropped as invented lettering, and the
fix set aside only a stop right after the mark; phase-4 re-verification
found '"…sofa" .', '"…sofa";' and '"…sofa",' still dropped that way. One
that quotes words the prompt does not have is dropped -
lettering nobody asked for, such as the sign example's "Black Forest
Labs" - and one that quotes the prompt's description gets a
`heuristic.rewrite_quote` warning where it sits: the rules skip quoted
text, so none of these reached the layer-1 re-check. Quoted words are held
to the prompt's lettering with the same folding as a quote, so “Joe’s
Diner” retyped with a straight apostrophe, or 24–7 with a hyphen, is the
same lettering; until the 2026-09-28 review such a rewrite was dropped as
inventing lettering. The review's optional warning that a rewrite changed
only its lettering's punctuation was not built. A rewrite with six or more
content words the prompt has in no form gets `heuristic.rewrite_adds`
(info), naming them: the instruction not to pad did not stop the card's
critic from adding lenses, light and scenery to short prompts, and a
comparison against such a rewrite tests new content, not only wording. What
validation takes out besides uncited issues is said on the page
(`PromptCritique.set_aside`): restated issues and ones speaking to a length
the check measured, order issues the documented order reverses or that
cannot all hold, parts the model named that are not the prompt's words or
its lettering, a rewrite or reordered version over the 4,000-character
limit, a rewrite that is the prompt unchanged, came back wrapped or quotes
words the prompt lacks, and a reordered version that changed words or that
Check's order does not take (above).

The answer is read from the content after the reasoning, but not only after
the last `</think>`: a prompt pasted from another model can hold one, and
the answer quotes it back. Qwen3's control and reasoning tokens as typed in
either message - `<|im_end|>` and every `<|...|>`, `<think>`,
`<tool_call>`, `<tool_response>` and their closers - reach llama-server
defused by a zero-width joiner after the "<"; one already there gets
another, so the escape is reversible. llama-server tokenizes the rendered
chat with special tokens parsed, so until the 2026-09-28 review a typed
`<|im_end|>` ended the user's turn and opened one of its own, and on klein,
Z-Image and Qwen-Image the control-token finding that quotes it ended the
app's system message early. `validate` takes the joiners out of the
answer's strings, so quotes and parts are looked for in the prompt as
typed. The rewrite's layer-1 findings are the rules and encoders run over
it by the route (`_lint`, without the tag lookup), added to `validate`'s
own - after `run_when_idle` returns, not inside it: the lint can load a
tokenizer the analysis failed to, from the Hub, waiting up to `LOAD_WAIT`
(30 s) for it, and inside the gate that held image jobs back;
`test_the_rewrite_is_linted_once_the_gate_is_open` holds it. What survives
all this is still a 4B model's reading - it misapplies sources and pads
prompts - which is why the page shows every version to be read before it is
used, and the comparison exists.

**The comparison is a grid, and its statistics are the caveat.** The
variants are one `POST /api/grid` with every other axis empty, the panel's
seed (or a drawn one, written back as grids do), and 4 seeds per version
by default, 8 offered. Its own Batch select offers 1 (default), 2 or 4:
`num_images` is Batch and grid `seeds` is the chosen total divided by Batch.
Each grid seed-axis step advances the base seed by Batch, so every version receives
the same 4 or 8 seeds and the image count stays versions times that total.
The preview gives the count, seeds per version, batch size and estimate before
anything runs, and the page opens that submission as the sheet when its
stream says done (`showJob(jobId, { fresh: true })`). A run can contain the
same prompts with original and yours swapped; `job:<id>` keeps each row's
meaning fixed. If Starred is ticked, it comes off first and the sheet's
note says so: nothing in the new comparison is starred yet, so
under the tick the sheet opened empty. The tick is removed rather than
bypassed because the gallery's filters are the sheet's, and Re-render sends
the tick with the search - a sheet of unstarred images under it would
re-render none of them. At Batch 1, for a 1 MP 28-step Krea [dev] image in fp8,
`verified.md` records 41.6-42.6 s (2026-09-27), so three versions on four
seeds take about 8.5 minutes. The 15.4 s once set beside it was an 8-step
image, and the parent commit 7a9c651 rendered at the same ~42 s through the
CLI, so neither points to a slowdown. The preview's estimate, from the
measured per-step time, is the one to go by.
The seeds are shared, so the rows are pairs, and the test that fits is a
paired sign test on the seeds where the two rows differ: four of four the
same way is p = 0.125, so four seeds cannot tell even a change on every seed
from chance; six of six is 0.031 and eight of eight 0.008 (seven of eight,
0.070, is not enough). An unpaired two-sided Fisher exact test on the hit
counts (0/4 against 4/4 is p = 0.029; at eight, 2/8 against 7/8 0.041) assumes
independent images, which shared seeds are not. Hence the line under the
Seeds select: 4 seeds is a first look - even a change on every seed could be
chance - and 8 can tell a consistent change from chance. The first wording,
"4 seeds only shows all-or-nothing differences", took the unpaired figure.
The fourth version, the user's own (`yours`), is what makes a comparison of
order dependable on a prose prompt: the rules never move one prose clause past
another, and on FLUX prose the critic's `reordered` rarely survived its check
(above). Scoring the images automatically (a VLM asked whether each element
appears) was left for later: nothing about its accuracy on these checkpoints
has been measured, and published judges are weakest exactly on text rendering
and style. Stars (`stats --job <id> --by variant`) and the run's `distinct`
count are the signals that exist.

**The page.** A retained rewrite shows **Changed from original**, with
character counts and an expandable exact diff. `promptChanges` aligns whole
words, whitespace and individual punctuation using a bounded common-subsequence
table after stripping common ends; it retains every character, including
escapes and emoji. Added and removed text is built with `textContent` in
`ins`/`del` elements, never HTML from the prompt. It compares against the
checked source even if the editor has since changed. An absent rewrite says
**No rewrite offered** alongside the existing set-aside explanations.
The diff stays closed by default - opened, it prints the whole rewrite a
second time directly above its own box - and a gist line after the
`<details>` (a closed one shows only its summary) gives the first three
changes, a few words of each, marked as the diff marks them ("Changed A
quiet → An empty · added film grain · moved masterpiece"). The 2026-10-04
review found the change hidden behind a click. The gist groups the diff's
runs first (`changeHunks`, `gistItems`): a word replaced is one change, not a
removal and an addition; edits only spaces and marks keep apart are one; and
words taken out in one place and put back in another, separators aside, are
one move. Run by run, a real rewrite's gist spent its three places on single
words ("Removed A · added An · removed quiet"). Each finding the check makes
on the rewrite says whether the prompt has one under the same rule and field
("also in your prompt") or not ("new"), and the prompt's findings the
rewrite no longer has follow as "No longer flagged: ..."; they are matched,
never collapsed, by rule, since the same rule can be another instance - on
Animagine the rewrite's order finding names other tags than the prompt's.
The rewrite's check asks the tag index nothing of its own (`prompt_check.lint`), so a
prompt finding that rests on it - the order, the style a known name reads as,
an artist's @ (`readsTagIndex`) - is never called fixed: gone from the rewrite,
it is listed as "Not flagged on the rewrite, whose check asks the tag index
nothing of its own - these may still apply". Review found "No longer flagged"
naming a Series-before-Character order the rewrite kept word for word. An
artist's @ is the exception (2026-10-05): the rewrite is checked with the
analysis's own artists, and where they answer for every artist the rewrite may
name (`rewrite_artists_checked`, above), a rewrite that adds the @ is called
fixed. The answer's parts list is captioned "How the critic
read it"; uncaptioned, it read as part of the issue under it.
`smoke-rewrite.json` checks the rendered diff, exact reconstruction,
escape-only changes, Unicode, markup as text, unchanged/absent rewrites, the
gist and the carried, no-longer-flagged and not-flagged marks;
`smoke-analysis-tags.json` the not-flagged one end to end on Animagine.

Check sits in the Prompt caption after Tags and before Expand
and sends what Generate would (`collect()`); its panel moves with the chips
(`placeTagValidation`), above the results on a two-column page - beside them
from 1800 px, docked by the stylesheet (below) - so its length cannot push
Generate down the sticky settings column. The move keeps
whatever had focus: `moveBefore` where the browser has it, elsewhere the
focused element and its selection put back by hand. The 2026-09-28 review
found the move re-appending the subtree, which blurred whoever was typing in
Your version or sitting on a chip when the window crossed 901 px, and the
next keystroke went to `<body>`. The panel describes one text - the prompt
and the negative prompt as `collect()` sent them, plus whether the panel's
CFG or True CFG applies that negative - because every offset in the answer
counts from that text and every Use and Apply is a whole replacement worked
out from it. Any change to either box marks it stale: typing, each
programmatic write through the one
`promptReplaced()` hook (an example, Reuse, Reset, a chip, an accepted
suggestion), `syncCfg()` greying or waking the negative, or changing its
effective activation while an open grid keeps the editor enabled for other
CFG values. The raw text and effective negative are captured before either
Check or the critic awaits a reply; later settings cannot relabel that answer.
Raw edits still invalidate a check when the enabled negative is inactive,
since some rules read what was typed and Apply must preserve edits. Adding or
removing a reference image also marks it stale (`setReferences`, or Reuse
through its `syncCfg()`): the check read those too, since klein's `guide.references`
counts them. Stale results stay visible, dimmed, with "Check again" and a
badge - "references changed since" where only the references moved, else
"prompt changed since", written only when its words change, not on every
keystroke; every Use, Apply and Run comparison goes dead, and a finding's
words are selected only while they are still where the check found them.
The first build compared the prompt alone, and "Apply to the negative
prompt" then overwrote a negative typed after the check; the 2026-09-28
review found the references still left out, so a third one left the panel
looking current, with no "image 3" note. The critic's answer is always
about the checked text: asking checks again first when either box or the
references moved on, a check of other text drops the answer, and an answer
that lands after an edit is shown dimmed and applies nothing. The critic's
hold starts at the click, that check included. The 2026-09-28 review found
it set after the check: Generate, Expand and Unload stayed live through it,
so a job started then got the critique a 409, and a Check pressed meanwhile
replaced the critic's check, so the critique was never sent, with no
message. Check and Check again are disabled while that check runs, and
re-enabled in a `finally`; a check replaced anyway says the critic was not
asked, while one a Reset or another checkpoint clears says nothing, since
the cleared panel shows nothing pending. Close pressed meanwhile hands focus
to the prompt: the disabled Check would drop it to `<body>`. A checkpoint
change clears the panel. Apply is offered on rule findings only - a rule's
`suggestion` is the field rewritten - while an encoder's suggestion (for
`((masterpiece:1.2))`, `Write "masterpiece" plainly; ...`) is advice and
shows as text. Use and Apply write through `replaceBoxText`, which selects
the box and inserts with `execCommand`, so Ctrl+Z brings back what was
typed; assigning `.value` emptied the undo history, and the re-check that
follows would have made the replacement the comparison's original, leaving
the typed prompt nowhere on the page. Use rewrite and Use reordered (the
critic's, and Check's own reordering) also put the prompt they replace into
Your version while that box is empty, with a note under it saying so: the
2026-10-04 review tried the rewrite and then compared it with what it had,
the obvious order, and found Compare offering only the new text and Run
comparison dead. A box holding words of the user's own is left alone, the
answer is still dropped by the re-check, and no new version label is added:
Your version already has the lifetime rules the replaced prompt needs. A
prompt finding's Apply does the same since 2026-10-05: applying a fix and
comparing it with what was there is then one click and a Run comparison
(U38's optional part, left out the day before as fixing one rule's complaint
rather than offering a version - which is exactly what wanted comparing). The
negative prompt has no Your version, and Ctrl+Z is its way back. The order
finding, whose suggestion is the Order section's reordering byte for byte,
says "Same as Use reordered above" instead of a second button for the one
action. Server offsets count
code points, as Python indexes a str (`PromptFinding` says so); the page's strings count
UTF-16 units, so `units()` converts before selecting - one emoji ahead of a
finding selected the wrong words, and the still-in-place guard, comparing
two JS slices, could not notice. The cut link's excerpt ("stops reading at
...") is cut in code points too: cut in UTF-16 units, as until the
2026-09-28 review, it could end in half an emoji wherever there was no space
to cut at, which on CJK is everywhere. An encoder row is drawn as a caution
only when the server says `pooled_only` (FLUX.1's CLIP-L); keyed on `role`,
it drew NetaYume's only encoder's cut as mild. The critic holds what Expand
holds - Generate and Run grid aria-disabled, Expand, Unload - and `startJob`
refuses while it runs. Its own button is aria-disabled while it runs
("Thinking…"), while Expand runs and while a job is queued or running, the
title saying why, and `askCritic` refuses each case; the 2026-09-28 review
found it disabled, which dropped a keyboard user's focus to `<body>` at the
click. An answer takes focus (`#critic-result`, tabindex=-1, a labelled
group) if focus is still on the button: nothing else announces it, since the
line under the button goes back to the cost. Check pressed on a two-column
page moves focus to the panel heading (tabindex=-1, its `scroll-margin-top`
clearing the sticky topbar) if focus is still on the button, because the
panel is about forty Tab stops away there; under 901 px it is three stops
away and focus stays. Check again hides itself once its check lands, so it
hands focus to the heading first. The comparison temporarily sent `num_images: 1`
because inheriting the panel's Batch 4 had made "4 seeds" sixteen. It now
uses its own Batch select and divides the grid seed count by that batch
size, preserving the total seeds per version. The main panel's Batch and
Repeat (the count once labelled Runs) still do not apply; that line and the
rest of the mechanics are a `?` badge after the comparison's Batch select,
its tip hung from the window (`floatTip`: placed inside the check, which
scrolls inside itself, it was cut off by the check's edge and lay under its
sticky head, and the `.compare-seeds .tip` rule that opened it upward is
gone), while the 4-or-8 caveat
stays in view beside it as the decision aid. It reads both comparison
selects directly for its preview and its run (a browser restoring form state sets a select
without a change event, and a copied variable had stayed at 4), never writes
the prompt box or the grid's variables, which are the user's and persisted,
and leaves out a version identical to another. Its Your version box is the
user's too: kept across checks (a check of other text still compares against
it), cleared with the rest of the panel, never persisted (autocomplete=off,
as are Seeds and comparison Batch, to discourage browser restoration).
`smoke-comparison-batch.json` checks all six Seeds/Batch pairings, preview
and request counts, select values changed without an event, and paired
output seeds, sheets and stale-run disabling. API tests cover paired saved
seeds, batch grouping and preview counts for all six pairings.
The Order section says "Not checked" under a documented order the rules did
not read (`order_checked`),
and the summary "nothing flagged · order not checked" rather than "nothing
to flag"; a prompt the rules placed only partly (`order_detail`: some tag
or clause kept every slot where that decides the order - the lookup failed
or stopped at its 64-tag cap, the index had no usable category for the tag,
a prose clause is never looked up) gets
"Partly checked: <why>" under the chain, naming those items, and the summary
adds "order partly checked" whatever else it counts. Until phase-4
re-verification the page keyed this on `categories_detail`, so an unknown
tag or a prose clause showed the order as checked beside a critic block
saying Check could not place every tag; the first fix then said it of a
lone tag or clause, and of an unknown tag no category could move, too. A negative prompt finding's Apply is dead while the negative box is
greyed, with a line saying what CFG would wake it: `collect()` sends nothing
from a greyed box, so the write changed nothing the re-check could see and
the same finding came back with the same button. The critic's `set_aside`
lines are shown under its answer, and the `dropped` line says what dropping
means - a citation that named nothing it was given, or an empty issue; an
unfound quote only loses its quote.

**Beside the results from 1800 px.** The 2026-10-04 review measured the
check at QHD: opened above the viewer it was 2,161 px wide and 669 px tall
after Check, 1,136 after the critic, and the image just made went from wholly
in view to 59% and then 61 px of it; at 2048x1040 at 125% none of it was left
after the answer - while about 1,270 px beside the image stood empty. So from
1800 px the stylesheet docks the open check in a column of its own beside the
viewer (`--dock-w`, 520-760 px) - except with Gallery beside, where it takes
the gallery's column instead (The gallery beside the viewer): `.stage` becomes
a grid while `#prompt-analysis` is open in its wide host (`:has()`), the chips
across both columns in the first row, then the viewer and its note on the left
and the check on the right across their rows, scrolling inside itself - and
everything from the gallery head on spans both columns in the rows after, in
document order. The check takes the height of the left column's rows rather
than setting it (`contain: size`, stretched), at most the window's height under
the topbar and the chips (`--chips-h`, the tray's height, which tags.js's
`chipsResized` writes on `.stage` from a ResizeObserver); the viewer, while the check is
docked, is at least 36rem or that height, whichever is less, so beside an
empty viewer the column is still one to read. (The minimum sat on the column
first, and since the column spans the note's `1fr` row, Chromium grew that
row by what the minimum lacked over the empty viewer's 380 px even beside a
917 px image.) The integration review found the first version's
two faults with it. It was sticky under the topbar, and a sticky grid item
is held by the whole stage, not by its rows: as the page scrolled it slid
down over the gallery head, the cards and an open comparison sheet, hiding
the search box, Starred and Sheet and taking their clicks. And it was
sized by its own content up to the window's height, so the critic's answer
always set the rows' height and put the gallery head past the fold at QHD,
beside an empty viewer as much as beside an image (1,313 px of 1,305); it
is now at 708 px beside the empty viewer and, beside a 1024x1024 image, at
1,049 with the check open or closed (866 of 1,040 at 2048x1040 at 125%).
The note's row is `1fr`, so a check taller than the viewer would have its
excess below the viewer rather than above it; with no
chips there is no row for them, since an empty one kept its gap and opening
the check moved the image down by it. The check's head - the title, the
stale badge, Check again, Close - sticks at the top of the column's own
scroll: the critic's answer takes focus and scrolls the column down to it,
and the head went with it, so a prompt edited after the answer left only
the light dimming to say the check no longer described it. It sits at the
scroller's very top (`top: 0`), the column's top padding moved into its
own: stuck 10 px up into a padding that stayed the column's, a sliver of
the findings scrolled past above it at 125%, their text cut in half. Close
is absolute in its top right corner; in the head's wrapping row it fell to a
line of its own under a long summary in the 614 px column. And beside an
idle viewer - nothing made this session - the 36rem minimum made the one
line an empty 576 px box with its sentence in the middle and dropped the
gallery head 535 px (U5 again, on a fresh session): there the viewer stays
its line, everything from the gallery head on goes into the first column
under it, and the check is as tall as what it holds up to the window under the
topbar and the chips (`contain: none`), spanning the rows down to the
gallery's, which is `1fr` and takes whatever the check needs past the rest. A
long run's strip stays in the ordinary layout: in the first column a
comparison's sheet would have been half its width. Check, focusing
the heading, puts the column back at its top by hand, since a stuck heading
is always in view and `focus()` no longer did. The image stays where it was
after Check, after the answer and for a Generate pressed meanwhile, chips or
none, and Generate does not move.

**The chips stay across both columns** (2026-10-05). They had moved from the
full width into the left column, where they wrapped onto more lines, and the
image moved down by them as the check opened: with the tray's own label,
`+ tag` and pinned hint, none for six tags at QHD (one line of chips either
way), 69 px for 40 tags at QHD and 65 px for 32 at 2048x1040 at 125%, where a
prompt of 40 tags - well past CLIP's 75 tokens - took the image's bottom 8 px
past the fold (U5's leftover). Keeping them across both columns was left out
then, because the check would start under them and its bottom, sized for a
column under the topbar, would sit below the fold by their height beside an
idle viewer. So the cap now leaves the chips out (`--chips-h`, above), and the
chips are as wide as with the check closed, the check starting under them: the
image does not move at all with 6, 32 or 40 tags at either size, and at QHD with
40 tags and the critic's answer the idle check ends at 1,295 of 1,305 px, where
it would have ended about 160 px past the fold. With `#recent` docked the chips
span all three columns. It is CSS but for that height: the DOM order - chips,
then the check, each in a wide host of its own (below) - is unchanged, the focus
rules still hold (the heading after Check, the answer after the critic:
`focus()` scrolls the panel's own box), and without `:has()` the check stays
above the viewer, as below 1800 px.

**Not built, by decision: a `.many` batch out of the dock, and a strip
minimum of its own.** The cost of the dock is a `.many` batch's slots at about
160 px while the check is open; the review's optional
`:not(:has(> #viewer.many))` would take such a batch out of the dock, flipping
the layout as the batch starts and bringing back the push-down U5 removed, and
it matters only between 1800 and 2000 px, below the capped strip. Beside a long
run's strip the docked check keeps the viewer's 36rem minimum too, which
outranks the strip's own `min(560px, 45vh)` cap. The way round it is the idle
layout, the gallery in the first column beside the check, and there a
comparison's sheet would open at half its width; the strip is capped already,
so the minimum stays.

**Tab goes as the dock is drawn** (2026-10-05 area review). In the DOM order the
check comes before the viewer, so docked, Tab went from the panel through the
check on the right before reaching the image just made on the left: at
2560x1305 the image was the 41st stop from the prompt, against the 26th with
the check closed. The docked grid first took `reading-flow: grid-rows`, but the
check starts in the image's row, so by rows the note under the image (a pass's
"Refined from #N") came after every stop in the check. It now reads in source
order (`reading-flow: source-order`, Chromium's; elsewhere the DOM order stands)
with the check's host after the stage's own items (`reading-order: 1` on
`#tag-validation-wide`, 2 on `.library`), so Tab goes: the chips, the image and
its tools, `#viewer-note`, `#recent` where it docks between, the check, the
gallery. Tag checkpoints needed more. The chips and the check shared one
`display: contents` host, and a reading-flow container orders such a host as one
item, where its first child is drawn (the chips' row), with its children in DOM
order, so no reading order could take the image ahead of the check. On a
two-column page each now has a host of its own, the chips' first:
`#tag-chips-wide`, then `#tag-validation-wide`, which keeps the id the check's
selectors use; `placeTagValidation` moves each to its own host, and both back
under the prompt below 901 px, chips first. The rules that read "no chips" from
the shared host (`#tag-validation[hidden] + #prompt-analysis`) now read the
chips' host, `:has(> #tag-chips-wide > #tag-validation[hidden])`; the old form
can no longer match, so CSS written against it fails silently. The reading
order is the accessibility tree's too, and a reading-flow container reads what
is out of its flow after all its items: an unseen "Results" heading before
`#viewer` was read after the gallery, which is why the viewer is a named region
instead (Headings and landmarks) and nothing absolute, fixed or `.sr-only` is a
child of `.stage`. With Gallery beside the open check is a fixed column at the
window's right, out of the stage's flow, and the stage takes `reading-flow:
flex-visual`, which visits a fixed box after the flow's items. Findings with
nothing to select are no longer tab stops (Show, below), which shortens the path
through the check too: on FLUX the image is now the first stop after the Grid
search summary, then its tools, then the check. In the
column the sections stack - side by side, Order had 356 px - and full width
above the results, from 1400 px, Encoders keeps a 260-360 px track and Order
the rest, where `auto-fit` had made seven 286 px tracks of which two were
used. Text is capped at 120 characters' width (`.analysis-list`,
`.analysis-text`, `.critic-diff`): at QHD a finding's words ended 800-1,400 px
from the badge saying what they rest on, and a version ran 400 characters a
line. A guide or author cite under 80 characters is in the line, after the
claim, as `<q class="analysis-cite">`; it was only the badge's tooltip, which
nothing but a hover shows. The badge keeps its title and link. The critic's
Rewrite and Reordered sit side by side where both fit at 60 characters a
line: full width above the results from 1400 px; in the docked column,
narrower than two such lines, they stack.
Badges are sentence case at the notes' size; Ask the critic and Run comparison
are lit while they would act (never while `disabled` or `aria-disabled`),
since as ghost buttons enabled differed from disabled only by opacity; Close
is a 28 px target. The reordering under Order is shown whole, scrolling past
six lines - clamped at three, it cut off what Use reordered writes - and a
version's tick under Compare outlines the box holding its text while it is
pointed at or focused (`data-version`).

The panel's own messages - a finding's words no longer where the check found
them, an answer set aside or never asked for - are written to a status slot
under Ask the critic (`analysisMessage`: unhidden before it is written, never
rewritten with the same words, and `#error` where the critic's section is
hidden), between the findings and the answer they are about; in `#error` they
sat under Generate, a column away. The slot stays at full strength while the
rest of a stale panel is dimmed - what it says is mostly why the panel went
stale, and its amber at 0.55 opacity fell under 4.5:1 - so the dimming is on
its siblings, since a child cannot undo an ancestor's opacity. Those are
dimmed lightly, to 0.8 with the quiet text raised from `--muted` to
`--text-3` under it (about 5:1 on the panel), not to 0.55, where the notes
fell under 3:1: stale results are still the last answer and must still read,
and the amber badge, Check again and the dead buttons say they are stale
(U89). The Refine dialog's own check goes stale by the same rule, one
declaration for both; its lines had kept 0.55, about 2.5:1. The refusals
("Enter a prompt first.", "Wait for ...") stay in `#error`. A dead Use or
Apply says why in its title, as Run comparison does, rather than keeping the
offer it refuses. They stay `disabled`, not `aria-disabled` as Ask the critic
and Refine are: they are dead as a set, a dozen at a time, while the stale
badge and Check again - at full strength in the head - say why to every
reader, and what makes them stale is an edit somewhere else - a prompt box,
the references, the CFG - so focus is seldom on one when it dies. A critique
that fails leaves a row in the critic's block in the
server's words ("The critic did not answer: ...") and takes focus like an
answer. `#error` gets that sentence only where the critic's block is not on
show (the panel closed while the critic thought): it had it every time at
first, and the one failure was then read out twice, from two slots a column
apart from 1800 px. Nothing generic follows the reason: the server's messages say
what to do where anything can be done (shorten the prompt, install a
library), and "Ask again", added to every row at first, contradicted a
refusal that asking again only repeats. The failure is `critiqueFailure`,
never a `critiqueResult`, which Use and Apply read as an answer, and is
cleared by the next Ask, a check of other text and a cleared panel. The route
sends `CriticError`'s and `CriticUnavailable`'s messages without the class
name, which led every one of them; the log line keeps it. A critique the
server never answered is the server's failure, not the critic's: it reads "The
critic did not answer: the server did not answer." (`critiqueOffline`), and
the status poll that answers again takes it back (`takeBackCriticOffline`), as
every offline message is; it had stayed in the critic's block for good as "No
answer from the Local Diffusion server — is it still running?.".
`critiqueFailureText` drops any closing stop, question mark or exclamation
mark of the reason's own, so a question no longer ends in "?.".

**Ask the critic is held for what the server would refuse** (2026-10-05).
Expand and Unload were held while the server did not answer, and Unload while
the image model loaded, but the critic was left live for both: a press got the
offline sentence in its block for good, or the route's 409 about the load
(`api_prompt.refuse_critique_now`; Expand, by contrast, waits for the load). It is now held (`criticHold`) while the server does not answer and
while the image model loads, then for a job (this page's or another tab's)
and for Expand, aria-disabled with the reason as its title and on
`#critic-cost`; `goOffline`, the poll that answers again and a change of the
model's state each sync it. A press says why: offline, the offline words
(`OFFLINE_MESSAGE`), which the poll that answers takes back; during the load,
a note taken back when the load ends (`criticLoadNote`); during this page's
job, a job note gone with the job (`showJobNote`); during another tab's job,
said until it ends (`elsewhereNote`).
While the critic runs, the seconds since the click count beside the line
(aria-hidden, outside its live region, so it is not read out every second),
in the line's run of text, at the end of its last line (`.critic-words`,
2026-10-05): a flex item of its own, the clock took a line to itself under a
first use's long sentence at QHD. It is kept whole and joined to the word
before it by a word joiner (U+2060) in its `::before` - its text stays
"· 12 s" - so a sentence that fills its line gives its last word to the
clock's line rather than leave the clock alone there;
and a first use's line says the download can take a few minutes: with nothing
moving, slow could not be told from stuck while every generation button was
held. The line was one polite live region, and it spoke at every job's start
and end about a button nobody had pressed; it is now a live half that answers
the press (`#critic-progress`: Downloading…, Thinking…) and a quiet half for
the cost and what holds the button - no server, the load, a job
(`#critic-cost`, 2026-10-05). After a
first answer it said a 4.5 GiB download was still to come until the next poll;
any answer now marks the critic ready at once (`promptCritiqueReady`), one set
aside because the prompt changed meanwhile included. Generate's, Run grid's and
Run comparison's titles say the critic holds them while it does
(`syncSubmitButtons`, whose reasons are `submitHold()`'s, and
`syncCompareRun`), and so does the Refine dialog's Refine (`syncRefineRun`,
which checks `submitHold()` ahead of its own reasons; a press says it in the
dialog as a sentence): it had looked live while the critic or Expand ran,
and only a press said why. A finding clicked, the cut link and a critic quote scroll the box to the
words they select (`scrollToSelection`, a hidden copy of the box laid out as
it is): `setSelectionRange` scrolls nothing in a box that has focus, and after
typing a long prompt the box sits at its end while the words are near its
start. The Order section's "Not checked" says the app checks word order only
in a tag checkpoint's comma-separated tags - of the app, not of "Check" in the
third person, and of a tag checkpoint's tags, since a comma tag prompt on FLUX
is not placed and a prose one on Animagine is. The comparison's preview counts
versions, the panel's word, not variants. `smoke-analysis.json` checks the
dock at 2560x1305 and 2048x1040 at 125%, and the stacking at 1400x900;
`smoke-analysis-tags.json` the dock on Animagine, where the image does not move
with 6, 32 or 40 tags at either size, the stacked check's cap under the chips at
1400x900 and 1280x720 and its floor at 1280x600, and the check in the gallery's
column with Gallery beside, over an idle viewer and beside an image.

**Above the results the check is capped** (2026-10-05). Below 1800 px, on a
two-column page, the check stacks over the viewer, and with the critic's answer
in it it ran past a thousand pixels and put the image just made off the screen -
U5's own observation, its optional 55vh cap left out the day before. It is
capped there and scrolls inside itself, its head sticky at the scroller's top
with the panel's top padding moved into the head's, as in the dock, where the
dock's own cap replaces this one. The cap is 55vh less a tag checkpoint's chips
above it (`--chips-h`), never under `clamp(8rem, 15vh, 12rem)`: at a plain 55vh,
40 tags' chips and the answer put the image's top at 817 of 900 px at 1400x900,
83 px of it in view, and it is at 622 now. A fixed 12rem floor overrode the cap
beside smoke-analysis-tags' 40 tags (chips 227 px), starting the image just made
at 0.76 of a 1280x720 window (175 px of it showing) and 0.91 of a 1280x600 one;
at 15% of the window it starts at 0.73 (197 px) and 0.80, and at 1280x600 the
floor itself sets the cap - the check is 8rem, a head and two or three lines of
findings, which scroll under it. Windows 1280 px tall and more keep 12rem. In the
panel under 901 px the check is as long as it is.

**Escape closes it** as Close does (`escapeAnalysis`, app.js's keydown), once
nothing nearer took the key: the Examples menu, the Tags helper, a suggestion
list or a chip editor (`defaultPrevented`), the lightbox and the Refine dialog
keep theirs, and a search box with text, or a composition, keeps its native
Escape. Focus inside the check (or on `<body>`) goes to Check - to the prompt
while Check is held - as it would otherwise fall to `<body>`. Focus elsewhere
stays, since the caret in a prompt being typed in should not jump to a button.

**A selection says where it is.** A finding, the cut link or a critic quote
outlines the box it selected in for 600 ms (`flashBox`, `.box-flash`;
2026-10-05, U50's option), which confirms a click whose words may be a column
away. Reduced motion takes away the flash, not the selection or the scroll;
`smoke-analysis.json` tests both through browser.mjs's `media` step, since the
flash expectation alone failed on a host with animation effects off. In klein's
references finding each "image N" is a link to that chip (`findingText`,
`referenceLink`, the words unchanged; U62's option): a press focuses the chip's
× (`revealInPanel` brings it into view) and flashes the chip, since a pointer
press leaves no focus ring. Past the last chip the status slot says image N is
not attached now; the link stays live while the check is stale. A later press
that finds its chip takes back only that sentence of its own
(`notAttachedSaid`): clearing the slot outright wiped sentences the user still
needed, such as why the critic's answer cannot be applied.

**Show, not a focusable finding** (2026-10-05 area review). Every finding and
critic issue was a tab stop (an `li` with tabIndex 0). One with words to select
read as a list item titled "Select these words", whose Enter and Space did what
no role said they would; one with nothing to select was a stop that did nothing.
A finding or issue with a span now has a Show button after its words
(`makeSelectable`, `.analysis-show`; the Refine dialog's check's word for the
same act), described by what the item says through `aria-describedby` to
`.analysis-said` - its severity in words and its message, or the issue's problem
and quote - and titled with the box it selects in: the prompt, the negative
prompt, or, for the rewrite's own findings, the rewrite. The item is no stop; a
click elsewhere on it still selects, for the pointer. Show stays live while the
panel is stale: it selects only while the words are still where the check found
them, and says so otherwise, as the click did. A negative-prompt finding's Show
pressed while CFG greyed the box did nothing at all, since a greyed box takes no
focus or selection; it is now aria-disabled with the reason as its title while
the box is greyed (`syncNegativeShow`, from `syncNegativeApply`) - a held
button by the page's rule rather than the dead Use/Apply exception, since it
comes back with the box and not with a Check - and any selection in a greyed box
(Show, a click on the finding, an encoder's cut link) says why in
`#analysis-status` (`selectIn`), with what would wake it. Show is a word in a
sentence, an inline target: from 1800 px a tiny button is 30 px tall, and inline
it stretched its line, so there it is the line's height (Design tokens). The
smoke steps that asked an encoder finding or the order finding for no button at
all now ask for no Apply (`[data-writes]`), since the cut's words and the order
finding's have a Show. The comparison's always-run Original box carries the
`aria-description` "Always runs: the other versions are compared against it".

**Open questions the build turned up**, for decisions rather than code:

- The tag autocomplete writes Danbooru names with A1111 escapes -
  `tags.display_tag` turns `(` into `\(` - to match the authors' own examples
  (Animagine's has them). Diffusers never unescapes, so CLIP reads
  `firefly \ ( honkai : star rail \ )`, ten tokens where the name is eight.
  The analyser does not flag what the app itself writes; whether the chips
  should write bare parentheses is undecided.
- Resolved 2026-09-28: schnell's guide said its T5 stops at 256 tokens,
  "about 180 words" - BFL's reference
  (`load_t5(..., max_length=256 if name == "flux-schnell" else 512)` in
  <https://github.com/black-forest-labs/flux/blob/main/src/flux/cli.py>) -
  while this app passes no `max_sequence_length`, so `FluxPipeline` encodes
  512 on schnell too and the encoder report says 511. The guide was reworded
  rather than the pipeline changed: "schnell's T5 encoder reads up to 512
  tokens in this app", and BFL's 256 is named as the reference in an
  `unsourced` sentence the critic is not given; the word rule stays out.
- Qwen-Image's guide says to sample "with True CFG 4 and a negative prompt -
  a single space if there is nothing to remove", but `generation.js`'s
  `collect()` sends `els.negative.value.trim() || null`, so a single space is
  sent as no negative prompt and no True CFG pass runs. The True CFG hint
  already says "a word will do".
- Resolved 2026-09-27: Krea [dev]'s "It answers best to photo language: a
  mood, a lens, a light, a film stock, a colour grade" and "Krea calls it
  opinionated - give it room rather than over-specifying" had no literal
  source in Krea's README, Krea's release post or BFL's, and the critic, held
  to BFL's pages, told users the opposite on the same page. They are out of
  the prose; what BFL's building page says about camera settings replaced
  them.
- Anima's guide prose leaves out the card's tag order - `[quality/meta/year/
  safety tags] [1girl/1boy/1other etc] [character] [series] [artist]
  [general tags]` - and its recommended prefix `masterpiece, best quality,
  score_7, safe, `, so Anima has no documented order in the app: the critic
  sees both as excerpts (A1, A3) and the page shows the order as the
  authors', unchecked. Added to the prose, they would be checkable.
  (`guide.artist_prefix`, which never fired as first shipped, now does: the
  route asks the tag index for Anima's artist categories.)
- NewBie's guide prose names a block's fields "name, gender, appearance, ...",
  but the authors' XML example and captioning instruction spell the first one
  `<n>`. The XML rule accepts both; whether the model reads `<name>` as it
  reads `<n>` has not been tried.

## The page scripts share one classic scope

`web/app.js` reached 3,200 lines - every section of the page in one
file - and was split on 2026-09-10 along its own section comments into
ten scripts `index.html` loads in order: `state` (the element table, every
`let` the page shares, the fetch helper), `sizes`, `status`, `generation`,
`panel`, `gallery`, `sheet`, `lightbox`, `tags` and last `app`, the wiring
and the boot; the prompt analyser added an eleventh, `analysis` (the check,
the critic and the comparison), between `tags` and `app` on 2026-09-27, and
the Refine dialog a twelfth, `refine`, between `lightbox` and `tags` on
2026-10-04. They stay *classic* scripts rather than ES modules on purpose.
Classic scripts share one global scope: a top-level `let`, `const` or
function in any of them is visible to all the others, so nothing needed an
export or an import, and - the reason that matters - a smoke step's `eval`
(`tools/browser.mjs`) still reaches `els`, `galleryRecords`, `startJob`,
`analysisResult` and the rest by name, which a module's scope would hide. The
contract that makes the order safe is that every script except `app.js` only *declares*:
a function in `sizes.js` may call one from `analysis.js` because every call
happens after all scripts have loaded, from a handler or from `app.js`'s
boot. A file that ran something at load would have to sit after everything
it touches; keep the wiring in `app.js`. The lightbox's pointer handlers
were once the exception, wired beside the code they belong to; the
2026-10-04 review's lightbox work (its comparison, strips and image
failures) grew them past "touching nothing outside it", and they are now
named functions in `lightbox.js` (`lbImageFailed`, `lbWheel`,
`lbPointerDown` and the rest) that `app.js` wires with every other handler.
`viewer.js` now holds the results viewer, and `lightbox.js` the shared
facts list (`factsList`) the viewer's column uses too. (2026-10-11: the
facts list and settings line moved to `facts.js`, and the comparison's
controls, divider keys and pass stepping to `lightbox-compare.js`;
`lightbox.js` keeps the view, layout, walk and gestures. The same day the
Examples menu moved out of `app.js` into `prompt-examples.js` and ERNIE's
Expand into `prompt-expand.js`, both declarations only, loaded after
`tag-chips.js`; their handlers stay wired in `app.js`, which is again only
the wiring and the boot, the last of 41 scripts.)

The same pass moved `tests/test_api.py` (1,300 lines) into `tests/fakes.py`,
the fixtures in `conftest.py`, `tests/api_helpers.py` and one `test_api*.py`
per area, and pulled `backend/search.py`, `backend/guides.py`,
`backend/backends/loading.py` and `backend/backends/schedule.py` out of the
three largest backend modules. `loading` takes `settings` as a parameter
rather than importing it: the tests replace the backend's `settings`, and a
second module-level copy would have gone on reading the real one.

The 2026-10-07 source audit split the larger feature files again, bringing
that list to 26 scripts. Submission, grid preview, streaming and job results
are separate; so are panel/history/viewer, gallery/actions/reuse, sheet
modelling/rendering/export, tag suggestions/token counts/chips, prompt
check/critic/comparison and the Refine dialog's prompt controls. Each original
group stays contiguous in `index.html`, with its executable declarations unchanged,
and the shared bindings and `app.js` boot contract remain the same. The eight
stylesheet links are consecutive sections of the old `style.css`, in its
original order, preserving the cascade exactly. Tokens and base layout remain
in `style.css`; the component rules live under `web/styles/`.

The same audit split the critic into runtime installation, process ownership,
context construction, pure reordering checks, answer validation and the coordinator; deterministic
prompt checks now share `prompt_text.py` and `prompt_order.py`. Public callers
still use `critic.py` and `prompt_rules.py`; tests patch the module that owns
the implementation, rather than an imported alias. Gallery and job routes
moved from `main.py` into router factories in `api_gallery.py` and
`api_jobs.py`. They receive callbacks for the app-owned worker and checkpoint
scope so they read current dependencies rather than capturing a second copy.
The generated OpenAPI contract is unchanged. Test modules were split by
responsibility too, retaining their test bodies and parameter cases. The full
inventory, reasons for keeping other long files together and verification are
in [the source audit](source-audit.md).

## uv.lock's hand-pinned torch hashes

**`uv.lock` carries hashes for the PyTorch wheels by hand.** The
`download.pytorch.org` index publishes none for torch 2.13's three wheels or
the Windows torchvision wheel, so `uv lock` cannot add them; the four were
downloaded, hashed and checked file-for-file (`RECORD`) against the Windows
and WSL venvs on 2026-09-08. `uv lock --check` and `--dry-run` accept the
edited entries; a `uv lock --upgrade` that moves torch will drop them.

## Verified on the card, 2026-09-08

Verified on the card 2026-09-08 beyond the above: a real Ctrl-C (SIGINT to
uvicorn's handler) 12 s into a load exits in 3 s with the lifespan run;
mid-sweep with a stream attached, `done` (cancelled) reaches the stream
0.2 s after the signal and the process is gone 3 s later; Pause is
acknowledged at the cell boundary, where a paused 512x512 grid held
2,046 MiB of the card (whole card, desktop included; a 1 MP fp8 denoise
reads 14,950 MiB); `reindex --from` over the real gallery
reproduces every hash and true CFG. Not verified: `LOCAL_DIFFUSION_DEVICE=cuda:1` (one
card here).
