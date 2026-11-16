# Plan: a local prompt-writing assistant

`PA#18`, commissioned 2026-11-16. Matt requested all three tasks: develop a
rough idea conversationally, improve an existing prompt, and turn a description
into candidate Danbooru tags. He authorized implementation up to GPU testing
and explicitly said not to run that testing yet. CPU tests and fake-browser
checks establish source behavior only.

## Scope

One optional **Write a prompt…** dialog holds Create, Improve and Find tags.
The selected image checkpoint's guide accompanies each question. Create may
suggest creative details; Improve is asked to preserve the source's subjects,
constraints, exclusions and exact lettering unless the instruction changes them.
Find tags proposes candidates and checks their names against the existing tag
index. These are suggestions, never proof that a prompt will make a better image.

Answers include drafts and notes, or candidate tags with authoritative lookup
verdicts. The user explicitly applies a draft or inserts selected usable tags.
Application uses the prompt editor's existing undo path and refuses an answer
whose source prompt or checkpoint has since changed. Nothing generates an image
or applies a draft automatically.

Conversation history lives in the open page's memory, with an explicit reset.
It is bounded rather than silently truncating away earlier instructions. It is
not saved in the gallery, browser storage, generation metadata or server logs.
Full text goes to the local llama.cpp child only. Tag verification sends the
candidate names to Danbooru, as the existing tag tools do; it is not fully offline.
An unavailable index leaves candidates unchecked, never verified by the LLM's
confidence. A model explanation is labelled separately from an index verdict.

## Runtime and admission

The first version reuses the critic's pinned Qwen3-4B Q8_0 and llama.cpp b11206,
including its verified downloads, loopback authentication, listener ownership
check, context sizing and process-tree cleanup. The existing
`AURADREAM_CRITIC_MODEL` and `AURADREAM_CRITIC_SERVER` overrides still select a
custom pair; this work does not download a newer candidate or change the pin.

Preparation runs outside the job gate. The actual answer runs through
`jobs.run_when_idle`, after another check for an image load or queued/running
job. It never calls `ensure_loaded`. The process must stop before releasing the
gate. The critic and assistant share that lifecycle, so shutdown and Unload
retain their existing owner. Requests remain model-dependent readers under
`hold_model_for_request`, including its revision-header check, and signed-out
requests remain closed by the existing passcode layer.

Every request has a ticket. Stop is scoped to that ticket, including a stop that
arrives before its request; closing the dialog requests Stop and discards late
answers. A stopped or superseded answer cannot overwrite a later conversation.

`POST /api/prompt/assist` takes the mode, instruction, source prompt, optional
negative prompt, bounded user/assistant history and ticket. It returns the
checkpoint and LLM labels, elapsed time, a message, drafts, tag candidates and
warnings. `POST /api/prompt/assist/{ticket}/stop` stops only that request.

## Validation before the card

- CPU contracts: bounds, checkpoint guidance, hostile dialogue treated as data,
  structured output refusal, authoritative tag verdicts and offline fallback.
- HTTP orchestration: cold requests, preparation outside the gate, a load/job
  beginning during preparation, shared gate ownership, ticket-scoped Stop and
  stop-before-start, revision and passcode admission, cleanup on failure.
- Real browser against the fake server: all three modes, a follow-up, explicit
  application and undo, tags, stale editor/model changes, errors and retry,
  Close/Stop/reopen, desktop and phone dialog layout.
- Repository checks: ruff, pinned Pyright, classic-script/shared-scope checks,
  documentation maps and relevant regression tests.

These checks use no weights, GPU inference or real gallery. First-use download,
real latency, memory peaks, semantic fidelity and real-process cancellation
remain unmeasured for this feature.

## GPU test handoff

`tools/prompt_assistant_pack.py` prints fixed cases and a future protocol,
writes a blank receipt only when requested, and audits a manually filled one.
It imports no backend and performs no network, process or card action. Printing
or checking a receipt never runs the test or grants qualification.

```powershell
uv run python tools/prompt_assistant_pack.py plan
uv run python tools/prompt_assistant_pack.py receipt --output <new-receipt.json>
uv run python tools/prompt_assistant_pack.py check <completed-receipt.json>
```

For a later authorized Windows session, use a new directory for
`AURADREAM_DB_PATH` and `AURADREAM_OUTPUTS_DIR`, explicitly set
`AURADREAM_PRELOAD=false`, and launch `uv run python -m backend.main --defaults`
in that process. Keep the shared Hugging Face cache for the cached baseline;
do not copy saved gallery choices into the scratch directory. Confirm the app
is unloaded before asking the assistant. The protocol's image-checkpoint
selection is guidance context and does not require loading image weights.

Start with the pinned model and a scratch gallery, no loaded image weights,
preload off, and a quiet machine. Use the existing `tools/card_guard.py` with
explicit preflight RAM and busy-card limits; retain whole-card, shared-memory,
process-RAM and commit evidence. Do not permit a shared-memory spill or relax
the guard. The critic's recorded roughly 5.5 GiB additional VRAM is a reference
for planning, not an assistant measurement or a RAM guarantee.

The later session should cover the fixed Create/Improve/Find tags cases and
follow-ups, cold and warm requests, first-use progress, all process cleanup,
Stop during startup and decoding, conflict refusals, and a bounded image job
after the LLM has released the card. Review retained subjects, exact lettering,
exclusions, useful variation and valid tag names by hand. Record failures as
well as successes. Test other LLMs only in separately scoped runs after the
baseline; this implementation does not qualify Qwen3.5 or change defaults.

**GPU testing has not been run or authorized by this implementation request.**

## Implementation and local results

Built 2026-11-16:

- `schemas_assistant.py` bounds the request, eight-message history, 12,000
  total input characters and structured response. `prompt_assistant.py` builds
  checkpoint-grounded questions, sizes the context, rejects malformed/trailing
  or duplicate-key JSON, adds omission notes and verifies tag candidates.
- `PromptCritic.assist` reuses the pinned authenticated transient server and
  counts the actual rendered question before inference. Cancellation watches
  its captured process only and joins that watcher before releasing the gate.
  `api_prompt_assistant.py` owns tickets, tombstones, preparation and post-answer
  lookup. A cancelled preparation may finish its daemon download but cannot
  continue into inference. Admitted inference drains through process cleanup.
  Post-answer tag lookup holds keep-awake on its own daemon worker until that
  worker ends, including after a stopped HTTP waiter.
- `prompt-assistant.js` owns the three tasks, conversation, undoable apply,
  source/revision holds and late-reply ownership. The follow-up box and Ask
  stay in place while replies scroll. Alias insertion keeps Anima's artist
  prefix and refuses an artist-prefixed alias resolving to a general tag.
- `prompt_assistant_pack.py` supplies the fixed future GPU protocol and blank
  receipt, including explicit idle-card and RAM admission, manual quality
  judgments, optional first-use downloads and separately scoped image tests.

Local checks (all without model weights or GPU inference):

- Full CPU suite: `uv run --no-sync --group dev pytest -n 4 --durations=15
  --timeout=600 -q` — **10,334 passed, 325 skipped**, exit 0, 305.42 seconds.
  The skips include uncached real tokenizers and platform-specific cases.
- Ruff: `uv run --no-sync --group dev ruff check .` — exit 0.
- Pinned Pyright: `uv run --no-sync --group dev python tools/typecheck.py` —
  214 sources, zero errors/warnings, exit 0.
- Final source/smoke-map, classic-script, section-bound, accessibility-audit,
  browser-output, smoke-runner and font checks — 342 passed, exit 0. The page
  now has 80 classic scripts. The Check-dismissal fixture includes the new
  assistant state; the prompt-analyser rules are split by topic under their
  existing parent anchor.

- Browser: **14 distinct focused smokes passed**. The final assistant smoke
  passes all 163 steps in 12.86 seconds, including all three real fake-API
  tasks, conversation, explicit Apply/native Undo, unknown/unchecked tags,
  input and Ask geometry through a pending reply, retained typed follow-up,
  Stop/Close/reopen, stale editor/offline/model holds, old-stop ownership,
  keyboard reachability, desktop/narrow/phone and accessibility audits.
- The existing `smoke-analysis`, `smoke-snippets`, `smoke-snippet-portability`,
  `smoke-library-write-recovery`, `smoke-saved-name-unicode`,
  `smoke-generation-presets`, `smoke-setup-list-recovery`,
  `smoke-setup-portability`, `smoke-phone-dialogs`, `smoke-phone-create`,
  `smoke-model-switch`, `smoke-execution-selector` and `smoke-execution-reuse`
  passed. The menu fixtures explicitly reveal their target inside the existing
  capped menu before a native click and stop on a failed hit-test prerequisite.
- Final static checks after the compose-area change: 136 passed. The final
  desktop draft, tag and phone screenshots were visually inspected. The fake
  servers, browser processes/profiles and owned ports were cleaned up.
- After the final tag-worker keep-awake scope, the assistant API/context/runtime
  and keep-awake-holder checks passed (93 passed, one platform skip), Ruff
  passed, and the documentation/section/smoke-runner checks passed (199 tests).

Retained browser evidence is under the system temporary directory:
`auradream-smoke-artifacts-ieng9taj/summary.json` (final assistant; screenshots
in `smoke-prompt-assistant-pntkee20/`),
`auradream-smoke-artifacts-9ncyer0i/summary.json` (eight corrected menu fixtures),
`auradream-smoke-artifacts-dtv7rhl6/summary.json` (the other model/phone/execution
passes; its original menu failures were diagnosed and corrected), and
`auradream-smoke-artifacts-qhx50zs0/summary.json` (analysis).

The remaining `PA#18` work is GPU qualification and Matt's review of generated
prompts. No weights were downloaded, real LLM/image inference run, or assistant
GPU measurements/qualification claimed by this implementation session.

## PR integration checks

PR #84's first hosted run (`43834820760`, 2026-11-16) passed Linux and all
five Windows CPU shards. Browser shard 5 found an old five-action phone-menu
expectation and a keyboard fixture race: the first short-viewport field was
sometimes obscured, while its late-content state passed. The phone-menu
expectation now includes Write a prompt… and retains ordered actions, Reset's
last/danger checks, Unload's title, enabled states and geometry.

The keyboard race also reproduced locally on More options' Steps before the
repair, rather than only the CI setup-name field. The supplied-viewport fixture
now waits for the open form's read and phone-list redraw, its queued viewport
reveal, and stable fitted geometry before one intentional focus. Its clear/hit
assertions and timing budgets are retained. Async setup steps fail fatally and
return true explicitly; failure rows now include geometry and focus diagnostics.
A deterministic Node-backed CPU test executes the real injected helper and
holds each unfinished owner, without a browser or model.

The corrected phone smoke passed (30.79 seconds) at
`auradream-smoke-artifacts-hzegxj3u/summary.json`; the keyboard smoke passed
(9 seconds) at `auradream-smoke-artifacts-nxk7qzkx/summary.json`. The original
local keyboard failure is retained under `auradream-smoke-artifacts-354bk9h7/`.
Both repairs stay on the original PR branch. No hosted job was rerun unchanged.
The final phone/keyboard pair also passed together with two smoke workers
(2/2, 38 seconds; `auradream-smoke-artifacts-83si8w1m/summary.json`).
