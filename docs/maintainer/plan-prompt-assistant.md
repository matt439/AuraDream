# Plan: a local prompt-writing assistant

`PA#18`, commissioned 2026-11-16. Matt requested all three tasks: develop a
rough idea conversationally, improve an existing prompt, and turn a description
into candidate Danbooru tags. He authorized implementation up to GPU testing
and explicitly said not to run that testing yet. CPU tests and fake-browser
checks establish source behavior only.

## Scope

One optional **Prompt assistant** right sidebar holds Create, Improve and Find tags.
The desktop topbar's rightmost **Assistant** button opens it directly, aligned
with the sidebar. Phones keep the same button in their prompt toolbar. It is
nonmodal on desktop, with the main editor available while it stays open. The
full-screen phone pane contains focus and holds its covered workspace inert.
The selected image checkpoint's guide accompanies each question. Create may
suggest creative details; Improve is asked to preserve the source's subjects,
constraints, exclusions and exact lettering unless the instruction changes them.
Find tags proposes candidates and checks their names against the existing tag
index. These are suggestions, never proof that a prompt will make a better image.

Answers include drafts and notes, candidate tags with authoritative lookup
verdicts, or a clarification question without forced draft content. The user
explicitly applies a draft or inserts selected usable tags.
Application uses the prompt editor's existing undo path and refuses an answer
whose source prompt or checkpoint has since changed. Nothing generates an image
or applies a draft automatically.
An explicit application keeps the sidebar open and uses its new editor text as
the next conversational source. An external edit or Undo still invalidates the
prepared suggestions. Continue explicitly adopts reviewed current text and a
short editable brief; old options remain held until a new answer. The previous
transcript stays visible while a new bounded model-history span starts.

Conversation history lives in the open page's memory, with an explicit reset.
It is bounded rather than silently truncating away earlier instructions. It is
not saved in the gallery, browser storage, generation metadata or server logs.
Message instructions and dialogue history go to the local llama.cpp child only.
Tag verification sends candidate names and draft comma items to Danbooru,
as the existing tag tools do; it is not fully offline.
An unavailable index leaves candidates unchecked, never verified by the LLM's
confidence. A model explanation is labelled separately from an index verdict.

## Runtime and admission

The first version reuses the critic's pinned Qwen3-4B Q8_0 and llama.cpp b11206,
including its verified downloads, loopback authentication, listener ownership
check, context sizing and process-tree cleanup. The existing
`AURADREAM_CRITIC_MODEL` and `AURADREAM_CRITIC_SERVER` overrides still select a
custom pair; this work does not download a newer candidate or change the pin.

On 2026-11-17 Matt requested two selectable alternatives. **Assistant model**
now offers the configured baseline, **Qwen3-8B Q4_K_M** (5,027,783,488 bytes)
and **Qwen3.5-4B Q8_0** (4,482,403,264 bytes). The optional GGUFs' exact
revisions and SHA256 pins live in `backend/assistant_models.py`. Metadata was
checked against the publishers' Hub APIs without fetching weights. Qwen3-8B
comes from Qwen; Qwen3.5-4B from llama.cpp's maintainers. The unchanged b11206
source includes both architectures. Its context stays bounded to 8,192 tokens:
Qwen3-8B uses the author's thinking recipe with the app's 2,048-token reasoning
budget, and Qwen3.5-4B uses the author's general non-thinking recipe.

The browser remembers only the choice id. Changing it between answers retains
the conversation, and each reply keeps its model label. Send prepares the
selected files; the model catalogue only inspects local files. Reopening a
retained conversation can resume unfinished draft tag checks. Preparation
returns immutable paths and a profile for that request and never changes the
critic's prepared baseline. Check keeps its configured model and overrides.
The selection is held while an answer is pending, with an explanatory press.
Download sizes are not memory measurements; both alternatives remain
unmeasured for this workflow. The existing GPU hold still applies.

Preparation runs outside the job gate. The actual answer runs through
`jobs.run_when_idle`, after another check for an image load or queued/running
job. It never calls `ensure_loaded`. The process must stop before releasing the
gate. The critic and assistant share that lifecycle, so shutdown and Unload
retain their existing owner. Requests remain model-dependent readers under
`hold_model_for_request`, including its revision-header check, except the
metadata-only ticket status read. Signed-out requests remain closed by the
existing passcode layer.

Every request has a ticket. Stop is scoped to that ticket, including a stop that
arrives before its request; closing the sidebar requests Stop and discards late
answers. A stopped or superseded answer cannot overwrite a later conversation.

`POST /api/prompt/assist` takes the mode, instruction, source prompt, optional
negative prompt, bounded user/assistant history, pinned requirements, a
continuation brief, optional focused draft and ticket. It returns the checkpoint
and LLM labels, elapsed time, a message, drafts, tag candidates, warnings and
server-computed comparisons/phrase checks. `POST /api/prompt/assist/{ticket}/stop`
stops only that request. `GET /api/prompt/assist/{ticket}/status` exposes its
phase and optional current-file transfer bytes, never the conversation.

## Validation before the card

- CPU contracts: bounds, checkpoint guidance, hostile dialogue treated as data,
  structured output refusal, authoritative tag verdicts and offline fallback.
- HTTP orchestration: cold requests, preparation outside the gate, a load/job
  beginning during preparation, shared gate ownership, ticket-scoped Stop and
  stop-before-start, revision and passcode admission, cleanup on failure.
- Real browser against the fake server: all three modes, a follow-up, explicit
  application and undo, tags, stale editor/model changes, errors and retry,
  Close/Stop/reopen, desktop and phone sidebar layout.
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

## Sidebar layout revision (2026-11-17)

Matt requested a familiar right sidebar, direct access outside Examples, and a
review of text and spacing. The Assistant button now opens a named nonmodal
sidebar. Messages and earlier full drafts stay visible above a fixed composer;
only the latest reply has Apply controls. Repeated help moves into About the
local assistant, with readable draft cards and compact status text. Applying
keeps the conversation available. The visible editor receives desktop focus
for native Undo; covered phone/narrow editors return focus to the Use control.

At 1280 px and above, the workspace reserves the sidebar's width. Smaller
windows overlay it; phones fill the visible viewport above their own fixed
bars. Touch Back and sidebar Escape close through the same ticketed Stop path.
Workspace keyboard commands retain their ordinary routes, and Tab is not
trapped. Short layouts retain the message box and actions above the keyboard.

Local validation, without weights, inference or live gallery data:

- 309 focused assistant/API/runtime, documentation, classic-script,
  accessibility-audit and browser-output tests passed. Ruff passed and pinned
  Pyright checked 218 sources with zero errors or warnings.
- Nine distinct browser smokes passed: prompt-assistant, analysis,
  shortcut-dispatch, shortcuts, snippets, generation-presets, phone-create,
  phone-dialogs and phone. The assistant's final 207 steps include native
  keyboard routing, explicit Apply/Undo and source rebasing, all three tasks,
  Stop/late replies, tag verdicts, sidebar-root accessibility, focus leaving
  the nonmodal sidebar, touch Back and control hit tests.
- Fresh screenshots were inspected at desktop/wide, portrait, 320 px wide,
  landscape and 393x300 keyboard-sized viewports. The new hit tests first
  exposed phone bars covering Send, then a 9 px overflow in the short layout;
  both were corrected and passed in the final fixture.

Final receipts in the system temporary directory:
`auradream-smoke-artifacts-7ueu4ny0/summary.json` (final assistant, screenshots
under `smoke-prompt-assistant-93jk0m_7/`; passed after the final action alignment),
`auradream-smoke-artifacts-jindd1wh/summary.json` (five desktop regressions),
`auradream-smoke-artifacts-rjz_6m89/summary.json` (phone Create/dialogs), and
`auradream-smoke-artifacts-nwvwm3u3/summary.json` (phone).

The GPU hold and physical phone/keyboard acceptance remain unchanged.

## Wider chat and compact controls (2026-11-17)

Matt's actual-page screenshots showed Reset wrapped below the prompt tools and
only two of three drafts fitting in the sidebar. The second layout pass widens
the desktop pane from 420/460-520 px to 600/680-900 px, puts task selection in
the header where it fits, and moves About to the footer. About opens upward
without changing the message box or Send position. The compact composer groups
the message and actions in one frame, with one focus ring, shorter status copy
and the long status text available to focus/scroll and as a title.

Draft titles share a row with Use prompt. Their optional explanations move into
native Notes disclosures; prompts retain their complete text in the one
conversation scroller. Reset stays beside Prompt/help on a stable caption row,
with writing tools in the explicit toolbar below it. Phone moves preserve each
node's new desktop home.

The strengthened assistant fixture uses three realistic tag prompts and two
notes each, matching Matt's example. At 1920x1080 and 2048x1152 it requires all
three Use controls to fit and be hit-testable, at least 60% of the sidebar for
conversation, and closed notes. Native Notes and footer About presses retain
their layout checks; all conversation, Stop, stale-source, Undo, pending-control
stability and phone guards remain. A narrowed Send button initially moved when
Thinking appeared; its stable width was restored before the final pass.

Local checks use fakes and no inference. Seven distinct smokes passed:
prompt-assistant, control-sizing (all 26 catalogue models, explicit tag-tool
views at 1400/1800/2048 px), phone-create, phone-dialogs, analysis,
compact-setup and shortcut-dispatch. Desktop three-draft, footer About, toolbar
and phone screenshots were visually inspected. Physical-device and GPU
acceptance remain separate.

Final follow-up checks: 139 documentation/classic-script/audit/browser-output
tests passed, Ruff passed, and pinned Pyright checked 218 sources with zero
errors or warnings. The final assistant pass completed all 226 steps after the
single composer focus ring was applied. Receipts under the system temporary
directory: `auradream-smoke-artifacts-3_fpgl12/summary.json` (assistant,
screenshots in `smoke-prompt-assistant-ynhgpevx/`),
`auradream-smoke-artifacts-k9bjg1lh/summary.json` (control sizing),
`auradream-smoke-artifacts-wyciob3z/summary.json` (phone Create), and
`auradream-smoke-artifacts-ckcpar2h/summary.json` (four regressions).

## Reset row, duplicate prose and card contrast (2026-11-17)

Matt clarified that Reset belongs on the same row as the writing controls,
and supplied a real reply showing each prompt twice: the model copied the
drafts into `message` as a numbered list as well as returning `drafts`.
Reset now ends the writing-tools row, aligned right, while Prompt/help remain
above it. All 26 catalogue combinations retain one tool row; the phone's hidden
Reset remains the topbar menu's action.

The model question and JSON schema explicitly reserve `message` for a brief
optional acknowledgement or follow-up. Draft text, draft notes and tag
explanations belong only in their structured fields. Validation preserves the
model's unique commentary rather than deleting it heuristically. For existing
repetitive replies, the page detects a complete whitespace-normalized draft
inside `message` and puts that full prose in a closed Assistant note below the
cards. Normal conversational messages stay visible, empty ones add no empty
paragraph, warnings stay visible, and the same rule renders earlier turns.
Conversation data and the note's unique questions are preserved intact.

Draft cards use a lighter token-mixed grey fill with the strong border and
secondary text adjusted for that fill. The real-example browser fixture checks
three Sakura/Aiko/Mei cards, a line-wrapped numbered echo, a unique question,
native note expansion, earlier history, empty/normal messages and card contrast.
Card fill and border contrast against the sidebar exceed 1.6:1 and 2.5:1;
prompt, title and Notes text exceed 4.5:1 on the fill. Fresh screenshots were
visually inspected; no repeated prompt list is visible above the cards.

Local checks: 202 focused assistant/API/runtime and documentation/classic-script/
audit/browser-output tests passed; Ruff passed; pinned Pyright checked 218
sources with zero errors or warnings. Five distinct smokes passed, all on
isolated fakes without inference: assistant (234 steps), control-sizing,
phone-create, analysis and phone-dialogs. Receipts under the system temporary
directory: `auradream-smoke-artifacts-0gamvo2e/summary.json` (assistant,
screenshots in `smoke-prompt-assistant-m5u045lz/`),
`auradream-smoke-artifacts-u_u1_z56/summary.json` (sizing and phone Create), and
`auradream-smoke-artifacts-rqv_wtv_/summary.json` (analysis/phone dialogs).
GPU testing remains held.

## Inline draft notes (2026-11-17)

Matt preferred removing the per-draft Notes disclosure because its row used
space that the notes themselves could occupy. Draft notes now appear directly
below the prompt as secondary text, without a Notes heading or disclosure.
Their content is unchanged; consecutive notes have a compact 4 px gap.

The final assistant smoke passed all 231 steps, including six visible inline
notes across the three cards, fully inside the conversation at 1920 and 2048 px.
The previous reply-note, About, contrast and interaction checks remain. The
three-draft and phone screenshots were inspected. Eight documentation-map
checks and the classic-script guard passed. Receipt under the system temporary
directory: `auradream-smoke-artifacts-noxva0y5/summary.json`, screenshots in
`smoke-prompt-assistant-72rqm4ol/`. No inference was run.

## Tag-coloured cards and smaller composer (2026-11-17)

Matt requested a smaller input, result backgrounds matching the existing tag
box, and stronger separation between prompt and notes. The message field is
now two lines rather than three, its frame padding is 4 px rather than 8 px,
and Send/Stop use the ordinary control height while retaining touch targets.
Drafts use `--panel-tags` for the prompt area. Notes remain visible, with their
own padded `--panel-2` section and a full-width strong divider above them.

The assistant smoke passed all 231 steps. It now checks exact agreement with
the tag surface token, distinct notes fill/divider/padding, text contrast on
each actual surface, and a two-line desktop field with a smaller message frame.
The three drafts and six notes still fit at 1920/2048 px; pending controls stay
stable, and phone/short-viewport guards pass. Desktop and phone screenshots were
inspected. Eight documentation-map tests, the classic-script guard and 39
screenshot-path checks passed. Receipt under the system temporary directory:
`auradream-smoke-artifacts-hrt9_h35/summary.json`, screenshots in
`smoke-prompt-assistant-3lljge9u/`. No inference was run.

## Numbered prompt options (2026-11-17)

Matt requested numbers on the option cards and number-key selection. The latest
drafts now carry 1-3 badges, with matching accessible button labels and
`aria-keyshortcuts`. Plain 1/2/3 within the assistant applies that option when
focus is outside an editor. Alt+1/2/3 also applies from its message field, so
plain numeric typing keeps its meaning. The footer and keyboard help describe
these routes. Earlier conversation cards have no active number badges.

The handler presses the ordinary Use button, preserving source, pending and
offline refusals and undo-aware application. It does not act outside the
assistant, under a dialog/menu/model picker, on modified Ctrl/Meta/Shift or
AltGr combinations, on a repeat, during composition, or on an already-handled
event. Numpad digits share the same logical key mapping.

The expanded assistant browser fixture checks native 1/2/3 with Undo, Alt+2
from the message field, normal numeric typing in the message/main prompt/task
selector, scope and overlay guards, held applications and keyboard help.
Synthetic browser events cover numpad metadata, repeats, composition and extra
modifiers; these are not physical-device acceptance. Existing interaction,
phone, layout, contrast and source guards remain.
After a successful shortcut the held number is captured through its release,
including after Apply focuses the main editor. A matching keyup or window blur
releases it; a fresh non-repeat press also retires a missed release. This keeps
auto-repeat from inserting digits into the newly applied prompt. The fixture
checks that focus-transfer edge and subsequent ordinary numeric typing.

Final local results: all 290 assistant smoke steps passed, with desktop/phone
number badges and narrow footer copy inspected. The existing shortcuts and
shortcut-dispatch smokes passed after the capture guard. The 139 focused
documentation/classic-script/audit/browser-output checks passed, the final
classic-script guard passed, and the eight documentation-map checks passed
after reconciliation. No inference or real gallery was used. Receipts under
the system temporary directory:
`auradream-smoke-artifacts-_qd2pkiv/summary.json` (assistant, screenshots in
`smoke-prompt-assistant-6bdy6jur/`) and
`auradream-smoke-artifacts-ja8g83fg/summary.json` (shortcut regressions).

## One surface per option (2026-11-17)

Matt found the prompt/notes dividing line and separate fill made three options
look like six boxes. Each option now has one continuous purple surface and one
outer border. Inline notes use smaller, softer text and a 12 px gap below the
prompt, without an internal rule, secondary fill or extra section padding.

The assistant smoke passed all 290 steps, including no internal border or
negative gutters, shared card surface, readable notes contrast, text/spacing
distinction, three-draft fit and existing keyboard/phone guards. Desktop and
phone screenshots were inspected. Eight documentation-map tests and 39
screenshot-path checks passed. Receipt under the system temporary directory:
`auradream-smoke-artifacts-fdbhn6mf/summary.json`, screenshots in
`smoke-prompt-assistant-4hy5ahew/`. No inference was run.

## Added and removed tags (2026-11-17)

Matt requested a visual indication of tag changes in each option. The server
now returns response-only `draft_changes`, aligned with the draft array. Each
entry has `added` and `removed` lists, or null when the existing prompt reader
cannot safely compare both texts as tags. An empty source is supported. Prose,
hybrid/structured text, quoted lettering and grid placeholders are withheld.
Exact `normalize_tag` name differences preserve original spelling and order
for display. Reordering, repeated instances and cosmetic case/underscore/escape
changes do not add or remove a name; weighting changes remain meaningful.
The field is absent from the LLM answer schema and cannot be authored by it.

Each card shows green Added (+) and red Removed (−) rows, with no extra box or
divider. An unchanged membership says No tags added or removed, since ordering
can still change. The comparison is tied to the request's actual prompt and
stays with the full page-memory answer after Apply, Undo, outside edits and
follow-ups. It is excluded from the bounded model-history payload. Applying
still writes the original draft string through the existing undo path.

Local checks: 226 focused assistant/API/runtime and documentation/classic-script/
audit/browser-output tests passed; Ruff passed; pinned Pyright checked 218
sources with zero errors or warnings. The HTTP check verifies serialized changes
from the real answer validator. The final assistant smoke passed all 294 steps,
including three visible options and notes at 1920/2048 px, phone/320 px changes,
unchanged/omitted data, contrast, literal text safety, fixed comparison history,
keyboard Apply/Undo and all prior guards. Desktop and phone screenshots were
inspected; 39 screenshot-path checks passed. Receipt under the system temporary
directory: `auradream-smoke-artifacts-e4t4vx53/summary.json`, screenshots in
`smoke-prompt-assistant-vpz6dzsi/`. No model weights, inference or real gallery
were used. A running server must restart to supply the new response field.

## Selectable writing models (2026-11-17)

Matt requested Qwen3-8B and Qwen3.5-4B alongside the configured baseline.
The selector, pinned catalogue, separate runtime profiles and request-owned
preparation described above are built. Only the selected id is remembered;
model changes between replies retain the conversation. The compact task names
Create, Improve and Find tags keep both selectors readable at 320 px. The wide
header keeps its controls on one row so three drafts and their notes still fit.

Local validation, without weights, GPU inference or real gallery data:

- The full CPU run completed with 11,237 passed and 325 skipped, plus one
  failed smoke-audit fixture check: the new reload needed an explicit
  accessibility check immediately before navigation. After that correction,
  all 293 focused audit, runtime, documentation, browser-output and smoke-runner
  checks passed. Two additional actual-token boundary cases passed in the
  runtime file: a 7,000-token input fits the non-thinking profile but is held
  when the thinking profile needs its reasoning reserve.
- Ruff passed. Pinned Pyright checked 219 Python sources with zero errors or
  warnings. Shared classic-script syntax/scope checks passed.
- The assistant browser smoke checks both alternatives through the fake API,
  preserved follow-ups, per-reply attribution, held native and scripted model
  changes, first-use information and a real page reload restoring only the id.
  Its 325 steps also retain three-draft density, Apply/Undo, Stop, phone,
  landscape and keyboard-sized viewports and accessibility checks.
- Six distinct regression smokes passed: analysis, model-switch,
  shortcut-dispatch, phone-create, phone-dialogs and phone. The phone fixture's
  old caption-membership assertion was diagnosed and corrected to the existing
  named prompt-tools row, including Assistant and Reset's last position. All
  other desktop and phone geometry/host checks were retained.

The first assistant smoke caught an extra wide header row reducing room for
the third draft's notes. Its original density assertion was retained and the
wide header corrected. Desktop, 320 px phone and restored-selection screenshots
were visually inspected. The final compact-label browser check passed all 325
steps (22 seconds in the browser, 29 seconds including the runner). Receipt:
`auradream-smoke-artifacts-cyh5z35k/summary.json` under the system temporary
directory; screenshots in `smoke-prompt-assistant-2y8bjjvl/`. The six regression
receipts are `auradream-smoke-artifacts-lwvqziy7/summary.json` (five passed,
the diagnosed phone fixture failure) and
`auradream-smoke-artifacts-j5rtdnd8/summary.json` (the corrected phone pass).
No assistant GPU session was run; prompt quality, latency, memory and
real-model cancellation remain held for later authorization.

## Message clearing and conversation following (2026-11-17)

Matt reported that Ctrl+Enter left the sent message in the input and new
entries stayed below the visible conversation. Admitted sends now clear the
field immediately, retaining the pending message above it. Replies preserve
newer drafts, including identical retyped text, and leave focus alone. Failed
or stopped messages restore their exact original text only to an untouched
empty field. A draft typed and then erased stays erased.

Send scrolls the conversation to its end. Replies follow when the reader is
within 48 px of the bottom; reading earlier messages preserves the scroll
position. Document scrolling and the fixed composer's geometry stay unchanged.
Plain Enter retains its newline behavior, and composing/already-handled events
cannot send.

Local validation used isolated fakes and no inference: all 336 assistant smoke
steps passed, along with shortcuts and shortcut-dispatch. The 139 focused
documentation, classic-script, accessibility-audit and browser-output checks
passed, as did Ruff. Native Ctrl+Enter and Send cover immediate clearing;
Cmd+Enter is checked with a synthetic browser event. Real overflow exercises
both bottom-follow and reading-position preservation. Desktop and phone
screenshots were inspected. Final receipts under the system temporary directory:
`auradream-smoke-artifacts-ryhtc3an/summary.json` (assistant) and
`auradream-smoke-artifacts-gj0hhjb9/summary.json` (keyboard regressions).

## Sidebar opener alignment (2026-11-17)

Matt found the left-side opener disconnected from the right sidebar. The
desktop Assistant button now sits at the right end of the topbar, directly
above the pane, at the same size as its neighbouring actions. The same node
and its expanded state, toggle and focus-return paths are retained. On phones
it still moves into the existing prompt toolbar for the full-screen assistant.
At 600 px and below the word mark gives the checkpoint status and actions
enough room; a native 400 px check caught overflow before this adjustment.

The existing browser fixtures now check the rightmost header action separately
from the prompt tools, including native hit tests at 800 and 400 px and the
phone-to-desktop return. Desktop and narrow screenshots were inspected.

Five focused browser smokes passed: prompt-assistant, control-sizing, phone,
phone-create and narrow. The narrow fixture's old y749 click points landed
1.5 px above Generate; measured right-edge centres and fatal hit prerequisites
replace them. Its load-state check now expects Assistant after Load/Unload.
All 106 focused documentation-map, section-bound, classic-script and browser
output tests passed, along with Ruff and `git diff --check`.

Receipts under the system temporary directory:
`auradream-smoke-artifacts-art6f6qg/summary.json` (final assistant pass;
its original narrow failure was diagnosed and corrected),
`auradream-smoke-artifacts-p6bxri_l/summary.json` (sizing and phone passes),
and `auradream-smoke-artifacts-z58gqqyg/summary.json` (final narrow pass).
The isolated fake servers and browser processes were stopped. No model weights
or GPU inference were used; the assistant's GPU hold remains in force.

## Conversation and draft workflow review (2026-11-17)

Matt asked for a review of the sidebar's features and UX, then commissioned all
of its recommendations. The review reproduced task changes clearing the chat,
manual edits blocking continuation, an abrupt context limit after five short
answers, question-only replies being refused and phone focus escaping behind
the full-screen pane. The changes retain the compact composer, numbered purple
draft cards, explicit Use and native Undo.

- Changing Create, Improve or Find tags now sets the next request's task without
  clearing the conversation. Continue opens the current source and an editable
  brief; acceptance rechecks the exact source, retains the earlier transcript
  and starts a new bounded model-history span. Old Use controls remain held
  until a new answer. When Refine selected an option, Continue also previews
  that focus and offers an explicit checked choice to keep it; neither source
  is written to the editor. The space indicator warns before the existing limits;
  no automatic truncation or unreviewed model summary replaces prior intent.
- Requests carry a brief of at most 2,000 characters, up to eight pinned
  requirements of at most 200 characters each, and an optional focused option.
  All count toward the existing 12,000-character input limit. Refine this option
  sets that focus for the next reply without applying a draft to the editor.
  A reply captures its requirement list: later pin changes hold Apply while
  permitting Send with the revised pins.
  The validator accepts nonempty clarification-only replies while retaining
  wrong-task and empty-answer refusals.
- The server supplies response-only literal prose changes and exact pinned
  phrase checks alongside the existing tag differences. Comparisons are
  lossless for the original and proposed texts, with a bounded broad fallback
  for expensive cases. Phrase presence is case-sensitive with boundaries at
  word-like edges, not semantic understanding or an image guarantee. More
  contains prose comparison; phrase checks are reviewable on each card.
- More offers Copy, Save as snippet, Refine, Check and Add to comparison on
  current and earlier drafts. Save prepares the ordinary snippet editor for an
  explicit Save, preserving its dirty/recovery state. CPU Check uses a separate
  dialog without editor writes or model loading. A comparison captures selected
  prompts and current settings, previews one image per prompt on four shared
  seeds, and submits only on Generate comparison. Prompt and grid fields are
  unchanged; Repeat, grid axes and Then refine do not apply.
  On a phone, a later snippet Insert closes the covering assistant before the
  ordinary editor rewrite, revealing its destination and retaining native Undo.
- Readiness/download cost is visible before Send. Ticket status names actual
  preparation, download, waiting, startup, writing and verification phases,
  with current-file transfer bytes where available. Status contains no prompt
  content, bypasses model-reader admission and does not count as keep-awake
  activity. Terminal records and stop tombstones expire after five minutes and
  their union is bounded to 256; late callbacks cannot alter a terminal ticket.
- An exact editor match marks Current prompt. Expanded message is optional;
  Jump to latest appears for an unread reply without disrupting someone reading
  history. Alias presentation retains both proposed and canonical spellings.
  On full-screen phones the pane owns dialog semantics, background inert state
  and native Tab/focus containment; desktop returns to the nonmodal aside.
  The scoped synchronous editor-write helper preserves native Undo, and native
  snippet, Check, continuation and comparison dialogs remain available.
  Requirements and Compare remain reachable in short panes when the ordinary
  About explanation and shortcut hints are hidden for space.

The source is split across the existing assistant script and three small
classic modules: `assistant-conversation.js`, `assistant-draft-tools.js` and
`assistant-phone.js`. The expanded `smoke-prompt-assistant.json` preserves its
existing density and interaction contracts. `smoke-assistant-workflow.json`
uses its own fresh FLUX fake on the shared 7960 row for the new workflows,
native phone focus/Undo, dialog handoffs and stale progress. Its generation
handoff is stubbed; it runs no model downloads or image inference and uses no
real gallery. GPU prompt quality, latency, memory and process cancellation
remain unmeasured and explicitly held for separate authorization.

Integrated local validation, without weights, GPU inference or real gallery:

- `uv run --group dev pytest -n 6 --durations=25`: **11,296 passed, 325
  skipped**, with 87 warnings, in 263.95 seconds. The first full run exposed a
  missing cost-table row for the new smoke; after adding it, the complete CPU
  gate passed. Log under the system temporary directory:
  `auradream-assistant-cpu-c5ce51c07e2045d89f4fd3f029077014.log`.
- Ruff passed. Pinned Pyright checked all 219 backend/tools Python sources with
  zero errors or warnings. The classic-script guard passed all 83 scripts.
- Eleven distinct browser smokes passed in 102 seconds: analysis, model-switch,
  phone, phone-create, prompt-assistant, phone-dialogs, assistant-workflow,
  shortcut-dispatch, shortcuts, snippets and control-sizing. Receipt under the
  system temporary directory: `auradream-smoke-artifacts-ocl_6xop/summary.json`.
  Comparison, continuation and phone screenshots were visually inspected;
  dialog surfaces use the application tokens.

The final focused browser pass passed both smokes: 343 assistant steps and 244
workflow steps, in 61 seconds including setup/cleanup. It covers native phone
snippet Insert/Undo, late snippet reads after dirty-editor reopen, About popup
bounds with selected comparisons, and New chat clearing comparison selections.
The final receipt is `auradream-smoke-artifacts-wh1wipmp/summary.json` under the
system temporary directory; final screenshots were visually inspected. The
final smoke/documentation/accessibility/harness subset passed 321 tests. These
local checks do not lift the GPU hold or establish physical-device acceptance.

## Tag verdicts inside Create and Improve drafts (2026-11-17)

Matt supplied a screenshot of generated options containing unrecognised
Danbooru names and asked for validation like the prompt editor. The draft
renderer now reuses `POST /api/tags/validate` on tag-vocabulary checkpoints,
with the exact draft prompt and first 64 comma items. No backend route,
assistant schema field, model inference or automatic text correction is added.

`assistant-draft-tags.js` keeps the prompt text exact while adding read-only
underlines: solid green for known names (the guide accent for checkpoint guide
tags), wavy red for unknown names, dashed amber for aliases and dotted for
unchecked items. Grid placeholders remain unmarked as tags. More > Tag
validation provides state counts, full verdicts and Recheck. Missing/offline
results and items after the 64-item cap remain unchecked. Server-classified
natural-language spans in hybrid prompts stay neutral; their dictionary details
remain available without counting the prose as validated tags. A known name
does not prove checkpoint training coverage or rendered image quality.

Each answer owns a WeakMap cache of exact draft/model/revision records. Current
and historical card renderings share finished verdicts; unfinished reads may
resume after reopening. Request identity, epoch and checkpoint guards discard
late responses, while Close, New chat and checkpoint changes abort outstanding
work. Use, Copy and follow-up content always retain the original draft; aliases
are visible suggestions, never silent replacements. About now explicitly says
that draft comma items and suggested tag names are sent to Danbooru.

`smoke-assistant-draft-tags.json` uses its own fresh Animagine fake on 7960,
then switches to Kohaku to exercise the real contextual prose reader. Its cases cover each verdict
state, Recheck, capped lookup, historical cache reuse, exact Use/Copy and
late-response retirement. The fixture uses no live tag lookup, weights, GPU
inference or real gallery. Test results belong to this follow-up's receipt;
the earlier integrated gate above predates these changes. The GPU hold remains
in force.

Validation: 411 focused tag, contextual-prose, API, classic-script,
documentation, smoke-runner, accessibility and browser-output tests passed;
Ruff and the 84-script guard passed. The new smoke passed all 79 steps in
9 seconds (17 seconds with setup/cleanup), with desktop and hybrid screenshots
inspected: `auradream-smoke-artifacts-jy890r8u/summary.json` under system temp.
The fake vocabulary is deterministic; it does not establish which names the
live Danbooru index currently accepts.

Four existing smokes also passed: prompt-assistant, assistant-workflow,
mixed-prompts and tags. The first three passed in
`auradream-smoke-artifacts-7yi5r1qe/summary.json`. The tags fixture's old 70 px
wheel stopped 19 px before its under-header prerequisite, independently of the
new draft renderer. Its native scroll now awaits the actual bottom-held
geometry as a fatal prerequisite, retaining the exact Alt+Arrow/Delete
stability assertions. That smoke passed in 21 seconds (30 seconds including
setup/cleanup): `auradream-smoke-artifacts-blcw399c/summary.json`; 41 additional
focused harness checks passed. All fake servers and owned browsers were stopped.
