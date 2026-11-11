# UI#30: copy-feedback ownership

**Status:** `UI#30` implemented and locally qualified, 2026-10-29 UTC.
Independent source/VM/native reviews, measured original controls, 57 focused
checks, the new native fixture, two related browser regressions and inspection
of both new screenshots are complete. Real clipboard integration/permissions
and user/device acceptance remain outside this qualification.

## Why this work

Matt reports J1 Copy prompt/seed Often, as well as F1/F3 inspection and L7
image menus. Requirements UX#6 keeps these actions at the inspected image;
the workbench requires local feedback without stealing focus or the caret.
The ranked source audit selected a concrete display defect rather than new
clipboard features:

1. A copy promise started on A could finish after walking to B or reopening
   the dialog and overwrite that inspection's message. Card/result errors
   also mirrored into an unrelated currently open lightbox.
2. Even an already completed lightbox copy note remained on the next image.

Both are one ownership gap. Captured prompt/seed payloads were already
correct. Clipboard image copying is Never (J2), so its existing PNG/fetch
handoff is preserved; only its shared helper's feedback gets the same guard.
No seed parsing, wire format, generation, controls, layout or security
permission policy changes are proposed. No competitor claim is needed.

## Source contract

`web/context-menu.js` keeps the existing clipboard callback and invokes it
once synchronously inside the chosen item's `run`, before its first await.
It adds no preflight await or automatic retry. A latest-press token per
original host controls only completion feedback. Constructing another menu
does not take ownership; normal menu closure does not retire a copy.

For the lightbox, the token must match the exact record object captured by
the item and the active inspection scope. `web/lightbox.js` synchronizes
that scope at render entry and retires it on close. A walk or a new record
object retires it permanently; walking A-to-B-to-A cannot revive it. An
already-open same-record-object background redraw retains it. A fresh open
retires it even if the record object is unchanged and the dialog's queued
native close event has not arrived yet.

A completed lightbox copy records its displayed text and the exact Text
node written by the existing `showError`. Retirement clears it only while
both still match. A later unrelated `showError` replaces that node even
when its words are identical, so it survives cleanup. This avoids editing
the shared feedback helpers or introducing a generic message framework.

A card/result retains completion feedback only while its original host is
connected, still carries the original record ID and is not a tombstone.
Its failure keeps one announced sentence in the gallery and a quiet local
gist, with `mirror:false` so another modal image is left alone. Current
lightbox failure is local to the lightbox. Held clipboard actions retain
their explanatory refusal without making a write attempt.

Bounds: latest copy per receiver and original inspection ownership, not a
unified all-action notification clock, cross-card ordering protocol or OS
clipboard completion-order guarantee. A later unrelated action within the
same still-current inspection is not made part of such a clock. Existing
direct message writers which retain and mutate the very same Text node
cannot be distinguished solely by that node; normal `showError` replacement
is the supported and tested ownership signal.

## Qualification scope

`tests/test_copy_feedback_ownership.py` authors 22 Node VM cases against the
actual declaration-only menu/lightbox modules and actual extracted
`showError`, `refuseInCard` and `cardNote`. Unrelated geometry/model helpers
are inert ports. The DOM port models `textContent` as replacement of Text
children; synthetic clipboard promises provide controlled success/error.
Native clipboard reads/writes, image-copy fallback, fetch and execCommand
are poisoned. No page, server, browser or real user data is involved.

Original-source controls below establish 16 ownership failures and six
ordinary controls across two measured runs. Controls cover exact prompt whitespace and
seed text, synchronous once-per-choice invocation, harmless redraw,
navigation, return to A, close/reopen same object, pre-close-event reopen,
stale menu source, latest-copy ordering, current/local errors, detached or
reused hosts, held copies and newer identical/different messages.

`tools/smoke-copy-feedback-ownership.json` is a normal fresh FLUX fake fixture
on port 7950. It discovers only the two actual planted other-model
rows through GETs, uses physical All models/card/menu/navigation/close
actions, and labels selector annotation, sentinel messages and promise
release as programmatic setup. It installs and verifies a wholly synthetic
clipboard before any copy. Each exact one-use write grant requires its own
trusted menu item. A wrapper checks the actual current native click via
`window.event`, calls the original chooser synchronously and clears its
press marker in `finally`. A preflight await therefore loses the marker;
browser capture/bubble microtask timing is not used as the proof. Missing
event evidence fails closed before the original action. All clipboard
reads, picture writes, ClipboardItem
construction and execCommand are poisoned; no Ctrl+C or native clipboard
fallback is used or restored.

The fixture covers seven synthetic text copies, delayed success/error,
same-A reopen, latest-copy supersession, settled-note cleanup, actual
same-text-node replacement and a card-origin error under another modal.
GET-only admission, unchanged row bodies/PNG hashes, empty job history and
unchanged composer/seed/search are final guards. The two new screenshots
were inspected as described below. No
browser flags, trust/security settings or historical denied artifact roots
are used.

Root's measured controls, independent reviews and positive checks are
recorded below. Native clipboard integration/permissions, OS contents and
manual user/device acceptance are outside this work and are not claimed
from the synthetic-port checks.

## Ownership and pending work

The packet contains two full web snapshots plus exact source/doc/AGENTS
anchors and hashes. Shared rules/design/AGENTS anchors must be applied and
partially staged without overwriting pending hunks. TODO is omitted and
root-owned. All pending GAL12/UI28 work, unknown outcomes, denied artifact
reads and stopped operations remain untouched. There is no browser run,
test execution, repository mutation, OS clipboard operation or qualification
claim by the proposal author.


## Original-source controls and validation repairs (2026-10-29 UTC)

The first 22-case actual-module run reports 18 failures and four passes in
1.42 seconds. Sixteen failures are the intended ownership/local-feedback
assertions, with no SyntaxError or ReferenceError. Two expected success
controls also fail because two host Promise turns do not settle a VM-realm
promise reached through the void-returning actual chooser. The root corrects
only their observation: capture the actual item.run promise synchronously
through the original chooser and await it after the unchanged synchronous
write/payload/menu assertions. Those two cases pass on original source in
0.30 seconds, with 20 deselected. Thus the measured original split is 16
ownership failures and six ordinary controls across these two runs; it is
not a claim of a second full run. An installed TestClient/httpx deprecation
warning is reported. The final VM poison factory returns an ordinary function
so unexpected constructor calls are recorded rather than merely rejected as
nonconstructible arrows.

Before any native execution, independent review finds two authored screenshot
keys unsupported by the actual harness. Fixture v2 changes only steps 50/53
from `screenshot` to `shot`, retaining their filenames. V3 replaces the
capture-listener/microtask press marker with the verified synchronous chooser
wrapper described above and binds each trusted item to its exact grant name.
These are validation changes, not application repairs. Root retains captured
v1 metadata/anchors; an author regeneration after registration produced an
invalid hypothetical duplicate AGENTS target, which was never applied.
Versioned corrected fixtures and the original execution output are retained
in the owned workspace. No product assertion was weakened.

The original v3 native fixture then fails exactly at fatal step 14: a real
physical menu choice starts one synthetic prompt write on A; physical Next
displays B, and the controlled A completion replaces B's explicitly labeled
unrelated-message sentinel. Browser exit is zero/no signal; runner exit one
is the intended assertion, in ten seconds total (browser three). All later
steps, screenshots and final guards are not reached and are not claimed as
evidence. The new artifact root is `auradream-smoke-artifacts-65pazf7h`, child
`smoke-copy-feedback-ownership-ol82wrh5`; no artifact read is needed for this
command-output result. No old denied artifact or unknown run was retried.

Independent source review is clear at menu `8c918ea2`, lightbox `ffc9e95f`;
final VM controls `bd97c947` and native v3 `49362c4d` are independently reviewed.
The original PNG-copy function body is unchanged. This fixture observes no
backup metadata: GET-only mutation admission prevents activation attempts,
but no measured backup-state assertion is claimed.


## Positive qualification (2026-10-29 UTC)

All 57 focused checks pass in 5.62 seconds: the 22 actual-module ownership
cases, classic-script shared-scope/syntax guard and documentation maps. One
installed Starlette TestClient/httpx deprecation warning is reported; no
application failure or dependency update is inferred from that warning.

All 53 native fixture steps, including 23 fatal checks, pass in 15 seconds
(browser eight). Seven exact synthetic writes start synchronously in their
own trusted menu choices. Actual physical navigation retires the old A
completion, clears a settled copy note, preserves a newer identical-word
message node, and keeps old failure out of a fresh same-A inspection. A
card-origin failure reports its quiet local gist and gallery sentence without
covering B's modal note; a current modal failure still appears locally.
Final guards prove unchanged actual planted row bodies/PNG hashes and
composer/seed/search, no jobs or selections, and no script errors or forbidden
operations. Same-record redraw/record-object replacement, stale closures,
detached/reused hosts and the pre-queued-close-event case have separate VM
evidence; the native fixture does not claim every one of those event orders.

Both new screenshots were read directly with the image tool and inspected
at 1400x1000: `ui30-copy-feedback-current-error.png` shows the readable current
error in the lightbox's right column, and `ui30-copy-feedback-gallery.png`
shows the original card's quiet `Copy failed` gist and the gallery sentence.
They are under `auradream-smoke-artifacts-gz0onfky/`
`smoke-copy-feedback-ownership-trksw95l`. No narrow-viewport, real model output
or system clipboard acceptance is inferred from these two images. Their
successful reads do not change any earlier denied artifact boundary.

Existing `smoke-lightbox` and `smoke-context-menu` both pass serially on fresh
fake data in 91 seconds total; context-menu reports 12 seconds. The lightbox
individual duration was truncated from command output and is not inferred.
Existing fake generation/refine/bin/download preparation in those regressions
is separate from the new GET-only planted-data fixture and real hardware.

The existing context-menu fixture now verifies a frozen own synthetic
clipboard port and an explicit execCommand refusal before any copy or job
setup. Its text/image write bodies remain unchanged; read/readText ports
throw. Independent comparison proves one setup change and one immediate
fatal guard, with every other parsed step identical (176 to 177 steps).
Its old programmatic item.click choices remain a separate legacy regression;
they are not relabeled as the new fixture's trusted physical choices. The
ordinary constructed ClipboardItem/Blob handoff uses owned in-memory data,
not the OS clipboard. The root's first preparation assertion used a wrong
step index and stopped before mutation; exact anchor lookup corrected that
helper, which then applied once. No application change resulted from that
preparation error.

No-cache Ruff passes. Pinned offline Pyright 1.1.407 checks 146 files with
zero errors or warnings; its measured duration and final docs check are
retained in the workspace journal. Source hashes remain menu `8c918ea2`,
lightbox `ffc9e95f`; qualified test `bd97c947`, native fixture `49362c4d`
and existing menu guard `78e2a357` are independently reviewed. Only owned
shared-document hunks are staged. Pending GAL12/UI28 source, their unknown
outcomes, GAL19's denied visual inspection and all other stopped operations
remain separate and untouched.
