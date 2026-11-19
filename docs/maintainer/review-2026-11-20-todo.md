# TODO audit, 2026-11-20

At `11580bf`, the index contained 33 unfinished items. Two were marked
`OPEN`, 12 needed Matt, nine called for card work, nine were deferred and
one was blocked. The two ready items both led into multi-stage image work.
The following findings add shorter jobs that can be finished with CPU,
temporary galleries, stubbed scripts or the fake browser server.

## Existing work

- **Tiled Refine (`GEN#49`).** The private four-region CPU prototype exists.
  The next change is the backend adapter that consumes the prepared canvas,
  proves its relationship to the original image and publishes one result.
  The plan also covers persisted intent and replay before card qualification.
  [Stage 2](plan-tiled-refine.md#2-exact-backend-integration-and-durable-intent)
- **Larger Kontext edits (`INP#13`).** The original 512-square edit passed
  real-server acceptance on November 18. The first deliverable is the
  1024-square recipe and a one-load protocol that includes the previously
  frozen aspects. The owner records the measured host-memory cost on which
  the new protocol must build. [Acceptance and next recipe](plan-instruction-editing.md#card-acceptance-result-2026-11-18)
- **Detailer (`INP#12`).** Matt chooses the gallery image IDs for the
  authorized quality and memory comparison. The next action belongs under
  `USER`. [Selection and card criteria](plan-detailer.md#remaining-card-acceptance)
- **ERNIE (`CARD#9`).** The projected 17,587 MiB for a 2048-square batch of
  two exceeds the 15,200 MiB ceiling. The row can be revisited when a viable
  hardware or recipe envelope exists. Until then it belongs under
  `DEFERRED`. [Measured rungs and refusal](../verified.md#ernie-reproduction-and-actual-first-use-expand)
- **Speed and workstation acceptance.** The Regional, attention and caching
  image pairs still need Matt's judgment. Dynamic fp8 still depends on
  Regional qualification. The writing assistant's GPU hold, hands-on
  console and device checks, and three remote 3070 measurements retain
  their existing conditions. [Regional decisions](plan-regional-compilation.md#matts-questions) ·
  [Dynamic fp8 dependency](plan-fp8-activations.md#current-dependency-2026-11-18) ·
  [Assistant handoff](plan-prompt-assistant.md)

## DEV#71 Recover an interrupted updater backup

The Windows and native Linux updaters give the pre-update database its final
name before copying the SQLite write-ahead log (WAL). On a retry at the same
commit they trust the presence of the database. A Windows scratch run held
the WAL open. The first update failed after copying the database, then the
retry left that incomplete copy in place and reached the stubbed Git pull.
In a second scratch database, the source held committed row 42 and the
advertised rollback copy lacked it after the retry. The Linux script has
the same publication order. [Windows updater](../../update.ps1) ·
[Linux updater](../../linux/update.sh)

**Done when** an interrupted backup on either platform can be retried and
restored with its last committed row. Interrupt the scratch copy before and
after copying the WAL, and cover a database without a WAL. A completed first
backup for the same commit survives a retry. Rotation and pull can proceed
once the restore point is complete. [Existing script fixtures](../../tests/test_install_scripts.py)

## DEV#72 Stop the Linux critic when its owning app dies

In the recorded WSL run, killing the owning app left its critic process and
listener alive for the full 160-second observation. Idle sleep reduced GPU
use while the listener remained open. The POSIX launcher starts a new
session. Its existing process tests exercise live-owner cleanup, including
a wrapper that exits before its descendant. [Measurement](../verified.md#wsl-critic-after-abrupt-owning-app-death-2026-11-18) ·
[Launcher](../../backend/critic_server.py) · [Process tests](../../tests/test_critic_process.py)

**Done when** a no-weight Linux test starts the wrapped dummy listener
through the production launcher, kills only its disposable app owner, and
observes the owned descendants and listener exit within a bounded wait.
An unrelated listener survives and ordinary Stop still cleans up. Record
the real WSL repeat separately under `CARD#11`.

## DEV#73 Recognize checkpoint deletion before stopping the Linux service

The server reports an in-progress cache deletion in `/api/status` as
`model_delete`. Given a status containing only that activity, the Linux
service script's busy check returned the same empty reason as an idle
server. Stop, restart and update with restart use that result. The backend
already waits for its deletion worker during shutdown. [Status producer](../../backend/main.py) ·
[Busy check](../../linux/common.sh) · [Service controls](../../linux/control.sh)

**Done when** stubbed stop, restart and update-with-restart report the
active deletion and leave the running service alone. Verify the same paths
with an idle status and retain the documented force behavior where offered.
This can be checked on temporary fixtures independently of Matt's Windows
cache acceptance.

## DEV#74 Bring the Assistant usage guide up to date

The guide puts the desktop opener beside the prompt and describes a
checkpoint change as holding an old answer. The opener is now on the right
of the top bar, and switching checkpoints clears the conversation. The
guide explains the critic comparison but omits the Assistant's own
four-shared-seed comparison. [User guide](../usage.md) ·
[Implemented conversation workflow](plan-prompt-assistant.md#conversation-and-draft-workflow-review-2026-11-17)

**Done when** the desktop and phone directions can be followed on the fake
server. Cover Continue after an ordinary source edit, pinned requirements,
and More → Add to comparison → Compare options → Generate comparison.
Describe the separate critic comparison and what survives an editor change
or checkpoint switch.

## DEV#75 Start the Windows launcher from a bracketed checkout path

With the actual PowerShell launcher in a scratch `[images]\AuraDream`
checkout and an unrelated current directory, wildcard-aware
`Push-Location` reported that the checkout did not exist. The launcher
continued, ran the stub `uv` in the caller's directory and returned success.
[PowerShell launcher](../../bin/auradream.ps1)

**Done when** a launcher invoked from elsewhere passes the checkout as
the working directory for a bracketed path, forwards arguments and
restores the caller's location. Failure to enter the checkout exits
before invoking `uv`.

## DEV#76 Exercise the changing scrollbar in the batch-strip fixture

Two real-server attempts observed a 15-pixel scrollbar throughout one job
and a zero-pixel scrollbar throughout the other. The pack logs the observed
widths. Stable tile and gallery geometry in those runs left the transition
itself untested. [Real observations](plan-card-harness.md#real-page-follow-up-2026-11-18) ·
[Current pack](../../tools/card/card10-7733f81.json)

**Done when** a disposable fake-server fixture observes the document
scrollbar change as the job ends, using native controls and ordinary page
layout. Assert the transition and the tile height, gallery head and
Generate position across it. A static-width run cannot pass the transition
check. `CARD#10` owns repeating it on the real server.

## GEN#73 Finish only the Refine passes the original jobs requested

In a scratch run, seed 1 belonged to a plain job and seed 2 to a job with
Then refine. Both base images were missing. The actual Finish handler and
fake worker saved four rows, a base and pass for each seed. The retained
requests called for three. The planner submits the combined missing cells
with one shared Then refine spec. The existing mixed-run test starts with
both bases already saved. [Finish planner](../../backend/api_runs.py) ·
[Then refine plan](plan-then-refine.md)

**Done when** Finish previews the same work the fake worker executes, saves
both bases and only seed 2's pass, and a second Finish finds nothing
missing. Keep recovery of a missing pass on a surviving base and the
refusal for incompatible retained specs.

## GEN#74 Publish pending Pause changes to other watchers

Two streams watched one running two-cell job. Pressing Pause and then
withdrawing it before the cell boundary changed the REST snapshot both
times, but neither stream received a frame for either change. Cancel
reached both streams in the same setup. Only the tab originating the
Pause action updates its pending control immediately.
[Job manager](../../backend/jobs.py) · [Stream handler](../../web/job-stream.js) ·
[Pause control](../../web/job-progress.js)

**Done when** a second tab can see and withdraw the first tab's pending
Pause before the next boundary. Both tabs then agree. Keep the current
step and progress visible and distinguish a requested pause from a worker
that has reached the boundary and stopped.

## GAL#32 Keep a damaged retained job from hiding healthy runs

A scratch database with one damaged compressed cell payload made the Runs
list raise `zlib.error`. Two readers miss this error, the lightweight
image-backed size reader and the full retained-job decoder. The existing
malformed-gzip test exercises a bad header. Damage in the compressed data
after a valid header follows a different exception path. [Summary reader](../../backend/gallery_reports.py) ·
[Full reader](../../backend/run_jobs.py) · [Recovery tests](../../tests/test_api_finish_run.py)

**Done when** a mixed scratch gallery still lists healthy runs, falls back
to image sizes for an affected run with outputs, and omits only the
unreadable zero-output run. A direct recovery request for the damaged job
returns the existing named refusal and queues no cells.

## GAL#33 Batch the Runs list reads for zero-output jobs

The `GAL#30` fix reads image-backed runs in one session. The later
zero-output recovery path calls `retained_run_jobs` separately for each
candidate. A scratch database with 200 zero-image runs of 40 cells used 202
gallery sessions for one expanded list. Review unfinished work reads 20
summaries initially and up to 200 on request. [Runs list](../../backend/gallery_reports.py) ·
[Recovery workflow](../design-notes.md#finding-older-unfinished-runs-2026-11-15)

**Done when** listing 20 or 200 zero-image runs uses a bounded session count
independent of the number listed. Preserve summary values, ordering, model
scope and bin behavior in mixed histories. Malformed requests, metadata or
outcomes still isolate the affected run. Keep the full eligibility check on
the user's deliberate Check or Finish action. [Existing recovery cases](../../tests/test_api_finish_run.py)

## GAL#34 Keep a reindex dry run read-only on older galleries

A scratch destination with an existing row and no `detailer` column gained
that column during `reindex --dry-run`. Its SQLite file hash changed.
Both PNG-only and database-backed previews use destination lookups that
open the normal storage connection, which runs migrations. The CLI help
promises a preview that changes nothing. [Reindex](../../backend/reindex.py) ·
[Storage connection](../../backend/storage.py)

**Done when** both preview modes read a populated legacy destination and
predict the real import while preserving its schema, database contents and
PNGs. Compare the stable database bytes on a closed, checkpointed fixture.
Retain existing behavior for a missing destination and for a real import.

## GAL#35 Verify an existing reindex image before mapping its dependants

A scratch database import had a same-name parent row with different pixels
on disk. Reindex counted the parent as present, imported its Refine child
and linked the child to the unrelated pixels. It reported zero conflicts
and zero unresolved references. In PNG-only recovery, a destination row
with a missing PNG also counted as present. [Existing-name branches and source mapping](../../backend/reindex.py)

**Done when** missing, unreadable or different same-name pixels produce a
named conflict and an imported child cannot silently resolve to that
parent. Cover ordinary references and Refine through preview and import.
Intact repeat imports keep their IDs, including when remapped metadata
makes the files bytewise different. Compare decoded size and pixels and
preserve the existing provenance rules for edits and detailer images.

## GAL#36 Check the confirmed image set for broad deletes

Delete unstarred and `clear --where` recheck the preview count at execution.
In a two-image scratch gallery, A was the sole unstarred match. After
preview, starring A and unstarring B kept the count at one. The old
confirmation binned B. The CLI shares the count check and deletes
permanently. Its usage guide promises that it deletes only what its
preview showed. [Selection and write](../../backend/gallery_delete.py) ·
[CLI](../../backend/clear.py) · [Usage](../usage.md)

**Done when** a same-count replacement makes both confirmed workflows ask
for a fresh preview and affects neither image. Check membership under the
write reservation. Preserve the typed count, existing scopes, favorite
protections and API compatibility. An unchanged selection still succeeds
in both modes.

## GAL#37 Bound decoded mask memory during reindex

Reindex's mask directory scan retains every restored grayscale image in
`_Masks.pictures` until the import finishes. With 24 distinct 128-square
masks, all 24 decoded buffers were still held before the first result PNG
was inspected, in both preview and import. Some result records never need
to read the mask pixels again. [Mask restoration and lookups](../../backend/reindex.py)

**Done when** peak live decoded mask buffers stay bounded as the number of
distinct masks grows. Verify that preview and import agree on hashes,
counts and provenance for ordinary masks, embedded masks, outpaint,
detailer and crop-region proof. The acceptance check can count live
decoded buffers using a temporary gallery.

## INP#14 Recover a removed adapter after detailer region review

Detail faces/hands carries its source LoRA inside the queued recipe. If the
adapter disappears after region review, the dialog keeps that name. A probe
of the actual handler with a coded missing-LoRA reply sent the same receipt
and missing adapter on two successive presses. Inpaint and Outpaint already
refresh their adapter lists and explain when a carried adapter has gone.
[Detailer submission](../../web/detailer.js) · [Pass recovery](../../web/refine-rules.js)

**Done when** the fake dialog handles that refusal, checks whether the
adapter still exists and shows its removal before a deliberate retry.
Keep the receipt, selected regions and eligible recipe values. The retry
queues the reviewed work without the missing adapter. A listed adapter or
an unrelated refusal keeps the original selection.

## INP#15 Clear a recovered detector-status error in the detailer

After a failed status read, Refresh status populated three detector choices
in a scratch browser. The old error remained visible and the normal summary
stayed hidden. The retry updates the detector list without retiring the
previous read's message. [Status, preparation and dialog feedback](../../web/detailer.js)

**Done when** a successful retry in the same dialog clears that read's
error and restores its current summary. A second failed read still shows
the reason. A successful background refresh after a failed Verify or repair
preserves the separate action error, and a late response after Close
cannot write into another source's dialog.

## PA#19 Show negative-prompt changes before continuing a conversation

Editing only the negative prompt holds Send and offers Continue to review
the current text. The review shows the positive prompt and any focused
draft. Accepting also adopts the negative prompt for the next request,
although it wasn't displayed. [Continuation dialog](../../web/assistant-conversation.js) ·
[Request builder](../../web/prompt-assistant.js)

**Done when** the fake browser shows the exact changed negative text, or
that it was cleared, before adoption. Cancel keeps Send held. Continue
uses the displayed context, including the existing empty-to-null request
behavior. Editing again while review is open still requires a fresh review.
The writing model can stay unloaded throughout.

## UI#57 Keep keyboard focus in Assistant after Jump to latest reply

In a scratch Chromium session, a reader scrolled back while a fake reply was
pending. Activating Jump to latest reply with Enter hid the focused button
and moved focus to the document body. The next Ctrl+Enter dispatched
workspace Generate. A counter captured the dispatch before any submission.
[Jump control](../../web/assistant-conversation.js) · [Shortcut dispatch](../../web/app.js)

**Done when** pressing Jump moves keyboard focus to a meaningful visible
target inside the sidebar and the next assistant shortcut stays there.
Also deliver a reply while the user is reading earlier messages and verify
that incoming text alone leaves their focus and scroll position intact.
