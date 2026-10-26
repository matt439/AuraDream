# More seeds: a direct, safe next step (GEN#23)

## Evidence and scope

Matt's 2026-10-26 request identifies obscure More seeds controls and apparent
inability to queue while generating. An isolated browser probe reproduced
successful active-run queuing: saved seeds 100–105, a running job holding
106–111, and one four-image extension assigned 112–115. The original job
remained watched and the extension appeared in Next. Running itself is not
a refusal; the existing admission lock and planned keys already allocate
distinct seeds across requests.

The visible problems are concrete: multi-image results lack a direct action;
the footer requires a selection or filter and looks like quiet text; the
dialog explains internal batch terminology before its simple count; one
ready row sits in a large mostly empty list; and a ready preview says
"queued" before submission. An unrelated pending Generate POST can also
leave Queue looking enabled although its handler returns silently. Static
review found no request-intent receipt for retry after a lost response.

## Contract

- Make More seeds reachable beside current results at ordinary and narrow
  widths, with an explicit target. Keep image, selected groups and whole-run
  meanings distinct; never extend unseen grid settings under a misleading
  label. Preserve left-click opening, modifier selection and range anchors.
- Lead the dialog with additional images and unchanged settings. Use plain
  "settings group" wording where more than one group matters. One count is
  images per unique group, never repeats or per selected image. Preserve
  original execution Batch, source, mask, crop, LoRA, checkpoint and run.
  Keep refusal details and multiple-group totals visible, with technical
  seed/batch detail secondary. Keep controls stable during asynchronous
  previews and refusals, keyboard operation and focus return.
- Allow queuing while a run is active. Distinguish that from a POST still
  pending; its hold must be visible and explained. Bound rapid presses and
  cancel or invalidate stale opening previews when their originating view
  goes away. A dismissed or changed context must not open an obsolete modal.
- Give one submission intent a bounded server receipt: replay of that same
  accepted intent returns its original job IDs; a new deliberate action is
  a new intent and allocates the next free seeds. Preserve uncertain intent
  on a lost response, so retry cannot silently add another extension. Keep
  scope/model guards and shared admission ordering. Do not claim durable
  exactly-once behavior across a server restart without durable evidence.
- No real weights, image generation on the GPU, model downloads, push or
  deployment. Use synthetic fake data and bounded serial browser checks.

## Verification

Independent static review; focused CPU tests for replay, conflicting reuse,
concurrency and existing seed allocation; browser checks for active Batch ×
Repeat, two intentional extensions, rapid presses, delayed/lost replies,
another tab, watched-job transitions, unrelated composer edits, stale
opening cancellation, narrow/normal/wide reachability and unchanged picks.
Use isolated galleries. Record exact passed checks and remaining limits here
before removing the TODO row in its scoped local commit.

## Status

Built and independently reviewed, 2026-10-26. The persistent results action
names the generated settings it extends, including while later images are
still running. The compact dialog leads with additional images and unchanged
settings; multiple groups and refusals remain explicit. Technical seed/Batch
details are secondary. Queue and Cancel keep fixed rectangles, and the
result caption reserves its line through image arrival and wide docking.

Preview-issued receipts bind the count, scope, expected ids and checkpoint.
One owned admission survives a disconnected caller; accepted retries recover
the original jobs. Distinct intentional requests allocate distinct seeds
under shared admission. A partial admission acknowledges the original jobs,
the unqueued count and a warning visible even over the lightbox. Receipts
are bounded to 512 entries, issued for 15 minutes and kept for one hour after
settlement. Expired/unknown tokens refuse; this is not durable execution
across a server restart.

Independent review found that a generic 409 after a lost accepted response
could discard the original intent. That path now preserves the exact body
and token through later errors; its regression checks lost reply, 503,
generic middleware 409 and successful recovery with no new preview. Opening
previews also reject stale results/lightbox/modal contexts and Escape cancels
them. Original Batch, settings and modifier selection behavior are kept.

Browser checks passed serially on isolated fake galleries: `smoke-more-seeds`,
`smoke-more-seeds-tabs`, `smoke-select`, `smoke-results`, `smoke-jobs`,
`smoke-refine`, `smoke-inpaint`, `smoke-outpaint`, `smoke-lightbox`,
`smoke-context-menu` and `smoke-queue-controls`. The two actual tabs prove
simultaneous previews allocate 1012-1013 and 1014-1015 behind paused work,
retry adds no extension, and each tab keeps its own composer/results through
queue handoff. Normal, narrow and wide reachability and compact-dialog
screenshots were checked. No real model, card acceptance or download is
claimed; hands-on accessibility checks remain `UI#7`.

The full CPU suite passed: **3745 passed, 5 skipped**; Ruff and
`git diff --check` passed. Receipt:
`%TEMP%/local-diffusion-ux-full-cpu-20261026-f1.log`. The first sandboxed
full-suite attempts were interrupted after Windows process creation stalled
on an existing one-byte dummy-executable test. That test passed outside the
sandbox, followed by this complete suite; no product workaround was added.

Receipts are in `%TEMP%/local-diffusion-ux-more-final-20261026.log`,
`local-diffusion-ux-caption-tabs-final-20261026.log`,
`local-diffusion-ux-tabs-regressions-20261026.log` (the lightbox/context-menu/
queue-control passes), and `local-diffusion-ux-resume-browser-20261026.log`
(selection pass). Earlier failures in those latter two receipts were fixed
and rerun as recorded above.
