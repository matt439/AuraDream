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

Investigation complete; implementation not started. The browser probe above
used the committed queue/click baseline and synthetic images. The current
size/crop batch is being qualified first.
