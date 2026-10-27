// Auradream's page script: a gallery card's batch, marked on hover.
// Declarations only; state.js has the load-order contract these files share.
'use strict';

// ------------------------------------------------------------ batch mates

// A card the mouse rests on faintly outlines the gallery's other cards of its
// batch: the same run (the job, for a record from before runs had ids) and the
// same settings but for the seed (`sameExcept`, sheet.js, with Seed skipped),
// so a Batch 4, its repeats and the More seeds that extended it (which join
// its run) are one set and a grid's other cells are not - the set More seeds
// takes as one batch. Only the cards the page holds, from the records it
// holds: presentation, so no request and no route. An outline, not a border,
// so nothing moves. A card alone in its run marks nothing.
const MATE_DELAY = 150;

let mateTimer;
// The card the pointer is on.
let mateOver = null;

const mateRun = (rec) => (rec?.run_id ? `run:${rec.run_id}` : rec?.job_id ? `job:${rec.job_id}` : null);

// The pointer onto a card (delegated from #gallery, app.js). The mouse only:
// a tap's pointerover would leave the marks with nothing to take them down.
function batchMatesOver(e) {
  if (e.pointerType && e.pointerType !== 'mouse') return;
  const cardEl = e.target.closest?.('.card');
  if (cardEl === mateOver) return;
  clearBatchMates();
  // Not in the recycle bin, whose images are no batch's any longer (bin.js),
  // nor the sheet, whose cells are not cards.
  if (!cardEl || cardEl.classList.contains('gone') || sheetMode || binMode) return;
  mateOver = cardEl;
  mateTimer = setTimeout(() => markBatchMates(cardEl), MATE_DELAY);
}

// Off the card, to anything that is not inside it.
function batchMatesOut(e) {
  if (!mateOver || (e.relatedTarget && mateOver.contains(e.relatedTarget))) return;
  clearBatchMates();
}

function clearBatchMates() {
  clearTimeout(mateTimer);
  mateOver = null;
  els.gallery.querySelectorAll('.card.batch-mate').forEach((el) => el.classList.remove('batch-mate'));
}

function markBatchMates(cardEl) {
  if (mateOver !== cardEl || !cardEl.isConnected) return;
  const byId = new Map(galleryRecords.map((r) => [r.id, r]));
  const rec = byId.get(+cardEl.dataset.id);
  const run = mateRun(rec);
  if (!run) return;
  for (const el of els.gallery.querySelectorAll('.card')) {
    if (el === cardEl || el.classList.contains('gone')) continue;
    const r = byId.get(+el.dataset.id);
    if (r && !isGone(r.id) && mateRun(r) === run && sameExcept(r, rec, ['seed'])) el.classList.add('batch-mate');
  }
}
