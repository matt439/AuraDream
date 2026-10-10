// Local Diffusion's page script: a gallery card's batch, shown on hover.
// Declarations only; state.js has the load-order contract these files share.
'use strict';

// ------------------------------------------------------------ batch peek

// A card held under the mouse for a moment shows the images made with it -
// its batch: the same job, and the same settings but for the seed (`sameExcept`,
// sheet.js, with Seed skipped), so a Batch 4 and its repeats are one set and a
// grid's other cells are not. A small panel of their pictures, over the page
// and never under the pointer's press (`pointer-events: none`), with the
// gallery's own cards of the set outlined. Worked out in the page from the
// records it holds and one gallery search per job (`job:=<id>`), as the
// lightbox's CFG ladder is: the grouping is presentation, so no route of its
// own. A card alone in its job shows nothing.
const PEEK_DELAY = 350;
// One page of the job's images; past it the panel says more were made.
const PEEK_LIMIT = 200;
// The pictures the panel draws; past them a "+N more" in their place.
const PEEK_SHOWN = 12;
// How long a job's answer stands: a job still running lands more images, and
// a hover a while later should see them.
const PEEK_FRESH_MS = 15000;

// What the gallery holds of each job: { records, more, at }.
const peekJobs = new Map();
// The job ids being asked for, so two hovers do not ask twice.
const peekFlights = new Map();
let peekTimer;
// The card the pointer is on, and the one the panel was drawn for.
let peekOver = null;
let peekShown = null;
// Where peekOver stood when the pointer reached it, so a gallery that moves
// it (cards prepended or deleted) is told from one that does not.
let peekAt = null;

// The images of `rec`'s batch the page holds, `rec` among them, oldest first
// (one generator per image, seeded `seed + i`, so by id is by seed too).
function batchMates(rec) {
  if (!rec?.job_id) return [];
  const fetched = peekJobs.get(rec.job_id)?.records ?? [];
  const seen = new Set();
  const mates = [];
  for (const r of [rec, ...galleryRecords, ...viewerRecords, ...sessionRecords, ...fetched]) {
    if (seen.has(r.id) || deletedIds.has(r.id)) continue;
    seen.add(r.id);
    if (r.job_id === rec.job_id && sameExcept(r, rec, ['seed'])) mates.push(r);
  }
  return mates.sort((a, b) => a.id - b.id);
}

// The job's images from the gallery, in their checkpoint's exact scope. A
// failure leaves the set to what the page holds.
async function fetchPeekJob(rec) {
  const job = rec.job_id;
  const held = peekJobs.get(job);
  if (held && Date.now() - held.at < PEEK_FRESH_MS) return;
  if (peekFlights.has(job)) return peekFlights.get(job);
  const params = new URLSearchParams({ limit: String(PEEK_LIMIT), model: rec.model_id, search: `job:=${job}` });
  const flight = api(`/api/gallery?${params}`)
    .then((page) => peekJobs.set(job, { records: page.items, more: !!page.has_more, at: Date.now() }))
    .catch(() => {})
    .finally(() => peekFlights.delete(job));
  peekFlights.set(job, flight);
  return flight;
}

const peekRecord = (cardEl) => galleryRecords.find((r) => r.id === +cardEl.dataset.id);

// The pointer onto a card (delegated from #gallery, app.js). The mouse only:
// a tap's pointerover would leave the panel up with nothing to take it down.
function batchPeekOver(e) {
  if (e.pointerType && e.pointerType !== 'mouse') return;
  const cardEl = e.target.closest?.('.card');
  if (cardEl === peekOver) return;
  hideBatchPeek();
  if (!cardEl || cardEl.classList.contains('gone') || sheetMode) return;
  const rec = peekRecord(cardEl);
  if (!rec?.job_id) return;
  peekOver = cardEl;
  const { top, left } = cardEl.getBoundingClientRect();
  peekAt = { top, left };
  peekTimer = setTimeout(async () => {
    await fetchPeekJob(rec);
    if (peekOver === cardEl && cardEl.isConnected) showBatchPeek(cardEl, rec);
  }, PEEK_DELAY);
}

// Off the card, to anything that is not inside it.
function batchPeekOut(e) {
  if (!peekOver || (e.relatedTarget && peekOver.contains(e.relatedTarget))) return;
  hideBatchPeek();
}

function hideBatchPeek() {
  clearTimeout(peekTimer);
  peekOver = null;
  peekShown = null;
  peekAt = null;
  const panel = $('batch-peek');
  if (panel) panel.hidden = true;
  els.gallery.querySelectorAll('.card.batch-mate').forEach((el) => el.classList.remove('batch-mate'));
}

function batchPeekPanel() {
  let panel = $('batch-peek');
  if (panel) return panel;
  panel = document.createElement('div');
  panel.id = 'batch-peek';
  panel.className = 'batch-peek';
  // The cards it repeats are the page's own, each named and reachable;
  // the panel is a picture of them for the pointer.
  panel.setAttribute('aria-hidden', 'true');
  panel.hidden = true;
  document.body.append(panel);
  return panel;
}

function showBatchPeek(cardEl, rec) {
  const mates = batchMates(rec);
  if (mates.length < 2) return;
  const more = peekJobs.get(rec.job_id)?.more;
  const panel = batchPeekPanel();
  const at = mates.findIndex((r) => r.id === rec.id);
  const head = document.createElement('div');
  head.className = 'batch-peek-head';
  const seeds = mates.map((r) => r.seed);
  // Past PEEK_LIMIT the seeds are the held ones' only, and the range says so.
  head.textContent = `#${rec.id} · ${at + 1} of ${mates.length}${more ? '+' : ''} in its batch`
    + (Number.isSafeInteger(seeds[0]) ? ` · seeds ${Math.min(...seeds)}–${Math.max(...seeds)}${more ? '…' : ''}` : '');
  const list = document.createElement('div');
  list.className = 'batch-peek-list';
  // The window of PEEK_SHOWN round the hovered image, so it is always drawn.
  const start = Math.max(0, Math.min(at - Math.floor(PEEK_SHOWN / 2), mates.length - PEEK_SHOWN));
  const shown = mates.slice(start, start + PEEK_SHOWN);
  for (const r of shown) {
    const img = document.createElement('img');
    img.src = r.thumb_url ?? r.url;
    img.alt = '';
    img.decoding = 'async';
    img.style.aspectRatio = `${r.width} / ${r.height}`;
    img.classList.toggle('current', r.id === rec.id);
    list.append(img);
  }
  const rest = mates.length - shown.length;
  if (rest > 0 || more) {
    const note = document.createElement('span');
    note.className = 'batch-peek-more';
    note.textContent = rest > 0 ? `+${rest} more${more ? '…' : ''}` : 'and more';
    list.append(note);
  }
  panel.replaceChildren(head, list);
  panel.hidden = false;
  placeBatchPeek(panel, cardEl);
  peekShown = cardEl;
  const ids = new Set(mates.map((r) => r.id));
  els.gallery.querySelectorAll('.card').forEach((el) => {
    el.classList.toggle('batch-mate', el !== cardEl && ids.has(+el.dataset.id));
  });
}

// Under the card where it fits in the window, else over it, its left edge
// with the card's and kept inside the window: never across the card itself.
function placeBatchPeek(panel, cardEl) {
  const margin = 8;
  // The window less its scrollbar, which innerWidth counts.
  const { clientWidth: across, clientHeight: down } = document.documentElement;
  const box = cardEl.getBoundingClientRect();
  panel.style.left = '0px';
  panel.style.top = '0px';
  const { width, height } = panel.getBoundingClientRect();
  const below = box.bottom + margin;
  const top = below + height <= down - margin || box.top - margin - height < margin
    ? below : box.top - margin - height;
  const left = Math.max(margin, Math.min(box.left, across - margin - width));
  panel.style.left = `${Math.round(left)}px`;
  panel.style.top = `${Math.round(Math.max(margin, top))}px`;
}

// A scroll moves the card out from under the panel; it goes, and comes back
// with the next card the pointer settles on.
function batchPeekScrolled() {
  if (peekOver || peekShown) hideBatchPeek();
}

// Cards prepended or deleted under a still mouse move the card from under
// the panel as a scroll does (a MutationObserver on #gallery, app.js); the
// next pointer move brings it back for whatever card is there.
function batchPeekShifted() {
  if (!peekOver) return;
  const { top, left } = peekOver.getBoundingClientRect();
  if (!peekOver.isConnected || top !== peekAt.top || left !== peekAt.left) hideBatchPeek();
}

// A press is for the card, not its batch: the panel goes until the pointer
// settles on a card again.
function batchPeekPressed() {
  if (peekOver || peekShown) hideBatchPeek();
}
