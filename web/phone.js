// AuraDream's page script: the phone's own places for nodes the page holds
// once (styles/phone.css draws them) - Generate in the Create view's action
// bar over the tab bar, the results under the settings that make them, the
// running job and the queue in a sheet - each moved there whole under
// PHONE_MEDIA, never copied, and home again off it, so `els`, `setQueue`
// and every writer keep writing the one node; and the bar's newest result
// and queue count. Declarations only; touch.js calls initPhone at boot and
// syncPhoneHosts whenever the tier changes.
'use strict';

// Each moved node's home: a comment left where it stood in the markup.
const phoneHomes = new Map();

// `node` into `host` before `before` (its end when null), with `moveBefore`
// where the browser has it, which keeps the node's state - its focus, a
// live region's - and elsewhere the focus put back by hand, as
// placeTagValidation does.
function phoneMove(node, host, before = null) {
  if (node.parentNode === host && (before ? node.nextSibling === before : true)) return;
  const focused = document.activeElement;
  const inside = node.contains(focused);
  let moved = false;
  if (typeof host.moveBefore === 'function') {
    try { host.moveBefore(node, before); moved = true; } catch { /* inserted below */ }
  }
  if (!moved) host.insertBefore(node, before);
  if (inside && focused.isConnected && focused !== document.activeElement) focused.focus({ preventScroll: true });
}

function phoneHome(node) {
  const mark = phoneHomes.get(node);
  if (mark && mark.nextSibling !== node) phoneMove(node, mark.parentNode, mark.nextSibling);
}

// The nodes that move, in the order each host holds them.
const phoneResultNodes = () => [els.viewer, els.viewerNote, els.recent];
const phoneQueueNodes = () => [els.progress, els.queueNote];

// On the phone, Generate in the bar before the queue's count and the
// results in their host; off it, both home and the queue's sheet shut, its
// nodes home too. From syncTier (touch.js): at boot and on every change.
function syncPhoneHosts() {
  if (isPhone()) {
    phoneMove(els.generate, $('phone-actions'), $('phone-queue'));
    for (const node of phoneResultNodes()) {
      if (node.parentNode !== $('phone-results')) phoneMove(node, $('phone-results'));
    }
  } else {
    const sheet = $('queue-sheet');
    if (sheet.open) sheet.close();
    for (const node of [els.generate, ...phoneResultNodes(), ...phoneQueueNodes()]) phoneHome(node);
  }
  syncPhoneDock();
  syncPhoneQueue();
}

// ------------------------------------------------------- the newest result

// What the bar's picture shows, and so what its press opens: 'live' for
// the running job's live preview, else the newest result's record, else
// null (nothing made on this page yet).
let phoneDockShows = null;

// The newest result: the viewer's last image not gone, else the newest of
// this page's earlier results (#recent's), with the list the lightbox
// walks from it.
function phoneNewest() {
  for (let i = viewerRecords.length - 1; i >= 0; i--) {
    if (!isGone(viewerRecords[i].id)) return { rec: viewerRecords[i], list: viewerRecords };
  }
  const rec = sessionRecords.find((r) => !isGone(r.id));
  return rec ? { rec, list: sessionRecords } : null;
}

// The picture, its ring and its words, from what the page shows already:
// the viewer's live preview and images (watched), and the topbar's job line
// and fraction, which job-progress.js writes (watched, as the tab bar's
// dot watches them).
function syncPhoneDock() {
  const dock = $('phone-dock');
  const img = $('phone-dock-img');
  const running = !els.topbarLine.hidden;
  const frame = els.viewer.querySelector(':scope > figure.live img.preview[src]');
  const newest = phoneNewest();
  phoneDockShows = frame ? 'live' : newest?.rec ?? null;
  const src = frame ? frame.getAttribute('src') : newest ? newest.rec.thumb_url ?? newest.rec.url : '';
  if (src && img.getAttribute('src') !== src) img.src = src;
  if (!src) img.removeAttribute('src');
  img.hidden = !src;
  dock.classList.toggle('running', running);
  dock.classList.toggle('live', !!frame);
  dock.classList.toggle('indeterminate', running && els.topbarLine.classList.contains('indeterminate'));
  const fraction = parseFloat(els.topbarFill.style.width) / 100;
  dock.style.setProperty('--ring', String(running && fraction > 0 ? Math.min(1, fraction) : 0));
  const step = running ? progressShown() : '';
  const job = running ? ` - a job running${step ? `, ${step}` : ''}` : '';
  let held = '';
  let title = '';
  if (phoneDockShows === 'live') title = 'Show the live preview in the results';
  else if (phoneDockShows) title = `Open #${phoneDockShows.id}, the newest result`;
  else held = running ? 'No image of this job yet' : 'Nothing generated on this page yet';
  dock.setAttribute('aria-disabled', String(!!held));
  dock.title = held || title;
  dock.setAttribute('aria-label', held ? `Newest result: ${held[0].toLowerCase()}${held.slice(1)}${job}` : `${title}${job}`);
}

// A press: the newest result in the lightbox, walking the viewer's strip
// (or this page's earlier results) from it, as the result's own picture
// does; the live preview's tile in the results, scrolled to by this press
// as New image's was (goToNewImage), while that is what the picture shows.
// Held, it says why above the bars. Never by itself.
function pressPhoneDock() {
  const dock = $('phone-dock');
  if (dock.getAttribute('aria-disabled') === 'true') {
    showError(`${dock.title}.`, true);
    sayAtGenerate();
    return;
  }
  if (phoneDockShows === 'live') { goToNewImage(); return; }
  const newest = phoneNewest();
  if (!newest) return;
  if (newest.list === viewerRecords) openViewerImage(newest.rec, 'dock');
  else openLightbox(newest.rec, sessionRecords, 'dock');
}

// --------------------------------------------------------------- the queue

// The count of jobs waiting behind the running one, on the bar's queue
// button, and its hold while nothing runs or waits.
function syncPhoneQueue() {
  const button = $('phone-queue');
  const count = $('phone-queue-count');
  const running = !els.progress.hidden;
  const waiting = els.queueNote.hidden ? 0 : els.queueList.children.length;
  count.hidden = !waiting;
  count.textContent = waiting > 99 ? '99+' : String(waiting);
  button.classList.toggle('running', running);
  const held = !running && !waiting;
  const words = [running ? 'a job running' : '', waiting ? `${waiting} waiting` : ''].filter(Boolean).join(', ');
  button.setAttribute('aria-disabled', String(held));
  button.title = held ? 'Nothing running or queued' : `Open the queue: ${words}`;
  button.setAttribute('aria-label', `Queue: ${held ? 'nothing running or queued' : words}`);
}

// The sheet holds the running job's progress (Pause, Skip, Cancel, the
// bars) and the queue's rows while it is open: the panel's own nodes, moved
// in and home again as it closes (by its ✕, Escape, Back, or the phone
// going). Held, the press says why above the bars.
function openQueueSheet() {
  const button = $('phone-queue');
  const sheet = $('queue-sheet');
  if (button.getAttribute('aria-disabled') === 'true') {
    showError(`${button.title}.`, true);
    sayAtGenerate();
    return;
  }
  if (!isPhone() || sheet.open) return;
  for (const node of phoneQueueNodes()) phoneMove(node, $('queue-sheet-body'));
  sheet.showModal();
}

function queueSheetClosed() {
  for (const node of phoneQueueNodes()) phoneHome(node);
}

function initPhone() {
  for (const node of [els.generate, ...phoneResultNodes(), ...phoneQueueNodes()]) {
    const mark = document.createComment(` ${node.id}'s place off the phone (phone.js) `);
    node.before(mark);
    phoneHomes.set(node, mark);
  }
  $('phone-dock').addEventListener('click', pressPhoneDock);
  $('phone-queue').addEventListener('click', openQueueSheet);
  $('queue-sheet-close').addEventListener('click', () => $('queue-sheet').close());
  $('queue-sheet').addEventListener('close', queueSheetClosed);
  const dockWatch = new MutationObserver(syncPhoneDock);
  dockWatch.observe(els.viewer, { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] });
  dockWatch.observe(els.recent, { childList: true, subtree: true });
  dockWatch.observe(els.topbarLine, { attributes: true, attributeFilter: ['hidden', 'class'] });
  dockWatch.observe(els.topbarFill, { attributes: true, attributeFilter: ['style'] });
  const queueWatch = new MutationObserver(syncPhoneQueue);
  queueWatch.observe(els.progress, { attributes: true, attributeFilter: ['hidden'] });
  queueWatch.observe(els.queueNote, { attributes: true, attributeFilter: ['hidden'] });
  queueWatch.observe(els.queueList, { childList: true });
}
