// AuraDream's page script: the phone's viewer - the lightbox's picture across
// the whole screen with bars over it (styles/phone-viewer.css). The top bar
// holds the cross (drawn as a ‹), the place in the walk and Share; the foot
// the lightbox's line and a bar of five - Star, Reuse and Refine, the
// lightbox's own buttons moved in whole, then Edit and More, whose action
// sheets press the lightbox's other buttons (Inpaint…, Outpaint…, Edit by
// instruction…; Use seed, More seeds…, Download, the copies, Delete). The
// facts are a sheet over the foot, a line of the prompt until it is raised.
// A tap on the picture shows or hides the bars (lightbox.js), a swipe up
// raises the facts, and the zoom's readout shows as a chip while it
// changes. Nodes move under PHONE_MEDIA and home again off it, as phone.js's
// do. Declarations only; phone.js calls initPhoneViewer at boot and
// syncPhoneViewer whenever the tier changes, lightbox.js
// syncPhoneViewerImage after each image is drawn.
'use strict';

// The facts raised as a sheet, and the bars hidden by a tap.
let phoneFactsUp = false;
let phoneBare = false;
// The zoom's readout as it was last written, for the chip (flashPhoneZoom).
let phoneZoomWas = '';
let phoneZoomTimer = 0;
// The first walk on a phone nudges the picture once: the walk is a swipe,
// and nothing else on the screen says so.
const PHONE_NUDGE_KEY = 'auradream.viewer.nudge.v1';

// The lightbox's nodes the phone's bars hold. The line (#lb-error) and F's
// said line go in the foot, over the bar, so what a press says is seen with
// the facts lowered; the zoom's readout and its said copy over the picture.
const viewerMoved = () => [els.lbClose, els.lbCount, els.lbFav, els.lbReuse, els.lbRefine, els.lbRestore,
  els.lbError, els.lbStarSaid, els.lbZoom, els.lbZoomSaid];

// Share where the browser shares a file (a secure context: 127.0.0.1, or
// HTTPS); on plain http to a LAN address it downloads instead, and says so.
const phoneCanShare = () => window.isSecureContext
  && typeof navigator.share === 'function' && typeof navigator.canShare === 'function';

function syncPhoneViewer() {
  if (!isPhone() && $('phone-preview-viewer').open) $('phone-preview-viewer').close();
  if (isPhone()) {
    const top = $('lbp-top'), share = $('lbp-share'), bar = $('lbp-bar'), foot = $('lbp-foot');
    phoneMove(els.lbClose, top, share);
    phoneMove(els.lbCount, top, share);
    for (const node of [els.lbFav, els.lbReuse, els.lbRefine]) phoneMove(node, bar, $('lbp-edit'));
    phoneMove(els.lbRestore, bar, $('lbp-more'));
    phoneMove(els.lbError, foot, bar);
    phoneMove(els.lbStarSaid, foot, bar);
    phoneMove(els.lbZoom, els.lbMain);
    phoneMove(els.lbZoomSaid, els.lbMain);
  } else {
    for (const node of viewerMoved()) phoneHome(node);
    setPhoneFacts(false);
    setPhoneBare(false);
    els.lbReuse.removeAttribute('aria-label');
  }
  const share = $('lbp-share');
  share.textContent = phoneCanShare() ? 'Share' : 'Download';
  share.title = phoneCanShare() ? 'Share the full-size PNG, its settings in it'
    : 'Download the full-size PNG with its saved settings (sharing needs HTTPS)';
  // The image as drawn for the other tier: its Reuse's words, its layout.
  if (els.lightbox.open) renderLightbox();
}

// After each image is drawn (renderLightbox): the peek's line, Edit's
// place (an image in the bin has Restore there), Reuse's word, and the
// nudge on a first open. `opening`: the lightbox was shut before this.
function syncPhoneViewerImage(opening = false) {
  const rec = lbList[lbIndex];
  if (!rec) return;
  $('lbp-edit').hidden = !!rec.binned_at;
  const peek = $('lbp-peek');
  const words = `${rec.prompt.split('\n')[0]} · seed ${rec.seed}`;
  if (peek.textContent !== words) peek.textContent = words;
  if (!isPhone()) return;
  // One word in a fifth of the bar: "Reuse settings" is its title's, and
  // the name kept whole for a screen reader (syncReuseButton writes it).
  const long = els.lbReuse.textContent;
  if (long !== '↻ Reuse') {
    els.lbReuse.setAttribute('aria-label', long);
    els.lbReuse.textContent = '↻ Reuse';
  }
  if (opening) nudgePhoneViewer();
}

function nudgePhoneViewer() {
  if (lbList.length < 2 || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  try {
    if (localStorage.getItem(PHONE_NUDGE_KEY)) return;
    localStorage.setItem(PHONE_NUDGE_KEY, '1');
  } catch { return; }
  els.lbImg.animate?.([{ translate: '0 0' }, { translate: '-24px 0' }, { translate: '0 0' }],
    { duration: 700, delay: 500, easing: 'ease-in-out' });
}

// The facts as a sheet (`up`) or a peek over the foot. Raised, it holds one
// history entry (touch.js), so Back lowers it before it closes the viewer.
function setPhoneFacts(up) {
  const want = !!up && isPhone() && els.lightbox.open;
  if (phoneFactsUp === want) return;
  phoneFactsUp = want;
  els.lightbox.toggleAttribute('data-facts', want);
  $('lbp-handle').setAttribute('aria-expanded', String(want));
  if (want) setPhoneBare(false);
  else document.querySelector('.lb-side').scrollTop = 0;
  queueBackSync();
}

function setPhoneBare(bare) {
  const want = !!bare && isPhone() && els.lightbox.open;
  if (phoneBare === want) return;
  phoneBare = want;
  els.lightbox.toggleAttribute('data-bare', want);
  // A control hidden with its bar gives the focus back to the dialog.
  if (want && els.lightbox.contains(document.activeElement) && !isShown(document.activeElement)) {
    els.lightbox.focus({ preventScroll: true });
  }
}

// A tap on the picture (lightbox.js, after a double tap's wait): the raised
// facts lower; else the bars hide, or come back.
function phoneViewerTap() {
  if (!els.lightbox.open) return;
  if (phoneFactsUp) setPhoneFacts(false);
  else setPhoneBare(!phoneBare);
}

// The zoom's readout as a chip over the picture, for a second and a bit
// after it changes - a pinch, a double tap - but not as a walk lands on
// another image at its fit.
function flashPhoneZoom() {
  const text = els.lbZoom.textContent;
  const was = phoneZoomWas;
  phoneZoomWas = text;
  if (!isPhone() || !text || !was || (text.startsWith('fit') && was.startsWith('fit'))) return;
  els.lbZoom.classList.add('shown');
  clearTimeout(phoneZoomTimer);
  phoneZoomTimer = setTimeout(() => els.lbZoom.classList.remove('shown'), 1200);
}

async function sharePhoneImage() {
  const rec = lbList[lbIndex];
  if (!els.lightbox.open || !rec) return;
  if (!phoneCanShare()) { void downloadLightboxImage(); return; }
  try {
    const res = await fetch(rec.url);
    if (!res.ok) throw new Error(`the image answered ${res.status}`);
    const file = new File([await res.blob()], lightboxFileName(rec), { type: 'image/png' });
    if (!navigator.canShare({ files: [file] })) { void downloadLightboxImage(); return; }
    await navigator.share({ files: [file], title: `#${rec.id}` });
  } catch (err) {
    // Shut by the user: nothing to say. Refused for want of the press (the
    // fetch outlived it): the download, which needs none.
    if (err?.name === 'AbortError') return;
    if (err?.name === 'NotAllowedError') { void downloadLightboxImage(); return; }
    if (els.lightbox.open && lbList[lbIndex] === rec) {
      showError(`Could not share #${rec.id}: ${err?.message || err}.`, false, { slot: els.lbError, mirror: false });
    }
  }
}

// Edit's action sheet: the ways to change this image, each the lightbox's
// own button or the menu's own item, held as they are.
function openViewerEdit(event) {
  const rec = lbList[lbIndex];
  if (!rec) return;
  showContextMenu([
    { heading: `Edit #${rec.id}` },
    buttonItem('✎ Inpaint…', els.lbInpaint),
    buttonItem('⬚ Outpaint…', els.lbOutpaint),
    syncedItem('✎ Edit by instruction…', (b) => syncEditTool(b, rec), () => editFromCard(rec)),
  ], 'Edit', event.currentTarget, 0, 0, event.detail === 0);
}

// More's: the image's own menu (context-menu.js) less what the bar and
// Edit hold, with Download beside Open PNG.
const VIEWER_BAR_ITEMS = new Set(['☆ Star', '★ Unstar', '↻ Reuse settings', '⤢ Refine…', '✎ Inpaint…',
  '⬚ Outpaint…', '✎ Edit by instruction…', '↶ Restore']);
function openViewerMore(event) {
  const rec = lbList[lbIndex];
  if (!rec) return;
  const items = imageItems({ kind: 'lightbox', rec, host: els.lbStage })
    .filter((item) => !item || !VIEWER_BAR_ITEMS.has(item.label));
  const png = items.findIndex((item) => item?.label === 'Open PNG in new tab');
  items.splice(png < 0 ? items.length : png, 0,
    { label: 'Download', title: els.lbDownload.title, run: () => downloadLightboxImage() });
  showContextMenu([{ heading: `#${rec.id}` }, ...items], `More for #${rec.id}`, event.currentTarget, 0, 0, event.detail === 0);
}

// A live frame has no record, metadata or image actions. Its viewer follows
// only the job opened by the dock's press, even when another job starts.
// A finished image replaces the frame; its ordinary lightbox opens only on
// the explicit Open finished image press. Cancellation keeps the last frame
// labelled as a preview, never as a saved result.
let phonePreviewJob = null;
let phonePreviewRecord = null;
function syncPhoneLivePreview() {
  const dialog = $('phone-preview-viewer');
  if (!dialog.open) return;
  const frame = currentJob === phonePreviewJob
    ? els.viewer.querySelector(':scope > figure.live img.preview[src]') : null;
  const belongs = (r) => r.job_id === phonePreviewJob && !isGone(r.id);
  const rec = viewerRecords.filter(belongs).at(-1) ?? sessionRecords.find(belongs)
    ?? (phonePreviewRecord && belongs(phonePreviewRecord) ? phonePreviewRecord : null);
  phonePreviewRecord = frame ? null : rec ?? null;
  const img = $('phone-preview-image');
  const src = frame?.getAttribute('src') ?? rec?.url;
  if (src && img.getAttribute('src') !== src) img.src = src;
  const running = currentJob === phonePreviewJob && busy;
  const title = frame || running ? 'Live preview' : rec ? `Finished image #${rec.id}` : 'Preview ended';
  const note = frame ? frame.closest('figure').querySelector('.foot').textContent
    : running ? 'Waiting for the next preview…'
    : rec ? 'The finished image is ready.' : 'This preview ended without a finished image.';
  if ($('phone-preview-title').textContent !== title) $('phone-preview-title').textContent = title;
  if ($('phone-preview-note').textContent !== note) $('phone-preview-note').textContent = note;
  img.alt = rec && !frame ? `Finished image #${rec.id}` : 'Live preview of the image being generated';
  $('phone-preview-result').hidden = !phonePreviewRecord;
}

function openPhoneLivePreview() {
  const dialog = $('phone-preview-viewer');
  if (!isPhone() || dialog.open || !currentJob || !els.viewer.querySelector(':scope > figure.live img.preview[src]')) return;
  phonePreviewJob = currentJob;
  phonePreviewRecord = null;
  dialog.showModal();
  syncPhoneLivePreview();
  $('phone-preview-close').focus({ preventScroll: true });
}

function initPhoneViewer() {
  const preview = $('phone-preview-viewer');
  $('phone-preview-close').addEventListener('click', () => preview.close());
  preview.addEventListener('close', () => {
    if (preview.open) return;
    // A late close event from the handoff must not steal the lightbox's focus.
    phonePreviewJob = null;
    phonePreviewRecord = null;
    $('phone-preview-image').removeAttribute('src');
    if (!els.lightbox.open && isPhone()) $('phone-dock').focus({ preventScroll: true });
  });
  $('phone-preview-result').addEventListener('click', () => {
    const rec = phonePreviewRecord;
    if (!rec || isGone(rec.id)) return;
    preview.close();
    openLightbox(rec, [rec], 'dock');
  });
  const progressWatch = new MutationObserver(syncPhoneLivePreview);
  for (const node of [els.progressText, els.stepText]) {
    progressWatch.observe(node, { childList: true, characterData: true, subtree: true });
  }
  for (const node of viewerMoved()) {
    const mark = document.createComment(` ${node.id}'s place off the phone (phone-viewer.js) `);
    node.before(mark);
    phoneHomes.set(node, mark);
  }
  $('lbp-share').addEventListener('click', () => { void sharePhoneImage(); });
  $('lbp-edit').addEventListener('click', openViewerEdit);
  $('lbp-more').addEventListener('click', openViewerMore);
  $('lbp-handle').addEventListener('click', () => setPhoneFacts(!phoneFactsUp));
  // Shut, the viewer starts again with its bars and its facts lowered.
  els.lightbox.addEventListener('close', () => {
    setPhoneFacts(false);
    setPhoneBare(false);
    clearTimeout(phoneZoomTimer);
    els.lbZoom.classList.remove('shown');
  });
  new MutationObserver(flashPhoneZoom).observe(els.lbZoom, { childList: true, characterData: true, subtree: true });
  // The facts' peek sits on the foot, whose height is the bar's and the
  // line's over it when it speaks.
  const foot = $('lbp-foot');
  new ResizeObserver(() => {
    els.lightbox.style.setProperty('--lbp-foot-h', `${foot.offsetHeight}px`);
  }).observe(foot);
}
