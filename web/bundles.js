// Gallery archive/import controls. Declarations only; app.js owns the wiring.
'use strict';

let bundlePreview = null;
let bundleStatus = null;
let bundleSending = false;
let bundleReading = false;
let bundleSettling = false;
let bundleRequestEpoch = 0;
let bundlePollPending = false;
let bundlePollTimer = 0;
let bundleHandled = null;
// Whether this page has had any status yet, and has seen the current
// operation run (or started it). A finished operation that the first status
// already holds ended before this page loaded: its gallery is already current.
let bundleStatusSeen = false;
let bundleSeenRunning = false;
// Whether the handled result has been on screen in the open dialog.
let bundleResultShown = false;
let bundleListKey = null;
let bundlePreviewEpoch = 0;
// The selected file's first bytes: a protected bundle's header, checked
// against the password before a large upload starts. Null until read.
let bundleFileHeader = null;
let bundleFileEpoch = 0;
// The AbortController of an upload whose body (or password check) is still
// being sent; null otherwise. Once the server answers 202 the import is its own.
let bundleUploadAbort = null;

// What a bundle holds: the entire gallery, which Bundle & clear alone can
// clear, or the picked or the starred images, which are only ever backed up.
// The server resolves both (`submission.resolve_selection`); the page sends
// the picked ids and shows the counts.
const BUNDLE_SCOPE_HINTS = {
  all: "Includes every model, every starred image and the recycle bin. The gallery's search and filters do not limit this action.",
  picked: 'Only the picked images, whatever the filter, with their thumbnails. The gallery is kept. A refine source or a reference travels only if it is picked too.',
  starred: 'The starred images of every model, outside the recycle bin, with their thumbnails. The gallery is kept. A refine source or a reference travels only if it is starred too.',
};

const BUNDLE_MAGIC = 'LDBUNDLE';
const BUNDLE_HEADER_BYTES = 64;
const BUNDLE_PHASES = {
  writing: 'Writing the bundle', verifying: 'Verifying the bundle', checking: 'Checking for changes made outside the app',
  clearing: 'Clearing the gallery', receiving: 'Receiving the upload', merging: 'Merging gallery records',
  installing: 'Installing images',
};

function bundleProtected(bytes) {
  return !!bytes && String.fromCharCode(...bytes.slice(0, BUNDLE_MAGIC.length)) === BUNDLE_MAGIC;
}

function bundleBase64(bytes) {
  let text = '';
  for (const byte of bytes) text += String.fromCharCode(byte);
  return btoa(text);
}

function bundleProgressText(progress) {
  if (!progress) return '';
  const label = BUNDLE_PHASES[progress.phase] || progress.phase;
  if (progress.phase === 'receiving') {
    return progress.total ? `${label}: ${humanBytes(progress.done)} of ${humanBytes(progress.total)}` : `${label}: ${humanBytes(progress.done)}`;
  }
  return progress.total && progress.done ? `${label}: ${progress.done} of ${progress.total}` : `${label}…`;
}

// "1 picked image", "3 starred images".
function bundleImages(n, kind) {
  return `${n} ${kind} image${n === 1 ? '' : 's'}`;
}

function bundleScope() {
  return document.querySelector('input[name="bundle-scope"]:checked')?.value || 'all';
}

// The gallery's picks: set aside while the recycle bin is shown, whose own
// picks are for Restore picked, so those are the ones a bundle takes.
function bundlePickedIds() {
  return binMode ? [...(stashedSelection?.picks.keys() ?? [])] : selectedIds();
}

function bundleHold() {
  return bundleSending || bundleStatus?.state === 'running' ? 'Wait for the gallery bundle operation to finish' : '';
}

function bundleActionHold() {
  return bundleHold() || (bundleSettling ? 'Refreshing the gallery after the bundle operation' : '')
    || (bundleReading ? 'Wait for the gallery counts' : '') || (serverOffline ? 'The server is not answering' : '')
    || (busy || queuedIds.length || elsewhereBusy || submitting ? 'Wait for all generation jobs to finish' : '')
    || (expandingPrompt || critiquing ? 'Wait for prompt expansion or the critic to finish' : '')
    || modelChangeHold();
}

function bundleSay(message, failed = false) {
  if (els.bundleStatus.textContent !== message) els.bundleStatus.textContent = message;
  els.bundleStatus.classList.toggle('failed', failed);
}

// Phase and counts go beside the live region, not in it: rewritten at every
// poll, they made a screen reader repeat the whole sentence each tick.
function bundleProgressSay(text) {
  if (els.bundleProgress.textContent !== text) els.bundleProgress.textContent = text;
}

function syncBundleControls() {
  const why = bundleActionHold();
  const hold = (el, reason, title) => {
    el.setAttribute('aria-disabled', String(!!reason));
    el.title = reason || title;
  };
  const count = bundlePreview?.records;
  const confirmed = Number.isInteger(count) && els.bundleCount.value.trim() === String(count);
  const password = els.bundlePassword.value;
  const locked = !els.bundleUnlock.value ? 'Enter the password for this protected bundle' : '';
  const scope = bundleScope();
  const picked = bundlePickedIds().length;
  const starred = bundlePreview?.starred;
  const label = (el, text) => { if (el.textContent !== text) el.textContent = text; };
  label(els.bundleScopePicked, `Picked (${picked})`);
  label(els.bundleScopeStarred, Number.isInteger(starred) ? `Starred (${starred})` : 'Starred');
  label(els.bundleScopeHint, BUNDLE_SCOPE_HINTS[scope]);
  const differ = password !== els.bundlePasswordAgain.value ? 'The two passwords differ' : '';
  hold(els.bundleExport, why || (bundleReading ? 'Wait for the gallery counts' : '')
    || (scope !== 'all' ? 'Only the entire gallery is cleared; Back up saves the picked or the starred images and keeps them' : '')
    || (!bundlePreview ? 'Refresh the gallery counts first' : '')
    || differ || (!confirmed ? `Type ${count} to confirm the entire gallery` : ''),
    password ? 'Save, encrypt and verify a bundle, then clear the entire gallery' : 'Save and verify a bundle, then clear the entire gallery');
  const what = scope === 'all' ? 'the entire gallery' : `the ${bundleImages(scope === 'picked' ? picked : starred, scope)}`;
  hold(els.bundleBackup, why || (bundleReading ? 'Wait for the gallery counts' : '')
    || (!bundlePreview ? 'Refresh the gallery counts first' : '')
    || (scope === 'picked' && !picked ? 'No cards are picked; pick them with Select first' : '')
    || (scope === 'starred' && !starred ? 'No image is starred' : '') || differ,
    `${password ? 'Save, encrypt and verify' : 'Save and verify'} a bundle of ${what}; the gallery is kept`);
  hold(els.bundleRefresh, bundleHold() || (bundleReading ? 'Reading the gallery counts' : ''), 'Read the current gallery counts');
  hold(els.bundleUpload, why || (!els.bundleFile.files.length ? 'Choose a bundle file first' : '')
    || (bundleProtected(bundleFileHeader) ? locked : ''), 'Merge this file into the gallery');
  hold(els.bundleUploadCancel, bundleUploadAbort && bundleSending ? '' : 'No upload is being sent',
    'Stop sending the selected file; nothing is imported');
  for (const button of els.bundleList.querySelectorAll('[data-bundle-import]')) {
    hold(button, why || ('bundleEncrypted' in button.dataset ? locked : ''), 'Merge this saved bundle into the gallery');
  }
}

function bundleRefused(button) {
  syncBundleControls();
  if (button.getAttribute('aria-disabled') !== 'true') return false;
  bundleSay(button.title + '.', true);
  return true;
}

function bundleFileURL(name) {
  return `/api/gallery/bundles/files/${encodeURIComponent(name)}`;
}

async function refreshBundlePreview() {
  if (bundleHold()) return;
  const epoch = ++bundlePreviewEpoch;
  bundleReading = true;
  syncBundleControls();
  try {
    const preview = await api('/api/gallery/bundles/preview');
    if (epoch !== bundlePreviewEpoch) return;
    if (bundlePreview?.records !== preview.records) els.bundleCount.value = '';
    bundlePreview = preview;
    // The recycle bin's rows are records too, and go into the bundle as binned.
    const binned = preview.binned ? ` (${preview.binned} in the recycle bin)` : '';
    els.bundlePreview.textContent = `${preview.records} gallery records${binned} · ${preview.images} full-size images · ${preview.thumbnails} thumbnails · ${humanBytes(preview.bytes)}`;
    els.bundleCountLabel.textContent = `Type ${preview.records} to confirm all gallery records will be removed`;
  } catch (err) {
    if (epoch !== bundlePreviewEpoch) return;
    bundlePreview = null;
    els.bundlePreview.textContent = 'The gallery counts could not be read.';
    // A row with no file names its id (`bundles.MissingImage`): Show it,
    // inline, so nothing below the line moves.
    const missing = err.body?.image_id;
    if (Number.isInteger(missing)) {
      const show = document.createElement('button');
      show.type = 'button';
      show.className = 'ghost small';
      show.id = 'bundle-show-missing';
      show.textContent = `Show #${missing}`;
      show.title = `Close this and search the ${err.body.binned ? 'recycle bin' : 'gallery'} for #${missing}`;
      show.onclick = () => showBundleMissing(missing, !!err.body.binned);
      els.bundlePreview.append(' ', show);
    }
    bundleSay(err.message, true);
  } finally {
    if (epoch === bundlePreviewEpoch) { bundleReading = false; syncBundleControls(); }
  }
}

// Show #N: the dialog closes and the gallery - the recycle bin, for a binned
// row - is searched for the one id the preview named, every checkpoint's and
// Starred off (the Models picker's See all way), so its card is there to bin
// or delete permanently. Its thumbnail can outlive the PNG; a filename is no
// search key.
function showBundleMissing(id, binned) {
  showBundleImages(`id:${id}`, binned);
}

// A deliberate navigation from the dialog: discard the old search, sheet
// return and filters so every model's matching rows can be reached. The
// bin has its own view; a binned-only import never opens an empty live one.
function showBundleImages(search, binned) {
  els.bundlesDialog.close();
  outsideView = null;
  if (!binned) leaveBin();
  els.search.value = search;
  els.favOnly.checked = false;
  els.allModels.checked = true;
  sheetReturn = null;
  if (binned && !binMode) enterBin();
  else if (sheetMode) void setSheet(false);
  else void resetGallery();
  if (!binned) loadRuns();
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  document.querySelector('.gallery-head').scrollIntoView({ block: 'start', behavior: reduce ? 'auto' : 'smooth' });
  els.search.focus({ preventScroll: true });
}

function bundleImportActions(result) {
  const first = result.first_id;
  const last = result.last_id;
  // Old server answers, duplicate-only imports and restored known files
  // have no new range. Never turn an absent or malformed range into a query.
  if (!result.imported || !Number.isSafeInteger(first) || !Number.isSafeInteger(last)
      || first < 1 || last < first || last - first + 1 !== result.imported) return;
  const add = (binned) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'ghost small';
    button.id = binned ? 'bundle-show-imported-bin' : 'bundle-show-imported';
    button.textContent = binned ? 'Show new images in bin' : 'Show new gallery images';
    button.title = `Close this and show only the newly added images in the ${binned ? 'recycle bin' : 'gallery'}, across all models`;
    button.onclick = () => showBundleImages(`id:${first}-${last}`, binned);
    els.bundleStatus.append(' ', button);
  };
  if (result.imported > (result.binned || 0)) add(false);
  if (result.binned) add(true);
}

async function refreshBundleList() {
  try {
    const result = await api('/api/gallery/bundles');
    els.bundleDirectory.textContent = `Saved on the server: ${result.directory}`;
    const key = JSON.stringify(result.bundles);
    if (key !== bundleListKey) {
      bundleListKey = key;
      const rows = result.bundles.map((bundle) => {
        const row = document.createElement('li');
        row.className = 'bundle-row';
        const name = document.createElement('span');
        name.className = 'bundle-name';
        name.textContent = bundle.name;
        const size = document.createElement('span');
        size.className = 'bundle-size';
        size.textContent = humanBytes(bundle.bytes);
        const tag = document.createElement('span');
        tag.className = 'bundle-tag';
        tag.textContent = 'Protected';
        tag.hidden = !bundle.encrypted;
        const link = document.createElement('a');
        link.href = bundleFileURL(bundle.name);
        link.download = bundle.name;
        link.textContent = 'Download';
        link.setAttribute('aria-label', `Download ${bundle.name}`);
        const button = document.createElement('button');
        button.type = 'button'; button.className = 'ghost small';
        button.dataset.bundleImport = bundle.name; button.textContent = 'Import';
        if (bundle.encrypted) button.dataset.bundleEncrypted = '';
        button.setAttribute('aria-label', `Import ${bundle.name}${bundle.encrypted ? ' (password-protected)' : ''}`);
        row.append(name, tag, size, link, button);
        return row;
      });
      if (!rows.length) {
        const empty = document.createElement('li');
        empty.textContent = 'No bundles saved yet.';
        rows.push(empty);
      }
      els.bundleList.replaceChildren(...rows);
    }
    syncBundleControls();
  } catch (err) { bundleSay(err.message, true); }
}

// `past` labels a result that ended before this page loaded, or one shown
// again when the dialog is reopened, so it never reads as the gallery's state.
function renderBundleResult(status, past = false) {
  const result = status.result;
  const say = (message, failed) => bundleSay(past ? `Last bundle operation: ${message}` : message, failed);
  els.bundleDownload.hidden = !result?.name;
  if (result?.name) {
    els.bundleDownload.href = bundleFileURL(result.name);
    els.bundleDownload.download = result.name;
    els.bundleDownload.textContent = `Download ${result.name}`;
  }
  const retained = result?.retained ?? [];
  els.bundleRetained.hidden = !retained.length;
  els.bundleRetained.querySelector('ul').replaceChildren(...retained.map((name) => {
    const item = document.createElement('li'); item.textContent = name; return item;
  }));
  if (status.state === 'failed') {
    say(`${status.error || 'The bundle operation failed.'}${result?.name ? ' The saved bundle is available below.' : ''}`, true);
  } else if (status.operation === 'backup') {
    // Nothing was cleared, which is the point: no part of this reads as a
    // failure. What a picked bundle left out, and the sources outside it,
    // are said as facts.
    const what = result.scope === 'all' ? `${result.images} images and ${result.records} gallery records`
      : bundleImages(result.records, result.scope);
    const left = result.left_out?.length || 0;
    const leftOut = left === 1 ? ' 1 picked image was left out: it is in the recycle bin or no longer in the gallery.'
      : left ? ` ${left} picked images were left out: they are in the recycle bin or no longer in the gallery.` : '';
    const links = result.links_outside || 0;
    const outside = links === 1 ? ' 1 link to a source image (a reference or a refine source) points outside the bundle; an import leaves it out.'
      : links ? ` ${links} links to source images (references or refine sources) point outside the bundle; an import leaves them out.` : '';
    say(`Backed up and verified ${what}. The gallery is unchanged.${leftOut}${outside}`, false);
  } else if (status.operation === 'export') {
    const end = result.cleared ? (past ? 'The gallery was cleared.' : 'The gallery is empty and ready for new images.')
      : 'The gallery was not completely cleared; review the files still on disk.';
    say(`Saved and verified ${result.images} images and ${result.records} gallery records. ${end}${result.cleanup_error ? ` ${result.cleanup_error}` : ''}`, !result.cleared);
  } else if (status.operation === 'import') {
    const missing = result.unresolved_references || 0;
    const restored = result.restored || 0;
    // A new record keeps the bin state it was bundled with, so the gallery
    // grows by fewer than were imported: say where the rest went.
    const binned = result.binned || 0;
    const inBin = binned === 1 ? ' 1 of them is in the recycle bin, as it was when bundled.'
      : binned ? ` ${binned} of them are in the recycle bin, as they were when bundled.` : '';
    say(`Imported ${result.imported} gallery records; ${result.present} already present. Existing images were kept.${!result.imported ? ' No new gallery records were added.' : ''}${inBin}${restored ? ` ${restored} missing image files were restored from the bundle.` : ''}${missing === 1 ? ' 1 link to a source image the bundle does not hold was left out.' : missing ? ` ${missing} links to source images the bundle does not hold were left out.` : ''}`, false);
    bundleImportActions(result);
  }
}

async function receiveBundleStatus(status) {
  const wasHeld = !!bundleHold();
  const first = !bundleStatusSeen;
  bundleStatusSeen = true;
  bundleStatus = status;
  if (status.state === 'running') {
    bundleSeenRunning = true;
    bundleSay({
      export: 'Creating and verifying the bundle, then clearing the gallery. You can close this dialog; the server will finish the operation.',
      backup: 'Creating and verifying the bundle; the gallery is kept. You can close this dialog; the server will finish the operation.',
    }[status.operation] || 'Checking and importing the bundle. You can close this dialog; the server will finish the operation.');
    bundleProgressSay(bundleProgressText(status.progress));
  } else bundleProgressSay('');
  if (status.id && ['done', 'failed'].includes(status.state) && bundleHandled !== status.id) {
    bundleHandled = status.id;
    if (first && !bundleSeenRunning) {
      // Ended before this page loaded (the server keeps its last result until
      // the next operation): show it as past, and leave the gallery alone.
      renderBundleResult(status, true);
      bundleResultShown = true;
    } else {
      bundleSeenRunning = false;
      bundleSettling = true;
      try {
        renderBundleResult(status);
        bundleResultShown = els.bundlesDialog.open;
        // The unlock password is kept after a failure, so a mistyped one can
        // be corrected, and goes once an import has used it.
        if (status.operation === 'import' && status.state === 'done') els.bundleUnlock.value = '';
        if (status.result?.deleted_ids?.length) forgetImages(status.result.deleted_ids);
        // A backup changes nothing in the gallery, so the typed count stays;
        // the views are still read again, since any gallery read made while
        // it ran was refused and left the grid empty under that refusal.
        if (status.operation !== 'backup') els.bundleCount.value = '';
        await Promise.all([resetGallery(), loadRuns(), syncBin()]);
        // A load may have landed while it ran, and the adapter list and the
        // previews its ready status asked for again were refused too.
        void refreshLoras();
        schedulePreview();
        if (refineSource) scheduleRefinePreview();
        if (inpaintSource) scheduleInpaintPreview();
        if (outpaintSource) scheduleOutpaintPreview();
        if (els.bundlesDialog.open) await Promise.all([refreshBundlePreview(), refreshBundleList()]);
      } finally { bundleSettling = false; }
    }
  }
  syncBundleControls();
  if (wasHeld !== !!bundleHold()) syncSubmitButtons();
}

async function pollBundleStatus() {
  clearTimeout(bundlePollTimer);
  if (bundlePollPending) return;
  bundlePollPending = true;
  const epoch = bundleRequestEpoch;
  try {
    const status = await api('/api/gallery/bundles/status');
    // A poll begun before this page's POST must not replace its pending state;
    // an upload's own progress is still shown while its body is sent.
    if (!bundleSending && epoch === bundleRequestEpoch) await receiveBundleStatus(status);
    else if (bundleSending && status.state === 'running' && status.progress?.phase === 'receiving') {
      bundleSay('Sending the bundle for import…');
      bundleProgressSay(bundleProgressText(status.progress));
    }
  } catch (err) {
    if (els.bundlesDialog.open) bundleSay(`${err.message} Reconnecting to the bundle operation…`, true);
  } finally {
    bundlePollPending = false;
    syncBundleControls();
    bundlePollTimer = setTimeout(pollBundleStatus, bundleHold() ? 750 : 3000);
  }
}

async function startBundleOperation(path, options, operation, before = null) {
  ++bundleRequestEpoch;
  bundleSending = true;
  bundleSeenRunning = true;
  bundleSay(operation !== 'import' ? 'Starting the gallery bundle…' : before ? 'Checking the bundle password…' : 'Sending the bundle for import…');
  syncBundleControls(); syncSubmitButtons();
  try {
    if (before) {
      await before();
      bundleSay('Sending the bundle for import…');
      void pollBundleStatus();
    }
    const status = await api(path, options);
    bundleSending = false;
    await receiveBundleStatus(status);
  } catch (err) {
    bundleSending = false;
    // The page's own Cancel: not a failure. A Cancel pressed once the whole
    // file had arrived may be too late, so whether anything was imported is
    // the server's to say: the poll below shows its words once it has them.
    if (err.name === 'AbortError') bundleSay('Upload cancelled.');
    else bundleSay(err.message, true);
    await pollBundleStatus();
    if (!bundleHold()) await refreshBundlePreview();
  } finally {
    bundleSending = false;
    syncBundleControls(); syncSubmitButtons();
    void pollBundleStatus();
  }
}

function clearBundlePasswords() {
  for (const input of [els.bundlePassword, els.bundlePasswordAgain, els.bundleUnlock]) input.value = '';
}

function exportGalleryBundle() {
  if (bundleRefused(els.bundleExport)) return;
  const password = els.bundlePassword.value;
  // The export password lives only in these fields and this one request;
  // the unlock field is cleared when an import it opened is done.
  els.bundlePassword.value = els.bundlePasswordAgain.value = '';
  void startBundleOperation('/api/gallery/bundles/export', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ expected: bundlePreview.records, ...(password ? { password } : {}) }),
  }, 'export');
}

// Back up: the same verified bundle, of the scope chosen, with nothing
// cleared, so no typed count. The picked ids are sent as they stand; one
// binned or deleted since is left out by the server and said in the result.
function backupGallery() {
  if (bundleRefused(els.bundleBackup)) return;
  const password = els.bundlePassword.value;
  const scope = bundleScope();
  els.bundlePassword.value = els.bundlePasswordAgain.value = '';
  void startBundleOperation('/api/gallery/bundles/export', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      keep_gallery: true, scope, ...(scope === 'picked' ? { ids: bundlePickedIds() } : {}),
      ...(password ? { password } : {}),
    }),
  }, 'backup');
}

async function readBundleFile() {
  const epoch = ++bundleFileEpoch;
  bundleFileHeader = null;
  syncBundleControls();
  const file = els.bundleFile.files[0];
  if (!file) return null;
  try {
    const bytes = new Uint8Array(await file.slice(0, BUNDLE_HEADER_BYTES).arrayBuffer());
    if (epoch === bundleFileEpoch) bundleFileHeader = bytes;
  } catch (err) {
    if (epoch === bundleFileEpoch) bundleSay(`The selected file could not be read: ${err.message}`, true);
  } finally {
    if (epoch === bundleFileEpoch) syncBundleControls();
  }
  return epoch === bundleFileEpoch ? bundleFileHeader : null;
}

async function uploadGalleryBundle() {
  if (bundleRefused(els.bundleUpload)) return;
  const file = els.bundleFile.files[0];
  // A press straight after choosing a file reads its header first.
  if (!bundleFileHeader && (!(await readBundleFile()) || bundleRefused(els.bundleUpload))) return;
  const headers = { 'Content-Type': 'application/octet-stream' };
  const abort = new AbortController();
  let check = null;
  if (bundleProtected(bundleFileHeader)) {
    const password = els.bundleUnlock.value;
    const header = bundleBase64(bundleFileHeader);
    headers['X-Bundle-Password'] = bundleBase64(new TextEncoder().encode(password));
    // A wrong password is refused here, before the whole file is sent.
    check = () => api('/api/gallery/bundles/password', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ header, password }),
      signal: abort.signal,
    });
  }
  bundleUploadAbort = abort;
  try {
    await startBundleOperation('/api/gallery/bundles/upload', { method: 'POST', headers, body: file, signal: abort.signal }, 'import', check);
  } finally {
    if (bundleUploadAbort === abort) bundleUploadAbort = null;
    syncBundleControls();
  }
}

function cancelBundleUpload() {
  if (bundleRefused(els.bundleUploadCancel)) return;
  bundleSay('Cancelling the upload…');
  bundleUploadAbort.abort();
}

async function openBundles() {
  if (!els.bundlesDialog.open) els.bundlesDialog.showModal();
  els.bundlesClose.focus({ preventScroll: true });
  // A result already read in this dialog is labelled as past on reopening;
  // one that arrived while it was closed is still news. A newer one the poll
  // below brings is shown as it comes.
  const status = bundleStatus;
  if (status?.id && status.id === bundleHandled && ['done', 'failed'].includes(status.state) && !bundleSending) {
    if (bundleResultShown) renderBundleResult(status, true);
    bundleResultShown = true;
  }
  syncBundleControls();
  await pollBundleStatus();
  await Promise.all([refreshBundlePreview(), refreshBundleList()]);
}

function initBundles() {
  els.bundles.onclick = () => { void openBundles(); };
  els.bundlesClose.onclick = () => els.bundlesDialog.close();
  els.bundlesDialog.addEventListener('close', clearBundlePasswords);
  els.bundleCount.oninput = syncBundleControls;
  for (const input of [els.bundlePassword, els.bundlePasswordAgain, els.bundleUnlock]) input.oninput = syncBundleControls;
  els.bundleFile.onchange = () => { void readBundleFile(); };
  els.bundleExport.onclick = exportGalleryBundle;
  els.bundleBackup.onclick = backupGallery;
  for (const radio of document.querySelectorAll('input[name="bundle-scope"]')) radio.onchange = syncBundleControls;
  els.bundleUpload.onclick = () => { void uploadGalleryBundle(); };
  els.bundleUploadCancel.onclick = cancelBundleUpload;
  els.bundleRefresh.onclick = () => { if (!bundleRefused(els.bundleRefresh)) void refreshBundlePreview(); };
  els.bundleList.onclick = (event) => {
    const button = event.target.closest('[data-bundle-import]');
    if (!button || bundleRefused(button)) return;
    const password = 'bundleEncrypted' in button.dataset ? els.bundleUnlock.value : '';
    void startBundleOperation('/api/gallery/bundles/import', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: button.dataset.bundleImport, ...(password ? { password } : {}) }),
    }, 'import');
  };
  void pollBundleStatus();
}
