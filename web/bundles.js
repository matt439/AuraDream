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
let bundleListKey = null;
let bundlePreviewEpoch = 0;

function bundleHold() {
  return bundleSending || bundleStatus?.state === 'running' ? 'Wait for the gallery bundle operation to finish' : '';
}

function bundleActionHold() {
  return bundleHold() || (bundleSettling ? 'Refreshing the gallery after the bundle operation' : '')
    || (bundleReading ? 'Wait for the gallery counts' : '') || (serverOffline ? 'The server is not answering' : '')
    || (busy || queuedIds.length || elsewhereBusy || submitting ? 'Wait for all generation jobs to finish' : '')
    || (expandingPrompt || critiquing ? 'Wait for prompt expansion or the critic to finish' : '')
    || (modelState === 'loading' ? 'Wait for the weights to finish loading' : '') || modelChangeHold();
}

function bundleSay(message, failed = false) {
  if (els.bundleStatus.textContent !== message) els.bundleStatus.textContent = message;
  els.bundleStatus.classList.toggle('failed', failed);
}

function syncBundleControls() {
  const why = bundleActionHold();
  const hold = (el, reason, title) => {
    el.setAttribute('aria-disabled', String(!!reason));
    el.title = reason || title;
  };
  const count = bundlePreview?.records;
  const confirmed = Number.isInteger(count) && els.bundleCount.value.trim() === String(count);
  hold(els.bundleExport, why || (bundleReading ? 'Wait for the gallery counts' : '')
    || (!bundlePreview ? 'Refresh the gallery counts first' : '')
    || (!confirmed ? `Type ${count} to confirm the entire gallery` : ''), 'Save and verify a bundle, then clear the entire gallery');
  hold(els.bundleRefresh, bundleHold() || (bundleReading ? 'Reading the gallery counts' : ''), 'Read the current gallery counts');
  hold(els.bundleUpload, why || (!els.bundleFile.files.length ? 'Choose a bundle ZIP file first' : ''), 'Merge this file into the gallery');
  for (const button of els.bundleList.querySelectorAll('[data-bundle-import]')) hold(button, why, 'Merge this saved bundle into the gallery');
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
    els.bundlePreview.textContent = `${preview.records} gallery records · ${preview.images} full-size images · ${preview.thumbnails} thumbnails · ${humanBytes(preview.bytes)}`;
    els.bundleCountLabel.textContent = `Type ${preview.records} to confirm all gallery records will be removed`;
  } catch (err) {
    if (epoch !== bundlePreviewEpoch) return;
    bundlePreview = null;
    els.bundlePreview.textContent = 'The gallery counts could not be read.';
    bundleSay(err.message, true);
  } finally {
    if (epoch === bundlePreviewEpoch) { bundleReading = false; syncBundleControls(); }
  }
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
        const link = document.createElement('a');
        link.href = bundleFileURL(bundle.name);
        link.download = bundle.name;
        link.textContent = 'Download';
        link.setAttribute('aria-label', `Download ${bundle.name}`);
        const button = document.createElement('button');
        button.type = 'button'; button.className = 'ghost small';
        button.dataset.bundleImport = bundle.name; button.textContent = 'Import';
        button.setAttribute('aria-label', `Import ${bundle.name}`);
        row.append(name, size, link, button);
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

function renderBundleResult(status) {
  const result = status.result;
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
    bundleSay(`${status.error || 'The bundle operation failed.'}${result?.name ? ' The saved bundle is available below.' : ''}`, true);
  } else if (status.operation === 'export') {
    const end = result.cleared ? 'The gallery is empty and ready for new images.'
      : 'The gallery was not completely cleared; review the files still on disk.';
    bundleSay(`Saved and verified ${result.images} images and ${result.records} gallery records. ${end}${result.cleanup_error ? ` ${result.cleanup_error}` : ''}`, !result.cleared);
  } else if (status.operation === 'import') {
    const missing = result.unresolved_references || 0;
    bundleSay(`Imported ${result.imported} gallery records; ${result.present} already present. Existing images were kept.${missing ? ` ${missing} links to missing source images could not be restored.` : ''}`, !!missing);
  }
}

async function receiveBundleStatus(status) {
  const wasHeld = !!bundleHold();
  bundleStatus = status;
  if (status.state === 'running') {
    bundleSay(status.operation === 'export'
      ? 'Creating and verifying the bundle, then clearing the gallery. You can close this dialog; the server will finish the operation.'
      : 'Checking and importing the bundle. You can close this dialog; the server will finish the operation.');
  }
  if (status.id && ['done', 'failed'].includes(status.state) && bundleHandled !== status.id) {
    bundleHandled = status.id;
    bundleSettling = true;
    try {
      renderBundleResult(status);
      if (status.result?.deleted_ids?.length) forgetImages(status.result.deleted_ids);
      els.bundleCount.value = '';
      await Promise.all([resetGallery(), loadRuns()]);
      if (els.bundlesDialog.open) await Promise.all([refreshBundlePreview(), refreshBundleList()]);
    } finally { bundleSettling = false; }
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
    // A poll begun before this page's POST must not replace its pending state.
    if (!bundleSending && epoch === bundleRequestEpoch) await receiveBundleStatus(status);
  } catch (err) {
    if (els.bundlesDialog.open) bundleSay(`${err.message} Reconnecting to the bundle operation…`, true);
  } finally {
    bundlePollPending = false;
    syncBundleControls();
    bundlePollTimer = setTimeout(pollBundleStatus, bundleHold() ? 750 : 3000);
  }
}

async function startBundleOperation(path, options, operation) {
  ++bundleRequestEpoch;
  bundleSending = true;
  bundleSay(operation === 'export' ? 'Starting the gallery bundle…' : 'Sending the bundle for import…');
  syncBundleControls(); syncSubmitButtons();
  try {
    const status = await api(path, options);
    bundleSending = false;
    await receiveBundleStatus(status);
  } catch (err) {
    bundleSending = false;
    bundleSay(err.message, true);
    await pollBundleStatus();
    if (!bundleHold()) await refreshBundlePreview();
  } finally {
    bundleSending = false;
    syncBundleControls(); syncSubmitButtons();
    void pollBundleStatus();
  }
}

function exportGalleryBundle() {
  if (bundleRefused(els.bundleExport)) return;
  void startBundleOperation('/api/gallery/bundles/export', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ expected: bundlePreview.records }),
  }, 'export');
}

function uploadGalleryBundle() {
  if (bundleRefused(els.bundleUpload)) return;
  void startBundleOperation('/api/gallery/bundles/upload', {
    method: 'POST', headers: { 'Content-Type': 'application/zip' }, body: els.bundleFile.files[0],
  }, 'import');
}

async function openBundles() {
  if (!els.bundlesDialog.open) els.bundlesDialog.showModal();
  els.bundlesClose.focus({ preventScroll: true });
  syncBundleControls();
  await pollBundleStatus();
  await Promise.all([refreshBundlePreview(), refreshBundleList()]);
}

function initBundles() {
  els.bundles.onclick = () => { void openBundles(); };
  els.bundlesClose.onclick = () => els.bundlesDialog.close();
  els.bundleCount.oninput = syncBundleControls;
  els.bundleFile.onchange = syncBundleControls;
  els.bundleExport.onclick = exportGalleryBundle;
  els.bundleUpload.onclick = uploadGalleryBundle;
  els.bundleRefresh.onclick = () => { if (!bundleRefused(els.bundleRefresh)) void refreshBundlePreview(); };
  els.bundleList.onclick = (event) => {
    const button = event.target.closest('[data-bundle-import]');
    if (!button || bundleRefused(button)) return;
    void startBundleOperation('/api/gallery/bundles/import', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: button.dataset.bundleImport }),
    }, 'import');
  };
  void pollBundleStatus();
}
