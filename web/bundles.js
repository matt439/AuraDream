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
// Automatic backup settings are read only while Bundles is open. Keep the
// edited draft and its revision apart from polls, including another tab's saves.
let automaticBackupStatus = null;
let automaticBackupDraftRevision = null;
let automaticBackupDraftDestination = null;
let automaticBackupDraftModeRevision = null;
let automaticBackupDraftCredentialGeneration = null;
let automaticBackupDirty = false;
let automaticBackupEditEpoch = 0;
let automaticBackupReading = false;
let automaticBackupSaving = false;
let automaticBackupUnavailable = false;
let automaticBackupEpoch = 0;

// The clear has its own lifetime: POST completion is not terminal cleanup.
// Never correlate an unanswered local POST with an arbitrary later GET id.
let bundleClearOwner = null;
const BUNDLE_CLEAR_GUIDANCE = 'Bin, Restore and delete actions stay held in this page. Let any server operation finish, inspect the gallery and saved bundles, and keep any unsent draft before manually reloading this page. Reloading does not cancel a server operation or prove the clear never ran. Do not repeat the request to test it.';

function bundleClearHold() {
  if (!bundleClearOwner) return '';
  return bundleClearOwner.phase === 'unresolved'
    ? 'The clear outcome is unconfirmed; inspect Bundles, then reload this page'
    : 'Wait for Bundle & clear to finish';
}

// Raw owners only: the public bin reasons include this clear's own hold.
function bundleClearBinReason() {
  return (binWriteIds.size || binBroadOwner)
    ? 'Wait for the current recycle bin change to finish' : '';
}

function bundleClearUnresolved(owner, reason) {
  if (bundleClearOwner !== owner) return;
  if (owner.phase !== 'unresolved') owner.unresolvedReason = reason;
  owner.phase = 'unresolved';
  bundleSay(owner.unresolvedReason + ' ' + BUNDLE_CLEAR_GUIDANCE, true);
  bundleProgressSay('');
  syncBinWriteHolds();
}

// Validate only the current export fields that rendering/forgetting consumes.
// A failed worker can legitimately have no result; failure does not establish
// which rows, if any, were deleted.
function validBundleClearStatus(status) {
  if (!status || typeof status !== 'object' || Array.isArray(status)
      || typeof status.id !== 'string' || !status.id.trim() || status.operation !== 'export'
      || !['running', 'done', 'failed'].includes(status.state)) return false;
  const result = status.result;
  if (result != null && (typeof result !== 'object' || Array.isArray(result))) return false;
  if (result?.deleted_ids !== undefined && (!Array.isArray(result.deleted_ids)
      || result.deleted_ids.some((id) => !Number.isSafeInteger(id) || id <= 0))) return false;
  if (result?.name !== undefined && typeof result.name !== 'string') return false;
  if (result?.retained !== undefined && (!Array.isArray(result.retained)
      || result.retained.some((name) => typeof name !== 'string'))) return false;
  return status.state !== 'done' || (!!result && typeof result.name === 'string'
    && Number.isSafeInteger(result.images) && result.images >= 0
    && Number.isSafeInteger(result.records) && result.records >= 0
    && typeof result.cleared === 'boolean' && Array.isArray(result.deleted_ids));
}

// True means this clear owns the status path, including deferred/unresolved
// observations. It must never fall through to generic terminal bookkeeping.
async function receiveBundleClearStatus(status, first) {
  let owner = bundleClearOwner;
  if (!owner) {
    if (status?.operation !== 'export' || (status.id && bundleHandled === status.id)) return false;
    // Preserve the existing first-observation past-result heuristic. A local
    // accepted receipt is already tracked and therefore never takes this path.
    if (first && !bundleSeenRunning && ['done', 'failed'].includes(status.state)) return false;
    owner = { phase: 'running', id: status.id, origin: 'observed',
      requestEpoch: bundleRequestEpoch, previousStatusId: bundleStatus?.id ?? null };
    bundleClearOwner = owner;
    syncBinWriteHolds();
  }
  if (owner.phase === 'unresolved') {
    bundleStatus = status;
    bundleSay(owner.unresolvedReason + ' ' + BUNDLE_CLEAR_GUIDANCE, true);
    return true;
  }
  // A status read while our POST is unacknowledged is inspection, not its receipt.
  if (owner.phase === 'sending' && owner.id === null) return true;
  // A terminal result already received is this clear's to apply: a concurrent
  // poll cannot apply it twice, and a later poll - another client's operation
  // replaces the server's latest status - cannot make it unconfirmed.
  if (owner.phase === 'settling') return true;
  if (owner.phase === 'draining') status = owner.terminal;
  if (!validBundleClearStatus(status) || status.id !== owner.id) {
    bundleStatus = status;
    bundleClearUnresolved(owner, 'The tracked clear result is no longer available.');
    return true;
  }
  bundleStatus = status;
  if (status.state === 'running') {
    owner.phase = 'running';
    bundleSeenRunning = true;
    bundleSay('Creating and verifying the bundle, then clearing the gallery. You can close this dialog; the server will finish the operation.');
    bundleProgressSay(bundleProgressText(status.progress));
    return true;
  }
  owner.phase = 'draining';
  owner.terminal = status;
  bundleProgressSay('');
  if (binWriteIds.size || binBroadOwner) {
    bundleSay('Waiting for earlier gallery changes before applying the clear result.');
    return true;
  }
  owner.phase = 'settling';
  bundleSettling = true;
  syncBundleControls();
  try {
    renderBundleResult(status);
    bundleResultShown = els.bundlesDialog.open;
    if (status.result?.deleted_ids?.length) forgetImages(status.result.deleted_ids);
    els.bundleCount.value = '';
    await Promise.all([resetGallery(), loadRuns(), syncBin()]);
    if (bundleClearOwner !== owner || owner.id !== status.id || owner.phase !== 'settling') return true;
    void refreshLoras();
    schedulePreview();
    if (refineSource) scheduleRefinePreview();
    if (inpaintSource) scheduleInpaintPreview();
    if (outpaintSource) scheduleOutpaintPreview();
    if (els.bundlesDialog.open) await Promise.all([refreshBundlePreview(), refreshBundleList()]);
    if (bundleClearOwner !== owner || owner.id !== status.id || owner.phase !== 'settling') return true;
    bundleHandled = status.id;
    bundleSeenRunning = false;
    bundleClearOwner = null;
    syncBinWriteHolds();
  } catch (err) {
    bundleClearUnresolved(owner, 'The clear result could not finish updating this page.');
  } finally {
    bundleSettling = false;
  }
  return true;
}

async function startBundleClear(path, options, owner) {
  owner.requestEpoch = ++bundleRequestEpoch;
  bundleSending = true;
  bundleSeenRunning = true;
  bundleSay('Starting the gallery bundle.');
  syncBundleControls(); syncSubmitButtons();
  try {
    let status;
    try {
      status = await api(path, options);
    } catch (err) {
      if (owner.requestEpoch === bundleRequestEpoch) bundleSending = false;
      // Only this POST await can supply a pre-reservation refusal. An error
      // in later result application can never release accepted ownership.
      if (bundleClearOwner === owner && owner.id === null && [409, 422].includes(err.status)) {
        bundleClearOwner = null;
        bundleSeenRunning = owner.previousSeenRunning;
        syncBinWriteHolds();
        bundleSay(err.message, true);
      } else bundleClearUnresolved(owner, "The clear request's outcome is unconfirmed.");
      await pollBundleStatus();
      if (!bundleHold()) await refreshBundlePreview();
      return;
    }
    if (owner.requestEpoch === bundleRequestEpoch) bundleSending = false;
    if (bundleClearOwner !== owner || owner.phase === 'unresolved') return;
    if (!validBundleClearStatus(status)) {
      bundleClearUnresolved(owner, "The clear request's response could not be confirmed.");
      return;
    }
    owner.id = status.id;
    owner.phase = 'running';
    await receiveBundleStatus(status);
  } catch (err) {
    bundleClearUnresolved(owner, 'The clear result could not finish updating this page.');
  } finally {
    // A refused attempt may have awaited inspection after releasing its hold;
    // its cleanup must not clear a newer operation's sending flag.
    if (owner.requestEpoch === bundleRequestEpoch) bundleSending = false;
    syncBundleControls(); syncSubmitButtons();
    void pollBundleStatus();
  }
}

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
  return bundleClearHold() || bundleHold() || (bundleSettling ? 'Refreshing the gallery after the bundle operation' : '')
    || (bundleReading ? 'Wait for the gallery counts' : '') || (serverOffline ? 'The server is not answering' : '')
    || (busy || queuedIds.length || elsewhereBusy || submitting ? 'Wait for all generation jobs to finish' : '')
    || (expandingPrompt || critiquing ? 'Wait for prompt expansion or the critic to finish' : '')
    || modelChangeHold();
}

function bundleSay(message, failed = false) {
  // Polls, reopening and read failures cannot silently replace this guidance.
  if (bundleClearOwner?.phase === 'unresolved') {
    message = bundleClearOwner.unresolvedReason + ' ' + BUNDLE_CLEAR_GUIDANCE;
    failed = true;
  }
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
  hold(els.bundleExport, why || bundleClearBinReason() || (bundleReading ? 'Wait for the gallery counts' : '')
    || (scope !== 'all' ? 'Only the entire gallery is cleared; Back up saves the picked or the starred images and keeps them' : '')
    || (!bundlePreview ? 'Refresh the gallery counts first' : '')
    || differ || (!confirmed ? `Type ${count} in the confirmation field to enable Bundle & clear` : ''),
    password ? 'Save, encrypt and verify a bundle, then clear the entire gallery' : 'Save and verify a bundle, then clear the entire gallery');
  const what = scope === 'all' ? 'the entire gallery' : `the ${bundleImages(scope === 'picked' ? picked : starred, scope)}`;
  hold(els.bundleBackup, why || (bundleReading ? 'Wait for the gallery counts' : '')
    || (!bundlePreview ? 'Refresh the gallery counts first' : '')
    || (scope === 'picked' && !picked ? 'No cards are picked; use Ctrl/Cmd+click or Select to pick them first' : '')
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
  syncAutomaticBackupControls();
}

function bundleRefused(button) {
  syncBundleControls();
  if (button.getAttribute('aria-disabled') !== 'true') return false;
  bundleSay(button.title + '.', true);
  // The count is the clear's safety check. A press with only that missing
  // puts the caret beside the explanation, revealing it inside the dialog
  // only where a narrow wrapped row has put it out of view.
  if (button === els.bundleExport && !bundleActionHold() && !bundleClearBinReason() && bundleScope() === 'all' && bundlePreview
      && els.bundlePassword.value === els.bundlePasswordAgain.value
      && els.bundleCount.value.trim() !== String(bundlePreview.records)) {
    focusBundleField(els.bundleCount);
  } else if (!bundleActionHold() && !els.bundleUnlock.value
      && ('bundleEncrypted' in button.dataset
        || (button === els.bundleUpload && els.bundleFile.files.length && bundleProtected(bundleFileHeader)))) {
    focusBundleUnlock();
  }
  return true;
}

function focusBundleField(input) {
  input.focus({ preventScroll: true });
  const box = els.bundlesDialog.getBoundingClientRect();
  const top = box.top + els.bundlesDialog.clientTop;
  const bottom = top + els.bundlesDialog.clientHeight;
  const field = input.getBoundingClientRect();
  if (field.top < top || field.bottom > bottom) input.scrollIntoView({ block: 'nearest' });
}

function focusBundleUnlock() {
  // Explicit navigation from a Protected row, whose import uses this one
  // shared field. No per-file secret or password state is kept on the row.
  focusBundleField(els.bundleUnlock);
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
    els.bundleCountLabel.textContent = `Type ${preview.records} to enable Bundle & clear (${preview.records} gallery records)`;
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
        if (bundle.encrypted) {
          button.dataset.bundleEncrypted = '';
          button.setAttribute('aria-describedby', 'bundle-unlock-hint');
        }
        button.setAttribute('aria-label', `Import ${bundle.name}${bundle.encrypted ? ' (password-protected)' : ''}`);
        row.append(name, tag, size, link);
        if (bundle.encrypted) {
          const unlock = document.createElement('button');
          unlock.type = 'button'; unlock.className = 'ghost tiny bundle-import-password';
          unlock.dataset.bundleUnlock = '';
          unlock.textContent = 'Uses Import password';
          unlock.title = 'Go to the shared Import password field above';
          unlock.setAttribute('aria-label', `Go to Import password for ${bundle.name}`);
          row.append(unlock);
        }
        row.append(button);
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
  // A settings save creates no archive and changes no gallery view or count.
  if (status.operation === 'backup_settings') return;
  const result = status.result;
  const say = (message, failed) => bundleSay(past ? `Last bundle operation: ${message}` : message, failed);
  const archive = status.operation === 'automatic_backup' ? result?.archive_name : result?.name;
  els.bundleDownload.hidden = !archive;
  if (archive) {
    els.bundleDownload.href = bundleFileURL(archive);
    els.bundleDownload.download = archive;
    els.bundleDownload.textContent = `Download ${archive}`;
  }
  const retained = result?.retained ?? [];
  els.bundleRetained.hidden = !retained.length;
  els.bundleRetained.querySelector('ul').replaceChildren(...retained.map((name) => {
    const item = document.createElement('li'); item.textContent = name; return item;
  }));
  if (status.state === 'failed') {
    say(`${status.error || 'The bundle operation failed.'}${result?.name ? ' The saved bundle is available below.' : ''}`, true);
  } else if (status.operation === 'automatic_backup') {
    const retention = result?.retention;
    const cleanup = retention ? ' ' + automaticBackupRetentionText({
      retention_state: retention.state, retention_reason: retention.reason,
      retention_marked: retention.marked, retention_retained: retention.retained,
      retention_pending: retention.pending, retention_warning: retention.warning,
      retention_uncertain: retention.uncertain,
    }) + (retention.warning ? ` ${retention.warning}` : '') : '';
    say(automaticBackupOutcomeText(result || {}) + cleanup, ['failed', 'unmanaged'].includes(result?.state));
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
  // Keep the ordinary status path synchronous until its existing await. A
  // needless await on a non-clear result could let a new clear overtake it.
  const clearApplies = bundleClearOwner || (status?.operation === 'export'
    && !(status.id && bundleHandled === status.id)
    && !(first && !bundleSeenRunning && ['done', 'failed'].includes(status.state)));
  if (clearApplies) {
    await receiveBundleClearStatus(status, first);
    syncBundleControls();
    if (wasHeld !== !!bundleHold()) syncSubmitButtons();
    return;
  }
  bundleStatus = status;
  if (status.state === 'running') {
    bundleSeenRunning = true;
    if (status.operation !== 'backup_settings') bundleSay({
      export: 'Creating and verifying the bundle, then clearing the gallery. You can close this dialog; the server will finish the operation.',
      backup: 'Creating and verifying the bundle; the gallery is kept. You can close this dialog; the server will finish the operation.',
      automatic_backup: 'Checking the automatic startup backup and saving a verified copy if due; the gallery is kept. You can close this dialog; the server will finish the operation.',
      backup_settings: 'Saving automatic startup backup settings. No backup is created now.',
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
        if (!['backup', 'automatic_backup', 'backup_settings'].includes(status.operation)) els.bundleCount.value = '';
        await Promise.all([resetGallery(), loadRuns(), syncBin()]);
        // Even a settings save briefly reserves admission, so reads made while
        // it ran may need repair. Keep its manual count/message, but recover
        // refused gallery, adapter and preview reads as for an ordinary backup.
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
    if (els.bundlesDialog.open) void refreshAutomaticBackup();
    const clearActive = bundleClearOwner && bundleClearOwner.phase !== 'unresolved';
    bundlePollTimer = setTimeout(pollBundleStatus, bundleHold() || clearActive ? 750 : 3000);
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

// Caps Lock as the last key event in a password field reported it. Only the
// modifier state is kept, in memory; focus has no modifier state of its own.
let bundleCapsLock = false;

function sayBundleCapsLock(input, on) {
  const note = input.closest('.bundle-password').querySelector('.caps-lock-note');
  const text = on ? 'Caps Lock is on' : '';
  if (note.textContent !== text) note.textContent = text;
}

function watchBundleCapsLock(input) {
  const read = (event) => {
    if (typeof event.getModifierState === 'function') bundleCapsLock = event.getModifierState('CapsLock');
    sayBundleCapsLock(input, bundleCapsLock);
  };
  input.addEventListener('keydown', read);
  input.addEventListener('keyup', read);
  input.addEventListener('focus', () => sayBundleCapsLock(input, bundleCapsLock));
  input.addEventListener('blur', () => sayBundleCapsLock(input, false));
}

function exportGalleryBundle() {
  if (bundleRefused(els.bundleExport)) return;
  // Admission is synchronous, before passwords leave the editor or POST starts.
  const held = bundleClearHold() || bundleClearBinReason();
  if (held) { bundleSay(held + '.', true); return; }
  const owner = { phase: 'sending', id: null, origin: 'local',
    requestEpoch: bundleRequestEpoch + 1, previousStatusId: bundleStatus?.id ?? null,
    previousSeenRunning: bundleSeenRunning };
  const expected = bundlePreview.records;
  const password = els.bundlePassword.value;
  bundleClearOwner = owner;
  syncBinWriteHolds();
  // The export password lives only in these fields and this one request.
  els.bundlePassword.value = els.bundlePasswordAgain.value = '';
  void startBundleClear('/api/gallery/bundles/export', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ expected, ...(password ? { password } : {}) }),
  }, owner);
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
  await Promise.all([refreshBundlePreview(), refreshBundleList(), refreshAutomaticBackup()]);
}

function initBundles() {
  initAutomaticBackup();
  els.bundles.onclick = () => { void openBundles(); };
  els.bundlesClose.onclick = () => els.bundlesDialog.close();
  els.bundlesDialog.addEventListener('close', clearBundlePasswords);
  els.bundleCount.oninput = syncBundleControls;
  for (const input of [els.bundlePassword, els.bundlePasswordAgain, els.bundleUnlock]) input.oninput = syncBundleControls;
  for (const input of [els.bundlePassword, els.bundlePasswordAgain, els.bundleUnlock]) watchBundleCapsLock(input);
  els.bundleFile.onchange = () => { void readBundleFile(); };
  els.bundleExport.onclick = exportGalleryBundle;
  els.bundleBackup.onclick = backupGallery;
  for (const radio of document.querySelectorAll('input[name="bundle-scope"]')) radio.onchange = syncBundleControls;
  els.bundleUpload.onclick = () => { void uploadGalleryBundle(); };
  els.bundleUploadCancel.onclick = cancelBundleUpload;
  els.bundleRefresh.onclick = () => { if (!bundleRefused(els.bundleRefresh)) void refreshBundlePreview(); };
  els.bundleList.onclick = (event) => {
    if (event.target.closest('[data-bundle-unlock]')) { focusBundleUnlock(); return; }
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

function automaticBackupElements() {
  return {
    enabled: document.getElementById('automatic-backup-enabled'),
    age: document.getElementById('automatic-backup-age'),
    keep: document.getElementById('automatic-backup-keep'),
    consent: document.getElementById('automatic-backup-consent'),
    protection: document.getElementById('automatic-backup-protection'),
    encryptedConsent: document.getElementById('automatic-backup-encrypted-consent'),
    recoveryConsent: document.getElementById('automatic-backup-recovery-consent'),
    credentialStatus: document.getElementById('automatic-backup-credential-status'),
    retentionEnabled: document.getElementById('automatic-backup-retention-enabled'),
    deleteConsent: document.getElementById('automatic-backup-delete-consent'),
    save: document.getElementById('automatic-backup-save'),
    reload: document.getElementById('automatic-backup-reload'),
    status: document.getElementById('automatic-backup-status'),
    warning: document.getElementById('automatic-backup-warning'),
    destination: document.getElementById('automatic-backup-destination'),
    mode: document.getElementById('automatic-backup-mode'),
    retentionStatus: document.getElementById('automatic-backup-retention-status'),
    retentionWarning: document.getElementById('automatic-backup-retention-warning'),
    feedback: document.getElementById('automatic-backup-feedback'),
  };
}

function automaticBackupSay(message, failed = false) {
  const feedback = automaticBackupElements().feedback;
  if (feedback.textContent !== message) feedback.textContent = message;
  feedback.classList.toggle('failed', failed);
}

function automaticBackupOutcomeText(status) {
  const reasons = {
    disabled: 'Automatic startup backups are off.',
    next_start: 'Automatic backup is enabled for the next server start. No backup is created by saving settings.',
    legacy_settings: 'Earlier automatic backup settings need explicit confirmation before they can run.',
    destination_changed: 'The backup destination changed. Confirm the new destination before automatic backups can run.',
    settings_unreadable: 'Automatic backups are off because their settings cannot be read. Repair the settings file before saving.',
    destination_unavailable: 'Automatic backups are off because the backup destination is unavailable.',
    not_started: 'This server has not attempted an automatic backup.',
    checking: 'Checking whether the automatic startup backup is due. The gallery is kept.',
    empty: 'Automatic startup backup skipped: the gallery is empty. Earlier copies are kept.',
    young: 'Automatic startup backup is not due: a verified automatic copy is recent enough.',
    clock_backwards: 'Automatic startup backup skipped because the clock moved backward. Earlier copies are kept.',
    admission_busy: 'Automatic startup backup deferred because other work is active. There is no automatic retry this session.',
    gallery_busy: 'Automatic startup backup deferred because another process is using the gallery. There is no automatic retry this session.',
    untrusted_inventory: 'Automatic startup backup skipped because archive ownership could not be confirmed.',
    operation_already_recorded: 'This automatic startup backup attempt was already recorded; no duplicate was created.',
    verified_publication: 'Automatic startup backup saved and verified. The gallery is unchanged.',
    receipt_not_saved: 'The automatic backup file was saved, but its ownership receipt could not be saved. It remains unmanaged; earlier copies are kept.',
    journal_unusable: 'Automatic startup backup failed because its receipt journal needs repair. Earlier copies are kept.',
    source_mismatch: 'Automatic startup backup failed because the gallery identity did not match. Earlier copies are kept.',
    destination_mismatch: 'Automatic startup backup failed because the destination did not match. Earlier copies are kept.',
    destination_identity_mismatch: 'Automatic startup backup failed because its saved destination identity did not match. Earlier copies are kept.',
    invalid_policy: 'Automatic startup backup failed because its saved policy is invalid. Earlier copies are kept.',
    invalid_identity_or_protection: 'Automatic startup backup failed because its identity or protection settings are invalid. Earlier copies are kept.',
    credential_pending: 'Automatic startup backup is inactive because offline credential setup is pending. Inspect the local setup outcome before enabling it.',
    credential_unavailable: 'Automatic encrypted backup could not obtain its configured credential. No plaintext fallback was used.',
    credential_missing: 'The configured encrypted-backup credential is missing. Inspect the local profile and repair its offline setup using the intended Windows account. No plaintext fallback was used.',
    credential_session_unavailable: 'This Windows session cannot read the configured backup credential. Use the intended Windows account and an available sign-in session before the next start. No plaintext fallback was used.',
    credential_access_denied: 'Windows refused access to the configured backup credential. Review credential-store access for the intended account before the next start. No plaintext fallback was used.',
    credential_unusable: 'The configured backup credential could not be used. Inspect the local profile and repair the offline credential setup before the next start. No plaintext fallback was used.',
    worker_cancelled_before_start: 'Automatic startup backup stopped before its worker started. No new backup was created.',
    worker_start_failed: 'Automatic startup backup could not start its worker. Earlier copies are kept.',
    backup_failed: 'Automatic startup backup failed. Earlier copies are kept; check the destination and settings before the next server start.',
  };
  return reasons[status.reason] || ({
    disabled: 'Automatic startup backups are off.',
    needs_confirmation: 'Automatic backup settings need explicit confirmation before they can run.',
    unavailable: 'Automatic startup backups are unavailable; review the warning below.',
    scheduled: 'Automatic backup is enabled for the next server start.',
    running: 'Automatic startup backup is running. The gallery is kept.',
    succeeded: 'Automatic startup backup saved and verified. The gallery is unchanged.',
    failed: 'Automatic startup backup failed. Earlier copies are kept.',
    deferred: 'Automatic startup backup was deferred. There is no automatic retry this session.',
    skipped: 'Automatic startup backup was skipped. Earlier copies are kept.',
    unmanaged: 'The saved automatic backup has no confirmed receipt. Earlier copies are kept.',
  }[status.state] || 'This server has not attempted an automatic backup.');
}

function automaticBackupRetentionText(status) {
  if (status.retention_pending) {
    return 'Automatic cleanup is blocked by an unresolved deletion attempt. Physical removal is unconfirmed; do not assume that copy was preserved. Repair the pending state before cleanup can run again. There is no automatic retry.';
  }
  if (status.retention_uncertain) {
    return 'The cleanup journal could not be inspected. Whether an earlier removal completed is unknown. Repair the journal before enabling cleanup again. There is no automatic retry.';
  }
  const marked = Number.isInteger(status.retention_marked) ? status.retention_marked : 0;
  const retained = Number.isInteger(status.retention_retained) ? status.retention_retained : 0;
  if (marked > 0) {
    return `${marked} older automatic ${marked === 1 ? 'copy was' : 'copies were'} marked for removal. Physical removal is unconfirmed.`
      + (retained > 0 ? ` ${retained} older ${retained === 1 ? 'candidate was' : 'candidates were'} not marked.` : '');
  }
  if (retained > 0) {
    return `No older automatic copies were marked for removal. ${retained} older ${retained === 1 ? 'candidate was' : 'candidates were'} not marked.`;
  }
  if (status.retention_state === 'succeeded' && status.retention_reason === 'completed') return 'Cleanup finished without marking an older automatic copy.';
  if (status.retention_state === 'skipped') return 'Automatic cleanup was skipped; no older copy was marked for removal this attempt. There is no automatic retry this session.';
  if (status.retention_warning || ['blocked', 'failed', 'unavailable'].includes(status.retention_state)) {
    return 'Automatic cleanup is unavailable or incomplete; review the cleanup warning. There is no automatic retry this session.';
  }
  if (status.retention_enabled) {
    return 'Cleanup is enabled after a new verified automatic startup backup. The new copy and newest confirmed copies are kept; manual, foreign, uncertain and busy copies are protected. Saving or reloading settings creates no backup and deletes no file.';
  }
  return 'Automatic cleanup is not enabled for future backups. The requested count is not enforced.';
}

function automaticBackupCredentialText(status) {
  if (status.credential_state === 'pending') {
    return 'Offline credential setup is pending. Startup backups cannot be enabled until the setup outcome is resolved. Use the offline inspect command for local metadata; this page never checks the credential store.';
  }
  if (status.credential_configured && status.credential_state === 'ready') {
    return `An encrypted credential reference is configured${status.credential_generation ? ` for generation ${status.credential_generation}` : ''}. Its availability has not been checked here. The backup worker reads it only when an encrypted backup is due; saving, reloading and status do not check the credential store.`;
  }
  return 'Encrypted startup backups need an offline credential setup for this gallery and destination. This page accepts no password and never reuses the manual bundle password fields.';
}

function automaticBackupEncryptionIssue() {
  const status = automaticBackupStatus;
  if (status?.credential_state === 'pending') return 'Resolve the pending offline credential setup before enabling encrypted backups';
  if (!status?.credential_configured || status.credential_state !== 'ready'
      || status.credential_provider !== 'windows-credential-manager-v1' || !status.credential_generation
      || status.protection !== 'encrypted' || status.reason === 'destination_changed') {
    return 'Complete a new offline encrypted credential setup for this gallery and destination, then reload the settings';
  }
  if (automaticBackupDraftCredentialGeneration !== status.credential_generation) {
    return 'Reload saved settings and confirm the changed credential generation before enabling encrypted backups';
  }
  return '';
}

function automaticBackupRetentionIssue() {
  if (!automaticBackupStatus || automaticBackupUnavailable) return 'Read the automatic backup settings before enabling cleanup';
  if (automaticBackupStatus.retention_pending) return 'Repair the unresolved deletion attempt before enabling cleanup';
  if (automaticBackupStatus.retention_uncertain) return 'Repair the unreadable cleanup journal before enabling cleanup';
  if (!automaticBackupStatus.mode_revision || !automaticBackupStatus.shared_mode_id) {
    return 'Install Windows shared archive mode offline before enabling cleanup, then reload these settings';
  }
  return '';
}

function receiveAutomaticBackup(status, replaceDraft = false) {
  automaticBackupStatus = status;
  automaticBackupUnavailable = false;
  const fields = automaticBackupElements();
  const stamp = status.verified_at ? ` Verified at ${new Date(status.verified_at).toLocaleString()}.` : '';
  const text = automaticBackupOutcomeText(status) + stamp;
  if (fields.status.textContent !== text) fields.status.textContent = text;
  fields.warning.textContent = status.warning || '';
  fields.warning.hidden = !status.warning;
  fields.retentionWarning.textContent = status.retention_warning || '';
  fields.retentionWarning.hidden = !status.retention_warning;
  fields.retentionStatus.textContent = automaticBackupRetentionText(status);
  fields.credentialStatus.textContent = automaticBackupCredentialText(status);
  fields.mode.textContent = status.shared_mode_id && status.mode_revision
    ? `Installed shared archive mode: ${status.shared_mode_id}. Cleanup permission applies to this mode and destination.`
    : 'Cleanup needs Windows shared archive mode installed offline while every server using this destination is stopped. Use the documented offline shared-mode installer, then Reload saved settings. This page never installs it.';
  fields.destination.textContent = status.destination
    ? `Automatic backup destination: ${status.destination}` : 'The automatic backup destination is unavailable.';
  if (replaceDraft || !automaticBackupDirty) {
    fields.enabled.checked = !!status.enabled;
    fields.age.value = status.minimum_age_hours ?? '';
    fields.keep.value = status.keep_last ?? '';
    fields.consent.checked = false;
    fields.protection.value = status.protection || '';
    fields.encryptedConsent.checked = false;
    fields.recoveryConsent.checked = false;
    fields.retentionEnabled.checked = !!status.retention_enabled;
    fields.deleteConsent.checked = false;
    automaticBackupDraftRevision = status.revision;
    automaticBackupDraftDestination = status.destination;
    automaticBackupDraftModeRevision = status.mode_revision || null;
    automaticBackupDraftCredentialGeneration = status.credential_generation || null;
    automaticBackupDirty = false;
  }
  syncAutomaticBackupControls();
}

async function refreshAutomaticBackup(replaceDraft = false) {
  if (!els.bundlesDialog.open || automaticBackupReading || automaticBackupSaving) return;
  const epoch = automaticBackupEpoch;
  const draftEpoch = automaticBackupEditEpoch;
  automaticBackupReading = true;
  syncAutomaticBackupControls();
  try {
    const status = await api('/api/gallery/automatic-backup' + (replaceDraft ? '/reload' : ''),
      replaceDraft ? { method: 'POST' } : undefined);
    if (epoch === automaticBackupEpoch) {
      const newerDraft = draftEpoch !== automaticBackupEditEpoch;
      receiveAutomaticBackup(status, replaceDraft && !newerDraft);
      if (replaceDraft) automaticBackupSay(newerDraft
        ? 'Saved settings were read. Your newer unsaved edits are kept.' : 'Saved settings reloaded.');
    }
  } catch (err) {
    if (epoch === automaticBackupEpoch) {
      automaticBackupUnavailable = true;
      automaticBackupSay(`Automatic backup settings could not be read. ${err.message}`, true);
    }
  } finally {
    automaticBackupReading = false;
    syncAutomaticBackupControls();
  }
}

function automaticBackupIssue() {
  const fields = automaticBackupElements();
  const held = automaticBackupSaving ? 'Wait for these settings to finish saving'
    : automaticBackupReading ? 'Wait for the automatic backup settings'
      : !automaticBackupStatus || automaticBackupUnavailable ? 'Read the automatic backup settings first'
        : automaticBackupStatus.revision === null ? 'Repair the unreadable automatic backup settings file first'
          : bundleActionHold();
  if (held) return { reason: held };
  if (!automaticBackupDirty) return { reason: 'Change a setting before saving' };
  if (!fields.enabled.checked) return { reason: '' };
  for (const [input, label] of [[fields.age, 'Minimum age'], [fields.keep, 'Requested copies']]) {
    const value = input.valueAsNumber;
    if (input.validity.badInput || !Number.isInteger(value) || value < 1 || value > Number(input.max)) {
      return { reason: `${label} must be a whole number from 1 to ${input.max}`, field: input };
    }
  }
  if (!automaticBackupStatus.destination) return { reason: 'Read a valid backup destination before enabling automatic backups' };
  if (automaticBackupDraftDestination !== automaticBackupStatus.destination) {
    return { reason: 'Reload saved settings and confirm the changed backup destination before enabling automatic backups' };
  }
  if (!['unencrypted', 'encrypted'].includes(fields.protection.value)) {
    return { reason: 'Choose archive protection before enabling automatic backups', field: fields.protection };
  }
  if (automaticBackupStatus.credential_state === 'pending') {
    return { reason: 'Resolve the pending offline credential setup before enabling automatic backups', field: fields.protection };
  }
  if (fields.protection.value === 'encrypted') {
    const encryptionIssue = automaticBackupEncryptionIssue();
    if (encryptionIssue) return { reason: encryptionIssue, field: fields.protection };
    if (!fields.encryptedConsent.checked) return { reason: 'Explicitly allow unattended encrypted archives with the configured credential reference', field: fields.encryptedConsent };
    if (!fields.recoveryConsent.checked) return { reason: 'Confirm independent password recovery outside this Windows account and PC', field: fields.recoveryConsent };
  } else if (!fields.consent.checked) {
    return { reason: 'Explicitly allow unencrypted images and prompts at the displayed destination', field: fields.consent };
  }
  if (fields.retentionEnabled.checked) {
    const retentionIssue = automaticBackupRetentionIssue();
    if (retentionIssue) return { reason: retentionIssue, field: fields.retentionEnabled };
    if (automaticBackupDraftModeRevision !== automaticBackupStatus.mode_revision) {
      return { reason: 'Reload saved settings and confirm the changed shared archive mode before enabling cleanup', field: fields.retentionEnabled };
    }
    if (!fields.deleteConsent.checked) {
      return { reason: 'Explicitly allow irreversible deletion at the displayed destination with the chosen count', field: fields.deleteConsent };
    }
  }
  return { reason: '' };
}

function syncAutomaticBackupControls() {
  const fields = automaticBackupElements();
  if (!fields.save) return;
  const issue = automaticBackupIssue();
  fields.save.setAttribute('aria-disabled', String(!!issue.reason));
  fields.save.title = issue.reason || 'Save these choices for the next server start; no backup is created now';
  const held = automaticBackupReading || automaticBackupSaving;
  fields.reload.setAttribute('aria-disabled', String(held));
  fields.reload.title = held ? 'Wait for the current settings request' : 'Discard unsaved edits and read the saved settings';
  const retentionIssue = automaticBackupRetentionIssue();
  fields.retentionEnabled.setAttribute('aria-disabled', String(!fields.retentionEnabled.checked && !!retentionIssue));
  fields.retentionEnabled.title = retentionIssue || 'Permit cleanup only after a new verified automatic backup; saving deletes no file';
}

async function saveAutomaticBackup() {
  const fields = automaticBackupElements();
  const issue = automaticBackupIssue();
  if (issue.reason) {
    automaticBackupSay(issue.reason + '.', true);
    if (issue.field) focusBundleField(issue.field);
    return;
  }
  const body = { revision: automaticBackupDraftRevision, enabled: fields.enabled.checked };
  if (body.enabled) Object.assign(body, {
    minimum_age_hours: fields.age.valueAsNumber, keep_last: fields.keep.valueAsNumber,
    protection: fields.protection.value, destination: automaticBackupDraftDestination,
    retention_enabled: fields.retentionEnabled.checked,
    allow_managed_deletion: fields.retentionEnabled.checked && fields.deleteConsent.checked,
    expected_mode_revision: fields.retentionEnabled.checked ? automaticBackupDraftModeRevision : null,
  });
  if (body.enabled && body.protection === 'encrypted') Object.assign(body, {
    allow_encrypted: fields.encryptedConsent.checked, confirm_recovery: fields.recoveryConsent.checked,
  });
  else if (body.enabled) body.allow_unencrypted = fields.consent.checked;
  const draftEpoch = automaticBackupEditEpoch;
  ++automaticBackupEpoch;
  automaticBackupSaving = true;
  fields.deleteConsent.checked = false;
  fields.encryptedConsent.checked = false;
  fields.recoveryConsent.checked = false;
  syncAutomaticBackupControls();
  automaticBackupSay('Saving settings for the next server start. No backup is created and no file is deleted now.');
  try {
    const status = await api('/api/gallery/automatic-backup', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const newerDraft = draftEpoch !== automaticBackupEditEpoch;
    receiveAutomaticBackup(status, !newerDraft);
    // This response acknowledges our own saved base. Keep edits made while
    // waiting, but let their next save build on that base without a false
    // conflict. A status poll or another tab's save never advances this token.
    if (newerDraft) automaticBackupDraftRevision = status.revision;
    automaticBackupSay((body.enabled
      ? 'Saved for the next server start. No backup was created and no file was deleted now. '
        + (body.retention_enabled ? 'Cleanup permission applies only after a new verified automatic backup.' : 'Automatic cleanup is not enabled for future backups; the requested count is not enforced.')
      : 'Automatic backups and cleanup are off for the next server start. Saving these settings deleted no file.')
      + (newerDraft ? ' Your newer unsaved edits are kept.' : ''));
  } catch (err) {
    automaticBackupSay(`${err.message} Your unsaved edits are kept; reload saved settings before trying again if another tab changed them.`, true);
  } finally {
    automaticBackupSaving = false;
    fields.deleteConsent.checked = false;
    fields.encryptedConsent.checked = false;
    fields.recoveryConsent.checked = false;
    syncAutomaticBackupControls();
  }
}

function initAutomaticBackup() {
  const fields = automaticBackupElements();
  for (const input of [fields.enabled, fields.age, fields.keep, fields.consent, fields.protection, fields.encryptedConsent, fields.recoveryConsent, fields.retentionEnabled, fields.deleteConsent]) input.addEventListener('input', () => {
    if (input === fields.protection) fields.consent.checked = false;
    if (input !== fields.deleteConsent) fields.deleteConsent.checked = false;
    if (![fields.encryptedConsent, fields.recoveryConsent, fields.deleteConsent].includes(input)) {
      fields.encryptedConsent.checked = false;
      fields.recoveryConsent.checked = false;
    }
    ++automaticBackupEditEpoch;
    automaticBackupDirty = true;
    syncAutomaticBackupControls();
  });
  fields.retentionEnabled.addEventListener('click', event => {
    const issue = automaticBackupRetentionIssue();
    if (fields.retentionEnabled.checked && issue) {
      event.preventDefault();
      automaticBackupSay(issue + '.', true);
    }
  });
  fields.save.onclick = () => { void saveAutomaticBackup(); };
  fields.reload.onclick = () => {
    if (fields.reload.getAttribute('aria-disabled') === 'true') {
      automaticBackupSay(fields.reload.title + '.', true);
      return;
    }
    void refreshAutomaticBackup(true);
  };
}
