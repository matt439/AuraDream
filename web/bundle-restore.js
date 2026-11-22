// The restore preview and its choices belong to the currently browsed image.
'use strict';

const BUNDLE_RESTORE_PATH = BUNDLE_BROWSE_PATH + '/restore';
const BUNDLE_DEPENDENCY_ROLES = {
  reference: 'Reference', refine: 'Refine source', edit: 'Instruction edit source',
  detailer: 'Detailer source', tiled_refine: 'Tiled Refine source', controlnet: 'ControlNet guide', mask: 'Mask',
};

function syncBundleRestoreControls(hold = bundleActionHold()) {
  const restore = bundleView?.detail?.restore;
  if (!restore) return;
  const sending = restore.sending ? 'The restore request is being sent' : '';
  const reason = sending || (restore.read ? 'Wait for the restore preview' : '')
    || (!restore.preview ? 'Read the restore preview first' : '') || hold;
  for (const [button, why] of [[restore.confirm, reason], [restore.button, sending], [restore.cancel, sending]]) {
    button.setAttribute('aria-disabled', String(!!why));
    button.title = why || button.textContent;
  }
  for (const input of restore.list.querySelectorAll('input')) input.disabled = restore.sending;
}

function bundleRestoreReplayHint(restore) {
  const active = restore.preview.dependencies.filter(link => link.used_by === restore.id || restore.selected.has(link.used_by));
  const notes = [];
  if (active.some(link => link.kind === 'image' && link.status === 'available' && !restore.selected.has(link.id))) {
    notes.push('Include required source images to keep replay available.');
  }
  if (active.some(link => link.status === 'missing')) notes.push('An unavailable source or mask may prevent replay.');
  if (active.some(link => link.binned)) notes.push('Bring required images out of the recycle bin before replay.');
  restore.replay.textContent = notes.join(' ');
  restore.replay.hidden = notes.length === 0;
}

function renderBundleRestoreDependencies(restore) {
  const byDependency = new Map();
  for (const link of restore.preview.dependencies) {
    // Keep an unproven use visible even when the same image can fill another role.
    const key = link.id == null ? link : `${link.kind}:${link.id}:${link.status}`;
    if (!byDependency.has(key)) byDependency.set(key, []);
    byDependency.get(key).push(link);
  }
  const rows = [...byDependency.values()].map(links => {
    const link = links[0];
    const image = link.kind === 'image';
    const title = image ? (link.id == null ? 'Source image' : `Image #${link.id}`)
      : (link.id == null ? 'Mask' : `Mask ${link.id.slice(0, 12)}`);
    const item = document.createElement('li');
    if (image && link.status === 'available') {
      const label = document.createElement('label');
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.dataset.bundleDependency = link.id;
      input.onchange = () => {
        if (input.checked) restore.selected.add(link.id);
        else restore.selected.delete(link.id);
        bundleRestoreReplayHint(restore);
      };
      label.append(input, `Include image #${link.id}`);
      item.append(label);
    } else {
      const availability = { present: 'Already present', missing: 'Unavailable',
        available: 'Available in the archive' }[link.status];
      item.append(`${title} · ${availability}`);
    }
    const detail = document.createElement('p');
    detail.className = 'hint';
    const purposes = links.map(({role, used_by}) => `${BUNDLE_DEPENDENCY_ROLES[role]} for #${used_by}`);
    detail.textContent = [...new Set(purposes), ...(link.binned ? ['In the recycle bin'] : []),
      ...(link.prompt ? [excerpt(link.prompt, 120)] : [])].join(' · ');
    item.append(detail);
    return item;
  });
  restore.list.replaceChildren(...rows);
  restore.list.hidden = !rows.length;
  restore.status.textContent = restore.list.querySelector('input') ? 'Choose any archived source images to include.'
    : rows.length ? 'Replay dependencies for this image.' : 'This image has no replay dependencies.';
  bundleRestoreReplayHint(restore);
}

async function readBundleRestorePreview(view, restore) {
  const read = new AbortController();
  restore.read = read;
  restore.preview = null;
  restore.selected.clear();
  restore.list.replaceChildren();
  restore.list.hidden = restore.retry.hidden = restore.replay.hidden = true;
  restore.status.textContent = 'Checking replay dependencies…';
  syncBundleRestoreControls();
  try {
    const data = await api(BUNDLE_RESTORE_PATH + '/preview', {
      method: 'POST', cache: 'no-store', signal: read.signal,
      headers: { 'X-Bundle-View': view.token, 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_id: restore.id }),
    });
    if (read.signal.aborted || bundleView !== view) return;
    armBundleBrowserExpiry(view);
    restore.preview = data;
    renderBundleRestoreDependencies(restore);
  } catch (err) {
    if (read.signal.aborted || bundleView !== view) return;
    if (err.code === 'bundle_view_closed') bundleBrowserError(view, err);
    else {
      restore.status.textContent = err.message;
      restore.retry.hidden = false;
    }
  } finally {
    if (restore.read === read) restore.read = null;
    syncBundleRestoreControls();
  }
}

async function confirmBundleRestore(view, restore) {
  syncBundleRestoreControls();
  if (restore.confirm.getAttribute('aria-disabled') === 'true') {
    restore.status.textContent = restore.confirm.title + '.';
    return;
  }
  restore.sending = true;
  restore.status.textContent = 'Starting the restore…';
  const { error } = await startBundleOperation(BUNDLE_RESTORE_PATH, {
    method: 'POST', cache: 'no-store',
    headers: { 'X-Bundle-View': view.token, 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id: restore.id, include_ids: [...restore.selected] }),
  }, 'restore');
  restore.sending = false;
  if (bundleView !== view || view.detail?.restore !== restore) return;
  if (!error) closeBundleBrowser();
  else if (error.code === 'bundle_view_closed') bundleBrowserError(view, error);
  else {
    restore.status.textContent = error.message;
    syncBundleRestoreControls();
  }
}

function bundleRestoreControl(view, record) {
  const panel = $('bundle-restore-template').content.firstElementChild.cloneNode(true);
  const restore = { panel, id: record.id, selected: new Set() };
  for (const part of ['title', 'status', 'confirm', 'retry', 'cancel', 'replay']) {
    restore[part] = panel.querySelector('#bundle-restore-' + part);
  }
  restore.list = panel.querySelector('#bundle-restore-dependencies');
  restore.title.textContent = `Restore image #${record.id}`;
  const button = document.createElement('button');
  button.id = 'bundle-browser-restore';
  button.type = 'button';
  button.className = 'ghost small';
  button.textContent = 'Restore…';
  button.setAttribute('aria-controls', panel.id);
  button.setAttribute('aria-expanded', 'false');
  restore.button = button;
  const cancel = () => {
    if (restore.sending) {
      restore.status.textContent = restore.cancel.title + '.';
      return;
    }
    restore.read?.abort();
    panel.hidden = true;
    button.setAttribute('aria-expanded', 'false');
    button.focus({ preventScroll: true });
  };
  button.onclick = () => {
    if (!panel.hidden) { cancel(); return; }
    panel.hidden = false;
    button.setAttribute('aria-expanded', 'true');
    restore.title.focus({ preventScroll: true });
    void readBundleRestorePreview(view, restore);
  };
  restore.cancel.onclick = cancel;
  restore.retry.onclick = () => {
    restore.title.focus({ preventScroll: true });
    void readBundleRestorePreview(view, restore);
  };
  restore.confirm.onclick = () => void confirmBundleRestore(view, restore);
  return restore;
}
