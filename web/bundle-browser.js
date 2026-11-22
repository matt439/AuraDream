// Tab-local view of one saved bundle. The view token travels in a header.
// Only the current page has records and thumbnail object URLs.
'use strict';

const BUNDLE_BROWSE_PATH = '/api/gallery/bundles/browse';
const BUNDLE_BROWSE_PAGE = 24;
let bundleView = null;

function bundleBrowserEl(part) { return $(part ? 'bundle-browser-' + part : 'bundle-browser'); }

function clearBundleBrowserFilters() {
  bundleBrowserEl('prompt').value = '';
  bundleBrowserEl('run').value = '';
  bundleBrowserEl('starred').checked = false;
  bundleBrowserEl('filters').hidden = true;
}

function releaseBundleView(token) {
  if (token) void fetch(BUNDLE_BROWSE_PATH, {
    method: 'DELETE', headers: { 'X-Bundle-View': token }, cache: 'no-store', keepalive: true,
  }).catch(() => {}); // A page leaving the network is also covered by server expiry.
}

function clearBundleBrowserPage(view) {
  view.read?.abort();
  view.read = null;
  clearTimeout(view.filterTimer);
  view.detail?.restore.read?.abort();
  view.detail = null;
  bundleBrowserEl('details').replaceChildren();
  bundleBrowserEl('details').hidden = true;
  bundleBrowserEl('list').replaceChildren();
  bundleBrowserEl('list').hidden = false;
  bundleBrowserEl('pages').hidden = true;
  for (const url of view.urls) URL.revokeObjectURL(url);
  view.urls.clear();
}

function forgetBundleView() {
  const view = bundleView;
  if (!view) return;
  bundleView = null;
  view.opening.abort();
  clearBundleBrowserPage(view);
  clearTimeout(view.expiry);
  releaseBundleView(view.token);
  clearBundleBrowserFilters();
  bundleBrowserEl('recovery').hidden = true;
  bundleBrowserEl('name').textContent = '';
  bundleBrowserEl('status').textContent = '';
  if (view.protected && !view.editPassword) els.bundleUnlock.value = '';
  syncBundleControls();
}

function closeBundleBrowser() {
  if (bundleBrowserEl('').open) bundleBrowserEl('').close();
  forgetBundleView();
}

function expireBundleBrowser(view) {
  if (bundleView !== view) return;
  forgetBundleView();
  bundleBrowserEl('status').textContent = 'This bundle view expired. Close it and choose Browse to open it again.';
  bundleBrowserEl('close').focus({preventScroll: true});
}

function armBundleBrowserExpiry(view) {
  clearTimeout(view.expiry);
  const delay = view.idleSeconds * 1000;
  view.deadline = Date.now() + delay;
  view.expiry = setTimeout(() => expireBundleBrowser(view), delay);
}

function bundleBrowserError(view, err) {
  if (bundleView !== view) return;
  clearBundleBrowserPage(view);
  clearBundleBrowserFilters();
  bundleBrowserEl('status').textContent = err.message;
  bundleBrowserEl('password').hidden = !view.protected;
  bundleBrowserEl('recovery').hidden = false;
  // A failed or expired view is released before another attempt. Preserve
  // its name alone, so the same saved archive can be explicitly retried.
  releaseBundleView(view.token);
  view.token = null;
  clearTimeout(view.expiry);
  if (!bundleBrowserEl('').contains(document.activeElement)) bundleBrowserEl('close').focus({preventScroll: true});
}

async function openBundleBrowser(name, protectedFile) {
  forgetBundleView();
  // Preallocate the capability so Close can revoke it even if verification
  // succeeded on the server but the response body never reaches this tab.
  const token = Array.from(crypto.getRandomValues(new Uint8Array(32)), byte => byte.toString(16).padStart(2, '0')).join('');
  const view = { name, protected: protectedFile, opening: new AbortController(), token, ready: false,
    read: null, urls: new Set(), offset: 0, detail: null };
  bundleView = view;
  bundleBrowserEl('name').textContent = name;
  bundleBrowserEl('status').textContent = 'Opening and verifying the saved bundle…';
  if (!bundleBrowserEl('').open) bundleBrowserEl('').showModal();
  bundleBrowserEl('close').focus({ preventScroll: true });
  try {
    const receipt = await api(BUNDLE_BROWSE_PATH, {
      method: 'POST', cache: 'no-store', signal: view.opening.signal,
      headers: { 'Content-Type': 'application/json', 'X-Bundle-View': token },
      body: JSON.stringify({ name, ...(protectedFile ? {password: els.bundleUnlock.value} : {}) }),
    });
    // A close, sign-out or replacement during verification discards this
    // reply but still gives the server its token back.
    if (bundleView !== view || !bundleBrowserEl('').open) {
      releaseBundleView(receipt.token);
      return;
    }
    if (receipt.token !== token) {
      releaseBundleView(receipt.token);
      throw new Error('The server returned an unexpected bundle view. Open the bundle again.');
    }
    view.ready = true;
    view.idleSeconds = receipt.idle_seconds;
    if (protectedFile) els.bundleUnlock.value = '';
    bundleBrowserEl('filters').hidden = false;
    armBundleBrowserExpiry(view);
    await readBundleBrowserPage(view, 0);
  } catch (err) {
    // An explicit rejection did not acquire the requested capability.
    if (!view.ready && err.status >= 400 && err.status < 500) view.token = null;
    bundleBrowserError(view, err);
  }
}

function bundleBrowserImageURL(id) {
  return BUNDLE_BROWSE_PATH + '/thumb/' + encodeURIComponent(id);
}

async function loadBundleThumbs(view, read, cards) {
  let next = 0;
  async function worker() {
    while (!read.signal.aborted && next < cards.length) {
      const {record, image} = cards[next++];
      try {
        const blob = await api(bundleBrowserImageURL(record.id), {
          headers: { 'X-Bundle-View': view.token }, cache: 'no-store', signal: read.signal,
        }, 'blob');
        if (read.signal.aborted) return;
        const url = URL.createObjectURL(blob);
        view.urls.add(url);
        image.src = url;
        if (view.detail?.id === record.id) view.detail.image.src = url;
        armBundleBrowserExpiry(view);
      } catch (err) {
        if (read.signal.aborted) return;
        if (err.code === 'bundle_view_closed') { bundleBrowserError(view, err); return; }
        image.alt = 'Thumbnail unavailable';
        if (view.detail?.id === record.id) view.detail.image.alt = image.alt;
      }
    }
  }
  await Promise.all(Array.from({length: Math.min(3, cards.length)}, worker));
}

function bundleBrowserImageLabel(record) {
  return '#' + record.id + (record.favorite ? ' · Starred' : '')
    + (record.settings.binned_at ? ' · In recycle bin' : '');
}

function showBundleBrowserDetails(view, record, card, thumbnail) {
  const detail = bundleBrowserEl('details');
  const back = document.createElement('button');
  back.type = 'button'; back.className = 'ghost small'; back.textContent = 'Back to images';
  const image = document.createElement('img');
  image.alt = thumbnail.alt || 'Archived image #' + record.id;
  if (thumbnail.hasAttribute('src')) image.src = thumbnail.src;
  const title = document.createElement('h3');
  title.textContent = 'Image ' + bundleBrowserImageLabel(record);
  const settings = document.createElement('dl');
  const fields = {Prompt: record.prompt, 'Run ID': record.run_id, ...record.settings};
  for (const [key, value] of Object.entries(fields)) {
    if (value == null || value === '' || key === 'prompt' || key === 'run_id') continue;
    const term = document.createElement('dt');
    const words = key.replaceAll('_', ' ');
    term.textContent = words[0].toUpperCase() + words.slice(1);
    const description = document.createElement('dd');
    description.textContent = typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value);
    settings.append(term, description);
  }
  const restore = bundleRestoreControl(view, record);
  const actions = document.createElement('div');
  actions.className = 'bundle-browser-actions';
  actions.append(back, restore.button);
  view.detail = {id: record.id, image, restore};
  detail.replaceChildren(title, actions, restore.panel, image, settings);
  detail.hidden = false;
  for (const part of ['filters', 'list', 'pages']) bundleBrowserEl(part).hidden = true;
  bundleBrowserEl('status').textContent = '';
  back.onclick = () => {
    restore.read?.abort();
    view.detail = null;
    detail.hidden = true;
    detail.replaceChildren();
    bundleBrowserEl('filters').hidden = bundleBrowserEl('list').hidden = false;
    bundleBrowserEl('pages').hidden = view.total <= BUNDLE_BROWSE_PAGE;
    bundleBrowserEl('status').textContent = view.summary;
    card.focus({preventScroll: true});
    card.scrollIntoView({block: 'nearest'});
  };
  back.focus({preventScroll: true});
  bundleBrowserEl('').scrollTop = 0;
}

async function readBundleBrowserPage(view, offset) {
  clearBundleBrowserPage(view);
  const read = new AbortController();
  view.read = read;
  view.offset = offset;
  bundleBrowserEl('status').textContent = 'Reading archived images…';
  try {
    const result = await api(BUNDLE_BROWSE_PATH + '/images', {
      method: 'POST', cache: 'no-store', signal: read.signal,
      headers: { 'X-Bundle-View': view.token, 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: bundleBrowserEl('prompt').value, run: bundleBrowserEl('run').value,
        starred: bundleBrowserEl('starred').checked, offset }),
    });
    if (read.signal.aborted || bundleView !== view) return;
    armBundleBrowserExpiry(view);
    view.total = result.total;
    const last = offset + result.images.length;
    view.summary = result.total ? (offset + 1) + '–' + last + ' of ' + result.total
      + (result.total === 1 ? ' image' : ' images') : 'No matching images.';
    bundleBrowserEl('status').textContent = view.summary;
    bundleBrowserEl('pages').hidden = result.total <= BUNDLE_BROWSE_PAGE;
    for (const [part, reason] of [['prev', offset ? '' : 'This is the first page'],
      ['next', last < result.total ? '' : 'This is the last page']]) {
      const button = bundleBrowserEl(part);
      button.setAttribute('aria-disabled', String(!!reason));
      button.title = reason || button.textContent;
    }
    const cards = result.images.map((record) => {
      const button = document.createElement('button');
      button.className = 'bundle-browser-card ghost'; button.type = 'button';
      button.dataset.bundleImage = record.id;
      const image = document.createElement('img');
      image.alt = '';
      const id = document.createElement('strong');
      id.textContent = bundleBrowserImageLabel(record);
      const prompt = document.createElement('span');
      prompt.textContent = record.settings.imported_name
        ? `Imported · ${record.settings.imported_name}`
        : record.prompt;
      button.append(image, id, prompt);
      button.onclick = () => showBundleBrowserDetails(view, record, button, image);
      return {button, image, record};
    });
    bundleBrowserEl('list').replaceChildren(...cards.map((c) => c.button));
    await loadBundleThumbs(view, read, cards);
  } catch (err) {
    if (!read.signal.aborted) bundleBrowserError(view, err);
  }
}

function initBundleBrowser() {
  const dialog = bundleBrowserEl('');
  dialog.addEventListener('close', () => { if (!dialog.open) forgetBundleView(); });
  bundleBrowserEl('close').onclick = closeBundleBrowser;
  els.bundlesDialog.addEventListener('close', closeBundleBrowser);
  window.addEventListener('pagehide', closeBundleBrowser);
  // Throttled timers can run late when a background tab wakes.
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden && bundleView?.token && Date.now() >= bundleView.deadline) expireBundleBrowser(bundleView);
  });
  bundleBrowserEl('password').onclick = () => {
    if (bundleView) bundleView.editPassword = true;
    closeBundleBrowser();
    focusBundleUnlock();
  };
  bundleBrowserEl('retry').onclick = () => {
    const view = bundleView;
    if (!view) return;
    if (view.protected && !els.bundleUnlock.value) {
      bundleBrowserEl('password').click();
      return;
    }
    view.editPassword = true;
    void openBundleBrowser(view.name, view.protected);
  };
  const filter = () => {
    const view = bundleView;
    if (!view?.ready || !view.token) return;
    clearBundleBrowserPage(view);
    bundleBrowserEl('status').textContent = 'Reading archived images…';
    view.filterTimer = setTimeout(() => { void readBundleBrowserPage(view, 0); }, 160);
  };
  for (const part of ['prompt', 'run']) bundleBrowserEl(part).oninput = filter;
  bundleBrowserEl('starred').onchange = filter;
  for (const [part, delta] of [['prev', -1], ['next', 1]]) {
    const button = bundleBrowserEl(part);
    button.onclick = () => {
      if (button.getAttribute('aria-disabled') === 'true') {
        bundleBrowserEl('status').textContent = button.title + '.';
        return;
      }
      if (bundleView?.ready && bundleView.token) void readBundleBrowserPage(bundleView, bundleView.offset + delta * BUNDLE_BROWSE_PAGE);
    };
  }
}
