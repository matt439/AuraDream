// Injected by smoke-bundle-browse. Build archives with the real exporter before
// the app starts polling so gallery maintenance can reserve the scratch server.
const __archivePrompt = 'moonlit wireframe observatory';
const __bundlePrefix = '/api/gallery/bundles';
const __allGallery = '/api/gallery?limit=100&all_models=true';

async function __waitFor(predicate, ms = 25000) {
  const end = Date.now() + ms;
  do {
    if (await predicate()) return true;
    await new Promise(resolve => setTimeout(resolve, 50));
  } while (Date.now() < end);
  return false;
}

async function __json(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  });
  if (!response.ok) throw Error(path + ' returned ' + response.status);
  return response.json();
}

async function __backup(password) {
  const operation = await __json(__bundlePrefix + '/export', {keep_gallery: true, password});
  return __waitFor(async () => {
    const status = await __json(__bundlePrefix + '/status');
    if (status.state === 'failed') throw Error(status.error);
    return status.id === operation.id && status.state === 'done' && status.result.records === 3;
  });
}

async function __prepareBrowseFixture() {
  const fake = await __json('/api/fake/activity', {});
  const original = await __json(__allGallery);
  if (!Array.isArray(fake.counted) || original.total !== 2
      || !original.items.every(row => row.run_id === 'planted')
      || (await __json(__bundlePrefix)).bundles.length !== 0) return false;

  const {job_id} = await __json('/api/generate', {
    prompt: __archivePrompt, negative_prompt: 'thick fog', seed: 79713,
    width: 256, height: 256, steps: 1, guidance_scale: 2, true_cfg: 3,
  });
  if (!await __waitFor(async () => (await __json('/api/jobs/' + job_id)).status === 'done')) return false;
  const archiveOnly = (await __json(__allGallery)).items.find(row =>
    row.job_id === job_id && row.prompt === __archivePrompt && row.seed === 79713);
  if (archiveOnly?.negative_prompt !== 'thick fog'
      || !await __backup(null) || !await __backup('lantern room')) return false;

  // This ID came from this smoke's own completed job. Keep the two original
  // records so their bytes and metadata can be compared after browsing.
  const removed = await fetch('/api/images/' + archiveOnly.id, {method: 'DELETE'});
  return removed.ok && (await __json(__allGallery)).total === 2;
}

async function __workingSnapshot() {
  const page = await __json(__allGallery);
  const bin = await __json('/api/gallery/bin?all_models=true');
  const runs = await __json('/api/runs?all_models=true');
  const hashes = await Promise.all(page.items.map(async row => {
    const response = await fetch(row.url, {cache: 'no-store'});
    if (!response.ok) throw Error('Missing working PNG ' + row.id);
    const hash = await crypto.subtle.digest('SHA-256', await response.arrayBuffer());
    return Array.from(new Uint8Array(hash)).join('.');
  }));
  return JSON.stringify({page, bin, runs, hashes});
}

function __input(selector, value) {
  const field = document.querySelector(selector);
  field.value = value;
  field.dispatchEvent(new Event('input', {bubbles: true}));
}

function __cardCount(count) {
  const cards = document.querySelectorAll('#bundle-browser-list [data-bundle-image]');
  return cards.length === count && [...cards].every(card => {
    const thumb = card.querySelector('img');
    return thumb && thumb.complete && thumb.naturalWidth > 0;
  });
}

function __scrollTo(selector) {
  document.querySelector(selector).scrollIntoView({block: 'center', behavior: 'instant'});
}

// Keep replies genuine. One successful open can be held after the server
// verified its archive so a real Close can overtake its delivery to the UI.
const __views = [];
const __viewResponses = [];
let __holdOpen = false;
let __deliverOpen;

function __observeBundleTraffic() {
  const original = window.fetch.bind(window);
  window.fetch = async (input, options) => {
    const response = await original(input, options);
    const path = new URL(response.url).pathname;
    if (!path.startsWith(__bundlePrefix + '/browse')) return response;
    const method = options?.method || 'GET';
    const token = new Headers(options?.headers).get('X-Bundle-View');
    __viewResponses.push({url: response.url, method, token, ok: response.ok, cache: response.headers.get('Cache-Control')});
    if (path === __bundlePrefix + '/browse' && method === 'POST' && response.ok) {
      __views.push((await response.clone().json()).token);
      if (__holdOpen) {
        __holdOpen = false;
        await new Promise(resolve => { __deliverOpen = resolve; });
      }
    }
    return response;
  };
}

function __releasedAllViews(count) {
  return __views.length === count && __views.every(token =>
    __viewResponses.some(row => row.method === 'DELETE' && row.ok && row.token === token));
}

async function __serverReleasedLastView() {
  const response = await fetch(__bundlePrefix + '/browse/images', {
    method: 'POST',
    headers: {'Content-Type': 'application/json', 'X-Bundle-View': __views.at(-1)},
    body: '{}',
  });
  return response.status === 410;
}
