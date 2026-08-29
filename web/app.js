'use strict';

const $ = (id) => document.getElementById(id);
const PAGE = 48;

const els = {
  prompt: $('prompt'), negative: $('negative'), negField: $('negative-field'),
  width: $('width'), height: $('height'), steps: $('steps'), cfg: $('cfg'),
  mu: $('mu'), seed: $('seed'), batch: $('batch'),
  generate: $('btn-generate'), dice: $('btn-dice'), unload: $('btn-unload'),
  progress: $('progress'), fill: $('progress-fill'), progressText: $('progress-text'),
  error: $('error'), viewer: $('viewer'), gallery: $('gallery'),
  count: $('gallery-count'), search: $('search'), favOnly: $('fav-only'),
  more: $('btn-more'), presets: $('presets'), sizeHint: $('size-hint'),
  dot: $('status-dot'), statusText: $('status-text'), vram: $('vram'),
  lightbox: $('lightbox'), lbImg: $('lb-img'), lbMeta: $('lb-meta'), lbClose: $('lb-close'),
  lbReuse: $('lb-reuse'),
};

const SIZES = [
  ['1:1', 1024, 1024], ['3:4', 896, 1200], ['4:3', 1200, 896],
  ['16:9', 1360, 768], ['9:16', 768, 1360], ['2K', 2048, 2048],
];

let offset = 0;
let total = 0;
let busy = false;
let defaultsApplied = false;
let statusPending = false;
let lightboxRec = null;

// ------------------------------------------------------------------ helpers

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* non-JSON body */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return res.status === 204 ? null : res.json();
}

function showError(message) {
  els.error.textContent = message || '';
  els.error.hidden = !message;
}

// -------------------------------------------------------------------- sizes

function buildPresets() {
  SIZES.forEach(([label, w, h]) => {
    const b = document.createElement('button');
    b.textContent = label;
    b.onclick = () => { els.width.value = w; els.height.value = h; syncSize(); };
    els.presets.appendChild(b);
  });
}

function syncSize() {
  const w = +els.width.value, h = +els.height.value;
  [...els.presets.children].forEach((b, i) => {
    b.classList.toggle('active', SIZES[i][1] === w && SIZES[i][2] === h);
  });
  const tight = w * h > 1600 * 1600;
  // 2K on a 16GB card is the edge of what fits once the DiT is resident.
  els.sizeHint.textContent = `${((w * h) / 1e6).toFixed(1)} MP` +
    (tight ? ' — may run tight on 16GB VRAM' : '');
  els.sizeHint.classList.toggle('warn', tight);
}

function syncCfg() {
  const off = +els.cfg.value <= 1;
  els.negField.classList.toggle('disabled', off);
  els.negative.disabled = off;
}

// ------------------------------------------------------------------- status

// Seed the controls from the server's profile for the loaded checkpoint. Turbo
// wants 8 steps / CFG 0 / mu 1.15; FLUX wants 28 / 3.5 and no mu at all, since
// it derives its own resolution-dependent shift. Applied once, before the user
// has had a chance to touch anything.
function applyDefaults(d) {
  if (defaultsApplied || !d || d.steps === undefined) return;
  defaultsApplied = true;
  els.steps.value = d.steps;
  els.cfg.value = d.guidance_scale;
  els.mu.value = d.mu ?? '';
  syncCfg();
}

// Anything the user picks - by hand or through Reuse - takes the controls off
// the profile. /api/status reads CUDA memory, which stalls behind a model load,
// so its reply can land minutes late and used to wipe steps/CFG/mu long after
// the page had settled.
function ownControls() { defaultsApplied = true; }

async function refreshStatus() {
  if (statusPending) return;  // a stalled poll must not queue more behind it
  statusPending = true;
  try {
    const s = await api('/api/status');
    applyDefaults(s.defaults);
    els.dot.className = `dot ${s.state}`;
    els.statusText.textContent =
      s.state === 'ready' ? s.model_id.split('/').pop()
      : s.state === 'loading' ? 'loading weights…'
      : s.state === 'error' ? 'load failed'
      : 'not loaded';
    if (s.state === 'error' && s.detail) showError(s.detail);
    els.vram.textContent = s.vram_total_mb
      ? `${s.device_name ?? ''} · ${(s.vram_used_mb / 1024).toFixed(1)}/${(s.vram_total_mb / 1024).toFixed(1)} GB · ${s.quantization}`
      : '';
  } catch { /* server not up yet */ } finally { statusPending = false; }
}

// --------------------------------------------------------------- generation

function collect() {
  const seed = els.seed.value.trim();
  return {
    prompt: els.prompt.value.trim(),
    negative_prompt: els.negative.value.trim() || null,
    width: +els.width.value,
    height: +els.height.value,
    steps: +els.steps.value,
    guidance_scale: +els.cfg.value,
    mu: els.mu.value === '' ? null : +els.mu.value,
    seed: seed === '' ? null : +seed,
    num_images: +els.batch.value,
  };
}

function setBusy(on) {
  busy = on;
  els.generate.disabled = on;
  els.generate.textContent = on ? 'Generating…' : 'Generate';
  els.progress.hidden = !on;
  if (!on) { els.fill.style.width = '0%'; els.progressText.textContent = ''; }
}

async function generate() {
  if (busy) return;
  const payload = collect();
  if (!payload.prompt) { showError('Enter a prompt first.'); return; }
  showError('');
  setBusy(true);
  els.progressText.textContent = 'queued…';

  try {
    const { job_id } = await api('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    await streamJob(job_id);
  } catch (err) {
    showError(err.message);
    setBusy(false);
  }
}

function streamJob(jobId) {
  return new Promise((resolve) => {
    const source = new EventSource(`/api/jobs/${jobId}/stream`);

    source.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'progress') {
        els.fill.style.width = `${(data.step / data.total_steps) * 100}%`;
        els.progressText.textContent = `${data.step}/${data.total_steps}`;
      } else if (data.type === 'status' && data.status === 'running') {
        els.progressText.textContent = 'denoising…';
      } else if (data.type === 'done') {
        source.close();
        setBusy(false);
        if (data.status === 'error') showError(data.error);
        else { showResults(data.images); resetGallery(); }
        resolve();
      }
    };

    source.onerror = () => {
      source.close();
      setBusy(false);
      showError('Lost connection to the generation stream.');
      resolve();
    };
  });
}

function showResults(images) {
  els.viewer.innerHTML = '';
  els.viewer.classList.toggle('multi', images.length > 1);
  images.forEach((rec) => {
    const img = document.createElement('img');
    img.src = rec.url;
    img.alt = rec.prompt;
    img.onclick = () => openLightbox(rec);
    els.viewer.appendChild(img);
  });
}

// -------------------------------------------------------------------gallery

function resetGallery() { offset = 0; els.gallery.innerHTML = ''; loadGallery(); }

async function loadGallery() {
  const params = new URLSearchParams({ limit: PAGE, offset });
  if (els.search.value.trim()) params.set('search', els.search.value.trim());
  if (els.favOnly.checked) params.set('favorites', 'true');

  const page = await api(`/api/gallery?${params}`);
  total = page.total;
  els.count.textContent = total ? `(${total})` : '';
  page.items.forEach((rec) => els.gallery.appendChild(card(rec)));
  offset += page.items.length;
  els.more.hidden = offset >= total;
}

function card(rec) {
  const div = document.createElement('div');
  div.className = `card${rec.favorite ? ' fav' : ''}`;

  const img = document.createElement('img');
  img.src = rec.url;
  img.alt = rec.prompt;
  img.loading = 'lazy';
  img.onclick = () => openLightbox(rec);

  const star = tool(rec.favorite ? '★' : '☆', 'Favorite', async () => {
    await api(`/api/images/${rec.id}/favorite?value=${!rec.favorite}`, { method: 'POST' });
    rec.favorite = !rec.favorite;
    div.classList.toggle('fav', rec.favorite);
    star.textContent = rec.favorite ? '★' : '☆';
  });

  const tools = document.createElement('div');
  tools.className = 'tools';
  tools.append(
    tool('↻', 'Reuse settings', () => reuse(rec)),
    star,
    tool('⌦', 'Delete', async () => {
      await api(`/api/images/${rec.id}`, { method: 'DELETE' });
      div.remove();
      total -= 1;
      els.count.textContent = total ? `(${total})` : '';
    }),
  );

  div.append(img, tools);
  return div;
}

function tool(label, title, onClick) {
  const b = document.createElement('button');
  b.textContent = label;
  b.title = title;
  b.onclick = onClick;
  return b;
}

function reuse(rec) {
  ownControls();
  els.prompt.value = rec.prompt;
  els.negative.value = rec.negative_prompt ?? '';
  els.width.value = rec.width;
  els.height.value = rec.height;
  els.steps.value = rec.steps;
  els.cfg.value = rec.guidance_scale;
  // Turbo's flow shift. Records from a pipeline that derives its own carry null.
  els.mu.value = rec.mu ?? '';
  els.seed.value = rec.seed;
  syncSize();
  syncCfg();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ----------------------------------------------------------------- lightbox

function openLightbox(rec) {
  lightboxRec = rec;
  els.lbImg.src = rec.url;
  els.lbMeta.textContent =
    `${rec.prompt} — ${rec.width}×${rec.height}, ${rec.steps} steps, ` +
    `CFG ${rec.guidance_scale}, ${rec.mu == null ? 'no mu' : `mu ${rec.mu}`}, ` +
    `seed ${rec.seed}, ${(rec.duration_ms / 1000).toFixed(1)}s`;
  els.lightbox.hidden = false;
}

function closeLightbox() { els.lightbox.hidden = true; els.lbImg.src = ''; lightboxRec = null; }

// --------------------------------------------------------------------- wire

let searchTimer;
els.generate.onclick = generate;
els.dice.onclick = () => { els.seed.value = ''; };
els.unload.onclick = async () => {
  await api('/api/model/unload', { method: 'POST' });
  refreshStatus();
};
els.more.onclick = loadGallery;
els.width.oninput = els.height.oninput = syncSize;
els.steps.oninput = els.mu.oninput = els.seed.oninput = ownControls;
els.cfg.oninput = () => { ownControls(); syncCfg(); };
els.favOnly.onchange = resetGallery;
els.search.oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(resetGallery, 250); };
els.lbClose.onclick = closeLightbox;
// The just-generated image only exists in the viewer, so reuse has to be
// reachable from the lightbox as well as from a gallery card.
els.lbReuse.onclick = () => { if (lightboxRec) { reuse(lightboxRec); closeLightbox(); } };
els.lightbox.onclick = (e) => { if (e.target === els.lightbox) closeLightbox(); };

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeLightbox();
  // Ctrl/Cmd+Enter generates from anywhere, including inside the prompt box.
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) generate();
});

buildPresets();
syncSize();
syncCfg();
refreshStatus();
loadGallery();
setInterval(refreshStatus, 3000);
