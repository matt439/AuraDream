// Image-first loop - prototype DOM moves and behaviour over the shipped page.
// Runs once on a live page (indirect eval): moves the gallery into a right
// "shelf", turns the stage into caption + actions / focus image / session
// strip / keys, docks the Refine dialog over the shelf (non-modal) and adds
// the compare view. Every app function it wraps keeps its original behaviour
// below 1600 CSS px.
(() => {
  if (window.__loop) return;
  window.__loop = true;
  const WIDE = '(min-width: 901px)';
  const SHELL = '(min-width: 1600px)';
  const shell = () => window.matchMedia(SHELL).matches;
  const wide = () => window.matchMedia(WIDE).matches;
  const h = (tag, cls, text) => {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text != null) el.textContent = text;
    return el;
  };
  const short = (id) => (id || '').split('/').pop();
  document.body.classList.add('loop');

  const main = document.querySelector('main');
  const stage = document.querySelector('.stage');
  const panel = document.querySelector('.panel');

  // ------------------------------------------------------------ the shelf
  const shelf = h('section', 'shelf');
  shelf.setAttribute('aria-label', 'Gallery');
  const head = document.querySelector('.gallery-head');
  const scroller = h('div', 'shelf-scroll');
  scroller.append(els.gallery, els.sheet, els.more);
  // The prompt check's wide host (display: contents) moves with it, so the
  // check covers the shelf rather than pushing the image down.
  shelf.append(head, scroller, els.tagValidationWide);
  main.append(shelf);

  // Card size: S / M / L (M86), per browser.
  const sizes = h('span', 'shelf-sizes');
  sizes.setAttribute('role', 'group');
  sizes.setAttribute('aria-label', 'Thumbnail size');
  const CARD = { S: 120, M: 0, L: 240 };
  for (const key of Object.keys(CARD)) {
    const b = h('button', '', key);
    b.type = 'button';
    b.setAttribute('aria-pressed', String(key === 'M'));
    b.onclick = () => {
      document.documentElement.style.setProperty('--card-min', CARD[key] ? `${CARD[key]}px` : '');
      if (!CARD[key]) document.documentElement.style.removeProperty('--card-min');
      [...sizes.children].forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    };
    sizes.append(b);
  }
  head.insertBefore(sizes, els.sheetBtn);

  // ------------------------------------------------------------ the stage
  const status = h('div', 'stage-status');
  status.setAttribute('aria-label', 'Job status');
  const idle = h('span', 'loop-idle', 'Idle. Ctrl+Enter generates from anywhere.');
  status.append(idle, els.progress, els.queueNote, els.cellNote, els.error, els.lastRun);

  const bar = h('div', 'stagebar');
  const cap = h('div', 'cap');
  const capMain = h('div', 'cap-main');
  const capPrompt = h('div', 'cap-prompt');
  cap.append(capMain, capPrompt);
  const acts = h('div', 'acts');
  bar.append(cap, acts);

  const focusEl = h('div', 'focus');
  focusEl.tabIndex = -1;
  focusEl.setAttribute('aria-label', 'Selected image');
  const frame = h('div', 'focus-frame');
  const layerB = h('div', 'focus-layer b');
  const layerA = h('div', 'focus-layer a');
  const imgB = h('img');
  const imgA = h('img');
  imgA.alt = ''; imgB.alt = '';
  imgA.draggable = false; imgB.draggable = false;
  layerB.append(imgB); layerA.append(imgA);
  const swipe = h('div', 'swipe');
  const tagL = h('div', 'focus-tag left');
  const tagR = h('div', 'focus-tag right');
  const crop = h('div', 'focus-crop');
  crop.hidden = true;
  const cropLabel = h('span');
  crop.append(cropLabel);
  frame.append(layerB, layerA, swipe, tagL, tagR, crop);
  const empty = h('div', 'focus-empty');
  empty.innerHTML = '<div>Nothing selected yet.</div><div style="font-size:13px">Write a prompt and press <kbd>Ctrl+Enter</kbd> - or pick an image in the gallery.</div>';
  const hud = h('div', 'focus-hud');
  const note = h('div', 'focus-note');
  note.hidden = true;
  focusEl.append(frame, empty, hud, note);

  // The bottom row: the job's status beside the rail's Generate, then the
  // session strip the job's images land in.
  const strip = h('div', 'strip');
  const stripLabel = h('span', 'strip-label', 'This session');
  strip.append(status, stripLabel, els.viewer, els.viewerNote);

  stage.append(bar, focusEl, strip);

  // Sampling and Output headings in the rail (M84).
  const steps = els.steps.closest('.row');
  const seedRow = els.seed.closest('.row');
  panel.insertBefore(h('div', 'loop-group', 'Sampling'), steps);
  panel.insertBefore(h('div', 'loop-group', 'Output'), seedRow);

  // ----------------------------------------------------------- the topbar
  const top = h('div', 'loop-top');
  const railBtn = h('button', 'ghost', 'Controls');
  railBtn.type = 'button';
  railBtn.setAttribute('aria-pressed', 'true');
  railBtn.append(Object.assign(h('kbd'), { textContent: '[' }));
  const genTop = h('button', 'primary', 'Generate');
  genTop.id = 'loop-generate-top';
  genTop.hidden = true;
  genTop.onclick = () => generate();
  const keysBtn = h('button', 'ghost', 'Keys');
  keysBtn.type = 'button';
  keysBtn.append(Object.assign(h('kbd'), { textContent: '?' }));
  top.append(genTop, railBtn, keysBtn);
  els.vram.after(top);
  const toggleRail = () => {
    const off = document.body.classList.toggle('rail-off');
    railBtn.setAttribute('aria-pressed', String(!off));
    genTop.hidden = !off;
    layout();
  };
  railBtn.onclick = toggleRail;

  const keys = h('div', 'keys-pop');
  keys.hidden = true;
  keys.setAttribute('role', 'dialog');
  keys.setAttribute('aria-label', 'Keyboard shortcuts');
  keys.innerHTML = '<h3>Keys</h3><dl>' + [
    ['Ctrl+Enter', 'Generate - in the Refine drawer, run the pass; in Grid search, Run grid'],
    ['← →', 'previous / next image in the list it came from'],
    ['↑ ↓', 'a gallery row up / down'],
    ['Enter', 'selected image full screen (zoom, 1:1)'],
    ['F', 'star / unstar'], ['R', 'reuse its settings in the controls'],
    ['E', 'refine it (drawer over the gallery)'],
    ['C', 'compare: refined vs source, or vs the pinned A'],
    ['A', 'pin this image as A'], ['B (hold)', 'flip to the other image'],
    ['S', 'its run as a sheet'], ['Delete', 'delete (asks first)'],
    ['[', 'hide / show the controls'], ['/', 'search the gallery'], ['Esc', 'leave a drawer, close the keys'],
  ].map(([k, t]) => `<dt><kbd>${k}</kbd></dt><dd>${t}</dd>`).join('') + '</dl>';
  document.body.append(keys);
  keysBtn.onclick = () => { keys.hidden = !keys.hidden; };

  // --------------------------------------------------- focus and compare
  const view = { scale: 1, x: 0, y: 0, key: null };
  let focusRec = null;
  let focusList = null;
  let pairRec = null;      // what the focus is compared against
  let pairKind = null;     // 'source' | 'pinned'
  let pinned = null;
  let mode = 'single';     // 'single' | 'swipe' | 'side'
  let swipeAt = 0.5;
  let lastFocus = null;

  const findRec = (id) => [...viewerRecords, ...galleryRecords, ...lbList].find((r) => r.id === id) || null;
  async function sourceOf(rec) {
    if (rec.refine == null) return null;
    const known = findRec(rec.refine);
    if (known) return known;
    try { return await api(`/api/images/${rec.refine}`); } catch { return null; }
  }

  function flash(text) {
    note.textContent = text;
    note.hidden = false;
    clearTimeout(flash.t);
    flash.t = setTimeout(() => { note.hidden = true; }, 2600);
  }

  function markSelected() {
    document.querySelectorAll('.card.selected, #viewer img.selected').forEach((el) => el.classList.remove('selected'));
    if (!focusRec) return;
    const card = els.gallery.querySelector(`.card[data-id="${focusRec.id}"]`);
    if (card) card.classList.add('selected');
    const tile = els.viewer.querySelector(`img[data-id="${focusRec.id}"]`);
    if (tile) tile.classList.add('selected');
  }

  async function setFocus(rec, list, { compare = false, keep = false } = {}) {
    if (focusRec && rec && focusRec.id !== rec.id) lastFocus = focusRec;
    focusRec = rec;
    focusList = list && rec && list.includes(rec) ? list : rec ? [rec] : null;
    const key = rec ? (rec.width / rec.height).toFixed(4) : null;
    if (!keep && view.key !== key) { view.scale = 1; view.x = view.y = 0; view.key = key; }
    if (rec) imgA.src = rec.url;
    if (!compare && mode !== 'single' && !keep) mode = 'single';
    pairRec = null; pairKind = null;
    if (rec && (compare || mode !== 'single')) await choosePair();
    markSelected();
    renderBar();
    layout();
  }

  async function choosePair() {
    const rec = focusRec;
    let pair = null; let kind = null;
    if (rec.refine != null) { pair = await sourceOf(rec); kind = 'source'; }
    if (!pair && pinned && pinned.id !== rec.id) { pair = pinned; kind = 'pinned'; }
    if (focusRec !== rec) return;
    if (!pair) { mode = 'single'; return false; }
    pairRec = pair; pairKind = kind;
    if (mode === 'single') mode = 'swipe';
    imgB.src = pair.url;
    imgB.style.objectFit = kind === 'source' ? 'cover' : 'contain';
    return true;
  }

  async function toggleCompare(force) {
    if (!focusRec) return;
    if (mode !== 'single' && force !== true) { mode = 'single'; pairRec = null; renderBar(); layout(); return; }
    const ok = await choosePair();
    if (!ok) { flash('Nothing to compare with: press A on another image to pin it, or pick a refined image.'); renderBar(); layout(); return; }
    renderBar(); layout();
  }

  function pinA() {
    if (!focusRec) return;
    pinned = pinned && pinned.id === focusRec.id ? null : focusRec;
    flash(pinned ? `Image ${pinned.id} pinned as A. Pick another and press C.` : 'Pin cleared.');
    renderBar();
  }

  // Caption and actions.
  const ACTS = [
    ['star', '☆ Star', 'F', () => focusRec && setFavorite(focusRec, !focusRec.favorite)],
    ['reuse', '↻ Reuse', 'R', () => focusRec && reuse(focusRec)],
    ['refine', '⤢ Refine…', 'E', () => focusRec && openRefine(focusRec)],
    ['compare', '⇆ Compare', 'C', () => toggleCompare()],
    ['pin', 'Pin A', 'A', pinA],
    ['sep'],
    ['reference', '❐ Reference', '', () => {
      if (!focusRec) return;
      if (referenceIds.includes(focusRec.id)) return;
      setReferences([...referenceIds, focusRec.id].slice(0, 4));
    }],
    ['run', '⊞ Sheet', 'S', () => {
      if (!focusRec) return;
      if (focusRec.run_id) showRun(focusRec.run_id); else if (focusRec.job_id) showJob(focusRec.job_id);
    }],
    ['zoom', '⛶ Zoom', 'Enter', () => focusRec && openLb(focusRec, focusList)],
    ['delete', '⌦', 'Del', () => focusRec && deleteImage(focusRec)],
  ];
  const actEls = {};
  for (const [name, label, key, fn] of ACTS) {
    if (name === 'sep') { acts.append(h('span', 'sep')); continue; }
    const b = h('button', `ghost act ${name}`);
    b.type = 'button';
    b.dataset.act = name;
    const span = h('span', 'lbl', label);
    b.append(span);
    if (key) { b.append(Object.assign(h('kbd'), { textContent: key })); b.setAttribute('aria-keyshortcuts', key === 'Del' ? 'Delete' : key); }
    b.onclick = () => { if (b.getAttribute('aria-disabled') !== 'true') fn(); };
    actEls[name] = b;
    acts.append(b);
  }
  actEls.delete.classList.add('danger');
  actEls.delete.setAttribute('aria-label', 'Delete');
  actEls.refine.classList.add('primaryish');

  function renderBar() {
    const rec = focusRec;
    bar.hidden = false;
    acts.querySelectorAll('button').forEach((b) => b.setAttribute('aria-disabled', String(!rec)));
    if (!rec) {
      capMain.textContent = 'No image selected';
      capPrompt.textContent = '';
      return;
    }
    capMain.replaceChildren();
    const b = h('b', '', `Image ${rec.id}`);
    capMain.append(b, h('span', 'dim', `  ·  ${rec.width}×${rec.height}  ·  seed ${rec.seed}  ·  ${rec.steps} steps  ·  CFG ${rec.guidance_scale}${rec.true_cfg != null ? `  ·  true CFG ${rec.true_cfg}` : ''}  ·  ${short(rec.model_id)}`));
    if (rec.refine != null) {
      const chip = h('button', 'lineage', `⤢ refined from ${rec.refine} at ${rec.strength}`);
      chip.type = 'button';
      chip.title = 'Select the source image';
      chip.onclick = async () => { const src = await sourceOf(rec); if (src) setFocus(src, findList(src)); };
      capMain.append(chip);
    }
    capPrompt.textContent = rec.prompt;
    capPrompt.title = rec.prompt;
    actEls.star.querySelector('.lbl').textContent = rec.favorite ? '★ Starred' : '☆ Star';
    actEls.star.setAttribute('aria-pressed', String(!!rec.favorite));
    actEls.reuse.querySelector('.lbl').textContent = rec.refine != null ? '↻ Reuse pass…' : '↻ Reuse';
    actEls.refine.setAttribute('aria-disabled', String(!refineSupported));
    actEls.refine.title = refineSupported ? 'Refine: scale up and re-denoise part of the way' : 'This checkpoint has no refine pass';
    actEls.compare.setAttribute('aria-pressed', String(mode !== 'single'));
    actEls.pin.setAttribute('aria-pressed', String(!!pinned && pinned.id === rec.id));
    actEls.pin.querySelector('.lbl').textContent = pinned && pinned.id !== rec.id ? `A = ${pinned.id}` : 'Pin A';
    actEls.reference.hidden = !referenceSupported;
    actEls.run.setAttribute('aria-disabled', String(!rec.run_id && !rec.job_id));
    renderHud();
  }

  function renderHud() {
    hud.replaceChildren();
    if (!focusRec) { hud.hidden = true; return; }
    hud.hidden = false;
    const modeBtn = (m, label) => {
      const b = h('button', '', label);
      b.type = 'button';
      b.setAttribute('aria-pressed', String(mode === m));
      b.onclick = async () => {
        if (m === 'single') { mode = 'single'; pairRec = null; }
        else { const prev = mode; mode = m; if (!pairRec && !(await choosePair())) { mode = prev; flash('Nothing to compare with: press A on another image to pin it, or pick a refined image.'); } }
        renderBar(); layout();
      };
      return b;
    };
    hud.append(modeBtn('single', 'Single'), modeBtn('swipe', 'Swipe'), modeBtn('side', 'Side by side'));
    const z = h('span', 'zoom');
    z.id = 'loop-zoom';
    const fit = h('button', '', 'Fit');
    fit.type = 'button';
    fit.onclick = () => { view.scale = 1; view.x = view.y = 0; layout(); };
    const one = h('button', '', '1:1');
    one.type = 'button';
    one.onclick = () => { zoomTo(oneScale()); };
    hud.append(h('span', 'hint-k', '·'), fit, one, z);
    const keysHint = h('span', 'hint-k keys');
    keysHint.innerHTML = (mode !== 'single' ? '<kbd>B</kbd> hold to flip &nbsp;' : '')
      + '<kbd>←</kbd><kbd>→</kbd> walk &nbsp;<kbd>?</kbd> keys';
    hud.append(keysHint);
  }

  // Geometry. Fit is capped at one image pixel per device pixel, so a 1 MP
  // image on QHD is shown at 1:1 without opening anything (M11).
  let base = { w: 0, h: 0, k: 1 };
  const oneScale = () => (1 / window.devicePixelRatio) / base.k;
  function layout() {
    if (!wide()) return;
    const rec = focusRec;
    empty.hidden = !!rec;
    frame.hidden = !rec;
    if (!rec) { crop.hidden = true; return; }
    const pad = 8;
    const hudRoom = 34;
    const aw = focusEl.clientWidth - 2 * pad;
    const ah = focusEl.clientHeight - 2 * pad - hudRoom;
    const side = mode === 'side' && pairRec;
    const gap = 14;
    const cellW = side ? (aw - gap) / 2 : aw;
    const k = Math.min(cellW / rec.width, ah / rec.height, 1 / window.devicePixelRatio);
    const w = rec.width * k;
    const hh = rec.height * k;
    base = { w, h: hh, k };
    const fw = side ? 2 * w + gap : w;
    Object.assign(frame.style, {
      width: `${fw}px`, height: `${hh}px`,
      left: `${pad + (aw - fw) / 2}px`, top: `${pad + (ah - hh) / 2}px`,
    });
    frame.classList.toggle('comparing', mode !== 'single' && !!pairRec);
    frame.classList.toggle('side', !!side);
    frame.classList.toggle('zoomed', view.scale > 1);
    Object.assign(layerB.style, { left: '0px', width: `${w}px`, height: `${hh}px` });
    Object.assign(layerA.style, { left: side ? `${w + gap}px` : '0px', width: `${w}px`, height: `${hh}px` });
    const iw = w * view.scale;
    const ih = hh * view.scale;
    const mx = Math.max(0, (iw - w) / 2);
    const my = Math.max(0, (ih - hh) / 2);
    view.x = Math.min(mx, Math.max(-mx, view.x));
    view.y = Math.min(my, Math.max(-my, view.y));
    for (const img of [imgA, imgB]) {
      Object.assign(img.style, {
        width: `${iw}px`, height: `${ih}px`,
        left: `${(w - iw) / 2 + view.x}px`, top: `${(hh - ih) / 2 + view.y}px`,
      });
    }
    layerA.style.clipPath = mode === 'swipe' && pairRec ? `inset(0 0 0 ${swipeAt * 100}%)` : 'none';
    swipe.style.left = `${swipeAt * w}px`;
    if (pairRec) {
      const pr = pairRec;
      tagL.innerHTML = '';
      tagL.append(h('b', '', pairKind === 'source' ? 'Source' : 'A'), document.createTextNode(`image ${pr.id} · ${pr.width}×${pr.height}`));
      tagR.innerHTML = '';
      tagR.append(h('b', '', pairKind === 'source' ? 'Refined' : 'B'), document.createTextNode(`image ${rec.id} · ${rec.width}×${rec.height}${rec.strength != null ? ` · strength ${rec.strength}` : ''}`));
      tagL.style.left = '10px';
      tagR.style.left = 'auto';
      tagR.style.right = '10px';
    }
    const z = document.getElementById('loop-zoom');
    if (z) {
      const pct = Math.round(k * view.scale * window.devicePixelRatio * 100);
      z.textContent = view.scale <= 1 ? `fit ${pct}%` : `${pct}%`;
    }
    drawCrop();
  }

  function zoomTo(scale, cx = 0, cy = 0) {
    const next = Math.min(Math.max(1, oneScale() * 8), Math.max(1, scale));
    const ratio = next / view.scale;
    view.x = cx - ratio * (cx - view.x);
    view.y = cy - ratio * (cy - view.y);
    view.scale = next;
    if (next <= 1) view.x = view.y = 0;
    layout();
  }

  // Pointer: hover moves the swipe line; the wheel zooms both layers about the
  // pointer; a drag pans them together; a click at fit opens the lightbox.
  let drag = null;
  const layerPoint = (e) => {
    const target = mode === 'side' && pairRec && e.clientX > layerA.getBoundingClientRect().left ? layerA : layerB;
    const box = target.getBoundingClientRect();
    return [e.clientX - box.left - box.width / 2, e.clientY - box.top - box.height / 2];
  };
  frame.addEventListener('wheel', (e) => {
    e.preventDefault();
    zoomTo(view.scale * Math.exp(-e.deltaY * 0.002), ...layerPoint(e));
  }, { passive: false });
  frame.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    drag = { x: e.clientX, y: e.clientY, moved: false };
    frame.setPointerCapture(e.pointerId);
  });
  frame.addEventListener('pointermove', (e) => {
    if (drag && view.scale > 1) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (!drag.moved && Math.hypot(dx, dy) < 4) return;
      drag.moved = true;
      view.x += dx; view.y += dy; drag.x = e.clientX; drag.y = e.clientY;
      layout();
      return;
    }
    if (mode === 'swipe' && pairRec) {
      const box = layerB.getBoundingClientRect();
      swipeAt = Math.min(1, Math.max(0, (e.clientX - box.left) / box.width));
      if (drag) drag.moved = true;
      layout();
    }
  });
  frame.addEventListener('pointerup', () => {
    const d = drag; drag = null;
    if (!d || d.moved) return;
    if (mode === 'single' && view.scale <= 1 && focusRec) openLb(focusRec, focusList);
  });
  window.addEventListener('resize', layout);
  new ResizeObserver(() => layout()).observe(focusEl);

  // The kept region of a pass over the source (M51), while the drawer is on it.
  function drawCrop() {
    const src = refineSource;
    if (!wide() || !src || !focusRec || focusRec.id !== src.id || mode !== 'single' || !els.refineDialog.open) { crop.hidden = true; return; }
    const W = +els.refineWidth.value, H = +els.refineHeight.value;
    if (!(W > 0 && H > 0)) { crop.hidden = true; return; }
    const s = Math.max(W / src.width, H / src.height);
    const fw = (W / s) / src.width, fh = (H / s) / src.height;
    const bw = base.w, bh = base.h;
    Object.assign(crop.style, {
      left: `${(1 - fw) / 2 * bw}px`, top: `${(1 - fh) / 2 * bh}px`, width: `${fw * bw}px`, height: `${fh * bh}px`,
    });
    const kept = fw > 0.995 && fh > 0.995 ? 'whole source kept' : `keeps ${Math.round(fw * 100)}% × ${Math.round(fh * 100)}% of the source`;
    cropLabel.textContent = `→ ${W}×${H} · ${(W * H / 1e6).toFixed(1)} MP · ${kept}`;
    crop.hidden = false;
  }

  // -------------------------------------------- wrapping the page's functions
  const findList = (rec) => (galleryRecords.includes(rec) ? galleryRecords : viewerRecords.includes(rec) ? viewerRecords : [rec]);
  const _openLightbox = openLightbox;
  const openLb = (rec, list) => _openLightbox(rec, list);
  // A thumbnail's click selects it into the stage; a double-click, or Enter
  // on the one already selected, opens the lightbox. The sheet keeps its
  // lightbox on a click.
  openLightbox = function (rec, list) {
    if (!wide() || sheetMode) return _openLightbox(rec, list);
    const viaKey = window.event && window.event.type === 'keydown';
    if (viaKey && focusRec && focusRec.id === rec.id) return _openLightbox(rec, list);
    setFocus(rec, list);
  };
  for (const host of [els.gallery, els.viewer]) {
    host.addEventListener('dblclick', (e) => {
      if (!wide() || sheetMode) return;
      const img = e.target.closest('img[data-id]');
      if (!img) return;
      const list = host === els.gallery ? galleryRecords : viewerRecords;
      const rec = list.find((r) => r.id === +img.dataset.id);
      if (rec) _openLightbox(rec, list);
    });
  }

  const _appendResult = appendResult;
  let jobStart = false;
  appendResult = function (rec) {
    _appendResult(rec);
    const tile = els.viewer.lastElementChild;
    if (tile && jobStart) { tile.classList.add('job-start'); jobStart = false; }
    if (!wide()) return;
    if (tile) tile.scrollIntoView({ block: 'nearest', inline: 'end' });
    setFocus(rec, viewerRecords, { compare: rec.refine != null });
    if (rec.refine != null && refineSource && refineSource.id === rec.refine && typeof loadPasses === 'function') void loadPasses(refineSource);
  };
  const _clearResults = clearResults;
  clearResults = function () {
    if (!wide()) return _clearResults();
    // The session strip keeps earlier jobs (M67); a job starts a new group.
    els.viewer.querySelector('.placeholder')?.remove();
    els.viewerNote.textContent = '';
    jobStart = true;
  };
  const ph = els.viewer.querySelector('.placeholder');
  if (ph) ph.textContent = 'Images made in this session collect here.';

  // The live preview shows in the stage, over the last image, not in the strip.
  new MutationObserver((list) => {
    for (const m of list) for (const node of m.addedNodes) {
      if (node.nodeType === 1 && node.matches('img.preview') && wide()) {
        node.classList.add('focus-preview');
        focusEl.append(node);
      }
    }
  }).observe(els.viewer, { childList: true });

  const _setFavorite = setFavorite;
  setFavorite = async function (rec, value) { await _setFavorite(rec, value); renderBar(); };
  const _forgetImages = forgetImages;
  forgetImages = function (ids) {
    const at = focusList ? focusList.indexOf(focusRec) : -1;
    const list = focusList;
    _forgetImages(ids);
    if (focusRec && ids.includes(focusRec.id)) {
      const next = list && list.length ? list[Math.min(Math.max(at, 0), list.length - 1)] : null;
      setFocus(next || null, list);
    }
  };

  // Gallery cards: refined marker, a caption on hover, selection ring.
  const decorate = (cardEl) => {
    const rec = galleryRecords.find((r) => r.id === +cardEl.dataset.id);
    if (!rec || cardEl.querySelector('.cap')) return;
    if (rec.refine != null) { cardEl.classList.add('refined'); cardEl.dataset.strength = rec.strength ?? ''; }
    const c = h('span', 'cap', `${rec.id} · ${rec.width}×${rec.height} · seed ${rec.seed}`);
    cardEl.append(c);
    if (focusRec && focusRec.id === rec.id) cardEl.classList.add('selected');
  };
  els.gallery.querySelectorAll('.card').forEach(decorate);
  new MutationObserver((list) => {
    for (const m of list) for (const node of m.addedNodes) if (node.nodeType === 1 && node.matches('.card')) decorate(node);
    if (!focusRec && galleryRecords.length && wide()) setFocus(galleryRecords[0], galleryRecords);
  }).observe(els.gallery, { childList: true });
  if (!focusRec && galleryRecords.length) setFocus(galleryRecords[0], galleryRecords);

  // Sheet mode spans the stage and the shelf.
  const _setSheet = setSheet;
  setSheet = function (on) { document.body.classList.toggle('sheet-on', on); return _setSheet(on); };
  document.body.classList.toggle('sheet-on', sheetMode);

  // Chips under the prompt (the rail scrolls and Generate is pinned, so they
  // no longer push it away); the check covers the shelf.
  const _place = placeTagValidation;
  placeTagValidation = function () {
    if (!wide()) return _place();
    const focused = document.activeElement;
    const range = typeof focused?.selectionStart === 'number' ? [focused.selectionStart, focused.selectionEnd, focused.selectionDirection] : null;
    for (const [node, host] of [[els.tagValidation, els.tagValidationHome], [els.promptAnalysis, els.tagValidationWide]]) {
      if (node.parentElement === host) continue;
      if (typeof host.moveBefore === 'function') host.moveBefore(node, null); else host.appendChild(node);
    }
    if (!focused || focused === document.activeElement || !focused.isConnected) return;
    focused.focus();
    if (range) focused.setSelectionRange(...range);
  };
  window.addEventListener('resize', () => placeTagValidation());
  placeTagValidation();

  // ------------------------------------------------- Refine as a drawer
  els.refineDialog.showModal = function () {
    if (shell()) this.show(); else HTMLDialogElement.prototype.showModal.call(this);
  };
  // Strength presets from the checkpoint's own default (M29).
  const presets = h('div', 'rf-strengths');
  els.refineStrength.closest('.row').after(presets);
  // Every pass already made from this source, newest first: a click compares
  // it against the source in the stage (M49) - the loop's memory.
  const passes = h('section', 'rf-passes');
  els.refinePreview.before(passes);
  async function loadPasses(src) {
    if (!src) return;
    const params = new URLSearchParams({ search: `refine:${src.id}`, all_models: 'true', limit: 24 });
    let page;
    try { page = await api(`/api/gallery?${params}`); } catch { return; }
    if (!refineSource || refineSource.id !== src.id) return;
    passes.replaceChildren(h('h3', '', `Passes of image ${src.id} (${page.total})`));
    const row = h('div', 'rf-pass-row');
    if (!page.items.length) row.append(h('p', 'hint', 'None yet. Each pass lands here and in the stage, swiped against this image.'));
    for (const rec of page.items) {
      const b = h('button', 'rf-pass');
      b.type = 'button';
      const img = h('img');
      img.src = rec.thumb_url ?? rec.url;
      img.alt = '';
      b.append(img, h('span', '', `${rec.strength} · ${rec.width}×${rec.height}${rec.favorite ? ' ★' : ''}`));
      b.title = `Image ${rec.id}: compare it with image ${src.id}`;
      b.onclick = () => setFocus(rec, page.items, { compare: true });
      row.append(b);
    }
    passes.append(row);
  }
  function buildPresets() {
    const d = refineDefaultStrength();
    const r = (x) => +x.toFixed(3);
    const opts = [[`${r(d / 2)}`, 'lighter'], [`${r(d)}`, 'default'], [`${r(d * 2)}`, 'stronger'],
      [`${r(d / 2)},${r(d)},${r(d * 1.5)},${r(d * 2)}`, 'sweep']];
    presets.replaceChildren();
    for (const [value, label] of opts) {
      const b = h('button', 'ghost', label === 'sweep' ? `sweep ${value}` : `${value} ${label}`);
      b.type = 'button';
      b.classList.toggle('on', els.refineStrength.value.trim() === value);
      b.onclick = () => { els.refineStrength.value = value; refineChanged(); buildPresets(); };
      presets.append(b);
    }
  }
  const _openRefine = openRefine;
  openRefine = function (rec, pass = null) {
    const ok = _openRefine(rec, pass);
    if (!ok || !wide()) return ok;
    buildPresets();
    passes.replaceChildren();
    void loadPasses(rec);
    if (!focusRec || focusRec.id !== rec.id) setFocus(rec, findList(rec)); else { mode = 'single'; renderBar(); layout(); }
    els.refineStrength.focus();
    els.refineStrength.select();
    drawCrop();
    return ok;
  };
  const _syncRefine = syncRefine;
  syncRefine = function () { _syncRefine(); drawCrop(); };
  els.refineStrength.addEventListener('input', buildPresets);

  // Refine keeps the drawer open on its source with what was typed (M28),
  // so the next strength is one edit away; the result lands in the stage as a
  // compare against the source.
  const FIELDS = ['refineWidth', 'refineHeight', 'refineStrength', 'refineVersions', 'refineSeed',
    'refinePrompt', 'refineNegative', 'refineSteps', 'refineCfg', 'refineMu', 'refineTrueCfg'];
  let keepOpen = null;
  const _runRefine = runRefine;
  runRefine = function () {
    if (!shell() || !els.refineDialog.open) return _runRefine();
    keepOpen = { src: refineSource, draft: FIELDS.map((k) => els[k].value), more: els.refineMore.open };
    const p = _runRefine();
    if (els.refineDialog.open) keepOpen = null;  // refused in place
    return p;
  };
  els.refineDialog.addEventListener('close', () => {
    const k = keepOpen;
    keepOpen = null;
    if (!k) { crop.hidden = true; return; }
    setTimeout(() => {
      _openRefine(k.src);
      FIELDS.forEach((f, i) => { els[f].value = k.draft[i]; });
      els.refineMore.open = k.more;
      refineChanged();
      buildPresets();
      void loadPasses(k.src);
    }, 0);
  });

  // Ctrl+Enter by where focus is: the drawer runs its pass, the grid fold
  // runs the grid (M7), anywhere else generates. Ahead of app.js's handler,
  // which would otherwise run the drawer's pass from the prompt box.
  window.addEventListener('keydown', (e) => {
    if (!(e.key === 'Enter' && (e.ctrlKey || e.metaKey))) return;
    if (els.refineDialog.open && els.refineDialog.matches(':modal')) return;  // app.js handles it
    e.preventDefault();
    e.stopPropagation();
    if (els.refineDialog.open && els.refineDialog.contains(e.target)) { void runRefine(); return; }
    if (els.gridSection.open && els.gridSection.contains(e.target)) { els.gridRun.click(); return; }
    generate();
  }, true);

  // Tab title follows the job (M24).
  new MutationObserver(() => {
    const t = els.progressText.textContent.trim();
    document.title = busy && t ? `${t} · Local Diffusion` : 'Local Diffusion';
  }).observe(els.progressText, { childList: true, characterData: true, subtree: true });

  // ------------------------------------------------------------ the keys
  const typing = (t) => t.closest && t.closest('input, textarea, select, [contenteditable="true"], .tag-validation, .menu, .tag-menu');
  const cols = () => getComputedStyle(els.gallery).gridTemplateColumns.split(' ').length;
  async function step(delta) {
    if (!focusRec || !focusList) return;
    let at = focusList.indexOf(focusRec) + delta;
    if (at >= focusList.length && focusList === galleryRecords && galleryMore) { await loadGallery(); }
    at = Math.max(0, Math.min(focusList.length - 1, at));
    const next = focusList[at];
    if (!next || next === focusRec) return;
    await setFocus(next, focusList, { keep: mode !== 'single' });
    const el = focusList === galleryRecords
      ? els.gallery.querySelector(`.card[data-id="${next.id}"] img`)
      : els.viewer.querySelector(`img[data-id="${next.id}"]`);
    if (el) { el.focus({ preventScroll: true }); el.scrollIntoView({ block: 'nearest', inline: 'nearest' }); }
  }
  document.addEventListener('keydown', (e) => {
    if (!wide() || els.lightbox.open || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === 'Escape') {
      if (!keys.hidden) { keys.hidden = true; return; }
      if (els.refineDialog.open && !els.refineDialog.matches(':modal')) {
        // Out of the drawer first, then shut it.
        if (els.refineDialog.contains(document.activeElement)) { focusEl.focus(); return; }
        els.refineDialog.close();
        return;
      }
      return;
    }
    if (typing(e.target)) return;
    if (e.target.closest && e.target.closest('#refine-dialog, .prompt-analysis, dialog')) return;
    const k = e.key;
    if (k === '?') { keys.hidden = !keys.hidden; e.preventDefault(); return; }
    if (k === '/') { e.preventDefault(); els.search.focus(); els.search.select(); return; }
    if (k === '[') { e.preventDefault(); toggleRail(); return; }
    if (k === 'ArrowLeft' || k === 'ArrowRight') { e.preventDefault(); step(k === 'ArrowLeft' ? -1 : 1); return; }
    if (k === 'ArrowUp' || k === 'ArrowDown') {
      if (focusList !== galleryRecords) return;
      e.preventDefault(); step((k === 'ArrowUp' ? -1 : 1) * cols()); return;
    }
    if (!focusRec) return;
    const lower = k.length === 1 ? k.toLowerCase() : k;
    if (lower === 'b' && !e.repeat && mode !== 'single') { frame.classList.add('blink'); return; }
    const map = { f: 'star', r: 'reuse', e: 'refine', c: 'compare', a: 'pin', s: 'run', Delete: 'delete' };
    if (map[lower]) { e.preventDefault(); actEls[map[lower]].click(); return; }
    if (k === 'Enter' && (e.target === document.body || e.target === focusEl)) { e.preventDefault(); actEls.zoom.click(); }
  });
  document.addEventListener('keyup', (e) => { if (e.key.toLowerCase() === 'b') frame.classList.remove('blink'); });

  renderBar();
  layout();
  // Re-sync the bar when the status flips a capability (Refine, references).
  setInterval(() => {
    if (!focusRec) return;
    actEls.refine.setAttribute('aria-disabled', String(!refineSupported));
    actEls.reference.hidden = !referenceSupported;
  }, 3000);
  window.__loopApi = { setFocus: (r, l) => setFocus(r, l), toggleCompare, pinA, layout, get focus() { return focusRec; }, get mode() { return mode; } };
})();
