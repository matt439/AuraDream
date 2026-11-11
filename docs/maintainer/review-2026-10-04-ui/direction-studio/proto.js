// Studio workbench prototype: DOM moves and a selection model over the real
// page. Everything the page's scripts reach through `els` keeps its id, so
// moving a node changes nothing they do. Run once, after the page has booted.
(() => {
  if (document.body.classList.contains('studio')) return;
  document.body.classList.add('studio');
  const main = document.querySelector('main');
  const panel = document.querySelector('.panel');
  const stage = document.querySelector('.stage');
  const make = (tag, props = {}, ...kids) => {
    const el = Object.assign(document.createElement(tag), props);
    el.append(...kids);
    return el;
  };
  const short = (id) => (id || '').split('/').pop();

  // ------------------------------------------------ 1. the control column
  const scroll = make('div', { className: 'studio-scroll' });
  while (panel.firstChild) scroll.appendChild(panel.firstChild);
  const kbd = make('kbd', { textContent: 'Ctrl+Enter' });
  const foot = make('div', { className: 'studio-foot' }, els.generate, els.gridRun, kbd);
  els.generate.setAttribute('aria-keyshortcuts', 'Control+Enter');
  panel.append(scroll, foot);
  // Group headings (M84): the column reads as Prompt / Frame / Sampling /
  // Conditioning / Output / Grid instead of one flat list.
  const group = (text, before) => before && before.before(make('h3', { className: 'studio-group', textContent: text }));
  group('Frame', els.presets.closest('.field'));
  group('Sampling', els.steps.closest('.row'));
  group('Conditioning', els.referenceField);
  group('Output', els.seed.closest('.row'));

  // ------------------------------------------------ 2. the stage
  const title = make('div', { id: 'studio-title' });
  const back = make('button', { id: 'studio-back', className: 'ghost small', textContent: '← Latest results', hidden: true });
  const status = make('div', { id: 'studio-status', role: 'group', ariaLabel: 'Canvas and run status' },
    title, els.progress, els.queueNote, els.cellNote, back, els.error, els.lastRun);
  stage.prepend(status);
  const focus = make('div', { id: 'studio-focus' });
  els.viewer.after(focus);

  // ------------------------------------------------ 3. the rail
  const insBody = make('div', { id: 'studio-insp-body' });
  const inspector = make('section', { id: 'studio-inspector', ariaLabel: 'Selected image' }, insBody, els.refineDialog);
  const expand = make('button', { id: 'studio-expand', className: 'ghost small', textContent: '⇱ Wide',
    title: 'Give the library the canvas column (the inspector stays)' });
  const head = document.querySelector('.gallery-head');
  head.append(expand);
  const library = make('section', { id: 'studio-library', ariaLabel: 'Library' }, head, els.gallery, els.more);
  const rail = make('aside', { id: 'studio-rail' }, inspector, library);
  main.append(rail);
  expand.onclick = () => {
    const on = main.classList.toggle('library');
    expand.textContent = on ? '⇲ Canvas' : '⇱ Wide';
  };

  // ------------------------------------------------ 4. selection
  // `mode`: 'latest' - the viewer, this session's newest results;
  // 'focus' - one record (and its source, for a pass) on the canvas;
  // 'refine' - the Refine dialog's source, with the crop it will keep.
  const sel = { rec: null, list: null, mode: 'latest', source: null };
  window.studioSel = sel;

  function setTitle(strong, rest) {
    title.replaceChildren(make('strong', { textContent: strong }), make('span', { textContent: rest || '' }));
  }

  function markSelected() {
    document.querySelectorAll('.studio-sel').forEach((el) => el.classList.remove('studio-sel'));
    if (!sel.rec) return;
    document.querySelectorAll(`.card[data-id="${sel.rec.id}"]`).forEach((el) => el.classList.add('studio-sel'));
    document.querySelectorAll(`.sheet-cell[data-id="${sel.rec.id}"]`).forEach((el) => el.classList.add('studio-sel'));
    if (sel.mode === 'latest' && viewerRecords.length > 1) {
      els.viewer.querySelectorAll(`img[data-id="${sel.rec.id}"]`).forEach((el) => el.classList.add('studio-sel'));
    }
  }

  function figure(rec, caption, list) {
    const img = make('img', { src: rec.url, alt: rec.prompt, draggable: false });
    img.width = rec.width; img.height = rec.height;
    img.title = 'Open full size (zoom, 1:1)';
    img.onclick = () => openLightbox(rec, list);
    const box = make('div', { className: 'sf-box' }, img);
    const cap = make('figcaption', { className: 'sf-cap' });
    cap.innerHTML = caption;
    return make('figure', { className: 'sf-fig' }, box, cap);
  }

  async function renderFocus() {
    const rec = sel.rec;
    focus.replaceChildren();
    if (!rec) return;
    const dims = (r) => `${r.width}×${r.height}`;
    if (sel.mode === 'refine') {
      focus.style.setProperty('--n', 1);
      focus.append(figure(rec, `<b>Source · image ${rec.id}</b> · ${dims(rec)} · <span id="sf-crop-note"></span>`, [rec]));
      cropOverlay();
      return;
    }
    if (rec.refine != null) {
      // A pass beside its source, at the same displayed size (M5).
      let source = sel.source && sel.source.id === rec.refine ? sel.source : null;
      if (!source) {
        try { source = await api(`/api/images/${rec.refine}`); } catch { source = null; }
        if (sel.rec !== rec) return;
        sel.source = source;
      }
      focus.style.setProperty('--n', source ? 2 : 1);
      const pair = source ? [source, rec] : [rec];
      if (source) focus.append(figure(source, `<b>Source · image ${source.id}</b> · ${dims(source)}`, pair));
      focus.append(figure(rec, `<b>Refine pass · image ${rec.id}</b> · ${dims(rec)} · strength ${rec.strength}`, pair));
      return;
    }
    focus.style.setProperty('--n', 1);
    focus.append(figure(rec, `<b>Image ${rec.id}</b> · ${dims(rec)} · seed ${rec.seed}`, sel.list || [rec]));
  }

  function fact(dt, dd, wide = false) {
    return [make('dt', { textContent: dt }), make('dd', { textContent: dd, className: wide ? 'wide' : '' })];
  }

  function act(label, title, onClick, cls = 'ghost') {
    const b = make('button', { className: cls, textContent: label, title });
    b.type = 'button';
    b.onclick = onClick;
    return b;
  }

  function renderInspector() {
    const rec = sel.rec;
    if (!rec) {
      insBody.replaceChildren(make('p', { className: 'si-empty',
        textContent: 'Select an image in the library, or generate one: its settings and actions appear here.' }));
      return;
    }
    const made = new Date(rec.created_at);
    const star = act(rec.favorite ? '★' : '☆', rec.favorite ? 'Unstar (F)' : 'Star (F)', async () => {
      await setFavorite(rec, !rec.favorite);
      renderInspector();
    }, `ghost si-star${rec.favorite ? ' on' : ''}`);
    star.setAttribute('aria-pressed', String(!!rec.favorite));
    const headEl = make('div', { className: 'si-head' },
      make('h2', { textContent: `Image ${rec.id}` }),
      make('span', { className: 'si-when', textContent: made.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) }),
      star);
    const parts = [headEl];
    if (rec.refine != null) {
      const chip = make('div', { className: 'si-chip' });
      const src = make('button', { textContent: `image ${rec.refine}`, title: 'Select the source' });
      src.onclick = async () => {
        try { const s = await api(`/api/images/${rec.refine}`); select(s, [s], 'focus'); } catch (err) { showError(err.message); }
      };
      chip.append('⤢ Refine pass of ', src, ` at strength ${rec.strength}`);
      parts.push(chip);
    }
    parts.push(make('p', { className: 'si-prompt', textContent: rec.prompt }));
    if (rec.negative_prompt) parts.push(make('p', { className: 'si-neg', textContent: `Negative: ${rec.negative_prompt}` }));
    const facts = make('dl', { className: 'si-facts' });
    facts.append(
      ...fact('Model', `${short(rec.model_id)}${rec.quantization ? ` · ${rec.quantization}` : ''}`, true),
      ...fact('Size', `${rec.width}×${rec.height}`), ...fact('Seed', String(rec.seed)),
      ...fact('Steps', String(rec.steps)), ...fact('CFG', String(rec.guidance_scale)),
      ...fact('True CFG', rec.true_cfg == null ? 'off' : String(rec.true_cfg)), ...fact('Mu', rec.mu == null ? 'auto' : String(rec.mu)),
      ...fact('LoRA', rec.lora ? `${rec.lora} @ ${rec.lora_scale}` : 'none'), ...fact('Time', `${(rec.duration_ms / 1000).toFixed(1)} s`),
    );
    parts.push(facts);
    const actions = make('div', { className: 'si-actions' },
      act('⤢ Refine…', 'Scale this image up and re-denoise it part of the way, here in the inspector', () => openRefine(rec), 'primary'),
      act('Open full size', 'Zoom and 1:1 in the lightbox', () => openLightbox(rec, sel.list || [rec])),
      act('↻ Reuse settings', 'Load prompt, size, sampling and seed into the controls', () => reuse(rec)),
      act('Use seed', `Put seed ${rec.seed} in the Seed box`, () => {
        els.seed.value = rec.seed; syncRuns(); schedulePreview(); persistPanel();
        showError(`Seed ${rec.seed} is in the Seed box.`, true);
      }),
      act(`Re-render at ${els.width.value}×${els.height.value}`, 'Same prompt, settings and seed at the size set in the controls (Frame)', () => {
        void startJob('/api/rerender', { search: `hash:${rec.content_hash}`, favorites: false, all_models: false,
          width: +els.width.value, height: +els.height.value });
      }),
      act('⊞ Run as sheet', 'Lay this image\'s run out as a contact sheet on the canvas', () => {
        if (!rec.run_id && !rec.job_id) { showError('That image belongs to no run.', true); return; }
        els.search.value = rec.run_id ? `run:${rec.run_id}` : `job:${rec.job_id}`;
        setSheet(true);
        back.hidden = false;
      }),
      act('⌦ Delete', 'Delete this image and its file', async () => {
        await deleteImage(rec);
        if (!recordsWithId(rec.id).length) { sel.rec = null; renderInspector(); renderFocus(); }
      }, 'ghost danger'),
    );
    actions.children[4].id = 'studio-rerender';
    parts.push(actions);
    insBody.replaceChildren(...parts);
  }

  function select(rec, list, mode) {
    sel.rec = rec; sel.list = list; sel.mode = mode;
    if (sel.source && rec && sel.source.id !== rec.refine) sel.source = null;
    document.body.classList.toggle('studio-focus-mode', mode !== 'latest');
    back.hidden = mode === 'latest' && !sheetMode;
    if (mode === 'latest') {
      setTitle('Latest result', viewerRecords.length > 1 ? `${viewerRecords.length} images · click one to inspect it, again to zoom` : (rec ? rec.prompt : ''));
    } else if (mode === 'refine') {
      setTitle(`Refining image ${rec.id}`, 'the dashed frame is what the pass keeps');
    } else {
      setTitle(`Image ${rec.id}`, rec.refine != null ? 'refine pass beside its source · click either to zoom' : 'from the library · click to zoom');
    }
    renderInspector();
    renderFocus();
    markSelected();
  }
  window.studioSelect = select;

  back.onclick = () => {
    if (sheetMode) { setSheet(false); els.search.value = ''; resetGallery(); }
    if (els.refineDialog.open) els.refineDialog.close();
    select(viewerRecords.at(-1) ?? null, viewerRecords, 'latest');
  };

  // A click on a card selects it; a second click, or Enter, opens the lightbox.
  els.gallery.addEventListener('click', (e) => {
    const img = e.target.closest('.card img');
    if (!img) return;
    e.stopImmediatePropagation();
    e.preventDefault();
    const rec = galleryRecords.find((r) => r.id === +img.dataset.id);
    if (!rec) return;
    if (sel.rec && sel.rec.id === rec.id && sel.mode === 'focus') { openLightbox(rec, galleryRecords); return; }
    // Refine is bound to the selection: picking another image re-points it.
    if (els.refineDialog.open) { openRefine(rec); return; }
    select(rec, galleryRecords, 'focus');
  }, true);
  // Keyboard: focusing a card shows it in the inspector without moving focus.
  els.gallery.addEventListener('focusin', (e) => {
    const img = e.target.closest('.card img');
    if (!img || els.refineDialog.open) return;
    const rec = galleryRecords.find((r) => r.id === +img.dataset.id);
    if (rec && sel.rec !== rec) select(rec, galleryRecords, 'focus');
  });
  // In a batch on the canvas, a click inspects that image; a second zooms.
  els.viewer.addEventListener('click', (e) => {
    const img = e.target.closest('img');
    if (!img || img.classList.contains('preview') || viewerRecords.length < 2) return;
    const rec = viewerRecords.find((r) => r.id === +img.dataset.id);
    if (!rec || sel.rec === rec) return;
    e.stopImmediatePropagation();
    sel.rec = rec;
    renderInspector();
    markSelected();
  }, true);
  // On the sheet a click inspects the cell (star, Reuse, Refine without the
  // lightbox round trip, M48); a second click on it zooms.
  els.sheetGrid.addEventListener('click', (e) => {
    const img = e.target.closest('.sheet-cell img');
    if (!img) return;
    const rec = sheetRecords.find((r) => r.id === +img.closest('.sheet-cell').dataset.id) ?? galleryRecords.find((r) => r.id === +img.closest('.sheet-cell').dataset.id);
    if (!rec || sel.rec === rec) return;
    e.stopImmediatePropagation();
    sel.rec = rec; sel.list = sheetRecords;
    renderInspector();
    markSelected();
  }, true);
  // Pass badges on cards (M34).
  const passBadges = () => {
    els.gallery.querySelectorAll('.card').forEach((cardEl) => {
      const rec = galleryRecords.find((r) => r.id === +cardEl.dataset.id);
      if (rec && rec.refine != null && !cardEl.querySelector('.studio-pass')) {
        cardEl.append(make('span', { className: 'studio-pass', textContent: `⤢ ${rec.strength}` }));
      }
    });
    markSelected();
  };
  new MutationObserver(passBadges).observe(els.gallery, { childList: true });
  passBadges();

  // New results take the canvas and the inspector.
  const appendOriginal = window.appendResult;
  window.appendResult = function (rec) {
    appendOriginal(rec);
    // A result landing while Refine is open does not take its canvas away:
    // the strip offers it instead.
    if (els.refineDialog.open) { back.hidden = false; back.textContent = '← Latest results (new)'; return; }
    back.textContent = '← Latest results';
    if (rec.refine != null && viewerRecords.length === 1) select(rec, viewerRecords, 'focus');
    else select(rec, viewerRecords, 'latest');
  };

  // ------------------------------------------------ 5. Refine, docked
  // Non-modal: the controls and the library stay live, and the canvas shows
  // the source large. refine.js still opens, previews, runs and closes it.
  els.refineDialog.showModal = function () { this.show(); };
  let beforeRefine = null;
  new MutationObserver(() => {
    if (els.refineDialog.open && refineSource) {
      if (sel.mode !== 'refine') beforeRefine = { rec: sel.rec, list: sel.list, mode: sel.mode };
      select(refineSource, [refineSource], 'refine');
      els.refineStrength.focus();
      els.refineStrength.select();
    } else if (!els.refineDialog.open && sel.mode === 'refine') {
      const b = beforeRefine;
      beforeRefine = null;
      // A run puts its result in through appendResult; a cancel goes back.
      setTimeout(() => {
        if (sel.mode !== 'refine') return;
        if (b && b.rec) select(b.rec, b.list, b.mode); else select(viewerRecords.at(-1) ?? null, viewerRecords, 'latest');
      }, 50);
    }
  }).observe(els.refineDialog, { attributes: true, attributeFilter: ['open'] });
  // openRefine on another image while open: the observer sees no change.
  const openRefineOriginal = window.openRefine;
  window.openRefine = function (rec, pass) {
    const opened = openRefineOriginal(rec, pass);
    if (opened && els.refineDialog.open && sel.mode === 'refine' && sel.rec !== refineSource) {
      select(refineSource, [refineSource], 'refine');
    }
    return opened;
  };
  const syncRefineOriginal = window.syncRefine;
  window.syncRefine = function () {
    syncRefineOriginal();
    if (!refineSource) return;
    [...els.refineScales.children].forEach((button, i) => {
      const [w, h] = refineScaled(refineSource.width, refineSource.height, REFINE_SCALES[i]);
      button.innerHTML = `<b>${REFINE_SCALES[i]}×</b><small>${w}×${h}</small>`;
    });
    cropOverlay();
  };
  // The kept region, drawn over the source: refine.fitted's cover-and-crop.
  function cropOverlay() {
    if (sel.mode !== 'refine' || !refineSource) return;
    const img = focus.querySelector('.sf-box img');
    if (!img) return;
    const place = () => {
      const box = img.parentElement;
      box.querySelector('.sf-crop')?.remove();
      const sw = refineSource.width, sh = refineSource.height;
      const w = +els.refineWidth.value, h = +els.refineHeight.value;
      const note = document.getElementById('sf-crop-note');
      if (!(w > 0 && h > 0)) return;
      const s = Math.max(w / sw, h / sh);
      const fx = Math.min(1, w / s / sw), fy = Math.min(1, h / s / sh);
      if (note) {
        note.textContent = `→ ${w}×${h} (${(w * h / 1e6).toFixed(1)} MP)`
          + (fx < 0.995 || fy < 0.995 ? ` · keeps the middle ${Math.round(fx * 100)}% × ${Math.round(fy * 100)}%` : ' · the whole frame is kept');
      }
      if (fx >= 0.995 && fy >= 0.995) return;
      const r = img.getBoundingClientRect(), b = box.getBoundingClientRect();
      const crop = make('div', { className: 'sf-crop' });
      Object.assign(crop.style, { left: `${r.left - b.left}px`, top: `${r.top - b.top}px`, width: `${r.width}px`, height: `${r.height}px` });
      const frame = make('i');
      Object.assign(frame.style, {
        left: `${(1 - fx) / 2 * r.width}px`, top: `${(1 - fy) / 2 * r.height}px`,
        width: `${fx * r.width}px`, height: `${fy * r.height}px`,
      });
      crop.append(frame);
      box.append(crop);
    };
    if (img.complete && img.naturalWidth) place(); else img.addEventListener('load', place, { once: true });
  }
  window.addEventListener('resize', cropOverlay);
  // Escape closes the docked panel; Ctrl+Enter outside it is Generate again.
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && els.refineDialog.open && !els.lightbox.open
        && els.examplesMenu.hidden && els.tagsMenu.hidden) els.refineDialog.close();
  });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && els.refineDialog.open && !els.refineDialog.contains(e.target)) {
      e.stopImmediatePropagation();
      e.preventDefault();
      generate();
    }
  }, true);

  // The sheet takes the canvas; the strip says what it is laid out from.
  const setSheetOriginal = window.setSheet;
  window.setSheet = function (on) {
    const done = setSheetOriginal(on);
    if (on) { setTitle('Contact sheet', els.search.value || 'the whole gallery'); back.hidden = false; }
    else select(sel.rec, sel.list, sel.mode === 'refine' ? 'focus' : sel.mode);
    return done;
  };
  // The inspector's Re-render names the size the Frame group holds.
  const syncSizeOriginal = window.syncSize;
  window.syncSize = function (...args) {
    const out = syncSizeOriginal(...args);
    const b = document.getElementById('studio-rerender');
    if (b) b.textContent = `Re-render at ${els.width.value}×${els.height.value}`;
    return out;
  };

  // ------------------------------------------------ 6. odds and ends
  // Help tips inside a scrolling column would be clipped: lift them out.
  const lift = (help) => {
    const tip = help.querySelector('.tip');
    if (!tip || !help.closest('.studio-scroll, #studio-inspector, #studio-library')) return;
    const r = help.getBoundingClientRect();
    tip.classList.add('studio-fixed');
    const width = tip.offsetWidth || 300;
    tip.style.left = `${Math.max(8, Math.min(innerWidth - width - 8, r.left - 6))}px`;
    tip.style.top = `${r.bottom + 7}px`;
    tip.style.right = 'auto';
  };
  document.addEventListener('mouseover', (e) => { const h = e.target.closest?.('.help'); if (h) lift(h); });
  document.addEventListener('focusin', (e) => { const h = e.target.closest?.('.help'); if (h) lift(h); });
  // Reuse scrolls the controls, not the page, and only when the prompt is out of view (M27).
  const reuseIntoPanelOriginal = window.reuseIntoPanel;
  window.reuseIntoPanel = function (...args) {
    const y = window.scrollY;
    reuseIntoPanelOriginal(...args);
    window.scrollTo({ top: y, behavior: 'auto' });
    const r = els.prompt.getBoundingClientRect(), s = scroll.getBoundingClientRect();
    if (r.top < s.top || r.bottom > s.bottom) scroll.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // The canvas is never an empty box (M35): with nothing made this session,
  // it opens on the newest image in the library.
  const newest = () => {
    if (viewerRecords.length || sel.rec || !galleryRecords.length) return false;
    select(galleryRecords[0], galleryRecords, 'focus');
    setTitle('Newest in the library', `image ${galleryRecords[0].id} · Generate puts this session's results here`);
    back.hidden = true;
    return true;
  };
  if (viewerRecords.length) select(viewerRecords.at(-1), viewerRecords, 'latest');
  else if (!newest()) {
    select(null, viewerRecords, 'latest');
    const wait = new MutationObserver(() => { if (newest()) wait.disconnect(); });
    wait.observe(els.gallery, { childList: true });
  }
})();
