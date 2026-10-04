// "Same bones, real scale" - the few DOM moves the prototype needs, run once
// on the live page after its own scripts. Each block says what the real
// change in web/ would be.
(() => {
  const wide = () => matchMedia('(min-width: 1800px)').matches;

  // 1. Group headings in the panel (index.html: two <h3 class="group-head">).
  const stepsRow = els.steps.closest('.row');
  const seedRow = els.seed.closest('.row');
  const head = (text) => {
    const h = document.createElement('h3');
    h.className = 'group-head';
    h.textContent = text;
    return h;
  };
  stepsRow.before(head('Sampling'));
  seedRow.before(head('Output'));

  // 2. Progressive disclosure: references and the adapter fold into one
  //    "Conditioning" section whose summary says what it holds, opened while
  //    it holds anything (index.html: a <details> around #reference-field and
  //    #lora-field; sizes.js syncConditioning writes the note and the open).
  const fold = document.createElement('details');
  fold.className = 'section';
  fold.id = 'conditioning-section';
  fold.innerHTML = '<summary>Conditioning <em id="conditioning-note"></em></summary>';
  els.referenceField.before(fold);
  fold.append(els.loraField, els.referenceField);
  const note = fold.querySelector('#conditioning-note');
  const syncFold = () => {
    const lora = els.lora.disabled ? 'no adapters'
      : els.lora.value ? `LoRA ${els.lora.value} × ${els.loraScale.value || 1}` : 'no LoRA';
    const refs = !referenceSupported ? 'references n/a here'
      : referenceIds.length ? `${referenceIds.length} reference${referenceIds.length > 1 ? 's' : ''}` : 'no references';
    note.textContent = `${lora} · ${refs}`;
    if (els.lora.value || (referenceSupported && referenceIds.length)) fold.open = true;
  };
  for (const ev of ['input', 'change']) { els.lora.addEventListener(ev, syncFold); els.loraScale.addEventListener(ev, syncFold); }
  new MutationObserver(syncFold).observe(els.referenceList, { childList: true });
  new MutationObserver(syncFold).observe(els.referenceField, { attributes: true, attributeFilter: ['class'] });
  syncFold();

  // 3. The sticky panel's top from its measured height, not a 900 px guess
  //    (M1): pinned under the topbar while it fits, pinned by its bottom while
  //    it does not, so Generate and Run grid are always reachable.
  const panel = document.querySelector('.panel');
  const topbar = document.querySelector('.topbar');
  const placePanel = () => {
    const gap = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--gutter')) || 18;
    const top = topbar.offsetHeight + gap;
    const room = innerHeight - top - gap;
    panel.style.setProperty('--panel-top', `${panel.offsetHeight <= room ? top : innerHeight - panel.offsetHeight - gap}px`);
  };
  new ResizeObserver(placePanel).observe(panel);
  addEventListener('resize', placePanel);
  placePanel();

  // 4. A word for each card tool (gallery.js: tool() takes a short label and
  //    sets data-tip; the long title stays the aria-label).
  const TIPS = { '↻': 'Reuse settings', '⚠': 'Reuse settings (other model)', '⤢': 'Refine…', '❐': 'Use as reference', '⊞': 'Show run as sheet', '★': 'Unstar', '☆': 'Star', '⌦': 'Delete' };
  const tipCard = (div) => {
    for (const b of div.querySelectorAll('.tools button')) b.dataset.tip = TIPS[b.textContent.trim()] ?? b.title;
    return div;
  };
  const baseCard = card;
  card = (rec) => tipCard(baseCard(rec));
  document.querySelectorAll('.card').forEach(tipCard);
  // The star's glyph flips on toggle; keep its word in step.
  new MutationObserver(() => document.querySelectorAll('.card .tools .star').forEach((b) => { b.dataset.tip = TIPS[b.textContent.trim()]; }))
    .observe(els.gallery, { subtree: true, characterData: true, childList: true });

  // 5. Refine: the source shown large, from the full image rather than the
  //    384 px thumbnail, with the region the pass keeps framed (refine.js:
  //    openRefine sets src to rec.url; syncRefine calls the frame).
  const stage = document.createElement('div');
  stage.className = 'rf-stage';
  els.refineThumb.before(stage);
  stage.append(els.refineThumb);
  const crop = document.createElement('div');
  crop.className = 'rf-crop';
  const caption = document.createElement('div');
  caption.className = 'rf-stage-caption';
  stage.append(crop, caption);
  const frame = () => {
    const src = refineSource;
    if (!src || !wide()) { crop.hidden = true; caption.textContent = ''; return; }
    const w = +els.refineWidth.value; const h = +els.refineHeight.value;
    const box = stage.getBoundingClientRect();
    if (!(w > 0 && h > 0) || !box.width) { crop.hidden = true; return; }
    // The image as object-fit: contain draws it.
    const a = src.width / src.height;
    const [dw, dh] = box.width / box.height > a ? [box.height * a, box.height] : [box.width, box.width / a];
    const ox = (box.width - dw) / 2; const oy = (box.height - dh) / 2;
    // refine.fitted: scaled to cover, centre-cropped.
    const s = Math.max(w / src.width, h / src.height);
    const kw = Math.min(1, w / s / src.width); const kh = Math.min(1, h / s / src.height);
    crop.hidden = false;
    Object.assign(crop.style, {
      left: `${ox + dw * (1 - kw) / 2}px`, top: `${oy + dh * (1 - kh) / 2}px`,
      width: `${dw * kw}px`, height: `${dh * kh}px`,
    });
    const cut = kw < 0.999 || kh < 0.999;
    crop.dataset.label = cut ? `kept: ${Math.round(kw * kh * 100)}% of the source` : '';
    crop.style.outlineColor = cut ? 'rgba(251, 191, 36, 0.95)' : 'transparent';
    caption.textContent = `Source ${src.width}×${src.height}  →  ${w}×${h} (${(w * h / 1e6).toFixed(1)} MP)`;
  };
  const baseOpen = openRefine;
  openRefine = (rec, pass = null) => {
    const opened = baseOpen(rec, pass);
    if (opened && wide()) {
      els.refineThumb.src = rec.url;
      els.refineThumb.onload = frame;
      requestAnimationFrame(frame);
    }
    return opened;
  };
  const baseSync = syncRefine;
  syncRefine = () => { baseSync(); frame(); };
  new ResizeObserver(frame).observe(stage);

  // 6. The one image just made, with what it is and what to do next beside
  //    it, in the viewer's empty width (panel.js: appendResult builds this
  //    when the strip holds one image; the buttons are gallery.js's own
  //    setFavorite, openRefine and reuseButton).
  const side = document.createElement('div');
  side.className = 'result-side';
  const renderSide = () => {
    const rec = viewerRecords.length === 1 ? viewerRecords[0] : null;
    if (!rec || !wide()) { side.remove(); els.viewer.classList.remove('single-result'); return; }
    const button = (label, onClick) => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'ghost'; b.textContent = label; b.onclick = onClick;
      return b;
    };
    const prompt = document.createElement('p');
    prompt.className = 'result-prompt';
    prompt.textContent = rec.prompt;
    const facts = document.createElement('p');
    facts.className = 'result-facts';
    facts.textContent = `${rec.width}×${rec.height} · seed ${rec.seed} · ${rec.steps} steps · CFG ${rec.guidance_scale}`;
    const actions = document.createElement('div');
    actions.className = 'result-actions';
    const reuseIt = reuseButton(rec, false);
    reuseIt.classList.add('ghost');
    actions.append(
      button(rec.favorite ? '★ Starred' : '☆ Star', async () => { await setFavorite(rec, !rec.favorite); renderSide(); }),
      button('⤢ Refine…', () => openRefine(rec)),
      reuseIt,
    );
    side.replaceChildren(prompt, facts, actions);
    els.viewer.classList.add('single-result');
    if (side.parentElement !== els.viewer) els.viewer.append(side);
  };
  const baseAppend = appendResult;
  appendResult = (rec) => { side.remove(); baseAppend(rec); renderSide(); };
  const baseClear = clearResults;
  clearResults = () => { side.remove(); els.viewer.classList.remove('single-result'); baseClear(); };
  renderSide();
})();
