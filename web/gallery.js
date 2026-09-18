// Local Diffusion's page script: the gallery grid, its cards and Reuse. Declarations only; state.js has
// the load-order contract these files share.
'use strict';

// -------------------------------------------------------------------gallery

function resetGallery() {
  // Anything already in flight was fetched under the old filters and against
  // the grid this is about to empty; the token tells it to drop itself.
  galleryEpoch += 1;
  galleryMore = false;
  galleryRecords = [];
  els.gallery.innerHTML = '';
  // The fresh query has to be issued *after* an in-flight page settles rather
  // than sharing its promise: that page carries the old epoch and drops
  // itself, so sharing it left the gallery empty with nothing outstanding and
  // nothing to retry. Its `.finally` nulls `galleryPending` first, so the
  // chained call finds a free slot.
  const pending = galleryPending;
  return pending ? pending.then(loadGallery) : loadGallery();
}

function loadGallery() {
  // One page at a time. Nothing else serialises this - a double-clicked "Load
  // more", or an arrow key held down in the lightbox - and two calls reading
  // the same anchor append the same 48 cards twice.
  if (galleryPending) return galleryPending;
  const epoch = galleryEpoch;
  const params = new URLSearchParams({ limit: sheetMode ? sheetPage : PAGE });
  // The sheet asks for the whole run at once, so only the grid ever pages.
  const last = galleryRecords[galleryRecords.length - 1];
  if (!sheetMode && last) params.set('before', last.id);
  if (els.search.value.trim()) params.set('search', els.search.value.trim());
  if (els.favOnly.checked) params.set('favorites', 'true');
  if (els.allModels.checked) params.set('all_models', 'true');
  els.more.disabled = true;
  els.more.textContent = 'Loading…';
  els.lbNext.disabled = true;

  galleryPending = api(`/api/gallery?${params}`)
    .then((page) => {
      if (epoch !== galleryEpoch) return;  // superseded by a reset
      total = page.total;
      els.count.textContent = total ? `(${total})` : '';
      // What the tick would add: the same filter's matches on other
      // checkpoints, so a run made on the other model reads as "+12" on the
      // tick rather than as an empty grid. 0 once it is on.
      els.otherModels.textContent = page.other_models ? ` (+${page.other_models})` : '';
      syncRerender();
      syncDeleteUnstarred();
      // Keyset paging cannot repeat a row; the guard stays for a stream that
      // reattaches and a page that overlap through the viewer's records.
      const have = new Set(galleryRecords.map((r) => r.id));
      page.items.forEach((rec) => {
        if (have.has(rec.id)) return;
        galleryRecords.push(rec);
        if (!sheetMode) els.gallery.appendChild(card(rec));
      });
      galleryMore = !!page.has_more;
      els.more.hidden = sheetMode || !galleryMore;
      watchMore();
      if (sheetMode) renderSheet();
    })
    .catch((err) => {
      // A filter term that is not a number, most often: report it rather than
      // leaving the gallery silently showing the last good page.
      if (epoch === galleryEpoch) showError(err.message);
    })
    .finally(() => {
      galleryPending = null;
      els.more.disabled = false;
      els.more.textContent = 'Load more';
      if (els.lightbox.open) renderLightbox();  // re-derives the nav buttons
    });
  return galleryPending;
}

// Re-arms the scroll trigger once a page has landed. The observer reports a
// *change* in whether the button is in view, and a page that does not push it
// past the fold - a wide grid, the tail of a short run - leaves it in view with
// nothing left to fire; observing afresh always reports the current state once.
// Called on success only: after a failed page the button is the retry, and
// re-arming would fetch again from the server that just refused.
function watchMore() {
  if (!moreWatcher) return;
  moreWatcher.unobserve(els.more);
  moreWatcher.observe(els.more);
}

function card(rec) {
  const div = document.createElement('div');
  div.className = `card${rec.favorite ? ' fav' : ''}`;
  // Tagged by id so an action taken elsewhere - the lightbox, mostly - can
  // find this card and keep it in step.
  div.dataset.id = rec.id;

  const img = thumb(rec, galleryRecords);

  const star = tool(rec.favorite ? '★' : '☆', 'Favorite', () => setFavorite(rec, !rec.favorite));
  star.className = 'star';

  const tools = document.createElement('div');
  tools.className = 'tools';
  tools.append(
    reuseButton(rec, true),
    tool('❐', 'Use as a reference image', () => {
      if (!referenceSupported) {
        showError(
          'This checkpoint takes no reference images - its pipeline has no `image` '
          + 'parameter. FLUX.2 [klein] is the one here that conditions on them.',
          true,
        );
        return;
      }
      if (referenceIds.includes(rec.id)) { setReferences(referenceIds); return; }
      if (referenceIds.length >= 4) {
        showError('Four reference images is the limit; each one lengthens the sequence.', true);
        return;
      }
      setReferences([...referenceIds, rec.id]);
    }),
    tool('⊞', 'Show this run as a sheet', () => {
      if (!rec.run_id && !rec.job_id) {
        showError('That image predates runs being grouped, so it belongs to no run.', true);
        return;
      }
      // The run id spans every job that worked on the grid - a night that died
      // and was resumed is one run - where the job id names only one of them.
      els.search.value = rec.run_id ? `run:${rec.run_id}` : `job:${rec.job_id}`;
      setSheet(true);
    }),
    star,
    tool('⌦', 'Delete', () => deleteImage(rec)),
  );

  div.append(img, tools);
  return div;
}

function tool(label, title, onClick) {
  const b = document.createElement('button');
  b.textContent = label;
  b.title = title;
  // A button is named by its contents, and `title` is never exposed - so these
  // announced as their glyph, or as a bare "button" for a glyph the screen
  // reader has no word for. The fourth of them deletes the image.
  b.setAttribute('aria-label', title);
  b.onclick = onClick;
  return b;
}

function modelsDiffer(modelId) {
  return !!loadedModel && !!modelId && modelId !== loadedModel;
}

// Warn before the click, not only after the panel has already been replaced.
// Gallery cards use a compact glyph; the lightbox has room for the full label.
// The data attributes let the first status response update buttons that were
// rendered while the server was still answering.
function syncReuseButton(button, modelId, compact = false) {
  button.dataset.reuseModel = modelId || '';
  button.dataset.reuseCompact = String(compact);
  button.classList.add('reuse-settings');
  const mismatch = modelsDiffer(modelId);
  const from = (modelId || '').split('/').pop();
  const current = (loadedModel || '').split('/').pop();
  button.classList.toggle('model-mismatch', mismatch);
  button.textContent = compact ? (mismatch ? '⚠' : '↻')
    : `${mismatch ? '⚠ ' : ''}Reuse settings`;
  button.title = mismatch
    ? `Warning: generated with ${from}; the current model is ${current}`
    : 'Load these settings into the panel';
  button.setAttribute('aria-label', mismatch
    ? `Reuse settings (warning: generated with ${from}; current model is ${current})`
    : 'Reuse settings');
}

function reuseButton(rec, compact = false) {
  const button = tool('', '', () => reuse(rec));
  syncReuseButton(button, rec.model_id, compact);
  return button;
}

function syncReuseButtons() {
  document.querySelectorAll('.reuse-settings').forEach((button) => {
    syncReuseButton(button, button.dataset.reuseModel, button.dataset.reuseCompact === 'true');
  });
}

// The same image can be on the page more than once - in the viewer, in the
// gallery grid, on the sheet, in the lightbox - as separate record objects
// fetched at different times. So both actions apply by id: to every record
// that carries it, and to every element tagged with it.
function recordsWithId(id) {
  return [...galleryRecords, ...viewerRecords, ...lbList].filter((r) => r.id === id);
}

async function setFavorite(rec, value) {
  try {
    await api(`/api/images/${rec.id}/favorite?value=${value}`, { method: 'POST' });
  } catch (err) {
    showError(err.message);
    return;
  }
  recordsWithId(rec.id).forEach((r) => { r.favorite = value; });
  document.querySelectorAll(`.card[data-id="${rec.id}"]`).forEach((div) => {
    div.classList.toggle('fav', value);
    const star = div.querySelector('.star');
    star.textContent = value ? '★' : '☆';
    star.setAttribute('aria-label', value ? 'Unfavorite' : 'Favorite');
  });
  // The sheet's cell too - by class, never by re-rendering the sheet, which
  // rebuilds up to a thousand <img> nodes and resets the scroll per keypress.
  document.querySelectorAll(`.sheet-cell[data-id="${rec.id}"]`).forEach((cell) => {
    cell.classList.toggle('fav', value);
  });
  if (els.lightbox.open) renderLightbox();
}

// The delete removes the PNG as well as the row, and on a card the button sits
// a few pixels from the star. Say which image, so a slip is caught here.
function confirmDelete(rec) {
  const snippet = rec.prompt.length > 60 ? `${rec.prompt.slice(0, 57)}…` : rec.prompt;
  const what = `${rec.favorite ? 'favourite, ' : ''}seed ${rec.seed}`;
  return window.confirm(`Delete this image (${what})?\n\n"${snippet}"\n\nThe file is removed from disk.`);
}

async function deleteImage(rec) {
  if (!confirmDelete(rec)) return;
  try {
    await api(`/api/images/${rec.id}`, { method: 'DELETE' });
  } catch (err) {
    showError(err.message);
    return;
  }
  forgetImages([rec.id]);
}

// Every trace of these ids on the page, once the server has let them go: the
// cards, the sheet cells, the viewer tiles, and the records behind them.
// Shared by the card's delete and Delete unstarred.
function forgetImages(ids) {
  const gone = new Set(ids);
  document.querySelectorAll('[data-id]').forEach((el) => { if (gone.has(+el.dataset.id)) el.remove(); });
  // Only the grid's counters mirror the server's: the viewer holds whatever
  // this session made, and the sheet lays itself out from the grid's records.
  const inGallery = galleryRecords.filter((r) => gone.has(r.id)).length;
  for (const list of new Set([galleryRecords, viewerRecords, lbList])) {
    for (let at = list.length - 1; at >= 0; at--) if (gone.has(list[at].id)) list.splice(at, 1);
  }
  // The badge follows the server's count; the page anchor is an id, which a
  // delete does not move.
  if (inGallery) {
    total -= inGallery;
    els.count.textContent = total ? `(${total})` : '';
    syncRerender();
    syncDeleteUnstarred();
  }
  els.viewer.classList.toggle('multi', viewerRecords.length > 1);
  els.viewer.classList.toggle('many', viewerRecords.length > 4);
  if (sheetMode) renderSheet();
  if (!els.lightbox.open) return;
  // The strip has closed up under the viewer: the next image already sits at
  // this index, unless this was the last one.
  if (!lbList.length) closeLightbox();
  else { lbIndex = Math.min(lbIndex, lbList.length - 1); renderLightbox(); }
}

// The inverse of the card's delete, for the night that made hundreds and
// earned a few stars: star those, then remove everything else the filter
// matches in one gesture. Counted server-side, as Re-render's selection is -
// the browser holds one page - and gated on typing that count back: the
// hazard is no longer "the wrong image" but "the wrong scope", a run not yet
// looked at, and the number is what tells them apart. The delete is pinned
// to it, so images a running job lands meanwhile are refused, not swept.
async function deleteUnstarred() {
  const search = els.search.value.trim();
  const params = new URLSearchParams();
  if (search) params.set('search', search);
  if (els.allModels.checked) params.set('all_models', 'true');
  let found;
  try {
    found = await api(`/api/gallery/unstarred?${params}`);
  } catch (err) {
    showError(err.message);
    return;
  }
  // The checkpoint scope is part of what is confirmed: with All models off
  // the sweep stops at this checkpoint's images, and the dialog says so by
  // the server's own name for it.
  const only = els.allModels.checked ? '' : (found.models[0] ?? loadedModel ?? '').split('/').pop();
  const scope = (search ? `matching "${search}"` : 'in the gallery') + (only ? ` (${only} only)` : '');
  if (!found.unstarred) {
    showError(`Nothing to delete: every image ${scope} is starred.`, true);
    return;
  }
  const plural = found.unstarred === 1 ? '' : 's';
  const prompts = found.prompts.slice(0, 3).map((p) => `  "${p.length > 60 ? `${p.slice(0, 57)}…` : p}"`);
  if (found.prompts.length > 3) prompts.push('  …');
  const message = [
    `Delete ${found.unstarred} unstarred image${plural} ${scope}, keeping ${found.favorites} starred?`,
    '',
    ...prompts,
    '',
    `The files are removed from disk. Type ${found.unstarred} to continue:`,
  ].join('\n');
  const typed = window.prompt(message, '');
  if (typed === null) return;
  if (typed.trim() !== String(found.unstarred)) {
    showError(`Nothing deleted: "${typed.trim()}" is not ${found.unstarred}.`, true);
    return;
  }
  params.set('expected', found.unstarred);
  let result;
  try {
    result = await api(`/api/gallery/unstarred?${params}`, { method: 'DELETE' });
  } catch (err) {
    showError(err.message);
    return;
  }
  forgetImages(result.deleted);
  showError(`Deleted ${result.deleted.length} image${plural}, keeping ${found.favorites} starred.`, true);
  // The page is now mostly holes; ask for it again from the top, which also
  // brings the badge back to the server's count.
  resetGallery();
}

// Shown while there is something it could act on: the filter matches images
// and the Favorites tick is off, since with it on every match is a keeper.
// Whether the search box reads as a strict scope is the server's call, at
// the click.
function syncDeleteUnstarred() {
  els.deleteUnstarred.hidden = els.favOnly.checked || total <= 0;
}

function reuse(rec) {
  ownControls();
  els.prompt.value = rec.prompt;
  clearPromptTranslation();
  els.negative.value = rec.negative_prompt ?? '';
  els.width.value = rec.width;
  els.height.value = rec.height;
  els.steps.value = rec.steps;
  els.cfg.value = rec.guidance_scale;
  // The scale the negative prompt ran at. Only where the box is a knob of its
  // own: on a checkpoint whose CFG is real CFG the record's value *is* the
  // CFG box, and the greyed box would only echo it.
  els.trueCfg.value = trueCfgSupported && rec.true_cfg != null ? rec.true_cfg : '';
  // Turbo's flow shift. Records from a pipeline that derives its own carry null.
  els.mu.value = rec.mu ?? '';
  // The conditioning that made it, where this server can still apply it: the
  // references are ids in this same gallery, and the adapter has to be a file
  // the select is offering or the request would be refused at submit.
  referenceIds = referenceSupported ? (rec.reference ?? []) : [];
  const listed = [...els.lora.options].some((o) => o.value === rec.lora);
  els.lora.value = loraSupported && rec.lora && listed ? rec.lora : '';
  els.loraScale.value = els.lora.value ? (rec.lora_scale ?? 1) : 1;
  els.seed.value = rec.seed;
  syncSize();
  syncCfg();
  syncConditioning();
  syncRuns();
  schedulePreview();
  scheduleTagValidation(0);
  persistPanel();
  // Steps, CFG and mu are checkpoint-specific: Turbo's 8 / 0 / 1.15 on FLUX is
  // a visibly broken image recorded under FLUX. The checkpoint is part of the
  // reproduction set as much as the seed, so say so - as a note, not a failure.
  const short = (id) => (id || '').split('/').pop();
  if (loadedModel && rec.model_id !== loadedModel) {
    showError(
      `These settings were made on ${short(rec.model_id)}; the server is running ` +
      `${short(loadedModel)}, whose steps, CFG and mu differ. Prompt, size and seed carry ` +
      'over; check the three sampling boxes before generating.',
      true,
    );
  } else {
    showError('');
  }
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  window.scrollTo({ top: 0, behavior: reduce ? 'auto' : 'smooth' });
}
