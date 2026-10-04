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
  // What the gallery said was about the grid this empties. Not the lightbox's
  // line: a job ending under an open lightbox resets the gallery too.
  showError('', false, { slot: els.galleryError, mirror: false });
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
      // Under a filter the sheet waits for its run's order - learnSheetOrder.
      if (sheetMode) return learnSheetOrder(epoch).then(() => { if (epoch === galleryEpoch) renderSheet(); });
    })
    .catch((err) => {
      // A filter term that is not a number, most often: report it rather than
      // leaving the gallery silently showing the last good page - and where
      // the empty grid is, without the count of the filter before it.
      if (epoch !== galleryEpoch) return;
      showError(err.message, false, { slot: els.galleryError });
      if (!galleryRecords.length) els.count.textContent = els.otherModels.textContent = '';
    })
    .finally(() => {
      galleryPending = null;
      els.more.disabled = false;
      els.more.textContent = 'Load more';
      // Re-derives the nav buttons - while there is an image to show: a
      // delete paging in behind the last one (`pageInBehind`) places itself.
      if (els.lightbox.open && lbList[lbIndex]) renderLightbox();
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
    tool('⤢', 'Refine: scale this image up and re-denoise it part of the way', () => openRefine(rec)),
    tool('❐', 'Use as a reference image', () => {
      if (!referenceSupported) {
        showError("This checkpoint can't use reference images; FLUX.2 [klein] can.", true, { slot: els.galleryError });
        return;
      }
      if (referenceIds.includes(rec.id)) { setReferences(referenceIds); return; }
      if (referenceIds.length >= 4) {
        showError('Four reference images is the most this panel sends; remove one first.', true, { slot: els.galleryError });
        return;
      }
      setReferences([...referenceIds, rec.id]);
    }),
    tool('⊞', 'Show this run as a sheet', () => {
      if (!rec.run_id && !rec.job_id) {
        showError('That image predates runs being grouped, so it belongs to no run.', true, { slot: els.galleryError });
        return;
      }
      // The run id spans every job that worked on the grid - a night that died
      // and was resumed is one run - where the job id names only one of them.
      showSheetScope(rec.run_id ? `run:${rec.run_id}` : `job:${rec.job_id}`);
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

// Gallery ids as the page names images: "#8", "#8 and #4", "#8, #4 and #2".
function imageNames(ids) {
  const names = ids.map((id) => `#${id}`);
  return names.length > 1 ? `${names.slice(0, -1).join(', ')} and ${names.at(-1)}` : names.join('');
}

// The delete removes the PNG as well as the row, and on a card the button sits
// a few pixels from the star. Say which image, so a slip is caught here: the
// cells of a sweep share a prompt and a seed, and a pass shares both with its
// source, so the settings line is what tells them apart. And say what else it
// costs - `passes` is the gallery's answer to `refine:<id>`, the passes made
// from this image, which could not be made again without it.
function confirmDelete(rec, passes) {
  const snippet = rec.prompt.length > 60 ? `${rec.prompt.slice(0, 57)}…` : rec.prompt;
  const costs = [];
  if (passes?.total === 1) {
    costs.push(`#${passes.items[0].id} was refined from this one; once it is gone, that pass cannot be made again.`);
  } else if (passes?.total > 1) {
    const named = passes.items.map((r) => `#${r.id}`).join(', ') + (passes.total > passes.items.length ? ', …' : '');
    costs.push(`${passes.total} refine passes were made from this one (${named}); once it is gone, they cannot be made again.`);
  }
  const at = referenceIds.indexOf(rec.id);
  if (at >= 0) costs.push(`It is reference image ${at + 1} in the panel, and leaves the references with it.`);
  return window.confirm([
    `Delete #${rec.id}${rec.favorite ? ' (starred)' : ''}?`,
    '',
    settingsLine(rec),
    `"${snippet}"`,
    ...(costs.length ? ['', ...costs] : []),
    '',
    'The PNG is deleted from disk, not moved to a recycle bin.',
  ].join('\n'));
}

// The ids whose delete is under way, from the press to the server's answer.
// The confirm waits on the question about passes now, and a second press in
// that time - a double-click on ⌦, the lightbox's Delete held down - asked
// about the same image twice and sent two deletes.
const deleting = new Set();

async function deleteImage(rec) {
  if (deleting.has(rec.id)) return;
  deleting.add(rec.id);
  try {
    await deleteOne(rec);
  } finally {
    deleting.delete(rec.id);
  }
}

async function deleteOne(rec) {
  // Asked before the confirm, so it can name the passes - on any checkpoint.
  // A failed look asks without them; the delete then says what is wrong.
  let passes = null;
  try {
    passes = await api(`/api/gallery?limit=3&all_models=true&search=refine:${rec.id}`);
  } catch { /* asked without */ }
  if (!confirmDelete(rec, passes)) return;
  const fromLightbox = els.lightbox.open;
  let already = false;
  try {
    await api(`/api/images/${rec.id}`, { method: 'DELETE' });
  } catch (err) {
    if (!/unknown image/.test(err.message)) {
      showError(err.message, false, { slot: els.galleryError });
      return;
    }
    // Gone already - from another tab, or a record a reattached job had
    // replayed: what the delete was for has happened, so it is a note.
    already = true;
  }
  const also = forgetImages([rec.id]);
  const said = already ? `#${rec.id} was already deleted.` : `Deleted #${rec.id} (seed ${rec.seed}).`;
  // A card's tombstone says it in place. The lightbox has moved on to the
  // next image, so it says which one went, until the next step.
  if (fromLightbox && els.lightbox.open) noteLightboxDelete(said + also);
  else if (already || also) showError((already ? said + also : also).trim(), true, { slot: els.galleryError });
  // The Runs picker counts images; it is the gallery's table of contents.
  loadRuns();
}

// After a delete in the lightbox the next image is in view at the same zoom,
// and on a sweep's near-identical cells nothing showed that anything had
// happened. Said in the gallery's slot, which mirrors into the lightbox's
// role=alert line, and taken back at the next key or press in the lightbox.
function noteLightboxDelete(text) {
  showError(text, true, { slot: els.galleryError });
  // Each copy only while it still says this: a job ending meanwhile resets
  // the gallery's line and leaves the lightbox's.
  const clear = () => {
    if (els.galleryError.textContent === text) showError('', false, { slot: els.galleryError });
    else if (els.lbError.textContent === text) { els.lbError.hidden = true; els.lbError.textContent = ''; }
  };
  els.lightbox.addEventListener('keydown', clear, { once: true });
  els.lightbox.addEventListener('pointerdown', clear, { once: true });
  if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    els.lbImg.animate([{ opacity: 0.25 }, { opacity: 1 }], { duration: 180 });
  }
}

// Every trace of these ids on the page, once the server has let them go: the
// cards, the sheet cells, the viewer tiles, the panel's references and the
// records behind them. Shared by the card's delete, the lightbox's, Delete
// unstarred and a tile whose picture has gone. A card and a viewer tile keep
// their place as a tombstone: removed, the next card slid under the pointer
// that pressed ⌦ - its own ⌦ lit - and an emptied viewer shrank the page by
// half a screen. Answers a sentence for the caller's note when references
// went with them, else ''.
function forgetImages(ids) {
  const gone = new Set(ids);
  const records = new Map([...galleryRecords, ...viewerRecords, ...lbList].map((r) => [r.id, r]));
  const focused = document.activeElement;
  let lost = null;
  document.querySelectorAll('[data-id]').forEach((el) => {
    if (!gone.has(+el.dataset.id)) return;
    if (!el.classList.contains('card') && el.parentElement !== els.viewer) { el.remove(); return; }
    const held = el.contains(focused);
    const kept = tombstone(el, records.get(+el.dataset.id));
    if (held) lost = kept;
  });
  // Only the grid's counters mirror the server's: the viewer holds whatever
  // this session made, and the sheet lays itself out from the grid's records.
  const inGallery = galleryRecords.filter((r) => gone.has(r.id)).length;
  const fromViewer = viewerRecords.filter((r) => gone.has(r.id)).map((r) => r.id);
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
  // Keyboard focus was on what went: it moves on to the next image in the
  // same strip, or back to the one before, rather than dropping to <body>.
  if (lost) {
    const strip = [...lost.parentElement.children].filter((el) => !el.matches('.gone, .tomb, .placeholder, .preview'));
    const after = strip.find((el) => lost.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_FOLLOWING);
    const next = after ?? strip.at(-1);
    (next?.matches('img') ? next : next?.querySelector('img'))?.focus();
  }
  // A strip with images left keeps its tombstones in their slots. One whose
  // last image has gone, with no job to fill it, says what it is for again -
  // in place of the tombstones, which beside it read as leftovers - at the
  // height the strip had, so the gallery below does not jump; the next run's
  // first image clears it, height and all.
  if (fromViewer.length && !viewerRecords.length && !busy) {
    const css = getComputedStyle(els.viewer);
    const placeholder = document.createElement('p');
    placeholder.className = 'placeholder held';
    placeholder.style.minHeight = `${els.viewer.clientHeight - parseFloat(css.paddingTop) - parseFloat(css.paddingBottom)}px`;
    const names = fromViewer.length > 3 ? `${fromViewer.length} images` : imageNames(fromViewer);
    placeholder.textContent = `${names} ${fromViewer.length === 1 ? 'was' : 'were'} deleted. Generated images appear here.`;
    els.viewer.replaceChildren(placeholder);
    els.viewer.classList.remove('multi', 'many');
    els.viewerNote.textContent = '';
  }
  for (const id of gone) if (sheetSources.has(id)) sheetSources.set(id, null);
  if (sheetMode) renderSheet();
  // A reference that has gone would be refused at submit, naming an id the
  // chips never showed.
  const dropped = referenceIds.filter((id) => gone.has(id));
  if (dropped.length) setReferences(referenceIds.filter((id) => !gone.has(id)));
  // Nor may a Reuse's Undo put one back.
  if (reuseNote) reuseNote.before.state.reference = reuseNote.before.state.reference.filter((id) => !gone.has(id));
  const note = !dropped.length ? ''
    : dropped.length === 1 ? ` ${imageNames(dropped)} was a reference image; it is no longer in the references.`
      : ` ${imageNames(dropped)} were reference images; they are no longer in the references.`;
  if (!els.lightbox.open) return note;
  // The strip has closed up under the viewer: the next image already sits at
  // this index - unless this was the last one loaded and the server has more,
  // when the next is on its next page, not the image before.
  if (lbIndex >= lbList.length && canPage()) pageInBehind();
  else if (!lbList.length) closeLightbox();
  else { lbIndex = Math.min(lbIndex, lbList.length - 1); renderLightbox(); }
  return note;
}

// A card or a viewer tile that has gone, kept as a box of its size that says
// so. Out of the id-tagged set and of the pointer's way: nothing acts on it.
// Answers the element that holds the place now.
function tombstone(el, rec) {
  const tomb = document.createElement('div');
  tomb.className = 'tomb';
  tomb.textContent = rec ? `Deleted · seed ${rec.seed}` : 'Deleted';
  if (el.classList.contains('card')) {
    el.replaceChildren(tomb);
    el.classList.remove('fav');
    el.classList.add('gone');
    delete el.dataset.id;
    return el;
  }
  const box = el.getBoundingClientRect();
  tomb.style.width = `${box.width}px`;
  tomb.style.height = `${box.height}px`;
  el.replaceWith(tomb);
  return tomb;
}

// The lightbox's last loaded image went while the server has more: the page
// after it is fetched and the lightbox steps onto its first image. Meanwhile
// the deleted image is off the screen, and with the index past the strip's
// end its buttons act on nothing (each asks `lbList[lbIndex]` first) - not
// disabled, which would drop the focus of the one just pressed to <body>.
function pageInBehind() {
  els.lbImg.removeAttribute('src');
  els.lbImg.alt = '';
  els.lbPrompt.textContent = 'Loading next…';
  els.lbSettings.textContent = '';
  els.lbCount.textContent = '';
  loadGallery().catch(() => {}).then(() => {
    if (!els.lightbox.open) return;
    // A page that failed, or brought nothing, leaves the one before.
    lbIndex = Math.min(lbIndex, lbList.length - 1);
    if (lbIndex >= 0) renderLightbox();
    else closeLightbox();
  });
}

// A viewer tile whose picture would not load: deleted from another tab, or
// replayed by a job from before its delete - or the server is only out of
// reach. Asked, and only "unknown image" forgets it.
async function forgetIfGone(id) {
  try {
    await api(`/api/images/${id}`);
  } catch (err) {
    if (!/unknown image/.test(err.message)) return;
    const note = forgetImages([id]);
    if (note) showError(note.trim(), true, { slot: els.galleryError });
  }
}

// The inverse of the card's delete, for the night that made hundreds and
// earned a few stars: star those, then remove everything else the filter
// matches in one gesture. Counted server-side, as Re-render's selection is -
// the browser holds one page - and gated on typing that count back: the
// hazard is no longer "the wrong image" but "the wrong scope", a run not yet
// looked at, and the number is what tells them apart. The delete is pinned
// to it, so images a running job lands meanwhile are refused, not swept.
// A starred image keeps what it was made from - a pass its source, a render
// its references - and the dialog names those. Everything it says, it says
// under the gallery's head, where the button is.
async function deleteUnstarred() {
  const slot = { slot: els.galleryError };
  const search = els.search.value.trim();
  const params = new URLSearchParams();
  if (search) params.set('search', search);
  if (els.allModels.checked) params.set('all_models', 'true');
  let found;
  try {
    found = await api(`/api/gallery/unstarred?${params}`);
  } catch (err) {
    showError(err.message, false, slot);
    return;
  }
  // The checkpoint scope is part of what is confirmed: with All models off
  // the sweep stops at this checkpoint's images, and the dialog says so by
  // the server's own name for it. With it on, the wider sweep names every
  // checkpoint it reaches (the server lists six at most).
  const named = (id) => (id || '').split('/').pop();
  let scope = search ? `matching "${search}"` : 'in the gallery';
  if (!els.allModels.checked) {
    const only = named(found.models[0] ?? loadedModel);
    if (only) scope += ` (${only} only)`;
  } else if (found.models.length) {
    const many = found.models.length >= 6;
    const count = many ? '6 or more' : found.models.length;
    scope += ` across ${count} checkpoint${found.models.length === 1 ? '' : 's'} `
      + `(${found.models.map(named).join(', ')}${many ? ', …' : ''})`;
  }
  const sources = found.sources_kept ?? 0;
  if (!found.unstarred) {
    const kept = sources ? ' or one a starred image was made from' : '';
    showError(`Nothing to delete: every image ${scope} is starred${kept}.`, true, slot);
    return;
  }
  const plural = found.unstarred === 1 ? '' : 's';
  const sourceNames = found.sources.map((id) => `#${id}`).join(', ') + (sources > found.sources.length ? ', …' : '');
  const keeping = `keeping ${found.favorites} starred`
    + (sources === 1 ? ` and ${sourceNames}, which a starred image was made from`
      : sources > 1 ? ` and ${sources} images starred ones were made from (${sourceNames})` : '');
  // How much of the gallery's history goes: the sample prompts are the first
  // three alphabetically and say nothing of how many runs that is, or when.
  const from = found.first ? when(Date.parse(found.first)) : '';
  const to = found.last ? when(Date.parse(found.last)) : '';
  const span = [
    found.runs ? `From ${found.runs} run${found.runs === 1 ? '' : 's'}` : '',
    from && (from === to ? `made ${from}` : `made ${from} to ${to}`),
  ].filter(Boolean).join(', ');
  const prompts = found.prompts.slice(0, 3).map((p) => `  "${p.length > 60 ? `${p.slice(0, 57)}…` : p}"`);
  if (found.prompts.length > 3) prompts.push('  …');
  const message = [
    `Delete ${found.unstarred} unstarred image${plural} ${scope}, ${keeping}?`,
    ...(span ? [`${span}.`] : []),
    '',
    ...prompts,
    '',
    // The count is pinned: an image a running job lands in scope before the
    // answer refuses the whole sweep, and that is better known before typing.
    ...(busy ? ['A job is still running; if it adds an image here before you confirm, nothing is deleted.'] : []),
    `The files are removed from disk. Type ${found.unstarred} to continue:`,
  ].join('\n');
  const typed = window.prompt(message, '');
  if (typed === null) return;
  if (typed.trim() !== String(found.unstarred)) {
    showError(`Nothing deleted: "${typed.trim()}" is not ${found.unstarred}.`, true, slot);
    return;
  }
  params.set('expected', found.unstarred);
  let result;
  try {
    result = await api(`/api/gallery/unstarred?${params}`, { method: 'DELETE' });
  } catch (err) {
    if (!/selection changed/i.test(err.message)) { showError(err.message, false, slot); return; }
    // The server counts something else now - images a job landed in scope,
    // most often. The page shows the server's count rather than the one the
    // message has just called stale.
    resetGallery();
    loadRuns();
    showError(err.message + (busy
      ? ' A running job is still adding images here: press Delete unstarred again, or let it finish first.'
      : ''), false, slot);
    return;
  }
  const also = forgetImages(result.deleted);
  // The page is now mostly holes; ask for it again from the top, which also
  // brings the badge back to the server's count - and clears the slot, so
  // the outcome is written after it.
  resetGallery();
  loadRuns();
  const went = result.deleted.length;
  showError(`Deleted ${went} image${went === 1 ? '' : 's'}, ${keeping}.${also}`, true, slot);
}

// Shown while there is something it could act on: the filter matches images
// and the Favorites tick is off, since with it on every match is a keeper.
// Whether the search box reads as a strict scope is the server's call, at
// the click.
function syncDeleteUnstarred() {
  els.deleteUnstarred.hidden = els.favOnly.checked || total <= 0;
}

// What made a record, into the panel: the prompt pair and the seed, and with
// `sampling` its steps, CFG, true CFG, mu and conditioning too. The size is
// not written here: a refine pass whose source has gone keeps the panel's.
// The two boxes are written as typing writes them (`replaceBoxText`), so
// Ctrl+Z gives back what they held: assigning `.value` emptied their undo
// history, and in Chromium a later Ctrl+Z could then splice the old prompt
// into the new one, which the panel went on to remember. The negative first,
// so the caret ends in the prompt. Synchronous, all of it: the references are
// in place when this returns.
function fillPanel(rec, sampling) {
  rewriteBox(els.negative, rec.negative_prompt ?? '');
  rewriteBox(els.prompt, rec.prompt);
  // The insert's `input` asked for tag suggestions at the caret, as typing
  // does; a reused prompt is not typing, and none open over it.
  clearTimeout(tagSuggestionTimer);
  hideTagSuggestions();
  promptReplaced();
  els.seed.value = rec.seed;
  if (!sampling) return;
  ownControls();
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
  referenceIds = referenceSupported ? [...(rec.reference ?? [])] : [];
  const listed = [...els.lora.options].some((o) => o.value === rec.lora);
  els.lora.value = loraSupported && rec.lora && listed ? rec.lora : '';
  els.loraScale.value = els.lora.value ? (rec.lora_scale ?? 1) : 1;
}

// `replaceBoxText`, with the box focused first without the browser's scroll to
// it. Writing as typing does takes the focus, and the browser then scrolled the
// page to the box as though a stuck panel moved with it: half a screen of the
// gallery lost at the page's end, nothing at all for a box focused already.
// Where the page goes after a Reuse is `revealReused`'s, and after its Undo,
// pressed at the panel's foot, nowhere.
function rewriteBox(el, text) {
  el.focus({ preventScroll: true });
  replaceBoxText(el, text);
}

// Reuse restores what made a record. A direct render's settings go into the
// panel; a refine pass is the Refine dialog's, so a refined record opens the
// dialog on its source with the pass as it ran (`reusePass`).
function reuse(rec) {
  if (rec.refine != null) { void reusePass(rec); return; }
  reuseIntoPanel(rec, true);
}

// The note the last Reuse left in the panel - its sentences, the panel as it
// was before, and the Undo button that puts it back - or null.
let reuseNote = null;

function reuseIntoPanel(rec, withSize, note = '') {
  // Ctrl+Z covers the two text boxes; Reuse also replaces the seed, the size,
  // steps/CFG/mu and which of those the user had taken off the profile, so
  // the note's Undo restores those from this.
  const before = {
    state: panelState(),
    values: { steps: els.steps.value, cfg: els.cfg.value, mu: els.mu.value, trueCfg: els.trueCfg.value },
    owned: { ...owned },
  };
  fillPanel(rec, true);
  if (withSize) {
    els.width.value = rec.width;
    els.height.value = rec.height;
  }
  syncSize();
  syncCfg();
  syncConditioning();
  syncRuns();
  schedulePreview();
  scheduleTagValidation(0);
  persistPanel();
  // Reuse used to replace a half-written prompt with no word, and the work
  // was gone; the note says so whenever there was other text to replace. Its
  // first sentence, the Undo after it, then whatever else there is to say:
  // the note is at the panel's foot, and where the window shows the prompt
  // and only the start of the note, the start is what is in sight.
  const replaced = before.state.prompt.trim() !== '' && before.state.prompt !== rec.prompt;
  const lead = `Settings from #${rec.id} loaded${replaced ? ' - your prompt was replaced' : ''}.`;
  const notes = [note];
  // Steps, CFG and mu are checkpoint-specific: Turbo's 8 / 0 / 1.15 on FLUX is
  // a visibly broken image recorded under FLUX. The checkpoint is part of the
  // reproduction set as much as the seed, so say so - as a note, not a failure.
  const short = (id) => (id || '').split('/').pop();
  if (loadedModel && rec.model_id !== loadedModel) {
    notes.push(
      `These settings were made on ${short(rec.model_id)}; the server is running ` +
      `${short(loadedModel)}, whose steps, CFG and mu differ. Prompt` +
      `${withSize ? ', size' : ''} and seed carry over; check the steps, CFG and mu ` +
      'boxes before generating.',
    );
  }
  // What fillPanel could not carry, said: a dropped reference renders a plain
  // text-to-image that looks like a success.
  const references = rec.reference ?? [];
  if (references.length && !referenceSupported) {
    const one = references.length === 1;
    notes.push(
      `Its ${one ? '' : `${references.length} `}reference image${one ? '' : 's'} (${imageNames(references)}) `
      + `${one ? 'is' : 'are'} not used: ${short(loadedModel) || 'this checkpoint'} takes none, `
      + 'so this renders from the prompt alone.',
    );
  }
  reuseNote = { lead, parts: notes, before, id: rec.id, button: null };
  showReuseNote();
  revealReused(reuseNote);
  void checkReused(reuseNote, rec);
}

// The Reuse note: its first sentence, the Undo, the rest. Written in the
// panel's slot - Reuse sends the user to the panel - and rewritten when a
// follow-up has more to say.
function showReuseNote() {
  const note = reuseNote;
  showError(note.lead, true);
  const undo = document.createElement('button');
  undo.type = 'button';
  undo.className = 'ghost tiny undo';
  undo.textContent = 'Undo';
  undo.title = 'Put the panel back as it was before this Reuse';
  undo.onclick = () => undoReuse(note);
  els.error.append(' ', undo);
  const rest = note.parts.filter(Boolean).join(' ');
  if (rest) els.error.append(` ${rest}`);
  note.button = undo;
}

// Where a Reuse leaves the page: the prompt in sight below the topbar, its
// label with it, and the note's Undo too where the window has room for both.
// Not the jump to the top of the page there used to be, which beside the
// gallery lost the user's place for nothing.
function revealReused(note) {
  const panel = document.querySelector('.panel');
  // The note has just made the panel taller: placed now rather than after
  // the frame, so what is measured below is where the panel stays.
  placePanel();
  // Beside the gallery the panel moves, not the page (panel.js's sticky top).
  revealInPanel(els.prompt);
  if (promptInSight()) { showUndo(note); return; }
  // In the page's flow - a one-column page, or a panel at the page's end,
  // which a sticky top cannot push down - the page moves, as far as puts the
  // panel where it rests: under the topbar as at the top of the page, or at
  // its sticky top.
  const rest = panelPos.top === null
    ? document.querySelector('.topbar').getBoundingClientRect().bottom + parseFloat(getComputedStyle(panel.parentElement).paddingTop)
    : panelPos.rest;
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  scrollTo({ top: scrollY + panel.getBoundingClientRect().top - rest, behavior: reduce ? 'auto' : 'smooth' });
  if (reduce) showUndo(note);
  else addEventListener('scrollend', () => showUndo(note), { once: true });
}

// The prompt's field, label and all, between the topbar and the window's foot.
function promptInSight() {
  const field = els.prompt.closest('.field').getBoundingClientRect();
  return field.top >= document.querySelector('.topbar').getBoundingClientRect().bottom - 0.5
    && els.prompt.getBoundingClientRect().bottom <= innerHeight + 0.5;
}

// The Undo, after the note's first sentence: a stuck panel goes up to bring
// it in sight - a gutter clear of the window's foot where there is room for
// that - only as far as the prompt's field stays clear of the topbar. On a
// window too short for both, the prompt is what Reuse is about, and Ctrl+Z
// in it still gives the text back.
function showUndo(note) {
  if (panelPos.top === null || reuseNote !== note || !note.button.isConnected) return;
  const room = els.prompt.closest('.field').getBoundingClientRect().top
    - document.querySelector('.topbar').getBoundingClientRect().bottom;
  const bottom = note.button.getBoundingClientRect().bottom;
  const lift = Math.min(bottom - (innerHeight - panelPos.gap), room);
  if (lift > 0 && bottom - lift <= innerHeight) setPanelTop(panelPos.top - lift);
}

// What a Reuse could only find out by asking, once the panel is filled: a
// restored reference that has since been deleted - dropped, and only on the
// server's "unknown image", so a passing failure never takes a live one out -
// and why an adapter the record ran with is not in the LoRA box.
async function checkReused(note, rec) {
  const extra = [];
  const restored = referenceSupported ? (rec.reference ?? []) : [];
  const answers = await Promise.all(restored.map((id) => api(`/api/images/${id}`)
    .then(() => null, (err) => (/unknown image/.test(err.message) ? id : null))));
  const dead = answers.filter((id) => id != null);
  if (dead.length) {
    setReferences(referenceIds.filter((id) => !dead.includes(id)));
    const which = dead.map((id) => `reference image ${restored.indexOf(id) + 1} (gallery #${id})`).join(' and ');
    extra.push(`Its ${which} ${dead.length === 1 ? 'is' : 'are'} no longer in the gallery, so `
      + `${dead.length === 1 ? 'it was' : 'they were'} left out.`);
  }
  if (rec.lora && els.lora.value !== rec.lora) {
    let reason = 'the file is no longer in the adapter directory';
    try {
      const body = await api('/api/loras');
      const entry = (body.loras ?? []).find((e) => e.name === rec.lora);
      if (!body.supported) reason = body.detail || 'this installation cannot load adapters';
      else if (entry?.detail) reason = entry.detail;
      else if (entry) reason = 'the adapter list had not loaded yet; pick it in the LoRA box';
    } catch { /* the plain reason stands */ }
    extra.push(`Its adapter ${rec.lora} @ ${rec.lora_scale ?? 1} is not applied here - ${reason}.`);
  }
  if (!extra.length) return;
  // Onto the note while it is still the one shown; on its own where nothing
  // has replaced it with something else to say.
  if (reuseNote === note && note.button?.isConnected) {
    note.parts.push(...extra);
    showReuseNote();
  } else if (els.error.hidden) showError(extra.join(' '), true);
}

// The panel as the Reuse found it: the boxes through `replaceBoxText` again,
// so Ctrl+Z still walks back through both, and the seed, the size, the
// sampling values with whose they were, the references and the adapter.
function undoReuse(note) {
  const { state, values } = note.before;
  if (reuseNote === note) reuseNote = null;
  rewriteBox(els.negative, state.negative);
  rewriteBox(els.prompt, state.prompt);
  clearTimeout(tagSuggestionTimer);
  hideTagSuggestions();
  els.seed.value = state.seed;
  els.width.value = state.width;
  els.height.value = state.height;
  Object.assign(owned, note.before.owned);
  els.steps.value = values.steps;
  els.cfg.value = values.cfg;
  els.mu.value = values.mu;
  els.trueCfg.value = values.trueCfg;
  referenceIds = state.reference;
  els.lora.value = state.lora;
  els.loraScale.value = state.loraScale;
  promptReplaced();
  syncSize();
  syncCfg();
  syncConditioning();
  syncRuns();
  schedulePreview();
  scheduleTagValidation(0);
  persistPanel();
  showError(`Reuse of #${note.id} undone: the panel is as it was.`, true);
}

// A refined record holds its source's id, not the source: asked for, so the
// dialog has the picture and the size to scale. A source that has been
// deleted cannot be refined again - the prompt and settings still go into
// the panel, at the panel's own size, since the pass's may be past what a
// direct render takes.
async function reusePass(rec) {
  let source;
  try {
    source = await api(`/api/images/${rec.refine}`);
  } catch (err) {
    if (!/unknown image/.test(err.message)) { showError(err.message); return; }
    reuseIntoPanel(rec, false,
      `Image ${rec.refine}, the source this was refined from, is no longer in the gallery, `
      + 'so the pass cannot be made again. The size is left as it was.');
    return;
  }
  openRefine(source, rec);
}
