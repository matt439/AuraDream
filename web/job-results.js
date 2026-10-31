// AuraDream's page script: job outcomes and newly completed images.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// The finished job's line, #finish-note: its outcome and what to do next -
// the seed and Use seed, Show sheet, Finish run. A line of its own,
// which only the next job's end replaces: in the cell line, a job queued
// behind cleared it the moment the watched one ended, and its recovery action
// with it - in the overnight queue the button exists for. While another job
// runs, or after one whose stream gave up, it says whose it is. Focus on it
// when it is replaced moves to the new line's first button, else Generate,
// rather than to the page's body.
let finishStale = false;
// Set by setBusy where the ending job's Pause or Cancel had the focus, which
// Generate holds until this line is drawn.
let finishTakesFocus = false;
function setFinish(nodes = []) {
  els.finishNote.title = '';
  els.finishNote.removeAttribute('aria-description');
  const focused = els.finishNote.contains(document.activeElement)
    || (finishTakesFocus && document.activeElement === els.generate);
  finishTakesFocus = false;
  const lead = document.createElement('span');
  lead.className = 'lead';
  els.finishNote.replaceChildren(...(nodes.length ? [lead, ...nodes] : []));
  finishStale = false;
  syncFinishLead();
  if (focused) (els.finishNote.querySelector('button') ?? els.generate).focus({ preventScroll: true });
}

function syncFinishLead() {
  const lead = els.finishNote.querySelector('.lead');
  if (lead) lead.textContent = busy || finishStale ? 'Previous job: ' : '';
  // The finished job's sentence in the panel's line says whose it is too.
  els.error.classList.toggle('previous', busy && jobOutcome !== null && els.error.textContent === jobOutcome);
}

// The panel's line (#error) about the watched job and about the finished one.
// A press on a held Pause or Unload says why (`jobNote`), which goes when the
// job ends. A finished job's outcome - a cancel, its time budget, an error,
// the cells it skipped (`jobOutcome`) - lasts as its finish line does: led by
// "Previous job:" while another runs (#error.previous, styles/generation.css), and gone
// once the next job's end is drawn. Both used to stay until the next press of
// Generate, and under a queued job's own finish line "Cancelled before
// anything finished." read as that job's outcome.
let jobNote = null;
let jobOutcome = null;

function showJobNote(text) {
  showError(text, true);
  jobNote = text;
}

function showOutcome(text, note = false) {
  showError(text, note);
  jobOutcome = text;
}

// The finished job's outcome: the finish line, the panel's line for what
// went wrong, the tab's title, and the gallery. `submitted` is the
// submission this page made; `sheet` a job that
// opens as its sheet; `foreign` another tab's job this page picked up
// (isForeign), whose end is said as that - no Use seed, no title held.
function finishJob(jobId, data, { submitted, sheet, foreign = false, staleBefore = false }) {
  const images = data.images ?? [];
  const made = data.image_count ?? images.length;
  // Taken in below: the status's newest id reaching them is this page's own.
  images.forEach((r) => ownImageIds.add(r.id));
  // The job before this one's outcome goes with its finish line.
  if (jobOutcome !== null && els.error.textContent === jobOutcome) showError('');
  jobOutcome = null;
  // A job that ended at the load is the status's to say - the viewer's card,
  // the topbar's Retry load - which it says once, with the steps; the panel
  // line used to give the same six lines again.
  const atLoad = data.status === 'error' && !data.started_at;
  const line = [];
  // Its finish line says it was not made, though: the press had queued an
  // image, and the panel went back to Generate with no word of what became
  // of it. The reason is the viewer's card, or the panel's line under this
  // where the viewer holds images (syncLoadFailure).
  // Marked, so the status can reword it once the load is no longer failed
  // (syncLoadFailure): the card and the line it pointed at go then.
  if (atLoad) {
    const notMade = document.createElement('span');
    notMade.className = 'not-made';
    notMade.dataset.lead = foreign ? `${ANOTHER_TABS_JOB} was not made` : 'Not made';
    notMade.textContent = notMadeText(notMade.dataset.lead, 'error');
    line.push(notMade);
  }
  // Nothing made - every cell skipped - leaves the line to the skipped cells.
  if (data.status === 'done' && data.started_at && data.finished_at && made) {
    // Fewer distinct pictures than images is an axis that changed nothing,
    // said in one number rather than found on the sheet.
    const distinct = typeof data.distinct_count === 'number' && made > 1 && data.distinct_count < made
      ? ` Only ${data.distinct_count} distinct.` : '';
    // What is left of it: the count is what the job made, which a delete
    // since does not undo.
    const deleted = data.deleted_count ? `, ${data.deleted_count} since deleted,` : '';
    // A plain job's seeds, which a blank Seed box drew and nothing on the
    // page showed short of the lightbox: Use seed keeps the composition for
    // the next prompt, and writes nothing else.
    const plain = !data.grid && data.refine == null && images.length;
    // More seeds' too, where the seeds they took are the news: a batch's
    // run on from its highest, the same span for each batch of one run.
    const seeded = plain || (data.more_seeds != null && images.length);
    const span = images.map((image) => image.seed);
    const seeds = !seeded ? ''
      : images.length === 1 ? `, seed ${images[0].seed}`
      : `, seeds ${Math.min(...span)}–${Math.max(...span)}`;
    const took = humanTime(data.finished_at - data.started_at);
    if (foreign) {
      // Its seeds are another tab's to use; its images join the gallery.
      line.push(`${ANOTHER_TABS_JOB} finished: ${plural(made, 'image')}${deleted.replace(/,$/, '')}.`);
    } else {
      // One of the jobs a press split by size (Swap tags, Mass refine) or by
      // run (More seeds): which one, since its count is that job's, not the
      // press's.
      const split = submitted?.split;
      const part = split && split.includes(jobId)
        ? ` Job ${split.indexOf(jobId) + 1} of ${split.length} of ${data.swap != null ? 'this swap' : data.refine_batch != null ? 'this batch refine' : data.more_seeds != null ? 'these more seeds' : 'this submission'}${data.width && data.height ? `, at ${data.width}×${data.height}` : ''}.`
        : '';
      line.push(`Finished ${plural(made, 'image')}${deleted} in ${took}${seeds}.${part}${distinct}`);
      if (plain) {
        line.push(noteButton('Use seed', () => useSeed(images[0].seed), `Put ${images[0].seed} in the Seed box`));
      }
      // More of what it made, the seeds following on: a plain job's, a More
      // seeds job's, and a single refine, inpaint or outpaint pass's - whose
      // strength, mask or spec its batch keeps - wider than `seeded`, which
      // is where the seeds are the news. Not a grid's (its run's below), a
      // swap's or a batch refine's, whose images are many batches of
      // another's. Held where every image has gone since, or the checkpoint
      // is another; the batches are the server's.
      const pass = !data.grid && data.refine != null && data.refine_batch == null && data.swap == null;
      if ((seeded || pass) && images.length) line.push(finishMoreSeeds(images));
    }
  }
  if (data.skipped) line.push(intentionalSkippedCells(data));
  // A grid's sheet is its payoff, and the way to it was the Runs picker or a
  // card's ⊞. Offered, never opened: a night ending under a user browsing
  // the gallery must not take the gallery away.
  if (data.grid && images.length && !sheet) {
    line.push(noteButton('Show sheet', () => (data.run_id ? showRun(data.run_id) : showJob(jobId)),
      'Lay this run out as a contact sheet'));
  }
  // A finished grid's next step: more seeds of every cell of its run.
  if (data.grid && data.status === 'done' && images.length && data.run_id && !foreign) {
    line.push(moreSeedsRunButton(data.run_id, { model: images[0].model_id, className: 'ghost tiny' }));
  }
  // The server kept the complete run, including Repeat, comparisons and
  // cells this tab never saw. Missing counts and holds never come from the
  // current composer or from this tab's old request payload.
  if (data.run_id && !foreign) line.push(finishRunButton(data.run_id, { className: 'ghost tiny', slot: els.error }));
  const executionCaption = executionDiagnosticCaption(data.execution_diagnostics);
  if (executionCaption) line.push(` ${executionCaption.short}.`);
  // Another tab's job keeps out of this page's finish line: that line is
  // this page's own last job, Use seed and all, and the foreign job's end
  // is said once, on the hidden line, as the title is given back below.
  // The line the foreign job's stream marked as a previous job's is this
  // page's latest again.
  if (foreign) {
    finishTakesFocus = false;
    finishStale = staleBefore;
    syncFinishLead();
  } else {
    setFinish(line);
    els.finishNote.title = executionCaption?.full ?? '';
    if (executionCaption) els.finishNote.setAttribute('aria-description', executionCaption.full);
  }

  // A cancel this page did not ask for - pressed in another tab or window,
  // which watched the same job - was reported as this page's own.
  const cancelledElsewhere = !cancelledHere.has(jobId);
  cancelledHere.delete(jobId);
  const whose = foreign ? `${ANOTHER_TABS_JOB}: ` : '';
  if (data.status === 'error') {
    if (!atLoad) showOutcome(`${whose}${data.error}`);
  } else if (data.status === 'cancelled') {
    // A stop at the time budget is the plan working, not a cancel: say
    // which, and that Finish run checks the rest. `run` is the cells
    // finished there - the budget is checked before a cell starts. A stop
    // at the server's shutdown is no one's cancel either.
    const kept = `kept ${plural(images.length, 'image')}`;
    const by = cancelledElsewhere ? ' from another tab or window' : '';
    showOutcome(whose + (data.stopped_by === 'budget'
      ? `Time budget reached after ${data.run}/${data.total_runs} cells — ${kept}. Finish run checks the rest.`
      : data.stopped_by === 'shutdown'
        ? `Stopped: the server is shutting down — ${images.length ? kept : 'nothing had finished'}.`
      : data.waiting_on === 'load' ? `Cancelled${by} — the weights keep loading, so the next Generate starts sooner.`
      : images.length ? `Cancelled${by} — ${kept}.` : `Cancelled${by} before anything finished.`), true);
  } else if (data.error) {
    // Done, but not every cell: the grid carried on past a failure and says
    // which, rather than dropping the rest of the night.
    showOutcome(`${whose}${skippedCells(data)}${data.skipped ? ` ${intentionalSkippedCells(data)}` : ''}`);
  } else if (data.skipped) {
    showOutcome(`${whose}${intentionalSkippedCells(data)}`, true);
  }
  // Said once to a screen reader where the panel's alert line has not said
  // the outcome already: the finish line's sentence, without its buttons. A
  // job that ended at the load has its reason there only where the viewer
  // holds images (syncLoadFailure); over the load card, which is no live
  // region, its "Not made" sentence is said here.
  const inAlert = data.status === 'cancelled'
    || (atLoad ? !!els.viewer.querySelector('img, figure, .tomb') : data.status === 'error' || !!data.error || !!data.skipped);
  if (!inAlert) {
    sayJob(line.filter((node) => node.nodeName !== 'BUTTON')
      .map((node) => (typeof node === 'string' ? node : node.textContent)).join('').trim());
  }
  // Another tab's job holds no title: the tab's own last result is what
  // its title was about, and gets it back where it has not been seen yet -
  // the foreign job's progress wrote over it (setBusy cleared it again).
  if (foreign) {
    if (titleHeld !== null) setTitle(titleHeld);
  } else {
    // Done with nothing made - every image skipped - is no ✓: "✓ Finished 0
    // images" read as a success in the tab strip.
    holdTitle(data.status === 'done' ? (made ? `✓ Finished ${plural(made, 'image')}` : 'Nothing made')
      : data.stopped_by === 'budget' ? 'Time budget reached'
      : data.stopped_by === 'shutdown' ? 'Server stopped'
      : data.status === 'cancelled' ? (cancelledElsewhere ? 'Cancelled elsewhere' : 'Cancelled')
      : atLoad ? 'Load failed' : 'Stopped');
    notifyJobDone(jobId, data, { atLoad, cancelledElsewhere });
  }
  // The status says how the load went, at once rather than at the next poll.
  // Said again, where it goes to the panel's line: the press that queued
  // this job cleared that line, and a detail already posted once was
  // otherwise never posted again (syncLoadFailure).
  if (atLoad) shownLoadDetail = null;
  void refreshStatus();
  invalidateFinishRuns();
  loadRuns();
  // A comparison opens this submission alone. Run ids name rendered
  // cells, so reversing original/yours on the same seeds shares a run
  // but must not mix the two meanings of either version label.
  // Another tab's images owed since this job took over (setElsewhereBusy)
  // come in with this job's, or alone where it made none.
  const owed = elsewhereCatchUp;
  elsewhereCatchUp = false;
  // Under a sheet or a filter they may join no strip: the ladders ask anyway.
  if (owed) noteUnseenImages();
  if (sheet && images.length) { showJob(jobId, { fresh: true, asker: submitted?.asker ?? null }); }
  else if (images.length || owed) {
    void Promise.resolve(prependFinished(made)).then((coming) => noteOutsideView(images, coming));
  }
}

// The finish line's sentence for a job that ended at the load, by where the
// load stands now: failed, it points at the reason - the viewer's card with
// Retry load, or the panel's line where the viewer holds images; loading
// again, or loaded since, that card and line have gone, and it said "why,
// and Retry load, are in the viewer" over a ready topbar (syncLoadFailure);
// so it did after a restart of the server, which starts unloaded.
function notMadeText(lead, state) {
  const now = state === 'error'
    ? `- ${els.viewer.querySelector('img, figure, .tomb') ? 'why is said below' : 'why, and Retry load, are in the viewer'}`
    : state === 'loading' ? 'then; it is loading now'
    : state === 'unloaded' ? 'then; it is not loaded now'
    : 'then; it has loaded since';
  return `${lead}: the model could not load ${now}.`;
}

// A finished job's images the gallery's filter leaves out - a comparison's
// or a sweep's sheet still scoped to its job:, a search, Starred, the
// recycle bin shown in the gallery's place (bin.js) - said
// under the gallery's head with a way to them: Generate after a comparison
// left the 12 images of its sheet on screen, the count unchanged, as though
// nothing had been saved. Counted across jobs while the filter is the same
// (`outsideView`); the gallery's line clears with the next reset.
let outsideView = null;
function viewFilter() {
  return [els.search.value.trim(), els.favOnly.checked, els.allModels.checked, sheetMode, binMode].join('|');
}

// `coming`: the ids prependFinished's page holds, whose cards may still be
// waiting for the pointer to leave the gallery's column (joinFinished) - in
// this view all the same, and said at once.
function noteOutsideView(images, coming) {
  if (!binMode && !els.search.value.trim() && !els.favOnly.checked) { outsideView = null; return; }
  const have = new Set([...galleryRecords.map((r) => r.id), ...(Array.isArray(coming) ? coming : [])]);
  const missing = images.filter((r) => !have.has(r.id)).map((r) => r.id);
  if (!missing.length) return;
  if (outsideView?.filter !== viewFilter()) outsideView = { filter: viewFilter(), ids: new Set() };
  missing.forEach((id) => outsideView.ids.add(id));
  const n = outsideView.ids.size;
  showError(`${plural(n, 'new image')} outside this view`, true, { slot: els.galleryError, mirror: false });
  const show = document.createElement('button');
  show.type = 'button';
  show.className = 'ghost tiny';
  show.textContent = 'Show all';
  show.title = binMode ? 'Leave the recycle bin, clear the search and Starred, to see every image'
    : 'Clear the search and Starred, and close the sheet, to see every image';
  show.onclick = showAllImages;
  els.galleryError.append(' · ', show);
}

// The gallery unfiltered: the search box emptied, Starred off, the sheet
// closed, the recycle bin left - the escape the scope note offers.
function showAllImages() {
  outsideView = null;
  leaveBin();
  els.search.value = '';
  els.favOnly.checked = false;
  sheetReturn = null;
  if (sheetMode) setSheet(false);
  else resetGallery();
  loadRuns();
}

// Intentional skips are missing work even when the job ended without an
// error. Name their seeds and keep Finish available for the missing cells.
function intentionalSkippedCells(data) {
  const count = data.skipped ?? 0;
  const unit = unitWord(data);
  const units = count === 1 ? unit : unit === 'batch' ? 'batches' : `${unit}s`;
  const cells = data.skipped_cells ?? [];
  const named = cells.slice(0, 3).map((cell) => [cell.label, `seed ${cell.seed}`].filter(Boolean).join(', '));
  const more = cells.length > 3 ? `; and ${cells.length - 3} more` : '';
  return `${count} ${units} skipped${named.length ? `: ${named.join('; ')}${more}` : ''}. Finish run checks the missing work.`;
}

// Failure-skipped cells retain their error and the existing retry wording.
function skippedCells(data) {
  const cells = data.failed_cells ?? [];
  if (!cells.length) return data.error;
  const named = cells.slice(0, 3).map((c) => [c.label, `seed ${c.seed}`].filter(Boolean).join(', '));
  const more = cells.length > 3 ? `; and ${cells.length - 3} more` : '';
  const last = cells.at(-1).error;
  const unit = unitWord(data);
  const units = data.total_runs === 1 ? unit : unit === 'batch' ? 'batches' : `${unit}s`;
  return `${cells.length} of ${data.total_runs} ${units} ${cells.length === 1 ? 'was' : 'were'} skipped: `
    + `${named.join('; ')}${more} (${cells.length === 1 ? '' : 'last: '}${last})`;
}

// The finish line's More seeds…: the batches of the images the job made, as
// they are at the press - those deleted since left out, and held with the
// reason once none is left (moreSeedsToolHold).
function finishMoreSeeds(images) {
  const ids = images.map((r) => r.id);
  const button = noteButton('More seeds…', null, '');
  button.setAttribute('aria-haspopup', 'dialog');
  syncMoreSeedsTool(button, {
    model: images[0].model_id,
    ids,
    title: `More images of ${ids.length === 1 ? 'this image\'s batch' : `the batches of these ${ids.length} images`}, the seeds following on - each at its own settings and Batch`,
  });
  button.onclick = () => {
    const live = ids.filter((id) => !isGone(id));
    pressMoreSeedsTool(button, {
      scope: { ids: live }, n: live.length,
      line: `From the ${live.length === 1 ? 'image' : `${live.length} images`} this job made.`,
    });
  };
  return button;
}

function noteButton(text, onClick, title) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'ghost tiny';
  button.textContent = text;
  button.title = title;
  button.onclick = onClick;
  return button;
}

// The finished image's seed into the Seed box, and nothing else: Reuse also
// rewrites the prompt and the size, and "same seed, another prompt" is the
// loop once something good turns up. The run hint then says it is fixed.
function useSeed(seed) {
  els.seed.value = seed;
  pageSeed = els.seed.value;
  syncRuns();
  schedulePreview();
  persistPanel();
}

// A finished job's images at the head of the gallery the page holds, under
// the same filters, rather than the gallery reset: that scrolled a user
// browsing old images during the wait back to the top, and dropped every page
// they had scrolled in, at every job's end. Asked once any page in flight has
// landed, and dropped if the filters changed meanwhile - the reset that
// changed them fetches the new images itself. One page reaches back past the
// job (`made` plus a page, the sheet's ceiling at most); where it does not
// join up with what the page holds, or the gallery is empty, the gallery is
// reset as before. Never in sheet mode, whose sheet lays itself out from the
// whole run. The paging anchor - the last record - is untouched. The head's
// count is drawn as the page lands and the cards by joinFinished, which can
// wait for the pointer; answers the ids that page holds (noteOutsideView).
// `now` puts them in without that wait: Delete unstarred brings the grid up
// to what it would sweep before asking. Its press was on the head's own
// button (or a key), not on the cards, and its modal question follows
// within one round trip, so no card moves under a pointer about to press it.
// Nothing in the recycle bin, which a job's images never join: the note
// says they are outside this view (noteOutsideView).
function prependFinished(made, { now = false, upTo = null } = {}) {
  if (binMode) return Promise.resolve(undefined);
  // A catch-up's images, up to the status's newest id (`upTo`), that the
  // grid's own read has already (galleryReadHead): nothing to fetch, and no
  // reset. Under load the status that sees another tab's images can land
  // after the gallery was read again over them - a script's jobs, then
  // Delete unstarred's reset - and the reset an empty grid takes cleared the
  // line its caller had just written ("Moved 2 images", smoke-delete).
  const covered = () => upTo !== null && galleryReadHead !== null && galleryReadHead >= upTo;
  if (sheetMode) return resetGallery();
  if (!galleryRecords.length && !galleryPending) return covered() ? Promise.resolve(undefined) : resetGallery();
  const epoch = galleryEpoch;
  return (async () => {
    // An empty grid with a page on its way is a reset already asking (Delete
    // unstarred's 409, a filter): that page lands first, then this weighs
    // what it read - a second reset there took "The selection changed" away.
    // A reset made while a page was out chains its own after it.
    while (galleryPending && epoch === galleryEpoch) await galleryPending;
    if (epoch !== galleryEpoch || sheetMode || binMode || covered()) return;
    if (!galleryRecords.length) return resetGallery();
    const params = new URLSearchParams({ limit: Math.min(made + PAGE, sheetPage) });
    if (els.search.value.trim()) params.set('search', els.search.value.trim());
    if (els.favOnly.checked) params.set('favorites', 'true');
    if (els.allModels.checked) params.set('all_models', 'true');
    let page;
    try {
      page = await api(`/api/gallery?${params}`);
    } catch (err) {
      // No server, no gallery to reload: what the page holds stays, and the
      // poll's first answer asks again (refreshStatus).
      return err.offline ? undefined : resetGallery();
    }
    if (epoch !== galleryEpoch || sheetMode) return;
    if (!joinsUp(page)) {
      // More came in than one page reaches back over: the gallery starts
      // again, but not under a pointer resting on its cards, as
      // joinFinished's wait (an import in another tab).
      if (!now) await galleryPointerGone();
      return epoch === galleryEpoch && !sheetMode && !binMode ? resetGallery() : undefined;
    }
    // What the head says, at once: it is above the cards, and with the
    // gallery beside the viewer above the column they scroll in. The read
    // head rises now, before joinFinished's pointer wait, and stays
    // exactly-once: a catch-up it covers skips, as that pending joinFinished
    // adds these cards; where the page no longer joins up by then, its
    // resetGallery nulls galleryReadHead and the gallery is read afresh.
    if (Number.isInteger(page.newest_image_id)) galleryReadHead = Math.max(galleryReadHead ?? 0, page.newest_image_id);
    total = page.total;
    els.count.textContent = total ? `(${total})` : '';
    showOtherModels(page.other_models);
    galleryOthers = page.other_models || 0;
    syncRerender();
    syncDeleteUnstarred();
    if (els.lightbox.open && lbList === galleryRecords) renderLightbox();
    void joinFinished(page, epoch, now);
    // For noteOutsideView: the images this page holds are the gallery's, in
    // view or about to be.
    return page.items.map((r) => r.id);
  })();
}

// Whether a page of the gallery's newest joins up with the cards it holds.
function joinsUp(page) {
  const have = new Set(galleryRecords.map((r) => r.id));
  return galleryRecords.length > 0 && page.items.some((r) => have.has(r.id));
}

// A finished job's cards, from the page prependFinished fetched, put in ahead
// of the gallery's. A pointer resting on the gallery's cards - beside the
// viewer or under it - keeps them where they are until it leaves
// (galleryPointerGone): one image ahead of them moves every card a place
// along the rows, which no scroll can take back. Only where there is
// something to put in.
async function joinFinished(page, epoch, now = false) {
  const newer = () => page.items.some((r) => r.id > galleryRecords[0].id && !galleryRecords.some((g) => g.id === r.id));
  if (!now && galleryRecords.length && newer()) await galleryPointerGone();
  if (epoch !== galleryEpoch || sheetMode) return;
  // Deleted from meanwhile, the gallery may no longer meet this page.
  if (!joinsUp(page)) { void resetGallery(); return; }
  const have = new Set(galleryRecords.map((r) => r.id));
  const head = galleryRecords[0];
  const fresh = page.items.filter((r) => r.id > head.id && !have.has(r.id));
  // In place: every card's lightbox walks this array. One open on it keeps
  // its image as the head grows.
  galleryRecords.unshift(...fresh);
  if (lbList === galleryRecords && fresh.length) lbIndex += fresh.length;
  // A lightbox open meanwhile: new CFG values for its ladders (lightbox-cfg.js).
  if (fresh.length) noteLadderImages(fresh);
  // With the day's lines as loadGallery draws them - over the first card
  // where its day is not today's, between two new cards, and between the
  // last of them and the old head (a job that ends after midnight under a
  // page loaded the day before). The old head is the grid's first card as
  // the page shows it, a tombstone included - its record has gone from
  // galleryRecords, and the line worked out from the records stood a
  // second time over it - unless the grid opens with a line already, which
  // is the old head's.
  const top = els.gallery.firstElementChild;
  const oldHead = top?.matches('.gallery-break') ? null : top;
  const firstLive = els.gallery.querySelector('.card[data-id]');
  const nodes = [];
  fresh.forEach((rec, i) => {
    if (i === 0 && !isToday(rec)) nodes.push(dayBreak(rec));
    nodes.push(card(rec));
    const next = fresh[i + 1];
    if (next ? localDay(next) !== localDay(rec) : oldHead && cardDay(oldHead) !== localDay(rec)) {
      nodes.push(dayBreak(next ?? { created_at: oldHead.dataset.made }));
    }
  });
  holdGalleryView(() => els.gallery.prepend(...nodes));
  // The grid keeps one way in from the keyboard (gallery.js): the newest
  // image, where the stop was the grid's default or the old head with focus
  // elsewhere; a card the user chose further down keeps it.
  const current = currentCard();
  if (fresh.length && (!galleryChosen || (current === firstLive && !els.gallery.contains(document.activeElement)))) {
    makeCurrent(els.gallery.querySelector('.card[data-id]'));
  } else {
    keepCurrentCard();
  }
  syncGalleryEmpty();
  if (els.lightbox.open && lbList === galleryRecords) renderLightbox();
}

// A job's first image or frame landing in a viewer out of sight - above the
// window, with a user browsing the gallery during the wait, or below it, in
// one column, where the viewer sits under the whole panel and a Generate
// pressed at the page's top put the image 150-800 px under the window's foot
// - is said with a press that goes to it: "New image ↑" or "↓". Never a
// scroll of the page's own, which would take the gallery from under the
// pointer. In one column the button is a pill fixed under the topbar
// (styles/generation.css): in the static panel it was as far out of view as the viewer.
// Gone once the viewer is in view, however it got there; its arrow follows a
// scroll that jumps past the viewer without showing it (End, Home).
let viewerWatch = null;
function viewerOutOfSight() {
  const top = document.querySelector('.topbar').getBoundingClientRect().bottom;
  const box = els.viewer.getBoundingClientRect();
  return box.bottom < top + 48 ? 'above' : box.top > innerHeight - 48 ? 'below' : null;
}

const NEW_IMAGE_WORDS = {
  above: ['New image ↑', 'New image: scroll up to it in the viewer'],
  below: ['New image ↓', 'New image: scroll down to it in the viewer'],
};
function syncNewImage() {
  if (els.newImage.hidden) return;
  const where = viewerOutOfSight();
  if (!where) { withdrawNewImage(newImageTarget()); return; }
  const [text, label] = NEW_IMAGE_WORDS[where];
  if (els.newImage.textContent === text) return;
  els.newImage.textContent = text;
  els.newImage.setAttribute('aria-label', label);
}

function offerNewImage() {
  if (!viewerOutOfSight()) return;
  els.newImage.hidden = false;
  syncNewImage();
  if (!viewerWatch) {
    viewerWatch = new IntersectionObserver(syncNewImage, { threshold: [0, 0.1, 0.25, 0.5, 1] });
    window.addEventListener('scroll', syncNewImage, { passive: true });
    const panel = document.querySelector('.panel');
    panel.addEventListener('pointerdown', newImageBackToPanel, true);
    panel.addEventListener('input', newImageBackToPanel, true);
  }
  viewerWatch.observe(els.viewer);
}

// The pill (one column) lies over whatever scrolls under the topbar - the
// prompt's first line, Check - once the page has moved at all, and a press
// meant for that went to the viewer. While the job runs it stays: the image
// is still coming. Once the job has ended, the user's own press in the panel
// or typing in one of its boxes is a return to the panel, and withdraws it;
// the finish line still names the result. Not a focus arriving: the finish
// line takes focus itself, and a Tab towards the pill passes through the
// panel. In two columns the button is in the panel's flow, over nothing.
function newImageBackToPanel(event) {
  if (busy || els.newImage.hidden || els.newImage.contains(event.target)) return;
  if (getComputedStyle(els.newImage).position !== 'fixed') return;
  withdrawNewImage();
}

// The button goes - the viewer in view, a new job, or a job that ended with
// nothing of it in the strip - and focus on it goes with it to `to`, else
// Generate: hidden with the focus, it fell to <body>.
function withdrawNewImage(to = null) {
  if (els.newImage.hidden) return;
  if (document.activeElement === els.newImage) focusTile(to ?? els.generate);
  els.newImage.hidden = true;
}

// A job whose stream gave up with nothing of it in the strip: its first frame
// had cleared the strip, the live preview went with the stream, and the
// viewer was an empty box with the only words red in the panel - still empty
// once the server was back. A card says it where the work is, as a failed
// load's does (status.js), with the same request a press away where this
// page sent it; its refusal - the server still gone - is said in the card.
// The next job's first frame clears it; an answering server rewords it
// (`lostRunBack`).
function showLostRun({ heading, text, run }, submitted) {
  const card = document.createElement('div');
  card.className = 'lost-run';
  card.setAttribute('role', 'group');
  card.setAttribute('aria-labelledby', 'lost-run-head');
  const head = document.createElement('h2');
  head.id = 'lost-run-head';
  head.textContent = heading;
  const line = document.createElement('p');
  line.textContent = text;
  card.append(head, line);
  const runId = run ?? submitted?.run_id;
  if (runId) {
    card.dataset.run = runId;
    const refused = document.createElement('p');
    refused.className = 'error';
    refused.setAttribute('role', 'alert');
    refused.hidden = true;
    const again = finishRunButton(runId, { className: 'primary', slot: refused });
    card.append(again, refused);
  }
  els.viewer.replaceChildren(card);
  els.viewer.classList.remove('multi', 'many', 'fitted');
  els.viewer.style.minHeight = els.viewer.style.maxHeight = '';
  els.viewerNoteText.replaceChildren();
  syncViewerMoreSeeds();
}

// The server answers again: the card says so rather than that it is gone.
function lostRunBack(restarted, s) {
  const card = els.viewer.querySelector(':scope > .lost-run');
  if (!card) return;
  const at = typeof s.started_at === 'number'
    ? new Date(s.started_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
  card.querySelector('h2').textContent = restarted ? `The server restarted${at ? ` at ${at}` : ''}` : 'The server is back';
  card.querySelector('p').textContent = card.dataset.run
    ? 'Finish run checks the saved cells of this run. Your composer stays as it is.'
    : "This job has no retained run id here. Find it in Runs; older runs use the panel's Resume tick.";
  const refused = card.querySelector('.error');
  if (refused) { refused.hidden = true; refused.textContent = ''; }
  invalidateFinishRuns();
}

// What New image goes to: the newest image; before any has landed - the
// first frame is what offered it - the live preview's tile, or a pass's
// stand-in, which streamJob hands on to the image that replaces it; where
// the viewer holds neither (an image deleted elsewhere), its held line.
// Null where the viewer is empty: a stream that reconnects.
function newImageTarget() {
  const images = els.viewer.querySelectorAll('img[data-id]');
  return images[images.length - 1]
    ?? els.viewer.querySelector(':scope > figure.live, :scope > figure.pending')
    ?? els.viewer.querySelector(':scope > .placeholder.held:not([hidden]), :scope > .tomb');
}

// Focus on an image or a button, or on a tile, a line or the viewer that
// takes it only for this (tabindex -1). The viewer's own is let go with the
// focus: kept, a click on its empty space would focus it.
function focusTile(el) {
  if (el === els.viewer) {
    el.tabIndex = -1;
    el.addEventListener('blur', () => el.removeAttribute('tabindex'), { once: true });
  } else if (!el.matches('img, button')) {
    el.tabIndex = -1;
  }
  el.focus({ preventScroll: true });
}

// To the newest image, focused - Enter opens it - since the button goes the
// moment the viewer is in view: focus left on it fell to <body> as the
// pressed button hid. An empty viewer takes the focus itself.
function goToNewImage() {
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  els.viewer.scrollIntoView({ block: 'start', behavior: reduce ? 'auto' : 'smooth' });
  focusTile(newImageTarget() ?? els.viewer);
}

// A job outlives the page that started it: the id lives on the server, so a
// reload - or a stream that gave up - can find its way back to a running sweep
// instead of leaving it unreachable until the process is killed.
async function resumeActiveJob() {
  if (busy || jobListing) return;
  // No answer is a server not up yet: nothing to say.
  const jobId = await adoptListedJob('reattaching…');
  if (jobId) { await jobStreams.get(jobId); return; }
  // Nothing live: say what the last one did instead, if the server still
  // knows - not where a Generate pressed meanwhile has taken the page.
  if (jobId === null && !busy) announceLastRun();
}
