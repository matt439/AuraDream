// Local Diffusion's page script, last of eleven: the event wiring and the boot.
// Everything called here is declared in the files index.html loads before
// it; state.js has the load-order contract.
'use strict';

// --------------------------------------------------------------------- wire

let searchTimer;
els.generate.onclick = generate;
els.cancel.onclick = cancelJob;
els.pause.onclick = togglePause;
els.cancelQueued.onclick = cancelQueued;
els.dice.onclick = () => { els.seed.value = ''; syncRuns(); schedulePreview(); persistPanel(); };
els.reset.onclick = resetPanel;
els.orientation.onclick = swapOrientation;

// The Examples menu is built on open from the guide's current list, so a
// status that lands late (behind a load) is not stale by the first click.
function showExamples(open) {
  if (open && !els.tagsMenu.hidden) showTagHelper(false);
  els.examplesBtn.setAttribute('aria-expanded', String(open));
  els.examplesMenu.hidden = !open;
  if (!open) return;
  els.examplesMenu.replaceChildren();
  if (!examples.length) {
    const li = document.createElement('li');
    li.className = 'empty';
    li.textContent = 'No examples for this checkpoint yet.';
    els.examplesMenu.appendChild(li);
    return;
  }
  examples.forEach((text) => {
    const li = document.createElement('li');
    li.setAttribute('role', 'menuitem');
    li.tabIndex = 0;
    const span = document.createElement('span');
    span.className = 'text';
    span.textContent = text;
    li.appendChild(span);
    li.title = text;
    const pick = () => {
      els.prompt.value = text;
      promptReplaced();
      showExamples(false);
      // Assigning .value fires no `input`: the preview and the remembered
      // panel are told the way `choose` tells them.
      schedulePreview();
      scheduleTagValidation(0);
      persistPanel();
      els.prompt.focus();
    };
    li.onclick = pick;
    li.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } };
    els.examplesMenu.appendChild(li);
  });
}

async function expandPrompt() {
  if (expandingPrompt) return;
  if (busy) { showError('Wait for the image queue to finish before expanding a prompt.'); return; }
  if (critiquing) { showError('Wait for the critic to finish.'); return; }
  const source = els.prompt.value;
  const prompt = source.trim();
  if (!prompt) { showError('Enter a prompt to expand first.'); els.prompt.focus(); return; }
  clearPromptTranslation();
  expandingPrompt = true;
  syncPromptExpansion();
  showError('Expanding with ERNIE, then translating locally…', true);
  try {
    const body = await api('/api/prompt/expand', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, width: +els.width.value, height: +els.height.value }),
    });
    // A multi-GB first load can take long enough for the user to continue
    // writing. Never replace work that happened after the click.
    if (els.prompt.value !== source) {
      showError('The prompt changed while expansion was running, so the result was not applied.', true);
      return;
    }
    els.prompt.value = body.expanded_prompt;
    syncAnalysisStale();
    if (body.translated_prompt) showPromptTranslation(body.expanded_prompt, body.translated_prompt);
    schedulePreview();
    persistPanel();
    els.prompt.focus();
    els.prompt.setSelectionRange(els.prompt.value.length, els.prompt.value.length);
    if (body.translation_detail) {
      showError(`Prompt expanded, but its English translation failed: ${body.translation_detail}`, true);
    } else {
      showError('Prompt expanded. Review the Chinese prompt and its English translation before generating.', true);
    }
  } catch (err) {
    showError(err.message);
  } finally {
    expandingPrompt = false;
    syncPromptExpansion();
  }
}
els.examplesBtn.onclick = () => showExamples(els.examplesMenu.hidden);
els.tagsBtn.onclick = () => showTagHelper(els.tagsMenu.hidden);
els.expandPrompt.onclick = expandPrompt;
// The prompt check: Check and Check again run it on the prompt as it is now;
// the rest of its buttons are built with the results they act on.
els.analyze.onclick = () => { void checkFromButton(); };
els.analysisAgain.onclick = () => { void checkPrompt(); };
els.analysisClose.onclick = closeAnalysis;
els.criticAsk.onclick = askCritic;
els.compareSeeds.onchange = () => scheduleComparePreview(0);
els.compareBatch.onchange = () => scheduleComparePreview(0);
els.compareRun.onclick = runComparison;
els.compareOwn.addEventListener('input', compareOwnChanged);
els.compareOwnFill.onclick = compareOwnFromPrompt;
els.tagValidation.addEventListener('contextmenu', (event) => {
  // Keep the ordinary text-editing menu inside the inline input. Everywhere
  // else in the validation tray, including the flex gaps, is an insert point.
  if (event.target.closest('.tag-editor')) return;
  event.preventDefault();
  openTagInsert(tagInsertionIndexAtPoint(event.clientX, event.clientY));
});
els.prompt.addEventListener('input', () => {
  if (translatedSource !== null && els.prompt.value !== translatedSource) clearPromptTranslation();
  syncAnalysisStale();
  syncTagSelection();
  scheduleTagSuggestions();
  scheduleTagValidation();
});
// The check read the negative prompt too; editing it dims the check as well.
els.negative.addEventListener('input', () => syncAnalysisStale());
els.prompt.addEventListener('click', () => scheduleTagSuggestions(0));
els.prompt.addEventListener('keyup', (event) => {
  if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) scheduleTagSuggestions(0);
});
els.prompt.addEventListener('keydown', (event) => {
  if (els.tagSuggestions.hidden || event.ctrlKey || event.metaKey) return;
  if (event.key === 'ArrowDown') {
    event.preventDefault();
    selectTagSuggestion(tagSuggestionIndex + 1);
  } else if (event.key === 'ArrowUp') {
    event.preventDefault();
    selectTagSuggestion(tagSuggestionIndex - 1);
  } else if ((event.key === 'Enter' || event.key === 'Tab') && tagSuggestionIndex >= 0) {
    event.preventDefault();
    applyTagSuggestion(tagSuggestions[tagSuggestionIndex]);
  } else if (event.key === 'Escape') {
    event.preventDefault();
    hideTagSuggestions();
  }
});
document.addEventListener('click', (e) => {
  if (!els.examplesMenu.hidden && !e.target.closest('#examples')) showExamples(false);
  if (!els.tagsMenu.hidden && !e.target.closest('#tag-helper')) showTagHelper(false);
  if (!e.target.closest('.prompt-editor')) hideTagSuggestions();
});
els.unload.onclick = async () => {
  try {
    await api('/api/model/unload', { method: 'POST' });
    showError('');
  } catch (err) {
    showError(err.message);  // 409 while a job is still running
  }
  refreshStatus();
};
els.more.onclick = () => loadGallery();
// Scrolling to the end of the gallery fetches the next page: the trigger is
// the button coming within a few hundred pixels of the viewport, so the page
// is usually there before the button is. Hidden - no older rows, or the
// sheet - it never intersects. `loadGallery` serialises a click racing this.
moreWatcher = new IntersectionObserver((entries) => {
  if (galleryMore && !sheetMode && entries.some((e) => e.isIntersecting)) loadGallery();
}, { rootMargin: '0px 0px 400px 0px' });
moreWatcher.observe(els.more);
els.gridRun.onclick = () => startJob('/api/grid', gridPayload());
// The preview reflects the panel as much as the axes: a blank axis is the
// control's own value, and both the size and the batch move the totals.
// The negative prompt is in the list because a `{name}` placeholder may live
// there - the server validates it - and Resume because the whole "N already
// done" half of the preview is what ticking it is for.
for (const el of [els.gridPrompt, els.gridSteps, els.gridCfg, els.gridMu, els.gridSeeds,
                  els.gridSeedList, els.gridBudget, els.prompt, els.negative, els.steps,
                  els.cfg, els.trueCfg, els.mu, els.batch, els.width, els.height, els.seed,
                  els.gridLoraScale, els.loraScale]) {
  el.addEventListener('input', schedulePreview);
}
// A listed seed axis makes the count and the base seed moot; say so in place.
els.gridSeedList.addEventListener('input', () => {
  els.gridSeeds.disabled = els.gridSeedList.value.trim() !== '';
});
els.load.onclick = async () => {
  try {
    await api('/api/model/load', { method: 'POST' });
    showError('');
  } catch (err) {
    showError(err.message);
  }
  refreshStatus();
};
els.gridResume.addEventListener('change', previewGrid);
els.gridSection.addEventListener('toggle', previewGrid);
els.width.oninput = els.height.oninput = syncSize;
// These run before the panel's delegated persist handler on the same event -
// the target's listener fires before its ancestors' - so the `owned` flags
// are current by the time the panel is written.
els.steps.oninput = () => own('steps');
els.mu.oninput = () => own('mu');
// The seed is not a sampling value and has no business freezing the profile.
els.seed.oninput = syncRuns;
els.batch.oninput = els.runs.oninput = syncRuns;
els.cfg.oninput = () => { own('cfg'); syncCfg(); };
// The weight boxes are dead without an adapter, so selecting one has to wake
// them - and clearing it has to stop a stale axis reaching the server.
els.lora.onchange = () => { syncConditioning(); schedulePreview(); };
els.trueCfg.oninput = () => { own('trueCfg'); syncCfg(); };
els.favOnly.onchange = resetGallery;
// The Runs picker is the gallery's table of contents, so it follows the scope.
els.allModels.onchange = () => { resetGallery(); loadRuns(); };
els.sheetBtn.onclick = () => setSheet(!sheetMode);
els.rerender.onclick = rerenderSelection;
els.deleteUnstarred.onclick = deleteUnstarred;
els.runList.onchange = () => { showRun(els.runList.value); els.runList.value = ''; };
els.lastRunShow.onclick = () => {
  const { run, job } = els.lastRun.dataset;
  if (run) showRun(run);
  else if (job) { els.search.value = `job:${job}`; setSheet(true); }
  dismissLastRun();
};
els.lastRunDismiss.onclick = dismissLastRun;
els.search.oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(resetGallery, 250); };
els.lbClose.onclick = closeLightbox;
els.lbPrev.onclick = () => stepLightbox(-1);
els.lbNext.onclick = () => stepLightbox(1);
// The just-generated image only exists in the viewer, so reuse has to be
// reachable from the lightbox as well as from a gallery card.
els.lbReuse.onclick = () => { const rec = lbList[lbIndex]; if (rec) { reuse(rec); closeLightbox(); } };
els.lbFav.onclick = () => { const rec = lbList[lbIndex]; if (rec) setFavorite(rec, !rec.favorite); };
els.lbDelete.onclick = () => { const rec = lbList[lbIndex]; if (rec) deleteImage(rec); };
els.lightbox.onclick = (e) => { if (e.target === els.lightbox) closeLightbox(); };

document.addEventListener('keydown', (e) => {
  // Ctrl/Cmd+Enter generates from anywhere, including inside the prompt box.
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { generate(); return; }
  if (e.key === 'Escape' && !els.examplesMenu.hidden) { showExamples(false); return; }
  if (e.key === 'Escape' && !els.tagsMenu.hidden) { showTagHelper(false); return; }
  if (!els.lightbox.open) return;
  // Arrows walk whichever strip the lightbox was opened from. Nothing else on
  // the page reads them while it is up, so the caret keys are free to take.
  if (e.key === 'Escape') { closeLightbox(); return; }
  if (e.key === 'ArrowLeft') { e.preventDefault(); stepLightbox(-1); return; }
  if (e.key === 'ArrowRight') { e.preventDefault(); stepLightbox(1); return; }
  // F, Delete and the zoom keys act on the image in view - unless a text field
  // still holds the caret, where they are typing.
  if (e.target.matches('input, textarea')) return;
  if (e.key === 'f' || e.key === 'F') els.lbFav.click();
  else if (e.key === 'Delete') els.lbDelete.click();
  else if (e.key === '+' || e.key === '=') { e.preventDefault(); zoomLightbox(ZOOM_STEP); }
  else if (e.key === '-' || e.key === '_') { e.preventDefault(); zoomLightbox(1 / ZOOM_STEP); }
  else if (e.key === '0') zoomFit();
  else if (e.key === '1') zoomOneToOne();
});

window.addEventListener('resize', placeTagValidation);
placeTagValidation();
buildPresets();
// Before the sync functions, which light the presets, the CFG note and the
// runs hint up from whatever is in the boxes.
restorePanel();
syncSize();
syncCfg();
syncRuns();
// Every keystroke and tick in the panel, plus the programmatic writes below
// - presets, Reuse, the dice, the seed written back after Run grid - which
// fire no input event.
document.querySelector('.panel').addEventListener('input', persistPanel);
document.querySelector('.panel').addEventListener('change', persistPanel);
els.gridSection.addEventListener('toggle', persistPanel);
refreshStatus();
loadGallery();
loadRuns();
resumeActiveJob();
setInterval(refreshStatus, 3000);
