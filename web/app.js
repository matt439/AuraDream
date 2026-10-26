// Local Diffusion's last page script: the event wiring and the boot.
// Everything called here is declared in the files index.html loads before
// it; state.js has the load-order contract.
'use strict';

// --------------------------------------------------------------------- wire

let searchTimer;
els.generate.onclick = generate;
els.cancel.onclick = cancelJob;
els.pause.onclick = togglePause;
els.cancelQueued.onclick = cancelQueued;
initQueueControls();
initSnippets();
els.notifyDone.onchange = () => { void toggleJobNotifications(); };
document.addEventListener('visibilitychange', () => syncJobNotifications());
els.newImage.onclick = goToNewImage;
els.dice.onclick = () => { els.seed.value = ''; syncRuns(); schedulePreview(); persistPanel(); };
els.reset.onclick = resetPanel;
els.orientation.onclick = swapOrientation;

// The menu's keyboard: ArrowDown and ArrowUp walk it (round the ends), Home
// and End jump; leaving it - Tab, a click elsewhere - closes it, rather than
// leaving it open over the prompt.
els.examplesBtn.addEventListener('keydown', (e) => {
  if (keyboardEventHeld(e) || keyboardCommandModifier(e) || e.shiftKey || e.key !== 'ArrowDown') return;
  e.preventDefault();
  if (els.examplesMenu.hidden) showExamples(true);
  els.examplesMenu.querySelector('[role="menuitem"]')?.focus();
});
els.examplesMenu.addEventListener('keydown', (e) => {
  if (keyboardEventHeld(e) || keyboardCommandModifier(e) || e.shiftKey) return;
  const items = [...els.examplesMenu.querySelectorAll('[role="menuitem"]')];
  const at = items.indexOf(document.activeElement);
  const next = { ArrowDown: at + 1, ArrowUp: at - 1, Home: 0, End: items.length - 1 }[e.key];
  if (next === undefined || !items.length) return;
  e.preventDefault();
  items[(next + items.length) % items.length].focus();
});
els.examplesBox.addEventListener('focusout', (e) => {
  if (!els.examplesMenu.hidden && !els.examplesBox.contains(e.relatedTarget)) showExamples(false);
});
els.promptNoteUndo.onclick = undoPromptWrite;
els.promptNoteDiscard.onclick = discardPromptNote;

// A click from the keyboard (detail 0) - Enter or Space on the button - takes
// focus into the menu on its first item, as ArrowDown does: its items are no
// Tab stops, and Tab from the button went on to Tags, closing it unread.
els.examplesBtn.onclick = (e) => {
  const open = els.examplesMenu.hidden;
  showExamples(open);
  if (open && e.detail === 0) els.examplesMenu.querySelector('[role="menuitem"]')?.focus();
};
// A click from the keyboard takes focus into the helper too.
els.tagsBtn.onclick = (e) => showTagHelper(els.tagsMenu.hidden, { focus: e.detail === 0 });
els.expandPrompt.onclick = expandPrompt;
els.promptExpandStop.onclick = stopExpandPrompt;
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
  if (event.target.closest('.tag-editor, .prompt-prose')) return;
  event.preventDefault();
  openTagInsert(tagInsertionIndexAtPoint(event.clientX, event.clientY));
});
// A gap takes a plain click too, and the pointer over one shows where the
// tag would go - the drop's bar, once a frame for the last position. Over the
// tray itself only, not a chip, the editor or + tag.
els.tagValidation.addEventListener('pointermove', (event) => {
  cancelAnimationFrame(tagHoverFrame);
  tagHoverFrame = 0;
  if (event.target !== els.tagValidation || draggedTagIndex !== null) {
    tagHoverPoint = null;
    clearTagDropMarkers();
    return;
  }
  const { clientX, clientY } = event;
  tagHoverPoint = { x: clientX, y: clientY };
  tagHoverFrame = requestAnimationFrame(() => { tagHoverFrame = 0; markTagGap(clientX, clientY); });
});
els.tagValidation.addEventListener('pointerleave', () => {
  cancelAnimationFrame(tagHoverFrame);
  tagHoverFrame = 0;
  tagHoverPoint = null;
  clearTagDropMarkers();
});
els.tagValidation.addEventListener('click', (event) => {
  if (event.target !== els.tagValidation) return;
  const index = tagInsertionIndexAtPoint(event.clientX, event.clientY, { gap: true });
  if (index === null) return;
  tagHoverPoint = null;
  clearTagDropMarkers();
  openTagInsert(index);
});
// Focus leaving the tray lets go of the chip it was on; a rebuild of the
// tray, which removes the focused chip, is not leaving it.
els.tagValidation.addEventListener('focusout', (event) => {
  if (tagRendering || els.tagValidation.contains(event.relatedTarget)) return;
  tagFocusIndex = null;
});
els.prompt.addEventListener('input', (event) => {
  if (translatedSource !== null && els.prompt.value !== translatedSource) clearPromptTranslation();
  syncAnalysisStale();
  noteTagTyping(event);
  syncTagSelection();
  scheduleTagSuggestions();
  scheduleTagValidation();
});
// The check read the negative prompt too; editing it dims the check as well.
els.negative.addEventListener('input', () => syncAnalysisStale());
// The caret moved: suggestions for the item it is in now, the item typed
// before it checked, and the Tags helper's note on where a tag goes.
els.prompt.addEventListener('click', () => { scheduleTagSuggestions(0); followTagCaret(); syncTagSelection(); });
els.prompt.addEventListener('keyup', (event) => {
  if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) scheduleTagSuggestions(0);
  if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) {
    followTagCaret();
    syncTagSelection();
  }
});
// Leaving the box ends the typing, and the list goes with it: Tab, with
// nothing selected, moves on, and must not leave suggestions open behind.
els.prompt.addEventListener('blur', () => { hideTagSuggestions(); endTagTyping(); });
els.prompt.addEventListener('keydown', (event) => {
  if (els.tagSuggestions.hidden || keyboardEventHeld(event) || keyboardCommandModifier(event) || event.shiftKey) return;
  if (event.key === 'ArrowDown') {
    event.preventDefault();
    selectTagSuggestion(tagSuggestionIndex + 1);
  } else if (event.key === 'ArrowUp') {
    event.preventDefault();
    selectTagSuggestion(tagSuggestionIndex - 1);
  } else if ((event.key === 'Enter' || event.key === 'Tab') && tagSuggestionIndex >= 0) {
    event.preventDefault();
    if (!event.repeat) applyTagSuggestion(tagSuggestions[tagSuggestionIndex]);
  } else if (event.key === 'Escape') {
    event.preventDefault();
    hideTagSuggestions();
  }
});
document.addEventListener('click', (e) => {
  if (!els.examplesMenu.hidden && !e.target.closest('#examples')) showExamples(false);
  if (!els.pickedMenu.hidden && !e.target.closest('#picked')) showPickedMenu(false);
  // The helper hangs under the prompt, which takes clicks to move its caret.
  if (!els.tagsMenu.hidden && !e.target.closest('#tag-helper, #tags-menu, #prompt')) showTagHelper(false);
  // Find tag's inserts go to the prompt's caret too, so a click there keeps
  // it, and its row in the suggestions opened it - a row the list has
  // dropped by the time the click reaches here, so no longer in the page.
  if (!els.tagFinderMenu.hidden && e.target.isConnected
      && !e.target.closest('#tag-finder, #prompt, #tag-suggestions')) showTagFinder(false);
  // The token line is in the editor (under the box) but no part of typing.
  if (!e.target.closest('.prompt-editor') || e.target === els.promptTokens) hideTagSuggestions();
});
els.unload.onclick = unloadModel;
els.more.onclick = () => loadGallery();
// One card at a time in the tab order, walked with the arrow keys (gallery.js).
els.gallery.addEventListener('keydown', galleryKey);
els.gallery.addEventListener('focusin', galleryFocus);
// A card's batch outlined while the mouse rests on it (gallery-batch.js).
els.gallery.addEventListener('pointerover', batchMatesOver);
els.gallery.addEventListener('pointerout', batchMatesOut);
// The selection's Select, Clear and Escape (gallery-select.js); a card's
// press picks through cardPress, which card() hands thumb, and a click
// between the cards empties it.
els.gallery.addEventListener('click', galleryBlankClick);
els.select.onclick = () => {
  if (els.select.getAttribute('aria-disabled') === 'true') { setSelectMode(true); return; }
  setSelectMode(!selectMode);
};
els.selectClear.onclick = clearSelection;
els.library.addEventListener('keydown', selectKey);
// The gallery beside the viewer, from 2000 px where ticked (gallery.js): the
// tick, the window crossing that width, and the tick as this browser left it.
els.besideTick.onchange = toggleBeside;
matchMedia(BESIDE_MEDIA).addEventListener('change', syncBeside);
// A finished job's images wait for the pointer to leave its column there.
els.libraryScroll.addEventListener('pointerenter', () => { galleryPointer = true; });
els.libraryScroll.addEventListener('pointerleave', galleryPointerLeft);
restoreBeside();
// Scrolling to the end of the gallery fetches the next page: the trigger is
// the button coming within a few hundred pixels of the viewport - or of the
// gallery's own column, beside the viewer - so the page is usually there
// before the button is. Hidden - no older rows, or the sheet - it never
// intersects. `loadGallery` serialises a click racing this.
if (!moreWatcher) armMore();
els.gridRun.onclick = runGrid;
// The preview reflects the panel as much as the axes: a blank axis is the
// control's own value, and both the size and the batch move the totals.
// The negative prompt is in the list because a `{name}` placeholder may live
// there - the server validates it - and Resume because the whole "N already
// done" half of the preview is what ticking it is for. Repeat (`runs`) moves
// nothing a grid runs, but the preview says so while it is above one.
for (const el of [els.gridPrompt, els.gridPromptMode, els.gridSize, els.gridSteps, els.gridCfg, els.gridMu, els.gridSeeds,
                  els.gridSeedList, els.gridBudget, els.prompt, els.negative, els.steps,
                  els.cfg, els.trueCfg, els.mu, els.batch, els.runs, els.width, els.height,
                  els.seed, els.gridLoraScale, els.loraScale]) {
  el.addEventListener('input', schedulePreview);
}
// The comparison refuses a version holding a variable the open grid declares
// (compareUnfilled): its preview says so as the box or the fold changes.
els.gridPrompt.addEventListener('input', () => scheduleComparePreview());
els.gridSection.addEventListener('toggle', () => scheduleComparePreview(0));
// A grid variable's chip names the values the fold declares, and asks about
// them (gridVariableTip): the tray reads the box again as it changes.
els.gridPrompt.addEventListener('input', () => { if (gridVariableChips()) scheduleTagValidation(); });
els.gridSection.addEventListener('toggle', () => { if (gridVariableChips()) scheduleTagValidation(0); });
// A listed seed axis makes the count and the base seed moot; say so in place.
els.gridSeedList.addEventListener('input', () => {
  els.gridSeeds.disabled = els.gridSeedList.value.trim() !== '';
});
els.load.onclick = loadModel;
restoreModelFavourites();
window.addEventListener('storage', receiveModelFavourites);
els.models.onclick = () => showModelPicker(els.modelPicker.hidden);
els.modelTable.addEventListener('change', (event) => {
  if (event.target.name === 'model-choice') modelPickerEdited('model_id');
});
// A row keeps its star and link tabbable while it holds the focus.
els.modelTable.addEventListener('focusout', () => setTimeout(syncModelTabStops, 0));
els.modelTable.addEventListener('click', (event) => {
  const star = event.target.closest('.model-star');
  if (star) toggleModelFavourite(star.closest('.model-row').dataset.model);
  // The gaps between a row's cells belong to the row, not its label: a press
  // there chooses the row too, as its hover promises.
  else if (event.target.matches('.model-row:not(.model-row-head)')) event.target.querySelector('input')?.click();
});
els.modelOffload.onchange = () => modelPickerEdited('offload', { catalogue: true });
els.modelLoadOnStartup.onchange = () => modelPickerEdited('load_on_startup');
els.modelQuantization.onchange = () => modelPickerEdited('quantization', { catalogue: true });
els.modelFormat.onchange = () => modelPickerEdited('format', { catalogue: true });
els.modelSwitch.onclick = () => switchModel();
els.modelSave.onclick = () => switchModel({ load: false });
els.modelRevert.onclick = revertModelPicker;
// The strip's pictures open in the lightbox over the picker, which stays open
// under it: a press or the focus in the lightbox (or a dialog over it) is not
// a move away.
els.modelShotsList.addEventListener('click', (event) => {
  const shot = event.target.closest('.model-shot');
  if (shot) openModelShot(shot);
});
els.modelShotsList.addEventListener('keydown', moveModelShotFocus);
els.modelShotsAll.onclick = showModelShotsGallery;
// The pictures' fit follows a resize while the picker is open.
window.addEventListener('resize', refitModelShots);
document.addEventListener('pointerdown', (event) => {
  if (els.lightbox.open) return;
  if (!els.modelPicker.hidden && !els.modelPicker.contains(event.target) && event.target !== els.models) {
    showModelPicker(false, { focus: false });
  }
});
els.modelPicker.addEventListener('focusout', () => {
  setTimeout(() => {
    if (els.lightbox.open) return;
    if (!els.modelPicker.contains(document.activeElement) && document.activeElement !== els.models) {
      showModelPicker(false, { focus: false });
    }
  }, 0);
});
els.gridResume.addEventListener('change', previewGrid);
els.gridSection.addEventListener('toggle', previewGrid);
// On the summary's click (Enter and Space on it are clicks too), not on the
// toggle event, which restorePanel and resetPanel set off as well; the fold
// is open by the next frame. And again once its preview lands (previewShown).
els.gridSection.querySelector('summary').addEventListener('click', () => {
  gridRevealPending = true;
  requestAnimationFrame(revealGridRun);
});
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
// The greyed negative box points at the box that wakes it.
els.negEnable.onclick = focusNegativeKnob;
els.favOnly.onchange = resetGallery;
// The Runs picker is the gallery's table of contents, so it follows the scope.
els.allModels.onchange = () => { resetGallery(); loadRuns(); };
els.sheetBtn.onclick = toggleSheet;
// How the cards are drawn (gallery.js): a press chooses; each pair is one
// tab stop whose arrows move focus along it, as a size strip's do.
els.cardView.addEventListener('click', cardViewClick);
for (const pair of els.cardView.querySelectorAll('[role="group"]')) pair.addEventListener('keydown', (e) => stepPreset(pair, e));
els.sheetGrid.addEventListener('keydown', (e) => { void sheetKey(e); });
// Keep one image and its tools in Tab's path, and remember Swipe CFG's cell.
els.sheetGrid.addEventListener('focusin', sheetFocus);
// The sheet's fit is measured against the window, so it follows a resize.
window.addEventListener('resize', refitSheet);
// The sheet's zoom and pan (sheet.js): Ctrl+wheel non-passive, so the page
// does not zoom with it; a drag's release no click, caught on the way down.
{
  const box = els.sheetGrid.parentElement;
  box.addEventListener('wheel', sheetWheel, { passive: false });
  box.addEventListener('pointerdown', sheetPointerDown);
  box.addEventListener('pointermove', sheetPointerMove);
  box.addEventListener('pointerup', sheetPointerUp);
  box.addEventListener('pointercancel', sheetPointerCancel);
  box.addEventListener('click', sheetClickAfterPan, true);
  box.addEventListener('dragstart', sheetDragStart);
}
els.rerender.onclick = rerenderSelection;
els.deleteUnstarred.onclick = deleteUnstarred;
// Re-render and Delete unstarred come and go once the pointer is off the
// head's controls - the footer row under them is not (showInHead,
// gallery-actions.js).
const galleryHead = els.library.querySelector('.gallery-head');
galleryHead.addEventListener('pointerover', (e) => pointHead(!e.target.closest('.gallery-footer')));
galleryHead.addEventListener('pointerleave', () => pointHead(false));
// Delete unstarred, held given way after the picks were emptied, waits for
// the pointer to leave the gallery as a whole (syncHeadForPicks).
els.library.addEventListener('pointerover', () => { libraryPointer = true; });
els.library.addEventListener('pointerleave', libraryPointerLeft);
initBundles();
initThenRefine();
// The right-click menu on an image (context-menu.js): the right button or
// the Menu key opens it, and a press elsewhere, a scroll or the window
// going closes it. Capture, so the lightbox's stage never sees the press
// that only closed the menu over it.
document.addEventListener('contextmenu', contextMenuEvent);
document.addEventListener('keydown', contextMenuKey, true);
document.addEventListener('pointerdown', contextMenuPointerDown, true);
document.addEventListener('scroll', contextMenuAway, true);
window.addEventListener('resize', () => closeContextMenu(false));
window.addEventListener('blur', () => closeContextMenu(false));
els.contextMenu.addEventListener('keydown', contextMenuKeydown);
els.contextMenu.addEventListener('click', (e) => chooseContextItem(e.target.closest('[role="menuitem"]')));
els.contextMenu.addEventListener('focusout', (e) => {
  if (!els.contextMenu.contains(e.relatedTarget) && e.relatedTarget) closeContextMenu(false);
});
// The Picked menu between Clear and Select (gallery-select.js): its keyboard
// is a menu's, and leaving it closes it.
els.pickedBtn.onclick = pickedPress;
els.picked.addEventListener('keydown', pickedKey);
els.pickedMenu.addEventListener('click', (e) => pickedChoose(e.target));
els.picked.addEventListener('focusout', (e) => {
  if (!els.pickedMenu.hidden && !els.picked.contains(e.relatedTarget)) showPickedMenu(false);
});
// Re-render takes the picked cards, and the head keeps its rows as they
// change (syncHeadForPicks, gallery-actions.js).
onSelectionChange(syncHeadForPicks);
// Swap tags in the gallery's footer row and its dialog (tag-swap.js).
setupSwap();
// Mass refine in the same row, and its dialog (refine-batch.js).
setupRefineBatch();
// More seeds after it, and its dialog (more-seeds.js).
setupMoreSeeds();
// Bin after them: the recycle bin, shown in the gallery's place (bin.js).
setupBin();
// The picker then names the run on screen: resetGallery follows the search box.
// Browsed from the keyboard, a run opens on Enter or as focus leaves (runs.js).
els.runList.onchange = runListChange;
els.runList.addEventListener('keydown', runListKey);
els.runList.addEventListener('blur', commitRunList);
els.lastRunShow.onclick = () => {
  const { run, job } = els.lastRun.dataset;
  if (run) showRun(run);
  else if (job) showJob(job);
  dismissLastRun();
};
els.lastRunMore.onclick = moreSeedsLastRun;
els.lastRunResume.onclick = resumeLastRun;
els.lastRunDismiss.onclick = dismissLastRun;
// A first run's empty viewer offers the checkpoint's examples.
els.viewerExamples.onclick = viewerExamples;
// A batch of two to four is laid out to fit the strip (viewer.js, fitViewer):
// again when its width moves, and with the window's height, which its 68vh is
// - as a long run's scrolling strip is let grow by its padding (settleStrip).
new ResizeObserver(viewerResized).observe(els.viewer);
window.addEventListener('resize', () => { fitViewer(); settleStrip(false); });
els.search.oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(resetGallery, 250); };
// The user's own closes - the cross, the backdrop, Escape below - hand focus
// to the image the walk ended on (dismissLightbox); every other close is
// closeLightbox alone.
els.lbClose.onclick = dismissLightbox;
els.lbPrev.onclick = () => stepLightbox(-1);
els.lbNext.onclick = () => stepLightbox(1);
// The just-generated image only exists in the viewer, so reuse has to be
// reachable from the lightbox as well as from a gallery card. Closed first:
// Reuse writes the prompt as typing would, which needs the focus a modal
// lightbox keeps to itself.
els.lbReuse.onclick = () => { const rec = lbList[lbIndex]; if (rec) { closeLightbox(); reuse(rec); } };
els.lbDownload.onclick = downloadLightboxImage;
// Refine likewise: openRefine closes the lightbox for its dialog, and a close
// that runs nothing gives it back (refineClosed). Where the checkpoint has no
// such pass the lightbox stays open, since the note saying so is mirrored
// into it.
els.lbRefine.onclick = () => { const rec = lbList[lbIndex]; if (rec && !refuseBinned(rec, 'refine')) openRefine(rec); };
// Inpaint likewise (inpaint.js): a close that runs nothing gives the lightbox
// back, and a held one says why in the lightbox's line.
els.lbInpaint.onclick = pressLightboxInpaint;
// And Outpaint (outpaint.js), the same way.
els.lbOutpaint.onclick = pressLightboxOutpaint;
els.lbFav.onclick = () => { const rec = lbList[lbIndex]; if (rec) setFavorite(rec, !rec.favorite); };
els.lbDelete.onclick = () => { const rec = lbList[lbIndex]; if (rec) deleteImage(rec); };
// Nor a held Enter's repeats on the focused button, which press it again for
// each - the image after the one just binned, and the next - as Del's would.
els.lbDelete.addEventListener('keydown', (e) => {
  if (e.repeat && (e.key === 'Enter' || e.key === ' ')) e.preventDefault();
});
els.lightbox.onclick = (e) => { if (e.target === els.lightbox) dismissLightbox(); };
// A pass against its source, or any image across CFG or steps
// (lightbox-cfg.js): each kind's button turns it on and off, or switches to
// its kind; a mode turns on the kind S would in that mode, or switches to it.
// Lower and Higher CFG step along CFG, Fewer and More steps along steps, and
// Against picks the partner.
els.lbCompareOn.onclick = () => { void toggleCompare(null, 'source'); };
els.lbCompareCfg.onclick = () => { void toggleCompare(null, pickedPair(lbList[lbIndex]) ? 'picked' : 'cfg'); };
els.lbSwipe.onclick = () => { void toggleCompare('swipe'); };
els.lbSideBySide.onclick = () => { void toggleCompare('side'); };
els.lbCompareSteps.onclick = () => { void toggleCompare(null, 'steps'); };
els.lbCfgLower.onclick = () => pressLadderStep('cfg', -1);
els.lbCfgHigher.onclick = () => pressLadderStep('cfg', 1);
els.lbStepsLower.onclick = () => pressLadderStep('steps', -1);
els.lbStepsHigher.onclick = () => pressLadderStep('steps', 1);
els.lbCompareAgainst.onchange = pickAgainst;
// Every close, the native Escape's included, resets the strip - unless the
// lightbox is open again by the time the event comes: the dialog's `close`
// is queued, not fired by close() itself, and a close followed at once by an
// open (a script's, a smoke's) had the late event shut the new lightbox, and
// reset its strip, under whatever it was doing - a delete in flight.
els.lightbox.addEventListener('close', () => { if (!els.lightbox.open) closeLightbox(); });
// The pictures: one that will not load says why, the next that loads takes it
// back, and a source that will not load ends the comparison.
els.lbImg.onerror = () => { void lbImageFailed(); };
els.lbImg.onload = lbImageLoaded;
els.lbSrcImg.onerror = lbSourceFailed;
// Zoom and pan: the wheel non-passive, or the page behind the dialog
// scrolls; a press, a drag, a release and a double-click on the stage; an
// arrow's press, for the stage to know a double-click begun on it.
els.lbMain.addEventListener('wheel', lbWheel, { passive: false });
for (const nav of [els.lbPrev, els.lbNext]) nav.addEventListener('pointerdown', lbArrowDown);
els.lbStage.onpointerdown = lbPointerDown;
els.lbStage.onpointermove = lbPointerMove;
els.lbStage.onpointerup = lbPointerUp;
els.lbStage.onpointercancel = lbPointerCancel;
els.lbStage.ondblclick = lbDoubleClick;
els.lbSplit.onkeydown = lbSplitKey;
els.lbFit.onclick = zoomFit;
els.lbOne.onclick = () => zoomOneToOne();
window.addEventListener('resize', layoutLightbox);
// And the stage's own size, whatever moves it: under 901 px a line in
// #lb-error (Use seed's, a delete's note) is a row of its own that took 30 px
// off the stage, and the picture laid out before kept its height, its foot
// cut off under a readout still saying "fit". Loop-safe: the layout writes
// the facts' room only when it changes (lbFactsRoom), from the stage as it
// would be with the facts at six lines.
new ResizeObserver(() => { if (els.lightbox.open) layoutLightbox(); }).observe(els.lbStage);

// The Refine dialog. Every box asks the server's preview again, as the grid's
// do; the scale buttons are built with the source they multiply. Keep shape
// moves the other side before the box's own refineChanged reads the pair.
els.refineRun.onclick = () => { void runRefine(); };
// Each close is held while the pass's POST is out (closeRefine), Escape's too.
els.refineCancel.onclick = els.refineClose.onclick = closeRefine;
els.refineDialog.addEventListener('cancel', refineCancelEvent);
els.refineDialog.addEventListener('close', refineClosed);
els.refineDialog.addEventListener('keydown', refineKeydown);
els.refineWidth.addEventListener('input', refineSideTyped);
els.refineHeight.addEventListener('input', refineSideTyped);
els.refineDice.onclick = () => { els.refineSeed.value = ''; refineChanged(); };
els.refineCompare.onclick = fillRefineCompare;
els.refineStartOver.onclick = startRefineOver;
els.refineMore.addEventListener('toggle', sizeRefinePrompt);
REFINE_WIDE.addEventListener('change', showRefineSource);
// The pass's own prompt check reads the text and whether the negative runs.
els.refinePrompt.addEventListener('input', () => { sizeRefinePrompt(); scheduleRefineTags(); });
// Its tag suggestions, on a tag checkpoint: typing, a click and the caret
// keys ask; leaving the box closes the list; the box's keys walk it, and an
// Escape it takes is not the dialog's.
els.refinePrompt.addEventListener('input', suggestRefineTags);
els.refinePrompt.addEventListener('click', suggestRefineTags);
els.refinePrompt.addEventListener('keyup', refinePromptCaret);
els.refinePrompt.addEventListener('keydown', refinePromptKeydown);
els.refinePrompt.addEventListener('blur', hideRefineSuggestions);
els.refineFromPanel.onclick = copyRefineFromPanel;
for (const el of [els.refinePrompt, els.refineNegative, els.refineCfg, els.refineTrueCfg]) {
  el.addEventListener('input', refineCheckChanged);
}
for (const el of [els.refineWidth, els.refineHeight, els.refineStrength, els.refineVersions,
                  els.refineSeed, els.refinePrompt, els.refineNegative, els.refineSteps,
                  els.refineCfg, els.refineMu, els.refineTrueCfg]) {
  el.addEventListener('input', refineChanged);
}
els.refineVersions.addEventListener('change', refineChanged);
els.refineUpscaler.addEventListener('change', refineChanged);
els.refineCropAnchor.addEventListener('change', refineChanged);

// The Inpaint dialog (inpaint.js) and its mask (inpaint-mask.js). The canvas
// paints with pointer events, captured for the stroke; its keys are its own
// while it has focus; a plain wheel scrolls, Alt+wheel sizes the brush.
els.inpaintRun.onclick = () => { void runInpaint(); };
els.inpaintCancel.onclick = els.inpaintClose.onclick = closeInpaint;
els.inpaintDialog.addEventListener('cancel', inpaintCancelEvent);
els.inpaintDialog.addEventListener('close', inpaintClosed);
els.inpaintDialog.addEventListener('keydown', inpaintKeydown);
els.inpaintDice.onclick = () => { els.inpaintSeed.value = ''; inpaintChanged(); };
els.inpaintStartOver.onclick = startInpaintOver;
for (const el of [els.inpaintStrength, els.inpaintVersions, els.inpaintSeed, els.inpaintPrompt,
                  els.inpaintNegative, els.inpaintSteps, els.inpaintCfg, els.inpaintMu, els.inpaintTrueCfg]) {
  el.addEventListener('input', inpaintChanged);
}
els.inpaintVersions.addEventListener('change', inpaintChanged);
for (const el of [els.inpaintPrompt, els.inpaintNegative, els.inpaintCfg, els.inpaintTrueCfg]) {
  el.addEventListener('input', () => { syncInpaintCheck(); scheduleInpaintCheck(); });
}
els.inpaintCanvas.addEventListener('pointerdown', maskPointerDown);
els.inpaintCanvas.addEventListener('pointermove', maskPointerMove);
els.inpaintCanvas.addEventListener('pointerup', maskPointerUp);
els.inpaintCanvas.addEventListener('pointercancel', maskPointerUp);
els.inpaintCanvas.addEventListener('lostpointercapture', maskPointerUp);
els.inpaintCanvas.addEventListener('pointerleave', maskPointerLeave);
els.inpaintCanvas.addEventListener('wheel', maskWheel, { passive: false });
els.inpaintCanvas.addEventListener('keydown', maskKeydown);
els.inpaintCanvas.addEventListener('keyup', maskKeyup);
els.inpaintCanvas.addEventListener('blur', maskBlur);
els.inpaintBrushTool.onclick = () => setMaskTool('brush');
els.inpaintEraserTool.onclick = () => setMaskTool('eraser');
els.inpaintInvert.onclick = invertMask;
els.inpaintClear.onclick = clearMask;
els.inpaintUndo.onclick = undoMask;
els.inpaintRedo.onclick = redoMask;
els.inpaintShow.addEventListener('change', drawMaskSoon);
els.inpaintSize.addEventListener('input', () => setMaskSize(+els.inpaintSize.value));
els.inpaintFeather.addEventListener('input', () => setMaskFeather(+els.inpaintFeather.value));

// The Outpaint dialog (outpaint.js). A side or band box is committed -
// rounded to its grid - on change (Enter, blur, an arrow's step), and a
// number already on the grid as it is typed; the presets replace the four
// sides; the frame follows its box's size in the wide tier.
els.outpaintRun.onclick = () => { void runOutpaint(); };
els.outpaintCancel.onclick = els.outpaintClose.onclick = closeOutpaint;
els.outpaintDialog.addEventListener('cancel', outpaintCancelEvent);
els.outpaintDialog.addEventListener('close', outpaintClosed);
els.outpaintDialog.addEventListener('keydown', outpaintKeydown);
els.outpaintDice.onclick = () => { els.outpaintSeed.value = ''; outpaintChanged(); };
els.outpaintStartOver.onclick = startOutpaintOver;
for (const side of OUTPAINT_SIDES) {
  const box = outpaintBox(side);
  box.addEventListener('input', () => commitOutpaintSide(side, { typing: true }));
  box.addEventListener('change', () => commitOutpaintSide(side));
}
els.outpaintBand.addEventListener('input', () => commitOutpaintBand({ typing: true }));
els.outpaintBand.addEventListener('change', () => commitOutpaintBand());
for (const button of [els.outpaintGrow, els.outpaintWide, els.outpaintTall]) {
  button.onclick = () => applyOutpaintPreset(button.dataset.preset);
}
// The To shape… list only picks: an arrow key moves a closed select and
// fires change on every press, so the preset applies on Apply (or Enter in
// the list), never on change - a keyboard user can reach every aspect.
els.outpaintShape.addEventListener('change', syncOutpaintPresets);
els.outpaintShapeApply.onclick = applyOutpaintShape;
for (const el of [els.outpaintStrength, els.outpaintVersions, els.outpaintSeed, els.outpaintPrompt,
                  els.outpaintNegative, els.outpaintSteps, els.outpaintCfg, els.outpaintMu, els.outpaintTrueCfg]) {
  el.addEventListener('input', outpaintChanged);
}
els.outpaintVersions.addEventListener('change', outpaintChanged);
for (const el of [els.outpaintPrompt, els.outpaintNegative, els.outpaintCfg, els.outpaintTrueCfg]) {
  el.addEventListener('input', () => { syncOutpaintCheck(); scheduleOutpaintCheck(); });
}
new ResizeObserver(() => { if (outpaintSource) drawOutpaintFrame(); }).observe(els.outpaintFrame);

// A delete's question (gallery-actions.js): Delete, Cancel, Enter in the count's box,
// a press outside it. Escape is the dialog's own cancel, answered as Cancel.
els.confirmOk.onclick = () => els.confirmDialog.close('ok');
els.confirmCancel.onclick = () => els.confirmDialog.close('');
els.confirmDialog.addEventListener('close', askClosed);
els.confirmDialog.addEventListener('click', askBackdrop);
els.confirmCount.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !keyboardEventHeld(e) && !keyboardCommandModifier(e) && !e.shiftKey) {
    e.preventDefault();
    if (!e.repeat) els.confirmDialog.close('ok');
  }
});

els.shortcuts.onclick = openShortcuts;
els.shortcutsClose.onclick = closeShortcuts;
els.shortcutsDialog.addEventListener('close', shortcutsClosed);
els.shortcutsDialog.addEventListener('click', shortcutsBackdrop);

// Native buttons also repeat Enter clicks. Discrete actions own one press;
// navigation, zoom, numeric stepping and the mask's held stroke still repeat.
document.addEventListener('keydown', (e) => {
  if (keyboardEventHeld(e)) return;
  const activation = e.key === 'Enter' || e.key === ' ';
  const button = e.target.closest?.('button, summary, [role="button"], [role="menuitem"], input[type="button"], input[type="submit"]');
  // Chromium activates a focused button even on Alt+Enter/Shift+Enter. These
  // are not alternate submit bindings; only the exact primary chord routes.
  if (activation && button && (keyboardCommandModifier(e) || e.shiftKey) && !keyboardSubmit(e)) {
    e.preventDefault();
    return;
  }
  if (!e.repeat) return;
  const repeatControl = e.target.closest?.('#lb-prev, #lb-next, #lb-fit, #lb-one, #lb-cfg-lower, #lb-cfg-higher, #lb-steps-lower, #lb-steps-higher, .sheet-zoom button');
  if (activation && button && !repeatControl && !keyboardCommandModifier(e) && !e.shiftKey) {
    e.preventDefault();
  }
  if (e.key === 'Escape' && !keyboardCommandModifier(e) && !e.shiftKey) e.preventDefault();
}, true);

document.addEventListener('keydown', (e) => {
  if (keyboardEventHeld(e)) return;
  shortcutKey(e);
  if (e.defaultPrevented) return;

  // The event's modal owns its commands. An unknown settings dialog blocks
  // background actions too; adding another dialog cannot opt into Generate.
  const dialog = e.target.closest?.('dialog[open]');
  const modal = dialog || document.querySelector('dialog[open]');
  if (els.shortcutsDialog.open || els.snippetsDialog.open) {
    if (keyboardSubmit(e)) e.preventDefault();
    return;
  }
  if (modelListOpen()) return;
  const inMenu = e.target.closest?.('[role="menu"], [role="listbox"]');
  if (keyboardSubmit(e)) {
    e.preventDefault();
    if (e.repeat || inMenu || e.target.closest?.('.tag-editor')
        || (!dialog && e.target.closest?.('[role="dialog"], #model-picker'))) return;
    if (modal) {
      if (dialog === els.refineDialog) void runRefine();
      else if (dialog === els.inpaintDialog) void runInpaint();
      else if (dialog === els.outpaintDialog) void runOutpaint();
      else if (dialog === els.refineBatchDialog) void runRefineBatch();
      else if (dialog === els.swapDialog) void queueSwap();
      else if (dialog === els.moreSeedsDialog) void runMoreSeeds();
    } else if (els.gridSection.open && e.target.closest?.('#grid-section')) els.gridRun.click();
    else generate();
    return;
  }
  // Native Escape cancels the top dialog; only the lightbox has a custom
  // close path so that browsing returns focus to the image actually reached.
  if (modal && dialog !== els.lightbox) return;
  const plainEscape = e.key === 'Escape' && !keyboardCommandModifier(e) && !e.shiftKey && !e.repeat;
  if (plainEscape && els.lightbox.open) { dismissLightbox(); return; }
  if (!modal && plainEscape) {
    if (!els.modelPicker.hidden) { e.preventDefault(); showModelPicker(false); return; }
    if (!els.examplesMenu.hidden) { showExamples(false); return; }
    if (!els.tagsMenu.hidden) { showTagHelper(false); return; }
    if (!els.tagFinderMenu.hidden) { showTagFinder(false); return; }
    if (!els.promptAnalysis.hidden && !(e.target.matches?.('input[type="search"]') && e.target.value)) {
      escapeAnalysis();
      return;
    }
  }
  if (!els.lightbox.open || inMenu || keyboardEditingTarget(e.target) || keyboardCommandModifier(e)) return;
  // Shift+arrows belongs to an active swipe. Alt/Ctrl/Meta arrows keep their
  // browser/OS meaning, and fields/selects retain all of their editing keys.
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
    e.preventDefault();
    const delta = e.key === 'ArrowLeft' ? -1 : 1;
    if (e.shiftKey ? nudgeSplit(delta) : stepPicked(delta) || stepPass(delta) || stepLadder(delta)) return;
    void stepLightbox(delta);
    return;
  }
  const key = e.key.toLowerCase();
  if (['f', 'r', 's', 'delete'].includes(key)) {
    if (e.shiftKey || e.repeat) return;
    e.preventDefault();
    if (key === 'f') { const rec = lbList[lbIndex]; if (rec) void setFavorite(rec, !rec.favorite, { say: true }); }
    else if (key === 'r') els.lbRefine.click();
    else if (key === 's') void toggleCompare();
    else els.lbDelete.click();
  } else if (e.key === '+' || e.key === '=') { e.preventDefault(); zoomLightbox(ZOOM_STEP); }
  else if (e.key === '-' || e.key === '_') { e.preventDefault(); zoomLightbox(1 / ZOOM_STEP); }
  else if (!e.shiftKey && e.key === '0') { e.preventDefault(); zoomFit(); }
  else if (!e.shiftKey && e.key === '1') { e.preventDefault(); zoomOneToOne(); }
});

// A file dragged over the page is a PNG whose settings fill the panel
// (png-settings.js), wherever it is dropped; the outline over the window
// says so. Never the browser's own drop, which opened the file in the app's
// place - and over a modal dialog, which keeps the page to itself, nothing.
for (const type of ['dragenter', 'dragover']) {
  document.addEventListener(type, (e) => {
    if (!e.dataTransfer?.types?.includes('Files')) return;
    e.preventDefault();
    const take = pngDropTarget(e);
    e.dataTransfer.dropEffect = take ? 'copy' : 'none';
    document.body.classList.toggle('png-drop', take);
  });
}
// Out of the window: the drag's leave names nothing it entered.
document.addEventListener('dragleave', (e) => { if (!e.relatedTarget) document.body.classList.remove('png-drop'); });
document.addEventListener('drop', (e) => {
  const page = pageDrag;
  pageDrag = false;
  if (!e.dataTransfer?.types?.includes('Files')) return;
  e.preventDefault();
  if (!page && pngDropTarget(e)) dropPng(e);
  else document.body.classList.remove('png-drop');
});
// A drag that starts on the page - a gallery card's image, a link - is
// none of the above, though Chromium lists 'Files' in its types: the card
// is no file the user brought. Over at its end, its drop, or the next
// mouse move where a removed card's end never reached the document.
document.addEventListener('dragstart', () => { pageDrag = true; });
for (const type of ['dragend', 'mousemove']) document.addEventListener(type, () => { pageDrag = false; });
// Load settings from PNG…, beside the panel's outcome line: the same,
// picked. The input is emptied once read, so the same file picked again is
// read again.
els.pngButton.addEventListener('click', () => els.pngFile.click());
els.pngFile.addEventListener('change', (e) => {
  e.stopPropagation();
  const file = els.pngFile.files[0];
  els.pngFile.value = '';
  if (file) void loadPngSettings(file);
});
// Not a panel change for `persistPanel`: what the file holds is, once read.
els.pngFile.addEventListener('input', (e) => e.stopPropagation());

window.addEventListener('resize', placeTagValidation);
placeTagValidation();
TRANSLATION_WIDE.addEventListener('change', placeTranslation);
placeTranslation();
new ResizeObserver(chipsResized).observe(els.tagValidation);
// The panel's sticky top (panel.js has the rules). It follows the panel's
// height - the grid fold, the progress, the last-run banner, the hints all
// change it - and the window's; the observer also reports once on observing,
// which places it at load. It moves with the user's scrolling, and for a
// control focused out of sight - but not for focus a press in the panel gave,
// where the control is under the pointer already. The press ends after the
// click, which is where a label hands its control focus.
new ResizeObserver(panelResized).observe(document.querySelector('.panel'));
window.addEventListener('resize', placePanel);
window.addEventListener('scroll', followScroll, { passive: true });
window.addEventListener('scrollend', () => { panelPos.scrolling = panelPos.dragging; });
window.addEventListener('wheel', userScrolls, { passive: true });
window.addEventListener('touchmove', userScrolls, { passive: true });
window.addEventListener('keydown', (e) => {
  if (e.defaultPrevented || !SCROLL_KEYS.has(e.key)) return;
  if (!e.target.closest('input, textarea, select, button, summary, a, [tabindex], [contenteditable]')) userScrolls();
});
window.addEventListener('pointerdown', (e) => {
  if (e.target !== document.documentElement && e.button !== 1) return;
  panelPos.dragging = e.target === document.documentElement;
  userScrolls();
});
document.querySelector('.panel').addEventListener('pointerenter', () => { panelPos.pointer = true; });
document.querySelector('.panel').addEventListener('pointerleave', () => {
  panelPos.pointer = false;
  if (panelPos.deferred) placePanel();
  promptTokensAimed = false;
  if (promptTokensHeld) syncPromptTokens();
});
// The token line under the prompt waits to show or go while the mouse is on a
// control it would move (tag-tokens.js, syncPromptTokens): one whose top is at or
// below the prompt box's bottom. Given back on leaving it and on a key.
const TOKENS_AIMED = 'button, a, input, select, textarea, summary, label, [role="option"], [role="menuitem"], [tabindex]';
document.querySelector('.panel').addEventListener('pointerover', (e) => {
  const control = e.pointerType === 'touch' ? null : e.target.closest(TOKENS_AIMED);
  promptTokensAimed = !!control && control !== els.prompt
    && control.getBoundingClientRect().top >= els.prompt.getBoundingClientRect().bottom - 0.5;
  if (promptTokensHeld && !promptTokensAimed) syncPromptTokens();
});
window.addEventListener('keydown', () => { if (promptTokensHeld) syncPromptTokens(true); }, true);
document.querySelector('.panel').addEventListener('pointerdown', () => { panelPos.press = true; });
window.addEventListener('pointerup', () => {
  panelPos.dragging = false;
  setTimeout(() => { panelPos.press = false; });
}, true);
window.addEventListener('pointercancel', () => { panelPos.press = panelPos.dragging = false; }, true);
document.querySelector('.panel').addEventListener('focusin', (e) => {
  if (!panelPos.press) revealInPanel(e.target);
});
// A tip is placed as it opens: entering its badge from outside, not moving
// between the badge and the tip it holds.
document.addEventListener('pointerover', (e) => {
  const help = e.target.closest?.('.help');
  if (help && !help.contains(e.relatedTarget)) placeTip(help);
});
document.addEventListener('focusin', (e) => { if (e.target.matches?.('.help')) placeTip(e.target); });
// A queue row a status poll adds or takes out waits while the pointer rests
// on what it would move (job-queue.js, pointQueue); off the window, nothing.
document.addEventListener('pointerover', (e) => pointQueue(e.target));
document.addEventListener('pointerout', (e) => { if (!e.relatedTarget) pointQueue(null); });
// A tip hung from the window by a badge in a scrolling box keeps up with it
// (panel.js, floatTip): any scroll, the box's own included.
document.addEventListener('scroll', followFloatingTip, { capture: true, passive: true });
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
restoreCardView();
restoreJobNotifications();
refreshStatus();
loadGallery();
loadRuns();
resumeActiveJob();
setInterval(refreshStatus, 3000);
