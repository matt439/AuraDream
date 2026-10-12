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
els.newImage.onclick = goToNewImage;
els.dice.onclick = () => { els.seed.value = ''; syncRuns(); schedulePreview(); persistPanel(); };
els.reset.onclick = resetPanel;
els.orientation.onclick = swapOrientation;

// The menu's keyboard: ArrowDown and ArrowUp walk it (round the ends), Home
// and End jump; leaving it - Tab, a click elsewhere - closes it, rather than
// leaving it open over the prompt.
els.examplesBtn.addEventListener('keydown', (e) => {
  if (e.key !== 'ArrowDown') return;
  e.preventDefault();
  if (els.examplesMenu.hidden) showExamples(true);
  els.examplesMenu.querySelector('[role="menuitem"]')?.focus();
});
els.examplesMenu.addEventListener('keydown', (e) => {
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
  if (!els.pickedMenu.hidden && !e.target.closest('#picked')) showPickedMenu(false);
  // The helper hangs under the prompt, which takes clicks to move its caret.
  if (!els.tagsMenu.hidden && !e.target.closest('#tag-helper, #tags-menu, #prompt')) showTagHelper(false);
  // The token line is in the editor (under the box) but no part of typing.
  if (!e.target.closest('.prompt-editor') || e.target === els.promptTokens) hideTagSuggestions();
});
els.unload.onclick = unloadModel;
els.more.onclick = () => loadGallery();
// One card at a time in the tab order, walked with the arrow keys (gallery.js).
els.gallery.addEventListener('keydown', galleryKey);
els.gallery.addEventListener('focusin', galleryFocus);
// A card's batch over the page while the mouse rests on it (gallery-batch.js);
// any scroll, press or card moving takes it down, the gallery's own column's
// scroll included.
els.gallery.addEventListener('pointerover', batchPeekOver);
els.gallery.addEventListener('pointerout', batchPeekOut);
els.gallery.addEventListener('pointerdown', batchPeekPressed);
document.addEventListener('scroll', batchPeekScrolled, { capture: true, passive: true });
new MutationObserver(batchPeekShifted).observe(els.gallery, { childList: true });
// The selection's Select, Clear and Escape (gallery-select.js); a card's
// press picks through cardPress, which card() hands thumb.
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
els.gridRun.onclick = () => startJob('/api/grid', gridPayload());
// The preview reflects the panel as much as the axes: a blank axis is the
// control's own value, and both the size and the batch move the totals.
// The negative prompt is in the list because a `{name}` placeholder may live
// there - the server validates it - and Resume because the whole "N already
// done" half of the preview is what ticking it is for. Repeat (`runs`) moves
// nothing a grid runs, but the preview says so while it is above one.
for (const el of [els.gridPrompt, els.gridSteps, els.gridCfg, els.gridMu, els.gridSeeds,
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
// The cell Swipe CFG starts from: the one the keyboard was last on.
els.sheetGrid.addEventListener('focusin', (e) => {
  const cell = e.target.closest?.('.sheet-cell[data-id]:not(.source)');
  if (cell) sheetCfgFrom = +cell.dataset.id;
});
// The sheet's fit is measured against the window, so it follows a resize.
window.addEventListener('resize', refitSheet);
els.rerender.onclick = rerenderSelection;
els.deleteUnstarred.onclick = deleteUnstarred;
// Re-render and Delete unstarred come and go once the pointer is off the
// head's controls - the footer row under them is not (showInHead,
// gallery-actions.js).
const galleryHead = els.library.querySelector('.gallery-head');
galleryHead.addEventListener('pointerover', (e) => pointHead(!e.target.closest('.gallery-footer')));
galleryHead.addEventListener('pointerleave', () => pointHead(false));
initBundles();
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
// Refine likewise: openRefine closes the lightbox for its dialog, and a close
// that runs nothing gives it back (refineClosed). Where the checkpoint has no
// such pass the lightbox stays open, since the note saying so is mirrored
// into it.
els.lbRefine.onclick = () => { const rec = lbList[lbIndex]; if (rec && !refuseBinned(rec, 'refine')) openRefine(rec); };
els.lbFav.onclick = () => { const rec = lbList[lbIndex]; if (rec) setFavorite(rec, !rec.favorite); };
els.lbDelete.onclick = () => { const rec = lbList[lbIndex]; if (rec) deleteImage(rec); };
// Nor a held Enter's repeats on the focused button, which press it again for
// each - the image after the one just binned, and the next - as Del's would.
els.lbDelete.addEventListener('keydown', (e) => {
  if (e.repeat && (e.key === 'Enter' || e.key === ' ')) e.preventDefault();
});
els.lightbox.onclick = (e) => { if (e.target === els.lightbox) dismissLightbox(); };
// A pass against its source, or any image across CFG (lightbox-cfg.js):
// each kind's button turns it on and off, or switches to its kind; a mode
// turns on the kind S would in that mode, or switches to it. Lower and
// Higher CFG step along CFG, and Against picks the partner.
els.lbCompareOn.onclick = () => { void toggleCompare(null, 'source'); };
els.lbCompareCfg.onclick = () => { void toggleCompare(null, 'cfg'); };
els.lbSwipe.onclick = () => { void toggleCompare('swipe'); };
els.lbSideBySide.onclick = () => { void toggleCompare('side'); };
els.lbCfgLower.onclick = () => pressCfgStep(-1);
els.lbCfgHigher.onclick = () => pressCfgStep(1);
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

// A delete's question (gallery-actions.js): Delete, Cancel, Enter in the count's box,
// a press outside it. Escape is the dialog's own cancel, answered as Cancel.
els.confirmOk.onclick = () => els.confirmDialog.close('ok');
els.confirmCancel.onclick = () => els.confirmDialog.close('');
els.confirmDialog.addEventListener('close', askClosed);
els.confirmDialog.addEventListener('click', askBackdrop);
els.confirmCount.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); els.confirmDialog.close('ok'); }
});

document.addEventListener('keydown', (e) => {
  // Nothing of the page's while a delete asks: Escape there is its Cancel,
  // which closed the lightbox under it as well, and Ctrl+Enter generated.
  if (els.confirmDialog.open || els.bundlesDialog.open) return;
  // An open loader list (quantization, offload, format) draws in the page, so its keys reach
  // here; they are the list's own (Escape closes only it), as they were when
  // the native popup kept them from the page.
  if (modelListOpen()) return;
  // Not under a dialog: Escape there is the dialog's (Refine, Mass refine,
  // Swap tags, More seeds), the picker staying open under it.
  if (e.key === 'Escape' && !e.defaultPrevented && !els.modelPicker.hidden
      && !els.lightbox.open && !els.refineDialog.open
      && !els.refineBatchDialog.open && !els.swapDialog.open && !els.moreSeedsDialog.open) {
    e.preventDefault();
    showModelPicker(false);
    return;
  }
  // Ctrl/Cmd+Enter submits whatever the focus is in: inside the Refine dialog
  // it runs the pass and inside the open grid fold it runs the grid, which is
  // what their boxes are for - from a grid box Generate would render the
  // prompt's {placeholders} literally. Anywhere else, the prompt box included,
  // it generates. Run grid's click, so its handler and guards are the press's.
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    // Taken, so the Enter goes nowhere else: the button a closing dialog
    // hands focus back to - the card's ⤢ - read it as a press of its own.
    e.preventDefault();
    if (els.refineDialog.open) void runRefine();
    else if (els.refineBatchDialog.open) void runRefineBatch();
    else if (els.swapDialog.open) void queueSwap();
    else if (els.moreSeedsDialog.open) void runMoreSeeds();
    else if (els.gridSection.open && e.target.closest?.('#grid-section')) els.gridRun.click();
    else generate();
    return;
  }
  // The lightbox's Escape before the menus': it is modal over a menu left
  // open under it - the Tags helper stays open when focus leaves it by
  // keyboard - and the menu's branch took the key, so the dialog's own
  // cancel closed the lightbox without dismissLightbox, focus going back to
  // the image it was opened on rather than the one the walk ended on. The
  // menu stays open under the page, as it was before the lightbox came up.
  if (e.key === 'Escape' && !e.defaultPrevented && els.lightbox.open) { dismissLightbox(); return; }
  // An Escape the prompt's suggestion list or a chip's editor took closes that
  // alone: the helper stays open under the box while its list comes and goes.
  if (e.key === 'Escape' && !e.defaultPrevented && !els.examplesMenu.hidden) { showExamples(false); return; }
  if (e.key === 'Escape' && !e.defaultPrevented && !els.tagsMenu.hidden) { showTagHelper(false); return; }
  // Then the prompt check, as its Close does - not under a dialog, whose
  // Escape is its own (the lightbox's below), nor in the middle of a
  // composition, nor in a search box with text, which Escape empties.
  if (e.key === 'Escape' && !e.defaultPrevented && !e.isComposing && !els.promptAnalysis.hidden
      && !els.lightbox.open && !els.refineDialog.open
      && !els.refineBatchDialog.open && !els.swapDialog.open && !els.moreSeedsDialog.open
      && !(e.target.matches?.('input[type="search"]') && e.target.value)) {
    escapeAnalysis();
    return;
  }
  if (!els.lightbox.open) return;
  // A key the lightbox's own control took - the swipe divider's arrows.
  if (e.defaultPrevented) return;
  // Arrows walk whichever strip the lightbox was opened from. Nothing else on
  // the page reads them while it is up, so the caret keys are free to take.
  // Comparing a pass, they swap to another pass of the same source where the
  // strip has one (`stepPass`), comparing across CFG they step along CFG
  // where the strip has another value (`stepCfg`), the comparison kept, and
  // Shift+arrows move a swipe's divider (`nudgeSplit`); otherwise they walk,
  // as ever. On Against they are the select's own: a closed select changes on
  // ← → as on ↑ ↓ on Windows, and taking them had walked to another image
  // under the focus.
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
    if (e.target.tagName === 'SELECT') return;
    e.preventDefault();
    const delta = e.key === 'ArrowLeft' ? -1 : 1;
    if (e.shiftKey ? nudgeSplit(delta) : stepPass(delta) || stepCfg(delta)) return;
    void stepLightbox(delta);
    return;
  }
  // F, S, Delete and the zoom keys act on the image in view - unless a text
  // field still holds the caret, where they are typing, or a modifier makes
  // them the browser's: Ctrl+F is find, Ctrl+plus the page's zoom.
  if (e.target.matches('input, textarea') || e.ctrlKey || e.metaKey || e.altKey) return;
  // F does what the Star button's press does, and says so where focus is not
  // on that button (setFavorite's `say`): its state alone changed.
  if (e.key === 'f' || e.key === 'F') { const rec = lbList[lbIndex]; if (rec) void setFavorite(rec, !rec.favorite, { say: true }); }
  else if (e.key === 's' || e.key === 'S') void toggleCompare();
  // Not a held key's repeats: the delete asks nothing now (it moves the
  // image to the recycle bin), and Del held down would bin a run of images.
  else if (e.key === 'Delete' && !e.repeat) els.lbDelete.click();
  else if (e.key === '+' || e.key === '=') { e.preventDefault(); zoomLightbox(ZOOM_STEP); }
  else if (e.key === '-' || e.key === '_') { e.preventDefault(); zoomLightbox(1 / ZOOM_STEP); }
  else if (e.key === '0') zoomFit();
  else if (e.key === '1') zoomOneToOne();
});

// A file dragged over the page is dropped nowhere: the page takes no files,
// and the browser's own drop opened the file in the app's place.
for (const type of ['dragover', 'drop']) {
  document.addEventListener(type, (e) => {
    if (!e.dataTransfer?.types?.includes('Files')) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'none';
  });
}

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
refreshStatus();
loadGallery();
loadRuns();
resumeActiveJob();
setInterval(refreshStatus, 3000);
