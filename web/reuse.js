// AuraDream's page script: reusing image settings and undoing that reuse.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

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
  promptReplaced();
  els.seed.value = rec.seed;
  if (!sampling) return;
  setStepCache('');
  ownControls();
  els.steps.value = rec.steps;
  els.cfg.value = rec.guidance_scale;
  // The scale the negative prompt ran at. Only where the box is a knob of its
  // own: on a checkpoint whose CFG is real CFG the record's value *is* the
  // CFG box, and the greyed box would only echo it.
  els.trueCfg.value = trueCfgSupported && rec.true_cfg != null ? rec.true_cfg : '';
  // Turbo's flow shift. Records from a pipeline that derives its own carry null.
  // Only where Mu applies, as True CFG above: on Anima, say, the greyed box
  // showed another checkpoint's 0.8 and the panel remembered it.
  els.mu.value = muSupported ? (rec.mu ?? '') : '';
  // The conditioning that made it, where this server can still apply it: the
  // references are ids in this same gallery, and the adapter has to be a file
  // the select is offering or the request would be refused at submit.
  referenceIds = referenceSupported ? [...(rec.reference ?? [])] : [];
  const listed = [...els.lora.options].some((o) => o.value === rec.lora);
  els.lora.value = loraSupported && rec.lora && listed ? rec.lora : '';
  pendingLora = '';
  els.loraScale.value = els.lora.value ? (rec.lora_scale ?? 1) : 1;
}

// tags.js's `writeBox` - the page's one whole-box write: as typing writes it,
// with no suggestion list or typing state over the new text - with the box
// focused first without the browser's scroll to it. Writing as typing does
// takes the focus, and the browser then scrolled the page to the box as
// though a stuck panel moved with it: half a screen of the gallery lost at
// the page's end, nothing at all for a box focused already. Where the page
// goes after a Reuse is `revealReused`'s, and after its Undo, pressed at the
// panel's foot, nowhere.
function rewriteBox(el, text) {
  el.focus({ preventScroll: true });
  writeBox(el, text);
}

// Reuse restores what made a record. A direct render's settings go into the
// panel; a refine pass is the Refine dialog's, so a refined record opens the
// dialog on its source with the pass as it ran (`reusePass`).
// An instruction edit's Reuse opens Edit by instruction… on its source
// (`reuseEdit`), never the panel: it is held only where that source cannot be
// had - never resolved to a gallery image, deleted, or in the recycle bin.
function editReuseReason(rec) {
  const edit = rec.edit;
  if (edit == null) return '';
  if (edit.source == null) {
    return 'The source of this instruction edit was never resolved to a gallery image, so the edit cannot be reopened.';
  }
  if (rec.edit_missing && rec.edit_binned) {
    return `#${edit.source}, the source this was edited from, is in the recycle bin; restore it to reopen the edit.`;
  }
  if (rec.edit_missing) {
    return `#${edit.source}, the source this was edited from, has been deleted, so the edit cannot be reopened.`;
  }
  return '';
}

function executionReuseReason(rec) {
  if (rec.edit != null) return editReuseReason(rec);
  const spec = rec.execution;
  if (spec == null) return '';
  const empty = typeof spec === 'object' && !Array.isArray(spec)
    && Object.keys(spec).every((name) => ['compilation', 'attention_backend', 'step_cache'].includes(name)
      && spec[name] == null);
  return empty ? '' : 'This image uses an execution recipe that is unavailable in this build. Its settings cannot be reused.';
}

function refuseExecutionReuse(rec) {
  const reason = executionReuseReason(rec);
  if (!reason) return false;
  // Keep the composer's current Reuse note and its Undo intact. Card and
  // lightbox presses share the gallery's slot; showError mirrors a modal.
  showError(reason, false, { slot: els.galleryError });
  return true;
}

// A pass's source lookup is an opening intent, not permission to replace
// whatever the user opens or writes while its answer is out. It owns no job.
let reusePassOpening = null;

function reusePassDraftKey() {
  // The existing raw UI snapshot includes transient Resume/Then refine and
  // unsupported performance intent too. Context has its own rules below.
  const draft = JSON.parse(setupComposerState());
  delete draft.model;
  delete draft.revision;
  delete draft.server;
  // A number box can display an unfinished edit while exposing the same
  // empty value as before it. Preserve that intent without parsing a payload.
  draft.badInput = [...panelNumberInputs(), [els.gridSeeds], [els.gridBudget]]
    .map(([el]) => el.validity.badInput);
  return JSON.stringify(draft);
}
const reusePassScopeKey = () => JSON.stringify([
  els.search.value.trim(), els.favOnly.checked, els.allModels.checked, binMode, sheetMode,
]);

// A write the page makes itself (a grid's drawn seed, a LoRA the server no
// longer offers) is not a newer draft: an opening that still matched the
// panel before it keeps its intent, on the panel as written.
function keepReusePassDraft(write) {
  const opening = reusePassOpening;
  const matched = opening && reusePassDraftKey() === opening.draft;
  write();
  if (matched && reusePassOpening === opening) opening.draft = reusePassDraftKey();
}

function cancelReusePassOpening() {
  const opening = reusePassOpening;
  if (!opening) return;
  reusePassOpening = null;
  opening.controller.abort();
}

function beginReusePassOpening(rec) {
  cancelReusePassOpening();
  const controller = new AbortController();
  const draft = reusePassDraftKey();
  const scope = reusePassScopeKey();
  const revision = modelRevision;
  const model = loadedModel;
  const startedAt = serverStartedAt;
  // A menu proxy or a just-closed lightbox can leave focus elsewhere. Only
  // bind a host that actually carries this image; body/menu items are not hosts.
  const focused = document.activeElement?.closest?.('#gallery .card[data-id], #viewer figure[data-id]');
  const host = focused && +focused.dataset.id === rec.id ? focused : null;
  const fromViewer = !!host?.closest?.('#viewer');
  const viewerContext = viewerRecords[0]?.job_id ?? viewerRecords[0]?.id ?? null;
  const lightboxId = els.lightbox.open ? lbList[lbIndex]?.id ?? null : null;
  const dialogs = [...document.querySelectorAll('dialog[open]')];
  const opening = { controller, draft, valid: () => {
    if (reusePassOpening !== opening || controller.signal.aborted) return false;
    if (modelRevision !== revision || loadedModel !== model) return false;
    if (startedAt != null && serverStartedAt != null && startedAt !== serverStartedAt) return false;
    if (reusePassDraftKey() !== opening.draft || reusePassScopeKey() !== scope) return false;
    if (host && (!host.isConnected || !isShown(host))) return false;
    if (fromViewer && (viewerRecords[0]?.job_id ?? viewerRecords[0]?.id ?? null) !== viewerContext) return false;
    if (lightboxId != null && (!els.lightbox.open || lbList[lbIndex]?.id !== lightboxId)) return false;
    return ![...document.querySelectorAll('dialog[open]')].some((dialog) => !dialogs.includes(dialog));
  } };
  reusePassOpening = opening;
  return opening;
}

function reuse(rec) {
  cancelReusePassOpening();
  if (refuseExecutionReuse(rec)) return;
  // An edit's is Edit by instruction…'s, on its source with its instruction.
  if (rec.edit != null) { void reuseEdit(rec); return; }
  // An outpaint's is the Outpaint dialog's, on its source with its sides -
  // asked first, since an outpaint has a mask as well.
  if (rec.refine != null && rec.outpaint) { void reuseOutpaint(rec); return; }
  // An inpaint's is the Inpaint dialog's, on its source with its mask.
  if (rec.refine != null && rec.mask) { void reuseInpaint(rec); return; }
  if (rec.refine != null) { void reusePass(rec); return; }
  reuseIntoPanel(rec, true);
}

// The note the last Reuse left in the panel - its sentences, the panel as it
// was before and as the Reuse left it, and the Undo button that puts it
// back - or null once there is no Undo to offer.
let reuseNote = null;

// What a Reuse's Undo puts back, as the panel holds it now.
function reuseFields() {
  return {
    prompt: els.prompt.value, negative: els.negative.value, seed: els.seed.value,
    width: els.width.value, height: els.height.value,
    steps: els.steps.value, cfg: els.cfg.value, mu: els.mu.value, trueCfg: els.trueCfg.value,
    owned: { ...owned }, reference: [...referenceIds], lora: els.lora.value || pendingLora, loraScale: els.loraScale.value,
    stepCache: stepCacheValue(),
  };
}

// The Examples' rule (tags.js's `syncPromptUndo`) for the whole of what the
// Undo restores: it is on offer only while the panel holds what the Reuse
// left there. It used to stay through any later edit, and pressing it then
// threw away the typing, the seed or the Example picked since, and said the
// panel was as it was. So the first change takes the button, and the note's
// sentences stay. Asked by `persistPanel`, which every change of the panel
// reaches - typing, a preset, the dice, an Example (whose own Undo then
// stands alone).
function syncReuseUndo() {
  if (reusePassOpening && reusePassDraftKey() !== reusePassOpening.draft) cancelReusePassOpening();
  const note = reuseNote;
  if (!note || JSON.stringify(reuseFields()) === JSON.stringify(note.after)) return;
  reuseNote = null;
  note.button?.remove();
}

// `name` is what the note calls the record - a gallery id, or a PNG's file
// name for Load settings from PNG (png-settings.js) - and `sampling` false
// writes only the prompt pair and the seed, as `fillPanel` takes it: an
// A1111 block's steps and CFG are another sampler's.
// The words for an edit offered to the panel: its instruction is about its
// source, and means nothing as a prompt of its own.
const EDIT_NOT_PANEL = 'An instruction edit is reopened in Edit by instruction… on its source; '
  + 'its source-dependent instruction never goes into the generation panel.';

function reuseIntoPanel(rec, withSize, note = '', { name = `#${rec.id}`, sampling = true } = {}) {
  cancelReusePassOpening();
  if (refuseExecutionReuse(rec)) return;
  if (rec.edit != null) { showError(EDIT_NOT_PANEL, false, { slot: els.galleryError }); return; }
  // One Undo on offer at a time: an Example's, for the text this replaces,
  // goes with it.
  dropPromptUndo();
  // Ctrl+Z covers the two text boxes; Reuse also replaces the seed, the size,
  // steps/CFG/mu and which of those the user had taken off the profile, so
  // the note's Undo restores those from this.
  const before = {
    state: panelState(),
    values: { steps: els.steps.value, cfg: els.cfg.value, mu: els.mu.value, trueCfg: els.trueCfg.value },
    owned: { ...owned },
  };
  fillPanel(rec, sampling);
  if (withSize) {
    els.width.value = rec.width;
    els.height.value = rec.height;
  }
  syncSize();
  syncCfg();
  syncConditioning();
  revealActiveSetup();
  syncRuns();
  syncExecutionControls();
  syncSubmitButtons();
  schedulePreview();
  scheduleTagValidation(0);
  persistPanel();
  // Reuse used to replace a half-written prompt with no word, and the work
  // was gone; the note says so whenever there was other text to replace. Its
  // first sentence, the Undo after it, then whatever else there is to say:
  // the note is at the panel's foot, and where the window shows the prompt
  // and only the start of the note, the start is what is in sight.
  const replaced = before.state.prompt.trim() !== '' && before.state.prompt !== rec.prompt;
  const lead = `Settings from ${name} loaded${replaced ? ' - your prompt was replaced' : ''}.`;
  const notes = [note];
  // Steps, CFG and mu are checkpoint-specific: Turbo's 8 / 0 / 1.15 on FLUX is
  // a visibly broken image recorded under FLUX. The checkpoint is part of the
  // reproduction set as much as the seed, so say so - as a note, not a failure.
  // A PNG from another tool names no checkpoint of this app's: its note
  // says itself what does not carry.
  if (loadedModel && rec.model_id && rec.model_id !== loadedModel) {
    // Mu named only where its box can be checked: on Anima it is greyed.
    const knobs = muSupported ? 'steps, CFG and mu' : 'steps and CFG';
    notes.push(
      `These settings were made on ${checkpointName(rec.model_id)}; the server is running ` +
      `${checkpointName(loadedModel)}, whose ${knobs} differ. Prompt` +
      `${withSize ? ', size' : ''} and seed carry over; check the ${knobs} ` +
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
      + `${one ? 'is' : 'are'} not used: ${checkpointName(loadedModel) || 'this checkpoint'} takes none, `
      + 'so this renders from the prompt alone.',
    );
  }
  reuseNote = { lead, parts: notes, before, after: reuseFields(), name, button: null };
  showReuseNote();
  revealReused(reuseNote);
  void checkReused(reuseNote, rec);
}

// The Reuse note: its first sentence, the Undo, the rest. Written in the
// panel's slot - Reuse sends the user to the panel - and rewritten when a
// follow-up has more to say.
function showReuseNote() {
  const note = reuseNote;
  reuseSaid = note.lead;
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

// The lead of the last Reuse note written, which outlives its Undo: the first
// change takes the button and keeps the words (syncReuseUndo). A Reset takes
// the words too, while they are still what the panel's line says - "Settings
// from #15 loaded - your prompt was replaced" stood over the emptied panel.
let reuseSaid = null;
function clearReuseNote() {
  // Reset reaches this even when its empty draft already matches the panel.
  cancelReusePassOpening();
  if (reuseSaid !== null && els.error.textContent.startsWith(reuseSaid)) showError('');
  reuseSaid = null;
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
    // Part of what the Reuse left, so not a change that retires its Undo.
    if (reuseNote === note) note.after.reference = note.after.reference.filter((id) => !dead.includes(id));
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
// sampling values with whose they were, the references, the adapter and an
// expansion's English.
function undoReuse(note) {
  const { state, values } = note.before;
  if (reuseNote === note) reuseNote = null;
  rewriteBox(els.negative, state.negative);
  rewriteBox(els.prompt, state.prompt);
  els.seed.value = state.seed;
  els.width.value = state.width;
  els.height.value = state.height;
  Object.assign(owned, note.before.owned);
  els.steps.value = values.steps;
  els.cfg.value = values.cfg;
  els.mu.value = values.mu;
  els.trueCfg.value = values.trueCfg;
  setStepCache(Object.hasOwn(state, 'stepCache') ? state.stepCache : '');
  referenceIds = state.reference;
  els.lora.value = state.lora;
  pendingLora = state.lora && !els.lora.value ? state.lora : '';
  els.loraScale.value = state.loraScale;
  promptReplaced();
  // An expansion's English, where the prompt put back is the one it
  // translates, as Reset's Undo brings it back.
  const t = state.translation;
  if (t && els.prompt.value === t.source) showPromptTranslation(t.source, t.text);
  syncSize();
  syncCfg();
  syncConditioning();
  revealActiveSetup();
  syncRuns();
  syncExecutionControls();
  syncSubmitButtons();
  schedulePreview();
  scheduleTagValidation(0);
  persistPanel();
  showError(`Reuse of ${note.name} undone: the panel is as it was.`, true);
}

// A refined record holds its source's id, not the source: asked for, so the
// dialog has the picture and the size to scale. A source that has been
// deleted cannot be refined again - the prompt and settings still go into
// the panel, at the panel's own size, since the pass's may be past what a
// direct render takes.
async function reusePass(rec) {
  const opening = beginReusePassOpening(rec);
  try {
    let source;
    try {
      source = await api(`/api/images/${rec.refine}`, { signal: opening.controller.signal });
    } catch (err) {
      if (!opening.valid() || err.name === 'AbortError') return;
      // Release before the current fallback: it is this intent's deliberate
      // panel replacement, whose own entry retires any older opening.
      reusePassOpening = null;
      if (!/unknown image/.test(err.message)) { showError(err.message); return; }
      reuseIntoPanel(rec, false,
        `#${rec.refine}, the source this was refined from, is no longer in the gallery, `
        + 'so the pass cannot be made again. The size is left as it was.');
      return;
    }
    if (!opening.valid()) return;
    // openRefine also retires pending lookups. This one has settled and must
    // not cancel its own deliberate open or clear a newer request's token.
    reusePassOpening = null;
    openRefine(source, rec);
  } finally {
    if (reusePassOpening === opening) reusePassOpening = null;
  }
}
