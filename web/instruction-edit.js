// AuraDream's page script: Edit by instruction… - the capability the status
// answers, the image menu's entry, the dialog, its preview, submission,
// close and Reuse. Declarations only; state.js has the shared-global
// load-order contract.
'use strict';

// ------------------------------------------------------------ instruction edits

// An instruction edit changes one gallery image by a written instruction,
// through the one frozen recipe (FLUX.1 Kontext [dev], `kontext-v1`): the
// whole picture redrawn at the size the server resolves from the source's
// aspect, with no mask, strength, negative prompt, adapter or reference. The
// server decides everything that is not presentation - whether an edit can
// run here at all (the running backend's own gate, then the selected
// checkpoint), the output size, the recipe's steps and guidance, and the
// source's proof - and the page shows its answers. The dialog opens wherever
// the editor is held as well, and says why, with a way to Models; a source
// whose aspect the card never measured says so in the preview's place. It
// never writes the panel's prompt, negative or references, and Generate never
// runs an edit.

const EDIT_RECIPE = 'kontext-v1';
const EDIT_MODEL_ID = 'black-forest-labs/FLUX.1-Kontext-dev';
const EDIT_MODEL_NAME = 'FLUX.1 Kontext [dev]';
const EDIT_NO_INSTRUCTION = 'Write an instruction first.';
// What holds Edit while a submit is in flight - this dialog's or the panel's.
const EDIT_IN_FLIGHT = 'Wait for the last submit to be taken';
const EDIT_KEPT = 8;

// The status's answer: whether an edit can queue on this server now, and the
// server's words where it cannot (`instruction_edit_reason`). Before the
// first status nothing is claimed either way: the dialog's preview says it.
let instructionEditSupported = false;
let instructionEditReason = '';
// A checkpoint that edits but does not generate from text holds Generate
// with the server's words (`text_to_image_reason`), else ''.
let textToImageReason = '';

// The record the dialog is open on, null while it is closed; the edit a
// Reuse opened it from; the server's preview of the source (size, steps,
// guidance, availability) or the refusal of it in words.
let editSource = null;
let editFrom = null;
let editPreview = null;
let editProblem = null;
let editPreviewStale = true;
let editEpoch = 0;
let editTimer;
let editWaiting = false;
let editSending = false;
let editRan = false;
// Where the lightbox was when the dialog opened from it (lightboxPlace), and
// whether Open Models… closed it, which opens the picker instead.
let editReturn = null;
let editToModels = false;
// The steps and guidance the dialog wrote from the preview, so a later
// answer replaces only what the user has not typed over.
let editFilled = { steps: '', guidance: '' };

// What a close that ran nothing left - the instruction and the numbers - in
// page memory only, for the next open of that image on the same checkpoint,
// as the Inpaint and Outpaint dialogs keep theirs. A switch to the editor
// carries the instruction and seed (`carried`), not the numbers.
const editKept = new Map();

// The editor's name where the page names a checkpoint: off the catalogue,
// the id's last part would be all checkpointName has.
function editModelName(id) {
  return id === EDIT_MODEL_ID ? EDIT_MODEL_NAME : checkpointName(id);
}

// ------------------------------------------------------------- the capability

// The status's answer, as it lands (status.js): the menu's item and an open
// dialog follow it, and the dialog asks its preview again.
function syncEditCapability(defaults) {
  if (!defaults) return;
  const textHold = defaults.text_to_image_supported === false
    ? (typeof defaults.text_to_image_reason === 'string' && defaults.text_to_image_reason)
      || 'This checkpoint does not generate from text.'
    : '';
  if (textHold !== textToImageReason) {
    textToImageReason = textHold;
    syncSubmitButtons();
  }
  if (typeof defaults.instruction_edit_supported !== 'boolean') return;
  const supported = defaults.instruction_edit_supported === true;
  const reason = supported ? ''
    : (typeof defaults.instruction_edit_reason === 'string' && defaults.instruction_edit_reason)
      || 'Instruction editing is not available on this server.';
  if (supported === instructionEditSupported && reason === instructionEditReason) return;
  instructionEditSupported = supported;
  instructionEditReason = reason;
  if (!editSource) return;
  editPreviewStale = true;
  scheduleEditPreview(0);
  syncEditDialog();
}

// The menu's Edit by instruction…: live wherever the image is, so a press
// opens the dialog and the dialog says what holds it; held only where the
// image itself cannot be a source - in the recycle bin - with that reason.
function syncEditTool(button, rec) {
  const binned = !!rec?.binned_at;
  const title = binned ? 'In the recycle bin: restore it first to edit it by instruction'
    : instructionEditSupported
      ? `Edit by instruction: write a change to #${rec.id} and ${EDIT_MODEL_NAME} redraws it`
      : `Edit by instruction: opens the dialog, which says what holds it - ${instructionEditReason
        || 'instruction editing is not available on this server'}`;
  button.setAttribute('aria-disabled', String(binned));
  button.title = title;
  button.setAttribute('aria-label', title);
  button.dataset.tip = binned ? 'In the recycle bin' : 'Edit by instruction…';
}

// A card's, a result's or the lightbox's menu item: one in the recycle bin
// says so where the gallery says things, and opens nothing.
function editFromCard(rec) {
  if (!rec || refuseBinned(rec, 'edit')) return;
  void openEdit(rec);
}

// ------------------------------------------------------------- opening

// Opens the dialog on `rec`, the image to edit. `from` is an edit to start
// from instead - Reuse on one - with its instruction, seed, steps and
// guidance. Otherwise an empty instruction, a random seed and the recipe's
// steps and guidance from the preview - or, after a close that left an
// instruction unsent on this image on this checkpoint, that, which it says.
// The lightbox, where it was opened from, closes for it and comes back on a
// close that ran nothing. Answers whether it opened.
async function openEdit(rec, from = null) {
  if (!rec) return false;
  if (refuseBinned(rec, 'edit')) return false;
  editReturn = lightboxPlace();
  if (els.lightbox.open) closeLightbox();
  showError('');
  clearTimeout(editTimer);
  editEpoch++;
  const kept = !from ? editKept.get(rec.id) : null;
  const draft = kept && kept.model === loadedModel ? kept : null;
  editKept.delete(rec.id);
  editSource = rec;
  editFrom = from;
  editPreview = null;
  editProblem = null;
  editPreviewStale = true;
  editRan = false;
  editToModels = false;
  editFilled = { steps: '', guidance: '' };
  els.editTitle.textContent = from ? `Edit #${rec.id} as #${from.id} was made` : `Edit #${rec.id} by instruction`;
  const src = rec.url ?? rec.thumb_url;
  if (els.editImage.getAttribute('src') !== src) els.editImage.src = src;
  els.editImage.alt = `#${rec.id}, the image to edit`;
  els.editStage.style.setProperty('--ip-ratio', String(rec.width / rec.height));
  writeEditSourceLine(rec);
  els.editReused.hidden = !from;
  els.editReused.textContent = from
    ? `Reusing #${from.id}'s instruction, seed ${from.seed}, ${from.steps} steps and guidance ${from.guidance_scale}` : '';
  const values = from
    ? { instruction: from.prompt ?? '', seed: String(from.seed ?? ''), steps: String(from.steps ?? ''), guidance: String(from.guidance_scale ?? '') }
    : draft ? draft.values : { instruction: '', seed: '', steps: '', guidance: '' };
  els.editInstruction.value = '';
  els.editSeed.value = values.seed;
  els.editSteps.value = values.steps;
  els.editGuidance.value = values.guidance;
  for (const el of [els.editSeed, els.editSteps, els.editGuidance]) el.removeAttribute('aria-invalid');
  editFailed('');
  els.editPreview.textContent = '';
  els.editPreview.classList.remove('warn');
  els.editPreview.hidden = false;
  // Scrolled to the top only as it opens.
  if (!els.editDialog.open) {
    els.editDialog.showModal();
    els.editDialog.scrollTop = 0;
  }
  // Focus on the instruction, what the dialog is opened to write; a carried
  // one written as typing writes it, so Ctrl+Z takes it back out.
  els.editInstruction.focus({ preventScroll: true });
  if (values.instruction) replaceBoxText(els.editInstruction, values.instruction);
  // Shown or not only as the dialog opens, as the other dialogs' draft lines:
  // hiding it on input would move every control below it under the pointer.
  els.editDraft.hidden = !draft;
  els.editDraft.textContent = !draft ? ''
    : draft.carried ? `Your unsent instruction and seed for #${rec.id}, carried to ${EDIT_MODEL_NAME}.`
      : `Your unsent instruction for #${rec.id}, as you left it.`;
  syncEditDialog();
  scheduleEditPreview(0);
  return true;
}

// "#N · 768×512 → 576×384 · seed S · checkpoint": the source, and the size
// the server resolved for it once the preview has answered.
function writeEditSourceLine(rec, gone = null) {
  const out = editPreview && editSource === rec ? ` → ${editPreview.width}×${editPreview.height}` : '';
  els.editSourceLine.textContent = [
    `#${rec.id} · ${rec.width}×${rec.height}${out}`, `seed ${rec.seed}`, checkpointName(rec.model_id),
    rec.edit ? `itself an edit of #${rec.edit.source ?? '?'}` : '',
    gone ? `#${rec.id} (${gone === 'binned' ? 'in the recycle bin' : 'deleted'})` : '',
  ].filter(Boolean).join(' · ');
}

// A refusal of the source itself - deleted or binned from another tab, or
// here, while the dialog was open - in the page's words, as the other
// dialogs say it; any other refusal as the server said it.
function editSourceGone(message, source = editSource) {
  let gone = sourceGoneKind(message, source);
  const edit = /^Instruction-edit source #(\d+) is missing or in the recycle bin\.?$/.exec(message ?? '');
  if (!gone && edit && source && +edit[1] === source.id) {
    gone = source.binned_at || binnedIds.has(source.id) ? 'binned' : 'deleted';
  }
  if (!gone) return message;
  if (editSource === source) writeEditSourceLine(source, gone);
  return sourceGoneWords(source, gone, 'edit');
}

// ------------------------------------------------------------- the preview

function scheduleEditPreview(delay = 250) {
  clearTimeout(editTimer);
  editTimer = setTimeout(() => { void previewEdit(); }, delay);
}

// The server's reading of the source: the size the recipe resolves from its
// aspect, the recipe's steps and guidance, and whether an edit can queue here
// now and why not. A source the recipe refuses - an aspect it does not take,
// a source gone - is said in the preview's place, and holds Edit.
async function previewEdit() {
  const src = editSource;
  if (!src || !els.editDialog.open) return;
  const epoch = ++editEpoch;
  let answer = null;
  let problem = null;
  try {
    answer = await api('/api/instruction-edit/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: src.id, recipe: EDIT_RECIPE }),
    });
  } catch (err) {
    problem = editSourceGone(err.message, src);
  }
  if (epoch !== editEpoch || editSource !== src) return;
  editPreview = answer;
  editProblem = problem;
  editPreviewStale = false;
  if (answer) {
    // The recipe's numbers into boxes the user has not typed in.
    const steps = String(answer.steps);
    const guidance = String(answer.guidance_scale);
    if (els.editSteps.value === editFilled.steps) els.editSteps.value = steps;
    if (els.editGuidance.value === editFilled.guidance) els.editGuidance.value = guidance;
    editFilled = { steps, guidance };
    els.editSteps.placeholder = steps;
    els.editGuidance.placeholder = guidance;
    writeEditSourceLine(src);
  }
  syncEditDialog();
}

// ------------------------------------------------------------- the dialog's state

// Why no edit can queue here now: the status's words, else the preview's -
// not its source's own (`editSourceReason`), which no Models choice changes.
function editCapabilityReason() {
  if (instructionEditReason) return instructionEditReason;
  if (!editPreview || editPreview.available) return '';
  const reason = editPreview.reason || 'Instruction editing is not available on this server.';
  return reason === editPreview.source_reason ? '' : reason;
}

// Why this source cannot be edited here though the server could edit
// another: an aspect the card never measured, in the server's words. Said in
// the preview's place, at its size, so nothing moves under the pointer.
function editSourceReason() {
  return (editPreview && typeof editPreview.source_reason === 'string' && editPreview.source_reason) || '';
}

// The numbers as the server takes them, or what is wrong with one.
function editNumberProblem() {
  const seed = els.editSeed.value.trim();
  if (seed !== '' && !(Number.isSafeInteger(+seed) && +seed >= 0)) {
    return { el: els.editSeed, text: `Seed must be a whole number from 0 to ${Number.MAX_SAFE_INTEGER}, or empty for a random one` };
  }
  const steps = els.editSteps.value.trim();
  if (steps !== '' && !(Number.isInteger(+steps) && +steps >= 1 && +steps <= 100)) {
    return { el: els.editSteps, text: 'Steps must be a whole number from 1 to 100' };
  }
  const guidance = els.editGuidance.value.trim();
  if (guidance !== '' && !(Number.isFinite(+guidance) && +guidance >= 0 && +guidance <= 20)) {
    return { el: els.editGuidance, text: 'Guidance must be a number from 0 to 20' };
  }
  return null;
}

// Held, with the reason as its title: first what holds Generate, a submit in
// flight, then what holds the editor here, then the preview's refusal, the
// numbers and an empty instruction.
function editHold() {
  return submitHold() || (submitting ? EDIT_IN_FLIGHT : '')
    || editCapabilityReason()
    || editProblem
    || (!editPreview ? 'Wait for the preview of the source' : '')
    || editSourceReason()
    || editNumberProblem()?.text
    || (!els.editInstruction.value.trim() ? EDIT_NO_INSTRUCTION : '');
}

function syncEditRun() {
  if (!els.editRun || !editSource) return;
  const problem = editHold();
  els.editRun.setAttribute('aria-disabled', String(!!problem));
  els.editRun.title = problem || 'Run the edit (Ctrl+Enter)';
}

function syncEditDialog() {
  const src = editSource;
  if (!src) return;
  const reason = editCapabilityReason();
  els.editUnavailable.hidden = !reason;
  els.editUnavailableText.textContent = reason;
  const selected = editPreview?.selected_model_id ?? loadedModel;
  els.editCheckpoint.textContent = `Runs on ${EDIT_MODEL_NAME} (${EDIT_RECIPE})`
    + (selected && selected !== EDIT_MODEL_ID ? ` · the server has ${editModelName(selected)} selected` : '');
  els.editCheckpoint.title = EDIT_MODEL_ID;
  const sourceReason = editSourceReason();
  if (editProblem || sourceReason) {
    els.editPreview.classList.add('warn');
    els.editPreview.textContent = editProblem || sourceReason;
  } else if (editPreview) {
    els.editPreview.classList.remove('warn');
    const seed = els.editSeed.value.trim();
    els.editPreview.textContent = `1 image · ${editPreview.width} × ${editPreview.height}`
      + ` · ${els.editSteps.value.trim() || editPreview.steps} steps`
      + ` · guidance ${els.editGuidance.value.trim() || editPreview.guidance_scale}`
      + ` · ${seed === '' ? 'a random seed' : `seed ${seed}`}`;
  } else {
    els.editPreview.classList.remove('warn');
    els.editPreview.textContent = '';
  }
  els.editRun.textContent = editSending ? 'Sending…' : busy ? 'Edit (queue)' : 'Edit';
  syncEditRun();
}

function editChanged() {
  for (const el of [els.editSeed, els.editSteps, els.editGuidance]) el.removeAttribute('aria-invalid');
  els.editPreview.hidden = false;
  editFailed('');
  syncEditDialog();
}

// A refusal or a held press, said in the preview's place and at its size, so
// Edit does not move under the pointer.
function editFailed(message) {
  els.editError.hidden = !message;
  els.editError.textContent = message || '';
  if (message) els.editPreview.hidden = true;
  else els.editPreview.hidden = false;
}

// Enter in a one-line box runs, as the other dialogs' do; the instruction
// keeps its Enter for a new line.
function editKeydown(event) {
  if (keyboardEventHeld(event) || keyboardCommandModifier(event)) return;
  if (event.key === 'Enter' && !event.shiftKey && event.target.matches?.('#edit-dialog input[type="number"]')) {
    event.preventDefault();
    if (!event.repeat) void runEdit();
  }
}

// ------------------------------------------------------------- submission

function editRequest() {
  const num = (el) => (el.value.trim() === '' ? undefined : +el.value);
  return {
    source: editSource.id,
    recipe: EDIT_RECIPE,
    prompt: els.editInstruction.value.trim(),
    width: editPreview?.width,
    height: editPreview?.height,
    seed: num(els.editSeed) ?? null,
    steps: num(els.editSteps),
    guidance_scale: num(els.editGuidance),
  };
}

// Edit: asks the preview again where it is stale, then queues the job on its
// own route like the other dialogs' passes, and the dialog closes once the
// server has taken it. A held press says why in the preview's place.
async function runEdit() {
  if (!editSource || !els.editDialog.open || editWaiting) return;
  if (submitting) { editFailed(`${EDIT_IN_FLIGHT}.`); return; }
  const source = editSource;
  if (editPreviewStale || (!editPreview && !editProblem)) {
    clearTimeout(editTimer);
    editWaiting = true;
    try { await previewEdit(); } finally { editWaiting = false; }
    if (editSource !== source || !els.editDialog.open || submitting) return;
  }
  syncEditDialog();
  const held = editHold();
  if (held) {
    const bad = editNumberProblem();
    if (bad && held === bad.text) {
      bad.el.setAttribute('aria-invalid', 'true');
      bad.el.focus({ preventScroll: true });
    }
    editFailed(/[.!?]$/.test(held) ? held : `${held}.`);
    if (serverOffline) void refreshStatus();
    return;
  }
  const body = editRequest();
  editSending = true;
  let accepted = false;
  const settle = () => {
    editSending = false;
    if (editSource) syncEditDialog();
  };
  const sent = startJob('/api/instruction-edit', body, {
    panel: false,
    slot: els.editError,
    placeholder: {
      src: source.url ?? source.thumb_url,
      width: body.width,
      height: body.height,
      label: `Editing #${source.id}: ${snippet(body.prompt, 48)}`,
    },
    onAccepted: () => {
      accepted = true;
      settle();
      if (editSource !== source || !els.editDialog.open) return;
      editRan = true;
      els.editDialog.close();
      requestAnimationFrame(() => {
        if (document.activeElement && document.activeElement !== document.body) return;
        handJobFocus(els.newImage.hidden ? els.generate : els.newImage);
      });
    },
  });
  if (editSending) els.editRun.textContent = 'Sending…';
  await sent;
  if (accepted) return;
  settle();
  const refused = els.editError.textContent;
  if (refused) {
    const said = editSourceGone(refused, source);
    if (said !== refused) editFailed(said);
    else editFailed(refused);
  }
  if (!els.editDialog.open && els.editError.textContent) showError(els.editError.textContent);
}

// ------------------------------------------------------------- closing

function closeEdit() {
  if (!editSending) els.editDialog.close();
}

function editCancelEvent(event) {
  if (editSending) event.preventDefault();
}

// Open Models…: the picker opens once the dialog has closed - its draft kept
// - on the editor's row, chosen and in view, and switches nothing by itself.
function editOpenModels() {
  if (editSending) return;
  editToModels = true;
  els.editDialog.close();
}

// Closed by Edit, Cancel, the cross, Escape or Open Models…. A close that
// ran nothing keeps what was typed for the next open of the same image on
// this checkpoint - in page memory - and, opened from the lightbox, gives the
// lightbox back where it was; where it ran, focus goes to the source's card.
function editClosed() {
  const src = editSource;
  const ran = editRan || editSending;
  if (src && !ran) {
    const values = {
      instruction: els.editInstruction.value, seed: els.editSeed.value,
      steps: els.editSteps.value, guidance: els.editGuidance.value,
    };
    const typed = values.instruction.trim() !== '' || values.seed.trim() !== ''
      || values.steps !== editFilled.steps || values.guidance !== editFilled.guidance;
    if (typed && !editFrom) {
      editKept.set(src.id, { model: loadedModel, values });
      while (editKept.size > EDIT_KEPT) editKept.delete(editKept.keys().next().value);
    }
  }
  editSource = null;
  editFrom = null;
  editPreview = null;
  editProblem = null;
  editEpoch++;
  clearTimeout(editTimer);
  const back = editReturn;
  editReturn = null;
  if (editToModels) {
    editToModels = false;
    showModelPickerAt(EDIT_MODEL_ID);
    return;
  }
  if (!back || els.lightbox.open) return;
  if (!ran) { returnToLightbox(back); return; }
  // The source's card (a Reuse opens on the source, not the record shown).
  const id = src ? src.id : back.rec.id;
  const card = [...document.querySelectorAll(`#gallery .card[data-id="${id}"] img`)].find(isShown);
  handJobFocus(card ?? els.generate);
}

// Images binned or deleted (forgetImages) and restored (unforgetImages): a
// draft kept on one deleted for good goes, and an open dialog on one of them
// asks its preview again, which says where the source went or that it is back.
function editSourcesMoved(ids, { deleted = false } = {}) {
  const moved = new Set(ids);
  if (deleted) for (const id of moved) editKept.delete(id);
  if (editSource && moved.has(editSource.id)) {
    editPreviewStale = true;
    scheduleEditPreview(0);
  }
}

// A checkpoint switch: what was kept for one checkpoint is not offered on
// another - except onto the editor, which an instruction is written for: a
// switch to it (Open Models… then Switch & load) keeps each unsent
// instruction and seed, and the recipe's steps and guidance come from the
// next preview. An open dialog asks again whether an edit can run now.
function editCheckpointChanged(model = loadedModel) {
  for (const [id, kept] of editKept) {
    const { instruction, seed } = kept.values;
    if (model !== EDIT_MODEL_ID || (!instruction.trim() && !seed.trim())) editKept.delete(id);
    else editKept.set(id, { model, carried: true, values: { instruction, seed, steps: '', guidance: '' } });
  }
  if (!editSource) return;
  editPreviewStale = true;
  scheduleEditPreview(0);
  syncEditDialog();
}

// ------------------------------------------------------------- Reuse

// An edit holds its source's id: the source asked for - the bin's rows too,
// so one there is offered a Restore rather than said to be gone - and the
// dialog opened on it with the edit's instruction, seed, steps and guidance.
// Never the panel: an instruction means nothing without its source.
async function reuseEdit(rec) {
  const id = rec.edit?.source;
  if (id == null) { showError(executionReuseReason(rec), false, { slot: els.galleryError }); return; }
  let source;
  try {
    source = await api(`/api/images/${id}?binned=true`);
  } catch (err) {
    const said = /unknown image/.test(err.message)
      ? `#${id}, the source this was edited from, is no longer in the gallery, so the edit cannot be reopened.`
      : err.message;
    showError(said, false, { slot: els.galleryError });
    return;
  }
  if (source.binned_at) { showError(sourceGoneWords(source, 'binned', 'edit'), true, { slot: els.galleryError }); return; }
  await openEdit(source, rec);
}

// ------------------------------------------------------------- lineage

// An edit's source as the page names it: a press opens it in the lightbox
// with the edit after it, unless it has gone - "(deleted)", or "(in the
// recycle bin)" where a Restore brings it back (gallery.js's editGoneWord).
function editSourceLink(rec) {
  const id = rec.edit?.source;
  if (id == null) return 'an unresolved source';
  if (rec.edit_missing) return `#${id} (${editGoneWord(rec)})`;
  return factButton(`#${id}`, `Open #${id}, the image this was edited from; → comes back to the edit`,
    () => { void openEditSource(rec); });
}

async function openEditSource(rec) {
  const id = rec.edit?.source;
  if (id == null) return;
  let source = [...lbList, ...galleryRecords, ...viewerRecords].find((r) => r.id === id);
  if (!source) {
    try {
      source = await api(`/api/images/${id}`);
    } catch (err) {
      showError(/unknown image/.test(err.message) ? `#${id} is no longer in the gallery.` : err.message);
      return;
    }
  }
  openLightbox(source, [source, rec]);
}
