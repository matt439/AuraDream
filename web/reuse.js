// Local Diffusion's page script: reusing image settings and undoing that reuse.
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
function reuse(rec) {
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
    owned: { ...owned }, reference: [...referenceIds], lora: els.lora.value, loraScale: els.loraScale.value,
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
  const note = reuseNote;
  if (!note || JSON.stringify(reuseFields()) === JSON.stringify(note.after)) return;
  reuseNote = null;
  note.button?.remove();
}

// `name` is what the note calls the record - a gallery id, or a PNG's file
// name for Load settings from PNG (png-settings.js) - and `sampling` false
// writes only the prompt pair and the seed, as `fillPanel` takes it: an
// A1111 block's steps and CFG are another sampler's.
function reuseIntoPanel(rec, withSize, note = '', { name = `#${rec.id}`, sampling = true } = {}) {
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
  referenceIds = state.reference;
  els.lora.value = state.lora;
  els.loraScale.value = state.loraScale;
  promptReplaced();
  // An expansion's English, where the prompt put back is the one it
  // translates, as Reset's Undo brings it back.
  const t = state.translation;
  if (t && els.prompt.value === t.source) showPromptTranslation(t.source, t.text);
  syncSize();
  syncCfg();
  syncConditioning();
  syncRuns();
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
  let source;
  try {
    source = await api(`/api/images/${rec.refine}`);
  } catch (err) {
    if (!/unknown image/.test(err.message)) { showError(err.message); return; }
    reuseIntoPanel(rec, false,
      `#${rec.refine}, the source this was refined from, is no longer in the gallery, `
      + 'so the pass cannot be made again. The size is left as it was.');
    return;
  }
  openRefine(source, rec);
}
