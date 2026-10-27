// AuraDream's page script: Then refine, a refine pass of every image a Generate or a grid makes.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// "Then refine: off" at the end of the Notify row under Generate - the one
// place in the panel with room and no height to add at 1400x900 - opens
// its dialog (#then-refine-dialog). With the dialog's tick on, a
// Generate or a Run grid carries `then_refine` - Mass refine's scale,
// strength and steps - and the server runs a pass of each image right after
// the cell that made it, as Mass refine would make it, under a run of its
// own (backend/then_refine.py, `JobManager._passes`). The server decides
// what it refuses and why; the page says the setting where it can be seen
// while it is on - the opener's words and the two buttons' labels - because
// it doubles every press. The tick starts off on every load; the scale and
// the strength are the Refine dialog's remembered ones (REFINE_KEY), so the
// three places that refine share what was last run on this checkpoint.

let trScale = REFINE_SCALE;

// No pass on this checkpoint: the tick is held with the reason, never hidden.
const thenRefineHeld = () => !refineSupported;
const thenRefineOn = () => els.thenRefine.checked && !thenRefineHeld();

// The strength the passes run at as the box says it: typed, else the one
// last run on this checkpoint, else the checkpoint's own (the server's).
function thenRefineStrengthText() {
  const typed = els.thenRefineStrength.value.trim();
  if (typed) return typed;
  const remembered = refinePrefs().strength?.[loadedModel];
  return remembered ? String(remembered) : String(refineDefaultStrength());
}

// What a press sends beside its request, or null with the tick off. A box
// the server cannot read goes as typed, and is refused in its words.
function thenRefinePayload() {
  if (!thenRefineOn()) return null;
  const strength = thenRefineStrengthText();
  const steps = els.thenRefineSteps.value.trim();
  return {
    scale: trScale,
    strength: Number.isFinite(+strength) && strength !== '' ? +strength : strength,
    steps: steps === '' ? null : Number.isInteger(+steps) ? +steps : steps,
  };
}

// Generate's and Run grid's words, the setting in them while it is on.
const submitLabel = (base) => `${base}${thenRefineOn() ? ' + refine' : ''}${busy ? ' (queue)' : ''}`;

function syncThenRefine() {
  if (!els.thenRefineScales.children.length) {
    for (const scale of REFINE_SCALES) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = `${scale}×`;
      button.onclick = () => chooseThenRefineScale(scale);
      els.thenRefineScales.append(button);
    }
  }
  const held = thenRefineHeld();
  els.thenRefineLabel.setAttribute('aria-disabled', String(held));
  els.thenRefine.setAttribute('aria-disabled', String(held));
  els.thenRefineLabel.title = held ? REFINE_NONE : 'A refine pass of every image right after the cell that made it';
  els.thenRefineStrength.placeholder = thenRefineStrengthText();
  const on = thenRefineOn();
  const strength = thenRefineStrengthText();
  const width = +els.width.value;
  const height = +els.height.value;
  const sizeAxis = els.gridSection.open && els.gridSize.value.trim();
  const size = !sizeAxis && width && height ? refineScaled(width, height, trScale) : null;
  [...els.thenRefineScales.children].forEach((button, i) => {
    const scale = REFINE_SCALES[i];
    button.classList.toggle('active', scale === trScale);
    button.setAttribute('aria-pressed', String(scale === trScale));
    const made = !sizeAxis && width && height ? refineScaled(width, height, scale) : null;
    button.title = made ? `${scale}×: ${width}×${height} becomes ${made[0]}×${made[1]}` : `${scale} times each image's size`;
  });
  els.thenRefineOpen.textContent = on ? `Refine ${trScale}× · ${strength}` : 'Then refine: off';
  els.thenRefineOpen.classList.toggle('on', on);
  els.thenRefineOpen.title = held ? `Then refine - ${REFINE_NONE}`
    : on ? `Each image is then refined at ${trScale}× and strength ${strength}; change or turn it off`
    : 'Follow each image Generate or Run grid makes with a refine pass';
  els.thenRefineHint.textContent = held ? REFINE_NONE
    : sizeAxis ? `Run grid refines every image at ${trScale}x its own size; the grid preview lists the pass sizes. Each pass keeps its image's prompt and seed.`
    : size && !refineFits(size) ? `${width}×${height} at ${trScale}× is ${size[0]}×${size[1]}, past what a refine pass renders.`
    : size ? `Each image is then refined at ${trScale}× to ${size[0]}×${size[1]} from its own prompt and seed; the passes join a run of their own.`
    : '';
  const said = on ? ` - then a refine pass of each image at ${trScale}× and strength ${strength}` : '';
  els.generate.textContent = submitLabel('Generate');
  els.gridRun.textContent = submitLabel('Run grid');
  // A held Generate keeps its reason as its title (syncSubmitButtons): the
  // status poll reaches here, and a checkpoint that only edits held Generate
  // under the words of a live one.
  const why = submitHold({ composer: true });
  els.generate.title = why || `Generate (Ctrl+Enter)${said}`;
  els.gridRun.title = why || `Run grid (Ctrl+Enter from a grid box)${said}`;
}

function chooseThenRefineScale(scale) {
  trScale = scale;
  const prefs = refinePrefs();
  prefs.scale = scale;
  remember(REFINE_KEY, JSON.stringify(prefs));
  syncThenRefine();
  schedulePreview();
}

// A strength typed here is the one last run on this checkpoint, as the
// Refine dialog's and Mass refine's are.
function rememberThenRefineStrength() {
  const text = els.thenRefineStrength.value.trim();
  if (!loadedModel || !SINGLE_STRENGTH.test(text)) return;
  const prefs = refinePrefs();
  const strengths = prefs.strength && typeof prefs.strength === 'object' ? prefs.strength : {};
  if (+text === refineDefaultStrength()) delete strengths[loadedModel];
  else strengths[loadedModel] = text;
  prefs.strength = strengths;
  remember(REFINE_KEY, JSON.stringify(prefs));
}

function initThenRefine() {
  const prefs = refinePrefs();
  if (REFINE_SCALES.includes(prefs.scale)) trScale = prefs.scale;
  els.thenRefine.checked = false;
  els.thenRefineOpen.onclick = () => {
    syncThenRefine();
    els.thenRefineDialog.showModal();
    els.thenRefine.focus({ preventScroll: true });
  };
  els.thenRefineClose.onclick = () => els.thenRefineDialog.close();
  // The backdrop closes it, as Escape does; a press on its own padding,
  // whose target is the dialog too, does not.
  els.thenRefineDialog.addEventListener('click', (event) => {
    if (event.target !== els.thenRefineDialog) return;
    const box = els.thenRefineDialog.getBoundingClientRect();
    const inside = event.clientX >= box.left && event.clientX <= box.right
      && event.clientY >= box.top && event.clientY <= box.bottom;
    if (!inside) els.thenRefineDialog.close();
  });
  // A held tick says why on a press, and stays off.
  els.thenRefine.addEventListener('click', (event) => {
    if (!thenRefineHeld()) return;
    event.preventDefault();
    els.thenRefineHint.textContent = '';
    els.thenRefineHint.textContent = REFINE_NONE;
  });
  // Each through the grid's preview, which counts and prices the passes
  // and syncs the fold (schedulePreview), as a size change does.
  els.thenRefine.addEventListener('change', schedulePreview);
  els.thenRefineStrength.addEventListener('input', schedulePreview);
  els.thenRefineStrength.addEventListener('change', rememberThenRefineStrength);
  els.thenRefineSteps.addEventListener('input', schedulePreview);
  syncThenRefine();
}
