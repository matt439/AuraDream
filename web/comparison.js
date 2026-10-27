// AuraDream's page script: prompt-version comparisons.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// ------------------------------------------------------------------ compare

// The versions a comparison can take: the checked text always, a reordering -
// the rules' own, which is exact, ahead of the critic's - the critic's
// rewrite, and the user's own version. Whether the critic's reordering may
// stand is the server's to decide (`critic.validate`), and the page takes what
// it kept: where Check placed every tag itself, only Check's own order; where
// it could not place some tag (never looked up, or unknown to the tag index),
// any order the same rule finds nothing wrong with. The user's own version is
// what isolates order where neither layer can offer a reorder: a sentence on
// FLUX, where the critic's `reordered` rarely survives its check, or a
// checkpoint whose guide states no order. A version that is the original word
// for word, or another one's, is left out: the server would render it twice.
function compareVariants() {
  if (!analysisResult) return [];
  const { source, body } = analysisResult;
  const variants = [{ label: 'original', name: 'Original', prompt: source }];
  const seen = new Set([source]);
  for (const [label, name, text] of [
    ['reordered', 'Reordered', body.reordered ?? critiqueResult?.body.reordered],
    ['rewrite', 'Rewrite', critiqueResult?.body.rewrite],
    ['yours', 'Your version', els.compareOwn.value],
  ]) {
    const prompt = (text ?? '').trim();
    if (!prompt || seen.has(prompt)) continue;
    seen.add(prompt);
    variants.push({ label, name, prompt });
  }
  return variants;
}

const comparedVariants = () =>
  compareVariants().filter((v) => v.label === 'original' || compareChoice[v.label]);

// Typing a version of your own: the ticks at once, the server's count and
// estimate once the typing pauses. The box no longer holds only what a Use
// put there, so its note goes.
function compareOwnChanged() {
  els.compareOwnNote.hidden = true;
  renderCompare(400);
}

// The checked prompt as the starting point for a version of your own - to
// move its clauses about, say - rather than typing it out again.
function compareOwnFromPrompt() {
  if (!analysisResult) return;
  els.compareOwn.value = analysisResult.source;
  els.compareOwn.focus();
  compareOwnChanged();
}

// A version's text outlined where the panel shows it while its Compare tick
// is pointed at or focused: the labels name the versions, the critic's boxes
// and the Order section hold them.
function linkVersion(version, on) {
  for (const el of els.promptAnalysis.querySelectorAll(`[data-version="${version}"]`)) el.classList.toggle('linked', on);
}

function renderCompare(previewDelay = 0) {
  const variants = compareVariants();
  // A label replaced under the pointer gets no pointerleave.
  for (const el of els.promptAnalysis.querySelectorAll('.linked')) el.classList.remove('linked');
  els.compareOptions.replaceChildren();
  for (const variant of variants) {
    const label = document.createElement('label');
    label.className = 'check';
    label.title = variant.prompt;
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.dataset.variant = variant.label;
    box.checked = variant.label === 'original' || compareChoice[variant.label];
    // The row every other one is compared against - and why it cannot be
    // unticked, which "Original, checked, unavailable" alone never said.
    box.disabled = variant.label === 'original';
    if (box.disabled) box.setAttribute('aria-description', 'Always runs: the other versions are compared against it');
    box.onchange = () => {
      compareChoice[variant.label] = box.checked;
      syncCompareRun();
      scheduleComparePreview(0);
    };
    for (const [type, on] of [['pointerenter', true], ['pointerleave', false], ['focusin', true], ['focusout', false]]) {
      label.addEventListener(type, () => linkVersion(variant.label, on));
    }
    label.append(box, ` ${variant.name}`);
    els.compareOptions.appendChild(label);
  }
  syncCompareRun();
  scheduleComparePreview(previewDelay);
}

function syncCompareRun() {
  const count = comparedVariants().length;
  els.compareRun.disabled = count < 2 || analysisStale;
  // Held like Generate and Run grid - the editor on the card, no server -
  // with their reason as its title (submitHold), which only a press said.
  const why = submitHold({ composer: true });
  els.compareRun.setAttribute('aria-disabled', String(!!why || submitting));
  els.compareRun.title = analysisStale ? STALE_REASON
    : count < 2 ? 'Needs a second version of the prompt to compare against'
    : why || 'One grid: each version of the prompt on the same seeds, then the sheet with a row per version';
}

// The grid's own payload - the panel's settings and seed - with every axis
// emptied: the versions are the one thing that varies, crossed with the
// seeds. Seeds is the total per version; the comparison's own Batch groups
// those seeds, so the grid's seed count is the number of batches. Every
// offered batch (1, 2, 4) divides both totals (4, 8). Read both selects here:
// a browser restoring form state sets them without a change event. Nothing is
// written into the prompt box or the grid's variables, which are the user's
// and remembered.
function comparePayload(variants) {
  const batch = +els.compareBatch.value;
  return {
    ...gridPayload(),
    num_images: batch,
    prompt: analysisResult.source,
    prompt_axis: '',
    steps_axis: '',
    cfg_axis: '',
    mu_axis: '',
    lora_scale_axis: '',
    seed_axis: '',
    seeds: +els.compareSeeds.value / batch,
    resume: false,
    budget_minutes: null,
    prompt_variants: variants.map(({ label, prompt }) => ({ label, prompt })),
  };
}

// A comparison runs each version verbatim (grid.resolve_prompts substitutes
// nothing), so with the grid open, a `{name}` its variables declare was
// rendered as braces into every image - a prompt Generate refuses in the
// same state. The refusal names the first such name, in a version or the
// negative; braces the grid does not declare (a JSON prompt's) pass.
function compareUnfilled(variants) {
  const declared = declaredGridVariables();
  if (!declared.size) return null;
  const negative = els.negative.disabled ? '' : els.negative.value;
  for (const text of [...variants.map((v) => v.prompt), negative]) {
    for (const [hole] of text.matchAll(new RegExp(GRID_PLACEHOLDER.source, 'g'))) {
      if (declared.has(hole.slice(1, -1))) {
        return `The comparison runs each version as written; fill or remove ${hole} first.`;
      }
    }
  }
  return null;
}

function scheduleComparePreview(delay = 250) {
  clearTimeout(comparePreviewTimer);
  if (!analysisResult || els.promptAnalysis.hidden) return;
  comparePreviewTimer = setTimeout(previewComparison, delay);
}

// The server's own count and estimate, as the grid section's preview is, with
// the same token against a slow reply landing over a newer one.
async function previewComparison() {
  const epoch = ++comparePreviewEpoch;
  const variants = comparedVariants();
  els.comparePreview.classList.remove('warn');
  if (variants.length < 2) {
    els.comparePreview.textContent = promptCritiqueSupported
      ? 'Only the original so far: write a version of your own above, or ask the critic for a rewrite.'
      : 'Only the original so far: write a version of your own above to compare it against.';
    return;
  }
  const unfilled = compareUnfilled(variants);
  if (unfilled) {
    els.comparePreview.classList.add('warn');
    els.comparePreview.textContent = unfilled;
    return;
  }
  try {
    const payload = comparePayload(variants);
    const p = await api('/api/grid/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (epoch !== comparePreviewEpoch) return;
    els.comparePreview.classList.toggle('warn', !!p.duplicates);
    // The grid's summary counts batch starts as seeds; the comparison's
    // Seeds select counts every image seed. Describe the request just sent,
    // so changing a select while it runs cannot relabel an earlier reply.
    els.comparePreview.textContent = `${plural(p.images, 'image')} · ${plural(variants.length, 'version')}`
      + ` × ${plural(payload.seeds * payload.num_images, 'seed')} · batch ${payload.num_images}`
      + (p.seconds ? ` · about ${humanTime(p.seconds)}` : '')
      + (p.duplicates ? ` · ${p.duplicates} would be identical to another cell` : '');
  } catch (err) {
    if (epoch !== comparePreviewEpoch) return;
    els.comparePreview.classList.add('warn');
    els.comparePreview.textContent = err.message;
  }
}

// One grid, then its submission's sheet: startJob keeps the job id and opens
// it when the stream reports done, with one row per version as written. Not
// the panel's grid (`panel: false`): a blank Seed box stays blank. Its base
// was written back for a Resume no comparison takes, and every plain
// Generate after it repeated that seed, said only in the run hint's grey.
function runComparison() {
  if (refuseComposerExecution()) return undefined;
  const variants = comparedVariants();
  if (!analysisResult || variants.length < 2 || analysisStale) return undefined;
  // The comparison supplies its own Batch and Seeds, and ignores Repeat,
  // the panel's grid Seeds and its Budget.
  if (refuseBadNumber(panelNumberInputs({ batch: false, runs: false }))) return undefined;
  const unfilled = compareUnfilled(variants);
  if (unfilled) { showError(unfilled); return undefined; }
  return startJob('/api/grid', comparePayload(variants), { sheet: true, panel: false });
}
