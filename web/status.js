// Local Diffusion's page script: /api/status, the profile defaults and the hints. Declarations only; state.js has
// the load-order contract these files share.
'use strict';

// ------------------------------------------------------------------- status

// Seed the controls from the server's profile for the loaded checkpoint. Turbo
// wants 8 steps / CFG 0 / mu 1.15; FLUX wants 28 / 3.5 and no mu at all, since
// it derives its own resolution-dependent shift. Each control follows the
// profile until the user touches *that* control; the checkpoint cannot change
// under a running server, so re-applying to an untouched box is idempotent.
let lastDefaults = null;
function applyDefaults(d) {
  if (!d || d.steps === undefined) return;
  lastDefaults = d;
  if (!owned.steps) els.steps.value = d.steps;
  if (!owned.cfg) els.cfg.value = d.guidance_scale;
  if (!owned.mu) els.mu.value = d.mu ?? '';
  syncCfg();
}

// What steps/CFG/mu mean depends on the checkpoint - 8 steps at CFG 0 is right
// for Turbo and ruinous for FLUX - so the tooltips come from the same server
// profile as the values, and unlike those values they keep following it: the
// user owning the controls does not make the explanation stale.
function applyHints(hints) {
  if (!hints) return;
  const tips = {
    steps: els.stepsTip, guidance_scale: els.cfgTip, mu: els.muTip, true_cfg: els.trueCfgTip,
    prompt: els.promptTip, size: els.sizeTip, refine: els.refineStrengthTip,
  };
  for (const [key, el] of Object.entries(tips)) {
    if (hints[key]) el.textContent = hints[key];
  }
}

// The rest of the guide: the size range the checkpoint's authors vouch for,
// which the size line compares the current size against, and the example
// prompts. Both follow the server the way the tooltips do.
function applyGuide(g) {
  if (!g) return;
  const range = Array.isArray(g.pixels) && g.pixels.length === 2 ? g.pixels : null;
  const word = g.pixels_word || 'documented';
  const aspects = Array.isArray(g.aspects) && g.aspects.length ? g.aspects : null;
  const resolutions = Array.isArray(g.resolutions) && g.resolutions.length ? g.resolutions : null;
  if (JSON.stringify(range) !== JSON.stringify(sizeRange) || word !== sizeRangeWord
      || JSON.stringify(aspects) !== JSON.stringify(sizeAspects)
      || JSON.stringify(resolutions) !== JSON.stringify(guideResolutions)) {
    sizeRange = range;
    sizeRangeWord = word;
    sizeAspects = aspects;
    guideResolutions = resolutions;
    syncSize();
  }
  if (Array.isArray(g.examples)) examples = g.examples;
  const helper = g.tag_helper || null;
  if (JSON.stringify(helper) !== JSON.stringify(tagHelper)) {
    tagHelper = helper;
    syncTagHelper();
  }
}

function syncMu() {
  els.mu.disabled = !muSupported;
  els.muField.classList.toggle('disabled', !muSupported);
  els.gridMu.disabled = !muSupported;
  els.gridMuField.classList.toggle('disabled', !muSupported);
  if (!muSupported) {
    els.mu.value = '';
    els.gridMu.value = '';
  }
}

// The editor's two actions on the card - ERNIE's rewriter and the prompt
// critic - and what either holds while it runs: the other one, Unload, and
// the submit buttons (aria-disabled, as setSubmitting holds them). Both wait
// for the image queue, so a job queued or running holds them in turn.
function syncPromptExpansion() {
  const editing = expandingPrompt || critiquing;
  els.expandPrompt.hidden = !promptExpansionSupported;
  els.expandPrompt.disabled = editing || busy;
  els.expandPrompt.textContent = expandingPrompt ? 'Expanding…' : 'Expand';
  els.unload.disabled = editing || busy;
  els.generate.setAttribute('aria-disabled', String(editing || submitting));
  els.gridRun.setAttribute('aria-disabled', String(editing || submitting));
  syncCritic();
}

function clearPromptTranslation() {
  translatedSource = null;
  els.promptTranslation.value = '';
  els.promptTranslationField.hidden = true;
}

// Every programmatic write to the prompt box - an example, Reuse, Reset, a tag
// chip, an accepted suggestion - fires no `input` event, so what describes the
// old text is told here instead: its translation goes, and the prompt check
// marks itself stale. Expand's write keeps the translation it brings, so it
// tells the check alone.
function promptReplaced() {
  clearPromptTranslation();
  syncAnalysisStale();
}

function showPromptTranslation(source, translation) {
  translatedSource = source;
  els.promptTranslation.value = translation;
  els.promptTranslationField.hidden = false;
}

// Anything the user picks - by hand or through Reuse - takes that control off
// the profile. /api/status reads CUDA memory, which stalls behind a model load,
// so its reply can land minutes late and used to wipe steps/CFG/mu long after
// the page had settled.
function own(...keys) { keys.forEach((key) => { owned[key] = true; }); }
function ownControls() { own('steps', 'cfg', 'mu', 'trueCfg'); }

async function refreshStatus() {
  if (statusPending) return;  // a stalled poll must not queue more behind it
  statusPending = true;
  try {
    const s = await api('/api/status');
    const previousModel = loadedModel;
    loadedModel = s.model_id ?? loadedModel;
    if (loadedModel !== previousModel) {
      syncReuseButtons();
      // The gallery is scoped to the checkpoint server-side, so a server
      // restarted under another one while this page stayed open is showing
      // the old one's images: ask again. The first answer is not a change -
      // the page loaded under this checkpoint's scope already.
      if (previousModel && !els.allModels.checked) { resetGallery(); loadRuns(); }
      // The check read another checkpoint's tokenizers and guide: none of it
      // is about this one.
      if (analysisResult && analysisResult.body.model_id !== loadedModel) clearAnalysis();
    }
    applyDefaults(s.defaults);
    // After the profile, so a remembered value set on this checkpoint wins
    // over it and a value set on another is left to the profile.
    if (loadedModel) restoreSampling(loadedModel);
    applyHints(s.hints);
    applyGuide(s.guide);
    if (typeof s.prompt_expansion_supported === 'boolean'
        && s.prompt_expansion_supported !== promptExpansionSupported) {
      promptExpansionSupported = s.prompt_expansion_supported;
      syncPromptExpansion();
    }
    // Ready flips once, when a first critique has downloaded the critic; the
    // line under its button then stops promising the download.
    if (typeof s.prompt_critique_supported === 'boolean'
        && (s.prompt_critique_supported !== promptCritiqueSupported
            || !!s.prompt_critique_ready !== promptCritiqueReady
            || JSON.stringify(s.prompt_critic ?? null) !== JSON.stringify(criticStatus))) {
      promptCritiqueSupported = s.prompt_critique_supported;
      promptCritiqueReady = !!s.prompt_critique_ready;
      criticStatus = s.prompt_critic ?? null;
      syncCritic();
    }
    if (s.defaults && typeof s.defaults.cfg_threshold === 'number'
        && s.defaults.cfg_threshold !== cfgThreshold) {
      cfgThreshold = s.defaults.cfg_threshold;
      syncCfg();
    }
    if (s.defaults && typeof s.defaults.negative_prompt_supported === 'boolean'
        && s.defaults.negative_prompt_supported !== negativeSupported) {
      negativeSupported = s.defaults.negative_prompt_supported;
      syncCfg();
    }
    if (s.defaults && typeof s.defaults.true_cfg_supported === 'boolean'
        && s.defaults.true_cfg_supported !== trueCfgSupported) {
      trueCfgSupported = s.defaults.true_cfg_supported;
      syncCfg();
    }
    if (s.defaults && typeof s.defaults.mu_supported === 'boolean'
        && s.defaults.mu_supported !== muSupported) {
      muSupported = s.defaults.mu_supported;
      syncMu();
    }
    if (s.defaults && typeof s.defaults.reference_supported === 'boolean'
        && s.defaults.reference_supported !== referenceSupported) {
      referenceSupported = s.defaults.reference_supported;
      syncConditioning();
      // collect() sends the references, or none, from here: the check counted them.
      syncAnalysisStale();
    }
    if (s.defaults && typeof s.defaults.lora_supported === 'boolean'
        && s.defaults.lora_supported !== loraSupported) {
      loraSupported = s.defaults.lora_supported;
      syncConditioning();
    }
    if (s.defaults && typeof s.defaults.refine_supported === 'boolean'
        && s.defaults.refine_supported !== refineSupported) {
      refineSupported = s.defaults.refine_supported;
      // An open dialog wakes, or says why it cannot run, and asks again.
      if (refineSource) scheduleRefinePreview();
      if (els.lightbox.open) renderLightbox();
    }
    // The dialog's negative prompt, Mu and True CFG follow the same answers
    // as the panel's boxes above.
    if (refineSource) syncRefine();
    // The adapter list is a directory read, so it rides on the first status
    // rather than on the 3 s poll: files appear when the user drops one in,
    // which is a page reload away either way. Once more when the load lands,
    // for the fit the live denoiser answers where the configs could not.
    if (!lorasLoaded || (s.state === 'ready' && !lorasLive)) {
      lorasLoaded = true;
      lorasLive = s.state === 'ready';
      void refreshLoras();
    }
    if (typeof s.max_grid_images === 'number') sheetPage = s.max_grid_images;
    // The grid preview's budget line says whether the load counts against it.
    if (els.dot.classList.contains('ready') !== (s.state === 'ready')) schedulePreview();
    els.dot.className = `dot ${s.state}`;
    // A way to start - or retry, after a gated-repo 403 - the load without
    // queueing an image to do it.
    els.load.hidden = !(s.state === 'unloaded' || s.state === 'error');
    els.statusText.textContent =
      s.state === 'ready' ? s.model_id.split('/').pop()
      : s.state === 'loading' ? 'loading weights…'
      : s.state === 'error' ? 'load failed'
      : 'not loaded';
    if (s.state === 'error' && s.detail) showError(s.detail);
    // Free space beside the VRAM: the disk is the other thing a night can
    // run out of, and the grid preview compares its size estimate against it.
    diskFreeMb = typeof s.disk_free_mb === 'number' ? s.disk_free_mb : null;
    const disk = diskFreeMb !== null ? `${(diskFreeMb / 1024).toFixed(1)} GB free` : '';
    els.vram.textContent = [
      s.vram_total_mb
        ? `${s.device_name ?? ''} · ${(s.vram_used_mb / 1024).toFixed(1)}/${(s.vram_total_mb / 1024).toFixed(1)} GB · ${s.quantization}`
        : '',
      disk,
    ].filter(Boolean).join(' · ');
  } catch { /* server not up yet */ } finally { statusPending = false; }
}

// The adapters on disk, into the select. The chosen one survives a refresh
// that still lists it - a name that has gone is dropped rather than sent to a
// server that would refuse it. An adapter for another checkpoint (a `detail`
// on the entry) is not offered at all: the tooltip names it and says why, so
// a file dropped into the directory that never shows up is explained rather
// than missing.
async function refreshLoras() {
  try {
    const body = await api('/api/loras');
    // The panel is restored at start-up and this list arrives later, so a
    // remembered name is a value the select has no option for yet - assigning
    // it silently does nothing. `pendingLora` carries it across the gap.
    const chosen = els.lora.value || pendingLora;
    els.lora.textContent = '';
    const none = document.createElement('option');
    none.value = '';
    none.textContent = 'none';
    els.lora.append(none);
    const hidden = [];
    for (const entry of body.loras ?? []) {
      if (entry.detail) { hidden.push(entry); continue; }
      const option = document.createElement('option');
      option.value = entry.name;
      option.textContent = `${entry.name} (${humanBytes(entry.bytes)})`;
      els.lora.append(option);
    }
    els.lora.value = [...els.lora.options].some((o) => o.value === chosen) ? chosen : '';
    pendingLora = '';
    if (body.detail) els.loraTip.textContent = body.detail;
    else if (!(body.loras ?? []).length) {
      els.loraTip.textContent =
        `No adapters found. Drop a .safetensors LoRA into ${body.directory} and reload.`;
    } else if (hidden.length) {
      const all = hidden.length === body.loras.length;
      els.loraTip.textContent =
        `${all ? 'Every adapter in the directory is for another checkpoint' : 'Not shown, for another checkpoint'}: `
        + hidden.map((e) => `${e.name} - ${e.detail}`).join('; ');
    } else els.loraTip.textContent = loraTipDefault;
    syncConditioning();
  } catch { /* the list is a convenience; the field simply stays empty */ }
}
