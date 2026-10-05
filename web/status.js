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
  // The grid's example compares no shift against this checkpoint's own, not
  // Turbo's 1.15 on every one: where the profile has none, forcing one is
  // what degrades FLUX, so there is no example to give.
  els.gridMu.placeholder = d.mu != null ? `none,${d.mu}` : '';
  syncCfg();
}

// What steps/CFG/mu mean depends on the checkpoint - 8 steps at CFG 0 is right
// for Turbo and ruinous for FLUX - so the tooltips come from the same server
// profile as the values, and unlike those values they keep following it: the
// user owning the controls does not make the explanation stale.
let lastHints = null;
function applyHints(hints) {
  if (!hints) return;
  lastHints = hints;
  const tips = {
    steps: els.stepsTip, guidance_scale: els.cfgTip, mu: els.muTip, true_cfg: els.trueCfgTip,
    prompt: els.promptTip, size: els.sizeTip, refine: els.refineStrengthTip,
  };
  for (const [key, el] of Object.entries(tips)) {
    if (!hints[key]) continue;
    el.textContent = hints[key];
    // A guide's prompt and size halves are paragraphs; style.css's `.tip.pre`.
    el.classList.toggle('pre', hints[key].includes('\n'));
  }
  // The one some checkpoints have and others not: a server restarted under
  // another must not leave FLUX.1's words under Strength.
  els.refineLarge.textContent = hints.refine_large ?? '';
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
  // A checkpoint whose guide wants Danbooru tags and not prose (Animagine; the
  // hybrid ones document prose as well) shows the start of its authors' first
  // example, cut at a comma: the prose placeholder failed its own style check.
  const tags = helper?.vocabulary === 'danbooru' && !helper.hybrid_prompt && examples.length;
  els.prompt.placeholder = tags ? tagPlaceholder(examples[0]) : promptPlaceholderDefault;
}

function tagPlaceholder(example) {
  if (example.length <= 60) return example;
  const cut = example.lastIndexOf(',', 60);
  return cut > 0 ? `${example.slice(0, cut)}, …` : example;
}

function syncMu() {
  // Blank is the pipeline's own shift; a dead box promises nothing.
  els.mu.placeholder = muSupported ? 'auto' : '';
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
  syncUnload();
  syncSubmitButtons();
  syncCritic();
}

// Unload's one writer. setBusy, Expand and the critic each used to set it,
// overwriting one another, and a load the page did not start left it live
// for a press the server refuses (a 409 under a job or a load). Held, with
// the reason, while it would be refused or could not be sent - aria-disabled,
// as Generate is, so it keeps focus and a press says why (unloadModel);
// disabled, the reason was in a title no keyboard could reach - and hidden
// while Load is shown, with nothing loaded to give back.
function syncUnload() {
  els.unload.hidden = modelState === 'unloaded' || modelState === 'error';
  const why = serverOffline ? 'The server is not answering'
    : busy ? 'A job is queued or running: it holds the weights until it ends'
    : expandingPrompt ? 'Wait for prompt expansion to finish'
    : critiquing ? 'Wait for the critic to finish'
    : modelState === 'loading' ? 'The weights are still loading'
    : '';
  els.unload.setAttribute('aria-disabled', String(!!why));
  els.unload.title = why || 'Free VRAM';
}

// The topbar's Load - and the failed load's Retry load - starts the load
// without queueing an image to do it, and the status says so at once.
async function loadModel() {
  try {
    await api('/api/model/load', { method: 'POST' });
    showError('');
  } catch (err) {
    showError(err.message);
  }
  refreshStatus();
}

// An Unload the server refused - a load another tab started - is said until
// the load lands, when it no longer says anything true.
let unloadRefusal = null;
async function unloadModel() {
  if (els.unload.getAttribute('aria-disabled') === 'true') {
    // As Generate's press: no answer is said in the offline words, which the
    // poll's first answer takes back, and the poll asks again at once.
    if (serverOffline) { showError(OFFLINE_MESSAGE); void refreshStatus(); }
    // Held by a job: said until that job ends (generation.js).
    else if (busy) showJobNote(els.unload.title);
    else showError(els.unload.title, true);
    return;
  }
  try {
    await api('/api/model/unload', { method: 'POST' });
    showError('');
    unloadRefusal = null;
  } catch (err) {
    showError(err.message);  // 409 while a job or a load is still live
    unloadRefusal = err.message;
  }
  refreshStatus();
}

// No answer to a status poll: fetch rejected, never a slow reply - a load's
// CUDA read can stall one for a while, which the statusPending guard absorbs.
// The topbar said "ready" in green over a dead server for as long as it was
// down; it says what is true, and what would send to the server is held.
function goOffline() {
  if (serverOffline) return;
  serverOffline = true;
  els.dot.className = 'dot error';
  els.statusText.textContent = LOST_PHRASE;
  els.load.hidden = true;
  syncUnload();
  syncSubmitButtons();
}

// The server that answers is not the one the page was talking to: its jobs,
// the queue and the weights went with the old process. Said once, where the
// panel's line has nothing more particular - a lost job's own account of
// the same restart stays.
function noteRestart(s) {
  queuedIds = [];
  syncQueueNote();
  if (els.error.textContent) return;
  const at = new Date(s.started_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  showError(`The server restarted at ${at} — ${s.state === 'ready' ? '' : 'the model is not loaded, and '}`
    + 'any job that was running ended with it.', true);
}

// A failed load is said where the work is: in the viewer's empty state, a
// card with the status's sentence and its steps, the model page, and Retry
// load - rather than red text a column away, re-posted on every poll over
// whatever else the line had to say, and into the lightbox. Where the viewer
// holds images, the panel's line says it, once per failure and never in the
// lightbox, which has nothing to do with a load.
// `shownLoadDetail` is the detail last posted there; a job that ended at the
// load forgets it (finishJob), since the press that queued the job cleared
// the line. A card that goes - its Retry load pressed, the load on its way -
// hands focus to Generate, the next thing to press, or to the new card's
// Retry load, rather than to the page's body.
let shownLoadDetail = null;
function syncLoadFailure(s) {
  const failed = s.state === 'error' && !!s.detail;
  let card = els.viewer.querySelector('.load-failed');
  let focused = false;
  if (card && (!failed || card.dataset.detail !== s.detail)) {
    focused = card.contains(document.activeElement);
    card.remove();
    card = null;
  }
  if (failed && !card && !els.viewer.querySelector('img, figure, .tomb')) {
    card = loadFailureCard(s);
    els.viewer.prepend(card);
  }
  if (focused) (card?.querySelector('button') ?? els.generate).focus({ preventScroll: true });
  els.viewer.querySelectorAll('.placeholder').forEach((p) => { p.hidden = !!card; });
  if (!failed) { shownLoadDetail = null; return; }
  if (s.detail === shownLoadDetail) return;
  shownLoadDetail = s.detail;
  if (!card) showError(s.detail, false, { mirror: false });
}

function loadFailureCard(s) {
  const card = document.createElement('div');
  card.className = 'load-failed';
  card.dataset.detail = s.detail;
  card.setAttribute('role', 'group');
  card.setAttribute('aria-labelledby', 'load-failed-head');
  const head = document.createElement('h2');
  head.id = 'load-failed-head';
  head.textContent = `Couldn't load ${(s.model_id ?? '').split('/').pop()}`;
  // A `command` in the sentence is something to type: set as code.
  const detail = document.createElement('p');
  s.detail.split('`').forEach((part, i) => {
    detail.append(i % 2 ? Object.assign(document.createElement('code'), { textContent: part }) : part);
  });
  card.append(head, detail);
  if (s.detail_url) {
    const link = document.createElement('a');
    link.href = s.detail_url;
    link.target = '_blank';
    link.rel = 'noopener';
    link.textContent = 'Open the model page';
    const line = document.createElement('p');
    line.append(link);
    card.append(line);
  }
  const retry = document.createElement('button');
  retry.type = 'button';
  retry.className = 'primary';
  retry.textContent = 'Retry load';
  retry.onclick = loadModel;
  card.append(retry);
  return card;
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
    // An answer after none: what was held lets go, and what said the server
    // was not answering goes. A restart is the start time changing - with or
    // without a poll that failed between, which a quick one leaves none of.
    const back = serverOffline;
    if (back) {
      serverOffline = false;
      clearOffline();
      retryLightboxImage();
      syncSubmitButtons();
    }
    const restarted = serverStartedAt !== null && typeof s.started_at === 'number'
      && s.started_at !== serverStartedAt;
    if (typeof s.started_at === 'number') serverStartedAt = s.started_at;
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
      // A ⚠ Reuse says whether another checkpoint's references carry over.
      syncReuseButtons();
    }
    if (s.defaults && typeof s.defaults.lora_supported === 'boolean'
        && s.defaults.lora_supported !== loraSupported) {
      loraSupported = s.defaults.lora_supported;
      syncConditioning();
      syncReuseButtons();
    }
    if (s.defaults && typeof s.defaults.refine_supported === 'boolean'
        && s.defaults.refine_supported !== refineSupported) {
      refineSupported = s.defaults.refine_supported;
      syncCardTools();
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
    // The grid preview's budget line says whether the load counts against it,
    // and an open Refine dialog's preview where its pass starts, which only
    // the loaded schedule answers (`start_sigma`): a load that lands with the
    // dialog open adds the line, an unload takes it away.
    if (els.dot.classList.contains('ready') !== (s.state === 'ready')) {
      schedulePreview();
      if (refineSource) scheduleRefinePreview();
    }
    els.dot.className = `dot ${s.state}`;
    const was = modelState;
    modelState = s.state;
    // A way to start - or retry, after a gated-repo 403 - the load without
    // queueing an image to do it. Load and Unload take each other's place,
    // so the press that hides one hands focus to the other, not to the
    // page's body.
    const pressed = [els.load, els.unload].find((b) => b === document.activeElement);
    els.load.hidden = !(s.state === 'unloaded' || s.state === 'error');
    els.load.textContent = s.state === 'error' ? 'Retry load' : 'Load';
    els.load.title = s.state === 'error' ? 'Try the load again, once what stopped it is fixed' : 'Load the weights now';
    syncUnload();
    if (pressed?.hidden) (pressed === els.load ? els.unload : els.load).focus({ preventScroll: true });
    // The checkpoint's name in every state: "loading weights…" for minutes
    // never said which.
    const name = (s.model_id ?? '').split('/').pop();
    els.statusText.textContent =
      s.state === 'ready' ? name
      : s.state === 'loading' ? `${name} · loading…`
      : s.state === 'error' ? `${name} · load failed`
      : `${name} · not loaded`;
    syncLoadFailure(s);
    if (s.state === 'ready' && was !== 'ready' && unloadRefusal) {
      if (els.error.textContent === unloadRefusal) showError('');
      unloadRefusal = null;
    }
    if (restarted) noteRestart(s);
    // A lost run's card in the viewer says the server is back, or restarted.
    if (back || restarted) lostRunBack(restarted, s);
    // What a job saved before the server went quiet - or before it died -
    // is in the gallery now, which the page could not ask for meanwhile.
    if ((back || restarted) && loadedModel === previousModel) { void prependFinished(0); loadRuns(); }
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
  } catch (err) {
    // An answer that refused is not a server gone; no answer at all is.
    if (err.offline) goOffline();
  } finally { statusPending = false; }
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
    // What an adapter is and where they come from, always; the files not
    // offered and why after it, in a wider tip, rather than in its place. A
    // detail names its file first, which the list already does.
    const about = body.directory
      ? `A LoRA adapter from ${body.directory}; Weight 0-2 scales its effect.` : loraTipDefault;
    const why = (e) => (e.detail.startsWith(e.name) ? e.detail.slice(e.name.length).trim() : e.detail);
    let tip = about;
    if (body.detail) tip = body.detail;
    else if (!(body.loras ?? []).length) {
      tip = `No adapters found. Drop a .safetensors LoRA into ${body.directory} and reload.`;
    } else if (hidden.length) {
      const all = hidden.length === body.loras.length;
      tip = `${about}\n\n`
        + `${all ? 'Every adapter in the directory is for another checkpoint' : 'Not shown, for another checkpoint'}: `
        + hidden.map((e) => `${e.name} - ${why(e)}`).join('; ');
    }
    els.loraTip.textContent = tip;
    els.loraTip.classList.toggle('wide', tip.includes('\n'));
    els.loraTip.classList.toggle('pre', tip.includes('\n'));
    syncConditioning();
    // A ⚠ Reuse names an adapter this list does not offer.
    syncReuseButtons();
  } catch { /* the list is a convenience; the field simply stays empty */ }
}
