// Local Diffusion's page script: /api/status, the profile defaults and the hints. Declarations only; state.js has
// the load-order contract these files share.
'use strict';

// ------------------------------------------------------------------- status

// Seed the controls from the server's profile for the loaded checkpoint. Turbo
// wants 8 steps / CFG 0 / mu 1.15; FLUX wants 28 / 3.5 and no mu at all, since
// it derives its own resolution-dependent shift. Each control follows the
// profile until the user touches *that* control. A checkpoint switch restores
// that checkpoint's own edits before later polls apply its defaults again.
let lastDefaults = null;
function applyDefaults(d) {
  if (!d || d.steps === undefined) return;
  lastDefaults = d;
  if (!owned.steps) els.steps.value = d.steps;
  if (!owned.cfg) els.cfg.value = d.guidance_scale;
  if (!owned.mu) els.mu.value = d.mu ?? '';
  // The grid's examples are this checkpoint's neighbourhood, as its Mu one
  // is, not Krea 2 Turbo's 6-12x2 and 0,0.5,1 on every checkpoint: those
  // were sweeps far outside Z-Anime's 28-50 steps and CFG 3-5, where CFG 0
  // turns guidance and the negative prompt off. The tips stay syntax help.
  const sweep = gridExamples(d);
  els.gridSteps.placeholder = sweep.steps;
  els.gridCfg.placeholder = sweep.cfg;
  syncMuWords();
  syncCfg();
}

// A quarter either side of the profile's steps, and a step of CFG either
// side of its CFG - clamped to the boxes' own bounds. A profile at CFG 0
// (Krea 2 Turbo, Z-Image Turbo) keeps the shipped 0,0.5,1: one below would
// be negative, and those checkpoints are distilled for 0. One whose CFG
// leaves guidance off (Anima Turbo and ERNIE Turbo at 1, a threshold that
// is CFG's own and not True CFG's) sweeps upwards from it: 0,1,2 spent a
// cell on CFG 0, the same image as the profile's own 1. Where the box never
// reaches the model (`cfg_ignored`: the distilled klein, schnell, Qwen-Image)
// there is no example: 1,2,3 was three copies of one image, and on klein a
// walk into the pipeline's warning above 1.
function gridExamples(d) {
  const most = (el, fallback) => (Number.isFinite(+el.max) && el.max !== '' ? +el.max : fallback);
  const s = d.steps, c = d.guidance_scale;
  const threshold = typeof d.cfg_threshold === 'number' ? d.cfg_threshold : cfgThreshold;
  const steps = [Math.round(0.75 * s), s, Math.round(1.25 * s)]
    .map((n) => Math.min(most(els.steps, 100), Math.max(1, n)));
  const top = most(els.cfg, 20);
  const cfg = d.cfg_ignored ? []
    : !(c > 0) ? [0, 0.5, 1]
    : (c <= threshold && !d.true_cfg_supported ? [c, c + 1, c + 2] : [Math.max(0, c - 1), c, c + 1])
      .map((v) => +Math.min(top, v).toFixed(2));
  return { steps: [...new Set(steps)].join(','), cfg: [...new Set(cfg)].join(',') };
}

// The shift a blank Mu leaves, where the checkpoint's scheduler has a fixed
// one: Z-Anime's, NetaYume's and NewBie's shift 6 (mu 1.79), Z-Image's 3
// (1.10) - the profile's `mu_static`. Null where the pipeline derives its
// own from the size (FLUX) or Mu does not apply.
function muStatic() {
  const v = lastDefaults?.mu_static;
  return muSupported && typeof v === 'number' ? v : null;
}
const muStaticText = (v) => v.toFixed(2);

// What a blank Mu box says: "auto" where the pipeline derives its own shift,
// and where that shift is a fixed number, the number - nothing there is
// automatic, and the tip beside the box says "fixed shift of 6 (mu 1.79)".
// The number alone: "auto (1.79)" is 66 px in a box with 58 at 1400x900, and
// 2048x1040 at 125% has no more room. Nothing on a dead box. The Refine
// dialog's Mu asks the same.
function muPlaceholder() {
  if (!muSupported) return '';
  const fixed = muStatic();
  return fixed === null ? 'auto' : muStaticText(fixed);
}

// The Mu boxes' words, which follow the status as the tips do: the panel's
// placeholder, and the grid axis's example - or, where Mu cannot apply, its
// reason. The axis sits far below the panel's badge and was greyed with its
// syntax help still in its tip ("none,1.15 compares..."): a dead box gives
// its reason in its badge and its wrapper's title, the panel's Mu tip's
// words (hints.mu), and the syntax comes back where Mu applies.
function syncMuWords() {
  els.mu.placeholder = muPlaceholder();
  const why = muSupported ? '' : (lastHints?.mu || 'Mu does not apply on this checkpoint.');
  const d = lastDefaults;
  els.gridMu.placeholder = !muSupported ? 'unavailable'
    // The grid's example compares no shift against this checkpoint's own,
    // not Turbo's 1.15 on every one: where the profile has none, forcing
    // one is what degrades FLUX, so there is no example to give.
    : d?.mu != null ? `none,${d.mu}` : '';
  els.gridMuField.title = why;
  els.gridMuTip.replaceChildren(...(why ? [why] : gridMuTipDefault));
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
  // The CFG box's name, the panel's, the grid's and the Refine dialog's:
  // FLUX.1's is a distilled guidance value, by its own tip, beside a True CFG
  // that is the real one - "CFG (distilled)" there, plain CFG wherever a
  // status gives no name.
  const cfgLabel = hints.cfg_label ?? 'CFG';
  for (const el of [els.cfgLabel, els.gridCfgLabel, els.refineCfgLabel]) {
    if (el.textContent !== cfgLabel) el.textContent = cfgLabel;
  }
  // A dead grid Mu axis quotes the Mu tip's reason.
  syncMuWords();
}

// The rest of the guide: the size range the checkpoint's authors vouch for,
// which the size line compares the current size against, and the example
// prompts. Both follow the server the way the tooltips do.
function applyGuide(g) {
  if (!g) return;
  const range = Array.isArray(g.pixels) && g.pixels.length === 2 ? g.pixels : null;
  const word = g.pixels_word || 'documented';
  const source = typeof g.pixels_source === 'string' && g.pixels_source ? g.pixels_source : null;
  const aspects = Array.isArray(g.aspects) && g.aspects.length ? g.aspects : null;
  const resolutions = Array.isArray(g.resolutions) && g.resolutions.length ? g.resolutions : null;
  if (JSON.stringify(range) !== JSON.stringify(sizeRange) || word !== sizeRangeWord
      || source !== sizeRangeSource
      || JSON.stringify(aspects) !== JSON.stringify(sizeAspects)
      || JSON.stringify(resolutions) !== JSON.stringify(guideResolutions)) {
    sizeRange = range;
    sizeRangeWord = word;
    sizeRangeSource = source;
    sizeAspects = aspects;
    guideResolutions = resolutions;
    syncSize();
  }
  if (Array.isArray(g.examples)) examples = g.examples;
  exampleNotes = g.example_notes || {};
  if (Array.isArray(g.slots)) {
    exampleSlots = g.slots.flatMap((pattern) => {
      try { return [new RegExp(pattern, 'gi')]; } catch { return []; }
    });
  }
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
  // Blank is the pipeline's own shift. The panel's dead box promises
  // nothing; the grid's says "unavailable", with its reason (syncMuWords).
  syncMuWords();
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
// for the image queue, so a job queued or running holds them in turn -
// another tab's included (elsewhereBusy), which the server refuses them
// under as it does this page's. That one arrives with a status poll, at a
// moment no one chose, so Expand is held for it aria-disabled with the
// reason, and keeps a keyboard user's focus: disabled, it dropped it to
// <body> with nothing said (expandPrompt says why; syncExpandButton).
const EXPAND_ELSEWHERE = 'A job started in another tab or window is running; Expand once it ends';
function syncPromptExpansion() {
  syncExpandButton();
  syncUnload();
  syncSubmitButtons();
  syncCritic();
}

// Expand's one writer. Held - while it runs, a job is queued or running (this
// page's, or another tab's: elsewhereBusy), the critic thinks or the server
// does not answer - with aria-disabled and the
// reason as its title, never disabled: a first use runs for minutes, and the
// disabled button had dropped a keyboard user's focus to <body> for all of
// them, and after a failure too; a press says why (expandPrompt). Running,
// it reads "Expand…": "Expanding…" was 23 px wider, and at 1400 px wrapped
// Reset onto a line of its own, moving the prompt box down under the pointer
// for the length of the request. Its title names what a first use costs.
const EXPAND_TITLE = els.expandPrompt.title;
function syncExpandButton() {
  els.expandPrompt.hidden = !promptExpansionSupported;
  const why = expandingPrompt ? 'Expanding…'
    : serverOffline ? 'The server is not answering'
    : modelChangeHold() ? modelChangeHold()
    : busy ? 'A job is queued or running: Expand waits for the image queue to finish'
    : elsewhereBusy ? EXPAND_ELSEWHERE
    : critiquing ? 'Wait for the critic to finish'
    : '';
  const cost = expandCost();
  els.expandPrompt.setAttribute('aria-disabled', String(!!why));
  els.expandPrompt.title = why || (cost ? `${EXPAND_TITLE}. ${cost}` : EXPAND_TITLE);
  els.expandPrompt.textContent = expandingPrompt ? 'Expand…' : 'Expand';
}

// What a first Expand costs before it does anything, from the status: the
// downloads a first use makes, else the rewriter's load into memory once a
// session - as the critic's line names its own first download.
function expandCost() {
  const s = expanderStatus;
  if (!s) return '';
  if (s.fetch?.length) {
    return `First use downloads ${s.fetch.map((f) => f.name).join(' and ')} (about ${humanBytes(s.fetch_bytes)}) once.`;
  }
  return s.loaded ? '' : "Its first use this session loads ERNIE's prompt rewriter into memory first.";
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
    : modelChangeHold() ? modelChangeHold()
    : busy ? 'A job is queued or running: it holds the weights until it ends'
    : elsewhereBusy ? 'A job started in another tab or window is running; Unload once it ends'
    : expandingPrompt ? 'Wait for prompt expansion to finish'
    : cardHolder === 'expand' && stoppedExpansionTickets.has(cardTicket) ? 'The abandoned expansion is finishing its current stage'
    : critiquing ? 'Wait for the critic to finish'
    : modelState === 'loading' ? 'The weights are still loading'
    : '';
  els.unload.setAttribute('aria-disabled', String(!!why));
  els.unload.title = why || 'Free VRAM';
  if (unloadHeldNote !== null && unloadHeldNote !== `${why}.`) {
    if (els.error.textContent === unloadHeldNote) showError('');
    unloadHeldNote = null;
  }
  syncModelSwitch();
}

// The topbar's Load - and the failed load's Retry load - starts the load
// without queueing an image to do it, and the status says so at once. It
// takes back only what was said about the load - its own refusal, a failed
// load's sentence or Expand's pointer to it (`loadLine`) - never the line's
// other messages: an Undo on offer, a job's outcome. Expand's words about the
// load said under the prompt instead, beside a Reuse's Undo (app.js,
// expandSay), go too.
let loadLine = null;
async function loadModel() {
  if (modelChangeHold()) { sayModelChangeHold(); return; }
  if (executionLoadHold()) { showError(executionLoadHold()); return; }
  try {
    await api('/api/model/load', { method: 'POST' });
    const said = els.error.textContent;
    if (said && (said === loadLine || said === shownLoadDetail)) showError('');
    loadLine = null;
    if (promptUndo?.aboutLoad) dropPromptUndo();
  } catch (err) {
    showError(err.message);
    loadLine = err.message;
  }
  refreshStatus();
}

// An Unload the server refused - a load another tab started - is said until
// the load lands, when it no longer says anything true.
let unloadRefusal = null;
// A press held for a reason of this page's (a load, Expand, the critic, a
// model change) is said until that reason stops holding it (syncUnload): the
// load's sentence stayed in the line after the weights had landed.
let unloadHeldNote = null;
async function unloadModel() {
  if (els.unload.getAttribute('aria-disabled') === 'true') {
    // As Generate's press: no answer is said in the offline words, which the
    // poll's first answer takes back, and the poll asks again at once.
    if (serverOffline) { showError(OFFLINE_MESSAGE); void refreshStatus(); }
    // Held by a job: said until that job ends (job-results.js) - another
    // tab's too (setElsewhereBusy). A sentence, as every held press says its
    // reason: the title is the reason without its stop.
    else if (busy) showJobNote(`${els.unload.title}.`);
    else if (elsewhereBusy) {
      const said = `${els.unload.title}.`;
      showError(said, true);
      elsewhereNote = said;
    } else {
      const said = `${els.unload.title}.`;
      showError(said, true);
      unloadHeldNote = said;
    }
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
  els.statusAside.hidden = true;
  els.load.hidden = true;
  syncUnload();
  syncSubmitButtons();
  syncExpandButton();
  syncCritic();
}

// The server that answers is not the one the page was talking to: its jobs,
// the queue and the weights went with the old process. Said once, where the
// panel's line has nothing more particular - a lost job's own account of
// the same restart stays.
function noteRestart(s) {
  setQueue([]);
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
// Retry load, rather than to the page's body. Once the load is no longer
// failed - retried here, or in another tab - the panel's line lets the
// detail go (`loadDetailInError`, the one this function posted), and the
// finish line of a job that ended at the load stops pointing at a card and
// a Retry load that have gone (notMadeText): it said "why, and Retry load,
// are in the viewer" over a ready topbar, and the other tab kept the red
// detail under it.
let shownLoadDetail = null;
let loadDetailInError = null;
function syncLoadFailure(s) {
  const failed = s.state === 'error' && !!s.detail;
  let card = els.viewer.querySelector('.load-failed');
  let focused = false;
  if (card && (!failed || card.dataset.detail !== s.detail)) {
    focused = card.contains(document.activeElement);
    card.remove();
    card = null;
  }
  if (failed && !card && loadFailureInViewer()) {
    card = loadFailureCard(s);
    els.viewer.prepend(card);
  }
  if (focused) (card?.querySelector('button') ?? els.generate).focus({ preventScroll: true });
  els.viewer.querySelectorAll('.placeholder').forEach((p) => { p.hidden = !!card; });
  const notMade = els.finishNote.querySelector('.not-made');
  if (notMade && (failed || ['loading', 'ready', 'unloaded'].includes(s.state))) {
    const text = notMadeText(notMade.dataset.lead, failed ? 'error' : s.state);
    if (notMade.textContent !== text) notMade.textContent = text;
  }
  if (!failed) {
    if (loadDetailInError !== null && els.error.textContent === loadDetailInError) {
      showError('', false, { mirror: false });
    }
    loadDetailInError = null;
    shownLoadDetail = null;
    return;
  }
  if (s.detail === shownLoadDetail) return;
  shownLoadDetail = s.detail;
  if (!card) {
    showError(s.detail, false, { mirror: false });
    loadDetailInError = s.detail;
  }
}

// Whether a failed load is said in the viewer (the card) rather than in the
// panel's line: the viewer holds nothing of this session's.
function loadFailureInViewer() {
  return !els.viewer.querySelector('img, figure, .tomb');
}

function loadFailureCard(s) {
  const card = document.createElement('div');
  card.className = 'load-failed';
  card.dataset.detail = s.detail;
  card.setAttribute('role', 'group');
  card.setAttribute('aria-labelledby', 'load-failed-head');
  const head = document.createElement('h2');
  head.id = 'load-failed-head';
  head.textContent = `Couldn't load ${checkpointName(s.model_id)}`;
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
// chip, an accepted suggestion, Expand - may fire no `input` event, so what
// describes the old text is told here instead: its translation goes, and the
// prompt check marks itself stale. Expand shows the translation it brings
// after this.
function promptReplaced() {
  clearPromptTranslation();
  syncAnalysisStale();
}

function showPromptTranslation(source, translation) {
  translatedSource = source;
  els.promptTranslation.value = translation;
  els.promptTranslationField.hidden = false;
}

// The translation's place. Under the prompt in the panel below 1800 px; from
// 1800 px at the head of the results instead, where it has the column's
// width and its own height (style.css): in the panel's sticky column its
// four rows showed a fifth of a long expansion's English, and the field
// pushed Generate below the window at 2048x1040 at 125% while the results
// column stood idle. Moved whole, with focus and a selection in it kept, as
// placeTagValidation keeps them.
const TRANSLATION_WIDE = matchMedia('(min-width: 1800px)');
function placeTranslation() {
  const field = els.promptTranslationField;
  const [host, before] = TRANSLATION_WIDE.matches
    ? [els.stage, els.viewer] : [els.negField.parentElement, els.negField];
  if (field.parentElement === host && field.nextElementSibling === before) return;
  const focused = document.activeElement;
  const inside = field.contains(focused);
  const range = inside && typeof focused.selectionStart === 'number'
    ? [focused.selectionStart, focused.selectionEnd, focused.selectionDirection] : null;
  if (typeof host.moveBefore === 'function') host.moveBefore(field, before);
  else host.insertBefore(field, before);
  if (!inside || focused === document.activeElement) return;
  focused.focus({ preventScroll: true });
  if (range) focused.setSelectionRange(...range);
}

// A translation the panel remembered (panel.js, restorePanel), shown once the
// first status says the checkpoint expands prompts, and only while the box
// still holds the prompt it translates.
let pendingTranslation = null;

// What the page calls a checkpoint: the start-up catalogue's name for the
// running one (`ModelStatus.model_label`) - "NetaYume Lumina v4", where the
// repository is named for the parent model - else the id's last part, which
// is all there is for another checkpoint's records and one off the
// catalogue. The id itself goes in the titles beside it.
function checkpointName(id) {
  return id && id === loadedModel && checkpointLabel ? checkpointLabel : (id ?? '').split('/').pop();
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
  // A job this page queues after this is not in the answer (resyncQueue).
  const asked = Date.now();
  const epoch = modelStatusEpoch;
  try {
    const s = await api('/api/status');
    if (epoch === modelStatusEpoch) receiveStatus(s, asked);
  } catch (err) {
    // An answer that refused is not a server gone; no answer at all is.
    if (err.offline) goOffline();
  } finally { statusPending = false; }
}

function receiveStatus(s, asked = Date.now()) {
    // An answer after none: what was held lets go, and what said the server
    // was not answering goes. A restart is the start time changing - with or
    // without a poll that failed between, which a quick one leaves none of.
    const back = serverOffline;
    if (back) {
      serverOffline = false;
      clearOffline();
      retryLightboxImage();
      takeBackCriticOffline();
      syncSubmitButtons();
      syncExpandButton();
      syncCritic();
    }
    const restarted = serverStartedAt !== null && typeof s.started_at === 'number'
      && s.started_at !== serverStartedAt;
    if (typeof s.started_at === 'number') serverStartedAt = s.started_at;
    if (typeof s.server_time === 'number') serverClockOffset = s.server_time - Date.now() / 1000;
    const previousModel = loadedModel;
    const revision = Number.isInteger(s.model_revision) ? s.model_revision : 0;
    if (!restarted && modelRevision !== null && revision < modelRevision) return;
    const changed = (s.model_id && s.model_id !== previousModel)
      || (modelRevision !== null && revision !== modelRevision);
    const wasSwitching = modelSwitching;
    modelSwitching = !!s.switching;
    modelRevision = revision;
    if (changed) checkpointChanged(previousModel);
    loadedModel = s.model_id ?? loadedModel;
    if (s.model_id) checkpointLabel = s.model_label ?? null;
    receiveExecutionStatus(s, asked);
    receiveModelSettings(s, { reset: changed || restarted });
    if (changed) {
      syncReuseButtons();
      if (previousModel) resetGallery();
      loadRuns();
    }
    applyDefaults(s.defaults);
    // After the profile, so a remembered value set on this checkpoint wins
    // over it and a value set on another is left to the profile.
    if (loadedModel) restoreSampling(loadedModel);
    applyHints(s.hints);
    applyGuide(s.guide);
    if (typeof s.prompt_expansion_supported === 'boolean'
        && (s.prompt_expansion_supported !== promptExpansionSupported
            || JSON.stringify(s.prompt_expander ?? null) !== JSON.stringify(expanderStatus))) {
      promptExpansionSupported = s.prompt_expansion_supported;
      expanderStatus = s.prompt_expander ?? null;
      syncPromptExpansion();
      // A download finished, the rewriter loaded: an Expand running says so.
      syncExpandRunning();
    }
    // What holds the card: an Expand waiting behind another tab's critic or
    // Expand says so, and stops saying it once its own press has the card.
    const holder = typeof s.card_holder === 'string' ? s.card_holder : null;
    const ticket = typeof s.card_ticket === 'string' ? s.card_ticket : null;
    if (holder !== cardHolder || ticket !== cardTicket) {
      cardHolder = holder;
      cardTicket = ticket;
      syncExpandRunning();
      syncUnload();
    }
    // A pause's clock, here or the job ahead's, moves with the poll.
    tickPauseClock();
    if (pendingTranslation) {
      const { source, text } = pendingTranslation;
      pendingTranslation = null;
      if (promptExpansionSupported && translatedSource === null && els.prompt.value === source) {
        showPromptTranslation(source, text);
      }
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
    // Then refine's tick is held where there is no pass, and its strength
    // is this checkpoint's (then-refine.js).
    syncThenRefine();
    // The dialog's negative prompt, Mu and True CFG follow the same answers
    // as the panel's boxes above.
    if (refineSource) syncRefine();
    // And the inpaint's, with the tools that open it (inpaint.js).
    if (s.defaults) syncInpaintCapability(s.defaults);
    // Mass refine's Refine… says the hedge or the final answer as the status
    // settles it, and an open dialog's Refine what holds it now.
    syncRefineBatchButton();
    // The adapter list is a directory read, so it rides on the first status
    // rather than on the 3 s poll: files appear when the user drops one in,
    // which is a page reload away either way. Once more when the load lands,
    // for the fit the live denoiser answers where the configs could not; and
    // at a refusal naming the chosen adapter (dropRefusedLora, dropRefineLora).
    if (!modelSwitching && (!lorasLoaded || (s.state === 'ready' && !lorasLive))) {
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
      if (inpaintSource) scheduleInpaintPreview();
      if (outpaintSource) scheduleOutpaintPreview();
    }
    els.dot.className = `dot ${s.state}`;
    const was = modelState;
    modelState = s.state;
    // An Expand waiting on the load says so, and stops saying it once it
    // lands; the critic is held for the load (criticHold), and what a press
    // on it said meanwhile goes with it.
    if (modelState !== was) {
      syncExpandRunning();
      syncCritic();
      if (modelState !== 'loading') takeBackCriticLoadNote();
    }
    // A way to start - or retry, after a gated-repo 403 - the load without
    // queueing an image to do it. Load and Unload take each other's place,
    // so the press that hides one hands focus to the other, not to the
    // page's body.
    const pressed = [els.load, els.unload].find((b) => b === document.activeElement);
    els.load.hidden = !(s.state === 'unloaded' || s.state === 'error');
    els.load.textContent = s.state === 'error' ? 'Retry load' : 'Load';
    els.load.title = s.state === 'error' ? 'Try the load again, once what stopped it is fixed' : 'Load the weights now';
    syncExecutionControls();
    // Another tab's job, or one queued behind this page's from elsewhere:
    // before Unload's reason and the line below are written.
    if (Array.isArray(s.active_jobs)) followServerJobs(s.active_jobs, asked);
    syncPromptExpansion();
    if (changed || (wasSwitching && !modelSwitching)) {
      schedulePreview();
      scheduleTagValidation(0);
      if (previousModel) persistPanel();
    }
    if (pressed?.hidden) (pressed === els.load ? els.unload : els.load).focus({ preventScroll: true });
    // The checkpoint's name in every state: "loading weights…" for minutes
    // never said which. A live region, so written only when it changes: the
    // same name every three seconds was a new node to read out at every
    // poll, all session long. Its id is the title. Another tab's job is
    // beside it, outside the live region (index.html, #status-aside).
    const name = checkpointName(s.model_id);
    const line = s.state === 'ready' ? name
      : s.state === 'loading' ? `${name} · loading…`
      : s.state === 'error' ? `${name} · load failed`
      : `${name} · not loaded`;
    if (els.statusText.textContent !== line) els.statusText.textContent = line;
    const aside = s.state === 'ready' && elsewhereBusy ? '· busy in another tab' : '';
    if (els.statusAside.textContent !== aside) els.statusAside.textContent = aside;
    els.statusAside.hidden = !aside;
    if (s.model_id && els.statusText.title !== s.model_id) els.statusText.title = s.model_id;
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
    if ((back || restarted) && loadedModel === previousModel) {
      elsewhereCatchUp = false;
      noteUnseenImages();
      void prependFinished(0);
      loadRuns();
    }
    // Images made between two polls by a job this page never saw - queued,
    // run and ended in that time - by the gallery's newest id moving, and
    // another tab's job's end (setElsewhereBusy, followServerJobs) taken in.
    // Settled where this answer took everything in already: the server back
    // or restarted, the checkpoint changed (its gallery reset above).
    followGalleryHead(s.newest_image_id, back || restarted || changed);
    // And the prompt's count, which a measure asked while it was gone could
    // not get: the line under the prompt kept "cut unknown", titled with the
    // offline sentence, until the next pause in typing (a checkpoint change
    // asked above).
    if (back && loadedModel === previousModel) scheduleTagMeasure(0);
    // Free space beside the VRAM: the disk is the other thing a night can
    // run out of, and the grid preview compares its size estimate against it.
    diskFreeMb = typeof s.disk_free_mb === 'number' ? s.disk_free_mb : null;
    const disk = diskFreeMb !== null ? `${(diskFreeMb / 1024).toFixed(1)} GiB free` : '';
    els.vram.textContent = [
      s.vram_total_mb
        ? `${s.device_name ?? ''} · ${(s.vram_used_mb / 1024).toFixed(1)}/${(s.vram_total_mb / 1024).toFixed(1)} GiB · ${s.quantization}`
        : '',
      disk,
    ].filter(Boolean).join(' · ');
    offerModelPicker(s);
    receiveFinishStatus(s);
}

// The adapters on disk, into the select. The chosen one survives a refresh
// that still lists it - a name that has gone is dropped rather than sent to a
// server that would refuse it. An adapter for another checkpoint (a `detail`
// on the entry) is not offered at all: the tooltip names it and says why, so
// a file dropped into the directory that never shows up is explained rather
// than missing. Answers the listing it applied, or null where it applied none
// (no answer, or a later read overtook it): a refusal of a gone adapter reads
// the entry there (dropRefusedLora, dropRefineLora).
let loraEpoch = 0;
async function refreshLoras() {
  const epoch = ++loraEpoch;
  try {
    const body = await api('/api/loras');
    if (epoch !== loraEpoch) return null;
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
    loraRefusal = body.supported === false && body.detail ? body.detail : null;
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
    return body;
  } catch {
    // The list is a convenience; the field simply stays empty.
    return null;
  }
}
