// Performance configuration and composer intent. Declarations only; app.js boots.
'use strict';

const EXECUTION_UNAVAILABLE = 'Execution recipes are not available in this build; use Eager, native attention and no step cache.';
const EXECUTION_DRAFT_KEY = 'local-diffusion.execution-draft.v1';
const INVALID_PERFORMANCE_VALUE = '__unavailable_saved_choice__';
const INVALID_CACHE_REASON = 'The saved step cache choice is unknown or invalid. Choose Off or a valid named choice explicitly.';
const INVALID_LOADER_REASON = 'The saved performance draft is unknown or invalid. Open Models to choose valid Compilation and Attention choices, or Revert.';
const invalidPerformanceChoices = new WeakMap();
let invalidExecutionDraft = null;
let replacedExecutionEnvelope = null;
let invalidDraftReplacements = new Set();
let selectedExecution = null;
let selectedExecutionReason = '';
let selectedExecutionAvailable = true;
let executionEnvironmentWarning = '';
let executionDraftBase = {};
let executionCacheOptions = null;
let modelExecutionObservation = null;
let modelExecutionAsked = -Infinity;

function readExecutionDiagnostics(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const keys = ['load_id', 'recipe', 'registration', 'first_call', 'failure_stage'];
  if (Object.keys(value).some((key) => !keys.includes(key))
      || !Number.isSafeInteger(value.load_id) || value.load_id <= 0
      || value.recipe !== 'regional-v1' || value.registration !== 'registered'
      || !['pending', 'running', 'returned', 'failed', 'cancelled'].includes(value.first_call)) return null;
  const failed = value.first_call === 'failed' || value.first_call === 'cancelled';
  const stage = value.failure_stage ?? null;
  if (failed ? !['forward', 'verification', 'policy_exit'].includes(stage) : stage !== null) return null;
  return Object.freeze({ load_id: value.load_id, recipe: 'regional-v1', registration: 'registered',
    first_call: value.first_call, failure_stage: stage });
}

function mergeExecutionDiagnostics(previous, value) {
  const next = readExecutionDiagnostics(value);
  const old = readExecutionDiagnostics(previous);
  if (!next || !old || next.load_id !== old.load_id) return next;
  // This is one first attempt, not a count of calls or an ordinal across
  // server restarts. Delayed snapshots cannot reverse its observed outcome.
  if (['returned', 'failed', 'cancelled'].includes(old.first_call)
      || (old.first_call === 'running' && next.first_call === 'pending')) return old;
  return next;
}

function executionDiagnosticCaption(value) {
  const diagnostic = readExecutionDiagnostics(value);
  if (!diagnostic) return null;
  const short = {
    pending: 'first call pending', running: 'first call running', returned: 'first call returned',
    failed: 'first call failed', cancelled: 'first call cancelled',
  }[diagnostic.first_call];
  const state = {
    pending: 'Last reported first-call state: pending; no forward attempt has been observed. The next update may wait for the first callback.',
    running: 'First call was observed running; its first callback may be delayed.',
    returned: 'First call returned after ownership checks and policy exit.',
    failed: 'First call failed.', cancelled: 'First call was cancelled.',
  }[diagnostic.first_call];
  const stage = { forward: 'pipeline call', verification: 'ownership verification', policy_exit: 'compiler policy exit' }[diagnostic.failure_stage];
  const detail = stage ? ` Failure stage: ${stage}. Select Eager before trying an ordinary load.` : '';
  return { short, full: `Regional registered. ${state}${detail} Warm kernels, cache hits and performance are unverified.` };
}

function receiveModelExecutionDiagnostics(status, asked) {
  if (!Number.isFinite(asked) || asked < modelExecutionAsked) return;
  modelExecutionAsked = asked;
  const scope = JSON.stringify([status.started_at ?? null, status.model_revision ?? null, status.model_id ?? null]);
  const prior = modelExecutionObservation?.scope === scope ? modelExecutionObservation.value : null;
  modelExecutionObservation = { scope, model: status.model_id, revision: status.model_revision,
    server: status.started_at, value: status.state === 'ready'
      ? mergeExecutionDiagnostics(prior, status.execution_diagnostics) : null };
}

function modelExecutionNote() {
  const observed = modelExecutionObservation;
  const runtime = modelRuntimeSettings?.model_id ?? loadedModel;
  if (serverOffline || modelState !== 'ready' || !observed || observed.model !== runtime
      || observed.revision !== modelRevision || observed.server !== serverStartedAt) return '';
  const caption = executionDiagnosticCaption(observed.value);
  return caption ? `Loaded ${modelLabel(runtime)}: ${caption.full}` : '';
}

function setExecutionProgress(jobId, value, spoken, shown = null) {
  if (jobId !== currentJob) return;
  const caption = executionDiagnosticCaption(value);
  setProgressText(caption ? `${spoken}; ${caption.short}.` : spoken,
    caption && shown !== null ? `${caption.short} · ${shown}` : shown);
  els.progressText.title = caption?.full ?? '';
  if (caption) els.progressText.setAttribute('aria-description', caption.full);
}

function setPerformanceChoice(control, value, allowed) {
  invalidPerformanceChoices.delete(control);
  for (const option of [...control.options]) {
    if (option.dataset.invalidPerformance === 'true') option.remove();
  }
  if (typeof value === 'string' && allowed.includes(value)) {
    control.value = value;
    return;
  }
  // Keep the raw value only in state. Neither an option label nor a URL gets
  // the arbitrary saved payload, even for malformed objects/future strings.
  invalidPerformanceChoices.set(control, { value });
  const option = new Option('Unavailable saved choice', INVALID_PERFORMANCE_VALUE);
  option.dataset.invalidPerformance = 'true';
  control.add(option);
  control.value = INVALID_PERFORMANCE_VALUE;
}

function performanceChoiceInvalid(control) {
  return control.value === INVALID_PERFORMANCE_VALUE && invalidPerformanceChoices.has(control);
}

function performanceChoiceValue(control) {
  return performanceChoiceInvalid(control) ? invalidPerformanceChoices.get(control).value : control.value;
}

function samePerformanceValue(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function setStepCache(value) {
  setPerformanceChoice(els.stepCache, value, ['', 'fbc-005-v1']);
}

function stepCacheValue() {
  return performanceChoiceValue(els.stepCache);
}

function setLoaderChoice(key, value) {
  const control = key === 'compilation' ? els.modelCompilation : els.modelAttention;
  setPerformanceChoice(control, value, key === 'compilation' ? ['', 'regional-v1'] : ['', 'cudnn-v1']);
}

function invalidLoaderDraftReason() {
  return invalidExecutionDraft !== null || performanceChoiceInvalid(els.modelCompilation)
    || performanceChoiceInvalid(els.modelAttention) ? INVALID_LOADER_REASON : '';
}

function acceptLoaderChoice(key) {
  const control = key === 'compilation' ? els.modelCompilation : els.modelAttention;
  if (performanceChoiceInvalid(control)) return;
  setLoaderChoice(key, control.value);
  if (invalidExecutionDraft !== null) {
    invalidDraftReplacements.add(key);
    if (invalidDraftReplacements.size === 2) {
      replacedExecutionEnvelope = invalidExecutionDraft.raw;
      invalidExecutionDraft = null;
      invalidDraftReplacements.clear();
    }
  }
}

function loaderExecution(compilation = '', attention = '') {
  const spec = {};
  if (compilation) spec.compilation = compilation;
  if (attention) spec.attention_backend = attention;
  return Object.keys(spec).length ? spec : null;
}

function executionSame(a, b) {
  return JSON.stringify(loaderExecution(a?.compilation, a?.attention_backend))
    === JSON.stringify(loaderExecution(b?.compilation, b?.attention_backend));
}

function executionLoadHold() {
  return invalidLoaderDraftReason() || (!selectedExecutionAvailable ? (selectedExecutionReason || EXECUTION_UNAVAILABLE) : '');
}

function executionSubmitHold() {
  return executionLoadHold() || (performanceChoiceInvalid(els.stepCache) ? INVALID_CACHE_REASON
    : els.stepCache.value ? EXECUTION_UNAVAILABLE : '');
}

function composerExecution() {
  const spec = { ...(selectedExecution ?? {}) };
  // The fixed invalid option keeps request builders refused without forwarding
  // arbitrary stored payloads. panelState/Undo retain the exact raw value.
  if (els.stepCache.value) spec.step_cache = els.stepCache.value;
  return Object.keys(spec).length ? spec : null;
}

function refuseComposerExecution() {
  if (!executionSubmitHold()) return false;
  showError(submitHold({ composer: true }));
  if (serverOffline) void refreshStatus();
  return true;
}

function executionDraftOptions() {
  return modelCatalogueDetails[modelChoice()]?.execution_options ?? modelExecutionOptions;
}

function executionConfigurationHold() {
  if (invalidLoaderDraftReason()) return INVALID_LOADER_REASON;
  const selected = modelPickerSettings();
  if (!selected.execution) return '';
  const detail = modelCatalogueDetails[modelChoice()];
  return detail?.execution_configuration_reason
    || (!detail?.execution_options ? 'The server has not validated this performance configuration.' : '');
}

function executionPickerUnavailable() {
  return invalidLoaderDraftReason() || (modelPickerSettings().execution ? EXECUTION_UNAVAILABLE : '');
}

function syncExecutionControls() {
  const choices = executionDraftOptions();
  for (const [control, help, key] of [
    [els.modelCompilation, els.modelCompilationHelp, 'compilation'],
    [els.modelAttention, els.modelAttentionHelp, 'attention'],
  ]) {
    const option = choices?.[key]?.find((item) => (item.value ?? '') === control.value);
    const base = invalidExecutionDraft !== null || performanceChoiceInvalid(control) ? INVALID_LOADER_REASON
      : option?.configuration_reason || option?.unavailable_reason
      || (control.value ? EXECUTION_UNAVAILABLE : key === 'compilation'
        ? 'Eager is the existing execution mode.' : 'Native is the default attention choice.');
    const text = base + (key === 'attention' && executionEnvironmentWarning ? ` ${executionEnvironmentWarning}` : '');
    if (help.textContent !== text) help.textContent = text;
    help.title = text;
    control.title = text;
  }
  const cacheOption = executionCacheOptions?.find((option) => (option.value ?? '') === els.stepCache.value);
  const cacheReason = cacheOption?.configuration_reason || cacheOption?.unavailable_reason || EXECUTION_UNAVAILABLE;
  const note = performanceChoiceInvalid(els.stepCache) ? INVALID_CACHE_REASON
    : els.stepCache.value ? `Saved in this browser. ${cacheReason} Choose Off to render.`
    : 'Off. FBC 0.05 v1 is experimental and unavailable in this build.';
  if (els.stepCacheNote.textContent !== note) els.stepCacheNote.textContent = note;
  els.stepCacheNote.classList.toggle('sr-only', !performanceChoiceInvalid(els.stepCache));
  els.stepCache.title = note;
  const why = executionLoadHold();
  els.load.setAttribute('aria-disabled', String(!!why));
  if (why) els.load.title = why;
}

function receiveExecutionStatus(status, asked = Date.now()) {
  receiveModelExecutionDiagnostics(status, asked);
  selectedExecution = loaderExecution(status.execution?.compilation, status.execution?.attention_backend);
  selectedExecutionReason = status.execution_unavailable_reason || '';
  selectedExecutionAvailable = typeof status.execution_available === 'boolean' ? status.execution_available : !selectedExecution;
  executionEnvironmentWarning = status.attention_environment_warning || '';
  executionCacheOptions = status.execution_options?.step_cache ?? null;
  syncExecutionControls();
}

function readExecutionDraft() {
  let raw;
  try { raw = localStorage.getItem(EXECUTION_DRAFT_KEY); }
  catch { return { raw: null, value: {}, malformed: false }; }
  if (raw === null) return { raw, value: {}, malformed: false };
  try {
    const value = JSON.parse(raw);
    const malformed = value === null || typeof value !== 'object' || Array.isArray(value)
      || Object.keys(value).some((key) => !['compilation', 'attention'].includes(key));
    return { raw, value, malformed };
  } catch { return { raw, value: null, malformed: true }; }
}

function rememberExecutionDraft() {
  // An unreadable/future envelope is retained byte-for-byte until both fields
  // have been explicitly replaced, or Revert deliberately resolves it.
  if (invalidExecutionDraft !== null) return;
  const prior = readExecutionDraft();
  if (prior.malformed && prior.raw !== replacedExecutionEnvelope) {
    invalidExecutionDraft = { raw: prior.raw };
    invalidDraftReplacements.clear();
    return; // An intervening malformed store is never overwritten by a merge.
  }
  const next = !prior.malformed ? { ...prior.value } : {};
  for (const [key, control] of [['compilation', els.modelCompilation], ['attention', els.modelAttention]]) {
    const value = performanceChoiceValue(control);
    if (modelPickerEditedKeys.has(key) && !samePerformanceValue(value, executionDraftBase[key])) next[key] = value;
  }
  try { localStorage.setItem(EXECUTION_DRAFT_KEY, JSON.stringify(next)); } catch { /* Session edits still work. */ }
  replacedExecutionEnvelope = null;
  executionDraftBase = { compilation: performanceChoiceValue(els.modelCompilation), attention: performanceChoiceValue(els.modelAttention) };
}

function restoreExecutionDraft() {
  const saved = readExecutionDraft();
  invalidDraftReplacements.clear();
  replacedExecutionEnvelope = null;
  invalidExecutionDraft = saved.malformed ? { raw: saved.raw } : null;
  executionDraftBase = !saved.malformed ? { ...saved.value } : {};
  for (const key of ['compilation', 'attention']) {
    if (saved.malformed || Object.hasOwn(saved.value, key)) {
      setLoaderChoice(key, saved.malformed ? saved.value : saved.value[key]);
      modelPickerEditedKeys.add(key);
    }
  }
}

function forgetExecutionDraft(selected = modelPickerSettings(), { revert = false } = {}) {
  const saved = readExecutionDraft();
  const draft = !saved.malformed ? { ...saved.value } : null;
  const ownedEnvelope = revert && invalidExecutionDraft !== null && saved.raw === invalidExecutionDraft.raw;
  if (draft !== null) {
    for (const [key, control] of [['compilation', els.modelCompilation], ['attention', els.modelAttention]]) {
      const value = revert ? performanceChoiceValue(control) : selected[key];
      if (Object.hasOwn(draft, key) && samePerformanceValue(draft[key], value)) delete draft[key];
    }
  }
  try {
    if (draft !== null && Object.keys(draft).length) localStorage.setItem(EXECUTION_DRAFT_KEY, JSON.stringify(draft));
    else if (draft !== null || ownedEnvelope) localStorage.removeItem(EXECUTION_DRAFT_KEY);
  } catch { /* No persisted draft. */ }
  if (revert) {
    invalidExecutionDraft = null;
    replacedExecutionEnvelope = null;
    invalidDraftReplacements.clear();
    setLoaderChoice('compilation', modelRuntimeSettings?.compilation ?? '');
    setLoaderChoice('attention', modelRuntimeSettings?.attention ?? '');
  }
  executionDraftBase = { compilation: selected.compilation, attention: selected.attention };
}
