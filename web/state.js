// Local Diffusion's page script, part 1 of 12: shared state, the element table and
// the fetch/error helpers. index.html loads these files in order as classic
// scripts, so they share one global scope: a top-level `let`, `const` or
// function in any of them is visible to all the others and to a smoke step's
// `eval` (tools/browser.mjs). Only app.js, the last, runs anything at load;
// the rest only declare. Order: state, sizes, status, generation, panel,
// gallery, sheet, lightbox, refine, tags, analysis, app.
'use strict';

const $ = (id) => document.getElementById(id);
const PAGE = 48;
// A sheet wants the whole run in one go - it is a matrix, and a matrix with a
// page boundary through it is a puzzle. The rows are small and the thumbnails
// load lazily, so a full grid is cheaper than it sounds. The size follows the
// server's grid ceiling (/api/status), so a raised LOCAL_DIFFUSION_MAX_GRID_IMAGES does
// not silently drop the earliest seeds off a sheet.
let sheetPage = 1024;

const els = {
  prompt: $('prompt'), negative: $('negative'), negField: $('negative-field'),
  negNote: $('neg-note'), promptTip: $('prompt-tip'), sizeTip: $('size-tip'),
  examplesBtn: $('btn-examples'), examplesMenu: $('examples-menu'), expandPrompt: $('btn-expand'),
  tagHelper: $('tag-helper'), tagsBtn: $('btn-tags'), tagsMenu: $('tags-menu'),
  tagSuggestions: $('tag-suggestions'), tagValidation: $('tag-validation'),
  tagValidationHome: $('tag-validation-home'), tagValidationWide: $('tag-validation-wide'),
  analyze: $('btn-analyze'), promptAnalysis: $('prompt-analysis'),
  analysisModel: $('analysis-model'), analysisSummary: $('analysis-summary'),
  analysisStale: $('analysis-stale'), analysisAgain: $('analysis-again'), analysisClose: $('analysis-close'),
  analysisEncoders: $('analysis-encoders'), analysisOrder: $('analysis-order'),
  analysisFindings: $('analysis-findings'), analysisCritic: $('analysis-critic'),
  criticAsk: $('critic-ask'), criticLine: $('critic-line'), criticResult: $('critic-result'),
  compareOptions: $('compare-options'), compareSeeds: $('compare-seeds'), compareBatch: $('compare-batch'),
  compareOwn: $('compare-own'), compareOwnFill: $('compare-own-fill'),
  comparePreview: $('compare-preview'), compareRun: $('compare-run'),
  promptTranslationField: $('prompt-translation-field'), promptTranslation: $('prompt-translation'),
  width: $('width'), height: $('height'), steps: $('steps'), cfg: $('cfg'),
  mu: $('mu'), muField: $('mu-field'), seed: $('seed'), batch: $('batch'), runs: $('runs'),
  trueCfg: $('true-cfg'), trueCfgField: $('true-cfg-field'), trueCfgTip: $('true-cfg-tip'),
  stepsTip: $('steps-tip'), cfgTip: $('cfg-tip'), muTip: $('mu-tip'),
  referenceField: $('reference-field'), referenceList: $('reference-list'),
  referenceNote: $('reference-note'), referenceTip: $('reference-tip'),
  lora: $('lora'), loraScale: $('lora-scale'), loraField: $('lora-field'),
  loraTip: $('lora-tip'), gridLoraScale: $('grid-lora-scale'),
  gridLoraField: $('grid-lora-field'),
  generate: $('btn-generate'), dice: $('btn-dice'), unload: $('btn-unload'), load: $('btn-load'),
  reset: $('btn-reset'),
  cancel: $('btn-cancel'), pause: $('btn-pause'),
  queueNote: $('queue-note'), queueText: $('queue-text'), cancelQueued: $('btn-cancel-queued'),
  progress: $('progress'), fill: $('progress-fill'), progressText: $('progress-text'),
  stepRow: $('step-row'), stepFill: $('step-fill'), stepText: $('step-text'),
  cellNote: $('cell-note'),
  gridSection: $('grid-section'), gridPrompt: $('grid-prompt'),
  gridSteps: $('grid-steps'), gridCfg: $('grid-cfg'),
  gridMu: $('grid-mu'), gridMuField: $('grid-mu-field'),
  gridSeeds: $('grid-seeds'), gridSeedList: $('grid-seed-list'),
  gridBudget: $('grid-budget'), gridPreview: $('grid-preview'),
  gridRun: $('btn-grid'),
  error: $('error'), viewer: $('viewer'), viewerNote: $('viewer-note'), gallery: $('gallery'),
  count: $('gallery-count'), search: $('search'), favOnly: $('fav-only'),
  allModels: $('all-models'), otherModels: $('other-models'),
  more: $('btn-more'), presets: $('presets'), tiers: $('tiers'),
  orientation: $('btn-orientation'), sizeHint: $('size-hint'),
  gridResume: $('grid-resume'),
  sheetBtn: $('btn-sheet'), rerender: $('btn-rerender'), deleteUnstarred: $('btn-delete-unstarred'),
  sheet: $('sheet'), sheetControls: $('sheet-controls'),
  sheetFixed: $('sheet-fixed'), sheetGrid: $('sheet-grid'), sheetNote: $('sheet-note'),
  runHint: $('run-hint'),
  lastRun: $('last-run'), lastRunText: $('last-run-text'),
  lastRunShow: $('last-run-show'), lastRunDismiss: $('last-run-dismiss'),
  runList: $('run-list'),
  dot: $('status-dot'), statusText: $('status-text'), vram: $('vram'),
  lightbox: $('lightbox'), lbImg: $('lb-img'), lbMeta: $('lb-meta'), lbClose: $('lb-close'),
  lbPrompt: $('lb-prompt'), lbSettings: $('lb-settings'), lbCount: $('lb-count'),
  lbError: $('lb-error'), lbStage: $('lb-stage'), lbZoom: $('lb-zoom'),
  lbFit: $('lb-fit'), lbOne: $('lb-one'),
  lbReuse: $('lb-reuse'), lbRefine: $('lb-refine'), lbPrev: $('lb-prev'), lbNext: $('lb-next'),
  lbFav: $('lb-fav'), lbDelete: $('lb-delete'),
  refineDialog: $('refine-dialog'), refineTitle: $('refine-title'), refineClose: $('refine-close'),
  refineThumb: $('refine-thumb'), refineSourceLine: $('refine-source-line'),
  refinePromptLine: $('refine-prompt-line'), refineUnavailable: $('refine-unavailable'),
  refineScales: $('refine-scales'), refineWidth: $('refine-width'), refineHeight: $('refine-height'),
  refineSizeHint: $('refine-size-hint'), refineStrength: $('refine-strength'),
  refineStrengthTip: $('refine-strength-tip'), refineVersions: $('refine-versions'),
  refineSeed: $('refine-seed'), refineSeedHint: $('refine-seed-hint'),
  refineMore: $('refine-more'), refineMoreNote: $('refine-more-note'),
  refinePrompt: $('refine-prompt'), refineNegative: $('refine-negative'),
  refineNegField: $('refine-negative-field'), refineNegNote: $('refine-neg-note'),
  refineSteps: $('refine-steps'), refineCfg: $('refine-cfg'),
  refineMu: $('refine-mu'), refineMuField: $('refine-mu-field'),
  refineTrueCfg: $('refine-true-cfg'), refineTrueCfgField: $('refine-true-cfg-field'),
  refineCarried: $('refine-carried'), refinePreview: $('refine-preview'),
  refineError: $('refine-error'), refineCancel: $('refine-cancel'), refineRun: $('refine-run'),
};

// Aspect × megapixel tier. Every side is a multiple of 16 (the VAE downsamples by
// 16) and every entry lands within a few percent of its tier. The whole 4 MP row
// is within 0.4% of the old 2K preset's pixel
// count, so it costs the same VRAM whichever framing you pick. The 1.5 MP row's
// non-square entries are Qwen-Image's README shapes, and the row sits inside
// FLUX.1 Krea's recommended 1024-1280 px (1-1.64 MP). The 1 MP row carries
// the Animagine, ERNIE-Image and Qwen-Image model-card aspects.
const SIZES = {
  '1:1':  { 1: [1024, 1024], 1.5: [1248, 1248], 2: [1440, 1440], 4: [2048, 2048] },
  '7:9':  { 1: [896, 1152],  1.5: [1120, 1440], 2: [1264, 1632], 4: [1808, 2320] },
  '9:7':  { 1: [1152, 896],  1.5: [1440, 1120], 2: [1632, 1264], 4: [2320, 1808] },
  '3:4':  { 1: [896, 1200],  1.5: [1104, 1472], 2: [1248, 1664], 4: [1776, 2368] },
  '4:3':  { 1: [1200, 896],  1.5: [1472, 1104], 2: [1664, 1248], 4: [2368, 1776] },
  '2:3':  { 1: [848, 1264],  1.5: [1056, 1584], 2: [1184, 1776], 4: [1680, 2496] },
  '3:2':  { 1: [1264, 848],  1.5: [1584, 1056], 2: [1776, 1184], 4: [2496, 1680] },
  '4:7':  { 1: [768, 1344],  1.5: [960, 1680],  2: [1088, 1904], 4: [1552, 2704] },
  '7:4':  { 1: [1344, 768],  1.5: [1680, 960],  2: [1904, 1088], 4: [2704, 1552] },
  '9:16': { 1: [768, 1360],  1.5: [928, 1664],  2: [1088, 1920], 4: [1536, 2736] },
  '16:9': { 1: [1360, 768],  1.5: [1664, 928],  2: [1920, 1088], 4: [2736, 1536] },
  '5:12': { 1: [640, 1536],  1.5: [800, 1920],  2: [928, 2224],  4: [1328, 3168] },
  '12:5': { 1: [1536, 640],  1.5: [1920, 800],  2: [2224, 928],  4: [3168, 1328] },
};
const ASPECTS = Object.keys(SIZES);
const TIERS = [1, 1.5, 2, 4];

// The two axes are picked independently - changing one holds the other - so the
// same framing at four sizes is four clicks in the same place.
let pickedAspect = '1:1';
let pickedTier = 1;

// Paging is by id, not by offset: the gallery is a table a sweep is inserting
// into, and an offset over it repeats the tail of one page at the head of the
// next and skips as many older rows for good. The anchor is the smallest id on
// the page (the last record - they arrive newest first), and the server says
// whether anything older exists, exactly.
let galleryMore = false;
// The grid pages itself: an IntersectionObserver on the "Load more" button,
// armed in app.js and re-armed by `watchMore` after every page, fetches the
// next page as the button scrolls near the viewport. The button stays as the
// sign that older rows exist and as the retry after a page that failed.
let moreWatcher = null;
let total = 0;
let busy = false;
// Which sampling controls the user has taken off the checkpoint's profile.
// Per control, not one flag for all three: touching Seed before the first
// /api/status used to freeze steps, CFG and mu at the page's fallback numbers,
// and on Turbo that is 28 steps at CFG 3.5 for a model trained for 8 at 0.
const owned = { steps: false, cfg: false, mu: false, trueCfg: false };
// The checkpoint the server is running, once /api/status has said. Reuse
// compares a record against it: steps, CFG and mu are checkpoint-specific.
let loadedModel = null;
let statusPending = false;
// The job whose stream the page is watching, and the ids queued behind it.
// The server was written for a queue - jobs wait on its gate, a queued one
// can be cancelled on the spot - and the browser used to refuse a second
// submission outright, so the queue was unreachable from the UI.
let currentJob = null;
let queuedIds = [];
let submitting = false;
// Jobs that open as a sheet when their stream reports done, by job id:
// the prompt check's comparison, which is a question the sheet answers. Kept
// here rather than on the stream, since the job may queue behind another and
// be picked up by watchNext long after its submission returned.
const jobsToShow = new Set();
// ERNIE's optional rewriter is an editor action, not part of image generation.
// Its first use may load a separate 7.7 GB component, so the button owns its
// own busy state and never lets a late response overwrite text edited meanwhile.
let promptExpansionSupported = false;
let expandingPrompt = false;
let translatedSource = null;
// The prompt critic is the other editor action on the card: a local model
// started for one answer and stopped after it, so it holds the same things
// Expand does while it runs. Supported and ready follow /api/status - ready
// is whether a first use still has llama.cpp and the model to download.
let promptCritiqueSupported = false;
let promptCritiqueReady = false;
// ModelStatus.prompt_critic: the model's name, whether it is the pinned pair
// and what a first use downloads - what the line under the button says.
let criticStatus = null;
let critiquing = false;
// One gallery page in flight at a time, and a token so a page that was already
// on its way when the filters changed can be dropped rather than appended.
let galleryPending = null;
let galleryEpoch = 0;
// The CFG above which a negative prompt does anything: 1 on FLUX, where
// guidance_scale is a conditioning value baked into the model and only
// true_cfg_scale runs an uncond pass, 0 on Krea 2, which guides from 0 up. Not
// a control, so unlike the seeded defaults it follows the server all session.
let cfgThreshold = 1;
// FLUX.2 klein's pipeline takes no negative prompt at any CFG; the field greys
// out for the checkpoint rather than for the value.
let negativeSupported = true;
// Whether real CFG is a knob of its own (FLUX.1, where the CFG box is the
// distilled guidance value the model takes as an input) or the CFG box is real
// CFG already (Krea 2, klein). Decides which box the negative prompt follows,
// and whether the True CFG box is live at all.
let trueCfgSupported = false;
// Whether this checkpoint uses a flow-matching schedule to which Mu applies.
// SDXL accepts custom sigmas but is not flow matching, so argument presence is
// not enough to make this control safe.
let muSupported = true;
// Free space on the images volume, from the last status poll; null if unknown.
let diskFreeMb = null;
// The gallery ids this run will be conditioned on, in the order they were
// added - which is the order their tokens are concatenated in. Ids and not
// files: the server resolves them through the images table, so nothing here
// can name a path.
let referenceIds = [];
// Whether the loaded pipeline takes reference images at all (klein does,
// nothing else here), and whether this installation can apply an adapter
// (peft, and a base that is not pre-quantized). Both follow the server all
// session, like the CFG threshold.
let referenceSupported = false;
let loraSupported = false;
// The refine pass: whether the checkpoint has one (diffusers ships an
// image-to-image pipeline beside its own - FLUX.1, SDXL, Z-Image, Qwen-Image),
// following the server like the two above, and the record the Refine dialog
// is open on - null while it is closed. Only the dialog sends a pass: the
// panel's requests never carry a source.
let refineSupported = false;
let refineSource = null;
// The adapter a pass carries over from its record, {name, scale}, or null:
// shown in the dialog rather than editable there, since the adapter is part
// of how the picture was made.
let refineLora = null;
// What the scale buttons offer, and what the dialog opens at:
// backend/refine.py's DEFAULT_SCALE. Strength starts at the checkpoint's
// profile (`defaults.refine_strength`): what a strength does depends on the
// schedule it indexes into. REFINE_STRENGTH is the server's own fallback,
// DEFAULT_STRENGTH, for the moment before the first status lands.
const REFINE_SCALES = [1, 1.25, 1.5, 2];
const REFINE_SCALE = 1.5;
const REFINE_STRENGTH = 0.4;
let lorasLoaded = false;
// Whether the adapter list was fetched with the model loaded: the fit check
// answers from the checkpoint's configs before that and from the live
// denoiser after, so the list is fetched once more when the load lands.
let lorasLive = false;
// The LoRA tooltip as the page shipped it, put back when a refetch has
// nothing to say over it.
const loraTipDefault = els.loraTip.textContent;
// The checkpoint's prompting guide: the pixel range its authors document
// (megapixels, [low, high], null until a status lands) for the size line's
// warning, and the example prompts behind the Examples button.
let sizeRange = null;
let sizeRangeWord = 'documented';
// Null means the checkpoint accepts any framing in its pixel range. Animagine
// publishes a finite table, so its guide names the aspect buttons too.
let sizeAspects = null;
let guideResolutions = null;
let examples = [];
// Null for prose checkpoints. A tag checkpoint supplies its own finite,
// documented vocabulary; the browser never guesses a generic tag cloud.
let tagHelper = null;
let tagVocabularyEnabled = false;
let tagSuggestionTimer = null;
let tagValidationTimer = null;
let tagSuggestionEpoch = 0;
let tagValidationEpoch = 0;
let tagSuggestions = [];
let tagSuggestionIndex = -1;
let lastTagValidationResults = [];
let lastTagValidationAvailable = true;
let lastTagValidationDetail = '';
let draggedTagIndex = null;
// The inline chip editor: at most one open, inserting before `tagInsertAt` or
// replacing the item at `tagEditAt`.
let tagInsertAt = null;
let tagEditAt = null;
let tagEditorSuggestionTimer = null;
let tagEditorSuggestionEpoch = 0;
// A remembered adapter name waiting for the list it belongs to.
let pendingLora = '';

const humanBytes = (n) => {
  let size = n;
  for (const unit of ['B', 'KB', 'MB', 'GB']) {
    if (size < 1024 || unit === 'GB') return `${size.toFixed(1)} ${unit}`;
    size /= 1024;
  }
  return `${size.toFixed(1)} GB`;
};

// The two strips the lightbox can walk: what this job produced, and what the
// gallery has paged in so far. Both are kept in the order they are displayed.
let viewerRecords = [];
let galleryRecords = [];
let lbList = [];
let lbIndex = -1;

// Contact sheet. `sheetAxes` is what the user picked; null means "work it out
// from the images", which is what happens on the first render of a new set.
let sheetMode = false;
let sheetAxes = { rows: null, cols: null, pins: {} };
let sheetRecords = [];

// ------------------------------------------------------------------ helpers

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* non-JSON body */ }
    throw new Error(readableDetail(detail));
  }
  return res.status === 204 ? null : res.json();
}

// FastAPI's own 422s carry a list of {loc, msg, ctx, input} objects, which as
// JSON.stringify is a wall of braces in the error slot. Ours are plain strings.
function readableDetail(detail) {
  if (typeof detail === 'string') return detail;
  if (!Array.isArray(detail)) return JSON.stringify(detail);
  return detail
    .map((e) => {
      const field = (e.loc ?? []).filter((part) => part !== 'body').join('.');
      return field ? `${field}: ${e.msg}` : e.msg;
    })
    .join('; ');
}

// `note` styles the same slot as muted rather than red, for outcomes that are
// not failures - a cancelled run in particular.
function showError(message, note = false) {
  // Unhide *before* writing: the slot is role=alert, and text written into a
  // display:none node is never announced.
  els.error.hidden = !message;
  els.error.textContent = message || '';
  els.error.classList.toggle('note', note);
  // The lightbox covers the panel's slot; mirror into its own while it is up.
  const mirror = els.lightbox.open;
  els.lbError.hidden = !(mirror && message);
  els.lbError.textContent = mirror ? message || '' : '';
  els.lbError.classList.toggle('note', note);
}
