// Local Diffusion's page script: shared state, the element table and
// the fetch/error helpers. index.html loads these files in order as classic
// scripts, so they share one global scope: a top-level `let`, `const` or
// function in any of them is visible to all the others and to a smoke step's
// `eval` (tools/browser.mjs). Only app.js, the last, runs anything at load;
// the rest only declare. index.html is the load-order list: state first,
// each feature's declarations together, and app last. Keep shared bindings
// at top level so the page and browser smokes can reach them by name.
'use strict';

const $ = (id) => document.getElementById(id);
// Whether an element is drawn - not display: none or inside one, and not
// visibility: hidden, as the gallery's column is under the prompt check with
// Gallery beside (style.css) - so that focus sent to it would land. True
// where the browser cannot say.
const isShown = (el) => el.checkVisibility?.({ visibilityProperty: true, checkVisibilityCSS: true }) ?? true;
const PAGE = 48;
// A sheet wants the whole run in one go - it is a matrix, and a matrix with a
// page boundary through it is a puzzle. The rows are small and the thumbnails
// load lazily, so a full grid is cheaper than it sounds. The size follows the
// server's grid ceiling (/api/status), so a raised LOCAL_DIFFUSION_MAX_GRID_IMAGES does
// not silently drop the earliest seeds off a sheet.
let sheetPage = 1024;

const els = {
  prompt: $('prompt'), negative: $('negative'), negField: $('negative-field'),
  negNote: $('neg-note'), negEnable: $('neg-enable'), promptTip: $('prompt-tip'), sizeTip: $('size-tip'),
  examplesBtn: $('btn-examples'), examplesMenu: $('examples-menu'), expandPrompt: $('btn-expand'),
  examplesBox: $('examples'),
  promptNote: $('prompt-note'), promptNoteText: $('prompt-note-text'), promptNoteUndo: $('prompt-note-undo'),
  promptNoteDiscard: $('prompt-note-discard'),
  tagHelper: $('tag-helper'), tagsBtn: $('btn-tags'), tagsMenu: $('tags-menu'),
  tagSuggestions: $('tag-suggestions'), tagValidation: $('tag-validation'),
  tagValidationHome: $('tag-validation-home'), tagValidationWide: $('tag-validation-wide'),
  tagChipsWide: $('tag-chips-wide'),
  tagTip: $('tag-tip'), tagSummary: $('tag-summary'), promptTokens: $('prompt-tokens'),
  analyze: $('btn-analyze'), promptAnalysis: $('prompt-analysis'),
  analysisModel: $('analysis-model'), analysisSummary: $('analysis-summary'),
  analysisStale: $('analysis-stale'), analysisAgain: $('analysis-again'), analysisClose: $('analysis-close'),
  analysisEncoders: $('analysis-encoders'), analysisOrder: $('analysis-order'),
  analysisFindings: $('analysis-findings'), analysisCritic: $('analysis-critic'),
  criticAsk: $('critic-ask'), criticLine: $('critic-line'), criticResult: $('critic-result'),
  criticProgress: $('critic-progress'), criticCost: $('critic-cost'),
  criticClock: $('critic-clock'), analysisStatus: $('analysis-status'),
  compareOptions: $('compare-options'), compareSeeds: $('compare-seeds'), compareBatch: $('compare-batch'),
  compareOwn: $('compare-own'), compareOwnFill: $('compare-own-fill'), compareOwnNote: $('compare-own-note'),
  comparePreview: $('compare-preview'), compareRun: $('compare-run'),
  promptTranslationField: $('prompt-translation-field'), promptTranslation: $('prompt-translation'),
  width: $('width'), height: $('height'), steps: $('steps'), cfg: $('cfg'),
  mu: $('mu'), muField: $('mu-field'), seed: $('seed'), batch: $('batch'), runs: $('runs'),
  trueCfg: $('true-cfg'), trueCfgField: $('true-cfg-field'), trueCfgTip: $('true-cfg-tip'),
  stepsTip: $('steps-tip'), cfgTip: $('cfg-tip'), muTip: $('mu-tip'),
  cfgLabel: $('cfg-label'), gridCfgLabel: $('grid-cfg-label'), refineCfgLabel: $('refine-cfg-label'),
  referenceField: $('reference-field'), referenceList: $('reference-list'),
  referenceNote: $('reference-note'), referenceTip: $('reference-tip'),
  lora: $('lora'), loraScale: $('lora-scale'), loraField: $('lora-field'),
  loraTip: $('lora-tip'), gridLoraScale: $('grid-lora-scale'),
  gridLoraField: $('grid-lora-field'), gridLoraTip: $('grid-lora-tip'),
  generate: $('btn-generate'), dice: $('btn-dice'), unload: $('btn-unload'), load: $('btn-load'),
  models: $('btn-models'), modelPicker: $('model-picker'),
  modelTable: $('model-table'), modelTableHead: $('model-table-head'),
  modelSwitch: $('btn-model-switch'), modelSave: $('btn-model-save'), modelPickerNote: $('model-picker-note'),
  modelPickerNoteText: $('model-picker-note-text'), modelRevert: $('btn-model-revert'), modelPickerHelp: $('model-picker-help'),
  modelQuantization: $('model-quantization'), modelOffload: $('model-offload'),
  modelQuantizationHelp: $('model-quantization-help'), modelOffloadHelp: $('model-offload-help'),
  modelFormat: $('model-format'), modelFormatField: $('model-format-field'), modelFormatHelp: $('model-format-help'),
  modelLoadOnStartup: $('model-load-on-startup'),
  modelPickerBody: $('model-picker-body'),
  modelShots: $('model-shots'), modelShotsLabel: $('model-shots-label'), modelShotsAll: $('btn-model-shots-all'),
  modelShotsList: $('model-shots-list'), modelShotsEmpty: $('model-shots-empty'),
  reset: $('btn-reset'), pngButton: $('btn-png'), pngFile: $('png-file'),
  cancel: $('btn-cancel'), pause: $('btn-pause'),
  queueNote: $('queue-note'), queueText: $('queue-text'), queueList: $('queue-list'),
  cancelQueued: $('btn-cancel-queued'),
  progress: $('progress'), fill: $('progress-fill'), progressText: $('progress-text'),
  progressNow: $('progress-now'), newImage: $('btn-new-image'),
  stepRow: $('step-row'), stepFill: $('step-fill'), stepText: $('step-text'),
  cellNote: $('cell-note'), finishNote: $('finish-note'), jobSaid: $('job-said'),
  gridSection: $('grid-section'), gridPrompt: $('grid-prompt'),
  gridSteps: $('grid-steps'), gridCfg: $('grid-cfg'),
  gridMu: $('grid-mu'), gridMuField: $('grid-mu-field'), gridMuTip: $('grid-mu-tip'),
  gridSeeds: $('grid-seeds'), gridSeedList: $('grid-seed-list'),
  gridBudget: $('grid-budget'), gridPreview: $('grid-preview'),
  gridRun: $('btn-grid'),
  error: $('error'), galleryError: $('gallery-error'), starSaid: $('star-said'),
  viewer: $('viewer'), viewerNote: $('viewer-note'), gallery: $('gallery'), galleryEmpty: $('gallery-empty'),
  confirmDialog: $('confirm-dialog'), confirmTitle: $('confirm-title'), confirmText: $('confirm-text'),
  confirmCountField: $('confirm-count-field'), confirmCountLabel: $('confirm-count-label'),
  confirmCount: $('confirm-count'), confirmCancel: $('confirm-cancel'), confirmOk: $('confirm-ok'),
  viewerExamples: $('viewer-examples'), recent: $('recent'), recentList: $('recent-list'),
  count: $('gallery-count'), search: $('search'), favOnly: $('fav-only'), cardView: $('card-view'),
  allModels: $('all-models'), otherModels: $('other-models'),
  stage: $('stage'), library: $('library'), libraryScroll: $('library-scroll'), besideTick: $('gallery-beside'),
  more: $('btn-more'), presets: $('presets'), tiers: $('tiers'),
  orientation: $('btn-orientation'), sizeHint: $('size-hint'),
  gridResume: $('grid-resume'),
  sheetBtn: $('btn-sheet'), rerender: $('btn-rerender'), deleteUnstarred: $('btn-delete-unstarred'),
  select: $('btn-select'), selectClear: $('btn-select-clear'), selectCount: $('select-count'),
  picked: $('picked'), pickedBtn: $('btn-picked'), pickedMenu: $('picked-menu'),
  swap: $('btn-swap'), swapDialog: $('swap-dialog'), swapScope: $('swap-scope'), swapRows: $('swap-rows'),
  swapAdd: $('swap-add'), swapNegative: $('swap-negative-label'), swapCount: $('swap-count'),
  swapResults: $('swap-results'), swapError: $('swap-error'), swapCancel: $('swap-cancel'), swapQueue: $('swap-queue'),
  refineBatch: $('btn-refine-batch'), refineBatchDialog: $('refine-batch-dialog'),
  refineBatchTitle: $('refine-batch-title'), refineBatchScope: $('refine-batch-scope'),
  refineBatchStrength: $('refine-batch-strength'),
  refineBatchScales: $('refine-batch-scales'), refineBatchSteps: $('refine-batch-steps'),
  refineBatchSkip: $('refine-batch-skip'), refineBatchForeign: $('refine-batch-foreign'),
  refineBatchForeignLabel: $('refine-batch-foreign-label'), refineBatchCount: $('refine-batch-count'),
  refineBatchResults: $('refine-batch-results'), refineBatchError: $('refine-batch-error'),
  refineBatchCancel: $('refine-batch-cancel'), refineBatchRun: $('refine-batch-run'),
  moreSeeds: $('btn-more-seeds'), moreSeedsDialog: $('more-seeds-dialog'),
  moreSeedsTitle: $('more-seeds-title'), moreSeedsScope: $('more-seeds-scope'),
  moreSeedsCount: $('more-seeds-count'), moreSeedsLine: $('more-seeds-line'),
  moreSeedsResults: $('more-seeds-results'), moreSeedsError: $('more-seeds-error'),
  moreSeedsCancel: $('more-seeds-cancel'), moreSeedsRun: $('more-seeds-run'),
  bin: $('btn-bin'), emptyBin: $('btn-empty-bin'), restorePicked: $('btn-restore-picked'),
  galleryTitle: $('gallery-title'), lbRestore: $('lb-restore'),
  bundles: $('btn-bundles'), bundlesDialog: $('bundles-dialog'), bundlesClose: $('bundles-close'),
  bundleRefresh: $('bundle-refresh'), bundlePreview: $('bundle-preview'), bundleCount: $('bundle-count'),
  bundleCountLabel: $('bundle-count-label'), bundleExport: $('bundle-export'), bundleFile: $('bundle-file'),
  bundleUpload: $('bundle-upload'), bundleUploadCancel: $('bundle-upload-cancel'), bundleStatus: $('bundle-status'),
  bundleProgress: $('bundle-progress'), bundleDownload: $('bundle-download'),
  bundleRetained: $('bundle-retained'), bundleDirectory: $('bundle-directory'), bundleList: $('bundle-list'),
  bundlePassword: $('bundle-password'), bundlePasswordAgain: $('bundle-password-again'), bundleUnlock: $('bundle-unlock'),
  bundleBackup: $('bundle-backup'), bundleScopePicked: $('bundle-scope-picked'),
  bundleScopeStarred: $('bundle-scope-starred'), bundleScopeHint: $('bundle-scope-hint'),
  sheet: $('sheet'), sheetControls: $('sheet-controls'),
  sheetFixed: $('sheet-fixed'), sheetGrid: $('sheet-grid'), sheetNote: $('sheet-note'), sheetSaid: $('sheet-said'),
  runHint: $('run-hint'),
  lastRun: $('last-run'), lastRunText: $('last-run-text'),
  lastRunShow: $('last-run-show'), lastRunResume: $('last-run-resume'), lastRunDismiss: $('last-run-dismiss'),
  runList: $('run-list'),
  dot: $('status-dot'), statusText: $('status-text'), statusAside: $('status-aside'), vram: $('vram'),
  jobMirror: $('job-mirror'), topbarLine: $('topbar-line'), topbarFill: $('topbar-fill'),
  lightbox: $('lightbox'), lbImg: $('lb-img'), lbMeta: $('lb-meta'), lbClose: $('lb-close'),
  lbPrompt: $('lb-prompt'), lbSettings: $('lb-settings'), lbCount: $('lb-count'),
  lbError: $('lb-error'), lbStage: $('lb-stage'), lbZoom: $('lb-zoom'), lbZoomSaid: $('lb-zoom-said'),
  lbFit: $('lb-fit'), lbOne: $('lb-one'),
  lbReuse: $('lb-reuse'), lbRefine: $('lb-refine'), lbPrev: $('lb-prev'), lbNext: $('lb-next'),
  lbFav: $('lb-fav'), lbDelete: $('lb-delete'), lbStarSaid: $('lb-star-said'),
  lbMain: $('lb-main'), lbDesc: $('lb-desc'), lbSrcImg: $('lb-src-img'), lbSplit: $('lb-split'),
  lbTagSrc: $('lb-tag-src'), lbTagPass: $('lb-tag-pass'), lbCompare: $('lb-compare'),
  lbCompareOn: $('lb-compare-on'), lbSwipe: $('lb-swipe'), lbSideBySide: $('lb-side-by-side'),
  lbCompareNote: $('lb-compare-note'), lbKeys: $('lb-keys'),
  lbCompareCfg: $('lb-compare-cfg'), lbCompareAgainst: $('lb-compare-against'),
  lbCfgSteps: $('lb-cfg-steps'), lbCfgLower: $('lb-cfg-lower'), lbCfgHigher: $('lb-cfg-higher'),
  lbAgainst: $('lb-against'),
  refineDialog: $('refine-dialog'), refineTitle: $('refine-title'), refineClose: $('refine-close'),
  refineThumb: $('refine-thumb'), refineSourceLine: $('refine-source-line'),
  refineFigure: $('refine-figure'), refineKept: $('refine-kept'), refineReused: $('refine-reused'),
  refineDraft: $('refine-draft'), refineDraftText: $('refine-draft-text'),
  refineStartOver: $('refine-start-over'), refinePasses: $('refine-passes'),
  refinePassesHead: $('refine-passes-head'), refinePassList: $('refine-pass-list'),
  refinePromptLine: $('refine-prompt-line'), refineUnavailable: $('refine-unavailable'),
  refineScales: $('refine-scales'), refineWidth: $('refine-width'), refineHeight: $('refine-height'),
  refineLock: $('refine-lock'),
  refineSizeHint: $('refine-size-hint'), refineStrength: $('refine-strength'),
  refineStrengthTip: $('refine-strength-tip'), refineCompare: $('refine-compare'),
  refineSweepHint: $('refine-sweep-hint'), refineLarge: $('refine-large'),
  refineVersions: $('refine-versions'),
  refineSeed: $('refine-seed'), refineDice: $('refine-dice'), refineSeedHint: $('refine-seed-hint'),
  refineMore: $('refine-more'), refineMoreNote: $('refine-more-note'),
  refinePrompt: $('refine-prompt'), refineTags: $('refine-tags'), refineNegative: $('refine-negative'),
  refineFromPanel: $('refine-from-panel'), refineFromNote: $('refine-from-note'),
  refineTagSuggestions: $('refine-tag-suggestions'),
  refineNegField: $('refine-negative-field'), refineNegNote: $('refine-neg-note'),
  refineSteps: $('refine-steps'), refineCfg: $('refine-cfg'),
  refineMu: $('refine-mu'), refineMuField: $('refine-mu-field'),
  refineTrueCfg: $('refine-true-cfg'), refineTrueCfgField: $('refine-true-cfg-field'),
  refineCarried: $('refine-carried'), refineCheck: $('refine-check'),
  refineCheckSaid: $('refine-check-said'), refinePreview: $('refine-preview'),
  refineError: $('refine-error'), refineCancel: $('refine-cancel'), refineRun: $('refine-run'),
  lbInpaint: $('lb-inpaint'),
  inpaintDialog: $('inpaint-dialog'), inpaintTitle: $('inpaint-title'), inpaintClose: $('inpaint-close'),
  inpaintStage: $('inpaint-stage'), inpaintImage: $('inpaint-image'), inpaintCanvas: $('inpaint-canvas'),
  inpaintBrush: $('inpaint-brush'), inpaintTools: $('inpaint-tools'),
  inpaintBrushTool: $('inpaint-brush-tool'), inpaintEraserTool: $('inpaint-eraser-tool'),
  inpaintInvert: $('inpaint-invert'), inpaintClear: $('inpaint-clear'),
  inpaintUndo: $('inpaint-undo'), inpaintRedo: $('inpaint-redo'), inpaintShow: $('inpaint-show'),
  inpaintSize: $('inpaint-size'), inpaintSizeOut: $('inpaint-size-out'),
  inpaintFeather: $('inpaint-feather'), inpaintFeatherOut: $('inpaint-feather-out'),
  inpaintStatus: $('inpaint-status'), inpaintSourceLine: $('inpaint-source-line'),
  inpaintReused: $('inpaint-reused'), inpaintDraft: $('inpaint-draft'), inpaintDraftText: $('inpaint-draft-text'),
  inpaintStartOver: $('inpaint-start-over'), inpaintUnavailable: $('inpaint-unavailable'),
  inpaintStrength: $('inpaint-strength'), inpaintStrengthNote: $('inpaint-strength-note'),
  inpaintVersions: $('inpaint-versions'), inpaintSeed: $('inpaint-seed'), inpaintDice: $('inpaint-dice'),
  inpaintSeedHint: $('inpaint-seed-hint'), inpaintMore: $('inpaint-more'), inpaintMoreNote: $('inpaint-more-note'),
  inpaintPrompt: $('inpaint-prompt'), inpaintNegative: $('inpaint-negative'),
  inpaintNegField: $('inpaint-negative-field'), inpaintNegNote: $('inpaint-neg-note'),
  inpaintSteps: $('inpaint-steps'), inpaintCfg: $('inpaint-cfg'),
  inpaintMu: $('inpaint-mu'), inpaintMuField: $('inpaint-mu-field'),
  inpaintTrueCfg: $('inpaint-true-cfg'), inpaintTrueCfgField: $('inpaint-true-cfg-field'),
  inpaintCarried: $('inpaint-carried'), inpaintCheck: $('inpaint-check'), inpaintCheckSaid: $('inpaint-check-said'),
  inpaintPreview: $('inpaint-preview'), inpaintError: $('inpaint-error'),
  inpaintCancel: $('inpaint-cancel'), inpaintRun: $('inpaint-run'),
  lbOutpaint: $('lb-outpaint'),
  outpaintDialog: $('outpaint-dialog'), outpaintTitle: $('outpaint-title'), outpaintClose: $('outpaint-close'),
  outpaintTop: $('outpaint-top'), outpaintLeft: $('outpaint-left'),
  outpaintRight: $('outpaint-right'), outpaintBottom: $('outpaint-bottom'),
  outpaintFrame: $('outpaint-frame'), outpaintCanvas: $('outpaint-canvas'), outpaintImage: $('outpaint-image'),
  outpaintBandEdge: $('outpaint-band-edge'), outpaintFrameLine: $('outpaint-frame-line'),
  outpaintStatus: $('outpaint-status'), outpaintGrow: $('outpaint-grow'), outpaintWide: $('outpaint-wide'),
  outpaintTall: $('outpaint-tall'), outpaintShape: $('outpaint-shape'),
  outpaintShapeApply: $('outpaint-shape-apply'),
  outpaintBand: $('outpaint-band'), outpaintBandNote: $('outpaint-band-note'),
  outpaintSourceLine: $('outpaint-source-line'), outpaintReused: $('outpaint-reused'),
  outpaintDraft: $('outpaint-draft'), outpaintDraftText: $('outpaint-draft-text'),
  outpaintStartOver: $('outpaint-start-over'), outpaintUnavailable: $('outpaint-unavailable'),
  outpaintStrength: $('outpaint-strength'), outpaintStrengthNote: $('outpaint-strength-note'),
  outpaintVersions: $('outpaint-versions'), outpaintSeed: $('outpaint-seed'), outpaintDice: $('outpaint-dice'),
  outpaintSeedHint: $('outpaint-seed-hint'), outpaintMore: $('outpaint-more'), outpaintMoreNote: $('outpaint-more-note'),
  outpaintPrompt: $('outpaint-prompt'), outpaintNegative: $('outpaint-negative'),
  outpaintNegField: $('outpaint-negative-field'), outpaintNegNote: $('outpaint-neg-note'),
  outpaintSteps: $('outpaint-steps'), outpaintCfg: $('outpaint-cfg'),
  outpaintMu: $('outpaint-mu'), outpaintMuField: $('outpaint-mu-field'),
  outpaintTrueCfg: $('outpaint-true-cfg'), outpaintTrueCfgField: $('outpaint-true-cfg-field'),
  outpaintCarried: $('outpaint-carried'), outpaintCheck: $('outpaint-check'), outpaintCheckSaid: $('outpaint-check-said'),
  outpaintPreview: $('outpaint-preview'), outpaintError: $('outpaint-error'),
  outpaintCancel: $('outpaint-cancel'), outpaintRun: $('outpaint-run'),
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
// made by `armMore` (from app.js, and again where the gallery's scroll moves
// to a column of its own) and re-armed by `watchMore` after every page,
// fetches the next page as the button scrolls near the viewport. The button
// stays as the sign that older rows exist and as the retry after a page that
// failed.
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
// And the start-up catalogue's name for it (`ModelStatus.model_label`), which
// the page shows in its place (checkpointName): the repository's name hid a
// pinned version. Null off the catalogue.
let loadedModel = null;
let checkpointLabel = null;
let statusPending = false;
// The image model's state as the last status gave it (null before the
// first), whether the last poll got no answer at all, and when the server
// that did answer started - a change is a restart, which ended every job.
let modelState = null;
let serverOffline = false;
let serverStartedAt = null;
// The job whose stream the page is watching, and the ids queued behind it.
// The server was written for a queue - jobs wait on its gate, a queued one
// can be cancelled on the spot - and the browser used to refuse a second
// submission outright, so the queue was unreachable from the UI.
let currentJob = null;
let queuedIds = [];
// A job another tab or window started is on the server while this page
// watches none (`ModelStatus.active_jobs`, job-queue.js's followServerJobs):
// Unload, the critic and Expand are held with that reason, and the gallery
// takes in its images once it has gone. Generate stays live - the server
// queues it.
let elsewhereBusy = false;
// The gallery's newest id as the last status said it (`ModelStatus.
// newest_image_id`, binned rows included), null before the first gallery
// page, which sets it (`GalleryPage.newest_image_id`, read with it): a job
// queued, run and ended between two polls never showed in `active_jobs`, so
// only this moving says another tab's (or the CLI's) images are in
// (job-queue.js's followGalleryHead). And the ids this page's watched jobs
// streamed to it (job-stream.js), which explain a move without a catch-up;
// pruned below what the status has said.
let newestImageSeen = null;
const ownImageIds = new Set();
// The listing a page between jobs waits on to pick its next one (watchNext,
// resumeActiveJob, through adoptListedJob), while it is out, and every job
// a stream has been opened for. Which job the page watches next is that
// listing's to say: a Generate pressed meanwhile took the page as well, and
// when the listing found the new job the page streamed it twice - two tiles
// of one image, a lost Retry, a second watch of the next job too.
let jobListing = null;
const watchedJobs = new Set();
// The stream of each job being watched, by id, until it ends: a submission
// the listing adopted before its POST answered waits on it, so startJob
// resolves at the job's end whichever of the two opened the stream.
const jobStreams = new Map();
let submitting = false;
// What this page knows of a job it submitted, by id: the submission itself
// ({path, payload, seed, sheet, at}) - a grid's Retry sends its own payload
// again, never the panel as it is now - the queue note's name for it, and
// what the viewer shows of a pass until its first frame (refine.js).
const submissions = new Map();
const jobLabels = new Map();
const jobPlaceholders = new Map();
// Job ids this tab has a word about, kept in sessionStorage: a reload keeps
// a tab's sessionStorage and another tab has its own, so what this tab did
// before a reload is still this tab's when the reloaded page sees the job.
// Read once, at load (an iframe of the page shares it); every change is
// written through, the newest 50 kept. Never the page's state: a store that
// throws or comes back empty leaves a set of this page's own memory.
class TabJobIds extends Set {
  constructor(key) {
    super();
    this.key = key;
    try {
      const ids = JSON.parse(sessionStorage.getItem(key) ?? '[]');
      if (Array.isArray(ids)) for (const id of ids) if (typeof id === 'string') super.add(id);
    } catch { /* unavailable */ }
  }
  add(id) { super.add(id); this.save(); return this; }
  delete(id) { const had = super.delete(id); if (had) this.save(); return had; }
  save() {
    try { sessionStorage.setItem(this.key, JSON.stringify([...this].slice(-50))); } catch { /* unavailable */ }
  }
}
// Every job this page's own submissions made, for good (a submission is
// forgotten at its end): a job the page watches that it did not submit -
// another tab's, picked up when its own ended - is said to be another tab's,
// and so is a queue row (queueRow). Kept across a reload, whose listing
// would otherwise call this tab's own queued jobs another tab's.
const ownJobs = new TabJobIds('local-diffusion.own-jobs');
// Cancels this page asked for - Cancel, a queued row's ×, Cancel queued - by
// job id: finishJob says any other as one from elsewhere. Kept across a
// reload: a Cancel pressed just before one is still this tab's when the
// reloaded page sees the job end.
const cancelledHere = new TabJobIds('local-diffusion.cancelled-jobs');
// The seed the page itself put in the Seed box - a grid's base, written back
// for Resume, or a finished image's through Use seed - while it is still
// there: the run hint says the next Generate repeats it.
let pageSeed = null;
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
// ModelStatus.prompt_expander: whether the rewriter is loaded and what a
// first use downloads - what Expand's title and its running line say.
let expanderStatus = null;
// What holds the card now (`ModelStatus.card_holder`: "job", "critic",
// "expand", or null) and the ticket an editor action holding it was sent
// with (`card_ticket`): an Expand of this page's whose ticket it is not
// waits for the card, and its running line says so (expandRunningText).
let cardHolder = null;
let cardTicket = null;
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
// The gallery's newest id as this epoch's grid last read it: its first page
// (loadGallery) or a head page joined to it (prependFinished), each
// `GalleryPage.newest_image_id`; null after a reset until that page lands.
// A catch-up for images up to it has nothing to fetch (prependFinished's
// `upTo`): the grid's own read had them.
let galleryReadHead = null;
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
// The Reference images tooltip for a checkpoint that takes none, and the
// prompt's placeholder for one that reads prose: what the page shipped.
const referenceTipDefault = els.referenceTip.textContent;
// The grid's Mu and LoRA weight axes' syntax help, as nodes (its examples are
// bold): put back where the axis applies again, after a reason stood in it.
const gridMuTipDefault = [...els.gridMuTip.childNodes];
const gridLoraTipDefault = [...els.gridLoraTip.childNodes];
const promptPlaceholderDefault = els.prompt.placeholder;
// The checkpoint's prompting guide: the pixel range its authors document
// (megapixels, [low, high], null until a status lands) for the size line's
// warning, and the example prompts behind the Examples button.
let sizeRange = null;
let sizeRangeWord = 'documented';
// Whose an "example" span's sizes are, as the size prose names them ("the
// author's reference sizes"); null for a range in writing.
let sizeRangeSource = null;
// Null means the checkpoint accepts any framing in its pixel range. Animagine
// publishes a finite table, so its guide names the aspect buttons too.
let sizeAspects = null;
let guideResolutions = null;
let examples = [];
let exampleNotes = {};
// The authors' template slots in those examples (`guide.slots`: Z-Anime's
// "[character]" and "[location]"), as RegExps, case aside.
let exampleSlots = [];
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
// 'pending' (a lookup on its way), 'checked' or 'offline' (the index unreachable).
let lastTagValidationState = 'checked';
let lastTagValidationDetail = '';
let draggedTagIndex = null;
// The inline chip editor: at most one open, inserting before `tagInsertAt` or
// replacing the item at `tagEditAt`.
let tagInsertAt = null;
let tagEditAt = null;
// The pending lookup of the editor's suggestion list - the chip editor's, or
// the Refine dialog's prompt's, which share it (tags.js,
// scheduleEditorSuggestions): never both at once, the dialog being modal.
let tagEditorSuggestionTimer = null;
let tagEditorSuggestionEpoch = 0;
// A remembered adapter name waiting for the list it belongs to.
let pendingLora = '';
// Why this server applies no adapter at all - /api/loras' detail, which the
// LoRA badge shows - or null: the grid's LoRA weight axis quotes it.
let loraRefusal = null;

// Every size the page shows is binary and says so: the card's memory, the
// free disk, a download, a grid's PNGs. The arithmetic was always 1024's and
// the labels decimal's, so a first use's 4,856,283,677 bytes read "about
// 4.5 GB" where the Hub lists 4.86 GB. Labelled as computed (GiB, MiB), as
// nvidia-smi reads the card, rather than recomputed to a decimal gigabyte,
// in which a 16 GiB card is 17.1 GB.
const humanBytes = (n) => {
  let size = n;
  for (const unit of ['B', 'KiB', 'MiB', 'GiB']) {
    if (size < 1024 || unit === 'GiB') return `${size.toFixed(1)} ${unit}`;
    size /= 1024;
  }
  return `${size.toFixed(1)} GiB`;
};

// The strips the lightbox can walk: what this job produced, and what the
// gallery has paged in so far, both in the order they are displayed - and
// what this page's earlier jobs produced, newest first (the viewer's
// `#recent`), which outlives a gallery reset or a search.
let viewerRecords = [];
let galleryRecords = [];
let sessionRecords = [];
let lbList = [];
let lbIndex = -1;

// Contact sheet. `sheetAxes` is what the user picked; null means "work it out
// from the images", which is what happens on the first render of a new set.
let sheetMode = false;
let sheetAxes = { rows: null, cols: null, pins: {} };
let sheetRecords = [];

// ------------------------------------------------------------------ helpers

// No answer at all - the server stopped, or never started - rather than an
// answer that refuses: fetch's own words for it are "Failed to fetch", which
// named neither the app nor what to do. `offline` marks it, so the status poll
// can take the page into its offline state and clear the message once the
// server answers again.
const OFFLINE_MESSAGE = 'No answer from the Local Diffusion server — is it still running?';
// The topbar's words for that state, which the watched job's line, the
// topbar's mirror of it and the tab's title use too while its stream
// reconnects (job-stream.js): one phrase, not three.
const LOST_PHRASE = 'server not answering — retrying';

// Every /api/ request carries the checkpoint revision the page last saw; the
// server alone decides which routes refuse a stale one, so a model-dependent
// route added there needs nothing here.
async function api(path, options) {
  const revision = modelRevision;
  if (path.startsWith('/api/') && revision !== null) {
    const headers = new Headers(options?.headers);
    if (!headers.has('X-Local-Diffusion-Model-Revision')) {
      headers.set('X-Local-Diffusion-Model-Revision', String(revision));
    }
    options = { ...options, headers };
  }
  let res;
  try {
    res = await fetch(path, options);
  } catch (err) {
    // fetch's own network failure is a TypeError; anything else - an abort -
    // is not the server's silence.
    if (!(err instanceof TypeError)) throw err;
    throw Object.assign(new Error(OFFLINE_MESSAGE), { offline: true });
  }
  if (!res.ok) {
    let detail = res.statusText;
    let body = null;
    try { body = await res.json(); detail = body?.detail ?? detail; } catch { /* non-JSON body */ }
    // The status too, for a caller that tells "gone" from "refused" (a job
    // the server no longer knows is a 404), and the body for one that reads a
    // field beside `detail` (the bundle preview's missing `image_id`).
    throw Object.assign(new Error(readableDetail(detail)), { status: res.status, body });
  }
  const body = res.status === 204 ? null : await res.json();
  const advisory = /^\/api\/(?:prompt\/|tags\/|loras$|grid\/preview$|swap-tags\/preview$|refine\/batch\/preview$|more-seeds\/preview$|png\/settings$)/.test(path.split('?')[0]);
  if (advisory && revision !== null && revision !== modelRevision) {
    throw Object.assign(new Error('The checkpoint changed while this request was running. Try again with the current checkpoint.'), { status: 409 });
  }
  return body;
}

// FastAPI's own 422s carry a list of {loc, msg, ctx, input} objects, which as
// JSON.stringify is a wall of braces in the error slot. Ours are plain strings.
// A model validator's message arrives behind pydantic's "Value error, ", which
// is the library talking, not the sentence it wraps.
function readableDetail(detail) {
  if (typeof detail === 'string') return detail;
  if (!Array.isArray(detail)) return JSON.stringify(detail);
  return detail
    .map((e) => {
      const field = (e.loc ?? []).filter((part) => part !== 'body').join('.');
      const msg = String(e.msg ?? '').replace(/^Value error, /, '');
      return field ? `${field}: ${msg}` : msg;
    })
    .join('; ');
}

// `note` styles the same slot as muted rather than red, for outcomes that are
// not failures - a cancelled run in particular. `slot` is where it is said:
// the panel's line under Generate, or the gallery's under its head
// (`els.galleryError`) for what a gallery control did - a click among the
// cards at QHD used to be answered 2,400 px away, in the other column. One
// slot per message: both are role=alert, and a message in two is heard twice.
// `offline` is a message about the server not answering - any that carries
// OFFLINE_MESSAGE, and those that say so in their own words - which the
// first status poll the server answers takes back (clearOffline): it used to
// stay up, red, long after the server was back.
const offlineSlots = new Map();
function showError(message, note = false, {
  slot = els.error, mirror = true, offline = !!message?.includes(OFFLINE_MESSAGE),
} = {}) {
  // Unhide *before* writing: the slot is role=alert, and text written into a
  // display:none node is never announced.
  slot.hidden = !message;
  slot.textContent = message || '';
  slot.classList.toggle('note', note);
  // A one-line slot (the gallery's) cuts a long message with an ellipsis
  // rather than growing; its whole text is the title.
  if (slot.classList.contains('one-line')) slot.title = message || '';
  // A new message is nobody's "Previous job:" (job-results.js, syncFinishLead).
  slot.classList.remove('previous');
  if (offline && message) offlineSlots.set(slot, message);
  else offlineSlots.delete(slot);
  if (!mirror) return;
  // The lightbox covers both slots; mirror into its own while it is up.
  const shown = els.lightbox.open;
  els.lbError.hidden = !(shown && message);
  els.lbError.textContent = shown ? message || '' : '';
  els.lbError.classList.toggle('note', note);
}

// The server answers again: every message about it not answering goes, where
// it is still the one shown - the lightbox's copy of one included.
function clearOffline() {
  for (const [slot, message] of offlineSlots) {
    if (slot.textContent === message) showError('', false, { slot, mirror: false });
    if (els.lbError.textContent === message) { els.lbError.hidden = true; els.lbError.textContent = ''; }
  }
  offlineSlots.clear();
}
