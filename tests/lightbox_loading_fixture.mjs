import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

class Picture {
  constructor() { this.attrs = new Map(); this.style = {}; this.hidden = false; this.complete = false; this.naturalWidth = 0; }
  getAttribute(name) { return this.attrs.get(name) ?? null; }
  setAttribute(name, value) { this.attrs.set(name, value); }
  removeAttribute(name) { this.attrs.delete(name); }
  set src(value) { this.attrs.set('src', value); this.complete = false; this.naturalWidth = 0; }
  get src() { return this.attrs.get('src') ?? ''; }
  before(child) { this.beforeImage = child; }
  close() { this.open = false; }
}

const timers = new Map(), retired = [], made = [];
let nextTimer = 1, asked = 0;
const els = new Proxy({ lightbox: Object.assign(new Picture(), { open: true }) }, {
  get(target, key) { return target[key] ??= new Picture(); },
});
const context = vm.createContext({
  els, lbList: [], lbIndex: -1, lbPickedPair: [], lbCfgShown: {}, galleryRecords: [],
  serverOffline: false, OFFLINE_MESSAGE: 'offline',
  document: { createElement() { return new Picture(); } },
  Image: class extends Picture { constructor() { super(); made.push(this); } },
  setTimeout(callback) { const id = nextTimer++; timers.set(id, callback); retired.push(callback); return id; },
  clearTimeout(id) { timers.delete(id); },
  api: async () => { asked++; return {}; },
  showError(text) { els.lbError.textContent = text; els.lbError.hidden = !text; },
  syncLightboxCopyScope() {}, forgetLadders() {},
});
vm.runInContext(fs.readFileSync(process.argv[2] + '/web/lightbox.js', 'utf8'), context);
const evaluate = source => vm.runInContext(source, context);
const records = Array.from({ length: 5 }, (_, id) => ({ id, url: `/full-${id}.png`, thumb_url: `/thumb-${id}.webp` }));
context.lbList = records;
function show(index) { context.lbIndex = index; context.showLightboxImage(records[index]); }
function loaded() { els.lbImg.complete = true; els.lbImg.naturalWidth = 256; context.lbImageLoaded(); }
function fire() { for (const [id, callback] of [...timers]) { timers.delete(id); callback(); } }
const urls = () => JSON.parse(evaluate('JSON.stringify([...lbNeighbourImages.keys()])')).sort();

show(2);
assert.equal(els.lbImg.style.opacity, '0');
assert.equal(els.lbImg.beforeImage.src, '/thumb-2.webp');
assert.equal(els.lbImg.beforeImage.hidden, false);
assert.equal(els.lbImg.beforeImage.getAttribute('aria-hidden'), 'true');
context.translateLightboxImage('12px 0');
assert.equal(els.lbImg.beforeImage.style.translate, '12px 0');
assert.equal(made.length, 0, 'neighbours wait for the current PNG');
loaded();
assert.equal(els.lbImg.style.opacity, '');
assert.equal(els.lbImg.beforeImage.hidden, true);
assert.equal(made.length, 0, 'neighbours wait for the settle timer');
fire();
assert.deepEqual(urls(), ['/full-1.png', '/full-3.png']);
const oldPreloads = [...made];

show(3);
assert.equal(els.lbImg.style.opacity, '0', 'the previous PNG is hidden immediately');
assert.equal(els.lbImg.beforeImage.src, '/thumb-3.webp', 'never the previous thumbnail');
context.lbImageLoaded(); // a previous image's queued native event
await context.lbImageFailed();
assert.equal(els.lbImg.style.opacity, '0', 'stale load cannot expose a pending PNG');
assert.equal(asked, 0, 'stale error cannot diagnose a pending PNG');
loaded(); fire();
assert.deepEqual(urls(), ['/full-2.png', '/full-4.png']);
assert.ok(oldPreloads.every(image => image.getAttribute('src') === null), 'retired preload owners release their sources');
show(4); loaded(); fire();
assert.deepEqual(urls(), ['/full-3.png'], 'the edge preloads only its one neighbour');

// An error's record probe can resolve after a walk away and back to the
// same record object. Its original request still owns no new feedback.
show(2);
els.lbImg.complete = true;
let release;
context.api = () => { asked++; return new Promise(resolve => { release = resolve; }); };
const staleError = context.lbImageFailed();
show(3); show(2);
release({}); await staleError;
assert.equal(evaluate('lbLoadError'), '');
assert.equal(els.lbImg.beforeImage.src, '/thumb-2.webp');
els.lbImg.complete = true;
context.api = async () => ({});
await context.lbImageFailed();
assert.equal(els.lbError.textContent, 'The file of #2 could not be loaded.');
assert.equal(els.lbImg.style.opacity, '0');
assert.equal(els.lbImg.beforeImage.hidden, false, 'a failed full image keeps its own thumbnail');
loaded();
assert.equal(els.lbError.textContent, '');

show(1);
els.lbImg.beforeImage.complete = true;
els.lbImg.beforeImage.onerror();
assert.equal(els.lbImg.beforeImage.hidden, true, 'a broken thumbnail is hidden');
assert.equal(els.lbImg.style.opacity, '0');
loaded(); fire();
assert.deepEqual(urls(), ['/full-0.png', '/full-2.png']);
const staleTimer = retired.at(-1);
context.closeLightbox();
staleTimer();
assert.deepEqual(urls(), []);
assert.equal(els.lbImg.beforeImage.hidden, true);
assert.equal(els.lbImg.beforeImage.getAttribute('src'), null);
console.log('lightbox image ownership, thumbnails and bounded neighbours passed');
