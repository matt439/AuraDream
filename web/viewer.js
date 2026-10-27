// AuraDream's page script: the results viewer and recent images.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// ------------------------------------------------------------- the viewer

// The most full-resolution images the strip keeps in the DOM. A 1024-cell
// night at 4 MP is ~5 GB of PNG; the strip is for watching images land, and
// the lightbox still walks every record of the run through `viewerRecords`.
const VIEWER_MAX = 48;
// This page's earlier results beside a single image (#recent): how many it
// shows, and how many it keeps for the lightbox to walk on from them.
const RECENT_SHOWN = 6;
const SESSION_MAX = VIEWER_MAX;

// The job the strip is laid out for: how many images it makes and at what
// size, from its status frame as its first frame or image lands - so the
// tier is the whole job's from the start, and the live preview takes the box
// the result will.
const viewerJob = { expected: 0, width: 0, height: 0 };

// A job's first frame or image clears the strip, which until then kept the
// last job's up through the queue wait and the load. What it held becomes
// the newest of this page's earlier results (#recent) - unless it is this
// job's own, which a stream that reattached shows again. `job` is the job's
// last status frame.
function clearResults(job = null) {
  const earlier = viewerRecords.filter((r) => !job || r.job_id !== job.job_id).reverse();
  const kept = new Set(earlier.map((r) => r.id));
  sessionRecords = [...earlier, ...sessionRecords.filter((r) => !kept.has(r.id))].slice(0, SESSION_MAX);
  viewerRecords = [];
  Object.assign(viewerJob, { expected: job?.planned_images ?? 0, width: job?.width ?? 0, height: job?.height ?? 0 });
  // Focus on what goes - a result's tool the Refine dialog handed it back
  // to, its picture - goes to Generate rather than to <body>, and a sheet the
  // job opens takes it on from there (handJobFocus).
  const focused = els.viewer.contains(document.activeElement);
  if (focused) handJobFocus(els.generate);
  els.viewer.innerHTML = '';
  els.viewer.scrollTop = 0;
  els.viewer.style.maxHeight = '';
  els.viewer.style.minHeight = '';
  viewerTier();
  els.viewerNoteText.replaceChildren();
  syncViewerMoreSeeds();
  // The gallery's cards of the run before are not what the strip shows now.
  els.gallery.querySelectorAll('.card.this-run').forEach((div) => div.classList.remove('this-run'));
}

// Three layouts by how many images the job makes, so a long sweep does not
// push the gallery a screen further down with every image it finishes: one
// at 68vh, with its column beside it from 1800 px; two to four as large as
// that height allows (`fitViewer`); more, a strip of small tiles. From the
// job's count, not the images so far: by the running count the strip was
// laid out three times in a sweep's first five images, and the gallery
// under it jumped each time. #recent goes with the single tier, and before
// the fit, which measures the width it leaves.
function viewerTier() {
  const n = Math.max(viewerJob.expected, viewerRecords.length);
  els.viewer.classList.toggle('multi', n > 1);
  const many = els.viewer.classList.toggle('many', n > 4);
  // A long run's tiles are one tab stop each, their pictures, as before the
  // tools came: four more a tile put a 48-image strip 240 presses deep. The
  // lightbox a picture opens has every action; the pointer has the row.
  els.viewer.querySelectorAll('.tools button').forEach((b) => { b.tabIndex = many ? -1 : 0; });
  renderRecent();
  fitViewer();
}

// Two to four images, best fit: the column count whose tile is largest
// within the strip's width and the single image's height, the rows' gaps and
// tools rows counted - two squares side by side at 68vh, four 16:9 images two
// by two. A fixed 34vh used 42% of a QHD strip for a batch of two. The tile
// is the job's shape; a strip whose images are not all of it wraps instead.
// Every row the job will fill is there from its first frame (`--rows`): a
// 2x2 held one row while the preview denoised, and grew by the second as
// the images landed, the gallery under it pushed 466 px down at QHD at the
// end of every such Generate. So is every column (`--col-w`): in `auto`
// columns the lone preview sat centred, and the images moved left by half a
// tile as each of the others landed.
let viewerWidth = 0;
// The room a page scrollbar takes where it takes any: a classic one's
// width, 0 for overlay scrollbars. Measured once, on a probe of its own.
let scrollbarRoom = null;
function pageScrollbarWidth() {
  if (scrollbarRoom === null) {
    const probe = document.createElement('div');
    probe.style.cssText = 'position: absolute; top: -200px; width: 100px; height: 100px; overflow: scroll; visibility: hidden;';
    document.body.append(probe);
    scrollbarRoom = probe.offsetWidth - probe.clientWidth;
    probe.remove();
  }
  return scrollbarRoom;
}
function fitViewer() {
  const v = els.viewer;
  const aspect = viewerJob.width && viewerJob.height ? viewerJob.width / viewerJob.height
    : viewerRecords.length ? viewerRecords[0].width / viewerRecords[0].height : 0;
  const alike = viewerRecords.every((r) => Math.abs(r.width / r.height - aspect) / aspect < 0.01);
  const fits = v.classList.contains('multi') && !v.classList.contains('many') && aspect > 0 && alike;
  v.classList.toggle('fitted', fits);
  if (!fits) {
    v.style.removeProperty('--cols');
    v.style.removeProperty('--rows');
    v.style.removeProperty('--tile-h');
    v.style.removeProperty('--col-w');
    return;
  }
  const css = getComputedStyle(v);
  // Sized as if the page's scrollbar were there, whether or not it is: a
  // gallery too short to scroll the page takes one while a job's progress
  // block is up and gives it back as the job ends, and the strip, a
  // scrollbar wider, refitted then - 491 to 494 px a tile at QHD, the
  // gallery head 3 px lower at the end of a batch of four. A stable gutter
  // on the page would hold it too, but in Chrome it shrinks every vw size on
  // every page by its width (the panel, the type ramp, the cards).
  const missing = document.documentElement.clientWidth === innerWidth ? pageScrollbarWidth() : 0;
  const width = v.clientWidth - parseFloat(css.paddingLeft) - parseFloat(css.paddingRight) - missing;
  viewerWidth = v.clientWidth;
  const gap = parseFloat(css.columnGap) || 0;
  // A tile's tools row, under its picture (style.css).
  const foot = parseFloat(css.getPropertyValue('--h-tool')) + parseFloat(css.getPropertyValue('--sp-1'));
  const height = innerHeight * 0.68 + foot;
  const n = Math.min(4, Math.max(viewerJob.expected, viewerRecords.length));
  let best = null;
  for (let cols = 1; cols <= n; cols++) {
    const rows = Math.ceil(n / cols);
    const w = Math.min((width - (cols - 1) * gap) / cols, ((height - (rows - 1) * gap) / rows - foot) * aspect);
    if (!best || w > best.w) best = { cols, w, rows };
  }
  const tileH = Math.max(48, Math.floor(best.w / aspect));
  v.style.setProperty('--cols', best.cols);
  v.style.setProperty('--rows', best.rows);
  v.style.setProperty('--tile-h', `${tileH}px`);
  // The tile's own width (style.css: --tile-h times its shape), so every
  // column the job will fill is there from its first frame.
  v.style.setProperty('--col-w', `${tileH * aspect}px`);
}

// The strip's ResizeObserver: its width moved - the window, the panel, the
// prompt check docked beside it - and the best fit with it, on the next
// frame: refitted inside the observer's own callback, the strip's new height
// was a resize the same frame could not deliver. #recent beside a docked
// check likewise (fitRecent); its own coming leaves the image as it was, so
// the refit it causes ends there.
function viewerResized() {
  if (!els.recent.hidden) requestAnimationFrame(fitRecent);
  // A long run's reserve, at the column count the new width has.
  if (els.viewer.style.minHeight && els.viewer.clientWidth !== reservedWidth) {
    requestAnimationFrame(() => { els.viewer.style.minHeight = ''; reserveStrip(); });
  }
  if (els.viewer.clientWidth === viewerWidth || !els.viewer.classList.contains('fitted')) return;
  requestAnimationFrame(fitViewer);
}

// A tile of the strip, at its picture's shape and its tier's size: a result,
// the live preview, a pass's stand-in. All three are the one box (style.css)
// - the picture, and under it the tools row or a line of the same height -
// so whichever replaces another leaves the gallery where it was: the
// preview was a fixed 320 px, and the strip went from a 917 px result down
// to it and back up at every Generate.
function viewerTile(className, width, height) {
  const figure = document.createElement('figure');
  figure.className = className;
  figure.style.setProperty('--w', width || 1);
  figure.style.setProperty('--h', height || 1);
  const tile = document.createElement('div');
  tile.className = 'tile';
  figure.append(tile);
  return { figure, tile };
}

// The live preview's tile: its frames in place of the picture, the step
// where the tools will be. Never a record; the result replaces it.
function previewTile(width, height) {
  const { figure, tile } = viewerTile('live', width, height);
  const img = document.createElement('img');
  img.className = 'preview';
  img.alt = 'Live preview of the image being generated';
  const foot = document.createElement('p');
  foot.className = 'foot';
  tile.append(img, foot);
  return figure;
}

// The many tier scrolls inside itself from 2000 px (style.css): a strip at
// its end stays there as tiles come, one scrolled back up stays where it is.
// Asked at every tile, the live preview's and a pass's stand-in's too
// (`stripAppend`, below): the preview's tile starts a row of its own
// a few steps before its image, and a strip asked only as images landed was
// by then a row short of its end - and so not at it, for the rest of the run.
function stripAtEnd() {
  const strip = els.viewer;
  return strip.scrollTop + strip.clientHeight >= strip.scrollHeight - 2;
}

function stripAppend(el) {
  const atEnd = stripAtEnd();
  els.viewer.appendChild(el);
  settleStrip(atEnd);
  // A docked check's #recent, beside the box the image will take.
  fitRecent();
}

// After the strip's tiles changed: back to its end if it was there. And a
// strip that holds every row but for its own bottom padding shows them all
// rather than scrolling: 28 images at QHD came to 3 px past the 560 px cap,
// and the strip drew a scrollbar for those 3 px while its note said to scroll
// it. Its scrollbar's room is kept whether it scrolls or not (style.css), so
// a scrollbar coming or going does not change how many tiles a row holds.
function settleStrip(atEnd) {
  const strip = els.viewer;
  strip.style.maxHeight = '';
  reserveStrip();
  if (getComputedStyle(strip).overflowY !== 'visible') {
    const over = strip.scrollHeight - strip.clientHeight;
    if (over > 0 && over <= parseFloat(getComputedStyle(strip).paddingBottom)) {
      strip.style.maxHeight = `${strip.offsetHeight + over}px`;
    }
  }
  if (atEnd) strip.scrollTop = strip.scrollHeight;
}

// A long run's strip from 2000 px, where it has a cap of its own: every row
// the job will fill, from its first tile (`min-height`, measured on that
// tile, at the column count the strip has), up to the cap - so the gallery
// under it does not move down a row at a time as the images land. The tier
// and the tile size were the job's from its first frame; now the height is
// too. Read before settleStrip's own max-height, and let go by clearResults.
let reservedWidth = 0;
function reserveStrip() {
  const strip = els.viewer;
  if (strip.style.minHeight || !strip.classList.contains('many') || viewerJob.expected <= 1) return;
  const css = getComputedStyle(strip);
  if (css.display !== 'grid' || css.overflowY === 'visible') return;
  const first = strip.querySelector(':scope > figure');
  if (!first) return;
  const cols = css.gridTemplateColumns.split(' ').filter(Boolean).length || 1;
  const rows = Math.ceil(Math.min(viewerJob.expected, VIEWER_MAX) / cols);
  const edge = (side) => parseFloat(css[`padding${side}`]) + parseFloat(css[`border${side}Width`]);
  const tall = rows * first.getBoundingClientRect().height + (rows - 1) * (parseFloat(css.rowGap) || 0)
    + edge('Top') + edge('Bottom');
  const cap = parseFloat(css.maxHeight);
  // Past the cap by no more than the bottom padding, the strip is let grow
  // by that rather than scroll (settleStrip), so that much is reserved too.
  const height = !(cap > 0) || tall - cap <= parseFloat(css.paddingBottom) ? tall : cap;
  // Never under the least the stylesheet gives it: beside a docked check
  // that is the check's room (style.css), which the inline value overrides.
  const least = parseFloat(css.minHeight) || 0;
  strip.style.minHeight = `${Math.ceil(Math.max(height, least))}px`;
  reservedWidth = strip.clientWidth;
}

function appendResult(rec) {
  const strip = els.viewer;
  const atEnd = stripAtEnd();
  viewerRecords.push(rec);
  viewerTier();
  const figure = resultFigure(rec);
  strip.appendChild(figure);
  // The column beside the image only while it is the strip's one.
  strip.querySelectorAll('.result-side').forEach((side) => side.remove());
  if (viewerRecords.length === 1 && viewerJob.expected <= 1) figure.append(resultSide(rec, figure));
  // Lazy on its own is not enough - the strip sets no height, so unloaded
  // images stack inside the viewport and all become eligible at once. The cap
  // is what bounds the fetches; the note lives outside the strip so it survives.
  // Images only: a deleted tile's tombstone holds a slot, and shows none.
  const tiles = () => strip.querySelectorAll('img[data-id]').length;
  while (tiles() > VIEWER_MAX) strip.firstElementChild.remove();
  settleStrip(atEnd);
  // A docked check's #recent, measured beside the image now in the strip.
  fitRecent();
  noteViewer();
  // The run's cards in the gallery say so (`this-run`, set in card() too),
  // so the strip and the gallery's first rows read as the same images.
  els.gallery.querySelector(`.card[data-id="${rec.id}"]`)?.classList.add('this-run');
}

// One result: its picture - thumb's, which opens the lightbox - and under
// it the gallery card's own tools, by the card's own helpers (gallery.js):
// Reuse, Refine, Star and Delete. The image just made had none: starring or
// refining it took the lightbox, or the same image found among the cards. A
// figure, never a .card - the gallery's counts and keyboard are the cards' -
// tagged by id, so setFavorite keeps its stars in step and forgetImages
// leaves a tombstone in its place.
function resultFigure(rec) {
  const { figure, tile } = viewerTile('result', rec.width, rec.height);
  figure.dataset.id = rec.id;
  // Named "#N" as a card is (card()), so its Star, Refine and the rest are
  // heard as this image's.
  figure.setAttribute('role', 'group');
  figure.setAttribute('aria-label', `#${rec.id}`);
  figure.classList.toggle('fav', !!rec.favorite);
  const img = thumb(rec, viewerRecords, rec.url, () => openViewerImage(rec));
  img.onerror = () => { void forgetIfGone(rec.id); };
  const star = tool('☆', '', () => setFavorite(rec, !rec.favorite), 'Star');
  star.classList.add('star');
  syncStar(star, !!rec.favorite);
  const refine = tool('⤢', '', () => refineResult(rec, tile));
  refine.classList.add('refine-tool');
  syncRefineTool(refine);
  const inpaint = tool('✎', '', () => inpaintFromCard(rec, tile));
  inpaint.classList.add('inpaint-tool');
  syncInpaintTool(inpaint);
  // A card's ❐, where the checkpoint takes references (syncCardTools keeps
  // it to the status): on klein a reference had to come from a card.
  const reference = tool('❐', '', () => toggleReference(tile, rec));
  reference.classList.add('ref-tool');
  syncReferenceTool(reference, rec.id);
  reference.hidden = !referenceSupported;
  const del = tool('⌦', `Move #${rec.id} to the recycle bin`, () => deleteImage(rec), 'Move to recycle bin');
  del.classList.add('del');
  const tools = document.createElement('div');
  tools.className = 'tools';
  tools.append(reuseButton(rec, true), refine, reference, star, inpaint, del);
  // Out of the tab order in a long run's strip (viewerTier).
  if (els.viewer.classList.contains('many')) tools.querySelectorAll('button').forEach((b) => { b.tabIndex = -1; });
  tile.append(img, tools);
  return figure;
}

// The card's ⤢ on a result: on a checkpoint without a pass it says so under
// the gallery's head and over the tile (`refuseInCard`), and opens nothing.
function refineResult(rec, tile) {
  if (!refineSupported) { refuseInCard(tile, REFINE_NONE, 'No refine pass here'); return; }
  openRefine(rec);
}

// The image just made, while the strip holds it alone, with what it is and
// what to do next in the width the picture leaves - shown from 1800 px
// (style.css), where a square result left 1,270 px of the strip empty: which
// image it refines, Star, Refine…, Reuse, a pass's Compare with source and
// Use seed and More seeds… in words, the prompt, and the lightbox's own facts (factsList, facts.js,
// without its Use seed: the row has one). In the figure, so a star and a
// delete reach it as they reach the tile; Delete stays the tile's.
function resultSide(rec, figure) {
  const side = document.createElement('div');
  side.className = 'result-side';
  if (rec.refine != null || rec.edit != null) side.append(passLineage(rec));
  const action = (button) => { button.classList.add('ghost', 'small'); return button; };
  // Its word drawn after the glyph (`data-word`): syncStar writes the glyph.
  const star = action(tool('☆', '', () => setFavorite(rec, !rec.favorite)));
  star.classList.add('star');
  star.dataset.word = 'Star';
  syncStar(star, !!rec.favorite);
  const refine = action(tool('⤢ Refine…', '', () => refineResult(rec, figure.querySelector('.tile'))));
  refine.classList.add('refine-tool');
  syncRefineTool(refine);
  const inpaint = action(tool('✎ Inpaint…', '', () => inpaintFromCard(rec, figure.querySelector('.tile'))));
  inpaint.classList.add('inpaint-tool');
  syncInpaintTool(inpaint);
  // Outpaint… beside it (outpaint.js), held the same way.
  const outpaint = action(tool('⬚ Outpaint…', '', () => outpaintFromCard(rec, figure.querySelector('.tile'))));
  outpaint.classList.add('outpaint-tool');
  syncOutpaintTool(outpaint);
  const reference = action(tool('❐ Reference', '', () => toggleReference(figure.querySelector('.tile'), rec)));
  reference.classList.add('ref-tool');
  syncReferenceTool(reference, rec.id);
  reference.hidden = !referenceSupported;
  // Named by its words, as the finish line's is; the title has the seed.
  const seed = document.createElement('button');
  seed.type = 'button';
  seed.className = 'ghost small';
  seed.textContent = 'Use seed';
  seed.title = `Put ${rec.seed} in the Seed box`;
  seed.onclick = () => useSeed(rec.seed);
  // More of its batch after it, the seeds following on (more-seeds.js).
  const more = document.createElement('button');
  more.type = 'button';
  more.className = 'ghost small';
  more.textContent = 'More seeds…';
  more.setAttribute('aria-haspopup', 'dialog');
  syncMoreSeedsFor(more, rec);
  more.onclick = () => pressMoreSeedsTool(more, moreSeedsImageTarget(rec), { host: figure.querySelector('.tile') });
  // What the facts say, in a line, where the row stands in for the column
  // under a docked check (style.css); the settings line whole in its title.
  const brief = document.createElement('span');
  brief.className = 'result-brief';
  brief.textContent = `seed ${rec.seed} · ${rec.width}×${rec.height} · ${rec.steps} steps · CFG ${rec.guidance_scale}`;
  brief.title = settingsLine(rec);
  // Delete in words at the row's end, red under the pointer, as the page's
  // other destructive buttons: a lone ⌦ under the picture's corner was the
  // one tool the column did not have.
  const del = action(tool('Delete', `Move #${rec.id} to the recycle bin`, () => deleteImage(rec)));
  del.classList.add('del', 'danger');
  const actions = document.createElement('div');
  actions.className = 'result-actions';
  actions.append(star, refine, inpaint, outpaint, action(reuseButton(rec)), reference);
  if (rec.refine != null) actions.append(compareButton(rec, figure));
  actions.append(seed, more, brief, del);
  const prompt = document.createElement('p');
  prompt.className = 'result-prompt';
  // An edit's prompt is its instruction, and says so (instruction-edit.css).
  prompt.classList.toggle('instruction', rec.edit != null);
  prompt.textContent = rec.prompt;
  side.append(actions, prompt, factsList(rec));
  return side;
}

// A pass's Compare with source: the lightbox on the pass, as its picture
// opens it (openViewerImage, so a close comes back here), with the
// comparison already on - the question a pass that has just landed asks, a
// press and an S away before. Dead where the source has gone, saying why
// (`syncCompareResult`), as the lightbox's own is. The column's alone: the
// row under the picture that stands in for it with the check docked leaves
// it out (style.css), as it cut a portrait pass's row to ellipses.
function compareButton(rec, figure) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'ghost small compare-source';
  button.textContent = 'Compare with source';
  button.onclick = () => {
    if (button.getAttribute('aria-disabled') === 'true') {
      refuseInCard(figure.querySelector('.tile'), `${button.title}.`, 'Source deleted');
      return;
    }
    openViewerImage(rec);
    void toggleCompare();
  };
  syncCompareResult(button, rec);
  return button;
}

function syncCompareResult(button, rec) {
  button.setAttribute('aria-disabled', String(!!rec.refine_missing));
  button.title = rec.refine_missing ? `#${rec.refine}, the image this was refined from, ${rec.refine_binned ? 'is in the recycle bin' : 'has been deleted'}`
    : `Compare this pass with #${rec.refine}, the image it was refined from, in the lightbox`;
}

// The column's first line for a pass: which image it refines, and how far.
function passLineage(rec) {
  const line = document.createElement('p');
  line.className = 'lineage';
  // An instruction edit: which image it changed, gone or not.
  if (rec.edit != null) {
    line.append('✐ Instruction edit of ', editSourceLink(rec));
    return line;
  }
  line.append(rec.outpaint ? '⬚ Outpaint of ' : rec.mask ? '✎ Inpaint of ' : '⤢ Refine pass of ', sourceLink(rec),
    rec.outpaint ? ` by ${outpaintSidesText(rec.outpaint)}` : '', ` at strength ${rec.strength}`);
  return line;
}

// A pass's source as the page names it: a press opens it in the lightbox
// with the pass after it (facts.js's openRefineSource), unless it has gone.
function sourceLink(rec) {
  if (rec.refine_missing) return `#${rec.refine} (${goneWord(rec)})`;
  return factButton(`#${rec.refine}`, `Open #${rec.refine}, the image this was refined from; → comes back to the pass`,
    () => { void openRefineSource(rec); });
}

// A pass's source deleted while the pass is in the strip (forgetImages, once
// its records say `refine_missing`): the note and the column's lineage and
// facts say "#N (deleted)" rather than keep a button whose press said only,
// in the panel's line a screen away, that #N had gone. And back, where the
// source is restored out of the recycle bin (unforgetImages).
function viewerSourcesGone() {
  noteViewer();
  els.viewer.querySelectorAll('.result > .result-side').forEach((side) => {
    const rec = viewerRecords.find((r) => r.id === +side.parentElement.dataset.id);
    if (rec?.refine == null && rec?.edit == null) return;
    side.querySelector('.lineage')?.replaceWith(passLineage(rec));
    side.querySelector('.facts')?.replaceWith(factsList(rec));
    const compare = side.querySelector('.compare-source');
    if (compare) syncCompareResult(compare, rec);
  });
}

// Under the strip: where a pass came from - "Refined from #10 at 0.5 ·
// 1344×1728", the source a press away (an empty note had said nothing of
// what the image was) - and how much of a long run the strip does not show.
// The image just made, alone and made from the prompt, is named as a card's
// caption names it, its seed with Use seed (`seed-line`): under 1800 px,
// where nothing beside the image said its seed but its title; from there
// the column beside it says it (style.css hides the line).
function noteViewer() {
  const parts = [];
  const first = viewerRecords[0];
  if (first?.refine != null && viewerRecords.every((r) => r.refine === first.refine)) {
    const strengths = [...new Set(viewerRecords.map((r) => r.strength))].join(', ');
    const line = document.createElement('span');
    line.className = 'lineage';
    // Named as the facts row names the pass (facts.js), where every pass agrees.
    const verb = viewerRecords.every((r) => r.outpaint) ? 'Outpainted from '
      : viewerRecords.every((r) => r.mask && !r.outpaint) ? 'Inpainted from ' : 'Refined from ';
    line.append(verb, sourceLink(first), ` at ${strengths} · ${first.width}×${first.height}`);
    parts.push(line);
  } else if (first?.edit != null && viewerRecords.every((r) => r.edit?.source === first.edit.source)) {
    // Edits of one image, named as the facts row names them.
    const line = document.createElement('span');
    line.className = 'lineage';
    line.append('Edited from ', editSourceLink(first), ` · ${first.width}×${first.height}`);
    parts.push(line);
  } else if (first && viewerRecords.length === 1 && viewerJob.expected <= 1) {
    const line = document.createElement('span');
    line.className = 'seed-line';
    line.append(`#${first.id} · ${first.width}×${first.height} · seed ${first.seed} · `,
      factButton('Use seed', `Put ${first.seed} in the Seed box`, () => useSeed(first.seed)));
    parts.push(line);
  }
  const shown = els.viewer.querySelectorAll('img[data-id]').length;
  const hidden = viewerRecords.length - shown;
  const scrolls = getComputedStyle(els.viewer).overflowY !== 'visible'
    && els.viewer.scrollHeight > els.viewer.clientHeight + 1;
  if (hidden > 0) {
    parts.push(`… and ${hidden} earlier image${hidden === 1 ? '' : 's'} this run — open one to walk them all`);
  } else if (scrolls) {
    parts.push(`${viewerRecords.length} images this run — scroll the strip, or open one to walk them all`);
  }
  els.viewerNoteText.replaceChildren(...parts.flatMap((part, i) => (i ? [' · ', part] : [part])));
  syncViewerMoreSeeds();
}

// One persistent control, outside the strip that receives images. It takes
// only settings already represented by completed results, never unseen grid
// cells. Updating its target keeps the button and keyboard focus in place.
function syncViewerMoreSeeds() {
  if (!els.viewerSeeds.isConnected || !els.viewerNoteText.isConnected) {
    els.viewerNote.replaceChildren(els.viewerSeeds, els.viewerNoteText);
  }
  const records = viewerRecords.filter((rec) => !isGone(rec.id));
  els.viewerSeeds.hidden = !records.length && !viewerJob.expected;
  const one = records.length === 1;
  els.viewerMoreSeedsTarget.textContent = one
    ? `For #${records[0].id}'s settings group`
    : records.length ? 'For the settings groups already generated here' : 'Available when an image is ready';
  const model = records[0]?.model_id ?? null;
  syncMoreSeedsTool(els.viewerMoreSeeds, {
    model, ids: records.map((rec) => rec.id),
    unavailable: records.length ? '' : 'Wait for a completed image to choose its settings',
    title: one ? `More images of #${records[0].id}'s settings group, with the same settings and new seeds`
      : 'More images of the settings groups already generated in these results; unseen grid settings are not included',
  });
}

// The strip's image opens the lightbox on the strip - and, where it is the
// only one, on the gallery from it, so → walks back through what came before
// it: alone, it opened a lightbox of one, with both arrows dead. Not in the
// gallery (a search, a sheet), it walks this page's earlier results.
function openViewerImage(rec) {
  if (viewerRecords.length === 1 && viewerRecords[0] === rec) {
    const copy = !sheetMode && galleryRecords.find((r) => r.id === rec.id);
    if (copy) { openLightbox(copy, galleryRecords, 'viewer'); return; }
    if (sessionRecords.length) { openLightbox(rec, [rec, ...sessionRecords], 'viewer'); return; }
  }
  openLightbox(rec, viewerRecords, 'viewer');
}

// #recent: beside a single image from 2000 px (style.css), up to six of this
// page's earlier results, each opening the lightbox on the image in the strip
// and those, newest first. Iterating on a prompt showed one picture at a time
// in a strip 2,000 px wide, the earlier ones a screen down among the cards -
// below the fold whenever the prompt check was open. Rebuilt only when what
// it shows changes, so a focused thumbnail keeps its focus.
let recentShown = null;
function renderRecent() {
  const single = !els.viewer.classList.contains('multi');
  const shown = single ? sessionRecords.slice(0, RECENT_SHOWN) : [];
  const current = single && viewerRecords.length === 1 ? viewerRecords[0] : null;
  els.recent.hidden = !shown.length;
  const key = [current?.id, ...shown.map((r) => r.id)].join();
  if (key !== recentShown) {
    recentShown = key;
    const walk = current ? [current, ...sessionRecords] : sessionRecords.slice();
    els.recentList.replaceChildren(...shown.map((rec) => thumb(rec, walk, undefined, () => openLightbox(rec, walk, 'recent'))));
  }
  fitRecent();
}

// With the prompt check docked (style.css), #recent stays beside the image
// in the width the image leaves at its own size - so neither the image nor
// the gallery under it moves as the check opens - and goes where that is
// under --recent-min. Measured with #recent out of the docked grid, the
// viewer then the whole of its column. Again whenever the strip's width
// moves (viewerResized): the check opening or closing, the window.
function fitRecent() {
  const stage = els.recent.parentElement;
  stage.classList.remove('recent-docked');
  // Docked in the stage, which it is not with the gallery beside the viewer:
  // the check takes the gallery's column there (style.css, `:not(.beside)`).
  const docked = matchMedia('(min-width: 2000px)').matches && !els.promptAnalysis.hidden
    && els.promptAnalysis.parentElement === els.tagValidationWide && !besideOn;
  // The image, or the preview in its box - or the row under the image, where
  // that is wider: a narrow portrait's row is as wide as its buttons need
  // (style.css), and #recent beside it cut them to ellipses again.
  const tile = els.viewer.querySelector(':scope > figure > .tile');
  if (els.recent.hidden || !docked || !tile) return;
  const row = tile.parentElement.querySelector(':scope > .result-side');
  const width = Math.max(tile.getBoundingClientRect().width, row ? row.getBoundingClientRect().width : 0);
  const css = getComputedStyle(els.viewer);
  const room = els.viewer.clientWidth - parseFloat(css.paddingLeft) - parseFloat(css.paddingRight)
    - width - parseFloat(getComputedStyle(stage).columnGap);
  if (room < parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--recent-min'))) return;
  stage.style.setProperty('--recent-room', `${Math.floor(room)}px`);
  stage.classList.add('recent-docked');
}

// While a job waits on the load and the strip is empty - its one idle line,
// or the line a delete left - the line says so, with the clock the progress
// row ticks (job-stream.js's ticker): after Generate the eye is on the
// viewer, where "Nothing generated this session" stayed for the minutes a
// first load takes. The clock is aria-hidden, as the progress row's is,
// which says "loading weights" once; the line stays one line (style.css).
// `clock` null puts the line's own words back: the wait over, cancelled or
// failed, or the stream lost. The first frame or image clears the strip, the
// line with it. Not over a line holding the focus (a first run's Examples).
let viewerWait = null;
function viewerWaiting(clock) {
  const line = els.viewer.querySelector(':scope > .placeholder');
  if (viewerWait && (clock == null || viewerWait.line !== line)) {
    if (viewerWait.line.isConnected) {
      viewerWait.line.replaceChildren(...viewerWait.words);
      viewerWait.line.classList.remove('waiting');
    }
    viewerWait = null;
  }
  if (clock == null || !line) return;
  if (!viewerWait) {
    if (line.contains(document.activeElement)) return;
    viewerWait = { line, words: [...line.childNodes] };
    const time = document.createElement('span');
    time.className = 'clock';
    time.setAttribute('aria-hidden', 'true');
    line.replaceChildren('Loading weights', time, ' - the first load of a session takes a few minutes.');
    line.classList.add('waiting');
  }
  line.querySelector('.clock').textContent = clock;
}

// A gallery with nothing in it at all - no image on any checkpoint, nothing
// filtered out - is a first run: the empty viewer stays the tall box, with
// the authors' examples a press away (`viewerExamples`), rather than the one
// line every later visit gets. From each unfiltered gallery page - not the
// recycle bin's (bin.js), whose emptiness says nothing of the gallery's.
function syncEmptyInstall(page) {
  if (els.search.value.trim() || els.favOnly.checked || binMode) return;
  els.viewer.classList.toggle('empty-install', page.total === 0 && !page.other_models);
}

// The first run's Examples: the panel's own menu, open on its first item.
// After the click is through, whose document handler closes an open menu.
function viewerExamples() {
  requestAnimationFrame(() => {
    showExamples(true);
    els.examplesMenu.querySelector('[role="menuitem"]')?.focus();
  });
}

// One <img> for every place a record is shown: the viewer strip, a gallery
// card, a sheet cell, #recent. Lazy and async-decoded, since all of them can
// hold hundreds; a thumbnail derivative where one exists, the full PNG
// otherwise. Focusable and activated by Enter or Space, so the lightbox -
// where the full image and its metadata are shown - is reachable without a
// mouse; `open` is what a press does, the lightbox on `list` unless said.
function thumb(rec, list, src = rec.thumb_url ?? rec.url, open = () => openLightbox(rec, list)) {
  const img = document.createElement('img');
  img.src = src;
  img.alt = rec.prompt;
  img.loading = 'lazy';
  img.decoding = 'async';
  img.dataset.id = rec.id;
  img.tabIndex = 0;
  img.setAttribute('role', 'button');
  img.setAttribute('aria-keyshortcuts', 'Enter Space M');
  // Named as the page names images, by gallery id - the cells of a sweep
  // share a prompt, and "Open: …" named none of them apart - and titled with
  // what tells it from the one beside it under the pointer (cardTitle): its
  // size, its seed and a pass's source, then the prompt. A sheet's cell is
  // named by what the sheet lays out instead (renderSheet).
  img.setAttribute('aria-label', `Open #${rec.id}: ${rec.prompt}`);
  titleImage(img, rec);
  img.onclick = open;
  // A plain Enter or Space only: Ctrl+Enter is Generate's from anywhere
  // (app.js), and on a focused image - where closing the lightbox leaves the
  // focus - it opened the lightbox and queued a job behind it.
  img.onkeydown = (e) => {
    if (keyboardEventHeld(e) || keyboardCommandModifier(e) || keyboardEditingTarget(e.target)) return;
    if (e.key === 'Enter' && e.shiftKey) return;
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      if (!e.repeat) open(e);
    }
  };
  return img;
}

// An image's title, cardTitle's facts and then its prompt, for the pointer,
// and its description, the facts alone: the title was the description too,
// so a long prompt was read twice an image, once in the name ("Open #N:
// <prompt>") and again after "#N · W×H · seed N". `aria-description` takes
// the title's place there, less any fact the name says already: a sheet's
// cell is named by its settings ("Open #16 · Steps 2 · Seed 7", renderSheet,
// which calls this again once it has named it), and its id and seed were
// read twice. Again where a deleted source changes the facts (forgetImages).
function titleImage(img, rec) {
  const title = cardTitle(rec);
  img.title = title;
  const named = new Set((img.getAttribute('aria-label') ?? '').replace(/^Open /, '').toLowerCase().split(' · '));
  const facts = title.split('\n')[0].split(' · ').filter((fact) => !named.has(fact.toLowerCase()));
  img.setAttribute('aria-description', facts.join(' · '));
}
