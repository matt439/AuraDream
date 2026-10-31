// AuraDream's page script: a record's facts - the settings line and the
// facts list the lightbox and the viewer share, and a card's title and the
// words for a pass and its gone source. Declarations only; state.js has
// the load-order contract these files share.
'use strict';

// ------------------------------------------------------------------- facts

function upscalerName(name) {
  return name === 'realesrgan-x4plus-anime6b' ? 'R-ESRGAN 4x+ Anime6B' : name;
}

// What made a record, in one line: the lightbox's description for a screen
// reader, and what the delete confirm names an image by - the cells of a
// sweep share a prompt and a seed, and differ only here.
function settingsLine(rec) {
  return (
    // The checkpoint is part of the reproduction set as much as the seed is -
    // the same prompt and settings on Turbo and on FLUX are different images.
    `${rec.model_id.split('/').pop()}, ` +
    `${rec.width}×${rec.height}, ${rec.steps} steps, ` +
    `CFG ${rec.guidance_scale}, ` +
    // Only where it differs from the CFG: on Krea 2 it is the same number.
    // Named as the panel's boxes are.
    (rec.true_cfg != null && rec.true_cfg !== rec.guidance_scale ? `True CFG ${rec.true_cfg}, ` : '') +
    `${rec.mu == null ? 'no Mu' : `Mu ${rec.mu}`}, ` +
    // Only when there was one: most records carry neither.
    (rec.lora ? `LoRA ${rec.lora} @ ${rec.lora_scale}, ` : '') +
    // Gallery images by the page's one name for them, "#N" - the Refine
    // dialog's title, the sheet's source column and the confirms say it so.
    (rec.reference?.length ? `ref ${rec.reference.map((id) => `#${id}`).join('+')}, ` : '') +
    // A refine pass: the image it started from and how far it was re-denoised,
    // and whether that image is still in the gallery to make it again from.
    // An inpaint is a pass with a mask, and says so: "inpaint of #N @ 0.6".
    // An outpaint says its padding too: "outpaint of #N +400/+0/+400/+0 @ 0.9999".
    (rec.refine != null ? `${passWord(rec)} of #${rec.refine}${outpaintOf(rec)}${rec.refine_missing ? ` (${goneWord(rec)})` : ''} @ ${rec.strength}, ` : '') +
    // An instruction edit: the image it changed, and whether that is still
    // in the gallery to reopen the edit on.
    (rec.edit != null ? `edit of #${rec.edit.source ?? '?'}${rec.edit_missing ? ` (${editGoneWord(rec)})` : ''}, ` : '') +
    (rec.upscaler ? `upscaler ${upscalerName(rec.upscaler)}, ` : '') +
    (rec.crop_anchor && rec.crop_anchor !== 'center' ? `crop ${rec.crop_anchor}, ` : '') +
    `seed ${rec.seed}, ${(rec.duration_ms / 1000).toFixed(1)}s`
  );
}

// A recorded negative prompt as the page shows it. A whitespace-only one is
// the single space a live True CFG box sends (negativeSent) and runs the
// uncond pass, so it is named rather than shown as a blank that reads as none.
function negativeShown(text) {
  return text.trim() ? text : 'a single space';
}

// What made a record, as a list (`dl.facts`): the lightbox's side column, and
// whatever else shows one image with its settings. The one line it replaced
// was 12 px of commas that buried the seed mid-sentence and left out the
// negative prompt it gave the True CFG of, the date and the run. One row a
// fact, in the reproduction set's order; a row the record has no value for is
// left out. Its buttons open the image it was refined from (with this one
// after it, so → comes back), each reference, and the run as a sheet - and,
// where the view has no Use seed of its own (`useSeed`: the lightbox, where
// older images' seeds are browsed; not the result column, whose row of
// actions has one), put the seed in the panel's Seed box. `search` (the
// lightbox's) ends each row the search grammar can say with a ⌕ that adds it
// to the gallery's search (gallery-search.js). `megapixels` (the lightbox's,
// where a picture is looked at full size) follows the Size with its
// megapixels to one decimal, "1536×1024 (1.6 MP)", as the size hints say them.
function factsList(rec, { useSeed: seedButton = false, search = false, megapixels = false } = {}) {
  const dl = document.createElement('dl');
  dl.className = 'facts';
  const row = (term, ...value) => {
    const div = document.createElement('div');
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = term;
    dd.append(...value);
    div.append(dt, dd);
    dl.append(div);
    return dd;
  };
  // The running checkpoint by the catalogue's name, as the topbar has it (checkpointName).
  row('Model', [checkpointName(rec.model_id), rec.quantization].filter(Boolean).join(' · ')).title = rec.model_id;
  row('Size', `${rec.width}×${rec.height}` + (megapixels ? ` (${(rec.width * rec.height / 1e6).toFixed(1)} MP)` : ''));
  // An instruction edit's recipe has a guidance and no Mu or negative.
  row('Sampling', (rec.edit != null ? [`${rec.steps} steps`, `guidance ${rec.guidance_scale}`] : [
    `${rec.steps} steps`, `CFG ${rec.guidance_scale}`,
    // As the settings line has it: only where it is a number of its own.
    ...(rec.true_cfg != null && rec.true_cfg !== rec.guidance_scale ? [`True CFG ${rec.true_cfg}`] : []),
    rec.mu == null ? 'no Mu' : `Mu ${rec.mu}`,
  ]).join(' · '));
  // An instruction edit: the image it changed - a press opens it with this
  // one after it - "(deleted)" where it has gone, with a Restore where it
  // waits in the recycle bin; and the recipe that ran it.
  if (rec.edit != null) {
    const restore = rec.edit_missing && rec.edit_binned && rec.edit.source != null
      ? [' · ', sourceRestoreButton(rec.edit.source)]
      : [];
    row('Edited from', editSourceLink(rec), ...restore);
    row('Recipe', `${rec.edit.recipe} · ${EDIT_MODEL_NAME}`);
  }
  if (rec.negative_prompt) row('Negative', negativeShown(rec.negative_prompt)).className = 'text';
  if (rec.lora) row('LoRA', `${rec.lora} @ ${rec.lora_scale}`);
  if (rec.reference?.length) row('References', referenceThumbs(rec));
  if (rec.upscaler) row('Upscaler', upscalerName(rec.upscaler));
  if (rec.refine != null && !rec.mask && !rec.outpaint) row('Crop anchor', rec.crop_anchor || 'center');
  if (rec.refine != null) {
    // A source that has gone cannot be opened: said, not linked - with a
    // Restore where it waits in the recycle bin (bin.js).
    const from = rec.refine_missing ? `#${rec.refine} (${goneWord(rec)})`
      : factButton(`#${rec.refine}`, `Open #${rec.refine}, the image this was refined from; → comes back here`,
        () => { void openRefineSource(rec); });
    const restore = rec.refine_missing && rec.refine_binned
      ? [' · ', sourceRestoreButton(rec.refine)]
      : [];
    row(rec.outpaint ? 'Outpainted from' : rec.mask ? 'Inpainted from' : 'Refined from', from, ` @ ${rec.strength}`, ...restore);
    // An outpaint's padding, band and fill, as the server stored them.
    if (rec.outpaint) {
      row('Outpaint', `${outpaintSidesText(rec.outpaint)}, band ${rec.outpaint.band} px, fill ${rec.outpaint.fill}`);
    }
    // The mask as a small outline over the picture it was painted on
    // (maskOutline), and by the start of its hash, as the A1111 line names
    // it; the whole mask is a click away in Reuse inpaint…. An outpaint's the
    // server built from the padding and the band.
    if (rec.mask) {
      row('Mask', maskOutline(rec), rec.mask.slice(0, 16)).title = rec.outpaint
        ? `${rec.mask} - built by the server from the padding and the band` : rec.mask;
    }
  }
  const vars = Object.entries(rec.variables ?? {});
  if (vars.length) {
    // A comparison's version by the name the sheet gives it.
    row('Variables', vars.map(([k, v]) => (k === 'variant' ? `version: ${VERSION_NAMES.get(v) ?? v}` : `${k} = ${v}`)).join(', '));
  }
  // The panel is under the dialog, so the press says what it did in the
  // lightbox's own line - not the panel's, where it would replace a job's
  // outcome - and the run hint says the rest once it closes ("Seed fixed
  // at …").
  row('Seed', String(rec.seed), ...(seedButton ? [' · ', factButton('Use seed', `Put ${rec.seed} in the Seed box`, () => {
    useSeed(rec.seed);
    const text = `Seed ${rec.seed} is in the Seed box.`;
    showError(text, true, { slot: els.lbError, mirror: false });
    // Until another image is drawn (renderLightbox), whose seed it is not.
    lbSeedNote = { id: rec.id, text };
  })] : []));
  const made = new Date(rec.created_at).toLocaleString([], {
    year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });
  row('Created', `${made} · took ${(rec.duration_ms / 1000).toFixed(1)}s`, ...peakFact(rec));
  // What the card's ⊞ does: the run id spans every job that worked on a grid,
  // a resumed night included; the job id names one of them.
  const run = rec.run_id ? `run:${rec.run_id}` : rec.job_id ? `job:${rec.job_id}` : null;
  row('Run', run
    ? factButton('⊞ Show this run', `Show the run that made this as a sheet (${run})`, () => {
      if (els.lightbox.open) closeLightbox();
      void showSheetScope(run);
    })
    : 'none: made before runs were grouped');
  if (search) addFactSearches(dl, rec);
  return dl;
}

// The most GPU memory the call that made the image held, as PyTorch's
// allocator counts it - a batch's images share their call's, so the words
// say so. Nothing at all where nothing was measured: a CPU, the fake, every
// row from before it, with no "unknown" on each.
function peakFact(rec) {
  if (rec.vram_peak_allocated == null) return [];
  const gib = (bytes) => (bytes / 1024 ** 3).toFixed(2);
  const span = document.createElement('span');
  span.className = 'lb-peak';
  span.textContent = `${gib(rec.vram_peak_allocated)} GiB peak${rec.batch > 1 ? ' of its batch' : ''}`;
  const reserved = rec.vram_peak_reserved == null ? '' : ` (${gib(rec.vram_peak_reserved)} GiB reserved)`;
  span.title = `Peak GPU memory PyTorch allocated during the call that made this${reserved}. `
    + 'Not the whole card: the desktop, the driver and shared-memory spill are not counted.';
  return [' · ', span];
}

function sourceRestoreButton(id) {
  const button = factButton('Restore', `Move #${id} back out of the recycle bin`, () => { void restoreImages([id]); });
  syncBinWriteButton(button, [id]);
  return button;
}

function factButton(text, title, onClick) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'lb-link';
  b.textContent = text;
  b.title = title;
  b.onclick = onClick;
  return b;
}

// The images a render was conditioned on, numbered as the panel's chips are -
// "image 2" in a klein prompt is the second - each opening in the lightbox
// ahead of this one, and the set put back in the panel in one press, for a
// new prompt on the same inputs without a Reuse that replaces the panel.
function referenceThumbs(rec) {
  const ids = rec.reference;
  const block = document.createElement('div');
  block.className = 'lb-refs';
  const list = document.createElement('div');
  list.className = 'lb-ref-list';
  const use = document.createElement('button');
  // Each slot's "is it gone?" by id, for the Use press to ask again.
  const goneBy = new Map();
  ids.forEach((id, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'lb-ref';
    b.dataset.id = id;
    b.title = `Open #${id}, reference image ${i + 1}; → comes back here`;
    const img = document.createElement('img');
    img.src = `/api/images/${id}/thumb`;
    img.alt = '';
    img.loading = 'lazy';
    // Only the server's "unknown image" is a deleted one - or one in the
    // recycle bin, where it says so: then the slot says which and opens
    // nothing.
    const gone = (err) => {
      if (!/unknown image/.test(err.message)) return false;
      const binned = /recycle bin/.test(err.message);
      if (img.isConnected) {
        img.replaceWith(Object.assign(document.createElement('span'), {
          className: 'gone', textContent: binned ? 'in the bin' : 'deleted',
        }));
      }
      b.setAttribute('aria-disabled', 'true');
      b.title = `#${id}, reference image ${i + 1}, ${binned ? 'is in the recycle bin' : 'has been deleted'}`;
      syncUseReferences(use);
      return true;
    };
    goneBy.set(id, gone);
    // A picture that will not load is asked about.
    img.onerror = () => { api(`/api/images/${id}`).catch(gone); };
    const caption = document.createElement('span');
    caption.textContent = `image ${i + 1} · #${id}`;
    b.append(img, caption);
    b.onclick = async () => {
      if (b.getAttribute('aria-disabled') === 'true') return;
      let other;
      try {
        other = await api(`/api/images/${id}`);
      } catch (err) {
        if (!gone(err)) showError(err.message);
        return;
      }
      openLightbox(other, [other, rec]);
    };
    list.append(b);
  });
  use.type = 'button';
  use.className = 'ghost small lb-use-refs';
  use.textContent = 'Use these references';
  // Every id is asked about again, as Reuse's are (`checkReused`): a slot
  // drawn from the cache says nothing of an image deleted since, and a
  // deleted one put in the panel was an empty chip and a refused Generate.
  // Only the server's "unknown image" leaves one out; anything else keeps it.
  use.onclick = async () => {
    syncUseReferences(use);
    if (use.getAttribute('aria-disabled') === 'true') { showError(use.title, true); return; }
    const kept = await Promise.all(ids.map((id) => api(`/api/images/${id}`).then(() => true, (err) => !goneBy.get(id)(err))));
    const live = ids.filter((_, i) => kept[i]);
    const dead = ids.filter((_, i) => !kept[i]);
    // Asked again after the answers: a load can change the checkpoint meanwhile.
    syncUseReferences(use);
    if (use.getAttribute('aria-disabled') === 'true') { showError(use.title, true); return; }
    setReferences(live, { reveal: true });
    const put = live.length === 1 ? `Reference ${imageNames(live)} is in the panel`
      : `References ${imageNames(live)} are in the panel`;
    showError(!dead.length ? `${put}.` : `${put}; ${imageNames(dead)} ${dead.length === 1
      ? 'is no longer in the gallery, so it was' : 'are no longer in the gallery, so they were'} left out.`, true);
  };
  block.append(list, use);
  syncUseReferences(use);
  return block;
}

// Use these references, dead and saying why where the checkpoint takes none
// or every one of them has gone - from the checkpoint as it is now, so a
// render of the same facts (a status after a load) brings it up to date.
function syncUseReferences(use) {
  const slots = [...use.parentElement?.querySelectorAll('.lb-ref') ?? []];
  const ids = slots.map((b) => b.dataset.id);
  const why = !referenceSupported ? REFERENCES_NONE
    : slots.length && slots.every((b) => b.getAttribute('aria-disabled') === 'true')
      ? `${imageNames(ids)} ${ids.length === 1 ? 'is' : 'are'} no longer in the gallery.`
      : '';
  use.setAttribute('aria-disabled', String(!!why));
  use.title = why || 'Put these images in the panel as its references, in this order';
}

// The Mask row's picture fits a square of this many CSS pixels.
const MASK_THUMB = 80;

// A record's mask as a tint and an outline over the source's thumbnail - the
// picture it was painted on, as the Inpaint dialog shows it - in a box of the
// mask's own shape. An outpaint's mask is its canvas, so the source sits in
// it at its place by the padding, the new border round it. A source deleted
// from the gallery (or one that will not load) leaves the record's own
// picture under the mask: the same size, and still where it was painted. The
// mask is the stored PNG (`GET /api/masks/{hash}`, cached for good), read to
// the canvas's pixels: whatever level above 0 is repainted, as the server
// counts it. Colours are the page's tokens, the dialog's tint and line.
function maskOutline(rec) {
  const scale = MASK_THUMB / Math.max(rec.width, rec.height);
  const w = Math.max(1, Math.round(rec.width * scale));
  const h = Math.max(1, Math.round(rec.height * scale));
  const box = document.createElement('span');
  box.className = 'lb-mask';
  box.setAttribute('role', 'img');
  box.setAttribute('aria-label', `Mask over #${rec.refine}`);
  box.style.width = `${w}px`;
  box.style.height = `${h}px`;
  const img = document.createElement('img');
  img.alt = '';
  const own = () => {
    img.onerror = null;
    img.removeAttribute('style');
    img.src = rec.thumb_url || rec.url;
  };
  if (rec.refine_missing && !rec.refine_binned) {
    own();
  } else {
    img.src = `/api/images/${rec.refine}/thumb${rec.refine_binned ? '?binned=true' : ''}`;
    img.onerror = own;
    const o = rec.outpaint;
    if (o) {
      const pct = (n, of) => `${(100 * n) / of}%`;
      Object.assign(img.style, {
        left: pct(o.left, rec.width), top: pct(o.top, rec.height),
        width: pct(rec.width - o.left - o.right, rec.width), height: pct(rec.height - o.top - o.bottom, rec.height),
      });
    }
  }
  const dpr = window.devicePixelRatio || 1;
  const canvas = canvas2d(Math.round(w * dpr), Math.round(h * dpr));
  canvas.setAttribute('aria-hidden', 'true');
  box.append(img, canvas);
  void drawMaskOutline(canvas, rec.mask, Math.max(1, Math.round(1.5 * dpr)));
  return box;
}

// The mask onto `canvas`: a light tint where it repaints and a line `k`
// device pixels thick just inside its edge. A picture's border is not an
// edge: an outpaint's band has its line only on the kept side. Marks the
// canvas `data-drawn` once drawn ("failed" where the mask would not load).
async function drawMaskOutline(canvas, hash, k) {
  const { width: w, height: h } = canvas;
  let levels;
  try {
    const res = await fetch(`/api/masks/${hash}`);
    if (!res.ok) throw new Error(String(res.status));
    // Resized by the decoder at its best: a plain draw samples a few source
    // pixels per thumbnail pixel, and a thin stroke fell between them.
    const bitmap = await createImageBitmap(await res.blob(), { resizeWidth: w, resizeHeight: h, resizeQuality: 'high' });
    const scratch = canvas2d(w, h).getContext('2d', { willReadFrequently: true });
    scratch.drawImage(bitmap, 0, 0);
    bitmap.close?.();
    levels = scratch.getImageData(0, 0, w, h).data;
  } catch {
    canvas.dataset.drawn = 'failed';
    return;
  }
  const inside = new Uint8Array(w * h);
  for (let i = 0; i < inside.length; i++) inside[i] = levels[4 * i] > 0 ? 1 : 0;
  const style = getComputedStyle(document.documentElement);
  const tint = cssRgb(style.getPropertyValue('--accent').trim() || '#7c5cff');
  const line = cssRgb(style.getPropertyValue('--accent-2').trim() || '#4ad6c8');
  const out = new ImageData(w, h);
  const px = out.data;
  const outside = (x, y) => x >= 0 && y >= 0 && x < w && y < h && !inside[y * w + x];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!inside[y * w + x]) continue;
      let edge = false;
      for (let d = 1; d <= k && !edge; d++) {
        edge = outside(x - d, y) || outside(x + d, y) || outside(x, y - d) || outside(x, y + d);
      }
      const at = 4 * (y * w + x);
      const [r, g, b] = edge ? line : tint;
      px[at] = r; px[at + 1] = g; px[at + 2] = b;
      px[at + 3] = edge ? 255 : 90;
    }
  }
  canvas.getContext('2d').putImageData(out, 0, 0);
  canvas.dataset.drawn = 'true';
}

// A CSS colour as [r, g, b], by a one-pixel canvas.
function cssRgb(color) {
  const ctx = canvas2d(1, 1).getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, 1, 1);
  return [...ctx.getImageData(0, 0, 1, 1).data.slice(0, 3)];
}

// The image a pass was refined from, by id: from a strip the page holds, else
// asked for. Opened with the pass after it, so → comes back.
async function openRefineSource(rec) {
  let source = [...lbList, ...galleryRecords, ...viewerRecords].find((r) => r.id === rec.refine);
  if (!source) {
    try {
      source = await api(`/api/images/${rec.refine}`);
    } catch (err) {
      showError(/unknown image/.test(err.message) ? `#${rec.refine} is no longer in the gallery.` : err.message);
      return;
    }
  }
  openLightbox(source, [source, rec]);
}

// A card's image title: the facts that tell it from the card beside it, a
// pass's source and strength among them - "(deleted)" where the source has
// gone, "(in the recycle bin)" where it waits there (goneWord), as the
// lightbox's facts say it.
function cardTitle(rec) {
  const pass = rec.refine != null
    ? ` · ${passWord(rec)} of #${rec.refine}${outpaintOf(rec)}${rec.refine_missing ? ` (${goneWord(rec)})` : ''} @ ${rec.strength}`
    // An instruction edit says its source, gone or not, as a pass does.
    : rec.edit != null
      ? ` · edit of #${rec.edit.source ?? '?'}${rec.edit_missing ? ` (${editGoneWord(rec)})` : ''}` : '';
  return `#${rec.id} · ${rec.width}×${rec.height} · seed ${rec.seed}${pass}\n${rec.prompt}`;
}

// What a pass is called - "outpaint", "inpaint" or "refine" - by what its
// row carries: an outpaint has a mask as well, built by the server.
function passWord(rec) {
  return rec.outpaint ? 'outpaint' : rec.mask ? 'inpaint' : 'refine';
}

// An outpaint's padding after its source, " +400/+0/+400/+0", else ''.
function outpaintOf(rec) {
  return rec.outpaint ? ` ${outpaintSidesText(rec.outpaint)}` : '';
}

// Where a pass's gone source went (`refine_missing`): "deleted", or "in the
// recycle bin" (`refine_binned`), from which a Restore brings it back.
function goneWord(rec) {
  return rec.refine_binned ? 'in the recycle bin' : 'deleted';
}

// The same for an instruction edit's source (`edit_missing`, `edit_binned`).
function editGoneWord(rec) {
  return rec.edit_binned ? 'in the recycle bin' : 'deleted';
}
