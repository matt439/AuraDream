// Local Diffusion's page script: a record's facts - the settings line and the
// facts list the lightbox and the viewer share. Declarations only; state.js has
// the load-order contract these files share.
'use strict';

// ------------------------------------------------------------------- facts

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
    (rec.refine != null ? `${rec.mask ? 'inpaint' : 'refine'} of #${rec.refine}${rec.refine_missing ? ` (${goneWord(rec)})` : ''} @ ${rec.strength}, ` : '') +
    `seed ${rec.seed}, ${(rec.duration_ms / 1000).toFixed(1)}s`
  );
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
  row('Sampling', [
    `${rec.steps} steps`, `CFG ${rec.guidance_scale}`,
    // As the settings line has it: only where it is a number of its own.
    ...(rec.true_cfg != null && rec.true_cfg !== rec.guidance_scale ? [`True CFG ${rec.true_cfg}`] : []),
    rec.mu == null ? 'no Mu' : `Mu ${rec.mu}`,
  ].join(' · '));
  if (rec.negative_prompt) row('Negative', rec.negative_prompt).className = 'text';
  if (rec.lora) row('LoRA', `${rec.lora} @ ${rec.lora_scale}`);
  if (rec.reference?.length) row('References', referenceThumbs(rec));
  if (rec.refine != null) {
    // A source that has gone cannot be opened: said, not linked - with a
    // Restore where it waits in the recycle bin (bin.js).
    const from = rec.refine_missing ? `#${rec.refine} (${goneWord(rec)})`
      : factButton(`#${rec.refine}`, `Open #${rec.refine}, the image this was refined from; → comes back here`,
        () => { void openRefineSource(rec); });
    const restore = rec.refine_missing && rec.refine_binned
      ? [' · ', factButton('Restore', `Move #${rec.refine} back out of the recycle bin`, () => { void restoreImages([rec.refine]); })]
      : [];
    row(rec.mask ? 'Inpainted from' : 'Refined from', from, ` @ ${rec.strength}`, ...restore);
    // The mask by the start of its hash, as the A1111 line names it; its
    // picture is a click away in Reuse inpaint….
    if (rec.mask) row('Mask', rec.mask.slice(0, 16)).title = rec.mask;
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
  row('Created', `${made} · took ${(rec.duration_ms / 1000).toFixed(1)}s`);
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
    setReferences(live);
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
