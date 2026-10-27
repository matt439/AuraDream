// AuraDream's page script: the lightbox's comparison across CFG, and
// across steps. Declarations only; state.js has the load-order contract these
// files share.
'use strict';

// ------------------------------------------------------------ ladders

// Any image is compared with another that differs from it in CFG alone: the
// same prompt, negative, seed, steps, mu, LoRA, references, strength, size,
// checkpoint, quantization and refine source (`sameExcept`, sheet.js, with
// CFG and True CFG skipped). The set of them, one image a CFG value and
// sorted by CFG, is its ladder. Worked out in the page from records the
// server has already made - the cells of a CFG grid, or separate Generates
// at one seed - with no route of its own: the pairing is presentation.
// Steps make a ladder the same way, Steps alone skipped: each axis is a
// `lbCompare.kind` of its own, with the words its controls say it in.
const CFG_SKIP = ['cfg', 'tcfg'];
const stepsWord = (n) => `${n} step${n === 1 ? '' : 's'}`;
const LADDER_AXES = {
  cfg: {
    kind: 'cfg', skip: CFG_SKIP, of: (r) => r.guidance_scale, name: (r) => `CFG ${r.guidance_scale}`,
    keys: 'CFG', value: 'CFG value', values: 'CFG values', lowest: 'lowest', highest: 'highest',
    lower: 'the lower CFG', first: 'The lower CFG', second: 'the higher', changed: 'CFG',
  },
  steps: {
    kind: 'steps', skip: ['steps'], of: (r) => r.steps, name: (r) => stepsWord(r.steps),
    keys: 'steps', value: 'step count', values: 'step counts', lowest: 'fewest', highest: 'most',
    lower: 'the fewer steps', first: 'Fewer steps', second: 'more steps', changed: 'the step count',
  },
};
// The axis of a comparison kind, or null for a pass's or a picked pair's.
const ladderAxis = (kind) => (Object.hasOwn(LADDER_AXES, kind) ? LADDER_AXES[kind] : null);
// The gallery's answer is up to LADDER_PAGES pages this long, read newest
// first; past them the held reason and the note say more may be in the
// gallery. One page was too few: a seed kept fixed across a few hundred
// prompts at one size filled it with the other prompts, and an image whose
// neighbour was older was told it had none.
const LADDER_LIMIT = 200;
const LADDER_PAGES = 5;

// What the gallery holds of each image's ladders, by query (ladderQuery's
// key, one an axis): { records, more, failed, model, seed, width, height }, asked once a
// session, emptied by closeLightbox, pruned by forgetImages and dropped when
// a job lands an image of the same checkpoint, seed and size.
const lbLadders = new Map();
let lbLadderTimer;
// The queries waiting out their 200 ms (ladderQueries'), so a re-render of
// the same image (a star, a status) does not put them back by 200 ms each
// time.
let lbLadderAsking = null;
// Every query on its way, by key: { match, stale }. A job landing an image
// it would have held, or the lightbox closing, marks it stale and takes it
// out, and its answer is then not kept - whichever image is in view by then,
// so a walk on to another seed does not let an out-of-date answer be cached
// for the rest of the session.
const lbLadderFlights = new Map();

const cfgName = (rec) => LADDER_AXES.cfg.name(rec);
// A picture's name on the stage: its value on the axis and its id, and its
// True CFG where that is a number of its own, as the facts list says it.
const ladderTag = (axis, rec) => `${axis.name(rec)} · #${rec.id}`
  + (axis.kind === 'cfg' && rec.true_cfg != null && rec.true_cfg !== rec.guidance_scale ? ` · True CFG ${rec.true_cfg}` : '');

// The images the page holds that a ladder can be drawn from, no request
// made: the strip being walked, the gallery's pages (the sheet's records are
// the same array), the viewer, this page's earlier results, the gallery's
// answers for this image's ladders, and the comparison's own ladder.
function ladderPool(rec) {
  const fetched = ladderQueries(rec).flatMap((q) => lbLadders.get(q.key)?.records ?? []);
  return [...lbList, ...galleryRecords, ...viewerRecords, ...sessionRecords, ...fetched, ...lbCompare.ladder];
}

// `rec`'s ladder along `axis`, ascending. Built once per comparison
// (compareLadder) and frozen while it is on, so stepping along it does not
// reorder it; off, worked out again on every sync from what the page holds.
// The images that agree with `rec` on everything but the axis (CFG's being
// CFG and True CFG), then: along CFG, where True CFG is not an echo of CFG
// on every one of them, only those at `rec`'s True CFG, so the ladder is CFG
// alone. One image a value: `rec` for its own, then any in `keep` (the
// partner a comparison has), then one ← → can reach (in lbList), then the
// newest.
function ladderOf(axis, rec, pool = ladderPool(rec), keep = []) {
  const seen = new Set();
  let set = [rec];
  seen.add(rec.id);
  for (const r of pool) {
    if (seen.has(r.id) || deletedIds.has(r.id)) continue;
    seen.add(r.id);
    if (sameExcept(r, rec, axis.skip)) set.push(r);
  }
  if (axis.kind === 'cfg' && !trueCfgEchoesCfg(set)) {
    set = set.filter((r) => (r.true_cfg ?? null) === (rec.true_cfg ?? null));
  }
  const walked = new Set(lbList.map((r) => r.id));
  const rank = (r) => (r.id === rec.id ? 4 : keep.includes(r.id) ? 3 : walked.has(r.id) ? 2 : 1);
  const byValue = new Map();
  for (const r of set) {
    const held = byValue.get(axis.of(r));
    if (!held || rank(r) > rank(held) || (rank(r) === rank(held) && r.id > held.id)) byValue.set(axis.of(r), r);
  }
  return [...byValue.values()].sort((a, b) => axis.of(a) - axis.of(b));
}
const cfgLadder = (rec, pool, keep) => ladderOf(LADDER_AXES.cfg, rec, pool, keep);

// The partner a comparison starts with: the value directly below, or the one
// above where this is the lowest.
function ladderNeighbour(ladder, rec) {
  const at = ladder.findIndex((r) => r.id === rec.id);
  if (at < 0) return ladder[0] ?? null;
  return ladder[at - 1] ?? ladder[at + 1] ?? null;
}

// The gallery search that finds the rest of `rec`'s ladder along `axis`:
// its seed, size, refine source, mu and LoRA, and the other axis's value -
// its steps for CFG's, its CFG for steps' - in its checkpoint's exact scope
// (the `model=` parameter), narrowed so a seed used for hundreds of prompts,
// or swept along the other axis, does not fill the pages with images the
// ladder filters out here anyway. One answer for both ladders, with neither
// term, let a steps sweep at one seed crowd an older CFG neighbour out of
// the five pages. A seed past 2**53 (a grid's `seed + i` near the top)
// cannot be said exactly from a JavaScript number: the run is asked for
// instead, and with neither the ladder is what the page holds. Null then.
function ladderQuery(rec, axis = LADDER_AXES.cfg) {
  const terms = [];
  if (Number.isSafeInteger(rec.seed)) terms.push(`seed:${rec.seed}`);
  else if (rec.run_id) terms.push(`run:${rec.run_id}`);
  else return null;
  // A number the grammar reads back exactly: an exponent's "-" would read as
  // a range.
  const exact = (n) => !/e/i.test(String(n));
  terms.push(`size:${rec.width}x${rec.height}`);
  if (axis.kind === 'cfg') terms.push(`steps:${rec.steps}`);
  else if (exact(rec.guidance_scale)) terms.push(`cfg:${rec.guidance_scale}`);
  terms.push(`refine:${rec.refine ?? 'none'}`);
  if (rec.mu == null) terms.push('mu:none');
  else if (exact(rec.mu)) terms.push(`mu:${rec.mu}`);
  // `lora:=` is the whole name, `lora:none` no adapter at all.
  if (!rec.lora) terms.push('lora:none');
  else if (!/\s/.test(rec.lora)) terms.push(`lora:=${rec.lora}`);
  const search = terms.join(' ');
  return {
    key: `${rec.model_id}\n${search}`, model: rec.model_id, search,
    match: { model: rec.model_id, seed: rec.seed, width: rec.width, height: rec.height },
  };
}
const ladderQueries = (rec) => Object.values(LADDER_AXES).map((axis) => ladderQuery(rec, axis)).filter(Boolean);
const asksFor = (rec, q) => ladderQueries(rec).some((own) => own.key === q.key);

// Asked a moment after the image lands, as its passes are (passesLine), so
// a walk does not ask for every image it passes; once a session per query,
// both axes' together. A failure leaves the ladder to what the page holds,
// said "may be incomplete"; it is a convenience, as the passes' count is.
function askLadder(rec) {
  const qs = ladderQueries(rec).filter((q) => !lbLadders.has(q.key) && !lbLadderFlights.has(q.key));
  if (!qs.length || qs.every((q) => lbLadderAsking?.some((a) => a.key === q.key))) return;
  clearTimeout(lbLadderTimer);
  lbLadderAsking = qs;
  lbLadderTimer = setTimeout(() => {
    if (lbLadderAsking === qs) lbLadderAsking = null;
    for (const q of qs) void fetchLadder(q);
  }, 200);
}

// The gallery's pages for `q`, newest first by the id cursor (`before`), as
// the grid pages, until one says there is no more or LADDER_PAGES are read.
async function fetchLadder(q) {
  const flight = { match: q.match, stale: false, needs: null };
  lbLadderFlights.set(q.key, flight);
  let entry;
  // The gallery's newest id as the first page read it (`GalleryPage.
  // newest_image_id`, under the same reservation): every image up to it is
  // in this answer, which noteUnseenImages weighs a catch-up against.
  let head = null;
  try {
    const records = [];
    let more = false;
    for (let n = 0; n < LADDER_PAGES && !flight.stale; n += 1) {
      const params = new URLSearchParams({ limit: String(LADDER_LIMIT), model: q.model, search: q.search });
      if (records.length) params.set('before', String(records.at(-1).id));
      const page = await api(`/api/gallery?${params}`);
      if (n === 0 && Number.isInteger(page.newest_image_id)) head = page.newest_image_id;
      records.push(...page.items);
      more = !!page.has_more && page.items.length > 0;
      if (!more) break;
    }
    entry = { records, more, failed: false };
  } catch {
    entry = { records: [], more: false, failed: true };
  }
  if (lbLadderFlights.get(q.key) === flight) lbLadderFlights.delete(q.key);
  if (flight.stale || !els.lightbox.open) return;
  const now = lbList[lbIndex];
  // Read before images a catch-up said are in (noteUnseenImages): asked
  // again, for the image in view where it is this ladder's.
  if (!entry.failed && flight.needs !== null && head !== null && head < flight.needs) {
    if (now && asksFor(now, q)) askLadder(now);
    return;
  }
  entry.records = entry.records.filter((r) => !deletedIds.has(r.id));
  lbLadders.set(q.key, { ...entry, ...q.match, head });
  if (now && asksFor(now, q)) ladderAnswered(now);
}

// The gallery's answer for the image in view: a comparison on takes in what
// it found - its partner and the image in view kept - and the controls say
// what there is.
function ladderAnswered(rec) {
  const axis = ladderAxis(lbCompare.kind);
  if (lbCompare.id === rec.id && axis && lbCompare.source) {
    lbCompare.ladder = ladderOf(axis, rec, ladderPool(rec), [lbCompare.source.id]);
  }
  syncCompare(rec);
}

// Whether the page knows all it is going to about `rec`'s ladder along
// `axis`: the gallery has answered, or there is nothing to ask it. `more`:
// the answer was a full page, or no answer came.
function ladderState(rec, axis = LADDER_AXES.cfg) {
  const q = ladderQuery(rec, axis);
  if (!q) return { known: true, more: false };
  const entry = lbLadders.get(q.key);
  return entry ? { known: true, more: entry.more || entry.failed, failed: entry.failed, count: entry.records.length }
    : { known: false, more: false };
}

// Why there is nothing to compare along `axis`: still asking, or none -
// said as far as the page knows it, where the gallery's answer was cut short
// or never came.
function ladderHeldReason(axis, rec, state = ladderState(rec, axis)) {
  if (!state.known) return `Looking for other ${axis.values} of #${rec.id}…`;
  if (state.failed) return `No other ${axis.value} of #${rec.id} in what the page holds: the gallery could not be asked`;
  if (state.more) {
    return `No other ${axis.value} of #${rec.id} in the newest ${state.count} at this seed and size: `
      + 'more may be in the gallery';
  }
  return `No other ${axis.value} of #${rec.id} at this seed, size and settings`;
}

function forgetLadders() {
  clearTimeout(lbLadderTimer);
  lbLadderAsking = null;
  for (const flight of lbLadderFlights.values()) flight.stale = true;
  lbLadderFlights.clear();
  lbLadders.clear();
}

// A job's images landed - streamed (job-stream.js), or put in ahead of the
// gallery's cards (job-results.js): every ladder of their checkpoint, seed
// and size is asked for again, and the image in view says what it has now.
function noteLadderImages(records) {
  const same = (match, rec) => match.model === rec.model_id && match.seed === rec.seed
    && match.width === rec.width && match.height === rec.height;
  for (const rec of records) {
    for (const [key, entry] of lbLadders) if (same(entry, rec)) lbLadders.delete(key);
    // Any answer on its way may have been read before this image was saved,
    // whichever image asked for it; one still waiting out its 200 ms reads
    // after, and stands.
    for (const [key, flight] of lbLadderFlights) {
      if (!same(flight.match, rec)) continue;
      flight.stale = true;
      lbLadderFlights.delete(key);
    }
  }
  const now = lbList[lbIndex];
  if (!els.lightbox.open || !now) return;
  askLadder(now);
  syncCompare(now);
}

// Images saved that this page was not handed one by one - another tab's
// job, the CLI's, taken in by a catch-up (catchUpGallery) whose view may
// leave them out: a sheet, the bin, Starred, a search. Any ladder may have a
// new value, so every one held that was read before `newest` (the status's
// newest id; an answer's `head` is the gallery's as it read it) is asked
// again, an answer on its way is kept only if it read that far, and the
// image in view says what it has now. A ladder read after those images were
// saved has them already: a catch-up landing late - a slow poll after
// another tab's jobs, the gallery already reset over them - re-asked every
// ladder and took the comparison's note away under the viewer
// (smoke-compare-cfg). Without a number (the server back, or restarted,
// with the gallery held), every one. Nothing to do with the lightbox shut:
// closing it forgets them (forgetLadders).
function noteUnseenImages(newest = null) {
  if (!lbLadders.size && !lbLadderFlights.size) return;
  const known = Number.isInteger(newest);
  let dropped = false;
  for (const [key, entry] of lbLadders) {
    if (known && Number.isInteger(entry.head) && entry.head >= newest) continue;
    lbLadders.delete(key);
    dropped = true;
  }
  for (const [key, flight] of lbLadderFlights) {
    if (known) { flight.needs = Math.max(flight.needs ?? newest, newest); continue; }
    flight.stale = true;
    lbLadderFlights.delete(key);
  }
  const now = lbList[lbIndex];
  if (!dropped && known) return;
  if (!els.lightbox.open || !now) return;
  askLadder(now);
  syncCompare(now);
}

// Images deleted (forgetImages, before the strips close up): out of every
// ladder, and a comparison whose partner went ends, saying so.
function forgetLadderImages(gone) {
  for (const entry of lbLadders.values()) entry.records = entry.records.filter((r) => !gone.has(r.id));
  if (lbPickedPair.some((r) => gone.has(r.id))) lbPickedPair = [];
  if (lbCompare.id === null || !(ladderAxis(lbCompare.kind) || lbCompare.kind === 'picked')) return;
  lbCompare.ladder = lbCompare.ladder.filter((r) => !gone.has(r.id));
  const partner = lbCompare.source;
  if (!partner || !gone.has(partner.id)) return;
  const at = lbCompare.id;
  setCompare(null);
  lbCompareSaid = { id: at, text: `#${partner.id} is no longer in the gallery` };
}

// ------------------------------------------------------------ stepping

// The index in lbList of the next value of `ladder` from `rec` in the
// direction of `delta` that the strip holds, or -1. Only the strip: an image
// the gallery's answer brought is a partner, never a step, so ← → never
// leave the strip they walk.
function ladderReach(rec, ladder, delta) {
  const at = ladder.findIndex((r) => r.id === rec.id);
  if (at < 0) return -1;
  for (let k = at + delta; k >= 0 && k < ladder.length; k += delta) {
    const i = lbList.findIndex((r) => r.id === ladder[k].id);
    if (i >= 0) return i;
  }
  return -1;
}

// Why there is no step that way: the end of the ladder, or values past it
// the strip does not hold - to be picked under Against where Against lists
// this axis (`against`), or once this axis's comparison is on.
function ladderEndReason(axis, rec, ladder, delta, against = true) {
  const at = ladder.findIndex((r) => r.id === rec.id);
  const beyond = ladder[at + delta];
  if (!beyond) return `${axis.name(rec)} is the ${delta < 0 ? axis.lowest : axis.highest} of these`;
  return `${axis.name(beyond)} (#${beyond.id}) is not in what ← → walk here: `
    + `${against ? '' : `turn on ${ladderControls(axis).label} to `}pick it under Against`;
}

// Whether ← → step along the axis of the comparison on, with another value
// the strip holds either way. Otherwise they walk, ending it.
function ladderArrows() {
  const rec = lbList[lbIndex];
  if (!rec || lbCompare.id !== rec.id || !ladderAxis(lbCompare.kind)) return false;
  return ladderReach(rec, lbCompare.ladder, -1) >= 0 || ladderReach(rec, lbCompare.ladder, 1) >= 0;
}

// ← → while a comparison across CFG or steps is on: the next value that way
// the strip holds, compared with the value below it, in the same mode, at
// the same divider and zoom. At an end of the ladder the press stays and
// says why - the ladder is ordered, so it does not wrap round as passes do.
// False where the strip holds no other value either way, and the arrows
// walk.
function stepLadder(delta) {
  const rec = lbList[lbIndex];
  const axis = ladderAxis(lbCompare.kind);
  if (!rec || lbCompare.id !== rec.id || !axis) return false;
  const ladder = lbCompare.ladder;
  const i = ladderReach(rec, ladder, delta);
  if (i >= 0) { goLadder(i, ladder); return true; }
  if (ladderReach(rec, ladder, -delta) < 0) return false;
  lbCompareSaid = { id: rec.id, text: ladderEndReason(axis, rec, ladder, delta) };
  syncCompare(rec);
  return true;
}

// To lbList[i], the comparison carried with its kind (renderLightbox ends
// one whose id is not the image's), against the value below it.
function goLadder(i, ladder) {
  const next = lbList[i];
  lbCompare.id = next.id;
  lbCompare.ladder = ladder;
  lbCompare.source = ladderNeighbour(ladder, next);
  lbIndex = i;
  renderLightbox();
}

// Each axis's own controls: its Compare button and its two steps.
const ladderControls = (axis) => (axis.kind === 'cfg'
  ? { button: els.lbCompareCfg, label: 'Compare CFG', row: els.lbCfgSteps, lower: els.lbCfgLower, higher: els.lbCfgHigher }
  : { button: els.lbCompareSteps, label: 'Compare steps', row: els.lbStepsSteps, lower: els.lbStepsLower, higher: els.lbStepsHigher });

// Lower / Higher CFG and Fewer / More steps: a step as ← → take while
// comparing along that axis, which turns that comparison on where it was
// off, or another one was on. Held, the press says why.
function pressLadderStep(kind, delta) {
  const rec = lbList[lbIndex];
  const axis = ladderAxis(kind);
  if (!rec || !axis) return;
  const { lower, higher } = ladderControls(axis);
  const button = delta < 0 ? lower : higher;
  if (button.getAttribute('aria-disabled') === 'true') {
    lbCompareSaid = { id: rec.id, text: button.title };
    syncCompare(rec);
    return;
  }
  const on = lbCompare.id === rec.id && lbCompare.kind === kind;
  const ladder = on ? lbCompare.ladder : ladderOf(axis, rec);
  const i = ladderReach(rec, ladder, delta);
  if (i < 0) return;
  if (on) { goLadder(i, ladder); return; }
  if (lbCompare.id !== null) setCompare(null);
  lbIndex = i;
  renderLightbox();
  const next = lbList[lbIndex];
  compareLadder(kind, next, ladderNeighbour(ladder, next));
}

// ------------------------------------------------------------ comparing

// The comparison along `kind`'s axis on, against `partner` or the default
// neighbour, in `mode` or the last one; with nothing to compare, the press
// says why.
function compareLadder(kind, rec, partner = null, mode = null) {
  const axis = ladderAxis(kind);
  const ladder = ladderOf(axis, rec, ladderPool(rec), partner ? [partner.id] : []);
  if (ladder.length < 2) {
    lbCompareSaid = { id: rec.id, text: ladderHeldReason(axis, rec) };
    syncCompare(rec);
    return;
  }
  const other = (partner && ladder.find((r) => r.id === partner.id)) ?? ladderNeighbour(ladder, rec);
  lbCompare.kind = kind;
  lbCompare.ladder = ladder;
  lbCompare.source = other;
  setCompare(rec.id, mode ?? lbCompare.mode);
}

// Against: the partner, from the ladder of the comparison on - CFG's while
// none is. Off, a pick turns the comparison on against it; on, it swaps the
// partner and keeps the rest.
function pickAgainst() {
  const rec = lbList[lbIndex];
  if (!rec) return;
  if (pickedPair(rec)) { comparePickedPair(rec); return; }
  const on = lbCompare.id === rec.id && !!ladderAxis(lbCompare.kind);
  const kind = on ? lbCompare.kind : 'cfg';
  const ladder = on ? lbCompare.ladder : ladderOf(LADDER_AXES[kind], rec);
  const partner = ladder.find((r) => String(r.id) === els.lbCompareAgainst.value);
  if (!partner || partner.id === rec.id) { syncCompare(rec); return; }
  if (!on) { compareLadder(kind, rec, partner); return; }
  lbCompare.source = partner;
  lbCompareSaid = null;
  syncCompare(rec);
  layoutLightbox();
}

// One axis's part of the comparison's controls: its Compare button (S on
// CFG's where the image is no pass, `sKey`) and its two steps. Each is held,
// its title saying why, while the gallery is still being asked and the page
// holds no other value, and where there is none; a press says so. Under
// 901 px drawn only where `shown` (ladderShown, lightbox-compare.js).
// `against`: whether Against lists this axis. Answers `held`, the reason a
// press cannot turn it on; `steps`, whether ← → step along it; `note`, the
// line while on; and the ladder and partner Against offers.
function syncAxisCompare(axis, rec, on, sKey, shown, against) {
  const { button, label, row, lower, higher } = ladderControls(axis);
  const state = ladderState(rec, axis);
  button.hidden = row.hidden = !shown;
  const ladder = on ? lbCompare.ladder : ladderOf(axis, rec);
  const partner = on ? lbCompare.source : ladderNeighbour(ladder, rec);
  const held = ladder.length > 1 && partner ? '' : ladderHeldReason(axis, rec, state);
  if (button.textContent !== label) button.textContent = label;
  button.setAttribute('aria-pressed', String(on));
  button.setAttribute('aria-disabled', String(!!held));
  if (sKey) button.setAttribute('aria-keyshortcuts', 'S');
  else button.removeAttribute('aria-keyshortcuts');
  button.title = held || `Compare ${axis.name(rec)} with ${axis.name(partner)}, #${partner.id}, at the same crop, `
    + `${axis.lower} on the left${sKey ? ' (S)' : ''}`;
  for (const [step, delta] of [[lower, -1], [higher, 1]]) {
    const i = held ? -1 : ladderReach(rec, ladder, delta);
    const why = held || (i < 0 ? ladderEndReason(axis, rec, ladder, delta, against) : '');
    step.setAttribute('aria-disabled', String(!!why));
    if (why) step.title = why;
    else {
      const next = lbList[i];
      const against = ladderNeighbour(ladder, ladder.find((r) => r.id === next.id) ?? next);
      step.title = `Show ${axis.name(next)}, #${next.id}, compared with ${axis.name(against)}${on ? ` (${delta < 0 ? '←' : '→'})` : ''}`;
    }
  }
  const steps = !held && (ladderReach(rec, ladder, -1) >= 0 || ladderReach(rec, ladder, 1) >= 0);
  let note = '';
  if (on && partner) {
    const at = ladder.findIndex((r) => r.id === rec.id);
    note = `${axis.name(partner)}, #${partner.id} · ${at + 1} of ${ladder.length} ${axis.values}`
      + (partner.content_hash && partner.content_hash === rec.content_hash ? ` · identical pixels: ${axis.changed} changed nothing here` : '')
      + (state.more ? ' · may be incomplete: more may be in the gallery' : '');
  }
  return { held, steps, note, ladder, partner };
}

// The ladders' part of the comparison's controls - CFG's, then steps' - and
// Against, for the axis on (CFG's while neither is), drawn with that axis's
// part. `onKind` is the kind of the comparison on for this image, or null;
// `shown` says per axis whether it is drawn (ladderShown). Answers what the
// shared part needs: `held`, the reason a mode press cannot turn CFG's on
// (the kind a mode turns on where the image is no pass); `steps`, whether
// ← → step along the axis on; `note`; `shown`, whether either is drawn.
function syncLadderCompare(rec, onKind, sKey, shown) {
  const axis = onKind === 'steps' ? LADDER_AXES.steps : LADDER_AXES.cfg;
  const cfg = syncAxisCompare(LADDER_AXES.cfg, rec, onKind === 'cfg', sKey, shown.cfg, axis.kind === 'cfg');
  const steps = syncAxisCompare(LADDER_AXES.steps, rec, onKind === 'steps', false, shown.steps, axis.kind === 'steps');
  els.lbAgainst.hidden = !shown[axis.kind];
  const along = onKind === 'steps' ? steps : cfg;
  syncAgainst(rec, along.ladder, along.partner, along.held, (r) => `${axis.name(r)} · #${r.id}`,
    `The image ${axis.name(rec)} is compared with; ${axis.lower} is on the left`);
  return { held: cfg.held, steps: along.steps, note: onKind ? along.note : '', shown: shown.cfg || shown.steps };
}

// Against's options, rewritten only when they change, so an open list is not
// pulled from under the pointer by a sync. The other values of the ladder,
// "CFG 2 · #41" or "20 steps · #41", the partner selected; one "none" where
// it is held.
function syncAgainst(rec, ladder, partner, held, option = (r) => `${cfgName(r)} · #${r.id}`, title = null) {
  const select = els.lbCompareAgainst;
  const options = held ? [['', 'none']] : ladder.filter((r) => r.id !== rec.id)
    .map((r) => [String(r.id), option(r)]);
  const signature = options.map(([v, t]) => `${v}\t${t}`).join('\n');
  if (select.dataset.options !== signature) {
    select.replaceChildren(...options.map(([value, text]) => Object.assign(document.createElement('option'), { value, text })));
    select.dataset.options = signature;
  }
  select.value = held ? '' : String(partner?.id ?? '');
  select.setAttribute('aria-disabled', String(!!held));
  select.title = held || title || `The image ${cfgName(rec)} is compared with; the lower CFG is on the left`;
}
