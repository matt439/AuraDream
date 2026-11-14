// The Models panel's On disk fold: what the model cache holds, from
// GET /api/models/storage (backend/model_cache_storage.py), and a whole
// checkpoint's delete from it (`GEN#72`, docs/maintainer/plan-model-delete.md).
// Read only when the fold opens (once a page), Refresh is pressed or a
// delete has run: never on a timer and never as the picker opens, since a
// scan reads every cached file's size. Each entry's row ends in Delete…,
// which reads that entry's plan, asks, and deletes it from this server's
// caches; a Nunchaku lite id's sub-line has none (it goes with its entry).
// "Not referenced" never means unused. Declarations only; app.js wires the
// fold after every script has loaded.
'use strict';

let modelStorageLoading = false;
let modelStorageRead = false;
let modelStorageEpoch = 0;
// The last inventory shown, for a re-render in place (a delete's result, a
// held reason) without reading the cache again.
let modelStorageBody = null;
// The entry whose Delete… is reading its plan or deleting from this page,
// and the label of the delete the server says is running (`/api/status`'s
// `model_delete`, from the status poll: no timer of its own).
let modelStoragePlanning = null;
let modelStorageDeleting = null;
let modelDeleteRunning = null;
// A busy refusal holds every Delete until the next status says what holds
// the server; a held plan (`model_delete_held`) a press met that the read
// did not (the inventory's rows carry `delete_held`) holds its row with
// the server's words until Refresh.
let modelStorageBusyHold = '';
const modelStorageHeldPlans = new Map();
// Deleted entries' result lines, by entry id: each keeps its row (and its
// sub-lines) in place across every read until the fold is closed or Refresh
// is pressed, so no row below moves up under the pointer.
const modelStorageDeleted = new Map();

function modelStorageCell(tag, text, className = '') {
  const cell = document.createElement(tag);
  if (className) cell.className = className;
  cell.textContent = text;
  return cell;
}

// A size, or a dash for none; null is the server's unknown.
function modelStorageBytes(n) {
  if (n === null || n === undefined) return 'Unknown';
  return n > 0 ? humanBytes(n) : '—';
}

function modelStorageFacts(body) {
  const list = document.createElement('dl');
  list.className = 'model-storage-facts';
  const facts = [
    ['Hub cache', modelStorageBytes(body.hub_bytes), body.cache_dir],
    ['Free on its drive', body.free_bytes == null ? 'Unknown' : humanBytes(body.free_bytes), ''],
    ['Converted copies', modelStorageBytes(body.converted_bytes), body.converted_dir],
    ['Incomplete downloads', modelStorageBytes(body.incomplete_bytes), 'Partial files a stopped download left; the next download of that file resumes from them'],
    ['Not referenced', modelStorageBytes((body.unreferenced_bytes || 0) + (body.converted_unreferenced_bytes || 0)),
      'Files no catalogue entry\'s download names. Not the same as unused: the prompt critic, helpers and other apps may need them'],
  ];
  for (const [label, value, title] of facts) {
    const term = modelStorageCell('dt', label);
    const detail = modelStorageCell('dd', value);
    if (title) detail.title = title;
    list.append(term, detail);
  }
  return list;
}

function modelStorageHasBytes(row) {
  return row.revision_bytes > 0 || row.other_revision_bytes > 0 || row.converted_bytes > 0;
}

// What a 4-bit lite id's row is called under its entry ("Krea 2 Turbo,
// Nunchaku NVFP4 files"); the server names the entry (`entry`, `label`,
// `variant`), never a bare lite-infer id.
function modelStorageRowLabel(row) {
  const name = row.label
    || modelCatalogueEntries.find((entry) => entry.model_id === (row.entry ?? row.model_id))?.label
    || row.entry || row.model_id;
  return row.variant ? `${name}, Nunchaku ${row.variant.toUpperCase()} files` : name;
}

// Catalogue entries with files in the cache, each variant (a Nunchaku lite
// id) a sub-line under its entry's row, which shows (dashes where it holds
// nothing itself) whenever any of its lines has bytes. A shared file counts
// under every entry that uses it, so the column is never summed. A deleted
// entry keeps its row and the sub-lines it had, as its result.
function modelStorageGroups(models) {
  const groups = new Map();
  for (const row of models) {
    const key = row.entry ?? row.model_id;
    if (!groups.has(key)) groups.set(key, { entry: null, variants: [] });
    if (row.variant) groups.get(key).variants.push(row);
    else groups.get(key).entry = row;
  }
  const rows = [];
  for (const [key, group] of groups) {
    const deleted = modelStorageDeleted.get(key);
    const variants = deleted ? group.variants.filter((row) => deleted.variants.includes(row.model_id))
      : group.variants.filter(modelStorageHasBytes);
    if (!deleted && !variants.length && !(group.entry && modelStorageHasBytes(group.entry))) continue;
    const variant = variants[0] ?? {};
    rows.push(group.entry ?? {
      model_id: key, entry: key, label: variant.label ?? null, variant: null, cached_bytes: 0,
      revision_bytes: 0, other_revision_bytes: 0, shared_bytes: 0, converted_bytes: 0,
    });
    rows.push(...variants);
  }
  return rows;
}

// The catalogue entry a model id belongs to, as the inventory names it (a
// lite id's row carries its entry), else the id itself.
function modelStorageEntryOf(id) {
  return (modelStorageBody?.models ?? []).find((row) => row.model_id === id)?.entry ?? id;
}

// Why a row's Delete… cannot run now, or '' when it can: what the server
// would refuse, read from what the page already knows (the status poll's
// selection, load, jobs, card, switch and delete; this fold's own read,
// whose rows carry their plan's hold as `delete_held`).
function modelStorageDeleteHeld(row) {
  const entry = row.entry ?? row.model_id;
  const selected = modelRuntimeSettings?.model_id ?? loadedModel;
  if (serverOffline) return 'The server is not answering';
  if (modelStorageDeleted.has(entry)) return 'Deleted';
  if (modelDeleteRunning || modelStorageDeleting) return 'A checkpoint is being deleted';
  if (modelStoragePlanning) return 'Another checkpoint\'s delete is being read';
  if (selected && modelStorageEntryOf(selected) === entry) return 'It is the selected checkpoint. Choose another checkpoint first';
  if (loadedModel && modelState === 'ready' && modelStorageEntryOf(loadedModel) === entry) return 'It is loaded. Choose another checkpoint first';
  if (modelChangeHold()) return modelChangeHold();
  if (busy || elsewhereBusy) return 'A job is queued or running; delete once it ends';
  if (cardHolder) return `The ${cardHolder} is using the card; delete once it is done`;
  if (modelState === 'loading') return 'The weights are loading; delete once they have loaded';
  if (modelStorageLoading) return 'The model cache is being read; wait for it to finish';
  if (bundleStatus?.state === 'running') return GALLERY_MAINTENANCE_WORDS;
  if (modelStorageHeldPlans.has(entry)) return modelStorageHeldPlans.get(entry);
  // The inventory's own read of the plan's hold (a shared repository, ...).
  if (row.delete_held) return row.delete_held.replace(/\.$/, '');
  if (modelStorageBusyHold) return modelStorageBusyHold;
  return '';
}

function modelStorageDeleteButton(row) {
  const button = modelStorageCell('button', 'Delete…', 'ghost tiny model-storage-delete');
  button.type = 'button';
  button.dataset.entry = row.entry ?? row.model_id;
  button.setAttribute('aria-label', `Delete ${modelStorageRowLabel(row)}…`);
  return button;
}

// Every Delete… says now whether it can run, by `aria-disabled` and its
// title (the reason), and the status line says a delete that is running.
function syncModelStorageDelete() {
  for (const button of els.modelStorageBody.querySelectorAll('.model-storage-delete')) {
    const row = modelStorageRow(button.dataset.entry);
    const why = row ? modelStorageDeleteHeld(row) : 'Read the model cache again';
    if (why) {
      if (button.getAttribute('aria-disabled') !== 'true') button.setAttribute('aria-disabled', 'true');
      if (button.title !== why) button.title = why;
    } else {
      button.removeAttribute('aria-disabled');
      const title = `Delete ${modelStorageRowLabel(row)}'s files from this PC's model cache`;
      if (button.title !== title) button.title = title;
    }
  }
  const running = modelStorageDeleting ?? modelDeleteRunning;
  if (running && !modelStorageLoading) {
    const words = `Deleting ${running}…`;
    if (els.modelStorageStatus.textContent !== words) els.modelStorageStatus.textContent = words;
  }
}

function modelStorageRow(entry) {
  return (modelStorageBody?.models ?? []).find((row) => (row.entry ?? row.model_id) === entry && !row.variant)
    ?? (modelStorageBody?.models ?? []).find((row) => (row.entry ?? row.model_id) === entry) ?? null;
}

function modelStorageModels(body) {
  const rows = modelStorageGroups(body.models ?? []);
  if (!rows.length) {
    return [modelStorageCell('p', 'No catalogue checkpoint has files in the cache.', 'model-storage-empty')];
  }
  const table = document.createElement('table');
  table.className = 'model-storage-table';
  const caption = modelStorageCell('caption', 'Catalogue checkpoints in the cache');
  const head = document.createElement('thead');
  const headRow = document.createElement('tr');
  for (const [text, title] of [
    ['Checkpoint', ''],
    ['Cached', 'Its selected files in the Hub cache; Unknown where the app does not know which files a load selects'],
    ['Shared', 'Of those, files another catalogue entry also uses'],
    ['Converted', 'A converted copy this app made from the download'],
  ]) {
    const cell = modelStorageCell('th', text);
    cell.scope = 'col';
    if (title) cell.title = title;
    headRow.append(cell);
  }
  // The controls' column, named for a screen reader alone.
  const action = document.createElement('th');
  action.scope = 'col';
  action.append(modelStorageCell('span', 'Delete', 'sr-only'));
  headRow.append(action);
  head.append(headRow);
  const rowsBody = document.createElement('tbody');
  for (const row of rows) {
    const line = document.createElement('tr');
    if (row.variant) line.className = 'model-storage-variant';
    const name = modelStorageCell('th', modelStorageRowLabel(row));
    name.scope = 'row';
    name.title = row.model_id;
    name.dataset.entry = row.entry ?? row.model_id;
    const deleted = modelStorageDeleted.get(row.entry ?? row.model_id);
    if (deleted) {
      // The result in the row's own place; its sub-lines stay as dashes.
      line.classList.add('model-storage-deleted');
      const result = modelStorageCell('td', row.variant ? '—' : deleted.text, 'model-storage-result');
      result.colSpan = 4;
      line.append(name, result);
      rowsBody.append(line);
      continue;
    }
    // Cached only at a commit the app has no file list for (GEN#71): its
    // bytes, but which of them a load would use is unknown, as Models says.
    const otherOnly = row.cached_bytes === null && !(row.revision_bytes > 0) && row.other_revision_bytes > 0;
    let cached;
    // An entry's row shown only as its variants' heading holds nothing itself.
    if (row.cached_bytes !== null || !modelStorageHasBytes(row)) {
      cached = modelStorageCell('td', modelStorageBytes(row.cached_bytes ?? 0));
    } else if (otherOnly) {
      cached = modelStorageCell('td', `Unknown (${humanBytes(row.other_revision_bytes)})`);
      // A lite id's files can also be under lite-infer's renamed repository.
      cached.title = row.variant
        ? 'Downloaded at a different commit, or under lite-infer\'s new repository name, from the one this app records, so which of these files a load uses is unknown'
        : 'Downloaded at a different commit from the one this app records, so which of these files a load uses is unknown';
    } else {
      cached = modelStorageCell('td', `${modelStorageBytes(row.revision_bytes)}+`);
      cached.title = row.other_revision_bytes > 0
        ? `At least this much, and ${humanBytes(row.other_revision_bytes)} more at a different commit from the one this app records`
        : 'At least this much: the files a load selects are not all known';
    }
    // One Delete… per entry; a sub-line's files go with it.
    const control = modelStorageCell('td', '', 'model-storage-action');
    if (!row.variant) control.append(modelStorageDeleteButton(row));
    line.append(name, cached, modelStorageCell('td', modelStorageBytes(row.shared_bytes)),
      modelStorageCell('td', modelStorageBytes(row.converted_bytes)), control);
    rowsBody.append(line);
  }
  table.append(caption, head, rowsBody);
  return [table];
}

// Repositories none of whose cached revisions a catalogue entry selects or
// owns at a commit it does not record.
function modelStorageOthers(body) {
  const others = (body.repositories ?? []).filter((repo) => repo.cached_bytes > 0
    && (repo.revisions ?? []).every((revision) => !(revision.catalogue_models ?? []).length
      && !(revision.other_revision_of ?? []).length));
  if (!others.length) return [];
  const heading = modelStorageCell('p', 'Other repositories in the cache', 'model-storage-heading');
  const list = document.createElement('ul');
  list.className = 'model-storage-others';
  for (const repo of others) {
    const item = document.createElement('li');
    item.append(modelStorageCell('span', repo.repo_id, 'model-storage-repo'),
      modelStorageCell('span', humanBytes(repo.cached_bytes), 'model-storage-size'));
    if (repo.repo_type !== 'model') item.title = `A ${repo.repo_type} repository`;
    list.append(item);
  }
  return [heading, list];
}

function renderModelStorage(body) {
  modelStorageBody = body;
  // The focus stays on the Delete… (or the row) it was on across a re-render.
  const focused = els.modelStorageBody.contains(document.activeElement) ? document.activeElement : null;
  const entry = focused?.dataset?.entry ?? null;
  const parts = [modelStorageFacts(body), ...modelStorageModels(body), ...modelStorageOthers(body)];
  for (const warning of body.warnings ?? []) parts.push(modelStorageCell('p', warning, 'model-storage-warning'));
  parts.push(modelStorageCell('p', 'Delete… removes one checkpoint\'s downloaded files and converted copy from this '
    + 'PC\'s cache, after asking; files another checkpoint on disk still uses are kept. Shared files count under every '
    + 'checkpoint that uses them, so the rows do not add up. "Not referenced" means no catalogue download names those '
    + 'files, not that nothing uses them: the prompt critic, helpers and other apps may.', 'model-storage-note'));
  els.modelStorageBody.replaceChildren(...parts);
  syncModelStorageDelete();
  if (focused && !focused.isConnected) {
    // Its Delete…, or the row's name where the row has none now (a result).
    const named = (selector) => [...els.modelStorageBody.querySelectorAll(selector)]
      .find((element) => element.dataset.entry === entry);
    const target = entry ? named('.model-storage-delete') ?? named('tbody th[scope=row]') : null;
    if (target) {
      if (!target.matches('button')) target.tabIndex = -1;
      target.focus({ preventScroll: true });
    }
  }
}

function holdModelStorageRefresh(on) {
  if (on) {
    els.btnModelStorageRefresh.setAttribute('aria-disabled', 'true');
    els.btnModelStorageRefresh.title = 'The model cache is being read; wait for it to finish';
  } else {
    els.btnModelStorageRefresh.removeAttribute('aria-disabled');
    els.btnModelStorageRefresh.title = 'Read the model cache again';
  }
}

// `fresh`: Refresh's press or a delete's own re-read; otherwise the fold's
// first opening this page. `polled`: asked again by the bundle status poll
// after a gallery bundle operation refused the read
// (afterGalleryMaintenance), never by a person.
async function loadModelStorage({ fresh = false, polled = false } = {}) {
  if (modelStorageLoading) {
    if (fresh) els.modelStorageStatus.textContent = 'Still reading the model cache; wait for it to finish.';
    return;
  }
  if (modelStorageRead && !fresh) return;
  const epoch = ++modelStorageEpoch;
  modelStorageLoading = true;
  holdModelStorageRefresh(true);
  syncModelStorageDelete();
  els.modelStorageStatus.textContent = 'Reading the model cache… a large cache can take a while.';
  try {
    const body = await api('/api/models/storage', polled ? polledRead() : undefined);
    if (epoch !== modelStorageEpoch) return;
    modelStorageLoading = false;
    renderModelStorage(body);
    modelStorageRead = true;
    els.modelStorageStatus.textContent = `Read at ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}.`;
  } catch (err) {
    if (epoch !== modelStorageEpoch) return;
    if (galleryMaintenance(err)) {
      els.modelStorageStatus.textContent = `${GALLERY_MAINTENANCE_WORDS}, so the model cache cannot be read now. It is read again when the operation ends.`;
      // Only while the fold is open; a closed one reads at its next opening.
      afterGalleryMaintenance('model-storage', () => (els.modelStorage.open ? loadModelStorage({ polled: true }) : null));
    } else els.modelStorageStatus.textContent = `The model cache could not be read: ${err.message}`;
  } finally {
    if (epoch === modelStorageEpoch) {
      modelStorageLoading = false;
      holdModelStorageRefresh(false);
      syncModelStorageDelete();
    }
  }
}

// Refresh's press: what earlier presses said (results, held plans) goes,
// and the cache is read again.
function refreshModelStorage() {
  if (modelStorageLoading) {
    els.modelStorageStatus.textContent = 'Still reading the model cache; wait for it to finish.';
    return;
  }
  modelStorageDeleted.clear();
  modelStorageHeldPlans.clear();
  modelStorageBusyHold = '';
  void loadModelStorage({ fresh: true });
}

// The fold's toggle: opening reads once a page; closing lets the results
// go, so the next opening shows the inventory as read.
function toggleModelStorage() {
  if (els.modelStorage.open) {
    void loadModelStorage();
    return;
  }
  if (!modelStorageDeleted.size || modelStorageDeleting) return;
  modelStorageDeleted.clear();
  if (modelStorageBody) {
    const body = modelStorageBody;
    modelStorageBody = null;
    renderModelStorage(body);
  }
}

// The status poll's part: a delete running (here or in another tab), and
// whatever else holds the server, re-said on every Delete….
function receiveModelDeleteStatus(s) {
  const running = typeof s.model_delete === 'string' ? s.model_delete : null;
  const ended = modelDeleteRunning && !running && !modelStorageDeleting;
  modelDeleteRunning = running;
  // A busy refusal holds until a status says what holds the server now.
  modelStorageBusyHold = '';
  if (ended && els.modelStorageStatus.textContent.startsWith('Deleting ')) {
    els.modelStorageStatus.textContent = 'A checkpoint was deleted; Refresh reads the model cache again.';
  }
  if (els.modelStorageBody.querySelector('.model-storage-delete')) syncModelStorageDelete();
}

function modelStorageSay(words) {
  els.modelStorageStatus.textContent = words;
}

// The question, as a saved bundle's Delete asks it: the first line its
// title, then what goes, what stays and that it cannot be undone.
function modelDeleteQuestion(plan) {
  const downloaded = plan.parts.filter((part) => part.kind === 'hub' || part.kind === 'lite')
    .reduce((sum, part) => sum + part.bytes, 0);
  const converted = plan.parts.filter((part) => part.kind === 'converted' || part.kind === 'converted-old')
    .reduce((sum, part) => sum + part.bytes, 0);
  const lite = plan.parts.some((part) => part.kind === 'lite');
  const what = [
    downloaded > 0 ? `downloaded files ${humanBytes(downloaded)}` : '',
    converted > 0 ? `converted copy ${humanBytes(converted)}` : '',
  ].filter(Boolean).join(', ');
  const lines = [
    `Delete ${plan.label} from ${lite ? 'this PC\'s cache' : 'this PC'}?`,
    `Frees about ${humanBytes(plan.frees_bytes)}${what ? `: ${what}` : ''}.`,
  ];
  for (const share of plan.kept_shared ?? []) {
    lines.push(`Kept: ${humanBytes(share.bytes)} shared with ${share.with.join(', ')}.`);
  }
  for (const share of plan.also_taken_shared ?? []) {
    lines.push(`Also removes ${humanBytes(share.bytes)} shared with ${share.with.join(', ')}, which `
      + `${share.with.length === 1 ? 'has' : 'have'} nothing else on disk.`);
  }
  if (plan.removes_expander) lines.push('Also removes its prompt rewriter (Expand).');
  if (lite) {
    lines.push('Only this PC\'s cache: the Nunchaku files a WSL or Linux server keeps in its own cache stay.');
  }
  if (plan.gallery_images > 0) {
    lines.push(`${plan.gallery_images} gallery ${plan.gallery_images === 1 ? 'image' : 'images'} used it; Re-render `
      + 'downloads it again.');
  }
  lines.push('Gallery images, setups and favourites stay. Using it again downloads it again, which needs your '
    + 'Hugging Face access to its licence.');
  lines.push('Other apps using this Hugging Face cache lose these files too.');
  lines.push('This cannot be undone.');
  return lines.join('\n');
}

function modelDeleteResult(result) {
  const freed = result.freed_bytes ?? result.planned_bytes;
  const left = (result.remaining ?? []).reduce((sum, part) => sum + (part.bytes || 0), 0);
  const rest = (result.remaining ?? []).length
    ? ` ${humanBytes(left)} is left: ${result.remaining[0].reason}` : '';
  return `Deleted ${result.label}: about ${humanBytes(freed)} freed.${rest}`;
}

// A refusal, by its code (state.js), never its words; true when the plan
// should be read and asked again.
function modelDeleteRefused(entry, err) {
  if (modelDeleteChanged(err)) return true;
  if (modelDeleteHeldByServer(err)) modelStorageHeldPlans.set(entry, err.message.replace(/\.$/, ''));
  else if (modelDeleteBusy(err)) modelStorageBusyHold = err.message.replace(/\.$/, '');
  else if (galleryMaintenance(err)) {
    modelStorageSay(`${GALLERY_MAINTENANCE_WORDS}; delete once it ends.`);
    syncModelStorageDelete();
    return false;
  }
  // Selected, a download still writing, a file open: the sentence says
  // what to do, and nothing is held for it.
  else if (modelDeleteNotNow(err)) modelStorageBusyHold = '';
  modelStorageSay(err.message);
  if (modelDeleteGone(err)) void loadModelStorage({ fresh: true });
  syncModelStorageDelete();
  return false;
}

// Delete…'s press: the plan, the question, the delete with the plan's pin.
async function deleteModelStorage(button) {
  const entry = button.dataset.entry;
  const row = modelStorageRow(entry);
  if (!row) return;
  const held = modelStorageDeleteHeld(row);
  if (held) {
    modelStorageSay(`${held}.`);
    return;
  }
  // A changed cache between the plan and the delete reads the plan and asks
  // again, once more at most.
  for (let asked = 0; asked < 2; asked += 1) {
    let plan;
    modelStoragePlanning = entry;
    syncModelStorageDelete();
    try {
      plan = await api(`/api/models/storage/plan?model_id=${encodeURIComponent(entry)}`);
    } catch (err) {
      modelStoragePlanning = null;
      if (modelDeleteRefused(entry, err)) continue;
      return;
    } finally {
      modelStoragePlanning = null;
      syncModelStorageDelete();
    }
    if (plan.held) {
      modelStorageHeldPlans.set(entry, plan.held.detail.replace(/\.$/, ''));
      modelStorageSay(plan.held.detail);
      syncModelStorageDelete();
      return;
    }
    // Only files a present sharer still uses are left: nothing to free.
    if (!plan.parts.length) {
      const sharers = [...new Set((plan.kept_shared ?? []).flatMap((share) => share.with))].join(', ');
      modelStorageSay(`${plan.label} has nothing of its own on disk to delete`
        + `${sharers ? `: its files here are kept for ${sharers}` : ''}.`);
      return;
    }
    if (plan.selected || plan.loaded || plan.load_on_startup) {
      modelStorageSay(plan.load_on_startup && !plan.selected
        ? `${plan.label} is saved to load automatically on the next start. Choose another checkpoint first.`
        : `${plan.label} is the selected checkpoint. Choose another checkpoint first.`);
      return;
    }
    if (!await askConfirm(modelDeleteQuestion(plan), { okLabel: 'Delete checkpoint', destroy: true })) return;
    // Anything that started while the question was up holds it now.
    const after = modelStorageDeleteHeld(row);
    if (after) {
      modelStorageSay(`${after}.`);
      return;
    }
    modelStorageDeleting = plan.label;
    syncModelStorageDelete();
    let result;
    try {
      result = await api('/api/models/storage/delete', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: entry, pin: plan.pin }),
      });
    } catch (err) {
      modelStorageDeleting = null;
      if (modelDeleteRefused(entry, err)) continue;
      return;
    } finally {
      modelStorageDeleting = null;
      // The POST answered, so this delete is over: a status poll that saw
      // it running would otherwise hold every Delete… and say "Deleting …"
      // over the re-read's line until the next poll.
      if (modelDeleteRunning === plan.label) modelDeleteRunning = null;
    }
    // The result in the row's place at once, and across the re-read.
    const variants = (modelStorageBody?.models ?? [])
      .filter((line) => line.variant && (line.entry ?? line.model_id) === entry && modelStorageHasBytes(line))
      .map((line) => line.model_id);
    modelStorageDeleted.set(entry, { text: modelDeleteResult(result), variants });
    if (modelStorageBody) renderModelStorage(modelStorageBody);
    modelStorageSay(modelDeleteResult(result));
    await loadModelStorage({ fresh: true });
    void refreshModelCacheFacts();
    return;
  }
  modelStorageSay('The model cache kept changing; Refresh and try again.');
}
