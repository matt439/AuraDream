// The Models panel's On disk fold: what the model cache holds, from
// GET /api/models/storage (backend/model_cache_storage.py). Read only when
// the fold opens (once a page) or Refresh is pressed: never on a timer and
// never as the picker opens, since a scan reads every cached file's size.
// Read-only: nothing here deletes, and "not referenced" never means unused.
// Declarations only; app.js wires the fold after every script has loaded.
'use strict';

let modelStorageLoading = false;
let modelStorageRead = false;
let modelStorageEpoch = 0;

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
// under every entry that uses it, so the column is never summed.
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
    const variants = group.variants.filter(modelStorageHasBytes);
    if (!variants.length && !(group.entry && modelStorageHasBytes(group.entry))) continue;
    const variant = variants[0] ?? {};
    rows.push(group.entry ?? {
      model_id: key, entry: key, label: variant.label ?? null, variant: null, cached_bytes: 0,
      revision_bytes: 0, other_revision_bytes: 0, shared_bytes: 0, converted_bytes: 0,
    });
    rows.push(...variants);
  }
  return rows;
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
  head.append(headRow);
  const rowsBody = document.createElement('tbody');
  for (const row of rows) {
    const line = document.createElement('tr');
    if (row.variant) line.className = 'model-storage-variant';
    const name = modelStorageCell('th', modelStorageRowLabel(row));
    name.scope = 'row';
    name.title = row.model_id;
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
    line.append(name, cached, modelStorageCell('td', modelStorageBytes(row.shared_bytes)),
      modelStorageCell('td', modelStorageBytes(row.converted_bytes)));
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
  const parts = [modelStorageFacts(body), ...modelStorageModels(body), ...modelStorageOthers(body)];
  for (const warning of body.warnings ?? []) parts.push(modelStorageCell('p', warning, 'model-storage-warning'));
  parts.push(modelStorageCell('p', 'Read-only: nothing here deletes files. Shared files count under every checkpoint '
    + 'that uses them, so the rows do not add up. "Not referenced" means no catalogue download names those files, '
    + 'not that nothing uses them: the prompt critic, helpers and other apps may.', 'model-storage-note'));
  els.modelStorageBody.replaceChildren(...parts);
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

// `fresh`: Refresh's press; otherwise the fold's first opening this page.
// `polled`: asked again by the bundle status poll after a gallery bundle
// operation refused the read (afterGalleryMaintenance), never by a person.
async function loadModelStorage({ fresh = false, polled = false } = {}) {
  if (modelStorageLoading) {
    if (fresh) els.modelStorageStatus.textContent = 'Still reading the model cache; wait for it to finish.';
    return;
  }
  if (modelStorageRead && !fresh) return;
  const epoch = ++modelStorageEpoch;
  modelStorageLoading = true;
  holdModelStorageRefresh(true);
  els.modelStorageStatus.textContent = 'Reading the model cache… a large cache can take a while.';
  try {
    const body = await api('/api/models/storage', polled ? polledRead() : undefined);
    if (epoch !== modelStorageEpoch) return;
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
    }
  }
}
