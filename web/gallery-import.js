// A separate write from Load settings from PNG, owned by the import dialog.
'use strict';

let importGalleryFile = null;
let importedGalleryImage = null;
let galleryImportPending = false;

function syncGalleryImport() {
  const choose = $('gallery-import-choose');
  const run = $('gallery-import-run');
  const waiting = galleryImportPending ? 'Wait for the image to finish importing' : '';
  const reason = waiting || (importGalleryFile ? '' : 'Choose, drop or paste an image first');
  choose.setAttribute('aria-disabled', String(!!waiting));
  choose.title = waiting;
  run.setAttribute('aria-disabled', String(!!reason));
  run.title = reason;
  $('gallery-import-view').hidden = !importedGalleryImage;
}

function galleryImportNote(text) {
  $('gallery-import-status').textContent = text;
}

function stageGalleryImport(files) {
  if (galleryImportPending) return;
  const [file] = files;
  if (!file) return;
  importedGalleryImage = null;
  importGalleryFile = files.length === 1 ? file : null;
  galleryImportNote(importGalleryFile ? file.name : 'Choose, drop or paste one image at a time.');
  syncGalleryImport();
}

async function runGalleryImport() {
  const file = importGalleryFile;
  if (!file || galleryImportPending) {
    galleryImportNote($('gallery-import-run').title);
    return;
  }
  galleryImportPending = true;
  galleryImportNote(`Importing ${file.name}…`);
  syncGalleryImport();
  try {
    const record = await api(`/api/gallery/import?name=${encodeURIComponent(file.name)}`, {
      method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: file,
    });
    importGalleryFile = null;
    importedGalleryImage = record;
    ownImageIds.add(record.id);
    galleryImportNote(`Imported ${record.imported_name} as #${record.id}.`);
    void resetGallery();
  } catch (err) {
    galleryImportNote(err.message);
  } finally {
    galleryImportPending = false;
    syncGalleryImport();
  }
}

function setupGalleryImport() {
  const button = $('btn-gallery-import');
  const dialog = $('gallery-import-dialog');
  const choose = $('gallery-import-choose');
  const input = $('gallery-import-file');
  button.onclick = () => {
    if (dialog.open) return;
    if (!galleryImportPending && !importedGalleryImage) galleryImportNote('');
    dialog.showModal();
    choose.focus({ preventScroll: true });
  };
  $('gallery-import-close').onclick = () => dialog.close();
  $('gallery-import-run').onclick = () => { void runGalleryImport(); };
  $('gallery-import-view').onclick = () => {
    const result = importedGalleryImage;
    const visible = galleryRecords.find((rec) => rec.id === result?.id);
    if (!result) return;
    dialog.close();
    const opener = isPhone() ? $('btn-gallery-more') : button;
    openLightbox(visible ?? result, visible ? galleryRecords : [result], opener);
  };
  choose.onclick = () => {
    if (galleryImportPending) galleryImportNote(choose.title);
    else input.click();
  };
  input.onchange = () => {
    stageGalleryImport([...input.files]);
    input.value = '';
  };
  const acceptDrop = (event) => {
    const transfer = event.dataTransfer;
    if (pageDrag || !transfer?.types.includes('Files')) return;
    event.preventDefault();
    event.stopPropagation();
    transfer.dropEffect = galleryImportPending ? 'none' : 'copy';
  };
  dialog.addEventListener('dragenter', acceptDrop);
  dialog.addEventListener('dragover', acceptDrop);
  dialog.addEventListener('drop', (event) => {
    const transfer = event.dataTransfer;
    if (pageDrag || !transfer?.types.includes('Files')) return;
    event.preventDefault();
    event.stopPropagation();
    stageGalleryImport([...transfer.files]);
  });
  dialog.addEventListener('paste', (event) => {
    const files = event.clipboardData?.files;
    if (!files?.length) return;
    event.preventDefault();
    stageGalleryImport([...files]);
  });
  dialog.addEventListener('close', () => {
    if (dialog.open) return;
    if (!galleryImportPending) {
      importGalleryFile = null;
      importedGalleryImage = null;
    }
    if (!els.lightbox.open) (isPhone() ? $('btn-gallery-more') : button).focus({ preventScroll: true });
    syncGalleryImport();
  });
}
