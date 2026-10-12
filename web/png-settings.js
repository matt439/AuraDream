// Local Diffusion's page script: Load settings from PNG - a dropped or picked
// PNG's text read here, its settings filled in through Reuse.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// A PNG's first eight bytes.
const PNG_SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];

// One text chunk's bytes as text: Latin-1 for tEXt and zTXt, as the format
// and Pillow read them, UTF-8 for iTXt.
function latin1(bytes) {
  let text = '';
  for (let at = 0; at < bytes.length; at += 0x8000) {
    text += String.fromCharCode(...bytes.subarray(at, at + 0x8000));
  }
  return text;
}

// zTXt's and a compressed iTXt's text: zlib, which DecompressionStream calls
// 'deflate'.
async function inflate(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// A tEXt, zTXt or iTXt chunk's keyword and text, or null for one that does
// not parse - skipped, as Pillow skips it, rather than failing the file.
async function textChunk(type, data) {
  const nul = data.indexOf(0);
  if (nul < 1) return null;
  const key = latin1(data.subarray(0, nul));
  try {
    if (type === 'tEXt') return [key, latin1(data.subarray(nul + 1))];
    if (type === 'zTXt') return [key, latin1(await inflate(data.subarray(nul + 2)))];
    // iTXt: keyword, compression flag and method, language tag, translated
    // keyword, then the text.
    const compressed = data[nul + 1] === 1;
    const language = data.indexOf(0, nul + 3);
    const translated = language < 0 ? -1 : data.indexOf(0, language + 1);
    if (translated < 0) return null;
    const body = data.subarray(translated + 1);
    return [key, new TextDecoder('utf-8', { fatal: true }).decode(compressed ? await inflate(body) : body)];
  } catch {
    return null;
  }
}

// A PNG's text chunks by keyword - the last of a repeated one, as Pillow's
// `.info` keeps it for `--from-image` - or null for a file that is not a
// PNG. Only the chunk headers are walked, up to IEND; the pixels are neither
// decoded nor sent anywhere.
async function pngTextChunks(buffer) {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 8 || PNG_SIGNATURE.some((b, i) => bytes[i] !== b)) return null;
  const view = new DataView(buffer);
  const chunks = {};
  for (let at = 8; at + 8 <= bytes.length;) {
    const length = view.getUint32(at);
    const type = latin1(bytes.subarray(at + 4, at + 8));
    const start = at + 8;
    if (start + length > bytes.length) break;
    if (type === 'IEND') break;
    if (type === 'tEXt' || type === 'zTXt' || type === 'iTXt') {
      const entry = await textChunk(type, bytes.subarray(start, start + length));
      if (entry) chunks[entry[0]] = entry[1];
    }
    at = start + length + 4; // the CRC
  }
  return chunks;
}

// The one in flight, so a second file dropped meanwhile wins and the first's
// answer is not written over it.
let pngEpoch = 0;

// Load settings from PNG: the file read here, its text chunks - never the
// image - sent to the server, which answers them as the panel's settings
// after `applied()` (api_png.py), and the panel filled through Reuse, so
// Ctrl+Z, the note's Undo and the other-checkpoint sentence are Reuse's own.
// Nothing is stored, and the image is no reference.
async function loadPngSettings(file) {
  const epoch = ++pngEpoch;
  const name = file.name || 'The file';
  let chunks = null;
  try {
    chunks = await pngTextChunks(await file.arrayBuffer());
  } catch { /* unreadable: said below as not a PNG */ }
  if (epoch !== pngEpoch) return;
  if (!chunks) {
    showError(`${name} is not a PNG: settings are read from a PNG's text, so only a PNG can fill the panel.`);
    return;
  }
  let answer;
  try {
    answer = await api('/api/png/settings', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name.slice(0, 255), chunks }),
    });
  } catch (err) {
    if (epoch === pngEpoch) showError(err.message);
    return;
  }
  if (epoch !== pngEpoch) return;
  const ran = answer.settings;
  // The box keeps a negative `applied()` left out, greyed, as it keeps a
  // typed one; a seed that does not carry (another tool's) is the panel's.
  const rec = {
    ...ran, model_id: answer.model_id, negative_prompt: answer.negative_prompt,
    seed: ran.seed ?? els.seed.value,
  };
  reuseIntoPanel(rec, answer.size, answer.notes.join(' '), { name: answer.name, sampling: answer.sampling });
}

// What a file dragged over the page would do: a modal dialog keeps the page
// to itself, so there - and for anything but files - nothing.
function pngDropTarget(e) {
  return !!e.dataTransfer?.types?.includes('Files') && !document.querySelector('dialog[open]:modal');
}

// The drop: one file, read as a PNG; more than one is said, since one panel
// takes one image's settings.
function dropPng(e) {
  document.body.classList.remove('png-drop');
  const files = [...(e.dataTransfer?.files ?? [])];
  if (!files.length) return;
  if (files.length > 1) {
    showError(`${files.length} files were dropped; drop one PNG at a time, since its settings fill the one panel.`);
    return;
  }
  void loadPngSettings(files[0]);
}
