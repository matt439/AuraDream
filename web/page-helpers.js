// AuraDream's page script: page-wide helpers that scripts across the page
// call - a count in words, a short excerpt, code-point offsets, a duration
// and the keyboard guards. Pure and declarations only; loaded right after
// state.js, before every script that calls them.
'use strict';

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

// A few words from the start of `text`, cut at a word boundary: enough to say
// where an encoder stops without printing the rest of the prompt again.
// Counted in code points, as `units()` counts: CJK has no space to cut at, and
// a cut by UTF-16 units through an emoji left half of it, a broken glyph.
function excerpt(text, max = 36) {
  const flat = [...text.trim().replace(/\s+/g, ' ')];
  if (flat.length <= max) return flat.join('');
  const at = flat.lastIndexOf(' ', max);
  return `${flat.slice(0, at > max / 2 ? at : max).join('')}…`;
}

// Every offset from the server counts code points, as Python indexes a str;
// the page's strings count UTF-16 units, where each character past the BMP -
// an emoji, a rare CJK ideograph - takes two. One offset into `text`, in the
// page's units.
const units = (text, points) => [...text].slice(0, points).join('').length;

function humanTime(seconds) {
  if (seconds < 90) return `${Math.round(seconds)}s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 90) return `${minutes} min`;
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`;
}

// Shared by local handlers and the document dispatcher. A key already owned
// by an editor/menu, or still composing, must never become a second command.
function keyboardEventHeld(event) {
  return event.defaultPrevented || event.isComposing || event.keyCode === 229;
}

function keyboardEditingTarget(target) {
  return !!target?.closest?.('input, textarea, select') || !!target?.isContentEditable;
}

function keyboardCommandModifier(event) {
  return event.ctrlKey || event.metaKey || event.altKey;
}

// A row of a result list - Swap tags', Mass refine's and More seeds' (the
// shared batch-* rules in styles/swap.css): the image's thumbnail, or a blank
// where it has gone, and a body opening on its line, "#N" and the status chip
// in `chips`' words. Every element keeps its dialog's `prefix` class too,
// which the smokes select on. Answers the row, its body and that line, for
// the dialog's own details.
function resultRow(image, prefix, chips) {
  const li = document.createElement('li');
  li.className = `${prefix}-result batch-result`;
  li.dataset.id = image.id;
  li.dataset.status = image.status;
  if (image.thumb_url) {
    const img = document.createElement('img');
    img.src = image.thumb_url;
    img.alt = '';
    img.loading = 'lazy';
    li.append(img);
  } else {
    li.append(Object.assign(document.createElement('span'), { className: `${prefix}-thumb-gone batch-thumb-gone` }));
  }
  const body = document.createElement('div');
  body.className = `${prefix}-body batch-body`;
  const head = document.createElement('p');
  head.className = `${prefix}-line batch-line`;
  const chip = Object.assign(document.createElement('span'), {
    className: `${prefix}-chip batch-chip`, textContent: chips[image.status] ?? image.status,
  });
  head.append(`#${image.id} `, chip);
  body.append(head);
  li.append(body);
  return { li, body, head };
}

// A result row's notes, on one line under the rest, where it has any.
function resultNotes(body, image, prefix) {
  if (!image.notes?.length) return;
  body.append(Object.assign(document.createElement('p'), {
    className: `${prefix}-notes batch-notes`, textContent: image.notes.join(' · '),
  }));
}
