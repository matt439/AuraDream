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
