// The accessibility audit: what the page's markup tells a screen reader as it
// is drawn now. Loaded into the page by a smoke's {"inject": "a11y_audit.js"}
// step (again after every navigate), then read from `log` and `expect` steps:
//
//   {"log": "__a11yAudit()"}                         the whole report
//   {"expect": "__a11yCheck()"}                      every check, saying what failed
//   {"eval": "__a11yWatch()"}                        audit each dialog as it opens and shuts
//   {"expect": "__a11yAudit({root: '#refine-dialog'}).unnamed.length === 0"}
//
// __a11yAudit(options) returns
//   unnamed     exposed controls with no accessible name: no aria-labelledby
//               text, aria-label, label, text or alt inside, title, or (a
//               field) placeholder
//   duplicates  ids more than one element of the document carries, exposed or
//               not (a label's `for` or an aria reference finds the first only)
//   unreasoned  exposed elements held with aria-disabled="true" and no reason:
//               no title, no aria-describedby text (the held-button
//               convention: the reason is the title, and a press says it)
//   dialogs     open dialogs with no accessible name (aria-labelledby text or
//               aria-label; a title does not count)
//   disabled    exposed controls that are natively `disabled`. The convention
//               has documented exceptions, so these are listed for a log and
//               never counted in `ok`
//   excepted    entries the `except` list kept out of unnamed and unreasoned
//   ok          true when unnamed, duplicates, unreasoned and dialogs are empty
// Each entry names the element (tag#id.class "label"); a duplicate is
// `#id x2`.
//
// Options:
//   root    a selector or element whose controls and holds are audited; the
//           body by default, what lies behind an open modal dialog included
//           (it is exposed again as the dialog shuts). The dialog and
//           duplicate checks are always the page's.
//   except  selectors left out of unnamed and unreasoned, added to the
//           standing exception: a phone list's line of no value ("No saved
//           setups in this view"), an option that is never a choice and
//           whose own words are why
// Exposed means in the accessibility tree: rendered (not display: none, not
// visibility: hidden) and not under [inert] or aria-hidden="true". An
// <option> is exposed as its <select> is. Opacity and size do not count: a
// visually hidden input is still read.
//
// A regression guard for the names and holds the markup carries, nothing
// more: it does not hear what a screen reader says, follow focus or measure
// contrast, and passing it is no evidence for the hands-on checks.
// Tests only: no app script loads this file.
globalThis.__a11yAudit = (options = {}) => {
  const STANDING = ['.phone-list-row[data-value=""]'];
  const except = [...STANDING, ...(options.except ?? [])];
  const CONTROLS = [
    'button', 'a[href]', 'input:not([type="hidden"])', 'select', 'textarea', 'summary',
    '[role="button"]', '[role="link"]', '[role="tab"]', '[role="menuitem"]',
    '[role="menuitemcheckbox"]', '[role="menuitemradio"]', '[role="option"]',
    '[role="checkbox"]', '[role="radio"]', '[role="switch"]', '[role="slider"]',
    '[role="textbox"]', '[role="combobox"]', '[contenteditable="true"]',
  ].join(', ');
  const DIALOGS = 'dialog[open], [role="dialog"], [role="alertdialog"]';

  // A display: contents element has no box, so checkVisibility() says false of
  // it though its children are drawn (the Models panel's row label): it is
  // rendered as its parent is.
  const rendered = (el) => el.checkVisibility({ visibilityProperty: true })
    || (getComputedStyle(el).display === 'contents' && !!el.parentElement && rendered(el.parentElement));
  const exposed = (el) => {
    if (el.closest('[inert], [aria-hidden="true"]')) return false;
    if (el.matches('option, optgroup')) {
      const select = el.closest('select');
      return !!select && rendered(select);
    }
    return rendered(el);
  };
  const tidy = (text) => (text ?? '').replace(/\s+/g, ' ').trim();
  // The text the ids of an aria-labelledby or aria-describedby name. A
  // referenced element is read even where it is not rendered.
  const referenced = (el, attr) => tidy((el.getAttribute(attr) ?? '').split(/\s+/).filter(Boolean)
    .map((id) => document.getElementById(id)?.textContent ?? '').join(' '));
  // The name an element takes from what it holds: its text, an image's alt,
  // an SVG's <title> and a descendant's aria-label, skipping what is hidden
  // from the tree.
  const within = (node, skip) => {
    if (node.nodeType === Node.TEXT_NODE) return node.textContent;
    if (node.nodeType !== Node.ELEMENT_NODE || node === skip) return '';
    if (node.getAttribute('aria-hidden') === 'true' || node.matches('script, style, template')) return '';
    if (!node.matches('option') && !rendered(node)) return '';
    const label = tidy(node.getAttribute('aria-label'));
    if (label) return label;
    if (node.matches('img, input[type="image"]')) return node.getAttribute('alt') ?? '';
    return [...node.childNodes].map((child) => within(child, skip)).join(' ');
  };
  const accessibleName = (el) => {
    const named = referenced(el, 'aria-labelledby') || tidy(el.getAttribute('aria-label'));
    if (named) return named;
    if (el.matches('input[type="button"], input[type="submit"], input[type="reset"]')) {
      // Submit and reset have the browser's own words where no value is set.
      return tidy(el.value) || (el.type === 'button' ? tidy(el.title) : el.type);
    }
    if (el.matches('input[type="image"]')) return tidy(el.alt) || tidy(el.title);
    if (el.matches('input, select, textarea')) {
      // A label that wraps its field is read without the field's own text.
      const labels = tidy([...(el.labels ?? [])].map((label) => within(label, el)).join(' '));
      return labels || tidy(el.title) || tidy(el.getAttribute('placeholder'));
    }
    return tidy([...el.childNodes].map((child) => within(child)).join(' ')) || tidy(el.title);
  };
  const name = (el) => {
    const id = el.id ? '#' + el.id : '';
    const cls = !id && typeof el.className === 'string' && el.className.trim()
      ? '.' + el.className.trim().split(/\s+/).slice(0, 2).join('.') : '';
    const label = accessibleName(el).slice(0, 32);
    return el.tagName.toLowerCase() + id + cls + (label ? ` "${label}"` : '');
  };

  let root = document.body;
  if (options.root) {
    root = typeof options.root === 'string' ? document.querySelector(options.root) : options.root;
    if (!root) throw new Error('__a11yAudit: no element for root ' + options.root);
  }
  const under = (selector) => [
    ...(root.matches(selector) ? [root] : []), ...root.querySelectorAll(selector),
  ].filter(exposed);
  const left = (el) => except.some((sel) => el.matches(sel));

  const unnamed = [], unreasoned = [], excepted = [], disabled = [];
  for (const el of under(CONTROLS)) {
    if (!accessibleName(el)) (left(el) ? excepted : unnamed).push(name(el));
    if (el.disabled === true) disabled.push(name(el));
  }
  for (const el of under('[aria-disabled="true"]')) {
    if (tidy(el.title) || referenced(el, 'aria-describedby')) continue;
    (left(el) ? excepted : unreasoned).push(name(el));
  }
  const dialogs = [...document.querySelectorAll(DIALOGS)].filter(exposed)
    .filter((el) => !referenced(el, 'aria-labelledby') && !tidy(el.getAttribute('aria-label')))
    .map((el) => name(el).replace(/ ".*$/, ''));
  const counts = new Map();
  for (const el of document.querySelectorAll('[id]')) {
    if (el.id) counts.set(el.id, (counts.get(el.id) ?? 0) + 1);
  }
  const duplicates = [...counts].filter(([, n]) => n > 1).map(([id, n]) => `#${id} x${n}`);
  return {
    ok: !unnamed.length && !duplicates.length && !unreasoned.length && !dialogs.length,
    unnamed, duplicates, unreasoned, dialogs, disabled, excepted,
  };
};

// The watch: every <dialog> audited as the smoke's own steps open and shut it,
// so a dialog-heavy smoke needs one step after its navigate and one at its end
// instead of a step per dialog:
//
//   {"inject": "a11y_audit.js"}, {"eval": "__a11yWatch()"}
//   ...the smoke...
//   {"expect": "__a11yCheck()"}
//
// __a11yWatch(options) audits each dialog, as `root`, once the task that
// called show() or showModal() has ended (what the opener wrote after
// showing it is in), and again as close() is called or Escape cancels it
// (what it had become). `options.except` is every audit's. Findings are kept
// once each, under the dialog's id; __a11yWatched() returns them as
// {unnamed, duplicates, unreasoned, dialogs, seen}, `seen` the ids of the
// dialogs audited. __a11yCheck() counts them with its own audit. A navigate
// drops the watch with the rest.
globalThis.__a11yWatch = (options = {}) => {
  if (globalThis.__a11yWatching) return true;
  const KEYS = ['unnamed', 'duplicates', 'unreasoned', 'dialogs'];
  const found = Object.fromEntries(KEYS.map((key) => [key, new Set()]));
  const seen = new Set();
  const audit = (dialog) => {
    if (!dialog.isConnected || !dialog.open) return;
    seen.add(dialog.id || 'dialog');
    const report = globalThis.__a11yAudit({ ...options, root: dialog });
    const where = dialog.id ? ` in #${dialog.id}` : '';
    for (const key of KEYS) {
      // A duplicate id or an unnamed dialog is the page's, whichever dialog was open.
      for (const entry of report[key]) found[key].add(key === 'duplicates' || key === 'dialogs' ? entry : entry + where);
    }
  };
  const proto = HTMLDialogElement.prototype;
  for (const method of ['show', 'showModal']) {
    const original = proto[method];
    proto[method] = function (...args) {
      const result = original.apply(this, args);
      queueMicrotask(() => audit(this));
      return result;
    };
  }
  const close = proto.close;
  proto.close = function (...args) {
    audit(this);
    return close.apply(this, args);
  };
  // Escape fires cancel and shuts the dialog without calling close().
  document.addEventListener('cancel', (event) => {
    if (event.target instanceof HTMLDialogElement) audit(event.target);
  }, true);
  globalThis.__a11yWatching = { found, seen };
  return true;
};

globalThis.__a11yWatched = () => {
  const { found, seen } = globalThis.__a11yWatching ?? { found: {}, seen: [] };
  return {
    ...Object.fromEntries(['unnamed', 'duplicates', 'unreasoned', 'dialogs'].map((key) => [key, [...(found[key] ?? [])]])),
    seen: [...seen],
  };
};

// The audit as one step: true when the page as it is now and everything the
// watch recorded are clean; otherwise it throws with the failed lists, which
// the driver prints under the failed expect.
globalThis.__a11yCheck = (options = {}) => {
  const report = globalThis.__a11yAudit(options);
  const watched = globalThis.__a11yWatched();
  const failed = ['unnamed', 'duplicates', 'unreasoned', 'dialogs']
    .map((key) => [key, [...new Set([...report[key], ...watched[key]])]])
    .filter(([, entries]) => entries.length).map(([key, entries]) => `${key}: ${entries.join('; ')}`);
  if (failed.length) throw new Error('__a11yCheck: ' + failed.join(' | '));
  return true;
};
