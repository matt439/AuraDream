// The phone audit: what a finger meets on the page as it is drawn now.
// Loaded into the page by a smoke's {"inject": "phone_audit.js"} step (again
// after every navigate), then read from `log` and `expect` steps:
//
//   {"log": "__phoneAudit()"}                       the whole report
//   {"expect": "__phoneAudit().overflow === 0"}     one of its checks
//   {"expect": "__phoneAudit({root: '#lightbox'}).small.length === 0"}
//
// __phoneAudit(options) returns
//   small     visible controls whose press box is under min x min (44 px): the
//             box, grown by an absolutely placed ::before/::after that reaches
//             past it, and for a checkbox or radio its label's box
//   offside   visible controls whose box leaves 0..innerWidth (a row that
//             runs off the screen, scrolled sideways or not)
//   dialogs   open dialogs and menus whose box leaves 0..innerWidth or
//             0..innerHeight (a footer past the foot of the screen)
//   overflow  how far the document scrolls sideways (0 when it does not)
//   excepted  controls left out of `small` by the exception list
//   ok        true when small, offside and dialogs are empty and overflow is 0
// Each entry names the element (tag#id.class "label") and its press box, w x h
// at left,top in CSS px.
//
// Options:
//   root    a selector or element whose controls are audited; by default the
//           open modal dialogs when there are any (nothing else can be
//           pressed), else the body. The dialog and overflow checks are always
//           the page's.
//   min     the smallest press box, 44 (Apple's HIG; WCAG 2.5.5)
//   except  selectors left out of `small`, added to the standing exceptions:
//           inline citation links in prose (.lb-link, .analysis-link), the
//           one documented exception to the 44 px rule
// Visible means rendered with a box, not visibility: hidden, not under
// opacity 0 or [inert], and larger than 1x1 (a visually hidden input).
// Tests only: no app script loads this file.
globalThis.__phoneAudit = (options = {}) => {
  const min = options.min ?? 44;
  const STANDING = ['.lb-link', '.analysis-link'];
  const except = [...STANDING, ...(options.except ?? [])];
  const CONTROLS = [
    'button', 'a[href]', 'input:not([type="hidden"])', 'select', 'textarea', 'summary',
    '[role="button"]', '[role="link"]', '[role="tab"]', '[role="menuitem"]',
    '[role="menuitemcheckbox"]', '[role="menuitemradio"]', '[role="option"]',
    '[role="checkbox"]', '[role="radio"]', '[role="switch"]', '[contenteditable="true"]',
  ].join(', ');
  const SURFACES = 'dialog[open], [role="dialog"], [role="alertdialog"], [role="menu"]';
  const vw = innerWidth, vh = innerHeight, slack = 0.5;

  const visible = (el) => {
    if (!el.getClientRects().length || el.closest('[inert]')) return false;
    if (!el.checkVisibility({ opacityProperty: true, visibilityProperty: true })) return false;
    const r = el.getBoundingClientRect();
    return r.width > 1 && r.height > 1;
  };
  const px = (v) => (v.endsWith('px') ? parseFloat(v) : NaN);
  // An absolutely placed pseudo-element reaching past its element: it is
  // pressed as the element, so it counts toward the press box. Its four
  // offsets read as used values, from the element's padding box.
  const reach = (el, r) => {
    const cs = getComputedStyle(el);
    if (cs.position === 'static') return [];
    const pad = {
      left: r.left + parseFloat(cs.borderLeftWidth), top: r.top + parseFloat(cs.borderTopWidth),
      right: r.right - parseFloat(cs.borderRightWidth), bottom: r.bottom - parseFloat(cs.borderBottomWidth),
    };
    return ['::before', '::after'].flatMap((which) => {
      const p = getComputedStyle(el, which);
      if (p.content === 'none' || p.content === 'normal' || p.display === 'none'
          || p.position !== 'absolute' || p.pointerEvents === 'none') return [];
      const [l, t, rt, b] = [p.left, p.top, p.right, p.bottom].map(px);
      if ([l, t, rt, b].some(Number.isNaN)) return [];
      return [{ left: pad.left + l, top: pad.top + t, right: pad.right - rt, bottom: pad.bottom - b }];
    });
  };
  const union = (boxes) => boxes.reduce((u, b) => ({
    left: Math.min(u.left, b.left), top: Math.min(u.top, b.top),
    right: Math.max(u.right, b.right), bottom: Math.max(u.bottom, b.bottom),
  }));
  const pressBox = (el) => {
    const r = el.getBoundingClientRect();
    const boxes = [r, ...reach(el, r)];
    if (el.matches('input[type="checkbox"], input[type="radio"]')) {
      for (const label of el.labels ?? []) if (visible(label)) boxes.push(label.getBoundingClientRect());
    }
    return union(boxes);
  };
  const name = (el) => {
    const id = el.id ? '#' + el.id : '';
    const cls = !id && typeof el.className === 'string' && el.className.trim()
      ? '.' + el.className.trim().split(/\s+/).slice(0, 2).join('.') : '';
    // A field is named by its label, never by what was typed into it.
    const field = el.matches('input, select, textarea');
    const label = (el.getAttribute('aria-label') || (field ? el.labels?.[0]?.textContent : el.textContent)
      || el.title || (field ? el.placeholder || el.name : '') || '')
      .replace(/\s+/g, ' ').trim().slice(0, 32);
    return el.tagName.toLowerCase() + id + cls + (label ? ` "${label}"` : '');
  };
  const at = (b) => `${Math.round(b.right - b.left)}x${Math.round(b.bottom - b.top)}`
    + ` at ${Math.round(b.left)},${Math.round(b.top)}`;

  let roots;
  if (options.root) {
    const root = typeof options.root === 'string' ? document.querySelector(options.root) : options.root;
    if (!root) throw new Error('__phoneAudit: no element for root ' + options.root);
    roots = [root];
  } else {
    const modal = [...document.querySelectorAll('dialog:modal')];
    roots = modal.length ? modal : [document.body];
  }
  const controls = [...new Set(roots.flatMap((root) => [
    ...(root.matches(CONTROLS) ? [root] : []), ...root.querySelectorAll(CONTROLS),
  ]))].filter(visible);

  const small = [], excepted = [], offside = [];
  for (const el of controls) {
    const box = pressBox(el);
    if (box.right - box.left < min - slack || box.bottom - box.top < min - slack) {
      (except.some((sel) => el.matches(sel)) ? excepted : small).push(`${name(el)} ${at(box)}`);
    }
    const r = el.getBoundingClientRect();
    if (r.left < -slack || r.right > vw + slack) offside.push(`${name(el)} ${at(r)}`);
  }
  const dialogs = [...document.querySelectorAll(SURFACES)].filter(visible).flatMap((el) => {
    const r = el.getBoundingClientRect();
    return r.left < -slack || r.top < -slack || r.right > vw + slack || r.bottom > vh + slack
      ? [`${name(el).replace(/ ".*$/, '')} ${at(r)}`] : [];
  });
  const doc = document.documentElement;
  const overflow = Math.max(0, doc.scrollWidth - doc.clientWidth);
  return {
    ok: !small.length && !offside.length && !dialogs.length && !overflow,
    small, offside, dialogs, overflow, excepted,
  };
};
