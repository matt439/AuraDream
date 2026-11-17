// Full-screen phone focus for the assistant; desktop remains a nonmodal aside.
// Declarations only: app.js installs the handlers once all scripts are ready.
'use strict';

let assistantPhoneHeld = false;
let assistantPhoneRole = null;
let assistantPhoneModal = null;
let assistantPhoneLastFocus = null;
let assistantPhoneWriting = false;
const assistantPhoneInert = new Map();

function assistantPhoneActive() {
  return isPhone() && assistantSidebarOpen();
}

function assistantPhoneFocusable() {
  return [...els.assistantSidebar.querySelectorAll(
    'button, a[href], input:not([type="hidden"]), select, textarea, summary, [tabindex]',
  )].filter((control) => control.tabIndex >= 0 && shortcutFocusReady(control));
}

function assistantPhoneFocusInside() {
  const previous = assistantPhoneLastFocus;
  const target = previous && els.assistantSidebar.contains(previous) && shortcutFocusReady(previous)
    ? previous : els.assistantClose;
  target.focus({ preventScroll: true });
}

function syncAssistantPhoneFocus() {
  const sidebar = els.assistantSidebar;
  if (!sidebar || assistantPhoneWriting) return;
  if (!assistantPhoneActive()) {
    if (!assistantPhoneHeld) return;
    for (const [surface, inert] of assistantPhoneInert) surface.inert = inert;
    assistantPhoneInert.clear();
    for (const [name, value] of [['role', assistantPhoneRole], ['aria-modal', assistantPhoneModal]]) {
      if (value === null) sidebar.removeAttribute(name);
      else sidebar.setAttribute(name, value);
    }
    assistantPhoneHeld = false;
    assistantPhoneLastFocus = null;
    return;
  }
  if (!assistantPhoneHeld) {
    assistantPhoneRole = sidebar.getAttribute('role');
    assistantPhoneModal = sidebar.getAttribute('aria-modal');
    assistantPhoneHeld = true;
  }
  sidebar.setAttribute('role', 'dialog');
  sidebar.setAttribute('aria-modal', 'true');
  // Native dialogs establish their own modal background. Explicitly making
  // them inert would also prevent their controls from being used when opened.
  for (const surface of document.body.children) {
    if (surface === sidebar || ['DIALOG', 'SCRIPT', 'LINK', 'STYLE'].includes(surface.tagName)) continue;
    if (!assistantPhoneInert.has(surface)) assistantPhoneInert.set(surface, surface.inert);
    surface.inert = true;
  }
  if (!document.querySelector('dialog[open]') && !sidebar.contains(document.activeElement)) {
    assistantPhoneFocusInside();
  }
}

function assistantPhoneWrite(write) {
  if (!assistantPhoneHeld || assistantPhoneWriting) return write();
  // The undo-aware writer must briefly focus the covered editor. This is a
  // synchronous application from the user's press, never an open background
  // interaction or a promise. Reapply the hold before returning to the caller.
  assistantPhoneWriting = true;
  for (const [surface, inert] of assistantPhoneInert) surface.inert = inert;
  try {
    return write();
  } finally {
    assistantPhoneWriting = false;
    syncAssistantPhoneFocus();
  }
}

function assistantPhoneKey(event) {
  if (assistantPhoneWriting || !assistantPhoneActive() || document.querySelector('dialog[open]')
      || keyboardEventHeld(event) || event.key !== 'Tab'
      || event.ctrlKey || event.metaKey || event.altKey) return;
  const controls = assistantPhoneFocusable();
  const first = controls[0], last = controls.at(-1);
  if (!first) {
    event.preventDefault();
    els.assistantClose.focus({ preventScroll: true });
    return;
  }
  const active = document.activeElement;
  if (!els.assistantSidebar.contains(active) || (event.shiftKey ? active === first : active === last)) {
    event.preventDefault();
    (event.shiftKey ? last : first).focus({ preventScroll: true });
  }
}

function initAssistantPhone() {
  document.addEventListener('keydown', assistantPhoneKey, true);
  document.addEventListener('focusin', (event) => {
    if (assistantPhoneWriting || !assistantPhoneActive() || document.querySelector('dialog[open]')) return;
    if (els.assistantSidebar.contains(event.target)) assistantPhoneLastFocus = event.target;
    else assistantPhoneFocusInside();
  });
  // A late-created workspace surface must be covered too. Dialogs remain
  // exempt, and no attributes are observed, so our inert writes cannot loop.
  new MutationObserver(syncAssistantPhoneFocus).observe(document.body, { childList: true });
  syncAssistantPhoneFocus();
}
