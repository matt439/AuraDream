// Drive a headless Chromium-family browser over the DevTools protocol with
// nothing but Node's own fetch and WebSocket - no npm install.
//
//     node tools/browser.mjs tools/smoke.json
//     LOCAL_DIFFUSION_BROWSER="C:/Program Files/Google/Chrome/Application/chrome.exe" node tools/browser.mjs steps.json
//
// The script is a JSON list of steps, run in order:
//   {"navigate": url}            load a page (the cache is disabled)
//   {"wait": ms}                 pause
//   {"eval": js}                 run JavaScript in the page (promises awaited)
//   {"log": js}                  run JavaScript and print its value
//   {"expect": js}               run JavaScript; a falsy value fails the run
//   {"key": "1"}                 press a key (single characters and names like "Escape";
//                                Escape also closes a modal <dialog>, as a real press does)
//   {"keycombo": ["Control", "Enter"]}   press a key with a modifier held
//   {"type": "text"}             type text into the focused element
//   {"wheel": [x, y, deltaY]}    scroll the wheel at a point
//   {"hover": [x, y]}            move the mouse there (tooltips, hover states)
//   {"click": [x, y]}            click a point; {"dblclick": [x, y]} double-clicks;
//                                {"drag": [x1, y1, x2, y2]} drags
//   {"hover": "selector"}, {"click": "selector"}   the same at the middle of that
//                                element's box as it is at that moment, wherever the
//                                layout has put it
//   {"viewport": [w, h, dpr]}    resize the page (dpr optional, default 1) and wait for layout
//   {"media": {"prefers-reduced-motion": "reduce"}}   emulate media features ("" for
//                                one drops its override and the host's setting is back)
//   {"shot": "file.png"}         screenshot the viewport; {"fullshot": "file.png"} the
//                                whole scrollable page; {"elshot": ["selector", "file.png"]}
//                                one element's box, scrolled into view
//
// The page opens at 1400x900, where the smokes are written. Two settings move it:
//   LOCAL_DIFFUSION_VIEWPORT=2560x1305   (a maximized QHD browser at 100% scaling)
//   LOCAL_DIFFUSION_DPR=1.25             (with 2048x1040, QHD at 125% scaling)
// and LOCAL_DIFFUSION_PORT_OFFSET=N adds N to every 127.0.0.1:<port> and
// localhost:<port> in the steps, so a second checkout can run the same smokes
// against fake servers of its own without meeting the first one's.
//
// Page-side names (`els`, `lbView`, `sheetRecords`, ...) are the app's own
// top-level bindings, reachable from `eval` because app.js is a classic script.
// An `eval` whose last expression is a promise is awaited - handy for a fetch,
// but `startJob(...)` resolves when the job *ends* - or, for one queued behind
// a job the page watches, once the server has taken it - so to look at a job
// while it runs write `void startJob(...)`.
// Exit status is 1 when any `expect` failed, so this can gate a check, and 2
// for a script refused before the browser starts - among them an expect that
// joins an async helper (`__until`, or a `window.__x = async` of the file's
// own) to more with && or ||, which would never wait: put the whole
// condition in the predicate.
import { spawn } from 'node:child_process';
import { existsSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';

const CANDIDATES = [
  process.env.LOCAL_DIFFUSION_BROWSER,
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/microsoft-edge',
].filter(Boolean);
const browser = CANDIDATES.find((p) => existsSync(p));
if (!browser) {
  console.error('no Edge or Chrome found; set LOCAL_DIFFUSION_BROWSER to the executable');
  process.exit(2);
}
const file = process.argv[2];
if (!file) { console.error('usage: node tools/browser.mjs <steps.json>'); process.exit(2); }
const OFFSET = Number(process.env.LOCAL_DIFFUSION_PORT_OFFSET || 0);
let text = readFileSync(file, 'utf8');
if (OFFSET) {
  text = text.replace(/\b(127\.0\.0\.1|localhost):(\d{2,5})\b/g, (_, host, port) => `${host}:${Number(port) + OFFSET}`);
}
const steps = JSON.parse(text);

// An expect that joins an async helper's call to more with && or || never
// waits: the promise is truthy, so the rest is evaluated at once, and only a
// promise that is the whole value is awaited. `__until(() => A) && B` read B
// a moment after the step began and failed whenever the page was slower
// than that. Refused before the browser starts: the whole condition goes in
// the predicate. The helpers are the ones the file's own evals define as
// `window.__name = async`.
const asyncHelpers = new Set(['__until']);
for (const step of steps) {
  for (const m of String(step.eval ?? '').matchAll(/window\.(__\w+)\s*=\s*async\b/g)) asyncHelpers.add(m[1]);
}
const closing = (s, open) => {
  let depth = 0;
  let quote = null;
  for (let j = open; j < s.length; j++) {
    const c = s[j];
    if (quote) { if (c === '\\') j++; else if (c === quote) quote = null; continue; }
    if (c === '\'' || c === '"' || c === '`') quote = c;
    else if (c === '(') depth++;
    else if (c === ')' && --depth === 0) return j;
  }
  return -1;
};
const unawaited = steps.flatMap((step, n) => {
  const e = step.expect;
  if (typeof e !== 'string') return [];
  return [...e.matchAll(/(?<![\w.$])(__\w+)\(/g)].filter((m) => {
    if (!asyncHelpers.has(m[1])) return false;
    const before = e.slice(0, m.index).trimEnd();
    if (before.endsWith('await')) return false;
    const after = e.slice(closing(e, m.index + m[0].length - 1) + 1).trimStart();
    return /^(&&|\|\|)/.test(after) || /(&&|\|\||!)$/.test(before);
  }).map((m) => `step ${n + 1}: ${m[1]}(...) joined with && or || is never awaited`);
});
if (unawaited.length) {
  console.error(unawaited.join('\n'));
  process.exit(2);
}

const [VW, VH] = (process.env.LOCAL_DIFFUSION_VIEWPORT || '1400x900').split('x').map(Number);
const DPR = Number(process.env.LOCAL_DIFFUSION_DPR || 1);
if (!(VW > 0 && VH > 0 && DPR > 0)) {
  console.error('LOCAL_DIFFUSION_VIEWPORT is WIDTHxHEIGHT and LOCAL_DIFFUSION_DPR a positive number');
  process.exit(2);
}

// A profile of this run's own. A shared one is held by whichever Edge is
// still alive from a run that crashed, and the next launch then hands its
// command line to that instance and exits without ever opening the port -
// which reads as "the browser did not come up" for every run after it.
const PROFILE = `${tmpdir()}/local-diffusion-browser-${process.pid}`;
// Port 0: the browser picks a free one and writes it to DevToolsActivePort in
// the profile. A port of our own choosing could be another run's - two runs on
// one port drive each other's pages.
const proc = spawn(browser, [
  '--headless=new', '--remote-debugging-port=0', `--window-size=${VW},${VH}`,
  '--no-first-run', '--no-default-browser-check', '--disable-sync', '--disable-gpu',
  `--user-data-dir=${PROFILE}`, 'about:blank',
], { stdio: 'ignore' });
const cleanup = () => { try { rmSync(PROFILE, { recursive: true, force: true }); } catch { /* locked */ } };

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;

async function target() {
  for (let i = 0; i < 75; i++) {
    try {
      const port = Number(readFileSync(`${PROFILE}/DevToolsActivePort`, 'utf8').split('\n')[0]);
      const list = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
      // The blank page asked for, not whatever else a fresh profile opens
      // first - Edge puts a sync prompt ahead of it.
      const pages = list.filter((t) => t.type === 'page');
      const page = pages.find((t) => t.url === 'about:blank') ?? pages[0];
      if (page) return page.webSocketDebuggerUrl;
    } catch { /* not up yet */ }
    await sleep(200);
  }
  proc.kill();
  cleanup();
  throw new Error('the browser did not come up');
}

const ws = new WebSocket(await target());
await new Promise((r) => { ws.onopen = r; });
let id = 0;
const pending = new Map();
ws.onmessage = (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
};
const send = (method, params = {}) => new Promise((resolve) => {
  const n = ++id;
  pending.set(n, resolve);
  ws.send(JSON.stringify({ id: n, method, params }));
});

const viewport = (width, height, deviceScaleFactor = 1) =>
  send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor, mobile: false });

await send('Page.enable');
await send('Network.enable');
await send('Network.setCacheDisabled', { cacheDisabled: true });
await send('Runtime.enable');
await viewport(VW, VH, DPR);

async function evaluate(expression) {
  const res = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  const details = res.result?.exceptionDetails;
  if (details) console.log('  ! ' + (details.exception?.description ?? details.text));
  return res.result?.result?.value;
}
const mouse = (type, x, y, extra = {}) =>
  send('Input.dispatchMouseEvent', { type, x, y, button: 'left', clickCount: 1, ...extra });
// A point, or a selector standing for the middle of its element's box.
async function point(at) {
  if (Array.isArray(at)) return at;
  const xy = await evaluate(`(() => { const e = document.querySelector(${JSON.stringify(at)}); if (!e) return null; const r = e.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; })()`);
  if (!xy) console.log('  ! no element for ' + at);
  return xy ?? [0, 0];
}
// Escape carries its key code, as a real press does: a modal <dialog>
// closes on that, not on the key's name alone. A plain Enter stays a bare
// key name (the smokes' chip and autocomplete steps are written against
// that); in a keycombo, Enter is sent as the real key, text and all.
const ESCAPE = { code: 'Escape', windowsVirtualKeyCode: 27 };
const COMBO_KEYS = { Enter: { code: 'Enter', windowsVirtualKeyCode: 13, text: '\r' }, Escape: ESCAPE };

for (const step of steps) {
  if (step.navigate) { await send('Page.navigate', { url: step.navigate }); await sleep(800); }
  else if (step.wait) await sleep(step.wait);
  else if (step.eval) await evaluate(step.eval);
  else if (step.log) console.log('  ' + step.log + ' => ' + JSON.stringify(await evaluate(step.log)));
  else if (step.expect) {
    const value = await evaluate(step.expect);
    if (!value) failures += 1;
    console.log(`  ${value ? 'ok  ' : 'FAIL'} ${step.expect}`);
  } else if (step.key) {
    const text = step.key.length === 1 ? step.key : undefined;
    const code = step.key === 'Escape' ? ESCAPE : {};
    await send('Input.dispatchKeyEvent', { type: 'keyDown', key: step.key, text, ...code });
    await send('Input.dispatchKeyEvent', { type: 'keyUp', key: step.key, ...code });
    await sleep(100);
  } else if (step.keycombo) {
    const [mod, key] = step.keycombo;
    const bit = { Alt: 1, Control: 2, Meta: 4, Shift: 8 }[mod] || 0;
    const code = COMBO_KEYS[key] ?? {};
    await send('Input.dispatchKeyEvent', { type: 'rawKeyDown', key: mod, modifiers: bit });
    await send('Input.dispatchKeyEvent', { type: 'keyDown', key, modifiers: bit, ...code });
    await send('Input.dispatchKeyEvent', { type: 'keyUp', key, modifiers: bit, ...code, text: undefined });
    await send('Input.dispatchKeyEvent', { type: 'keyUp', key: mod, modifiers: 0 });
    await sleep(100);
  } else if (step.type) {
    await send('Input.insertText', { text: step.type });
    await sleep(100);
  } else if (step.wheel) {
    const [x, y, deltaY] = step.wheel;
    await mouse('mouseMoved', x, y);
    await send('Input.dispatchMouseEvent', { type: 'mouseWheel', x, y, deltaX: 0, deltaY });
    await sleep(100);
  } else if (step.hover) {
    const [x, y] = await point(step.hover);
    await mouse('mouseMoved', x, y);
    await sleep(400);
  } else if (step.click) {
    const [x, y] = await point(step.click);
    await mouse('mouseMoved', x, y); await mouse('mousePressed', x, y); await mouse('mouseReleased', x, y);
    await sleep(150);
  } else if (step.dblclick) {
    const [x, y] = step.dblclick;
    await mouse('mouseMoved', x, y);
    for (const clickCount of [1, 2]) {
      await mouse('mousePressed', x, y, { clickCount }); await mouse('mouseReleased', x, y, { clickCount });
    }
    await sleep(150);
  } else if (step.drag) {
    const [x1, y1, x2, y2] = step.drag;
    await mouse('mouseMoved', x1, y1); await mouse('mousePressed', x1, y1);
    for (let i = 1; i <= 5; i++) await mouse('mouseMoved', x1 + (x2 - x1) * i / 5, y1 + (y2 - y1) * i / 5, { buttons: 1 });
    await mouse('mouseReleased', x2, y2);
    await sleep(150);
  } else if (step.viewport) {
    const [w, h, dpr = 1] = step.viewport;
    await viewport(w, h, dpr);
    await sleep(300);
  } else if (step.media) {
    const features = Object.entries(step.media).map(([name, value]) => ({ name, value }));
    await send('Emulation.setEmulatedMedia', { features });
  } else if (step.shot) {
    const res = await send('Page.captureScreenshot', { format: 'png' });
    writeFileSync(step.shot, Buffer.from(res.result.data, 'base64'));
    console.log('  wrote ' + step.shot);
  } else if (step.fullshot) {
    const res = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
    writeFileSync(step.fullshot, Buffer.from(res.result.data, 'base64'));
    console.log('  wrote ' + step.fullshot);
  } else if (step.elshot) {
    const [sel, out] = step.elshot;
    const box = await evaluate(`(() => { const e = document.querySelector(${JSON.stringify(sel)}); if (!e) return null; e.scrollIntoView({block: 'nearest'}); const r = e.getBoundingClientRect(); return {x: r.x + scrollX, y: r.y + scrollY, width: r.width, height: r.height}; })()`);
    if (!box || !box.width || !box.height) console.log('  ! elshot: no visible element for ' + sel);
    else {
      const res = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, clip: { ...box, scale: 1 } });
      writeFileSync(out, Buffer.from(res.result.data, 'base64'));
      console.log('  wrote ' + out + ' (' + Math.round(box.width) + 'x' + Math.round(box.height) + ')');
    }
  }
}

ws.close();
proc.kill();
await sleep(500);
cleanup();
if (failures) console.log(`${failures} expectation(s) failed`);
process.exit(failures ? 1 : 0);
