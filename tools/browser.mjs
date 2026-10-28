// Drive a headless Chromium-family browser over the DevTools protocol with
// nothing but Node's own fetch and WebSocket - no npm install.
//
//     node tools/browser.mjs tools/smoke.json
//     AURADREAM_BROWSER="C:/Program Files/Google/Chrome/Application/chrome.exe" node tools/browser.mjs steps.json
//
// The script is a JSON list of steps, run in order:
//   {"navigate": url}            load a page (the cache is disabled)
//   {"wait": ms}                 pause
//   {"eval": js}                 run JavaScript in the page (promises awaited)
//   {"log": js}                  run JavaScript and print its value
//   {"expect": js}               run JavaScript; a falsy value fails the run
//   {"expect": js, "fatal": true}  prerequisite: stop after a failing expectation
//                                before later steps; return true from async setup
//   {"key": "1"}                 press a key (single characters and names like "Escape";
//                                Escape also closes a modal <dialog>, as a real press does;
//                                with "native": true an ArrowDown/ArrowUp carries its key
//                                code, so a focused <select> moves as on a real press, and
//                                an Enter its code and text, so a form submits and a button
//                                is pressed)
//                                "repeat": true sends a held key's native repeated keydown
//   {"keycombo": ["Control", "Enter"]}   press a key with a modifier held
//   {"keycombo": ["Control", "Shift", "Enter"]}   several unique modifiers, then the key
//   {"type": "text"}             type text into the focused element
//   {"wheel": [x, y, deltaY]}    scroll the wheel at a point
//   {"wheel": ["selector", deltaY]}   the same at the middle of that element's box
//   {"hover": [x, y]}            move the mouse there (tooltips, hover states)
//   {"click": [x, y]}            click a point; {"dblclick": [x, y]} double-clicks;
//                                {"drag": [x1, y1, x2, y2]} drags
//   {"rightclick": [x, y]}       a right-button press there (the page's contextmenu);
//                                a selector as for click; "shift": true holds Shift
//   {"hover": "selector"}, {"click": "selector"}, {"dblclick": "selector"}   the same at the middle of that
//                                element's box as it is at that moment, wherever the
//                                layout has put it
//   {"viewport": [w, h, dpr]}    resize the page (dpr optional, default 1) and wait for layout
//   {"media": {"prefers-reduced-motion": "reduce"}}   emulate media features ("" for
//                                one drops its override and the host's setting is back)
//   {"shot": "file.png"}         screenshot the viewport; {"fullshot": "file.png"} the
//                                whole scrollable page; {"elshot": ["selector", "file.png"]}
//                                one element's box, scrolled into view
// Screenshot names must be portable relative paths, under
// AURADREAM_SCREENSHOT_DIR (a fresh reported temp directory if unset).
//
// The page opens at 1400x900, where the smokes are written. Two settings move it:
//   AURADREAM_VIEWPORT=2560x1305   (a maximized QHD browser at 100% scaling)
//   AURADREAM_DPR=1.25             (with 2048x1040, QHD at 125% scaling)
// and AURADREAM_PORT_OFFSET=N adds N to every 127.0.0.1:<port> and
// localhost:<port> in the steps, so a second checkout can run the same smokes
// against fake servers of its own without meeting the first one's.
//
// Page-side names (`els`, `lbView`, `sheetRecords`, ...) are the app's own
// top-level bindings, reachable from `eval` because app.js is a classic script.
// An `eval` whose last expression is a promise is awaited - handy for a fetch,
// but `startJob(...)` resolves when the job *ends* - or, for one queued behind
// a job the page watches, once the server has taken it - so to look at a job
// while it runs write `void startJob(...)`.
// Exit status is 1 when any `expect` failed - an expect that throws, or whose
// promise rejects, has failed - or the browser left a call unanswered for
// five minutes (the step it was on is printed), so this can gate a check, and 2
// for a script refused before the browser starts - among them an expect that
// joins an async helper (`__until`, or a `window.__x = async` of the file's
// own) to more with && or ||, which would never wait: put the whole
// condition in the predicate.
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, readdirSync, readFileSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { screenshotRoot, writeScreenshot } from './browser-output.mjs';

// Fail before even looking for a browser or preparing script/output paths.
// Check capabilities rather than guessing a supported Node version.
const missingApis = ['fetch', 'WebSocket'].filter((name) => typeof globalThis[name] !== 'function');
if (missingApis.length) {
  console.error('browser.mjs requires callable native Node APIs: ' + missingApis.join(', ')
    + '. Use a Node runtime providing both fetch and WebSocket.');
  process.exit(2);
}

const file = process.argv[2];
if (!file) { console.error('usage: node tools/browser.mjs <steps.json>'); process.exit(2); }
const OFFSET = Number(process.env.AURADREAM_PORT_OFFSET || 0);
if (!Number.isSafeInteger(OFFSET)) {
  console.error('AURADREAM_PORT_OFFSET must be a finite safe integer');
  process.exit(2);
}
let text;
try {
  text = readFileSync(file, 'utf8');
} catch {
  console.error('could not read browser steps file; check its path and read access');
  process.exit(2);
}
let invalidLocalPort = false;
text = text.replace(/\b(127\.0\.0\.1|localhost):(\d{2,5})\b/g, (match, host, port) => {
  const effective = Number(port) + OFFSET;
  if (!Number.isSafeInteger(effective)
      || effective < 1 || effective > 65535) {
    invalidLocalPort = true;
    return match;
  }
  // A zero offset has always left the fixture text unchanged, including
  // leading-zero ports and numeric literals embedded in eval expressions.
  return OFFSET ? `${host}:${effective}` : match;
});
if (invalidLocalPort) {
  console.error('effective local ports must be integers in 1..65535 after AURADREAM_PORT_OFFSET');
  process.exit(2);
}
let steps;
try {
  steps = JSON.parse(text);
} catch (error) {
  if (!(error instanceof SyntaxError)) throw error;
  console.error('browser steps file must contain valid JSON');
  process.exit(2);
}
// A non-null primitive or nested-array entry otherwise has no action properties and
// silently disappears from the run. Validate the whole list before any
// browser lookup or output/profile preparation; action metadata stays with
// the existing per-action rules.
if (!Array.isArray(steps)) {
  console.error('browser steps must be a JSON array');
  process.exit(2);
}
const invalidSteps = steps.flatMap((step, index) =>
  step === null || typeof step !== 'object' || Array.isArray(step)
    ? [`step ${index + 1}: step must be a JSON object`] : []);
if (invalidSteps.length) {
  console.error(invalidSteps.join('\n'));
  process.exit(2);
}
// A malformed resize otherwise throws only after the browser and any earlier
// steps have run. Refuse every declared viewport before browser discovery or
// output/profile preparation. Keep the documented two/three-number form and
// the runtime's omitted-DPR default; unrelated step/container policy is unchanged.
const invalidViewports = Array.isArray(steps) ? steps.flatMap((step, index) => {
  if (step === null || typeof step !== 'object' || !Object.hasOwn(step, 'viewport')) return [];
  const values = step.viewport;
  const valid = Array.isArray(values) && (values.length === 2 || values.length === 3)
    && Number.isInteger(values[0]) && values[0] >= 0 && values[0] <= 10000000
    && Number.isInteger(values[1]) && values[1] >= 0 && values[1] <= 10000000
    && (values.length === 2 || (Number.isFinite(values[2]) && values[2] >= 0));
  return valid ? [] : [`step ${index + 1}: viewport needs [width, height] or [width, height, dpr] with integer dimensions from 0 to 10000000 and a finite nonnegative dpr`];
}) : [];
if (invalidViewports.length) {
  console.error(invalidViewports.join('\n'));
  process.exit(2);
}

// Optional fatal metadata belongs only to an unambiguous expectation. Validate
// before preparing screenshot roots, profiles or browser processes. Unrelated
// malformed step/container handling remains with the existing preflight.
const invalidFatal = Array.isArray(steps) ? steps.flatMap((step, index) => {
  if (step === null || typeof step !== 'object' || !Object.hasOwn(step, 'fatal')) return [];
  const valid = typeof step.fatal === 'boolean'
    && typeof step.expect === 'string' && !!step.expect.trim()
    && Object.keys(step).every((key) => key === 'expect' || key === 'fatal');
  return valid ? [] : [`step ${index + 1}: fatal must be a boolean on an expect-only step with a nonempty expression`];
}) : [];
if (invalidFatal.length) {
  console.error(invalidFatal.join('\n'));
  process.exit(2);
}
const KEY_MODIFIERS = { Alt: 1, Control: 2, Meta: 4, Shift: 8 };
const invalidCombos = steps.flatMap((step, index) => {
  if (!Object.hasOwn(step, 'keycombo')) return [];
  const combo = step.keycombo;
  if (!Array.isArray(combo) || combo.length < 2
      || typeof combo.at(-1) !== 'string' || !combo.at(-1).length) {
    return [`step ${index + 1}: keycombo needs modifiers followed by a key`];
  }
  const modifiers = combo.slice(0, -1);
  return modifiers.some((modifier) => typeof modifier !== 'string' || !Object.hasOwn(KEY_MODIFIERS, modifier))
      || new Set(modifiers).size !== modifiers.length
    ? [`step ${index + 1}: keycombo modifiers must be known and unique`] : [];
});
if (invalidCombos.length) {
  console.error(invalidCombos.join('\n'));
  process.exit(2);
}

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

// Startup uses positive geometry; viewport steps retain their zero resets.
// Match CDP's integer size bounds before browser flags or metrics are applied.
const STARTUP_SIZE = (process.env.AURADREAM_VIEWPORT || '1400x900').split('x').map(Number);
const [VW, VH] = STARTUP_SIZE;
const DPR = Number(process.env.AURADREAM_DPR || 1);
if (STARTUP_SIZE.length !== 2
    || !Number.isInteger(VW) || VW < 1 || VW > 10000000
    || !Number.isInteger(VH) || VH < 1 || VH > 10000000
    || !Number.isFinite(DPR) || DPR <= 0) {
  console.error('AURADREAM_VIEWPORT needs WIDTHxHEIGHT with integer dimensions in 1..10000000; AURADREAM_DPR needs a finite positive number');
  process.exit(2);
}

// Preparation starts after the existing script and startup checks.
// Keep discovery, output behavior and runtime launch policy unchanged.
const CANDIDATES = [
  process.env.AURADREAM_BROWSER,
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/microsoft-edge',
].filter(Boolean);
const browser = CANDIDATES.find((p) => existsSync(p));
if (!browser) {
  console.error('no Edge or Chrome found; set AURADREAM_BROWSER to the executable');
  process.exit(2);
}
let screenshots;
try {
  screenshots = screenshotRoot(steps);
} catch (error) {
  console.error(error.message);
  process.exit(2);
}
if (screenshots) console.log('screenshots: ' + screenshots);

// A profile of this run's own. A shared one is held by whichever Edge is
// still alive from a run that crashed, and the next launch then hands its
// command line to that instance and exits without ever opening the port -
// which reads as "the browser did not come up" for every run after it.
// Named for this run alone, not its pid: a profile Edge still held at the
// end is left behind, and Windows hands the pid out again - a later run on
// the same port then opened with that run's localStorage (smoke-model-switch
// found another run's favourites, under eight runs at once).
const PREFIX = 'auradream-browser-';
const PROFILE = `${tmpdir()}/${PREFIX}${process.pid}-${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
// Every profile is removed on the way out, but a run killed outright (a
// closed terminal, Task Manager) never gets there, and those piled up in
// Windows' Temp to 49 GiB. One untouched for six hours is no live run's - the
// longest smoke takes minutes - so each run sweeps those first; one an orphaned
// browser still holds stays until it has gone.
// The sweep also takes the prefix runs used before the rename to AuraDream.
const STALE_MS = 6 * 3600 * 1000;
const SWEPT = [PREFIX, 'local-diffusion-browser-'];
for (const name of (() => { try { return readdirSync(tmpdir()); } catch { return []; } })()) {
  if (!SWEPT.some((prefix) => name.startsWith(prefix))) continue;
  const dir = `${tmpdir()}/${name}`;
  try {
    if (Date.now() - statSync(dir).mtimeMs > STALE_MS) rmSync(dir, { recursive: true, force: true });
  } catch { /* held, or another run's sweep took it */ }
}
// Port 0: the browser picks a free one and writes it to DevToolsActivePort in
// the profile. A port of our own choosing could be another run's - two runs on
// one port drive each other's pages. The component, extension and background
// downloads are off: each fresh profile fetched them again, and they were
// most of what a left-behind profile weighed. On Linux the browser leads a
// process group of its own, so the whole tree can be stopped at once.
const WINDOWS = process.platform === 'win32';
// Explicit diagnostics for an owned scratch run only. Chromium normally writes
// stderr noisily; retain at most its tail and print it only on a failed run.
const STARTUP_DIAGNOSTICS = process.env.AURADREAM_BROWSER_DIAGNOSTICS === '1';
const STDERR_LIMIT = 16384;
let browserStderr = '';
const proc = spawn(browser, [
  '--headless=new', '--remote-debugging-port=0', `--window-size=${VW},${VH}`,
  '--no-first-run', '--no-default-browser-check', '--disable-sync', '--disable-gpu',
  '--disable-component-update', '--disable-extensions', '--disable-background-networking',
  `--user-data-dir=${PROFILE}`, 'about:blank',
], { stdio: STARTUP_DIAGNOSTICS ? ['ignore', 'ignore', 'pipe'] : 'ignore', detached: !WINDOWS });
if (STARTUP_DIAGNOSTICS) {
  proc.stderr.setEncoding('utf8');
  proc.stderr.on('data', (chunk) => { browserStderr = (browserStderr + chunk).slice(-STDERR_LIMIT); });
  process.on('exit', (code) => {
    if (!code) return;
    console.error(`browser diagnostics: exit=${proc.exitCode ?? 'running'} signal=${proc.signalCode ?? 'none'}`);
    if (browserStderr) console.error('browser stderr tail:\n' + browserStderr + '\nbrowser stderr tail [end]');
  });
}
let exited = false;
proc.on('exit', () => { exited = true; });
// The browser's whole tree, not its first process: Edge's renderers and
// utility processes outlived proc.kill() and held the profile open, so the
// remove that followed failed quietly and the profile stayed for good.
// On Linux the group outlives its leader, so it is stopped either way.
function killBrowser() {
  if (proc.pid === undefined) return;
  try {
    if (!WINDOWS) process.kill(-proc.pid, 'SIGKILL');
    else if (!exited) spawnSync('taskkill', ['/pid', String(proc.pid), '/T', '/F'], { stdio: 'ignore', windowsHide: true });
  } catch { /* already gone */ }
}
// Synchronous, so it runs from the 'exit' handler: every way out - the end of
// the steps, a failed expect, an abandoned call, a throw, Ctrl-C - ends here.
// Windows lets a killed process's handles go a moment after it has exited;
// the retries wait that out (5 s at most), and a profile still there after
// them is named, not dropped in silence.
let cleaned = false;
function cleanup() {
  if (cleaned) return;
  cleaned = true;
  killBrowser();
  try { rmSync(PROFILE, { recursive: true, force: true, maxRetries: 20, retryDelay: 250 }); } catch { /* reported below */ }
  if (existsSync(PROFILE)) console.error(`  ! the browser profile ${PROFILE} could not be removed; a later run sweeps it after six hours`);
}
process.on('exit', cleanup);
for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP', 'SIGBREAK']) {
  try { process.on(signal, () => process.exit(130)); } catch { /* not on this platform */ }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;

async function target() {
  for (let i = 0; i < 75; i++) {
    if (STARTUP_DIAGNOSTICS && exited) throw new Error('the browser exited before DevTools was ready');
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
// A call the browser never answers - a renderer hung or gone, an eval
// awaiting a promise that never settles - held the run until whatever ran it
// gave up and killed it, and everything it had printed went with it, so a
// stalled smoke never said where it stopped. Five minutes is far past any
// step the smokes take (their longest wait is a 40 s __until); the run then
// ends there, failed, naming the step, and so does a browser that closes the
// connection with a call unanswered.
const ANSWER_MS = 300000;
let at = 'starting the browser';
let abandoned = false;
function abandon(reason) {
  if (abandoned) return;
  abandoned = true;
  console.log(`  ! ${reason}, at ${at}`);
  try { ws.close(); } catch { /* already closed */ }
  process.exit(1);
}
ws.onclose = () => { if (pending.size) abandon('the browser closed the connection'); };
const send = (method, params = {}) => new Promise((resolve) => {
  const n = ++id;
  const timer = setTimeout(() => abandon(`no answer to ${method} in ${ANSWER_MS / 1000} s`), ANSWER_MS);
  pending.set(n, (msg) => { clearTimeout(timer); resolve(msg); });
  ws.send(JSON.stringify({ id: n, method, params }));
});

const viewport = (width, height, deviceScaleFactor = 1) =>
  send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor, mobile: false });

await send('Page.enable');
await send('Network.enable');
await send('Network.setCacheDisabled', { cacheDisabled: true });
await send('Runtime.enable');
await viewport(VW, VH, DPR);

// What threw, or a promise that rejected, is no value: returnByValue turned
// the Error into a truthy {}, and an expect that threw - a null element
// dereferenced, a predicate throwing inside __until - passed.
async function evaluate(expression) {
  const res = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  const details = res.result?.exceptionDetails;
  if (details) {
    console.log('  ! ' + (details.exception?.description ?? details.text));
    return undefined;
  }
  return res.result?.result?.value;
}
const mouse = (type, x, y, extra = {}) =>
  send('Input.dispatchMouseEvent', { type, x, y, button: 'left', clickCount: 1, ...extra });
// A point, or a selector standing for the middle of its element's box.
// The centre once it holds still across two reads 50 ms apart (2 s at
// most): a press made while a smooth scroll is still moving the page - Show
// new gallery images scrolls to the gallery's head - landed where the
// element had been, and the next click missed its button now and then.
async function point(at) {
  if (Array.isArray(at)) return at;
  const read = () => evaluate(`(() => { const e = document.querySelector(${JSON.stringify(at)}); if (!e) return null; const r = e.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; })()`);
  let xy = await read();
  for (let n = 0; xy && n < 40; n++) {
    await sleep(50);
    const again = await read();
    if (again && again[0] === xy[0] && again[1] === xy[1]) break;
    xy = again;
  }
  if (!xy) console.log('  ! no element for ' + at);
  return xy ?? [0, 0];
}
// Escape carries its key code, as a real press does: a modal <dialog>
// closes on that, not on the key's name alone. A plain Enter stays a bare
// key name (the smokes' chip and autocomplete steps are written against
// that); in a keycombo, Enter is sent as the real key, text and all.
const ESCAPE = { code: 'Escape', windowsVirtualKeyCode: 27 };
const COMBO_KEYS = {
  Enter: { code: 'Enter', windowsVirtualKeyCode: 13, text: '\r' }, Escape: ESCAPE,
  // Ctrl+A must also exercise the browser's native text selection, so a
  // smoke can prove an application shortcut leaves text fields alone.
  a: { code: 'KeyA', windowsVirtualKeyCode: 65 }, A: { code: 'KeyA', windowsVirtualKeyCode: 65 },
  // Native Undo needs its physical key code too; a key name alone only
  // exercises application handlers, leaving a textarea's history untouched.
  z: { code: 'KeyZ', windowsVirtualKeyCode: 90 }, Z: { code: 'KeyZ', windowsVirtualKeyCode: 90 },
};
// The codes a {"key", "native": true} step adds, so the focused control's
// own default action runs as on a real press (a closed <select>'s arrows).
const NATIVE_KEYS = {
  ArrowDown: { code: 'ArrowDown', windowsVirtualKeyCode: 40 },
  ArrowUp: { code: 'ArrowUp', windowsVirtualKeyCode: 38 },
  // A form's own submit on Enter (the Tags helper's finder) needs the key's
  // text as well as its code.
  Enter: { code: 'Enter', windowsVirtualKeyCode: 13, text: '\r' },
};

for (const [n, step] of steps.entries()) {
  at = `step ${n + 1}: ${JSON.stringify(step).slice(0, 160)}`;
  if (step.navigate) { await send('Page.navigate', { url: step.navigate }); await sleep(800); }
  else if (step.wait) await sleep(step.wait);
  else if (step.eval) await evaluate(step.eval);
  else if (step.log) console.log('  ' + step.log + ' => ' + JSON.stringify(await evaluate(step.log)));
  else if (step.expect) {
    const value = await evaluate(step.expect);
    if (!value) failures += 1;
    console.log(`  ${value ? 'ok  ' : 'FAIL'} ${step.expect}`);
    if (!value && step.fatal === true) {
      console.log(`  ! fatal expectation failed at step ${n + 1}; later steps were not run`);
      break;
    }
  } else if (step.key) {
    const text = step.key.length === 1 ? step.key : undefined;
    const code = step.key === 'Escape' ? ESCAPE : step.native ? (NATIVE_KEYS[step.key] ?? {}) : {};
    await send('Input.dispatchKeyEvent', {
      type: 'keyDown', key: step.key, text, ...code, ...(step.repeat === true ? { autoRepeat: true } : {}),
    });
    await send('Input.dispatchKeyEvent', { type: 'keyUp', key: step.key, ...code });
    await sleep(100);
  } else if (step.keycombo) {
    const mods = step.keycombo.slice(0, -1);
    const key = step.keycombo.at(-1);
    const code = COMBO_KEYS[key] ?? {};
    const held = [];
    let modifiers = 0;
    try {
      for (const mod of mods) {
        modifiers |= KEY_MODIFIERS[mod];
        held.push(mod);
        await send('Input.dispatchKeyEvent', { type: 'rawKeyDown', key: mod, modifiers });
      }
      await send('Input.dispatchKeyEvent', { type: 'keyDown', key, modifiers, ...code });
      await send('Input.dispatchKeyEvent', { type: 'keyUp', key, modifiers, ...code, text: undefined });
    } finally {
      for (const mod of held.reverse()) {
        modifiers &= ~KEY_MODIFIERS[mod];
        await send('Input.dispatchKeyEvent', { type: 'keyUp', key: mod, modifiers });
      }
    }
    await sleep(100);
  } else if (step.type) {
    await send('Input.insertText', { text: step.type });
    await sleep(100);
  } else if (step.wheel) {
    const [x, y] = await point(typeof step.wheel[0] === 'string' ? step.wheel[0] : step.wheel.slice(0, 2));
    const deltaY = step.wheel[step.wheel.length - 1];
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
  } else if (step.rightclick) {
    const [x, y] = await point(step.rightclick);
    const modifiers = step.shift ? 8 : 0;
    await mouse('mouseMoved', x, y);
    await mouse('mousePressed', x, y, { button: 'right', buttons: 2, modifiers });
    await mouse('mouseReleased', x, y, { button: 'right', modifiers });
    await sleep(150);
  } else if (step.dblclick) {
    const [x, y] = await point(step.dblclick);
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
    console.log('  wrote ' + writeScreenshot(screenshots, step.shot, res.result.data));
  } else if (step.fullshot) {
    const res = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
    console.log('  wrote ' + writeScreenshot(screenshots, step.fullshot, res.result.data));
  } else if (step.elshot) {
    const [sel, out] = step.elshot;
    const box = await evaluate(`(() => { const e = document.querySelector(${JSON.stringify(sel)}); if (!e) return null; e.scrollIntoView({block: 'nearest'}); const r = e.getBoundingClientRect(); return {x: r.x + scrollX, y: r.y + scrollY, width: r.width, height: r.height}; })()`);
    if (!box || !box.width || !box.height) console.log('  ! elshot: no visible element for ' + sel);
    else {
      const res = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, clip: { ...box, scale: 1 } });
      const destination = writeScreenshot(screenshots, out, res.result.data);
      console.log('  wrote ' + destination + ' (' + Math.round(box.width) + 'x' + Math.round(box.height) + ')');
    }
  }
}

// Asked to close, the browser shuts every process down and lets go of the
// profile itself; cleanup() then kills only what is still there after 5 s.
pending.clear();
try { ws.send(JSON.stringify({ id: ++id, method: 'Browser.close' })); } catch { /* connection gone */ }
for (let i = 0; i < 50 && !exited; i++) await sleep(100);
try { ws.close(); } catch { /* already closed */ }
cleanup();
if (failures) console.log(`${failures} expectation(s) failed`);
process.exit(failures ? 1 : 0);
