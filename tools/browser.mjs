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
//   {"wheel": [x, y, deltaY]}    scroll the wheel at a point
//   {"click": [x, y]}            click a point; {"dblclick": [x, y]} double-clicks;
//                                {"drag": [x1, y1, x2, y2]} drags
//   {"shot": "file.png"}         screenshot the viewport
//
// Page-side names (`els`, `lbView`, `sheetRecords`, ...) are the app's own
// top-level bindings, reachable from `eval` because app.js is a classic script.
// An `eval` whose last expression is a promise is awaited - handy for a fetch,
// but `startJob(...)` resolves when the job *ends*, so to look at a job while
// it runs write `void startJob(...)`.
// Exit status is 1 when any `expect` failed, so this can gate a check.
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
const steps = JSON.parse(readFileSync(file, 'utf8'));

const PORT = 9333 + Math.floor(Math.random() * 500);
// A profile of this run's own. A shared one is held by whichever Edge is
// still alive from a run that crashed, and the next launch then hands its
// command line to that instance and exits without ever opening the port -
// which reads as "the browser did not come up" for every run after it.
const PROFILE = `${tmpdir()}/local-diffusion-browser-${process.pid}`;
const proc = spawn(browser, [
  '--headless=new', `--remote-debugging-port=${PORT}`, '--window-size=1400,900',
  '--no-first-run', '--no-default-browser-check', '--disable-sync', '--disable-gpu',
  `--user-data-dir=${PROFILE}`, 'about:blank',
], { stdio: 'ignore' });
const cleanup = () => { try { rmSync(PROFILE, { recursive: true, force: true }); } catch { /* locked */ } };

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;

async function target() {
  for (let i = 0; i < 50; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json();
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

await send('Page.enable');
await send('Network.enable');
await send('Network.setCacheDisabled', { cacheDisabled: true });
await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1400, height: 900, deviceScaleFactor: 1, mobile: false });

async function evaluate(expression) {
  const res = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  const details = res.result?.exceptionDetails;
  if (details) console.log('  ! ' + (details.exception?.description ?? details.text));
  return res.result?.result?.value;
}
const mouse = (type, x, y, extra = {}) =>
  send('Input.dispatchMouseEvent', { type, x, y, button: 'left', clickCount: 1, ...extra });

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
    // Escape carries its key code, as a real press does: a modal <dialog>
    // closes on that, not on the key's name alone.
    const code = step.key === 'Escape' ? { code: 'Escape', windowsVirtualKeyCode: 27 } : {};
    await send('Input.dispatchKeyEvent', { type: 'keyDown', key: step.key, text, ...code });
    await send('Input.dispatchKeyEvent', { type: 'keyUp', key: step.key, ...code });
    await sleep(100);
  } else if (step.wheel) {
    const [x, y, deltaY] = step.wheel;
    await mouse('mouseMoved', x, y);
    await send('Input.dispatchMouseEvent', { type: 'mouseWheel', x, y, deltaX: 0, deltaY });
    await sleep(100);
  } else if (step.click) {
    const [x, y] = step.click;
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
  } else if (step.shot) {
    const res = await send('Page.captureScreenshot', { format: 'png' });
    writeFileSync(step.shot, Buffer.from(res.result.data, 'base64'));
    console.log('  wrote ' + step.shot);
  }
}

ws.close();
proc.kill();
await sleep(500);
cleanup();
if (failures) console.log(`${failures} expectation(s) failed`);
process.exit(failures ? 1 : 0);
