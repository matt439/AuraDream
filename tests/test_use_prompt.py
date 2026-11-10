"""Use prompt (GEN#53): reuse.js's real `usePrompt` on inert boxes.

No page boot, browser or server. `smoke-context-menu` owns the menu, the
focus, Ctrl+Z and the note as a real browser shows them.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent.parent

SCRIPT = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = process.argv[2];
const read = name => fs.readFileSync(path.join(root, 'web', name), 'utf8').replace(/\r\n/g, '\n');
const source = read('reuse.js');
const start = '// Use prompt (the image menu', end = '// Reuse restores what made a record.';
assert.equal(source.split(start).length - 1, 1);
assert.equal(source.split(end).length - 1, 1);
const slice = source.slice(source.indexOf(start), source.indexOf(end));
// Whole-box writes go through rewriteBox, never `.value` (AGENTS.md, Frontend).
assert.doesNotMatch(slice, /\.value\s*=[^=]/);

function page(prompt, negative, lightboxOpen = false) {
  const log = [];
  const box = (name, value) => ({ name, value, focus() { log.push('focus ' + name); } });
  const els = { prompt: box('prompt', prompt), negative: box('negative', negative), lightbox: { open: lightboxOpen } };
  const offers = [];
  const sandbox = vm.createContext({ els,
    rewriteBox: (el, text) => { log.push('write ' + el.name); el.value = text; },
    promptReplaced: () => log.push('replaced'), syncCfg: () => log.push('cfg'),
    schedulePreview: () => log.push('preview'), scheduleTagValidation: () => log.push('tags'),
    persistPanel: () => log.push('persist'), showCreateView: () => log.push('create'),
    dropPromptUndo: () => log.push('drop'), revealInPanel: () => log.push('reveal'),
    closeLightbox: () => { els.lightbox.open = false; log.push('close'); },
    offerPromptUndo: (message, undo, options) => offers.push({ message, undo, options }),
  });
  new vm.Script(slice + '\nthis.usePrompt = usePrompt; this.usePromptReason = usePromptReason;'
    + ' this.USE_PROMPT_EDIT = USE_PROMPT_EDIT;', { filename: 'actual-use-prompt.js' }).runInContext(sandbox);
  return { els, log, offers, sandbox };
}
const rec = { id: 7, prompt: 'a red fox in snow', negative_prompt: 'blurry', seed: 41 };

// Empty boxes: both written, the negative first, nothing to give back.
let p = page('', '');
assert.equal(p.sandbox.usePrompt(rec), true);
assert.deepEqual([p.els.prompt.value, p.els.negative.value], ['a red fox in snow', 'blurry']);
assert.deepEqual(p.log.filter(line => line.startsWith('write')), ['write negative', 'write prompt']);
assert.ok(p.log.includes('replaced') && p.log.includes('persist') && p.log.includes('cfg'));
assert.ok(p.log.indexOf('drop') < p.log.indexOf('write negative'));
assert.equal(p.offers.length, 0);

// Only whitespace in the box is nothing to give back either.
p = page('  \n', '');
p.sandbox.usePrompt(rec);
assert.equal(p.offers.length, 0);

// A record with no negative empties the box, and the Undo gives both back.
p = page('my own words', 'grainy');
p.sandbox.usePrompt({ id: 9, prompt: 'a lighthouse', negative_prompt: null });
assert.deepEqual([p.els.prompt.value, p.els.negative.value], ['a lighthouse', '']);
assert.equal(p.offers.length, 1);
assert.equal(p.offers[0].message, 'The prompt of #9 replaced yours.');
p.offers[0].undo();
assert.deepEqual([p.els.prompt.value, p.els.negative.value], ['my own words', 'grainy']);

// The Undo leaves a negative written since alone.
p = page('my own words', 'grainy');
p.sandbox.usePrompt(rec);
p.els.negative.value = 'typed since';
p.offers[0].undo();
assert.deepEqual([p.els.prompt.value, p.els.negative.value], ['my own words', 'typed since']);

// A negative alone is something to give back.
p = page('', 'grainy');
p.sandbox.usePrompt(rec);
assert.equal(p.offers.length, 1);

// The same pair already: no write, so no empty step in either box's Ctrl+Z.
p = page('a red fox in snow', 'blurry');
assert.equal(p.sandbox.usePrompt(rec), true);
assert.deepEqual(p.log.filter(line => line.startsWith('write')), []);
assert.ok(!p.log.includes('replaced') && p.log.includes('focus prompt'));
assert.equal(p.offers.length, 0);

// The lightbox is closed before anything is written.
p = page('', '', true);
p.sandbox.usePrompt(rec);
assert.equal(p.els.lightbox.open, false);
assert.ok(p.log.indexOf('close') < p.log.indexOf('write negative'));

// An instruction edit: held with its reason, nothing written or closed.
p = page('my own words', 'grainy', true);
const edit = { ...rec, edit: { source: 3 } };
assert.equal(p.sandbox.usePromptReason(edit), p.sandbox.USE_PROMPT_EDIT);
assert.equal(p.sandbox.usePromptReason(rec), '');
assert.equal(p.sandbox.usePrompt(edit), false);
assert.deepEqual(p.log, []);
assert.deepEqual([p.els.prompt.value, p.els.negative.value, p.els.lightbox.open], ['my own words', 'grainy', true]);

// The menu: Use prompt is the item after Use seed, on every image's menu.
const menu = read('context-menu.js');
const items = menu.slice(menu.indexOf('function imageItems('), menu.indexOf('const dangerItem'));
const seed = items.indexOf("label: 'Use seed'"), use = items.indexOf('usePromptItem(rec, host, lightbox)');
const more = items.indexOf("'More seeds of this batch…'");
assert.ok(seed > 0 && seed < use && use < more);
assert.doesNotMatch(items.slice(seed, use), /\bnull,/);
console.log('passed');
"""


def test_use_prompt_writes_the_pair_and_gives_it_back(tmp_path: Path):
    node = shutil.which("node")
    assert node is not None, "Node is required for the Use prompt checks"
    script = tmp_path / "use_prompt.cjs"
    script.write_text(SCRIPT, encoding="utf-8")
    done = subprocess.run(
        [node, str(script), str(ROOT)], cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8", timeout=20,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip() == "passed"
