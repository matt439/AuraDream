// Parse originals, then instantiate declarations in their shared classic scope.
// The leading throw stops before any source statement, including app.js boot.
import { readFileSync } from 'node:fs';
import { Script, constants, createContext } from 'node:vm';

const scripts = JSON.parse(readFileSync(0, 'utf8'));
// An ordinary global object: from Node 26 a contextified sandbox no longer
// records a function declaration as a var name, so a later const passed.
// Before Node 22.8 there is no such constant and the sandbox still catches it.
const context = createContext(constants?.DONT_CONTEXTIFY ?? {});
const checked = [];

for (const { filename, source } of scripts) {
  try {
    // Check the untouched script first, preserving its own directive prologue.
    new Script(source, { filename });

    // Ask the parser whether the original is strict: appending a with statement
    // is legal only in sloppy scripts. No probe or source body is ever run.
    let strict = false;
    try {
      new Script(`${source}\n;with ({}) {}`, { filename });
    } catch (error) {
      if (!(error instanceof SyntaxError)) throw error;
      strict = true;
    }

    // A hashbang is legal only at the start; keep its line, but remove its text
    // before adding the stopping statement. Preserve original source line numbers.
    const body = source.replace(/^#![^\r\n]*/, '');
    const prefix = strict ? '"use strict"; throw null;\n' : 'throw null;\n';
    try {
      new Script(prefix + body, { filename, lineOffset: -1 }).runInContext(context, { timeout: 1000 });
      throw new Error('the declaration check did not stop before the source body');
    } catch (error) {
      if (error !== null) throw error;
    }
    checked.push(filename);
  } catch (error) {
    console.error(`Classic-script check failed in ${filename}`);
    if (checked.length) console.error(`Earlier scripts, in order: ${checked.join(', ')}`);
    console.error(error.stack || String(error));
    process.exit(1);
  }
}
console.log(`Checked syntax and shared declarations in ${checked.length} classic scripts; no source bodies ran.`);
