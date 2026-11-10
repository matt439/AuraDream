// Parse originals, then instantiate declarations in their shared classic scope.
// The leading throw stops before any source statement, including app.js boot.
// Then a no-undef pass: every name a script reads or assigns resolves to a
// declaration in its own scopes, a top-level binding of any script on the
// page, an ECMAScript built-in or a browser global listed below. A misspelt
// name inside a function body parses and declares nothing, so it passed
// both checks above and threw only when that line ran in a browser.
// The pass reads Node's own bundled parser (acorn), reachable only under
// `node --expose-internals`, which tools/check_classic_scripts.py passes:
// there is no npm install here. It executes nothing either.
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { Script, constants, createContext, runInNewContext } from 'node:vm';

// The browser's globals the page's scripts use. A name joins the list by
// hand when a script first needs it: the list is all the check knows of the
// browser, and a wider one would pass a misspelling that happens to be some
// other API's name.
const BROWSER_GLOBALS = `
  AbortController Blob CSS ClipboardItem DecompressionStream Element Event EventSource File Headers
  Image ImageData InputEvent IntersectionObserver MutationObserver Node Option ResizeObserver
  Response TextDecoder TextEncoder URL URLSearchParams
  addEventListener btoa cancelAnimationFrame clearInterval clearTimeout createImageBitmap crypto
  devicePixelRatio document fetch getComputedStyle getSelection history innerHeight innerWidth
  localStorage location matchMedia navigator performance queueMicrotask removeEventListener
  requestAnimationFrame scrollTo scrollX scrollY sessionStorage setInterval setTimeout
  structuredClone window
`.trim().split(/\s+/).filter(Boolean);

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

// ---------------------------------------------------------------- no-undef

function loadParser() {
  try {
    return createRequire(import.meta.url)('internal/deps/acorn/acorn/dist/acorn');
  } catch {
    console.error("Classic-script check failed: Node's bundled parser is not reachable. "
      + 'Run tools/check_classic_scripts.py, which starts Node with --expose-internals.');
    process.exit(1);
  }
}

const FUNCTIONS = new Set(['FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression']);
const isNode = (value) => value !== null && typeof value === 'object' && typeof value.type === 'string';
function children(node) {
  const out = [];
  for (const [key, value] of Object.entries(node)) {
    if (key === 'loc') continue;
    if (Array.isArray(value)) out.push(...value.filter(isNode));
    else if (isNode(value)) out.push(value);
  }
  return out;
}
// The names a declaration's target binds: an identifier, or every one in a
// destructuring pattern.
function bound(pattern, names) {
  if (!pattern) return;
  if (pattern.type === 'Identifier') names.add(pattern.name);
  else if (pattern.type === 'ObjectPattern') {
    for (const p of pattern.properties) bound(p.type === 'RestElement' ? p.argument : p.value, names);
  } else if (pattern.type === 'ArrayPattern') for (const e of pattern.elements) bound(e, names);
  else if (pattern.type === 'AssignmentPattern') bound(pattern.left, names);
  else if (pattern.type === 'RestElement') bound(pattern.argument, names);
}
// `var` names anywhere in a function or script, not crossing into a nested
// function; in a sloppy script a block's function declarations too (Annex B
// gives each a var of its name, as the declaration check above knows).
function hoisted(node, names, sloppy) {
  for (const child of children(node)) {
    if (child.type === 'VariableDeclaration' && child.kind === 'var') {
      for (const d of child.declarations) bound(d.id, names);
    }
    if (child.type === 'FunctionDeclaration' && sloppy) names.add(child.id.name);
    if (!FUNCTIONS.has(child.type)) hoisted(child, names, sloppy);
  }
}
// What a statement list declares for its own block.
function lexical(statements, names) {
  for (const s of statements) {
    if (s.type === 'VariableDeclaration') for (const d of s.declarations) bound(d.id, names);
    else if (s.type === 'FunctionDeclaration' || s.type === 'ClassDeclaration') names.add(s.id.name);
  }
}

function undefinedNames(program, globals, sloppy) {
  const found = [];
  const resolves = (name, scope) => {
    for (let s = scope; s; s = s.parent) if (s.names.has(name)) return true;
    return globals.has(name);
  };
  const inner = (parent) => ({ names: new Set(), parent });
  function visit(node, scope) {
    switch (node.type) {
      case 'Identifier':
        if (!resolves(node.name, scope)) {
          found.push({ name: node.name, line: node.loc.start.line, column: node.loc.start.column + 1 });
        }
        return;
      case 'FunctionDeclaration': case 'FunctionExpression': case 'ArrowFunctionExpression': {
        const own = inner(scope);
        if (node.type !== 'ArrowFunctionExpression') own.names.add('arguments');
        if (node.type === 'FunctionExpression' && node.id) own.names.add(node.id.name);
        for (const param of node.params) bound(param, own.names);
        hoisted(node.body, own.names, sloppy);
        // A parameter's default or computed key is read in the function's scope.
        for (const param of node.params) visit(param, own);
        visit(node.body, own);
        return;
      }
      case 'ClassDeclaration': case 'ClassExpression': {
        const own = inner(scope);
        if (node.id) own.names.add(node.id.name);
        if (node.superClass) visit(node.superClass, own);
        visit(node.body, own);
        return;
      }
      case 'BlockStatement': case 'StaticBlock': case 'Program': {
        // The program's own names are already among the globals.
        const own = node.type === 'Program' ? scope : inner(scope);
        if (node.type !== 'Program') lexical(node.body, own.names);
        for (const s of node.body) visit(s, own);
        return;
      }
      case 'SwitchStatement': {
        visit(node.discriminant, scope);
        const own = inner(scope);
        for (const c of node.cases) lexical(c.consequent, own.names);
        for (const c of node.cases) visit(c, own);
        return;
      }
      case 'ForStatement': case 'ForInStatement': case 'ForOfStatement': {
        const own = inner(scope);
        const head = node.type === 'ForStatement' ? node.init : node.left;
        if (head?.type === 'VariableDeclaration') lexical([head], own.names);
        for (const child of children(node)) visit(child, own);
        return;
      }
      case 'CatchClause': {
        const own = inner(scope);
        bound(node.param, own.names);
        if (node.param) visit(node.param, own);
        visit(node.body, own);
        return;
      }
      case 'MemberExpression':
        visit(node.object, scope);
        if (node.computed) visit(node.property, scope);
        return;
      case 'Property': case 'MethodDefinition': case 'PropertyDefinition':
        if (node.computed) visit(node.key, scope);
        if (node.value) visit(node.value, scope);
        return;
      case 'LabeledStatement':
        visit(node.body, scope);
        return;
      case 'BreakStatement': case 'ContinueStatement': case 'MetaProperty':
        return;
      case 'UnaryExpression':
        // `typeof name` is how a script asks whether a name exists at all.
        if (node.operator === 'typeof' && node.argument.type === 'Identifier') return;
        visit(node.argument, scope);
        return;
      default:
        for (const child of children(node)) visit(child, scope);
    }
  }
  visit(program, { names: new Set(), parent: null });
  return found;
}

const acorn = loadParser();
const globals = new Set([...runInNewContext('Object.getOwnPropertyNames(globalThis)'), ...BROWSER_GLOBALS]);
const programs = [];
for (const { filename, source } of scripts) {
  let program;
  try {
    program = acorn.parse(source, { ecmaVersion: 'latest', sourceType: 'script', locations: true, allowHashBang: true });
  } catch (error) {
    console.error(`Classic-script check failed in ${filename}: Node's bundled parser (acorn ${acorn.version}) could not read it`);
    console.error(String(error));
    process.exit(1);
  }
  // The directive prologue: the leading string-literal statements.
  const prologue = program.body.findIndex((s) => typeof s.directive !== 'string');
  const sloppy = !program.body.slice(0, prologue < 0 ? program.body.length : prologue)
    .some((s) => s.directive === 'use strict');
  // Every script's top-level names are every script's: one global scope.
  lexical(program.body, globals);
  hoisted(program, globals, sloppy);
  programs.push({ filename, program, sloppy });
}
const undefinedFound = programs.flatMap(({ filename, program, sloppy }) =>
  undefinedNames(program, globals, sloppy).map((u) => `${filename}:${u.line}:${u.column} '${u.name}' is not defined`));
if (undefinedFound.length) {
  console.error('Classic-script check failed: a name no script declares and no listed global provides');
  for (const line of undefinedFound) console.error('  ' + line);
  console.error('A browser global the page newly uses joins BROWSER_GLOBALS in tools/check_classic_scripts.mjs.');
  process.exit(1);
}
console.log(`Checked syntax, shared declarations and undefined names in ${checked.length} classic scripts; no source bodies ran.`);
