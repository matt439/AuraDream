// Screenshot destinations have one portable interpretation on every host.
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve, win32 } from 'node:path';

export function screenshotNames(steps) {
  return steps.flatMap((step) => [
    ...(['shot', 'fullshot'].filter((key) => Object.hasOwn(step, key)).map((key) => step[key])),
    ...(Object.hasOwn(step, 'elshot') ? [step.elshot?.[1]] : []),
  ]);
}

export function screenshotPath(root, name) {
  // Refuse drive-relative paths too (C:file.png), and backslashes rather
  // than treating them as filename characters on POSIX and separators on Windows.
  if (typeof name !== 'string' || !name.trim() || name.includes('\\')
      || /[<>:"|?*\x00-\x1f]/.test(name) || win32.isAbsolute(name)
      || name.startsWith('/') || name.split('/').some((part) => part === '..')
      || name.split('/').some((part) => part !== '.' && (
        /[ .]$/.test(part) || /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(part)
      ))
      || name.endsWith('/') || name.split('/').at(-1) === '.') {
    throw new Error(`screenshot destination must be a portable relative filename: ${JSON.stringify(name)}`);
  }
  return resolve(root, name);
}

export function screenshotRoot(steps, requested = process.env.AURADREAM_SCREENSHOT_DIR) {
  const names = screenshotNames(steps);
  // Validate before creating any directories or starting a browser.
  for (const name of names) screenshotPath('.', name);
  if (!names.length) return null;
  const root = requested ? resolve(requested) : mkdtempSync(join(tmpdir(), 'auradream-shots-'));
  mkdirSync(root, { recursive: true });
  return root;
}

export function writeScreenshot(root, name, data) {
  const destination = screenshotPath(root, name);
  mkdirSync(dirname(destination), { recursive: true });
  writeFileSync(destination, Buffer.from(data, 'base64'));
  return destination;
}
