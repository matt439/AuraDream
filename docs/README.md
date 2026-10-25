# Documentation

## Using the app

- [usage.md](usage.md): prompts, batches, grids, refine, inpaint and outpaint,
  the gallery, the CLI and recovery.
- [anime-model-guide.md](anime-model-guide.md): choosing an anime checkpoint
  and its starting settings.
- [verified.md](verified.md): what has been measured on the GPU, and what has
  not.

## Working on the code

- [rules.md](rules.md): every rule in full, the module map and what each smoke
  covers. [AGENTS.md](../AGENTS.md) is its short version.
- [design-notes.md](design-notes.md): why things are built as they are, model
  sources, and what went wrong before.
- [source-audit.md](source-audit.md): the file splits and why the long files
  stay whole.
- `plan-*.md`: feature plans. Code comments, the rules and
  [TODO.md](../TODO.md) cite them as the owners of their designs; what was
  built differently is recorded in the design notes.
- [design/](design/README.md): the product design study (requirements,
  research, a prototype and trial material).
- [testing-notes.md](testing-notes.md): observations from use, not yet looked
  into.

## Maintainer records

`review-*.md`, `review-*-findings.json` and the `review-2026-10-04-ui/`
folder are dated reviews of the code and the UI, written with AI assistance. Their findings were triaged into
[TODO.md](../TODO.md) and fixed or deferred there; they are kept as the
provenance that commit messages and the docs cite, not as current
documentation. Paths in them to a "scratchpad" refer to the reviewer's
temporary folder, which is not part of the repository.
