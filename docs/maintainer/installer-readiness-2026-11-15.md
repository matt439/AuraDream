# AD-4: source-package and launcher readiness - 2026-11-15

Source baseline `ac78e436`, isolated `work/packaging-readiness`. Supported
Windows delivery here is the extracted GitHub source ZIP or Git checkout,
not a claim of a stand-alone wheel, signed executable or offline installer.
No live machine installation or GPU qualification is established by this work.

## Demonstrated gap and local fix

The installer previously checked only `pyproject.toml` and `backend/main.py`.
A partial source folder missing `uv.lock`, `web/index.html` or
`bin/auradream.cmd` could pass its dry-run with exit 0, including a proposed
shortcut to a missing launcher. Three inert-stub reproductions failed their
expected-refusal assertions before the fix.

The folder preflight now requires those five paths as files and lists the
missing ones. A normal run stops before tools/install/shortcuts; DryRun keeps
its established behavior of reporting the stop, showing the remaining
planned steps read-only, and exiting 1. Recovery guidance says to restore
application files or extract the whole ZIP separately, preserving existing
gallery/settings. This is a bounded partial-extraction check, not a
cryptographic completeness check of every source asset.

## Offline/disposable evidence

Existing installer dry-run/wrapper selection passed **6 tests** before the
change. Existing launch/startup-refusal tests passed **56 tests**. Those
launcher rehearsals use temporary paths (including spaces), inert uv stand-ins,
exit-code/argument assertions and isolated homes; they do not install packages.
Final installer selection includes the new missing-file cases in normal and
dry-run modes, all with explicit inert NVIDIA/uv/Git/winget tools: **12 passed,
32 deselected**. `git diff --check` was clean. Tests emitted one existing
Starlette/httpx deprecation warning.

An initial discarded new negative test mistakenly omitted DryRun and stubs.
It was interrupted. Inspection found uv cache metadata under that disposable
test home, but no installed package, `.venv`, shortcut or surviving child.
It is not counted as offline acceptance evidence. The replacement tests
explicitly stub tools even in normal mode; post-fix normal-mode cases assert
zero tool calls. This boundary is recorded rather than claiming every
attempt was inert.

## Supported launch and recovery sequence

1. Obtain the full repository ZIP from the project or a checkout; extract to
   a short local path outside OneDrive. Keep the directory tree together.
2. Run `install.cmd` interactively. It checks Windows/NVIDIA/memory, installs
   missing uv/Git, selects cache locations, syncs packages, checks CUDA and
   runs the doctor. These actual steps were NOT performed by this audit.
3. Sign in only for a gated checkpoint when desired; a fresh unsigned-in
   setup suggests open Animagine XL 4.0. Model weights are a separate first
   use download, not bundled or qualified here.
4. Start the installed shortcut or `bin\auradream.cmd --open` from the
   repository. Launcher resolves uv from PATH, user `.local\bin`, then
   WinGet Links and runs `uv run python -m backend.main` from project root.
5. Keep the server window open. Ctrl-C stops it; acknowledge the batch
   prompt if shown. A launcher failure preserves its error code and keeps
   the error readable. Missing uv explicitly points back to `install.cmd`.
6. For an incomplete folder, restore the named application files, not the
   gallery. For dependencies/driver issues, follow the installer's actual
   refusal and rerun once addressed. Use `uv run python -m backend.doctor`
   from an already prepared environment for read-only diagnosis.
7. Stop AuraDream before `update.cmd`; it backs up the database then pulls
   and syncs. No update, remote fetch, backup rotation or service change
   was performed in this audit.

## Boundaries

No clean Windows VM, network/package resolution, installed CUDA runtime,
model download, real first image, SmartScreen interaction, WSL or Linux
service install was qualified. Stub success is not evidence those external
steps succeed. The next acceptance stage needs an explicitly authorized
clean-machine installation with its own machine/driver/network evidence.
