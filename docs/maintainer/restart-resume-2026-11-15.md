# AD-3 continuation: actual process restart and missing-cell resume

Source under test: `ac78e436`, isolated `work/recovery-provenance` worktree.
This stacks after the cancellation-recovery record without replacing it.

## Reproducible probe

`tools/recovery_restart_probe.py` starts the existing fake via the trial kit's
owned worker protocol. It refuses existing data directories and occupied
ports. A fresh token and runtime acknowledgement identify the actual process
rather than assuming the Windows venv launcher is the server. It verifies
the returned bundle directory before mutation, terminates only that owned
runtime, verifies port closure, then starts a new process against the same
scratch state. Final shutdown runs in `finally`; no gallery deletion occurs.

```powershell
python tools/recovery_restart_probe.py --data <NEW-scratch-path> --port 8346
python -m pytest tests/test_recovery_restart_probe.py -q
```

The fixed task is a 12-cell grid: two prompt-variable values (`mist`, `rain`)
x six explicit seeds (834600..834605), 256x256, Steps 8, CFG 3. The fake
paces steps at 0.08 seconds until at least two images are persisted while
the job is still running. The probe terminates the actual owned process,
not an API cancellation and not a mocked restart response. On Windows this
is abrupt process termination; it is not a physical power-loss test.

After restart the live queue is empty. The probe reads the persisted rows,
asks the production grid preview for Resume, submits that exact resume, and
checks the final gallery. This qualifies the backend Resume path, not a
physical click on the browser's Finish action or UI reconnection messaging.

## Assertions and observed result

- Two cells survived restart. The preview reported two skipped, zero queued,
  and ten missing cells/images.
- The resumed job made exactly those ten missing `(prompt, seed)` pairs.
- Final run contained exactly the expected twelve pairs, no duplicates.
- All generated rows retained the original run ID, exact weather-variable
  attribution, 256x256 dimensions and eight steps.
- Previously persisted rows were unchanged, including IDs and metadata;
  new image IDs did not overlap them.
- Completed Resume preview reported zero remaining and twelve skipped;
  another submission returned HTTP 409 rather than creating duplicate work.
- Different acknowledged server PIDs establish actual stop/start. The owned
  loopback port was closed after both process lifetimes.

The final run passed; the prior development run also passed the count/ID
checks before the stricter exact cell/variable assertions were added.
Cold refusal tests: **5 passed**. No production defect was demonstrated and
no production code changed. `git diff --check` was clean.

Evidence JSON (`restart-evidence.json`) and server logs remain in each new
owned scratch directory. The JSON records process and job IDs, run ID,
preview and final counts. No packages/models were downloaded, and no live
gallery, GPU, remote service, Refine implementation or machine setting was
used or changed.

## Remaining boundaries

- Browser Finish discovery/click/reconnect after the process restart is not
  checked here; previous cancellation smoke covers only its separately
  described browser recovery contract.
- This tests persistence after completed cell writes, not a kill precisely
  during an image/metadata transaction or file corruption.
- One checkpoint's synthetic backend and fixed prompt/seed grid do not
  qualify GPU resource release, real model determinism, mixed-model runs,
  reference-image provenance or other grid axes.
