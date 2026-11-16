# Open-tab reconnect proof

Base: main `49c7521` after #79. Production code is unchanged.

Existing `smoke-jobs.json` already covers controlled fetch/SSE failures,
removal of stale steps/ETA, lost-run wording and recovery. `smoke-offline.json`
checks offline navigation and retry. #79 checks a fresh browser after a
real process restart. This slice covers the remaining same-document boundary
without duplicating those mocks.

`recovery_restart_probe.py --open-tab` pauses its partially completed owned
fake job while a fresh browser attaches. The browser creates an unrelated
unsent draft and signals readiness through its retained log. Only then does
the probe terminate the positively acknowledged fake runtime. The probe waits
for the browser to assert offline state before starting a replacement process
on the same scratch gallery. Neither fetch nor EventSource is replaced.

Six fatal browser assertions check:

- An actual active job is attached before interruption.
- The open page observes both offline status and stream loss.
- The topbar is error/offline, Generate is held, stale steps/ETA/mirror are
  hidden, and the unsent draft is unchanged.
- After the replacement answers, its server identity differs, and the page
  settles with no busy/listing state.
- The same document still exists, draft is unchanged, ready/Generate return,
  restart wording remains truthful and the queue is empty.
- The actual replacement server has no jobs: reconnection did not resubmit.

The existing API Resume then checks exact missing cells, twelve final unique
cells, unchanged retained metadata and 409 on repeated completed Resume.
The first development run retained three cells and resumed exactly nine.
Pausing can allow another completed cell; counts are measured, not assumed.
Raw JSON and browser/server logs are retained in new external scratch paths.

```powershell
python tools/recovery_restart_probe.py --data <NEW-scratch-path> --port 8357 --open-tab
```

CPU checks: 29 cold-refusal/trial-kit/docs-map tests passed, Ruff passed,
pinned Pyright 1.1.407 analyzed 204 files with no errors or warnings using
the existing virtualenv and a temporary repo-local config. No downloads.

## Limits

No production defect was reproduced. Draft setup is programmatic and the
browser attaches to a paused synthetic job; this is not a human-input,
physical-network-loss, sleeping-PC, real GPU, live-gallery or security-session
qualification. The final resume is API-driven, not another claim about the
Finish button. A fresh browser profile is used, but there is no navigation
or reload between attach, offline and recovery. Full local regression and
all smoke matrices were not run for this narrow probe.
