# Review and plan: early fake CLI numeric refusals

**Status:** `DEV#13` implemented and locally CPU-qualified, 2026-10-28.

## Evidence and selection

The [smoke port review](review-2026-10-28-smoke-port-plan.md) ranked standalone
fake CLI validation next, followed by finite nonnegative CLI timing values.
Source tracing confirmed that main prepares data/environment and imports the
application before passing --port to uvicorn. CLI --step and --critic-delay
pass unchanged into test fakes, which send truthy values directly to time.sleep.
Negative/nonfinite delays therefore fail after startup during a job or critique.
The HTTP fake-step knob has its own clamp; it is outside this repair.

The new baseline had 26 failures in 0.38 seconds: 24 range/timing cases reached a
poisoned preparation boundary, and two noninteger ports already refused but used
the old generic wording. These were cold synthetic-module probes with counted
filesystem/environment/import/provider-preparation refusals, not actual launches,
sleeps, sockets or credential operations.

## Bounded behavior and acceptance

Use argparse converters before existing main preparation. Preserve ports
0..65535, including zero's existing automatic-listener meaning and ordinary
integer spellings. Reject out-of-range/noninteger ports with a concrete message.
Require CLI timing values to be finite and at least zero, preserving signed
zero, positive finite values and defaults. Do not invent a maximum finite delay,
change the HTTP fake knobs/test fakes, or claim every very large finite value is
portable to every operating system's sleep implementation.

Retain the existing fourteen fake-tool isolation checks. Forty-five new cold
cases cover 26 main refusals with fresh/explicit sentinel data and counted startup
poisons; five valid port cases; ten valid timing cases; and four valid-main cases
that stop at the existing data-preparation boundary. The invalid paths assert
zero attempts and unchanged sentinels/environment; valid parser cases start no
application. Run the mocked smoke runner and documentation maps too, no-cache
Ruff and installed pinned Pyright. Independent source/test review was clear.
No real gallery, browser, model, provider or OS setting is involved; all earlier
stopped actions, pending GAL12 work and unapplied setup proposals stay untouched.

## Implemented result

The CLI converters now reject invalid ports and negative/nonfinite delays before
preparation, preserving port zero, signed zero and ordinary valid values. The
runtime HTTP knobs and test fakes remain unchanged. Independent review found no
blocking issue. All forty-five new cases and existing fake, mocked runner and
documentation checks passed **100 tests in 1.22 seconds**. No-cache Ruff passed
across the repository. Installed pinned Pyright 1.1.407 analyzed **146 files with
zero diagnostics**, in 11.339 seconds. No application, browser, real socket,
credential provider, model or blocked operation was started by these checks.
