# Review and plan: smoke runner port preflight

**Status:** `DEV#12` implemented and locally CPU-qualified, 2026-10-28.

## Evidence and choice

Independent source review ranked (1) selected smoke-plan port validation,
(2) standalone fake CLI port validation before data/environment preparation,
and (3) finite nonnegative fake timing arguments. The first is bounded and
prevents an avoidable partial startup. It is independent of the pending GAL12
repair, unknown browser result, stopped artifact reads/equivalent searches and
unapplied setup API/UI proposals. No competitor assertion is needed.

`--offset` accepts arbitrary integers. Previously main allocated artifacts
before run_row passed an effective port to socket.connect_ex. Negative and
above-65535 values can raise OverflowError; zero cannot name the fixed listener
the browser expects. A later +100 worker band can overflow even if the first
band is valid. Future selected rows a worker span apart can collide across
bands, despite each individual base port being valid. The current table fits
within one span; no current default collision is claimed.

Seven new baseline cases failed in 0.50 seconds when real main reached poisoned
startup boundaries. No real out-of-range socket, process or artifact allocation
was attempted. Runtime exception behavior is established by the source contract;
this baseline proves the missing cold preflight without starting a failed fake.

## Bounded behavior

Validate every selected row against every possible band after the existing jobs
clamp and before artifact mkdir/allocation, sockets, pool or process startup.
Refuse with exit 2 and a concrete reason unless each effective port is 1..65535
and different bands cannot use the same port. A shared base port in the same
band remains safe because a band has one owner at a time. Do not reject a wide
row range without an actual overlap, unselected rows, or extra unclamped bands.
Reuse the validated bands for scheduling. Preserve list-only behavior, existing
jobs<=0 clamp, offsets, receipts and cleanup. Runtime occupied-port checks stay
necessary; this does not remove the probe/bind race or establish server identity.

## Qualification

Seventeen new cold cases cover invalid ranges, later-band overflow, overlapping
bands, exact boundaries, serial/shared-base and wide noncolliding rows, selected
filtering, jobs clamping and list-only behavior. Invalid cases poison all startup
boundaries and assert zero attempts. Valid cases drive main with an inline pool
and mocked row runner writing owned scratch receipts; real sockets/processes are
poisoned. Retain the existing eighteen runner cases, fake-tool argument guards,
documentation maps, no-cache Ruff and installed pinned Pyright. No actual browser,
fake server, operating-system credential, model or GPU operation is needed.

## Implemented result

The pure preflight runs before any startup effect and the scheduler reuses its
validated bands. Independent review found no blocking issue. Targeted runner,
fake-tool and documentation checks passed **55 tests in 0.73 seconds**: all
seventeen new cases, eighteen existing runner cases, fourteen fake-tool cases
and six documentation maps. No-cache Ruff passed across the repository. Pinned
Pyright 1.1.407 analyzed **146 files with zero diagnostics**, in 10.937 seconds.
All seven cold baseline failures now refuse normally. No actual browser, fake
server, socket or external process was launched by these mocked runner checks.
