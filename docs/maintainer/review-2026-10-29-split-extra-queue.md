# Review: a split receipt's watched extra job, 2026-10-29

**Status:** closed 2026-10-29 (`GEN#31`). Repaired and qualified on a fresh
fake; one related, pre-existing race is left documented below.

## Defect

A press the server splits into several jobs (Swap tags, one per size) answers
its first job and lists the rest in `more`. If the page was busy, `startJob`
queued the answer's jobs as "Next:" rows: `setQueue([...queuedIds, job_id,
...extras()])`. `extras()` skipped only ids already in `queuedIds`.

Sequence: the page watches predecessor P, and a Swap of two differently sized
sources commits A and extra B behind it. The POST answer is still out when P
ends. A runs and ends, and the page's next-job listing adopts the running B as
the watched current job. When the delayed answer then lands, B is added as a
"Next:" row behind itself, and that row's × would cancel the job on screen.

## Repair

`extras()` also skips ids in `watchedJobs`. An extra the listing already made
the watched job is the current job, not a queued one. An extra watched to its
end is also not re-queued, and its finish was already said.
(`web/generation.js`, `startJob`.)

## Qualification

`smoke-split-extra-queue.json` (7951, fresh FLUX fake, `--step 0.3`) drives
that sequence with real owned jobs: two sources, a paused two-cell P, a native
Swap whose actual receipt is held until the listing has adopted B. It then
waits for ordinary completion, with no fabricated receipt, status, stream or
cancel.

- With the fix: passes in 81 s; the final history is five owned done jobs
  and six images, and no streams or grants remain.
- With only the filter reverted: fails fatally at the defect check. The
  delivered receipt's queue write lists B while B is current.

The two "watched" screenshots, wide and 600 px narrow, were inspected. B
(384×256) is the running job, and the only "Next:" row is the terminal A
(256×256), which the fixture explicitly tolerates: the existing history replay
reports it after B ends. The completion shot shows no queue.

## Left as is

An independent review found a related race that this change neither caused
nor fixed. The *first* job of the answer, or another extra, can end unwatched
before the listing runs. It is then still added as a "Next:" row, and
"Queued: …" is announced. The next status poll's `resyncQueue` drops the row
within seconds, and the job's finish line is still replayed through the
listing's finished path. The window is narrow, the effect transient, and no
work is lost. Closing it would mean routing the whole queueing branch through
the listing's pending/finished replay, which is a wider change than this slice.

The review's other claim was refuted: an ended watched extra's leftover
`submissions` entry is never read again.
