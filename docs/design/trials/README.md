# Disposable formative-trial kit

Prepared for `DEV#3`, 2026-10-18; refreshed for the current app by `DEV#21`,
2026-10-30 (fixture version 2). No human trial has been conducted. This kit
supports the [validation protocol](../validation.md) and Matt's
[reported priorities](../preferences.md). It does not choose a redesign.

Use [task cards](task-cards.md), a fresh copy of the
[blank observation record](observation-template.md), and the separately labelled
[agent expert baseline](expert-baseline-2026-10-30.md) (the
[2026-10-18 one](expert-baseline-2026-10-18.md) is history, for the version 1
fixture). Keep the participant's
record blank until observed. An automated assertion is not Matt's prediction,
press count, timing, quote or task outcome.

## Current app: prepare, launch and reset

Run from the repository root in PowerShell. Check available space first; the
fixture itself requires 1 GiB of free headroom and writes small 256px fake PNGs.
Use an unoccupied loopback port. The default kit port is 8312, separate from the
documented smoke ports and their +400 runs.

```powershell
Get-PSDrive C
$trialData = Join-Path $env:TEMP ('auradream-trial-A-' + [guid]::NewGuid().ToString('N'))
uv run --no-sync python tools/trial_fixture.py --data $trialData --port 8312 --variant A
Get-Content (Join-Path $trialData 'fixture.json')
uv run --no-sync python tools/fake_server.py --data $trialData --port 8312 --model black-forest-labs/FLUX.1-Krea-dev
```

Open `http://127.0.0.1:8312/` in a fresh browser profile or private window.
Record the build (`git rev-parse HEAD`), `git status --short`, browser/version,
CSS viewport, DPR, zoom, desktop resolution and actual Windows scaling. A
headless DPR=1 run cannot establish the participant's Windows scaling. Use
Matt's normal display and input method for the human trial, and record short
window checks separately.

The setup tool starts and stops its own fake server. Before its first mutation
it checks the returned bundle directory matches the new scratch directory.
It never deletes or reuses an existing path, and will refuse an occupied port.
The later launch command opens only that prepared fake gallery. Ctrl+C stops
that server. Do not use `--fresh`, point it at a live gallery, reset another
server, download weights, or delete previous evidence as a reset mechanism.

**Reset between interface/variant attempts:** stop this owned fake server;
choose a new UUID path and change `--variant A` to `B` (and the path label).
Prepare/launch again, then use a new private browser window/profile. This resets
the gallery and the page's local choices without deleting existing artifacts.
Do not reuse a gallery changed by the archive task as another baseline.

`fixture.json` is authoritative for IDs/counts. On a successful fresh setup it
names:

| Alias | Stable definition |
|---|---|
| `complete` | 60 cells: two prompt-variable values × Steps 1,2 × CFG 2,3,4 × five seeds; larger than a 48-card page |
| `interrupted` | Twelve planned cells, cancelled after at least two saved images; the exact made count is recorded, never assumed |
| `keeper`, `complete_last` | First and last images of the completed grid |
| `source`, `child` | A direct render and an 8-step, strength 0.6 refine of it |
| `binned_reject`, `live_reject` | One recoverable reject in the bin and one live reject |
| `foreign` | Two planted Krea 2 Turbo rows; one starred, no weights loaded |
| `pick_in_order` | Keeper then child; a preparation instruction, not persisted server selection |
| `then_refine_base`, `then_refine_pass` | One Then refine job: a base and its 1.5×, strength 0.5 pass, each in a run of its own (`jobs.then_refine`) |
| `setup` | One saved setup (`/api/presets`): id, name, prompt, 512×768 and its seed |
| `saved_search` | One saved gallery search (`/api/gallery/saved-searches`): `run:` the complete grid, Starred and All models off |
| `snippet` | One personal prompt snippet (`/api/snippets`) |

`refusals` records what the server answered to three requests it must
refuse: `foreign_rerender` and `foreign_more_seeds` (another checkpoint's
image) and `binned_refine_source`, each 422 with the server's words.

Variant A uses amber paper lighthouse, places quay/ridge, base grid seed 4100;
B uses cobalt ceramic observatory, shore/terrace, base 6100. Interrupted seeds
start at base+100, source/child at base+200/+201, rejects at base+300/+301,
the Then refine base at base+400 and the saved setup at base+500. Setup,
search and snippet names carry the variant (`Lighthouse setup A`,
`Observatory grid B`, ...). Run/job IDs are generated; image IDs must be read
from the manifest. Version 2 manifests prepared on 2026-10-30 (A and B)
counted 69 live + 1 binned = 70 records with interruption making two, and
one setup, one saved search and one snippet. Two current-model stars plus
the foreign star are present. The setup, search and snippet are sidecars
beside the database, outside bundles: archive-and-clear leaves them.

**Observer setup, outside scored tasks:** press Reset once; clear search, turn
off Starred and All models, and leave Sheet off. To prepare retained picks,
search `id:<keeper>`, turn on Select, click the card; search `id:<child>` and
click it. Check the selection count says two, then apply the task's requested
filter. Use real input; page-only picks are not a fixture database feature.
Before archive, leave both picks, Starred and a current-model/search filter
active. Whole-gallery scope must still include the foreign rows and the bin.

Cases the actual app must refuse: Re-render/More seeds of another checkpoint;
a binned source used for a new refine/reference; and archive clear without
the current typed record count or with mismatching passwords. Retained
definition/Finish behavior depends on the tested build: use the server's
current plan and held reason rather than inventing missing cells from the
composer. A binned result still counts as completed for recovery.

## Concept: launch and reset

```powershell
uv run --no-sync python -m http.server 8311 --bind 127.0.0.1 --directory docs/design
```

Open `http://127.0.0.1:8311/prototype/index.html`. Reload resets its in-memory
draft, picks, jobs and synthetic outcomes. Use a new page for each variant.
The self-check is repeatable without installing software:

```powershell
$env:AURADREAM_PORT_OFFSET = '400'
node tools/browser.mjs docs/design/prototype/check.json
Remove-Item Env:AURADREAM_PORT_OFFSET
```

The existing concept is a bounded sample, not the fake gallery rendered in a
different layout. Its Run A is four WAI v17 samples #101–104 (seeds 120–123),
Run B has the interrupted sample definition, #107 is a refine child of #101,
and #111 is the other checkpoint. Stars start on #101 and #105. Use keeper
#101, child #107, source #101 and picks #101 then #107; record these alternate
values on the task card. Do not alter prototype data to simulate a backend.

The concept has geometric placeholders and no API, gallery persistence,
encryption, real planner, archive verification or restart/reconnect. Its
J12/J13 watched-job exercise is independent of its sample run outcomes.
Re-render/archive/Finish routes demonstrate a labelled interaction boundary;
they cannot be scored as equivalent execution. Mark unsupported steps **not
represented**, compare only the shared interaction, and never infer personal
speed or production durability from the concept checks.

## Evidence and next boundary

The [agent record](expert-baseline-2026-10-30.md) records exact fixture/build
and bounded browser evidence for the version 2 fixture; the
[2026-10-18 record](expert-baseline-2026-10-18.md) stays as version 1 history. Prepare a fresh fixture for Matt; do not teach
the candidate before the current-app baseline. Alternate A/B values and
candidate-layout order on repeats, recording exposure and assistance.
`DEV#1` remains the hands-on human trial. Screen-reader/touch operation and
GPU pixels/timing/memory remain their separate verification boundaries.

To repeat the **agent** current-app walkthrough on a newly prepared fixture,
start its fake server as above, then run in another terminal:

```powershell
$trialManifest = Join-Path $trialData 'fixture.json'
$trialSteps = Join-Path $trialData 'walkthrough.json'
uv run --no-sync python tools/trial_walkthrough.py --manifest $trialManifest --output $trialSteps
Remove-Item Env:AURADREAM_PORT_OFFSET -ErrorAction SilentlyContinue
$env:AURADREAM_VIEWPORT = '1400x900'
$env:AURADREAM_DPR = '1'
node tools/browser.mjs $trialSteps 2>&1 | Tee-Object -FilePath (Join-Path $trialData 'walkthrough.log')
```

The manifest already contains the actual port, so do not add a port offset.
Use a new fixture for each full run; the driver verifies the scratch owner
before acting and refuses archive count drift. It refuses a version 1
manifest. Its steps run tasks 1-5 and then the optional 6-8; the import in
task 5 renumbers gallery ids, so 6-8 read sidecar ids and runtime job ids.
Every press first checks that its target is what lies under the pointer, and
every fill that focus landed in its field: a missed press fails the step
instead of typing into whatever kept focus. Its reported counts are
trusted automation input events, not Matt's presses. Scroll-to-target helper
steps do not establish human reachability or scrolling effort. FLUX also
needs a real True CFG adjustment to enable the negative box; the WAI concept
does not, so record that checkpoint-specific detour separately.

## Refresh for the current app (DEV#21)

**Status:** stages 1-3 built and run on the fake, 2026-10-30 (branch
`wip/dev21-trial-kit`); the [result](#result-2026-10-30) is below. Matt's
hands-on trial stays `DEV#1` (USER) and is not part of this item, which only
makes the kit match the app he would be trialled on. Recorded 2026-10-29
from the TODO audit; the table below is that day's reading, kept as written.

### Why the kit is stale

The kit was committed on 2026-10-18 (`9199031`); since then only the rename
(`613ca30`) and a documentation alignment (`0604f5a`) touched
`tools/trial_fixture.py`, `tools/trial_walkthrough.py` or this folder. The
app has since gained keyboard routes and compact setup (`UI#17`-`UI#19`,
[plan](../../plan-shortcut-workflow.md)), Then refine
([plan](../../plan-then-refine.md)), named saved setups (`GEN#27`,
[plan](../../plan-generation-presets.md)), saved gallery searches (`GAL#7`,
[plan](../../plan-saved-searches.md)) and snippet/setup export and import
(`UI#20`, [plan](../../plan-snippet-portability.md); `UI#25`,
[plan](../../plan-setup-portability.md)). Matt rates
keyboard shortcuts (`L6`) and Ctrl/Shift selection (`L8`) Often in
[preferences](../preferences.md), as he does prompt-word search (`E5`),
search from a setting (`E9`), refine at size (`H1`), enlarge by Refine
(`H2`) and Reuse (`G1`). The kit exercises none of the routes built for
them. `preferences.md` has no row for saved setups, saved searches or
portability, so this refresh gives them no rating.

The parts known stale from source reading on 2026-10-29 (the walkthrough was
not run on this tree, so stage 1 may find more):

| Part | Stale since | What is wrong |
|---|---|---|
| `trial_walkthrough.py:101-102`, `:176-177` | `UI#19` compact setup | They type into `#width`/`#height`, which now sit inside the closed `<details id="custom-size">` disclosure (`web/index.html:356-370`). The click is expected to miss a hidden input. A participant has the same extra press, so the driver must open the summary with real input and record it as a detour, not skip it. |
| `trial_walkthrough.py:123` | Queue controls | It takes the first `button` in a queue row. Rows now hold cancel then the ↑ move-to-front button (`web/job-queue.js:149-175`). Cancel is still first, so this works by order alone; target `[data-queue-action="cancel"]` instead. |
| `trial_walkthrough.py` tasks 1-5 | `UI#17`-`UI#19` | No step uses `/` (focus search), M (More seeds preview) or `?` (contextual help). Task 3's sheet and task 4's searches are pointer-only. |
| `trial_fixture.py:153-156` (`roles`) | `GEN#27`, `GAL#7`, Then refine | The fixture plants no saved setup, no saved search, no personal snippet and no Then refine job (a base and its pass in one job). The manifest has no alias for any of them. |
| `trial_fixture.py:161-174` (`refusals`) | 2026-10-18 | Only the foreign Re-render and the binned refine source are recorded. The list above also names More seeds of another checkpoint, which is never recorded as a refusal. |
| `trial_fixture.py:80`, `trial_walkthrough.py:226` | - | `fixture_version` 1. New roles need version 2, and the walkthrough must refuse a version 1 manifest rather than read it with roles missing. |
| [Task cards](task-cards.md) task 5 | `GEN#27`, `GAL#7`, `UI#20` | Setups, saved searches and snippets have their own sidecars, outside bundles and gallery maintenance (`backend/api_bundles.py:248-253`; [setups plan](../../plan-generation-presets.md) "Saved contract"). The card does not ask for a prediction of what archive-and-clear leaves behind. |
| [Task cards](task-cards.md):41 | All of the above | The "do not require" list predates Then refine, saved setups and searches, and portability, so it neither includes nor excludes them. |
| [Observation record](observation-template.md) | `UI#17`-`UI#19` | Input method is recorded once per session. There is no per-task field for the route taken (pointer, keyboard shortcut, context menu). |
| [Agent baseline](expert-baseline-2026-10-18.md) | 2026-10-18 | Pinned to commit `17c7ef0` and the version 1 fixture. It is historical evidence and stays unchanged. |
| This README, "Current app" | Fixture changes | The alias table and the "67 live + 1 binned = 68" count describe version 1 and must be updated from a version 2 manifest, never assumed. |

### Design and constraints

- Fixture data goes through the fake's real routes only, as now: setups via
  `POST /api/presets`, saved searches via `POST /api/gallery/saved-searches`,
  snippets via `POST /api/snippets`, Then refine via `POST /api/generate`
  with `then_refine` (`backend/schemas_generate.py:709`). No direct SQLite
  or sidecar writes. Keep the 1 GiB headroom check, the new-path-only rule,
  the owned-runtime identity check and port 8312, away from the smoke ports
  and their +400 runs.
- Task mutations stay real browser input; `eval` only observes, scrolls,
  counts events and sets fake pacing. Prompt text is typed, never written
  through `.value`, so the page's `writeBox`/`replaceBoxText` undo paths are
  what get exercised.
- A held control is `aria-disabled` with its reason as the title
  (`AGENTS.md`, Frontend). The driver records a press on a held control and
  its stated reason; it never treats one as a success.
- No application change to make the driver pass, as the 2026-10-18 baseline
  recorded. A real defect found on the way becomes its own TODO row.
- Smokes run at 1400×900, where nothing may move under 1800 px; the driver
  uses the same viewport and DPR 1, and records any scroll it needs as a
  helper step, not as human reachability.
- New aliases (for example `setup`, `saved_search`, `snippet`,
  `then_refine_base`, `then_refine_pass`) come from the manifest, never from
  guessed ids or names.

### Non-goals

- Running or scoring Matt's trial (`DEV#1`), or filling the observation
  record with agent results.
- Choosing a redesign, or changing the first five tasks' outcomes. A new
  task for Then refine, saved setups or searches is an optional second set
  that the observer offers; Matt decides whether it is part of his trial.
- Portability tasks (export/import of setups or snippets). They have no
  preference rating; they enter the cards only if Matt asks.
- Updating the concept prototype (`docs/design/prototype/`).
- Screen-reader, touch or GPU pixel/timing/memory evidence.

### Stages and completion criteria

1. **Fixture version 2 and a working driver.** Add the roles above to
   `trial_fixture.py`, bump `fixture_version`, record the More seeds
   refusal, and repair `trial_walkthrough.py` (custom-size disclosure by real
   input, the named cancel button, version 1 refused). Closes when: a cold
   pytest builds `steps_for()` from a synthetic version 2 manifest and checks
   that every `#id` it clicks or fills exists in `web/index.html` and that a
   version 1 manifest is refused; on a fresh fake at a new scratch path the
   fixture prepares without error and `fixture.json` names every new role;
   `node tools/browser.mjs` runs the generated steps to exit 0 at 1400×900,
   DPR 1. The alias table and counts above are updated from that manifest.
2. **Cards and record.** Update task 5 to ask for a prediction of what
   archive-and-clear leaves (sidecars versus gallery), widen the
   "do not require" list, add a per-task route row to the observation
   record, and write optional tasks 6-8 (restore a saved setup then Undo;
   reopen a run through a saved search; Generate with Then refine and find
   the pass beside its base), each with A/B values and walkthrough steps.
   Closes when: the stage 1 checks still pass with the new steps, and the
   cards name only aliases the version 2 manifest contains.
3. **New agent baseline.** Record a new dated
   `expert-baseline-<date>.md` from a full run on a fresh fixture: commit,
   working-tree state, browser, viewport, DPR, the walkthrough log and the
   detours it found, labelled as agent observations. The 2026-10-18 file is
   left as it is. Closes when: the file exists and this README and the
   [validation protocol](../validation.md#trial-preparation) link it.

Matt's hands-on session then remains `DEV#1`, unchanged by this item.

### Result, 2026-10-30

**Stage 1 (fixture version 2, driver).** `tools/trial_fixture.py` writes
`fixture_version` 2: the roles above plus `then_refine_base`,
`then_refine_pass`, `setup`, `saved_search` and `snippet`, each made through
its own route (`/api/generate` with `then_refine`, `/api/presets`,
`/api/gallery/saved-searches`, `/api/snippets`), and the `foreign_more_seeds`
refusal (422, "no selected batch can have more seeds (1 foreign)"). A dead
runtime pid no longer hides the real setup failure behind WinError 87.
`tools/trial_walkthrough.py` refuses any other version; presses Custom
dimensions' summary before W/H; cancels the waiting row by
`[data-queue-action="cancel"]`; uses `/` (task 3's run search and task 4's
searches), `?` (task 3) and M (task 4, the keeper's lightbox: the More seeds
preview opens and nothing queues); records any press on an `aria-disabled`
control with its title; reinstalls its request log after the reload; checks
every press's target and every fill's focus; and checks task 4's rerenders
are 512×512 and task 5 leaves the three sidecars. Checked:
`tests/test_trial_kit.py` (cold: synthetic version 2 manifest, every pressed
`#id` in `web/index.html` or made by a page script, versions 1/None/3
refused, the disclosure and cancel targets, the key routes, roles the driver
reads are ones the fixture writes); fresh fixtures A and B prepared without
error at new scratch paths on ports 8159/8160, their manifests naming every
new role; `node tools/browser.mjs` on each one's generated steps exited 0,
206 of 206 expectations, at 1400×900, DPR 1. The alias table and counts in
"Current app" above come from those manifests.

**Stage 2 (cards and record).** Task 5 now asks for a prediction of what
archive-and-clear leaves; the "do not require" list names Then refine,
saved setups and searches, snippets and portability; the record has a
per-task route row; optional tasks 6-8 (restore a setup then Undo, reopen a
run through a saved search, Generate with Then refine and find the pass
beside its base) have A/B values and walkthrough steps, which the same two
runs exercised. The cards name only `setup`, `saved_search`, `snippet`,
`then_refine_base`/`then_refine_pass` and the version 1 aliases.

**Stage 3 (agent baseline).**
[`expert-baseline-2026-10-30.md`](expert-baseline-2026-10-30.md), linked
here and from the [validation protocol](../validation.md#trial-preparation).

**Left:** Matt's own session (`DEV#1`), with his normal display, scaling
and input, and any decision about offering tasks 6-8. No app source was
changed; the detours the baseline found are observations, not defects
filed.
