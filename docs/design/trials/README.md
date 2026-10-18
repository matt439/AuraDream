# Disposable formative-trial kit

Prepared for `DEV#3`, 2026-10-18. No human trial has been conducted. This kit
supports the [validation protocol](../validation.md) and Matt's
[reported priorities](../preferences.md). It does not choose a redesign.

Use [task cards](task-cards.md), a fresh copy of the
[blank observation record](observation-template.md), and the separately labelled
[agent expert baseline](expert-baseline-2026-10-18.md). Keep the participant's
record blank until observed. An automated assertion is not Matt's prediction,
press count, timing, quote or task outcome.

## Current app: prepare, launch and reset

Run from the repository root in PowerShell. Check available space first; the
fixture itself requires 1 GiB of free headroom and writes small 256px fake PNGs.
Use an unoccupied loopback port. The default kit port is 8312, separate from the
documented smoke ports and their +400 runs.

```powershell
Get-PSDrive C
$trialData = Join-Path $env:TEMP ('local-diffusion-trial-A-' + [guid]::NewGuid().ToString('N'))
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

Variant A uses amber paper lighthouse, places quay/ridge, base grid seed 4100;
B uses cobalt ceramic observatory, shore/terrace, base 6100. Interrupted seeds
start at base+100, source/child at base+200/+201 and rejects at base+300/+301.
Run/job IDs are generated; image IDs must be read from the manifest. The
usual initial count is 67 live + 1 binned = 68 records when interruption made
two. Two current-model stars plus the foreign star are present.

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
$env:LOCAL_DIFFUSION_PORT_OFFSET = '400'
node tools/browser.mjs docs/design/prototype/check.json
Remove-Item Env:LOCAL_DIFFUSION_PORT_OFFSET
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

The [agent record](expert-baseline-2026-10-18.md) records exact fixture/build
and bounded browser evidence. Prepare a fresh fixture for Matt; do not teach
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
Remove-Item Env:LOCAL_DIFFUSION_PORT_OFFSET -ErrorAction SilentlyContinue
$env:LOCAL_DIFFUSION_VIEWPORT = '1400x900'
$env:LOCAL_DIFFUSION_DPR = '1'
node tools/browser.mjs $trialSteps 2>&1 | Tee-Object -FilePath (Join-Path $trialData 'walkthrough.log')
```

The manifest already contains the actual port, so do not add a port offset.
Use a new fixture for each full run; the driver verifies the scratch owner
before acting and refuses archive count drift. Its reported counts are
trusted automation input events, not Matt's presses. Scroll-to-target helper
steps do not establish human reachability or scrolling effort. FLUX also
needs a real True CFG adjustment to enable the negative box; the WAI concept
does not, so record that checkpoint-specific detour separately.
