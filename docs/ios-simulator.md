# iPhone and iPad Simulator testing from Windows

The Windows checkout serves AuraDream's actual web assets on its fake backend.
SSH connects to the MacBook, where Xcode and Appium run native iOS Safari.
A reverse SSH tunnel makes the Windows fake available on the Mac's loopback
address. The Mac does not run AuraDream's backend or load model weights.

The [2026-11-07 results](review-2026-11-05-phone.md#6-safari-simulator-check-2026-11-07)
record 35 passed checks, three Refine keyboard failures and two not-applicable
checks. P44 remains open. Keep new results separate from that dated evidence.

## Prepared machines

Working values verified on **2026-11-07**, not permanent network addresses:

| Item | Working value |
|---|---|
| Windows checkout | `C:\Projects\AuraDream` |
| Mac SSH connection | `matt@192.168.0.103` |
| Mac tool checkout | `/Users/matt/DeckLabSafari/decklab` |
| Mac prerequisite launcher | `/Users/matt/DeckLabSafari/run-safari-decklab` |
| Private toolchain | `/Users/matt/DeckLabSafari/toolchain/` |
| Retained AuraDream harness | `/Users/matt/DeckLabSafari/auradream-20261107/` |
| Local harness and raw evidence | `work/safari-20261107/` (gitignored) |
| App origin on both machines | `http://127.0.0.1:17961` |

The prepared toolchain is Xcode 27.2 (27B52), iOS 27.0 (24A434), Python
3.14.8, Node 24.14.0 / npm 11.9.0, Appium 3.8.0 and XCUITest 12.22.7.
Python 3.14 is only the Mac's test client; AuraDream still uses its Windows
Python 3.13 environment. Appium's packages belong to the DeckLab checkout.
These are measured versions, not a request to upgrade or install packages.

Keep the Mac awake, logged in, plugged in and with its lid open. Run one
Simulator suite at a time. Close classic Simulator windows first; Xcode 27's
Device Hub can stay open. The automation is headless, but native Safari,
software keyboards and share sheets still run inside the disposable devices.

From Windows PowerShell:

```powershell
$macTarget = 'matt@192.168.0.103'
$macRunDir = '/Users/matt/DeckLabSafari/auradream-20261107'
ssh -o BatchMode=yes -o ConnectTimeout=10 $macTarget 'uname -s; whoami; xcodebuild -version'
ssh -o BatchMode=yes $macTarget "/Users/matt/DeckLabSafari/run-safari-decklab --check --runtime 27.0 --ipad-model 'iPad Air 11-inch (M2)'"
```

`--check` verifies the shared Mac tooling and available device matrix. Running
that launcher without `--check` runs **DeckLab's** scenarios, not AuraDream's.
Neither a readiness check nor Windows browser emulation is a Safari pass.
If SSH times out, recheck the Mac's current address, awake state and Remote
Login setting. Keep SSH host-key verification enabled.

## Serve a fresh Windows fixture

In a dedicated PowerShell terminal, from the AuraDream checkout:

```powershell
uv run python tools/fake_server.py --fresh --port 17961 --model black-forest-labs/FLUX.1-Krea-dev --step 0.02
```

Keep this terminal open. `--fresh` prints and retains a new scratch gallery;
use it rather than the shared default or a real gallery. The explicit model
is important: the fake's default claims Krea 2 Turbo, while these test flows
use the FLUX.1 Krea profile. No model weights are loaded.

In a second PowerShell terminal, wait for `/api/status` to answer, then seed
two completed images for the viewer's swipe check:

```powershell
$ErrorActionPreference = 'Stop'
$origin = 'http://127.0.0.1:17961'
Invoke-RestMethod "$origin/api/status" | Select-Object state, model_id
foreach ($seed in @(11, 12)) {
    $body = @{prompt='a lighthouse on a cliff at dusk'; width=512; height=512; steps=4; seed=$seed} | ConvertTo-Json
    $job = Invoke-RestMethod "$origin/api/generate" -Method Post -ContentType application/json -Body $body
    $state = $null
    for ($attempt = 0; $attempt -lt 300; $attempt++) {
        $state = Invoke-RestMethod "$origin/api/jobs/$($job.job_id)"
        if ($state.status -notin @('queued', 'running')) { break }
        Start-Sleep -Milliseconds 100
    }
    if ($state.status -ne 'done') { throw "Fixture job did not finish: $($state.status)" }
}
```

The generate POST alone is not completion. Confirm both jobs finished before
starting the Safari cases. Keep the served application files unchanged until
the run ends.

## Connect Mac loopback to the fixture

The forward is **remote** (`-R`): Mac loopback port 17961 forwards through SSH
to Windows loopback port 17961. The loopback bind leaves the fixture local to
the two machines. First verify transport; this command closes its tunnel on
completion:

```powershell
$macTarget = 'matt@192.168.0.103'
ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -R 127.0.0.1:17961:127.0.0.1:17961 $macTarget 'curl --fail --silent --show-error http://127.0.0.1:17961/api/status'
if ($LASTEXITCODE -ne 0) { throw 'Mac loopback cannot reach the Windows fake' }
```

The run command below holds its own tunnel for the entire test process.
For a separate manual/Appium session, keep a dedicated tunnel terminal open
instead: use the same SSH options with `-N` and no remote command. Open
`http://127.0.0.1:17961/` in the intended Simulator's Safari.

Loopback is a secure context for the phone's native PNG Share path. Ordinary
HTTP to a LAN address exercises its Download fallback instead. Appium has a
separate, automatically selected loopback port; 17961 is the app transport.

## Repeat the retained automated run

There is no checked-in AuraDream Simulator runner. The dated `run.py`,
`flows.py`, `phone_audit.js` and `start.sh` remain in the local and Mac harness
directories above; accepted raw run directories also contain `harness/`
snapshots. A fresh clone does not contain these ignored files. Restore or
review them before using this section; do not treat them as a supported CLI.

The retained runner imports DeckLab's Appium client, disposable-device
lifecycle and tool discovery, but supplies AuraDream's own flows. Its paths,
origin, runtime and matrix are fixed in the script. The final flow version is
`2026-11-07.6-visible-safari`; inspect it before reuse. The accepted phone rows
used `.4`; `.6` was validated on iPad only. A phone run of `.6` is new
validation. `--ipad-only` runs just the two iPad orientations; otherwise the
matrix is iPhone 15 Pro and iPad Air 11-inch (M2), each portrait and landscape.

Refresh `source.json` for **every** new run. The historical script has an old
revision fallback if that file is absent; never use that fallback as new
provenance. From a clean, committed Windows checkout:

```powershell
$ErrorActionPreference = 'Stop'
Set-Location C:\Projects\AuraDream
$sourceStatus = git status --porcelain
if ($LASTEXITCODE -ne 0) { throw 'Cannot read the source checkout' }
if ($sourceStatus) { throw 'Commit or resolve source changes before recording a new run' }
$harnessDir = 'work/safari-20261107'
$manifest = @{
    repository = 'AuraDream'
    revision = (git rev-parse HEAD)
    dirty = $false
    changed_files = @()
    application_assets_unmodified = $true
    web_tree = (git rev-parse HEAD:web)
    backend_tree = (git rev-parse HEAD:backend)
}
foreach ($value in @($manifest.revision, $manifest.web_tree, $manifest.backend_tree)) {
    if ($value -notmatch '^[0-9a-f]{40}$') { throw 'Cannot identify the tested source' }
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 "$harnessDir/source.json"
$inputs = @('run.py', 'flows.py', 'phone_audit.js', 'start.sh', 'source.json') | ForEach-Object { Join-Path $harnessDir $_ }
foreach ($path in $inputs) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing harness input: $path" }
}
$macTarget = 'matt@192.168.0.103'
$macRunDir = '/Users/matt/DeckLabSafari/auradream-20261107'
scp $inputs "${macTarget}:$macRunDir/"
if ($LASTEXITCODE -ne 0) { throw 'Harness transfer failed; do not run with older inputs' }
```

Unlike DeckLab's own suite, this workflow does not transfer an AuraDream Git
checkout to the Mac: the Windows fake serves the tested source. `source.json`
identifies that source; the runner snapshots and hashes its test inputs too.
The launcher selects the private Python/Node toolchain and uses `caffeinate`:

```powershell
ssh -t -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -R 127.0.0.1:17961:127.0.0.1:17961 $macTarget "sh $macRunDir/start.sh"
$LASTEXITCODE
```

Leave both this terminal and the Windows fake terminal open. The runner prints
its new `results-<UTC-time>` directory. Read its rows, errors and cleanup fields;
an exit code alone does not establish complete coverage. The retained script
uses exit 1 for failed checks, interrupted/incomplete coverage or setup errors.

## Native Safari pitfalls and evidence

- Open each exact test URL from `NATIVE_APP` after activating Safari, then
  select the web context whose full URL matches and whose
  `document.visibilityState` is `visible`. A background AuraDream tab can pass
  DOM checks while the native screen still shows Safari Start Page. Inspect
  native screenshots as well as DOM results.
- Appium's web `send_keys` can re-click the field and fail in its tap
  calibration overlay. The retained flows tap actual native keyboard keys.
  iPad controls use native touch coordinates measured from the current
  WebView and a visible app control, with DOM hit tests. Keep those checks;
  fixed toolbar offsets and stale coordinate translations are unreliable.
- A keyboard check must compare the actual native action rectangles with the
  software keyboard and visible screen. A nonzero `--kb-inset` is not itself
  required, and accessibility `visible=true` does not prove a button is above
  the keyboard. Preserve screenshots and native XML for failures.
- History Back in these flows uses `history.back()`. Share proves that the
  native activity UI opens, not that a file was delivered. Keep physical
  hardware, Android Back, edge-swipe Back, Split View, long press/pinch, other
  dialogs and GPU acceptance separate.

Copy the complete printed run directory, including `report.json`,
`appium.log`, per-case command logs, screenshots/XML and `harness/`:

```powershell
$runId = 'results-REPLACE-WITH-PRINTED-UTC-TIME'
$localResults = 'work/safari-20261107'
scp -r "${macTarget}:$macRunDir/$runId" $localResults
if ($LASTEXITCODE -ne 0) { throw 'Results copy failed' }
Get-FileHash -Algorithm SHA256 "$localResults/$runId/report.json"
```

Record reviewed results in their owner with the tested revision, harness
hashes, per-configuration coverage and raw report hash. The old
`summarize.py` selects specific historical run IDs; do not reuse it unchanged
to summarize a new run or combine away a failure.

## Stop and verify cleanup

Press Ctrl+C once in the attached run terminal and allow its `finally` blocks
to finish. If SSH is lost, identify the exact owned Python runner and its
working directory before sending it SIGINT; do not start another run while
cleanup is uncertain. The runner removes only its created Simulator UUIDs
and stops its own Appium process group. Never delete all Simulators or kill
unrelated Appium/Xcode processes.

After the run, check that its recorded UUIDs are absent from
`xcrun simctl list devices --json`, its Appium port is closed and no owned
runner remains. Stop the Windows fake with Ctrl+C, then check
`Get-NetTCPConnection -State Listen | Where-Object LocalPort -eq 17961`:
no output confirms the listener is gone. Closing the SSH run also closes the
reverse tunnel; stop any separate `ssh -N` tunnel too.
Retain the fresh scratch gallery and raw evidence for diagnosis rather than
deleting them as part of process cleanup.
