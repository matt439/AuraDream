<#
.SYNOPSIS
    Update AuraDream to the latest version. Safe to run again.

.DESCRIPTION
    Double-click update.cmd, which runs this script. It refuses while
    AuraDream is running, copies the gallery database to its before-update
    name (the way back, since a new version may add to the database on its
    first start), then runs `git pull --ff-only` and `uv sync`. It never
    discards a change: a pull Git refuses is explained, and nothing is reset.

.PARAMETER DryRun
    Print what it would do and change nothing.
#>
[CmdletBinding()]
param(
    [switch]$DryRun
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

# Test seam, as in install.ps1: a temporary project folder.
$TestRoot = $env:AURADREAM_INSTALL_TEST_ROOT
if ($TestRoot) { $Root = $TestRoot } else { $Root = $PSScriptRoot }
$Root = (Resolve-Path -LiteralPath $Root).Path.TrimEnd('\')

function Say([string]$Text) { Write-Host $Text }
function Ok([string]$Text) { Write-Host "  [ok]   $Text" }
function Note([string]$Text) { Write-Host "  [note] $Text" }
function Would([string]$Text) { Write-Host "  [dry run] would $Text" }
function Stop-Update([string]$Text, [string]$WhatToDo) {
    Write-Host "  [stop] $Text"
    if ($WhatToDo) { Write-Host "         What to do: $WhatToDo" }
    Write-Host ''
    Write-Host 'Update stopped. Nothing was changed by this step.'
    exit 1
}

function Invoke-Capture([string]$Exe, [string[]]$Arguments) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $lines = & $Exe @Arguments 2>&1 | ForEach-Object { "$_" }
        $code = $LASTEXITCODE
    } catch {
        $lines = @("$_")
        $code = 1
    } finally {
        $ErrorActionPreference = $old
    }
    if ($null -eq $code) { $code = 0 }
    return [pscustomobject]@{ Code = $code; Text = (@($lines) -join "`n") }
}

# One argument quoted for a Windows command line (the C runtime's rules).
function ConvertTo-Argument([string]$Value) {
    if ($Value -ne '' -and $Value -notmatch '[\s"]') { return $Value }
    $escaped = $Value -replace '(\\*)"', '$1$1\"'
    $escaped = $escaped -replace '(\\+)$', '$1$1'
    return '"' + $escaped + '"'
}

# A program that shows progress: started on this console directly, not
# through a PowerShell pipe, so its output is never this function's result.
function Invoke-Live([string]$Exe, [string[]]$Arguments) {
    try {
        $argLine = (@($Arguments) | ForEach-Object { ConvertTo-Argument $_ }) -join ' '
        $start = @{ FilePath = $Exe; NoNewWindow = $true; PassThru = $true; WorkingDirectory = (Get-Location).Path }
        if ($argLine) { $start.ArgumentList = $argLine }
        $process = Start-Process @start
        $null = $process.Handle   # held now, or 5.1 can lose the exit code
        $process.WaitForExit()
        return [int]$process.ExitCode
    } catch {
        Write-Host "  Could not run ${Exe}: $($_.Exception.Message)"
        return 1
    }
}

# The server's own second-launch test (backend/server_probe.py): something
# listens on the port, and its page has AuraDream's title. Never through a
# proxy, which would answer for loopback in AuraDream's place.
function Test-AuraDreamAt([int]$Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $connect = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if (-not $connect.AsyncWaitHandle.WaitOne(1000) -or -not $client.Connected) { return 'free' }
        $client.EndConnect($connect)
    } catch {
        return 'free'
    } finally {
        $client.Close()
    }
    try {
        $request = [System.Net.WebRequest]::Create("http://127.0.0.1:$Port/")
        $request.Proxy = $null
        $request.Timeout = 2000
        $response = $request.GetResponse()
        try {
            $reader = New-Object System.IO.StreamReader ($response.GetResponseStream())
            $page = $reader.ReadToEnd()
        } finally { $response.Close() }
        if ($page -match '<title>AuraDream</title>') { return 'auradream' }
    } catch { }
    return 'other'
}

Say 'AuraDream update'
Say "Project folder: $Root"
if ($DryRun) { Say 'Dry run: nothing will be changed; every action is printed instead.' }

# 1. Not while AuraDream runs -------------------------------------------------
$port = 7860
if ($env:AURADREAM_PORT -match '^\d+$') { $port = [int]$env:AURADREAM_PORT }
$state = Test-AuraDreamAt $port
if ($state -eq 'auradream') {
    Stop-Update "AuraDream is running at http://127.0.0.1:$port/. Updating under it could break the gallery database copy and the packages it is using." 'Stop it first: press Ctrl-C in its window (answer Y if asked), wait for the window to finish, then run update.cmd again.'
}
if ($state -eq 'other') { Note "Another program is using port $port; it is not AuraDream, so carrying on." }
$venvDir = Join-Path $Root '.venv'
try {
    $running = @(Get-CimInstance -ClassName Win32_Process -ErrorAction Stop | Where-Object {
        $_.ExecutablePath -and $_.ExecutablePath.StartsWith($venvDir + '\', [StringComparison]::OrdinalIgnoreCase)
    })
} catch { $running = @() }
if ($running.Count -gt 0) {
    $which = ($running | ForEach-Object { "$($_.Name) (process $($_.ProcessId))" }) -join ', '
    Stop-Update "A program from this folder's .venv is still running: $which. It may be AuraDream on another port, or a command such as generate." 'Stop it (Ctrl-C in its window), then run update.cmd again.'
}
Ok 'AuraDream is not running.'

# 2. Git ----------------------------------------------------------------------
if (-not (Test-Path -LiteralPath (Join-Path $Root '.git'))) {
    Stop-Update 'This folder has no .git: it came from a ZIP download, which cannot update itself.' 'Run install.cmd, which offers to turn the folder into a Git checkout, or download the new ZIP (docs/setup.md, Updating).'
}
$git = Get-Command git -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $git) { Stop-Update 'Git is not installed (or not on PATH in this window).' 'Run install.cmd, which installs it, or install it from https://git-scm.com/download/win; then open a new window.' }
$git = $git.Source
$uv = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $uv) { Stop-Update 'uv is not installed (or not on PATH in this window).' 'Run install.cmd, which installs it; then open a new window.' }
$uv = $uv.Source
$dirty = Invoke-Capture $git @('-C', $Root, 'status', '--porcelain', '--untracked-files=no')
if ($dirty.Code -eq 0 -and $dirty.Text.Trim()) {
    Note 'Some code files in this folder were changed. Git keeps them; the update goes ahead unless the new version changes the same files:'
    $dirty.Text -split "`n" | ForEach-Object { Say "         $_" }
}

# 3. The gallery database copy ------------------------------------------------
# The default database, as backend/legacy_names.py chooses it: auradream.db,
# or local-diffusion.db while that is the only one; AURADREAM_DB_PATH wins.
if ($env:AURADREAM_DB_PATH) {
    $db = [Environment]::ExpandEnvironmentVariables($env:AURADREAM_DB_PATH)
} else {
    $db = Join-Path $Root 'auradream.db'
    $legacy = Join-Path $Root 'local-diffusion.db'
    if (-not (Test-Path -LiteralPath $db) -and (Test-Path -LiteralPath $legacy)) { $db = $legacy }
}
if (Test-Path -LiteralPath $db) {
    $dir = Split-Path -Parent $db
    $stem = [IO.Path]::GetFileNameWithoutExtension($db)
    $copy = Join-Path $dir "$stem-before-update.db"
    $wal = "$db-wal"
    $hasWal = (Test-Path -LiteralPath $wal) -and ((Get-Item -LiteralPath $wal).Length -gt 0)
    if ($DryRun) {
        Would "copy $db to $copy (replacing an older copy)"
        if ($hasWal) { Would "copy $wal to $copy-wal" }
    } else {
        Copy-Item -LiteralPath $db -Destination $copy -Force
        # A journal left from an older copy would be applied to this one.
        Remove-Item -LiteralPath "$copy-wal", "$copy-shm" -Force -ErrorAction SilentlyContinue
        if ($hasWal) { Copy-Item -LiteralPath $wal -Destination "$copy-wal" -Force }
        Ok "Gallery database copied to $copy. If the new version misbehaves, that copy is the way back."
    }
} else {
    Note "No gallery database yet ($db), so there is nothing to copy."
}

# 4. git pull --ff-only -------------------------------------------------------
if ($DryRun) {
    Would "run in ${Root}: git pull --ff-only"
} else {
    Say '  Getting the latest version (git pull --ff-only)...'
    $pull = Invoke-Capture $git @('-C', $Root, 'pull', '--ff-only')
    if ($pull.Text.Trim()) { $pull.Text -split "`n" | ForEach-Object { Say "    $_" } }
    if ($pull.Code -ne 0) {
        $text = $pull.Text
        if ($text -match 'would be overwritten') {
            Stop-Update 'Git did not update: code files changed in this folder would be overwritten by the new version. Nothing was discarded.' 'Git names the files above. To keep your change, copy those files somewhere else first. Then put back the original with: git restore <file> (this discards your change to that file) and run update.cmd again.'
        } elseif ($text -match 'Not possible to fast-forward|diverging|non-fast-forward|fatal: Need to specify how to reconcile') {
            Stop-Update 'Git did not update: this folder has commits of its own that the new version does not include. Nothing was discarded.' 'If you made those commits, merge or rebase them yourself (git pull). If you did not, ask on the project page before changing anything.'
        } elseif ($text -match 'no tracking information|There is no tracking') {
            Stop-Update 'Git did not update: this branch does not follow one on GitHub.' 'Switch to main with: git checkout main, then run update.cmd again.'
        } elseif ($text -match 'Could not resolve host|unable to access|Failed to connect|timed out') {
            Stop-Update 'Git could not reach GitHub.' 'Check the internet connection (and any proxy or firewall), then run update.cmd again.'
        } else {
            Stop-Update "git pull --ff-only failed (exit $($pull.Code)). Nothing was discarded." 'Read Git''s message above; git status shows the state of the folder.'
        }
    }
    Ok 'The code is up to date.'
}

# 5. uv sync ------------------------------------------------------------------
if ($DryRun) {
    Would "run in ${Root}: uv sync"
} else {
    Say '  Installing any changed packages (uv sync); this can take a while when PyTorch changes...'
    Push-Location -LiteralPath $Root
    try { $code = Invoke-Live $uv @('sync') } finally { Pop-Location }
    if ($code -ne 0) {
        Say "  [warn] uv sync failed (exit $code). The code is updated; the next start runs the same install, or run update.cmd again."
        Say '         A proxy, firewall or antivirus scanning the download can break it.'
        exit 1
    }
    Ok 'Packages up to date.'
}

Say ''
if ($DryRun) { Say 'Dry run finished. Nothing was changed.'; exit 0 }
Say 'Update finished. Start AuraDream as usual (the Start-menu shortcut, or bin\auradream.cmd --open).'
exit 0
