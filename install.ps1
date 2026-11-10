<#
.SYNOPSIS
    Install AuraDream on this Windows machine. Safe to run again.

.DESCRIPTION
    Double-click install.cmd, which runs this script. It checks the NVIDIA
    driver, the card and memory, installs uv and Git only if they are missing
    (winget, or uv's own installer at a pinned version), asks where model
    downloads go (HF_HOME) before anything downloads, runs `uv sync` and a
    CUDA check, runs `python -m backend.doctor`, optionally guides the Hugging
    Face sign-in, makes Start-menu (and, if you like, desktop) shortcuts and
    offers to start AuraDream.

    It never downloads model weights, writes auradream.settings.json, moves
    the gallery, sets AURADREAM_* variables for your user, touches WSL or
    needs administrator rights. HF_HOME and UV_CACHE_DIR are the only user
    variables it may set, and only when you choose a non-default folder.

.PARAMETER DryRun
    Print every action it would take and change nothing. Read-only checks
    (nvidia-smi, RAM, disk, git status, the doctor if .venv exists) still run.

.PARAMETER Yes
    Accept every default without asking (keep the default download folder,
    skip the Hugging Face sign-in, make both shortcuts, start AuraDream).
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$Yes
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

# uv's own installer, pinned: the version this project was last set up with.
$UvVersion = '0.12.7'
$UvInstaller = "https://astral.sh/uv/$UvVersion/install.ps1"
$RepoUrl = 'https://github.com/matt439/AuraDream.git'
$DriverUrl = 'https://www.nvidia.com/Download/index.aspx'
$OpenModel = 'cagliostrolab/animagine-xl-4.0'
$GatedModel = 'black-forest-labs/FLUX.1-Krea-dev'

# Test seam: tests point this at a temporary project folder and put stub
# nvidia-smi/winget/uv/git scripts first on PATH. The desktop is then taken
# from USERPROFILE rather than the shell's own folder setting.
$TestRoot = $env:AURADREAM_INSTALL_TEST_ROOT
if ($TestRoot) { $Root = $TestRoot } else { $Root = $PSScriptRoot }
# ProviderPath: a plain path even on a network share, never PowerShell's
# provider-qualified form (step 1 refuses a share anyway).
$Root = (Resolve-Path -LiteralPath $Root).ProviderPath.TrimEnd('\')

$script:WouldStop = @()
$script:StepNumber = 0
$StepCount = 12

# --- Output -------------------------------------------------------------------

function Say([string]$Text) { Write-Host $Text }
function Step([string]$Title) {
    $script:StepNumber++
    Write-Host ''
    Write-Host ("== Step {0} of {1}: {2}" -f $script:StepNumber, $StepCount, $Title)
}
function Ok([string]$Text) { Write-Host "  [ok]   $Text" }
function Note([string]$Text) { Write-Host "  [note] $Text" }
function Warn([string]$Text) { Write-Host "  [warn] $Text" }
function Would([string]$Text) { Write-Host "  [dry run] would $Text" }

# A problem that makes AuraDream unusable: a real run stops here; a dry run
# says so and carries on, so the rest of the plan is still shown.
function Stop-Install([string]$Text, [string]$WhatToDo) {
    Write-Host "  [stop] $Text"
    if ($WhatToDo) { Write-Host "         What to do: $WhatToDo" }
    if ($DryRun) {
        Write-Host '         (dry run: a real run would stop here; carrying on to show the rest)'
        $script:WouldStop += $Text
        return
    }
    Write-Host ''
    Write-Host 'Install stopped. Nothing after this step was done; run install.cmd again once it is fixed.'
    exit 1
}

function Ask-YesNo([string]$Question, [bool]$Default) {
    if ($Default) { $hint = '[Y/n]'; $word = 'yes' } else { $hint = '[y/N]'; $word = 'no' }
    if ($Yes) {
        Write-Host "  $Question $hint -> $word (-Yes)"
        return $Default
    }
    while ($true) {
        $answer = Read-Host "  $Question $hint"
        if ([string]::IsNullOrWhiteSpace($answer)) { return $Default }
        if ($answer -match '^\s*(y|yes)\s*$') { return $true }
        if ($answer -match '^\s*(n|no)\s*$') { return $false }
        Write-Host '  Please answer y or n.'
    }
}

function Ask-Text([string]$Question, [string]$Default) {
    if ($Yes) {
        Write-Host "  $Question [$Default] -> $Default (-Yes)"
        return $Default
    }
    $answer = Read-Host "  $Question [$Default]"
    if ([string]::IsNullOrWhiteSpace($answer)) { return $Default }
    return $answer.Trim().Trim('"')
}

function Wait-Enter([string]$Text) {
    if ($Yes) { return }
    [void](Read-Host "  $Text (press Enter)")
}

# --- Running other programs ---------------------------------------------------

# Windows PowerShell 5.1 turns a native program's stderr into errors, which
# 'Stop' would make fatal: native calls run under 'Continue' and are judged
# by their exit code.
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

# A program that talks to you (progress, prompts): started on this console
# directly, not through a PowerShell pipe, so its prompts show as they are
# written and its output is never mistaken for this function's result.
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

# Folders where a tool lands when this window's PATH predates its install.
function Get-KnownToolDirs([string]$Name) {
    $dirs = @()
    if ($Name -eq 'uv') {
        if ($env:USERPROFILE) { $dirs += (Join-Path $env:USERPROFILE '.local\bin') }
        if ($env:LOCALAPPDATA) { $dirs += (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links') }
    }
    if ($Name -eq 'git' -and -not $TestRoot) {
        if ($env:ProgramFiles) { $dirs += (Join-Path $env:ProgramFiles 'Git\cmd') }
        if ($env:LOCALAPPDATA) { $dirs += (Join-Path $env:LOCALAPPDATA 'Programs\Git\cmd') }
    }
    return $dirs
}

function Find-Tool([string]$Name) {
    $found = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { return $found.Source }
    foreach ($dir in (Get-KnownToolDirs $Name)) {
        foreach ($ext in @('.exe', '.cmd')) {
            $candidate = Join-Path $dir ($Name + $ext)
            if (Test-Path -LiteralPath $candidate) {
                $env:Path = "$dir;$env:Path"
                return $candidate
            }
        }
    }
    return $null
}

# After an install: add the user and machine PATH entries this window lacks,
# keeping what it already has first.
function Update-SessionPath {
    $have = @($env:Path -split ';' | Where-Object { $_ })
    $more = @()
    foreach ($scope in @('User', 'Machine')) {
        $value = [Environment]::GetEnvironmentVariable('Path', $scope)
        if ($value) { $more += @($value -split ';' | Where-Object { $_ }) }
    }
    foreach ($entry in $more) {
        $expanded = [Environment]::ExpandEnvironmentVariables($entry)
        if ($have -notcontains $expanded) { $have += $expanded }
    }
    $env:Path = ($have -join ';')
}

# A variable saved for your user (what a new window would see). Under the
# test seam the machine's own settings are not consulted.
function Get-UserVariable([string]$Name) {
    if ($TestRoot) { return $null }
    return [Environment]::GetEnvironmentVariable($Name, 'User')
}

function Set-UserVariable([string]$Name, [string]$Value) {
    if ($DryRun) {
        Would "set $Name=$Value for your user (setx) and for this window"
        return
    }
    $result = Invoke-Capture 'setx' @($Name, $Value)
    if ($result.Code -ne 0) {
        Warn "setx could not save $Name ($($result.Text)). Set it yourself later: setx $Name `"$Value`""
    } else {
        Ok "$Name is now $Value for your user (new windows see it) and for this install."
    }
    Set-Item -Path "Env:$Name" -Value $Value
}

function Get-FreeGB([string]$Path) {
    try {
        $qualifier = Split-Path -Qualifier $Path
        $drive = New-Object System.IO.DriveInfo ($qualifier)
        if ($drive.IsReady) { return [math]::Floor($drive.AvailableFreeSpace / 1e9) }
    } catch { }
    return $null
}

# A full path on a drive this machine has, such as D:\huggingface.
function Test-DrivePath([string]$Path) {
    try {
        if ($Path -notmatch '^[A-Za-z]:\\') { return $false }
        return (Test-Path -LiteralPath (Split-Path -Qualifier $Path))
    } catch { return $false }
}

function Test-Interactive {
    try { return [Environment]::UserInteractive -and -not [Console]::IsInputRedirected } catch { return $false }
}

# The server's own second-launch test (backend/server_probe.py), as in
# update.ps1: something listens on the port, and its page has AuraDream's
# title. Never through a proxy, which would answer for loopback.
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
        $scheme = if ($env:AURADREAM_SSL_CERTFILE) { 'https' } else { 'http' }
        $request = [System.Net.WebRequest]::Create("${scheme}://127.0.0.1:$Port/")
        $request.Proxy = $null
        $request.AllowAutoRedirect = $false
        $request.Timeout = 2000
        $response = $request.GetResponse()
        try {
            $reader = New-Object System.IO.StreamReader ($response.GetResponseStream())
            $page = $reader.ReadToEnd()
        } finally { $response.Close() }
        if ($page -match '<title>AuraDream</title>') { return 'auradream' }
    } catch {
        $failure = $_.Exception
        while ($null -ne $failure) {
            if ($env:AURADREAM_SSL_CERTFILE -and $failure -is [System.Net.WebException] -and $failure.Status -in @('TrustFailure', 'SecureChannelFailure')) { return 'tls_unverified' }
            $failure = $failure.InnerException
        }
    }
    return 'other'
}

# Programs running from this folder's .venv (AuraDream on another port, or a
# command such as generate), as update.ps1 checks.
function Get-VenvProcesses {
    $venvDir = Join-Path $Root '.venv'
    try {
        return @(Get-CimInstance -ClassName Win32_Process -ErrorAction Stop | Where-Object {
            $_.ExecutablePath -and $_.ExecutablePath.StartsWith($venvDir + '\', [StringComparison]::OrdinalIgnoreCase)
        })
    } catch { return @() }
}

# The saved browser choices beside the gallery database (auradream.settings.json,
# or the pre-rename local-diffusion.settings.json), where a start restores them.
function Test-SavedChoices {
    if ($env:AURADREAM_DB_PATH) {
        $dbDir = Split-Path -Parent ([Environment]::ExpandEnvironmentVariables($env:AURADREAM_DB_PATH))
    } else { $dbDir = $Root }
    # Relative to the project folder, where bin\auradream.cmd starts the server.
    if (-not $dbDir) { $dbDir = $Root } elseif (-not [IO.Path]::IsPathRooted($dbDir)) { $dbDir = Join-Path $Root $dbDir }
    foreach ($name in @('auradream.settings.json', 'local-diffusion.settings.json')) {
        if ((Test-Path -LiteralPath (Join-Path $dbDir $name))) { return $true }
    }
    return $false
}

# --- Start --------------------------------------------------------------------

Say 'AuraDream installer'
Say "Project folder: $Root"
if ($DryRun) { Say 'Dry run: nothing will be installed or changed; every action is printed instead.' }
if ($Yes) { Say '-Yes: every question takes its default answer.' }
if (-not $Yes -and -not (Test-Interactive)) {
    Say 'No keyboard input here, so every question takes its default answer (as with -Yes).'
    $Yes = $true
}

# 1. Windows and the folder ---------------------------------------------------
Step 'Windows and the project folder'
if ([Environment]::OSVersion.Platform -ne 'Win32NT') {
    Stop-Install 'This installer is for Windows.' 'On Linux or WSL 2, use wsl/setup.sh (README: Run under WSL 2).'
} else {
    $build = [Environment]::OSVersion.Version.Build
    if ($build -ge 22000) { Ok "Windows 11 (build $build)." }
    elseif ($build -ge 10240) { Note "Windows 10 (build $build). AuraDream is tested on Windows 11; winget may be missing here, which is handled." }
    else { Warn "Windows build $build is older than Windows 10 and is not supported." }
}
if (-not ((Test-Path -LiteralPath (Join-Path $Root 'pyproject.toml')) -and (Test-Path -LiteralPath (Join-Path $Root 'backend\main.py')))) {
    Stop-Install "This folder does not look like AuraDream ($Root has no pyproject.toml and backend\main.py)." 'Run install.cmd from inside the AuraDream folder you downloaded or cloned.'
}
if ($Root.StartsWith('\\')) {
    Stop-Install "This folder is on a network share ($Root). The packages, the gallery database and the shortcuts need a folder on a drive of this machine." 'Copy the folder to a local drive, such as C:\AuraDream, and run install.cmd there.'
}
# A re-run may replace code files (step 5) and packages (step 8), which a
# running AuraDream is using, as update.cmd refuses to.
$port = 7860
if ($env:AURADREAM_PORT -match '^\d+$') { $port = [int]$env:AURADREAM_PORT }
$scheme = if ($env:AURADREAM_SSL_CERTFILE) { 'https' } else { 'http' }
$serverUrl = "${scheme}://127.0.0.1:$port/"
$state = Test-AuraDreamAt $port
if ($state -eq 'tls_unverified') {
    Stop-Install "The configured HTTPS listener at $serverUrl could not be verified." 'Use a trusted certificate including 127.0.0.1, or stop that listener before installing. Certificate verification stays on.'
} elseif ($state -eq 'auradream') {
    Stop-Install "AuraDream is running at $serverUrl. Installing under it could break the packages it is using." 'Stop it first: press Ctrl-C in its window (answer Y if asked), wait for the window to finish, then run install.cmd again.'
} else {
    $running = @(Get-VenvProcesses)
    if ($running.Count -gt 0) {
        $which = ($running | ForEach-Object { "$($_.Name) (process $($_.ProcessId))" }) -join ', '
        Stop-Install "A program from this folder's .venv is still running: $which. It may be AuraDream on another port, or a command such as generate." 'Stop it (Ctrl-C in its window), then run install.cmd again.'
    }
}
$oneDrives = @($env:OneDrive, $env:OneDriveConsumer, $env:OneDriveCommercial) | Where-Object { $_ }
$inOneDrive = $Root -match '\\OneDrive( - [^\\]+)?(\\|$)'
foreach ($od in $oneDrives) { if ($Root.StartsWith($od.TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase)) { $inOneDrive = $true } }
if ($inOneDrive) {
    Warn 'This folder is synced by OneDrive. OneDrive uploads the gallery database and thousands of package files while AuraDream uses them, which slows everything and can break the database.'
    Warn 'Better: move the folder somewhere OneDrive does not sync, such as C:\AuraDream, and run install.cmd there.'
}
if ($Root.Length -gt 80) {
    Warn "The folder path is long ($($Root.Length) characters). Package files nest deeply, and Windows' 260-character limit can break the install. A short folder such as C:\AuraDream is safer."
}
if ($Root -match '[^\x00-\x7F]') {
    Warn 'The folder path has non-English characters. Some tools mishandle them; a plain folder such as C:\AuraDream is safer.'
}
if (-not $inOneDrive -and $Root.Length -le 80 -and $Root -notmatch '[^\x00-\x7F]') { Ok 'The folder path is short, plain and not synced by OneDrive.' }

# 2. GPU and driver -----------------------------------------------------------
Step 'NVIDIA GPU and driver'
$vramGB = $null
$ramGiB = $null
$smi = Find-Tool 'nvidia-smi'
if (-not $smi) {
    Stop-Install 'nvidia-smi was not found. AuraDream needs an NVIDIA GPU; AMD and Intel graphics are not supported.' "If this PC has an NVIDIA card, install its current driver from $DriverUrl, reboot, and run install.cmd again. Installing that driver does not make an AMD or Intel card work."
} else {
    $header = Invoke-Capture $smi @()
    if ($header.Code -ne 0) {
        Stop-Install "nvidia-smi is installed but failed (exit $($header.Code)): the driver is not working." "Reinstall the current driver from $DriverUrl and reboot."
    } elseif ($header.Text -match 'CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)') {
        $cudaMajor = [int]$Matches[1]; $cudaMinor = [int]$Matches[2]
        if ($cudaMajor -lt 13) {
            Stop-Install "The driver supports CUDA $cudaMajor.$cudaMinor; AuraDream's PyTorch needs CUDA 13.0 or higher." "Update the NVIDIA driver from $DriverUrl (the current Game Ready or Studio driver), reboot, and run install.cmd again. You do not need the CUDA Toolkit."
        } else {
            Ok "The driver supports CUDA $cudaMajor.$cudaMinor (13.0 or higher is needed)."
        }
    } else {
        Warn 'nvidia-smi ran, but its CUDA version could not be read. Carrying on; the CUDA check after the install is the one that counts.'
    }
    $query = Invoke-Capture $smi @('--query-gpu=name,compute_cap,memory.total', '--format=csv,noheader,nounits')
    $first = @($query.Text -split "`n" | Where-Object { $_ -match ',' }) | Select-Object -First 1
    if ($query.Code -ne 0 -or -not $first) {
        Warn 'Could not read the card model and generation from nvidia-smi. Carrying on; the CUDA check after the install will tell.'
    } else {
        $fields = @($first -split ',' | ForEach-Object { $_.Trim() })
        $gpuName = $fields[0]
        Say "  Card: $gpuName"
        if ($fields.Count -ge 2 -and $fields[1] -match '^(\d+)\.(\d+)$') {
            $capMajor = [int]$Matches[1]; $capMinor = [int]$Matches[2]
            if ($capMajor -lt 7 -or ($capMajor -eq 7 -and $capMinor -lt 5)) {
                Stop-Install "$gpuName has compute capability $capMajor.$capMinor. The CUDA 13 PyTorch AuraDream installs supports 7.5 to 12.0 (sm_75 to sm_120): an RTX 20 or GTX 16 series card or newer. This card is not supported." 'AuraDream cannot run on this card. A newer NVIDIA card is needed.'
            } else {
                Ok "Compute capability $capMajor.$capMinor is supported (7.5 to 12.0)."
            }
        } else {
            Warn "Could not read the card's compute capability from '$first'. Carrying on; the CUDA check after the install will tell."
        }
        if ($fields.Count -ge 3 -and $fields[2] -match '^\d+$') {
            $vramGB = [math]::Round([int]$fields[2] / 1024, 1)
            if ($vramGB -ge 15.5) {
                Ok "$vramGB GB of video memory. The measured card is a 16 GB RTX 5080."
            } else {
                Note "$vramGB GB of video memory. Animagine XL 4.0 was measured on an 8 GB RTX 3070 under native Linux; the other measured card is a 16 GB RTX 5080 on Windows. This does not establish that every checkpoint fits 8 GB. The SDXL anime checkpoints are the lightest; sequential offload (under Models) uses less video memory, more slowly."
            }
        }
        $more = @($query.Text -split "`n" | Where-Object { $_ -match ',' }).Count
        if ($more -gt 1) { Note "$more NVIDIA cards found; AuraDream uses the first one listed." }
    }
}

# 3. Memory -------------------------------------------------------------------
Step 'Memory'
try {
    $ramBytes = (Get-CimInstance -ClassName Win32_ComputerSystem -ErrorAction Stop).TotalPhysicalMemory
    $ramGiB = [math]::Round($ramBytes / 1GB, 1)
    Say "  This machine has $ramGiB GiB of RAM."
    Say '  For comparison, the default checkpoint (FLUX.1 Krea [dev]) peaked at 27.9 GiB working set and 35.4 GiB'
    Say '  private memory while loading, and 39.9 GiB private during the first render, on a 64 GB machine.'
    if ($ramGiB -lt 40) {
        Note 'Less RAM than that has not been measured. Start with an SDXL checkpoint, such as Animagine XL 4.0, which needs far less.'
    } else {
        Ok 'That is at least what the measured machine used. Other checkpoints and settings may use more.'
    }
} catch {
    Warn "Could not read the amount of RAM ($($_.Exception.Message))."
}

# 4. uv and Git ---------------------------------------------------------------
Step 'uv and Git'
$winget = Find-Tool 'winget'
$uv = Find-Tool 'uv'
$git = Find-Tool 'git'
if ($winget) { Note 'winget is available for installing what is missing.' }

if ($uv) {
    $ver = Invoke-Capture $uv @('--version')
    Ok "uv is installed ($($ver.Text.Trim()))."
} elseif ($winget) {
    if ($DryRun) {
        Would 'install uv: winget install --id astral-sh.uv -e --accept-source-agreements --accept-package-agreements'
    } else {
        Say '  Installing uv with winget...'
        $code = Invoke-Live $winget @('install', '--id', 'astral-sh.uv', '-e', '--accept-source-agreements', '--accept-package-agreements')
        Update-SessionPath
        $uv = Find-Tool 'uv'
        if (-not $uv) {
            Stop-Install "winget could not install uv (exit $code)." "Install it from https://docs.astral.sh/uv/ (or run: winget install --id astral-sh.uv -e), then run install.cmd again."
        } else { Ok 'uv installed.' }
    }
} else {
    if ($DryRun) {
        Would "install uv with uv's own installer, version ${UvVersion}: powershell -NoProfile -ExecutionPolicy Bypass -Command `"irm $UvInstaller | iex`" (winget is not available here)"
    } else {
        Say "  winget is not available here, so installing uv $UvVersion with uv's own installer ($UvInstaller)..."
        $code = Invoke-Live 'powershell.exe' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', "irm $UvInstaller | iex")
        Update-SessionPath
        $uv = Find-Tool 'uv'
        if (-not $uv) {
            Stop-Install "uv's installer did not finish (exit $code). A proxy, firewall or antivirus may have blocked the download." 'Install uv by hand from https://docs.astral.sh/uv/getting-started/installation/ and run install.cmd again.'
        } else { Ok 'uv installed.' }
    }
}

if ($git) {
    $ver = Invoke-Capture $git @('--version')
    Ok "Git is installed ($($ver.Text.Trim()))."
} elseif ($winget) {
    if ($DryRun) {
        Would 'install Git: winget install --id Git.Git -e --scope user --accept-source-agreements --accept-package-agreements'
    } else {
        Say '  Installing Git with winget for your user only (used to get updates; no administrator approval)...'
        $code = Invoke-Live $winget @('install', '--id', 'Git.Git', '-e', '--scope', 'user', '--accept-source-agreements', '--accept-package-agreements')
        Update-SessionPath
        $git = Find-Tool 'git'
        if (-not $git) { Warn "winget could not install Git for your user (exit $code). AuraDream runs without it, but update.cmd needs it: install it from https://git-scm.com/download/win later." }
        else { Ok 'Git installed.' }
    }
} else {
    Warn 'Git is not installed and winget is not available to install it. AuraDream runs without Git; update.cmd needs it. Install it from https://git-scm.com/download/win and run install.cmd again to set up updates.'
}

# 5. Where the code came from -------------------------------------------------
Step 'Updates (Git checkout or ZIP download)'
$isCheckout = Test-Path -LiteralPath (Join-Path $Root '.git')
if ($isCheckout) {
    if ($git) {
        $origin = Invoke-Capture $git @('-C', $Root, 'remote', 'get-url', 'origin')
        $branch = Invoke-Capture $git @('-C', $Root, 'rev-parse', '--abbrev-ref', 'HEAD')
        Ok ("This folder is a Git checkout (branch {0}, from {1}), so update.cmd can update it." -f $branch.Text.Trim(), $origin.Text.Trim())
    } else {
        Ok 'This folder is a Git checkout; update.cmd will work once Git is installed.'
    }
} else {
    Note 'This folder has no .git: it came from a ZIP download, which cannot update itself.'
    Say '  Git keeps the code history and lets update.cmd fetch new versions without downloading another ZIP.'
    Say '  This folder can be turned into a Git checkout of the latest AuraDream. That downloads'
    Say '  the code history into a .git folder and makes the code files match the latest version. Your own'
    Say '  files (images, gallery, settings, LoRAs) are not part of the code and are left alone; any code file'
    Say '  that differs from the latest version is first copied to a zip-files-replaced-<date> folder here.'
    $convert = Ask-YesNo 'Turn this folder into a Git checkout so update.cmd works?' $true
    if (-not $convert) {
        Note 'Carrying on without updates. To update later, download the new ZIP (docs/setup.md, Updating).'
    } elseif (-not $git) {
        if ($DryRun) {
            Would "turn the folder into a Git checkout of $RepoUrl once Git is installed (git init, git remote add origin, git fetch origin main, git reset origin/main)"
        } else {
            Warn 'Git is not available, so the folder stays a ZIP download. Install Git and run install.cmd again to set up updates.'
        }
    } elseif ($DryRun) {
        Would "run: git init; git remote add origin $RepoUrl; git fetch origin main"
        Would 'run: git reset origin/main (moves Git to the latest version without touching any file on disk)'
        Would 'copy any code file that differs from the latest version to zip-files-replaced-<date>, then restore it from Git'
    } else {
        $gitDir = Join-Path $Root '.git'
        $failed = $null
        $code = Invoke-Live $git @('-C', $Root, 'init', '-q', '-b', 'main')
        if ($code -ne 0) { $failed = 'git init' }
        if (-not $failed) { $code = Invoke-Live $git @('-C', $Root, 'remote', 'add', 'origin', $RepoUrl); if ($code -ne 0) { $failed = 'git remote add' } }
        if (-not $failed) {
            Say '  Downloading the code history...'
            $code = Invoke-Live $git @('-C', $Root, 'fetch', 'origin', 'main')
            if ($code -ne 0) { $failed = 'git fetch (check the internet connection, proxy or firewall)' }
        }
        if (-not $failed) { $code = Invoke-Live $git @('-C', $Root, 'reset', '-q', 'origin/main'); if ($code -ne 0) { $failed = 'git reset' } }
        if ($failed) {
            # Nothing on disk was touched yet: drop the half-made .git so the folder is as it was.
            Remove-Item -LiteralPath $gitDir -Recurse -Force -ErrorAction SilentlyContinue
            Warn "Could not set up updates: $failed failed. The folder is unchanged and still works; run install.cmd again later to retry."
        } else {
            [void](Invoke-Live $git @('-C', $Root, 'branch', '-q', '--set-upstream-to=origin/main', 'main'))
            $diff = Invoke-Capture $git @('-C', $Root, 'diff', '--name-only')
            $changed = @($diff.Text -split "`n" | Where-Object { $_.Trim() })
            if ($changed.Count -gt 0) {
                $backup = Join-Path $Root ('zip-files-replaced-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
                foreach ($rel in $changed) {
                    $source = Join-Path $Root $rel
                    if (Test-Path -LiteralPath $source) {
                        $target = Join-Path $backup $rel
                        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
                        Copy-Item -LiteralPath $source -Destination $target -Force
                    }
                }
                [void](Invoke-Live $git @('-C', $Root, 'checkout', '--', '.'))
                Note "$($changed.Count) code file(s) differed from the latest version; the old copies are in $backup."
            }
            Ok 'This folder is now a Git checkout of the latest AuraDream; update.cmd can update it.'
            $isCheckout = $true
        }
    }
}

# 6. Where model downloads go -------------------------------------------------
Step 'Where model downloads go (HF_HOME)'
$defaultHf = Join-Path $env:USERPROFILE '.cache\huggingface'
$userHf = Get-UserVariable 'HF_HOME'
if ($env:HF_HOME) {
    Ok "HF_HOME is already set: model downloads go to $env:HF_HOME. Not asking again."
    $hfHome = $env:HF_HOME
} elseif ($userHf) {
    $env:HF_HOME = $userHf
    Ok "HF_HOME is already set for your user: model downloads go to $userHf. Not asking again."
    $hfHome = $userHf
} else {
    Say '  Checkpoint weights, converted copies, the optional prompt critic and your Hugging Face sign-in are'
    Say '  kept in one folder. Plan on roughly 45-50 GB free for the default checkpoint with the critic (an'
    Say '  estimate: about 32 GiB of weights, more for conversions), or several GB for an SDXL checkpoint.'
    Say '  Free space now:'
    try {
        Get-CimInstance -ClassName Win32_LogicalDisk -Filter 'DriveType=3' -ErrorAction Stop | ForEach-Object {
            Say ('    {0}  {1,6:N0} GB free of {2:N0} GB' -f $_.DeviceID, ($_.FreeSpace / 1e9), ($_.Size / 1e9))
        }
    } catch { Say '    (could not list the drives)' }
    $hfHome = $null
    while (-not $hfHome) {
        $answer = Ask-Text 'Folder for model downloads (Enter keeps the default)' $defaultHf
        if (-not (Test-DrivePath $answer)) {
            Say "  '$answer' is not a full path on a drive this machine has, such as D:\huggingface."
            if ($Yes) { $answer = $defaultHf } else { continue }
        }
        $hfHome = $answer.TrimEnd('\')
        # A bare drive root would leave 'D:', a path relative to the current
        # folder on D:; a folder of its own keeps the cache together.
        if ($hfHome -match '^[A-Za-z]:$') {
            $hfHome = "$hfHome\huggingface"
            Say "  A whole drive would mix the cache with everything else on it, so using $hfHome."
        }
    }
    $free = Get-FreeGB $hfHome
    if ($null -ne $free) { Say "  $free GB free on that drive." }
    if ($null -ne $free -and $free -lt 50) { Note 'That is less than the roughly 50 GB the default checkpoint with the critic needs; an SDXL checkpoint needs several GB.' }
    if ($hfHome -ieq $defaultHf) {
        Ok "Keeping the default, $defaultHf. Nothing to set."
    } else {
        if (Test-Path -LiteralPath (Join-Path $defaultHf 'token')) {
            Note 'You are signed in to Hugging Face in the default folder; after this change, sign in again (step 10 offers it).'
        }
        Set-UserVariable 'HF_HOME' $hfHome
    }
}

# 7. uv's package cache -------------------------------------------------------
Step "uv's package cache"
$userUvCache = Get-UserVariable 'UV_CACHE_DIR'
if (-not $env:UV_CACHE_DIR -and $userUvCache) { $env:UV_CACHE_DIR = $userUvCache }
$rootDrive = Split-Path -Qualifier $Root
if ($env:UV_CACHE_DIR) {
    $cacheDir = $env:UV_CACHE_DIR
    Ok "UV_CACHE_DIR is set: uv keeps its packages in $cacheDir."
} else {
    $cacheDir = Join-Path $env:LOCALAPPDATA 'uv\cache'
    $cacheDrive = Split-Path -Qualifier $cacheDir
    if ($cacheDrive -ieq $rootDrive) {
        Ok "uv's package cache ($cacheDir) is on the same drive as the project, so packages are linked, not copied."
    } else {
        Say "  uv keeps downloaded packages (several GB, mostly PyTorch) in $cacheDir on $cacheDrive. The project"
        Say "  is on $rootDrive, and uv cannot link files across drives, so it would copy them into .venv and the"
        Say '  space would be used twice. Setting UV_CACHE_DIR keeps the cache on the same drive as the project.'
        $suggested = "$rootDrive\uv-cache"
        if (Ask-YesNo "Keep uv's cache at $suggested?" $true) {
            Set-UserVariable 'UV_CACHE_DIR' $suggested
            $cacheDir = $suggested
        } else {
            Note 'Leaving the cache where it is; the packages will take their space twice.'
        }
    }
}
$freeRoot = Get-FreeGB $Root
if ($null -ne $freeRoot) {
    if ($freeRoot -lt 10) { Warn "Only $freeRoot GB free on $rootDrive. The packages need several GB (about 3 GB in .venv plus uv's cache)." }
    else { Ok "$freeRoot GB free on $rootDrive for the packages (about 3 GB in .venv plus uv's cache)." }
}

# 8. Packages and the CUDA check ----------------------------------------------
Step 'Install the packages (uv sync) and check CUDA'
$venvPython = Join-Path $Root '.venv\Scripts\python.exe'
$cudaCheck = "import torch; ok = torch.cuda.is_available(); print(ok, torch.cuda.get_device_name(0) if ok else '')"
if ($DryRun) {
    Would "run in ${Root}: uv sync (creates .venv and downloads a few GB, PyTorch from the CUDA 13 index; never model weights)"
} elseif (-not $uv) {
    Stop-Install 'uv is not available, so the packages cannot be installed.' 'Install uv (step 4) and run install.cmd again.'
} else {
    Say '  Installing the packages. The first time downloads a few GB; this can take a while.'
    Push-Location -LiteralPath $Root
    try { $code = Invoke-Live $uv @('sync') } finally { Pop-Location }
    if ($code -ne 0) {
        Stop-Install "uv sync failed (exit $code)." 'Check the internet connection. A proxy, firewall or antivirus scanning the download can break it; try again, or pause real-time scanning for the project folder. Run install.cmd again: it carries on where it stopped.'
    } else { Ok 'Packages installed.' }
}
if (Test-Path -LiteralPath $venvPython) {
    Say '  Checking that PyTorch sees the GPU...'
    $check = Invoke-Capture $venvPython @('-c', $cudaCheck)
    $out = $check.Text
    if ($check.Code -ne 0) {
        Stop-Install "The CUDA check could not run (exit $($check.Code)): $out" 'Run install.cmd again; if it repeats, delete the .venv folder and run it again.'
    } elseif ($out -match 'is not compatible with the current PyTorch') {
        Stop-Install 'PyTorch sees the card but says it is not compatible: the card is older than the CUDA 13 build supports (RTX 20 / GTX 16 series or newer).' 'A newer NVIDIA card is needed.'
    } elseif ($out -match '(?m)^True\s+(.+)$') {
        Ok "PyTorch can use the GPU: $($Matches[1].Trim())."
    } else {
        Stop-Install 'PyTorch cannot use the GPU (torch.cuda.is_available() is False). The driver is too old for CUDA 13, or not working.' "Update the NVIDIA driver from $DriverUrl, reboot, and run install.cmd again. Do not install the CUDA Toolkit."
    }
} elseif ($DryRun) {
    Would "check that PyTorch sees the GPU: .venv\Scripts\python.exe -c `"$cudaCheck`""
}

# 9. Doctor -------------------------------------------------------------------
Step 'Machine check (python -m backend.doctor)'
if (Test-Path -LiteralPath $venvPython) {
    Push-Location -LiteralPath $Root
    try { $code = Invoke-Live $venvPython @('-m', 'backend.doctor') } finally { Pop-Location }
    if ($code -ne 0) { Warn 'The check found a problem (FAIL above). AuraDream may not work until it is fixed; the line says what to do.' }
    else { Ok 'No FAIL rows. Read any WARN rows above; they may need a choice or a follow-up.' }
} else {
    Would 'run the read-only machine check: .venv\Scripts\python.exe -m backend.doctor'
}

# 10. Hugging Face ------------------------------------------------------------
Step 'Hugging Face sign-in (only for gated checkpoints)'
$hfEffective = $env:HF_HOME
if (-not $hfEffective) { $hfEffective = $defaultHf }
$wantGated = $false
# Where huggingface_hub looks, as the doctor does: HF_TOKEN (or the legacy
# HUGGING_FACE_HUB_TOKEN), then the token file (HF_TOKEN_PATH, or
# HF_HOME\token). Present only; never read.
$hfTokenSet = [bool]$env:HF_TOKEN -or [bool](Get-UserVariable 'HF_TOKEN') -or
    [bool]$env:HUGGING_FACE_HUB_TOKEN -or [bool](Get-UserVariable 'HUGGING_FACE_HUB_TOKEN')
$hfTokenFile = $env:HF_TOKEN_PATH
if (-not $hfTokenFile) { $hfTokenFile = Get-UserVariable 'HF_TOKEN_PATH' }
if (-not $hfTokenFile) { $hfTokenFile = Join-Path $hfEffective 'token' }
if ($hfTokenSet) {
    Ok 'A Hugging Face token is present (a token is set in HF_TOKEN or HUGGING_FACE_HUB_TOKEN).'
} elseif (Test-Path -LiteralPath $hfTokenFile -PathType Leaf) {
    Ok "A Hugging Face token is present (a token is saved in $(Split-Path -Parent $hfTokenFile))."
} else {
    Say '  The default checkpoint, FLUX.1 Krea [dev], is gated: it needs a free Hugging Face account, accepting'
    Say '  its licence (non-commercial) and signing in. Open checkpoints such as Animagine XL 4.0 need none,'
    Say '  are a much smaller download and lighter on memory: a good first start.'
    Say '  At the login menu, press Enter for "Log in with your browser", then open its URL and enter its code.'
    Say '  No copied token is needed. The fallback "Paste an access token" takes a Read token; pasted input stays hidden.'
    $wantGated = Ask-YesNo 'Set up the Hugging Face sign-in for the default checkpoint now?' $false
    if (-not $wantGated -and (Test-SavedChoices)) {
        Ok 'Skipped. AuraDream starts on the choices saved in its Models panel.'
    } elseif (-not $wantGated) {
        Ok "Skipped. AuraDream will start on an open checkpoint ($OpenModel)."
    } elseif ($DryRun) {
        Would 'open https://huggingface.co/join and https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev in the browser'
        Would 'run: uv run hf auth login (choose Log in with your browser; Hugging Face saves the sign-in)'
    } else {
        Say '  1. Create a free account (skip if you have one).'
        Start-Process 'https://huggingface.co/join'
        Wait-Enter 'When you have an account and are signed in'
        Say "  2. Accept the licence on the checkpoint's page."
        Start-Process "https://huggingface.co/$GatedModel"
        Wait-Enter 'When you have accepted it'
        Say '  3. Press Enter for "Log in with your browser" below, then follow the URL and code it shows.'
        Say '  If the code expires, run the login again for a new one. Hugging Face saves the sign-in.'
        Say '  To use "Paste an access token" instead, create a Read token at https://huggingface.co/settings/tokens.'
        Say '  Pasted input stays hidden. This script never sees your token.'
        Push-Location -LiteralPath $Root
        try { $code = Invoke-Live $uv @('run', 'hf', 'auth', 'login') } finally { Pop-Location }
        if ($code -ne 0) {
            Warn "The sign-in did not finish (exit $code)."
            $wantGated = $false
        } else { Ok 'Signed in to Hugging Face.' }
    }
    if (-not $wantGated) {
        Say "  To sign in later, open PowerShell in '$Root' and run: uv run hf auth login"
        Say '  Or stop AuraDream, double-click install.cmd in that folder again and answer y at the Hugging Face step (docs/setup.md step 6).'
    }
}
if ($hfTokenSet -or (Test-Path -LiteralPath $hfTokenFile -PathType Leaf)) {
    if (Test-SavedChoices) {
        Note 'The first start keeps the checkpoint already saved in Models.'
    } else {
        Say '  A token does not prove access to FLUX.1 Krea [dev]: accept its licence on Hugging Face first.'
        Say '  Animagine XL 4.0 is open, a smaller download and lighter on memory.'
        $gatedDefault = $null -ne $vramGB -and $vramGB -ge 15.5 -and $null -ne $ramGiB -and $ramGiB -ge 40
        if (-not $gatedDefault) { Note 'The open checkpoint is recommended: this PC has less than the measured default needs, or its memory could not be checked.' }
        $wantGated = Ask-YesNo 'Start on FLUX.1 Krea [dev] instead of Animagine XL 4.0?' $gatedDefault
    }
}

# 11. Shortcuts ---------------------------------------------------------------
Step 'Shortcuts'
# ZIP downloads can mark PowerShell launchers as internet files. Unblock
# only the app's own scripts, so RemoteSigned permits the PATH launchers.
foreach ($scriptName in @('install.ps1', 'update.ps1', 'bin\auradream.ps1', 'bin\auradream-wsl.ps1', 'bin\auradream-server.ps1')) {
    $scriptPath = Join-Path $Root $scriptName
    if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) { continue }
    if ($DryRun) { Would "remove the downloaded-file mark from $scriptPath (Unblock-File)" }
    else { Unblock-File -LiteralPath $scriptPath }
}
$launcher = Join-Path $Root 'bin\auradream.cmd'
$programs = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
if ($TestRoot) { $desktop = Join-Path $env:USERPROFILE 'Desktop' } else { $desktop = [Environment]::GetFolderPath('Desktop') }
$description = 'Start AuraDream and open it in the browser. To stop it safely, press Ctrl-C in its window (answer Y if asked); closing the window also stops any queued work.'

function New-AuraShortcut([string]$Path) {
    if ($DryRun) {
        Would "create the shortcut $Path -> `"$launcher`" --open (starts in $Root)"
        return
    }
    try {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
        $shell = New-Object -ComObject WScript.Shell
        $link = $shell.CreateShortcut($Path)
        $link.TargetPath = $launcher
        $link.Arguments = '--open'
        $link.WorkingDirectory = $Root
        $link.Description = $description
        $link.Save()
        Ok "Shortcut: $Path"
    } catch {
        Warn "Could not create the shortcut $Path ($($_.Exception.Message)). Start AuraDream with bin\auradream.cmd --open instead."
    }
}

New-AuraShortcut (Join-Path $programs 'AuraDream.lnk')
if (Ask-YesNo 'Also put a shortcut on the desktop?' $true) {
    New-AuraShortcut (Join-Path $desktop 'AuraDream.lnk')
}
Say '  The shortcut opens a window running AuraDream and then the browser. Keep the window open while you use'
Say '  it. To stop, press Ctrl-C in that window (answer Y if asked "Terminate batch job"); closing the window'
Say '  also stops it, and any queued work with it.'

# 12. Start -------------------------------------------------------------------
Step 'Start AuraDream'
# An explicit --model sets every saved choice aside for that start, so a
# re-run never passes one over choices already saved in Models.
$savedChoices = Test-SavedChoices
if ($wantGated -or $savedChoices) { $startArgs = @('--open') } else { $startArgs = @('--model', $OpenModel, '--open') }
if ($savedChoices) {
    Say ('  Start: bin\auradream.cmd {0} (it restores the choices saved in the Models panel)' -f ($startArgs -join ' '))
} else {
    Say ('  First start: bin\auradream.cmd {0}' -f ($startArgs -join ' '))
    Say '  On the first visit the Models panel opens. Press Switch & load: it downloads and loads the checkpoint'
    Say '  and saves the choice, so the shortcut comes back to it. The first download takes a while (32 GiB for'
    Say '  the default is about 45 minutes at 100 Mbit/s); the window shows the progress.'
}
if (Ask-YesNo 'Start AuraDream now?' $true) {
    if ($DryRun) {
        Would ('start "{0}" {1} in a new window' -f $launcher, ($startArgs -join ' '))
    } elseif (-not (Test-Path -LiteralPath $venvPython)) {
        Warn 'The packages are not installed, so AuraDream cannot start yet.'
    } else {
        Start-Process -FilePath $launcher -ArgumentList $startArgs -WorkingDirectory $Root
        Ok 'Started in a new window; the browser opens when it is ready.'
    }
} else {
    Say '  Start it later from the Start menu (AuraDream) or the desktop shortcut.'
}

Say ''
if ($DryRun) {
    if ($script:WouldStop.Count -gt 0) {
        Say 'Dry run finished. Nothing was changed. A real run would stop at:'
        foreach ($reason in $script:WouldStop) { Say "  - $reason" }
        exit 1
    }
    Say 'Dry run finished. Nothing was changed.'
    exit 0
}
Say 'AuraDream is installed. Update it later with update.cmd (stop AuraDream first).'
exit 0
