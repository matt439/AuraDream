# Update and control a Linux AuraDream server (linux/, a systemd user
# service) from this Windows PC, over ssh:
#
#     auradream-server status            # the service, the checkout, the server's own status
#     auradream-server start | stop | restart [-Force]
#     auradream-server update [-DryRun]  # bash linux/update.sh --restart (or --dry-run)
#     auradream-server logs [-Follow] [-Lines N]
#     auradream-server doctor            # the read-only machine check, on the server
#     auradream-server connect [-Command '<cmd>']   # a shell in the checkout, or one command
#     auradream-server open              # the page in this PC's browser
#
# Settings come from environment variables only (read by this script, never
# by the app): AURADREAM_SERVER (user@host, required), AURADREAM_SERVER_DIR
# (the checkout there, default ~/AuraDream), AURADREAM_SERVER_PORT (default
# 7860) and AURADREAM_SERVER_WAKE (an http(s) URL to POST, or a MAC address
# for a Wake-on-LAN packet). Each command but help first wakes the server
# when its ssh port does not answer. docs/setup.md, "From another PC".
#
# Test seam (tests/test_install_scripts.py): AURADREAM_INSTALL_TEST_NO_BROWSER
# makes open print the address without starting a browser. Dot-sourced, the
# script defines its functions and runs nothing.
param(
    [Parameter(Position = 0)][string]$Action = '',
    [switch]$Force,
    [switch]$DryRun,
    [switch]$Follow,
    [int]$Lines = 0,
    [string]$Command = $null,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest = @()
)

$ErrorActionPreference = 'Stop'
# The script's own options (inside a function, $PSBoundParameters is the function's).
$Bound = $PSBoundParameters

$Usage = @'
Usage: auradream-server <command> [options]

Commands (each runs on the server, over ssh, from its checkout):
  status                 the service, the checkout, the server's own status
  start                  start the service and wait for the page
  stop [-Force]          stop it; refused while it is busy unless -Force
  restart [-Force]       stop it (same check), start it, wait for the page
  update [-DryRun]       bash linux/update.sh --restart (-DryRun: --dry-run)
  logs [-Follow] [-Lines N]   its log, the last 100 lines by default
  doctor                 the read-only machine check, with the service's settings
  connect [-Command 'cmd']    a shell in the checkout, or one command there
  open                   open the page in this PC's browser
  help                   this text

Settings (environment variables):
  AURADREAM_SERVER       required: user@host (or host), e.g. me@my-box
  AURADREAM_SERVER_DIR   the checkout on the server (default ~/AuraDream)
  AURADREAM_SERVER_PORT  the server's port (default 7860)
  AURADREAM_SERVER_WAKE  optional: a URL to POST to wake it
                         (e.g. http://192.168.1.5:8650/__wake), or its MAC
                         address for a Wake-on-LAN packet (aa:bb:cc:dd:ee:ff)
'@

# BatchMode: never stop to ask for a password; keepalives so a long run
# notices a dropped link.
$SshOptions = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6')

function Write-Problem([string]$Text) {
    [Console]::Error.WriteLine($Text)
}

function Stop-WithUsage([string]$Text) {
    Write-Problem $Text
    Write-Problem ''
    Write-Problem $Usage
    exit 2
}

function ConvertTo-BashWord([string]$Text) {
    # One argument for the remote bash, single-quoted. Windows PowerShell 5.1
    # mangles double quotes passed to ssh.exe, so those are refused rather
    # than sent wrong.
    if ($Text.Contains('"')) { throw "Arguments with double quotes can't be sent safely over ssh from Windows PowerShell: $Text" }
    return "'" + $Text.Replace("'", "'\''") + "'"
}

function ConvertTo-BashPath([string]$Path) {
    # A folder for the remote cd: a leading ~ stays bare (bash expands it
    # only unquoted), the rest is one quoted word.
    if ($Path -eq '~') { return '~' }
    if ($Path.StartsWith('~/')) {
        $rest = $Path.Substring(2)
        if ($rest -eq '') { return '~/' }
        return '~/' + (ConvertTo-BashWord $rest)
    }
    return ConvertTo-BashWord $Path
}

function Get-WakeSetting([string]$Value) {
    # $null when unset; @{ Kind = 'url' | 'mac'; Value = ... }; throws on
    # anything else.
    if ([string]::IsNullOrWhiteSpace($Value)) { return $null }
    $text = $Value.Trim()
    $uri = $null
    if ($text -match '^(?i)https?://' -and [Uri]::TryCreate($text, [UriKind]::Absolute, [ref]$uri)) {
        return @{ Kind = 'url'; Value = $uri.AbsoluteUri }
    }
    if ($text -match '^[0-9A-Fa-f]{2}(:[0-9A-Fa-f]{2}){5}$' -or $text -match '^[0-9A-Fa-f]{2}(-[0-9A-Fa-f]{2}){5}$') {
        return @{ Kind = 'mac'; Value = $text }
    }
    throw "AURADREAM_SERVER_WAKE is neither an http(s) URL nor a MAC address: $Value (for example http://192.168.1.5:8650/__wake or aa:bb:cc:dd:ee:ff)"
}

function New-MagicPacket([string]$Mac) {
    # Wake-on-LAN: six 0xFF bytes, then the MAC address sixteen times (102 bytes).
    $hex = $Mac -replace '[:-]', ''
    if ($hex -notmatch '^[0-9A-Fa-f]{12}$') { throw "Not a MAC address: $Mac" }
    $address = New-Object byte[] 6
    for ($i = 0; $i -lt 6; $i++) { $address[$i] = [Convert]::ToByte($hex.Substring($i * 2, 2), 16) }
    $packet = New-Object byte[] 102
    for ($i = 0; $i -lt 6; $i++) { $packet[$i] = 0xFF }
    for ($i = 0; $i -lt 16; $i++) { [Array]::Copy($address, 0, $packet, 6 + $i * 6, 6) }
    return , $packet
}

function Send-Wake($Wake) {
    if ($Wake.Kind -eq 'url') {
        try {
            Invoke-WebRequest -Method Post -Uri $Wake.Value -UseBasicParsing -TimeoutSec 5 | Out-Null
        } catch {
            Write-Problem "  [warn] The wake address did not take the request: $($_.Exception.Message)"
        }
        return
    }
    $packet = New-MagicPacket $Wake.Value
    $udp = New-Object Net.Sockets.UdpClient
    try {
        $udp.EnableBroadcast = $true
        [void]$udp.Send($packet, $packet.Length, '255.255.255.255', 9)
    } catch {
        Write-Problem "  [warn] The Wake-on-LAN packet could not be sent: $($_.Exception.Message)"
    } finally {
        $udp.Close()
    }
}

function Get-SshTarget([string]$Ssh, [string]$Server) {
    # The host name and port ssh itself would connect to (ssh -G applies
    # ~/.ssh/config, so a Host alias works); else the part after @ and 22.
    $name = ($Server -split '@')[-1]
    $target = @{ Host = $name; Port = 22 }
    $previous = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $config = & $Ssh -G $Server 2>$null
        if ($LASTEXITCODE -eq 0) {
            foreach ($line in $config) {
                if ($line -match '^hostname\s+(\S+)$') { $target.Host = $Matches[1] }
                elseif ($line -match '^port\s+(\d+)$') { $target.Port = [int]$Matches[1] }
            }
        }
    } catch {
    } finally {
        $ErrorActionPreference = $previous
    }
    return $target
}

function Test-SshPort($Target, [int]$Milliseconds = 3000) {
    # Whether the server's sshd answers within a few seconds.
    $client = New-Object Net.Sockets.TcpClient
    try {
        $connect = $client.BeginConnect($Target.Host, $Target.Port, $null, $null)
        if (-not $connect.AsyncWaitHandle.WaitOne($Milliseconds)) { return $false }
        $client.EndConnect($connect)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Wait-ServerAwake($Target, $Wake, [string]$Server, [int]$TimeoutSeconds = 180) {
    # Wakes the server if its ssh port does not answer, and waits for it. If
    # it never answers, says so and lets ssh try (its own timeout decides).
    if (Test-SshPort $Target) { return }
    $where = "$($Target.Host) port $($Target.Port)"
    if ($null -eq $Wake) {
        Write-Host "$Server is not answering ($where), and AURADREAM_SERVER_WAKE is not set to wake it; trying ssh anyway."
        return
    }
    $how = 'a Wake-on-LAN packet'
    if ($Wake.Kind -eq 'url') { $how = $Wake.Value }
    Write-Host "$Server is not answering ($where): waking it ($how) and waiting up to $TimeoutSeconds s..."
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $nextWake = Get-Date
    while ((Get-Date) -lt $deadline) {
        if ((Get-Date) -ge $nextWake) {
            Send-Wake $Wake
            $nextWake = (Get-Date).AddSeconds(15)
        }
        Start-Sleep -Seconds 2
        if (Test-SshPort $Target) {
            Write-Host 'The server is awake.'
            return
        }
    }
    Write-Host "The server did not answer within $TimeoutSeconds s (is it on?); trying ssh anyway."
}

function Invoke-Ssh([string]$Ssh, [string[]]$Arguments) {
    # Runs ssh with the console reading UTF-8 (the server writes it), then
    # puts the console back; $LASTEXITCODE is ssh's afterwards. Called as a
    # statement, never assigned, so ssh writes straight to this console (a
    # shell or journalctl -f stays interactive).
    $encoding = $null
    try { $encoding = [Console]::OutputEncoding; [Console]::OutputEncoding = New-Object Text.UTF8Encoding $false } catch { $encoding = $null }
    try {
        & $Ssh @Arguments
    } finally {
        if ($null -ne $encoding) { try { [Console]::OutputEncoding = $encoding } catch { } }
    }
}

function Get-RemoteCommand([string]$Directory, [string]$Script, [string[]]$Words) {
    # cd into the checkout, then bash linux/<script> with quoted words; a
    # missing folder or script is said plainly (127 for the script).
    $cd = ConvertTo-BashPath $Directory
    $quoted = @($Words | ForEach-Object { ConvertTo-BashWord $_ })
    $run = (@('bash', "linux/$Script") + $quoted) -join ' '
    return "cd $cd || exit 2; [ -f linux/$Script ] || { echo 'auradream-server: this checkout has no linux/$Script' >&2; exit 127; }; $run"
}

function Invoke-Main {
    $action = $Action.ToLowerInvariant()
    if ($Rest.Count -gt 0) { Stop-WithUsage "Unexpected argument(s): $($Rest -join ' ')" }
    $known = @('status', 'start', 'stop', 'restart', 'update', 'logs', 'doctor', 'connect', 'open', 'help')
    if ($action -eq '' -or $action -eq 'help' -or $action -eq '-h' -or $action -eq '--help') {
        Write-Host $Usage
        exit 0
    }
    if ($known -notcontains $action) { Stop-WithUsage "Unknown command: $Action" }
    if ($Force -and @('stop', 'restart') -notcontains $action) { Stop-WithUsage '-Force is for stop and restart only.' }
    if ($DryRun -and $action -ne 'update') { Stop-WithUsage '-DryRun is for update only.' }
    if (($Follow -or $Bound.ContainsKey('Lines')) -and $action -ne 'logs') { Stop-WithUsage '-Follow and -Lines are for logs only.' }
    if ($Bound.ContainsKey('Command') -and $action -ne 'connect') { Stop-WithUsage '-Command is for connect only.' }
    if ($Bound.ContainsKey('Lines') -and $Lines -lt 1) { Stop-WithUsage '-Lines takes a whole number of 1 or more.' }

    $server = $env:AURADREAM_SERVER
    if ([string]::IsNullOrWhiteSpace($server)) {
        Write-Problem 'AURADREAM_SERVER is not set: it names the Linux server to control, as user@host (or host).'
        Write-Problem "Set it for this window with: `$env:AURADREAM_SERVER = 'me@my-box'"
        Write-Problem "or for your user with: setx AURADREAM_SERVER me@my-box (then open a new window)."
        exit 2
    }
    $server = $server.Trim()
    if ($server -match '[\s"''`]' -or $server.StartsWith('-')) {
        Write-Problem "AURADREAM_SERVER is not a user@host: $server"
        exit 2
    }
    $directory = $env:AURADREAM_SERVER_DIR
    if ([string]::IsNullOrWhiteSpace($directory)) { $directory = '~/AuraDream' }
    $directory = $directory.Trim()
    $portText = $env:AURADREAM_SERVER_PORT
    $port = 7860
    if (-not [string]::IsNullOrWhiteSpace($portText)) {
        if ($portText.Trim() -notmatch '^\d{1,5}$' -or [int]$portText.Trim() -lt 1 -or [int]$portText.Trim() -gt 65535) {
            Write-Problem "AURADREAM_SERVER_PORT is not a port number: $portText"
            exit 2
        }
        $port = [int]$portText.Trim()
    }
    try {
        $wake = Get-WakeSetting $env:AURADREAM_SERVER_WAKE
    } catch {
        Write-Problem $_.Exception.Message
        exit 2
    }

    # The words are checked before anything wakes or connects.
    try {
        switch ($action) {
            'update' {
                $flag = '--restart'
                if ($DryRun) { $flag = '--dry-run' }
                $remote = Get-RemoteCommand $directory 'update.sh' @($flag)
            }
            'connect' {
                $cd = ConvertTo-BashPath $directory
                if ($Bound.ContainsKey('Command') -and -not [string]::IsNullOrWhiteSpace($Command)) {
                    if ($Command.Contains('"')) { throw "Use single quotes inside -Command: Windows PowerShell 5.1 mangles double quotes passed to ssh." }
                    $remote = "cd $cd && $Command"
                } else {
                    $remote = "cd $cd && exec bash -l"
                }
            }
            'open' { $remote = $null }
            default {
                $words = @($action)
                if ($Force) { $words += '--force' }
                if ($action -eq 'logs') {
                    if ($Follow) { $words += '--follow' }
                    if ($Bound.ContainsKey('Lines')) { $words += @('--lines', [string]$Lines) }
                }
                $remote = Get-RemoteCommand $directory 'control.sh' $words
            }
        }
        # A trailing backslash would escape the closing quote 5.1 adds.
        if ($null -ne $remote -and $remote.EndsWith('\')) { throw 'The command ends with a backslash, which cannot be sent safely over ssh from Windows PowerShell.' }
    } catch {
        Write-Problem $_.Exception.Message
        exit 2
    }

    $sshCommand = Get-Command ssh -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $sshCommand) {
        Write-Problem 'ssh is not installed: add the Windows OpenSSH Client (Settings > System > Optional features), then open a new window.'
        exit 2
    }
    $ssh = $sshCommand.Source

    $target = Get-SshTarget $ssh $server
    Wait-ServerAwake $target $wake $server

    if ($action -eq 'open') {
        $url = "http://$($target.Host):$port/"
        Write-Host "Opening $url"
        if (-not $env:AURADREAM_INSTALL_TEST_NO_BROWSER) { Start-Process $url }
        exit 0
    }

    $arguments = @()
    if ($action -eq 'connect') {
        # Interactive: a password prompt may be wanted, so no BatchMode.
        $arguments = @('-t', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6', $server, $remote)
    } elseif ($action -eq 'logs' -and $Follow) {
        # A terminal, so Ctrl-C reaches journalctl on the server.
        $arguments = @('-t') + $SshOptions + @($server, $remote)
    } else {
        $arguments = $SshOptions + @($server, $remote)
    }
    Invoke-Ssh $ssh $arguments
    $code = $LASTEXITCODE
    if ($code -eq 255) {
        Write-Problem ''
        Write-Problem "ssh could not reach $server (exit 255). Check AURADREAM_SERVER (user@host), that the server is on and"
        Write-Problem 'answering, and that this PC''s key is in the server''s ~/.ssh/authorized_keys (ssh-keygen, then add'
        Write-Problem 'the .pub file there): these commands use ssh with BatchMode, which never asks for a password.'
    } elseif ($code -eq 127 -and $action -ne 'connect') {
        Write-Problem ''
        if ($action -eq 'update') {
            Write-Problem "The checkout on the server ($directory) has no linux/update.sh: it is older than the Linux scripts."
            Write-Problem 'Update it once by hand: auradream-server connect -Command ''git pull --ff-only && uv sync'''
        } else {
            Write-Problem "The checkout on the server ($directory) has no linux/control.sh: it is an older version."
            Write-Problem 'auradream-server update brings it (or set AURADREAM_SERVER_DIR to the checkout).'
        }
    }
    exit $code
}

# Dot-sourced (the tests' packet check): the functions only.
if ($MyInvocation.InvocationName -eq '.') { return }
Invoke-Main
