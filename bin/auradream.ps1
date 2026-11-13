# Start the AuraDream server (Windows venv) from any folder.
# PowerShell picks this over auradream.cmd on the PATH: with no cmd.exe
# in the chain, Ctrl-C stops the server without "Terminate batch job (Y/N)?".
# Arguments pass through to backend.main: --open, --defaults, --menu, --model ...
# uv is looked up as auradream.cmd looks it up (PATH, then ~\.local\bin, then
# WinGet's Links): a window opened before install.cmd put it on PATH.
$uv = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if ($uv) {
    $uv = $uv.Source
} else {
    $uv = $null
    foreach ($dir in @(
        $(if ($env:USERPROFILE) { Join-Path $env:USERPROFILE '.local\bin' }),
        $(if ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links' })
    )) {
        if ($dir -and (Test-Path -LiteralPath (Join-Path $dir 'uv.exe'))) {
            $uv = Join-Path $dir 'uv.exe'
            break
        }
    }
}
if (-not $uv) {
    Write-Host 'uv was not found. Run install.cmd from the AuraDream folder, then try again.'
    exit 1
}
$rc = 1
Push-Location (Join-Path $PSScriptRoot '..')
try {
    & $uv run python -m backend.main @args
    # A run that never set an exit code still counts as a failure.
    if ($null -ne $LASTEXITCODE) { $rc = $LASTEXITCODE }
} finally {
    Pop-Location
}
exit $rc
