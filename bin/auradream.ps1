# Start the AuraDream server (Windows venv) from any folder.
# PowerShell picks this over auradream.cmd on the PATH: with no cmd.exe
# in the chain, Ctrl-C stops the server without "Terminate batch job (Y/N)?".
# Arguments pass through to backend.main: --defaults, --menu, --model ...
Push-Location (Join-Path $PSScriptRoot '..')
try {
    & uv run python -m backend.main @args
    $rc = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $rc
