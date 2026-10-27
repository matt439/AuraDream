# Start the AuraDream server in WSL 2 (4-bit Nunchaku) from any folder.
# PowerShell picks this over auradream-wsl.cmd on the PATH: with no
# cmd.exe in the chain, Ctrl-C stops the server without "Terminate batch job".
# A login shell puts ~/.local/bin (uv) on PATH; no quoted bash -c string, which
# Windows PowerShell 5.1 would pass to wsl.exe unescaped.
# Arguments pass through to backend.main: --defaults, --menu, --model ...
& wsl.exe --cd (Join-Path $PSScriptRoot '..') -e bash --login wsl/serve.sh @args
exit $LASTEXITCODE
