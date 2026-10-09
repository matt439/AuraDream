@echo off
rem Start the Local Diffusion server in WSL 2 (4-bit Nunchaku) from any folder.
rem Runs wsl/serve.sh in a login shell so ~/.local/bin (uv) is on PATH.
rem Arguments pass through to backend.main: --defaults, --menu, --model ...
setlocal
wsl.exe --cd "%~dp0.." -e bash -lc "exec bash wsl/serve.sh \"$@\"" local-diffusion-wsl %*
exit /b %ERRORLEVEL%
