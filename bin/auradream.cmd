@echo off
rem Start the Auradream server (Windows venv) from any folder.
rem Arguments pass through to backend.main: --defaults, --menu, --model ...
setlocal
pushd "%~dp0.." || exit /b 1
uv run python -m backend.main %*
set "rc=%ERRORLEVEL%"
popd
exit /b %rc%
