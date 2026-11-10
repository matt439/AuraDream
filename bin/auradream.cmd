@echo off
rem Start the AuraDream server (Windows venv) from any folder.
rem Arguments pass through to backend.main: --open, --defaults, --menu, --model ...
setlocal
pushd "%~dp0.." || exit /b 1
uv run python -m backend.main %*
set "rc=%ERRORLEVEL%"
popd
rem A shortcut's window otherwise closes before a start-up refusal can be read.
rem Ctrl-C can be Python's 130 or Windows' signed/unsigned STATUS_CONTROL_C_EXIT.
if "%rc%"=="0" goto done
if "%rc%"=="130" goto done
if "%rc%"=="-1073741510" goto done
if "%rc%"=="3221225786" goto done
echo.
echo AuraDream exited with an error ^(code %rc%^). Read the message above.
echo Press any key to close this window.
pause >nul
:done
exit /b %rc%
