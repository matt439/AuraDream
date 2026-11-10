@echo off
rem Start the AuraDream server (Windows venv) from any folder.
rem Arguments pass through to backend.main: --open, --defaults, --menu, --model ...
setlocal
pushd "%~dp0.." || exit /b 1
set "auradream_uv=uv"
where uv >nul 2>nul && goto run
if exist "%USERPROFILE%\.local\bin\uv.exe" (
  set "auradream_uv=%USERPROFILE%\.local\bin\uv.exe"
  goto run
)
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe" (
  set "auradream_uv=%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe"
  goto run
)
echo uv was not found. Run install.cmd from the AuraDream folder, then try the shortcut again.
set "rc=1"
goto finish
:run
"%auradream_uv%" run python -m backend.main %*
set "rc=%ERRORLEVEL%"
:finish
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
