@echo off
rem Update AuraDream: double-click this file, or run it from a terminal.
rem Stop AuraDream first. It runs update.ps1 with Windows PowerShell,
rem bypassing the execution policy for this one script only; arguments pass
rem through (-DryRun). See docs\setup.md, Updating.
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0update.ps1" %*
set "rc=%ERRORLEVEL%"
echo.
rem Keep a double-clicked window open so its last lines can be read.
if not defined AURADREAM_INSTALL_NO_PAUSE pause
exit /b %rc%
