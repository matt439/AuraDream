@echo off
rem Install AuraDream: double-click this file, or run it from a terminal.
rem It runs install.ps1 with Windows PowerShell, bypassing the execution
rem policy for this one script only. Arguments pass through (-DryRun, -Yes).
rem Safe to run again; see docs\setup.md, Easy install.
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set "rc=%ERRORLEVEL%"
echo.
rem Keep a double-clicked window open so its last lines can be read.
if not defined AURADREAM_INSTALL_NO_PAUSE pause
exit /b %rc%
