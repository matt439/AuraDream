@echo off
rem Update AuraDream: double-click this file, or run it from a terminal.
rem Stop AuraDream first. It runs update.ps1 with Windows PowerShell,
rem bypassing the execution policy for this one script only; arguments pass
rem through (-DryRun). See docs\setup.md, Updating.
rem
rem Everything from the PowerShell call on is one bracketed block, which
rem cmd.exe reads whole before running it: git pull may replace this file
rem while it runs, and cmd.exe would otherwise read on in the new file at
rem the old position.
setlocal
(
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0update.ps1" %*
  call set "rc=%%ERRORLEVEL%%"
  echo.
  rem Keep a double-clicked window open so its last lines can be read.
  if not defined AURADREAM_INSTALL_NO_PAUSE pause
  setlocal EnableDelayedExpansion
  exit /b !rc!
)
