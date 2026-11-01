@echo off
rem Install AuraDream: double-click this file, or run it from a terminal.
rem It runs install.ps1 with Windows PowerShell, bypassing the execution
rem policy for this one script only. Arguments pass through (-DryRun, -Yes).
rem Safe to run again; see docs\setup.md, Easy install.
rem
rem Everything from the PowerShell call on is one bracketed block, which
rem cmd.exe reads whole before running it: turning a ZIP folder into a Git
rem checkout may replace this file while it runs, and cmd.exe would
rem otherwise read on in the new file at the old position.
setlocal
(
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
  call set "rc=%%ERRORLEVEL%%"
  echo.
  rem Keep a double-clicked window open so its last lines can be read.
  if not defined AURADREAM_INSTALL_NO_PAUSE pause
  setlocal EnableDelayedExpansion
  exit /b !rc!
)
