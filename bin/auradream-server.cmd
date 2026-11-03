@echo off
rem Update and control a Linux AuraDream server over ssh, from any folder.
rem The work is auradream-server.ps1's (PowerShell picks that one on the PATH);
rem this runs it for cmd.exe, arguments passed through: status, -Force ...
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0auradream-server.ps1" %*
exit /b %ERRORLEVEL%
