@echo off
chcp 65001 >nul
set "EXTRA="
if /I "%~1"=="--no-browser" set "EXTRA=-NoBrowser"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0studio_ctl.ps1" start %EXTRA%
if errorlevel 1 (
  pause
) else (
  if "%EXTRA%"=="" timeout /t 2 /nobreak >nul
)
