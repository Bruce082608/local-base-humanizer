@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0studio_ctl.ps1" stop
if /I not "%~1"=="--quiet" timeout /t 2 /nobreak >nul
