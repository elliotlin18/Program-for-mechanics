@echo off
REM Dubbelklicka den har filen for att installera och starta Batvarde.
REM All logik ligger i install.ps1.
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"

if errorlevel 1 (
  echo.
  echo Nagot gick fel. Felmeddelandet star ovanfor.
  pause
)
