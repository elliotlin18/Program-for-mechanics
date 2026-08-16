@echo off
REM Dubbelklicka den har filen for att starta Batvarde.
REM All logik ligger i start.ps1 - batch klarar inte portkoll och Read-Host ordentligt.
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"

if errorlevel 1 (
  echo.
  echo Nagot gick fel. Felmeddelandet star ovanfor.
  pause
)
