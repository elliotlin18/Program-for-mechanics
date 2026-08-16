@echo off
REM Installerar Batvarde och startar det. Dubbelklicka den har filen en gang.
REM Efter det ligger en genvag pa skrivbordet som startar programmet.
setlocal enabledelayedexpansion

set "REPO_URL=https://github.com/elliotlin18/Program-for-mechanics.git"
set "BRANCH=claude/batvarde-project-setup-ub31pi"
if "%BATVARDE_DIR%"=="" set "BATVARDE_DIR=%USERPROFILE%\Batvarde"
set "APP_DIR=%BATVARDE_DIR%\batvarde"

echo.
echo Installerar Batvarde
echo.

where git >nul 2>&1
if errorlevel 1 (
  echo Git saknas. Installera fran https://git-scm.com/download/win och kor om det har.
  pause
  exit /b 1
)

where node >nul 2>&1
if errorlevel 1 (
  echo Node.js saknas. Installera fran https://nodejs.org och kor om det har.
  pause
  exit /b 1
)

if exist "%BATVARDE_DIR%\.git" (
  echo Uppdaterar koden i %BATVARDE_DIR%
  git -C "%BATVARDE_DIR%" fetch origin %BRANCH% --quiet
  git -C "%BATVARDE_DIR%" checkout %BRANCH% --quiet
  git -C "%BATVARDE_DIR%" merge --ff-only origin/%BRANCH% --quiet
) else (
  echo Hamtar koden till %BATVARDE_DIR%
  git clone --branch %BRANCH% "%REPO_URL%" "%BATVARDE_DIR%" --quiet
)

REM Genvag pa skrivbordet
set "DESKTOP=%USERPROFILE%\Desktop"
if exist "%DESKTOP%" (
  powershell -NoProfile -Command ^
    "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%DESKTOP%\Batvarde.lnk');" ^
    "$s.TargetPath='%APP_DIR%\start.bat';" ^
    "$s.WorkingDirectory='%APP_DIR%';" ^
    "$s.Description='Vad gar baten for?';" ^
    "$s.Save()"
  echo Genvag skapad: %DESKTOP%\Batvarde.lnk
)

echo.
echo Startar
call "%APP_DIR%\start.bat"
