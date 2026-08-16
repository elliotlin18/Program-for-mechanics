@echo off
REM Dubbelklicka den har filen i Utforskaren for att starta Batvarde.
setlocal
cd /d "%~dp0"

echo.
echo Batvarde
echo.

where node >nul 2>&1
if errorlevel 1 (
  echo Node.js saknas. Installera fran https://nodejs.org och kor igen.
  pause
  exit /b 1
)

if not exist .env copy .env.example .env >nul

if not exist node_modules (
  echo Installerar paket - det har tar nagon minut forsta gangen
  call npm install
)

if not exist data\boats.db (
  echo Skapar databasen
  call npx prisma db push --skip-generate
  call npx prisma generate >nul
  echo Laser in data\seed.csv
  call npm run db:seed
  echo.
  echo data\seed.csv innehaller fran borjan bara fem EXEMPEL-rader.
  echo data\seed_demo.csv innehaller 190 PAHITTADE annonser att bygga och testa med.
  echo De ar inte marknadsdata och ska aldrig visas for nagon utomstaende.
  set /p ANSWER="Fylla pa med demodata? [j/N] "
  if /i "%ANSWER%"=="j" (
    call npm run db:seed -- data/seed_demo.csv
    python pipeline\stats.py
  )
)

set PORT=3000
echo.
echo Startar pa http://localhost:%PORT% - stang med Ctrl+C
start "" http://localhost:%PORT%
call npx next dev -p %PORT%
