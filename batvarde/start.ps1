# Startar Båtvärde på Windows. Dubbelklicka start.bat, eller kör:
#   powershell -ExecutionPolicy Bypass -File start.ps1
# Gör allt som behövs första gången och hoppar över det nästa gång.

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Say  ($t) { Write-Host "`n$t" -ForegroundColor Cyan }
function Warn ($t) { Write-Host $t -ForegroundColor Yellow }

Say 'Båtvärde'

# --- Hämta senaste versionen -------------------------------------------------
# Hoppas över med BATVARDE_NO_UPDATE=1. --ff-only gör att egna ändringar aldrig
# skrivs över.

if ($env:BATVARDE_NO_UPDATE -ne '1' -and (Get-Command git -ErrorAction SilentlyContinue)) {
    $branch = (git rev-parse --abbrev-ref HEAD 2>$null)
    if ($LASTEXITCODE -eq 0 -and $branch) {
        git fetch origin $branch --quiet 2>$null
        if ($LASTEXITCODE -eq 0) {
            $behind = (git rev-list --count "HEAD..origin/$branch" 2>$null)
            if ($behind -and [int]$behind -gt 0) {
                git merge --ff-only "origin/$branch" --quiet 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "Uppdaterade till senaste versionen ($behind nya andringar)"
                    if (Test-Path .next) { Remove-Item .next -Recurse -Force }
                } else {
                    Warn "$behind nya andringar finns, men du har egna andringar. Kor som den ar."
                }
            }
        }
    }
}

# --- Krav --------------------------------------------------------------------

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Warn 'Node.js saknas. Installera fran https://nodejs.org (valj LTS) och kor igen.'
    Read-Host 'Tryck enter for att stanga' | Out-Null
    exit 1
}

# Versionen delas i PowerShell, inte i JavaScript: PowerShell plockar bort
# citattecken i argument till program, så node -p '...split(".")...' kom fram
# som split(.) och blev ett syntaxfel.
$nodeVersion = (node -p "process.versions.node")
$nodeMajor = [int]($nodeVersion.Split('.')[0])

if ($nodeMajor -lt 18) {
    Warn "Node $nodeMajor ar for gammal, minst 18 behovs. Uppdatera fran https://nodejs.org."
    Read-Host 'Tryck enter for att stanga' | Out-Null
    exit 1
}

# --- Förberedelser -----------------------------------------------------------

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host 'Skapade .env'
}

if (-not (Test-Path node_modules)) {
    Say 'Installerar paket - det har tar nagon minut forsta gangen'
    npm install
    if ($LASTEXITCODE -ne 0) { Warn 'npm install misslyckades.'; Read-Host | Out-Null; exit 1 }
}

if (-not (Test-Path 'data\boats.db')) {
    Say 'Skapar databasen'
    npx prisma db push --skip-generate
    npx prisma generate | Out-Null

    Say 'Laser in data\seed.csv'
    npm run db:seed
}

# --- Demodata ----------------------------------------------------------------
# data/seed.csv innehaller fran borjan bara fem EXEMPEL-rader. Med sa lite data
# ser skarmarna tomma ut, sa vi erbjuder demodatan - men bara efter en fraga,
# eftersom den ar pahittad.

$listings = 0
try {
    $listings = [int](node scripts\count-listings.mjs)
} catch { $listings = 0 }

if ($listings -lt 50 -and (Test-Path 'data\seed_demo.csv')) {
    Warn ''
    Warn "Bara $listings annonser i databasen. Skarmarna blir tunna med sa lite underlag."
    Warn 'data\seed_demo.csv innehaller 190 PAHITTADE annonser att bygga och testa med.'
    Warn 'De ar inte marknadsdata och ska aldrig visas for nagon utomstaende.'
    $answer = Read-Host 'Fylla pa med demodata? [j/N]'
    if ($answer -match '^[jJ]') {
        # Direkt mot tsx i stället för "npm run db:seed --": PowerShell behandlar
        # -- som slut på parametrar och argumentet efter kan tappas bort.
        npx tsx prisma/seed.ts data/seed_demo.csv
        if (Get-Command python -ErrorAction SilentlyContinue) {
            python pipeline\stats.py
        } else {
            Warn 'Python saknas - hoppar over pipeline/stats.py. Statistiken raknas anda av seed-skriptet.'
        }
    }
}

# --- Hitta en ledig port -----------------------------------------------------

function Test-PortFree([int]$port) {
    try {
        $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $port)
        $listener.Start()
        $listener.Stop()
        return $true
    } catch {
        return $false
    }
}

$port = 3000
if ($env:PORT) { $port = [int]$env:PORT }
while (-not (Test-PortFree $port) -and $port -lt 3010) { $port++ }
$url = "http://localhost:$port"

# --- Öppna webblasaren nar servern svarar ------------------------------------

Start-Job -ScriptBlock {
    param($u)
    for ($i = 0; $i -lt 60; $i++) {
        try {
            Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 2 | Out-Null
            Start-Process $u
            break
        } catch {
            Start-Sleep -Seconds 1
        }
    }
} -ArgumentList $url | Out-Null

Say "Startar pa $url - stang med Ctrl+C"
npx next dev -p $port
