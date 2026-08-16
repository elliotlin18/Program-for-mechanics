# Installerar Båtvärde på Windows och startar det. Kör en gång i PowerShell:
#
 iex$| iex
#
# Efter det ligger en genväg på skrivbordet som startar programmet med ett dubbelklick.
# Byt installationsmapp med $env:BATVARDE_DIR.

$ErrorActionPreference = 'Stop'

$repoUrl = 'https://github.com/elliotlin18/Program-for-mechanics.git'
$branch  = 'claude/batvarde-project-setup-ub31pi'
$installDir = if ($env:BATVARDE_DIR) { $env:BATVARDE_DIR } else { Join-Path $env:USERPROFILE 'Batvarde' }
$appDir = Join-Path $installDir 'batvarde'

function Say  ($t) { Write-Host "`n$t" -ForegroundColor Cyan }
function Warn ($t) { Write-Host $t -ForegroundColor Yellow }

Say 'Installerar Batvarde'

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Warn 'Git saknas. Installera fran https://git-scm.com/download/win och kor om det har.'
    Read-Host 'Tryck enter for att stanga' | Out-Null
    exit 1
}

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Warn 'Node.js saknas. Installera LTS fran https://nodejs.org och kor om det har.'
    Read-Host 'Tryck enter for att stanga' | Out-Null
    exit 1
}

# --- Hamta eller uppdatera koden ---------------------------------------------

if (Test-Path (Join-Path $installDir '.git')) {
    Say "Uppdaterar koden i $installDir"
    git -C $installDir fetch origin $branch --quiet
    git -C $installDir checkout $branch --quiet
    git -C $installDir merge --ff-only "origin/$branch" --quiet
    if ($LASTEXITCODE -ne 0) {
        Warn 'Kunde inte uppdatera automatiskt - du har egna andringar. Koden lamnas som den ar.'
    }
} else {
    Say "Hamtar koden till $installDir"
    git clone --branch $branch $repoUrl $installDir --quiet
    if ($LASTEXITCODE -ne 0) { Warn 'git clone misslyckades.'; Read-Host | Out-Null; exit 1 }
}

# --- Genvag pa skrivbordet ---------------------------------------------------

$desktop = [Environment]::GetFolderPath('Desktop')
if ($desktop -and (Test-Path $desktop)) {
    $linkPath = Join-Path $desktop 'Batvarde.lnk'
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($linkPath)
    $shortcut.TargetPath = Join-Path $appDir 'start.bat'
    $shortcut.WorkingDirectory = $appDir
    $shortcut.Description = 'Vad gar baten for?'
    $shortcut.Save()
    Say "Genvag skapad: $linkPath"
} else {
    Warn "Hittade ingen skrivbordsmapp. Starta med: $appDir\start.bat"
}

Say 'Startar'
& (Join-Path $appDir 'start.bat')
