#!/usr/bin/env bash
# Startar Båtvärde. Dubbelklicka start.command (Mac) eller kör ./start.sh.
# Gör allt som behövs första gången och hoppar över det nästa gång.

set -euo pipefail
cd "$(dirname "$0")"

say() { printf "\n\033[1m%s\033[0m\n" "$1"; }
warn() { printf "\033[33m%s\033[0m\n" "$1"; }

say "Båtvärde"

# --- Hämta senaste versionen --------------------------------------------------
# Hoppas över med BATVARDE_NO_UPDATE=1. --ff-only gör att egna ändringar aldrig
# skrivs över – då lämnas koden som den är.

if [ "${BATVARDE_NO_UPDATE:-0}" != "1" ] && git rev-parse --git-dir >/dev/null 2>&1; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
  if [ -n "$BRANCH" ] && git remote get-url origin >/dev/null 2>&1; then
    if git fetch origin "$BRANCH" --quiet 2>/dev/null; then
      BEHIND="$(git rev-list --count "HEAD..origin/$BRANCH" 2>/dev/null || echo 0)"
      if [ "$BEHIND" -gt 0 ]; then
        if git merge --ff-only "origin/$BRANCH" --quiet 2>/dev/null; then
          echo "Uppdaterade till senaste versionen ($BEHIND nya ändringar)"
          rm -rf .next
        else
          warn "$BEHIND nya ändringar finns, men du har egna ändringar. Kör som den är."
        fi
      fi
    fi
  fi
fi

# --- Krav ---------------------------------------------------------------------

if ! command -v node >/dev/null 2>&1; then
  warn "Node.js saknas. Installera från https://nodejs.org (välj LTS) och kör igen."
  read -r -p "Tryck enter för att stänga." _
  exit 1
fi

NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
if [ "$NODE_MAJOR" -lt 18 ]; then
  warn "Node $NODE_MAJOR är för gammal, minst 18 behövs. Uppdatera från https://nodejs.org."
  read -r -p "Tryck enter för att stänga." _
  exit 1
fi

# --- Förberedelser ------------------------------------------------------------

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Skapade .env"
fi

if [ ! -d node_modules ]; then
  say "Installerar paket – det här tar någon minut första gången"
  npm install
fi

if [ ! -f data/boats.db ]; then
  say "Skapar databasen"
  npx prisma db push --skip-generate
  npx prisma generate >/dev/null

  say "Läser in data/seed.csv"
  npm run db:seed
fi

LISTINGS="$(node scripts/count-listings.mjs 2>/dev/null || echo 0)"

# data/seed.csv innehåller från början bara fem EXEMPEL-rader. Med så lite data
# ser skärmarna tomma ut, så vi erbjuder demodatan – men bara efter en fråga,
# eftersom den är påhittad.
if [ "$LISTINGS" -lt 50 ] && [ -f data/seed_demo.csv ]; then
  warn ""
  warn "Bara $LISTINGS annonser i databasen. Skärmarna blir tunna med så lite underlag."
  warn "data/seed_demo.csv innehåller 190 PÅHITTADE annonser att bygga och testa med."
  warn "De är inte marknadsdata och ska aldrig visas för någon utomstående."
  # Läs från terminalen, inte stdin. Startas skriptet via "curl | bash" är stdin
  # pipen med skriptet i, och frågan skulle besvaras med tystnad. Att /dev/tty
  # går att testa med -r betyder inte att den går att öppna, så vi provar.
  # Testet görs i en subshell: en misslyckad omdirigering skriver sitt felmeddelande
  # innan ett 2>/dev/null på samma rad hinner gälla.
  ANSWER="n"
  if ( : < /dev/tty ) 2>/dev/null; then
    printf "Fylla på med demodata? [j/N] "
    read -r ANSWER < /dev/tty || ANSWER="n"
  else
    warn "Ingen terminal att fråga i – hoppar över demodatan."
    warn "Kör 'npm run db:seed -- data/seed_demo.csv' när du vill ha den."
  fi

  case "$ANSWER" in
    [jJ]*)
      npm run db:seed -- data/seed_demo.csv
      python3 pipeline/stats.py 2>/dev/null || true
      ;;
  esac
fi

# --- Hitta en ledig port ------------------------------------------------------

PORT="${PORT:-3000}"
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if node -e "
    const net=require('net'), s=net.createServer();
    s.once('error',()=>process.exit(1));
    s.once('listening',()=>{s.close(()=>process.exit(0))});
    s.listen(Number(process.argv[1]),'127.0.0.1');
  " "$PORT" >/dev/null 2>&1; then
    break
  fi
  PORT=$((PORT + 1))
done

URL="http://localhost:$PORT"

# --- Öppna webbläsaren när servern svarar -------------------------------------

(
  for _ in $(seq 1 60); do
    if curl -sf -o /dev/null "$URL" 2>/dev/null; then
      if command -v open >/dev/null 2>&1; then open "$URL"
      elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1
      fi
      break
    fi
    sleep 1
  done
) &

say "Startar på $URL – stäng med Ctrl+C"
exec npx next dev -p "$PORT"
