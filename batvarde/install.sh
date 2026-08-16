#!/usr/bin/env bash
# Installerar Båtvärde och startar det. Kör en gång:
#
#   curl -fsSL https://raw.githubusercontent.com/elliotlin18/Program-for-mechanics/claude/batvarde-project-setup-ub31pi/batvarde/install.sh | bash
#
# Efter det ligger en ikon på skrivbordet som startar programmet med ett dubbelklick.
# Byt installationsmapp med BATVARDE_DIR=/annan/plats.

set -euo pipefail

REPO_URL="https://github.com/elliotlin18/Program-for-mechanics.git"
BRANCH="claude/batvarde-project-setup-ub31pi"
INSTALL_DIR="${BATVARDE_DIR:-$HOME/Batvarde}"
APP_DIR="$INSTALL_DIR/batvarde"

say() { printf "\n\033[1m%s\033[0m\n" "$1"; }
warn() { printf "\033[33m%s\033[0m\n" "$1"; }

say "Installerar Båtvärde"

for tool in git node; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    warn "$tool saknas."
    case "$tool" in
      git) warn "Mac: installera Xcode Command Line Tools med 'xcode-select --install'." ;;
      node) warn "Ladda ner Node.js LTS från https://nodejs.org och kör om det här." ;;
    esac
    exit 1
  fi
done

# --- Hämta eller uppdatera koden ---------------------------------------------

if [ -d "$APP_DIR/.git" ] || [ -d "$INSTALL_DIR/.git" ]; then
  say "Uppdaterar koden i $INSTALL_DIR"
  git -C "$INSTALL_DIR" fetch origin "$BRANCH" --quiet
  git -C "$INSTALL_DIR" checkout "$BRANCH" --quiet
  git -C "$INSTALL_DIR" merge --ff-only "origin/$BRANCH" --quiet || {
    warn "Kunde inte uppdatera automatiskt – du har egna ändringar. Koden lämnas som den är."
  }
else
  say "Hämtar koden till $INSTALL_DIR"
  git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR" --quiet
fi

# --- Ikon på skrivbordet ------------------------------------------------------

desktop_dir() {
  if command -v xdg-user-dir >/dev/null 2>&1; then
    xdg-user-dir DESKTOP 2>/dev/null && return
  fi
  echo "$HOME/Desktop"
}

DESKTOP="$(desktop_dir)"

if [ -d "$DESKTOP" ]; then
  if [ "$(uname)" = "Darwin" ]; then
    LAUNCHER="$DESKTOP/Båtvärde.command"
    cat > "$LAUNCHER" <<LAUNCHER_EOF
#!/usr/bin/env bash
# Startar Båtvärde. Skapad av install.sh – ta bort filen om du vill bli av med ikonen.
exec "$APP_DIR/start.sh"
LAUNCHER_EOF
    chmod +x "$LAUNCHER"
    say "Ikon skapad: $LAUNCHER"
  else
    LAUNCHER="$DESKTOP/batvarde.desktop"
    cat > "$LAUNCHER" <<LAUNCHER_EOF
[Desktop Entry]
Type=Application
Name=Båtvärde
Comment=Vad går båten för?
Exec=$APP_DIR/start.sh
Path=$APP_DIR
Terminal=true
Categories=Development;
LAUNCHER_EOF
    chmod +x "$LAUNCHER"
    say "Ikon skapad: $LAUNCHER"
  fi
else
  warn "Hittade ingen skrivbordsmapp. Starta med: $APP_DIR/start.sh"
fi

say "Startar"
exec "$APP_DIR/start.sh"
