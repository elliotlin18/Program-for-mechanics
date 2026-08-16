"""Hämtar båtannonser, normaliserar mot aliases.yaml och sparar i listing.

Kör från projektroten:
    python pipeline/scrape.py                 # alla modeller, max 20 annonser var
    python pipeline/scrape.py --limit 1       # en annons – använd den här första gången
    python pipeline/scrape.py --model "Yamarin 79 DC"
    python pipeline/scrape.py --offline       # bara cachad HTML, inga anrop

Regler (docs/PROTOTYP.md och CLAUDE.md): max en request varannan sekund, all HTML
cachas i data/cache/, robots.txt respekteras, ingen inloggning.

Körningen skrivs till scrape_run. pipeline/detect_removed.py läser den tabellen
för att avgöra vilka annonser som saknats i tillräckligt många körningar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
import yaml

import blocket
from blocket import ParseError

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "boats.db"
CACHE_DIR = ROOT / "data" / "cache"
ALIASES_PATH = ROOT / "pipeline" / "aliases.yaml"

REQUEST_DELAY_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 30
USER_AGENT = "batvarde-prototyp/0.1 (+https://github.com/elliotlin18/Program-for-mechanics)"


def now_ms() -> int:
    """Prisma lagrar DATETIME som millisekunder sedan epoch i SQLite."""
    return int(time.time() * 1000)


def today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class Fetcher:
    """Hämtar sidor snällt: robots.txt, en request varannan sekund, allt cachat."""

    def __init__(self, offline: bool = False):
        self.offline = offline
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last_request_at = 0.0
        self._robots: dict[str, RobotFileParser | None] = {}
        self.requests_made = 0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, url: str) -> Path:
        return CACHE_DIR / f"{hashlib.sha1(url.encode()).hexdigest()}.html"

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < REQUEST_DELAY_SECONDS:
            time.sleep(REQUEST_DELAY_SECONDS - elapsed)
        self._last_request_at = time.monotonic()

    def _robots_for(self, url: str) -> RobotFileParser | None:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin in self._robots:
            return self._robots[origin]

        parser: RobotFileParser | None = None
        try:
            self._wait()
            response = self.session.get(
                f"{origin}/robots.txt", timeout=REQUEST_TIMEOUT_SECONDS
            )
            self.requests_made += 1
            if response.status_code == 200:
                parser = RobotFileParser()
                parser.parse(response.text.splitlines())
            else:
                print(f"  robots.txt svarade {response.status_code}, fortsätter försiktigt")
        except requests.RequestException as error:
            print(f"  kunde inte hämta robots.txt ({error}), fortsätter försiktigt")

        self._robots[origin] = parser
        return parser

    def allowed(self, url: str) -> bool:
        if self.offline:
            return True
        parser = self._robots_for(url)
        # Ingen läsbar robots.txt -> vi låter det passera men kör lika snällt.
        return True if parser is None else parser.can_fetch(USER_AGENT, url)

    def get(self, url: str) -> str | None:
        """HTML för url, eller None om den inte gick att hämta."""
        cache_path = self._cache_path(url)

        if self.offline:
            if not cache_path.exists():
                print(f"  ingen cache för {url}")
                return None
            return cache_path.read_text(encoding="utf-8")

        if not self.allowed(url):
            print(f"  robots.txt tillåter inte {url}")
            return None

        self._wait()
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            self.requests_made += 1
        except requests.RequestException as error:
            print(f"  fel vid hämtning av {url}: {error}")
            return None

        if response.status_code != 200:
            print(f"  {response.status_code} för {url}")
            return None

        html = response.text
        cache_path.write_text(f"<!-- {url} -->\n{html}", encoding="utf-8")
        return html


def load_models(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    """Modeller ur databasen, med alias från aliases.yaml som fallback."""
    rows = connection.execute("SELECT id, brand, model, aliases FROM boats_model").fetchall()
    if rows:
        return rows

    print(f"boats_model är tom. Kör npm run db:seed först ({ALIASES_PATH.name}).")
    return []


def match_model(title: str, models: list[sqlite3.Row]) -> sqlite3.Row | None:
    """Fritext -> modell via alias. Längsta träffen vinner, så att
    "yamarin 79dc" inte fastnar på ett kortare alias för en annan modell."""
    lowered = title.lower()
    best: sqlite3.Row | None = None
    best_length = 0

    for model in models:
        candidates = [f"{model['brand']} {model['model']}".lower()]
        try:
            candidates += [str(a).lower() for a in json.loads(model["aliases"] or "[]")]
        except json.JSONDecodeError:
            pass

        for alias in candidates:
            if alias and alias in lowered and len(alias) > best_length:
                best, best_length = model, len(alias)

    return best


def save_listing(
    connection: sqlite3.Connection, ad: blocket.ParsedAd, model_id: int, source: str, seen_at: int
) -> str:
    """Skapar eller uppdaterar en annons. Returnerar 'ny', 'prisändring' eller 'oförändrad'."""
    existing = connection.execute(
        "SELECT id, price, price_history, removed_at FROM listing WHERE url = ?", (ad.url,)
    ).fetchone()

    if existing is None:
        connection.execute(
            "INSERT INTO listing (source, source_id, url, model_id, year, price,"
            " engine_brand, engine_hp, hours, region, title_raw, first_seen, last_seen,"
            " removed_at, price_history, seen_runs, missing_runs)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,?,0,0)",
            (
                source,
                ad.url.rstrip("/").rsplit("/", 1)[-1],
                ad.url,
                model_id,
                ad.year,
                ad.price,
                ad.engine_brand,
                ad.engine_hp,
                ad.hours,
                ad.region,
                ad.title,
                seen_at,
                seen_at,
                json.dumps([{"date": today_iso(), "price": ad.price}], ensure_ascii=False),
            ),
        )
        return "ny"

    changed = ad.price != existing["price"]
    history = json.loads(existing["price_history"] or "[]")
    if changed:
        history.append({"date": today_iso(), "price": ad.price})

    connection.execute(
        "UPDATE listing SET price = ?, price_history = ?, last_seen = ?, missing_runs = 0,"
        " removed_at = NULL, year = COALESCE(?, year), engine_brand = COALESCE(?, engine_brand),"
        " engine_hp = COALESCE(?, engine_hp), hours = COALESCE(?, hours),"
        " region = COALESCE(?, region), title_raw = ? WHERE id = ?",
        (
            ad.price,
            json.dumps(history, ensure_ascii=False),
            seen_at,
            ad.year,
            ad.engine_brand,
            ad.engine_hp,
            ad.hours,
            ad.region,
            ad.title,
            existing["id"],
        ),
    )

    if existing["removed_at"] is not None:
        return "återuppstånden"
    return "prisändring" if changed else "oförändrad"


def scrape(args: argparse.Namespace) -> int:
    if not DB_PATH.exists():
        print(f"Hittar ingen databas på {DB_PATH}. Kör npm run db:push först.", file=sys.stderr)
        return 1

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    models = load_models(connection)
    if not models:
        return 1

    if args.model:
        wanted = args.model.lower()
        models = [m for m in models if f"{m['brand']} {m['model']}".lower() == wanted]
        if not models:
            print(f"Ingen modell heter {args.model!r}.", file=sys.stderr)
            return 1

    started_at = now_ms()
    cursor = connection.execute(
        "INSERT INTO scrape_run (source, started_at, n_seen) VALUES (?, ?, 0)",
        (args.source, started_at),
    )
    run_id = cursor.lastrowid
    connection.commit()

    fetcher = Fetcher(offline=args.offline)
    seen_urls: set[str] = set()
    parse_errors = 0
    counts = {"ny": 0, "prisändring": 0, "oförändrad": 0, "återuppstånden": 0}

    for model in models:
        query = f"{model['brand']} {model['model']}"
        print(f"\n{query}")

        html = fetcher.get(blocket.search_url(query))
        if html is None:
            continue

        links = blocket.extract_ad_links(html)
        if not links:
            print("  inga annonslänkar i sökresultatet – kontrollera search_url() i blocket.py")
            continue

        print(f"  {len(links)} annonslänkar, hämtar högst {args.limit}")

        for url in links[: args.limit]:
            if url in seen_urls:
                continue

            ad_html = fetcher.get(url)
            if ad_html is None:
                continue

            try:
                ad = blocket.parse_ad(ad_html, url)
            except ParseError as error:
                parse_errors += 1
                print(f"  {error}")
                if parse_errors >= 3:
                    print("\nAvbryter: tre annonser i rad gick inte att tolka.")
                    print("Läs diagnosen ovan och justera pipeline/blocket.py.")
                    break
                continue

            matched = match_model(ad.title, models) or model
            outcome = save_listing(connection, ad, matched["id"], args.source, started_at)
            counts[outcome] += 1
            seen_urls.add(url)
            print(f"  [{outcome}] {ad.price} kr – {ad.title[:60]} (via {ad.via})")

        if parse_errors >= 3:
            break

    connection.execute(
        "UPDATE scrape_run SET finished_at = ?, n_seen = ? WHERE id = ?",
        (now_ms(), len(seen_urls), run_id),
    )
    connection.commit()
    connection.close()

    print(
        f"\nKörning {run_id} klar: {len(seen_urls)} annonser sedda "
        f"({counts['ny']} nya, {counts['prisändring']} prisändringar, "
        f"{counts['återuppstånden']} återuppståndna), {fetcher.requests_made} requests."
    )
    if parse_errors:
        print(f"{parse_errors} annonser gick inte att tolka.")
    print("Kör pipeline/detect_removed.py och pipeline/stats.py efteråt.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Hämtar båtannonser till listing.")
    parser.add_argument("--limit", type=int, default=20, help="max annonser per modell")
    parser.add_argument("--model", help='t.ex. "Yamarin 79 DC"')
    parser.add_argument("--source", default="blocket")
    parser.add_argument(
        "--offline", action="store_true", help="läs bara cachad HTML, gör inga anrop"
    )
    return scrape(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
