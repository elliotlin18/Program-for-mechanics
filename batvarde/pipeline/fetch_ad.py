"""Hämtar och tolkar EN annons, skriver resultatet som JSON på stdout.

Kör från projektroten:
    python pipeline/fetch_ad.py https://www.blocket.se/annons/...
    python pipeline/fetch_ad.py <url> --offline    # bara ur data/cache/

Finns för att /kolla ska kunna använda exakt samma parser som scrapern i stället
för att en andra tolkning skrivs i TypeScript. Samma Fetcher används också, så
robots.txt, en request varannan sekund och HTML-cachen gäller även här.

Utdata:
    {"ok": true, "ad": {...}}
    {"ok": false, "error": "...", "diagnosis": "..."}
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

import blocket  # noqa: E402
from scrape import DB_PATH, Fetcher, match_model  # noqa: E402

ALLOWED_HOSTS = {"www.blocket.se", "blocket.se", "www.sokbat.se", "sokbat.se"}


def matched_model_id(title: str) -> int | None:
    """Samma alias-matchning som scrapern använder."""
    import sqlite3

    if not DB_PATH.exists():
        return None
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    models = connection.execute("SELECT id, brand, model, aliases FROM boats_model").fetchall()
    connection.close()

    match = match_model(title, models)
    return match["id"] if match else None


def fail(error: str, diagnosis: str = "") -> int:
    json.dump({"ok": False, "error": error, "diagnosis": diagnosis}, sys.stdout, ensure_ascii=False)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Hämtar och tolkar en annons.")
    parser.add_argument("url")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    parsed = urlparse(args.url)
    if parsed.scheme not in ("http", "https") or parsed.netloc not in ALLOWED_HOSTS:
        return fail(f"Vi läser bara annonser från {', '.join(sorted(ALLOWED_HOSTS))}.")

    # Fetcher skriver sin diagnostik med print(). Stdout är reserverad för JSON,
    # så den styrs om till stderr.
    with contextlib.redirect_stdout(sys.stderr):
        html = Fetcher(offline=args.offline).get(args.url)

    if html is None:
        return fail("Kunde inte hämta annonsen.")

    try:
        ad = blocket.parse_ad(html, args.url)
    except blocket.ParseError as error:
        return fail("Kunde inte tolka annonsen.", str(error))

    payload = dataclasses.asdict(ad)
    payload["model_id"] = matched_model_id(ad.title)

    json.dump({"ok": True, "ad": payload}, sys.stdout, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
