"""Markerar annonser som borttagna enligt 2+2-regeln.

Kör från projektroten, efter pipeline/scrape.py:
    python pipeline/detect_removed.py
    python pipeline/detect_removed.py --dry-run

Regeln (docs/PROTOTYP.md avsnitt 3): en annons som funnits i minst två körningar
och sedan saknats i två körningar räknas som borttagen. removed_at sätts till
sista gången vi faktiskt såg annonsen, och priset den hade då är vårt
"troligen sålt"-pris.

Skriptet är idempotent: varje körning i scrape_run räknas exakt en gång, vilket
scrape_run.detected_at håller reda på. Kör det två gånger i rad och andra
gången gör ingenting.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "boats.db"

MIN_SEEN_RUNS = 2
MIN_MISSING_RUNS = 2


def format_ts(ms: int | None) -> str:
    if ms is None:
        return "-"
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")


def process_run(connection: sqlite3.Connection, run: sqlite3.Row) -> tuple[int, int]:
    """Räknar en körning. Returnerar (antal sedda, antal saknade)."""
    # Bara annonser som fanns när körningen startade kan sakna sig i den.
    candidates = connection.execute(
        "SELECT id, last_seen FROM listing"
        " WHERE source = ? AND removed_at IS NULL AND first_seen <= ?",
        (run["source"], run["started_at"]),
    ).fetchall()

    seen_ids = [row["id"] for row in candidates if row["last_seen"] >= run["started_at"]]
    missing_ids = [row["id"] for row in candidates if row["last_seen"] < run["started_at"]]

    if seen_ids:
        connection.executemany(
            "UPDATE listing SET seen_runs = seen_runs + 1, missing_runs = 0 WHERE id = ?",
            [(i,) for i in seen_ids],
        )
    if missing_ids:
        connection.executemany(
            "UPDATE listing SET missing_runs = missing_runs + 1 WHERE id = ?",
            [(i,) for i in missing_ids],
        )

    connection.execute(
        "UPDATE scrape_run SET detected_at = ? WHERE id = ?",
        (int(time.time() * 1000), run["id"]),
    )
    return len(seen_ids), len(missing_ids)


def mark_removed(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    """Sätter removed_at på allt som passerat 2+2. Returnerar de markerade."""
    rows = connection.execute(
        "SELECT id, url, price, last_seen, seen_runs, missing_runs FROM listing"
        " WHERE removed_at IS NULL AND seen_runs >= ? AND missing_runs >= ?",
        (MIN_SEEN_RUNS, MIN_MISSING_RUNS),
    ).fetchall()

    if rows:
        connection.executemany(
            "UPDATE listing SET removed_at = last_seen WHERE id = ?",
            [(row["id"],) for row in rows],
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Markerar borttagna annonser (2+2-regeln).")
    parser.add_argument("--source", default="blocket")
    parser.add_argument(
        "--dry-run", action="store_true", help="visa vad som skulle hända, skriv inget"
    )
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"Hittar ingen databas på {DB_PATH}. Kör npm run db:push först.", file=sys.stderr)
        return 1

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    runs = connection.execute(
        "SELECT id, source, started_at FROM scrape_run"
        " WHERE source = ? AND detected_at IS NULL AND finished_at IS NOT NULL"
        " ORDER BY started_at",
        (args.source,),
    ).fetchall()

    if not runs:
        print(f"Inga oräknade körningar för {args.source}. Kör pipeline/scrape.py först.")
        connection.close()
        return 0

    for run in runs:
        seen, missing = process_run(connection, run)
        print(
            f"Körning {run['id']} ({format_ts(run['started_at'])}): "
            f"{seen} sedda, {missing} saknade"
        )

    removed = mark_removed(connection)

    total_runs = connection.execute(
        "SELECT COUNT(*) FROM scrape_run WHERE source = ?", (args.source,)
    ).fetchone()[0]

    if args.dry_run:
        connection.rollback()
        print("\n--dry-run: inget sparat.")
    else:
        connection.commit()

    print(f"\n{len(runs)} körningar räknade (totalt {total_runs} för {args.source}).")
    if removed:
        print(f"{len(removed)} annonser markerade som borttagna:")
        for row in removed:
            print(
                f"  {row['price']} kr, sist sedd {format_ts(row['last_seen'])} – {row['url']}"
            )
        print("\nKör pipeline/stats.py för att räkna om model_stats.")
    else:
        print("Inga annonser passerade 2+2-regeln den här gången.")

    connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
