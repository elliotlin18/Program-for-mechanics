"""Räknar om model_stats för varje modell.

Kör från projektroten: python pipeline/stats.py

Samma regler som lib/stats.ts, som seed-skriptet använder. Ändras den ena måste
den andra ändras med, annars visar modellsidan andra siffror än pipelinen.

Prisma lagrar DATETIME som millisekunder sedan epoch i SQLite, så all
tidsräkning här sker i millisekunder.
"""

import json
import math
import sqlite3
import sys
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "boats.db"

WINDOW_DAYS = 90
DAY_MS = 86_400_000


def js_round(value: float) -> int:
    """Math.round i JavaScript avrundar halva uppåt. Pythons round() gör det inte."""
    return math.floor(value + 0.5)


def percentile(values: list[int], p: float) -> int | None:
    """Linjärt interpolerad percentil. p anges 0-1."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    pos = (len(ordered) - 1) * p
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return js_round(ordered[lower])
    return js_round(ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower))


def confidence_for(observations: int) -> str:
    """hög >= 30 obs, medel 10-29, låg < 10 (docs/PROTOTYP.md steg 1)."""
    if observations >= 30:
        return "hög"
    if observations >= 10:
        return "medel"
    return "låg"


def days_between(from_ms: int, to_ms: int) -> int:
    return max(0, js_round((to_ms - from_ms) / DAY_MS))


def parse_price_history(raw: str) -> list[dict]:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def compute_model_stats(listings: list[sqlite3.Row], now_ms: int) -> dict:
    """Aktiva annonser = listpriser, borttagna = troligen sålt-priser."""
    cutoff = now_ms - WINDOW_DAYS * DAY_MS

    active = [
        row for row in listings if row["removed_at"] is None and row["last_seen"] >= cutoff
    ]
    removed = [
        row for row in listings if row["removed_at"] is not None and row["removed_at"] >= cutoff
    ]

    active_prices = [row["price"] for row in active]
    removed_prices = [row["price"] for row in removed]

    # Dagar på marknaden räknas i första hand på borttagna annonser - de är avslutade
    # spelningar. Finns inga borttagna ännu använder vi hur länge de aktiva legat ute.
    days_source = removed if removed else active
    days_on_market = [
        days_between(row["first_seen"], row["removed_at"] or now_ms) for row in days_source
    ]

    drop_pcts = []
    for row in active + removed:
        history = parse_price_history(row["price_history"])
        if len(history) < 2:
            continue
        first = history[0]["price"]
        last = history[-1]["price"]
        if first > 0 and last < first:
            drop_pcts.append((first - last) / first * 100)

    avg_drop = js_round(sum(drop_pcts) / len(drop_pcts) * 10) / 10 if drop_pcts else None

    return {
        "window_days": WINDOW_DAYS,
        "n_active": len(active),
        "n_removed": len(removed),
        "p25": percentile(active_prices, 0.25),
        "median": percentile(active_prices, 0.5),
        "p75": percentile(active_prices, 0.75),
        "p25_removed": percentile(removed_prices, 0.25),
        "median_removed": percentile(removed_prices, 0.5),
        "p75_removed": percentile(removed_prices, 0.75),
        "median_days_on_market": percentile(days_on_market, 0.5),
        "avg_price_drop_pct": avg_drop,
        "confidence": confidence_for(len(active) + len(removed)),
    }


UPSERT = """
INSERT INTO model_stats (
    model_id, window_days, n_active, n_removed,
    p25, median, p75, p25_removed, median_removed, p75_removed,
    median_days_on_market, avg_price_drop_pct, confidence
) VALUES (
    :model_id, :window_days, :n_active, :n_removed,
    :p25, :median, :p75, :p25_removed, :median_removed, :p75_removed,
    :median_days_on_market, :avg_price_drop_pct, :confidence
)
ON CONFLICT (model_id, window_days) DO UPDATE SET
    n_active = excluded.n_active,
    n_removed = excluded.n_removed,
    p25 = excluded.p25,
    median = excluded.median,
    p75 = excluded.p75,
    p25_removed = excluded.p25_removed,
    median_removed = excluded.median_removed,
    p75_removed = excluded.p75_removed,
    median_days_on_market = excluded.median_days_on_market,
    avg_price_drop_pct = excluded.avg_price_drop_pct,
    confidence = excluded.confidence
"""


def format_sek(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:,}".replace(",", " ") + " kr"


def main() -> int:
    if not DB_PATH.exists():
        print(f"Hittar ingen databas på {DB_PATH}. Kör npm run db:push först.", file=sys.stderr)
        return 1

    now_ms = int(time.time() * 1000)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    models = connection.execute(
        "SELECT id, brand, model FROM boats_model ORDER BY brand, model"
    ).fetchall()

    for model in models:
        listings = connection.execute(
            "SELECT price, first_seen, last_seen, removed_at, price_history"
            " FROM listing WHERE model_id = ?",
            (model["id"],),
        ).fetchall()

        stats = compute_model_stats(listings, now_ms)
        connection.execute(UPSERT, {**stats, "model_id": model["id"]})

        name = f"{model['brand']} {model['model']}"
        print(
            f"  {name:<16} {stats['n_active']:>3} aktiva / {stats['n_removed']:>3} borttagna"
            f"   median {format_sek(stats['median'])} / {format_sek(stats['median_removed'])}"
            f"   konfidens {stats['confidence']}"
        )

    connection.commit()
    connection.close()

    print(f"\nmodel_stats uppdaterad för {len(models)} modeller ({WINDOW_DAYS} dagars fönster).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
