"""Tester för pipelinen. Kör från projektroten: python pipeline/test_pipeline.py

De finns för att blocket.py skrevs utan tillgång till Blockets riktiga HTML.
Testerna bevisar att maskineriet runt omkring stämmer – 2+2-regeln, alias-
matchningen, prishistoriken, robots-spärren och de tre vägarna in i en sida –
så att det enda som återstår att verifiera mot skarp sajt är själva HTML:en.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import blocket  # noqa: E402
import detect_removed  # noqa: E402
import scrape  # noqa: E402

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "boats.db"


def temp_db() -> sqlite3.Connection:
    """Minnesdatabas med samma schema som den riktiga – hämtat därifrån, så att
    testerna aldrig kan glida isär från prisma/schema.prisma."""
    source = sqlite3.connect(DB_PATH)
    ddl = [
        row[0]
        for row in source.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
        )
    ]
    source.close()

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    for statement in ddl:
        connection.execute(statement)
    return connection


# --- Tolkning av annonssidor ---------------------------------------------------


JSON_LD_PAGE = """
<html><head><title>Yamarin 79 DC 2019 – Blocket</title>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Product","name":"Yamarin 79 DC 2019",
 "offers":{"@type":"Offer","price":"895000","priceCurrency":"SEK"}}
</script></head>
<body><p>Yamarin 79 DC 2019, Yamaha 300 hk, 240 timmar. Stockholm.</p></body></html>
"""

NEXT_DATA_PAGE = """
<html><head><title>Buster XL</title>
<script id="__NEXT_DATA__" type="application/json">
{"props":{"pageProps":{"ad":{"subject":"Buster XL 2018 Yamaha 150hk",
 "price":{"value":329000,"suffix":"kr"},"location":"Skåne"}}}}
</script></head>
<body><p>Buster XL 2018, Yamaha 150 hk, 180 timmar, Skåne.</p></body></html>
"""

META_PAGE = """
<html><head><title>Nimbus 27</title>
<meta property="og:title" content="Nimbus 27 Nova 2015">
<meta property="product:price:amount" content="1195000">
</head><body><p>Volvo Penta 300 hk. 520 motortimmar. Stockholm.</p></body></html>
"""

USELESS_PAGE = "<html><head><title>Sidan finns inte</title></head><body>Hoppsan</body></html>"


class TestParseAd(unittest.TestCase):
    def test_json_ld(self):
        ad = blocket.parse_ad(JSON_LD_PAGE, "https://x/annons/1")
        self.assertEqual(ad.via, "json-ld")
        self.assertEqual(ad.price, 895000)
        # Namnet sitter på Product, priset i Offer under den.
        self.assertEqual(ad.title, "Yamarin 79 DC 2019")
        self.assertEqual(ad.year, 2019)
        self.assertEqual(ad.engine_brand, "Yamaha")
        self.assertEqual(ad.engine_hp, 300)
        self.assertEqual(ad.hours, 240)
        self.assertEqual(ad.region, "Stockholm")

    def test_next_data(self):
        ad = blocket.parse_ad(NEXT_DATA_PAGE, "https://x/annons/2")
        self.assertEqual(ad.via, "__NEXT_DATA__")
        self.assertEqual(ad.price, 329000)
        self.assertEqual(ad.title, "Buster XL 2018 Yamaha 150hk")
        self.assertEqual(ad.year, 2018)
        self.assertEqual(ad.engine_hp, 150)
        self.assertEqual(ad.region, "Skåne")

    def test_meta(self):
        ad = blocket.parse_ad(META_PAGE, "https://x/annons/3")
        self.assertEqual(ad.via, "meta")
        self.assertEqual(ad.price, 1195000)
        self.assertEqual(ad.title, "Nimbus 27 Nova 2015")
        self.assertEqual(ad.engine_brand, "Volvo Penta")
        self.assertEqual(ad.hours, 520)

    def test_missing_price_raises_with_diagnosis(self):
        with self.assertRaises(blocket.ParseError) as caught:
            blocket.parse_ad(USELESS_PAGE, "https://x/annons/4")
        message = str(caught.exception)
        self.assertIn("json-ld: False", message)
        self.assertIn("__NEXT_DATA__: False", message)
        self.assertIn("Sidan finns inte", message)

    def test_implausible_price_is_rejected(self):
        page = JSON_LD_PAGE.replace('"895000"', '"12"')
        with self.assertRaises(blocket.ParseError):
            blocket.parse_ad(page, "https://x/annons/5")

    def test_parse_int_sv_handles_swedish_formatting(self):
        self.assertEqual(blocket.parse_int_sv("1\xa0234\xa0567 kr"), 1234567)
        self.assertEqual(blocket.parse_int_sv("895 000"), 895000)
        self.assertEqual(blocket.parse_int_sv("329000,00"), 329000)
        self.assertIsNone(blocket.parse_int_sv("kontakta säljaren"))

    def test_year_takes_the_last_plausible_year(self):
        # "79" i modellnamnet får inte bli årsmodell, och 2019 ska vinna över 1980.
        self.assertEqual(blocket.extract_year("Yamarin 79 DC 2019"), 2019)
        self.assertIsNone(blocket.extract_year("Buster XL"))


class TestAdLinks(unittest.TestCase):
    SEARCH_PAGE = """
    <html><body>
      <a href="/annons/yamarin-79-dc/abc123">Yamarin</a>
      <a href="/annons/yamarin-79-dc/abc123?utm=1">Samma annons igen</a>
      <a href="https://www.blocket.se/annons/buster-xl/def456">Buster</a>
      <a href="/om-oss">Om oss</a>
      <a href="/annonsera">Annonsera</a>
    </body></html>
    """

    def test_finds_ads_and_drops_duplicates_and_noise(self):
        links = blocket.extract_ad_links(self.SEARCH_PAGE)
        self.assertEqual(
            links,
            [
                "https://www.blocket.se/annons/yamarin-79-dc/abc123",
                "https://www.blocket.se/annons/buster-xl/def456",
            ],
        )


# --- Normalisering -------------------------------------------------------------


class TestMatchModel(unittest.TestCase):
    def setUp(self):
        self.connection = temp_db()
        self.connection.execute(
            "INSERT INTO boats_model (id, brand, model, type, length_m, aliases)"
            " VALUES (1,'Yamarin','79 DC','motorbåt',7.9,?)",
            (json.dumps(["yamarin 79", "79dc", "79 dc"]),),
        )
        self.connection.execute(
            "INSERT INTO boats_model (id, brand, model, type, length_m, aliases)"
            " VALUES (2,'Albin','Vega','segelbåt',8.25,?)",
            (json.dumps(["albin vega", "vega"]),),
        )
        self.models = self.connection.execute(
            "SELECT id, brand, model, aliases FROM boats_model"
        ).fetchall()

    def tearDown(self):
        self.connection.close()

    def test_matches_alias(self):
        self.assertEqual(scrape.match_model("Snygg Yamarin 79dc 2019", self.models)["id"], 1)

    def test_longest_alias_wins(self):
        # "vega" och "albin vega" matchar båda – den längre ska vinna.
        self.assertEqual(scrape.match_model("Albin Vega 1974", self.models)["id"], 2)

    def test_no_match_returns_none(self):
        self.assertIsNone(scrape.match_model("Uttern D68 2018", self.models))


class TestSaveListing(unittest.TestCase):
    def setUp(self):
        self.connection = temp_db()
        self.connection.execute(
            "INSERT INTO boats_model (id, brand, model, type, aliases)"
            " VALUES (1,'Yamarin','79 DC','motorbåt','[]')"
        )
        self.ad = blocket.ParsedAd(
            url="https://www.blocket.se/annons/a1",
            title="Yamarin 79 DC 2019",
            price=895000,
            year=2019,
        )

    def tearDown(self):
        self.connection.close()

    def row(self):
        return self.connection.execute("SELECT * FROM listing WHERE url = ?", (self.ad.url,)).fetchone()

    def test_new_listing(self):
        self.assertEqual(scrape.save_listing(self.connection, self.ad, 1, "blocket", 1000), "ny")
        row = self.row()
        self.assertEqual(row["price"], 895000)
        self.assertEqual(row["first_seen"], 1000)
        self.assertEqual(row["last_seen"], 1000)
        self.assertEqual(len(json.loads(row["price_history"])), 1)

    def test_price_change_appends_history(self):
        scrape.save_listing(self.connection, self.ad, 1, "blocket", 1000)
        self.ad.price = 849000
        self.assertEqual(
            scrape.save_listing(self.connection, self.ad, 1, "blocket", 2000), "prisändring"
        )
        row = self.row()
        history = json.loads(row["price_history"])
        self.assertEqual([h["price"] for h in history], [895000, 849000])
        self.assertEqual(row["price"], 849000)
        self.assertEqual(row["first_seen"], 1000, "first_seen ska inte flyttas")
        self.assertEqual(row["last_seen"], 2000)

    def test_unchanged_price_does_not_append(self):
        scrape.save_listing(self.connection, self.ad, 1, "blocket", 1000)
        self.assertEqual(
            scrape.save_listing(self.connection, self.ad, 1, "blocket", 2000), "oförändrad"
        )
        self.assertEqual(len(json.loads(self.row()["price_history"])), 1)

    def test_listing_that_comes_back_is_un_removed(self):
        scrape.save_listing(self.connection, self.ad, 1, "blocket", 1000)
        self.connection.execute("UPDATE listing SET removed_at = 1500, missing_runs = 2")
        self.assertEqual(
            scrape.save_listing(self.connection, self.ad, 1, "blocket", 2000), "återuppstånden"
        )
        row = self.row()
        self.assertIsNone(row["removed_at"])
        self.assertEqual(row["missing_runs"], 0)


# --- 2+2-regeln ----------------------------------------------------------------

DAY = 86_400_000


class TestDetectRemoved(unittest.TestCase):
    def setUp(self):
        self.connection = temp_db()
        self.connection.execute(
            "INSERT INTO boats_model (id, brand, model, type, aliases)"
            " VALUES (1,'Yamarin','79 DC','motorbåt','[]')"
        )
        self.run_number = 0

    def tearDown(self):
        self.connection.close()

    def add_listing(self, url: str, first_seen: int) -> int:
        cursor = self.connection.execute(
            "INSERT INTO listing (source, source_id, url, model_id, price, first_seen,"
            " last_seen, price_history, seen_runs, missing_runs)"
            " VALUES ('blocket', ?, ?, 1, 500000, ?, ?, '[]', 0, 0)",
            (url, url, first_seen, first_seen),
        )
        return cursor.lastrowid

    def run_scrape(self, at: int, seen_urls: list[str]) -> None:
        """Låtsas att scrape.py körde vid `at` och såg `seen_urls`."""
        self.run_number += 1
        for url in seen_urls:
            self.connection.execute(
                "UPDATE listing SET last_seen = ?, missing_runs = 0 WHERE url = ?", (at, url)
            )
        cursor = self.connection.execute(
            "INSERT INTO scrape_run (source, started_at, finished_at, n_seen)"
            " VALUES ('blocket', ?, ?, ?)",
            (at, at + 60_000, len(seen_urls)),
        )
        run = self.connection.execute(
            "SELECT id, source, started_at FROM scrape_run WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        detect_removed.process_run(self.connection, run)

    def state(self, url: str):
        return self.connection.execute(
            "SELECT seen_runs, missing_runs, removed_at, last_seen FROM listing WHERE url = ?",
            (url,),
        ).fetchone()

    def test_two_seen_then_two_missing_marks_removed(self):
        self.add_listing("a", 1 * DAY)

        self.run_scrape(1 * DAY, ["a"])
        self.run_scrape(2 * DAY, ["a"])
        self.assertEqual(self.state("a")["seen_runs"], 2)

        self.run_scrape(3 * DAY, [])
        self.assertEqual(detect_removed.mark_removed(self.connection), [])
        self.assertIsNone(self.state("a")["removed_at"], "en missad körning räcker inte")

        self.run_scrape(4 * DAY, [])
        removed = detect_removed.mark_removed(self.connection)
        self.assertEqual(len(removed), 1)
        # removed_at = sista gången vi faktiskt såg annonsen, dvs körning 2.
        self.assertEqual(self.state("a")["removed_at"], 2 * DAY)

    def test_seen_only_once_is_not_removed(self):
        self.add_listing("b", 1 * DAY)
        self.run_scrape(1 * DAY, ["b"])
        self.run_scrape(2 * DAY, [])
        self.run_scrape(3 * DAY, [])
        self.assertEqual(self.state("b")["seen_runs"], 1)
        self.assertEqual(detect_removed.mark_removed(self.connection), [])

    def test_missing_counter_resets_when_ad_reappears(self):
        self.add_listing("c", 1 * DAY)
        self.run_scrape(1 * DAY, ["c"])
        self.run_scrape(2 * DAY, ["c"])
        self.run_scrape(3 * DAY, [])
        self.assertEqual(self.state("c")["missing_runs"], 1)
        self.run_scrape(4 * DAY, ["c"])
        self.assertEqual(self.state("c")["missing_runs"], 0)
        self.run_scrape(5 * DAY, [])
        self.run_scrape(6 * DAY, [])
        self.assertEqual(len(detect_removed.mark_removed(self.connection)), 1)

    def test_ad_added_after_a_run_is_not_counted_as_missing(self):
        self.run_scrape(1 * DAY, [])
        self.add_listing("d", 2 * DAY)
        self.run_scrape(3 * DAY, ["d"])
        self.assertEqual(self.state("d")["missing_runs"], 0)
        self.assertEqual(self.state("d")["seen_runs"], 1)

    def test_processing_is_idempotent(self):
        self.add_listing("e", 1 * DAY)
        self.run_scrape(1 * DAY, ["e"])
        unprocessed = self.connection.execute(
            "SELECT COUNT(*) FROM scrape_run WHERE detected_at IS NULL"
        ).fetchone()[0]
        self.assertEqual(unprocessed, 0, "process_run ska stämpla körningen som räknad")


# --- Robots och rate limit -----------------------------------------------------


class TestFetcherPoliteness(unittest.TestCase):
    def test_robots_disallow_is_honoured(self):
        from urllib.robotparser import RobotFileParser

        fetcher = scrape.Fetcher()
        parser = RobotFileParser()
        parser.parse(["User-agent: *", "Disallow: /annons/"])
        fetcher._robots["https://www.blocket.se"] = parser

        self.assertFalse(fetcher.allowed("https://www.blocket.se/annons/abc"))
        self.assertTrue(fetcher.allowed("https://www.blocket.se/annonser/hela_sverige"))

    def test_delay_is_two_seconds(self):
        self.assertGreaterEqual(scrape.REQUEST_DELAY_SECONDS, 2.0)

    def test_offline_mode_never_calls_out(self):
        fetcher = scrape.Fetcher(offline=True)
        self.assertIsNone(fetcher.get("https://www.blocket.se/annons/finns-inte-i-cachen"))
        self.assertEqual(fetcher.requests_made, 0)


if __name__ == "__main__":
    if not DB_PATH.exists():
        print(f"Hittar ingen databas på {DB_PATH}. Kör npm run db:push först.", file=sys.stderr)
        sys.exit(1)
    unittest.main(verbosity=2)
