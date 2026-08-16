# Prototyp: "Vad går båten för?" – byggplan för Claude Code

Syfte: en klickbar demo på riktig data som visar Matija idén på fem minuter. Inte en produkt. Bygg det som gör poängen tydlig – **sålda priser slår listpriser** – och inget mer.

Tidsbudget: 15–25 timmar över 1–2 veckor. Allt under "Inte nu" väntar.

---

## 1. Vad demon ska bevisa

1. Vi kan samla båtannonser automatiskt och normalisera dem per modell/årgång.
2. Vi kan se när annonser försvinner (proxy för sålt) och vad sista priset var.
3. Vi kan visa en modellsida med prisintervall, prishistorik och tid på marknaden.
4. Vi kan svara på "är den här annonsen rätt prissatt?" – det som ingen annan gör.

Om de fyra punkterna syns i demon är den klar.

---

## 2. Scope: tre skärmar och en pipeline

**Skärm A – Modellsida** (t.ex. `/bat/yamarin/79-dc`)
- Namn, bild-placeholder, kort spec (längd, typ).
- Prisintervall aktiva annonser (min/median/max) och prisintervall "troligen sålda".
- Graf: annonspris över tid (12 veckor räcker).
- Tabell: senaste 20 annonserna – pris, år, region, dagar ute, status (aktiv/borttagen), ev. prissänkningar.
- Liten ruta: "Konfidens: hög/medel/låg" baserat på antal observationer.

**Skärm B – Värdera min båt** (`/vardera`)
- Fält: märke, modell, årsmodell, motor (märke + hk), timmar (valfritt), region, skick (1–5).
- Svar: intervall + median + "baserat på N observationer, senaste 90 dagarna" + 5 jämförbara annonser.
- Ingen e-post, inget konto.

**Skärm C – Annonskollen** (`/kolla`)
- Klistra in en Blocket-länk (eller ange märke/modell/år/pris manuellt om länken inte kan läsas).
- Svar: "Annonsen ligger X % över/under median för modellen", dagar ute, antal prissänkningar, jämförbara.

**Pipeline (körs som skript, visas i terminal under demon)**
- Hämtar annonser → normaliserar → sparar → detekterar borttagna → räknar om statistik.

---

## 3. Data: så gör ni det på 1–2 veckor utan att fastna

**Källor för prototypen (i prioritetsordning)**
1. **Blocket Båtar** – primär. Börja med 10 modeller (se lista), hämta sökresultat + annonssidor. Kör snällt: 1 request/2 sek, cache allt lokalt, respektera robots.txt, ingen inloggning. Om blockering: fall tillbaka på källa 2–3, poängen är densamma.
2. **Sokbat.se** – enklare struktur, färre annonser, bra taxonomi.
3. **Boat24 (SE-filter)** – liten volym, men internationell prisreferens.
4. **Manuellt seed-set** – om scraping strular första dagen: lägg 100–200 annonser i en CSV för hand (30 min) så att UI-arbetet aldrig blockeras av datainsamlingen.

**Tio modeller att starta med** (stor volym, standardiserade, 5–10 m motorbåtar + två segelbåtar för bredd):
Yamarin 79 DC, Buster XL, Nimbus 27, Uttern D68, Bella 703, Ryds 548, Anytec 750, Flipper 640, Maxi 77 (segel), Albin Vega (segel).

**Datamodell (minimum)**
```
boats_model(id, brand, model, type, length_m, aliases[])
listing(id, source, source_id, url, model_id, year, price, engine_brand, engine_hp,
        hours, region, title_raw, first_seen, last_seen, removed_at, price_history jsonb)
model_stats(model_id, window_days, n_active, n_removed, p25, median, p75,
            median_days_on_market, avg_price_drop_pct, confidence)
```

**Borttagningslogik**: kör hämtningen dagligen (eller flera gånger under demoveckan). Annons som funnits ≥ 2 körningar och sedan saknas i 2 körningar → `removed_at` sätts, sista pris = "troligen sålt-pris". Det räcker för att illustrera principen.

**Normalisering**: mappa fritext till modell via alias-lista (regex på titel: "yamarin 79", "79dc", "79 dc"). Manuell alias-lista för 10 modeller är 20 minuters jobb; gör den utbyggbar.

---

## 4. Stack (håll det tråkigt)

- **Next.js 14 (App Router) + TypeScript + Tailwind** för UI. shadcn/ui för komponenter, Recharts för grafen.
- **SQLite via Prisma** (en fil, noll drift). Byt till Postgres senare.
- **Scraper i Python** (`requests` + `beautifulsoup4`/`playwright` om JS krävs) i `/pipeline`, skriver till samma SQLite-fil. Alternativt Node om du vill ha ett språk – spelar ingen roll för demon.
- **Kör lokalt.** Deploy till Vercel + Turso/Neon bara om ni vill visa på mobil.

Repo-struktur:
```
/app            Next.js
/pipeline       scrape.py, normalize.py, detect_removed.py, stats.py, aliases.yaml
/data           boats.db, seed.csv
/docs           DEMO.md (demoscriptet nedan)
```

---

## 5. Byggordning – i den här ordningen, en Claude Code-session per steg

**Steg 0 (30 min) – Repo och seed**
Prompt: "Skapa Next.js 14-projekt med TypeScript, Tailwind, shadcn/ui, Prisma+SQLite. Lägg in datamodellen ovan. Skapa seed-skript som läser data/seed.csv med kolumnerna source,url,brand,model,year,price,engine_brand,engine_hp,hours,region,first_seen." Fyll seed.csv med 100–200 rader (kopiera för hand från Blocket, tar 30 min). Nu kan UI byggas oavsett scraping.

**Steg 1 (3–4 h) – Modellsida**
Prompt: "Bygg /bat/[brand]/[model]: hämta model_stats + listings, visa prisintervall aktiva vs borttagna, Recharts-linje över pris per vecka, tabell med senaste 20 annonser (pris, år, region, dagar ute, status, prissänkningar). Konfidens: hög ≥30 obs, medel 10–29, låg <10." Gör den snygg direkt – det är den skärm Matija kommer att titta längst på.

**Steg 2 (2–3 h) – Statistik**
Prompt: "Skriv pipeline/stats.py som per modell räknar p25/median/p75 för aktiva och borttagna senaste 90 dagar, median dagar på marknaden, snittprissänkning i procent, konfidens. Skriv till model_stats." Kör mot seed-datan.

**Steg 3 (4–6 h) – Scraper + borttagning**
Prompt: "Skriv pipeline/scrape.py som för varje modell i aliases.yaml hämtar Blocket-sökresultat för båtar, går in på annonssidor, plockar pris/år/motor/region/titel, sparar/uppdaterar listing (uppdatera last_seen, lägg pris i price_history vid ändring). Rate limit 1 req/2 s, cache HTML i /data/cache. Skriv detect_removed.py enligt regeln 2+2 körningar." Kör 2–3 gånger per dag under demoveckan så att några annonser hinner försvinna. Om Blocket blockerar: byt till Sokbat, ändra inte något annat.

**Steg 4 (2–3 h) – Värdera min båt**
Prompt: "Bygg /vardera med formulär (märke/modell dropdown från DB, år, motor hk, timmar, region, skick 1–5). Backend: filtrera listings på modell ±2 år, vikta borttagna högre än aktiva, justera ±5 % per skicksteg från 3, returnera intervall/median/N och 5 närmaste jämförbara. Visa resultatet utan att kräva e-post."

**Steg 5 (2–3 h) – Annonskollen**
Prompt: "Bygg /kolla: input Blocket-URL. Försök hämta annonsen (samma parser som scrapern); om det misslyckas, visa manuellt formulär. Matcha modell via aliases, jämför pris mot median för modell/år, visa avvikelse i %, dagar ute, prissänkningar, jämförbara. Färgkoda: grön under −5 %, gul ±5 %, röd över +5 %."

**Steg 6 (1–2 h) – Startsida + polish**
Enkel startsida: sök modell, två knappar (Värdera / Kolla annons), tre "heta modeller"-kort med median och trend. Lägg en liten "Så räknar vi"-sida med tre stycken. Kör igenom demoscriptet.

---

## 6. Demoscript för Matija (5 minuter)

1. Startsidan: "Det här är Bilpriser för båtar." Klicka Yamarin 79 DC.
2. Modellsidan: peka på skillnaden mellan aktiva och borttagna priser. "Annonspris X, troligen sålt Y. Ingen annan visar Y."
3. Grafen + dagar på marknaden: "Vi ser prissänkningar och hur länge båtar ligger ute."
4. Kolla annons: klistra in en riktig Blocket-annons, visa "12 % över marknad".
5. Terminalen: kör `python pipeline/scrape.py` live, visa att nya annonser trillar in.
6. Avsluta med affären i en mening: "Konsumenten får detta gratis. Mäklaren, banken och försäkringsbolaget betalar för leads och datan bakom. SäljaDinBåt gör bara mäklarbiten och på listpriser."

---

## 7. Inte nu

Inloggning, mäklarportal, leads, affiliate, e-post, bevakningar, PDF-intyg, motorer, Norge, app, deploy, SEO. Allt detta finns i backloggen (våg 2). Prototypen ska bara bevisa datamotorn och de tre skärmarna.

---

## 8. Klart-kriterier

- Minst 10 modeller, minst 300 annonser, minst 20 markerade som borttagna.
- Modellsida, värdering och annonskoll fungerar utan krasch på alla 10 modeller.
- Scraper kan köras live under demon (eller seed-datan visas transparent som seed).
- Demon tar under 5 minuter och slutar med affärsmeningen.
