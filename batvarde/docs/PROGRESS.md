# Progress

## Steg 0 – Repo och seed (klart)
- Next.js 14 + TypeScript + Tailwind + Prisma/SQLite uppsatt i `batvarde/`; datamodellen från specen ligger i `prisma/schema.prisma` som tabellerna `boats_model`, `listing`, `model_stats`.
- `npm run db:seed` läser `pipeline/aliases.yaml` (10 modeller) och `data/seed.csv`, sätter `removed_at = first_seen + 14 dagar` för `status=removed`, `last_seen = idag` för aktiva, och räknar om `model_stats`.
- Återstår: `data/seed.csv` innehåller fortfarande de fem EXEMPEL-raderna från startkitet – de ska bytas mot 100–200 riktiga rader från Blocket.

## Steg 1 – Modellsida (klart)
- `/bat/[brand]/[model]` visar spec, prisintervall p25/median/p75 för aktiva (listpris) vs borttagna (troligen sålt), Recharts-linje med medianpris per vecka i 12 veckor, och tabell över de senaste 20 annonserna (pris, år, region, dagar ute, status, prissänkningar).
- Konfidens enligt specen: hög ≥ 30 observationer, medel 10–29, låg < 10, räknat på aktiva + borttagna de senaste 90 dagarna. `/` är tills vidare ett enkelt modellindex – riktig startsida byggs i steg 6.
- Verifierat: `npm run typecheck`, `npm run lint` och `npm run build` går igenom, och alla tio modellsidor svarar 200 mot produktionsservern.

## Steg 2 – Statistik (klart)
- `pipeline/stats.py` räknar per modell p25/median/p75 för aktiva och borttagna senaste 90 dagarna, median dagar på marknaden, snittprissänkning i procent och konfidens, och upsertar till `model_stats`. Bara standardbiblioteket – inga pip-beroenden.
- Prisma lagrar DATETIME som millisekunder i SQLite, så all tidsräkning sker i millisekunder. `js_round()` finns för att Pythons `round()` avrundar halva till jämnt medan JavaScripts `Math.round()` avrundar uppåt.
- Verifierat mot `lib/stats.ts`: noll skillnader på alla tolv fälten för alla tio modellerna, både på seedad data och på injicerade edge-fall (modell utan annonser, modell utan borttagna, modell med en enda annons, annonser utanför 90-dagarsfönstret, prishistorik med både sänkningar och en höjning).

## Steg 3 – Scraper och borttagning (delvis klart – läs "Vad som återstår")
- `pipeline/scrape.py`: hämtar sökresultat per modell, går in på annonssidorna, matchar mot modell via alias (längsta träffen vinner), och skapar eller uppdaterar `listing`. Prisändring läggs till i `price_history`, `last_seen` uppdateras, en annons som dyker upp igen får `removed_at` nollställd. Max en request varannan sekund, robots.txt läses och respekteras, all HTML cachas i `data/cache/`, ingen inloggning. Varje körning skrivs till nya tabellen `scrape_run`.
- `pipeline/detect_removed.py`: 2+2-regeln. Räknare `seen_runs`/`missing_runs` på `listing`, och `scrape_run.detected_at` gör steget idempotent – samma körning kan aldrig höja `missing_runs` två gånger. `removed_at` sätts till sista gången annonsen faktiskt syntes, så priset den hade då blir vårt "troligen sålt"-pris.
- Verifierat: 23 tester i `pipeline/test_pipeline.py` (`python pipeline/test_pipeline.py`), plus en helt igenom-repetition offline där fixtur-HTML lades i cachen och den riktiga koden kördes: fyra körningar, en prissänkning 895 000 → 849 000 hamnade i `price_history`, annonsen som försvann markerades borttagen på exakt fjärde körningen, `stats.py` räknade snittprissänkning 5,1 % och modellsidan visade den.

### Vad som återstår i steg 3
Sessionen som byggde det här kom inte ut på nätet – nätverkspolicyn svarade 403 på både blocket.se och sokbat.se. Därför är Blockets riktiga HTML aldrig sedd, och två saker är overifierade:

1. **`search_url()` i `pipeline/blocket.py`** – sökvägen och kategori-id:t (`cg=1060`) är inte bekräftade mot sajten.
2. **Fältextraheringen** – i stället för gissade CSS-klasser provas tre standardformat i tur och ordning: JSON-LD, `__NEXT_DATA__` och Open Graph-meta. Hittas inget pris kastas `ParseError` med en diagnos som säger vad sidan faktiskt innehöll.

Gör så här första gången du kör mot skarp sajt: `python pipeline/scrape.py --limit 1`. Fungerar det står det `via json-ld` (eller `__NEXT_DATA__`/`meta`) i utskriften. Fungerar det inte får du diagnosen, och då är det bara `pipeline/blocket.py` som behöver ändras – resten av kedjan är testad. Blockerar Blocket: skriv om samma fil mot sokbat.se, inget annat.

## Steg 4 – Värdera min båt (klart)
- `/vardera` med formulär: modell (dropdown ur databasen, grupperad på märke), årsmodell, motormärke, hk, motortimmar (valfritt), region och skick 1–5. Region- och motorlistorna byggs av `SELECT DISTINCT` på annonserna i stället för hårdkodade listor, så de speglar alltid vad vi har data för. Ingen e-post, inget konto.
- Beräkningen ligger i `lib/valuation.ts`: annonser för modellen inom ±2 årsmodeller de senaste 90 dagarna, borttagna annonser räknas som två observationer, skick justerar ±5 % per steg från 3. Svaret är median, p25–p75, antal observationer uppdelat på aktiva och borttagna, konfidens och de fem närmaste jämförbara annonserna.
- Verifierat: 11 tester (`npm test`), `typecheck`, `lint` och `build` går igenom, och sidan svarar 200 på tomt formulär, okänd modell, årsmodell utan träffar samt skräpinput (`ar=abc`, `skick=9`).

### Två val värda att känna till
- **Formuläret är en vanlig GET-form utan JavaScript.** Resultatet renderas på servern från query-parametrarna, så en värdering har en egen URL som går att skicka vidare eller lägga i en demo.
- **Värdet avrundas till närmaste tusen.** En värdering på kronan (792 488 kr) låtsas om en precision vi inte har. Rådata före skickjusteringen avrundas inte, så siffran går att stämma av mot annonserna.
- Viktningen är gjord genom att räkna en borttagen annons som två observationer, inte med en egen viktad percentilfunktion. Det gör att `/vardera` och modellsidan använder exakt samma `percentile()` och blir lätt att förklara på en "Så räknar vi"-sida i steg 6.

## Steg 5 – Annonskollen (klart, med samma förbehåll som steg 3)
- `/kolla` tar en annonslänk och svarar med avvikelse i procent mot medianen för modell och årsmodell ±2 år, plus dagar ute, prissänkningar och jämförbara annonser. Färgkodning enligt specen: grön under −5 %, gul ±5 %, röd över +5 %. Misslyckas hämtningen visas ett manuellt formulär med felet och parserns diagnos utskriven.
- Ordningen är: känner vi redan igen URL:en i `listing` använder vi vår egen data – då vet vi dessutom dagar ute och antal prissänkningar. Annars hämtas annonsen via `pipeline/fetch_ad.py`.
- Verifierat: alla tre färgerna (+31,1 % röd, −20,2 % grön, +0,6 % gul), databasvägen, det manuella formuläret, misslyckad hämtning, otillåten domän och okänd modell. `typecheck`, `lint`, `npm test` (11) och `python pipeline/test_pipeline.py` (23) går igenom.

### Två val värda att känna till
- **"Samma parser som scrapern" tas bokstavligt.** Parsern finns i Python, appen i TypeScript. I stället för att skriva en andra tolkning i TypeScript – som garanterat skulle glida isär från den första – anropar `lib/fetch-ad.ts` skriptet `pipeline/fetch_ad.py` med `execFile`. Då gäller samma extrahering, samma alias-matchning, samma robots.txt-kontroll och samma en-request-varannan-sekund även här. URL:en valideras mot en lista tillåtna domäner innan skriptet startas, och `execFile` går inte via något skal.
- **Annonsen som kollas räknas bort ur sitt eget jämförelseunderlag.** Annars jämförs den mot en median den själv drar åt sitt håll – med sex observationer var effekten direkt synlig.

Kvarstår: själva hämtningen av en ny annons är otestad av samma skäl som i steg 3 – ingen nätverksåtkomst i byggmiljön. Databasvägen, det manuella formuläret och hela felhanteringen är verifierade, och `fetch_ad.py` är verifierad mot cachad HTML (`--offline`).

## Öppna punkter inför nästa steg
- **`data/seed_demo.csv` är påhittad demodata**, genererad för att kunna bygga och testa UI:t innan riktig data finns. Källa `demo`, URL:er på `example.invalid`. Visa den aldrig för Matija som marknadsdata – seeda om från `data/seed.csv` när de riktiga raderna finns.
- "Median dagar på marknaden" blir 14 för alla modeller så länge datan bara är seedad, eftersom seed-regeln sätter `removed_at = first_seen + 14 dagar`. Riktig spridning kommer när `detect_removed.py` fått köra ett par dagar.
- "Snittprissänkning" och kolumnen "Prissänkningar" är 0/– på seedad data: `seed.csv` har ingen priskolumn över tid, så `price_history` får en enda post. Fylls av `scrape.py`.
- Seedade annonser med källan `blocket` räknas in i 2+2-regeln så fort scrapern kört mot samma källa. Det är rätt beteende när `seed.csv` innehåller riktiga Blocket-annonser som scrapern hittar igen, men handknackade rader som scrapern inte kan se kommer att markeras som borttagna efter fyra körningar.
- Statistiken finns i två implementationer: `lib/stats.ts` (som seed-skriptet använder) och `pipeline/stats.py`. De ska ge identiska siffror – ändras den ena måste den andra ändras med.

## Nästa steg
Kör `python pipeline/scrape.py --limit 1` mot skarp sajt och stäm av `pipeline/blocket.py` (se "Vad som återstår i steg 3"). Därefter steg 5 enligt `docs/PROTOTYP.md`: `/kolla` (annonskollen).
