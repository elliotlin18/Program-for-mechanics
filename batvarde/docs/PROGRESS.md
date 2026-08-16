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

## Öppna punkter inför nästa steg
- **`data/seed_demo.csv` är påhittad demodata**, genererad för att kunna bygga och testa UI:t innan riktig data finns. Källa `demo`, URL:er på `example.invalid`. Visa den aldrig för Matija som marknadsdata – seeda om från `data/seed.csv` när de riktiga raderna finns.
- "Median dagar på marknaden" blir 14 för alla modeller så länge datan är seedad, eftersom seed-regeln sätter `removed_at = first_seen + 14 dagar`. Riktig spridning kommer först när `detect_removed.py` sätter datumen (steg 3).
- "Snittprissänkning" och kolumnen "Prissänkningar" är 0/– på seedad data: `seed.csv` har ingen priskolumn över tid, så `price_history` får en enda post. Fylls av `scrape.py` i steg 3.
- Statistiken finns i två implementationer: `lib/stats.ts` (som seed-skriptet använder) och `pipeline/stats.py`. De ska ge identiska siffror – ändras den ena måste den andra ändras med.

## Nästa steg
Steg 3 enligt `docs/PROTOTYP.md`: `pipeline/scrape.py` och `pipeline/detect_removed.py`.
