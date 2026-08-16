# Progress

## Steg 0 – Repo och seed (klart)
- Next.js 14 + TypeScript + Tailwind + Prisma/SQLite uppsatt i `batvarde/`; datamodellen från specen ligger i `prisma/schema.prisma` som tabellerna `boats_model`, `listing`, `model_stats`.
- `npm run db:seed` läser `pipeline/aliases.yaml` (10 modeller) och `data/seed.csv`, sätter `removed_at = first_seen + 14 dagar` för `status=removed`, `last_seen = idag` för aktiva, och räknar om `model_stats`.
- Återstår: `data/seed.csv` innehåller fortfarande de fem EXEMPEL-raderna från startkitet – de ska bytas mot 100–200 riktiga rader från Blocket.

## Steg 1 – Modellsida (klart)
- `/bat/[brand]/[model]` visar spec, prisintervall p25/median/p75 för aktiva (listpris) vs borttagna (troligen sålt), Recharts-linje med medianpris per vecka i 12 veckor, och tabell över de senaste 20 annonserna (pris, år, region, dagar ute, status, prissänkningar).
- Konfidens enligt specen: hög ≥ 30 observationer, medel 10–29, låg < 10, räknat på aktiva + borttagna de senaste 90 dagarna. `/` är tills vidare ett enkelt modellindex – riktig startsida byggs i steg 6.
- Verifierat: `npm run typecheck`, `npm run lint` och `npm run build` går igenom, och alla tio modellsidor svarar 200 mot produktionsservern.

## Öppna punkter inför nästa steg
- **`data/seed_demo.csv` är påhittad demodata**, genererad för att kunna bygga och testa UI:t innan riktig data finns. Källa `demo`, URL:er på `example.invalid`. Visa den aldrig för Matija som marknadsdata – seeda om från `data/seed.csv` när de riktiga raderna finns.
- "Median dagar på marknaden" blir 14 för alla modeller så länge datan är seedad, eftersom seed-regeln sätter `removed_at = first_seen + 14 dagar`. Riktig spridning kommer först när `detect_removed.py` sätter datumen (steg 3).
- "Snittprissänkning" och kolumnen "Prissänkningar" är 0/– på seedad data: `seed.csv` har ingen priskolumn över tid, så `price_history` får en enda post. Fylls av `scrape.py` i steg 3.
- Statistiken räknas i `lib/stats.ts` och skrivs till `model_stats` av seed-skriptet. `pipeline/stats.py` i steg 2 ska räkna exakt samma sak och skriva till samma tabell.

## Nästa steg
Steg 2 enligt `docs/PROTOTYP.md`: `pipeline/stats.py`.
