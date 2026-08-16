# Båtvärde – prototyp

Läs docs/PROTOTYP.md först. Det är specen. Bygg exakt det, inget mer.
Bakgrund och affärslogik finns i docs/BAKGRUND.md – läs den en gång så du förstår varför.

## Vad vi bygger
En demo som visar vad begagnade båtar faktiskt säljs för (sålda/borttagna annonser vs listpriser).
Tre skärmar: modellsida, "värdera min båt", "annonskollen". En pipeline som hämtar annonser,
normaliserar, upptäcker borttagna och räknar statistik.

## Stack – ändra inte utan att fråga
- Next.js 14 App Router, TypeScript, Tailwind, shadcn/ui, Recharts
- Prisma + SQLite (data/boats.db). En fil. Ingen Postgres i prototypen.
- Pipeline i Python 3.11 (/pipeline): requests, beautifulsoup4, pyyaml. Playwright bara om nödvändigt.
- Allt körs lokalt. Ingen deploy, ingen auth, ingen e-post.

## Regler
- Följ byggordningen i docs/PROTOTYP.md steg 0–6. Ett steg i taget. Fråga innan du hoppar.
- Scraping: max 1 request per 2 sekunder, cacha all HTML i data/cache/, respektera robots.txt,
  ingen inloggning. Om Blocket blockerar: byt källa till sokbat.se, ändra inget annat.
- Seed-data (data/seed.csv) ska alltid fungera som fallback så att UI aldrig blockeras av scraping.
- Svenska i UI och kommentarer. Kod och variabelnamn på engelska.
- Håll det enkelt. Ingen abstraktion för framtida behov. Prototypen slängs eller skrivs om.
- Kör `npm run typecheck` och `npm run lint` innan du säger att ett steg är klart.
- Efter varje steg: skriv 3 rader i docs/PROGRESS.md om vad som gjordes och vad som återstår.

## Datamodell
Se avsnitt 3 i docs/PROTOTYP.md. Använd exakt de tabellerna: boats_model, listing, model_stats.

## Kommandon
- `npm run dev` – app
- `npm run db:seed` – läs data/seed.csv
- `python pipeline/scrape.py` – hämta annonser
- `python pipeline/detect_removed.py` – markera borttagna
- `python pipeline/stats.py` – räkna om model_stats
