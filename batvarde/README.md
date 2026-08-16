# Båtvärde – prototyp

- CLAUDE.md – regler för Claude Code
- docs/BAKGRUND.md – varför vi bygger det här
- docs/PROTOTYP.md – specen och byggordningen (steg 0–6)
- docs/FORSTA_PROMPT.md – första prompten
- docs/PROGRESS.md – vad som är gjort och vad som återstår
- pipeline/aliases.yaml – de tio startmodellerna
- data/seed.csv – seed-data (byt EXEMPEL-raderna mot 100–200 riktiga från Blocket)

## Kom igång

```bash
npm install
cp .env.example .env   # DATABASE_URL till data/boats.db
npm run db:push        # skapar data/boats.db enligt prisma/schema.prisma
npm run db:seed        # läser pipeline/aliases.yaml + data/seed.csv
npm run db:count       # verifierar modeller, annonser och statistik
npm run dev            # http://localhost:3000
```

Klart hittills: modellsidan `/bat/[brand]/[model]` (steg 1). `/` är ett enkelt modellindex
tills startsidan byggs i steg 6.

## Demodata

`data/seed_demo.csv` är **påhittad** – 190 rader genererade för att kunna bygga och testa UI:t
innan riktig data finns. Den känns igen på källan `demo` och URL:er på `example.invalid`.

```bash
npm run db:seed -- data/seed_demo.csv
```

Använd den aldrig i en demo för någon utomstående – det är inte marknadsdata. Så fort
`data/seed.csv` har riktiga Blocket-rader: ta bort `data/boats.db`, kör `db:push` och `db:seed`
igen, och radera `data/seed_demo.csv`.
