# Båtvärde – prototyp

- CLAUDE.md – regler för Claude Code
- docs/BAKGRUND.md – varför vi bygger det här
- docs/PROTOTYP.md – specen och byggordningen (steg 0–6)
- docs/FORSTA_PROMPT.md – första prompten
- docs/PROGRESS.md – vad som är gjort och vad som återstår
- pipeline/aliases.yaml – de tio startmodellerna
- data/seed.csv – seed-data (byt EXEMPEL-raderna mot 100–200 riktiga från Blocket)

## Installera – en gång

**Windows.** Klistra in i PowerShell:

```powershell
irm https://raw.githubusercontent.com/elliotlin18/Program-for-mechanics/claude/batvarde-project-setup-ub31pi/batvarde/install.ps1 -OutFile "$env:TEMP\batvarde-install.ps1"; powershell -ExecutionPolicy Bypass -File "$env:TEMP\batvarde-install.ps1"
```

I PowerShell är `curl` ett alias för `Invoke-WebRequest` och förstår varken `-fsSL` eller
`| bash` – använd raden ovan, inte curl-raden. Filen sparas i temp-mappen först, så du kan
öppna och läsa den innan du kör om du vill.

**Mac och Linux.** Klistra in i Terminal:

```bash
curl -fsSL https://raw.githubusercontent.com/elliotlin18/Program-for-mechanics/claude/batvarde-project-setup-ub31pi/batvarde/install.sh | bash
```

Installationen hämtar koden till `~/Batvarde`, lägger en **Båtvärde-ikon på skrivbordet** och
startar programmet. Kräver git och Node.js 18+; saknas något säger den till med länk.

Vill du läsa skriptet innan du kör det – rimligt – hämta det först med `curl -O <adressen ovan>`,
öppna filen och kör `bash install.sh`.

## Starta

Dubbelklicka **Båtvärde-ikonen på skrivbordet**. Den hämtar senaste versionen automatiskt och
startar. Sätt `BATVARDE_NO_UPDATE=1` om du vill hoppa över uppdateringen.

Utan ikon går det lika bra att dubbelklicka **`start.command`** (Mac) eller **`start.bat`**
(Windows) i projektmappen. På Linux: `./start.sh`.

Windows-filerna (`install.bat`, `start.bat`) är tunna omslag kring `install.ps1` och `start.ps1`
– portkoll, väntan på servern och frågan om demodata går inte att göra pålitligt i batch.

Filen installerar paket, skapar databasen, läser in `data/seed.csv`, startar servern och öppnar
webbläsaren. Nästa gång hoppar den över allt som redan är gjort och startar direkt. Är port 3000
upptagen tar den nästa lediga. Stäng med Ctrl+C i fönstret.

Är `data/seed.csv` fortfarande de fem EXEMPEL-raderna frågar den om du vill fylla på med
`data/seed_demo.csv` – 190 påhittade annonser, se avsnittet Demodata nedan.

Node.js 18 eller senare krävs. Saknas det säger startfilen till och länkar till nedladdningen.

## Kom igång manuellt

```bash
npm install
cp .env.example .env   # DATABASE_URL till data/boats.db
npm run db:push        # skapar data/boats.db enligt prisma/schema.prisma
npm run db:seed        # läser pipeline/aliases.yaml + data/seed.csv
npm run db:count       # verifierar modeller, annonser och statistik
npm test               # tester för värderingen
npm run dev            # http://localhost:3000
```

Skärmar: `/` (startsida med sök och heta modeller), `/bat/[brand]/[model]` (modellsida),
`/vardera` (värdering), `/kolla` (annonskollen) och `/sa-raknar-vi`.

## Pipeline

```bash
pip install -r pipeline/requirements.txt

python pipeline/scrape.py --limit 1     # kör så här första gången, se docs/PROGRESS.md
python pipeline/scrape.py               # alla modeller
python pipeline/detect_removed.py       # 2+2-regeln
python pipeline/stats.py                # räknar om model_stats
python pipeline/test_pipeline.py        # 23 tester, kräver ingen nätverksåtkomst
```

`scrape.py --offline` kör mot cachad HTML i `data/cache/` utan att göra några anrop.

`pipeline/blocket.py` är skriven utan tillgång till Blockets riktiga HTML och behöver
stämmas av mot sajten – läs "Vad som återstår i steg 3" i `docs/PROGRESS.md` först.

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
