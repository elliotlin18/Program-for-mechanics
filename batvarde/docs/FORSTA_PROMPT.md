# Första prompten att klistra in i Claude Code

Läs CLAUDE.md, docs/BAKGRUND.md och docs/PROTOTYP.md. Vi gör Steg 0.

Sätt upp Next.js 14 med TypeScript, Tailwind och shadcn/ui i /app, Prisma med SQLite
(data/boats.db) och datamodellen från specen (boats_model, listing, model_stats).

Skriv ett seed-skript (`npm run db:seed`) som:
1. läser pipeline/aliases.yaml och skapar boats_model-rader,
2. läser data/seed.csv och skapar listing-rader (matcha brand+model mot boats_model;
   sätt removed_at = first_seen + 14 dagar om status=removed; last_seen = idag för active).

Lägg till npm-scripts för dev, typecheck, lint, db:seed. Skapa tom docs/PROGRESS.md.
Kör seed och verifiera med ett count-skript att modeller och annonser finns.

Fråga mig innan du gör något utanför Steg 0. När Steg 0 är grönt: skriv i PROGRESS.md och stanna.
