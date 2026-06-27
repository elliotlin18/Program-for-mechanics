# Trygg Ekonomi

Consent-based financial caregiving for the Nordic market. An adult child gets
discreet, early-warning visibility into an aging parent's finances — without
taking over. Read-only (Open Banking / AISP via Tink), BankID for identity and
consent, a rule-based alert engine, and a senior-controlled transparency panel.

See `BUILD_SPEC` and the market analysis for architecture, data model, and the
M1→M5 milestone plan.

## What's implemented

- **Auth & onboarding** — BankID login (via broker, mocked in dev), caregiver
  creates a CareLink, senior accepts with one approval, grants consent, connects
  their bank. Sessions are server-side with signed httpOnly cookies.
- **Open Banking sync** — Tink Link connection + account/transaction sync,
  webhook endpoint with signature verification. Runs fully against deterministic
  **mock data** when `PROVIDER_MODE=mock` (the default).
- **Alert engine** — all six MVP rules, including `BALANCE_DROP` (via balance
  snapshots) and `MISSED_RECURRING` (via a recurring-payment detector). Pure,
  deterministic, unit-tested (`src/lib/alerts.test.ts`).
- **Caregiver dashboard** — balances, recent transactions, active alerts
  (dismiss/seen), per-rule settings and thresholds.
- **Senior transparency panel** — see exactly what is shared and with whom,
  revoke instantly, optionally erase all stored data (GDPR).
- **Security & privacy** — AES-256-GCM encryption-at-rest for sensitive fields,
  per-CareLink authorization on every access, immutable audit log, rate limiting,
  CSRF-protected OAuth state, security headers. No raw personal numbers stored.

## Öppna programmet (snabbast — mock-läge, inga externa konton)

Förutsätter **Node 18+** och **Docker** (för en lokal Postgres). På din egen dator:

```bash
cd trygg-ekonomi
npm install      # installerar beroenden
npm run setup    # skapar .env + nycklar, startar DB, migrerar, seedar
npm run dev      # startar appen
```

Öppna sedan **<http://localhost:3000>** i webbläsaren.

> Har du ingen Docker? Starta en egen Postgres, peka `DATABASE_URL` i `.env`
> mot den och kör `npm run setup` igen. `npm run setup` är säkert att köra om.

### Quick start (manuellt, om du hellre gör stegen själv)

```bash
docker compose up -d            # Postgres
cp .env.example .env            # PROVIDER_MODE stannar "mock"
npm run db:migrate              # skapa databasen
npm run dev
```

Then walk the flow:

1. Open <http://localhost:3000> → **Kom igång med BankID** → sign in as e.g.
   "Demo Vårdgivare" (mock BankID).
2. **Bjud in närstående** → enter a name → you get an invite link.
3. Open the invite link (incognito works) → **Godkänn med BankID** as the parent
   → **Anslut testbank**. Mock data loads and the alert engine runs.
4. Back as the caregiver, the dashboard shows balances, transactions and alerts
   (large withdrawal, new payee, duplicate charge from the fixtures).

`npm test` runs the alert-engine unit tests.

## Going live

Set `PROVIDER_MODE=live` and fill in the BankID broker (Criipto/Signicat) and
Tink **sandbox** credentials in `.env`. Then:

- Implement broker JWT signature verification (`src/lib/bankid.ts`) against the
  issuer JWKS before trusting id_tokens.
- Wire Tink access-token storage/refresh for `syncCareLink` and the webhook
  (`src/lib/sync.ts`, `src/app/api/webhooks/tink/route.ts`).
- Plug a real notification provider into `src/lib/notify.ts`.

Search for `TODO(live)` / `TODO(claude-code)` for each integration point.

## Compliance note

Launch on Tink's AISP license (no own Finansinspektionen license needed for the
MVP). Read-only only — no money movement. GDPR + a light AML policy required.
Framtidsfullmakt signing must be physical + witnessed (lag 2017:310); the digital
feature covers preparation, storage and activation, not e-signing.
