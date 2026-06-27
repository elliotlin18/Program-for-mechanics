// Thin Tink (Open Banking / AISP) client. You build on Tink's license — Tink is
// the regulated party. Use the SANDBOX environment first: https://console.tink.com
// Docs: https://docs.tink.com
//
// PROVIDER_MODE=mock returns deterministic fake data so account connection,
// sync, and the alert engine can be exercised end-to-end without Tink.

import { env, isMock } from "@/lib/env";
import { randomToken } from "@/lib/crypto";

const TINK_BASE = "https://api.tink.com";

export interface TinkAccount {
  tinkAccountId: string;
  name: string;
  maskedNumber?: string;
  balance: number;
  currency: string;
}

export interface TinkTransaction {
  tinkTxId: string;
  accountId: string; // tinkAccountId
  bookedAt: Date;
  amount: number; // negative = debit
  currency: string;
  payee?: string;
  description?: string;
}

/** URL that starts the hosted account-connection flow. */
export function buildTinkLinkUrl(state: string): string {
  if (isMock) {
    return `${env.APP_URL}/mock/tink?state=${encodeURIComponent(state)}`;
  }
  const params = new URLSearchParams({
    client_id: env.TINK_CLIENT_ID ?? "",
    redirect_uri: env.TINK_REDIRECT_URI ?? "",
    market: "SE",
    locale: "sv_SE",
    scope: "accounts:read,transactions:read",
    state,
  });
  return `https://link.tink.com/1.0/transactions/connect-accounts?${params.toString()}`;
}

/** Exchange the authorization code from the Tink Link callback for a token. */
export async function exchangeCodeForToken(code: string): Promise<{ accessToken: string }> {
  if (isMock) return { accessToken: `mock-token-${randomToken(8)}` };

  const res = await fetch(`${TINK_BASE}/api/v1/oauth/token`, {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      code,
      client_id: env.TINK_CLIENT_ID ?? "",
      client_secret: env.TINK_CLIENT_SECRET ?? "",
      grant_type: "authorization_code",
    }),
  });
  if (!res.ok) throw new Error(`Tink token exchange failed: ${res.status}`);
  const json = (await res.json()) as { access_token: string };
  return { accessToken: json.access_token };
}

export async function fetchAccounts(accessToken: string): Promise<TinkAccount[]> {
  if (isMock) return mockAccounts(accessToken);

  const res = await fetch(`${TINK_BASE}/data/v2/accounts`, {
    headers: { authorization: `Bearer ${accessToken}` },
  });
  if (!res.ok) throw new Error(`Tink accounts fetch failed: ${res.status}`);
  const json = (await res.json()) as { accounts: TinkRawAccount[] };
  return json.accounts.map((a) => ({
    tinkAccountId: a.id,
    name: a.name,
    maskedNumber: a.identifiers?.iban?.iban ?? a.identifiers?.financialInstitution?.accountNumber,
    balance: scaledToNumber(a.balances?.booked?.amount?.value),
    currency: a.balances?.booked?.amount?.currencyCode ?? "SEK",
  }));
}

export async function fetchTransactions(
  accessToken: string,
  tinkAccountId: string
): Promise<TinkTransaction[]> {
  if (isMock) return mockTransactions(tinkAccountId);

  const res = await fetch(
    `${TINK_BASE}/data/v2/transactions?accountIdIn=${encodeURIComponent(tinkAccountId)}`,
    { headers: { authorization: `Bearer ${accessToken}` } }
  );
  if (!res.ok) throw new Error(`Tink transactions fetch failed: ${res.status}`);
  const json = (await res.json()) as { transactions: TinkRawTx[] };
  return json.transactions.map((t) => ({
    tinkTxId: t.id,
    accountId: t.accountId,
    bookedAt: new Date(t.dates?.booked ?? t.dates?.value ?? Date.now()),
    amount: scaledToNumber(t.amount?.value),
    currency: t.amount?.currencyCode ?? "SEK",
    payee: t.merchantInformation?.merchantName ?? t.descriptions?.display,
    description: t.descriptions?.original,
  }));
}

/** Tink returns amounts as { unscaledValue, scale }. Convert to a JS number. */
function scaledToNumber(v?: { unscaledValue?: string; scale?: string }): number {
  if (!v?.unscaledValue) return 0;
  const unscaled = Number(v.unscaledValue);
  const scale = Number(v.scale ?? "0");
  return unscaled / Math.pow(10, scale);
}

// ---- Minimal shapes of the Tink v2 responses we read --------------------------
interface TinkRawAccount {
  id: string;
  name: string;
  identifiers?: {
    iban?: { iban?: string };
    financialInstitution?: { accountNumber?: string };
  };
  balances?: { booked?: { amount?: { value?: { unscaledValue?: string; scale?: string }; currencyCode?: string } } };
}
interface TinkRawTx {
  id: string;
  accountId: string;
  amount?: { value?: { unscaledValue?: string; scale?: string }; currencyCode?: string };
  dates?: { booked?: string; value?: string };
  descriptions?: { display?: string; original?: string };
  merchantInformation?: { merchantName?: string };
}

// ---- Deterministic mock data --------------------------------------------------
function mockAccounts(_token: string): TinkAccount[] {
  return [
    {
      tinkAccountId: "mock-acc-lonekonto",
      name: "Lönekonto",
      maskedNumber: "•••• 4821",
      balance: 48250.75,
      currency: "SEK",
    },
    {
      tinkAccountId: "mock-acc-sparkonto",
      name: "Sparkonto",
      maskedNumber: "•••• 9930",
      balance: 152000.0,
      currency: "SEK",
    },
  ];
}

function mockTransactions(tinkAccountId: string): TinkTransaction[] {
  if (tinkAccountId !== "mock-acc-lonekonto") return [];
  const day = (d: number) => new Date(2026, 5, d, 10, 0, 0); // June 2026
  const t = (
    id: string,
    d: number,
    amount: number,
    payee: string,
    description?: string
  ): TinkTransaction => ({
    tinkTxId: `mock-tx-${id}`,
    accountId: tinkAccountId,
    bookedAt: day(d),
    amount,
    currency: "SEK",
    payee,
    description,
  });
  return [
    // Recurring + ordinary history
    t("rent-04", 1, -8500, "Hyresvärden AB", "Hyra juni"),
    t("ica-1", 3, -642.5, "ICA Maxi", "Matinköp"),
    t("pension-in", 25, 21300, "Pensionsmyndigheten", "Pension"),
    t("pharmacy", 8, -312, "Apotek Hjärtat"),
    t("ica-2", 10, -489.9, "ICA Maxi", "Matinköp"),
    t("electric", 12, -1145, "Vattenfall", "El"),
    // Suspicious recent events the engine should catch:
    t("unknown-large", 19, -15000, "Okänd Mottagare", "Överföring"),
    t("new-payee", 20, -2400, "SnabbLån Direkt AB"),
    t("dup-1", 20, -799, "TV-Tjänst", "Prenumeration"),
    t("dup-2", 20, -799, "TV-Tjänst", "Prenumeration"),
  ];
}
