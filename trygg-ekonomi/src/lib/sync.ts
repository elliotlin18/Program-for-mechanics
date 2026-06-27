// Orchestrates a data sync for one CareLink:
//   fetch (Tink) → persist (encrypted) → snapshot balance → evaluate rules →
//   create deduped alerts → notify caregiver. Refuses to run without consent.

import { prisma } from "@/lib/db";
import { encrypt, decrypt } from "@/lib/crypto";
import { hasActiveConsent } from "@/lib/consent";
import { loadRuleConfig } from "@/lib/rules";
import { evaluate } from "@/lib/alerts";
import { sendNotification } from "@/lib/notify";
import { audit } from "@/lib/audit";
import { fetchAccounts, fetchTransactions } from "@/lib/tink";
import type { Tx, BalancePoint, EvaluatedAlert } from "@/types";

export interface SyncResult {
  skipped?: "no-consent";
  accounts: number;
  newTransactions: number;
  newAlerts: number;
}

export async function syncCareLink(
  careLinkId: string,
  opts: { accessToken?: string; actorId?: string | null } = {}
): Promise<SyncResult> {
  if (!(await hasActiveConsent(careLinkId))) {
    return { skipped: "no-consent", accounts: 0, newTransactions: 0, newAlerts: 0 };
  }

  // In live mode the access token comes from the stored Tink connection.
  const token = opts.accessToken ?? "mock";

  const careLink = await prisma.careLink.findUnique({
    where: { id: careLinkId },
    include: { caregiver: true },
  });
  if (!careLink) throw new Error("CareLink not found");

  const remoteAccounts = await fetchAccounts(token);
  let newTransactions = 0;
  let newAlertCount = 0;

  for (const ra of remoteAccounts) {
    const account = await prisma.account.upsert({
      where: { careLinkId_tinkAccountId: { careLinkId, tinkAccountId: ra.tinkAccountId } },
      update: { name: ra.name, balance: ra.balance, currency: ra.currency },
      create: {
        careLinkId,
        tinkAccountId: ra.tinkAccountId,
        name: ra.name,
        maskedNumber: encrypt(ra.maskedNumber),
        balance: ra.balance,
        currency: ra.currency,
      },
    });

    await prisma.balanceSnapshot.create({
      data: { accountId: account.id, balance: ra.balance },
    });

    const remoteTxs = await fetchTransactions(token, ra.tinkAccountId);
    const existing = await prisma.transaction.findMany({
      where: { accountId: account.id },
      select: { tinkTxId: true },
    });
    const known = new Set(existing.map((e) => e.tinkTxId));
    const fresh = remoteTxs.filter((t) => !known.has(t.tinkTxId));
    // The first import only establishes a baseline; novelty/baseline rules
    // (new payee, missed recurring, frequency, balance drop) would otherwise
    // fire for every legitimate transaction. Absolute rules still apply.
    const isFirstSync = existing.length === 0;

    // Persist new transactions (payee/description encrypted at rest).
    for (const t of fresh) {
      await prisma.transaction.create({
        data: {
          accountId: account.id,
          tinkTxId: t.tinkTxId,
          bookedAt: t.bookedAt,
          amount: t.amount,
          currency: t.currency,
          payee: encrypt(t.payee),
          description: encrypt(t.description),
        },
      });
    }
    newTransactions += fresh.length;

    // Build decrypted history + recent for the engine. Tx.id is our DB id so
    // alert.transactionId maps straight back to the row.
    const dbTxs = await prisma.transaction.findMany({
      where: { accountId: account.id },
      orderBy: { bookedAt: "asc" },
    });
    const all: Tx[] = dbTxs.map((t) => ({
      id: t.id,
      bookedAt: t.bookedAt,
      amount: Number(t.amount),
      currency: t.currency,
      payee: decrypt(t.payee),
      description: decrypt(t.description),
    }));
    const freshTinkIds = new Set(fresh.map((f) => f.tinkTxId));
    const freshDbIds = new Set(
      dbTxs.filter((t) => freshTinkIds.has(t.tinkTxId)).map((t) => t.id)
    );
    const recent = all.filter((t) => freshDbIds.has(t.id));
    const history = all.filter((t) => !freshDbIds.has(t.id));

    const snaps = await prisma.balanceSnapshot.findMany({
      where: { accountId: account.id },
      orderBy: { takenAt: "asc" },
    });
    const balanceHistory: BalancePoint[] = snaps.map((s) => ({
      balance: Number(s.balance),
      takenAt: s.takenAt,
    }));

    const cfg = await loadRuleConfig(careLinkId);
    if (isFirstSync) {
      cfg.enabled = {
        ...cfg.enabled,
        NEW_PAYEE: false,
        MISSED_RECURRING: false,
        FREQUENCY_SPIKE: false,
        BALANCE_DROP: false,
      };
    }
    const evaluated = evaluate(
      recent,
      history,
      { id: account.id, balance: ra.balance, currency: ra.currency },
      cfg,
      { balanceHistory }
    );

    newAlertCount += await persistAlerts(careLinkId, evaluated, careLink.caregiver.email);
  }

  await audit({
    action: "DATA_SYNC",
    actorId: opts.actorId,
    careLinkId,
    metadata: { accounts: remoteAccounts.length, newTransactions, newAlerts: newAlertCount },
  });

  return { accounts: remoteAccounts.length, newTransactions, newAlerts: newAlertCount };
}

async function persistAlerts(
  careLinkId: string,
  evaluated: EvaluatedAlert[],
  caregiverEmail: string | null
): Promise<number> {
  let created = 0;
  for (const ev of evaluated) {
    // Dedupe on (careLinkId, dedupeKey) — see the unique index in the schema.
    const existing = await prisma.alert.findFirst({
      where: { careLinkId, dedupeKey: ev.dedupeKey },
    });
    if (existing) continue;

    const alert = await prisma.alert.create({
      data: {
        careLinkId,
        type: ev.type,
        severity: ev.severity,
        message: ev.message,
        dedupeKey: ev.dedupeKey,
        transactionId: ev.transactionId,
      },
    });
    created += 1;

    // Notify the caregiver for anything above "low".
    if (caregiverEmail && ev.severity !== "low") {
      await sendNotification({
        alertId: alert.id,
        channel: "email",
        to: caregiverEmail,
        subject: "Trygg Ekonomi — ny varning",
        body: ev.message,
      });
    }
  }
  return created;
}
