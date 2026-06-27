import type {
  Tx,
  AccountSnapshot,
  RuleConfig,
  EvaluatedAlert,
  AlertType,
  EvaluationContext,
} from "@/types";
import { detectRecurring } from "@/lib/recurring";

const DEFAULTS = {
  largeWithdrawalAbsolute: 10000,
  largeWithdrawalMultiplier: 5,
  balanceDropPct: 0.4,
  balanceDropDays: 3,
  frequencySpikeMultiplier: 3,
  missedRecurringGraceDays: 5,
};

const DAY = 86_400_000;

const debit = (t: Tx) => t.amount < 0;
const abs = (t: Tx) => Math.abs(t.amount);
const dayKey = (d: Date) => d.toISOString().slice(0, 10);
const kr = (n: number) => n.toLocaleString("sv-SE");

/**
 * Pure, deterministic alert engine. `recent` is the new batch to evaluate;
 * `history` is the trailing window used for baselines (e.g. last 12 months).
 * `ctx` carries balance snapshots and an injectable clock. No DB, no IO —
 * fully unit-testable. See alerts.test.ts.
 */
export function evaluate(
  recent: Tx[],
  history: Tx[],
  account: AccountSnapshot,
  cfg: RuleConfig,
  ctx: EvaluationContext = {}
): EvaluatedAlert[] {
  const alerts: EvaluatedAlert[] = [];
  const on = (t: AlertType) => cfg.enabled?.[t];
  const now = ctx.now ?? new Date();

  const debits = history.filter(debit);
  const avgDebit =
    debits.length > 0 ? debits.reduce((s, t) => s + abs(t), 0) / debits.length : 0;

  // ---- LARGE_WITHDRAWAL ----------------------------------------------------
  if (on("LARGE_WITHDRAWAL")) {
    const absLimit = cfg.largeWithdrawalAbsolute ?? DEFAULTS.largeWithdrawalAbsolute;
    const mult = cfg.largeWithdrawalMultiplier ?? DEFAULTS.largeWithdrawalMultiplier;
    for (const t of recent.filter(debit)) {
      if (abs(t) >= absLimit || (avgDebit > 0 && abs(t) >= avgDebit * mult)) {
        alerts.push({
          type: "LARGE_WITHDRAWAL",
          severity: "high",
          message: `Stor transaktion: ${kr(abs(t))} ${t.currency}${
            t.payee ? ` till ${t.payee}` : ""
          }.`,
          transactionId: t.id,
          dedupeKey: `LARGE_WITHDRAWAL:${t.id}`,
        });
      }
    }
  }

  // ---- NEW_PAYEE -----------------------------------------------------------
  if (on("NEW_PAYEE")) {
    const knownPayees = new Set(
      history.map((t) => (t.payee ?? "").toLowerCase()).filter(Boolean)
    );
    for (const t of recent.filter((t) => debit(t) && t.payee)) {
      if (!knownPayees.has((t.payee ?? "").toLowerCase())) {
        alerts.push({
          type: "NEW_PAYEE",
          severity: "medium",
          message: `Ny mottagare: ${t.payee}.`,
          transactionId: t.id,
          // Dedupe per payee, not per transaction — one alert per new recipient.
          dedupeKey: `NEW_PAYEE:${(t.payee ?? "").toLowerCase()}`,
        });
      }
    }
  }

  // ---- DUPLICATE_CHARGE — same amount + payee within 48h -------------------
  if (on("DUPLICATE_CHARGE")) {
    const all = [...history, ...recent].filter(debit);
    for (const t of recent.filter(debit)) {
      const dup = all.find(
        (o) =>
          o.id !== t.id &&
          o.payee === t.payee &&
          abs(o) === abs(t) &&
          Math.abs(o.bookedAt.getTime() - t.bookedAt.getTime()) <= 48 * 3600 * 1000
      );
      if (dup) {
        alerts.push({
          type: "DUPLICATE_CHARGE",
          severity: "low",
          message: `Möjlig dubbelbetalning: ${kr(abs(t))} ${t.currency}${
            t.payee ? ` till ${t.payee}` : ""
          }.`,
          transactionId: t.id,
          // Canonical key so both halves of a duplicate pair collapse into one.
          dedupeKey: `DUPLICATE_CHARGE:${(t.payee ?? "").toLowerCase()}:${abs(t)}:${dayKey(
            t.bookedAt
          )}`,
        });
      }
    }
  }

  // ---- FREQUENCY_SPIKE — daily count vs history daily average -------------
  if (on("FREQUENCY_SPIKE")) {
    const mult = cfg.frequencySpikeMultiplier ?? DEFAULTS.frequencySpikeMultiplier;
    const histDays = new Map<string, number>();
    for (const t of history)
      histDays.set(dayKey(t.bookedAt), (histDays.get(dayKey(t.bookedAt)) ?? 0) + 1);
    const avgPerDay =
      histDays.size > 0
        ? [...histDays.values()].reduce((a, b) => a + b, 0) / histDays.size
        : 0;
    const recentDays = new Map<string, number>();
    for (const t of recent)
      recentDays.set(dayKey(t.bookedAt), (recentDays.get(dayKey(t.bookedAt)) ?? 0) + 1);
    for (const [day, count] of recentDays) {
      if (avgPerDay > 0 && count >= avgPerDay * mult) {
        alerts.push({
          type: "FREQUENCY_SPIKE",
          severity: "low",
          message: `Ovanligt många transaktioner ${day} (${count} st).`,
          dedupeKey: `FREQUENCY_SPIKE:${day}`,
        });
      }
    }
  }

  // ---- BALANCE_DROP — balance fell > X% within Y days ----------------------
  if (on("BALANCE_DROP")) {
    const pct = cfg.balanceDropPct ?? DEFAULTS.balanceDropPct;
    const days = cfg.balanceDropDays ?? DEFAULTS.balanceDropDays;
    const points = (ctx.balanceHistory ?? [])
      .slice()
      .sort((a, b) => a.takenAt.getTime() - b.takenAt.getTime());
    const current = account.balance;
    const windowStart = now.getTime() - days * DAY;
    // Highest balance within the window acts as the baseline to compare against.
    const baseline = points
      .filter((p) => p.takenAt.getTime() >= windowStart)
      .reduce((max, p) => Math.max(max, p.balance), Number.NEGATIVE_INFINITY);
    if (baseline > 0 && current < baseline) {
      const drop = (baseline - current) / baseline;
      if (drop >= pct) {
        alerts.push({
          type: "BALANCE_DROP",
          severity: "high",
          message: `Saldot har fallit ${Math.round(drop * 100)}% på ${days} dagar (${kr(
            Math.round(baseline)
          )} → ${kr(Math.round(current))} ${account.currency}).`,
          dedupeKey: `BALANCE_DROP:${account.id}:${dayKey(now)}`,
        });
      }
    }
  }

  // ---- MISSED_RECURRING — a known recurring bill didn't arrive on time -----
  if (on("MISSED_RECURRING")) {
    const grace = cfg.missedRecurringGraceDays ?? DEFAULTS.missedRecurringGraceDays;
    const patterns = detectRecurring(history);
    const recentPayees = new Set(
      recent.filter(debit).map((t) => (t.payee ?? "").toLowerCase())
    );
    for (const p of patterns) {
      const expected = p.lastSeen.getTime() + p.cadenceDays * DAY;
      const overdueBy = (now.getTime() - expected) / DAY;
      const paidRecently = recentPayees.has(p.payee.toLowerCase());
      // Overdue beyond the grace window, and not seen in the recent batch.
      if (!paidRecently && overdueBy > grace) {
        alerts.push({
          type: "MISSED_RECURRING",
          severity: "medium",
          message: `Återkommande betalning till ${p.payee} (~${kr(
            Math.round(p.amount)
          )} ${account.currency}) verkar ha uteblivit.`,
          dedupeKey: `MISSED_RECURRING:${p.payee.toLowerCase()}:${dayKey(
            new Date(expected)
          )}`,
        });
      }
    }
  }

  // Collapse any duplicates produced within this batch (keep the first).
  const seen = new Set<string>();
  return alerts.filter((a) => {
    if (seen.has(a.dedupeKey)) return false;
    seen.add(a.dedupeKey);
    return true;
  });
}

export const defaultRuleConfig = (): RuleConfig => ({
  ...DEFAULTS,
  enabled: {
    LARGE_WITHDRAWAL: true,
    NEW_PAYEE: true,
    MISSED_RECURRING: true,
    BALANCE_DROP: true,
    FREQUENCY_SPIKE: true,
    DUPLICATE_CHARGE: true,
  },
});
