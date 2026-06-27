// Recurring-payment detection over a transaction history. Used by the
// MISSED_RECURRING rule. Pure and deterministic — no IO.
//
// Heuristic: group debits by payee, look for a roughly monthly cadence
// (median gap 25–35 days) with a similar amount, over at least 3 occurrences.

import type { Tx } from "@/types";

export interface RecurringPattern {
  payee: string;
  /** Median absolute amount of the recurring debit. */
  amount: number;
  /** Median gap in days between occurrences. */
  cadenceDays: number;
  /** Booking date of the most recent occurrence. */
  lastSeen: Date;
  occurrences: number;
}

const DAY = 86_400_000;

function median(nums: number[]): number {
  if (nums.length === 0) return 0;
  const s = [...nums].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

export function detectRecurring(history: Tx[]): RecurringPattern[] {
  const byPayee = new Map<string, Tx[]>();
  for (const t of history) {
    if (t.amount >= 0 || !t.payee) continue; // debits with a payee only
    const key = t.payee.toLowerCase();
    (byPayee.get(key) ?? byPayee.set(key, []).get(key)!).push(t);
  }

  const patterns: RecurringPattern[] = [];
  for (const txs of byPayee.values()) {
    if (txs.length < 3) continue;
    const sorted = [...txs].sort(
      (a, b) => a.bookedAt.getTime() - b.bookedAt.getTime()
    );

    const gaps: number[] = [];
    for (let i = 1; i < sorted.length; i++) {
      gaps.push(
        (sorted[i]!.bookedAt.getTime() - sorted[i - 1]!.bookedAt.getTime()) / DAY
      );
    }
    const cadence = median(gaps);
    if (cadence < 25 || cadence > 35) continue; // ~monthly only, for MVP

    const amounts = sorted.map((t) => Math.abs(t.amount));
    const last = sorted[sorted.length - 1]!;
    patterns.push({
      payee: last.payee!,
      amount: median(amounts),
      cadenceDays: cadence,
      lastSeen: last.bookedAt,
      occurrences: sorted.length,
    });
  }
  return patterns;
}
