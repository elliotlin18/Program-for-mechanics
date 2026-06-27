export type Severity = "low" | "medium" | "high";

export type AlertType =
  | "LARGE_WITHDRAWAL"
  | "NEW_PAYEE"
  | "MISSED_RECURRING"
  | "BALANCE_DROP"
  | "FREQUENCY_SPIKE"
  | "DUPLICATE_CHARGE";

export const ALERT_TYPES: AlertType[] = [
  "LARGE_WITHDRAWAL",
  "NEW_PAYEE",
  "MISSED_RECURRING",
  "BALANCE_DROP",
  "FREQUENCY_SPIKE",
  "DUPLICATE_CHARGE",
];

export interface Tx {
  id: string;
  bookedAt: Date;
  amount: number; // negative = debit
  currency: string;
  payee?: string | null;
  description?: string | null;
}

export interface AccountSnapshot {
  id: string;
  balance: number;
  currency: string;
}

/** A historical balance reading, used to detect rapid drops. */
export interface BalancePoint {
  balance: number;
  takenAt: Date;
}

export interface RuleConfig {
  largeWithdrawalAbsolute?: number; // e.g. 10000
  largeWithdrawalMultiplier?: number; // e.g. 5 (x 90-day avg debit)
  balanceDropPct?: number; // e.g. 0.4
  balanceDropDays?: number; // e.g. 3
  frequencySpikeMultiplier?: number; // e.g. 3
  missedRecurringGraceDays?: number; // e.g. 5 (window past expected date)
  enabled: Record<AlertType, boolean>;
}

export interface EvaluatedAlert {
  type: AlertType;
  severity: Severity;
  message: string;
  transactionId?: string;
  /** Stable key so the same condition doesn't create duplicate Alert rows. */
  dedupeKey: string;
}

/** Extra context the engine needs beyond the transaction batch. */
export interface EvaluationContext {
  /** Balance readings over time (most recent last), incl. the current balance. */
  balanceHistory?: BalancePoint[];
  /** "Now" for deterministic testing; defaults to Date.now(). */
  now?: Date;
}
