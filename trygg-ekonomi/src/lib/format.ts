import type { Severity } from "@/types";

export function kr(amount: number, currency = "SEK"): string {
  return new Intl.NumberFormat("sv-SE", {
    style: "currency",
    currency,
    maximumFractionDigits: 2,
  }).format(amount);
}

export function date(d: Date): string {
  return new Intl.DateTimeFormat("sv-SE", { dateStyle: "medium" }).format(d);
}

export function dateTime(d: Date): string {
  return new Intl.DateTimeFormat("sv-SE", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(d);
}

export const severityLabel: Record<Severity, string> = {
  low: "Låg",
  medium: "Medel",
  high: "Hög",
};

export function severityClasses(sev: string): string {
  switch (sev) {
    case "high":
      return "bg-red-50 text-red-700 border-red-200";
    case "medium":
      return "bg-amber-50 text-amber-700 border-amber-200";
    default:
      return "bg-neutral-100 text-neutral-600 border-neutral-200";
  }
}

export const alertTypeLabel: Record<string, string> = {
  LARGE_WITHDRAWAL: "Stort uttag",
  NEW_PAYEE: "Ny mottagare",
  MISSED_RECURRING: "Missad räkning",
  BALANCE_DROP: "Fallande saldo",
  FREQUENCY_SPIKE: "Många transaktioner",
  DUPLICATE_CHARGE: "Dubbelbetalning",
};
