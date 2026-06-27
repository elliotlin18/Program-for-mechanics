import { describe, it, expect } from "vitest";
import { evaluate, defaultRuleConfig } from "./alerts";
import { detectRecurring } from "./recurring";
import type { Tx, AccountSnapshot, BalancePoint } from "@/types";

const acc: AccountSnapshot = { id: "a1", balance: 50000, currency: "SEK" };
const tx = (p: Partial<Tx>): Tx => ({
  id: Math.random().toString(36).slice(2),
  bookedAt: new Date("2026-06-20T10:00:00Z"),
  amount: -100,
  currency: "SEK",
  payee: "ICA",
  description: null,
  ...p,
});

describe("LARGE_WITHDRAWAL", () => {
  it("flags a large withdrawal over the absolute limit", () => {
    const recent = [tx({ amount: -15000, payee: "Okänd" })];
    const out = evaluate(recent, [], acc, defaultRuleConfig());
    expect(out.some((a) => a.type === "LARGE_WITHDRAWAL")).toBe(true);
  });

  it("flags a withdrawal that is N× the historical average debit", () => {
    const history = Array.from({ length: 10 }, () => tx({ amount: -200 }));
    const recent = [tx({ amount: -2000, payee: "ICA" })]; // 10× avg, under abs limit
    const out = evaluate(recent, history, acc, defaultRuleConfig());
    expect(out.some((a) => a.type === "LARGE_WITHDRAWAL")).toBe(true);
  });

  it("respects disabled rules", () => {
    const cfg = defaultRuleConfig();
    cfg.enabled.LARGE_WITHDRAWAL = false;
    const out = evaluate([tx({ amount: -99999 })], [], acc, cfg);
    expect(out.some((a) => a.type === "LARGE_WITHDRAWAL")).toBe(false);
  });
});

describe("NEW_PAYEE", () => {
  it("flags a new payee not seen in history", () => {
    const history = [tx({ payee: "ICA" }), tx({ payee: "Hyra" })];
    const recent = [tx({ amount: -500, payee: "NyMottagareAB" })];
    const out = evaluate(recent, history, acc, defaultRuleConfig());
    expect(out.some((a) => a.type === "NEW_PAYEE")).toBe(true);
  });

  it("does not flag a known payee", () => {
    const history = [tx({ payee: "ICA" })];
    const recent = [tx({ amount: -200, payee: "ICA" })];
    const out = evaluate(recent, history, acc, defaultRuleConfig());
    expect(out.some((a) => a.type === "NEW_PAYEE")).toBe(false);
  });
});

describe("DUPLICATE_CHARGE", () => {
  it("flags two identical charges within 48h", () => {
    const a = tx({ amount: -499, payee: "Streaming", bookedAt: new Date("2026-06-20T08:00:00Z") });
    const b = tx({ amount: -499, payee: "Streaming", bookedAt: new Date("2026-06-20T20:00:00Z") });
    const out = evaluate([a, b], [], acc, defaultRuleConfig());
    expect(out.some((x) => x.type === "DUPLICATE_CHARGE")).toBe(true);
  });
});

describe("FREQUENCY_SPIKE", () => {
  it("flags an unusual number of transactions in one day", () => {
    // History: ~1/day across 10 distinct days.
    const history = Array.from({ length: 10 }, (_, i) =>
      tx({ bookedAt: new Date(`2026-05-${String(i + 1).padStart(2, "0")}T10:00:00Z`) })
    );
    const recent = Array.from({ length: 6 }, () =>
      tx({ bookedAt: new Date("2026-06-20T10:00:00Z") })
    );
    const out = evaluate(recent, history, acc, defaultRuleConfig());
    expect(out.some((a) => a.type === "FREQUENCY_SPIKE")).toBe(true);
  });
});

describe("BALANCE_DROP", () => {
  it("flags a >40% drop within the window", () => {
    const now = new Date("2026-06-20T12:00:00Z");
    const balanceHistory: BalancePoint[] = [
      { balance: 100000, takenAt: new Date("2026-06-18T12:00:00Z") },
      { balance: 90000, takenAt: new Date("2026-06-19T12:00:00Z") },
    ];
    const lowAcc = { ...acc, balance: 40000 }; // 60% below the 100k baseline
    const out = evaluate([], [], lowAcc, defaultRuleConfig(), { balanceHistory, now });
    expect(out.some((a) => a.type === "BALANCE_DROP")).toBe(true);
  });

  it("does not flag a modest drop", () => {
    const now = new Date("2026-06-20T12:00:00Z");
    const balanceHistory: BalancePoint[] = [
      { balance: 100000, takenAt: new Date("2026-06-19T12:00:00Z") },
    ];
    const out = evaluate([], [], { ...acc, balance: 95000 }, defaultRuleConfig(), {
      balanceHistory,
      now,
    });
    expect(out.some((a) => a.type === "BALANCE_DROP")).toBe(false);
  });
});

describe("MISSED_RECURRING", () => {
  const monthly = (payee: string, day: string, amount = -1200) =>
    tx({ payee, amount, bookedAt: new Date(`${day}T09:00:00Z`) });

  it("detects a monthly recurring pattern", () => {
    const history = [
      monthly("Hyresvärden", "2026-03-01"),
      monthly("Hyresvärden", "2026-04-01"),
      monthly("Hyresvärden", "2026-05-01"),
    ];
    const patterns = detectRecurring(history);
    expect(patterns.find((p) => p.payee === "Hyresvärden")).toBeTruthy();
  });

  it("flags a recurring bill that is overdue beyond the grace window", () => {
    const history = [
      monthly("Hyresvärden", "2026-03-01"),
      monthly("Hyresvärden", "2026-04-01"),
      monthly("Hyresvärden", "2026-05-01"),
    ];
    // Expected ~2026-06-01; now is well past grace and no payment in recent.
    const now = new Date("2026-06-20T12:00:00Z");
    const out = evaluate([], history, acc, defaultRuleConfig(), { now });
    expect(out.some((a) => a.type === "MISSED_RECURRING")).toBe(true);
  });

  it("does not flag when the recurring bill was paid", () => {
    const history = [
      monthly("Hyresvärden", "2026-03-01"),
      monthly("Hyresvärden", "2026-04-01"),
      monthly("Hyresvärden", "2026-05-01"),
    ];
    const now = new Date("2026-06-20T12:00:00Z");
    const recent = [monthly("Hyresvärden", "2026-06-01")];
    const out = evaluate(recent, history, acc, defaultRuleConfig(), { now });
    expect(out.some((a) => a.type === "MISSED_RECURRING")).toBe(false);
  });
});

describe("dedupe keys", () => {
  it("emits a stable dedupeKey per alert", () => {
    const out = evaluate([tx({ amount: -15000, id: "fixed-id" })], [], acc, defaultRuleConfig());
    const large = out.find((a) => a.type === "LARGE_WITHDRAWAL");
    expect(large?.dedupeKey).toBe("LARGE_WITHDRAWAL:fixed-id");
  });

  it("flags a new payee only once even across several transactions", () => {
    const recent = [
      tx({ amount: -100, payee: "NyButik" }),
      tx({ amount: -250, payee: "NyButik" }),
    ];
    const out = evaluate(recent, [tx({ payee: "ICA" })], acc, defaultRuleConfig());
    expect(out.filter((a) => a.type === "NEW_PAYEE")).toHaveLength(1);
  });

  it("collapses both halves of a duplicate pair into one alert", () => {
    const a = tx({ amount: -499, payee: "Strm", bookedAt: new Date("2026-06-20T08:00:00Z") });
    const b = tx({ amount: -499, payee: "Strm", bookedAt: new Date("2026-06-20T20:00:00Z") });
    const out = evaluate([a, b], [], acc, defaultRuleConfig());
    expect(out.filter((x) => x.type === "DUPLICATE_CHARGE")).toHaveLength(1);
  });
});
