// Application-layer encryption-at-rest for sensitive free-text fields and
// helpers for signing tokens / hashing identifiers.
//
// Uses Node's built-in crypto only (no third-party dependency = smaller supply
// chain). AES-256-GCM gives confidentiality + integrity. The stored format is:
//   v1.<iv-base64>.<authTag-base64>.<ciphertext-base64>
// so the scheme version is self-describing and future-proof.

import crypto from "node:crypto";
import { env } from "@/lib/env";

const SCHEME = "v1";

function getKey(): Buffer {
  const raw = env.APP_ENCRYPTION_KEY;
  if (!raw) {
    if (env.NODE_ENV === "production") {
      throw new Error("APP_ENCRYPTION_KEY is required in production");
    }
    // Deterministic, clearly-insecure dev fallback so the app runs without setup.
    // Never used in production (guarded above).
    return crypto.createHash("sha256").update("trygg-ekonomi-dev-key").digest();
  }
  const key = Buffer.from(raw, "base64");
  if (key.length !== 32) {
    throw new Error("APP_ENCRYPTION_KEY must be 32 bytes (base64-encoded)");
  }
  return key;
}

/** Encrypt a UTF-8 string. Returns null for null/undefined input (passthrough). */
export function encrypt(plain: string | null | undefined): string | null {
  if (plain === null || plain === undefined) return null;
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv("aes-256-gcm", getKey(), iv);
  const enc = Buffer.concat([cipher.update(plain, "utf8"), cipher.final()]);
  const tag = cipher.getAuthTag();
  return `${SCHEME}.${iv.toString("base64")}.${tag.toString("base64")}.${enc.toString(
    "base64"
  )}`;
}

/** Decrypt a value produced by `encrypt`. Returns null for null input. */
export function decrypt(stored: string | null | undefined): string | null {
  if (stored === null || stored === undefined) return null;
  const parts = stored.split(".");
  if (parts.length !== 4 || parts[0] !== SCHEME) {
    // Not in our format (e.g. legacy plaintext) — return as-is to avoid data loss.
    return stored;
  }
  const [, ivB64, tagB64, dataB64] = parts;
  const decipher = crypto.createDecipheriv(
    "aes-256-gcm",
    getKey(),
    Buffer.from(ivB64!, "base64")
  );
  decipher.setAuthTag(Buffer.from(tagB64!, "base64"));
  const dec = Buffer.concat([
    decipher.update(Buffer.from(dataB64!, "base64")),
    decipher.final(),
  ]);
  return dec.toString("utf8");
}

function getSessionSecret(): Buffer {
  const raw = env.SESSION_SECRET;
  if (raw) return Buffer.from(raw, "base64");
  if (env.NODE_ENV === "production") {
    throw new Error("SESSION_SECRET is required in production");
  }
  return crypto.createHash("sha256").update("trygg-ekonomi-dev-session").digest();
}

/** Sign a payload with HMAC-SHA256, returning `<payload>.<sig>` (base64url). */
export function sign(payload: string): string {
  const sig = crypto
    .createHmac("sha256", getSessionSecret())
    .update(payload)
    .digest("base64url");
  return `${payload}.${sig}`;
}

/** Verify a signed token; returns the payload or null if tampered/invalid. */
export function unsign(token: string | undefined | null): string | null {
  if (!token) return null;
  const idx = token.lastIndexOf(".");
  if (idx < 0) return null;
  const payload = token.slice(0, idx);
  const sig = token.slice(idx + 1);
  const expected = crypto
    .createHmac("sha256", getSessionSecret())
    .update(payload)
    .digest("base64url");
  const a = Buffer.from(sig);
  const b = Buffer.from(expected);
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) return null;
  return payload;
}

/** Stable, non-reversible reference for a personal identifier (BankID `sub`). */
export function hashSubject(sub: string): string {
  return crypto
    .createHmac("sha256", getSessionSecret())
    .update(`pno-ref:${sub}`)
    .digest("hex");
}

/** URL-safe random token for invites, state params, etc. */
export function randomToken(bytes = 32): string {
  return crypto.randomBytes(bytes).toString("base64url");
}

/** Constant-time string comparison helper. */
export function safeEqual(a: string, b: string): boolean {
  const ba = Buffer.from(a);
  const bb = Buffer.from(b);
  if (ba.length !== bb.length) return false;
  return crypto.timingSafeEqual(ba, bb);
}
