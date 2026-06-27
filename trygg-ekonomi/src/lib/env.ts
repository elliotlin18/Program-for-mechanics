// Central, validated access to environment configuration.
// Fail fast in production if security-critical secrets are missing.

import { z } from "zod";

const schema = z.object({
  NODE_ENV: z.enum(["development", "test", "production"]).default("development"),
  APP_URL: z.string().url().default("http://localhost:3000"),
  DATABASE_URL: z.string().optional(),
  APP_ENCRYPTION_KEY: z.string().optional(),
  SESSION_SECRET: z.string().optional(),
  PROVIDER_MODE: z.enum(["mock", "live"]).default("mock"),

  BANKID_BROKER_CLIENT_ID: z.string().optional(),
  BANKID_BROKER_CLIENT_SECRET: z.string().optional(),
  BANKID_BROKER_ISSUER_URL: z.string().optional(),
  BANKID_BROKER_REDIRECT_URI: z.string().optional(),

  TINK_CLIENT_ID: z.string().optional(),
  TINK_CLIENT_SECRET: z.string().optional(),
  TINK_REDIRECT_URI: z.string().optional(),
  TINK_WEBHOOK_SECRET: z.string().optional(),

  NOTIFY_API_KEY: z.string().optional(),
  NOTIFY_FROM: z.string().default("trygg@example.com"),
});

const parsed = schema.safeParse(process.env);

if (!parsed.success) {
  // Don't leak values; just surface which keys are malformed.
  console.error("Invalid environment configuration:", parsed.error.flatten().fieldErrors);
  throw new Error("Invalid environment configuration");
}

export const env = parsed.data;

export const isProd = env.NODE_ENV === "production";
export const isMock = env.PROVIDER_MODE === "mock";

// In production, security secrets are mandatory.
if (isProd) {
  const missing: string[] = [];
  if (!env.APP_ENCRYPTION_KEY) missing.push("APP_ENCRYPTION_KEY");
  if (!env.SESSION_SECRET) missing.push("SESSION_SECRET");
  if (!env.DATABASE_URL) missing.push("DATABASE_URL");
  if (missing.length) {
    throw new Error(`Missing required production secrets: ${missing.join(", ")}`);
  }
}
