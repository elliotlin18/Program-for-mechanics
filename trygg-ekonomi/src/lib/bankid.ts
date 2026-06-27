// BankID via a broker (Criipto or Signicat) — OIDC. We never integrate raw
// BankID, and we never persist the raw personal number: handleCallback returns
// an opaque, stable subject ref (HMAC of the broker `sub`).
//
// PROVIDER_MODE=mock runs a local stand-in (see /mock/bankid) so the whole
// onboarding flow is demoable without broker credentials.

import { env, isMock } from "@/lib/env";
import { hashSubject } from "@/lib/crypto";

/** Build the URL the browser is redirected to in order to authenticate. */
export function buildAuthUrl(state: string): string {
  if (isMock) {
    return `${env.APP_URL}/mock/bankid?state=${encodeURIComponent(state)}`;
  }
  const params = new URLSearchParams({
    client_id: env.BANKID_BROKER_CLIENT_ID ?? "",
    redirect_uri: env.BANKID_BROKER_REDIRECT_URI ?? "",
    response_type: "code",
    scope: "openid",
    // Force Swedish BankID at the broker. Exact value is broker-specific.
    acr_values: "urn:grn:authn:se:bankid",
    state,
  });
  return `${env.BANKID_BROKER_ISSUER_URL}/authorize?${params.toString()}`;
}

export interface BankIdIdentity {
  subjectRef: string;
  name?: string;
}

/**
 * Complete authentication. In live mode `params.code` is exchanged for an
 * id_token at the broker. In mock mode we read a display name supplied by the
 * local mock page and synthesize a stable subject ref from it.
 */
export async function handleCallback(params: {
  code?: string | null;
  mockName?: string | null;
}): Promise<BankIdIdentity> {
  if (isMock) {
    const name = (params.mockName ?? "Testperson").trim() || "Testperson";
    return { subjectRef: hashSubject(`mock:${name.toLowerCase()}`), name };
  }

  if (!params.code) throw new Error("Missing authorization code");

  const tokenRes = await fetch(`${env.BANKID_BROKER_ISSUER_URL}/token`, {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "authorization_code",
      code: params.code,
      redirect_uri: env.BANKID_BROKER_REDIRECT_URI ?? "",
      client_id: env.BANKID_BROKER_CLIENT_ID ?? "",
      client_secret: env.BANKID_BROKER_CLIENT_SECRET ?? "",
    }),
  });
  if (!tokenRes.ok) throw new Error(`BankID token exchange failed: ${tokenRes.status}`);
  const token = (await tokenRes.json()) as { id_token?: string };
  if (!token.id_token) throw new Error("No id_token in broker response");

  // Decode the JWT payload (signature verification against the broker JWKS is a
  // TODO before production — use `jose` + the issuer's JWKS endpoint).
  const claims = decodeJwtPayload(token.id_token);
  const sub = claims.sub;
  if (!sub) throw new Error("No sub claim in id_token");
  return {
    subjectRef: hashSubject(sub),
    name: claims.name ?? claims.given_name,
  };
}

function decodeJwtPayload(jwt: string): {
  sub?: string;
  name?: string;
  given_name?: string;
} {
  const part = jwt.split(".")[1];
  if (!part) throw new Error("Malformed id_token");
  return JSON.parse(Buffer.from(part, "base64url").toString("utf8"));
}
