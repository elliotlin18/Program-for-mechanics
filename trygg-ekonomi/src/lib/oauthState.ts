// Helper for the OAuth/OIDC "state" round-trip with CSRF protection.
// The signed state lives in an httpOnly cookie; the redirect carries only the
// nonce. On callback we require the cookie's nonce to equal the returned nonce.

import { cookies } from "next/headers";
import { sign, unsign, randomToken } from "@/lib/crypto";
import { isProd } from "@/lib/env";

const COOKIE = "te_oauth";

export interface OAuthState {
  nonce: string;
  intent: "login" | "accept";
  token?: string; // invite token for the accept flow
}

export function beginState(intent: OAuthState["intent"], token?: string): string {
  const nonce = randomToken(16);
  const state: OAuthState = { nonce, intent, token };
  cookies().set(COOKIE, sign(JSON.stringify(state)), {
    httpOnly: true,
    secure: isProd,
    sameSite: "lax",
    path: "/",
    maxAge: 600, // 10 minutes to complete auth
  });
  return nonce;
}

export function consumeState(returnedNonce: string | null): OAuthState | null {
  const raw = cookies().get(COOKIE)?.value;
  cookies().delete(COOKIE);
  const payload = unsign(raw);
  if (!payload || !returnedNonce) return null;
  let state: OAuthState;
  try {
    state = JSON.parse(payload) as OAuthState;
  } catch {
    return null;
  }
  if (state.nonce !== returnedNonce) return null;
  return state;
}
