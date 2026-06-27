// Server-side session management. The cookie carries only a signed session id;
// all session state lives in the DB so we can revoke sessions and audit them.

import { cookies } from "next/headers";
import { prisma } from "@/lib/db";
import { sign, unsign } from "@/lib/crypto";
import { isProd } from "@/lib/env";

const COOKIE = "te_session";
const MAX_AGE_SECONDS = 60 * 60 * 24 * 14; // 14 days

export type SessionUser = {
  id: string;
  name: string | null;
  email: string | null;
};

export async function createSession(
  userId: string,
  meta?: { userAgent?: string; ip?: string }
): Promise<void> {
  const expiresAt = new Date(Date.now() + MAX_AGE_SECONDS * 1000);
  const session = await prisma.session.create({
    data: { userId, expiresAt, userAgent: meta?.userAgent, ip: meta?.ip },
  });
  cookies().set(COOKIE, sign(session.id), {
    httpOnly: true,
    secure: isProd,
    sameSite: "lax",
    path: "/",
    maxAge: MAX_AGE_SECONDS,
  });
}

export async function getCurrentUser(): Promise<SessionUser | null> {
  const raw = cookies().get(COOKIE)?.value;
  const sessionId = unsign(raw);
  if (!sessionId) return null;

  const session = await prisma.session.findUnique({
    where: { id: sessionId },
    include: { user: true },
  });
  if (!session || session.expiresAt < new Date()) return null;

  return {
    id: session.user.id,
    name: session.user.name,
    email: session.user.email,
  };
}

/** Throwing variant for routes/pages that must be authenticated. */
export async function requireUser(): Promise<SessionUser> {
  const user = await getCurrentUser();
  if (!user) throw new UnauthorizedError();
  return user;
}

export async function destroySession(): Promise<void> {
  const raw = cookies().get(COOKIE)?.value;
  const sessionId = unsign(raw);
  if (sessionId) {
    await prisma.session.deleteMany({ where: { id: sessionId } });
  }
  cookies().delete(COOKIE);
}

export class UnauthorizedError extends Error {
  constructor() {
    super("Unauthorized");
    this.name = "UnauthorizedError";
  }
}
