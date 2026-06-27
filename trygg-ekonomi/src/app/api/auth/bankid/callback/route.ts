import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { handleCallback } from "@/lib/bankid";
import { consumeState } from "@/lib/oauthState";
import { createSession } from "@/lib/session";
import { grantConsent } from "@/lib/consent";
import { audit } from "@/lib/audit";
import { clientIp } from "@/lib/ratelimit";
import { env } from "@/lib/env";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const sp = req.nextUrl.searchParams;
  const state = consumeState(sp.get("state"));
  if (!state) {
    return NextResponse.redirect(`${env.APP_URL}/?error=state`);
  }

  const ip = clientIp(req);
  let identity;
  try {
    identity = await handleCallback({
      code: sp.get("code"),
      mockName: sp.get("mock_name"),
    });
  } catch {
    return NextResponse.redirect(`${env.APP_URL}/?error=auth`);
  }

  // Upsert the user by their opaque personal-id ref (never the raw PNO).
  const user = await prisma.user.upsert({
    where: { personalIdRef: identity.subjectRef },
    update: identity.name ? { name: identity.name } : {},
    create: { personalIdRef: identity.subjectRef, name: identity.name },
  });

  await createSession(user.id, { userAgent: req.headers.get("user-agent") ?? undefined, ip });
  await audit({ action: "LOGIN", actorId: user.id, ip });

  // Senior accepting an invite: link them, grant consent, then connect the bank.
  if (state.intent === "accept" && state.token) {
    const link = await prisma.careLink.findUnique({ where: { inviteToken: state.token } });
    if (link && link.status === "invited") {
      await prisma.careLink.update({
        where: { id: link.id },
        data: { seniorId: user.id, status: "active", activatedAt: new Date() },
      });
      await grantConsent(link.id, "accounts:read,transactions:read", {
        actorId: user.id,
        ip,
      });
      return NextResponse.redirect(`${env.APP_URL}/api/tink/start?careLinkId=${link.id}`);
    }
    // Invite invalid/already used — send them to their panel.
    return NextResponse.redirect(`${env.APP_URL}/senior?error=invite`);
  }

  return NextResponse.redirect(`${env.APP_URL}/dashboard`);
}
