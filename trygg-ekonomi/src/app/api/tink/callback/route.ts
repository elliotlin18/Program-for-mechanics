import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";
import { prisma } from "@/lib/db";
import { exchangeCodeForToken } from "@/lib/tink";
import { syncCareLink } from "@/lib/sync";
import { requireUser } from "@/lib/session";
import { authorizeCareLink } from "@/lib/authz";
import { unsign } from "@/lib/crypto";
import { audit } from "@/lib/audit";
import { env } from "@/lib/env";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const sp = req.nextUrl.searchParams;
  const raw = cookies().get("te_tink")?.value;
  cookies().delete("te_tink");
  const payload = unsign(raw);
  if (!payload) return NextResponse.redirect(`${env.APP_URL}/senior?error=tink_state`);

  let parsed: { nonce: string; careLinkId: string };
  try {
    parsed = JSON.parse(payload);
  } catch {
    return NextResponse.redirect(`${env.APP_URL}/senior?error=tink_state`);
  }
  if (parsed.nonce !== sp.get("state")) {
    return NextResponse.redirect(`${env.APP_URL}/senior?error=tink_state`);
  }

  const user = await requireUser();
  await authorizeCareLink(user.id, parsed.careLinkId, ["senior"]);

  const { accessToken } = await exchangeCodeForToken(sp.get("code") ?? "mock");
  // We store only a non-sensitive connection reference, never the access token.
  await prisma.careLink.update({
    where: { id: parsed.careLinkId },
    data: { tinkConnectionId: `conn-${Date.now()}` },
  });
  await audit({ action: "BANK_CONNECT", actorId: user.id, careLinkId: parsed.careLinkId });

  // First sync runs immediately so the caregiver sees data right away.
  await syncCareLink(parsed.careLinkId, { accessToken, actorId: user.id });

  return NextResponse.redirect(`${env.APP_URL}/senior?connected=1`);
}
