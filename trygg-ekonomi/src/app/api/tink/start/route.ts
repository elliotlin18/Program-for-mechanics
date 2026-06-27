import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";
import { buildTinkLinkUrl } from "@/lib/tink";
import { requireUser, UnauthorizedError } from "@/lib/session";
import { authorizeCareLink, ForbiddenError } from "@/lib/authz";
import { sign, randomToken } from "@/lib/crypto";
import { isProd, env } from "@/lib/env";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const careLinkId = req.nextUrl.searchParams.get("careLinkId");
  if (!careLinkId) return NextResponse.json({ error: "careLinkId required" }, { status: 400 });

  try {
    const user = await requireUser();
    // Only the senior connects their own bank.
    await authorizeCareLink(user.id, careLinkId, ["senior"]);
  } catch (e) {
    if (e instanceof UnauthorizedError) return NextResponse.redirect(`${env.APP_URL}/`);
    if (e instanceof ForbiddenError) return NextResponse.json({ error: "forbidden" }, { status: 403 });
    throw e;
  }

  const nonce = randomToken(16);
  cookies().set("te_tink", sign(JSON.stringify({ nonce, careLinkId })), {
    httpOnly: true,
    secure: isProd,
    sameSite: "lax",
    path: "/",
    maxAge: 600,
  });
  return NextResponse.redirect(buildTinkLinkUrl(nonce));
}
