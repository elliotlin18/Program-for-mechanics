import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { grantConsent, revokeConsent } from "@/lib/consent";
import { requireUser, UnauthorizedError } from "@/lib/session";
import { authorizeCareLink, ForbiddenError } from "@/lib/authz";
import { clientIp, rateLimit } from "@/lib/ratelimit";

export const dynamic = "force-dynamic";

const Body = z.object({
  careLinkId: z.string().min(1),
  action: z.enum(["grant", "revoke"]),
  scope: z.string().optional(),
  purge: z.boolean().optional(),
});

export async function POST(req: NextRequest) {
  const ip = clientIp(req);
  if (!rateLimit(`consent:${ip}`, 30, 60_000).ok) {
    return NextResponse.json({ error: "rate limited" }, { status: 429 });
  }

  let user;
  try {
    user = await requireUser();
  } catch (e) {
    if (e instanceof UnauthorizedError)
      return NextResponse.json({ error: "unauthorized" }, { status: 401 });
    throw e;
  }

  const parsed = Body.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return NextResponse.json({ error: "bad request" }, { status: 400 });
  const { careLinkId, action, scope, purge } = parsed.data;

  try {
    // Grant requires the senior; revoke is allowed from either side.
    await authorizeCareLink(
      user.id,
      careLinkId,
      action === "grant" ? ["senior"] : ["caregiver", "senior"]
    );
  } catch (e) {
    if (e instanceof ForbiddenError)
      return NextResponse.json({ error: "forbidden" }, { status: 403 });
    throw e;
  }

  if (action === "grant") {
    const c = await grantConsent(careLinkId, scope ?? "accounts:read,transactions:read", {
      actorId: user.id,
      ip,
    });
    return NextResponse.json({ ok: true, consentId: c.id });
  }

  await revokeConsent(careLinkId, { actorId: user.id, ip, purgeData: purge });
  return NextResponse.json({ ok: true });
}
