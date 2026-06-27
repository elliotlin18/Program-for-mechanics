import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { requireUser, UnauthorizedError } from "@/lib/session";
import { authorizeCareLink, ForbiddenError } from "@/lib/authz";
import { audit } from "@/lib/audit";
import { clientIp } from "@/lib/ratelimit";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const careLinkId = req.nextUrl.searchParams.get("careLinkId");
  if (!careLinkId) return NextResponse.json({ error: "careLinkId required" }, { status: 400 });

  let user;
  try {
    user = await requireUser();
  } catch (e) {
    if (e instanceof UnauthorizedError)
      return NextResponse.json({ error: "unauthorized" }, { status: 401 });
    throw e;
  }

  try {
    await authorizeCareLink(user.id, careLinkId, ["caregiver", "senior"]);
  } catch (e) {
    if (e instanceof ForbiddenError)
      return NextResponse.json({ error: "forbidden" }, { status: 403 });
    throw e;
  }

  const alerts = await prisma.alert.findMany({
    where: { careLinkId },
    orderBy: { createdAt: "desc" },
    take: 50,
  });
  await audit({
    action: "DATA_ACCESS",
    actorId: user.id,
    careLinkId,
    targetType: "Alert",
    ip: clientIp(req),
  });
  return NextResponse.json({ alerts });
}
