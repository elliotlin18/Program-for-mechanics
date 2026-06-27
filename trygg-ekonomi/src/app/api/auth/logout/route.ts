import { NextRequest, NextResponse } from "next/server";
import { destroySession, getCurrentUser } from "@/lib/session";
import { audit } from "@/lib/audit";
import { env } from "@/lib/env";

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  const user = await getCurrentUser();
  if (user) await audit({ action: "LOGOUT", actorId: user.id });
  await destroySession();
  return NextResponse.redirect(`${env.APP_URL}/`, { status: 303 });
}
