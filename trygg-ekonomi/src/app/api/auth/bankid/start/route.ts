import { NextRequest, NextResponse } from "next/server";
import { buildAuthUrl } from "@/lib/bankid";
import { beginState } from "@/lib/oauthState";
import { rateLimit, clientIp } from "@/lib/ratelimit";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const ip = clientIp(req);
  if (!rateLimit(`bankid-start:${ip}`, 20, 60_000).ok) {
    return NextResponse.json({ error: "rate limited" }, { status: 429 });
  }

  const intentParam = req.nextUrl.searchParams.get("intent");
  const intent = intentParam === "accept" ? "accept" : "login";
  const token = req.nextUrl.searchParams.get("token") ?? undefined;

  const nonce = beginState(intent, token);
  return NextResponse.redirect(buildAuthUrl(nonce));
}
