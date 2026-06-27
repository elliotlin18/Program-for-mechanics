import { NextRequest, NextResponse } from "next/server";
import crypto from "node:crypto";
import { prisma } from "@/lib/db";
import { syncCareLink } from "@/lib/sync";
import { env } from "@/lib/env";

export const dynamic = "force-dynamic";

/**
 * Tink refresh webhook. We verify the signature against TINK_WEBHOOK_SECRET
 * before trusting anything, then trigger a sync for the affected CareLink.
 */
export async function POST(req: NextRequest) {
  const bodyText = await req.text();

  if (env.TINK_WEBHOOK_SECRET) {
    const signature = req.headers.get("x-tink-signature") ?? "";
    if (!verifySignature(bodyText, signature, env.TINK_WEBHOOK_SECRET)) {
      return NextResponse.json({ error: "bad signature" }, { status: 401 });
    }
  } else if (env.NODE_ENV === "production") {
    return NextResponse.json({ error: "webhook secret not configured" }, { status: 500 });
  }

  let event: { type?: string; context?: { externalUserId?: string } } | null;
  try {
    event = JSON.parse(bodyText);
  } catch {
    return NextResponse.json({ error: "bad payload" }, { status: 400 });
  }
  if (!event) return NextResponse.json({ error: "bad payload" }, { status: 400 });

  // On a refresh event, map the external user id to a CareLink and sync.
  // (How you map depends on how you provision Tink users; for the MVP we sync
  // all active links touched by the connection.)
  if (event.type?.includes("refreshed")) {
    const links = await prisma.careLink.findMany({
      where: { status: "active", tinkConnectionId: { not: null } },
      select: { id: true },
    });
    for (const l of links) {
      await syncCareLink(l.id).catch((e) => console.error("webhook sync failed", l.id, e));
    }
  }

  return NextResponse.json({ received: true });
}

function verifySignature(body: string, signature: string, secret: string): boolean {
  const expected = crypto.createHmac("sha256", secret).update(body).digest("hex");
  const a = Buffer.from(signature);
  const b = Buffer.from(expected);
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}
