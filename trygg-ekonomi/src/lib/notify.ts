// Notification delivery + audit. In mock mode (or with no NOTIFY_API_KEY) we log
// instead of sending, and still record a NotificationLog row for the audit trail.

import { prisma } from "@/lib/db";
import { env, isMock } from "@/lib/env";

export type Channel = "email" | "sms" | "push";

export interface NotifyInput {
  alertId: string;
  channel: Channel;
  to: string; // email / phone / device ref
  subject: string;
  body: string;
}

export async function sendNotification(input: NotifyInput): Promise<boolean> {
  let success = false;
  let error: string | null = null;

  try {
    if (isMock || !env.NOTIFY_API_KEY) {
      console.info(
        `[notify:${input.channel}] → ${input.to}: ${input.subject} — ${input.body}`
      );
      success = true;
    } else {
      // TODO(live): call your provider (e.g. Postmark/SendGrid for email,
      // 46elks/Twilio for SMS). Keep payloads minimal — no full account data.
      success = await deliverViaProvider(input);
    }
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
    success = false;
  }

  await prisma.notificationLog.create({
    data: {
      alertId: input.alertId,
      channel: input.channel,
      toRef: maskRecipient(input.to),
      success,
      error,
    },
  });
  return success;
}

async function deliverViaProvider(_input: NotifyInput): Promise<boolean> {
  throw new Error("Live notification provider not configured");
}

/** Store a masked recipient in logs, never the full address/number. */
function maskRecipient(to: string): string {
  if (to.includes("@")) {
    const [user, domain] = to.split("@");
    return `${(user ?? "").slice(0, 2)}***@${domain ?? ""}`;
  }
  return to.length > 4 ? `***${to.slice(-4)}` : "***";
}
