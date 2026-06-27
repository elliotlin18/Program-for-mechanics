"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { requireUser } from "@/lib/session";
import { authorizeCareLink } from "@/lib/authz";
import { randomToken } from "@/lib/crypto";
import { seedDefaultRules, setRuleEnabled, updateThresholds } from "@/lib/rules";
import { revokeConsent, purgeCareLinkData } from "@/lib/consent";
import { syncCareLink } from "@/lib/sync";
import { audit } from "@/lib/audit";
import { ALERT_TYPES, type AlertType } from "@/types";

const CreateLink = z.object({
  seniorName: z.string().trim().min(1).max(120),
  seniorEmail: z.string().email().optional().or(z.literal("")),
  seniorPhone: z.string().trim().max(40).optional().or(z.literal("")),
});

/** Caregiver creates a CareLink and gets an invite link to share. */
export async function createCareLink(formData: FormData) {
  const user = await requireUser();
  const parsed = CreateLink.safeParse({
    seniorName: formData.get("seniorName"),
    seniorEmail: formData.get("seniorEmail") ?? "",
    seniorPhone: formData.get("seniorPhone") ?? "",
  });
  if (!parsed.success) throw new Error("Ogiltiga uppgifter");

  const inviteToken = randomToken(24);
  const link = await prisma.careLink.create({
    data: {
      caregiverId: user.id,
      seniorName: parsed.data.seniorName,
      seniorEmail: parsed.data.seniorEmail || null,
      seniorPhone: parsed.data.seniorPhone || null,
      inviteToken,
    },
  });
  await seedDefaultRules(link.id);
  await audit({
    action: "CARELINK_CREATE",
    actorId: user.id,
    careLinkId: link.id,
    metadata: { seniorName: parsed.data.seniorName },
  });
  // TODO(live): deliver the invite via the notification provider (SMS/email).
  redirect(`/dashboard/${link.id}`);
}

/** Caregiver triggers a manual sync (background refresh otherwise). */
export async function triggerSync(formData: FormData) {
  const user = await requireUser();
  const careLinkId = String(formData.get("careLinkId"));
  await authorizeCareLink(user.id, careLinkId, ["caregiver"]);
  await syncCareLink(careLinkId, { actorId: user.id });
  revalidatePath(`/dashboard/${careLinkId}`);
}

export async function dismissAlert(formData: FormData) {
  const user = await requireUser();
  const alertId = String(formData.get("alertId"));
  const alert = await prisma.alert.findUnique({ where: { id: alertId } });
  if (!alert) throw new Error("Saknas");
  await authorizeCareLink(user.id, alert.careLinkId, ["caregiver"]);
  await prisma.alert.update({
    where: { id: alertId },
    data: { status: "dismissed", dismissedAt: new Date() },
  });
  await audit({ action: "ALERT_ACK", actorId: user.id, careLinkId: alert.careLinkId, targetId: alertId });
  revalidatePath(`/dashboard/${alert.careLinkId}`);
}

export async function markAlertsSeen(careLinkId: string) {
  const user = await requireUser();
  await authorizeCareLink(user.id, careLinkId, ["caregiver"]);
  await prisma.alert.updateMany({
    where: { careLinkId, status: "new" },
    data: { status: "seen", seenAt: new Date() },
  });
}

export async function toggleRule(formData: FormData) {
  const user = await requireUser();
  const careLinkId = String(formData.get("careLinkId"));
  const type = String(formData.get("type")) as AlertType;
  const enabled = formData.get("enabled") === "on";
  if (!ALERT_TYPES.includes(type)) throw new Error("Okänd regel");
  await authorizeCareLink(user.id, careLinkId, ["caregiver"]);
  await setRuleEnabled(careLinkId, type, enabled);
  await audit({ action: "RULE_UPDATE", actorId: user.id, careLinkId, metadata: { type, enabled } });
  revalidatePath(`/dashboard/${careLinkId}/settings`);
}

const Thresholds = z.object({
  largeWithdrawalAbsolute: z.coerce.number().positive().optional(),
  balanceDropPct: z.coerce.number().min(0.05).max(0.95).optional(),
});

export async function saveThresholds(formData: FormData) {
  const user = await requireUser();
  const careLinkId = String(formData.get("careLinkId"));
  await authorizeCareLink(user.id, careLinkId, ["caregiver"]);
  const parsed = Thresholds.safeParse({
    largeWithdrawalAbsolute: formData.get("largeWithdrawalAbsolute") || undefined,
    balanceDropPct: formData.get("balanceDropPct") || undefined,
  });
  if (!parsed.success) throw new Error("Ogiltiga trösklar");
  await updateThresholds(careLinkId, parsed.data);
  await audit({ action: "RULE_UPDATE", actorId: user.id, careLinkId, metadata: parsed.data });
  revalidatePath(`/dashboard/${careLinkId}/settings`);
}

/** Senior (or caregiver) revokes the link. Senior may also purge all data. */
export async function revokeLink(formData: FormData) {
  const user = await requireUser();
  const careLinkId = String(formData.get("careLinkId"));
  const purge = formData.get("purge") === "on";
  const { role } = await authorizeCareLink(user.id, careLinkId, ["caregiver", "senior"]);
  await revokeConsent(careLinkId, { actorId: user.id, purgeData: purge });
  revalidatePath(role === "senior" ? "/senior" : "/dashboard");
  redirect(role === "senior" ? "/senior" : "/dashboard");
}

/** Senior exercises GDPR erasure without revoking (rare, but supported). */
export async function purgeData(formData: FormData) {
  const user = await requireUser();
  const careLinkId = String(formData.get("careLinkId"));
  await authorizeCareLink(user.id, careLinkId, ["senior"]);
  await purgeCareLinkData(careLinkId, { actorId: user.id });
  revalidatePath("/senior");
}
