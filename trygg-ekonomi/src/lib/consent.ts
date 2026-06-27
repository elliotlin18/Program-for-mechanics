import { prisma } from "@/lib/db";
import { audit } from "@/lib/audit";

const PSD2_MAX_DAYS = 180;

export async function grantConsent(
  careLinkId: string,
  scope: string,
  opts: { days?: number; actorId?: string | null; ip?: string | null } = {}
) {
  const days = Math.min(opts.days ?? PSD2_MAX_DAYS, PSD2_MAX_DAYS);
  const expiresAt = new Date(Date.now() + days * 86_400_000);
  const consent = await prisma.consent.create({
    data: { careLinkId, scope, expiresAt },
  });
  await audit({
    action: "CONSENT_GRANT",
    actorId: opts.actorId,
    careLinkId,
    targetType: "Consent",
    targetId: consent.id,
    metadata: { scope, expiresAt },
    ip: opts.ip,
  });
  return consent;
}

/**
 * Revoke all active consents and stop the link. Sync jobs must check
 * `hasActiveConsent` and refuse to run when this returns false. Optionally purge
 * stored financial data (GDPR right to erasure).
 */
export async function revokeConsent(
  careLinkId: string,
  opts: { actorId?: string | null; ip?: string | null; purgeData?: boolean } = {}
) {
  await prisma.consent.updateMany({
    where: { careLinkId, active: true },
    data: { active: false, revokedAt: new Date() },
  });
  await prisma.careLink.update({
    where: { id: careLinkId },
    data: { status: "revoked", revokedAt: new Date() },
  });
  await audit({
    action: "CONSENT_REVOKE",
    actorId: opts.actorId,
    careLinkId,
    ip: opts.ip,
  });

  if (opts.purgeData) {
    await purgeCareLinkData(careLinkId, opts);
  }
}

/** Delete all synced financial data for a link (keeps consent/audit history). */
export async function purgeCareLinkData(
  careLinkId: string,
  opts: { actorId?: string | null; ip?: string | null } = {}
) {
  const accounts = await prisma.account.findMany({
    where: { careLinkId },
    select: { id: true },
  });
  const accountIds = accounts.map((a) => a.id);
  await prisma.$transaction([
    prisma.transaction.deleteMany({ where: { accountId: { in: accountIds } } }),
    prisma.balanceSnapshot.deleteMany({ where: { accountId: { in: accountIds } } }),
    prisma.account.deleteMany({ where: { careLinkId } }),
  ]);
  await audit({
    action: "DATA_PURGE",
    actorId: opts.actorId,
    careLinkId,
    metadata: { accounts: accountIds.length },
    ip: opts.ip,
  });
}

export async function hasActiveConsent(careLinkId: string): Promise<boolean> {
  const c = await prisma.consent.findFirst({
    where: { careLinkId, active: true, expiresAt: { gt: new Date() } },
  });
  return Boolean(c);
}
