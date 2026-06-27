// Append-only audit logging. Every privacy-relevant action (consent changes,
// data access, purges, logins) must call this. Failures here are logged but
// never block the primary operation.

import { prisma } from "@/lib/db";

export type AuditAction =
  | "LOGIN"
  | "LOGOUT"
  | "CARELINK_CREATE"
  | "CONSENT_GRANT"
  | "CONSENT_REVOKE"
  | "BANK_CONNECT"
  | "DATA_ACCESS"
  | "DATA_SYNC"
  | "DATA_PURGE"
  | "ALERT_ACK"
  | "RULE_UPDATE";

export async function audit(params: {
  action: AuditAction;
  actorId?: string | null;
  careLinkId?: string | null;
  targetType?: string;
  targetId?: string;
  metadata?: Record<string, unknown>;
  ip?: string | null;
}): Promise<void> {
  try {
    await prisma.auditLog.create({
      data: {
        action: params.action,
        actorId: params.actorId ?? null,
        careLinkId: params.careLinkId ?? null,
        targetType: params.targetType,
        targetId: params.targetId,
        metadata: params.metadata as object | undefined,
        ip: params.ip ?? null,
      },
    });
  } catch (err) {
    console.error("audit log failed", { action: params.action, err });
  }
}
