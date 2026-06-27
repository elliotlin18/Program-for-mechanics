// Load/save the per-CareLink RuleConfig from AlertRule rows. Falls back to
// sensible defaults so a freshly created link is protected out of the box.

import { prisma } from "@/lib/db";
import { defaultRuleConfig } from "@/lib/alerts";
import { ALERT_TYPES, type AlertType, type RuleConfig } from "@/types";

export async function loadRuleConfig(careLinkId: string): Promise<RuleConfig> {
  const rows = await prisma.alertRule.findMany({ where: { careLinkId } });
  const cfg = defaultRuleConfig();
  for (const row of rows) {
    if (!ALERT_TYPES.includes(row.type as AlertType)) continue;
    cfg.enabled[row.type as AlertType] = row.enabled;
    const c = (row.config ?? {}) as Partial<RuleConfig>;
    Object.assign(cfg, stripEnabled(c));
  }
  return cfg;
}

/** Ensure default AlertRule rows exist for a new CareLink. */
export async function seedDefaultRules(careLinkId: string): Promise<void> {
  const cfg = defaultRuleConfig();
  await prisma.$transaction(
    ALERT_TYPES.map((type) =>
      prisma.alertRule.upsert({
        where: { careLinkId_type: { careLinkId, type } },
        update: {},
        create: { careLinkId, type, enabled: cfg.enabled[type], config: thresholds(cfg) },
      })
    )
  );
}

export async function setRuleEnabled(
  careLinkId: string,
  type: AlertType,
  enabled: boolean
): Promise<void> {
  await prisma.alertRule.upsert({
    where: { careLinkId_type: { careLinkId, type } },
    update: { enabled },
    create: { careLinkId, type, enabled, config: {} },
  });
}

export async function updateThresholds(
  careLinkId: string,
  patch: Partial<Omit<RuleConfig, "enabled">>
): Promise<void> {
  const rows = await prisma.alertRule.findMany({ where: { careLinkId } });
  await prisma.$transaction(
    rows.map((row) =>
      prisma.alertRule.update({
        where: { id: row.id },
        data: {
          config: { ...(row.config as object), ...patch },
        },
      })
    )
  );
}

function thresholds(cfg: RuleConfig): Record<string, number | undefined> {
  const { enabled, ...rest } = cfg;
  void enabled;
  return rest;
}

function stripEnabled(c: Partial<RuleConfig>): Partial<Omit<RuleConfig, "enabled">> {
  const { enabled, ...rest } = c;
  void enabled;
  return rest;
}
