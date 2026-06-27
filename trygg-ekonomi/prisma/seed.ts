// Optional demo seed. Creates a caregiver, an invited CareLink and default
// rules so you can jump straight to the invite flow. Run: npm run db:seed
// (Requires a reachable DATABASE_URL.)

import { PrismaClient } from "@prisma/client";
import crypto from "node:crypto";

const prisma = new PrismaClient();

async function main() {
  const caregiver = await prisma.user.upsert({
    where: { personalIdRef: "seed-caregiver" },
    update: {},
    create: {
      personalIdRef: "seed-caregiver",
      name: "Demo Vårdgivare",
      email: "demo@example.com",
    },
  });

  const inviteToken = crypto.randomBytes(24).toString("base64url");
  const link = await prisma.careLink.create({
    data: {
      caregiverId: caregiver.id,
      seniorName: "Anna Andersson",
      inviteToken,
    },
  });

  const types = [
    "LARGE_WITHDRAWAL",
    "NEW_PAYEE",
    "MISSED_RECURRING",
    "BALANCE_DROP",
    "FREQUENCY_SPIKE",
    "DUPLICATE_CHARGE",
  ];
  for (const type of types) {
    await prisma.alertRule.create({
      data: { careLinkId: link.id, type, enabled: true, config: {} },
    });
  }

  console.log("Seeded. Invite link:");
  console.log(`  ${process.env.APP_URL ?? "http://localhost:3000"}/invite/${inviteToken}`);
}

main()
  .then(() => prisma.$disconnect())
  .catch(async (e) => {
    console.error(e);
    await prisma.$disconnect();
    process.exit(1);
  });
