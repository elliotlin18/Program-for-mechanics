// Authorization: a user may only touch a CareLink they participate in.
// Every server action / API route that reads or mutates CareLink data must go
// through one of these guards.

import { prisma } from "@/lib/db";
import type { CareLink } from "@prisma/client";

export class ForbiddenError extends Error {
  constructor(message = "Forbidden") {
    super(message);
    this.name = "ForbiddenError";
  }
}

export type CareRole = "caregiver" | "senior";

export interface CareLinkAccess {
  careLink: CareLink;
  role: CareRole;
}

/**
 * Returns the CareLink and the caller's role on it, or throws ForbiddenError.
 * `allow` restricts which roles are permitted (default: either side).
 */
export async function authorizeCareLink(
  userId: string,
  careLinkId: string,
  allow: CareRole[] = ["caregiver", "senior"]
): Promise<CareLinkAccess> {
  const careLink = await prisma.careLink.findUnique({ where: { id: careLinkId } });
  if (!careLink) throw new ForbiddenError("CareLink not found");

  let role: CareRole | null = null;
  if (careLink.caregiverId === userId) role = "caregiver";
  else if (careLink.seniorId === userId) role = "senior";

  if (!role || !allow.includes(role)) {
    throw new ForbiddenError("Not a participant in this CareLink");
  }
  return { careLink, role };
}

/** List CareLinks where the user is the caregiver. */
export function caregiverLinks(userId: string) {
  return prisma.careLink.findMany({
    where: { caregiverId: userId },
    orderBy: { createdAt: "desc" },
  });
}

/** List CareLinks where the user is the senior (the shared-from side). */
export function seniorLinks(userId: string) {
  return prisma.careLink.findMany({
    where: { seniorId: userId },
    orderBy: { createdAt: "desc" },
  });
}
