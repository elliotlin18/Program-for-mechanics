import { redirect } from "next/navigation";
import { getCurrentUser } from "@/lib/session";
import { prisma } from "@/lib/db";
import { date } from "@/lib/format";
import { revokeLink } from "@/app/actions";

export const dynamic = "force-dynamic";

export default async function SeniorPanel({
  searchParams,
}: {
  searchParams: { connected?: string; error?: string };
}) {
  const user = await getCurrentUser();
  if (!user) redirect("/api/auth/bankid/start?intent=login");

  const links = await prisma.careLink.findMany({
    where: { seniorId: user.id },
    include: {
      caregiver: true,
      accounts: { select: { id: true, name: true } },
      consents: { where: { active: true }, orderBy: { grantedAt: "desc" }, take: 1 },
    },
    orderBy: { createdAt: "desc" },
  });

  return (
    <main className="mx-auto max-w-xl px-6 py-12">
      <h1 className="text-senior-lg font-bold">Mina delningar</h1>
      <p className="mt-2 text-senior text-neutral-700">
        Här ser du exakt vad du delar och med vem. Du kan dra tillbaka åtkomsten när
        som helst.
      </p>

      {searchParams.connected && (
        <div className="card mt-4 border-green-200 bg-green-50 text-senior">
          Klart! Din ekonomi delas nu säkert med din familj.
        </div>
      )}
      {searchParams.error && (
        <div className="card mt-4 border-amber-200 bg-amber-50 text-senior">
          Något gick inte som väntat. Försök igen eller be om en ny inbjudan.
        </div>
      )}

      {links.length === 0 ? (
        <p className="card mt-6 text-senior text-neutral-600">
          Du delar ingenting just nu.
        </p>
      ) : (
        <ul className="mt-6 grid gap-4">
          {links.map((l) => (
            <li key={l.id} className="card">
              <p className="text-senior font-semibold">
                Delas med {l.caregiver.name ?? "din familj"}
              </p>
              <p className="mt-1 text-neutral-600">
                Status:{" "}
                {l.status === "active"
                  ? "Aktiv"
                  : l.status === "invited"
                    ? "Inte godkänd än"
                    : "Återkallad"}
              </p>
              <ul className="mt-2 list-disc pl-6 text-neutral-700">
                <li>Endast läsbehörighet — ingen kan flytta dina pengar.</li>
                <li>
                  Konton som delas:{" "}
                  {l.accounts.length > 0
                    ? l.accounts.map((a) => a.name).join(", ")
                    : "inga än"}
                </li>
                {l.consents[0] && (
                  <li>Samtycket gäller till {date(l.consents[0].expiresAt)}.</li>
                )}
              </ul>

              {l.status !== "revoked" && (
                <form action={revokeLink} className="mt-4">
                  <input type="hidden" name="careLinkId" value={l.id} />
                  <label className="flex items-center gap-2 text-neutral-700">
                    <input type="checkbox" name="purge" className="h-5 w-5" />
                    Radera även all sparad information om mig
                  </label>
                  <button className="mt-3 rounded-lg border border-red-300 bg-white px-5 py-3 text-senior text-red-700 hover:bg-red-50">
                    Dra tillbaka åtkomsten
                  </button>
                </form>
              )}
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
