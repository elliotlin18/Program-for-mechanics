import Link from "next/link";
import { redirect, notFound } from "next/navigation";
import { getCurrentUser } from "@/lib/session";
import { authorizeCareLink, ForbiddenError } from "@/lib/authz";
import { prisma } from "@/lib/db";
import { decrypt } from "@/lib/crypto";
import { env } from "@/lib/env";
import { kr, date, dateTime, severityClasses, alertTypeLabel } from "@/lib/format";
import { triggerSync, dismissAlert, markAlertsSeen } from "@/app/actions";

export const dynamic = "force-dynamic";

export default async function CareLinkDetail({
  params,
}: {
  params: { careLinkId: string };
}) {
  const user = await getCurrentUser();
  if (!user) redirect("/api/auth/bankid/start?intent=login");

  try {
    await authorizeCareLink(user.id, params.careLinkId, ["caregiver"]);
  } catch (e) {
    if (e instanceof ForbiddenError) notFound();
    throw e;
  }

  const link = await prisma.careLink.findUnique({
    where: { id: params.careLinkId },
    include: {
      accounts: { include: { transactions: { orderBy: { bookedAt: "desc" }, take: 8 } } },
      alerts: { orderBy: { createdAt: "desc" }, take: 30 },
    },
  });
  if (!link) notFound();

  // Viewing the dashboard marks new alerts as seen (audited via markAlertsSeen).
  await markAlertsSeen(params.careLinkId);

  const inviteUrl = `${env.APP_URL}/invite/${link.inviteToken}`;
  const activeAlerts = link.alerts.filter((a) => a.status !== "dismissed");

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <div className="flex items-center justify-between">
        <div>
          <Link href="/dashboard" className="text-sm text-neutral-500 hover:underline">
            ← Översikt
          </Link>
          <h1 className="text-2xl font-bold">{link.seniorName ?? "Närstående"}</h1>
        </div>
        <div className="flex gap-2">
          <Link href={`/dashboard/${link.id}/settings`} className="btn-secondary">
            Inställningar
          </Link>
          {link.status === "active" && (
            <form action={triggerSync}>
              <input type="hidden" name="careLinkId" value={link.id} />
              <button className="btn-primary">Uppdatera</button>
            </form>
          )}
        </div>
      </div>

      {link.status === "invited" && (
        <div className="card mt-6 border-amber-200 bg-amber-50">
          <h2 className="font-semibold">Väntar på godkännande</h2>
          <p className="mt-1 text-sm text-neutral-700">
            Dela den här länken med {link.seniorName}. De godkänner med BankID och väljer
            vad som delas.
          </p>
          <code className="mt-3 block break-all rounded-lg border border-amber-200 bg-white px-3 py-2 text-sm">
            {inviteUrl}
          </code>
        </div>
      )}

      {/* Alerts */}
      <section className="mt-8">
        <h2 className="font-semibold">Aktiva varningar</h2>
        {activeAlerts.length === 0 ? (
          <p className="card mt-2 text-neutral-500">Inga varningar just nu.</p>
        ) : (
          <ul className="mt-2 grid gap-3">
            {activeAlerts.map((a) => (
              <li key={a.id} className={`card border ${severityClasses(a.severity)}`}>
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="text-xs font-medium uppercase tracking-wide">
                      {alertTypeLabel[a.type] ?? a.type}
                    </p>
                    <p className="mt-1">{a.message}</p>
                    <p className="mt-1 text-xs text-neutral-500">{dateTime(a.createdAt)}</p>
                  </div>
                  <form action={dismissAlert}>
                    <input type="hidden" name="alertId" value={a.id} />
                    <button className="text-sm text-neutral-500 hover:underline">
                      Avfärda
                    </button>
                  </form>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* Accounts */}
      <section className="mt-8">
        <h2 className="font-semibold">Konton</h2>
        {link.accounts.length === 0 ? (
          <p className="card mt-2 text-neutral-500">Inga konton anslutna än.</p>
        ) : (
          <div className="mt-2 grid gap-4">
            {link.accounts.map((acc) => (
              <div key={acc.id} className="card">
                <div className="flex items-baseline justify-between">
                  <div>
                    <p className="font-semibold">{acc.name}</p>
                    <p className="text-sm text-neutral-500">{decrypt(acc.maskedNumber)}</p>
                  </div>
                  <p className="text-xl font-bold">{kr(Number(acc.balance), acc.currency)}</p>
                </div>
                {acc.transactions.length > 0 && (
                  <ul className="mt-3 divide-y divide-neutral-100 border-t border-neutral-100">
                    {acc.transactions.map((t) => (
                      <li key={t.id} className="flex items-center justify-between py-2 text-sm">
                        <span>
                          <span className="font-medium">{decrypt(t.payee) ?? "—"}</span>
                          <span className="ml-2 text-neutral-400">{date(t.bookedAt)}</span>
                        </span>
                        <span
                          className={Number(t.amount) < 0 ? "text-neutral-900" : "text-green-700"}
                        >
                          {kr(Number(t.amount), t.currency)}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
