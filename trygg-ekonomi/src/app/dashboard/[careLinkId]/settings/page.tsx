import Link from "next/link";
import { redirect, notFound } from "next/navigation";
import { getCurrentUser } from "@/lib/session";
import { authorizeCareLink, ForbiddenError } from "@/lib/authz";
import { loadRuleConfig } from "@/lib/rules";
import { toggleRule, saveThresholds, revokeLink } from "@/app/actions";
import { ALERT_TYPES } from "@/types";
import { alertTypeLabel } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function Settings({ params }: { params: { careLinkId: string } }) {
  const user = await getCurrentUser();
  if (!user) redirect("/api/auth/bankid/start?intent=login");

  try {
    await authorizeCareLink(user.id, params.careLinkId, ["caregiver"]);
  } catch (e) {
    if (e instanceof ForbiddenError) notFound();
    throw e;
  }

  const cfg = await loadRuleConfig(params.careLinkId);

  return (
    <main className="mx-auto max-w-2xl px-6 py-12">
      <Link href={`/dashboard/${params.careLinkId}`} className="text-sm text-neutral-500 hover:underline">
        ← Tillbaka
      </Link>
      <h1 className="mt-1 text-2xl font-bold">Inställningar</h1>

      <section className="mt-6">
        <h2 className="font-semibold">Varningar</h2>
        <ul className="mt-2 grid gap-2">
          {ALERT_TYPES.map((type) => (
            <li key={type} className="card flex items-center justify-between py-3">
              <span>{alertTypeLabel[type] ?? type}</span>
              <form action={toggleRule} className="flex items-center gap-2">
                <input type="hidden" name="careLinkId" value={params.careLinkId} />
                <input type="hidden" name="type" value={type} />
                <input
                  type="checkbox"
                  name="enabled"
                  defaultChecked={cfg.enabled[type]}
                  className="h-5 w-5"
                />
                <button className="text-sm text-brand hover:underline">Spara</button>
              </form>
            </li>
          ))}
        </ul>
      </section>

      <section className="mt-8">
        <h2 className="font-semibold">Trösklar</h2>
        <form action={saveThresholds} className="card mt-2 grid gap-4">
          <input type="hidden" name="careLinkId" value={params.careLinkId} />
          <label className="grid gap-1">
            <span className="text-sm font-medium">Gräns för stort uttag (SEK)</span>
            <input
              name="largeWithdrawalAbsolute"
              type="number"
              min={0}
              defaultValue={cfg.largeWithdrawalAbsolute}
              className="rounded-lg border border-neutral-300 px-3 py-2"
            />
          </label>
          <label className="grid gap-1">
            <span className="text-sm font-medium">Larma vid saldofall (andel, t.ex. 0.4 = 40%)</span>
            <input
              name="balanceDropPct"
              type="number"
              step="0.05"
              min={0.05}
              max={0.95}
              defaultValue={cfg.balanceDropPct}
              className="rounded-lg border border-neutral-300 px-3 py-2"
            />
          </label>
          <button className="btn-primary justify-self-start">Spara trösklar</button>
        </form>
      </section>

      <section className="mt-8">
        <h2 className="font-semibold text-red-700">Avsluta</h2>
        <form action={revokeLink} className="card mt-2 border-red-200">
          <input type="hidden" name="careLinkId" value={params.careLinkId} />
          <p className="text-sm text-neutral-600">
            Återkalla länken. Datainsamlingen stoppas omedelbart.
          </p>
          <label className="mt-3 flex items-center gap-2 text-sm">
            <input type="checkbox" name="purge" className="h-4 w-4" />
            Radera även all sparad transaktionsdata (GDPR)
          </label>
          <button className="mt-4 rounded-lg border border-red-300 bg-white px-4 py-2 text-red-700 hover:bg-red-50">
            Återkalla länken
          </button>
        </form>
      </section>
    </main>
  );
}
