import Link from "next/link";
import { redirect } from "next/navigation";
import { getCurrentUser } from "@/lib/session";
import { caregiverLinks } from "@/lib/authz";
import { prisma } from "@/lib/db";

export const dynamic = "force-dynamic";

export default async function Dashboard() {
  const user = await getCurrentUser();
  if (!user) redirect("/api/auth/bankid/start?intent=login");

  const links = await caregiverLinks(user.id);
  const newCounts = await prisma.alert.groupBy({
    by: ["careLinkId"],
    where: { careLinkId: { in: links.map((l) => l.id) }, status: "new" },
    _count: { _all: true },
  });
  const countFor = (id: string) =>
    newCounts.find((c) => c.careLinkId === id)?._count._all ?? 0;

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Översikt</h1>
        <Link href="/dashboard/new" className="btn-primary">
          Bjud in närstående
        </Link>
      </div>

      {links.length === 0 ? (
        <div className="card mt-8 text-center">
          <p className="text-neutral-600">
            Du följer ingen ännu. Bjud in din förälder för att komma igång.
          </p>
          <Link href="/dashboard/new" className="btn-primary mt-4 inline-block">
            Bjud in närstående
          </Link>
        </div>
      ) : (
        <ul className="mt-8 grid gap-4">
          {links.map((l) => (
            <li key={l.id}>
              <Link
                href={`/dashboard/${l.id}`}
                className="card flex items-center justify-between transition hover:border-brand"
              >
                <div>
                  <p className="font-semibold">{l.seniorName ?? "Närstående"}</p>
                  <p className="text-sm text-neutral-500">
                    {l.status === "invited" && "Väntar på godkännande"}
                    {l.status === "active" && "Aktiv"}
                    {l.status === "revoked" && "Återkallad"}
                  </p>
                </div>
                {countFor(l.id) > 0 && (
                  <span className="rounded-full bg-red-100 px-3 py-1 text-sm font-medium text-red-700">
                    {countFor(l.id)} nya
                  </span>
                )}
              </Link>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
