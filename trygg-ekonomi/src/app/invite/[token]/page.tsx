// Senior-facing accept screen. Large, calm, single-purpose (one approval).
import { prisma } from "@/lib/db";

export const dynamic = "force-dynamic";

export default async function Invite({ params }: { params: { token: string } }) {
  const link = await prisma.careLink.findUnique({
    where: { inviteToken: params.token },
    include: { caregiver: true },
  });

  if (!link || link.status === "revoked") {
    return (
      <main className="mx-auto max-w-xl px-6 py-16">
        <h1 className="text-senior-lg font-bold">Inbjudan saknas</h1>
        <p className="mt-4 text-senior text-neutral-700">
          Den här inbjudan är inte längre giltig. Be din familj skapa en ny.
        </p>
      </main>
    );
  }

  if (link.status === "active") {
    return (
      <main className="mx-auto max-w-xl px-6 py-16">
        <h1 className="text-senior-lg font-bold">Redan godkänd</h1>
        <p className="mt-4 text-senior text-neutral-700">
          Du har redan godkänt den här delningen.{" "}
          <a href="/senior" className="text-brand underline">
            Se vad du delar
          </a>
          .
        </p>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-xl px-6 py-16">
      <h1 className="text-senior-lg font-bold">
        {link.caregiver.name ?? "Din familj"} vill hjälpa dig hålla koll
      </h1>
      <p className="mt-4 text-senior text-neutral-700">
        Du bestämmer vad som delas och kan när som helst ångra dig.
      </p>
      <ul className="mt-6 list-disc pl-6 text-senior text-neutral-700">
        <li>Endast läsbehörighet — ingen kan flytta dina pengar.</li>
        <li>Du ser exakt vad som delas.</li>
        <li>Du kan dra tillbaka åtkomsten direkt.</li>
      </ul>

      <a
        href={`/api/auth/bankid/start?intent=accept&token=${encodeURIComponent(params.token)}`}
        className="mt-8 inline-block rounded-lg bg-brand px-6 py-4 text-senior font-medium text-white hover:bg-brand-dark"
      >
        Godkänn med BankID
      </a>
    </main>
  );
}
