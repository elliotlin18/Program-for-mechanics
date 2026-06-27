import { redirect } from "next/navigation";
import { getCurrentUser } from "@/lib/session";
import { createCareLink } from "@/app/actions";

export const dynamic = "force-dynamic";

export default async function NewCareLink() {
  const user = await getCurrentUser();
  if (!user) redirect("/api/auth/bankid/start?intent=login");

  return (
    <main className="mx-auto max-w-xl px-6 py-12">
      <h1 className="text-2xl font-bold">Bjud in en närstående</h1>
      <p className="mt-2 text-neutral-600">
        Ange din förälders uppgifter. Vi skapar en inbjudan som du kan dela. Din
        förälder godkänner själv med BankID och bestämmer vad som delas.
      </p>

      <form action={createCareLink} className="card mt-6 grid gap-4">
        <label className="grid gap-1">
          <span className="text-sm font-medium">Namn</span>
          <input
            name="seniorName"
            required
            maxLength={120}
            className="rounded-lg border border-neutral-300 px-3 py-2"
            placeholder="t.ex. Anna Andersson"
          />
        </label>
        <label className="grid gap-1">
          <span className="text-sm font-medium">E-post (valfritt)</span>
          <input
            name="seniorEmail"
            type="email"
            className="rounded-lg border border-neutral-300 px-3 py-2"
            placeholder="anna@example.com"
          />
        </label>
        <label className="grid gap-1">
          <span className="text-sm font-medium">Mobilnummer (valfritt)</span>
          <input
            name="seniorPhone"
            className="rounded-lg border border-neutral-300 px-3 py-2"
            placeholder="070-123 45 67"
          />
        </label>
        <button className="btn-primary justify-self-start">Skapa inbjudan</button>
      </form>
    </main>
  );
}
