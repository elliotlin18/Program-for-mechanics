import Link from "next/link";
import { getCurrentUser } from "@/lib/session";

export default async function Home() {
  const user = await getCurrentUser();
  return (
    <main className="mx-auto max-w-2xl px-6 py-20">
      <h1 className="text-4xl font-bold tracking-tight">Trygg Ekonomi</h1>
      <p className="mt-4 text-lg text-neutral-600">
        Se till att mamma och pappa har koll — utan att ta över. Diskret tidig
        varning vid ovanliga händelser, med full insyn och kontroll för din förälder.
      </p>

      <div className="mt-8 flex gap-3">
        {user ? (
          <Link href="/dashboard" className="btn-primary">
            Till översikten
          </Link>
        ) : (
          <Link href="/api/auth/bankid/start?intent=login" className="btn-primary">
            Kom igång med BankID
          </Link>
        )}
        <a href="#hur" className="btn-secondary">
          Så funkar det
        </a>
      </div>

      <section id="hur" className="mt-16 grid gap-4">
        <Step
          n={1}
          title="Bjud in din förälder"
          body="Du skapar en länk och bjuder in din förälder via SMS eller e-post."
        />
        <Step
          n={2}
          title="Föräldern godkänner med BankID"
          body="Din förälder ser exakt vad som delas och godkänner med ett swipe. Bara läsbehörighet — ingen kan flytta pengar."
        />
        <Step
          n={3}
          title="Du får diskreta varningar"
          body="Ovanligt stora uttag, nya mottagare, missade räkningar eller snabbt fallande saldo — du blir notifierad tidigt."
        />
      </section>
    </main>
  );
}

function Step({ n, title, body }: { n: number; title: string; body: string }) {
  return (
    <div className="card flex gap-4">
      <div className="flex h-9 w-9 flex-none items-center justify-center rounded-full bg-brand font-semibold text-white">
        {n}
      </div>
      <div>
        <h2 className="font-semibold">{title}</h2>
        <p className="mt-1 text-neutral-600">{body}</p>
      </div>
    </div>
  );
}
