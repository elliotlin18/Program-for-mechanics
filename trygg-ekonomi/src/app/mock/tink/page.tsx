// Local stand-in for the hosted Tink Link account-connection flow.
export const dynamic = "force-dynamic";

export default function MockTink({ searchParams }: { searchParams: { state?: string } }) {
  const state = searchParams.state ?? "";
  return (
    <main className="mx-auto max-w-md px-6 py-16">
      <div className="card">
        <p className="text-xs font-semibold uppercase tracking-wide text-amber-600">
          Testläge — låtsas-bank
        </p>
        <h1 className="mt-1 text-xl font-bold">Anslut din bank</h1>
        <p className="mt-2 text-sm text-neutral-600">
          I skarp drift väljer du bank och loggar in med BankID via Tink. Här ansluter
          du ett testkonto med exempeldata.
        </p>
        <a
          href={`/api/tink/callback?state=${encodeURIComponent(state)}&code=mock`}
          className="btn-primary mt-4 inline-block"
        >
          Anslut testbank
        </a>
      </div>
    </main>
  );
}
