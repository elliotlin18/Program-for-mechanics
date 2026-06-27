// Local stand-in for the BankID broker (PROVIDER_MODE=mock). Lets you "sign in"
// as any test person so the whole flow is demoable without broker credentials.
export const dynamic = "force-dynamic";

export default function MockBankId({ searchParams }: { searchParams: { state?: string } }) {
  return (
    <main className="mx-auto max-w-md px-6 py-16">
      <div className="card">
        <p className="text-xs font-semibold uppercase tracking-wide text-amber-600">
          Testläge — låtsas-BankID
        </p>
        <h1 className="mt-1 text-xl font-bold">Identifiera dig</h1>
        <p className="mt-2 text-sm text-neutral-600">
          I skarp drift sker detta i BankID-appen. Här skriver du bara ett namn.
        </p>

        <form action="/api/auth/bankid/callback" method="get" className="mt-4 grid gap-3">
          <input type="hidden" name="state" value={searchParams.state ?? ""} />
          <input type="hidden" name="code" value="mock" />
          <input
            name="mock_name"
            required
            placeholder="Ditt namn, t.ex. Anna Andersson"
            className="rounded-lg border border-neutral-300 px-3 py-2"
          />
          <button className="btn-primary">Signera</button>
        </form>
      </div>
    </main>
  );
}
