import Link from "next/link";
import { getCurrentUser } from "@/lib/session";

export const dynamic = "force-dynamic";

export default async function Home() {
  const user = await getCurrentUser();
  const primaryHref = user ? "/dashboard" : "/api/auth/bankid/start?intent=login";
  const primaryLabel = user ? "Till översikten" : "Kom igång med BankID";

  return (
    <main>
      {/* ── Hero (continues the forest band from the header) ─────────────── */}
      <section className="bg-forest text-[#EFEDE4]">
        <div className="mx-auto max-w-[1140px] px-6 pb-36 pt-16 md:px-10">
          <p className="mb-6 flex items-center gap-3.5 font-mono text-xs uppercase tracking-[0.2em] text-[#84A096]">
            <span className="h-px w-14 bg-gold-2/70" /> 01 — Trygghet på avstånd
          </p>
          <h1 className="max-w-[16ch] font-serif text-5xl font-semibold leading-[1.02] tracking-[-0.015em] md:text-[74px]">
            Att vaka över en förälders ekonomi, <em className="not-italic text-gold-2 [font-style:italic]">utan</em> att ta över den.
          </h1>
          <p className="mt-7 max-w-[46ch] text-xl leading-relaxed text-[#C7D6CE]">
            Banknivå-säker insyn med förälderns samtycke. Du ser tidiga tecken på att något
            inte stämmer. Hon behåller varje uns av kontrollen.
          </p>
          <div className="mt-9 flex flex-wrap items-center gap-4">
            <Link href={primaryHref} className="btn-gold">
              {primaryLabel}
            </Link>
            <a href="#sa-funkar" className="btn-line-dark">
              Så funkar det
            </a>
          </div>
          <p className="mt-4 font-mono text-xs uppercase tracking-wide text-[#9FB6AC]">
            BankID · Endast läsbehörighet · Inga pengar kan flyttas
          </p>
        </div>
      </section>

      {/* ── Ledger panel, overlapping the band edge ──────────────────────── */}
      <div className="mx-auto max-w-[1140px] px-6 md:px-10">
        <div className="relative mx-auto -mt-28 max-w-[560px] overflow-hidden rounded-lg border border-hair bg-paper-2 shadow-[0_30px_60px_-40px_rgba(0,0,0,0.5)]">
          <div className="flex items-baseline justify-between border-b border-hair px-6 py-4">
            <span className="font-mono text-[11.5px] uppercase tracking-[0.2em] text-muted">
              Kontoutdrag — Lönekonto •••• 4821
            </span>
            <span className="font-serif text-2xl font-semibold">48 250 kr</span>
          </div>
          <LedgerRow date="25 jun" desc="Pensionsmyndigheten" amount="+21 300,00" />
          <LedgerRow date="20 jun" desc="Okänd mottagare" amount="−15 000,00" flag />
          <LedgerRow date="12 jun" desc="Vattenfall — el" amount="−1 145,00" />
          <LedgerRow date="01 jun" desc="Hyresvärden AB" amount="−8 500,00" last />
        </div>
      </div>

      {/* ── Trust line ───────────────────────────────────────────────────── */}
      <div className="mt-[74px] border-y border-hair">
        <div className="mx-auto grid max-w-[1140px] grid-cols-2 md:grid-cols-4">
          <TrustItem label="Behörighet" value="Endast läsa" />
          <TrustItem label="Identitet" value="BankID" />
          <TrustItem label="Infrastruktur" value="Tinks AISP-licens" />
          <TrustItem label="Skydd" value="GDPR · krypterat" last />
        </div>
      </div>

      {/* ── What it gives (numbered manifesto) ───────────────────────────── */}
      <section id="sa-funkar" className="mx-auto max-w-[1140px] px-6 py-24 md:px-10">
        <p className="kicker">Vad det ger</p>
        <h2 className="mt-2.5 font-serif text-4xl font-semibold tracking-[-0.01em]">
          Inte ett bedrägerilarm. En trygghet.
        </h2>
        <div className="mt-8">
          <Entry
            no="01"
            title="Lugn för dig"
            body="Diskret tidig varning vid det som brukar vara de första tecknen — ett ovanligt stort uttag, en ny mottagare, en räkning som uteblev, ett saldo som faller fort. Annars hör vi inte av oss."
          />
          <Entry
            no="02"
            title="Värdighet för föräldern"
            body="Hon ger sitt samtycke med BankID, ser exakt vad som delas och kan dra tillbaka det på en sekund. Det här är ett samarbete mellan er — inte en kamera riktad mot henne."
          />
          <Entry
            no="03"
            title="Byggt på licens, inte löften"
            body="Vi rör aldrig pengarna. Insynen sker genom Tinks reglerade kontoinformationslicens (PSD2) — samma infrastruktur som bankerna själva bygger på."
            last
          />
        </div>
      </section>

      {/* ── Stats (forest band) ──────────────────────────────────────────── */}
      <div className="bg-forest text-[#EFEDE4]">
        <div className="mx-auto grid max-w-[1140px] grid-cols-1 sm:grid-cols-3">
          <Stat n="232 862" t="anmälda bedrägeribrott i Sverige under 2025." src="Källa — Brå" />
          <Stat n="+112 %" t="fler kvinnliga målsägare 65+ sedan 2019." src="Källa — Brå" />
          <Stat n="180 dgr" t="gäller samtycket enligt PSD2, och förnyas av föräldern." src="EU PSD2" last />
        </div>
      </div>

      {/* ── Dignity ──────────────────────────────────────────────────────── */}
      <section className="mx-auto grid max-w-[1140px] items-start gap-14 px-6 py-24 md:grid-cols-2 md:px-10">
        <div>
          <p className="kicker">Förälderns kontroll</p>
          <h2 className="mt-3 font-serif text-[42px] font-semibold leading-[1.08] tracking-[-0.01em]">
            Din förälder bestämmer allt.
          </h2>
          <p className="mt-4 text-lg text-muted">
            Integritet är inte en eftertanke här — det är själva grunden tjänsten är byggd på.
          </p>
          <ul className="mt-6 border-t border-hair">
            <DignityItem n="i." text="Ser i realtid vad som delas, och med vem." />
            <DignityItem n="ii." text="Kan dra tillbaka åtkomsten direkt, utan att fråga någon." />
            <DignityItem n="iii." text="Ingen kan flytta pengar — det går bara att läsa." />
          </ul>
        </div>
        <figure className="font-serif text-[27px] italic leading-[1.42]">
          <div className="mb-6 h-0.5 w-12 bg-gold" />
          <blockquote>
            ”Jag ville bara veta att allt var okej med mamma. Nu slipper jag ligga vaken — och hon
            känner sig respekterad, inte övervakad.”
          </blockquote>
          <figcaption className="mt-5 font-mono text-xs uppercase not-italic tracking-[0.12em] text-muted">
            Karin, 54 — dotter och användare
          </figcaption>
        </figure>
      </section>
    </main>
  );
}

function LedgerRow({
  date,
  desc,
  amount,
  flag,
  last,
}: {
  date: string;
  desc: string;
  amount: string;
  flag?: boolean;
  last?: boolean;
}) {
  return (
    <div
      className={`grid grid-cols-[84px_1fr_auto] items-center gap-3.5 px-6 py-3.5 ${
        last ? "" : "border-b border-hair"
      } ${flag ? "-ml-px border-l-[3px] border-l-gold bg-[#FBF4E4]" : ""}`}
    >
      <span className="font-mono text-[13px] text-muted">{date}</span>
      <span className="text-base">
        {desc}
        {flag && (
          <span className="ml-2.5 rounded-[3px] border border-gold px-1.5 py-px font-mono text-[10px] uppercase tracking-[0.14em] text-gold">
            Avvikelse
          </span>
        )}
      </span>
      <span className="text-right font-mono text-[15px] tabular-nums">{amount}</span>
    </div>
  );
}

function TrustItem({ label, value, last }: { label: string; value: string; last?: boolean }) {
  return (
    <div className={`px-6 py-5 ${last ? "" : "border-r border-hair"}`}>
      <span className="font-mono text-xs uppercase tracking-[0.1em] text-muted">{label}</span>
      <b className="mt-1.5 block font-serif text-[17px] font-semibold">{value}</b>
    </div>
  );
}

function Entry({
  no,
  title,
  body,
  last,
}: {
  no: string;
  title: string;
  body: string;
  last?: boolean;
}) {
  return (
    <div
      className={`grid grid-cols-[88px_1fr] items-start gap-6 border-t border-hair py-9 ${
        last ? "border-b" : ""
      }`}
    >
      <div className="pt-2 font-mono text-sm tracking-[0.1em] text-gold">{no}</div>
      <div>
        <h3 className="mb-2 font-serif text-[27px] font-semibold">{title}</h3>
        <p className="max-w-[60ch] text-[18.5px] text-muted">{body}</p>
      </div>
    </div>
  );
}

function Stat({ n, t, src, last }: { n: string; t: string; src: string; last?: boolean }) {
  return (
    <div className={`px-8 py-14 ${last ? "" : "border-b border-white/15 sm:border-b-0 sm:border-r"}`}>
      <div className="font-serif text-[52px] font-semibold leading-none text-white">{n}</div>
      <div className="mt-3.5 max-w-[26ch] text-[15.5px] text-[#A9C0B6]">{t}</div>
      <div className="mt-2.5 font-mono text-[10.5px] uppercase tracking-[0.12em] text-[#6E897E]">
        {src}
      </div>
    </div>
  );
}

function DignityItem({ n, text }: { n: string; text: string }) {
  return (
    <li className="flex items-baseline gap-3.5 border-b border-hair py-[18px] text-lg">
      <span className="font-mono text-[13px] text-gold">{n}</span>
      {text}
    </li>
  );
}
