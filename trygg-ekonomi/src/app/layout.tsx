import "./globals.css";
import type { Metadata } from "next";
import Link from "next/link";
import { getCurrentUser } from "@/lib/session";

export const metadata: Metadata = {
  title: "Trygg Ekonomi",
  description: "Håll koll på en närståendes ekonomi — utan att ta över.",
};

function Mark() {
  return (
    <span className="grid h-7 w-7 place-items-center border border-gold-2/80">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
        <path
          d="M12 3l7 3v5c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6l7-3z"
          stroke="#C8A24A"
          strokeWidth="2"
        />
      </svg>
    </span>
  );
}

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const user = await getCurrentUser();
  return (
    <html lang="sv">
      <body>
        {/* Utility bar — forest band, serif wordmark, mono meta. */}
        <header className="bg-forest text-[#CFE0D8]">
          <div className="mx-auto flex h-[58px] max-w-[1140px] items-center justify-between px-6 md:px-10">
            <Link href="/" className="flex items-center gap-3 font-serif text-xl font-bold text-white">
              <Mark />
              Trygg Ekonomi
            </Link>
            <span className="hidden font-mono text-[11.5px] uppercase tracking-[0.18em] text-[#9FB6AC] md:block">
              Samtyckesbaserad ekonomisk tillsyn
            </span>
            <nav className="flex items-center gap-5 text-sm">
              {user ? (
                <>
                  <Link href="/dashboard" className="text-[#CFE0D8] hover:text-white">
                    Översikt
                  </Link>
                  <Link href="/senior" className="hidden text-[#CFE0D8] hover:text-white sm:block">
                    Mina delningar
                  </Link>
                  <form action="/api/auth/logout" method="post">
                    <button className="border-b border-transparent text-[#9FB6AC] hover:border-gold-2 hover:text-white">
                      Logga ut
                    </button>
                  </form>
                </>
              ) : (
                <Link
                  href="/api/auth/bankid/start?intent=login"
                  className="border-b border-transparent text-[#CFE0D8] hover:border-gold-2"
                >
                  Logga in&nbsp;→
                </Link>
              )}
            </nav>
          </div>
        </header>

        {children}

        <footer className="border-t border-hair">
          <div className="mx-auto flex max-w-[1140px] flex-wrap justify-between gap-3 px-6 py-8 font-mono text-xs uppercase tracking-[0.08em] text-muted md:px-10">
            <span>Trygg Ekonomi</span>
            <span>Endast läsbehörighet · Samtyckesbaserat · GDPR</span>
            <span>Inga pengar kan flyttas via tjänsten</span>
          </div>
        </footer>
      </body>
    </html>
  );
}
