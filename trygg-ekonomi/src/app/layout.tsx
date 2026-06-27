import "./globals.css";
import type { Metadata } from "next";
import Link from "next/link";
import { getCurrentUser } from "@/lib/session";

export const metadata: Metadata = {
  title: "Trygg Ekonomi",
  description: "Håll koll på en närståendes ekonomi — utan att ta över.",
};

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const user = await getCurrentUser();
  return (
    <html lang="sv">
      <body>
        <header className="border-b border-neutral-200 bg-white">
          <div className="mx-auto flex max-w-4xl items-center justify-between px-6 py-4">
            <Link href="/" className="text-lg font-bold text-brand">
              Trygg Ekonomi
            </Link>
            <nav className="flex items-center gap-4 text-sm">
              {user ? (
                <>
                  <Link href="/dashboard" className="hover:underline">
                    Översikt
                  </Link>
                  <Link href="/senior" className="hover:underline">
                    Mina delningar
                  </Link>
                  <span className="text-neutral-500">{user.name ?? "Inloggad"}</span>
                  <form action="/api/auth/logout" method="post">
                    <button className="text-neutral-500 hover:underline">Logga ut</button>
                  </form>
                </>
              ) : (
                <Link href="/api/auth/bankid/start" className="hover:underline">
                  Logga in
                </Link>
              )}
            </nav>
          </div>
        </header>
        {children}
        <footer className="mx-auto max-w-4xl px-6 py-10 text-xs text-neutral-400">
          Endast läsbehörighet · Samtyckesbaserat · GDPR. Inga pengar kan flyttas via
          tjänsten.
        </footer>
      </body>
    </html>
  );
}
