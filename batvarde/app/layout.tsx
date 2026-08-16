import type { Metadata } from 'next'
import Link from 'next/link'
import './globals.css'

export const metadata: Metadata = {
  title: 'Båtvärde',
  description: 'Vad går båten för? Sålda priser, inte listpriser.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="sv">
      <body className="min-h-screen antialiased">
        <header className="border-b border-border bg-card">
          <div className="mx-auto flex max-w-5xl flex-wrap items-baseline gap-x-5 gap-y-2 px-6 py-4">
            <Link href="/" className="text-lg font-semibold">
              Båtvärde
            </Link>
            <span className="text-sm text-muted-foreground">Vad går båten för?</span>
            <nav className="ml-auto flex gap-5 text-sm">
              <Link href="/vardera" className="hover:underline">
                Värdera min båt
              </Link>
              <Link href="/kolla" className="hover:underline">
                Kolla en annons
              </Link>
            </nav>
          </div>
        </header>

        <main className="mx-auto max-w-5xl px-6 py-8">{children}</main>

        <footer className="mx-auto max-w-5xl px-6 pb-10 text-sm text-muted-foreground">
          <Link href="/sa-raknar-vi" className="underline">
            Så räknar vi
          </Link>
          <span className="ml-3">Prototyp – litet underlag, tio modeller.</span>
        </footer>
      </body>
    </html>
  )
}
