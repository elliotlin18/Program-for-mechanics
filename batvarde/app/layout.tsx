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
          <div className="mx-auto flex max-w-5xl items-baseline gap-3 px-6 py-4">
            <Link href="/" className="text-lg font-semibold">
              Båtvärde
            </Link>
            <span className="text-sm text-muted-foreground">Vad går båten för?</span>
            <Link href="/vardera" className="ml-auto text-sm underline">
              Värdera min båt
            </Link>
          </div>
        </header>
        <main className="mx-auto max-w-5xl px-6 py-8">{children}</main>
      </body>
    </html>
  )
}
