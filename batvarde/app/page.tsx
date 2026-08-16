import Link from 'next/link'
import { prisma } from '@/lib/db'
import { Card, CardContent } from '@/components/ui/card'
import { slugify } from '@/lib/utils'

// Enkel modellindex. Riktig startsida byggs i steg 6.
export const dynamic = 'force-dynamic'

export default async function Home() {
  const models = await prisma.boatsModel.findMany({
    orderBy: [{ brand: 'asc' }, { model: 'asc' }],
    include: { _count: { select: { listings: true } } },
  })

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold">Modeller</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Prototypens tio startmodeller. Välj en modell för prisintervall, prishistorik och tid på
          marknaden.
        </p>
      </div>

      {models.length === 0 ? (
        <Card>
          <CardContent className="pt-5 text-sm text-muted-foreground">
            Databasen är tom. Kör <code className="font-mono">npm run db:seed</code>.
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2">
          {models.map((model) => (
            <Link
              key={model.id}
              href={`/bat/${slugify(model.brand)}/${slugify(model.model)}`}
              className="rounded-lg border border-border bg-card p-4 transition hover:border-primary"
            >
              <div className="font-medium">
                {model.brand} {model.model}
              </div>
              <div className="mt-1 text-sm text-muted-foreground">
                {model.type}
                {model.lengthM ? ` · ${model.lengthM} m` : ''} · {model._count.listings} annonser
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
