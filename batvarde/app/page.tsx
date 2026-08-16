import Link from 'next/link'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { prisma } from '@/lib/db'
import { WINDOW_DAYS, priceTrendPct, weeklyMedians, type Confidence } from '@/lib/stats'
import { formatPercent, formatSek, slugify } from '@/lib/utils'

export const dynamic = 'force-dynamic'

const HOT_MODELS = 3

const confidenceVariant: Record<Confidence, 'high' | 'medium' | 'low'> = {
  hög: 'high',
  medel: 'medium',
  låg: 'low',
}

function modelHref(brand: string, model: string) {
  return `/bat/${slugify(brand)}/${slugify(model)}`
}

export default async function Home({
  searchParams,
}: {
  searchParams: Record<string, string | undefined>
}) {
  const query = searchParams.q?.trim() ?? ''

  const models = await prisma.boatsModel.findMany({
    orderBy: [{ brand: 'asc' }, { model: 'asc' }],
    include: {
      _count: { select: { listings: true } },
      stats: { where: { windowDays: WINDOW_DAYS } },
    },
  })

  const matching = query
    ? models.filter((model) =>
        `${model.brand} ${model.model}`.toLowerCase().includes(query.toLowerCase()),
      )
    : models

  // "Heta modeller" = de med flest observationer i fönstret, alltså där vi har
  // mest att säga.
  const hot = [...models]
    .filter((model) => model.stats[0])
    .sort(
      (a, b) =>
        b.stats[0].nActive + b.stats[0].nRemoved - (a.stats[0].nActive + a.stats[0].nRemoved),
    )
    .slice(0, HOT_MODELS)

  const now = new Date()
  const trends = new Map<number, number | null>()
  for (const model of hot) {
    const listings = await prisma.listing.findMany({
      where: { modelId: model.id },
      select: { price: true, firstSeen: true, lastSeen: true, removedAt: true, priceHistory: true },
    })
    trends.set(model.id, priceTrendPct(weeklyMedians(listings, now)))
  }

  return (
    <div className="flex flex-col gap-10">
      <section className="flex flex-col gap-5">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Vad går båten för?</h1>
          <p className="mt-2 max-w-2xl text-muted-foreground">
            Vi följer båtannonser över tid och ser vilka som försvinner från marknaden. Det ger
            något ingen annan visar: inte bara vad säljare begär, utan vad båtar faktiskt går
            för.
          </p>
        </div>

        <form className="flex flex-col gap-3 sm:flex-row">
          <input
            name="q"
            type="search"
            placeholder="Sök modell, t.ex. Yamarin 79 DC"
            defaultValue={query}
            className="w-full rounded-md border border-border bg-card px-4 py-2.5 text-sm sm:max-w-md"
          />
          <button
            type="submit"
            className="rounded-md border border-border bg-card px-4 py-2.5 text-sm font-medium"
          >
            Sök
          </button>
        </form>

        <div className="flex flex-wrap gap-3">
          <Link
            href="/vardera"
            className="rounded-md bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground"
          >
            Värdera min båt
          </Link>
          <Link
            href="/kolla"
            className="rounded-md border border-border bg-card px-5 py-2.5 text-sm font-medium"
          >
            Kolla en annons
          </Link>
        </div>
      </section>

      {!query && hot.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold">Heta modeller</h2>
          <div className="mt-3 grid gap-3 sm:grid-cols-3">
            {hot.map((model) => {
              const stats = model.stats[0]
              const trend = trends.get(model.id) ?? null
              return (
                <Link key={model.id} href={modelHref(model.brand, model.model)}>
                  <Card className="h-full transition hover:border-primary">
                    <CardContent className="pt-5">
                      <div className="font-medium">
                        {model.brand} {model.model}
                      </div>
                      <div className="mt-2 text-2xl font-semibold">
                        {formatSek(stats.median)}
                      </div>
                      <div className="mt-1 text-xs text-muted-foreground">
                        median aktiva annonser
                      </div>

                      <div className="mt-3 flex items-center gap-2 text-sm">
                        {/* Neutral färg med avsikt: grönt och rött betyder "rimligt pris"
                            i annonskollen och ska inte betyda något annat här. */}
                        <span className="text-muted-foreground">
                          {trend === null
                            ? 'Trend: för lite data'
                            : `${trend > 0 ? '↑' : trend < 0 ? '↓' : '→'} ${formatPercent(trend)} på ${WINDOW_DAYS} dagar`}
                        </span>
                      </div>

                      <div className="mt-3">
                        <Badge variant={confidenceVariant[stats.confidence as Confidence]}>
                          {stats.nActive + stats.nRemoved} observationer
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>
                </Link>
              )
            })}
          </div>
        </section>
      )}

      <section>
        <h2 className="text-lg font-semibold">
          {query ? `Modeller som matchar "${query}"` : 'Alla modeller'}
        </h2>

        {matching.length === 0 ? (
          <p className="mt-3 text-sm text-muted-foreground">
            Ingen modell matchar. Prototypen följer tio modeller –{' '}
            <Link className="underline" href="/">
              visa alla
            </Link>
            .
          </p>
        ) : (
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            {matching.map((model) => (
              <Link
                key={model.id}
                href={modelHref(model.brand, model.model)}
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
      </section>
    </div>
  )
}
