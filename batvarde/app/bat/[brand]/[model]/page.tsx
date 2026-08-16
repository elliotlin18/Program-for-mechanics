import { notFound } from 'next/navigation'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { PriceChart } from '@/components/price-chart'
import { prisma } from '@/lib/db'
import {
  CHART_WEEKS,
  WINDOW_DAYS,
  computeModelStats,
  confidenceFor,
  priceDrops,
  weeklyMedians,
  type Confidence,
} from '@/lib/stats'
import { daysBetween, formatPercent, formatSek, slugify } from '@/lib/utils'

export const dynamic = 'force-dynamic'

const confidenceVariant: Record<Confidence, 'high' | 'medium' | 'low'> = {
  hög: 'high',
  medel: 'medium',
  låg: 'low',
}

export default async function ModelPage({
  params,
}: {
  params: { brand: string; model: string }
}) {
  // Slug -> modell. Tio modeller i prototypen, så en full läsning duger gott.
  const models = await prisma.boatsModel.findMany()
  const boatsModel = models.find(
    (m) => slugify(m.brand) === params.brand && slugify(m.model) === params.model,
  )
  if (!boatsModel) notFound()

  const listings = await prisma.listing.findMany({
    where: { modelId: boatsModel.id },
    orderBy: { firstSeen: 'desc' },
  })

  const now = new Date()

  // model_stats skrivs av seed och (från steg 2) av pipeline/stats.py. Saknas raden
  // räknar vi om på plats så att sidan aldrig kraschar på en modell utan statistik.
  const stored = await prisma.modelStats.findUnique({
    where: { modelId_windowDays: { modelId: boatsModel.id, windowDays: WINDOW_DAYS } },
  })
  const stats = stored ?? computeModelStats(listings, now)
  const confidence = (
    stored ? (stored.confidence as Confidence) : confidenceFor(stats.nActive + stats.nRemoved)
  ) satisfies Confidence

  const chartData = weeklyMedians(listings, now, CHART_WEEKS)
  const latest = listings.slice(0, 20)

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
        <div
          aria-hidden
          className="flex h-24 w-40 shrink-0 items-center justify-center rounded-lg border border-border bg-muted text-xs text-muted-foreground"
        >
          bild
        </div>
        <div>
          <h1 className="text-2xl font-semibold">
            {boatsModel.brand} {boatsModel.model}
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {boatsModel.type}
            {boatsModel.lengthM ? ` · ${boatsModel.lengthM} m` : ''} · {listings.length} annonser
            totalt
          </p>
          <div className="mt-2 flex items-center gap-2">
            <Badge variant={confidenceVariant[confidence]}>Konfidens: {confidence}</Badge>
            <span className="text-xs text-muted-foreground">
              {stats.nActive + stats.nRemoved} observationer senaste {WINDOW_DAYS} dagarna
            </span>
          </div>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Aktiva annonser – listpris</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-semibold">{formatSek(stats.median)}</div>
            <div className="mt-1 text-sm text-muted-foreground">
              {formatSek(stats.p25)} – {formatSek(stats.p75)} (p25–p75)
            </div>
            <div className="mt-2 text-xs text-muted-foreground">
              {stats.nActive} aktiva annonser
            </div>
          </CardContent>
        </Card>

        <Card className="border-emerald-200 bg-emerald-50/50">
          <CardHeader>
            <CardTitle>Borttagna annonser – troligen sålt</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-semibold">{formatSek(stats.medianRemoved)}</div>
            <div className="mt-1 text-sm text-muted-foreground">
              {formatSek(stats.p25Removed)} – {formatSek(stats.p75Removed)} (p25–p75)
            </div>
            <div className="mt-2 text-xs text-muted-foreground">
              {stats.nRemoved} borttagna annonser
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Median dagar på marknaden</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-semibold">
              {stats.medianDaysOnMarket === null ? '–' : `${stats.medianDaysOnMarket} dagar`}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Snittprissänkning</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-semibold">{formatPercent(stats.avgPriceDropPct)}</div>
            <div className="mt-1 text-xs text-muted-foreground">
              Bland annonser som sänkt priset minst en gång.
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Annonspris per vecka – {CHART_WEEKS} veckor</CardTitle>
        </CardHeader>
        <CardContent>
          <PriceChart data={chartData} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Senaste {latest.length} annonserna</CardTitle>
        </CardHeader>
        <CardContent>
          {latest.length === 0 ? (
            <p className="py-6 text-sm text-muted-foreground">Inga annonser för den här modellen.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Pris</TableHead>
                  <TableHead>År</TableHead>
                  <TableHead>Region</TableHead>
                  <TableHead>Dagar ute</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Prissänkningar</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {latest.map((listing) => {
                  const removed = listing.removedAt !== null
                  return (
                    <TableRow key={listing.id}>
                      <TableCell className="font-medium">{formatSek(listing.price)}</TableCell>
                      <TableCell>{listing.year ?? '–'}</TableCell>
                      <TableCell>{listing.region ?? '–'}</TableCell>
                      <TableCell>
                        {daysBetween(listing.firstSeen, listing.removedAt ?? now)}
                      </TableCell>
                      <TableCell>
                        <Badge variant={removed ? 'removed' : 'active'}>
                          {removed ? 'borttagen' : 'aktiv'}
                        </Badge>
                      </TableCell>
                      <TableCell>{priceDrops(listing.priceHistory)}</TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
