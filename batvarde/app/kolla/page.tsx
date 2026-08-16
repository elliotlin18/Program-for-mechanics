import Link from 'next/link'
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
import { prisma } from '@/lib/db'
import { fetchAd } from '@/lib/fetch-ad'
import { WINDOW_DAYS, priceDrops } from '@/lib/stats'
import {
  DEFAULT_CONDITION,
  DEVIATION_THRESHOLD,
  YEAR_SPAN,
  priceDeviation,
  valuate,
  verdictFor,
  type Verdict,
} from '@/lib/valuation'
import { daysBetween, formatPercent, formatSek, slugify } from '@/lib/utils'

export const dynamic = 'force-dynamic'

const field = 'w-full rounded-md border border-border bg-card px-3 py-2 text-sm'
const label = 'mb-1 block text-xs font-medium text-muted-foreground'

const verdictStyle: Record<Verdict, { box: string; text: string; heading: string }> = {
  under: {
    box: 'border-emerald-300 bg-emerald-50',
    text: 'text-emerald-900',
    heading: 'under marknaden',
  },
  rimlig: { box: 'border-amber-300 bg-amber-50', text: 'text-amber-900', heading: 'i nivå med marknaden' },
  over: { box: 'border-rose-300 bg-rose-50', text: 'text-rose-900', heading: 'över marknaden' },
}

function toInt(value: string | undefined): number | null {
  if (!value) return null
  const parsed = Number.parseInt(value, 10)
  return Number.isNaN(parsed) ? null : parsed
}

type Checked = {
  title: string
  price: number
  year: number | null
  region: string | null
  modelId: number | null
  url: string | null
  /** Vad vi vet om annonsens historik – bara om vi har sett den själva. */
  daysOnMarket: number | null
  drops: number | null
  origin: string
}

export default async function KollaPage({
  searchParams,
}: {
  searchParams: Record<string, string | undefined>
}) {
  const models = await prisma.boatsModel.findMany({ orderBy: [{ brand: 'asc' }, { model: 'asc' }] })

  const url = searchParams.url?.trim() || null
  const manualModelId = toInt(searchParams.modell)
  const manualYear = toInt(searchParams.ar)
  const manualPrice = toInt(searchParams.pris)

  let checked: Checked | null = null
  let error: string | null = null
  let diagnosis: string | null = null

  if (url) {
    // Har vi redan sett annonsen behöver vi inte hämta den igen – och då vet vi
    // dessutom hur länge den legat ute och om priset sänkts.
    const known = await prisma.listing.findUnique({ where: { url } })

    if (known) {
      checked = {
        title: known.titleRaw ?? url,
        price: known.price,
        year: known.year,
        region: known.region,
        modelId: known.modelId,
        url,
        daysOnMarket: daysBetween(known.firstSeen, known.removedAt ?? new Date()),
        drops: priceDrops(known.priceHistory),
        origin: 'ur vår egen databas',
      }
    } else {
      const result = await fetchAd(url)
      if (result.ok) {
        checked = {
          title: result.ad.title,
          price: result.ad.price,
          year: result.ad.year,
          region: result.ad.region,
          modelId: result.ad.model_id,
          url,
          daysOnMarket: null,
          drops: null,
          origin: `hämtad nu (via ${result.ad.via})`,
        }
      } else {
        error = result.error
        diagnosis = result.diagnosis ?? null
      }
    }
  } else if (manualModelId && manualYear && manualPrice) {
    const model = models.find((m) => m.id === manualModelId)
    checked = {
      title: model ? `${model.brand} ${model.model} ${manualYear}` : `Annons ${manualYear}`,
      price: manualPrice,
      year: manualYear,
      region: searchParams.region || null,
      modelId: manualModelId,
      url: null,
      daysOnMarket: null,
      drops: null,
      origin: 'ifylld för hand',
    }
  }

  const ad: Checked | null = checked
  const boatsModel = ad?.modelId ? models.find((m) => m.id === ad.modelId) : undefined

  const listings =
    boatsModel && ad?.year
      ? await prisma.listing.findMany({
          // Annonsen vi kollar får inte ingå i medianen den jämförs mot.
          where: { modelId: boatsModel.id, ...(ad.url ? { url: { not: ad.url } } : {}) },
          select: {
            id: true,
            price: true,
            year: true,
            region: true,
            engineBrand: true,
            engineHp: true,
            hours: true,
            lastSeen: true,
            removedAt: true,
          },
        })
      : []

  const valuation =
    boatsModel && ad?.year
      ? valuate(
          listings,
          { year: ad.year, region: ad.region, condition: DEFAULT_CONDITION },
          new Date(),
        )
      : null

  const deviation =
    valuation?.median && ad ? priceDeviation(ad.price, valuation.median) : null
  const verdict = deviation === null ? null : verdictFor(deviation)

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold">Annonskollen</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Klistra in en annonslänk så jämför vi priset med vad liknande båtar ligger på – och
          med vad de som försvunnit från marknaden låg på.
        </p>
      </div>

      <Card>
        <CardContent className="pt-5">
          <form className="flex flex-col gap-3 sm:flex-row sm:items-end">
            <div className="flex-1">
              <label className={label} htmlFor="url">
                Länk till annonsen
              </label>
              <input
                id="url"
                name="url"
                type="url"
                placeholder="https://www.blocket.se/annons/..."
                className={field}
                defaultValue={url ?? ''}
              />
            </div>
            <button
              type="submit"
              className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground"
            >
              Kolla annonsen
            </button>
          </form>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-amber-300 bg-amber-50">
          <CardContent className="pt-5">
            <p className="text-sm font-medium">{error}</p>
            {diagnosis && (
              <pre className="mt-2 overflow-x-auto whitespace-pre-wrap text-xs text-muted-foreground">
                {diagnosis}
              </pre>
            )}
            <p className="mt-3 text-sm">Fyll i uppgifterna för hand i stället:</p>
          </CardContent>
        </Card>
      )}

      {(error || !checked) && (
        <Card>
          <CardHeader>
            <CardTitle>Ange annonsen för hand</CardTitle>
          </CardHeader>
          <CardContent>
            <form className="grid gap-4 sm:grid-cols-3">
              <div>
                <label className={label} htmlFor="modell">
                  Märke och modell
                </label>
                <select id="modell" name="modell" className={field} defaultValue={manualModelId ?? ''}>
                  <option value="">Välj modell</option>
                  {models.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.brand} {model.model}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className={label} htmlFor="ar">
                  Årsmodell
                </label>
                <input
                  id="ar"
                  name="ar"
                  type="number"
                  min={1950}
                  max={2030}
                  className={field}
                  defaultValue={manualYear ?? ''}
                />
              </div>
              <div>
                <label className={label} htmlFor="pris">
                  Begärt pris, kr
                </label>
                <input
                  id="pris"
                  name="pris"
                  type="number"
                  min={1000}
                  className={field}
                  defaultValue={manualPrice ?? ''}
                />
              </div>
              <div className="sm:col-span-3">
                <button
                  type="submit"
                  className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground"
                >
                  Jämför
                </button>
              </div>
            </form>
          </CardContent>
        </Card>
      )}

      {ad && !boatsModel && (
        <Card>
          <CardContent className="pt-5 text-sm">
            Vi känner inte igen modellen i <span className="font-medium">{ad.title}</span>.
            Prototypen följer tio modeller – välj en av dem i formuläret ovan.
          </CardContent>
        </Card>
      )}

      {ad && boatsModel && valuation && (
        <>
          <Card className={verdict ? verdictStyle[verdict].box : undefined}>
            <CardHeader>
              <CardTitle>{ad.title}</CardTitle>
            </CardHeader>
            <CardContent>
              {deviation === null || verdict === null ? (
                <p className="text-sm">
                  Vi har inga jämförbara annonser för {boatsModel.brand} {boatsModel.model} från{' '}
                  {ad.year} ±{YEAR_SPAN} år de senaste {WINDOW_DAYS} dagarna, så vi vill
                  inte uttala oss.
                </p>
              ) : (
                <>
                  <div className={`text-3xl font-semibold ${verdictStyle[verdict].text}`}>
                    {deviation > 0 ? '+' : ''}
                    {formatPercent(Math.round(deviation * 10) / 10)}{' '}
                    <span className="text-xl font-medium">{verdictStyle[verdict].heading}</span>
                  </div>
                  <p className="mt-2 text-sm">
                    Annonsen ligger på {formatSek(ad.price)}. Medianen för{' '}
                    {boatsModel.brand} {boatsModel.model} {ad.year} ±{YEAR_SPAN} år är{' '}
                    {formatSek(valuation.median)}.
                  </p>
                </>
              )}

              <div className="mt-3 flex flex-wrap gap-4 text-sm">
                <span>
                  <span className="text-muted-foreground">Dagar ute: </span>
                  {ad.daysOnMarket ?? '–'}
                </span>
                <span>
                  <span className="text-muted-foreground">Prissänkningar: </span>
                  {ad.drops ?? '–'}
                </span>
                <span className="text-muted-foreground">
                  {valuation.n} observationer ({valuation.nActive} aktiva,{' '}
                  {valuation.nRemoved} borttagna) · {ad.origin}
                </span>
              </div>

              {ad.daysOnMarket === null && (
                <p className="mt-2 text-xs text-muted-foreground">
                  Dagar ute och prissänkningar vet vi bara för annonser vi själva följt över tid.
                  Kör <code className="font-mono">python pipeline/scrape.py</code> så börjar vi
                  följa den här.
                </p>
              )}

              <p className="mt-3 text-xs text-muted-foreground">
                Grön under −{DEVIATION_THRESHOLD} %, gul ±{DEVIATION_THRESHOLD} %, röd över +
                {DEVIATION_THRESHOLD} %.{' '}
                <Link
                  className="underline"
                  href={`/bat/${slugify(boatsModel.brand)}/${slugify(boatsModel.model)}`}
                >
                  Se hela modellsidan
                </Link>
                .
              </p>
            </CardContent>
          </Card>

          {valuation.comparables.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Jämförbara annonser</CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Pris</TableHead>
                      <TableHead>År</TableHead>
                      <TableHead>Region</TableHead>
                      <TableHead>Motor</TableHead>
                      <TableHead>Status</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {valuation.comparables.map((listing) => (
                      <TableRow key={listing.id}>
                        <TableCell className="font-medium">{formatSek(listing.price)}</TableCell>
                        <TableCell>{listing.year ?? '–'}</TableCell>
                        <TableCell>{listing.region ?? '–'}</TableCell>
                        <TableCell>
                          {listing.engineBrand ?? '–'}
                          {listing.engineHp ? ` ${listing.engineHp} hk` : ''}
                        </TableCell>
                        <TableCell>
                          <Badge variant={listing.removedAt ? 'removed' : 'active'}>
                            {listing.removedAt ? 'borttagen' : 'aktiv'}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
