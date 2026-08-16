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
import { WINDOW_DAYS, type Confidence } from '@/lib/stats'
import { DEFAULT_CONDITION, YEAR_SPAN, valuate } from '@/lib/valuation'
import { formatSek, slugify } from '@/lib/utils'

export const dynamic = 'force-dynamic'

const CONDITIONS = [
  { value: 1, label: '1 – renoveringsobjekt' },
  { value: 2, label: '2 – slitet' },
  { value: 3, label: '3 – normalt för åldern' },
  { value: 4, label: '4 – välvårdat' },
  { value: 5, label: '5 – nyskick' },
]

const confidenceVariant: Record<Confidence, 'high' | 'medium' | 'low'> = {
  hög: 'high',
  medel: 'medium',
  låg: 'low',
}

const field = 'w-full rounded-md border border-border bg-card px-3 py-2 text-sm'
const label = 'mb-1 block text-xs font-medium text-muted-foreground'

function toInt(value: string | undefined): number | null {
  if (!value) return null
  const parsed = Number.parseInt(value, 10)
  return Number.isNaN(parsed) ? null : parsed
}

export default async function VarderaPage({
  searchParams,
}: {
  searchParams: Record<string, string | undefined>
}) {
  const models = await prisma.boatsModel.findMany({
    orderBy: [{ brand: 'asc' }, { model: 'asc' }],
  })

  // Region- och motorlistor kommer ur datan i stället för en hårdkodad lista,
  // så de speglar alltid vad vi faktiskt har annonser för.
  const distinct = await prisma.listing.findMany({
    where: { modelId: { not: null } },
    select: { region: true, engineBrand: true },
    distinct: ['region', 'engineBrand'],
  })
  const unique = (values: (string | null)[]) =>
    Array.from(new Set(values.filter((v): v is string => Boolean(v)))).sort()

  const regions = unique(distinct.map((l) => l.region))
  const engineBrands = unique(distinct.map((l) => l.engineBrand))

  const modelId = toInt(searchParams.modell)
  const year = toInt(searchParams.ar)
  const engineHp = toInt(searchParams.hk)
  const hours = toInt(searchParams.timmar)
  const region = searchParams.region || null
  const engineBrand = searchParams.motormarke || null
  const condition = Math.min(5, Math.max(1, toInt(searchParams.skick) ?? DEFAULT_CONDITION))

  const boatsModel = modelId ? models.find((m) => m.id === modelId) : undefined
  const submitted = Boolean(boatsModel && year)

  const listings = submitted
    ? await prisma.listing.findMany({
        where: { modelId: boatsModel!.id },
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

  const result = submitted
    ? valuate(listings, { year: year!, engineBrand, engineHp, hours, region, condition }, new Date())
    : null

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold">Värdera min båt</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Vi jämför med annonser för samma modell inom ±{YEAR_SPAN} årsmodeller de senaste{' '}
          {WINDOW_DAYS} dagarna. Borttagna annonser väger tyngre än aktiva – de säger något om
          vad båtar faktiskt går för.
        </p>
      </div>

      <Card>
        <CardContent className="pt-5">
          <form className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <div className="sm:col-span-2 lg:col-span-1">
              <label className={label} htmlFor="modell">
                Märke och modell
              </label>
              <select id="modell" name="modell" className={field} defaultValue={modelId ?? ''}>
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
                placeholder="2018"
                className={field}
                defaultValue={year ?? ''}
              />
            </div>

            <div>
              <label className={label} htmlFor="motormarke">
                Motormärke
              </label>
              <select
                id="motormarke"
                name="motormarke"
                className={field}
                defaultValue={engineBrand ?? ''}
              >
                <option value="">Vet ej</option>
                {engineBrands.map((brand) => (
                  <option key={brand} value={brand}>
                    {brand}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className={label} htmlFor="hk">
                Motor, hk
              </label>
              <input
                id="hk"
                name="hk"
                type="number"
                min={1}
                max={2000}
                placeholder="200"
                className={field}
                defaultValue={engineHp ?? ''}
              />
            </div>

            <div>
              <label className={label} htmlFor="timmar">
                Motortimmar <span className="font-normal">(valfritt)</span>
              </label>
              <input
                id="timmar"
                name="timmar"
                type="number"
                min={0}
                max={50000}
                placeholder="300"
                className={field}
                defaultValue={hours ?? ''}
              />
            </div>

            <div>
              <label className={label} htmlFor="region">
                Region
              </label>
              <select id="region" name="region" className={field} defaultValue={region ?? ''}>
                <option value="">Hela Sverige</option>
                {regions.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className={label} htmlFor="skick">
                Skick
              </label>
              <select id="skick" name="skick" className={field} defaultValue={condition}>
                {CONDITIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-end sm:col-span-2 lg:col-span-3">
              <button
                type="submit"
                className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground"
              >
                Värdera
              </button>
              <span className="ml-3 text-xs text-muted-foreground">
                Ingen e-post, inget konto.
              </span>
            </div>
          </form>
        </CardContent>
      </Card>

      {result === null ? (
        <p className="text-sm text-muted-foreground">
          Fyll i märke, modell och årsmodell så räknar vi.
        </p>
      ) : result.n === 0 ? (
        <Card>
          <CardContent className="pt-5">
            <p className="text-sm">
              Vi har inga annonser för {boatsModel!.brand} {boatsModel!.model} från {year} ±
              {YEAR_SPAN} år de senaste {WINDOW_DAYS} dagarna, så vi vill inte gissa.
            </p>
            <p className="mt-2 text-sm text-muted-foreground">
              Prova en annan årsmodell, eller se{' '}
              <Link
                className="underline"
                href={`/bat/${slugify(boatsModel!.brand)}/${slugify(boatsModel!.model)}`}
              >
                allt vi har för modellen
              </Link>
              .
            </p>
          </CardContent>
        </Card>
      ) : (
        <>
          <Card>
            <CardHeader>
              <CardTitle>
                {boatsModel!.brand} {boatsModel!.model} {year}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-semibold">{formatSek(result.median)}</div>
              <div className="mt-1 text-sm text-muted-foreground">
                Sannolikt intervall {formatSek(result.low)} – {formatSek(result.high)}
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-2">
                <Badge variant={confidenceVariant[result.confidence]}>
                  Konfidens: {result.confidence}
                </Badge>
                <span className="text-xs text-muted-foreground">
                  Baserat på {result.n} observationer senaste {WINDOW_DAYS} dagarna –{' '}
                  {result.nActive} aktiva och {result.nRemoved} borttagna annonser.
                </span>
              </div>

              {condition !== DEFAULT_CONDITION && (
                <p className="mt-3 text-xs text-muted-foreground">
                  Skick {condition} justerar medianen{' '}
                  {Math.round((result.conditionFactor - 1) * 100)} % från{' '}
                  {formatSek(result.medianBeforeCondition)}.
                </p>
              )}
            </CardContent>
          </Card>

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
                    <TableHead>Timmar</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {result.comparables.map((listing) => (
                    <TableRow key={listing.id}>
                      <TableCell className="font-medium">{formatSek(listing.price)}</TableCell>
                      <TableCell>{listing.year ?? '–'}</TableCell>
                      <TableCell>{listing.region ?? '–'}</TableCell>
                      <TableCell>
                        {listing.engineBrand ?? '–'}
                        {listing.engineHp ? ` ${listing.engineHp} hk` : ''}
                      </TableCell>
                      <TableCell>{listing.hours ?? '–'}</TableCell>
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
        </>
      )}
    </div>
  )
}
