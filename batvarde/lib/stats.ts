import { daysBetween } from './utils'

export const WINDOW_DAYS = 90
export const CHART_WEEKS = 12

export type PricePoint = { date: string; price: number }

/** Minsta gemensamma nämnare av en listing – allt statistiken behöver. */
export type StatListing = {
  price: number
  firstSeen: Date
  lastSeen: Date
  removedAt: Date | null
  priceHistory: string
}

export type ModelStatsResult = {
  windowDays: number
  nActive: number
  nRemoved: number
  p25: number | null
  median: number | null
  p75: number | null
  p25Removed: number | null
  medianRemoved: number | null
  p75Removed: number | null
  medianDaysOnMarket: number | null
  avgPriceDropPct: number | null
  confidence: Confidence
}

export type Confidence = 'hög' | 'medel' | 'låg'

export function parsePriceHistory(raw: string): PricePoint[] {
  try {
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? (parsed as PricePoint[]) : []
  } catch {
    return []
  }
}

/** Antal gånger priset sänkts i annonsens historik. */
export function priceDrops(raw: string): number {
  const history = parsePriceHistory(raw)
  let drops = 0
  for (let i = 1; i < history.length; i++) {
    if (history[i].price < history[i - 1].price) drops++
  }
  return drops
}

/** Linjärt interpolerad percentil. p anges 0–1. */
export function percentile(values: number[], p: number): number | null {
  if (values.length === 0) return null
  const sorted = [...values].sort((a, b) => a - b)
  if (sorted.length === 1) return sorted[0]
  const pos = (sorted.length - 1) * p
  const lower = Math.floor(pos)
  const upper = Math.ceil(pos)
  if (lower === upper) return Math.round(sorted[lower])
  return Math.round(sorted[lower] + (sorted[upper] - sorted[lower]) * (pos - lower))
}

/** hög >= 30 observationer, medel 10–29, låg < 10 (docs/PROTOTYP.md steg 1). */
export function confidenceFor(observations: number): Confidence {
  if (observations >= 30) return 'hög'
  if (observations >= 10) return 'medel'
  return 'låg'
}

export function isActive(listing: StatListing): boolean {
  return listing.removedAt === null
}

/**
 * Statistik per modell. Aktiva annonser = listpriser, borttagna = troligen sålt-priser.
 * Samma regler som pipeline/stats.py ska skriva till model_stats i steg 2.
 */
export function computeModelStats(
  listings: StatListing[],
  now: Date,
  windowDays: number = WINDOW_DAYS,
): ModelStatsResult {
  const cutoff = new Date(now.getTime() - windowDays * 86_400_000)

  const active = listings.filter((l) => isActive(l) && l.lastSeen >= cutoff)
  const removed = listings.filter((l) => l.removedAt !== null && l.removedAt >= cutoff)

  const activePrices = active.map((l) => l.price)
  const removedPrices = removed.map((l) => l.price)

  // Dagar på marknaden räknas i första hand på borttagna annonser – de är avslutade
  // spelningar. Finns inga borttagna ännu använder vi hur länge de aktiva legat ute.
  const daysSource = removed.length > 0 ? removed : active
  const daysOnMarket = daysSource.map((l) => daysBetween(l.firstSeen, l.removedAt ?? now))

  const dropPcts: number[] = []
  for (const listing of [...active, ...removed]) {
    const history = parsePriceHistory(listing.priceHistory)
    if (history.length < 2) continue
    const first = history[0].price
    const last = history[history.length - 1].price
    if (first > 0 && last < first) dropPcts.push(((first - last) / first) * 100)
  }

  return {
    windowDays,
    nActive: active.length,
    nRemoved: removed.length,
    p25: percentile(activePrices, 0.25),
    median: percentile(activePrices, 0.5),
    p75: percentile(activePrices, 0.75),
    p25Removed: percentile(removedPrices, 0.25),
    medianRemoved: percentile(removedPrices, 0.5),
    p75Removed: percentile(removedPrices, 0.75),
    medianDaysOnMarket: percentile(daysOnMarket, 0.5),
    avgPriceDropPct:
      dropPcts.length > 0
        ? Math.round((dropPcts.reduce((a, b) => a + b, 0) / dropPcts.length) * 10) / 10
        : null,
    confidence: confidenceFor(active.length + removed.length),
  }
}

export type WeekPoint = { week: string; median: number | null; antal: number }

/**
 * Medianpris per vecka för annonser som låg ute den veckan.
 * Ger grafen på modellsidan – CHART_WEEKS veckor bakåt.
 */
export function weeklyMedians(
  listings: StatListing[],
  now: Date,
  weeks: number = CHART_WEEKS,
): WeekPoint[] {
  const points: WeekPoint[] = []
  const week = 7 * 86_400_000

  for (let i = weeks - 1; i >= 0; i--) {
    const end = new Date(now.getTime() - i * week)
    const start = new Date(end.getTime() - week)

    const onMarket = listings.filter(
      (l) => l.firstSeen <= end && (l.removedAt === null || l.removedAt >= start),
    )

    points.push({
      week: start.toISOString().slice(5, 10).replace('-', '/'),
      median: percentile(
        onMarket.map((l) => l.price),
        0.5,
      ),
      antal: onMarket.length,
    })
  }

  return points
}
