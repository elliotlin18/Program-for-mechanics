import { WINDOW_DAYS, confidenceFor, percentile, type Confidence } from './stats'

/** Vi jämför med annonser inom ±2 årsmodeller. */
export const YEAR_SPAN = 2

/** Skick 1–5. Varje steg från 3 justerar värdet 5 %. */
export const CONDITION_STEP = 0.05
export const DEFAULT_CONDITION = 3

/**
 * Borttagna annonser väger tyngre än aktiva: en borttagen annons är ett pris
 * någon faktiskt slutade annonsera på, en aktiv är bara ett önskemål. Vikten är
 * ett heltal så att vi kan räkna den som två observationer och återanvända
 * percentile() från stats.ts i stället för att ha två percentilberäkningar.
 */
export const REMOVED_WEIGHT = 2

export type ValuationListing = {
  id: number
  price: number
  year: number | null
  region: string | null
  engineBrand: string | null
  engineHp: number | null
  hours: number | null
  lastSeen: Date
  removedAt: Date | null
}

export type ValuationInput = {
  year: number
  engineBrand?: string | null
  engineHp?: number | null
  hours?: number | null
  region?: string | null
  condition: number
}

export type ValuationResult = {
  n: number
  nActive: number
  nRemoved: number
  low: number | null
  median: number | null
  high: number | null
  /** Omedianen före skickjusteringen – används för att förklara justeringen. */
  medianBeforeCondition: number | null
  conditionFactor: number
  confidence: Confidence
  comparables: ValuationListing[]
}

export function conditionFactor(condition: number): number {
  return 1 + CONDITION_STEP * (condition - DEFAULT_CONDITION)
}

function inWindow(listing: ValuationListing, cutoff: Date): boolean {
  return listing.removedAt ? listing.removedAt >= cutoff : listing.lastSeen >= cutoff
}

/**
 * Hur nära en annons ligger den båt användaren beskrivit. Lägre är närmare.
 * Årsmodell väger tyngst, sedan region, motorstyrka, motormärke och timmar.
 */
export function comparableDistance(listing: ValuationListing, input: ValuationInput): number {
  let distance = Math.abs((listing.year ?? input.year) - input.year) * 3

  if (input.region && listing.region && listing.region !== input.region) distance += 2

  if (input.engineHp && listing.engineHp) {
    distance += Math.abs(listing.engineHp - input.engineHp) / 50
  }

  if (input.engineBrand && listing.engineBrand && listing.engineBrand !== input.engineBrand) {
    distance += 1
  }

  if (input.hours && listing.hours) {
    distance += Math.abs(listing.hours - input.hours) / 500
  }

  return distance
}

/**
 * Värdering: annonser för modellen inom ±2 år de senaste 90 dagarna, där
 * borttagna räknas dubbelt, justerat med skick.
 */
export function valuate(
  listings: ValuationListing[],
  input: ValuationInput,
  now: Date,
): ValuationResult {
  const cutoff = new Date(now.getTime() - WINDOW_DAYS * 86_400_000)

  const relevant = listings.filter(
    (listing) =>
      listing.year !== null &&
      Math.abs(listing.year - input.year) <= YEAR_SPAN &&
      inWindow(listing, cutoff),
  )

  const nRemoved = relevant.filter((listing) => listing.removedAt !== null).length

  // Borttagna annonser läggs in en extra gång – det är hela viktningen.
  const weighted: number[] = []
  for (const listing of relevant) {
    const weight = listing.removedAt !== null ? REMOVED_WEIGHT : 1
    for (let i = 0; i < weight; i++) weighted.push(listing.price)
  }

  const factor = conditionFactor(input.condition)
  // Avrundas till närmaste tusen. En värdering på kronan låtsas om en precision
  // vi inte har.
  const adjust = (value: number | null) =>
    value === null ? null : Math.round((value * factor) / 1000) * 1000
  const median = percentile(weighted, 0.5)

  return {
    n: relevant.length,
    nActive: relevant.length - nRemoved,
    nRemoved,
    low: adjust(percentile(weighted, 0.25)),
    median: adjust(median),
    high: adjust(percentile(weighted, 0.75)),
    medianBeforeCondition: median,
    conditionFactor: factor,
    confidence: confidenceFor(relevant.length),
    comparables: [...relevant]
      .sort((a, b) => comparableDistance(a, input) - comparableDistance(b, input))
      .slice(0, 5),
  }
}
