import assert from 'node:assert/strict'
import test from 'node:test'
import {
  REMOVED_WEIGHT,
  comparableDistance,
  conditionFactor,
  valuate,
  type ValuationListing,
} from './valuation'

const NOW = new Date('2026-08-16T00:00:00.000Z')
const RECENT = new Date('2026-08-01T00:00:00.000Z')
const OLD = new Date('2026-01-01T00:00:00.000Z')

let nextId = 1

function listing(overrides: Partial<ValuationListing> = {}): ValuationListing {
  return {
    id: nextId++,
    price: 500_000,
    year: 2018,
    region: 'Stockholm',
    engineBrand: 'Yamaha',
    engineHp: 200,
    hours: 300,
    lastSeen: RECENT,
    removedAt: null,
    ...overrides,
  }
}

const input = { year: 2018, condition: 3 }

test('skickjusteringen är 5 procent per steg från 3', () => {
  assert.equal(conditionFactor(3), 1)
  assert.equal(Math.round(conditionFactor(5) * 100) / 100, 1.1)
  assert.equal(Math.round(conditionFactor(1) * 100) / 100, 0.9)
})

test('bara annonser inom ±2 årsmodeller räknas', () => {
  const listings = [
    listing({ year: 2016, price: 100_000 }),
    listing({ year: 2020, price: 200_000 }),
    listing({ year: 2015, price: 999_000 }),
    listing({ year: 2021, price: 999_000 }),
    listing({ year: null, price: 999_000 }),
  ]
  const result = valuate(listings, input, NOW)
  assert.equal(result.n, 2)
  assert.equal(result.median, 150_000)
})

test('annonser utanför 90-dagarsfönstret räknas inte', () => {
  const listings = [
    listing({ price: 400_000, lastSeen: RECENT }),
    listing({ price: 900_000, lastSeen: OLD }),
    listing({ price: 900_000, lastSeen: OLD, removedAt: OLD }),
  ]
  const result = valuate(listings, input, NOW)
  assert.equal(result.n, 1)
  assert.equal(result.median, 400_000)
})

test('borttagna annonser väger tyngre än aktiva', () => {
  // En aktiv på 900k och en borttagen på 500k. Med lika vikt hade medianen
  // blivit 700k; borttagen väger dubbelt och drar den nedåt.
  const listings = [
    listing({ price: 900_000 }),
    listing({ price: 500_000, removedAt: RECENT }),
  ]
  const result = valuate(listings, input, NOW)
  assert.equal(REMOVED_WEIGHT, 2)
  assert.equal(result.nActive, 1)
  assert.equal(result.nRemoved, 1)
  assert.equal(result.median, 500_000)
  assert.ok(result.median! < 700_000, 'borttagen ska dra medianen nedåt')
})

test('n räknar annonser, inte viktade observationer', () => {
  const listings = [listing({ removedAt: RECENT }), listing({ removedAt: RECENT })]
  const result = valuate(listings, input, NOW)
  assert.equal(result.n, 2)
})

test('skick justerar intervallet men inte antalet', () => {
  const listings = [listing({ price: 400_000 }), listing({ price: 600_000 })]

  const neutral = valuate(listings, { year: 2018, condition: 3 }, NOW)
  const good = valuate(listings, { year: 2018, condition: 5 }, NOW)
  const bad = valuate(listings, { year: 2018, condition: 1 }, NOW)

  assert.equal(neutral.median, 500_000)
  assert.equal(good.median, 550_000)
  assert.equal(bad.median, 450_000)
  assert.equal(good.n, neutral.n)
  assert.equal(good.medianBeforeCondition, 500_000)
})

test('värdet avrundas till närmaste tusen', () => {
  // 333 333 * 1,05 = 349 999,65 -> 350 000, inte 349 999.
  const listings = [listing({ price: 333_333 })]
  const result = valuate(listings, { year: 2018, condition: 4 }, NOW)
  assert.equal(result.median, 350_000)
  assert.equal(result.median! % 1000, 0)
  // Rådata före justeringen avrundas inte – den ska gå att stämma av mot annonsen.
  assert.equal(result.medianBeforeCondition, 333_333)
})

test('tomt underlag ger inga siffror i stället för att krascha', () => {
  const result = valuate([], input, NOW)
  assert.equal(result.n, 0)
  assert.equal(result.median, null)
  assert.equal(result.low, null)
  assert.equal(result.high, null)
  assert.equal(result.confidence, 'låg')
  assert.deepEqual(result.comparables, [])
})

test('konfidens följer samma gränser som modellsidan', () => {
  const many = Array.from({ length: 30 }, () => listing())
  assert.equal(valuate(many, input, NOW).confidence, 'hög')
  assert.equal(valuate(many.slice(0, 10), input, NOW).confidence, 'medel')
  assert.equal(valuate(many.slice(0, 9), input, NOW).confidence, 'låg')
})

test('jämförbara sorteras efter närhet och begränsas till fem', () => {
  const exact = listing({ year: 2018, region: 'Stockholm', engineHp: 200, hours: 300 })
  const others = Array.from({ length: 8 }, (_, i) =>
    listing({ year: 2020, region: 'Skåne', engineHp: 100, hours: 2000 + i }),
  )
  const result = valuate([...others, exact], { ...input, region: 'Stockholm', engineHp: 200, hours: 300 }, NOW)

  assert.equal(result.comparables.length, 5)
  assert.equal(result.comparables[0].id, exact.id)
})

test('årsmodell väger tyngre än region i närhetsmåttet', () => {
  const sameYearOtherRegion = listing({ year: 2018, region: 'Skåne' })
  const otherYearSameRegion = listing({ year: 2020, region: 'Stockholm' })
  const withRegion = { ...input, region: 'Stockholm' }

  assert.ok(
    comparableDistance(sameYearOtherRegion, withRegion) <
      comparableDistance(otherYearSameRegion, withRegion),
  )
})
