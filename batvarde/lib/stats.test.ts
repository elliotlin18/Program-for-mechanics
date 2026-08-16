import assert from 'node:assert/strict'
import test from 'node:test'
import {
  TREND_WEEKS,
  confidenceFor,
  percentile,
  priceDrops,
  priceTrendPct,
  weeklyMedians,
  type StatListing,
  type WeekPoint,
} from './stats'

function weeks(medians: (number | null)[]): WeekPoint[] {
  return medians.map((median, i) => ({ week: `v${i}`, median, antal: median === null ? 0 : 1 }))
}

test('percentil interpolerar linjärt', () => {
  assert.equal(percentile([100, 200], 0.5), 150)
  assert.equal(percentile([100, 200, 300, 400], 0.25), 175)
  assert.equal(percentile([500], 0.5), 500)
  assert.equal(percentile([], 0.5), null)
})

test('konfidensgränserna är 30 och 10', () => {
  assert.equal(confidenceFor(30), 'hög')
  assert.equal(confidenceFor(29), 'medel')
  assert.equal(confidenceFor(10), 'medel')
  assert.equal(confidenceFor(9), 'låg')
})

test('prissänkningar räknar bara nedgångar', () => {
  const history = JSON.stringify([
    { date: '2026-06-01', price: 900_000 },
    { date: '2026-07-01', price: 850_000 },
    { date: '2026-07-15', price: 875_000 },
    { date: '2026-08-01', price: 800_000 },
  ])
  assert.equal(priceDrops(history), 2)
  assert.equal(priceDrops('[]'), 0)
  assert.equal(priceDrops('trasig json'), 0)
})

test('trenden jämför sista veckorna mot de första', () => {
  const stigande = weeks([100, 100, 100, 100, 110, 110, 110, 110])
  assert.equal(priceTrendPct(stigande), 10)

  const fallande = weeks([200, 200, 200, 200, 180, 180, 180, 180])
  assert.equal(priceTrendPct(fallande), -10)

  const platt = weeks([100, 100, 100, 100, 100, 100, 100, 100])
  assert.equal(priceTrendPct(platt), 0)
})

test('trenden är null när underlaget inte räcker till båda ändarna', () => {
  assert.equal(priceTrendPct(weeks([100, 100, 100])), null)
  assert.equal(priceTrendPct(weeks(Array(TREND_WEEKS * 2 - 1).fill(100))), null)
  assert.equal(priceTrendPct(weeks([null, null, null, null, 100, 100, 100, 100])), null)
})

test('veckomedianer räknar annonser som låg ute den veckan', () => {
  const day = 86_400_000
  const now = new Date('2026-08-16T00:00:00.000Z')

  const listing = (over: Partial<StatListing>): StatListing => ({
    price: 100_000,
    firstSeen: new Date(now.getTime() - 80 * day),
    lastSeen: now,
    removedAt: null,
    priceHistory: '[]',
    ...over,
  })

  // En annons som togs bort för länge sedan ska inte synas i de sista veckorna.
  const points = weeklyMedians(
    [
      listing({ price: 100_000 }),
      listing({ price: 900_000, removedAt: new Date(now.getTime() - 70 * day) }),
    ],
    now,
    12,
  )

  assert.equal(points.length, 12)
  assert.equal(points[points.length - 1].median, 100_000)
  assert.equal(points[points.length - 1].antal, 1)
})
