import { PrismaClient } from '@prisma/client'
import { WINDOW_DAYS } from '../lib/stats'
import { formatSek, slugify } from '../lib/utils'

const prisma = new PrismaClient()

async function main() {
  const models = await prisma.boatsModel.findMany({
    orderBy: [{ brand: 'asc' }, { model: 'asc' }],
    include: { listings: true, stats: true },
  })

  const listings = await prisma.listing.count()
  const removed = await prisma.listing.count({ where: { NOT: { removedAt: null } } })

  console.log(`Modeller:  ${models.length}`)
  console.log(`Annonser:  ${listings} (varav ${removed} borttagna)`)
  console.log('')

  for (const model of models) {
    const stats = model.stats.find((s) => s.windowDays === WINDOW_DAYS)
    const url = `/bat/${slugify(model.brand)}/${slugify(model.model)}`
    const summary = stats
      ? `median ${formatSek(stats.median)} aktiva / ${formatSek(stats.medianRemoved)} borttagna, konfidens ${stats.confidence}`
      : 'ingen statistik'
    console.log(`  ${url.padEnd(26)} ${String(model.listings.length).padStart(3)} annonser  ${summary}`)
  }

  // Klart-kriterier för prototypen, docs/PROTOTYP.md avsnitt 8.
  console.log('')
  console.log(`Klart-kriterier: >=10 modeller: ${models.length >= 10 ? 'ja' : 'nej'}`)
  console.log(`                 >=300 annonser: ${listings >= 300 ? 'ja' : 'nej'}`)
  console.log(`                 >=20 borttagna: ${removed >= 20 ? 'ja' : 'nej'}`)
}

main()
  .catch((error) => {
    console.error(error)
    process.exitCode = 1
  })
  .finally(() => prisma.$disconnect())
