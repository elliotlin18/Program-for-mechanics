import { readFileSync } from 'node:fs'
import path from 'node:path'
import { parse } from 'csv-parse/sync'
import YAML from 'yaml'
import { PrismaClient } from '@prisma/client'
import { WINDOW_DAYS, computeModelStats } from '../lib/stats'
import { formatDate } from '../lib/utils'

const prisma = new PrismaClient()

const ROOT = process.cwd()
const ALIASES_PATH = path.join(ROOT, 'pipeline', 'aliases.yaml')
// Standard är data/seed.csv. `npm run db:seed -- data/seed_demo.csv` läser en annan fil.
const CSV_PATH = path.join(ROOT, process.argv[2] ?? path.join('data', 'seed.csv'))

/** Annons utan removed_at ligger ute i 14 dagar innan vi kallar den borttagen (steg 0). */
const REMOVED_AFTER_DAYS = 14

type AliasEntry = {
  brand: string
  // YAML läser "27" och "640" som tal – normaliseras med String() nedan.
  model: string | number
  type: string
  length_m?: number
  aliases?: string[]
}

type SeedRow = {
  source: string
  url: string
  brand: string
  model: string
  year: string
  price: string
  engine_brand: string
  engine_hp: string
  hours: string
  region: string
  first_seen: string
  status: string
}

function optionalInt(value: string): number | null {
  const trimmed = value?.trim()
  if (!trimmed) return null
  const parsed = Number.parseInt(trimmed, 10)
  return Number.isNaN(parsed) ? null : parsed
}

function addDays(date: Date, days: number): Date {
  return new Date(date.getTime() + days * 86_400_000)
}

/** Sista biten av URL:en duger som källans id i prototypen. */
function sourceIdFromUrl(url: string): string {
  return url.replace(/\/+$/, '').split('/').pop() || url
}

async function seedModels(): Promise<Map<string, number>> {
  const entries = YAML.parse(readFileSync(ALIASES_PATH, 'utf8')) as AliasEntry[]
  const byKey = new Map<string, number>()

  for (const entry of entries) {
    const brand = String(entry.brand)
    const model = String(entry.model)

    const record = await prisma.boatsModel.upsert({
      where: { brand_model: { brand, model } },
      update: {
        type: entry.type,
        lengthM: entry.length_m ?? null,
        aliases: JSON.stringify(entry.aliases ?? []),
      },
      create: {
        brand,
        model,
        type: entry.type,
        lengthM: entry.length_m ?? null,
        aliases: JSON.stringify(entry.aliases ?? []),
      },
    })

    byKey.set(`${brand} ${model}`.toLowerCase(), record.id)
    for (const alias of entry.aliases ?? []) byKey.set(alias.toLowerCase(), record.id)
  }

  return byKey
}

async function seedListings(modelIds: Map<string, number>): Promise<number> {
  const rows = parse(readFileSync(CSV_PATH, 'utf8'), {
    columns: true,
    skip_empty_lines: true,
    trim: true,
  }) as SeedRow[]

  const today = new Date(`${formatDate(new Date())}T00:00:00.000Z`)
  let imported = 0

  for (const row of rows) {
    const key = `${row.brand} ${row.model}`.toLowerCase()
    const modelId = modelIds.get(key) ?? null
    if (modelId === null) {
      console.warn(`  ! okänd modell, hoppar över: ${row.brand} ${row.model} (${row.url})`)
      continue
    }

    const firstSeen = new Date(`${row.first_seen}T00:00:00.000Z`)
    const removed = row.status?.trim().toLowerCase() === 'removed'
    const removedAt = removed ? addDays(firstSeen, REMOVED_AFTER_DAYS) : null
    const price = Number.parseInt(row.price, 10)

    const data = {
      source: row.source,
      sourceId: sourceIdFromUrl(row.url),
      url: row.url,
      modelId,
      year: optionalInt(row.year),
      price,
      engineBrand: row.engine_brand || null,
      engineHp: optionalInt(row.engine_hp),
      hours: optionalInt(row.hours),
      region: row.region || null,
      titleRaw: `${row.brand} ${row.model} ${row.year}`.trim(),
      firstSeen,
      lastSeen: removedAt ?? today,
      removedAt,
      priceHistory: JSON.stringify([{ date: row.first_seen, price }]),
    }

    await prisma.listing.upsert({ where: { url: row.url }, update: data, create: data })
    imported++
  }

  return imported
}

/** Samma beräkning som pipeline/stats.py ska göra i steg 2. */
async function writeModelStats() {
  const models = await prisma.boatsModel.findMany({ include: { listings: true } })
  const now = new Date()

  for (const model of models) {
    const stats = computeModelStats(model.listings, now, WINDOW_DAYS)
    const data = { ...stats, modelId: model.id }

    await prisma.modelStats.upsert({
      where: { modelId_windowDays: { modelId: model.id, windowDays: WINDOW_DAYS } },
      update: data,
      create: data,
    })
  }
}

async function main() {
  console.log(`Modeller från ${path.relative(ROOT, ALIASES_PATH)}`)
  const modelIds = await seedModels()

  console.log(`Annonser från ${path.relative(ROOT, CSV_PATH)}`)
  const imported = await seedListings(modelIds)

  console.log('Räknar model_stats')
  await writeModelStats()

  console.log(`Klart: ${imported} annonser importerade.`)
}

main()
  .catch((error) => {
    console.error(error)
    process.exitCode = 1
  })
  .finally(() => prisma.$disconnect())
