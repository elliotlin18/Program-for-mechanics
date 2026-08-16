import { PrismaClient } from '@prisma/client'

// En instans i dev, annars skapar Next hot reload nya klienter tills SQLite säger ifrån.
const globalForPrisma = globalThis as unknown as { prisma?: PrismaClient }

export const prisma = globalForPrisma.prisma ?? new PrismaClient()

if (process.env.NODE_ENV !== 'production') globalForPrisma.prisma = prisma
