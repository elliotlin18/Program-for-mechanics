// Skriver ut antalet annonser i databasen, inget annat. Används av start.sh och
// start.ps1 för att avgöra om underlaget är för tunt.
//
// Ligger i en fil i stället för inline i startskripten: PowerShell plockar bort
// citattecken när argument skickas till program, så inline-JS med strängar i
// blir tyst felaktig i stället för att fela.

import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

try {
  console.log(await prisma.listing.count())
} catch {
  console.log(0)
} finally {
  await prisma.$disconnect()
}
