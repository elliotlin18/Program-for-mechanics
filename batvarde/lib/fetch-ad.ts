import { execFile } from 'node:child_process'
import path from 'node:path'
import { promisify } from 'node:util'

const run = promisify(execFile)

const SCRIPT = path.join(process.cwd(), 'pipeline', 'fetch_ad.py')

export type FetchedAd = {
  url: string
  title: string
  price: number
  year: number | null
  engine_brand: string | null
  engine_hp: number | null
  hours: number | null
  region: string | null
  via: string
  /** Satt av samma alias-matchning som scrapern använder. */
  model_id: number | null
}

export type FetchAdResult =
  | { ok: true; ad: FetchedAd }
  | { ok: false; error: string; diagnosis?: string }

/**
 * Hämtar en annons via pipeline/fetch_ad.py, så att /kolla använder exakt samma
 * parser och samma robots- och rate limit-regler som scrapern. Ett andra
 * tolkningsförsök i TypeScript skulle garanterat glida isär från det första.
 */
export async function fetchAd(url: string): Promise<FetchAdResult> {
  try {
    const { stdout } = await run('python3', [SCRIPT, url], {
      timeout: 45_000,
      maxBuffer: 4 * 1024 * 1024,
    })
    return JSON.parse(stdout) as FetchAdResult
  } catch (error) {
    // Skriptet avslutar med kod 1 när det misslyckas, men skriver ändå sin JSON.
    const stdout = (error as { stdout?: string }).stdout
    if (stdout) {
      try {
        return JSON.parse(stdout) as FetchAdResult
      } catch {
        // faller igenom till felet nedan
      }
    }
    return {
      ok: false,
      error: 'Kunde inte köra pipeline/fetch_ad.py. Är python3 och beroendena installerade?',
      diagnosis: error instanceof Error ? error.message : String(error),
    }
  }
}
