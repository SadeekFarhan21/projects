'use server'

import { revalidatePath } from 'next/cache'
import { resolveRange, syncRange } from '@/lib/wakatime'

// Refresh the last N days (including today) via the WakaTime sync.
// Runs server-side, so it calls the sync directly — no secret needs to reach
// the browser, and the public /api/wakatime/sync route stays locked down.
export async function refreshRecent(days = 2): Promise<{ synced: number; error?: string }> {
  const { start, end } = resolveRange({ days })
  try {
    const result = await syncRange(start, end, true)
    if (result.error) return { synced: 0, error: result.error }
    revalidatePath('/dashboard')
    return { synced: result.synced }
  } catch (err) {
    return { synced: 0, error: err instanceof Error ? err.message : 'Refresh failed' }
  }
}
