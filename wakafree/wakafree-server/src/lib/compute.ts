const TIMEOUT_SECONDS = 15 * 60

interface RawHeartbeat {
  time: number
  project?: string | null
  language?: string | null
}

export function computeTotalSeconds(heartbeats: RawHeartbeat[]): number {
  if (heartbeats.length === 0) return 0
  const sorted = [...heartbeats].sort((a, b) => a.time - b.time)
  let total = 0
  for (let i = 1; i < sorted.length; i++) {
    const diff = sorted[i].time - sorted[i - 1].time
    if (diff < TIMEOUT_SECONDS) total += diff
  }
  return Math.round(total)
}

export function computeByKey(
  heartbeats: RawHeartbeat[],
  key: 'project' | 'language'
): Array<{ name: string; seconds: number }> {
  const groups = new Map<string, RawHeartbeat[]>()
  for (const hb of heartbeats) {
    const value = hb[key] ?? 'Unknown'
    if (!groups.has(value)) groups.set(value, [])
    groups.get(value)!.push(hb)
  }
  return Array.from(groups.entries())
    .map(([name, hbs]) => ({ name, seconds: computeTotalSeconds(hbs) }))
    .sort((a, b) => b.seconds - a.seconds)
}

export function formatSeconds(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  if (h > 0) return `${h} hr ${m} min`
  return `${m} min`
}

export function compactNumber(n: number): string {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return String(Math.round(n))
}
