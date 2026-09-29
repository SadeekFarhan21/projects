import { supabase } from '@/lib/supabase'
import { currentLocalDate } from '@/lib/timezone'

const WAKATIME_API = 'https://wakatime.com/api/v1/users/current'

function wakaHeaders(): HeadersInit {
  const key = process.env.WAKATIME_API_KEY
  if (!key) throw new Error('WAKATIME_API_KEY env var is not set')
  return {
    Authorization: `Basic ${Buffer.from(key).toString('base64')}`,
  }
}

// Returns the calendar date `daysAgo` days before "today" in the timezone the
// user is currently in (see lib/timezone).
// Anchoring to the local day — not UTC — keeps today/yesterday aligned with how
// WakaTime buckets activity: in the evening in the Americas, UTC has already rolled to the next
// day, so a UTC-based date would sync the wrong day.
export function offsetDate(daysAgo: number): string {
  const [y, m, d] = currentLocalDate().split('-').map(Number)
  // Anchor at UTC noon so subtracting whole days never crosses a DST/midnight edge.
  const dt = new Date(Date.UTC(y, m - 1, d, 12))
  dt.setUTCDate(dt.getUTCDate() - daysAgo)
  return dt.toISOString().split('T')[0]
}

// Fetch a WakaTime endpoint and return its `data` field (or the whole body).
// Returns null on any failure so one bad endpoint never aborts the whole sync.
async function fetchWaka<T = unknown>(path: string): Promise<T | null> {
  try {
    const res = await fetch(`${WAKATIME_API}${path}`, { headers: wakaHeaders() })
    if (!res.ok) return null
    const json = await res.json()
    return (json?.data ?? json) as T
  } catch {
    return null
  }
}

// Like fetchWaka but returns the whole response body (used for durations, where
// we need the start/end/timezone wrapper to anchor the timeline, not just data).
async function fetchWakaFull<T = unknown>(path: string): Promise<T | null> {
  try {
    const res = await fetch(`${WAKATIME_API}${path}`, { headers: wakaHeaders() })
    if (!res.ok) return null
    return (await res.json()) as T
  } catch {
    return null
  }
}

// Pull every account-level dataset WakaTime exposes and upsert it into waka_meta.
async function syncMeta(): Promise<string[]> {
  const synced_at = new Date().toISOString()

  const endpoints: Record<string, string> = {
    profile: '',
    all_time: '/all_time_since_today',
    goals: '/goals',
    projects: '/projects',
    machine_names: '/machine_names',
    user_agents: '/user_agents',
    stats_last_7_days: '/stats/last_7_days',
    stats_last_30_days: '/stats/last_30_days',
    stats_all_time: '/stats/all_time',
  }

  const results = await Promise.all(
    Object.entries(endpoints).map(async ([key, path]) => {
      const data = await fetchWaka(path)
      return data === null ? null : { key, data, synced_at }
    })
  )

  const rows = results.filter((r): r is NonNullable<typeof r> => r !== null)
  if (rows.length > 0) {
    await supabase.from('waka_meta').upsert(rows, { onConflict: 'key' })
  }
  return rows.map((r) => r.key)
}

export interface RangeParams {
  date?: string | null
  days?: number | null
  backfill?: number | null
}

// Resolve a [start, end] date range from the supported param shapes.
export function resolveRange({ date, days, backfill }: RangeParams): { start: string; end: string } {
  if (date) return { start: date, end: date }
  if (days && days > 0) {
    // Recent N days, anchored to today (so today's in-progress data refreshes too).
    return { start: offsetDate(days - 1), end: offsetDate(0) }
  }
  const end = offsetDate(1)
  const n = Math.min(Math.max(backfill ?? 1, 1), 365)
  return { start: n > 1 ? offsetDate(n) : end, end }
}

export interface SyncResult {
  synced: number
  dates: string[]
  durations_captured: number
  meta_synced: string[]
  error?: string
  status?: number
}

// Core sync: pull summaries + durations for the range, plus account snapshots.
// Shared by the API route and the dashboard refresh server action.
export async function syncRange(
  start: string,
  end: string,
  includeMeta = true
): Promise<SyncResult> {
  const empty: SyncResult = { synced: 0, dates: [], durations_captured: 0, meta_synced: [] }

  // 1. Daily summaries for the whole range (single batched call).
  const res = await fetch(`${WAKATIME_API}/summaries?start=${start}&end=${end}`, {
    headers: wakaHeaders(),
  })
  if (!res.ok) {
    const text = await res.text()
    return { ...empty, error: `WakaTime ${res.status}: ${text.slice(0, 200)}`, status: 502 }
  }

  const json = (await res.json()) as { data: Array<{ range: { date: string } }> }
  const days = json.data ?? []

  // 2. Per-day durations (the timeline) — project-sliced plus every extra
  // slice the dashboard's "Segment By" picker offers, all in parallel.
  const TIMELINE_SLICES = ['category', 'language', 'editor', 'os', 'machine'] as const
  const durationsByDate = new Map<string, unknown>()
  const slicesByDate = new Map<string, Record<string, unknown>>()
  await Promise.all(
    days.flatMap((day) => {
      const d = day.range.date
      return [
        fetchWakaFull(`/durations?date=${d}`).then((durations) => {
          if (durations !== null) durationsByDate.set(d, durations)
        }),
        ...TIMELINE_SLICES.map((slice) =>
          fetchWakaFull(`/durations?date=${d}&slice_by=${slice}`).then((durations) => {
            if (durations !== null) {
              const cur = slicesByDate.get(d) ?? {}
              cur[slice] = durations
              slicesByDate.set(d, cur)
            }
          })
        ),
      ]
    })
  )

  const synced_at = new Date().toISOString()
  const rows = days.map((day) => ({
    date: day.range.date,
    data: day,
    durations: durationsByDate.get(day.range.date) ?? null,
    durations_category: slicesByDate.get(day.range.date)?.category ?? null,
    durations_slices: slicesByDate.get(day.range.date) ?? null,
    synced_at,
  }))

  if (rows.length > 0) {
    const { error } = await supabase.from('waka_daily').upsert(rows, { onConflict: 'date' })
    if (error) return { ...empty, error: error.message, status: 500 }
  }

  // 3. Account-level snapshots.
  const meta_synced = includeMeta ? await syncMeta() : []

  return {
    synced: rows.length,
    dates: rows.map((r) => r.date),
    durations_captured: durationsByDate.size,
    meta_synced,
  }
}
