process.env.WAKATIME_API_KEY = 'test'
const S = process.env.WF + '/wakafree-server/src/lib/'
const c = await import(S + 'compute.ts')
const tz = await import(S + 'timezone.ts')
const out = {}
// compute: heartbeats at 0, 5, 10 min, then 20-min gap, then 2 more 5 min apart
const m = (x) => x * 60
const hb = [0, m(5), m(10), m(30), m(35)].map((t) => ({ time: 1e9 + t, project: 'p' }))
out.compute_total_seconds_gap_20min = c.computeTotalSeconds(hb)
out.compute_total_seconds_gap_14min = c.computeTotalSeconds([0, m(14)].map((t) => ({ time: t })))
out.compute_total_seconds_gap_15min = c.computeTotalSeconds([0, m(15)].map((t) => ({ time: t })))
out.formatSeconds_3725 = c.formatSeconds(3725)
out.compactNumber_1234567 = c.compactNumber(1234567)
// timezone: local midnight across DST (generic zone used for the check)
const z = 'America/New_York'
out.midnight_utc_epoch = tz.localMidnightEpochSeconds('2026-01-15', 'UTC')
out.midnight_ny_jan = tz.localMidnightEpochSeconds('2026-01-15', z) - tz.localMidnightEpochSeconds('2026-01-15', 'UTC')
out.midnight_ny_jul = tz.localMidnightEpochSeconds('2026-07-15', z) - tz.localMidnightEpochSeconds('2026-07-15', 'UTC')
// sync fan-out with a counting fetch
const urls = []
globalThis.fetch = async (u) => {
  urls.push(u)
  const url = new URL(u)
  if (url.pathname.endsWith('/summaries')) {
    const s = new Date(url.searchParams.get('start') + 'T00:00:00Z'), e = new Date(url.searchParams.get('end') + 'T00:00:00Z')
    const data = []
    for (let d = s; d <= e; d = new Date(d.getTime() + 864e5)) data.push({ range: { date: d.toISOString().slice(0, 10) } })
    return { ok: true, json: async () => ({ data }) }
  }
  return { ok: true, json: async () => ({ data: {} }) }
}
const w = await import(S + 'wakatime.ts')
const { calls } = await import('./stubs/supabase.mjs')
for (const [label, days, meta] of [['days2_meta0', 2, false], ['days2_meta1', 2, true], ['days7_meta0', 7, false], ['backfill365_meta1', 365, true]]) {
  urls.length = 0; calls.length = 0
  const r = await w.syncRange('2026-01-01', new Date(Date.UTC(2026, 0, days)).toISOString().slice(0, 10), meta)
  out['sync_' + label] = { days: r.synced, fetches: urls.length, upsert_rows: calls.map(x => x.t + ':' + x.n) }
}
out.resolveRange_backfill_999_days_span = (() => { const r = w.resolveRange({ backfill: 999 }); return Math.round((new Date(r.end) - new Date(r.start)) / 864e5) + 1 })()
console.log(JSON.stringify(out, null, 2))
