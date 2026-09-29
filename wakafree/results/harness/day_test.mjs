// Edge cases behind the post: the 15-minute gap rule, the UTC-vs-local date,
// and how long a local day lasts, using compute.ts and timezone.ts from this copy.
const S = process.env.WF + '/wakafree-server/src/lib/'
const c = await import(S + 'compute.ts')
const tz = await import(S + 'timezone.ts')
const out = {}

// compute: heartbeats every 2 min for an hour, a 20-min break, then another hour
const hb = []
for (let t = 0; t <= 3600; t += 120) hb.push({ time: t })
for (let t = 3600 + 1200; t <= 3600 + 1200 + 3600; t += 120) hb.push({ time: t })
out.two_hours_with_20min_break_heartbeats = hb.length
out.two_hours_with_20min_break_total_seconds = c.computeTotalSeconds(hb)
out.gap_899s_total_seconds = c.computeTotalSeconds([{ time: 0 }, { time: 899 }])
out.gap_900s_total_seconds = c.computeTotalSeconds([{ time: 0 }, { time: 900 }])
out.single_heartbeat_total_seconds = c.computeTotalSeconds([{ time: 0 }])

// UTC date vs local date at 19:00 on 2026-01-15 in a zone five hours behind UTC
const zone5w = 'Etc/GMT+5' // IANA sign convention: Etc/GMT+5 is UTC-5
const instant = new Date(Date.UTC(2026, 0, 16, 0, 0, 0))
const localDate = (d, z) => new Intl.DateTimeFormat('en-CA', { timeZone: z, year: 'numeric', month: '2-digit', day: '2-digit' }).format(d)
const localTime = (d, z) => new Intl.DateTimeFormat('en-GB', { timeZone: z, hour: '2-digit', minute: '2-digit', hour12: false }).format(d)
out.utc5w_instant_local_time = localTime(instant, zone5w)
out.utc5w_instant_local_date = localDate(instant, zone5w)
out.utc5w_instant_utc_date = instant.toISOString().slice(0, 10)
out.utc5w_local_midnight_minus_utc_midnight_hours =
  (tz.localMidnightEpochSeconds('2026-01-15', zone5w) - tz.localMidnightEpochSeconds('2026-01-15', 'UTC')) / 3600

// Local day length = next local midnight minus this local midnight, in hours
const len = (d0, z0, d1, z1) => (tz.localMidnightEpochSeconds(d1, z1) - tz.localMidnightEpochSeconds(d0, z0)) / 3600
const ny = 'America/New_York'
out.day_hours_ordinary_2026_01_15_new_york = len('2026-01-15', ny, '2026-01-16', ny)
out.day_hours_spring_forward_2026_03_08_new_york = len('2026-03-08', ny, '2026-03-09', ny)
out.day_hours_fall_back_2026_11_01_new_york = len('2026-11-01', ny, '2026-11-02', ny)
// A day that starts in one zone and ends in another (generic fixed-offset zones)
out.day_hours_utc_to_two_hours_west = len('2026-01-15', 'UTC', '2026-01-16', 'Etc/GMT+2')
out.day_hours_utc_to_two_hours_east = len('2026-01-15', 'UTC', '2026-01-16', 'Etc/GMT-2')
out.day_hours_next_start_by_adding_86400_after_spring_forward_off_by_hours =
  (tz.localMidnightEpochSeconds('2026-03-08', ny) + 86400 - tz.localMidnightEpochSeconds('2026-03-09', ny)) / 3600
console.log(JSON.stringify(out, null, 2))
