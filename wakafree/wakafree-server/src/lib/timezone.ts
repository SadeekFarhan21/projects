// Where the user was over time, so each day renders in the timezone they were
// actually in. IANA zones handle daylight-saving transitions automatically, so
// only the location changes are encoded here.
//
// Ordered by date; timezoneForDate() picks the last segment that has started.
// To record a travel or timezone change, append one line.
const TZ_SCHEDULE: Array<{ from: string; tz: string }> = [
  { from: '0000-01-01', tz: 'UTC' }, // replace with your own schedule
]

// IANA timezone the user was in on the given YYYY-MM-DD date.
export function timezoneForDate(dateStr: string): string {
  let tz = TZ_SCHEDULE[0].tz
  for (const seg of TZ_SCHEDULE) {
    if (dateStr >= seg.from) tz = seg.tz
    else break
  }
  return tz
}

// Minutes that `tz` is offset from UTC at the given instant (local - UTC).
// e.g. PDT → -420, EDT → -240.
function tzOffsetMinutes(date: Date, tz: string): number {
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat('en-US', {
      timeZone: tz,
      hour12: false,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
      .formatToParts(date)
      .map((p) => [p.type, p.value])
  )
  const asUTC = Date.UTC(
    Number(parts.year),
    Number(parts.month) - 1,
    Number(parts.day),
    Number(parts.hour),
    Number(parts.minute),
    Number(parts.second)
  )
  return (asUTC - date.getTime()) / 60000
}

// Epoch seconds of local midnight for `dateStr` in `tz` — the start of that
// calendar day as experienced in that timezone.
export function localMidnightEpochSeconds(dateStr: string, tz: string): number {
  const utcMidnight = new Date(`${dateStr}T00:00:00Z`)
  const offsetMin = tzOffsetMinutes(utcMidnight, tz)
  // Local midnight occurs `offsetMin` after UTC midnight (offset is negative in the Americas).
  return Math.round((utcMidnight.getTime() - offsetMin * 60000) / 1000)
}

// Today's YYYY-MM-DD in the timezone the user is currently in (per the schedule).
export function currentLocalDate(): string {
  const fmt = (tz: string) =>
    new Intl.DateTimeFormat('en-CA', {
      timeZone: tz,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    }).format(new Date())
  // Rough pass to pick the segment, then format in the resolved zone.
  const rough = fmt(TZ_SCHEDULE[TZ_SCHEDULE.length - 1].tz)
  return fmt(timezoneForDate(rough))
}
