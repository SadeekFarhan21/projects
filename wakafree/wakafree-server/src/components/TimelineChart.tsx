import { timezoneForDate, localMidnightEpochSeconds } from '@/lib/timezone'

interface DurationBlock {
  time: number // epoch seconds (block start)
  duration: number // seconds
  project?: string | null
}

interface Props {
  date: string // YYYY-MM-DD — used to anchor the day in the right timezone
  blocks: DurationBlock[]
  // Gaps shorter than this merge into one continuous session block, and lane
  // totals are recomputed from the merged spans (WakaTime-style timeout).
  timeoutMinutes?: number
}

// Merge same-lane blocks whose gap is under the keystroke timeout.
function mergeBlocks(blocks: DurationBlock[], timeoutSeconds: number): DurationBlock[] {
  const sorted = [...blocks].sort((a, b) => a.time - b.time)
  const merged: DurationBlock[] = []
  for (const b of sorted) {
    const last = merged[merged.length - 1]
    if (last && b.time - (last.time + last.duration) < timeoutSeconds) {
      last.duration = Math.max(last.duration, b.time + b.duration - last.time)
    } else {
      merged.push({ ...b })
    }
  }
  return merged
}

// Muted Nord-style family (matches BreakdownPie)
const PALETTE = [
  '#b48ead', '#88c0d0', '#a3be8c', '#ebcb8b', '#d08770',
  '#8fbcbb', '#81a1c1', '#bf616a', '#d5b3cf', '#94a3b8',
]

// Hour labels across the day (12a, 3a, 6a, ... 12a)
const HOUR_MARKS = [0, 3, 6, 9, 12, 15, 18, 21, 24]
function hourLabel(h: number): string {
  const hh = h % 24
  if (hh === 0) return '12a'
  if (hh === 12) return '12p'
  return hh < 12 ? `${hh}a` : `${hh - 12}p`
}

function fmtShort(s: number): string {
  const h = Math.floor(s / 3600)
  const m = Math.round((s % 3600) / 60)
  return `${h}:${String(m).padStart(2, '0')}`
}

export default function TimelineChart({ date, blocks, timeoutMinutes = 60 }: Props) {
  // Anchor the day to local midnight in the timezone the user was in on this
  // date (per the timezone schedule), so the hour axis is correct.
  const tz = timezoneForDate(date)
  const startEpoch = localMidnightEpochSeconds(date, tz)
  const span = 86400 // one full day in seconds

  if (blocks.length === 0) {
    return <p className="text-outline text-sm text-center py-8">No timeline for this day.</p>
  }

  // One swimlane per project; blocks within the keystroke timeout merge into
  // continuous sessions, and lane totals come from the merged spans.
  const byProject = new Map<string, DurationBlock[]>()
  for (const b of blocks) {
    const name = b.project ?? 'Unknown'
    byProject.set(name, [...(byProject.get(name) ?? []), b])
  }
  const rows = Array.from(byProject.entries())
    .map(([name, laneBlocks]) => {
      const merged = mergeBlocks(laneBlocks, timeoutMinutes * 60)
      return {
        name,
        blocks: merged,
        seconds: merged.reduce((s, b) => s + b.duration, 0),
      }
    })
    .sort((a, b) => b.seconds - a.seconds)

  return (
    // Label column hugs the longest name (capped at 11rem) so short labels
    // like "Coding" don't leave a wide empty gutter.
    <div className="grid grid-cols-[fit-content(11rem)_2.5rem_minmax(0,1fr)] items-center gap-x-3 gap-y-1">
      {/* Hour axis */}
      <div />
      <div />
      <div className="relative h-4">
        {HOUR_MARKS.map((h) => (
          <span
            key={h}
            className="absolute -translate-x-1/2 font-mono text-[10px] text-outline"
            style={{ left: `${(h / 24) * 100}%` }}
          >
            {hourLabel(h)}
          </span>
        ))}
      </div>

      {/* Swimlanes */}
      {rows.map((row, ri) => (
        <div key={row.name} className="contents">
          <div className="truncate text-right text-xs text-onsurface-variant" title={row.name}>
            {row.name}
          </div>
          <div className="font-mono text-[11px] tracking-[0.02em] text-outline">
            {fmtShort(row.seconds)}
          </div>
          <div className="relative h-6 overflow-hidden rounded border border-line/60 bg-container-high/40">
            {row.blocks.map((b, i) => {
              const left = ((b.time - startEpoch) / span) * 100
              const width = (b.duration / span) * 100
              const mins = Math.round(b.duration / 60)
              return (
                <div
                  key={i}
                  className="absolute top-0 h-full"
                  style={{
                    left: `${Math.max(0, left)}%`,
                    width: `${Math.max(0.35, width)}%`,
                    backgroundColor: PALETTE[ri % PALETTE.length],
                  }}
                  title={`${row.name} · ${mins}m`}
                />
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}
