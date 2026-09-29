'use client'
import { useRef, useState } from 'react'
import TimelineChart from '@/components/TimelineChart'
import type { TimelinePayload } from '@/lib/timelineData'

const OPTIONS: Array<{ key: string; label: string }> = [
  { key: 'category', label: 'Category' },
  { key: 'language', label: 'Language' },
  { key: 'editor', label: 'Editor' },
  { key: 'os', label: 'Operating System' },
  { key: 'machine', label: 'Machine' },
]

function shiftDate(date: string, days: number): string {
  const [y, m, d] = date.split('-').map(Number)
  const dt = new Date(Date.UTC(y, m - 1, d, 12))
  dt.setUTCDate(dt.getUTCDate() + days)
  return dt.toISOString().split('T')[0]
}

// "2026-07-02" → "Thu Jul 2nd 2026"
function prettyDate(date: string): string {
  const dt = new Date(date + 'T12:00:00Z')
  const day = dt.getUTCDate()
  const suffix =
    day % 10 === 1 && day !== 11 ? 'st' : day % 10 === 2 && day !== 12 ? 'nd' : day % 10 === 3 && day !== 13 ? 'rd' : 'th'
  const weekday = dt.toLocaleDateString('en-US', { weekday: 'short', timeZone: 'UTC' })
  const month = dt.toLocaleDateString('en-US', { month: 'short', timeZone: 'UTC' })
  return `${weekday} ${month} ${day}${suffix} ${dt.getUTCFullYear()}`
}

function Chevron({ dir }: { dir: 'left' | 'right' }) {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      {dir === 'left' ? <polyline points="15 18 9 12 15 6" /> : <polyline points="9 18 15 12 9 6" />}
    </svg>
  )
}

function GearIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
      <path d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0z" />
    </svg>
  )
}

export default function TimelineCard({
  variant,
  initial,
  minDate,
  maxDate,
}: {
  variant: 'projects' | 'segmented'
  initial: TimelinePayload
  minDate: string
  maxDate: string
}) {
  const [payload, setPayload] = useState(initial)
  const [loading, setLoading] = useState(false)
  const [segment, setSegment] = useState('category')
  const [open, setOpen] = useState(false)
  const cache = useRef(new Map<string, TimelinePayload>([[initial.date, initial]]))

  const canPrev = payload.date > minDate
  const canNext = payload.date < maxDate

  async function go(delta: number) {
    const date = shiftDate(payload.date, delta)
    if (date < minDate || date > maxDate) return
    const cached = cache.current.get(date)
    if (cached) return setPayload(cached)
    setLoading(true)
    try {
      const res = await fetch(`/api/timeline?date=${date}`)
      if (!res.ok) return
      const next = (await res.json()) as TimelinePayload
      cache.current.set(date, next)
      setPayload(next)
    } finally {
      setLoading(false)
    }
  }

  const available = OPTIONS.filter((o) => (payload.slices[o.key] ?? []).length > 0)
  const activeSegment = available.some((o) => o.key === segment)
    ? segment
    : (available[0]?.key ?? segment)
  const blocks = variant === 'projects' ? payload.projectBlocks : (payload.slices[activeSegment] ?? [])

  const navBtn =
    'rounded-full p-1.5 text-outline transition-colors hover:bg-container-high hover:text-onsurface disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-outline'

  return (
    <div className={loading ? 'opacity-60 transition-opacity' : 'transition-opacity'}>
      {/* Nav header: ‹  total + date  › (+ gear for the segmented card) */}
      <div className="relative mb-4 flex items-center justify-center gap-3">
        <button onClick={() => go(-1)} disabled={!canPrev || loading} aria-label="Previous day" className={navBtn}>
          <Chevron dir="left" />
        </button>
        <p className="text-sm text-onsurface-variant">
          <span className="font-semibold text-onsurface">{payload.totalText ?? '0 mins'}</span>{' '}
          {prettyDate(payload.date)}
        </p>
        <button onClick={() => go(1)} disabled={!canNext || loading} aria-label="Next day" className={navBtn}>
          <Chevron dir="right" />
        </button>

        {variant === 'segmented' && (
          <div className="absolute right-0">
            <button
              onClick={() => setOpen(!open)}
              aria-label="Segment by"
              className={`rounded p-1 transition-colors hover:bg-container-high hover:text-onsurface ${
                open ? 'bg-container-high text-onsurface' : 'text-outline'
              }`}
            >
              <GearIcon />
            </button>
            {open && (
              <>
                <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
                <div className="absolute right-0 z-20 mt-2 w-56 rounded-lg border border-outline-variant bg-container shadow-[0px_20px_40px_rgba(0,0,0,0.4)]">
                  <p className="border-b border-line px-4 py-2.5 text-center text-sm font-medium text-onsurface">
                    Segment By
                  </p>
                  <div className="p-2">
                    {(available.length > 0 ? available : OPTIONS).map((o) => {
                      const selected = o.key === activeSegment
                      return (
                        <button
                          key={o.key}
                          onClick={() => {
                            setSegment(o.key)
                            setOpen(false)
                          }}
                          className="flex w-full items-center gap-2.5 rounded px-2.5 py-2 text-left text-sm text-onsurface-variant transition-colors hover:bg-container-high hover:text-onsurface"
                        >
                          <span
                            className={`flex h-3.5 w-3.5 items-center justify-center rounded-full border ${
                              selected ? 'border-primary-dim' : 'border-outline'
                            }`}
                          >
                            {selected && <span className="h-2 w-2 rounded-full bg-primary-dim" />}
                          </span>
                          {o.label}
                        </button>
                      )
                    })}
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {blocks.length > 0 ? (
        <TimelineChart date={payload.date} blocks={blocks} />
      ) : (
        <p className="text-outline text-sm text-center py-8">
          {variant === 'segmented' && payload.projectBlocks.length > 0
            ? 'Segments not synced for this day yet — re-sync it to load them.'
            : 'No timeline for this day.'}
        </p>
      )}
    </div>
  )
}
