'use client'
import { useState, useTransition } from 'react'
import { useRouter } from 'next/navigation'
import { RANGE_OPTIONS } from '@/lib/ranges'

function CalendarIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
      <line x1="16" y1="2" x2="16" y2="6" />
      <line x1="8" y1="2" x2="8" y2="6" />
      <line x1="3" y1="10" x2="21" y2="10" />
    </svg>
  )
}

export default function RangePicker({ current }: { current: number }) {
  const router = useRouter()
  const [open, setOpen] = useState(false)
  const [isPending, startTransition] = useTransition()

  function select(days: number) {
    setOpen(false)
    startTransition(() => {
      router.push(days === 7 ? '/dashboard' : `/dashboard?range=${days}`, { scroll: false })
    })
  }

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        disabled={isPending}
        className={`flex items-center gap-2 rounded-lg border border-line px-3 py-1.5 text-sm font-medium transition-colors hover:border-outline-variant hover:text-onsurface disabled:opacity-60 ${
          open ? 'border-outline-variant text-onsurface' : 'text-onsurface-variant'
        }`}
      >
        <CalendarIcon />
        {isPending ? 'Loading…' : `Last ${current} Days`}
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div className="absolute right-0 z-20 mt-2 w-44 rounded-lg border border-outline-variant bg-container shadow-[0px_20px_40px_rgba(0,0,0,0.4)]">
            <div className="p-2">
              {RANGE_OPTIONS.map((days) => {
                const selected = days === current
                return (
                  <button
                    key={days}
                    onClick={() => select(days)}
                    className="flex w-full items-center gap-2.5 rounded px-2.5 py-2 text-left text-sm text-onsurface-variant transition-colors hover:bg-container-high hover:text-onsurface"
                  >
                    <span
                      className={`flex h-3.5 w-3.5 items-center justify-center rounded-full border ${
                        selected ? 'border-primary-dim' : 'border-outline'
                      }`}
                    >
                      {selected && <span className="h-2 w-2 rounded-full bg-primary-dim" />}
                    </span>
                    Last {days} Days
                  </button>
                )
              })}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
