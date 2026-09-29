'use client'
import { useState, useTransition } from 'react'
import { useRouter } from 'next/navigation'
import { refreshRecent } from '@/app/dashboard/actions'

export default function RefreshButton() {
  const router = useRouter()
  const [isPending, startTransition] = useTransition()
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<string | null>(null)

  const busy = loading || isPending

  async function refresh() {
    setLoading(true)
    setStatus(null)
    try {
      const result = await refreshRecent(2)
      if (result.error) throw new Error(result.error)
      setStatus('Refreshed')
      setTimeout(() => setStatus(null), 3000)
      // Re-render the server component with fresh data.
      startTransition(() => router.refresh())
    } catch (err) {
      setStatus(err instanceof Error ? err.message : 'Refresh failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex items-center gap-3">
      {status && <span className="font-mono text-[11px] tracking-[0.02em] text-outline">{status}</span>}
      <button
        onClick={refresh}
        disabled={busy}
        aria-label="Refresh data"
        title="Refresh data"
        className="rounded-lg border border-line bg-transparent p-2 text-onsurface-variant hover:border-outline-variant hover:text-onsurface disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        <svg
          className={`w-4 h-4 ${busy ? 'animate-spin' : ''}`}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
          <path d="M21 3v5h-5" />
          <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
          <path d="M3 21v-5h5" />
        </svg>
      </button>
    </div>
  )
}
