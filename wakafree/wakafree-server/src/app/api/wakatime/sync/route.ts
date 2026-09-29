import { NextRequest, NextResponse } from 'next/server'
import { checkSecret } from '@/lib/apiAuth'
import { resolveRange, syncRange } from '@/lib/wakatime'

// GET /api/wakatime/sync   (requires SYNC_SECRET / CRON_SECRET)
// ?date=YYYY-MM-DD       → sync a specific date
// ?days=N                → refresh the last N days INCLUDING today (max 365)
// ?backfill=N            → sync the N days ending yesterday (max 365)
// ?meta=0                → skip the account-level snapshot refresh
// no params              → sync yesterday
export async function GET(req: NextRequest) {
  const denied = checkSecret(req)
  if (denied) return denied

  const { searchParams } = new URL(req.url)
  const { start, end } = resolveRange({
    date: searchParams.get('date'),
    days: searchParams.get('days') ? Number(searchParams.get('days')) : null,
    backfill: searchParams.get('backfill') ? Number(searchParams.get('backfill')) : null,
  })
  const includeMeta = searchParams.get('meta') !== '0'

  try {
    const result = await syncRange(start, end, includeMeta)
    if (result.error) {
      return NextResponse.json({ error: result.error }, { status: result.status ?? 500 })
    }
    return NextResponse.json(result)
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
