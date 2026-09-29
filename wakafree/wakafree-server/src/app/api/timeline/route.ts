import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'
import { mapTimelineRow, type TimelineRow } from '@/lib/timelineData'

export const dynamic = 'force-dynamic'

// GET /api/timeline?date=YYYY-MM-DD
// Timeline payload (project blocks + segment slices) for one day, used by the
// dashboard's timeline cards when navigating between dates.
export async function GET(req: NextRequest) {
  const date = new URL(req.url).searchParams.get('date') ?? ''
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    return NextResponse.json({ error: 'Invalid date' }, { status: 400 })
  }

  const { data: rows, error } = await supabase
    .from('waka_daily')
    .select('data, durations, durations_category, durations_slices')
    .eq('date', date)
    .limit(1)

  if (error) return NextResponse.json({ error: error.message }, { status: 500 })

  const row = (rows?.[0] ?? null) as TimelineRow | null
  return NextResponse.json(mapTimelineRow(date, row))
}
