import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'
import { validateApiKey } from '@/lib/auth'
import { computeTotalSeconds, computeByKey, formatSeconds } from '@/lib/compute'

const RANGE_DAYS: Record<string, number> = {
  last_7_days: 7,
  last_30_days: 30,
  last_6_months: 180,
  last_year: 365,
  all_time: 36500,
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ range: string }> }
) {
  const userId = await validateApiKey(request)
  if (!userId) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const { range } = await params
  const days = RANGE_DAYS[range] ?? 7
  const startTs = Date.now() / 1000 - days * 86400

  const { data, error } = await supabase
    .from('heartbeats')
    .select('time, project, language, entity')
    .eq('user_id', userId)
    .gte('time', startTs)
    .order('time')

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }

  const heartbeats = data ?? []
  const totalSeconds = computeTotalSeconds(heartbeats)

  return NextResponse.json({
    data: {
      total_seconds: totalSeconds,
      human_readable_total: formatSeconds(totalSeconds),
      daily_average: Math.round(totalSeconds / days),
      human_readable_daily_average: formatSeconds(Math.round(totalSeconds / days)),
      projects: computeByKey(heartbeats, 'project').map((p) => ({
        name: p.name,
        total_seconds: p.seconds,
        text: formatSeconds(p.seconds),
      })),
      languages: computeByKey(heartbeats, 'language').map((l) => ({
        name: l.name,
        total_seconds: l.seconds,
        text: formatSeconds(l.seconds),
      })),
      range,
    },
  })
}
