import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'
import { validateApiKey } from '@/lib/auth'
import { computeTotalSeconds, computeByKey, formatSeconds } from '@/lib/compute'

export async function GET(request: NextRequest) {
  const userId = await validateApiKey(request)
  if (!userId) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const { searchParams } = new URL(request.url)
  const start = searchParams.get('start')
  const end = searchParams.get('end')

  if (!start || !end) {
    return NextResponse.json(
      { error: 'start and end query params are required' },
      { status: 400 }
    )
  }

  const startTs = new Date(start).getTime() / 1000
  const endTs = new Date(`${end}T23:59:59Z`).getTime() / 1000

  const { data, error } = await supabase
    .from('heartbeats')
    .select('time, project, language, entity')
    .eq('user_id', userId)
    .gte('time', startTs)
    .lte('time', endTs)
    .order('time')

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }

  const heartbeats = data ?? []
  const totalSeconds = computeTotalSeconds(heartbeats)

  return NextResponse.json({
    data: [
      {
        grand_total: {
          total_seconds: totalSeconds,
          hours: Math.floor(totalSeconds / 3600),
          minutes: Math.floor((totalSeconds % 3600) / 60),
          text: formatSeconds(totalSeconds),
        },
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
        range: { start, end },
      },
    ],
  })
}
