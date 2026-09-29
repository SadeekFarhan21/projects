import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'
import { validateApiKey } from '@/lib/auth'

export async function POST(request: NextRequest) {
  const userId = await validateApiKey(request)
  if (!userId) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const body = await request.json()
  const arr = Array.isArray(body) ? body : [body]
  const machine = request.headers.get('X-Machine-Name') ?? null

  const rows = arr
    .filter((hb) => hb.entity)
    .map((hb: Record<string, unknown>) => ({
      user_id: userId,
      entity: hb.entity as string,
      type: (hb.type as string) ?? 'file',
      time:
        typeof hb.time === 'string'
          ? parseFloat(hb.time as string)
          : (hb.time as number),
      project: (hb.project as string) ?? null,
      language: (hb.language as string) ?? null,
      branch: (hb.branch as string) ?? null,
      is_write: (hb.is_write as boolean) ?? false,
      editor: (hb.editor as string) ?? null,
      operating_system: (hb.operating_system as string) ?? null,
      machine: machine ?? (hb.machine as string) ?? null,
      category: (hb.category as string) ?? 'coding',
      lines: (hb.lines as number) ?? null,
      lineno: (hb.lineno as number) ?? null,
      cursorpos: (hb.cursorpos as number) ?? null,
    }))

  if (rows.length === 0) {
    return NextResponse.json({ responses: [] }, { status: 201 })
  }

  const { data, error } = await supabase
    .from('heartbeats')
    .insert(rows)
    .select()

  if (error) {
    console.error('Heartbeat insert error:', error)
    return NextResponse.json({ error: error.message }, { status: 500 })
  }

  const responses = (data ?? []).map((hb: Record<string, unknown>) => [
    { data: { id: hb.id, entity: hb.entity, time: hb.time } },
    201,
  ])

  return NextResponse.json({ responses }, { status: 201 })
}
