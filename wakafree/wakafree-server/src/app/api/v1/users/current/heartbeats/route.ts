import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'
import { validateApiKey } from '@/lib/auth'

export async function POST(request: NextRequest) {
  const userId = await validateApiKey(request)
  if (!userId) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const body = await request.json()

  const { data, error } = await supabase
    .from('heartbeats')
    .insert({
      user_id: userId,
      entity: body.entity,
      type: body.type ?? 'file',
      time: typeof body.time === 'string' ? parseFloat(body.time) : body.time,
      project: body.project ?? null,
      language: body.language ?? null,
      branch: body.branch ?? null,
      is_write: body.is_write ?? false,
      editor: body.editor ?? null,
      operating_system: body.operating_system ?? null,
      machine: request.headers.get('X-Machine-Name') ?? body.machine ?? null,
      category: body.category ?? 'coding',
      lines: body.lines ?? null,
      lineno: body.lineno ?? null,
      cursorpos: body.cursorpos ?? null,
    })
    .select()
    .single()

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }

  return NextResponse.json({ data }, { status: 201 })
}
