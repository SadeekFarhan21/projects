import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'
import { validateApiKey } from '@/lib/auth'

export async function GET(request: NextRequest) {
  const userId = await validateApiKey(request)
  if (!userId) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const { data, error } = await supabase
    .from('heartbeats')
    .select('project')
    .eq('user_id', userId)
    .not('project', 'is', null)

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }

  const projects = [
    ...new Set((data ?? []).map((r: { project: string }) => r.project)),
  ]
    .filter(Boolean)
    .map((name) => ({ id: name, name }))

  return NextResponse.json({ data: projects })
}
