import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'
import { validateApiKey } from '@/lib/auth'

export async function GET(request: NextRequest) {
  const userId = await validateApiKey(request)
  if (!userId) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const { data, error } = await supabase
    .from('users')
    .select('id, email, username, display_name, timezone, created_at, modified_at')
    .eq('id', userId)
    .single()

  if (error || !data) {
    return NextResponse.json({ error: 'User not found' }, { status: 404 })
  }

  return NextResponse.json({
    data: {
      id: data.id,
      email: data.email,
      display_name: data.display_name ?? data.username ?? "Farhan's WakaTime User",
      username: data.username,
      timezone: data.timezone ?? 'UTC',
      plan: 'free',
      created_at: data.created_at,
      modified_at: data.modified_at,
      has_premium_features: false,
      is_email_confirmed: true,
    },
  })
}
