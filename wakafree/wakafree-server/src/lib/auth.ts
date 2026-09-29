import { supabase } from './supabase'

export async function validateApiKey(request: Request): Promise<string | null> {
  const authHeader = request.headers.get('Authorization')
  if (!authHeader) return null

  let apiKey: string | null = null

  if (authHeader.startsWith('Basic ')) {
    try {
      apiKey = Buffer.from(authHeader.slice(6), 'base64').toString('utf-8')
    } catch {
      return null
    }
  } else if (authHeader.startsWith('Bearer ')) {
    apiKey = authHeader.slice(7)
  }

  if (!apiKey) return null

  const { data } = await supabase
    .from('users')
    .select('id')
    .eq('api_key', apiKey)
    .single()

  return data?.id ?? null
}
