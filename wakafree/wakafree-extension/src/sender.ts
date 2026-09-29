import { getApiUrl, getApiKey } from './config'

export interface HeartbeatPayload {
  entity: string
  type: 'file'
  time: number
  project?: string
  language?: string
  branch?: string
  is_write: boolean
  editor: string
  operating_system: string
  machine: string
  lines?: number
  lineno?: number
  cursorpos?: number
}

export async function sendHeartbeats(heartbeats: HeartbeatPayload[]): Promise<void> {
  const apiKey = getApiKey()
  if (!apiKey) {
    throw new Error('No API key configured — set wakafree.apiKey in VS Code settings')
  }

  const apiUrl = getApiUrl()
  const encoded = Buffer.from(apiKey).toString('base64')

  const res = await fetch(`${apiUrl}/users/current/heartbeats.bulk`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Basic ${encoded}`,
      'X-Machine-Name': heartbeats[0]?.machine ?? '',
    },
    body: JSON.stringify(heartbeats),
  })

  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`Server ${res.status}: ${text.slice(0, 120)}`)
  }
}

export async function fetchTodayStats(): Promise<string> {
  const apiKey = getApiKey()
  if (!apiKey) return 'No API key set'

  const apiUrl = getApiUrl()
  const encoded = Buffer.from(apiKey).toString('base64')
  const today = new Date().toISOString().split('T')[0]

  const res = await fetch(
    `${apiUrl}/users/current/summaries?start=${today}&end=${today}`,
    { headers: { Authorization: `Basic ${encoded}` } }
  )
  if (!res.ok) return `Server error ${res.status}`

  const json = (await res.json()) as {
    data: Array<{ grand_total: { text: string } }>
  }
  return json.data?.[0]?.grand_total?.text ?? '0 min'
}
