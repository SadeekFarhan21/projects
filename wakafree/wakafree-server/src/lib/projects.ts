export interface WakaProject {
  name: string
  total_seconds: number
  text?: string
  ai_sessions?: number
  ai_additions?: number
  ai_deletions?: number
  human_additions?: number
  human_deletions?: number
  ai_input_tokens?: number
  ai_output_tokens?: number
  ai_agent_total_cost?: number
  ai_prompt_length_avg?: number
  ai_prompt_events_total?: number
  ai_prompt_events_avg_per_session?: number
  ai_agent_breakdown?: Array<{ name: string; cost: number; lines: number }>
}

export interface ProjectDetail {
  name: string
  seconds: number
  aiChanges: number
  humanChanges: number
  aiPercent: number
  humanPercent: number
  aiPrompts: number
  promptAvgChars: number
  aiSessions: number
  avgPromptsPerSession: number
  inputTokens: number
  outputTokens: number
  aiSpend: number
  agentBreakdown: Array<{ name: string; cost: number }>
}

interface RowWithProjects {
  data: { projects?: WakaProject[] }
}

// Aggregate full per-project AI/coding detail across the given rows.
export function aggregateProjectDetails(rows: RowWithProjects[], limit: number): ProjectDetail[] {
  const map = new Map<
    string,
    Required<Pick<WakaProject, 'name'>> & {
      seconds: number
      ai_sessions: number
      ai_additions: number
      ai_deletions: number
      human_additions: number
      human_deletions: number
      ai_input_tokens: number
      ai_output_tokens: number
      ai_agent_total_cost: number
      ai_prompt_events_total: number
      agentCosts: Map<string, number>
    }
  >()

  for (const row of rows) {
    const projects = row.data.projects ?? []
    for (const p of projects) {
      const existing = map.get(p.name) ?? {
        name: p.name,
        seconds: 0,
        ai_sessions: 0,
        ai_additions: 0,
        ai_deletions: 0,
        human_additions: 0,
        human_deletions: 0,
        ai_input_tokens: 0,
        ai_output_tokens: 0,
        ai_agent_total_cost: 0,
        ai_prompt_events_total: 0,
        agentCosts: new Map<string, number>(),
      }

      existing.seconds += p.total_seconds ?? 0
      existing.ai_sessions += p.ai_sessions ?? 0
      existing.ai_additions += p.ai_additions ?? 0
      existing.ai_deletions += p.ai_deletions ?? 0
      existing.human_additions += p.human_additions ?? 0
      existing.human_deletions += p.human_deletions ?? 0
      existing.ai_input_tokens += p.ai_input_tokens ?? 0
      existing.ai_output_tokens += p.ai_output_tokens ?? 0
      existing.ai_agent_total_cost += p.ai_agent_total_cost ?? 0
      existing.ai_prompt_events_total += p.ai_prompt_events_total ?? 0

      for (const agent of p.ai_agent_breakdown ?? []) {
        existing.agentCosts.set(agent.name, (existing.agentCosts.get(agent.name) ?? 0) + agent.cost)
      }

      map.set(p.name, existing)
    }
  }

  return Array.from(map.values())
    .sort((a, b) => b.seconds - a.seconds)
    .slice(0, limit)
    .map((p) => {
      const aiChanges = p.ai_additions + p.ai_deletions
      const humanChanges = p.human_additions + p.human_deletions
      const totalChanges = aiChanges + humanChanges
      const aiPrompts = p.ai_prompt_events_total
      const aiSessions = p.ai_sessions
      return {
        name: p.name,
        seconds: p.seconds,
        aiChanges,
        humanChanges,
        aiPercent: totalChanges > 0 ? (aiChanges / totalChanges) * 100 : 0,
        humanPercent: totalChanges > 0 ? (humanChanges / totalChanges) * 100 : 0,
        aiPrompts,
        promptAvgChars: 0,
        aiSessions,
        avgPromptsPerSession: aiSessions > 0 ? Math.round(aiPrompts / aiSessions) : 0,
        inputTokens: p.ai_input_tokens,
        outputTokens: p.ai_output_tokens,
        aiSpend: p.ai_agent_total_cost,
        agentBreakdown: Array.from(p.agentCosts.entries())
          .map(([name, cost]) => ({ name, cost }))
          .sort((a, b) => b.cost - a.cost),
      }
    })
}
