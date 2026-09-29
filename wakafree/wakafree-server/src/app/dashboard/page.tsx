import { supabase } from '@/lib/supabase'
import { formatSeconds, compactNumber } from '@/lib/compute'
import { currentLocalDate } from '@/lib/timezone'
import ActivityChart from '@/components/ActivityChart'
import BreakdownPie from '@/components/BreakdownPie'
import { AICodingChart, WeekdaysChart, TodayGauge, AiHumanByDayChart } from '@/components/DashboardCharts'
import TimelineCard from '@/components/TimelineCard'
import { mapTimelineRow, normalizeMachineName } from '@/lib/timelineData'
import { ProjectsDailyChart, CategoriesDailyChart, type DailyStackRow } from '@/components/StackedDaily'
import RefreshButton from '@/components/RefreshButton'
import RangePicker from '@/components/RangePicker'
import { RANGE_OPTIONS } from '@/lib/ranges'

export const dynamic = 'force-dynamic'

interface WakaEntry {
  name: string
  total_seconds: number
  percent?: number
}

interface WakaDay {
  grand_total: {
    total_seconds: number
    text: string
    ai_agent_total_cost?: number
    ai_model_total_cost?: number
    ai_input_tokens?: number
    ai_cached_input_tokens?: number
    ai_output_tokens?: number
    ai_prompt_events_total?: number
    ai_sessions?: number
    ai_agent_breakdown?: Array<{ name: string; cost?: number; lines?: number }>
    ai_model_breakdown?: Array<{ name: string; cost?: number; lines?: number }>
    ai_additions?: number
    ai_deletions?: number
    human_additions?: number
    human_deletions?: number
  }
  projects: WakaEntry[]
  languages: WakaEntry[]
  editors: WakaEntry[]
  categories: WakaEntry[]
  machines: WakaEntry[]
  operating_systems?: WakaEntry[]
  dependencies?: WakaEntry[]
  range: { date: string; text: string }
}

interface DailyRow {
  date: string
  data: WakaDay
}

// AI fields carried on each per-project daily entry
interface AIProjectEntry {
  ai_input_tokens?: number
  ai_output_tokens?: number
  ai_prompt_events_total?: number
  ai_sessions?: number
  ai_agent_breakdown?: Array<{ name: string; cost: number; lines?: number }>
}


function aggregateEntries(rows: DailyRow[], key: keyof WakaDay, limit: number) {
  const map = new Map<string, number>()
  for (const row of rows) {
    const entries = row.data[key] as WakaEntry[] | undefined
    if (!entries) continue
    for (const e of entries) {
      map.set(e.name, (map.get(e.name) ?? 0) + e.total_seconds)
    }
  }
  return Array.from(map.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([name, seconds]) => ({ name, seconds }))
}


interface DurationsWrapper {
  start: string
  end: string
  timezone?: string
  data: Array<{ time: number; duration: number; project?: string | null; category?: string | null }>
}

interface MetaRow {
  key: string
  data: unknown
}

async function getData(rangeDays: number) {
  // The heavy JSONB timeline columns (durations, durations_slices) are only
  // needed for the most recent day — fetching them for all 365 rows made every
  // page load pull tens of MB from Supabase. Fetch summaries for the year and
  // timelines for just the last few days.
  const [{ data: rows, error }, { data: metaRows }, { data: timelineRows }] = await Promise.all([
    supabase
      .from('waka_daily')
      .select('date, data')
      .order('date', { ascending: false })
      .limit(365),
    supabase.from('waka_meta').select('key, data'),
    supabase
      .from('waka_daily')
      .select('date, data, durations, durations_category, durations_slices')
      .order('date', { ascending: false })
      .limit(3),
  ])

  if (error || !rows || rows.length === 0) return null

  const todayLocal = currentLocalDate()
  // Drop any stale rows dated in the future relative to our dedicated timezone
  // (e.g. a UTC-rolled "tomorrow" row created before the timezone fix).
  const typed = (rows as DailyRow[]).filter((r) => r.date <= todayLocal)
  if (typed.length === 0) return null

  const typedTimelines = (
    (timelineRows ?? []) as Array<
      DailyRow & {
        durations: DurationsWrapper | null
        durations_category: DurationsWrapper | null
        durations_slices: Record<string, DurationsWrapper | null> | null
      }
    >
  ).filter((r) => r.date <= todayLocal)

  const meta = new Map<string, unknown>()
  for (const m of (metaRows ?? []) as MetaRow[]) meta.set(m.key, m.data)

  const allTime = meta.get('all_time') as
    | { text?: string; daily_average?: number; range?: { start_date?: string } }
    | undefined
  const stats7 = meta.get('stats_last_7_days') as
    | { human_readable_total?: string; human_readable_daily_average?: string }
    | undefined
  const goalsRaw = meta.get('goals')
  const goalsList = Array.isArray(goalsRaw) ? goalsRaw : []

  // Most recent day that actually has a timeline.
  const latestWithTimeline = typedTimelines.find(
    (r) => r.durations && Array.isArray(r.durations.data) && r.durations.data.length > 0
  )

  const latest = typed[0]
  const rangeRows = typed.slice(0, rangeDays)

  // Current day = the row for today in our dedicated timezone (0 if none yet).
  const todayRow = typed.find((r) => r.date === todayLocal)
  const currentDayText = todayRow?.data.grand_total?.text ?? '0 secs'

  // Most active = the day with the most tracked time across all history.
  const mostActive = rangeRows.reduce((best, r) =>
    (r.data.grand_total?.total_seconds ?? 0) > (best.data.grand_total?.total_seconds ?? 0) ? r : best
  )
  const mostActiveLabel = mostActive.data.range?.text ?? mostActive.date

  const allTimeSeconds = typed.reduce((s, r) => s + (r.data.grand_total?.total_seconds ?? 0), 0)
  const rangeSeconds = rangeRows.reduce((s, r) => s + (r.data.grand_total?.total_seconds ?? 0), 0)

  // Average of the 6 days before today — today is excluded so an in-progress
  // day doesn't drag its own comparison baseline down.
  const priorRows = typed.filter((r) => r.date !== todayLocal).slice(0, 6)
  const priorAvgSeconds = priorRows.length
    ? Math.round(priorRows.reduce((s, r) => s + (r.data.grand_total?.total_seconds ?? 0), 0) / priorRows.length)
    : 0

  const totalAiCost = rangeRows.reduce(
    (s, r) => s + (r.data.grand_total?.ai_model_total_cost ?? r.data.grand_total?.ai_agent_total_cost ?? 0),
    0
  )
  // WakaTime's line-change totals include both additions and deletions.
  const totalAiLines = rangeRows.reduce(
    (s, r) => s + (r.data.grand_total?.ai_additions ?? 0) + (r.data.grand_total?.ai_deletions ?? 0),
    0
  )
  const totalHumanLines = rangeRows.reduce(
    (s, r) => s + (r.data.grand_total?.human_additions ?? 0) + (r.data.grand_total?.human_deletions ?? 0),
    0
  )

  // Daily activity chart follows the selected range
  const dailyActivity = [...rangeRows]
    .reverse()
    .map((r) => ({ date: r.date, seconds: r.data.grand_total?.total_seconds ?? 0 }))

  // AI vs Human by day over the range (deletions negative, below the axis)
  const aiHumanDaily = [...rangeRows]
    .reverse()
    .map((r) => ({
      date: r.date.slice(5),
      humanAdd: r.data.grand_total?.human_additions ?? 0,
      humanDel: -(r.data.grand_total?.human_deletions ?? 0),
      aiAdd: r.data.grand_total?.ai_additions ?? 0,
      aiDel: -(r.data.grand_total?.ai_deletions ?? 0),
    }))

  const weekdayData = buildWeekdayData(rangeRows)

  // Range daily stacks: per-project and per-category
  const stackKeys = (key: 'projects' | 'categories', limit: number) =>
    aggregateEntries(rangeRows, key, limit).map((e) => e.name)
  // Anything outside the top names lands in Miscellaneous so day totals stay honest.
  const buildDailyStacks = (key: 'projects' | 'categories', names: string[]): DailyStackRow[] =>
    [...rangeRows].reverse().map((r) => {
      const row: DailyStackRow = { date: r.date.slice(5) }
      for (const name of names) row[name] = 0
      let misc = 0
      for (const e of (r.data[key] ?? []) as WakaEntry[]) {
        if (names.includes(e.name)) row[e.name] = e.total_seconds
        else misc += e.total_seconds
      }
      row['Miscellaneous'] = misc
      return row
    })
  const projectStackKeys = [...stackKeys('projects', 6), 'Miscellaneous']
  const categoryStackKeys = [...stackKeys('categories', 6), 'Miscellaneous']

  // Range AI totals + per-agent/model lines. WakaTime's daily grand_total is
  // authoritative; project entries are retained only for older payloads.
  let aiInputTokens = 0
  let aiCachedInputTokens = 0
  let aiOutputTokens = 0
  let aiPrompts = 0
  let aiSessions = 0
  const agentLines = new Map<string, number>()
  for (const r of rangeRows) {
    const projects = (r.data.projects ?? []) as unknown as AIProjectEntry[]
    const grandTotal = r.data.grand_total
    const projectTotals = projects.reduce(
      (totals, p) => ({
        input: totals.input + (p.ai_input_tokens ?? 0),
        cached: totals.cached + 0,
        output: totals.output + (p.ai_output_tokens ?? 0),
        prompts: totals.prompts + (p.ai_prompt_events_total ?? 0),
        sessions: totals.sessions + (p.ai_sessions ?? 0),
      }),
      { input: 0, cached: 0, output: 0, prompts: 0, sessions: 0 }
    )
    aiInputTokens += grandTotal.ai_input_tokens ?? projectTotals.input
    aiCachedInputTokens += grandTotal.ai_cached_input_tokens ?? projectTotals.cached
    aiOutputTokens += grandTotal.ai_output_tokens ?? projectTotals.output
    aiPrompts += grandTotal.ai_prompt_events_total ?? projectTotals.prompts
    aiSessions += grandTotal.ai_sessions ?? projectTotals.sessions

    const dailyAgents = grandTotal.ai_model_breakdown?.length
      ? grandTotal.ai_model_breakdown
      : grandTotal.ai_agent_breakdown?.length
        ? grandTotal.ai_agent_breakdown
        : projects.flatMap((p) => p.ai_agent_breakdown ?? [])
    if (dailyAgents?.length) {
      for (const a of dailyAgents) {
        const name = a.name.replace(/-/g, ' ')
        agentLines.set(name, (agentLines.get(name) ?? 0) + (a.lines ?? 0))
      }
    } else {
      for (const p of projects) for (const a of p.ai_agent_breakdown ?? []) {
        // WakaTime reports agent slugs like "Claude-Code" — display with spaces.
        const name = a.name.replace(/-/g, ' ')
        agentLines.set(name, (agentLines.get(name) ?? 0) + (a.lines ?? 0))
      }
    }
  }
  const topAgents = Array.from(agentLines.entries())
    .filter(([, lines]) => lines > 0)
    .map(([name, lines]) => ({ name, seconds: lines }))
    .sort((a, b) => b.seconds - a.seconds)
    .slice(0, 8)

  return {
    latestDate: latest.data.range.text,
    latestSeconds: latest.data.grand_total?.total_seconds ?? 0,
    latestText: latest.data.grand_total?.text ?? '0 min',
    latestAiCost: latest.data.grand_total?.ai_agent_total_cost ?? 0,
    currentDayText,
    mostActiveLabel,
    rangeDays,
    rangeSeconds,
    allTimeSeconds,
    totalAiCost,
    daysStored: typed.length,
    dailyActivity,
    aiHumanDaily,
    weekdayData,
    totalAiLines,
    totalHumanLines,
    todaySeconds: todayRow?.data.grand_total?.total_seconds ?? 0,
    priorAvgSeconds,
    aiInputTokens,
    aiCachedInputTokens,
    aiOutputTokens,
    aiPrompts,
    aiSessions,
    topAgents,
    projectStackKeys,
    projectDaily: buildDailyStacks('projects', projectStackKeys),
    categoryStackKeys,
    categoryDaily: buildDailyStacks('categories', categoryStackKeys),
    categoryTotals7: aggregateEntries(rangeRows, 'categories', 6),
    // aggregated over the selected range
    topProjects: aggregateEntries(rangeRows, 'projects', 8),
    topLanguages: aggregateEntries(rangeRows, 'languages', 8),
    topEditors: aggregateEntries(rangeRows, 'editors', 6),
    topCategories: aggregateEntries(rangeRows, 'categories', 4),
    topMachines: (() => {
      // Merge hostnames that map to the same friendly name (see MACHINE_ALIASES).
      const merged = new Map<string, number>()
      for (const m of aggregateEntries(rangeRows, 'machines', 50)) {
        if (m.name === 'Mac.lan') continue
        const name = normalizeMachineName(m.name)
        merged.set(name, (merged.get(name) ?? 0) + m.seconds)
      }
      return Array.from(merged.entries())
        .map(([name, seconds]) => ({ name, seconds }))
        .sort((a, b) => b.seconds - a.seconds)
        .slice(0, 6)
    })(),
    topOSs: aggregateEntries(rangeRows, 'operating_systems', 6),
    // most recent day breakdown
    latestProjects: (latest.data.projects ?? []).slice(0, 8).map((p) => ({ name: p.name, seconds: p.total_seconds })),
    latestLanguages: (latest.data.languages ?? []).slice(0, 8).map((l) => ({ name: l.name, seconds: l.total_seconds })),
    latestEditors: (latest.data.editors ?? []).slice(0, 4).map((e) => ({ name: e.name, seconds: e.total_seconds })),
    latestCategories: (latest.data.categories ?? []).slice(0, 4).map((c) => ({ name: c.name, seconds: c.total_seconds, percent: c.percent })),
    // account-level snapshots (from waka_meta)
    allTimeText: allTime?.text ?? null,
    allTimeDailyAvg: allTime?.daily_average ? formatSeconds(allTime.daily_average) : null,
    allTimeStart: allTime?.range?.start_date ?? null,
    stats7Total: stats7?.human_readable_total ?? null,
    stats7Avg: stats7?.human_readable_daily_average ?? null,
    goals: goalsList,
    // most recent day's timeline payload + navigable date bounds
    timelineInitial: latestWithTimeline
      ? mapTimelineRow(latestWithTimeline.date, latestWithTimeline)
      : null,
    timelineMinDate: typed[typed.length - 1].date,
    timelineMaxDate: typed[0].date,
  }
}

function buildWeekdayData(rows: DailyRow[]) {
  const weekdayMap = new Map<string, number>()
  const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

  for (const day of days) {
    weekdayMap.set(day, 0)
  }

  for (const row of rows) {
    const date = new Date(row.date + 'T00:00:00Z')
    const dayName = days[date.getUTCDay() === 0 ? 6 : date.getUTCDay() - 1]
    const current = weekdayMap.get(dayName) ?? 0
    weekdayMap.set(dayName, current + (row.data.grand_total?.total_seconds ?? 0))
  }

  return Array.from(weekdayMap.entries()).map(([day, seconds]) => ({
    day,
    hours: Math.round((seconds / 3600) * 10) / 10,
  }))
}

function AIStat({
  label,
  value,
  sub,
  bar,
  barColor,
}: {
  label: string
  value: string
  sub?: string
  bar?: number
  barColor?: string
}) {
  return (
    <div className="border border-line rounded-lg p-4">
      <p className="font-mono text-[11px] font-medium uppercase tracking-[0.08em] text-outline mb-1.5">{label}</p>
      <p className="text-xl font-semibold text-onsurface">{value}</p>
      {bar !== undefined && (
        <div className="mt-2 h-1 rounded-full bg-container-high">
          <div className="h-1 rounded-full" style={{ width: `${bar}%`, backgroundColor: barColor }} />
        </div>
      )}
      {sub && <p className="text-outline text-xs mt-1.5">{sub}</p>}
    </div>
  )
}

export default async function DashboardPage({
  searchParams,
}: {
  searchParams: Promise<{ range?: string }>
}) {
  const { range } = await searchParams
  const rangeDays = RANGE_OPTIONS.includes(Number(range)) ? Number(range) : 7
  const data = await getData(rangeDays)
  const aiPct =
    data && data.totalAiLines > 0
      ? Math.round((data.totalAiLines / (data.totalAiLines + data.totalHumanLines)) * 100)
      : 0

  if (!data) {
    return (
      <main className="min-h-screen bg-surface p-6 max-w-7xl mx-auto flex items-center justify-center">
        <div className="text-center">
          <div className="w-10 h-10 bg-primary-dim rounded-xl flex items-center justify-center text-lg font-bold mx-auto mb-4">W</div>
          <h1 className="text-onsurface text-xl font-semibold mb-2">No data yet</h1>
          <p className="text-outline text-sm max-w-sm">
            Trigger your first sync by visiting{' '}
            <code className="bg-container-high px-1 rounded font-mono text-[13px] text-primary">/api/wakatime/sync</code>.
            <br />
            To backfill history:{' '}
            <code className="bg-container-high px-1 rounded font-mono text-[13px] text-primary">/api/wakatime/sync?backfill=30</code>
          </p>
        </div>
      </main>
    )
  }

  return (
    <main className="p-4 sm:p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-8 mt-2">
        <h1 className="text-2xl font-semibold tracking-tight text-onsurface">Activity Overview</h1>
        <div className="flex items-center gap-3">
          <RefreshButton />
          <RangePicker current={data.rangeDays} />
        </div>
      </div>

      {/* Top summary bar: 7-day hero + inset stat boxes */}
      <div className="mb-8 flex flex-col justify-between gap-6 rounded-lg border border-line bg-container-low p-6 lg:flex-row lg:items-center">
        <div>
          <p className="font-mono text-[11px] font-medium uppercase tracking-[0.08em] text-outline mb-2">Over the Last {data.rangeDays} Days</p>
          <p className="text-[44px] leading-[52px] font-semibold tracking-[-0.02em] text-onsurface">{formatSeconds(data.rangeSeconds)}</p>
        </div>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 lg:w-[60%]">
          <div className="rounded-lg border border-outline-variant/60 bg-container p-6">
            <p className="font-mono text-[11px] font-medium uppercase tracking-[0.08em] text-outline mb-1.5">Current Day</p>
            <p className="text-xl font-semibold text-onsurface">{data.currentDayText}</p>
            <p className="text-outline text-xs mt-1">Today</p>
          </div>
          <div className="rounded-lg border border-outline-variant/60 bg-container p-6">
            <p className="font-mono text-[11px] font-medium uppercase tracking-[0.08em] text-outline mb-1.5">Daily Average</p>
            <p className="text-xl font-semibold text-onsurface">{formatSeconds(Math.round(data.rangeSeconds / data.rangeDays))}</p>
            <p className="text-outline text-xs mt-1">over {data.rangeDays} days</p>
          </div>
          <div className="rounded-lg border border-outline-variant/60 bg-container p-6">
            <p className="font-mono text-[11px] font-medium uppercase tracking-[0.08em] text-outline mb-1.5">Most Active</p>
            <p className="text-xl font-semibold text-onsurface">{data.mostActiveLabel}</p>
            <p className="text-outline text-xs mt-1">top day</p>
          </div>
        </div>
      </div>

      {/* AI Coding Section */}
      <div id="ai-metrics" className="scroll-mt-24 bg-container-low border border-line rounded-lg p-6 mb-8">
        <div className="mb-6 flex items-center justify-between">
          <h2 className="text-xl font-medium tracking-tight text-onsurface">AI Coding</h2>
        </div>
        <div className="flex flex-col items-center gap-6 xl:flex-row">
          <div className="shrink-0">
            <AICodingChart aiPercent={aiPct} size={150} />
          </div>
          <div className="grid w-full flex-1 grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-6">
            <AIStat
              label="AI lines"
              value={compactNumber(data.totalAiLines)}
              bar={aiPct}
              barColor="#b48ead"
            />
            <AIStat
              label="Human lines"
              value={compactNumber(data.totalHumanLines)}
              bar={100 - aiPct}
              barColor="#94a3b8"
            />
            <AIStat
              label="Tokens"
              value={compactNumber(data.aiInputTokens + data.aiCachedInputTokens + data.aiOutputTokens)}
              sub={`${compactNumber(data.aiInputTokens)} uncached · ${compactNumber(data.aiCachedInputTokens)} cached · ${compactNumber(data.aiOutputTokens)} out`}
            />
            <AIStat
              label="Cost"
              value={`$${data.totalAiCost.toFixed(2)}`}
              sub="estimated agent spend"
            />
            <AIStat
              label="Prompts"
              value={data.aiPrompts.toLocaleString()}
              sub="AI prompt events"
            />
            <AIStat
              label="Sessions"
              value={data.aiSessions.toLocaleString()}
              sub="AI coding sessions"
            />
          </div>
        </div>
      </div>

      {/* Daily stacks: projects + categories — hidden on phones */}
      <div className="hidden md:grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="bg-container-low border border-line rounded-lg p-6">
          <h3 className="text-base font-medium text-onsurface mb-4">Projects</h3>
          <ProjectsDailyChart data={data.projectDaily} seriesKeys={data.projectStackKeys} />
        </div>

        <div className="bg-container-low border border-line rounded-lg p-6">
          <h3 className="text-base font-medium text-onsurface mb-4">Categories</h3>
          <CategoriesDailyChart
            data={data.categoryDaily}
            seriesKeys={data.categoryStackKeys}
            totals={data.categoryTotals7}
          />
        </div>
      </div>

      {/* Daily timelines: by project + segmentable, both date-navigable */}
      {data.timelineInitial && (
        <div id="timeline" className="scroll-mt-24 hidden md:grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-container-low border border-line rounded-lg p-6">
            <TimelineCard
              variant="projects"
              initial={data.timelineInitial}
              minDate={data.timelineMinDate}
              maxDate={data.timelineMaxDate}
            />
          </div>

          <div className="bg-container-low border border-line rounded-lg p-6">
            <TimelineCard
              variant="segmented"
              initial={data.timelineInitial}
              minDate={data.timelineMinDate}
              maxDate={data.timelineMaxDate}
            />
          </div>
        </div>
      )}

      {/* Charts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {/* Projects */}
        <div className="bg-container-low border border-line rounded-lg p-6">
          <h3 className="text-base font-medium text-onsurface text-center mb-4">Projects</h3>
          <div className="space-y-3">
            {data.topProjects.slice(0, 5).map((project, i) => (
              <div key={project.name}>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium text-onsurface">{project.name}</span>
                  <span className="font-mono text-[11px] tracking-[0.02em] text-outline">{formatSeconds(project.seconds)}</span>
                </div>
                <div className="h-1.5 bg-container-high rounded-full">
                  <div
                    className="h-1.5 rounded-full"
                    style={{
                      width: `${Math.round((project.seconds / data.topProjects[0].seconds) * 100)}%`,
                      backgroundColor: i === 0 ? '#b48ead' : '#64748b',
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* AI Agents */}
        <div className="bg-container-low border border-line rounded-lg p-6">
          <h3 className="text-base font-medium text-onsurface text-center mb-4">Models</h3>
          {data.topAgents.length > 0 ? (
            <BreakdownPie data={data.topAgents} valueKind="lines" />
          ) : (
            <p className="text-outline text-sm text-center py-10">No agent data yet.</p>
          )}
        </div>
      </div>

      <div id="statistics" className="scroll-mt-24 grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {/* Categories */}
        <div className="bg-container-low border border-line rounded-lg p-6">
          <h3 className="text-base font-medium text-onsurface text-center mb-4">Categories</h3>
          <BreakdownPie data={data.topCategories} totalSeconds={data.rangeSeconds} />
        </div>

        {/* Editors */}
        <div className="bg-container-low border border-line rounded-lg p-6">
          <h3 className="text-base font-medium text-onsurface text-center mb-4">Editors</h3>
          <BreakdownPie data={data.topEditors} totalSeconds={data.rangeSeconds} />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {/* Languages */}
        <div className="bg-container-low border border-line rounded-lg p-6">
          <h3 className="text-base font-medium text-onsurface text-center mb-4">Languages</h3>
          <BreakdownPie data={data.topLanguages} totalSeconds={data.rangeSeconds} />
        </div>

        {/* Operating Systems */}
        <div className="bg-container-low border border-line rounded-lg p-6">
          <h3 className="text-base font-medium text-onsurface text-center mb-4">Operating Systems</h3>
          <BreakdownPie data={data.topOSs} totalSeconds={data.rangeSeconds} />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {/* Machines */}
        <div className="bg-container-low border border-line rounded-lg p-6">
          <h3 className="text-base font-medium text-onsurface text-center mb-4">Machines</h3>
          <BreakdownPie data={data.topMachines} totalSeconds={data.rangeSeconds} />
        </div>

        {/* Today vs daily average gauge */}
        <div className="bg-container-low border border-line rounded-lg p-6">
          <TodayGauge
            todaySeconds={data.todaySeconds}
            compareSeconds={data.priorAvgSeconds}
            todayText={data.currentDayText}
            avgText={formatSeconds(Math.round(data.rangeSeconds / data.rangeDays))}
            mostActiveLabel={data.mostActiveLabel}
          />
        </div>
      </div>

      {/* AI vs Human by day + Weekdays — hidden on phones */}
      <div className="hidden md:grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="bg-container-low border border-line rounded-lg p-6">
          <h3 className="text-base font-medium text-onsurface text-center mb-4">AI vs Human by Day</h3>
          <AiHumanByDayChart data={data.aiHumanDaily} />
        </div>

        <div className="bg-container-low border border-line rounded-lg p-6">
          <h3 className="text-base font-medium text-onsurface text-center mb-4">Weekdays</h3>
          <WeekdaysChart data={data.weekdayData} />
        </div>
      </div>

      {/* Full activity history — hidden on phones */}
      <div className="hidden md:block bg-container-low border border-line rounded-lg p-6 mb-8">
        <h3 className="text-base font-medium text-onsurface text-center mb-4">Daily Activity</h3>
        <ActivityChart data={data.dailyActivity} />
      </div>

    </main>
  )
}
