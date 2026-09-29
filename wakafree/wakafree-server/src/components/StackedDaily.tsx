'use client'
import { useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

// Muted Nord-style family (matches BreakdownPie)
const PROJECT_COLORS = [
  '#b48ead', '#88c0d0', '#a3be8c', '#ebcb8b', '#d08770',
  '#8fbcbb', '#81a1c1', '#bf616a', '#d5b3cf', '#94a3b8',
]

// Fixed hues for the three WakaTime categories; extras fall back to the palette
const CATEGORY_COLORS: Record<string, string> = {
  'AI Coding': '#b48ead',
  Coding: '#88c0d0',
  'Writing Docs': '#a3be8c',
}

export interface DailyStackRow {
  date: string
  [series: string]: string | number
}

function fmtTime(s: number): string {
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  return h > 0 ? `${h}h ${m}m` : `${m}m`
}

const TOOLTIP_STYLE = {
  backgroundColor: '#000000',
  border: '1px solid #3d434e',
  borderRadius: '8px',
  color: '#e6e8ec',
  fontFamily: 'var(--font-mono), monospace',
  fontSize: '12px',
} as const

function colorFor(key: string, i: number, kind: 'project' | 'category') {
  if (key === 'Miscellaneous') return '#8fbcbb'
  if (kind === 'category') return CATEGORY_COLORS[key] ?? PROJECT_COLORS[i % PROJECT_COLORS.length]
  return PROJECT_COLORS[i % PROJECT_COLORS.length]
}

// Custom tooltip: hides zero-time series, sorts the rest longest-first.
function StackTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean
  payload?: Array<{ name?: string; value?: number; color?: string }>
  label?: string
}) {
  if (!active || !payload?.length) return null
  const items = payload
    .filter((p) => (p.value ?? 0) >= 60)
    .sort((a, b) => (b.value ?? 0) - (a.value ?? 0))
  if (items.length === 0) return null
  return (
    <div style={TOOLTIP_STYLE} className="px-3 py-2">
      <p className="mb-1 text-[#8b919d]">{label}</p>
      {items.map((p) => (
        <p key={p.name} className="flex items-center gap-1.5 text-[#e6e8ec]">
          <span
            className="inline-block h-2 w-2 rounded-[2px]"
            style={{ backgroundColor: p.color }}
          />
          {p.name} : {fmtTime(Number(p.value))}
        </p>
      ))}
    </div>
  )
}

function StackedBars({
  data,
  seriesKeys,
  kind,
}: {
  data: DailyStackRow[]
  seriesKeys: string[]
  kind: 'project' | 'category'
}) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 4, right: 4, left: -16, bottom: 0 }}>
        <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.06)" />
        <XAxis
          dataKey="date"
          tickLine={false}
          axisLine={{ stroke: '#3d434e' }}
          tick={{ fill: '#8b919d', fontSize: 11, fontFamily: 'var(--font-mono), monospace' }}
        />
        <YAxis
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => `${Math.round(Number(v) / 3600)}h`}
          tick={{ fill: '#8b919d', fontSize: 11, fontFamily: 'var(--font-mono), monospace' }}
        />
        <Tooltip
          content={<StackTooltip />}
          cursor={{ fill: 'rgba(255,255,255,0.04)' }}
        />
        {seriesKeys.map((key, i) => (
          <Bar
            key={key}
            dataKey={key}
            stackId="day"
            fill={colorFor(key, i, kind)}
            stroke="#12161d"
            strokeWidth={2}
            isAnimationActive={false}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}

function LegendRow({ keys, kind }: { keys: string[]; kind: 'project' | 'category' }) {
  return (
    <div className="mt-2 flex flex-wrap justify-center gap-x-4 gap-y-1">
      {keys.map((key, i) => (
        <span key={key} className="flex items-center gap-1.5">
          <span
            className="h-2 w-2 rounded-[2px]"
            style={{ backgroundColor: colorFor(key, i, kind) }}
          />
          <span className="font-mono text-[11px] text-onsurface-variant">{key}</span>
        </span>
      ))}
    </div>
  )
}

// Daily coding time stacked by project.
export function ProjectsDailyChart({
  data,
  seriesKeys,
}: {
  data: DailyStackRow[]
  seriesKeys: string[]
}) {
  return (
    <div>
      <StackedBars data={data} seriesKeys={seriesKeys} kind="project" />
      <LegendRow keys={seriesKeys} kind="project" />
    </div>
  )
}

// Total split bar + daily stacks by category.
export function CategoriesDailyChart({
  data,
  seriesKeys,
  totals,
}: {
  data: DailyStackRow[]
  seriesKeys: string[]
  totals: Array<{ name: string; seconds: number }>
}) {
  const [active, setActive] = useState<number | null>(null)
  const grand = totals.reduce((s, t) => s + t.seconds, 0)

  // Midpoint of the hovered segment, as a % of the bar width (for the tooltip)
  const midPercent = (i: number) => {
    let before = 0
    for (let k = 0; k < i; k++) before += totals[k].seconds
    return ((before + totals[i].seconds / 2) / grand) * 100
  }

  return (
    <div>
      {/* Aggregate split over the whole range */}
      {grand > 0 && (
        <div className="relative mb-4" onMouseLeave={() => setActive(null)}>
          <div className="flex h-5 w-full overflow-hidden rounded">
            {totals.map((t, i) => (
              <div
                key={t.name}
                onMouseEnter={() => setActive(i)}
                className="transition-opacity duration-150"
                style={{
                  width: `${(t.seconds / grand) * 100}%`,
                  backgroundColor: colorFor(t.name, i, 'category'),
                  opacity: active === null || active === i ? 1 : 0.35,
                }}
              />
            ))}
          </div>
          {active !== null && (
            <div
              className="pointer-events-none absolute top-full z-10 mt-1.5 -translate-x-1/2 whitespace-nowrap rounded-lg border border-outline-variant bg-black px-2.5 py-1.5 font-mono text-[12px] text-onsurface"
              style={{ left: `clamp(72px, ${midPercent(active)}%, calc(100% - 72px))` }}
            >
              <span
                className="mr-1.5 inline-block h-2 w-2 rounded-[2px] align-middle"
                style={{ backgroundColor: colorFor(totals[active].name, active, 'category') }}
              />
              {totals[active].name} · {fmtTime(totals[active].seconds)} (
              {((totals[active].seconds / grand) * 100).toFixed(1)}%)
            </div>
          )}
        </div>
      )}
      <StackedBars data={data} seriesKeys={seriesKeys} kind="category" />
      <LegendRow keys={seriesKeys} kind="category" />
    </div>
  )
}
