'use client'
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

export function AICodingChart({ aiPercent, size = 200 }: { aiPercent: number; size?: number }) {
  const data = [
    { name: 'AI', value: aiPercent },
    { name: 'Human', value: 100 - aiPercent },
  ]
  const outer = size / 2 - 10
  const inner = outer - Math.max(20, size * 0.14)

  return (
    <div className="relative flex items-center justify-center">
      <ResponsiveContainer width={size} height={size}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={inner}
            outerRadius={outer}
            dataKey="value"
            paddingAngle={2}
            stroke="#12161d"
            strokeWidth={2}
            cornerRadius={4}
            isAnimationActive={false}
            startAngle={90}
            endAngle={450}
          >
            <Cell fill="#b48ead" />
            <Cell fill="#2a303a" />
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center text-center">
        <p
          className={`font-semibold tracking-[-0.02em] text-onsurface ${size < 180 ? 'text-2xl' : 'text-3xl'}`}
        >
          {aiPercent}%
        </p>
        <p className="font-mono text-[10px] tracking-[0.02em] text-outline">AI-driven</p>
      </div>
    </div>
  )
}

export function WeekdaysChart({ data }: { data: Array<{ day: string; hours: number }> }) {
  const max = Math.max(...data.map((d) => d.hours))

  return (
    <ResponsiveContainer width="100%" height={250}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
        <defs>
          <linearGradient id="weekdayFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#64748b" />
            <stop offset="100%" stopColor="#475569" stopOpacity={0.9} />
          </linearGradient>
        </defs>
        <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.06)" />
        <XAxis
          dataKey="day"
          tickFormatter={(d: string) => d.slice(0, 3)}
          tickLine={false}
          axisLine={{ stroke: '#3d434e' }}
          tick={{ fill: '#8b919d', fontSize: 11, fontFamily: 'var(--font-mono), monospace' }}
        />
        <YAxis
          tickLine={false}
          axisLine={false}
          tick={{ fill: '#8b919d', fontSize: 11, fontFamily: 'var(--font-mono), monospace' }}
          unit="h"
        />
        <Tooltip
          formatter={(value) => [`${value}h`, 'Time']}
          cursor={{ fill: 'rgba(255,255,255,0.04)' }}
          labelStyle={{ color: '#8b919d', marginBottom: 4 }}
          itemStyle={{ color: '#e6e8ec', padding: 0 }}
          contentStyle={{
            backgroundColor: '#000000',
            border: '1px solid #3d434e',
            borderRadius: '8px',
            color: '#e6e8ec',
            fontFamily: 'var(--font-mono), monospace',
            fontSize: '12px',
          }}
        />
        <Bar isAnimationActive={false} dataKey="hours" radius={[3, 3, 0, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.hours === max ? '#b48ead' : 'url(#weekdayFill)'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// Semicircle gauge comparing today against the average of the past 6 days
// (compareSeconds), while displaying the range's daily average (avgText).
export function TodayGauge({
  todaySeconds,
  compareSeconds,
  todayText,
  avgText,
  mostActiveLabel,
}: {
  todaySeconds: number
  compareSeconds: number
  todayText: string
  avgText: string
  mostActiveLabel: string
}) {
  const ratio = compareSeconds > 0 ? todaySeconds / compareSeconds : 0
  const pct = Math.round(Math.abs(ratio - 1) * 100)
  const down = ratio < 1
  const color = down ? '#bf616a' : '#a3be8c'
  const gauge = [
    { value: Math.min(ratio, 1) },
    { value: 1 - Math.min(ratio, 1) },
  ]

  return (
    <div className="flex h-full flex-col items-center justify-center py-2">
      <p className="text-sm text-onsurface">
        <span className="font-semibold">{todayText}</span>{' '}
        <span className="text-onsurface-variant">Today</span>
      </p>
      <div className="relative h-[110px] w-[200px]">
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie
              data={gauge}
              cx="50%"
              cy="50%"
              startAngle={180}
              endAngle={0}
              innerRadius={62}
              outerRadius={84}
              dataKey="value"
              stroke="none"
              cornerRadius={4}
              isAnimationActive={false}
            >
              <Cell fill={color} />
              <Cell fill="#2a303a" />
            </Pie>
          </PieChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-1 font-mono text-[13px] font-medium tracking-[0.02em]" style={{ color }}>
        {down ? '↓' : '↑'} {pct}% {down ? 'Decrease' : 'Increase'}
      </p>
      <div className="mt-4 space-y-1 text-center">
        <p className="text-sm text-onsurface-variant">
          <span className="font-semibold text-onsurface">{avgText}</span> Daily Average
        </p>
        <p className="text-sm text-onsurface-variant">
          <span className="font-semibold text-onsurface">{mostActiveLabel}</span> Most Active Day
        </p>
      </div>
    </div>
  )
}

// Additions above the axis, deletions below, human vs AI stacks side by side.
export function AiHumanByDayChart({
  data,
}: {
  data: Array<{ date: string; humanAdd: number; humanDel: number; aiAdd: number; aiDel: number }>
}) {
  const fmtTick = (v: number) => {
    const a = Math.abs(v)
    return a >= 1000 ? `${(a / 1000).toFixed(a % 1000 === 0 ? 0 : 1)}k` : String(a)
  }
  return (
    <div>
      <ResponsiveContainer width="100%" height={230}>
        <BarChart data={data} stackOffset="sign" margin={{ top: 4, right: 4, left: -8, bottom: 0 }}>
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
            tickFormatter={fmtTick}
            tick={{ fill: '#8b919d', fontSize: 11, fontFamily: 'var(--font-mono), monospace' }}
          />
          <Tooltip
            formatter={(value, name) => [Math.abs(Number(value)).toLocaleString(), name]}
            cursor={{ fill: 'rgba(255,255,255,0.04)' }}
            labelStyle={{ color: '#8b919d', marginBottom: 4 }}
            itemStyle={{ color: '#e6e8ec', padding: 0 }}
            contentStyle={{
              backgroundColor: '#000000',
              border: '1px solid #3d434e',
              borderRadius: '8px',
              color: '#e6e8ec',
              fontFamily: 'var(--font-mono), monospace',
              fontSize: '12px',
            }}
          />
          <Bar isAnimationActive={false} dataKey="humanAdd" name="Human Additions" stackId="human" fill="#94a3b8" />
          <Bar isAnimationActive={false} dataKey="humanDel" name="Human Deletions" stackId="human" fill="#94a3b8" fillOpacity={0.35} />
          <Bar isAnimationActive={false} dataKey="aiAdd" name="AI Additions" stackId="ai" fill="#b48ead" />
          <Bar isAnimationActive={false} dataKey="aiDel" name="AI Deletions" stackId="ai" fill="#b48ead" fillOpacity={0.35} />
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-2 flex flex-wrap justify-center gap-x-4 gap-y-1">
        {[
          ['Human Additions', '#94a3b8', 1],
          ['Human Deletions', '#94a3b8', 0.35],
          ['AI Additions', '#b48ead', 1],
          ['AI Deletions', '#b48ead', 0.35],
        ].map(([label, color, opacity]) => (
          <span key={label as string} className="flex items-center gap-1.5">
            <span
              className="h-2 w-2 rounded-[2px]"
              style={{ backgroundColor: color as string, opacity: opacity as number }}
            />
            <span className="font-mono text-[11px] text-onsurface-variant">{label}</span>
          </span>
        ))}
      </div>
    </div>
  )
}
