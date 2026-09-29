'use client'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

interface Props {
  data: Array<{ date: string; seconds: number }>
}

export default function ActivityChart({ data }: Props) {
  const chartData = data.map((d) => ({
    date: d.date.slice(5),
    hours: Math.round(d.seconds / 360) / 10,
  }))

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={chartData} margin={{ top: 4, right: 4, left: -16, bottom: 0 }}>
        <defs>
          <linearGradient id="activityFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#88c0d0" />
            <stop offset="100%" stopColor="#6797a6" stopOpacity={0.9} />
          </linearGradient>
        </defs>
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
          tick={{ fill: '#8b919d', fontSize: 11, fontFamily: 'var(--font-mono), monospace' }}
          unit="h"
        />
        <Tooltip
          formatter={(value) => [`${value}h`, 'Coding']}
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
        <Bar isAnimationActive={false} dataKey="hours" fill="url(#activityFill)" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}
