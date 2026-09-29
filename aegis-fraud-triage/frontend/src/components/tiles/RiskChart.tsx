import React, { useMemo } from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { TransactionEvent } from '../../types/events';
import { Tile } from '../layout/Tile';
import { LoadingSpinner } from '../ui/LoadingSpinner';

interface RiskChartProps {
  transactions: TransactionEvent[];
}

export function RiskChart({ transactions }: RiskChartProps) {
  const chartData = useMemo(() => {
    return transactions
      .slice(0, 50)
      .reverse()
      .map((txn, i) => ({
        index: i,
        time: new Date(txn.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        risk: Math.round(txn.risk_score * 100),
        amount: txn.amount,
        isAnomaly: txn.is_anomaly,
        merchant: txn.merchant_name,
      }));
  }, [transactions]);

  const anomalyCount = chartData.filter(d => d.isAnomaly).length;

  const badge = anomalyCount > 0 ? (
    <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-red-900/30 text-red-400 border border-red-500/30 font-mono">
      {anomalyCount} anomalies
    </span>
  ) : null;

  return (
    <Tile title="Risk Timeline" accentColor="#ffffff" badge={badge}>
      <div className="h-full w-full">
        {chartData.length === 0 ? (
          <LoadingSpinner label="Collecting data..." />
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4} />
                  <stop offset="40%" stopColor="#ffffff" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#ffffff" stopOpacity={0.02} />
                </linearGradient>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
              <XAxis
                dataKey="time"
                tick={{ fill: '#4b5563', fontSize: 9 }}
                axisLine={{ stroke: '#1f2937' }}
                tickLine={false}
                interval="preserveStartEnd"
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fill: '#4b5563', fontSize: 9 }}
                axisLine={{ stroke: '#1f2937' }}
                tickLine={false}
                width={28}
                tickFormatter={(v) => `${v}%`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(17, 17, 27, 0.95)',
                  border: '1px solid rgba(75, 85, 99, 0.3)',
                  borderRadius: '10px',
                  fontSize: 11,
                  backdropFilter: 'blur(10px)',
                  boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
                }}
                labelStyle={{ color: '#9ca3af' }}
                formatter={(value: any, name: any) => {
                  if (name === 'risk') return [`${value}%`, 'Risk Score'];
                  return [value, name];
                }}
              />
              {/* Anomaly threshold line */}
              <ReferenceLine
                y={55}
                stroke="#ef4444"
                strokeDasharray="4 4"
                strokeOpacity={0.4}
                label={{
                  value: 'ANOMALY',
                  fill: '#ef4444',
                  fontSize: 9,
                  opacity: 0.5,
                  position: 'right',
                }}
              />
              <Area
                type="monotone"
                dataKey="risk"
                stroke="#ffffff"
                strokeWidth={2}
                fill="url(#riskGradient)"
                animationDuration={300}
                dot={(props: any) => {
                  const { cx, cy, payload } = props;
                  if (payload.isAnomaly) {
                    return (
                      <g key={`dot-${payload.index}`}>
                        {/* Pulsing ring */}
                        <circle
                          cx={cx} cy={cy} r={8}
                          fill="none"
                          stroke="#ef4444"
                          strokeWidth={1.5}
                          strokeOpacity={0.3}
                        >
                          <animate attributeName="r" values="4;12;4" dur="2s" repeatCount="indefinite" />
                          <animate attributeName="stroke-opacity" values="0.5;0;0.5" dur="2s" repeatCount="indefinite" />
                        </circle>
                        {/* Solid dot */}
                        <circle
                          cx={cx} cy={cy} r={4}
                          fill="#ef4444"
                          stroke="#ef4444"
                          strokeWidth={2}
                          strokeOpacity={0.5}
                          filter="url(#glow)"
                        />
                      </g>
                    );
                  }
                  // Small white dot for normal transactions
                  return (
                    <circle
                      key={`dot-${payload.index}`}
                      cx={cx} cy={cy} r={1.5}
                      fill="#ffffff"
                      fillOpacity={0.3}
                    />
                  );
                }}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </Tile>
  );
}
