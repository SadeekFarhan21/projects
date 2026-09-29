import React from 'react';
import { TransactionEvent } from '../../types/events';
import { formatCurrency, timeAgo, getRiskColor } from '../../utils/formatters';
import { Tile } from '../layout/Tile';
import { LoadingSpinner } from '../ui/LoadingSpinner';

interface TransactionFeedProps {
  transactions: TransactionEvent[];
}

function RiskBar({ score }: { score: number }) {
  return (
    <div className="w-12 h-1 bg-gray-800 rounded-full overflow-hidden flex-shrink-0">
      <div
        className="h-full rounded-full transition-all duration-500"
        style={{
          width: `${Math.max(score * 100, 3)}%`,
          backgroundColor: getRiskColor(score),
        }}
      />
    </div>
  );
}

export function TransactionFeed({ transactions }: TransactionFeedProps) {
  const liveBadge = (
    <div className="flex items-center gap-1.5 text-xs text-white/70">
      <span className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
      LIVE
    </div>
  );

  return (
    <Tile title="Live Transactions" className="row-span-2" accentColor="#ffffff" badge={liveBadge}>
      <div className="overflow-y-auto h-full space-y-1 pr-1 scrollbar-thin">
        {transactions.length === 0 && (
          <LoadingSpinner label="Waiting for transactions..." />
        )}
        {transactions.slice(0, 50).map((txn, i) => (
          <div
            key={txn.id + i}
            className={`flex items-center gap-2.5 px-3 py-2 rounded-xl transition-all duration-300 ${
              txn.is_anomaly
                ? 'bg-red-950/15 border border-red-900/20'
                : 'bg-gray-800/20 border border-transparent hover:bg-gray-800/40 hover:border-gray-700/30'
            }`}
            style={{
              animation: i === 0 ? 'slideIn 0.3s cubic-bezier(0.16, 1, 0.3, 1)' : undefined,
            }}
          >
            {/* Merchant & category */}
            <div className="flex-1 min-w-0">
              <div className="text-sm text-white truncate font-medium">{txn.merchant_name}</div>
              <div className="text-[11px] text-gray-500">
                {txn.category} · {txn.city || 'Online'}{txn.country === 'US' && txn.state ? `, ${txn.state}` : `, ${txn.country}`}
              </div>
            </div>

            {/* Risk mini-bar */}
            <RiskBar score={txn.risk_score} />

            {/* Amount */}
            <div className={`text-sm font-mono font-bold tabular-nums ${txn.is_anomaly ? 'text-red-400/60' : 'text-white'}`}>
              {formatCurrency(txn.amount)}
            </div>

            {/* Time */}
            <div className="text-[11px] text-gray-600 w-14 text-right flex-shrink-0 font-mono">
              {timeAgo(txn.timestamp)}
            </div>
          </div>
        ))}
      </div>
    </Tile>
  );
}
