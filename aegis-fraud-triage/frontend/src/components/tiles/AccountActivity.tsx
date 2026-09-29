import React, { useMemo, useState, useEffect, useRef, useCallback } from 'react';
import { TransactionEvent, KYCResult } from '../../types/events';
import { formatCurrency, timeAgo, getRiskColor } from '../../utils/formatters';
import { Tile } from '../layout/Tile';
import { LoadingSpinner } from '../ui/LoadingSpinner';
import { API_URL } from '../../config';
import { Shield, ChevronDown, ChevronRight } from 'lucide-react';

interface AccountActivityProps {
  transactions: TransactionEvent[];
}

interface AccountSummary {
  id: string;
  txnCount: number;
  totalSpend: number;
  anomalyCount: number;
  avgRisk: number;
  lastCity: string;
}

const KYC_COLORS: Record<string, string> = {
  low: '#9ca3af',
  medium: '#d1d5db',
  high: '#ef4444',
  critical: '#dc2626',
};

function KYCBadge({ accountId }: { accountId: string }) {
  const [kyc, setKyc] = useState<KYCResult | null>(null);
  const [expanded, setExpanded] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch(`${API_URL}/api/kyc/${accountId}`)
      .then(r => r.json())
      .then(data => setKyc(data))
      .catch(() => {});
  }, [accountId]);

  // Click-outside-to-dismiss
  useEffect(() => {
    if (!expanded) return;
    function handleClickOutside(e: MouseEvent) {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        setExpanded(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [expanded]);

  if (!kyc) return null;

  return (
    <div className="relative" ref={popoverRef}>
      <button
        onClick={(e) => { e.stopPropagation(); setExpanded(!expanded); }}
        className="flex items-center gap-0.5 px-1.5 py-0.5 rounded text-xs font-semibold transition-all hover:scale-105"
        style={{
          backgroundColor: `${KYC_COLORS[kyc.risk_level]}15`,
          color: KYC_COLORS[kyc.risk_level],
          border: `1px solid ${KYC_COLORS[kyc.risk_level]}30`,
        }}
        title={`KYC: ${kyc.risk_level.toUpperCase()} (${kyc.risk_score}/100)`}
      >
        <Shield size={10} />
        KYC
      </button>
      {expanded && (
        <div className="absolute right-0 top-7 z-50 w-64 bg-gray-900/95 backdrop-blur-xl border border-gray-700 rounded-lg p-3 shadow-xl text-xs animate-fade-up">
          <div className="flex justify-between items-center mb-2">
            <span className="font-semibold text-gray-300">KYC Assessment</span>
            <span
              className="px-1.5 py-0.5 rounded font-bold"
              style={{
                backgroundColor: `${KYC_COLORS[kyc.risk_level]}20`,
                color: KYC_COLORS[kyc.risk_level],
              }}
            >
              {kyc.risk_level.toUpperCase()} ({kyc.risk_score})
            </span>
          </div>
          <div className="space-y-1 text-gray-400">
            <div>Location Familiar: {kyc.location_familiar ?
              <span className="text-white/70">Yes</span> :
              <span className="text-red-400">No - New Location</span>}
            </div>
            {kyc.familiar_locations && kyc.familiar_locations.length > 0 && (
              <div className="text-xs text-gray-500">Known: {kyc.familiar_locations.slice(0, 5).map(l => l.charAt(0).toUpperCase() + l.slice(1)).join(', ')}</div>
            )}
            {kyc.customer_id && (
              <div>Customer: <span className="font-mono text-gray-300">{kyc.customer_id.slice(0, 10)}...</span></div>
            )}
            <div className="mt-2 font-semibold text-gray-300">Flags:</div>
            {kyc.flags.map((flag, i) => (
              <div key={i} className="text-gray-400 pl-2">- {flag}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export function AccountActivity({ transactions }: AccountActivityProps) {
  const [expandedAccountId, setExpandedAccountId] = useState<string | null>(null);

  const accounts = useMemo(() => {
    const map = new Map<string, AccountSummary>();

    for (const txn of transactions) {
      const existing = map.get(txn.account_id);
      if (existing) {
        existing.txnCount += 1;
        existing.totalSpend += txn.amount;
        existing.avgRisk = (existing.avgRisk * (existing.txnCount - 1) + txn.risk_score) / existing.txnCount;
        if (txn.is_anomaly) existing.anomalyCount += 1;
        existing.lastCity = txn.city || existing.lastCity;
      } else {
        map.set(txn.account_id, {
          id: txn.account_id,
          txnCount: 1,
          totalSpend: txn.amount,
          anomalyCount: txn.is_anomaly ? 1 : 0,
          avgRisk: txn.risk_score,
          lastCity: txn.city || 'Online',
        });
      }
    }

    return Array.from(map.values())
      .sort((a, b) => b.anomalyCount - a.anomalyCount || b.txnCount - a.txnCount);
  }, [transactions]);

  // Get transactions for an expanded account, sorted newest first
  const getAccountTransactions = useCallback((accountId: string) => {
    return transactions
      .filter(t => t.account_id === accountId)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }, [transactions]);

  return (
    <Tile title="Account Activity" accentColor="#ffffff">
      <div className="overflow-y-auto h-full space-y-1.5 pr-1 scrollbar-thin">
        {accounts.length === 0 && (
          <LoadingSpinner label="Loading accounts..." />
        )}
        {accounts.map((acc) => {
          const isExpanded = expandedAccountId === acc.id;
          const accountTxns = isExpanded ? getAccountTransactions(acc.id) : [];

          return (
            <div key={acc.id} className="space-y-0">
              {/* Account row — clickable */}
              <div
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs cursor-pointer transition-all duration-200 ${
                  isExpanded
                    ? 'bg-white/5 border border-white/10'
                    : acc.anomalyCount > 0
                      ? 'bg-red-950/20 border border-red-500/20 hover:border-red-500/30'
                      : 'bg-gray-800/30 border border-transparent hover:bg-gray-800/50 hover:border-gray-700/30'
                }`}
                onClick={() => setExpandedAccountId(isExpanded ? null : acc.id)}
              >
                {/* Left group: identity + badges */}
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  {/* Expand chevron */}
                  {isExpanded
                    ? <ChevronDown size={12} className="text-white/50 flex-shrink-0" />
                    : <ChevronRight size={12} className="text-gray-500 flex-shrink-0" />
                  }

                  {/* Risk dot */}
                  <div
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ backgroundColor: getRiskColor(acc.avgRisk) }}
                  />

                  {/* Account ID */}
                  <div className="min-w-0">
                    <div className="text-gray-300 font-mono truncate">
                      {acc.id.slice(0, 8)}...{acc.id.slice(-4)}
                    </div>
                    <div className="text-gray-500">
                      {acc.txnCount} txns &middot; {acc.lastCity}
                    </div>
                  </div>

                  {/* Anomaly badge — next to account ID */}
                  {acc.anomalyCount > 0 && (
                    <span className="text-red-400 bg-red-900/40 px-1.5 py-0.5 rounded text-xs font-bold flex-shrink-0">
                      {acc.anomalyCount}
                    </span>
                  )}

                  {/* KYC badge */}
                  <KYCBadge accountId={acc.id} />
                </div>

                {/* Right group: total spend */}
                <div className="text-gray-400 text-right flex-shrink-0 font-mono">
                  {formatCurrency(acc.totalSpend)}
                </div>
              </div>

              {/* Expanded: Transaction list for this account */}
              {isExpanded && (
                <div className="ml-4 mr-1 mt-1 mb-1 space-y-0.5 border-l-2 border-white/10 pl-2 animate-fade-up">
                  {/* Header row */}
                  <div className="flex items-center justify-between px-2 py-1">
                    <span className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">
                      {accountTxns.length} Transaction{accountTxns.length !== 1 ? 's' : ''}
                    </span>
                    <span className="text-[10px] text-gray-500">
                      Avg Risk: <span style={{ color: getRiskColor(acc.avgRisk) }}>
                        {(acc.avgRisk * 100).toFixed(0)}%
                      </span>
                    </span>
                  </div>

                  {/* Transaction rows */}
                  {accountTxns.slice(0, 20).map((txn, i) => (
                    <div
                      key={txn.id + i}
                      className={`flex items-center gap-2 px-2 py-1.5 rounded-lg text-xs transition-all ${
                        txn.is_anomaly
                          ? 'bg-red-950/30 border border-red-500/20'
                          : 'bg-gray-800/20 border border-transparent'
                      }`}
                    >
                      {/* Merchant name + category */}
                      <div className="flex-1 min-w-0">
                        <div className="text-gray-300 truncate">{txn.merchant_name}</div>
                        <div className="text-[10px] text-gray-600">
                          {txn.category} · {txn.city || 'Online'}, {txn.country}
                        </div>
                      </div>

                      {/* Risk mini bar */}
                      <div className="w-8 h-1 bg-gray-800 rounded-full overflow-hidden flex-shrink-0">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${Math.max(txn.risk_score * 100, 3)}%`,
                            backgroundColor: getRiskColor(txn.risk_score),
                          }}
                        />
                      </div>

                      {/* Amount */}
                      <div className={`font-mono font-bold tabular-nums flex-shrink-0 ${
                        txn.is_anomaly ? 'text-red-400' : 'text-gray-300'
                      }`}>
                        {formatCurrency(txn.amount)}
                      </div>

                      {/* Time ago */}
                      <div className="text-[10px] text-gray-600 w-12 text-right flex-shrink-0 font-mono">
                        {timeAgo(txn.timestamp)}
                      </div>
                    </div>
                  ))}

                  {accountTxns.length > 20 && (
                    <div className="text-[10px] text-gray-500 text-center py-1">
                      +{accountTxns.length - 20} more transactions
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </Tile>
  );
}
