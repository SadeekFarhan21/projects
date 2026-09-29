import React from 'react';
import { DisputeEvent } from '../../types/events';
import { formatCurrency, timeAgo, stripMarkdown } from '../../utils/formatters';
import { Tile } from '../layout/Tile';
import { Gavel, Clock, CheckCircle, XCircle, AlertTriangle, Loader2 } from 'lucide-react';

interface DisputePanelProps {
  disputes: DisputeEvent[];
}

const REASON_LABELS: Record<string, { label: string; color: string }> = {
  unauthorized_charge: { label: 'Unauthorized', color: '#9ca3af' },
  wrong_amount: { label: 'Wrong Amount', color: '#9ca3af' },
  never_received: { label: 'Not Received', color: '#9ca3af' },
  duplicate: { label: 'Duplicate', color: '#9ca3af' },
  fraud_claim: { label: 'Fraud Claim', color: '#9ca3af' },
};

const STATUS_CONFIG: Record<string, { icon: React.ReactNode; color: string; bg: string }> = {
  pending: {
    icon: <Clock size={12} />,
    color: '#9ca3af',
    bg: 'bg-gray-800/40 border-gray-600/30',
  },
  investigating: {
    icon: <Loader2 size={12} className="animate-spin" />,
    color: '#d1d5db',
    bg: 'bg-gray-800/30 border-gray-600/20',
  },
  approved: {
    icon: <CheckCircle size={12} />,
    color: '#22c55e',
    bg: 'bg-green-900/20 border-green-500/20',
  },
  denied: {
    icon: <XCircle size={12} />,
    color: '#ef4444',
    bg: 'bg-red-900/30 border-red-500/30',
  },
  escalated: {
    icon: <AlertTriangle size={12} />,
    color: '#9ca3af',
    bg: 'bg-gray-800/40 border-gray-600/30',
  },
};

export function DisputePanel({ disputes }: DisputePanelProps) {
  return (
    <Tile title="Customer Disputes" accentColor="#ffffff">
      <div className="overflow-y-auto h-full space-y-2 pr-1 scrollbar-thin">
        {disputes.length === 0 && (
          <div className="text-gray-500 text-sm text-center py-8 flex flex-col items-center gap-2">
            <Gavel size={20} className="text-white/30" />
            <span>No disputes filed yet</span>
            <span className="text-xs text-gray-600">Disputes appear when customers challenge transactions</span>
          </div>
        )}
        {disputes.map((dispute, i) => {
          const reason = REASON_LABELS[dispute.reason] || { label: dispute.reason, color: '#9ca3af' };
          const status = STATUS_CONFIG[dispute.status] || STATUS_CONFIG.pending;

          return (
            <div
              key={dispute.id}
              className={`rounded-xl p-3 border transition-all duration-300 ${
                dispute.status === 'investigating'
                  ? 'bg-gray-800/30 border-gray-600/20'
                  : dispute.status === 'approved'
                  ? 'bg-gray-800/20 border-gray-600/15'
                  : dispute.status === 'denied'
                  ? 'bg-red-950/20 border-red-500/20'
                  : 'bg-gray-800/30 border-gray-700/30'
              }`}
              style={{
                animation: i === 0 ? 'slideIn 0.3s ease-out' : undefined,
              }}
            >
              {/* Header row: merchant + status */}
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2 min-w-0">
                  <Gavel size={13} style={{ color: reason.color }} className="flex-shrink-0" />
                  <span className="text-sm font-semibold text-white truncate">
                    {dispute.merchant_name}
                  </span>
                </div>
                <div
                  className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold border ${status.bg}`}
                  style={{ color: status.color }}
                >
                  {status.icon}
                  {dispute.status.toUpperCase()}
                </div>
              </div>

              {/* Amount + reason badge */}
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-lg font-mono font-bold text-white">
                  {formatCurrency(dispute.amount)}
                </span>
                <span
                  className="text-xs px-2 py-0.5 rounded-full font-medium"
                  style={{
                    backgroundColor: `${reason.color}15`,
                    color: reason.color,
                    border: `1px solid ${reason.color}30`,
                  }}
                >
                  {reason.label}
                </span>
              </div>

              {/* Customer statement */}
              <div className="text-xs text-gray-400 italic mb-1.5 leading-relaxed">
                "{dispute.customer_statement.slice(0, 120)}{dispute.customer_statement.length > 120 ? '...' : ''}"
              </div>

              {/* Resolution summary (when resolved) */}
              {dispute.resolution_summary && (
                <div
                  className="text-xs rounded-lg p-2 mt-1 border-l-2"
                  style={{
                    borderColor: status.color,
                    backgroundColor: `${status.color}10`,
                    color: '#d1d5db',
                  }}
                >
                  <div className="font-semibold mb-0.5" style={{ color: status.color }}>
                    AI Resolution
                  </div>
                  {stripMarkdown(dispute.resolution_summary).slice(0, 200)}
                </div>
              )}

              {/* Footer: time */}
              <div className="flex items-center justify-between mt-1.5">
                <span className="text-xs text-gray-600 font-mono">
                  {dispute.account_id.slice(0, 8)}...
                </span>
                <span className="text-xs text-gray-500">
                  {timeAgo(dispute.timestamp)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </Tile>
  );
}
