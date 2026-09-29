import React, { useState, useEffect } from 'react';
import { AlertEvent, AgentTrace, KYCResult } from '../../types/events';
import { formatCurrency, timeAgo, getRiskColor, getRiskLevel, getTraceColor, getTracePrefix, stripMarkdown } from '../../utils/formatters';
import { Tile } from '../layout/Tile';
import { LoadingSpinner } from '../ui/LoadingSpinner';
import { ChevronDown, ChevronRight, Shield, UserCheck, CheckCircle, XCircle, Brain } from 'lucide-react';
import { API_URL } from '../../config';

interface AlertPanelProps {
  alerts: AlertEvent[];
  agentTraces?: AgentTrace[];
  activeInvestigationIds?: Set<string>;
}

function ConfidenceBadge({ score }: { score: number | null }) {
  if (score == null) return null;
  const color = score >= 75 ? '#d1d5db' : score >= 50 ? '#9ca3af' : '#ef4444';
  return (
    <span
      className="text-[10px] px-1.5 py-0.5 rounded-full font-bold font-mono border"
      style={{
        color,
        background: `${color}15`,
        borderColor: `${color}30`,
      }}
    >
      {score.toFixed(0)}%
    </span>
  );
}

function StatusBadge({ status, reviewStatus, confidence }: { status: string; reviewStatus?: string | null; confidence?: number | null }) {
  const colors: Record<string, string> = {
    pending: 'bg-gray-800/50 text-gray-400 border-gray-600/30',
    investigating: 'bg-gray-800/40 text-gray-300 border-gray-600/20',
    blocked: 'bg-red-900/50 text-red-400 border-red-500/30',
    cleared: 'bg-gray-800/40 text-gray-300 border-gray-600/20',
    flagged: 'bg-gray-800/50 text-gray-400 border-gray-600/30',
    awaiting_review: 'bg-gray-700/50 text-white border-gray-500/40 animate-pulse',
  };

  const label = status === 'awaiting_review' ? 'AWAITING REVIEW' : status.toUpperCase();

  // After human review: show the final status + a small "by human" tag
  const isHumanResolved = reviewStatus === 'resolved';

  return (
    <div className="flex items-center gap-1.5">
      <span className={`text-xs px-2 py-0.5 rounded-full border ${colors[status] || colors.pending}`}>
        {label}
      </span>
      {confidence != null && (status === 'blocked' || status === 'cleared' || status === 'awaiting_review') && (
        <ConfidenceBadge score={confidence} />
      )}
      {isHumanResolved && (
        <span className="text-xs px-1.5 py-0.5 rounded-full border bg-gray-700/40 text-gray-300 border-gray-500/30 flex items-center gap-0.5">
          <UserCheck size={9} /> HUMAN
        </span>
      )}
    </div>
  );
}

function HumanReviewPanel({ alert }: { alert: AlertEvent }) {
  const [reason, setReason] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const submitDecision = async (decision: 'blocked' | 'cleared') => {
    setSubmitting(true);
    try {
      await fetch(`${API_URL}/api/alerts/${alert.id}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'override',
          override_action: decision,
          reason: reason || undefined,
        }),
      });
    } catch (e) {
      console.error('Failed to submit review:', e);
    }
    setSubmitting(false);
    setReason('');
  };

  // Parse the agent's recommendation from the verdict
  const agentRecommendation = alert.agent_verdict
    ? alert.agent_verdict.includes('BLOCK') ? 'blocked'
      : alert.agent_verdict.includes('CLEAR') ? 'cleared'
      : null
    : null;

  return (
    <div className="mt-2 p-3 bg-gray-800/50 border border-gray-500/30 rounded-lg space-y-2.5" onClick={(e) => e.stopPropagation()}>
      <div className="flex items-center gap-1.5 text-gray-200 text-xs font-bold uppercase tracking-wider">
        <UserCheck size={12} />
        Human Decision Required
      </div>

      {/* Show agent's full reasoning prominently */}
      {alert.agent_verdict && (
        <div className="p-2.5 bg-gray-900/60 rounded-lg border border-gray-700/30 space-y-1.5">
          <div className="flex items-center gap-1.5 text-xs text-white/70 font-semibold">
            <Brain size={11} />
            AI Agent Analysis
          </div>
          <p className="text-xs text-gray-300 leading-relaxed">
            {stripMarkdown(alert.agent_verdict).slice(0, 500)}{alert.agent_verdict.length > 500 ? '...' : ''}
          </p>
        </div>
      )}

      <input
        type="text"
        placeholder="Reason (optional)"
        value={reason}
        onChange={(e) => setReason(e.target.value)}
        className="w-full px-2 py-1 text-xs bg-gray-900/80 text-gray-300 border border-gray-700/50 rounded
          placeholder-gray-600 focus:outline-none focus:border-gray-500/50"
      />

      <div className="flex gap-2">
        <button
          onClick={() => submitDecision('blocked')}
          disabled={submitting}
          className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-md text-xs font-bold
            bg-red-900/50 text-red-400 border border-red-500/40
            hover:bg-red-800/60 hover:border-red-400/60 transition-all
            disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <XCircle size={13} />
          {submitting ? '...' : 'Block Account'}
        </button>
        {agentRecommendation && (
          <button
            onClick={() => submitDecision(agentRecommendation as 'blocked' | 'cleared')}
            disabled={submitting}
            className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-md text-xs font-bold
              bg-white/8 text-white/70 border border-white/15
              hover:bg-white/10 hover:border-white/20 transition-all
              disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Brain size={13} />
            {submitting ? '...' : 'Confirm AI'}
          </button>
        )}
        <button
          onClick={() => submitDecision('cleared')}
          disabled={submitting}
          className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-md text-xs font-bold
            bg-white/8 text-white/70 border border-white/15
            hover:bg-white/10 hover:border-white/20 transition-all
            disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <CheckCircle size={13} />
          {submitting ? '...' : 'Clear Transaction'}
        </button>
      </div>
    </div>
  );
}

function AlertKYC({ accountId }: { accountId: string }) {
  const [kyc, setKyc] = useState<KYCResult | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/api/kyc/${accountId}`)
      .then(r => r.json())
      .then(data => setKyc(data))
      .catch(() => {});
  }, [accountId]);

  if (!kyc) return null;

  const kycColors: Record<string, string> = {
    low: 'text-gray-300 border-gray-600/20 bg-gray-800/30',
    medium: 'text-gray-300 border-gray-500/30 bg-gray-800/40',
    high: 'text-red-400 border-red-500/30 bg-red-900/20',
    critical: 'text-red-400 border-red-500/50 bg-red-900/30',
  };

  return (
    <div className={`text-xs rounded p-2 border ${kycColors[kyc.risk_level] || kycColors.low}`}>
      <div className="flex items-center gap-1 font-semibold mb-1">
        <Shield size={10} />
        KYC Identity Check: {kyc.risk_level.toUpperCase()} ({kyc.risk_score}/100)
      </div>
      {kyc.flags.map((flag, i) => (
        <div key={i} className="text-gray-400 pl-3">- {flag}</div>
      ))}
    </div>
  );
}

export function AlertPanel({ alerts, agentTraces = [], activeInvestigationIds }: AlertPanelProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // Auto-expand awaiting_review alerts
  useEffect(() => {
    const awaitingReview = alerts.find(a => a.status === 'awaiting_review' && a.review_status !== 'resolved');
    if (awaitingReview && expandedId !== awaitingReview.id) {
      setExpandedId(awaitingReview.id);
    }
  }, [alerts, expandedId]);

  // Sort alerts: awaiting_review first, then active investigations, then by recency
  // Limit to 15 visible alerts to prevent infinite stacking
  const sortedAlerts = [...alerts].sort((a, b) => {
    const aAwait = a.status === 'awaiting_review' && a.review_status !== 'resolved';
    const bAwait = b.status === 'awaiting_review' && b.review_status !== 'resolved';
    if (aAwait && !bAwait) return -1;
    if (!aAwait && bAwait) return 1;
    const aActive = activeInvestigationIds?.has(a.id) ?? false;
    const bActive = activeInvestigationIds?.has(b.id) ?? false;
    if (aActive && !bActive) return -1;
    if (!aActive && bActive) return 1;
    return 0; // preserve original order (newest first)
  }).slice(0, 5);

  // Count resolved alerts that are hidden
  const hiddenCount = alerts.length > 15 ? alerts.length - 15 : 0;

  return (
    <Tile title="Anomaly Alerts" accentColor="#ffffff">
      <div className="overflow-y-auto h-full space-y-2 pr-1 scrollbar-thin">
        {alerts.length === 0 && (
          <LoadingSpinner label="Monitoring for anomalies..." />
        )}
        {sortedAlerts.map((alert, i) => {
          const isExpanded = expandedId === alert.id;
          const alertTraces = agentTraces.filter(t => t.alert_id === alert.id);
          const isBeingInvestigated = activeInvestigationIds?.has(alert.id) ?? false;
          const isAwaitingReview = alert.status === 'awaiting_review' && alert.review_status !== 'resolved';
          const isResolved = (alert.status === 'blocked' || alert.status === 'cleared') && !isExpanded;

          // Resolved alerts render as compact one-liners unless expanded
          if (isResolved) {
            return (
              <div
                key={alert.id}
                className="flex items-center justify-between gap-2 bg-gray-900/30 border border-gray-700/20 rounded-lg px-3 py-1.5 cursor-pointer hover:border-gray-600/30 transition-all"
                onClick={() => setExpandedId(alert.id)}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <ChevronRight size={10} className="text-gray-600 flex-shrink-0" />
                  <span className="text-xs text-gray-400 truncate">{alert.transaction.merchant_name}</span>
                  <span className="text-xs font-mono text-gray-500">{formatCurrency(alert.transaction.amount)}</span>
                </div>
                <StatusBadge status={alert.status} reviewStatus={alert.review_status} confidence={alert.confidence_score} />
              </div>
            );
          }

          return (
            <div
              key={alert.id}
              className={`bg-red-950/30 border rounded-lg p-3 space-y-2 cursor-pointer transition-all ${
                isAwaitingReview
                  ? 'border-gray-400/40 ring-2 ring-white/10'
                  : isExpanded ? 'border-red-500/40' : 'border-red-500/20 hover:border-red-500/30'
              } ${isBeingInvestigated ? 'investigation-focus' : ''}`}
              style={{
                animation: i === 0 && !isResolved ? 'alertSlideIn 0.3s ease-out' : undefined,
              }}
              onClick={() => setExpandedId(isExpanded ? null : alert.id)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  {isExpanded
                    ? <ChevronDown size={12} className="text-gray-500" />
                    : <ChevronRight size={12} className="text-gray-500" />
                  }
                  <span className="text-sm font-semibold text-red-300">
                    {alert.transaction.merchant_name}
                  </span>
                </div>
                <StatusBadge status={alert.status} reviewStatus={alert.review_status} confidence={alert.confidence_score} />
              </div>

              <div className="flex items-center justify-between">
                <span className="text-lg font-mono font-bold text-red-400">
                  {formatCurrency(alert.transaction.amount)}
                </span>
                <span className="text-xs text-gray-400">
                  {timeAgo(alert.timestamp)}
                </span>
              </div>

              {/* Risk score bar */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-400">Risk Score</span>
                  <span style={{ color: getRiskColor(alert.risk_score) }}>
                    {(alert.risk_score * 100).toFixed(0)}% - {getRiskLevel(alert.risk_score)}
                  </span>
                </div>
                <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${alert.risk_score * 100}%`,
                      backgroundColor: getRiskColor(alert.risk_score),
                    }}
                  />
                </div>
              </div>

              {/* Risk factors (always show top 2, show all when expanded) */}
              <div className="space-y-1">
                {alert.risk_factors.slice(0, isExpanded ? undefined : 2).map((factor, j) => (
                  <div key={j} className="text-xs text-gray-400 flex items-start gap-1">
                    <span className="text-red-500 mt-0.5">!</span>
                    <span>{factor}</span>
                  </div>
                ))}
              </div>

              {/* Expanded section */}
              {isExpanded && (
                <div className="mt-2 space-y-2 border-t border-red-500/10 pt-2">
                  {/* Transaction details */}
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                    <div>
                      <span className="text-gray-500">Account: </span>
                      <span className="text-gray-300 font-mono">{alert.transaction.account_id}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Merchant ID: </span>
                      <span className="text-gray-300 font-mono">{alert.transaction.merchant_id}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Location: </span>
                      <span className="text-gray-300">
                        {alert.transaction.city || 'Unknown'}, {alert.transaction.country}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">Category: </span>
                      <span className="text-gray-300">{alert.transaction.category}</span>
                    </div>
                    {alert.transaction.latitude && (
                      <div className="col-span-2">
                        <span className="text-gray-500">Coordinates: </span>
                        <span className="text-gray-300 font-mono">
                          {alert.transaction.latitude.toFixed(4)}, {alert.transaction.longitude?.toFixed(4)}
                        </span>
                      </div>
                    )}
                  </div>

                  {/* KYC Identity Risk */}
                  <AlertKYC accountId={alert.transaction.account_id} />

                  {/* Agent verdict */}
                  {alert.agent_verdict && alert.status !== 'awaiting_review' && (
                    <div className="text-xs bg-gray-800/50 rounded p-2 text-gray-300 border-l-2 border-white/30">
                      <div className="text-white/70 font-semibold mb-1 flex items-center gap-1.5">
                        AI Agent Verdict
                        {alert.confidence_score != null && (
                          <ConfidenceBadge score={alert.confidence_score} />
                        )}
                      </div>
                      {stripMarkdown(alert.agent_verdict).slice(0, 300)}{alert.agent_verdict.length > 300 ? '...' : ''}
                    </div>
                  )}

                  {/* Agent investigation trace */}
                  {alertTraces.length > 0 && (
                    <div className="space-y-1">
                      <div className="text-xs text-gray-500 font-semibold uppercase tracking-wider">
                        Investigation Trace
                      </div>
                      <div className="max-h-32 overflow-y-auto space-y-0.5 scrollbar-thin">
                        {alertTraces.map((trace, k) => (
                          <div key={k} className="text-xs flex gap-1">
                            <span
                              style={{ color: getTraceColor(trace.step_type) }}
                              className="font-bold flex-shrink-0"
                            >
                              {getTracePrefix(trace.step_type)}
                            </span>
                            <span className="text-gray-400 truncate">
                              {trace.content.slice(0, 120)}{trace.content.length > 120 ? '...' : ''}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* HITL: Human Review Panel - shown when awaiting review */}
                  {isAwaitingReview && (
                    <HumanReviewPanel alert={alert} />
                  )}

                  {/* Human review result */}
                  {alert.review_status === 'resolved' && alert.human_reason && (
                    <div className="text-xs rounded p-2 border-l-2 bg-gray-800/40 border-gray-400 text-gray-300">
                      <div className="font-semibold mb-0.5 flex items-center gap-1">
                        <UserCheck size={10} />
                        Human Decision: {alert.human_override?.toUpperCase()}
                      </div>
                      <span className="text-gray-400">{alert.human_reason}</span>
                    </div>
                  )}
                </div>
              )}

            </div>
          );
        })}
        {hiddenCount > 0 && (
          <div className="text-center text-[10px] text-gray-600 py-1 font-mono">
            +{hiddenCount} older alert{hiddenCount !== 1 ? 's' : ''}
          </div>
        )}
      </div>
    </Tile>
  );
}
