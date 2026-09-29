import React, { useRef, useEffect, useState, useMemo } from 'react';
import { AgentTrace } from '../../types/events';
import { getTraceColor, getTracePrefix, formatTime, getVerdictColor, parseVerdict, parseStepNumber } from '../../utils/formatters';
import { Tile } from '../layout/Tile';
import { Brain, Zap, ClipboardList, Target, Scale } from 'lucide-react';

interface AgentConsoleProps {
  traces: AgentTrace[];
}

function StepIcon({ type }: { type: string }) {
  const color = getTraceColor(type);
  const props = { size: 12, style: { color }, strokeWidth: 2 };
  switch (type) {
    case 'thinking': return <Brain {...props} />;
    case 'tool_call': return <Zap {...props} />;
    case 'tool_result': return <ClipboardList {...props} />;
    case 'action': return <Target {...props} />;
    case 'verdict': return <Scale {...props} />;
    default: return <Zap {...props} />;
  }
}

export function AgentConsole({ traces }: AgentConsoleProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [selectedTab, setSelectedTab] = useState<string | null>(null);
  const [dismissedTabs, setDismissedTabs] = useState<Set<string>>(new Set());
  const dismissTimers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  // Group traces by alert_id into ALL investigations (including dismissed)
  const allInvestigations = useMemo(() => {
    const groups = new Map<string, AgentTrace[]>();
    for (const trace of traces) {
      const existing = groups.get(trace.alert_id) || [];
      existing.push(trace);
      groups.set(trace.alert_id, existing);
    }
    return Array.from(groups.entries());
  }, [traces]);

  // Auto-dismiss completed tabs after 15 seconds
  useEffect(() => {
    const timers = dismissTimers.current;
    for (const [alertId, alertTraces] of allInvestigations) {
      const hasVerdict = alertTraces.some(t => t.step_type === 'verdict');
      if (hasVerdict && !dismissedTabs.has(alertId) && !timers.has(alertId)) {
        const timer = setTimeout(() => {
          setDismissedTabs(prev => {
            const next = new Set(prev);
            next.add(alertId);
            return next;
          });
          timers.delete(alertId);
        }, 15000);
        timers.set(alertId, timer);
      }
    }
    // Cleanup timers on unmount
    return () => {
      timers.forEach(timer => clearTimeout(timer));
    };
  }, [allInvestigations, dismissedTabs]);

  // Filter out dismissed tabs (but keep the currently selected one visible)
  const investigations = useMemo(() => {
    return allInvestigations.filter(
      ([id]) => !dismissedTabs.has(id) || id === selectedTab
    );
  }, [allInvestigations, dismissedTabs, selectedTab]);

  // Auto-select tab: newest active investigation, or stay on current if still active
  useEffect(() => {
    if (investigations.length === 0) {
      setSelectedTab(null);
      return;
    }

    // If nothing selected, pick the newest
    if (!selectedTab) {
      setSelectedTab(investigations[investigations.length - 1][0]);
      return;
    }

    // If current selection still exists and is active, keep it
    const currentTraces = investigations.find(([id]) => id === selectedTab)?.[1];
    if (currentTraces && !currentTraces.some(t => t.step_type === 'verdict')) {
      return; // Stay on active investigation
    }

    // Current tab completed — switch to newest active, or newest overall
    const newestActive = [...investigations].reverse().find(
      ([, alertTraces]) => !alertTraces.some(t => t.step_type === 'verdict')
    );
    if (newestActive) {
      setSelectedTab(newestActive[0]);
    } else {
      // All completed, stay on current if it exists, else newest
      const currentExists = investigations.some(([id]) => id === selectedTab);
      if (!currentExists) {
        setSelectedTab(investigations[investigations.length - 1][0]);
      }
    }
  }, [investigations, selectedTab]);

  // Auto-scroll when new traces arrive for the selected tab
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [traces, selectedTab]);

  // Get selected investigation's traces
  const selectedTraces = useMemo(() => {
    if (!selectedTab) return [];
    return investigations.find(([id]) => id === selectedTab)?.[1] || [];
  }, [investigations, selectedTab]);

  // Derive status info for the selected tab
  const selectedHasVerdict = selectedTraces.some(t => t.step_type === 'verdict');
  const selectedIsActive = selectedTraces.length > 0 && !selectedHasVerdict;

  // Get step progress for selected investigation
  const selectedStepInfo = useMemo(() => {
    if (selectedTraces.length === 0) return null;
    const last = selectedTraces[selectedTraces.length - 1];
    return parseStepNumber(last.content);
  }, [selectedTraces]);

  // Count active investigations
  const activeCount = useMemo(() => {
    return investigations.filter(
      ([, alertTraces]) => !alertTraces.some(t => t.step_type === 'verdict')
    ).length;
  }, [investigations]);

  const statusBadge = (
    <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium ${
      activeCount > 0
        ? 'bg-white/5 text-white/70 border border-white/10'
        : 'bg-gray-800/50 text-gray-500 border border-gray-700/30'
    }`}>
      <span className={`w-1.5 h-1.5 rounded-full ${activeCount > 0 ? 'bg-white animate-pulse' : 'bg-gray-600'}`} />
      {activeCount > 0
        ? (activeCount === 1
          ? (selectedIsActive && selectedStepInfo ? `Step ${selectedStepInfo.step}/${selectedStepInfo.total}` : 'ACTIVE')
          : `${activeCount} ACTIVE`)
        : 'IDLE'
      }
    </div>
  );

  // Render a single trace line
  const renderTrace = (trace: AgentTrace, i: number, isLast: boolean) => {
    const isVerdict = trace.step_type === 'verdict';
    const stepData = parseStepNumber(trace.content);
    const displayContent = stepData ? stepData.text : trace.content;

    // Special verdict card rendering
    if (isVerdict) {
      const verdict = parseVerdict(displayContent);
      const verdictColor = verdict ? getVerdictColor(verdict.action) : '#d1d5db';
      const scanClass = verdict?.action.includes('BLOCK') ? 'verdict-block'
        : verdict?.action.includes('CLEAR') ? 'verdict-clear' : 'verdict-flag';

      return (
        <div
          key={i}
          className={`mt-2 rounded-lg border overflow-hidden ${scanClass}`}
          style={{
            borderColor: `${verdictColor}30`,
            background: `linear-gradient(135deg, ${verdictColor}08 0%, transparent 100%)`,
            animation: isLast ? 'traceSlideIn 0.3s ease-out' : undefined,
          }}
        >
          <div className="flex items-center gap-2 px-3 py-2">
            <StepIcon type="verdict" />
            {verdict && (
              <span
                className="px-2 py-0.5 rounded text-[11px] font-bold tracking-wider"
                style={{
                  color: verdictColor,
                  background: `${verdictColor}15`,
                  border: `1px solid ${verdictColor}30`,
                }}
              >
                {verdict.action}
              </span>
            )}
            <span className="text-gray-400 text-[10px] font-mono ml-auto">
              {formatTime(trace.timestamp)}
            </span>
          </div>
          {verdict && (
            <div className="px-3 pb-2 space-y-1.5">
              <p className="text-gray-300 text-xs leading-relaxed">{verdict.reason}</p>
              {verdict.evidence.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {verdict.evidence.map((e, j) => (
                    <span key={j} className="text-[10px] px-1.5 py-0.5 rounded bg-gray-800/60 text-gray-400 border border-gray-700/30">
                      {e}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
          {!verdict && (
            <div className="px-3 pb-2">
              <p className="text-white text-xs font-medium leading-relaxed">{displayContent}</p>
            </div>
          )}
        </div>
      );
    }

    // Regular trace line with icon + color
    return (
      <div
        key={i}
        className="flex gap-2 py-0.5 px-1 rounded hover:bg-gray-800/20 transition-colors"
        style={{ animation: isLast ? 'traceSlideIn 0.3s ease-out' : undefined }}
      >
        <span className="text-gray-700 flex-shrink-0 w-14 text-[10px] pt-0.5">
          {formatTime(trace.timestamp)}
        </span>
        <span className="flex-shrink-0 pt-0.5" style={{ width: '18px' }}>
          <StepIcon type={trace.step_type} />
        </span>
        <span
          className="font-bold flex-shrink-0 text-[10px] pt-0.5 uppercase tracking-wider"
          style={{ color: getTraceColor(trace.step_type), width: '52px' }}
        >
          {getTracePrefix(trace.step_type)}
        </span>
        <span
          className="break-words leading-relaxed text-xs"
          style={{
            color: trace.step_type === 'tool_result' ? '#9ca3af' : '#d1d5db',
            fontStyle: trace.step_type === 'tool_result' ? 'italic' : undefined,
          }}
        >
          {trace.step_type === 'tool_result' ? renderToolResult(displayContent) : displayContent}
        </span>
      </div>
    );
  };

  // Highlight key-value pairs and keywords in tool results
  const renderToolResult = (content: string) => {
    return content.split('\n').map((line, i) => {
      if (/WARNING|NOT REGISTERED|FRAUD|IMPOSSIBLE/i.test(line)) {
        return (
          <span key={i} className="block">
            <span className="text-red-400">{line}</span>
          </span>
        );
      }
      if (/VERIFIED|FEASIBLE|REGISTERED|SAFE/i.test(line)) {
        return (
          <span key={i} className="block">
            <span className="text-white/70">{line}</span>
          </span>
        );
      }

      const kvMatch = line.match(/^(\s*[\w\s]+?):\s+(.+)$/);
      if (kvMatch) {
        return (
          <span key={i} className="block">
            <span className="text-gray-500">{kvMatch[1]}:</span>{' '}
            <span className="text-gray-300">{kvMatch[2]}</span>
          </span>
        );
      }

      return <span key={i} className="block">{line}</span>;
    });
  };

  return (
    <Tile title="Agent Brain Trace" className="col-span-2" accentColor="#ffffff" badge={statusBadge}>
      <div className="h-full flex flex-col">
        {/* Terminal window */}
        <div className="flex-1 rounded-xl overflow-hidden border border-gray-800/30 flex flex-col">
          {/* Terminal header bar */}
          <div className="flex items-center gap-2 px-3 py-2 bg-gray-900/80 border-b border-gray-800/30">
            <span className="text-[10px] text-gray-500 font-mono">
              aegis-agent {selectedIsActive ? <>&mdash; investigating {selectedTab?.slice(0, 16)}</> : <>&mdash; monitoring</>}
            </span>
            {selectedIsActive && selectedStepInfo && (
              <div className="ml-auto flex items-center gap-2">
                <div className="w-24 h-1 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${(selectedStepInfo.step / selectedStepInfo.total) * 100}%`,
                      background: 'linear-gradient(90deg, #ffffff, #a1a1aa)',
                    }}
                  />
                </div>
                <span className="text-[10px] text-white/40 font-mono">{selectedStepInfo.step}/{selectedStepInfo.total}</span>
              </div>
            )}
          </div>

          {/* Tab bar — only show when there are investigations */}
          {investigations.length > 0 && (
            <div className="flex items-center bg-gray-900/60 border-b border-gray-800/30 overflow-x-auto scrollbar-thin">
              {investigations.map(([alertId, alertTraces]) => {
                const isSelected = alertId === selectedTab;
                const hasVerdict = alertTraces.some(t => t.step_type === 'verdict');
                const verdictTrace = alertTraces.find(t => t.step_type === 'verdict');
                const verdict = verdictTrace ? parseVerdict(
                  parseStepNumber(verdictTrace.content)?.text || verdictTrace.content
                ) : null;
                const verdictColor = verdict ? getVerdictColor(verdict.action) : null;
                const isActiveInvestigation = !hasVerdict;

                return (
                  <button
                    key={alertId}
                    onClick={() => setSelectedTab(alertId)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-mono whitespace-nowrap border-b-2 transition-all flex-shrink-0 ${
                      isSelected
                        ? 'border-white/30 bg-white/5 text-gray-200'
                        : 'border-transparent text-gray-500 hover:text-gray-300 hover:bg-gray-800/30'
                    }`}
                  >
                    {/* Status dot */}
                    {isActiveInvestigation ? (
                      <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse flex-shrink-0" />
                    ) : (
                      <span
                        className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                        style={{ backgroundColor: verdictColor || '#6b7280' }}
                      />
                    )}

                    {/* Alert ID */}
                    <span>{alertId.replace('alert-', '').slice(0, 8)}</span>

                    {/* Verdict label */}
                    {hasVerdict && verdict && (
                      <span
                        className="text-[9px] px-1 py-0 rounded font-bold"
                        style={{
                          color: verdictColor || '#d1d5db',
                          background: `${verdictColor}15`,
                        }}
                      >
                        {verdict.action}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          )}

          {/* Console output — shows selected tab's traces only */}
          <div
            ref={scrollRef}
            className="flex-1 overflow-y-auto font-mono text-xs space-y-0.5 p-3 bg-[#0a0a10] scrollbar-thin"
          >
            {traces.length === 0 && (
              <div className="text-center py-6">
                <pre className="text-white/20 text-[10px] leading-tight">{`\u256D\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u256E
\u2502  AEGIS AUTONOMOUS AGENT v2.1    \u2502
\u2502  Model: Llama 3 \u00B7 ReAct Loop    \u2502
\u2502  Status: MONITORING             \u2502
\u2502  Awaiting anomaly trigger...    \u2502
\u2570\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u256F`}</pre>
                <div className="text-gray-600 mt-3 text-xs cursor-blink">
                  Ready
                </div>
              </div>
            )}

            {/* Render only selected tab's traces */}
            {selectedTraces.map((trace, i) =>
              renderTrace(trace, i, i === selectedTraces.length - 1)
            )}

            {/* Active investigation cursor */}
            {selectedIsActive && (
              <div className="flex gap-2 py-0.5 px-1">
                <span className="text-gray-700 flex-shrink-0 w-14 text-[10px]" />
                <span className="flex-shrink-0" style={{ width: '18px' }} />
                <span className="flex-shrink-0" style={{ width: '52px' }} />
                <span className="text-white cursor-blink" />
              </div>
            )}
          </div>

          {/* Bottom status bar */}
          <div className="flex items-center justify-between px-3 py-1.5 bg-gray-900/80 border-t border-gray-800/30 text-[10px] text-gray-600 font-mono">
            <span>
              {activeCount > 0 && <><span className="text-white/40">{activeCount} active</span> &middot; </>}
              {investigations.length} investigation{investigations.length !== 1 ? 's' : ''} &middot; {selectedTraces.length} traces
            </span>
            <span className={activeCount > 0 ? 'text-white/40' : ''}>{activeCount > 0 ? 'ACTIVE' : 'STANDBY'}</span>
          </div>
        </div>
      </div>
    </Tile>
  );
}
