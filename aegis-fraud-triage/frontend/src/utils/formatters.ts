export function formatCurrency(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(amount);
}

export function formatTime(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

export function timeAgo(timestamp: string): string {
  const seconds = Math.floor((Date.now() - new Date(timestamp).getTime()) / 1000);
  if (seconds < 5) return 'just now';
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}

export function getRiskColor(score: number): string {
  if (score >= 0.7) return '#ef4444'; // red — critical
  if (score >= 0.4) return '#d1d5db'; // light gray — high
  if (score >= 0.2) return '#9ca3af'; // gray — medium
  return '#6b7280'; // dim gray — low risk
}

export function getRiskLevel(score: number): string {
  if (score >= 0.7) return 'Critical';
  if (score >= 0.4) return 'High';
  if (score >= 0.2) return 'Medium';
  return 'Low';
}

export function getTraceColor(stepType: string): string {
  switch (stepType) {
    case 'thinking': return '#d1d5db';  // light gray — AI reasoning
    case 'tool_call': return '#ffffff'; // white — AI action
    case 'tool_result': return '#6b7280'; // gray — data
    case 'action': return '#e5e7eb';    // light gray — decision
    case 'verdict': return '#ffffff';   // white — final verdict
    default: return '#9ca3af';
  }
}

export function getTracePrefix(stepType: string): string {
  switch (stepType) {
    case 'thinking': return 'THOUGHT';
    case 'tool_call': return 'ACTION';
    case 'tool_result': return 'RESULT';
    case 'action': return 'DECIDE';
    case 'verdict': return 'VERDICT';
    default: return '>';
  }
}

export function getVerdictColor(action: string): string {
  if (action.includes('BLOCK')) return '#ef4444';
  if (action.includes('CLEAR')) return '#22c55e';
  return '#d1d5db'; // FLAG_FOR_REVIEW — monochrome
}

export function parseVerdict(content: string): {
  action: string;
  reason: string;
  evidence: string[];
} | null {
  // Try structured format: "ACTION - reason. Evidence: item1, item2."
  const match = content.match(/(BLOCK|CLEAR|FLAG_FOR_REVIEW)\s*[-—]\s*(.*?)(?:Evidence:\s*(.*?))?$/s);
  if (match) {
    const evidence = match[3]
      ? match[3].split(',').map(e => e.trim().replace(/\.$/, '')).filter(Boolean)
      : [];
    return {
      action: match[1],
      reason: match[2].trim().replace(/\.\s*$/, ''),
      evidence,
    };
  }
  // Fallback: just detect the action
  for (const act of ['BLOCK', 'CLEAR', 'FLAG_FOR_REVIEW']) {
    if (content.toUpperCase().includes(act)) {
      return { action: act, reason: content.slice(0, 150), evidence: [] };
    }
  }
  return null;
}

export function stripMarkdown(text: string): string {
  return text
    .replace(/\*\*(.*?)\*\*/g, '$1')      // **bold**
    .replace(/\*(.*?)\*/g, '$1')           // *italic*
    .replace(/#{1,6}\s+/g, '')             // ### headings
    .replace(/^[-*+]\s+/gm, '')            // - bullet points
    .replace(/^\d+\.\s+/gm, '')            // 1. numbered lists
    .replace(/`([^`]+)`/g, '$1')           // `code`
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // [link](url)
    .replace(/\n{3,}/g, '\n\n')            // collapse excessive newlines
    .trim();
}

export function parseStepNumber(content: string): { step: number; total: number; text: string } | null {
  const match = content.match(/^\[(\d+)\/(\d+)\]\s*(.*)/s);
  if (match) {
    return { step: parseInt(match[1]), total: parseInt(match[2]), text: match[3] };
  }
  return null;
}
