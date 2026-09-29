export interface TransactionEvent {
  id: string;
  account_id: string;
  merchant_name: string;
  merchant_id: string;
  amount: number;
  currency: string;
  category: string;
  latitude: number | null;
  longitude: number | null;
  city: string | null;
  state?: string | null;
  country: string;
  timestamp: string;
  risk_score: number;
  is_anomaly: boolean;
}

export interface AlertEvent {
  id: string;
  transaction: TransactionEvent;
  risk_score: number;
  risk_factors: string[];
  status: 'pending' | 'investigating' | 'blocked' | 'cleared' | 'flagged' | 'awaiting_review';
  agent_verdict: string | null;
  review_status: 'awaiting_review' | 'resolved' | null;
  human_override: string | null;
  human_reason: string | null;
  confidence_score: number | null;
  timestamp: string;
}

export interface AgentTrace {
  alert_id: string;
  step_type: 'thinking' | 'tool_call' | 'tool_result' | 'action' | 'verdict';
  content: string;
  timestamp: string;
}

export interface DisputeEvent {
  id: string;
  transaction_id: string;
  account_id: string;
  amount: number;
  merchant_name: string;
  reason: 'unauthorized_charge' | 'wrong_amount' | 'never_received' | 'duplicate' | 'fraud_claim';
  customer_statement: string;
  status: 'pending' | 'investigating' | 'approved' | 'denied' | 'escalated';
  agent_resolution: string | null;
  resolution_summary: string | null;
  timestamp: string;
  resolved_at: string | null;
}

export interface DashboardStats {
  total_transactions: number;
  flagged_count: number;
  blocked_count: number;
  cleared_count: number;
  avg_risk_score: number;
  total_amount: number;
  money_saved: number;
  investigations_completed: number;
  disputes_filed: number;
  disputes_approved: number;
  disputes_denied: number;
}

export interface KYCResult {
  account_id: string;
  customer_id: string | null;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  risk_score: number;
  flags: string[];
  location_familiar: boolean;
  familiar_locations: string[];
  account_age_days: number;
  assessed_at: string;
}

// Fraud network visualization types (from /api/graph/network)
export interface GraphNode {
  id: string;
  type: 'account' | 'merchant';
  label: string;
  // Account fields
  status?: string;           // blocked | flagged | awaiting_review
  alert_count?: number;
  total_amount?: number;
  // Merchant fields
  merchant_id?: string;
  category?: string;
  city?: string;
  country?: string;
  fraud_count?: number;
  total_fraud_amount?: number;
  shared_accounts?: number;
  is_ring_node?: boolean;
}

export interface GraphEdge {
  source: string;
  target: string;
  amount: number;
  status: string;             // blocked | flagged | awaiting_review
  confidence: number | null;
  risk_score: number;
  alert_id: string;
}

export interface GraphNetworkResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  total_accounts: number;
  total_merchants: number;
  ring_merchants: number;
}
