"""Pydantic models for API request/response serialization"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class TransactionEvent(BaseModel):
    id: str
    account_id: str
    merchant_name: str
    merchant_id: str
    amount: float
    currency: str = "USD"
    category: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: str = "US"
    timestamp: str
    risk_score: float = 0.0
    is_anomaly: bool = False


class AlertEvent(BaseModel):
    id: str
    transaction: TransactionEvent
    risk_score: float
    risk_factors: list[str]
    status: str = "pending"  # pending, investigating, blocked, cleared, awaiting_review
    agent_verdict: Optional[str] = None
    review_status: Optional[str] = None  # awaiting_review, confirmed, overridden
    human_override: Optional[str] = None  # blocked, cleared (the human's chosen action)
    human_reason: Optional[str] = None
    confidence_score: Optional[float] = None  # 0-100
    timestamp: str


class AlertReviewRequest(BaseModel):
    action: str  # "confirm" or "override"
    override_action: Optional[str] = None  # "blocked" or "cleared" (required if action=override)
    reason: Optional[str] = None


class AgentTrace(BaseModel):
    alert_id: str
    step_type: str  # thinking, tool_call, tool_result, action, verdict
    content: str
    timestamp: str


class DisputeEvent(BaseModel):
    id: str
    transaction_id: str
    account_id: str
    amount: float
    merchant_name: str
    reason: str  # unauthorized_charge, wrong_amount, never_received, duplicate, fraud_claim
    customer_statement: str
    status: str = "pending"  # pending → investigating → approved/denied/escalated
    agent_resolution: Optional[str] = None
    resolution_summary: Optional[str] = None
    timestamp: str
    resolved_at: Optional[str] = None


class DashboardStats(BaseModel):
    total_transactions: int = 0
    flagged_count: int = 0
    blocked_count: int = 0
    cleared_count: int = 0
    avg_risk_score: float = 0.0
    total_amount: float = 0.0
    money_saved: float = 0.0
    investigations_completed: int = 0
    disputes_filed: int = 0
    disputes_approved: int = 0
    disputes_denied: int = 0


class KYCResult(BaseModel):
    account_id: str
    customer_id: Optional[str] = None
    risk_level: str = "low"  # low, medium, high, critical
    risk_score: float = 0.0  # 0-100
    flags: list[str] = []
    location_familiar: bool = True
    familiar_locations: list[str] = []
    account_age_days: int = 30
    assessed_at: str = ""
