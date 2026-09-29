"""Tools for the dispute investigation agent"""

import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Dispute history per account: account_id -> list of past dispute outcomes
_dispute_histories: dict[str, list] = {}

# Reference to alert/transaction buffers (set by server.py at startup)
_alert_buffer = None
_transaction_buffer = None


def set_dispute_buffers(alert_buf, txn_buf):
    """Wire in the server's alert and transaction buffers for context lookups"""
    global _alert_buffer, _transaction_buffer
    _alert_buffer = alert_buf
    _transaction_buffer = txn_buf


def record_dispute_outcome(account_id: str, outcome: dict):
    """Record a dispute outcome for history tracking"""
    if account_id not in _dispute_histories:
        _dispute_histories[account_id] = []
    _dispute_histories[account_id].append(outcome)


@tool
def get_dispute_context(transaction_id: str) -> str:
    """Get the original transaction details and any fraud alert/verdict associated with it.
    Helps understand whether this transaction was already flagged by the fraud system."""
    # Search transaction buffer
    txn_data = None
    if _transaction_buffer:
        for t in _transaction_buffer:
            if t.get("id") == transaction_id:
                txn_data = t
                break

    # Search alert buffer for related alert
    alert_data = None
    if _alert_buffer:
        for a in _alert_buffer:
            atxn = a.get("transaction", {})
            if atxn.get("id") == transaction_id:
                alert_data = a
                break

    lines = [f"Dispute Context for Transaction {transaction_id}:"]

    if txn_data:
        lines.append(f"  Merchant: {txn_data.get('merchant_name', 'Unknown')}")
        lines.append(f"  Amount: ${txn_data.get('amount', 0):.2f}")
        lines.append(f"  Category: {txn_data.get('category', 'Unknown')}")
        lines.append(f"  Location: {txn_data.get('city', 'Unknown')}, {txn_data.get('country', 'US')}")
        lines.append(f"  Timestamp: {txn_data.get('timestamp', 'Unknown')}")
        lines.append(f"  Risk Score: {txn_data.get('risk_score', 0):.3f}")
        lines.append(f"  Was Anomaly: {txn_data.get('is_anomaly', False)}")
    else:
        lines.append("  Transaction not found in recent buffer (may have scrolled out)")

    if alert_data:
        lines.append(f"\n  FRAUD ALERT FOUND:")
        lines.append(f"    Alert Status: {alert_data.get('status', 'unknown')}")
        lines.append(f"    Risk Score: {alert_data.get('risk_score', 0):.3f}")
        lines.append(f"    Risk Factors: {', '.join(alert_data.get('risk_factors', []))}")
        verdict = alert_data.get('agent_verdict')
        if verdict:
            lines.append(f"    Agent Verdict: {verdict[:200]}")
    else:
        lines.append(f"\n  No fraud alert associated with this transaction.")

    return "\n".join(lines)


@tool
def check_customer_dispute_history(account_id: str) -> str:
    """Check the customer's past dispute history - how many disputes filed, approval rate, any patterns."""
    history = _dispute_histories.get(account_id, [])

    lines = [f"Dispute History for Account {account_id}:"]
    lines.append(f"  Total Past Disputes: {len(history)}")

    if not history:
        lines.append("  No previous disputes on record.")
        lines.append("  Customer dispute profile: CLEAN")
        return "\n".join(lines)

    approved = sum(1 for d in history if d.get("action") == "APPROVE")
    denied = sum(1 for d in history if d.get("action") == "DENY")
    escalated = sum(1 for d in history if d.get("action") == "ESCALATE")
    approval_rate = approved / len(history) * 100 if history else 0

    lines.append(f"  Approved: {approved}")
    lines.append(f"  Denied: {denied}")
    lines.append(f"  Escalated: {escalated}")
    lines.append(f"  Approval Rate: {approval_rate:.0f}%")

    # Flag suspicious patterns
    if len(history) >= 3 and approval_rate < 30:
        lines.append(f"  WARNING: Frequent disputes with low approval rate - possible abuse pattern")
    elif len(history) >= 5:
        lines.append(f"  WARNING: High dispute frequency ({len(history)} disputes)")
    else:
        lines.append(f"  Customer dispute profile: NORMAL")

    # Show recent disputes
    for d in history[-3:]:
        lines.append(f"    - {d.get('reason', 'unknown')}: ${d.get('amount', 0):.2f} -> {d.get('action', 'unknown')}")

    return "\n".join(lines)


@tool
def find_duplicate_transactions(transaction_id: str, account_id: str, amount: float, merchant_name: str) -> str:
    """Search for duplicate charges: same merchant, same amount, same account.
    Use this when the customer claims a duplicate or double charge."""
    if not _transaction_buffer:
        return "Transaction buffer not available. Cannot search for duplicates."

    # Find the disputed transaction
    disputed_txn = None
    for t in _transaction_buffer:
        if t.get("id") == transaction_id:
            disputed_txn = t
            break

    # Search for matches: same account + same merchant + same amount (±$0.01)
    matches = []
    for t in _transaction_buffer:
        if t.get("id") == transaction_id:
            continue  # Skip the disputed transaction itself
        if (t.get("account_id") == account_id
                and t.get("merchant_name", "").lower() == merchant_name.lower()
                and abs(t.get("amount", 0) - amount) < 0.02):
            matches.append(t)

    lines = [f"Duplicate Transaction Search for {transaction_id}:"]
    lines.append(f"  Looking for: {merchant_name} | ${amount:.2f} | Account {account_id}")

    if matches:
        lines.append(f"\n  DUPLICATE FOUND: {len(matches)} matching transaction(s):")
        for m in matches[:5]:
            time_str = m.get("timestamp", "Unknown")[:19]
            lines.append(
                f"    - ID: {m.get('id', '?')} | ${m.get('amount', 0):.2f} | "
                f"{m.get('merchant_name', '?')} | {time_str}"
            )
        if disputed_txn:
            lines.append(f"\n  Disputed transaction timestamp: {disputed_txn.get('timestamp', '?')[:19]}")
            lines.append(f"  Duplicate transaction timestamp:  {matches[0].get('timestamp', '?')[:19]}")
        lines.append(f"\n  CONCLUSION: This appears to be a genuine duplicate charge.")
    else:
        lines.append(f"\n  NO DUPLICATE FOUND in recent transaction history.")
        lines.append(f"  No other transactions match this merchant + amount for this account.")
        lines.append(f"  CONCLUSION: No evidence of duplicate charge in available data.")

    return "\n".join(lines)


@tool
def resolve_dispute(action: str, summary: str) -> str:
    """Submit your final resolution for the customer dispute.
    action: must be one of 'APPROVE' (refund customer), 'DENY' (reject dispute), or 'ESCALATE' (needs human review)
    summary: detailed explanation of your decision"""
    valid_actions = ["APPROVE", "DENY", "ESCALATE"]
    if action.upper() not in valid_actions:
        return f"Invalid action. Must be one of: {valid_actions}"
    return f"DISPUTE RESOLUTION: {action.upper()}\nSummary: {summary}"
