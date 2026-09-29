"""LangChain tools for the fraud investigation agent"""

import random
import logging
from datetime import datetime, timedelta
from langchain_core.tools import tool

from backend.anomaly_detection.features import haversine

logger = logging.getLogger(__name__)

# In-memory data store for tool responses (populated by the producer or Nessie)
_account_histories = {}
_merchant_data = {}
_nessie_client = None


def set_account_history(account_id: str, transactions: list):
    _account_histories[account_id] = transactions


def set_merchant_data(merchant_id: str, data: dict):
    _merchant_data[merchant_id] = data


def set_nessie_client(client):
    """Set the Nessie API client for real-time data lookups"""
    global _nessie_client
    _nessie_client = client


@tool
def check_account_history(account_id: str) -> str:
    """Check the recent transaction history for a bank account. Returns the last 20 transactions."""
    history = _account_histories.get(account_id, [])
    if not history:
        # Generate realistic mock history as fallback
        history = _generate_mock_history(account_id)

    source = "Nessie API" if account_id in _account_histories and _nessie_client else "simulated"
    recent = history[-20:]
    lines = [f"Recent transactions for account {account_id} ({len(recent)} shown, source: {source}):"]
    total = 0.0
    for txn in recent:
        total += txn.get("amount", 0)
        # Show "City, STATE" for US (domestic), "City, COUNTRY" for international
        country = txn.get("country", "US")
        state = txn.get("state", "")
        if country == "US" and state:
            location = f"{txn.get('city', 'Unknown')}, {state} (US)"
        elif country == "US":
            location = f"{txn.get('city', 'Unknown')}, US"
        else:
            location = f"{txn.get('city', 'Unknown')}, {country} (INTERNATIONAL)"
        lines.append(
            f"  - {txn.get('timestamp', 'N/A')[:16]} | "
            f"${txn.get('amount', 0):.2f} | "
            f"{txn.get('merchant_name', 'Unknown')} | "
            f"{location}"
        )
    lines.append(f"\nAverage transaction: ${total / max(len(recent), 1):.2f}")
    lines.append(f"Total in period: ${total:.2f}")
    return "\n".join(lines)


@tool
def verify_merchant(merchant_id: str) -> str:
    """Verify a merchant's legitimacy and business details. Returns merchant information."""
    data = _merchant_data.get(merchant_id)
    if not data:
        data = _generate_mock_merchant(merchant_id)

    source = "Nessie API" if merchant_id in _merchant_data and _nessie_client else "simulated"
    lines = [f"Merchant Verification Report for {merchant_id} (source: {source}):"]
    lines.append(f"  Name: {data.get('name', 'Unknown')}")
    lines.append(f"  Category: {data.get('category', 'Unknown')}")
    # Show state for US merchants, country for international
    m_country = data.get("country", "Unknown")
    m_state = data.get("state", "")
    if m_country == "US" and m_state:
        lines.append(f"  Location: {data.get('city', 'Unknown')}, {m_state} (US - Domestic)")
    elif m_country == "US":
        lines.append(f"  Location: {data.get('city', 'Unknown')}, US (Domestic)")
    else:
        lines.append(f"  Location: {data.get('city', 'Unknown')}, {m_country} (INTERNATIONAL)")
    lines.append(f"  Registration Status: {data.get('registration', 'Not Found')}")
    lines.append(f"  Years in Business: {data.get('years', 'Unknown')}")
    lines.append(f"  Fraud Reports: {data.get('fraud_reports', 0)}")
    lines.append(f"  Trust Score: {data.get('trust_score', 'N/A')}/100")
    return "\n".join(lines)


@tool
def check_travel_feasibility(
    lat1: float, lon1: float, lat2: float, lon2: float, time_gap_minutes: float
) -> str:
    """Check if travel between two locations is physically possible given the time gap.
    Provide coordinates of both locations and the time difference in minutes."""
    distance_km = haversine(lat1, lon1, lat2, lon2)
    if time_gap_minutes <= 0:
        time_gap_minutes = 1

    required_speed_kmh = distance_km / (time_gap_minutes / 60)

    result = [f"Travel Feasibility Analysis:"]
    result.append(f"  Distance: {distance_km:.1f} km")
    result.append(f"  Time gap: {time_gap_minutes:.0f} minutes")
    result.append(f"  Required speed: {required_speed_kmh:.0f} km/h")

    if required_speed_kmh > 900:
        result.append(f"  Verdict: IMPOSSIBLE - Would require faster than commercial flight speed")
        result.append(f"  Max feasible with flight: ~900 km/h")
    elif required_speed_kmh > 300:
        result.append(f"  Verdict: HIGHLY UNLIKELY - Would require high-speed transport with no delays")
    elif required_speed_kmh > 120:
        result.append(f"  Verdict: UNLIKELY - Would require sustained highway speed with no stops")
    else:
        result.append(f"  Verdict: FEASIBLE - Normal travel speed")

    return "\n".join(result)


@tool
def get_account_risk_profile(account_id: str) -> str:
    """Get the overall risk profile and statistics for a bank account."""
    history = _account_histories.get(account_id, [])

    if history:
        # Calculate real stats from actual transaction history
        amounts = [t.get("amount", 0) for t in history]
        countries = [t.get("country", "US") for t in history]
        international_count = sum(1 for c in countries if c != "US")
        international_pct = round(international_count / max(len(countries), 1) * 100, 1)

        # Gather unique locations (show state for US, country for international)
        locations = list(set(
            f"{t.get('city', 'Unknown')}, {t.get('state', '')}" if t.get("country", "US") == "US" and t.get("state")
            else f"{t.get('city', 'Unknown')}, {t.get('country', 'US')}"
            for t in history[-20:]
        ))

        # Count previous anomalies
        anomaly_count = sum(
            1 for t in history
            if t.get("is_anomaly", False) or t.get("risk_score", 0) > 0.5
        )

        # Determine risk tier based on data
        if anomaly_count > 2 or international_pct > 30:
            risk_tier = "High"
        elif anomaly_count > 0 or international_pct > 15:
            risk_tier = "Medium"
        else:
            risk_tier = "Low"

        profile = {
            "total_transactions": len(history),
            "avg_transaction": round(sum(amounts) / max(len(amounts), 1), 2),
            "max_transaction": round(max(amounts), 2) if amounts else 0,
            "previous_fraud_alerts": anomaly_count,
            "international_txn_pct": international_pct,
            "usual_locations": locations[:5],
            "risk_tier": risk_tier,
        }
        source = "Nessie API" if _nessie_client else "cached data"
    else:
        # Fallback to generated profile
        num_txns = random.randint(50, 200)
        profile = {
            "total_transactions": num_txns,
            "avg_transaction": round(random.uniform(30, 150), 2),
            "max_transaction": round(random.uniform(200, 800), 2),
            "previous_fraud_alerts": random.randint(0, 2),
            "international_txn_pct": round(random.uniform(0, 15), 1),
            "usual_locations": ["Pittsburgh, PA", "Philadelphia, PA", "Cleveland, OH", "New York, NY"],
            "risk_tier": random.choice(["Low", "Low", "Low", "Medium", "Medium"]),
        }
        source = "simulated"

    lines = [f"Risk Profile for {account_id} (source: {source}):"]
    lines.append(f"  Total transactions: {profile['total_transactions']}")
    lines.append(f"  Avg transaction: ${profile['avg_transaction']:.2f}")
    lines.append(f"  Max transaction: ${profile['max_transaction']:.2f}")
    lines.append(f"  Previous fraud alerts: {profile['previous_fraud_alerts']}")
    lines.append(f"  International transactions: {profile['international_txn_pct']}%")
    lines.append(f"  Usual locations: {', '.join(profile['usual_locations'])}")
    lines.append(f"  Risk tier: {profile['risk_tier']}")
    return "\n".join(lines)


@tool
def run_kyc_check(account_id: str) -> str:
    """Run a KYC (Know Your Customer) identity fraud risk assessment for a bank account.
    Returns identity risk level, score, and any fraud flags detected."""
    from backend.kyc.engine import kyc_engine
    history = _account_histories.get(account_id, [])
    result = kyc_engine.assess(account_id, history)

    lines = [f"KYC Identity Risk Assessment for {account_id}:"]
    lines.append(f"  Risk Level: {result['risk_level'].upper()}")
    lines.append(f"  Risk Score: {result['risk_score']}/100")
    lines.append(f"  Location Familiar: {'Yes' if result['location_familiar'] else 'NO - NEW LOCATION'}")
    if result.get('familiar_locations'):
        lines.append(f"  Known Locations: {', '.join(loc.title() for loc in result['familiar_locations'][:6])}")
    lines.append(f"  Customer ID: {result['customer_id'] or 'Unknown'}")
    lines.append(f"  Flags:")
    for flag in result["flags"]:
        lines.append(f"    - {flag}")
    return "\n".join(lines)


@tool
def recommend_action(action: str, summary: str) -> str:
    """Submit your final recommendation for the investigated alert.
    action: must be one of 'BLOCK', 'FLAG_FOR_REVIEW', or 'CLEAR'
    summary: a detailed explanation of your decision"""
    valid_actions = ["BLOCK", "FLAG_FOR_REVIEW", "CLEAR"]
    if action.upper() not in valid_actions:
        return f"Invalid action. Must be one of: {valid_actions}"
    return f"RECOMMENDATION SUBMITTED: {action.upper()}\nSummary: {summary}"


@tool
def find_similar_transactions(account_id: str) -> str:
    """Search the vector database for transactions similar to the current alert being investigated.
    Finds past transactions with similar spending patterns, merchant types, locations, and risk profiles.
    Use this to identify fraud patterns or establish that this type of transaction is normal for the account."""
    from backend.vector_store.faiss_store import vector_store

    if not vector_store.enabled:
        return "Vector search is not available (FAISS not initialized)."

    # Get the most recent transaction for this account (the one under investigation)
    history = _account_histories.get(account_id, [])
    if not history:
        return f"No transaction data available for account {account_id} to search against."

    current_txn = history[-1]  # Most recent = the one being investigated

    # Query for similar anomalous transactions across ALL accounts
    similar_anomalies = vector_store.query_similar_transactions(
        query_txn=current_txn,
        top_k=5,
        exclude_id=current_txn.get("id"),
        anomalies_only=True,
    )

    # Also query for similar normal transactions for context
    similar_normal = vector_store.query_similar_transactions(
        query_txn=current_txn,
        top_k=3,
        exclude_id=current_txn.get("id"),
        anomalies_only=False,
    )

    if not similar_anomalies and not similar_normal:
        return (
            f"Vector Search Results for account {account_id}:\n"
            f"  No similar transactions found in the database yet. "
            f"(Database may still be building up transaction history.)"
        )

    lines = [f"Vector Similarity Search Results for account {account_id}:"]
    lines.append(f"  Query: {vector_store.transaction_to_text(current_txn)}")
    lines.append("")

    if similar_anomalies:
        lines.append(f"  SIMILAR FLAGGED TRANSACTIONS ({len(similar_anomalies)} found):")
        for i, s in enumerate(similar_anomalies, 1):
            lines.append(
                f"    {i}. [{s['similarity_score']:.0%} match] "
                f"${s.get('amount', 0):.2f} at {s.get('merchant_name', 'Unknown')} "
                f"in {s.get('city', '?')}, {s.get('country', '?')} "
                f"(risk: {s.get('risk_score', 0):.2f}, account: {s.get('account_id', '?')[:8]}...)"
            )
        lines.append("")

    if similar_normal:
        lines.append(f"  SIMILAR NORMAL TRANSACTIONS ({len(similar_normal)} found):")
        for i, s in enumerate(similar_normal, 1):
            lines.append(
                f"    {i}. [{s['similarity_score']:.0%} match] "
                f"${s.get('amount', 0):.2f} at {s.get('merchant_name', 'Unknown')} "
                f"in {s.get('city', '?')}, {s.get('country', '?')} "
                f"(risk: {s.get('risk_score', 0):.2f})"
            )

    # Add analysis hints for the agent
    if similar_anomalies and similar_anomalies[0]["similarity_score"] > 0.85:
        lines.append(
            "\n  WARNING: High similarity to past flagged transactions suggests a known fraud pattern."
        )
    elif similar_normal and similar_normal[0]["similarity_score"] > 0.9:
        lines.append(
            "\n  NOTE: Very similar to normal transactions on record. "
            "This spending pattern may be legitimate."
        )

    return "\n".join(lines)


@tool
def find_similar_investigations(account_id: str) -> str:
    """Search for similar past fraud investigations and their outcomes.
    Returns verdicts from previous cases that resemble the current alert.
    Use this to see how similar cases were resolved (BLOCK, FLAG_FOR_REVIEW, or CLEAR)."""
    from backend.vector_store.faiss_store import vector_store

    if not vector_store.enabled:
        return "Vector search is not available (FAISS not initialized)."

    history = _account_histories.get(account_id, [])
    if not history:
        return f"No transaction data available for account {account_id}."

    current_txn = history[-1]
    similar = vector_store.query_similar_investigations(current_txn, top_k=3)

    if not similar:
        return (
            f"Investigation Memory for account {account_id}:\n"
            f"  No similar past investigations found. This may be a novel case."
        )

    lines = [f"Similar Past Investigations ({len(similar)} found):"]
    for i, inv in enumerate(similar, 1):
        lines.append(
            f"  {i}. [{inv['similarity_score']:.0%} match] "
            f"Alert {inv.get('alert_id', '?')} - "
            f"${inv.get('amount', 0):.2f} at {inv.get('merchant_name', '?')} - "
            f"Verdict: {inv.get('verdict', '?')}"
        )
        if inv.get("summary"):
            lines.append(f"     Reason: {inv['summary'][:150]}")

    return "\n".join(lines)


@tool
def detect_fraud_ring(account_id: str) -> str:
    """Detect potential fraud ring connections by analyzing the graph database.
    Finds accounts that share suspicious merchant connections with the target account.
    Use this to identify coordinated fraud across multiple accounts."""
    from backend.graph.neo4j_client import graph_client

    if not graph_client.enabled:
        return "Graph database not available. Cannot perform fraud ring analysis."

    result = graph_client.detect_fraud_ring(account_id)

    if not result.get("ring_detected") and not result.get("accounts"):
        if result.get("reason"):
            return f"Fraud Ring Analysis for {account_id}:\n  {result['reason']}"
        return (
            f"Fraud Ring Analysis for {account_id}:\n"
            f"  No fraud ring connections detected.\n"
            f"  This account does not share suspicious merchant patterns with other accounts."
        )

    lines = [f"Fraud Ring Analysis for {account_id}:"]
    lines.append(f"  Ring Detected: {'YES' if result['ring_detected'] else 'NO'}")
    lines.append(f"  Ring Risk Score: {result.get('ring_risk_score', 0):.2f}")
    lines.append(f"  Connected Accounts: {result.get('connected_accounts', 0)}")
    lines.append("")

    for acc in result.get("accounts", []):
        shared = ", ".join(acc["shared_merchants"][:3])
        if len(acc["shared_merchants"]) > 3:
            shared += f" (+{len(acc['shared_merchants']) - 3} more)"
        lines.append(
            f"  - Account {acc['account_id'][:12]}... | "
            f"Shared merchants: {shared} | "
            f"Txns: {acc['transaction_count']} | "
            f"Anomalies: {acc['anomaly_count']}"
        )

    if result.get("ring_detected"):
        lines.append(
            "\n  WARNING: Accounts sharing multiple merchants with anomalous transactions "
            "may indicate a coordinated fraud ring."
        )

    return "\n".join(lines)


def _generate_mock_history(account_id: str) -> list:
    """Generate realistic transaction history for an account across multiple US cities"""
    # (name, category, city, state, country)
    merchants = [
        ("Starbucks", "Coffee", "Pittsburgh", "PA", "US"),
        ("Giant Eagle", "Grocery", "Pittsburgh", "PA", "US"),
        ("Shell Gas", "Gas", "Pittsburgh", "PA", "US"),
        ("Chipotle", "Restaurant", "Pittsburgh", "PA", "US"),
        ("Amazon", "Online", "Seattle", "WA", "US"),
        ("Target", "Retail", "Pittsburgh", "PA", "US"),
        ("Walmart", "Retail", "Philadelphia", "PA", "US"),
        ("Whole Foods", "Grocery", "New York", "NY", "US"),
        ("Home Depot", "Retail", "Cleveland", "OH", "US"),
        ("Costco", "Grocery", "Chicago", "IL", "US"),
    ]
    history = []
    base_time = datetime.utcnow() - timedelta(days=30)
    for i in range(20):
        m = random.choice(merchants)
        history.append({
            "merchant_name": m[0],
            "category": m[1],
            "city": m[2],
            "state": m[3],
            "country": m[4],
            "amount": round(random.uniform(5, 120), 2),
            "timestamp": (base_time + timedelta(days=i * 1.5, hours=random.randint(8, 20))).isoformat(),
        })
    _account_histories[account_id] = history
    return history


def _generate_mock_merchant(merchant_id: str) -> dict:
    """Generate mock merchant data — uses real ANOMALY_MERCHANTS info for suspicious IDs"""
    from backend.streaming.producer import ANOMALY_MERCHANTS

    # Build lookup from ANOMALY_MERCHANTS (keyed by id)
    anomaly_lookup = {m["id"]: m for m in ANOMALY_MERCHANTS}

    if merchant_id in anomaly_lookup:
        m = anomaly_lookup[merchant_id]
        data = {
            "name": m["name"],
            "category": m["category"],
            "city": m["city"],
            "state": m.get("state", ""),
            "country": m["country"],
            "registration": "NOT REGISTERED",
            "years": "< 1 year",
            "fraud_reports": random.randint(5, 25),
            "trust_score": random.randint(5, 25),
        }
    else:
        data = {
            "name": f"Verified Business #{merchant_id}",
            "category": "Retail",
            "city": "Pittsburgh",
            "state": "PA",
            "country": "US",
            "registration": "VERIFIED",
            "years": f"{random.randint(3, 20)} years",
            "fraud_reports": 0,
            "trust_score": random.randint(80, 98),
        }
    _merchant_data[merchant_id] = data
    return data
