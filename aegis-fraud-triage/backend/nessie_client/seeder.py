"""Nessie API Data Seeder - Seeds the system with real banking data on startup"""

import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from backend.nessie_client.client import NessieClient
from backend.agent.tools import set_account_history, set_merchant_data
from backend.kyc.engine import set_customer_data, set_account_customer_mapping
from backend.streaming.producer import CATEGORY_MAP

logger = logging.getLogger(__name__)

# Merchant data cache: merchant_id -> merchant details (populated from Nessie)
_merchant_cache: Dict[str, dict] = {}


async def seed_nessie_data(client: NessieClient) -> Dict[str, Any]:
    """
    Fetch real accounts, merchants, and transaction data from Nessie API.
    Creates initial purchase history if accounts have none.
    Populates agent tools with real data for investigations.
    """
    result = {
        "account_ids": [],
        "accounts": [],
        "merchants": [],
    }

    # Step 1: Fetch existing accounts
    accounts = await client.get_accounts(limit=25)
    logger.info(f"Nessie: Found {len(accounts)} existing accounts")

    # Step 2: If no accounts exist, create test customers and accounts
    if not accounts:
        logger.info("Nessie: No accounts found, creating test customers...")
        accounts = await _create_test_data(client)

    if not accounts:
        logger.warning("Nessie: Could not get or create accounts")
        return result

    result["accounts"] = accounts
    result["account_ids"] = [acc.id for acc in accounts]

    # Step 3: Fetch merchants and build lookup cache
    merchants = await _fetch_merchants(client)
    result["merchants"] = merchants

    # Step 4: Fetch customer profiles for KYC and map accounts to customers
    seen_customers = set()
    for acc in accounts:
        if acc.user_id and acc.user_id not in seen_customers:
            seen_customers.add(acc.user_id)
            try:
                customer_data = await client.get_customer(acc.user_id)
                if customer_data:
                    set_customer_data(acc.user_id, customer_data)
                    logger.info(
                        f"Nessie: Loaded customer profile {acc.user_id} "
                        f"({customer_data.get('first_name', '')} {customer_data.get('last_name', '')})"
                    )
            except Exception as e:
                logger.warning(f"Nessie: Failed to fetch customer {acc.user_id}: {e}")

        if acc.user_id:
            set_account_customer_mapping(acc.id, acc.user_id)

    # Step 5: Fetch existing purchases, or seed initial history for checking accounts
    checking_accounts = [acc for acc in accounts if acc.account_type == "Checking"]
    for acc in checking_accounts:
        try:
            purchases = await client.get_purchases(acc.id, limit=30)
            if purchases:
                history = _purchases_to_history(purchases)
                set_account_history(acc.id, history)
                logger.info(f"Nessie: Loaded {len(history)} existing purchases for {acc.id}")
            else:
                # No purchases exist - seed initial history via the API
                logger.info(f"Nessie: No purchases for {acc.id}, seeding initial history...")
                history = await _seed_purchase_history(client, acc.id, count=15)
                if history:
                    set_account_history(acc.id, history)
                    logger.info(f"Nessie: Created {len(history)} seed purchases for {acc.id}")
        except Exception as e:
            logger.warning(f"Nessie: Failed to handle purchases for {acc.id}: {e}")

    logger.info(
        f"Nessie: Seeding complete - {len(result['account_ids'])} accounts, "
        f"{len(merchants)} merchants, {len(_merchant_cache)} cached"
    )
    return result


def _purchases_to_history(purchases) -> List[dict]:
    """Convert Nessie purchase objects to the format agent tools expect"""
    history = []
    for p in purchases:
        merchant_id = p.merchant_id or p.merchant or ""
        merchant_info = _merchant_cache.get(merchant_id, {})

        raw_cat = merchant_info.get("category", "Purchase")
        clean_cat = CATEGORY_MAP.get(raw_cat, raw_cat)

        history.append({
            "id": p.id,
            "merchant_name": merchant_info.get("name", merchant_id),
            "merchant_id": merchant_id,
            "category": clean_cat,
            "city": merchant_info.get("city", "Unknown"),
            "country": merchant_info.get("country", "US"),
            "amount": p.amount,
            "timestamp": p.date or "",
            "latitude": p.latitude or merchant_info.get("lat"),
            "longitude": p.longitude or merchant_info.get("lng"),
        })
    return history


async def _seed_purchase_history(
    client: NessieClient, account_id: str, count: int = 15
) -> List[dict]:
    """Create real purchases in Nessie to build initial transaction history"""
    if not _merchant_cache:
        return []

    merchant_ids = list(_merchant_cache.keys())
    history = []
    base_date = datetime.utcnow() - timedelta(days=30)

    for i in range(count):
        merchant_id = random.choice(merchant_ids)
        merchant_info = _merchant_cache[merchant_id]
        category = merchant_info.get("category", "Retail")
        category = CATEGORY_MAP.get(category, category)  # Normalize raw categories

        # Realistic amount based on category
        amount_ranges = {
            "Coffee": (3.0, 8.0),
            "Grocery": (20.0, 150.0),
            "Food": (5.0, 80.0),
            "Restaurant": (10.0, 50.0),
            "Retail": (15.0, 200.0),
            "Bar & Lounge": (8.0, 40.0),
            "Automotive": (50.0, 300.0),
            "Electronics": (50.0, 500.0),
            "Clothing": (20.0, 200.0),
            "Lodging": (80.0, 300.0),
            "Other": (10.0, 50.0),
        }
        amount_range = amount_ranges.get(category, (5.0, 100.0))
        amount = round(random.uniform(*amount_range), 2)

        # Spread purchases across the last 30 days
        purchase_date = (base_date + timedelta(days=i * 2, hours=random.randint(8, 20))).strftime("%Y-%m-%d")

        try:
            purchase = await client.create_purchase(
                account_id=account_id,
                merchant_id=merchant_id,
                amount=amount,
                purchase_date=purchase_date,
                description=f"{merchant_info.get('name', 'Purchase')} - {category}",
            )
            if purchase:
                history.append({
                    "id": purchase.get("_id", ""),
                    "merchant_name": merchant_info.get("name", merchant_id),
                    "merchant_id": merchant_id,
                    "category": category,
                    "city": merchant_info.get("city", "Unknown"),
                    "country": "US",
                    "amount": amount,
                    "timestamp": purchase_date,
                    "latitude": merchant_info.get("lat"),
                    "longitude": merchant_info.get("lng"),
                })
        except Exception as e:
            logger.warning(f"Nessie: Failed to create seed purchase: {e}")

    return history


async def _create_test_data(client: NessieClient) -> list:
    """Create test customers and accounts in Nessie"""
    test_customers = [
        {"first_name": "Alice", "last_name": "Johnson"},
        {"first_name": "Bob", "last_name": "Smith"},
        {"first_name": "Carol", "last_name": "Williams"},
        {"first_name": "David", "last_name": "Chen"},
        {"first_name": "Elena", "last_name": "Rodriguez"},
        {"first_name": "Frank", "last_name": "Patel"},
        {"first_name": "Grace", "last_name": "Kim"},
        {"first_name": "Hassan", "last_name": "Ali"},
    ]

    created_accounts = []
    for customer_info in test_customers:
        try:
            customer_id = await _create_customer(client, customer_info)
            if not customer_id:
                continue

            account = await _create_account(client, customer_id, "Checking", 5000.0)
            if account:
                created_accounts.append(account)

            account2 = await _create_account(client, customer_id, "Savings", 15000.0)
            if account2:
                created_accounts.append(account2)
        except Exception as e:
            logger.warning(f"Nessie: Failed to create test data for {customer_info}: {e}")

    return created_accounts


async def _create_customer(client: NessieClient, info: dict) -> Optional[str]:
    """Create a customer in Nessie and return the customer ID"""
    try:
        response = await client.client.post(
            f"{client.base_url}/customers",
            json={
                "first_name": info["first_name"],
                "last_name": info["last_name"],
                "address": {
                    "street_number": "123",
                    "street_name": "Forbes Ave",
                    "city": "Pittsburgh",
                    "state": "PA",
                    "zip": "15213",
                },
            },
            params={"key": client.api_key},
        )
        response.raise_for_status()
        data = response.json()
        customer_id = data.get("objectCreated", {}).get("_id")
        if customer_id:
            logger.info(f"Nessie: Created customer {info['first_name']} {info['last_name']} ({customer_id})")
        return customer_id
    except Exception as e:
        logger.warning(f"Nessie: Failed to create customer: {e}")
        return None


async def _create_account(client: NessieClient, customer_id: str, account_type: str, balance: float):
    """Create an account for a customer"""
    from backend.nessie_client.client import Account
    try:
        response = await client.client.post(
            f"{client.base_url}/customers/{customer_id}/accounts",
            json={
                "type": account_type,
                "nickname": f"{account_type} Account",
                "rewards": 0,
                "balance": balance,
            },
            params={"key": client.api_key},
        )
        response.raise_for_status()
        data = response.json()
        acc_data = data.get("objectCreated", {})
        if acc_data and acc_data.get("_id"):
            return Account(
                id=acc_data["_id"],
                user_id=customer_id,
                account_type=account_type,
                balance=balance,
            )
        return None
    except Exception as e:
        logger.warning(f"Nessie: Failed to create account: {e}")
        return None


async def _fetch_merchants(client: NessieClient) -> list:
    """Fetch merchants from Nessie and populate the merchant data store + cache"""
    try:
        response = await client.client.get(
            f"{client.base_url}/merchants",
            params={"key": client.api_key},
        )
        response.raise_for_status()
        merchants = response.json()

        # Use first 50 merchants for variety in the transaction feed
        for m in merchants[:50]:
            merchant_id = m.get("_id", "")
            geocode = m.get("geocode", {})
            address = m.get("address", {})
            categories = m.get("category", [])
            cat = categories[0] if isinstance(categories, list) and categories else (
                categories if isinstance(categories, str) else "Retail"
            )
            cat = CATEGORY_MAP.get(cat, cat)  # Normalize raw Nessie categories

            merchant_info = {
                "name": m.get("name", "Unknown"),
                "category": cat,
                "city": address.get("city", "Unknown"),
                "state": address.get("state", "PA"),
                "country": "US",
                "registration": "VERIFIED",
                "years": "Established",
                "fraud_reports": 0,
                "trust_score": 85,
                "lat": geocode.get("lat") if geocode else None,
                "lng": geocode.get("lng") if geocode else None,
            }

            set_merchant_data(merchant_id, merchant_info)
            _merchant_cache[merchant_id] = merchant_info

        logger.info(f"Nessie: Loaded {len(merchants[:50])} merchants")
        return merchants[:50]
    except Exception as e:
        logger.warning(f"Nessie: Failed to fetch merchants: {e}")
        return []
