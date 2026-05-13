"""
Agent Tools — the concrete actions agents can take.

Product Discovery (1 tool):
  • search_snackstack_menu  – RAG semantic search

Sales Support (2 tools):
  • get_order_status        – order lookup by ID or email
  • escalate_to_human       – HITL ticket + optional email via Resend
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.config import get_logger
from src.data import ORDER_DATABASE, ORDER_BY_TRACKING, ORDER_BY_EMAIL
from src.rag import menu_vectorstore

import re

logger = get_logger("tools")


# ── helpers ──────────────────────────────────────────────────

def normalise_order_id(raw: str) -> str:
    """Accept 'ORD101', 'ORD-101', 'ord-101' → 'ORD-101'."""
    upper = raw.upper().strip()
    clean = upper.replace("ORD-", "").replace("ORD", "").strip()
    return f"ORD-{clean}"


def normalise_tracking_id(raw: str) -> str:
    """Extracts only the digits and wraps them in the SS###TRK format."""
    # Find all digits in the string and join them together
    numbers = "".join(re.findall(r"\d+", raw))

    if not numbers:
        return ""  # Or handle as an invalid ID

    return f"SS{numbers}TRK"
# ═══════════════════════════════════════════════════════════
#  PRODUCT DISCOVERY TOOL
# ═══════════════════════════════════════════════════════════

@tool
def search_snackstack_menu(query: str) -> str:
    """Search the SnackStack menu using semantic search (RAG).

    Args:
        query: natural-language search, e.g. "french fries"
    """
    logger.info("search_snackstack_menu query=%r", query)
    try:
        docs = menu_vectorstore.similarity_search(query, k=3)
        if not docs:
            return "No items found matching your query."
        results = "Found the following items:\n\n"
        for i, doc in enumerate(docs, 1):
            results += f"Item {i}:\n{doc.page_content}\n\n"
        return results
    except Exception as exc:
        logger.exception("Menu search failed")
        return f"Error searching menu: {exc}"


# ═══════════════════════════════════════════════════════════
#  SALES SUPPORT TOOLS
# ═══════════════════════════════════════════════════════════

@tool
def get_order_status(identifier: str) -> str:
    """Look up the current status of a customer order.

    Args:
        identifier: an order ID (e.g. "ORD101") OR a customer email address
    """
    logger.info("get_order_status  identifier=%r", identifier)
    order = None
    oid = None

    if "@" in identifier:
        order = ORDER_BY_EMAIL.get(identifier.lower().strip())
        if order: oid = order["order_id"]
        logger.info("searched by email")
    elif "ord" in identifier.lower():
        oid = normalise_order_id(identifier)
        order = ORDER_DATABASE.get(oid)
        logger.info("searched by order")
    else: # default to check for tracking #
        trk = normalise_tracking_id(identifier)
        order = ORDER_BY_TRACKING.get(trk)
        if order: oid = order["order_id"]
        logger.info("searched by tracking")

    if not order:
        return f"No order found for: {identifier}"


    info = (
        f"Order {oid}:\n"
        f"  Customer : {order['customer_name']} ({order['customer_email']})\n"
        f"  Item     : {order['item_name']}\n"
        f"  Price    : ₹{order['price']:,}\n"
        f"  Status   : {order['status']}\n"
        f"  Tracking : {order['tracking_id']}\n"
        f"  Ordered  : {order['order_date']}\n"
        f"  ETA      : {order['estimated_delivery']}"
    )

    logger.info(info)

    return info
