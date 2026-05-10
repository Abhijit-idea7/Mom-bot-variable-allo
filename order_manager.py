"""
order_manager.py
----------------
Sends CNC delivery order webhooks to stocksdeveloper.in.

Features:
  - Up to 3 attempts per order with exponential back-off (5s, 15s, 30s).
    Retries on network errors and 429/503 responses only; other HTTP errors
    (4xx except 429) fail immediately (no point retrying a bad request).
  - HTTP 200 is treated as success at the transport layer.
    stocksdeveloper does not return a structured order-ID in the body;
    actual broker acceptance/rejection is confirmed via the broker's order book.
  - Account credentials are passed as an AccountConfig object, never read
    from module-level globals — enabling multi-account trading.

Webhook payload:
{
    "command": "PLACE_ORDERS",
    "orders": [{
        "variety":     "REGULAR",
        "exchange":    "NSE",
        "symbol":      "TATAMOTORS",
        "tradeType":   "BUY" | "SELL",
        "orderType":   "MARKET",
        "productType": "DELIVERY",
        "quantity":    50
    }]
}
"""

import logging
import time

import requests

from accounts import AccountConfig
from config import (
    EXCHANGE,
    ORDER_TYPE,
    POSITION_SIZE_INR,
    PRODUCT_TYPE,
    STOCKSDEVELOPER_URL,
    VARIETY,
)

logger = logging.getLogger(__name__)

_TIMEOUT      = 15   # seconds per attempt
_MAX_RETRIES  = 3
_RETRY_DELAYS = (5, 15, 30)   # seconds between attempts (exponential back-off)
_RETRY_STATUS = {429, 503, 502, 504}   # transient HTTP errors worth retrying


def _build_payload(symbol: str, trade_type: str, quantity: int) -> dict:
    return {
        "command": "PLACE_ORDERS",
        "orders": [
            {
                "variety":     VARIETY,
                "exchange":    EXCHANGE,
                "symbol":      symbol,
                "tradeType":   trade_type,
                "orderType":   ORDER_TYPE,
                "productType": PRODUCT_TYPE,
                "quantity":    quantity,
            }
        ],
    }


def _send_webhook(payload: dict, account: AccountConfig) -> bool:
    """POST to stocksdeveloper with up to _MAX_RETRIES attempts.

    Retries on network errors and transient HTTP status codes (429/502/503/504).
    Returns True on HTTP 200; False on permanent failure or exhausted retries.
    """
    params = {
        "apiKey":  account.api_key,
        "account": account.account_id,
        "group":   "false",
    }
    order = payload["orders"][0]
    label = f"{order['tradeType']} {order['symbol']} ×{order['quantity']} [{account.name}]"

    last_exc: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.post(
                STOCKSDEVELOPER_URL,
                params  = params,
                json    = payload,
                timeout = _TIMEOUT,
            )

            if resp.status_code == 200:
                logger.info(
                    f"Webhook OK [{resp.status_code}] — {label} "
                    f"(attempt {attempt}/{_MAX_RETRIES})"
                )
                return True

            if resp.status_code in _RETRY_STATUS and attempt < _MAX_RETRIES:
                delay = _RETRY_DELAYS[attempt - 1]
                logger.warning(
                    f"Webhook transient error [{resp.status_code}] — {label}. "
                    f"Retrying in {delay}s... (attempt {attempt}/{_MAX_RETRIES})"
                )
                time.sleep(delay)
                continue

            # Permanent HTTP error (4xx except 429, 5xx except above)
            logger.error(
                f"Webhook FAILED [{resp.status_code}] — {label}: {resp.text[:200]}"
            )
            return False

        except requests.RequestException as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                delay = _RETRY_DELAYS[attempt - 1]
                logger.warning(
                    f"Webhook exception — {label}: {exc}. "
                    f"Retrying in {delay}s... (attempt {attempt}/{_MAX_RETRIES})"
                )
                time.sleep(delay)
                continue

    logger.error(
        f"Webhook FAILED after {_MAX_RETRIES} attempts — {label}. "
        f"Last error: {last_exc}. "
        f"CHECK BROKER ORDER BOOK MANUALLY before next rebalance."
    )
    return False


def buy_delivery(symbol: str, quantity: int, account: AccountConfig) -> bool:
    """Place a CNC BUY order via stocksdeveloper webhook."""
    if quantity < 1:
        logger.warning(f"{symbol}: quantity {quantity} < 1, skipping BUY.")
        return False
    payload = _build_payload(symbol, "BUY", quantity)
    logger.info(f"→ BUY  (CNC) {symbol} ×{quantity} [{account.name}]")
    return _send_webhook(payload, account)


def sell_delivery(symbol: str, quantity: int, account: AccountConfig) -> bool:
    """Place a CNC SELL order via stocksdeveloper webhook."""
    if quantity < 1:
        logger.warning(f"{symbol}: quantity {quantity} < 1, skipping SELL.")
        return False
    payload = _build_payload(symbol, "SELL", quantity)
    logger.info(f"→ SELL (CNC) {symbol} ×{quantity} [{account.name}]")
    return _send_webhook(payload, account)


def calculate_quantity(price: float, allocation_inr: float = POSITION_SIZE_INR) -> int:
    """Return the maximum whole shares buyable for the given rupee allocation."""
    if price <= 0:
        return 0
    return int(allocation_inr // price)
