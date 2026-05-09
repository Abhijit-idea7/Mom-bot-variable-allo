"""
order_manager.py
----------------
Sends CNC delivery order webhooks to stocksdeveloper.in.

Identical webhook format to the intraday bot, but with:
  productType: "DELIVERY"   →  Zerodha CNC (Cash and Carry / Delivery)

Payload format:
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

import requests

from config import (
    EXCHANGE,
    ORDER_TYPE,
    POSITION_SIZE_INR,
    PRODUCT_TYPE,
    STOCKSDEVELOPER_ACCOUNT,
    STOCKSDEVELOPER_API_KEY,
    STOCKSDEVELOPER_URL,
    VARIETY,
)

logger = logging.getLogger(__name__)

_TIMEOUT = 15  # seconds


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
                "productType": PRODUCT_TYPE,   # "DELIVERY" = CNC delivery
                "quantity":    quantity,
            }
        ],
    }


def _send_webhook(payload: dict) -> bool:
    """POST to stocksdeveloper and return True on success."""
    params = {
        "apiKey":  STOCKSDEVELOPER_API_KEY,
        "account": STOCKSDEVELOPER_ACCOUNT,
        "group":   "false",
    }
    try:
        resp = requests.post(
            STOCKSDEVELOPER_URL,
            params=params,
            json=payload,
            timeout=_TIMEOUT,
        )
        order = payload["orders"][0]
        if resp.status_code == 200:
            logger.info(
                f"Webhook OK [{resp.status_code}] — "
                f"{order['tradeType']} {order['symbol']} × {order['quantity']} "
                f"[{order['productType']}]"
            )
            return True
        else:
            logger.error(
                f"Webhook FAILED [{resp.status_code}]: {resp.text} — "
                f"{order['tradeType']} {order['symbol']}"
            )
            return False
    except requests.RequestException as e:
        logger.error(f"Webhook exception: {e}")
        return False


def buy_delivery(symbol: str, quantity: int) -> bool:
    """Place a CNC BUY order for delivery."""
    if quantity < 1:
        logger.warning(f"{symbol}: quantity {quantity} invalid, skipping BUY.")
        return False
    payload = _build_payload(symbol, "BUY", quantity)
    logger.info(f"→ BUY (CNC) {symbol} × {quantity}")
    return _send_webhook(payload)


def sell_delivery(symbol: str, quantity: int) -> bool:
    """Place a CNC SELL order to exit a delivery position."""
    if quantity < 1:
        logger.warning(f"{symbol}: quantity {quantity} invalid, skipping SELL.")
        return False
    payload = _build_payload(symbol, "SELL", quantity)
    logger.info(f"→ SELL (CNC) {symbol} × {quantity}")
    return _send_webhook(payload)


def calculate_quantity(price: float, allocation_inr: float = POSITION_SIZE_INR) -> int:
    """Floor division of allocation by current price."""
    if price <= 0:
        return 0
    return int(allocation_inr // price)
