"""
trade_logger.py
---------------
Appends every BUY/SELL trade to a universe-specific trade log CSV
which is committed to the repo after every rebalance.

CSV columns:
  date, symbol, action, price, entry_price, quantity, value_inr,
  reason, momentum_return, rank, pnl_inr, pnl_pct

Notes:
  - entry_price is stored on SELL rows so every row is self-contained
    (no cross-referencing required to compute P&L or hard-stop history).
  - value_inr is the gross notional value (price × quantity).
    It does not include brokerage / STT / exchange fees.
  - pnl_inr and pnl_pct use the stored entry_price (approximate fill price
    from yfinance at the time of the order, not the actual broker fill price).
"""

import csv
import logging
from datetime import date
from pathlib import Path

from config import TRADE_LOG_FILE as _DEFAULT_TRADE_LOG_FILE

logger = logging.getLogger(__name__)

_FIELDS = [
    "date", "symbol", "action", "price", "entry_price", "quantity", "value_inr",
    "reason", "momentum_return", "rank", "pnl_inr", "pnl_pct",
]


def log_buy(
    symbol:          str,
    price:           float,
    quantity:        int,
    momentum_return: float,
    rank:            int,
    reason:          str = "MOMENTUM_ENTRY",
    trade_log_file:  str = _DEFAULT_TRADE_LOG_FILE,
) -> None:
    """Append a BUY trade to the trade log."""
    _append_row(
        symbol          = symbol,
        action          = "BUY",
        price           = price,
        entry_price     = price,   # entry_price == price on a buy
        quantity        = quantity,
        reason          = reason,
        momentum_return = momentum_return,
        rank            = rank,
        pnl_inr         = 0.0,
        pnl_pct         = 0.0,
        trade_log_file  = trade_log_file,
    )


def log_sell(
    symbol:          str,
    price:           float,
    quantity:        int,
    entry_price:     float,
    momentum_return: float,
    rank:            int,
    reason:          str,
    trade_log_file:  str = _DEFAULT_TRADE_LOG_FILE,
) -> None:
    """Append a SELL trade to the trade log, computing realised P&L."""
    pnl_inr = round((price - entry_price) * quantity, 2)
    pnl_pct = round(
        (price - entry_price) / entry_price * 100, 2
    ) if entry_price > 0 else 0.0

    flag = "OK" if pnl_inr >= 0 else "XX"
    logger.info(
        f"[TRADE] {flag} SELL {symbol} ×{quantity} | "
        f"entry=Rs{entry_price:.2f}  exit=Rs{price:.2f} | "
        f"P&L=Rs{pnl_inr:+,.0f} ({pnl_pct:+.1f}%) | reason={reason}"
    )

    _append_row(
        symbol          = symbol,
        action          = "SELL",
        price           = price,
        entry_price     = entry_price,
        quantity        = quantity,
        reason          = reason,
        momentum_return = momentum_return,
        rank            = rank,
        pnl_inr         = pnl_inr,
        pnl_pct         = pnl_pct,
        trade_log_file  = trade_log_file,
    )


def _append_row(
    symbol:          str,
    action:          str,
    price:           float,
    entry_price:     float,
    quantity:        int,
    reason:          str,
    momentum_return: float,
    rank:            int,
    pnl_inr:         float,
    pnl_pct:         float,
    trade_log_file:  str = _DEFAULT_TRADE_LOG_FILE,
) -> None:
    path = Path(trade_log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    value_inr   = round(price * quantity, 2)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "date":            date.today().isoformat(),
            "symbol":          symbol,
            "action":          action,
            "price":           round(price, 2),
            "entry_price":     round(entry_price, 2),
            "quantity":        quantity,
            "value_inr":       value_inr,
            "reason":          reason,
            "momentum_return": round(momentum_return, 4),
            "rank":            rank,
            "pnl_inr":         pnl_inr,
            "pnl_pct":         pnl_pct,
        })


def print_session_summary(
    session_buys:  list[dict],
    session_sells: list[dict],
) -> None:
    """Log a formatted summary of this month's rebalance actions."""
    sep = "=" * 64
    logger.info(sep)
    logger.info(f"  MONTHLY REBALANCE SUMMARY — {date.today().isoformat()}")
    logger.info(sep)
    logger.info(f"  Sells executed : {len(session_sells)}")
    logger.info(f"  Buys executed  : {len(session_buys)}")

    if session_sells:
        total_pnl = sum(t.get("pnl_inr", 0) for t in session_sells)
        logger.info(f"  Realised P&L   : Rs{total_pnl:+,.0f}")
        logger.info("  SELLS:")
        for t in session_sells:
            flag = "+" if t.get("pnl_inr", 0) >= 0 else "-"
            logger.info(
                f"    [{flag}] {t['symbol']:<14} ×{t['quantity']:<5} "
                f"Rs{t['entry_price']:,.2f} → Rs{t['price']:,.2f}  "
                f"P&L=Rs{t.get('pnl_inr', 0):+,.0f} ({t.get('pnl_pct', 0):+.1f}%)  "
                f"[{t['reason']}]"
            )

    if session_buys:
        logger.info("  BUYS:")
        for t in session_buys:
            logger.info(
                f"    [^] {t['symbol']:<14} ×{t['quantity']:<5} "
                f"Rs{t['price']:,.2f}  12M-1M={t['momentum_return']:+.1%}  "
                f"rank=#{t['rank']}"
            )

    logger.info(sep)
