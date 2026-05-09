"""
trade_logger.py
---------------
Appends every BUY/SELL trade to a universe-specific trade_log CSV
which is committed to the repo after every rebalance.

CSV columns:
  date, symbol, action, price, quantity, value_inr,
  reason, momentum_return, rank, pnl_inr, pnl_pct

Pass trade_log_file=ucfg.trade_log_file to route to the correct CSV.
Defaults to the legacy TRADE_LOG_FILE from config for backward compatibility.
"""

import csv
import logging
from datetime import date
from pathlib import Path

from config import TRADE_LOG_FILE as _DEFAULT_TRADE_LOG_FILE

logger = logging.getLogger(__name__)

_FIELDS = [
    "date", "symbol", "action", "price", "quantity", "value_inr",
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
    """Log a BUY trade to the universe trade log."""
    _append_row(
        symbol          = symbol,
        action          = "BUY",
        price           = price,
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
    """Log a SELL trade to the universe trade log, computing realised P&L."""
    pnl_inr = round((price - entry_price) * quantity, 2)
    pnl_pct = round((price - entry_price) / entry_price * 100, 2) if entry_price > 0 else 0.0

    emoji = "OK" if pnl_inr >= 0 else "XX"
    logger.info(
        f"[TRADE] {emoji} SELL {symbol} x{quantity} | "
        f"entry=Rs{entry_price:.2f} exit=Rs{price:.2f} | "
        f"P&L=Rs{pnl_inr:+,.0f} ({pnl_pct:+.1f}%) | reason={reason}"
    )

    _append_row(
        symbol          = symbol,
        action          = "SELL",
        price           = price,
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
    quantity:        int,
    reason:          str,
    momentum_return: float,
    rank:            int,
    pnl_inr:         float,
    pnl_pct:         float,
    trade_log_file:  str = _DEFAULT_TRADE_LOG_FILE,
) -> None:
    path = Path(trade_log_file)
    file_exists = path.exists() and path.stat().st_size > 0
    value_inr = round(price * quantity, 2)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "date":            date.today().isoformat(),
            "symbol":          symbol,
            "action":          action,
            "price":           round(price, 2),
            "quantity":        quantity,
            "value_inr":       value_inr,
            "reason":          reason,
            "momentum_return": round(momentum_return, 4),
            "rank":            rank,
            "pnl_inr":         pnl_inr,
            "pnl_pct":         pnl_pct,
        })


def print_session_summary(session_buys: list[dict], session_sells: list[dict]) -> None:
    """Print a formatted summary of this month's rebalance actions."""
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
                f"    [{flag}] {t['symbol']:<14} x{t['quantity']:<5} "
                f"Rs{t['entry_price']:,.2f} -> Rs{t['price']:,.2f}  "
                f"P&L=Rs{t.get('pnl_inr', 0):+,.0f} ({t.get('pnl_pct', 0):+.1f}%)  "
                f"[{t['reason']}]"
            )

    if session_buys:
        logger.info("  BUYS:")
        for t in session_buys:
            logger.info(
                f"    [^] {t['symbol']:<14} x{t['quantity']:<5} "
                f"Rs{t['price']:,.2f}  12M-1M={t['momentum_return']:+.1%}  "
                f"rank=#{t['rank']}"
            )

    logger.info(sep)
