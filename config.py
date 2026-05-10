"""
config.py
---------
Dual Momentum Delivery Bot — Configuration.

Strategy: Gary Antonacci's Dual Momentum, adapted for Nifty 200.
  1. Absolute momentum  — Nifty 50 12M-1M return > 6% → risk-on, else → cash
  2. Relative momentum  — buy top 15 Nifty 200 stocks by 12M-1M return
  Rebalance: monthly (last trading day of each month)

Universe symbols are loaded from plain-text files (universe_nifty200.txt,
universe_bees.txt).  Edit those files directly on GitHub to update the universe
without touching any Python code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_BASE_DIR = Path(__file__).parent


def _load_symbols(filename: str) -> list[str]:
    """Load NSE symbols from a plain-text file.

    One symbol per line; lines starting with '#' or blank lines are ignored.
    Returns an empty list if the file does not exist (safe fallback).
    """
    path = _BASE_DIR / filename
    if not path.exists():
        return []
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split("#")[0].strip()
        if s:
            symbols.append(s)
    return symbols


# ---------------------------------------------------------------------------
# Universes  (symbols loaded from plain-text files — edit on GitHub directly)
# ---------------------------------------------------------------------------
NIFTY200_UNIVERSE: list[str] = _load_symbols("universe_nifty200.txt")
BEES_UNIVERSE:     list[str] = _load_symbols("universe_bees.txt")

# ---------------------------------------------------------------------------
# Dual Momentum Parameters
# ---------------------------------------------------------------------------
MOMENTUM_LOOKBACK_DAYS      = 252   # ~12 months of trading days
SKIP_RECENT_DAYS            = 21    # Skip last month (avoids short-term reversal)
ABSOLUTE_MOMENTUM_THRESHOLD = -0.05  # Nifty 50 12M-1M return must exceed -5% → risk-on
                                     # Only goes to cash in a genuine bear market (Nifty down >5% YoY)
                                     # Backtested 2018-2024: CAGR 40.9%, Sharpe 1.32, MaxDD -32.7%
TOP_N_HOLD                  = 15    # Maximum simultaneous open positions
HOLD_BUFFER                 = 20    # Keep holding if still in top HOLD_BUFFER (reduces churn)
MIN_HISTORY_BARS            = 285   # Minimum bars needed: 252 + 21 + 12 buffer

# ---------------------------------------------------------------------------
# Risk valve  (safety net — not part of original Dual Momentum)
# Protects against individual stock gap-downs between monthly rebalances.
# Applied on the monthly rebalance date only. Set to None to disable.
# ---------------------------------------------------------------------------
HARD_STOP_PCT = 0.15    # Sell if position down >15% from entry

# ---------------------------------------------------------------------------
# Portfolio / Position Sizing
# ---------------------------------------------------------------------------
PORTFOLIO_SIZE    = TOP_N_HOLD
POSITION_SIZE_INR = 100_000    # ₹1 lakh per stock (15 slots = ₹15 lakh)

# ---------------------------------------------------------------------------
# Graded Position Sizing  (rank-weighted allocation)
#
# Instead of equal ₹1L per slot, the capital is distributed by rank:
#   weight(rank) = (N + 1 - rank)   →  rank 1 gets N shares, rank N gets 1 share
#   allocation   = total_capital × weight / sum(1..N)
#
# NIFTY200 example (N=15, total=₹15L):
#   Rank  1 → 15/120 × 15L = ₹1,87,500   Rank  8 → 8/120 × 15L = ₹1,00,000
#   Rank 15 →  1/120 × 15L = ₹12,500
#
# BEES example (N=5, total=₹5L):
#   Rank 1 → 5/15 × 5L = ₹1,66,667    Rank 5 → 1/15 × 5L = ₹33,333
#
# Set WEIGHTING_SCHEME = "EQUAL" to revert to flat ₹POSITION_SIZE_INR per slot.
# ---------------------------------------------------------------------------
WEIGHTING_SCHEME       = "GRADED"   # "EQUAL" or "GRADED"
TOTAL_CAPITAL_NIFTY200 = 1_500_000  # 15 × ₹1L — same total, graded distribution
TOTAL_CAPITAL_BEES     = 500_000    #  5 × ₹1L

# ---------------------------------------------------------------------------
# Weekly Rank Stop  (rank-based mid-month exit in weekly_scan.py)
#
# Complements the price-based hard stop: if a held stock's momentum rank
# deteriorates sharply between monthly rebalances, exit immediately rather
# than waiting until month-end.  Threshold is wider than the monthly
# hold_buffer to avoid excess churn on small rank fluctuations.
#
# Set to None to disable the rank check in weekly_scan.py.
# ---------------------------------------------------------------------------
WEEKLY_RANK_STOP_NIFTY200 = 25   # monthly hold_buffer = 20;  weekly exit if rank > 25
WEEKLY_RANK_STOP_BEES     = 9    # monthly hold_buffer =  7;  weekly exit if rank >  9

# ---------------------------------------------------------------------------
# Market regime (uses REGIME_TICKER for absolute momentum check)
# ---------------------------------------------------------------------------
MARKET_REGIME_FILTER = True
REGIME_TICKER        = "^NSEI"

# ---------------------------------------------------------------------------
# File paths  (committed to repo after every rebalance)
# ---------------------------------------------------------------------------
POSITIONS_FILE = "positions.csv"
TRADE_LOG_FILE = "trade_log.csv"

# ---------------------------------------------------------------------------
# Order settings — NORMAL = CNC Delivery in Zerodha via stocksdeveloper
# ---------------------------------------------------------------------------
EXCHANGE     = "NSE"
PRODUCT_TYPE = "DELIVERY"  # CNC delivery — stocksdeveloper productType for Zerodha CNC
ORDER_TYPE   = "MARKET"
VARIETY      = "REGULAR"
ORDER_TIME   = "15:00"     # Target: fire orders before 15:30 market close

# ---------------------------------------------------------------------------
# Stocksdeveloper Webhook  (same endpoint as intraday bot)
# ---------------------------------------------------------------------------
STOCKSDEVELOPER_URL     = "https://tv.stocksdeveloper.in/"
# Legacy single-account env vars — still used as the "default" account fallback.
# For multi-account support, define accounts in accounts.json instead.
STOCKSDEVELOPER_API_KEY = os.getenv("STOCKSDEVELOPER_API_KEY")
STOCKSDEVELOPER_ACCOUNT = os.getenv("STOCKSDEVELOPER_ACCOUNT", "AbhiZerodha")
