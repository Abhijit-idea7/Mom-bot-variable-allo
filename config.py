"""
config.py
---------
Infrastructure configuration for the Dual Momentum Delivery Bot.

This file handles:
  1. Loading universe symbol lists from plain-text files
  2. Exposing environment variables (API keys, broker settings)
  3. Re-exporting strategy parameter constants from strategy.py

STRATEGY PARAMETERS → strategy.py   (lookback, skip, hold_buffer, etc.)
UNIVERSE SYMBOLS    → universe_nifty200.txt / universe_bees.txt
API CREDENTIALS     → .env / GitHub Actions secrets → accounts.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Import all strategy defaults so other modules can still do:
#   from config import MOMENTUM_LOOKBACK_DAYS, HARD_STOP_PCT, ...
# without needing to change their import lines.
from strategy import NIFTY200_DEFAULTS as _n200, BEES_DEFAULTS as _bees

load_dotenv()

_BASE_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Universe symbol loading  (edit universe_*.txt files directly on GitHub)
# ---------------------------------------------------------------------------

def _load_symbols(filename: str) -> list[str]:
    """Load NSE symbols from a plain-text universe file.

    Rules:
      - One symbol per line.
      - Everything after '#' on a line is a comment and is ignored.
      - Blank lines are ignored.
      - Returns [] (with a warning) if the file does not exist.
    """
    path = _BASE_DIR / filename
    if not path.exists():
        import logging
        logging.getLogger(__name__).warning(
            f"Universe file '{filename}' not found in {_BASE_DIR}. "
            f"Returning empty symbol list. "
            f"Create the file or restore it from the repository."
        )
        return []
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split("#")[0].strip()
        if s:
            symbols.append(s)
    return symbols


NIFTY200_UNIVERSE: list[str] = _load_symbols("universe_nifty200.txt")
BEES_UNIVERSE:     list[str] = _load_symbols("universe_bees.txt")


# ---------------------------------------------------------------------------
# Strategy parameters — re-exported from strategy.py for backward compat
#
# All values below are sourced from NIFTY200_DEFAULTS in strategy.py.
# To change them, edit strategy.py. Do NOT override them here.
# ---------------------------------------------------------------------------

MOMENTUM_LOOKBACK_DAYS      = _n200.lookback_days       # 252
SKIP_RECENT_DAYS            = _n200.skip_days            # 21
ABSOLUTE_MOMENTUM_THRESHOLD = _n200.abs_threshold        # -0.05
MARKET_REGIME_FILTER        = _n200.regime_filter        # True
REGIME_TICKER               = _n200.regime_ticker        # "^NSEI"
MIN_HISTORY_BARS            = _n200.min_history_bars     # 285
TOP_N_HOLD                  = _n200.top_n_hold           # 15
HOLD_BUFFER                 = _n200.hold_buffer          # 20
HARD_STOP_PCT               = _n200.hard_stop_pct        # 0.15
WEIGHTING_SCHEME            = _n200.weighting            # "GRADED"
TOTAL_CAPITAL_NIFTY200      = _n200.total_capital        # 1_500_000
TOTAL_CAPITAL_BEES          = _bees.total_capital        # 500_000
WEEKLY_RANK_STOP_NIFTY200   = _n200.weekly_rank_stop     # 25
WEEKLY_RANK_STOP_BEES       = _bees.weekly_rank_stop     # 9
TRANSACTION_COST            = _n200.transaction_cost     # 0.0025
RISK_FREE_RATE              = _n200.risk_free_rate       # 0.065

# Legacy position-size constant (kept for backward compat; use total_capital now)
PORTFOLIO_SIZE    = TOP_N_HOLD
POSITION_SIZE_INR = TOTAL_CAPITAL_NIFTY200 // TOP_N_HOLD   # ₹1,00,000


# ---------------------------------------------------------------------------
# Order settings — CNC Delivery via stocksdeveloper → Zerodha
# ---------------------------------------------------------------------------

EXCHANGE     = "NSE"
PRODUCT_TYPE = "DELIVERY"   # CNC delivery
ORDER_TYPE   = "MARKET"
VARIETY      = "REGULAR"
ORDER_TIME   = "15:00"      # Target: fire orders before 15:30 market close


# ---------------------------------------------------------------------------
# Stocksdeveloper webhook endpoint
# ---------------------------------------------------------------------------

STOCKSDEVELOPER_URL = "https://tv.stocksdeveloper.in/"

# Legacy single-account env vars — used as the "default" account fallback.
# For multi-account support, define accounts in accounts.json instead.
# Do NOT raise here — account validation happens in accounts.py at run time.
STOCKSDEVELOPER_API_KEY = os.getenv("STOCKSDEVELOPER_API_KEY")
STOCKSDEVELOPER_ACCOUNT = os.getenv("STOCKSDEVELOPER_ACCOUNT", "AbhiZerodha")


# ---------------------------------------------------------------------------
# Legacy file paths (kept for backward compat; use ucfg.positions_file)
# ---------------------------------------------------------------------------

POSITIONS_FILE = "positions.csv"
TRADE_LOG_FILE = "trade_log.csv"
