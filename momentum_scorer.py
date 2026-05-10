"""
momentum_scorer.py
------------------
Dual Momentum cross-sectional ranker.

Scores each stock by its (lookback - skip) return, skipping the most recent
'skip' trading days to avoid short-term reversal (Jegadeesh & Titman 1993).

Formula:
  momentum_return = (Close[-(skip+1)] - Close[-(lookback+skip+1)])
                    / Close[-(lookback+skip+1)]

Defaults:
  skip     = 21  trading days (~1 month)
  lookback = 252 trading days (~12 months)

→ Measures the return from ~13 months ago to ~1 month ago.
→ current_price (Close[-1]) is fetched for order sizing only.

The backtest engine (backtest.py) calls rank_universe() with explicit lookback
and skip values so the same ranking code runs in both live and backtest modes.
"""

import logging
from dataclasses import dataclass

import pandas as pd

from strategy import NIFTY200_DEFAULTS as _defaults

logger = logging.getLogger(__name__)

_DEFAULT_LOOKBACK = _defaults.lookback_days   # 252
_DEFAULT_SKIP     = _defaults.skip_days       # 21


@dataclass
class MomentumRank:
    symbol:          str
    momentum_return: float   # 12M-1M total return — the sole ranking signal
    current_price:   float   # latest close — for order sizing only
    rank:            int     # 1 = highest momentum in universe


def rank_universe(
    prices_df: pd.DataFrame,
    lookback:  int = _DEFAULT_LOOKBACK,
    skip:      int = _DEFAULT_SKIP,
) -> list[MomentumRank]:
    """Rank all stocks in prices_df by (lookback − skip) momentum return.

    Accepts custom lookback and skip values so the same function can be used
    by both the live bot (default params) and the backtest (CLI-overridable params).

    Args:
        prices_df: DataFrame with dates as index, symbols as columns.
                   Typically a slice of the full price history ending at the
                   rebalance date (prices_df.iloc[:bar_idx+1] in the backtest).
        lookback:  Momentum window in trading days (default 252 = ~12 months).
        skip:      Days to skip at the right end (default 21 = ~1 month).

    Returns:
        List of MomentumRank sorted best → worst (rank 1 = best momentum).
        Stocks with insufficient history are silently excluded.
    """
    required = lookback + skip + 5
    results  = []

    for symbol in prices_df.columns:
        series = prices_df[symbol].dropna()
        if len(series) < required:
            continue

        try:
            end_idx   = -(skip + 1)                    # ~1 month ago
            start_idx = -(lookback + skip + 1)         # ~13 months ago

            price_end   = float(series.iloc[end_idx])
            price_start = float(series.iloc[start_idx])
            current     = float(series.iloc[-1])       # latest close for sizing
        except (IndexError, ValueError):
            continue

        if price_start <= 0 or current <= 0:
            continue

        mom = (price_end - price_start) / price_start
        results.append(MomentumRank(
            symbol          = symbol,
            momentum_return = mom,
            current_price   = current,
            rank            = 0,
        ))

    # Sort best → worst, assign ranks 1..N
    results.sort(key=lambda x: x.momentum_return, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1

    logger.info(
        f"Ranked {len(results)} stocks by {lookback}-{skip} momentum "
        f"(lookback={lookback}d, skip={skip}d)"
    )
    return results


def graded_allocation(rank: int, total_slots: int, total_capital: int) -> int:
    """Rank-weighted position sizing.

    Distributes total_capital across total_slots positions with linear decay:
      weight(rank) = total_slots + 1 − rank
      sum(weights) = total_slots * (total_slots + 1) / 2

    Rank 1 (best momentum) gets the largest slice; rank N gets the smallest.

    Args:
        rank:          Rank of this stock (1 = best, N = worst held).
        total_slots:   Total portfolio slots (e.g. 15 for NIFTY200).
        total_capital: Total Rs allocated across all slots.

    Returns:
        Integer Rs allocation for this position.

    Examples (NIFTY200: slots=15, capital=₹15L):
        rank  1 → weight 15/120 → ₹1,87,500
        rank  8 → weight  8/120 → ₹1,00,000
        rank 15 → weight  1/120 → ₹  12,500

    Examples (BEES: slots=5, capital=₹5L):
        rank  1 → weight 5/15 → ₹1,66,667
        rank  5 → weight 1/15 → ₹   33,333
    """
    if rank < 1 or rank > total_slots:
        return 0
    weight       = total_slots + 1 - rank
    total_weight = total_slots * (total_slots + 1) // 2   # N(N+1)/2
    return int(total_capital * weight / total_weight)


def print_ranked_table(ranked: list[MomentumRank], top_n: int = 25) -> None:
    """Log the top-N ranked stocks as a formatted table."""
    logger.info(f"\n  {'Rank':<6} {'Symbol':<16} {'12M-1M Return':>14} {'Price (Rs)':>12}")
    logger.info("  " + "-" * 52)
    for r in ranked[:top_n]:
        logger.info(
            f"  {r.rank:<6} {r.symbol:<16} {r.momentum_return:>+13.1%} "
            f"{r.current_price:>12.2f}"
        )
    if len(ranked) > top_n:
        logger.info(f"  ... ({len(ranked) - top_n} more stocks ranked below)")
