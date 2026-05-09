"""
momentum_scorer.py
------------------
Dual Momentum cross-sectional ranker.

Scores each stock by its 12-month return excluding the most recent 1 month
(the "skip-month" effect documented by Jegadeesh & Titman 1993 to avoid
short-term reversal noise). This is the single ranking signal — no composite
scoring, no sub-scores, no quality gates.

Formula:
  momentum_return = (Close[-(SKIP+1)] - Close[-(LOOKBACK+SKIP+1)])
                    / Close[-(LOOKBACK+SKIP+1)]

  Where SKIP    = 21  trading days (~1 month)
        LOOKBACK= 252 trading days (~12 months)

  → Measures the return from ~13 months ago to ~1 month ago.
  → current_price (Close[-1]) is fetched for order sizing only.
"""

import logging
from dataclasses import dataclass

import pandas as pd

from config import MOMENTUM_LOOKBACK_DAYS, SKIP_RECENT_DAYS

logger = logging.getLogger(__name__)

_REQUIRED_BARS = MOMENTUM_LOOKBACK_DAYS + SKIP_RECENT_DAYS + 5


@dataclass
class MomentumRank:
    symbol:           str
    momentum_return:  float   # 12M-1M total return — the sole ranking signal
    current_price:    float   # Latest close — for order sizing only
    rank:             int     # 1 = highest momentum in universe


def rank_universe(prices_df: pd.DataFrame) -> list[MomentumRank]:
    """
    Rank all stocks in prices_df by 12M-1M momentum return.
    Returns list sorted best → worst (rank 1 = best).
    Stocks with insufficient price history are silently excluded.
    """
    results = []

    for symbol in prices_df.columns:
        series = prices_df[symbol].dropna()
        if len(series) < _REQUIRED_BARS:
            continue

        try:
            end_idx   = -(SKIP_RECENT_DAYS + 1)                           # ~1 month ago
            start_idx = -(MOMENTUM_LOOKBACK_DAYS + SKIP_RECENT_DAYS + 1)  # ~13 months ago

            price_end   = float(series.iloc[end_idx])
            price_start = float(series.iloc[start_idx])
            current     = float(series.iloc[-1])
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

    logger.info(f"Ranked {len(results)} stocks by 12M-1M momentum")
    return results


def graded_allocation(rank: int, total_slots: int, total_capital: int) -> int:
    """
    Rank-based position sizing for graded weighting.

    Distributes total_capital across total_slots positions using linear decay:
      weight(rank) = total_slots + 1 - rank
      sum of all weights = total_slots * (total_slots + 1) / 2

    Rank 1 (best momentum) gets the largest slice; rank N gets the smallest.

    Args:
        rank:          Current rank of this stock (1 = best).
        total_slots:   Total number of portfolio slots (e.g. 15 for NIFTY200).
        total_capital: Total Rs allocated across all slots (e.g. 15,00,000).

    Returns:
        Rs allocation for this position (integer, floor division).

    Examples (NIFTY200: total_slots=15, total_capital=15,00,000):
        rank  1 → weight 15 → Rs 1,87,500
        rank  8 → weight  8 → Rs 1,00,000
        rank 15 → weight  1 → Rs    12,500
    """
    if rank < 1 or rank > total_slots:
        return 0
    weight       = total_slots + 1 - rank
    total_weight = total_slots * (total_slots + 1) // 2   # sum 1..N = N(N+1)/2
    return int(total_capital * weight / total_weight)


def print_ranked_table(ranked: list[MomentumRank], top_n: int = 25) -> None:
    """Log the top N ranked stocks in a formatted table."""
    logger.info(f"\n  {'Rank':<6} {'Symbol':<16} {'12M-1M Return':>14} {'Price (Rs)':>12}")
    logger.info("  " + "-" * 52)
    for r in ranked[:top_n]:
        logger.info(
            f"  {r.rank:<6} {r.symbol:<16} {r.momentum_return:>+13.1%} {r.current_price:>12.2f}"
        )
    if len(ranked) > top_n:
        logger.info(f"  ... ({len(ranked) - top_n} more stocks ranked below)")
