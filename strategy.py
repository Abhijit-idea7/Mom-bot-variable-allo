"""
strategy.py
-----------
SINGLE SOURCE OF TRUTH for all Dual Momentum strategy parameters.

Both live trading (main.py, weekly_scan.py) and the backtest engine
(backtest.py) import parameter defaults exclusively from here.
Universe-specific overrides are defined at the bottom of this file and
consumed by universes.py.

THE RULE: if a strategy parameter exists anywhere else in the codebase,
that is a bug. Change it here; everywhere picks it up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _replace
from typing import Optional


@dataclass(frozen=True)
class StrategyParams:
    """All tunable Dual Momentum parameters in one immutable dataclass.

    Instances are frozen after creation.  Use .replace(**kwargs) to
    produce a modified copy for a specific universe, tranche, or CLI override.
    """

    # ── Momentum calculation ───────────────────────────────────────────────
    lookback_days:    int   = 252
    """~12 months of trading days for the momentum window."""

    skip_days:        int   = 21
    """Skip last ~1 month to avoid short-term reversal (Jegadeesh & Titman 1993)."""

    abs_threshold:    float = -0.05
    """Nifty 50 12M-1M return must exceed this → risk-on.
    -0.05 = -5%: only go to cash in a genuine bear market.
    Original Antonacci used 0.0; -0.05 reduces whipsaw."""

    regime_filter:    bool  = True
    """Enable absolute momentum (risk-off) check against the benchmark."""

    regime_ticker:    str   = "^NSEI"
    """Yahoo Finance ticker for the absolute momentum benchmark (Nifty 50)."""

    min_history_bars: int   = 285
    """Minimum bars required before a stock is rankable: lookback + skip + buffer."""

    # ── Portfolio construction ─────────────────────────────────────────────
    top_n_hold:       int   = 15
    """Maximum simultaneous open positions."""

    hold_buffer:      int   = 20
    """Monthly rebalance: keep holding if rank <= this (reduces churn).
    Must be >= top_n_hold."""

    weekly_rank_stop: Optional[int] = 25
    """Weekly scan: exit if rank > this mid-month.
    Should be >= hold_buffer to avoid firing on every weekly check.
    None = disable the weekly rank-based exit."""

    # ── Risk management ────────────────────────────────────────────────────
    hard_stop_pct:    float = 0.15
    """Exit if position is down more than this fraction from entry price.
    Checked at monthly rebalance AND weekly scan. 0.15 = 15%."""

    warn_pct:         float = 0.10
    """Log a warning if position is down more than this fraction (approaching stop).
    Defaults to ~2/3 of hard_stop_pct."""

    # ── Position sizing ────────────────────────────────────────────────────
    weighting:        str   = "GRADED"
    """Position sizing scheme: 'GRADED' (rank-weighted) or 'EQUAL' (flat per slot).

    GRADED: weight(rank) = N+1−rank; rank 1 gets the largest allocation.
    Total weights sum to N*(N+1)/2 so the full capital is always deployed.

    EQUAL: every slot gets total_capital / top_n_hold (simple flat sizing)."""

    total_capital:    int   = 1_500_000
    """Total rupees allocated to this universe/tranche.
    For GRADED weighting this is the denominator of the allocation formula.
    For EQUAL weighting this is divided by top_n_hold to get per-slot size.
    Override per-tranche with --capital when running the workflow."""

    # ── Backtest / performance analysis ────────────────────────────────────
    transaction_cost: float = 0.0025
    """One-way cost fraction applied in the backtest (buy AND sell).
    0.0025 = 0.25% covers brokerage, STT, exchange charges, and slippage."""

    risk_free_rate:   float = 0.065
    """Annualised risk-free rate used for Sharpe ratio calculation.
    6.5% approximates the Indian 10-year G-Sec yield / liquid fund return."""

    def replace(self, **kwargs: object) -> "StrategyParams":
        """Return a modified copy (thin wrapper around dataclasses.replace)."""
        return _replace(self, **kwargs)

    def __post_init__(self) -> None:
        """Validate parameter relationships on construction."""
        if self.hold_buffer < self.top_n_hold:
            raise ValueError(
                f"hold_buffer ({self.hold_buffer}) must be >= top_n_hold ({self.top_n_hold}). "
                f"Otherwise the monthly rebalance would sell stocks it just bought."
            )
        if self.weekly_rank_stop is not None and self.weekly_rank_stop < self.hold_buffer:
            raise ValueError(
                f"weekly_rank_stop ({self.weekly_rank_stop}) should be >= hold_buffer "
                f"({self.hold_buffer}). A tighter weekly stop fires more often than the "
                f"monthly hold buffer, making it the effective exit trigger instead."
            )
        if not self.weighting.upper() in ("GRADED", "EQUAL"):
            raise ValueError(f"weighting must be 'GRADED' or 'EQUAL', got '{self.weighting}'")
        if self.hard_stop_pct <= 0 or self.hard_stop_pct >= 1:
            raise ValueError(f"hard_stop_pct must be between 0 and 1, got {self.hard_stop_pct}")


# ── Universe-specific parameter defaults ──────────────────────────────────────
#
# These are the ONLY place where NIFTY200 and BEES differ in strategy.
# All other parameters inherit the StrategyParams defaults above.

NIFTY200_DEFAULTS = StrategyParams(
    # 180+ stocks → hold top 15, buffer to 20, weekly stop at 25
    top_n_hold       = 15,
    hold_buffer      = 20,
    weekly_rank_stop = 25,
    total_capital    = 1_500_000,   # ₹15 lakh default
    warn_pct         = 0.10,        # warn at ~10% loss (2/3 of 15% hard stop)
)

BEES_DEFAULTS = StrategyParams(
    # 18 ETFs → hold top 5, buffer to 7, weekly stop at 9
    top_n_hold       = 5,
    hold_buffer      = 7,
    weekly_rank_stop = 9,
    total_capital    = 500_000,     # ₹5 lakh default
    warn_pct         = 0.10,
)
