"""
universes.py
------------
Universe registry for the Dual Momentum Delivery Bot.

Each universe bundles:
  - Symbol list          (loaded from universe_*.txt)
  - StrategyParams       (all tunable parameters for that universe)
  - State file paths     (positions CSV, trade-log CSV)

The StrategyParams inside UniverseConfig is the authoritative source for every
strategy number used in live trading and backtesting for that universe.
There are no separately-hardcoded copies of hold_buffer, top_n_hold, etc.

Select universe at runtime:  --universe NIFTY200  or  --universe BEES

File naming convention for multi-account / multi-tranche:
  account=default, tranche=     → positions_nifty200.csv
  account=default, tranche=T1   → positions_nifty200_t1.csv
  account=abhi2,   tranche=     → positions_nifty200_abhi2.csv
  account=abhi2,   tranche=T1   → positions_nifty200_abhi2_t1.csv
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace

from config import BEES_UNIVERSE, NIFTY200_UNIVERSE
from strategy import BEES_DEFAULTS, NIFTY200_DEFAULTS, StrategyParams


# ── Universe config ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class UniverseConfig:
    """All configuration for one trading universe (NIFTY200 or BEES).

    Strategy parameters are embedded in `params` — a frozen StrategyParams
    instance.  Use the convenience properties (.top_n_hold, .hold_buffer, etc.)
    to access them; they delegate to params so all code that already uses
    ucfg.top_n_hold continues to work without modification.
    """

    name:           str             # "NIFTY200" | "BEES"
    display_name:   str             # human-readable label
    symbols:        tuple           # immutable tuple of NSE symbols
    positions_file: str             # CSV tracking open positions
    trade_log_file: str             # CSV recording every BUY/SELL
    params:         StrategyParams  # ALL tunable strategy parameters

    # ── Convenience properties (delegate to params) ───────────────────────
    # These keep backward compatibility with all existing call sites that
    # use ucfg.top_n_hold, ucfg.hold_buffer, ucfg.total_capital, etc.

    @property
    def top_n_hold(self) -> int:
        return self.params.top_n_hold

    @property
    def hold_buffer(self) -> int:
        return self.params.hold_buffer

    @property
    def total_capital(self) -> int:
        return self.params.total_capital

    @property
    def weekly_rank_stop(self):
        return self.params.weekly_rank_stop

    @property
    def weighting(self) -> str:
        return self.params.weighting

    @property
    def hard_stop_pct(self) -> float:
        return self.params.hard_stop_pct


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, UniverseConfig] = {
    "NIFTY200": UniverseConfig(
        name           = "NIFTY200",
        display_name   = "Nifty 200 Stocks",
        symbols        = tuple(NIFTY200_UNIVERSE),
        positions_file = "positions_nifty200.csv",
        trade_log_file = "trade_log_nifty200.csv",
        params         = NIFTY200_DEFAULTS,
    ),
    "BEES": UniverseConfig(
        name           = "BEES",
        display_name   = "BEES ETFs",
        symbols        = tuple(BEES_UNIVERSE),
        positions_file = "positions_bees.csv",
        trade_log_file = "trade_log_bees.csv",
        params         = BEES_DEFAULTS,
    ),
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_universe(name: str) -> UniverseConfig:
    """Return the UniverseConfig for the given name (case-insensitive).

    Raises ValueError for unrecognised names.
    """
    key = name.upper()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown universe '{name}'. "
            f"Valid choices: {list(_REGISTRY.keys())}"
        )
    ucfg = _REGISTRY[key]
    if not ucfg.symbols:
        import logging
        logging.getLogger(__name__).warning(
            f"Universe '{name}' loaded with 0 symbols. "
            f"Check that universe_{name.lower()}.txt exists and is non-empty."
        )
    return ucfg


def list_universes() -> list[str]:
    """Return all registered universe names."""
    return list(_REGISTRY.keys())


def apply_tranche(
    ucfg:    UniverseConfig,
    tranche: str,
    capital: int | None = None,
    account: str        = "",
) -> UniverseConfig:
    """Return a copy of ucfg namespaced to a specific account and/or tranche.

    File naming:
        account=default, tranche=      → positions_nifty200.csv    (unchanged)
        account=default, tranche=T1    → positions_nifty200_t1.csv
        account=abhi2,   tranche=      → positions_nifty200_abhi2.csv
        account=abhi2,   tranche=T1    → positions_nifty200_abhi2_t1.csv

    Args:
        ucfg:    Base universe config (from get_universe()).
        tranche: Tranche label ("T1", "JAN2025", …). Empty = no suffix.
        capital: Override total_capital inside params (Rs).
                 Affects graded allocation sizes for this tranche.
        account: Account name. "default" or empty → no account suffix
                 (backward compatible with single-account deployments).

    Returns:
        A new (frozen) UniverseConfig with namespaced paths and/or updated capital.
    """
    changes: dict = {}

    # Build filename suffix: [account_]tranche  (skip "default" account)
    suffix_parts: list[str] = []
    if account and account.strip().lower() != "default":
        suffix_parts.append(account.strip().lower())
    if tranche:
        suffix_parts.append(tranche.strip().lower())

    if suffix_parts:
        suffix = "_".join(suffix_parts)
        n = ucfg.name.lower()
        changes["positions_file"] = f"positions_{n}_{suffix}.csv"
        changes["trade_log_file"] = f"trade_log_{n}_{suffix}.csv"

    # Update capital inside params (keeps all other params unchanged)
    if capital is not None:
        changes["params"] = ucfg.params.replace(total_capital=capital)

    return _dc_replace(ucfg, **changes) if changes else ucfg
