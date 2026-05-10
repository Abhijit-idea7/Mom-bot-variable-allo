"""
universes.py
------------
Universe registry for the Dual Momentum Delivery Bot.

Each universe defines its own:
  - Symbol list (stocks or ETFs)
  - Portfolio sizing params  (top_n_hold, hold_buffer)
  - State file paths         (positions CSV, trade-log CSV)

Select at runtime via --universe NIFTY200 or --universe BEES.
Both universes use identical strategy logic — only the universe params differ.

Tracked separately:
  NIFTY200 → positions_nifty200.csv + trade_log_nifty200.csv
  BEES     → positions_bees.csv     + trade_log_bees.csv
"""

from dataclasses import dataclass, replace as _dc_replace

from config import (
    BEES_UNIVERSE,
    NIFTY200_UNIVERSE,
    TOTAL_CAPITAL_BEES,
    TOTAL_CAPITAL_NIFTY200,
    WEEKLY_RANK_STOP_BEES,
    WEEKLY_RANK_STOP_NIFTY200,
)


# ── Universe config dataclass ─────────────────────────────────────────────────

@dataclass(frozen=True)
class UniverseConfig:
    name:              str
    display_name:      str
    symbols:           tuple        # frozen tuple — immutable after creation
    top_n_hold:        int          # maximum simultaneous positions
    hold_buffer:       int          # keep holding until rank exceeds this (monthly)
    positions_file:    str          # CSV that tracks open positions
    trade_log_file:    str          # CSV that records every BUY/SELL
    total_capital:     int          # total Rs allocated across all slots
    weekly_rank_stop:  int          # weekly_scan exits if rank > this (None = disabled)


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, UniverseConfig] = {
    "NIFTY200": UniverseConfig(
        name              = "NIFTY200",
        display_name      = "Nifty 200 Stocks",
        symbols           = tuple(NIFTY200_UNIVERSE),
        top_n_hold        = 15,
        hold_buffer       = 20,
        positions_file    = "positions_nifty200.csv",
        trade_log_file    = "trade_log_nifty200.csv",
        total_capital     = TOTAL_CAPITAL_NIFTY200,
        weekly_rank_stop  = WEEKLY_RANK_STOP_NIFTY200,
    ),
    "BEES": UniverseConfig(
        name              = "BEES",
        display_name      = "BEES ETFs",
        symbols           = tuple(BEES_UNIVERSE),
        top_n_hold        = 5,
        hold_buffer       = 7,
        positions_file    = "positions_bees.csv",
        trade_log_file    = "trade_log_bees.csv",
        total_capital     = TOTAL_CAPITAL_BEES,
        weekly_rank_stop  = WEEKLY_RANK_STOP_BEES,
    ),
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_universe(name: str) -> UniverseConfig:
    """Return the UniverseConfig for the given universe name.

    Args:
        name: 'NIFTY200' or 'BEES'  (case-insensitive)

    Raises:
        ValueError: if name is not recognised
    """
    key = name.upper()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown universe '{name}'. "
            f"Valid choices: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key]


def list_universes() -> list[str]:
    """Return all registered universe names."""
    return list(_REGISTRY.keys())


def apply_tranche(
    ucfg:    UniverseConfig,
    tranche: str,
    capital: int | None = None,
    account: str        = "",
) -> UniverseConfig:
    """
    Return a copy of ucfg namespaced to a specific account and/or tranche.

    File naming convention:
        account="",        tranche=""    →  positions_nifty200.csv           (default, no suffix)
        account="default", tranche=""    →  positions_nifty200.csv           (backward compat)
        account="",        tranche="T1"  →  positions_nifty200_t1.csv
        account="abhi2",   tranche=""    →  positions_nifty200_abhi2.csv
        account="abhi2",   tranche="T1"  →  positions_nifty200_abhi2_t1.csv

    Args:
        ucfg:    Base universe config (from get_universe()).
        tranche: Tranche label (e.g. "T1", "JAN2025"). Empty = no tranche suffix.
        capital: Total Rs for this tranche. Overrides ucfg.total_capital.
        account: Account name. "default" or empty = no account suffix (backward compat).

    Returns:
        A new (frozen) UniverseConfig with namespaced file paths.
    """
    changes: dict = {}

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

    if capital is not None:
        changes["total_capital"] = capital

    return _dc_replace(ucfg, **changes) if changes else ucfg
