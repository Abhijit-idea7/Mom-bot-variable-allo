"""
portfolio_state.py
------------------
Persists open delivery positions across monthly rebalances.

CSV committed to repo after every rebalance:
  positions_<universe>[_<account>][_<tranche>].csv

Atomic writes:
  save() writes to a temp file in the same directory, then uses os.replace()
  to atomically swap it into place. This prevents CSV corruption if the process
  is killed mid-write (e.g., GitHub Actions timeout). os.replace() is atomic on
  POSIX filesystems and best-effort on Windows (always used on Ubuntu runners).

No cooldown tracking — Dual Momentum has no re-entry gate.
Stocks exit because they fell out of the top-ranked tier, and
may re-enter next month if they rank highly again.
"""

import csv
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from config import POSITIONS_FILE as _DEFAULT_POSITIONS_FILE

logger = logging.getLogger(__name__)

_POS_FIELDS = [
    "symbol", "entry_date", "entry_price", "quantity",
    "momentum_return_at_entry", "rank_at_entry",
]


@dataclass
class DeliveryPosition:
    symbol:                   str
    entry_date:               str     # "YYYY-MM-DD"
    entry_price:              float   # approximate (last yfinance close before order)
    quantity:                 int
    momentum_return_at_entry: float   # 12M-1M return on the month we bought
    rank_at_entry:            int     # Rank in universe on the month we bought


class PortfolioState:
    """Loads and saves open positions from/to a universe-specific CSV.

    Args:
        positions_file: Path to the CSV.  Pass ucfg.positions_file for the
            correct universe/tranche/account-specific file.
    """

    def __init__(self, positions_file: str = _DEFAULT_POSITIONS_FILE) -> None:
        self._file = positions_file
        self._positions: dict[str, DeliveryPosition] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        path = Path(self._file)
        if not path.exists():
            logger.info(f"{self._file} not found — starting with empty portfolio.")
            return
        try:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if not row.get("symbol"):
                        continue
                    pos = DeliveryPosition(
                        symbol                   = row["symbol"],
                        entry_date               = row["entry_date"],
                        entry_price              = float(row["entry_price"]),
                        quantity                 = int(row["quantity"]),
                        momentum_return_at_entry = float(row["momentum_return_at_entry"]),
                        rank_at_entry            = int(row["rank_at_entry"]),
                    )
                    self._positions[pos.symbol] = pos
            logger.info(
                f"Loaded {len(self._positions)} open positions from {self._file}"
            )
        except Exception as e:
            logger.error(
                f"Failed to load positions from {self._file}: {e}. "
                f"Starting with empty portfolio — check the file manually."
            )

    def save(self) -> None:
        """Persist current state to CSV using an atomic write.

        Writes to a temporary file in the same directory, then renames it
        over the target file.  This means the target file is never left in
        a partially-written state if the process is interrupted.
        """
        path = Path(self._file)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to a temp file in the same directory (same filesystem → atomic rename)
        fd, tmp_path = tempfile.mkstemp(
            dir    = str(path.parent),
            prefix = f".{path.name}.",
            suffix = ".tmp",
        )
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_POS_FIELDS)
                writer.writeheader()
                for pos in self._positions.values():
                    writer.writerow(asdict(pos))
            os.replace(tmp_path, str(path))   # atomic on POSIX, best-effort on Windows
        except Exception:
            # Clean up temp file on failure; re-raise so the caller knows
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.info(
            f"Saved {len(self._positions)} open positions to {self._file} (atomic write)"
        )

    # ── Queries ───────────────────────────────────────────────────────────────

    def all(self) -> list[DeliveryPosition]:
        return list(self._positions.values())

    def symbols(self) -> set[str]:
        return set(self._positions.keys())

    def has(self, symbol: str) -> bool:
        return symbol in self._positions

    def get(self, symbol: str) -> DeliveryPosition | None:
        return self._positions.get(symbol)

    def count(self) -> int:
        return len(self._positions)

    # ── Mutations ─────────────────────────────────────────────────────────────

    def add(
        self,
        symbol:                   str,
        entry_price:              float,
        quantity:                 int,
        momentum_return_at_entry: float,
        rank_at_entry:            int,
    ) -> None:
        self._positions[symbol] = DeliveryPosition(
            symbol                   = symbol,
            entry_date               = date.today().isoformat(),
            entry_price              = round(entry_price, 2),
            quantity                 = quantity,
            momentum_return_at_entry = round(momentum_return_at_entry, 4),
            rank_at_entry            = rank_at_entry,
        )
        logger.info(
            f"[PORTFOLIO] +BUY  {symbol:<14} qty={quantity} @ Rs{entry_price:.2f}  "
            f"12M-1M={momentum_return_at_entry:+.1%}  rank=#{rank_at_entry}"
        )

    def remove(self, symbol: str) -> DeliveryPosition | None:
        pos = self._positions.pop(symbol, None)
        if pos:
            logger.info(f"[PORTFOLIO] -SELL {symbol:<14} (removed from portfolio)")
        return pos

    # ── Display ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = []
        today = date.today()
        if not self._positions:
            lines.append("Portfolio: empty (all cash)")
        else:
            lines.append(f"Open positions ({self.count()}):")
            for p in sorted(
                self._positions.values(),
                key=lambda x: x.momentum_return_at_entry,
                reverse=True,
            ):
                entry_dt  = date.fromisoformat(p.entry_date)
                days_held = (today - entry_dt).days
                lines.append(
                    f"  {p.symbol:<14} qty={p.quantity:<5} "
                    f"entry=Rs{p.entry_price:,.2f}  held={days_held}d  "
                    f"12M-1M={p.momentum_return_at_entry:+.1%}  rank=#{p.rank_at_entry}"
                )
        return "\n".join(lines)
