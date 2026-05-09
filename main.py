"""
main.py
-------
Dual Momentum Delivery Bot — Monthly Rebalance.
Run on the LAST TRADING DAY of each month via GitHub Actions (manual trigger).

Strategy (Gary Antonacci's Dual Momentum):

  STEP 1 — ABSOLUTE MOMENTUM CHECK:
    Is Nifty 50's 12-month return (excluding last 21 days) > -5%?
    -> NO:   Sell ALL open positions. Stay in cash. Re-check next month.
    -> YES:  Proceed to relative momentum.

  STEP 2 — RELATIVE MOMENTUM (cross-sectional):
    Rank all universe stocks/ETFs by their 12M-1M return.
    SELL: holdings that have dropped below rank HOLD_BUFFER.
    BUY:  top TOP_N_HOLD ranked stocks/ETFs not already held.

  RISK VALVE (monthly hard stop — safety net, not part of original DM):
    Also sell any position down > HARD_STOP_PCT (15%) from entry price.

Universes (select via --universe):
  NIFTY200  — 200 Nifty stocks | top 15 held | sell if rank > 20
  BEES      — BEES ETFs        | top 5 held  | sell if rank > 7
"""

import argparse
import logging
import sys
from datetime import date, datetime

import pytz

from config import (
    HARD_STOP_PCT,
    MARKET_REGIME_FILTER,
    POSITION_SIZE_INR,
    WEIGHTING_SCHEME,
)
from data_feed import fetch_universe_prices, get_absolute_momentum
from momentum_scorer import MomentumRank, graded_allocation, print_ranked_table, rank_universe
from order_manager import buy_delivery, calculate_quantity, sell_delivery
from portfolio_state import DeliveryPosition, PortfolioState
from trade_logger import log_buy, log_sell, print_session_summary
from universes import UniverseConfig, apply_tranche, get_universe, list_universes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")
IST = pytz.timezone("Asia/Kolkata")


def parse_args():
    p = argparse.ArgumentParser(description="Dual Momentum — Monthly Rebalance")
    p.add_argument(
        "--universe", "-u",
        choices=list_universes(),
        default="NIFTY200",
        help=(
            "Which universe to trade. "
            "NIFTY200 = top 15 of 200 stocks. "
            "BEES = top 5 BEES ETFs. "
            "(default: NIFTY200)"
        ),
    )
    p.add_argument(
        "--tranche", "-t",
        default="",
        metavar="LABEL",
        help=(
            "Tranche label for this portfolio slice (e.g. T1, T2, JAN2025). "
            "Each tranche keeps independent positions and trade log files. "
            "Omit to use the default single-tranche files (backward compatible)."
        ),
    )
    p.add_argument(
        "--capital",
        default=None,
        type=int,
        metavar="RS",
        help=(
            "Total capital for this tranche in Rs (e.g. 1500000 for ₹15L). "
            "Overrides the universe default. Affects graded allocation sizes. "
            "Default: universe config (15,00,000 for NIFTY200, 5,00,000 for BEES)."
        ),
    )
    p.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview all actions without placing any orders or saving state. "
             "Safe to run any time — no orders are sent to the broker.",
    )
    return p.parse_args()


def ist_now() -> datetime:
    return datetime.now(IST)


# ---------------------------------------------------------------------------
# Exit evaluation
# ---------------------------------------------------------------------------

def check_exit_reason(
    pos:         DeliveryPosition,
    rank_map:    dict[str, MomentumRank],
    risk_on:     bool,
    hold_buffer: int,
) -> str | None:
    """
    Return an exit reason string if this position should be sold, else None.

    Exit triggers (in priority order):
      1. RISK_OFF     — absolute momentum failed; sell everything
      2. HARD_STOP    — position down >15% from entry (monthly safety check)
      3. NOT_RANKED   — stock/ETF dropped out of scoreable universe
      4. RANK_EXIT    — stock/ETF dropped below hold_buffer rank
    """
    # 1. Absolute momentum failed — global risk-off, sell all
    if not risk_on:
        return "RISK_OFF"

    cur = rank_map.get(pos.symbol)

    # 2. Hard stop — individual position down >15% from entry
    if HARD_STOP_PCT is not None and cur is not None:
        loss_pct = (pos.entry_price - cur.current_price) / pos.entry_price
        if loss_pct >= HARD_STOP_PCT:
            return f"HARD_STOP({loss_pct:.1%})"

    # 3. Stock/ETF no longer scoreable (too little data, delisted, etc.)
    if cur is None:
        return "NOT_RANKED"

    # 4. Rank has fallen below the hold buffer
    if cur.rank > hold_buffer:
        return f"RANK_EXIT(rank={cur.rank})"

    return None   # keep holding


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    dry_run:       bool      = False,
    universe_name: str       = "NIFTY200",
    tranche:       str       = "",
    capital:       int | None = None,
) -> None:
    ucfg: UniverseConfig = apply_tranche(get_universe(universe_name), tranche, capital)
    now = ist_now()
    tranche_label = f" [{tranche.upper()}]" if tranche else ""

    sep = "=" * 66
    logger.info(sep)
    if dry_run:
        logger.info(f"  DUAL MOMENTUM BOT [{ucfg.name}]{tranche_label} — DRY RUN (no orders placed)")
    else:
        logger.info(f"  DUAL MOMENTUM BOT [{ucfg.name}]{tranche_label} — MONTHLY REBALANCE")
    logger.info(f"  Universe     : {ucfg.display_name}  ({len(ucfg.symbols)} symbols)")
    logger.info(f"  Run time     : {now.strftime('%Y-%m-%d %H:%M:%S IST')}")
    logger.info(f"  Lookback     : 12M-1M (252 days, skip 21 days)")
    if WEIGHTING_SCHEME == "GRADED":
        logger.info(
            f"  Portfolio    : top {ucfg.top_n_hold} | Rs{ucfg.total_capital:,.0f} total "
            f"(graded by rank — rank 1 gets most, rank {ucfg.top_n_hold} gets least)"
        )
    else:
        logger.info(f"  Portfolio    : top {ucfg.top_n_hold} | Rs{POSITION_SIZE_INR:,.0f}/slot (equal weight)")
    logger.info(f"  Hold buffer  : sell if rank > {ucfg.hold_buffer}")
    logger.info(f"  Hard stop    : {HARD_STOP_PCT:.0%} from entry (monthly check)")
    logger.info(f"  Positions    : {ucfg.positions_file}")
    logger.info(f"  Trade log    : {ucfg.trade_log_file}")
    if dry_run:
        logger.info("  *** DRY RUN — orders logged but NOT sent to broker ***")
    logger.info(sep)

    # ── 1. Price data ─────────────────────────────────────────────────────────
    logger.info(f"\n[1/5] Downloading price history for {ucfg.display_name} (3y)...")
    prices_df = fetch_universe_prices(symbols=list(ucfg.symbols))
    if prices_df.empty:
        logger.error("Price download failed. Aborting.")
        sys.exit(1)

    # ── 2. Absolute momentum check ────────────────────────────────────────────
    logger.info("\n[2/5] Absolute momentum check (Nifty 50 12M-1M vs -5%)...")
    if MARKET_REGIME_FILTER:
        risk_on, abs_return = get_absolute_momentum()
    else:
        risk_on, abs_return = True, 0.0
        logger.info("  Regime filter disabled — treating as RISK-ON.")

    # ── 3. Rank universe ──────────────────────────────────────────────────────
    logger.info(f"\n[3/5] Ranking {ucfg.display_name} by 12M-1M momentum...")
    ranked   = rank_universe(prices_df)
    rank_map = {r.symbol: r for r in ranked}
    print_ranked_table(ranked, top_n=min(25, len(ranked)))

    if risk_on:
        logger.info(f"\n  RISK-ON: Nifty 12M-1M = {abs_return:+.1%} > -5% threshold")
        logger.info(f"  Targets: top {ucfg.top_n_hold} | hold if rank <= {ucfg.hold_buffer}")
    else:
        logger.info(f"\n  RISK-OFF: Nifty 12M-1M = {abs_return:+.1%} <= -5% threshold")
        logger.info("  Selling ALL positions — moving to cash until next month.")

    # ── 4. Load portfolio ─────────────────────────────────────────────────────
    logger.info(f"\n[4/5] Loading portfolio ({ucfg.positions_file})...")
    portfolio = PortfolioState(positions_file=ucfg.positions_file)
    logger.info(portfolio.summary())

    # ── 5. Rebalance ──────────────────────────────────────────────────────────
    # Wrapped in try/finally so portfolio.save() always runs, even if an
    # unexpected exception occurs mid-loop after some orders have been placed.
    # This keeps positions CSV and trade log CSV in sync at all times.
    logger.info("\n[5/5] Rebalancing...")
    session_sells: list[dict] = []
    session_buys:  list[dict] = []

    try:
        # ── EXITS ─────────────────────────────────────────────────────────────
        for pos in portfolio.all():
            reason = check_exit_reason(pos, rank_map, risk_on, ucfg.hold_buffer)

            if reason is None:
                cur   = rank_map[pos.symbol]
                gain  = (cur.current_price - pos.entry_price) / pos.entry_price
                logger.info(
                    f"  HOLD {pos.symbol:<16} rank=#{cur.rank:<4} "
                    f"12M-1M={cur.momentum_return:+.1%}  unrealised={gain:+.1%}"
                )
                continue

            cur       = rank_map.get(pos.symbol)
            cur_price = cur.current_price if cur else pos.entry_price
            mom_ret   = cur.momentum_return if cur else pos.momentum_return_at_entry
            cur_rank  = cur.rank if cur else 0
            pnl_inr   = round((cur_price - pos.entry_price) * pos.quantity, 2)
            pnl_pct   = round((cur_price - pos.entry_price) / pos.entry_price * 100, 2)

            if dry_run:
                logger.info(
                    f"  [DRY-SELL] {pos.symbol:<16} qty={pos.quantity}  "
                    f"entry=Rs{pos.entry_price:.2f}  now=Rs{cur_price:.2f}  "
                    f"P&L=Rs{pnl_inr:+,.0f} ({pnl_pct:+.1f}%)  reason={reason}"
                )
                session_sells.append({
                    "symbol":      pos.symbol,
                    "price":       cur_price,
                    "entry_price": pos.entry_price,
                    "quantity":    pos.quantity,
                    "pnl_inr":     pnl_inr,
                    "pnl_pct":     pnl_pct,
                    "reason":      reason,
                })
            else:
                ok = sell_delivery(pos.symbol, pos.quantity)
                if ok:
                    log_sell(
                        symbol          = pos.symbol,
                        price           = cur_price,
                        quantity        = pos.quantity,
                        entry_price     = pos.entry_price,
                        momentum_return = mom_ret,
                        rank            = cur_rank,
                        reason          = reason,
                        trade_log_file  = ucfg.trade_log_file,
                    )
                    session_sells.append({
                        "symbol":      pos.symbol,
                        "price":       cur_price,
                        "entry_price": pos.entry_price,
                        "quantity":    pos.quantity,
                        "pnl_inr":     pnl_inr,
                        "pnl_pct":     pnl_pct,
                        "reason":      reason,
                    })
                    portfolio.remove(pos.symbol)
                else:
                    logger.error(f"  SELL FAILED {pos.symbol} — keeping in portfolio.")

        # ── ENTRIES (only if risk-on) ──────────────────────────────────────────
        if not risk_on:
            logger.info("  RISK-OFF — no new buys this month.")
        else:
            slots = ucfg.top_n_hold - portfolio.count()
            held  = portfolio.symbols()
            candidates = [r for r in ranked[:ucfg.top_n_hold] if r.symbol not in held][:slots]

            if not candidates:
                logger.info("  No new entries — portfolio already at capacity or no candidates.")
            else:
                logger.info(f"  {len(candidates)} new entry candidate(s):")
                for r in candidates:
                    logger.info(
                        f"    ^ {r.symbol:<16} rank=#{r.rank:<4} "
                        f"12M-1M={r.momentum_return:+.1%}  price=Rs{r.current_price:.2f}"
                    )

            for r in candidates:
                if WEIGHTING_SCHEME == "GRADED":
                    alloc = graded_allocation(r.rank, ucfg.top_n_hold, ucfg.total_capital)
                else:
                    alloc = ucfg.total_capital // ucfg.top_n_hold  # equal share of tranche capital
                qty = calculate_quantity(r.current_price, alloc)
                if qty < 1:
                    logger.warning(
                        f"  {r.symbol}: price Rs{r.current_price:.2f} too high "
                        f"for Rs{alloc:,.0f} allocation (rank #{r.rank}) — skipping."
                    )
                    continue

                if dry_run:
                    logger.info(
                        f"  [DRY-BUY ] {r.symbol:<16} rank=#{r.rank:<4} qty={qty}  "
                        f"price=Rs{r.current_price:.2f}  "
                        f"alloc=Rs{alloc:,.0f}  value=Rs{qty * r.current_price:,.0f}  "
                        f"12M-1M={r.momentum_return:+.1%}"
                    )
                    session_buys.append({
                        "symbol":          r.symbol,
                        "price":           r.current_price,
                        "quantity":        qty,
                        "momentum_return": r.momentum_return,
                        "rank":            r.rank,
                    })
                else:
                    ok = buy_delivery(r.symbol, qty)
                    if ok:
                        log_buy(
                            symbol          = r.symbol,
                            price           = r.current_price,
                            quantity        = qty,
                            momentum_return = r.momentum_return,
                            rank            = r.rank,
                            trade_log_file  = ucfg.trade_log_file,
                        )
                        portfolio.add(
                            symbol                   = r.symbol,
                            entry_price              = r.current_price,
                            quantity                 = qty,
                            momentum_return_at_entry = r.momentum_return,
                            rank_at_entry            = r.rank,
                        )
                        session_buys.append({
                            "symbol":          r.symbol,
                            "price":           r.current_price,
                            "quantity":        qty,
                            "momentum_return": r.momentum_return,
                            "rank":            r.rank,
                        })
                    else:
                        logger.error(f"  BUY FAILED {r.symbol}.")

    finally:
        # Always persist portfolio state after the rebalance loop — even if an
        # unexpected exception fired after some orders were placed.  This keeps
        # positions_*.csv in sync with trade_log_*.csv at all times.
        # Skipped in dry-run (no state changes to persist).
        if not dry_run:
            portfolio.save()

    # ── Summary (only reached when rebalance completed without exception) ──────
    print_session_summary(session_buys, session_sells)

    if dry_run:
        logger.info("\n--- DRY RUN COMPLETE ---")
        logger.info(
            f"  Would have placed {len(session_buys)} BUY and "
            f"{len(session_sells)} SELL order(s)."
        )
        logger.info("  No orders sent. Run without --dry-run on a trading day to execute.")
    else:
        logger.info("\nPortfolio after rebalance:")
        logger.info(portfolio.summary())
        if not session_sells and not session_buys:
            logger.info("\nNo trades this month — portfolio unchanged.")
        logger.info(
            f"\nDone. {ucfg.positions_file} and {ucfg.trade_log_file} "
            f"committed by workflow."
        )


if __name__ == "__main__":
    args = parse_args()
    run(
        dry_run       = args.dry_run,
        universe_name = args.universe,
        tranche       = args.tranche,
        capital       = args.capital,
    )
