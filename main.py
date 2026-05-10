"""
main.py
-------
Dual Momentum Delivery Bot — Monthly Rebalance.
Run on the LAST TRADING DAY of each month via GitHub Actions (manual trigger).

Strategy (Gary Antonacci's Dual Momentum):

  STEP 1 — ABSOLUTE MOMENTUM CHECK:
    Is Nifty 50's 12M-1M return > abs_threshold (-5%)?
    → NO:   Sell ALL open positions. Stay in cash. Re-check next month.
    → YES:  Proceed to relative momentum.

  STEP 2 — RELATIVE MOMENTUM (cross-sectional):
    Rank all universe stocks/ETFs by 12M-1M return.
    SELL: holdings ranked below hold_buffer.
    BUY:  fill empty slots with top-ranked stocks not already held.

  RISK VALVE (monthly hard stop — supplementary, not part of original DM):
    Also sell any position down > hard_stop_pct (15%) from entry price.

  ALLOCATION (GRADED by default):
    Rank 1 gets most capital, rank N gets least.
    Formula: weight(rank) = N+1−rank; allocation = total_capital × weight / N(N+1)/2

All strategy parameters are read from the universe's StrategyParams (strategy.py).
CLI flags override the defaults per-run without changing source code.
"""

import argparse
import logging
import sys
from datetime import date, datetime

import pytz

from accounts import AccountConfig, get_account
from config import POSITION_SIZE_INR
from data_feed import fetch_universe_prices, get_absolute_momentum
from momentum_scorer import MomentumRank, graded_allocation, print_ranked_table, rank_universe
from order_manager import buy_delivery, calculate_quantity, sell_delivery
from portfolio_state import DeliveryPosition, PortfolioState
from strategy import StrategyParams
from trade_logger import log_buy, log_sell, print_session_summary
from universes import UniverseConfig, apply_tranche, get_universe, list_universes

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")
IST = pytz.timezone("Asia/Kolkata")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Dual Momentum — Monthly Rebalance.\n"
            "All strategy parameters default to the universe config (strategy.py). "
            "CLI flags override per-run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--universe", "-u",
        choices = list_universes(),
        default = "NIFTY200",
        help    = "Universe to trade: NIFTY200 (stocks) or BEES (ETFs). (default: NIFTY200)",
    )
    p.add_argument(
        "--account", "-a",
        default = "default",
        metavar = "NAME",
        help    = "Account name from accounts.json (default: 'default'). "
                  "Each account uses its own API key and namespaced position files.",
    )
    p.add_argument(
        "--tranche", "-t",
        default = "",
        metavar = "LABEL",
        help    = "Tranche label (e.g. T1, T2, JAN2025). Each tranche is tracked "
                  "independently. Omit for the default single-tranche portfolio.",
    )
    p.add_argument(
        "--capital",
        default = None,
        type    = int,
        metavar = "RS",
        help    = "Total capital for this tranche in Rs (e.g. 1500000). "
                  "Overrides the universe default. Affects all allocation sizes.",
    )
    # ── Strategy overrides ─────────────────────────────────────────────────
    p.add_argument(
        "--weighting",
        choices = ["graded", "equal"],
        default = None,
        help    = "Position sizing: 'graded' (rank-weighted) or 'equal' (flat). "
                  "Default: universe config (GRADED).",
    )
    p.add_argument(
        "--hard-stop",
        default = None,
        type    = float,
        metavar = "FRACTION",
        help    = "Hard stop from entry, e.g. 0.15 = 15%%. "
                  "Default: universe config (0.15).",
    )
    p.add_argument(
        "--abs-threshold",
        default = None,
        type    = float,
        metavar = "FRACTION",
        help    = "Absolute momentum threshold, e.g. -0.05 = -5%%. "
                  "Default: universe config (-0.05).",
    )
    p.add_argument(
        "--lookback",
        default = None,
        type    = int,
        metavar = "DAYS",
        help    = "Momentum lookback in trading days (default: 252).",
    )
    p.add_argument(
        "--skip",
        default = None,
        type    = int,
        metavar = "DAYS",
        help    = "Skip-month days to avoid reversal (default: 21).",
    )
    p.add_argument(
        "--no-abs-filter",
        action  = "store_true",
        help    = "Disable absolute momentum check (pure relative momentum). "
                  "Use for testing only — disables the risk-off safety mechanism.",
    )
    p.add_argument(
        "--dry-run", "-n",
        action = "store_true",
        help   = "Preview all actions without placing any orders or saving state. "
                 "Safe to run any time.",
    )
    return p.parse_args()


def ist_now() -> datetime:
    return datetime.now(IST)


# ── Exit evaluation ───────────────────────────────────────────────────────────

def check_exit_reason(
    pos:         DeliveryPosition,
    rank_map:    dict[str, MomentumRank],
    risk_on:     bool,
    params:      StrategyParams,
) -> str | None:
    """Return an exit reason string if this position should be sold, else None.

    Exit triggers (in priority order):
      1. RISK_OFF     — absolute momentum failed; sell everything
      2. HARD_STOP    — position down > hard_stop_pct from entry
      3. NOT_RANKED   — stock/ETF dropped out of scoreable universe
      4. RANK_EXIT    — stock/ETF ranked below hold_buffer
    """
    if not risk_on:
        return "RISK_OFF"

    cur = rank_map.get(pos.symbol)

    # Hard stop — checked before rank so a crashing stock exits immediately
    if cur is not None:
        loss_pct = (pos.entry_price - cur.current_price) / pos.entry_price
        if loss_pct >= params.hard_stop_pct:
            return f"HARD_STOP({loss_pct:.1%})"

    if cur is None:
        return "NOT_RANKED"

    if cur.rank > params.hold_buffer:
        return f"RANK_EXIT(rank={cur.rank})"

    return None   # keep holding


# ── Main rebalance function ───────────────────────────────────────────────────

def run(
    dry_run:      bool      = False,
    universe_name: str      = "NIFTY200",
    tranche:      str       = "",
    capital:      int | None = None,
    account_name: str       = "default",
    weighting:    str | None = None,
    hard_stop:    float | None = None,
    abs_threshold: float | None = None,
    lookback:     int | None = None,
    skip:         int | None = None,
    no_abs_filter: bool     = False,
) -> None:
    account: AccountConfig = get_account(account_name, dry_run=dry_run)
    ucfg: UniverseConfig   = apply_tranche(
        get_universe(universe_name), tranche, capital, account=account_name
    )

    # Apply any per-run strategy overrides
    params = ucfg.params
    overrides: dict = {}
    if weighting     is not None: overrides["weighting"]      = weighting.upper()
    if hard_stop     is not None: overrides["hard_stop_pct"]  = hard_stop
    if abs_threshold is not None: overrides["abs_threshold"]  = abs_threshold
    if lookback      is not None: overrides["lookback_days"]  = lookback
    if skip          is not None: overrides["skip_days"]      = skip
    if no_abs_filter:              overrides["regime_filter"] = False
    if overrides:
        params = params.replace(**overrides)

    now = ist_now()
    tranche_label = f" [{tranche.upper()}]" if tranche else ""

    sep = "=" * 68
    logger.info(sep)
    if dry_run:
        logger.info(f"  DUAL MOMENTUM BOT [{ucfg.name}]{tranche_label} — DRY RUN")
    else:
        logger.info(f"  DUAL MOMENTUM BOT [{ucfg.name}]{tranche_label} — MONTHLY REBALANCE")
    logger.info(f"  Universe     : {ucfg.display_name}  ({len(ucfg.symbols)} symbols)")
    logger.info(f"  Run time     : {now.strftime('%Y-%m-%d %H:%M:%S IST')}")
    logger.info(f"  Account      : {account.display_name} ({account.account_id})")
    logger.info(f"  Capital      : Rs{params.total_capital:,.0f}")
    logger.info(f"  Weighting    : {params.weighting}")
    logger.info(f"  Portfolio    : top {params.top_n_hold} | "
                f"hold if rank ≤ {params.hold_buffer}")
    logger.info(f"  Lookback     : {params.lookback_days}d − {params.skip_days}d skip")
    logger.info(f"  Abs threshold: {params.abs_threshold:+.1%}"
                + ("  [DISABLED]" if not params.regime_filter else ""))
    logger.info(f"  Hard stop    : {params.hard_stop_pct:.0%} from entry")
    logger.info(f"  Positions    : {ucfg.positions_file}")
    logger.info(f"  Trade log    : {ucfg.trade_log_file}")
    if dry_run:
        logger.info("  *** DRY RUN — orders NOT sent to broker ***")
    logger.info(sep)

    # ── Validate universe has symbols ──────────────────────────────────────────
    if not ucfg.symbols:
        logger.error(
            f"Universe '{universe_name}' has 0 symbols. "
            f"Check universe_{universe_name.lower()}.txt exists and is non-empty."
        )
        sys.exit(1)

    # ── 1. Price data ──────────────────────────────────────────────────────────
    logger.info(f"\n[1/5] Downloading price history for {ucfg.display_name} (3y)...")
    prices_df = fetch_universe_prices(symbols=list(ucfg.symbols))
    if prices_df.empty:
        logger.error(
            "Price download failed — could not retrieve any data from Yahoo Finance. "
            "Check internet connectivity and try again."
        )
        sys.exit(1)

    # ── 2. Absolute momentum check ─────────────────────────────────────────────
    logger.info(
        f"\n[2/5] Absolute momentum check "
        f"({params.regime_ticker} 12M-1M vs {params.abs_threshold:+.1%})..."
    )
    if params.regime_filter:
        risk_on, abs_return = get_absolute_momentum(
            lookback  = params.lookback_days,
            skip      = params.skip_days,
            threshold = params.abs_threshold,
            ticker    = params.regime_ticker,
        )
    else:
        risk_on, abs_return = True, 0.0
        logger.info("  Regime filter disabled — treating as RISK-ON.")

    # ── 3. Rank universe ───────────────────────────────────────────────────────
    logger.info(f"\n[3/5] Ranking {ucfg.display_name} by {params.lookback_days}-{params.skip_days}d momentum...")
    ranked   = rank_universe(prices_df, lookback=params.lookback_days, skip=params.skip_days)
    rank_map = {r.symbol: r for r in ranked}
    print_ranked_table(ranked, top_n=min(25, len(ranked)))

    if risk_on:
        logger.info(f"\n  RISK-ON: 12M-1M = {abs_return:+.1%} > {params.abs_threshold:+.1%}")
        logger.info(f"  Targets: top {params.top_n_hold} | hold if rank ≤ {params.hold_buffer}")
    else:
        logger.info(f"\n  RISK-OFF: 12M-1M = {abs_return:+.1%} ≤ {params.abs_threshold:+.1%}")
        logger.info("  Selling ALL positions — moving to cash until next month.")

    # ── 4. Load portfolio ──────────────────────────────────────────────────────
    logger.info(f"\n[4/5] Loading portfolio ({ucfg.positions_file})...")
    portfolio = PortfolioState(positions_file=ucfg.positions_file)
    logger.info(portfolio.summary())

    # ── 5. Rebalance ───────────────────────────────────────────────────────────
    # try/finally guarantees portfolio.save() runs even after an unexpected
    # exception fired after some orders were already sent to the broker.
    logger.info("\n[5/5] Rebalancing...")
    session_sells: list[dict] = []
    session_buys:  list[dict] = []

    try:
        # ── EXITS ─────────────────────────────────────────────────────────────
        for pos in portfolio.all():
            reason = check_exit_reason(pos, rank_map, risk_on, params)

            if reason is None:
                cur  = rank_map[pos.symbol]
                gain = (cur.current_price - pos.entry_price) / pos.entry_price
                logger.info(
                    f"  HOLD {pos.symbol:<16} rank=#{cur.rank:<4} "
                    f"12M-1M={cur.momentum_return:+.1%}  unrealised={gain:+.1%}"
                )
                continue

            cur       = rank_map.get(pos.symbol)
            cur_price = cur.current_price   if cur else pos.entry_price
            mom_ret   = cur.momentum_return if cur else pos.momentum_return_at_entry
            cur_rank  = cur.rank            if cur else 0
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
                ok = sell_delivery(pos.symbol, pos.quantity, account)
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
                    logger.error(
                        f"  SELL FAILED {pos.symbol} — "
                        f"position KEPT in portfolio. Check broker order book."
                    )

        # ── ENTRIES (only if risk-on) ──────────────────────────────────────────
        if not risk_on:
            logger.info("  RISK-OFF — no new buys this month.")
        else:
            slots      = params.top_n_hold - portfolio.count()
            held       = portfolio.symbols()
            candidates = [r for r in ranked[:params.top_n_hold] if r.symbol not in held][:slots]

            if not candidates:
                logger.info(
                    "  No new entries — portfolio already at capacity or no candidates."
                )
            else:
                logger.info(f"  {len(candidates)} new entry candidate(s):")
                for r in candidates:
                    logger.info(
                        f"    ^ {r.symbol:<16} rank=#{r.rank:<4} "
                        f"12M-1M={r.momentum_return:+.1%}  price=Rs{r.current_price:.2f}"
                    )

            for r in candidates:
                if params.weighting.upper() == "GRADED":
                    alloc = graded_allocation(r.rank, params.top_n_hold, params.total_capital)
                else:
                    alloc = params.total_capital // params.top_n_hold
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
                        f"price=Rs{r.current_price:.2f}  alloc=Rs{alloc:,.0f}  "
                        f"value=Rs{qty * r.current_price:,.0f}  "
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
                    ok = buy_delivery(r.symbol, qty, account)
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
                        logger.error(f"  BUY FAILED {r.symbol} — order not placed.")

    finally:
        # Always persist portfolio state, even after an unexpected exception
        # mid-loop (after some orders may already have been sent to the broker).
        # This keeps positions_*.csv in sync with trade_log_*.csv at all times.
        if not dry_run:
            portfolio.save()

    # ── Summary ───────────────────────────────────────────────────────────────
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
            f"will be committed by the workflow."
        )


if __name__ == "__main__":
    args = parse_args()
    run(
        dry_run       = args.dry_run,
        universe_name = args.universe,
        tranche       = args.tranche,
        capital       = args.capital,
        account_name  = args.account,
        weighting     = args.weighting,
        hard_stop     = args.hard_stop,
        abs_threshold = args.abs_threshold,
        lookback      = args.lookback,
        skip          = args.skip,
        no_abs_filter = args.no_abs_filter,
    )
