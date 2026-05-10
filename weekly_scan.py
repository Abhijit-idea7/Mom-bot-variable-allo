"""
weekly_scan.py
--------------
Weekly Hard Stop Scanner — run every Friday before 3:30 PM IST.

Two independent circuit breakers checked for every open position:

  1. PRICE HARD STOP
     Sell if the position is down > hard_stop_pct (15%) from entry price.
     Prevents a gap-down disaster from compounding between monthly rebalances.

  2. RANK STOP (default enabled)
     Download full universe price history, rank by 12M-1M momentum, and sell
     if a held stock's rank > weekly_rank_stop (default 25 for NIFTY200, 9 for BEES).
     Catches rapid momentum reversals before month-end.
     Disable with --no-rank-stop for price-only mode.

Does NOT:
  - Open new positions (entries happen at monthly rebalance only).
  - Apply the absolute momentum (risk-off) filter.
    Mid-month risk-off exits would be caught at the next monthly rebalance.

Run via GitHub Actions: Actions → "Dual Momentum — Weekly Hard Stop Scan" → Run workflow
"""

import argparse
import logging
import sys
from datetime import date, datetime

import pytz

from accounts import AccountConfig, get_account
from data_feed import fetch_universe_prices, get_current_price
from momentum_scorer import rank_universe
from order_manager import sell_delivery
from portfolio_state import PortfolioState
from trade_logger import log_sell
from universes import UniverseConfig, apply_tranche, get_universe, list_universes

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("weekly_scan")
IST = pytz.timezone("Asia/Kolkata")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Dual Momentum — Weekly Hard Stop Scan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--universe", "-u",
        choices = list_universes(),
        default = "NIFTY200",
        help    = "Universe to scan: NIFTY200 or BEES (default: NIFTY200)",
    )
    p.add_argument(
        "--account", "-a",
        default = "default",
        metavar = "NAME",
        help    = "Account name from accounts.json. Must match the account "
                  "used at trade time. (default: 'default')",
    )
    p.add_argument(
        "--tranche", "-t",
        default = "",
        metavar = "LABEL",
        help    = "Tranche label to scan. Must match the label used at trade time. "
                  "Omit for the default single-tranche portfolio.",
    )
    p.add_argument(
        "--no-rank-stop",
        action = "store_true",
        help   = "Disable the weekly rank-based exit check. "
                 "Only the price hard stop will run.",
    )
    p.add_argument(
        "--dry-run", "-n",
        action = "store_true",
        help   = "Preview stop breaches without placing any sell orders or saving state.",
    )
    return p.parse_args()


# ── Scanner ───────────────────────────────────────────────────────────────────

def run(
    dry_run:      bool = False,
    universe_name: str = "NIFTY200",
    tranche:      str  = "",
    no_rank_stop: bool = False,
    account_name: str  = "default",
) -> None:
    account: AccountConfig = get_account(account_name, dry_run=dry_run)
    ucfg: UniverseConfig   = apply_tranche(
        get_universe(universe_name), tranche, account=account_name
    )
    params = ucfg.params
    now    = datetime.now(IST)
    sep    = "=" * 64
    tranche_label = f" [{tranche.upper()}]" if tranche else ""

    use_rank_stop = (not no_rank_stop) and (params.weekly_rank_stop is not None)

    logger.info(sep)
    if dry_run:
        logger.info(f"  WEEKLY STOP SCANNER [{ucfg.name}]{tranche_label} — DRY RUN")
    else:
        logger.info(f"  WEEKLY STOP SCANNER [{ucfg.name}]{tranche_label}")
    logger.info(f"  Universe    : {ucfg.display_name}")
    logger.info(f"  Account     : {account.display_name} ({account.account_id})")
    logger.info(f"  Scan time   : {now.strftime('%Y-%m-%d %H:%M:%S IST')}")
    logger.info(f"  Price stop  : exit if down > {params.hard_stop_pct:.0%} from entry")
    logger.info(f"  Warn at     : down > {params.warn_pct:.0%} (approaching price stop)")
    if use_rank_stop:
        logger.info(
            f"  Rank stop   : exit if rank > {params.weekly_rank_stop} "
            f"(monthly hold_buffer = {params.hold_buffer})"
        )
    else:
        logger.info("  Rank stop   : DISABLED (--no-rank-stop)")
    logger.info(f"  Positions   : {ucfg.positions_file}")
    if dry_run:
        logger.info("  *** DRY RUN — sell orders will NOT be sent to broker ***")
    logger.info(sep)

    portfolio = PortfolioState(positions_file=ucfg.positions_file)

    if portfolio.count() == 0:
        logger.info(f"\nPortfolio is empty ({ucfg.positions_file}) — nothing to scan.")
        return

    # ── Download prices + rank universe (only when rank stop is active) ────────
    rank_map: dict | None = None
    if use_rank_stop:
        if not ucfg.symbols:
            logger.warning(
                f"Universe '{universe_name}' has 0 symbols — rank stop DISABLED. "
                f"Check universe_{universe_name.lower()}.txt."
            )
        else:
            logger.info(
                f"\nDownloading price history for {ucfg.display_name} (3y) "
                f"— rank check..."
            )
            prices_df = fetch_universe_prices(symbols=list(ucfg.symbols))
            if prices_df.empty:
                logger.warning(
                    "  Price download failed — rank stop DISABLED for this run. "
                    "Price hard stop will still fire."
                )
            else:
                ranked   = rank_universe(
                    prices_df,
                    lookback = params.lookback_days,
                    skip     = params.skip_days,
                )
                rank_map = {r.symbol: r for r in ranked}
                logger.info(
                    f"  Ranked {len(rank_map)} stocks. "
                    f"Weekly rank threshold: rank > {params.weekly_rank_stop}"
                )

    logger.info(f"\nScanning {portfolio.count()} open position(s)...\n")

    # Track original count before any removals (for accurate summary)
    original_count = portfolio.count()
    stops_fired:  list[dict] = []
    fetch_errors: list[str]  = []

    for pos in sorted(portfolio.all(), key=lambda p: p.symbol):

        # Resolve current price — reuse rank_map price if available (avoids
        # a second Yahoo Finance call for stocks we just ranked)
        if rank_map is not None and pos.symbol in rank_map:
            cur_price = rank_map[pos.symbol].current_price
        else:
            cur_price = get_current_price(pos.symbol)

        if cur_price is None:
            logger.warning(
                f"  [????] {pos.symbol:<16} — price fetch failed, "
                f"skipping (check manually)"
            )
            fetch_errors.append(pos.symbol)
            continue

        loss_pct  = (pos.entry_price - cur_price) / pos.entry_price
        gain_pct  = (cur_price - pos.entry_price) / pos.entry_price
        days_held = (date.today() - date.fromisoformat(pos.entry_date)).days
        pnl_inr   = round((cur_price - pos.entry_price) * pos.quantity, 2)

        # ── Determine exit trigger ─────────────────────────────────────────────
        exit_reason: str | None = None
        cur_rank:    int | None = None

        if loss_pct >= params.hard_stop_pct:
            exit_reason = f"WEEKLY_HARD_STOP({loss_pct:.1%})"
        elif use_rank_stop and rank_map is not None:
            rank_entry = rank_map.get(pos.symbol)
            if rank_entry is None:
                exit_reason = "WEEKLY_RANK_EXIT(unranked)"
            else:
                cur_rank = rank_entry.rank
                if cur_rank > params.weekly_rank_stop:
                    exit_reason = f"WEEKLY_RANK_EXIT(rank={cur_rank})"

        # ── Execute exit ───────────────────────────────────────────────────────
        if exit_reason is not None:
            logger.warning(
                f"  [EXIT] {pos.symbol:<16}  "
                f"entry=Rs{pos.entry_price:>8.2f}  now=Rs{cur_price:>8.2f}  "
                f"{'loss' if loss_pct >= 0 else 'gain'}={abs(loss_pct):.1%}  "
                f"P&L=Rs{pnl_inr:+,.0f}  held={days_held}d  reason={exit_reason}"
            )
            stop_rec = {
                "symbol":      pos.symbol,
                "entry_price": pos.entry_price,
                "exit_price":  cur_price,
                "quantity":    pos.quantity,
                "loss_pct":    loss_pct,
                "pnl_inr":     pnl_inr,
                "days_held":   days_held,
                "reason":      exit_reason,
            }
            if dry_run:
                logger.warning(
                    f"  [DRY ] Would SELL {pos.symbol} ×{pos.quantity} "
                    f"@ Rs{cur_price:.2f} — order NOT sent"
                )
                stops_fired.append(stop_rec)
            else:
                ok = sell_delivery(pos.symbol, pos.quantity, account)
                if ok:
                    log_sell(
                        symbol          = pos.symbol,
                        price           = cur_price,
                        quantity        = pos.quantity,
                        entry_price     = pos.entry_price,
                        momentum_return = pos.momentum_return_at_entry,
                        rank            = pos.rank_at_entry,
                        reason          = exit_reason,
                        trade_log_file  = ucfg.trade_log_file,
                    )
                    portfolio.remove(pos.symbol)
                    stops_fired.append(stop_rec)
                else:
                    logger.error(
                        f"  [FAIL] Sell order FAILED for {pos.symbol} — "
                        f"position kept. Check broker and retry manually."
                    )

        elif loss_pct >= params.warn_pct:
            logger.warning(
                f"  [WARN] {pos.symbol:<16}  "
                f"entry=Rs{pos.entry_price:>8.2f}  now=Rs{cur_price:>8.2f}  "
                f"loss={loss_pct:.1%}  P&L=Rs{pnl_inr:+,.0f}  held={days_held}d  "
                f"(price stop at {params.hard_stop_pct:.0%}"
                + (f", rank=#{cur_rank}" if cur_rank else "") + ")"
            )
        elif gain_pct >= 0:
            rank_str = f"  rank=#{cur_rank}" if cur_rank else ""
            logger.info(
                f"  [ OK ] {pos.symbol:<16}  "
                f"entry=Rs{pos.entry_price:>8.2f}  now=Rs{cur_price:>8.2f}  "
                f"gain={gain_pct:+.1%}  P&L=Rs{pnl_inr:+,.0f}  held={days_held}d"
                + rank_str
            )
        else:
            rank_str = f"  rank=#{cur_rank}" if cur_rank else ""
            logger.info(
                f"  [HOLD] {pos.symbol:<16}  "
                f"entry=Rs{pos.entry_price:>8.2f}  now=Rs{cur_price:>8.2f}  "
                f"loss={loss_pct:.1%}  P&L=Rs{pnl_inr:+,.0f}  held={days_held}d"
                + rank_str
            )

    # ── Persist changes (skipped in dry-run) ──────────────────────────────────
    if stops_fired and not dry_run:
        portfolio.save()

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(f"\n{sep}")
    logger.info(f"  SCAN SUMMARY [{ucfg.name}] — {date.today().isoformat()}")
    logger.info(sep)
    # Use original_count (captured before loop) to avoid double-counting in dry-run
    logger.info(f"  Positions scanned  : {original_count}")
    logger.info(f"  Exits fired        : {len(stops_fired)}")
    logger.info(f"  Price fetch errors : {len(fetch_errors)}")

    if stops_fired:
        total_pnl   = sum(s["pnl_inr"] for s in stops_fired)
        price_stops = [s for s in stops_fired if "HARD_STOP"  in s["reason"]]
        rank_stops  = [s for s in stops_fired if "RANK_EXIT"  in s["reason"]]
        logger.info(f"  Total realised P&L : Rs{total_pnl:+,.0f}")
        logger.info(f"\n  Positions cut this week:")
        for s in stops_fired:
            logger.info(
                f"    XX {s['symbol']:<16}  "
                f"Rs{s['entry_price']:.2f} → Rs{s['exit_price']:.2f}  "
                f"loss={s['loss_pct']:.1%}  P&L=Rs{s['pnl_inr']:+,.0f}  "
                f"held={s['days_held']}d  [{s['reason']}]"
            )
        if price_stops:
            logger.info(f"\n  Price hard stops : {len(price_stops)}")
        if rank_stops:
            logger.info(f"  Rank exits       : {len(rank_stops)}")
        logger.info(
            f"\n  {len(stops_fired)} slot(s) freed — "
            f"will be refilled at next monthly rebalance."
        )
    else:
        logger.info("\n  No exits triggered — all positions within thresholds.")

    if fetch_errors:
        logger.warning(
            f"\n  Price fetch failures — check manually: {', '.join(fetch_errors)}"
        )

    if dry_run:
        logger.info("\n  *** DRY RUN COMPLETE — no orders sent, no files changed ***")

    logger.info(sep)


if __name__ == "__main__":
    args = parse_args()
    run(
        dry_run       = args.dry_run,
        universe_name = args.universe,
        tranche       = args.tranche,
        no_rank_stop  = args.no_rank_stop,
        account_name  = args.account,
    )
