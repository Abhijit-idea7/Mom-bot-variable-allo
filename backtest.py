"""
backtest.py
-----------
Dual Momentum Delivery Bot — Monthly Backtest Engine.

Simulates the exact same logic as main.py across historical data.

Rebalance logic (run on the last trading day of each month):
  1. Absolute momentum: Is Nifty 50's 12M-1M return > abs_threshold?
     NO  -> sell all, sit in cash until next month's check
     YES -> proceed to relative momentum
  2. Relative momentum: Rank Nifty 200 by 12M-1M return
     SELL: holdings ranked below hold_buffer
     BUY:  fill slots with top portfolio_size stocks not already held
  3. Hard stop: sell any position down > hard_stop from entry (checked monthly)

NAV is marked-to-market every trading day, but trades only happen on
month-end dates. This gives accurate Sharpe/max-drawdown calculations.

Usage:
  python backtest.py
  python backtest.py --start 2018-01-01 --end 2024-12-31
  python backtest.py --abs-threshold 0.0 --no-abs-filter
  python backtest.py --lookback 252 --skip 21 --hold-buffer 20

Outputs (uploaded as GitHub Actions artifacts):
  backtest_results.csv     — daily NAV
  backtest_trades.csv      — every BUY/SELL with hold months, reason, P&L
  backtest_performance.csv — metrics vs Nifty 50 benchmark
"""

import argparse
import logging
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    ABSOLUTE_MOMENTUM_THRESHOLD,
    HARD_STOP_PCT,
    MIN_HISTORY_BARS,
    MOMENTUM_LOOKBACK_DAYS,
    SKIP_RECENT_DAYS,
    WEIGHTING_SCHEME,
    WEEKLY_RANK_STOP_NIFTY200,
)
from universes import get_universe, list_universes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backtest")

BENCHMARK_TICKER = "^NSEI"
TRANSACTION_COST = 0.0025    # 0.25% one-way: brokerage + STT + slippage
INITIAL_CAPITAL  = 1_500_000
RISK_FREE_RATE   = 0.065


# ── Data download ─────────────────────────────────────────────────────────────

def download_prices(symbols: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.Series]:
    """Download all stock + benchmark closes for the backtest period."""
    # Add extra history for the lookback window before start date
    extended_start = pd.Timestamp(start) - timedelta(days=430)  # ~14 months buffer
    tickers = [f"{s}.NS" for s in symbols] + [BENCHMARK_TICKER]

    logger.info(f"Downloading {len(tickers)} tickers {extended_start.date()} -> {end}...")
    raw = yf.download(
        tickers,
        start=extended_start.strftime("%Y-%m-%d"),
        end=end,
        auto_adjust=True,
        progress=True,
    )
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    closes.index = pd.to_datetime(closes.index).tz_localize(None)

    bench  = closes[BENCHMARK_TICKER].dropna() if BENCHMARK_TICKER in closes.columns else None
    stocks = closes.drop(columns=[BENCHMARK_TICKER], errors="ignore").copy()
    stocks.columns = [c.replace(".NS", "") for c in stocks.columns]

    threshold = int(0.60 * len(stocks))
    stocks = stocks.dropna(axis=1, thresh=threshold).ffill().bfill()
    logger.info(f"Retained {len(stocks.columns)} stocks after quality filter")
    return stocks, bench


def find_month_ends(index: pd.DatetimeIndex, start: str, end: str) -> list[pd.Timestamp]:
    """Return the last available trading date for each calendar month in [start, end]."""
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)
    in_range = index[(index >= start_ts) & (index <= end_ts)]
    df = pd.DataFrame({"date": in_range})
    df["month"] = df["date"].dt.to_period("M")
    return df.groupby("month")["date"].last().tolist()


# ── Per-bar momentum ranking ──────────────────────────────────────────────────

def rank_at_bar(
    prices_df: pd.DataFrame,
    bar_idx:   int,
    lookback:  int,
    skip:      int,
) -> pd.DataFrame:
    """
    Compute 12M-1M momentum return for every stock at a given bar index.
    Returns DataFrame indexed by symbol with columns: momentum_return, price, rank.
    """
    required = lookback + skip + 5
    rows = []
    slice_df = prices_df.iloc[: bar_idx + 1]

    for sym in slice_df.columns:
        series = slice_df[sym].dropna()
        if len(series) < required:
            continue
        try:
            price_end   = float(series.iloc[-(skip + 1)])
            price_start = float(series.iloc[-(lookback + skip + 1)])
            current     = float(series.iloc[-1])
        except (IndexError, ValueError):
            continue
        if price_start <= 0 or current <= 0:
            continue
        mom = (price_end - price_start) / price_start
        rows.append({"symbol": sym, "momentum_return": mom, "price": current})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("symbol")
    df.sort_values("momentum_return", ascending=False, inplace=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def abs_momentum_at_bar(
    bench:     pd.Series,
    bar_idx:   int,
    lookback:  int,
    skip:      int,
    threshold: float,
) -> tuple[bool, float]:
    """
    Compute Nifty 50 absolute momentum at a given bar index.
    Returns (is_risk_on, abs_return).
    """
    if bench is None:
        return True, 0.0
    required = lookback + skip + 5
    series = bench.iloc[: bar_idx + 1].dropna()
    if len(series) < required:
        return True, 0.0
    try:
        price_end   = float(series.iloc[-(skip + 1)])
        price_start = float(series.iloc[-(lookback + skip + 1)])
    except (IndexError, ValueError):
        return True, 0.0
    if price_start <= 0:
        return True, 0.0
    abs_ret = (price_end - price_start) / price_start
    return abs_ret > threshold, abs_ret


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_backtest(
    prices_df:        pd.DataFrame,
    bench:            pd.Series,
    start:            str,
    end:              str,
    portfolio_size:   int,
    initial_capital:  float,
    lookback:         int,
    skip:             int,
    abs_threshold:    float,
    hold_buffer:      int,
    hard_stop:        float | None,
    use_abs_filter:   bool,
    use_weekly_stop:  bool = True,
    use_graded:       bool = True,
    weekly_rank_stop: int | None = None,
) -> dict:
    required_bars = lookback + skip + 5

    cash      = initial_capital
    holdings  = {}   # symbol -> {shares, entry_price, entry_date}
    nav_daily = {}
    trades    = []

    # Pre-compute month-end rebalance dates
    month_ends     = find_month_ends(prices_df.index, start, end)
    month_end_set  = set(month_ends)
    total_days     = len(prices_df)

    # Align benchmark index for fast lookup
    bench_idx_map = (
        {ts: i for i, ts in enumerate(bench.index)} if bench is not None else {}
    )

    for bar_idx, date in enumerate(prices_df.index):

        # Skip bars before we have enough history
        if bar_idx < required_bars:
            nav_daily[date] = initial_capital
            continue

        # Skip bars before the requested start date
        if date < pd.Timestamp(start):
            nav_daily[date] = initial_capital
            continue

        if bar_idx % 60 == 0:
            logger.info(f"  {bar_idx}/{total_days} bars ({date.date()})...")

        day_prices = prices_df.iloc[bar_idx]

        # ── Daily mark-to-market NAV ──────────────────────────────────────────
        port_value = cash
        for sym, pos in holdings.items():
            p = day_prices.get(sym)
            if p is not None and not np.isnan(p):
                port_value += pos["shares"] * p

        # ── Weekly stops — every Friday (mirrors weekly_scan.py) ─────────────
        # Runs BEFORE the monthly rebalance so that if month-end falls on a
        # Friday, weekly stops fire first and the monthly can refill freed slots.
        # Checks both the price hard stop and the rank-based stop.
        is_friday = date.dayofweek == 4   # 0 = Monday, 4 = Friday
        if use_weekly_stop and is_friday and holdings:
            # Rank universe on this Friday if weekly rank stop is active
            friday_scores = pd.DataFrame()
            if weekly_rank_stop is not None:
                friday_scores = rank_at_bar(prices_df, bar_idx, lookback, skip)

            weekly_exits = []
            for sym, pos in holdings.items():
                cur_p = day_prices.get(sym)
                if cur_p is None or np.isnan(cur_p):
                    continue

                # Price hard stop
                if hard_stop is not None:
                    loss_pct = (pos["entry_price"] - cur_p) / pos["entry_price"]
                    if loss_pct >= hard_stop:
                        weekly_exits.append((sym, cur_p, "WEEKLY_HARD_STOP"))
                        continue

                # Rank-based stop
                if weekly_rank_stop is not None and not friday_scores.empty:
                    if sym not in friday_scores.index:
                        weekly_exits.append((sym, cur_p, "WEEKLY_RANK_EXIT"))
                    elif friday_scores.loc[sym, "rank"] > weekly_rank_stop:
                        weekly_exits.append((sym, cur_p, "WEEKLY_RANK_EXIT"))

            for sym, cur_p, rsn in weekly_exits:
                pos      = holdings.pop(sym)
                proceeds = pos["shares"] * cur_p * (1 - TRANSACTION_COST)
                cash    += proceeds
                pnl_inr  = round((cur_p - pos["entry_price"]) * pos["shares"], 2)
                pnl_pct  = round((cur_p - pos["entry_price"]) / pos["entry_price"] * 100, 2)
                sym_rank = (
                    int(friday_scores.loc[sym, "rank"])
                    if not friday_scores.empty and sym in friday_scores.index else 0
                )
                trades.append({
                    "date":            date.date().isoformat(),
                    "symbol":          sym,
                    "action":          "SELL",
                    "price":           round(cur_p, 2),
                    "shares":          pos["shares"],
                    "value_inr":       round(proceeds, 0),
                    "entry_price":     round(pos["entry_price"], 2),
                    "hold_days":       (date - pos["entry_date"]).days,
                    "reason":          rsn,
                    "momentum_return": 0.0,
                    "rank":            sym_rank,
                    "pnl_inr":         pnl_inr,
                    "pnl_pct":         pnl_pct,
                })

        # ── Monthly rebalance on month-end dates only ─────────────────────────
        if date in month_end_set:

            # Bench bar index
            b_idx = bench_idx_map.get(date, bar_idx)

            # Absolute momentum
            if use_abs_filter:
                risk_on, abs_ret = abs_momentum_at_bar(bench, b_idx, lookback, skip, abs_threshold)
            else:
                risk_on, abs_ret = True, 0.0

            # Relative momentum ranking
            scores = rank_at_bar(prices_df, bar_idx, lookback, skip)

            # ── EXITS ─────────────────────────────────────────────────────────
            to_sell = []
            for sym, pos in list(holdings.items()):
                cur_p = day_prices.get(sym)
                if cur_p is None or np.isnan(cur_p):
                    to_sell.append((sym, "DATA_GAP"))
                    continue

                # Risk-off → sell everything
                if not risk_on:
                    to_sell.append((sym, "RISK_OFF"))
                    continue

                # Monthly hard stop (backup — catches any breach missed between Fridays)
                if hard_stop is not None:
                    loss_pct = (pos["entry_price"] - cur_p) / pos["entry_price"]
                    if loss_pct >= hard_stop:
                        rsn = "MONTHLY_HARD_STOP" if use_weekly_stop else "HARD_STOP"
                        to_sell.append((sym, rsn))
                        continue

                # Rank exit
                if sym not in scores.index:
                    to_sell.append((sym, "NOT_RANKED"))
                    continue

                if scores.loc[sym, "rank"] > hold_buffer:
                    to_sell.append((sym, f"RANK_EXIT"))
                    continue

            for sym, rsn in to_sell:
                cur_p = day_prices.get(sym)
                if cur_p is None or np.isnan(cur_p):
                    cur_p = holdings[sym]["entry_price"]
                pos      = holdings.pop(sym)
                proceeds = pos["shares"] * cur_p * (1 - TRANSACTION_COST)
                cash    += proceeds
                pnl_inr  = round((cur_p - pos["entry_price"]) * pos["shares"], 2)
                pnl_pct  = round((cur_p - pos["entry_price"]) / pos["entry_price"] * 100, 2)
                hold_days = (date - pos["entry_date"]).days
                mom_ret  = (scores.loc[sym, "momentum_return"]
                            if sym in scores.index else 0.0)
                trades.append({
                    "date":        date.date().isoformat(),
                    "symbol":      sym,
                    "action":      "SELL",
                    "price":       round(cur_p, 2),
                    "shares":      pos["shares"],
                    "value_inr":   round(proceeds, 0),
                    "entry_price": round(pos["entry_price"], 2),
                    "hold_days":   hold_days,
                    "reason":      rsn,
                    "momentum_return": round(mom_ret, 4),
                    "rank":        scores.loc[sym, "rank"] if sym in scores.index else 0,
                    "pnl_inr":     pnl_inr,
                    "pnl_pct":     pnl_pct,
                })

            # ── ENTRIES ───────────────────────────────────────────────────────
            if risk_on and not scores.empty:
                slots        = portfolio_size - len(holdings)
                held_symbols = set(holdings.keys())
                top_ranked   = scores[scores["rank"] <= portfolio_size]
                candidates   = [s for s in top_ranked.index if s not in held_symbols][:slots]

                total_weight = portfolio_size * (portfolio_size + 1) // 2  # N*(N+1)/2

                for sym in candidates:
                    cur_p = day_prices.get(sym)
                    if cur_p is None or np.isnan(cur_p) or cur_p <= 0:
                        continue
                    cost_ps  = cur_p * (1 + TRANSACTION_COST)
                    sym_rank = int(scores.loc[sym, "rank"])

                    if use_graded:
                        # Graded: rank 1 gets most, rank N gets least
                        weight = portfolio_size + 1 - sym_rank
                        alloc  = port_value * weight / total_weight
                    else:
                        alloc  = port_value / portfolio_size

                    shares = int(alloc / cost_ps)
                    if shares < 1:
                        continue
                    cash -= shares * cost_ps
                    holdings[sym] = {
                        "shares":      shares,
                        "entry_price": cur_p,
                        "entry_date":  date,
                    }
                    trades.append({
                        "date":        date.date().isoformat(),
                        "symbol":      sym,
                        "action":      "BUY",
                        "price":       round(cur_p, 2),
                        "shares":      shares,
                        "value_inr":   round(shares * cost_ps, 0),
                        "entry_price": round(cur_p, 2),
                        "hold_days":   0,
                        "reason":      "MOMENTUM_ENTRY",
                        "momentum_return": round(scores.loc[sym, "momentum_return"], 4),
                        "rank":        sym_rank,
                        "pnl_inr":     0,
                        "pnl_pct":     0,
                    })

        # ── Revalue + record NAV ──────────────────────────────────────────────
        port_value = cash
        for sym, pos in holdings.items():
            p = day_prices.get(sym)
            if p is not None and not np.isnan(p):
                port_value += pos["shares"] * p
        nav_daily[date] = port_value

    return {
        "nav":    pd.Series(nav_daily),
        "trades": pd.DataFrame(trades) if trades else pd.DataFrame(),
    }


# ── Performance metrics ───────────────────────────────────────────────────────

def calc_metrics(series: pd.Series, name: str) -> dict:
    if len(series) < 2:
        return {"Strategy": name}
    ret     = series.pct_change().dropna()
    n_years = (series.index[-1] - series.index[0]).days / 365.25
    total   = (series.iloc[-1] / series.iloc[0]) - 1
    cagr    = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
    vol     = ret.std() * np.sqrt(252)
    sharpe  = (cagr - RISK_FREE_RATE) / vol if vol > 0 else 0
    roll_max = series.cummax()
    dd       = (series - roll_max) / roll_max
    max_dd   = dd.min()
    calmar   = cagr / abs(max_dd) if max_dd < 0 else 0
    monthly  = series.resample("ME").last().pct_change().dropna()
    wins     = (monthly > 0).sum()
    return {
        "Strategy":     name,
        "Total Return": f"{total:.1%}",
        "CAGR":         f"{cagr:.1%}",
        "Volatility":   f"{vol:.1%}",
        "Sharpe":       f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.1%}",
        "Calmar":       f"{calmar:.2f}",
        "Monthly Win":  (f"{wins}/{len(monthly)} ({wins/len(monthly):.0%})"
                         if len(monthly) else "N/A"),
    }


def print_trade_stats(trades: pd.DataFrame) -> None:
    sells = trades[trades["action"] == "SELL"]
    buys  = trades[trades["action"] == "BUY"]
    n     = len(sells)
    if n == 0:
        print("  No closed trades.")
        return
    wins      = (sells["pnl_inr"] > 0).sum()
    total_pnl = sells["pnl_inr"].sum()
    avg_hold  = sells["hold_days"].mean()
    reasons   = sells["reason"].value_counts()

    print(f"  Total BUY trades     : {len(buys)}")
    print(f"  Total SELL trades    : {n}")
    print(f"  Win rate (sells)     : {wins}/{n} ({wins/n:.0%})")
    print(f"  Total realised P&L   : Rs{total_pnl:+,.0f}")
    print(f"  Avg hold (days)      : {avg_hold:.1f}")
    print(f"  Exit reason breakdown:")
    for rsn, cnt in reasons.items():
        pct = cnt / n * 100
        print(f"    {rsn:<22}: {cnt:>4}  ({pct:.0f}%)")
    if n > 0:
        best  = sells.loc[sells["pnl_inr"].idxmax()]
        worst = sells.loc[sells["pnl_inr"].idxmin()]
        print(f"  Best trade  : {best['symbol']} Rs{best['pnl_inr']:+,.0f} "
              f"({best['pnl_pct']:+.1f}%) [{best['reason']}]")
        print(f"  Worst trade : {worst['symbol']} Rs{worst['pnl_inr']:+,.0f} "
              f"({worst['pnl_pct']:+.1f}%) [{worst['reason']}]")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Dual Momentum Delivery — Monthly Backtest")
    p.add_argument("--universe",       choices=list_universes(), default="NIFTY200",
                   help="Universe to backtest: NIFTY200 or BEES (default: NIFTY200)")
    p.add_argument("--start",          default="2018-01-01",
                   help="Backtest start date (YYYY-MM-DD)")
    p.add_argument("--end",            default="2024-12-31",
                   help="Backtest end date (YYYY-MM-DD)")
    p.add_argument("--portfolio-size", default=None,              type=int,
                   help="Max positions (default: universe default — 15 for NIFTY200, 5 for BEES)")
    p.add_argument("--capital",        default=INITIAL_CAPITAL,   type=float,
                   help=f"Initial capital in Rs (default {INITIAL_CAPITAL:,.0f})")
    p.add_argument("--lookback",       default=MOMENTUM_LOOKBACK_DAYS, type=int,
                   help=f"Momentum lookback in trading days (default {MOMENTUM_LOOKBACK_DAYS})")
    p.add_argument("--skip",           default=SKIP_RECENT_DAYS,  type=int,
                   help=f"Skip-month days (default {SKIP_RECENT_DAYS})")
    p.add_argument("--abs-threshold",  default=ABSOLUTE_MOMENTUM_THRESHOLD, type=float,
                   help=f"Absolute momentum threshold, e.g. -0.05 = -5%% (default {ABSOLUTE_MOMENTUM_THRESHOLD})")
    p.add_argument("--hold-buffer",    default=None,              type=int,
                   help="Sell if rank > this (default: universe default — 20 for NIFTY200, 7 for BEES)")
    p.add_argument("--hard-stop",      default=HARD_STOP_PCT,     type=float,
                   help=f"Hard stop fraction e.g. 0.15 (default {HARD_STOP_PCT})")
    p.add_argument("--no-abs-filter",    action="store_true",
                   help="Disable absolute momentum filter (pure relative momentum)")
    p.add_argument("--no-weekly-stop",  action="store_true",
                   help="Disable weekly Friday stop simulation (default: enabled)")
    p.add_argument("--tranche",          default="",
                   help=(
                       "Tranche label for output file naming (e.g. T1, T2). "
                       "Produces backtest_results_nifty200_t1.csv etc. "
                       "Useful when comparing independent tranche parameters side-by-side."
                   ))
    p.add_argument("--weighting",       choices=["graded", "equal"], default=None,
                   help=(
                       "Position sizing: 'graded' (rank-weighted, default from config) "
                       "or 'equal' (flat per slot). "
                       f"Config default: {WEIGHTING_SCHEME.lower()}"
                   ))
    p.add_argument("--weekly-rank-stop", default=None, type=int,
                   help=(
                       "Exit mid-week if rank > this (default: from universe config). "
                       "Pass 0 to disable weekly rank stop in backtest."
                   ))
    return p.parse_args()


def main():
    args       = parse_args()
    ucfg       = get_universe(args.universe)
    use_abs    = not args.no_abs_filter
    use_weekly = not args.no_weekly_stop
    hard_stop  = args.hard_stop if args.hard_stop and args.hard_stop > 0 else None

    # Universe-specific defaults (overridable via CLI)
    port_size   = args.portfolio_size if args.portfolio_size is not None else ucfg.top_n_hold
    hold_buffer = args.hold_buffer    if args.hold_buffer    is not None else ucfg.hold_buffer

    # Graded weighting — default from config, overridable via --weighting
    if args.weighting is not None:
        use_graded = (args.weighting == "graded")
    else:
        use_graded = (WEIGHTING_SCHEME == "GRADED")

    # Weekly rank stop — default from universe config, overridable via --weekly-rank-stop
    if args.weekly_rank_stop is not None:
        weekly_rank_stop = args.weekly_rank_stop if args.weekly_rank_stop > 0 else None
    else:
        weekly_rank_stop = ucfg.weekly_rank_stop

    # Output file names include universe + optional tranche suffix
    tranche_suffix = f"_{args.tranche.strip().lower()}" if args.tranche.strip() else ""
    out_results = f"backtest_results_{ucfg.name.lower()}{tranche_suffix}.csv"
    out_trades  = f"backtest_trades_{ucfg.name.lower()}{tranche_suffix}.csv"
    out_perf    = f"backtest_performance_{ucfg.name.lower()}{tranche_suffix}.csv"

    sep = "=" * 68
    weighting_label = "GRADED (rank-weighted)" if use_graded else "EQUAL (flat per slot)"
    rank_stop_label = (
        f"ON — exit if rank > {weekly_rank_stop} (monthly hold_buffer={hold_buffer})"
        if use_weekly and weekly_rank_stop else
        ("ON — price stop only" if use_weekly else "OFF")
    )
    print(sep)
    print("  DUAL MOMENTUM DELIVERY BOT — MONTHLY BACKTEST")
    print(f"  Period          : {args.start}  ->  {args.end}")
    print(f"  Universe        : {ucfg.display_name}  ({len(ucfg.symbols)} symbols)")
    print(f"  Portfolio size  : {port_size} slots  ({weighting_label})")
    print(f"  Lookback        : {args.lookback} days (~12 months)")
    print(f"  Skip recent     : {args.skip} days (~1 month, avoids reversal)")
    print(f"  Abs. momentum   : {'ON — threshold ' + str(args.abs_threshold) if use_abs else 'OFF (pure relative)'}")
    print(f"  Hold buffer     : sell if rank > {hold_buffer}")
    print(f"  Hard stop       : {hard_stop:.0%}" if hard_stop else "  Hard stop       : OFF")
    print(f"  Weekly stop     : {rank_stop_label}")
    print(f"  Transaction cost: {TRANSACTION_COST*100:.2f}% per trade (one-way)")
    print(f"  Initial capital : Rs{args.capital:,.0f}")
    print(sep)

    prices_df, bench = download_prices(list(ucfg.symbols), args.start, args.end)
    if prices_df.empty:
        logger.error("No price data. Exiting.")
        sys.exit(1)

    logger.info(f"Running monthly rebalances from {args.start} to {args.end}...")
    result = run_backtest(
        prices_df        = prices_df,
        bench            = bench,
        start            = args.start,
        end              = args.end,
        portfolio_size   = port_size,
        initial_capital  = args.capital,
        lookback         = args.lookback,
        skip             = args.skip,
        abs_threshold    = args.abs_threshold,
        hold_buffer      = hold_buffer,
        hard_stop        = hard_stop,
        use_abs_filter   = use_abs,
        use_weekly_stop  = use_weekly,
        use_graded       = use_graded,
        weekly_rank_stop = weekly_rank_stop,
    )
    nav    = result["nav"]
    trades = result["trades"]

    # ── Trim NAV to requested date range ─────────────────────────────────────
    nav = nav[nav.index >= pd.Timestamp(args.start)]

    # ── Benchmark ─────────────────────────────────────────────────────────────
    bench_aligned = None
    if bench is not None and len(nav) > 0:
        bd = [d for d in nav.index if d in bench.index]
        if bd:
            bv = bench.reindex(bd).dropna()
            bench_aligned = bv / bv.iloc[0] * args.capital

    # ── Strategy metrics ──────────────────────────────────────────────────────
    sm = calc_metrics(nav, "Dual Momentum (Monthly)")
    bm = calc_metrics(bench_aligned, "Nifty 50 (Benchmark)") if bench_aligned is not None else None

    print(f"\n{sep}")
    print("  STRATEGY PERFORMANCE")
    print(sep)
    for k, v in sm.items():
        print(f"  {k:<22}: {v}")

    if bm:
        print(f"\n{sep}")
        print("  BENCHMARK — NIFTY 50")
        print(sep)
        for k, v in bm.items():
            print(f"  {k:<22}: {v}")

    # ── Annual returns ────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  ANNUAL RETURNS — {ucfg.display_name}")
    print(sep)
    print(f"  {'Year':<8} {'Strategy':>12} {'Nifty 50':>12} {'Alpha':>10}")
    print("  " + "-" * 46)
    for yr in sorted(nav.index.year.unique()):
        s_yr = nav[nav.index.year == yr]
        if len(s_yr) < 2:
            continue
        sr    = (s_yr.iloc[-1] / s_yr.iloc[0]) - 1
        b_str = a_str = "N/A"
        if bench_aligned is not None:
            b_yr = bench_aligned[bench_aligned.index.year == yr]
            if len(b_yr) >= 2:
                br    = (b_yr.iloc[-1] / b_yr.iloc[0]) - 1
                b_str = f"{br:+.1%}"
                a_str = f"{sr - br:+.1%}"
        print(f"  {yr:<8} {sr:>+12.1%} {b_str:>12} {a_str:>10}")

    # ── Regime periods summary ────────────────────────────────────────────────
    if not trades.empty:
        risk_off_months = trades[trades["reason"] == "RISK_OFF"]["date"].nunique()
        print(f"\n{sep}")
        print("  TRADE STATISTICS")
        print(sep)
        print_trade_stats(trades)
        if risk_off_months:
            print(f"\n  Risk-off months (all-cash): ~{risk_off_months}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    nav_df = nav.reset_index()
    nav_df.columns = ["date", "nav"]
    nav_df.to_csv(out_results, index=False)

    if not trades.empty:
        trades.to_csv(out_trades, index=False)

    rows = [sm]
    if bm:
        rows.append(bm)
    pd.DataFrame(rows).to_csv(out_perf, index=False)

    print(f"\n  Saved: {out_results} | {out_trades} | {out_perf}")
    print(f"\n{sep}")
    print("  BACKTEST COMPLETE")
    print(sep)


if __name__ == "__main__":
    main()
