"""
backtest.py
-----------
Dual Momentum Delivery Bot — Monthly Backtest Engine.

Simulates the EXACT same logic as main.py + weekly_scan.py:

  Monthly rebalance (last trading day of each month):
    1. Absolute momentum: Is Nifty 50 12M-1M return > abs_threshold?
       NO  → sell all, sit in cash until next month's check
       YES → proceed to relative momentum
    2. Relative momentum: Rank universe by 12M-1M return
       SELL: holdings ranked below hold_buffer
       BUY:  fill slots with top-N stocks not already held
    3. Monthly hard stop: sell any position down > hard_stop_pct from entry

  Weekly hard stops (every Friday, mirrors weekly_scan.py):
    1. Price hard stop: sell if down > hard_stop_pct from entry
    2. Rank stop: sell if current rank > weekly_rank_stop

  RANKING: uses momentum_scorer.rank_universe() — the SAME function as
  live trading. There is no separate backtest ranking implementation.

ALLOCATION MODEL:
  Default (--allocation-mode fixed):
    alloc = total_capital * weight / sum_weights
    This matches the live bot exactly. Capital does not compound — gains
    accumulate as uninvested cash. This is what most retail investors do
    (they commit a fixed sum, not a percentage of growing wealth).

  Optional (--allocation-mode nav):
    alloc = portfolio_NAV * weight / sum_weights
    Capital grows with wins, shrinks with losses. This is the "reinvest all
    gains" model. Produces higher CAGR in backtests but diverges from how
    the live bot actually behaves.

  Use --allocation-mode nav only if you understand the divergence.

NAV is marked-to-market every trading day for accurate Sharpe/drawdown.
"""

import argparse
import logging
import sys
import time
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from momentum_scorer import MomentumRank, graded_allocation, rank_universe
from strategy import NIFTY200_DEFAULTS, StrategyParams
from universes import UniverseConfig, get_universe, list_universes

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] — %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backtest")


# ── Data download ─────────────────────────────────────────────────────────────

_BATCH_SIZE        = 40    # tickers per yfinance call (same as data_feed.py)
_DOWNLOAD_RETRIES  = 3     # attempts per batch before giving up
_DOWNLOAD_DELAY    = 5     # seconds between retry attempts


def download_prices(
    symbols:  list[str],
    start:    str,
    end:      str,
    lookback: int,
    skip:     int,
    regime_ticker: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Download adjusted close prices for universe + benchmark.

    Uses period="max" + group_by="ticker" — the SAME API path as data_feed.py.
    This is critical: Yahoo Finance has two API endpoints:

      period=  → /v8/finance/chart   (permissive, works from GitHub Actions)
      start=/end= → download API    (aggressively rate-limited from DC IPs)

    The live bot's data_feed.py uses period= and never has download failures.
    Using start=/end= here caused all 166 tickers to fail simultaneously on
    GitHub Actions, returning column names with 0 rows — a silent failure.

    After downloading "max" history we slice to the required date window.
    Extends the start date by (lookback + skip + 30) calendar days so enough
    history exists before the first rebalance bar.

    Returns:
        (stocks_df, bench_series) — both with tz-naive DatetimeIndex.
        stocks_df is an empty DataFrame on total failure.
        bench_series is None if the benchmark download fails.
    """
    buffer_days    = int((lookback + skip + 30) * 1.45)
    extended_start = pd.Timestamp(start) - timedelta(days=buffer_days)
    end_ts         = pd.Timestamp(end)
    ns_tickers     = [f"{s}.NS" for s in symbols]

    logger.info(
        f"Downloading {len(ns_tickers)} universe tickers in batches of {_BATCH_SIZE} "
        f"(period=max, then filtering to {extended_start.date()} → {end})..."
    )

    # ── Universe: batched download with retry ─────────────────────────────────
    # Mirror data_feed.fetch_universe_prices() exactly: period=, group_by="ticker",
    # xs("Close", level=1) — this is the proven path that avoids rate-limiting.
    all_dfs: list[pd.DataFrame] = []
    batches = [
        ns_tickers[i : i + _BATCH_SIZE]
        for i in range(0, len(ns_tickers), _BATCH_SIZE)
    ]

    for b_num, batch in enumerate(batches, start=1):
        for attempt in range(1, _DOWNLOAD_RETRIES + 1):
            try:
                raw = yf.download(
                    batch,
                    period      = "max",     # ← chart API, not download API
                    interval    = "1d",
                    auto_adjust = True,
                    progress    = False,
                    group_by    = "ticker",  # ← (Ticker, OHLCV) MultiIndex
                )
                if raw.empty:
                    raise ValueError("yfinance returned empty DataFrame")

                if isinstance(raw.columns, pd.MultiIndex):
                    closes = raw.xs("Close", level=1, axis=1).copy()
                else:
                    # Single-ticker: flat OHLCV columns
                    closes = raw[["Close"]].copy()
                    closes.columns = batch

                closes.columns = [c.replace(".NS", "") for c in closes.columns]
                closes.index   = pd.to_datetime(closes.index).tz_localize(None)

                if len(closes) == 0:
                    raise ValueError("DataFrame has columns but 0 data rows")

                all_dfs.append(closes)
                logger.info(
                    f"  Batch {b_num}/{len(batches)} OK — "
                    f"{len(closes.columns)} tickers, {len(closes)} rows"
                )
                break

            except Exception as exc:
                if attempt < _DOWNLOAD_RETRIES:
                    logger.warning(
                        f"  Batch {b_num}/{len(batches)} attempt {attempt} failed "
                        f"({exc}). Retrying in {_DOWNLOAD_DELAY}s..."
                    )
                    time.sleep(_DOWNLOAD_DELAY)
                else:
                    logger.error(
                        f"  Batch {b_num}/{len(batches)} failed after "
                        f"{_DOWNLOAD_RETRIES} attempts — skipping."
                    )

    if not all_dfs:
        logger.error(
            "All batches failed — cannot download universe prices. "
            "Yahoo Finance may be temporarily unavailable. Try again in a few minutes."
        )
        return pd.DataFrame(), None

    stocks = pd.concat(all_dfs, axis=1)

    if len(stocks) == 0:
        logger.error("Combined DataFrame has 0 rows after concatenation.")
        return pd.DataFrame(), None

    # ── Benchmark: use yf.Ticker().history() — same as data_feed.get_absolute_momentum()
    bench: pd.Series | None = None
    for attempt in range(1, _DOWNLOAD_RETRIES + 1):
        try:
            raw_b = yf.Ticker(regime_ticker).history(
                period      = "max",
                interval    = "1d",
                auto_adjust = True,
            )
            if raw_b.empty:
                raise ValueError("empty response")

            b_close       = raw_b["Close"]
            b_close.index = pd.to_datetime(b_close.index).tz_localize(None)
            bench         = b_close.dropna()
            logger.info(f"  Benchmark {regime_ticker} OK — {len(bench)} rows")
            break

        except Exception as exc:
            if attempt < _DOWNLOAD_RETRIES:
                logger.warning(
                    f"  Benchmark attempt {attempt} failed ({exc}). "
                    f"Retrying in {_DOWNLOAD_DELAY}s..."
                )
                time.sleep(_DOWNLOAD_DELAY)
            else:
                logger.warning(
                    f"  Benchmark {regime_ticker} unavailable after "
                    f"{_DOWNLOAD_RETRIES} attempts — absolute momentum "
                    f"check will default to RISK-ON."
                )

    # ── Slice to required date window ─────────────────────────────────────────
    # We downloaded full history; trim to [extended_start, end] now.
    stocks = stocks.loc[
        (stocks.index >= extended_start) & (stocks.index <= end_ts)
    ]
    if bench is not None:
        bench = bench.loc[
            (bench.index >= extended_start) & (bench.index <= end_ts)
        ]

    if len(stocks) == 0:
        logger.error(
            f"No trading days found between {extended_start.date()} and {end} "
            "after date filtering. Check your --start/--end dates."
        )
        return pd.DataFrame(), bench

    # ── Quality filter (identical to data_feed.fetch_universe_prices) ─────────
    # Drop symbols with >40% missing rows; forward-fill remaining gaps.
    threshold = int(0.60 * len(stocks))
    stocks    = stocks.dropna(axis=1, thresh=threshold)
    stocks    = stocks.ffill()

    logger.info(
        f"Universe ready: {len(stocks.columns)} symbols × {len(stocks)} trading days "
        f"({stocks.index[0].date()} → {stocks.index[-1].date()})"
    )
    return stocks, bench


def find_month_ends(
    index: pd.DatetimeIndex, start: str, end: str
) -> list[pd.Timestamp]:
    """Return the last available trading date for each calendar month in [start, end]."""
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)
    in_range = index[(index >= start_ts) & (index <= end_ts)]
    df = pd.DataFrame({"date": in_range})
    df["month"] = df["date"].dt.to_period("M")
    return df.groupby("month")["date"].last().tolist()


# ── Benchmark momentum ────────────────────────────────────────────────────────

def abs_momentum_at_bar(
    bench:     Optional[pd.Series],
    bar_idx:   int,
    lookback:  int,
    skip:      int,
    threshold: float,
) -> tuple[bool, float]:
    """Compute absolute momentum for the benchmark at the given bar index."""
    if bench is None:
        return True, 0.0
    required = lookback + skip + 5
    series   = bench.iloc[: bar_idx + 1].dropna()
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


# ── Helper: convert rank_universe output to DataFrame ─────────────────────────

def _ranked_to_df(ranked: list[MomentumRank]) -> pd.DataFrame:
    """Convert rank_universe() list → DataFrame indexed by symbol.
    Columns: momentum_return, price, rank.
    """
    if not ranked:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "momentum_return": [r.momentum_return for r in ranked],
            "price":           [r.current_price   for r in ranked],
            "rank":            [r.rank             for r in ranked],
        },
        index=[r.symbol for r in ranked],
    )


# ── Main backtest engine ──────────────────────────────────────────────────────

def run_backtest(
    prices_df:       pd.DataFrame,
    bench:           Optional[pd.Series],
    start:           str,
    end:             str,
    params:          StrategyParams,
    allocation_mode: str = "fixed",   # "fixed" | "nav"
) -> dict:
    """Simulate the full Dual Momentum strategy over historical data.

    Args:
        prices_df:       Daily close prices (symbols as columns).
        bench:           Benchmark daily close series (for abs momentum).
        start, end:      Date range for the backtest (YYYY-MM-DD strings).
        params:          StrategyParams — all tunable strategy parameters.
        allocation_mode: "fixed" = use params.total_capital (matches live bot).
                         "nav"   = use current portfolio NAV (compounding mode).

    Returns:
        dict with keys "nav" (pd.Series) and "trades" (pd.DataFrame).
    """
    lookback    = params.lookback_days
    skip        = params.skip_days
    required    = lookback + skip + 5
    port_size   = params.top_n_hold
    hold_buffer = params.hold_buffer
    hard_stop   = params.hard_stop_pct
    use_abs     = params.regime_filter
    use_graded  = params.weighting.upper() == "GRADED"
    wk_stop     = params.weekly_rank_stop
    tc          = params.transaction_cost
    capital     = float(params.total_capital)
    abs_thr     = params.abs_threshold
    abs_ticker  = params.regime_ticker

    cash      = capital
    holdings  : dict[str, dict] = {}  # sym → {shares, entry_price, entry_date}
    nav_daily : dict            = {}
    trades    : list[dict]      = []

    month_ends    = find_month_ends(prices_df.index, start, end)
    month_end_set = set(month_ends)
    total_days    = len(prices_df.index)

    bench_idx_map = (
        {ts: i for i, ts in enumerate(bench.index)} if bench is not None else {}
    )

    # Pre-compute total_weight for graded allocation
    total_weight_grad = port_size * (port_size + 1) // 2  # N*(N+1)/2

    for bar_idx, dt in enumerate(prices_df.index):

        if bar_idx < required:
            nav_daily[dt] = capital
            continue
        if dt < pd.Timestamp(start):
            nav_daily[dt] = capital
            continue

        if bar_idx % 60 == 0:
            logger.info(f"  {bar_idx:>5}/{total_days} bars  ({dt.date()})...")

        day_prices = prices_df.iloc[bar_idx]

        # ── Mark-to-market portfolio NAV ──────────────────────────────────────
        port_value = cash
        for sym, pos in holdings.items():
            p = day_prices.get(sym)
            if p is not None and not np.isnan(p):
                port_value += pos["shares"] * p

        # ── Weekly hard stops — every Friday ──────────────────────────────────
        # Runs BEFORE the monthly block so that on a Friday month-end,
        # weekly stops fire first and the monthly can refill freed slots.
        is_friday = (dt.dayofweek == 4)
        if is_friday and holdings:
            # Re-rank on Fridays only if weekly rank stop is active
            friday_scores = pd.DataFrame()
            if wk_stop is not None:
                slice_df      = prices_df.iloc[: bar_idx + 1]
                ranked_fri    = rank_universe(slice_df, lookback=lookback, skip=skip)
                friday_scores = _ranked_to_df(ranked_fri)

            weekly_exits: list[tuple] = []
            for sym, pos in holdings.items():
                cur_p = day_prices.get(sym)
                if cur_p is None or np.isnan(cur_p):
                    continue

                # 1. Price hard stop (highest priority)
                loss_pct = (pos["entry_price"] - cur_p) / pos["entry_price"]
                if loss_pct >= hard_stop:
                    weekly_exits.append((sym, cur_p, f"WEEKLY_HARD_STOP({loss_pct:.1%})"))
                    continue

                # 2. Rank-based stop
                if wk_stop is not None and not friday_scores.empty:
                    if sym not in friday_scores.index:
                        weekly_exits.append((sym, cur_p, "WEEKLY_RANK_EXIT(unranked)"))
                    elif friday_scores.loc[sym, "rank"] > wk_stop:
                        r = int(friday_scores.loc[sym, "rank"])
                        weekly_exits.append((sym, cur_p, f"WEEKLY_RANK_EXIT(rank={r})"))

            for sym, cur_p, reason in weekly_exits:
                pos      = holdings.pop(sym)
                proceeds = pos["shares"] * cur_p * (1 - tc)
                cash    += proceeds
                pnl_inr  = round((cur_p - pos["entry_price"]) * pos["shares"], 2)
                pnl_pct  = round(
                    (cur_p - pos["entry_price"]) / pos["entry_price"] * 100, 2
                )
                sym_rank = (
                    int(friday_scores.loc[sym, "rank"])
                    if not friday_scores.empty and sym in friday_scores.index else 0
                )
                trades.append({
                    "date":            dt.date().isoformat(),
                    "symbol":          sym,
                    "action":          "SELL",
                    "price":           round(cur_p, 2),
                    "entry_price":     round(pos["entry_price"], 2),
                    "shares":          pos["shares"],
                    "value_inr":       round(proceeds, 0),
                    "hold_days":       (dt - pos["entry_date"]).days,
                    "reason":          reason,
                    "momentum_return": 0.0,
                    "rank":            sym_rank,
                    "pnl_inr":         pnl_inr,
                    "pnl_pct":         pnl_pct,
                })

        # ── Monthly rebalance on month-end dates only ─────────────────────────
        if dt in month_end_set:

            b_idx = bench_idx_map.get(dt, bar_idx)

            # Absolute momentum check
            if use_abs:
                risk_on, abs_ret = abs_momentum_at_bar(
                    bench, b_idx, lookback, skip, abs_thr
                )
            else:
                risk_on, abs_ret = True, 0.0

            # Relative momentum ranking (same function as live bot)
            slice_df = prices_df.iloc[: bar_idx + 1]
            ranked   = rank_universe(slice_df, lookback=lookback, skip=skip)
            scores   = _ranked_to_df(ranked)

            # ── EXITS ─────────────────────────────────────────────────────────
            to_sell: list[tuple] = []
            for sym, pos in list(holdings.items()):
                cur_p = day_prices.get(sym)
                if cur_p is None or np.isnan(cur_p):
                    to_sell.append((sym, "DATA_GAP"))
                    continue

                if not risk_on:
                    to_sell.append((sym, "RISK_OFF"))
                    continue

                # Monthly hard stop (catches any breach between weekly scans)
                loss_pct = (pos["entry_price"] - cur_p) / pos["entry_price"]
                if loss_pct >= hard_stop:
                    label = "MONTHLY_HARD_STOP" if wk_stop else "HARD_STOP"
                    to_sell.append((sym, f"{label}({loss_pct:.1%})"))
                    continue

                if sym not in scores.index:
                    to_sell.append((sym, "NOT_RANKED"))
                    continue

                if scores.loc[sym, "rank"] > hold_buffer:
                    r = int(scores.loc[sym, "rank"])
                    to_sell.append((sym, f"RANK_EXIT(rank={r})"))

            for sym, reason in to_sell:
                cur_p = day_prices.get(sym)
                if cur_p is None or np.isnan(cur_p):
                    cur_p = holdings[sym]["entry_price"]   # fallback at zero P&L
                pos      = holdings.pop(sym)
                proceeds = pos["shares"] * cur_p * (1 - tc)
                cash    += proceeds
                pnl_inr  = round((cur_p - pos["entry_price"]) * pos["shares"], 2)
                pnl_pct  = round(
                    (cur_p - pos["entry_price"]) / pos["entry_price"] * 100, 2
                )
                mom_ret  = (
                    scores.loc[sym, "momentum_return"]
                    if sym in scores.index else 0.0
                )
                trades.append({
                    "date":            dt.date().isoformat(),
                    "symbol":          sym,
                    "action":          "SELL",
                    "price":           round(cur_p, 2),
                    "entry_price":     round(pos["entry_price"], 2),
                    "shares":          pos["shares"],
                    "value_inr":       round(proceeds, 0),
                    "hold_days":       (dt - pos["entry_date"]).days,
                    "reason":          reason,
                    "momentum_return": round(mom_ret, 4),
                    "rank":            scores.loc[sym, "rank"] if sym in scores.index else 0,
                    "pnl_inr":         pnl_inr,
                    "pnl_pct":         pnl_pct,
                })

            # ── ENTRIES ───────────────────────────────────────────────────────
            if risk_on and not scores.empty:
                slots        = port_size - len(holdings)
                held_symbols = set(holdings.keys())
                top_ranked   = scores[scores["rank"] <= port_size]
                candidates   = [s for s in top_ranked.index if s not in held_symbols][:slots]

                # Allocation base:
                #   "fixed" → configured capital (matches live bot)
                #   "nav"   → current portfolio NAV (compounding mode)
                alloc_base = capital if allocation_mode == "fixed" else port_value

                for sym in candidates:
                    cur_p = day_prices.get(sym)
                    if cur_p is None or np.isnan(cur_p) or cur_p <= 0:
                        continue
                    cost_ps  = cur_p * (1 + tc)
                    sym_rank = int(scores.loc[sym, "rank"])

                    if use_graded:
                        alloc = graded_allocation(sym_rank, port_size, int(alloc_base))
                    else:
                        alloc = alloc_base / port_size

                    shares = int(alloc / cost_ps)
                    if shares < 1:
                        continue

                    cash -= shares * cost_ps
                    holdings[sym] = {
                        "shares":      shares,
                        "entry_price": cur_p,
                        "entry_date":  dt,
                    }
                    trades.append({
                        "date":            dt.date().isoformat(),
                        "symbol":          sym,
                        "action":          "BUY",
                        "price":           round(cur_p, 2),
                        "entry_price":     round(cur_p, 2),
                        "shares":          shares,
                        "value_inr":       round(shares * cost_ps, 0),
                        "hold_days":       0,
                        "reason":          "MOMENTUM_ENTRY",
                        "momentum_return": round(scores.loc[sym, "momentum_return"], 4),
                        "rank":            sym_rank,
                        "pnl_inr":         0,
                        "pnl_pct":         0,
                    })

        # ── Revalue + record NAV ──────────────────────────────────────────────
        port_value = cash
        for sym, pos in holdings.items():
            p = day_prices.get(sym)
            if p is not None and not np.isnan(p):
                port_value += pos["shares"] * p
        nav_daily[dt] = port_value

    return {
        "nav":    pd.Series(nav_daily),
        "trades": pd.DataFrame(trades) if trades else pd.DataFrame(),
    }


# ── Performance metrics ───────────────────────────────────────────────────────

def calc_metrics(series: pd.Series, name: str, risk_free: float) -> dict:
    if len(series) < 2:
        return {"Strategy": name}
    ret     = series.pct_change().dropna()
    n_years = (series.index[-1] - series.index[0]).days / 365.25
    total   = (series.iloc[-1] / series.iloc[0]) - 1
    cagr    = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
    vol     = ret.std() * np.sqrt(252)
    sharpe  = (cagr - risk_free) / vol if vol > 0 else 0
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
    print(f"  Total BUY  trades    : {len(buys)}")
    print(f"  Total SELL trades    : {n}")
    print(f"  Win rate (sells)     : {wins}/{n} ({wins/n:.0%})")
    print(f"  Total realised P&L   : Rs{total_pnl:+,.0f}")
    print(f"  Avg hold (days)      : {avg_hold:.1f}")
    print("  Exit reason breakdown:")
    for rsn, cnt in reasons.items():
        print(f"    {rsn:<30}: {cnt:>4}  ({cnt/n:.0%})")
    best  = sells.loc[sells["pnl_inr"].idxmax()]
    worst = sells.loc[sells["pnl_inr"].idxmin()]
    print(f"  Best trade  : {best['symbol']} Rs{best['pnl_inr']:+,.0f} "
          f"({best['pnl_pct']:+.1f}%) [{best['reason']}]")
    print(f"  Worst trade : {worst['symbol']} Rs{worst['pnl_inr']:+,.0f} "
          f"({worst['pnl_pct']:+.1f}%) [{worst['reason']}]")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _p(parser: argparse.ArgumentParser, *args, **kwargs):
    """Thin helper: add_argument with positional forwarding."""
    parser.add_argument(*args, **kwargs)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Dual Momentum Delivery — Monthly Backtest.\n"
            "All parameters default to the universe's StrategyParams. "
            "CLI flags override those defaults.\n"
            "Use --allocation-mode nav to enable compounding "
            "(reinvest gains); default 'fixed' matches the live bot."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    d = NIFTY200_DEFAULTS   # readable shorthand for help strings

    p.add_argument("--universe",        choices=list_universes(), default="NIFTY200",
                   help="Universe: NIFTY200 or BEES (default: NIFTY200)")
    p.add_argument("--start",           default="2018-01-01",
                   help="Backtest start date (YYYY-MM-DD, default: 2018-01-01)")
    p.add_argument("--end",             default="2024-12-31",
                   help="Backtest end date (YYYY-MM-DD, default: 2024-12-31)")

    # ── Capital and sizing ─────────────────────────────────────────────────
    p.add_argument("--capital",         default=None, type=float,
                   help=f"Initial capital in Rs (default: universe default "
                        f"— {d.total_capital:,.0f} for NIFTY200, {500_000:,.0f} for BEES)")
    p.add_argument("--portfolio-size",  default=None, type=int,
                   help="Max open positions (default: universe default)")
    p.add_argument("--allocation-mode", default="fixed",
                   choices=["fixed", "nav"],
                   help=(
                       "'fixed' = always size against configured capital — "
                       "matches live bot (default). "
                       "'nav' = size against current portfolio NAV — "
                       "compounding / reinvest-gains mode."
                   ))

    # ── Momentum parameters ────────────────────────────────────────────────
    p.add_argument("--lookback",        default=None, type=int,
                   help=f"Momentum lookback in trading days (default: {d.lookback_days})")
    p.add_argument("--skip",            default=None, type=int,
                   help=f"Skip-month days to avoid reversal (default: {d.skip_days})")
    p.add_argument("--abs-threshold",   default=None, type=float,
                   help=f"Absolute momentum threshold, e.g. -0.05 = -5%% (default: {d.abs_threshold})")
    p.add_argument("--no-abs-filter",   action="store_true",
                   help="Disable absolute momentum filter (pure relative momentum)")

    # ── Portfolio management ───────────────────────────────────────────────
    p.add_argument("--hold-buffer",     default=None, type=int,
                   help="Sell if rank > this (default: universe default)")
    p.add_argument("--hard-stop",       default=None, type=float,
                   help=f"Hard stop fraction, e.g. 0.15 = 15%% (default: {d.hard_stop_pct}). "
                        "Pass 0 to disable.")
    p.add_argument("--weighting",       choices=["graded", "equal"], default=None,
                   help=f"Position sizing: 'graded' (rank-weighted) or 'equal' "
                        f"(default: {d.weighting.lower()})")
    p.add_argument("--no-weekly-stop",  action="store_true",
                   help="Disable weekly Friday stop simulation")
    p.add_argument("--weekly-rank-stop", default=None, type=int,
                   help="Weekly exit if rank > this (default: universe default). "
                        "Pass 0 to disable rank-based weekly exit.")

    # ── Backtest analysis ──────────────────────────────────────────────────
    p.add_argument("--transaction-cost", default=None, type=float,
                   help=f"One-way cost fraction (default: {d.transaction_cost})")
    p.add_argument("--risk-free-rate",  default=None, type=float,
                   help=f"Risk-free rate for Sharpe (default: {d.risk_free_rate})")

    # ── Output naming ──────────────────────────────────────────────────────
    p.add_argument("--account",         default="",
                   help="Account label for output file naming (e.g. abhi2). "
                        "Leave empty for default.")
    p.add_argument("--tranche",         default="",
                   help="Tranche label for output file naming (e.g. T1, T2).")

    return p.parse_args()


def main():
    args = parse_args()
    ucfg = get_universe(args.universe)

    # ── Build StrategyParams: universe defaults + CLI overrides ───────────────
    params = ucfg.params

    overrides: dict = {}
    if args.lookback            is not None: overrides["lookback_days"]    = args.lookback
    if args.skip                is not None: overrides["skip_days"]        = args.skip
    if args.abs_threshold       is not None: overrides["abs_threshold"]    = args.abs_threshold
    if args.no_abs_filter:                   overrides["regime_filter"]    = False
    if args.portfolio_size      is not None: overrides["top_n_hold"]       = args.portfolio_size
    if args.hold_buffer         is not None: overrides["hold_buffer"]      = args.hold_buffer
    if args.hard_stop           is not None:
        overrides["hard_stop_pct"] = args.hard_stop if args.hard_stop > 0 else 0.999
    if args.weighting           is not None: overrides["weighting"]        = args.weighting.upper()
    if args.weekly_rank_stop    is not None:
        overrides["weekly_rank_stop"] = (
            args.weekly_rank_stop if args.weekly_rank_stop > 0 else None
        )
    if args.no_weekly_stop:                  overrides["weekly_rank_stop"] = None
    if args.capital             is not None: overrides["total_capital"]    = int(args.capital)
    if args.transaction_cost    is not None: overrides["transaction_cost"] = args.transaction_cost
    if args.risk_free_rate      is not None: overrides["risk_free_rate"]   = args.risk_free_rate

    if overrides:
        try:
            params = params.replace(**overrides)
        except ValueError as e:
            logger.error(f"Invalid parameter combination: {e}")
            sys.exit(1)

    # If --no-weekly-stop, also ensure weekly_rank_stop is None
    use_weekly = not args.no_weekly_stop

    # ── Output file naming ────────────────────────────────────────────────────
    suffix_parts = []
    if args.account.strip() and args.account.strip().lower() != "default":
        suffix_parts.append(args.account.strip().lower())
    if args.tranche.strip():
        suffix_parts.append(args.tranche.strip().lower())
    file_suffix  = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    base_name    = ucfg.name.lower()
    out_results  = f"backtest_results_{base_name}{file_suffix}.csv"
    out_trades   = f"backtest_trades_{base_name}{file_suffix}.csv"
    out_perf     = f"backtest_performance_{base_name}{file_suffix}.csv"

    # ── Print run configuration ───────────────────────────────────────────────
    sep = "=" * 72
    print(sep)
    print("  DUAL MOMENTUM DELIVERY BOT — MONTHLY BACKTEST")
    print(sep)
    print(f"  Universe         : {ucfg.display_name}  ({len(ucfg.symbols)} symbols)")
    print(f"  Period           : {args.start}  →  {args.end}")
    print(f"  Capital          : Rs{params.total_capital:,.0f}")
    print(f"  Allocation mode  : {args.allocation_mode.upper()}")
    print(f"  Portfolio size   : {params.top_n_hold} slots")
    print(f"  Hold buffer      : sell if rank > {params.hold_buffer}")
    print(f"  Weighting        : {params.weighting}")
    print(f"  Lookback / Skip  : {params.lookback_days}d / {params.skip_days}d")
    print(f"  Abs threshold    : {params.abs_threshold:+.1%}"
          + ("  [DISABLED]" if not params.regime_filter else ""))
    print(f"  Hard stop        : {params.hard_stop_pct:.0%} from entry")
    wk_lbl = (
        f"rank > {params.weekly_rank_stop} OR price > {params.hard_stop_pct:.0%}"
        if (use_weekly and params.weekly_rank_stop)
        else (f"price > {params.hard_stop_pct:.0%} only" if use_weekly else "DISABLED")
    )
    print(f"  Weekly stop      : {wk_lbl}")
    print(f"  Transaction cost : {params.transaction_cost:.2%} one-way")
    print(f"  Risk-free rate   : {params.risk_free_rate:.1%}")
    print(f"  Output files     : {out_results}, {out_trades}, {out_perf}")
    print(sep)

    # ── Validate universe has symbols ─────────────────────────────────────────
    if not ucfg.symbols:
        logger.error(
            f"Universe '{args.universe}' has 0 symbols. "
            f"Check universe_{args.universe.lower()}.txt exists and is not empty."
        )
        sys.exit(1)

    # ── Download price data ───────────────────────────────────────────────────
    stocks, bench = download_prices(
        symbols       = list(ucfg.symbols),
        start         = args.start,
        end           = args.end,
        lookback      = params.lookback_days,
        skip          = params.skip_days,
        regime_ticker = params.regime_ticker,
    )

    if stocks.empty:
        logger.error(
            "Price download returned no data. "
            "Possible causes: (1) Yahoo Finance rate-limited this IP — "
            "wait a few minutes and re-run; "
            "(2) universe_*.txt is empty or all symbols are invalid; "
            "(3) the date range is too short for the lookback window."
        )
        sys.exit(1)

    # When weekly stop is disabled, set weekly_rank_stop to None in a temp params copy
    run_params = params
    if not use_weekly:
        run_params = params.replace(weekly_rank_stop=None)

    # ── Run backtest ──────────────────────────────────────────────────────────
    logger.info("Starting backtest simulation...")
    result = run_backtest(
        prices_df       = stocks,
        bench           = bench,
        start           = args.start,
        end             = args.end,
        params          = run_params,
        allocation_mode = args.allocation_mode,
    )

    nav    = result["nav"]
    trades = result["trades"]

    if nav.empty:
        logger.error("Backtest produced no NAV data. Check start/end dates.")
        sys.exit(1)

    # ── Benchmark NAV ─────────────────────────────────────────────────────────
    bench_nav = None
    if bench is not None:
        b_in_range = bench[(bench.index >= pd.Timestamp(args.start)) &
                           (bench.index <= pd.Timestamp(args.end))]
        if not b_in_range.empty:
            bench_nav = b_in_range / b_in_range.iloc[0] * params.total_capital

    # ── Performance metrics ───────────────────────────────────────────────────
    rf  = params.risk_free_rate
    strat_metrics = calc_metrics(nav, f"Strategy ({ucfg.name})", rf)
    bench_metrics = (
        calc_metrics(bench_nav, f"Benchmark (Nifty 50)", rf)
        if bench_nav is not None else {}
    )

    print("\n")
    print(sep)
    print("  PERFORMANCE SUMMARY")
    print(sep)
    for k, v in strat_metrics.items():
        bench_v = bench_metrics.get(k, "")
        bench_lbl = f"  |  Nifty: {bench_v}" if bench_v and k != "Strategy" else ""
        print(f"  {k:<18}: {v}{bench_lbl}")
    print()

    if not trades.empty:
        print_trade_stats(trades)

    # ── Save CSV outputs ──────────────────────────────────────────────────────
    nav.to_csv(out_results, header=["nav_inr"])
    print(f"\n  NAV saved          → {out_results}  ({len(nav)} rows)")

    if not trades.empty:
        trades.to_csv(out_trades, index=False)
        print(f"  Trades saved       → {out_trades}  ({len(trades)} rows)")

    perf_rows = [strat_metrics]
    if bench_metrics:
        perf_rows.append(bench_metrics)
    pd.DataFrame(perf_rows).to_csv(out_perf, index=False)
    print(f"  Performance saved  → {out_perf}")
    print(sep)


if __name__ == "__main__":
    main()
