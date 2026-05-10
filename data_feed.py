"""
data_feed.py
------------
Fetches daily OHLCV price data from Yahoo Finance for NSE-listed stocks.

Key functions:
  fetch_universe_prices()  — batch download closes for the full universe
  get_absolute_momentum()  — Nifty 50 absolute momentum (risk-on / risk-off)
  get_current_price()      — latest close for a single symbol

Design notes:
  - Forward-fill (ffill) is applied to handle exchange holidays and
    short trading halts. Backward-fill (bfill) is intentionally NOT applied
    because it would propagate future prices backward into the start of a
    stock's history — a subtle look-ahead bias.
  - Stocks with >40% missing data over the download period are dropped.
    This threshold is the same in both live feed and backtest download.
  - get_current_price() warns if the most recent close is more than
    MAX_PRICE_STALE_DAYS old (e.g., illiquid BEES ETFs on a thin week).
"""

import logging
import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from strategy import NIFTY200_DEFAULTS as _defaults

logger = logging.getLogger(__name__)

_BATCH_SIZE          = 40    # yfinance handles ~40 tickers per batch efficiently
MAX_PRICE_STALE_DAYS = 4     # warn if single-stock price is older than this many days


def _ns(symbol: str) -> str:
    """Return Yahoo Finance ticker string for NSE (adds .NS suffix)."""
    return f"{symbol}.NS"


def fetch_universe_prices(
    symbols: list[str],
    period:  str = "3y",
) -> pd.DataFrame:
    """Batch download adjusted close prices for the full universe.

    Args:
        symbols: List of NSE symbols (no .NS suffix needed).
        period:  yfinance period string (default "3y" — needed for 12M+1M window).

    Returns:
        DataFrame: dates as index, symbols as columns.
        Symbols with >40% missing rows are dropped.
        Remaining gaps are forward-filled (no backward-fill).
        Returns empty DataFrame on total failure.
    """
    if not symbols:
        logger.error(
            "fetch_universe_prices() called with empty symbol list. "
            "Check that your universe_*.txt file exists and is not empty."
        )
        return pd.DataFrame()

    tickers = [_ns(s) for s in symbols]
    logger.info(f"Downloading {len(tickers)} tickers (period={period})...")

    all_dfs = []
    for i in range(0, len(tickers), _BATCH_SIZE):
        batch = tickers[i : i + _BATCH_SIZE]
        for attempt in range(3):
            try:
                raw = yf.download(
                    batch,
                    period   = period,
                    interval = "1d",
                    auto_adjust = True,
                    progress = False,
                    group_by = "ticker",
                )
                if isinstance(raw.columns, pd.MultiIndex):
                    closes = raw.xs("Close", level=1, axis=1)
                else:
                    # Single-ticker download returns flat columns
                    closes = raw[["Close"]].copy()
                    closes.columns = batch

                closes.columns = [c.replace(".NS", "") for c in closes.columns]
                all_dfs.append(closes)
                break
            except Exception as e:
                logger.warning(f"Batch download attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(3)

    if not all_dfs:
        logger.error("All batch downloads failed — cannot proceed.")
        return pd.DataFrame()

    prices = pd.concat(all_dfs, axis=1)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    # Drop symbols with >40% missing data over the download window.
    # Threshold is applied against the number of rows in the downloaded frame —
    # the same base as the backtest's download_prices() function.
    threshold = int(0.60 * len(prices))
    prices = prices.dropna(axis=1, thresh=threshold)

    # Forward-fill only — propagates last known price across holidays / halts.
    # NOT backward-filled: bfill would inject look-ahead prices at the start
    # of a newly-listed stock's history.
    prices = prices.ffill()

    logger.info(f"Universe: {len(prices.columns)} symbols retained after quality filter")
    return prices


def get_absolute_momentum(
    lookback:  int   = _defaults.lookback_days,
    skip:      int   = _defaults.skip_days,
    threshold: float = _defaults.abs_threshold,
    ticker:    str   = _defaults.regime_ticker,
) -> tuple[bool, float]:
    """Absolute momentum check for the market regime benchmark (Nifty 50).

    Computes the benchmark's (lookback − skip) return and compares it to
    the threshold.

      True  = risk-on  (return > threshold) → proceed to relative momentum
      False = risk-off (return ≤ threshold) → sell all, go to cash

    Args:
        lookback:  Momentum window in trading days (default: 252).
        skip:      Days to skip at the right end (default: 21).
        threshold: Risk-off trigger (default: -0.05 = -5%).
                   Only goes to cash in a genuine bear market.
        ticker:    Yahoo Finance ticker for the benchmark (default: "^NSEI").

    Returns:
        (is_risk_on: bool, actual_return: float)
        On download failure defaults to (True, 0.0) — fail-safe / risk-on.
    """
    required = lookback + skip + 10

    try:
        df = yf.Ticker(ticker).history(period="3y", interval="1d", auto_adjust=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)

        if len(df) < required:
            logger.warning(
                f"Insufficient benchmark data ({len(df)} bars, need {required}) "
                f"— defaulting to RISK-ON."
            )
            return True, 0.0

        price_end   = float(df["Close"].iloc[-(skip + 1)])            # ~1 month ago
        price_start = float(df["Close"].iloc[-(lookback + skip + 1)]) # ~13 months ago
        abs_return  = (price_end - price_start) / price_start

        is_risk_on = abs_return > threshold
        logger.info(
            f"Absolute momentum ({ticker}): {lookback}-{skip}d return = "
            f"{abs_return:+.1%}  |  threshold = {threshold:+.1%}  "
            f"→  {'RISK-ON' if is_risk_on else 'RISK-OFF'}"
        )
        return is_risk_on, abs_return

    except Exception as e:
        logger.warning(f"Absolute momentum check failed ({e}) — defaulting to RISK-ON.")
        return True, 0.0


def get_current_price(symbol: str) -> float | None:
    """Fetch the latest available close price for a single NSE symbol.

    Uses a 5-day window so we get at least one trading day even after
    a long weekend. Warns if the most recent close is older than
    MAX_PRICE_STALE_DAYS days (e.g., suspended or illiquid ETF).

    Returns None on any fetch error.
    """
    try:
        df = yf.Ticker(_ns(symbol)).history(
            period="5d", interval="1d", auto_adjust=True
        )
        if df.empty:
            logger.warning(f"{symbol}: empty price history returned.")
            return None

        last_ts = df.index[-1]
        last_date = last_ts.date() if hasattr(last_ts, "date") else last_ts

        stale_days = (date.today() - last_date).days
        if stale_days > MAX_PRICE_STALE_DAYS:
            logger.warning(
                f"{symbol}: last close is {stale_days} days old "
                f"({last_date}) — may be illiquid or suspended."
            )

        return float(df["Close"].iloc[-1])

    except Exception as e:
        logger.warning(f"{symbol}: price fetch error — {e}")
        return None
