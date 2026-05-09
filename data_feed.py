"""
data_feed.py
------------
Fetches daily OHLCV price data from Yahoo Finance for NSE-listed stocks.
Used by the momentum ranker to compute 12M-1M momentum returns.

Key functions:
  fetch_universe_prices()  — batch download closes for the full universe
  get_absolute_momentum()  — check Nifty 50 absolute momentum (risk-on/off)
  get_current_price()      — latest close for a single stock
"""

import logging
import time

import pandas as pd
import yfinance as yf

from config import (
    ABSOLUTE_MOMENTUM_THRESHOLD,
    MIN_HISTORY_BARS,
    MOMENTUM_LOOKBACK_DAYS,
    NIFTY200_UNIVERSE,
    REGIME_TICKER,
    SKIP_RECENT_DAYS,
)

logger = logging.getLogger(__name__)

_BATCH_SIZE = 40   # yfinance handles ~40 tickers per batch efficiently


def _ns(symbol: str) -> str:
    """Return Yahoo Finance ticker string for NSE."""
    return f"{symbol}.NS"


def fetch_universe_prices(symbols: list[str] = None, period: str = "3y") -> pd.DataFrame:
    """
    Batch download close prices for the full universe.
    Returns a DataFrame with dates as index and symbols as columns.
    Drops symbols with >40% missing data. Uses 3y period (need 13+ months of data).
    """
    if symbols is None:
        symbols = NIFTY200_UNIVERSE

    tickers = [_ns(s) for s in symbols]
    logger.info(f"Downloading {len(tickers)} tickers (period={period})...")

    all_dfs = []
    for i in range(0, len(tickers), _BATCH_SIZE):
        batch = tickers[i: i + _BATCH_SIZE]
        for attempt in range(3):
            try:
                raw = yf.download(
                    batch,
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    group_by="ticker",
                )
                if isinstance(raw.columns, pd.MultiIndex):
                    closes = raw.xs("Close", level=1, axis=1)
                else:
                    closes = raw[["Close"]]
                    closes.columns = batch

                closes.columns = [c.replace(".NS", "") for c in closes.columns]
                all_dfs.append(closes)
                break
            except Exception as e:
                logger.warning(f"Batch download error attempt {attempt + 1}: {e}")
                time.sleep(3)

    if not all_dfs:
        logger.error("All batch downloads failed.")
        return pd.DataFrame()

    prices = pd.concat(all_dfs, axis=1)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    # Drop symbols with >40% missing data
    threshold = int(0.60 * len(prices))
    prices = prices.dropna(axis=1, thresh=threshold)
    prices = prices.ffill().bfill()

    logger.info(f"Universe: {len(prices.columns)} stocks with sufficient data retained")
    return prices


def get_absolute_momentum(
    lookback: int = None,
    skip: int = None,
    threshold: float = None,
) -> tuple[bool, float]:
    """
    Dual Momentum absolute momentum check for Nifty 50.

    Computes Nifty 50's 12M-1M return and compares it to a threshold.
      True  = risk-on  (return > threshold) → proceed to relative momentum
      False = risk-off (return <= threshold) → sell all, go to cash

    Args:
        lookback:  Trading days for lookback window (default: MOMENTUM_LOOKBACK_DAYS = 252)
        skip:      Trading days to skip at end (default: SKIP_RECENT_DAYS = 21)
        threshold: Minimum return required (default: ABSOLUTE_MOMENTUM_THRESHOLD = 0.06)

    Returns:
        (is_risk_on: bool, actual_return: float)
    """
    if lookback is None:
        lookback = MOMENTUM_LOOKBACK_DAYS
    if skip is None:
        skip = SKIP_RECENT_DAYS
    if threshold is None:
        threshold = ABSOLUTE_MOMENTUM_THRESHOLD

    required = lookback + skip + 10

    try:
        df = yf.Ticker(REGIME_TICKER).history(
            period="3y", interval="1d", auto_adjust=True
        )
        df.index = pd.to_datetime(df.index).tz_localize(None)

        if len(df) < required:
            logger.warning(
                f"Insufficient Nifty data ({len(df)} bars, need {required}) "
                f"— defaulting to RISK-ON."
            )
            return True, 0.0

        price_end   = float(df["Close"].iloc[-(skip + 1)])           # ~1 month ago
        price_start = float(df["Close"].iloc[-(lookback + skip + 1)]) # ~13 months ago
        abs_return  = (price_end - price_start) / price_start

        is_risk_on = abs_return > threshold
        logger.info(
            f"Absolute momentum: Nifty 12M-1M = {abs_return:+.1%}  "
            f"threshold = {threshold:.0%}  "
            f"-> {'RISK-ON  (proceed to buy)' if is_risk_on else 'RISK-OFF (go to cash)'}"
        )
        return is_risk_on, abs_return

    except Exception as e:
        logger.warning(f"Absolute momentum check error: {e} — defaulting to RISK-ON.")
        return True, 0.0


def get_current_price(symbol: str) -> float | None:
    """Fetch the latest available close price for a single symbol."""
    try:
        df = yf.Ticker(_ns(symbol)).history(period="5d", interval="1d", auto_adjust=True)
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception as e:
        logger.warning(f"{symbol}: price fetch error — {e}")
        return None
