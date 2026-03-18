"""Data fetching module - yfinance for market data, FRED for macro data."""

import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from config import ASSETS, MACRO_SERIES

load_dotenv()


def fetch_asset_data(asset_key: str, period: str = "1y") -> pd.DataFrame | None:
    """Fetch OHLCV data for a single asset.

    Args:
        asset_key: Key from ASSETS config (e.g., 'GOLD', 'BTC')
        period: yfinance period string ('6mo', '1y', '2y', '5y')

    Returns:
        DataFrame with OHLCV data, or None on failure.
    """
    asset = ASSETS.get(asset_key)
    if asset is None:
        return None

    if asset["source"] == "fred":
        return _fetch_fred_series(asset.get("fred_series"), period)

    try:
        ticker = yf.Ticker(asset["ticker"])
        df = ticker.history(period=period)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return None


def fetch_all_assets(period: str = "1y") -> dict[str, pd.DataFrame]:
    """Fetch data for all configured assets."""
    results = {}
    for key in ASSETS:
        df = fetch_asset_data(key, period)
        if df is not None and not df.empty:
            results[key] = df
    return results


def fetch_macro_data(period_years: int = 5) -> dict[str, pd.DataFrame]:
    """Fetch macroeconomic data from FRED.

    Returns dict of series_key -> DataFrame.
    Falls back gracefully if FRED API key is not set.
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        return _fetch_macro_fallback()

    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
    except Exception:
        return _fetch_macro_fallback()

    end = datetime.now()
    start = end - timedelta(days=period_years * 365)
    results = {}

    for key, info in MACRO_SERIES.items():
        try:
            series = fred.get_series(info["series_id"], start, end)
            if series is not None and len(series) > 0:
                df = series.to_frame(name="value")
                df.index = pd.to_datetime(df.index)
                df = df.dropna()
                results[key] = df
        except Exception:
            continue

    return results


def _fetch_macro_fallback() -> dict[str, pd.DataFrame]:
    """Fallback: fetch VIX and treasury yields from yfinance when FRED key is missing."""
    fallback_tickers = {
        "VIX": "^VIX",
        "US10Y": "^TNX",
    }
    results = {}
    for key, ticker in fallback_tickers.items():
        try:
            df = yf.Ticker(ticker).history(period="2y")
            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                results[key] = df[["Close"]].rename(columns={"Close": "value"})
        except Exception:
            continue
    return results


def _fetch_fred_series(series_id: str | None, period: str = "1y") -> pd.DataFrame | None:
    """Fetch a single FRED series and format as OHLCV-like DataFrame."""
    if not series_id:
        return None

    api_key = os.getenv("FRED_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        return None

    period_map = {"6mo": 0.5, "1y": 1, "2y": 2, "5y": 5}
    years = period_map.get(period, 1)

    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        end = datetime.now()
        start = end - timedelta(days=int(years * 365))
        series = fred.get_series(series_id, start, end)
        if series is None or len(series) == 0:
            return None

        df = series.to_frame(name="Close")
        df.index = pd.to_datetime(df.index)
        df = df.dropna()
        # Create OHLCV-like structure for compatibility
        df["Open"] = df["Close"]
        df["High"] = df["Close"]
        df["Low"] = df["Close"]
        df["Volume"] = 0
        return df
    except Exception:
        return None


def get_latest_prices(data: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Extract latest price info for each asset."""
    results = {}
    for key, df in data.items():
        if df.empty:
            continue
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        close = latest["Close"]
        prev_close = prev["Close"]
        change = close - prev_close
        change_pct = (change / prev_close * 100) if prev_close != 0 else 0

        results[key] = {
            "close": close,
            "open": latest.get("Open", close),
            "high": latest.get("High", close),
            "low": latest.get("Low", close),
            "volume": latest.get("Volume", 0),
            "change": change,
            "change_pct": change_pct,
            "date": df.index[-1],
        }
    return results
