"""Smart data windowing - each analysis module gets the optimal time window.

Strategy:
- Technical analysis: 1 year (indicators use 14-200 day windows)
- Forecast (Prophet): 2 years (enough pattern, avoids old regime noise)
- Seasonal analysis: 5 years full data, but with recency weighting
- Macro correlation: 1 year (correlations shift with regime changes)
- Dashboard / charts: 1 year for display, full data available
"""

from datetime import datetime, timedelta

import pandas as pd


def window_for_technical(df: pd.DataFrame) -> pd.DataFrame:
    """Return last ~1 year of data for technical analysis.
    SMA 200 needs ~200 trading days, so 1 year (252 trading days) is ideal.
    """
    return _trim_days(df, days=365)


def window_for_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Return last ~2 years for Prophet forecast.
    2 years captures recent regime while providing enough seasonal data.
    Old regimes (pre-inflation, pre-COVID) are excluded.
    """
    return _trim_days(df, days=730)


def window_for_seasonal(df: pd.DataFrame) -> pd.DataFrame:
    """Return full data for seasonal analysis (up to 5 years).
    The seasonal module itself applies recency weighting.
    """
    return _trim_days(df, days=1825)


def window_for_macro_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Return last ~1 year for macro correlation.
    Correlations are regime-dependent; recent data is most relevant.
    """
    return _trim_days(df, days=365)


def window_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return last ~1 year for chart display."""
    return _trim_days(df, days=365)


def _trim_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Trim DataFrame to last N calendar days."""
    if df is None or df.empty:
        return df
    cutoff = df.index[-1] - timedelta(days=days)
    trimmed = df[df.index >= cutoff]
    # Ensure we have enough data, fall back to full if too little
    if len(trimmed) < 30:
        return df
    return trimmed
