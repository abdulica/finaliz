"""Finaliz - Configuration and asset definitions."""

# Asset definitions: all priced in USD
ASSETS = {
    "DXY": {
        "ticker": "DX=F",
        "name_tr": "Dolar Endeksi (DXY)",
        "name_en": "US Dollar Index (DXY)",
        "category": "currency",
        "color": "#1f77b4",
        "source": "yfinance",
    },
    "USDTRY": {
        "ticker": "USDTRY=X",
        "name_tr": "Dolar/TL",
        "name_en": "USD/TRY",
        "category": "currency",
        "color": "#e63946",
        "source": "yfinance",
    },
    "EURUSD": {
        "ticker": "EURUSD=X",
        "name_tr": "Euro/Dolar",
        "name_en": "EUR/USD",
        "category": "currency",
        "color": "#ff7f0e",
        "source": "yfinance",
    },
    "GOLD": {
        "ticker": "GC=F",
        "name_tr": "Altın (USD/oz)",
        "name_en": "Gold (USD/oz)",
        "category": "commodity",
        "color": "#FFD700",
        "source": "yfinance",
    },
    "SILVER": {
        "ticker": "SI=F",
        "name_tr": "Gümüş (USD/oz)",
        "name_en": "Silver (USD/oz)",
        "category": "commodity",
        "color": "#C0C0C0",
        "source": "yfinance",
    },
    "BTC": {
        "ticker": "BTC-USD",
        "name_tr": "Bitcoin (USD)",
        "name_en": "Bitcoin (USD)",
        "category": "crypto",
        "color": "#F7931A",
        "source": "yfinance",
    },
    "OIL": {
        "ticker": "CL=F",
        "name_tr": "Ham Petrol (USD/varil)",
        "name_en": "Crude Oil (USD/barrel)",
        "category": "commodity",
        "color": "#2ca02c",
        "source": "yfinance",
    },
    "IRON": {
        "ticker": None,
        "fred_series": "PIORECRUSDM",
        "name_tr": "Demir Cevheri (USD/ton)",
        "name_en": "Iron Ore (USD/ton)",
        "category": "commodity",
        "color": "#8B4513",
        "source": "fred",
    },
}

# FRED macro data series
MACRO_SERIES = {
    "FED_RATE": {
        "series_id": "FEDFUNDS",
        "name_tr": "FED Faiz Oranı",
        "name_en": "Federal Funds Rate",
    },
    "CPI": {
        "series_id": "CPIAUCSL",
        "name_tr": "Tüketici Fiyat Endeksi (CPI)",
        "name_en": "Consumer Price Index (CPI)",
    },
    "UNEMPLOYMENT": {
        "series_id": "UNRATE",
        "name_tr": "İşsizlik Oranı",
        "name_en": "Unemployment Rate",
    },
    "M2": {
        "series_id": "M2SL",
        "name_tr": "M2 Para Arzı",
        "name_en": "M2 Money Supply",
    },
    "US10Y": {
        "series_id": "DGS10",
        "name_tr": "ABD 10 Yıllık Tahvil Faizi",
        "name_en": "US 10-Year Treasury Yield",
    },
    "VIX": {
        "series_id": "VIXCLS",
        "name_tr": "Korku Endeksi (VIX)",
        "name_en": "Volatility Index (VIX)",
    },
}

# Technical analysis parameters
TA_PARAMS = {
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2,
    "sma_periods": [20, 50, 200],
    "ema_periods": [12, 26],
    "atr_period": 14,
    "stoch_rsi_period": 14,
}

# Forecast horizons in days
FORECAST_HORIZONS = {
    "1d": {"days": 1, "label_tr": "1 Günlük", "label_en": "1 Day"},
    "3d": {"days": 3, "label_tr": "3 Günlük", "label_en": "3 Day"},
    "1w": {"days": 7, "label_tr": "Haftalık", "label_en": "Weekly"},
    "2w": {"days": 14, "label_tr": "2 Haftalık", "label_en": "2 Week"},
    "4w": {"days": 28, "label_tr": "4 Haftalık", "label_en": "4 Week"},
}

# Data period options
DATA_PERIODS = {
    "6mo": {"label_tr": "6 Ay", "label_en": "6 Months"},
    "1y": {"label_tr": "1 Yıl", "label_en": "1 Year"},
    "2y": {"label_tr": "2 Yıl", "label_en": "2 Years"},
    "5y": {"label_tr": "5 Yıl", "label_en": "5 Years"},
}

# Disclaimer
DISCLAIMER_TR = (
    "⚠️ Bu platform yalnızca bilgilendirme ve eğitim amaçlıdır. "
    "Burada sunulan analizler, tahminler ve yorumlar kesinlikle yatırım tavsiyesi değildir. "
    "Finansal kararlarınızı almadan önce mutlaka lisanslı bir finansal danışmana başvurunuz."
)

DISCLAIMER_EN = (
    "⚠️ This platform is for informational and educational purposes only. "
    "The analyses, forecasts, and commentary presented here do not constitute investment advice. "
    "Please consult a licensed financial advisor before making any financial decisions."
)
