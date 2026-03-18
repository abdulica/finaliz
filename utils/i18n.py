"""Internationalization support for Turkish and English."""

TRANSLATIONS = {
    # Navigation
    "nav_dashboard": {"tr": "Genel Bakış", "en": "Dashboard"},
    "nav_asset_detail": {"tr": "Varlık Detay", "en": "Asset Detail"},
    "nav_forecast": {"tr": "Tahmin", "en": "Forecast"},
    "nav_comparison": {"tr": "Karşılaştırma", "en": "Comparison"},
    "nav_macro": {"tr": "Makro Göstergeler", "en": "Macro Indicators"},
    # Sidebar
    "sidebar_title": {"tr": "Finaliz", "en": "Finaliz"},
    "sidebar_language": {"tr": "Dil / Language", "en": "Language / Dil"},
    "sidebar_refresh": {"tr": "🔄 Verileri Güncelle", "en": "🔄 Refresh Data"},
    "sidebar_last_update": {"tr": "Son güncelleme", "en": "Last update"},
    "sidebar_period": {"tr": "Veri Periyodu", "en": "Data Period"},
    "sidebar_asset": {"tr": "Varlık Seçin", "en": "Select Asset"},
    "sidebar_assets": {"tr": "Varlık Seçin (Çoklu)", "en": "Select Assets (Multiple)"},
    # Dashboard
    "dash_title": {"tr": "Piyasa Genel Bakış", "en": "Market Overview"},
    "dash_price": {"tr": "Fiyat", "en": "Price"},
    "dash_change": {"tr": "Değişim", "en": "Change"},
    "dash_change_pct": {"tr": "Değişim %", "en": "Change %"},
    "dash_volume": {"tr": "Hacim", "en": "Volume"},
    "dash_high": {"tr": "Yüksek", "en": "High"},
    "dash_low": {"tr": "Düşük", "en": "Low"},
    "dash_sentiment": {"tr": "Piyasa Duyarlılığı", "en": "Market Sentiment"},
    # Technical Analysis
    "ta_title": {"tr": "Teknik Analiz", "en": "Technical Analysis"},
    "ta_rsi": {"tr": "RSI (Göreceli Güç Endeksi)", "en": "RSI (Relative Strength Index)"},
    "ta_macd": {"tr": "MACD", "en": "MACD"},
    "ta_bollinger": {"tr": "Bollinger Bantları", "en": "Bollinger Bands"},
    "ta_ma": {"tr": "Hareketli Ortalamalar", "en": "Moving Averages"},
    "ta_support_resistance": {"tr": "Destek / Direnç", "en": "Support / Resistance"},
    "ta_signal_buy": {"tr": "ALIM sinyali", "en": "BUY signal"},
    "ta_signal_sell": {"tr": "SATIM sinyali", "en": "SELL signal"},
    "ta_signal_neutral": {"tr": "NÖTR", "en": "NEUTRAL"},
    "ta_overbought": {"tr": "Aşırı alım bölgesi", "en": "Overbought zone"},
    "ta_oversold": {"tr": "Aşırı satım bölgesi", "en": "Oversold zone"},
    "ta_summary": {"tr": "Teknik Özet", "en": "Technical Summary"},
    # Forecast
    "fc_title": {"tr": "Tahmin", "en": "Forecast"},
    "fc_horizon": {"tr": "Tahmin Periyodu", "en": "Forecast Horizon"},
    "fc_predicted": {"tr": "Tahmini Fiyat", "en": "Predicted Price"},
    "fc_confidence": {"tr": "Güven Aralığı", "en": "Confidence Interval"},
    "fc_upper": {"tr": "Üst Sınır", "en": "Upper Bound"},
    "fc_lower": {"tr": "Alt Sınır", "en": "Lower Bound"},
    "fc_direction": {"tr": "Yön", "en": "Direction"},
    "fc_up": {"tr": "Yükseliş", "en": "Bullish"},
    "fc_down": {"tr": "Düşüş", "en": "Bearish"},
    "fc_sideways": {"tr": "Yatay", "en": "Sideways"},
    # Macro
    "macro_title": {"tr": "Makroekonomik Göstergeler", "en": "Macroeconomic Indicators"},
    "macro_correlation": {"tr": "Korelasyon Matrisi", "en": "Correlation Matrix"},
    "macro_impact": {"tr": "Makro Etki Analizi", "en": "Macro Impact Analysis"},
    # Comparison
    "comp_title": {"tr": "Varlık Karşılaştırma", "en": "Asset Comparison"},
    "comp_normalized": {"tr": "Normalize Edilmiş Performans", "en": "Normalized Performance"},
    "comp_correlation": {"tr": "Korelasyon", "en": "Correlation"},
    # Analysis commentary
    "analysis_interesting": {"tr": "İlginç bir durum var", "en": "Here's an interesting situation"},
    "analysis_note": {"tr": "Dikkat çekici bir nokta", "en": "A notable point"},
    "analysis_historical": {"tr": "Geçmişte benzer dönemlerde", "en": "In similar historical periods"},
    "analysis_contradiction": {"tr": "Çelişkili sinyaller görünüyor", "en": "Contradictory signals detected"},
    "analysis_consider": {"tr": "Şunu da göz önünde bulundurmak lazım", "en": "It's also worth considering"},
    # Seasonal
    "seasonal_title": {"tr": "Mevsimsel Analiz", "en": "Seasonal Analysis"},
    "seasonal_pattern": {"tr": "Mevsimsel Patern", "en": "Seasonal Pattern"},
    "seasonal_avg": {"tr": "Tarihsel Ortalama", "en": "Historical Average"},
    # General
    "loading": {"tr": "Veriler yükleniyor...", "en": "Loading data..."},
    "error_data": {"tr": "Veri çekilemedi", "en": "Failed to fetch data"},
    "no_data": {"tr": "Veri bulunamadı", "en": "No data available"},
}


def t(key: str, lang: str = "tr") -> str:
    """Get translation for a key."""
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    return entry.get(lang, entry.get("en", key))


def get_asset_name(asset_key: str, asset_config: dict, lang: str = "tr") -> str:
    """Get localized asset name."""
    return asset_config.get(f"name_{lang}", asset_config.get("name_en", asset_key))
