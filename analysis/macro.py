"""Macroeconomic analysis and commentary generation."""

import numpy as np
import pandas as pd

from config import MACRO_SERIES


def analyze_macro_environment(macro_data: dict[str, pd.DataFrame], lang: str = "tr") -> list[str]:
    """Analyze current macro environment and generate commentary.

    Returns list of commentary paragraphs.
    """
    comments = []

    # FED Rate analysis
    fed = macro_data.get("FED_RATE")
    if fed is not None and not fed.empty:
        current = fed["value"].iloc[-1]
        prev_6m = fed["value"].iloc[-min(6, len(fed))]
        trend = "rising" if current > prev_6m else "falling" if current < prev_6m else "stable"

        if lang == "tr":
            trend_tr = {"rising": "yükseliyor", "falling": "düşüyor", "stable": "sabit"}[trend]
            comments.append(
                f"🏦 FED faiz oranı şu an %{current:.2f} seviyesinde ve {trend_tr}. "
                + _fed_impact_commentary(trend, lang)
            )
        else:
            comments.append(
                f"🏦 Fed funds rate is currently at {current:.2f}% and {trend}. "
                + _fed_impact_commentary(trend, lang)
            )

    # CPI / Inflation
    cpi = macro_data.get("CPI")
    if cpi is not None and len(cpi) > 12:
        current_cpi = cpi["value"].iloc[-1]
        year_ago_cpi = cpi["value"].iloc[-12]
        yoy_inflation = (current_cpi - year_ago_cpi) / year_ago_cpi * 100

        if lang == "tr":
            comments.append(
                f"📊 Yıllık enflasyon (CPI bazlı) %{yoy_inflation:.1f} seviyesinde. "
                + _inflation_impact(yoy_inflation, lang)
            )
        else:
            comments.append(
                f"📊 Year-over-year inflation (CPI-based) is at {yoy_inflation:.1f}%. "
                + _inflation_impact(yoy_inflation, lang)
            )

    # VIX / Fear index
    vix = macro_data.get("VIX")
    if vix is not None and not vix.empty:
        current_vix = vix["value"].iloc[-1]
        if lang == "tr":
            comments.append(
                f"😰 VIX (Korku Endeksi) {current_vix:.1f} seviyesinde. "
                + _vix_commentary(current_vix, lang)
            )
        else:
            comments.append(
                f"😰 VIX (Fear Index) is at {current_vix:.1f}. "
                + _vix_commentary(current_vix, lang)
            )

    # US 10Y
    us10y = macro_data.get("US10Y")
    if us10y is not None and not us10y.empty:
        current_yield = us10y["value"].iloc[-1]
        if lang == "tr":
            comments.append(
                f"📈 ABD 10 yıllık tahvil faizi %{current_yield:.2f} seviyesinde. "
                + _yield_commentary(current_yield, lang)
            )
        else:
            comments.append(
                f"📈 US 10-year Treasury yield is at {current_yield:.2f}%. "
                + _yield_commentary(current_yield, lang)
            )

    # M2 Money Supply
    m2 = macro_data.get("M2")
    if m2 is not None and len(m2) > 12:
        current_m2 = m2["value"].iloc[-1]
        year_ago_m2 = m2["value"].iloc[-12]
        m2_growth = (current_m2 - year_ago_m2) / year_ago_m2 * 100

        if lang == "tr":
            comments.append(
                f"💰 M2 para arzı yıllık %{m2_growth:.1f} değişim gösteriyor. "
                + _m2_commentary(m2_growth, lang)
            )
        else:
            comments.append(
                f"💰 M2 money supply has changed {m2_growth:.1f}% year-over-year. "
                + _m2_commentary(m2_growth, lang)
            )

    return comments


def compute_macro_correlations(
    asset_data: dict[str, pd.DataFrame],
    macro_data: dict[str, pd.DataFrame],
) -> pd.DataFrame | None:
    """Compute correlation matrix between assets and macro indicators."""
    series_dict = {}

    for key, df in asset_data.items():
        if df is not None and not df.empty:
            monthly = df["Close"].resample("M").last()
            pct = monthly.pct_change().dropna()
            if len(pct) > 3:
                series_dict[key] = pct

    for key, df in macro_data.items():
        if df is not None and not df.empty:
            monthly = df["value"].resample("M").last()
            pct = monthly.pct_change().dropna()
            if len(pct) > 3:
                label = MACRO_SERIES.get(key, {}).get("name_en", key)
                series_dict[label] = pct

    if len(series_dict) < 2:
        return None

    combined = pd.DataFrame(series_dict).dropna()
    if len(combined) < 3:
        return None

    return combined.corr()


def _fed_impact_commentary(trend: str, lang: str) -> str:
    if lang == "tr":
        if trend == "rising":
            return (
                "Yükselen faizler genelde doları güçlendirir, altın ve hisse senetleri üzerinde "
                "baskı oluşturur. Ancak piyasa beklentileri zaten fiyatlanmış olabilir."
            )
        elif trend == "falling":
            return (
                "Düşen faizler genelde risk iştahını artırır, altın ve BTC gibi varlıkları "
                "destekler, doları zayıflatır."
            )
        return "Sabit faiz ortamı piyasaları bekle-gör moduna sokabilir."
    else:
        if trend == "rising":
            return (
                "Rising rates typically strengthen the dollar and put pressure on gold and equities. "
                "However, market expectations may already be priced in."
            )
        elif trend == "falling":
            return (
                "Falling rates tend to increase risk appetite, supporting assets like gold and BTC "
                "while weakening the dollar."
            )
        return "A stable rate environment may keep markets in wait-and-see mode."


def _inflation_impact(inflation: float, lang: str) -> str:
    if lang == "tr":
        if inflation > 5:
            return (
                "Yüksek enflasyon, altın gibi enflasyon korunma araçlarını desteklerken, "
                "FED'in daha şahin bir tutum sergilemesine neden olabilir."
            )
        elif inflation > 3:
            return "Orta seviye enflasyon devam ediyor. FED'in politikası belirleyici olacak."
        return "Düşük enflasyon, gevşek para politikası olasılığını artırıyor."
    else:
        if inflation > 5:
            return (
                "High inflation supports inflation hedges like gold, "
                "but may push the Fed toward a more hawkish stance."
            )
        elif inflation > 3:
            return "Moderate inflation persists. Fed policy will be the key driver."
        return "Low inflation increases the probability of accommodative monetary policy."


def _vix_commentary(vix: float, lang: str) -> str:
    if lang == "tr":
        if vix > 30:
            return "Piyasada ciddi bir korku hâkim. Bu seviyeler genelde dip yakınlarına denk gelir ama daha da kötüleşebilir."
        elif vix > 20:
            return "Ortalama üstü volatilite var. Piyasa tedirgin ama panik seviyesinde değil."
        return "Düşük volatilite, piyasanın sakin olduğunu gösteriyor. Ama bu sakinlik yanıltıcı olabilir."
    else:
        if vix > 30:
            return "Significant fear in the market. These levels often coincide with bottoms, but things could get worse."
        elif vix > 20:
            return "Above-average volatility. Market is nervous but not panicking."
        return "Low volatility suggests calm markets. But this calm can be deceptive."


def _yield_commentary(yield_val: float, lang: str) -> str:
    if lang == "tr":
        if yield_val > 4.5:
            return (
                "Yüksek tahvil faizi, altın ve hisse senetleri için rakip oluşturuyor. "
                "Yatırımcılar güvenli getiri için tahvillere yönelebilir."
            )
        elif yield_val > 3:
            return "Orta seviye tahvil faizi, piyasalar için nötr bir ortam oluşturuyor."
        return "Düşük tahvil faizi, risk varlıklarını ve altını destekleyici."
    else:
        if yield_val > 4.5:
            return (
                "High bond yields compete with gold and equities. "
                "Investors may shift to bonds for safe returns."
            )
        elif yield_val > 3:
            return "Moderate bond yields create a neutral environment for markets."
        return "Low bond yields are supportive of risk assets and gold."


def _m2_commentary(growth: float, lang: str) -> str:
    if lang == "tr":
        if growth > 5:
            return (
                "Para arzı artışı likiditeyi genişletiyor. Bu genelde varlık fiyatlarını "
                "yukarı iter, ama enflasyonu da besler."
            )
        elif growth < 0:
            return (
                "Para arzı daralmakta. Bu sıkılaştırıcı bir sinyal ve varlık fiyatları "
                "üzerinde baskı oluşturabilir."
            )
        return "Para arzı ılımlı büyüyor. Piyasalar için nötr bir ortam."
    else:
        if growth > 5:
            return (
                "Money supply growth is expanding liquidity. This typically pushes asset prices "
                "higher but also fuels inflation."
            )
        elif growth < 0:
            return (
                "Money supply is contracting. This is a tightening signal and may put "
                "pressure on asset prices."
            )
        return "Money supply is growing moderately. A neutral environment for markets."
