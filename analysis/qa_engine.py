"""Question-Answer engine for financial analysis.

Detects whether user input is a question or context, and generates
data-driven answers using technical indicators and factor×asset matrix.
"""

import re
import numpy as np
import pandas as pd


# --- Question detection ---

_TR_QUESTION_PATTERNS = [
    r"\?$",
    r"^(ne olur|ne olacak|ne beklenir)",
    r"^(neden|niye|niçin|nicin)",
    r"^(nasıl|nasil)",
    r"^(hangi|kaç|kac)",
    r"(ne olur|ne yapar|ne etkisi|etkisi ne|etkiler mi|etkisi var mı)",
    r"(nereye gider|nereye kadar|ne kadar)",
    r"(yükselir mi|düşer mi|yukselir mi|duser mi|artar mı|azalır mı)",
    r"(mantıklı mı|mantikli mi|doğru mu|dogru mu|risk.* mi)",
    r"(almalı mıyım|alınır mı|satmalı mıyım|satılır mı)",
    r"(ne zaman|nezaman)",
    r"(olur mu|olabilir mi|mümkün mü|mumkun mu)",
    r"(sence|sizce|tahmin)",
    r"(beklenti|öngörü|ongoru)",
    r"(açılırsa|kapanırsa|acilarsa|kapanarsa|artarsa|düşerse|duserse|yükselirse|yukselirse)",
    r"(olursa ne|gelirse ne|çıkarsa ne|cikarsa ne)",
]

_EN_QUESTION_PATTERNS = [
    r"\?$",
    r"^(what|why|how|when|where|which|will|would|could|should|can|is|are|do|does)",
    r"(what happens|what if|what would)",
    r"(go up|go down|rise|fall|drop|increase|decrease|crash|rally|spike)",
    r"(should i|is it safe|is it risky|makes sense)",
    r"(expect|forecast|predict|outlook)",
    r"(impact|affect|effect|influence)",
]


def is_question(text: str) -> bool:
    """Detect if the user input is a question rather than context."""
    text_lower = text.strip().lower()
    if not text_lower:
        return False

    all_patterns = _TR_QUESTION_PATTERNS + _EN_QUESTION_PATTERNS
    for pattern in all_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


# --- Question classification ---

_QUESTION_TYPES = {
    "what_if": {
        "tr": [r"(olursa|açılırsa|kapanırsa|artarsa|düşerse|yükselirse|gelirse|çıkarsa|kalkarsa|gevşerse|sıkılaşırsa|başlarsa|biterse|değişirse|kırılırsa)"],
        "en": [r"(what if|what happens if|if .+ then|suppose|assuming|in case)"],
    },
    "why": {
        "tr": [r"(neden|niye|niçin|sebebi ne|nedeni ne)"],
        "en": [r"(why|reason|cause|what caused|how come)"],
    },
    "direction": {
        "tr": [r"(yükselir mi|düşer mi|artar mı|azalır mı|nereye gider|nereye kadar|ne kadar)"],
        "en": [r"(go up|go down|rise|fall|direction|where.*head|target|how far|how high|how low)"],
    },
    "timing": {
        "tr": [r"(ne zaman|nezaman|sürer mi|surer mi|kadar devam|bitecek)"],
        "en": [r"(when|how long|duration|until when|timeline)"],
    },
    "comparison": {
        "tr": [r"(hangisi|karşılaştır|kıyasla|daha iyi|daha kötü|fark ne)"],
        "en": [r"(which.*better|compare|versus|vs|difference between)"],
    },
    "risk": {
        "tr": [r"(risk|tehlike|mantıklı mı|güvenli mi|güvenmeli mi|dikkat|uyarı)"],
        "en": [r"(risk|danger|safe|careful|warning|cautious|worried)"],
    },
}


def classify_question(text: str) -> str:
    """Classify question type."""
    text_lower = text.strip().lower()
    for qtype, patterns in _QUESTION_TYPES.items():
        for lang_patterns in patterns.values():
            for pattern in lang_patterns:
                if re.search(pattern, text_lower):
                    return qtype
    return "general"


# --- Extract hypothetical factor from what-if questions ---

def _extract_factor_from_question(text: str) -> str:
    """Extract the hypothetical factor/event from a what-if question."""
    text_lower = text.strip().lower()
    # Remove question marks and common prefixes
    text_clean = re.sub(r"\?+$", "", text_lower).strip()
    # TR patterns
    text_clean = re.sub(r"^(peki|ee|ya|acaba|sence|sizce)\s+", "", text_clean)
    # Extract the condition part (before "ne olur", "etkisi ne" etc.)
    for splitter in ["ne olur", "ne olacak", "etkisi ne", "nasıl etkiler", "what happens",
                     "what would", "how would", "what if"]:
        if splitter in text_clean:
            parts = text_clean.split(splitter)
            return parts[0].strip().rstrip(",. ")
    return text_clean


# --- Answer generation ---

def generate_answer(
    question: str,
    df: pd.DataFrame,
    asset_name: str,
    asset_key: str,
    lang: str = "tr",
) -> str:
    """Generate a data-driven answer to the user's question about the asset."""
    qtype = classify_question(question)
    row = df.iloc[-1]

    # Gather technical state
    close = row.get("Close", 0)
    rsi = row.get("RSI", 50)
    sma_50 = row.get("SMA_50")
    sma_200 = row.get("SMA_200")
    bb_upper = row.get("BB_upper")
    bb_lower = row.get("BB_lower")
    macd = row.get("MACD")
    macd_signal = row.get("MACD_signal")
    stoch_k = row.get("Stoch_K", 50)

    # Trend
    if sma_50 is not None and sma_200 is not None and not np.isnan(sma_50) and not np.isnan(sma_200):
        if close > sma_50 > sma_200:
            trend = "strong_up"
        elif close < sma_50 < sma_200:
            trend = "strong_down"
        elif sma_50 > sma_200:
            trend = "up"
        else:
            trend = "down"
    else:
        trend = "neutral"

    # Recent performance
    if len(df) > 5:
        pct_5d = ((close - df["Close"].iloc[-6]) / df["Close"].iloc[-6]) * 100
    else:
        pct_5d = 0
    if len(df) > 20:
        pct_20d = ((close - df["Close"].iloc[-21]) / df["Close"].iloc[-21]) * 100
    else:
        pct_20d = 0

    # Volatility
    if len(df) > 20:
        volatility = df["Close"].iloc[-20:].pct_change().std() * 100
    else:
        volatility = 0

    # Support/resistance from Bollinger
    bb_width = None
    if bb_upper is not None and bb_lower is not None:
        if not np.isnan(bb_upper) and not np.isnan(bb_lower):
            bb_width = ((bb_upper - bb_lower) / close) * 100

    if lang == "tr":
        return _generate_answer_tr(
            qtype, question, asset_name, asset_key, close, rsi, trend,
            sma_50, sma_200, bb_upper, bb_lower, bb_width,
            macd, macd_signal, stoch_k, pct_5d, pct_20d, volatility,
        )
    else:
        return _generate_answer_en(
            qtype, question, asset_name, asset_key, close, rsi, trend,
            sma_50, sma_200, bb_upper, bb_lower, bb_width,
            macd, macd_signal, stoch_k, pct_5d, pct_20d, volatility,
        )


def _generate_answer_tr(
    qtype, question, name, asset_key, close, rsi, trend,
    sma_50, sma_200, bb_upper, bb_lower, bb_width,
    macd, macd_signal, stoch_k, pct_5d, pct_20d, volatility,
) -> str:
    """Generate Turkish answer based on question type and technical data."""

    trend_desc = {
        "strong_up": "güçlü yükseliş trendinde",
        "strong_down": "sert düşüş trendinde",
        "up": "yukarı eğilimli",
        "down": "aşağı baskı altında",
        "neutral": "yatay seyirde",
    }[trend]

    momentum_desc = ""
    if rsi > 70:
        momentum_desc = "aşırı alım bölgesinde (RSI {:.0f})".format(rsi)
    elif rsi < 30:
        momentum_desc = "aşırı satım bölgesinde (RSI {:.0f})".format(rsi)
    elif rsi > 55:
        momentum_desc = "pozitif momentumlu (RSI {:.0f})".format(rsi)
    elif rsi < 45:
        momentum_desc = "zayıf momentumlu (RSI {:.0f})".format(rsi)
    else:
        momentum_desc = "nötr momentumlu (RSI {:.0f})".format(rsi)

    if qtype == "what_if":
        factor = _extract_factor_from_question(question)
        return (
            f"Sorunu veriler ışığında değerlendirelim. {name} şu an {close:,.2f} seviyesinde, "
            f"{trend_desc} ve {momentum_desc}. "
            f"Son 5 günde %{pct_5d:+.1f}, son 20 günde %{pct_20d:+.1f} değişim var. "
            f"\n\n"
            f"Eğer \"{factor}\" gerçekleşirse, şu senaryolar ortaya çıkabilir:\n\n"
            f"**Senaryo 1 (Olumlu etki):** Mevcut {trend_desc} yapı güçlenir. "
            f"{'RSI zaten aşırı alımda olduğu için yükseliş hızlanabilir ama düzeltme riski de artar.' if rsi > 70 else 'RSI hâlâ hareket alanı bırakıyor, yükseliş devam edebilir.' if rsi < 65 else 'RSI yüksek ama henüz aşırı değil.'} "
            f"Bollinger üst bandı ({bb_upper:,.2f}) ilk hedef olabilir.\n\n"
            f"**Senaryo 2 (Olumsuz etki):** Mevcut yapı bozulur. "
            f"SMA 50 ({sma_50:,.2f}) ilk destek, kırılırsa SMA 200 ({sma_200:,.2f}) devreye girer. "
            f"{'Zaten zayıf olan momentum daha da çözülebilir.' if rsi < 45 else 'Momentum henüz güçlü ama sert bir kırılma hızlı dönüş yapabilir.'}\n\n"
            f"**Senaryo 3 (Sınırlı etki):** Piyasa haberi fiyatlamış olabilir. "
            f"Günlük volatilite %{volatility:.2f} — {'düşük volatilite büyük bir hareketin biriktiğini gösterebilir' if volatility < 1 else 'mevcut volatilite zaten yüksek, yeni bir katalizör etkisi sınırlı kalabilir' if volatility > 2 else 'orta düzey volatilite, her iki yöne hareket mümkün'}."
        ) if sma_50 and sma_200 and bb_upper else (
            f"Sorunu veriler ışığında değerlendirelim. {name} şu an {close:,.2f} seviyesinde, "
            f"{trend_desc} ve {momentum_desc}. Eğer \"{factor}\" gerçekleşirse, "
            f"mevcut yapıda önemli bir değişiklik olabilir. Son 5 günde %{pct_5d:+.1f} değişim var — "
            f"bu hareketin devamı veya dönüşü söz konusu faktörün gücüne bağlı."
        )

    elif qtype == "why":
        parts = [
            f"{name}'ın mevcut durumunu verilerle açıklayalım. Fiyat {close:,.2f} seviyesinde ve {trend_desc}."
        ]
        # Explain based on what data shows
        if abs(pct_5d) > 2:
            parts.append(
                f"Son 5 günde %{pct_5d:+.1f} hareket var — bu {'güçlü bir alım dalgasına' if pct_5d > 0 else 'yoğun satış baskısına'} işaret ediyor."
            )
        if sma_50 and sma_200 and not np.isnan(sma_50) and not np.isnan(sma_200):
            if sma_50 > sma_200:
                parts.append(f"50 günlük ortalama ({sma_50:,.2f}) 200 günlüğün ({sma_200:,.2f}) üzerinde — bu uzun vadeli yükseliş yapısı korunuyor demek.")
            else:
                parts.append(f"50 günlük ortalama ({sma_50:,.2f}) 200 günlüğün ({sma_200:,.2f}) altında — uzun vadeli yapı zayıf.")
        if rsi > 65:
            parts.append(f"RSI {rsi:.0f} ile alıcılar hâlâ dominant. Bu yükseliş talebinin güçlü olduğunu gösteriyor.")
        elif rsi < 35:
            parts.append(f"RSI {rsi:.0f} ile satıcılar baskın. Ama bu seviye tükenme sinyali de olabilir.")
        if macd is not None and macd_signal is not None and not np.isnan(macd) and not np.isnan(macd_signal):
            if macd > macd_signal:
                parts.append("MACD sinyal çizgisinin üzerinde — momentum hâlâ yukarı yönlü.")
            else:
                parts.append("MACD sinyal çizgisinin altında — momentum aşağı dönmüş.")
        # No filler closing
        return " ".join(parts)

    elif qtype == "direction":
        parts = [f"{name} {close:,.2f} seviyesinde. Yön analizi:"]

        # Bull case
        bull_points = []
        if trend in ("strong_up", "up"):
            bull_points.append("trend yukarı")
        if rsi > 50 and rsi < 70:
            bull_points.append("momentum pozitif ama henüz aşırı değil")
        if macd is not None and macd_signal is not None and not np.isnan(macd) and not np.isnan(macd_signal) and macd > macd_signal:
            bull_points.append("MACD pozitif")
        if pct_5d > 0:
            bull_points.append(f"son 5 gün %{pct_5d:+.1f}")

        # Bear case
        bear_points = []
        if trend in ("strong_down", "down"):
            bear_points.append("trend aşağı")
        if rsi > 70:
            bear_points.append("RSI aşırı alımda — düzeltme riski")
        elif rsi < 40:
            bear_points.append("momentum zayıf")
        if macd is not None and macd_signal is not None and not np.isnan(macd) and not np.isnan(macd_signal) and macd < macd_signal:
            bear_points.append("MACD negatif")
        if pct_5d < 0:
            bear_points.append(f"son 5 gün %{pct_5d:+.1f}")

        bull_score = len(bull_points)
        bear_score = len(bear_points)

        if bull_points:
            parts.append(f"\n**Yukarı yönü destekleyen:** {', '.join(bull_points)}.")
        if bear_points:
            parts.append(f"\n**Aşağı yönü destekleyen:** {', '.join(bear_points)}.")

        if bb_upper and bb_lower and not np.isnan(bb_upper) and not np.isnan(bb_lower):
            parts.append(f"\nBollinger bandı: alt {bb_lower:,.2f} — üst {bb_upper:,.2f}. Fiyat bu band içinde hareket etme eğiliminde.")

        if bull_score > bear_score + 1:
            parts.append(f"\nVeriler ağırlıklı olarak yukarı yönü işaret ediyor ({bull_score} vs {bear_score}).")
        elif bear_score > bull_score + 1:
            parts.append(f"\nVeriler ağırlıklı olarak aşağı yönü işaret ediyor ({bear_score} vs {bull_score}). Dikkatli olmakta fayda var.")
        else:
            parts.append(f"\nSinyaller neredeyse eşit ({bull_score} yukarı vs {bear_score} aşağı). Kararsız bir dönemdeyiz, bir katalizör bekliyor olabilir.")

        return " ".join(parts)

    elif qtype == "timing":
        return (
            f"{name} için zamanlama sorusu zor ama verilerin söylediklerine bakalım. "
            f"Fiyat {close:,.2f}, {trend_desc}. "
            f"{'Bollinger bantları sıkışmış (genişlik %{:.1f}) — bu genelde büyük bir hareketin yakın olduğuna işaret eder. Genellikle 1-2 hafta içinde kırılma gelir.'.format(bb_width) if bb_width and bb_width < 3 else 'Bollinger bantları geniş (genişlik %{:.1f}) — volatilite yüksek, hareket zaten başlamış olabilir.'.format(bb_width) if bb_width and bb_width > 8 else ''} "
            f"{'RSI aşırı bölgede ({:.0f}) — bu seviyeler genelde uzun sürmez, birkaç gün içinde tepki olasılığı yüksek.'.format(rsi) if rsi > 75 or rsi < 25 else ''} "
            f"Son 20 günlük volatilite %{volatility:.2f} — "
            f"{'düşük volatilite genelde sakinlik öncesi fırtına demek.' if volatility < 0.8 else 'yüksek volatilite hareketlerin devam edeceğini gösteriyor.' if volatility > 2 else 'normal düzeyde, keskin bir zamanlama tahmini yapmak zor.'}"
        )

    elif qtype == "risk":
        risk_factors = []
        safe_factors = []

        if rsi > 75:
            risk_factors.append(f"RSI {rsi:.0f} — aşırı alım, düzeltme riski yüksek")
        elif rsi < 25:
            risk_factors.append(f"RSI {rsi:.0f} — aşırı satım, dip avcılığı riskli olabilir")
        if volatility > 2.5:
            risk_factors.append(f"volatilite yüksek (%{volatility:.2f}) — sert hareketler beklenmeli")
        if trend == "strong_down":
            risk_factors.append("güçlü düşüş trendi — düşen bıçağı tutmak tehlikeli")
        if bb_width and bb_width < 2.5:
            risk_factors.append("Bollinger sıkışması — büyük hareket kapıda, yön belirsiz")

        if trend == "strong_up":
            safe_factors.append("güçlü yükseliş trendi — momentum arkada")
        if 40 < rsi < 60:
            safe_factors.append(f"RSI dengeli ({rsi:.0f}) — aşırı uçlarda değil")
        if volatility < 1.5:
            safe_factors.append("volatilite makul — sürpriz hareket olasılığı düşük")

        parts = [f"{name} için risk değerlendirmesi ({close:,.2f}):"]
        if risk_factors:
            parts.append(f"\n⚠️ **Dikkat edilmesi gerekenler:** {'; '.join(risk_factors)}.")
        if safe_factors:
            parts.append(f"\n✅ **Olumlu taraflar:** {'; '.join(safe_factors)}.")

        total = len(risk_factors) + len(safe_factors)
        if total > 0:
            risk_pct = len(risk_factors) / total * 100
            if risk_pct > 60:
                parts.append("\nGenel olarak risk faktörleri ağır basıyor. Temkinli olmakta fayda var.")
            elif risk_pct < 40:
                parts.append("\nTeknik tablo görece güvenli görünüyor.")
            else:
                parts.append("\nDengeli bir tablo — ne aşırı riskli ne de güvenli. Pozisyon boyutuna dikkat etmek mantıklı.")

        return " ".join(parts)

    else:  # general
        return (
            f"{name} hakkında genel durumu özetleyeyim. Fiyat {close:,.2f} seviyesinde, {trend_desc} ve {momentum_desc}. "
            f"Son 5 günde %{pct_5d:+.1f}, son 20 günde %{pct_20d:+.1f} değişim var. "
            f"{'Volatilite düşük — sakin bir dönemdeyiz ama bu aldatıcı olabilir.' if volatility < 1 else 'Volatilite yüksek — hareketli bir dönemdeyiz.' if volatility > 2 else 'Volatilite normal seviyelerde.'} "
            f"{'Bollinger bantları sıkışmış, büyük bir hareket yakın olabilir.' if bb_width and bb_width < 3 else ''} "
            f""
        )


def _generate_answer_en(
    qtype, question, name, asset_key, close, rsi, trend,
    sma_50, sma_200, bb_upper, bb_lower, bb_width,
    macd, macd_signal, stoch_k, pct_5d, pct_20d, volatility,
) -> str:
    """Generate English answer based on question type and technical data."""

    trend_desc = {
        "strong_up": "in a strong uptrend",
        "strong_down": "in a sharp downtrend",
        "up": "with an upward bias",
        "down": "under downward pressure",
        "neutral": "trading sideways",
    }[trend]

    momentum_desc = ""
    if rsi > 70:
        momentum_desc = "overbought (RSI {:.0f})".format(rsi)
    elif rsi < 30:
        momentum_desc = "oversold (RSI {:.0f})".format(rsi)
    elif rsi > 55:
        momentum_desc = "positive momentum (RSI {:.0f})".format(rsi)
    elif rsi < 45:
        momentum_desc = "weak momentum (RSI {:.0f})".format(rsi)
    else:
        momentum_desc = "neutral momentum (RSI {:.0f})".format(rsi)

    if qtype == "what_if":
        factor = _extract_factor_from_question(question)
        parts = [
            f"Let's evaluate your question against the data. {name} is at {close:,.2f}, "
            f"{trend_desc} with {momentum_desc}. "
            f"5-day change: {pct_5d:+.1f}%, 20-day change: {pct_20d:+.1f}%.",
            f"\n\nIf \"{factor}\" materializes, here are the scenarios:\n",
        ]
        if sma_50 and sma_200 and bb_upper and not np.isnan(sma_50) and not np.isnan(sma_200) and not np.isnan(bb_upper):
            parts.append(
                f"**Scenario 1 (Bullish impact):** Current {trend_desc} structure strengthens. "
                f"{'RSI already overbought — rally could accelerate but correction risk increases.' if rsi > 70 else 'RSI still has room, upside can continue.' if rsi < 65 else 'RSI elevated but not extreme.'} "
                f"Upper Bollinger ({bb_upper:,.2f}) is the first target.\n\n"
                f"**Scenario 2 (Bearish impact):** Structure breaks down. "
                f"SMA 50 ({sma_50:,.2f}) is first support, if broken SMA 200 ({sma_200:,.2f}) comes into play. "
                f"{'Already weak momentum could deteriorate further.' if rsi < 45 else 'Momentum is still strong but a sharp break can reverse quickly.'}\n\n"
                f"**Scenario 3 (Muted impact):** Market may have already priced it in. "
                f"Daily volatility is {volatility:.2f}% — "
                f"{'low volatility suggests a big move is building' if volatility < 1 else 'high volatility means another catalyst may have limited additional impact' if volatility > 2 else 'moderate volatility, moves possible in either direction'}."
            )
        else:
            parts.append(
                f"Current trend is {trend_desc}. The 5-day change of {pct_5d:+.1f}% suggests "
                f"{'momentum is behind the move' if abs(pct_5d) > 2 else 'the market is relatively calm'}. "
                f"The factor's impact depends on its magnitude and market positioning."
            )
        return " ".join(parts)

    elif qtype == "why":
        parts = [f"Let's break down {name}'s current state with data. Price is at {close:,.2f}, {trend_desc}."]
        if abs(pct_5d) > 2:
            parts.append(f"5-day move of {pct_5d:+.1f}% indicates {'strong buying' if pct_5d > 0 else 'heavy selling'} pressure.")
        if sma_50 and sma_200 and not np.isnan(sma_50) and not np.isnan(sma_200):
            if sma_50 > sma_200:
                parts.append(f"50-day MA ({sma_50:,.2f}) above 200-day ({sma_200:,.2f}) — long-term bullish structure intact.")
            else:
                parts.append(f"50-day MA ({sma_50:,.2f}) below 200-day ({sma_200:,.2f}) — long-term structure is weak.")
        if macd is not None and macd_signal is not None and not np.isnan(macd) and not np.isnan(macd_signal):
            parts.append(f"MACD is {'above' if macd > macd_signal else 'below'} signal — momentum is {'up' if macd > macd_signal else 'down'}.")
        # No filler closing
        return " ".join(parts)

    elif qtype == "direction":
        bull_points, bear_points = [], []
        if trend in ("strong_up", "up"): bull_points.append("trend is up")
        if 50 < rsi < 70: bull_points.append("positive momentum without being overbought")
        if macd and macd_signal and not np.isnan(macd) and not np.isnan(macd_signal) and macd > macd_signal: bull_points.append("MACD positive")
        if pct_5d > 0: bull_points.append(f"5-day: {pct_5d:+.1f}%")
        if trend in ("strong_down", "down"): bear_points.append("trend is down")
        if rsi > 70: bear_points.append("RSI overbought — correction risk")
        elif rsi < 40: bear_points.append("weak momentum")
        if macd and macd_signal and not np.isnan(macd) and not np.isnan(macd_signal) and macd < macd_signal: bear_points.append("MACD negative")
        if pct_5d < 0: bear_points.append(f"5-day: {pct_5d:+.1f}%")

        parts = [f"{name} at {close:,.2f} — direction analysis:"]
        if bull_points: parts.append(f"\n**Bullish case:** {', '.join(bull_points)}.")
        if bear_points: parts.append(f"\n**Bearish case:** {', '.join(bear_points)}.")

        bull_s, bear_s = len(bull_points), len(bear_points)
        if bull_s > bear_s + 1:
            parts.append(f"\nData leans bullish ({bull_s} vs {bear_s}).")
        elif bear_s > bull_s + 1:
            parts.append(f"\nData leans bearish ({bear_s} vs {bull_s}). Caution warranted.")
        else:
            parts.append(f"\nSignals roughly balanced ({bull_s} up vs {bear_s} down). Waiting for a catalyst.")
        return " ".join(parts)

    elif qtype == "risk":
        risk_f, safe_f = [], []
        if rsi > 75: risk_f.append(f"RSI {rsi:.0f} — overbought, correction risk high")
        elif rsi < 25: risk_f.append(f"RSI {rsi:.0f} — oversold, catching a falling knife is risky")
        if volatility > 2.5: risk_f.append(f"high volatility ({volatility:.2f}%)")
        if trend == "strong_down": risk_f.append("strong downtrend")
        if trend == "strong_up": safe_f.append("strong uptrend — momentum behind you")
        if 40 < rsi < 60: safe_f.append(f"balanced RSI ({rsi:.0f})")
        if volatility < 1.5: safe_f.append("reasonable volatility")

        parts = [f"{name} risk assessment ({close:,.2f}):"]
        if risk_f: parts.append(f"\n⚠️ **Watch out:** {'; '.join(risk_f)}.")
        if safe_f: parts.append(f"\n✅ **Positive:** {'; '.join(safe_f)}.")
        return " ".join(parts)

    else:
        return (
            f"{name} summary: price at {close:,.2f}, {trend_desc}, {momentum_desc}. "
            f"5-day: {pct_5d:+.1f}%, 20-day: {pct_20d:+.1f}%. "
            f"{'Low volatility — calm before a storm?' if volatility < 1 else 'High volatility — expect sharp moves.' if volatility > 2 else 'Normal volatility.'} "
            f""
        )
