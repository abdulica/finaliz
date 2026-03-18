"""Technical analysis engine with auto-generated commentary."""

import numpy as np
import pandas as pd
import ta

from config import TA_PARAMS


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators on OHLCV DataFrame."""
    if df is None or df.empty or len(df) < 30:
        return df

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df.get("Volume", pd.Series(0, index=df.index))

    # RSI
    df["RSI"] = ta.momentum.rsi(close, window=TA_PARAMS["rsi_period"])

    # MACD
    macd = ta.trend.MACD(
        close,
        window_slow=TA_PARAMS["macd_slow"],
        window_fast=TA_PARAMS["macd_fast"],
        window_sign=TA_PARAMS["macd_signal"],
    )
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(
        close, window=TA_PARAMS["bb_period"], window_dev=TA_PARAMS["bb_std"]
    )
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]

    # Moving Averages
    for p in TA_PARAMS["sma_periods"]:
        df[f"SMA_{p}"] = ta.trend.sma_indicator(close, window=p)
    for p in TA_PARAMS["ema_periods"]:
        df[f"EMA_{p}"] = ta.trend.ema_indicator(close, window=p)

    # Stochastic RSI
    stoch = ta.momentum.StochRSIIndicator(close, window=TA_PARAMS["stoch_rsi_period"])
    df["StochRSI_K"] = stoch.stochrsi_k()
    df["StochRSI_D"] = stoch.stochrsi_d()

    # ATR
    df["ATR"] = ta.volatility.average_true_range(
        high, low, close, window=TA_PARAMS["atr_period"]
    )

    # Volume SMA
    if volume.sum() > 0:
        df["Volume_SMA_20"] = ta.trend.sma_indicator(volume.astype(float), window=20)

    return df


def compute_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """Compute support and resistance levels using pivot points and recent extremes."""
    if df is None or df.empty or len(df) < window:
        return {"support": [], "resistance": []}

    recent = df.tail(window)
    high = recent["High"].max()
    low = recent["Low"].min()
    close = recent["Close"].iloc[-1]

    # Classic pivot point
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)

    return {
        "pivot": pivot,
        "support": [round(s1, 4), round(s2, 4)],
        "resistance": [round(r1, 4), round(r2, 4)],
    }


def get_signal_summary(df: pd.DataFrame) -> dict:
    """Generate a summary of all technical signals.

    Returns dict with counts and overall signal.
    """
    if df is None or df.empty or len(df) < 30:
        return {"buy": 0, "sell": 0, "neutral": 0, "overall": "neutral"}

    last = df.iloc[-1]
    prev = df.iloc[-2]
    buy, sell, neutral = 0, 0, 0
    signals = []

    # RSI signal
    rsi = last.get("RSI")
    if rsi is not None and not np.isnan(rsi):
        if rsi < 30:
            buy += 1
            signals.append(("RSI", "buy", rsi))
        elif rsi > 70:
            sell += 1
            signals.append(("RSI", "sell", rsi))
        else:
            neutral += 1
            signals.append(("RSI", "neutral", rsi))

    # MACD signal
    macd_h = last.get("MACD_Hist")
    prev_macd_h = prev.get("MACD_Hist")
    if macd_h is not None and not np.isnan(macd_h):
        if macd_h > 0 and (prev_macd_h is not None and prev_macd_h <= 0):
            buy += 1
            signals.append(("MACD", "buy", macd_h))
        elif macd_h < 0 and (prev_macd_h is not None and prev_macd_h >= 0):
            sell += 1
            signals.append(("MACD", "sell", macd_h))
        elif macd_h > 0:
            buy += 1
            signals.append(("MACD", "buy", macd_h))
        else:
            sell += 1
            signals.append(("MACD", "sell", macd_h))

    # Bollinger Band signal
    close = last["Close"]
    bb_upper = last.get("BB_Upper")
    bb_lower = last.get("BB_Lower")
    if bb_upper is not None and not np.isnan(bb_upper):
        if close <= bb_lower:
            buy += 1
            signals.append(("BB", "buy", close))
        elif close >= bb_upper:
            sell += 1
            signals.append(("BB", "sell", close))
        else:
            neutral += 1
            signals.append(("BB", "neutral", close))

    # SMA signals
    for p in TA_PARAMS["sma_periods"]:
        sma = last.get(f"SMA_{p}")
        if sma is not None and not np.isnan(sma):
            if close > sma:
                buy += 1
                signals.append((f"SMA_{p}", "buy", sma))
            else:
                sell += 1
                signals.append((f"SMA_{p}", "sell", sma))

    # Stochastic RSI
    stoch_k = last.get("StochRSI_K")
    if stoch_k is not None and not np.isnan(stoch_k):
        if stoch_k < 0.2:
            buy += 1
            signals.append(("StochRSI", "buy", stoch_k))
        elif stoch_k > 0.8:
            sell += 1
            signals.append(("StochRSI", "sell", stoch_k))
        else:
            neutral += 1
            signals.append(("StochRSI", "neutral", stoch_k))

    total = buy + sell + neutral
    if total == 0:
        overall = "neutral"
    elif buy > sell and buy > neutral:
        overall = "buy"
    elif sell > buy and sell > neutral:
        overall = "sell"
    else:
        overall = "neutral"

    return {
        "buy": buy,
        "sell": sell,
        "neutral": neutral,
        "overall": overall,
        "signals": signals,
    }


def generate_commentary(
    df: pd.DataFrame,
    asset_name: str,
    lang: str = "tr",
    external_context: str = "",
) -> list[str]:
    """Generate a conversational 1-2 paragraph overview of the technical picture.

    Instead of listing each indicator separately, this weaves all signals
    into a cohesive narrative — like a friend explaining what they see on the chart.

    If external_context is provided (user-supplied qualitative info like geopolitical
    factors, policy interventions, etc.), the commentary acknowledges that the pure
    technical picture may not tell the full story and integrates the external context.
    """
    if df is None or df.empty or len(df) < 30:
        return []

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    close = last["Close"]
    summary = get_signal_summary(df)
    sr = compute_support_resistance(df)

    # --- Gather all the raw observations ---
    rsi = _safe(last, "RSI")
    macd_h = _safe(last, "MACD_Hist")
    prev_macd_h = _safe(prev, "MACD_Hist")
    bb_width = _safe(last, "BB_Width")
    avg_bb_width = df["BB_Width"].tail(50).mean() if "BB_Width" in df else None
    sma_50 = _safe(last, "SMA_50")
    sma_200 = _safe(last, "SMA_200")
    stoch_k = _safe(last, "StochRSI_K")

    # Trend direction
    if sma_50 is not None and sma_200 is not None:
        if sma_50 > sma_200 and close > sma_200:
            trend = "strong_up"
        elif sma_50 > sma_200:
            trend = "up"
        elif sma_50 < sma_200 and close < sma_200:
            trend = "strong_down"
        elif sma_50 < sma_200:
            trend = "down"
        else:
            trend = "neutral"
    else:
        trend = "unknown"

    # Momentum state
    if rsi is not None:
        if rsi > 70:
            momentum = "overbought"
        elif rsi < 30:
            momentum = "oversold"
        elif rsi > 55:
            momentum = "bullish"
        elif rsi < 45:
            momentum = "bearish"
        else:
            momentum = "neutral"
    else:
        momentum = "unknown"

    # MACD event
    macd_event = None
    if macd_h is not None and prev_macd_h is not None:
        if macd_h > 0 and prev_macd_h <= 0:
            macd_event = "bullish_cross"
        elif macd_h < 0 and prev_macd_h >= 0:
            macd_event = "bearish_cross"
        elif macd_h > 0:
            macd_event = "positive"
        else:
            macd_event = "negative"

    # Volatility state
    bb_squeeze = False
    if bb_width is not None and avg_bb_width is not None:
        bb_squeeze = bb_width < avg_bb_width * 0.6

    # Support/resistance proximity
    near_support = near_resistance = False
    support_val = resistance_val = None
    if sr["support"] and sr["resistance"]:
        support_val = sr["support"][0]
        resistance_val = sr["resistance"][0]
        near_support = abs(close - support_val) / close * 100 < 1.5
        near_resistance = abs(close - resistance_val) / close * 100 < 1.5

    # Signal agreement
    total = summary["buy"] + summary["sell"] + summary["neutral"]
    if total > 0:
        agreement = max(summary["buy"], summary["sell"], summary["neutral"]) / total
    else:
        agreement = 0
    signals_mixed = agreement < 0.5

    # --- Build the narrative ---
    ext = external_context.strip() if external_context else ""

    if lang == "tr":
        paragraphs = _build_narrative_tr(
            asset_name, close, rsi, trend, momentum, macd_event,
            bb_squeeze, near_support, near_resistance, support_val,
            resistance_val, signals_mixed, summary, sma_50, sma_200, stoch_k,
            ext,
        )
    else:
        paragraphs = _build_narrative_en(
            asset_name, close, rsi, trend, momentum, macd_event,
            bb_squeeze, near_support, near_resistance, support_val,
            resistance_val, signals_mixed, summary, sma_50, sma_200, stoch_k,
            ext,
        )

    return paragraphs


def _safe(row, col):
    """Safely get a value, returning None if NaN."""
    val = row.get(col)
    if val is not None and not np.isnan(val):
        return val
    return None


def _build_narrative_tr(
    name, close, rsi, trend, momentum, macd_event,
    bb_squeeze, near_support, near_resistance, support_val,
    resistance_val, signals_mixed, summary, sma_50, sma_200, stoch_k,
    ext="",
) -> list[str]:
    """Build Turkish conversational narrative. If ext (external context) is provided,
    every observation is reframed through that lens."""
    has_ext = bool(ext)

    # --- Paragraph 1: The big picture ---
    p1_parts = []

    # Simple prefix if external context exists — no repeating the input
    if has_ext:
        p1_parts.append("Sunduğun ek bilgi göz önüne alındığında:")

    # Opening with general state — reframed if ext
    if trend == "strong_up":
        if has_ext:
            p1_parts.append(
                f"Fiyat {close:,.2f} seviyesinde, 50 ve 200 günlük ortalamaların üzerinde. "
                f"Teknik olarak güçlü görünüyor ama asıl soru bu trendin organik mi yoksa yapay mı olduğu. "
                f"Doğal arz-talep yerine dışsal müdahaleyle şekilleniyorsa, bu 'güç' yanıltıcı olabilir."
            )
        else:
            p1_parts.append(
                f"Şu an {name}'a baktığımda genel olarak güçlü bir yükseliş trendi görüyorum. "
                f"Fiyat {close:,.2f} seviyesinde ve hem 50 hem 200 günlük ortalamaların üzerinde seyrediyor, "
                f"bu teknik açıdan sağlam bir yapıya işaret ediyor."
            )
    elif trend == "strong_down":
        if has_ext:
            p1_parts.append(
                f"Fiyat {close:,.2f} seviyesinde, her iki büyük ortalamanın da altında. "
                f"Bu düşüş piyasa dinamiklerinden mi yoksa dışsal baskıdan mı kaynaklanıyor, "
                f"onu ayırt etmek kritik. Dışsal baskı varsa toparlanma sinyalleri güvenilir olmayabilir."
            )
        else:
            p1_parts.append(
                f"{name}'da işler pek parlak görünmüyor açıkçası. "
                f"Fiyat {close:,.2f} seviyesinde, hem 50 hem 200 günlük ortalamaların altında — "
                f"yani klasik tabiriyle 'death cross' bölgesindeyiz. Aşağı yönlü baskı hakim."
            )
    elif trend == "up":
        if has_ext:
            p1_parts.append(
                f"{name} {close:,.2f} seviyesinde, teknik olarak yukarı eğilimli. "
                f"Ancak bu yükselişin ne kadarı gerçek piyasa hareketi, ne kadarı dışsal etki — "
                f"bunu ayrıştırmak lazım."
            )
        else:
            p1_parts.append(
                f"{name} şu an {close:,.2f} seviyesinde ve genel eğilim yukarı yönlü. "
                f"50 günlük ortalama 200 günlüğün üzerinde, bu olumlu bir işaret."
            )
    elif trend == "down":
        if has_ext:
            p1_parts.append(
                f"{name} {close:,.2f} seviyesinde, kısa vadeli ortalamalar uzun vadelilerin altında. "
                f"Bu teknik zayıflık, dışsal faktörle birlikte değerlendirildiğinde tablonun "
                f"göründüğünden daha derin olabileceğini düşündürüyor."
            )
        else:
            p1_parts.append(
                f"{name} {close:,.2f} seviyesinde ve kısa vadeli ortalamalar uzun vadelilerin altına sarkmış durumda. "
                f"Genel tablo biraz tedirgin edici."
            )
    else:
        p1_parts.append(f"{name} şu an {close:,.2f} seviyesinde işlem görüyor.")

    # Momentum — reframed if ext
    if momentum == "overbought":
        if has_ext:
            p1_parts.append(
                f"RSI {rsi:.0f} ile aşırı alım bölgesinde ama doğal geri çekilme mekanizması "
                f"baskılanmış olabilir."
            )
        else:
            p1_parts.append(
                f"RSI {rsi:.0f} ile aşırı alım bölgesinde — "
                f"yani piyasa biraz fazla ısınmış durumda, bir nefes alma ihtimali var."
            )
    elif momentum == "oversold":
        if has_ext:
            p1_parts.append(
                f"RSI {rsi:.0f} ile aşırı satım bölgesinde ama dışsal baskı sürdükçe "
                f"bu seviye 'dip' anlamına gelmeyebilir."
            )
        else:
            p1_parts.append(
                f"RSI {rsi:.0f} seviyesiyle aşırı satım bölgesine girmiş. "
                f"Satıcılar yorulmuş olabilir, buradan bir tepki gelmesi şaşırtıcı olmaz."
            )
    elif momentum == "bullish":
        if has_ext:
            p1_parts.append(
                f"RSI {rsi:.0f} ile pozitif momentum var ama organik mi yoksa yapay destekli mi, o belirsiz."
            )
        else:
            p1_parts.append(f"RSI {rsi:.0f} ile pozitif momentum devam ediyor.")
    elif momentum == "bearish":
        if has_ext:
            p1_parts.append(
                f"RSI {rsi:.0f} ile momentum zayıflıyor — yapısal mı geçici mi, dışsal faktör belirleyecek."
            )
        else:
            p1_parts.append(f"RSI {rsi:.0f} ile momentum zayıflıyor.")

    # MACD
    if macd_event == "bullish_cross":
        if has_ext:
            p1_parts.append(
                "MACD'de pozitif kesişim var ama müdahale altındaki piyasalarda "
                "MACD kesişimleri sıklıkla sahte sinyal üretir."
            )
        else:
            p1_parts.append(
                "Üstelik MACD'de tam da şimdi pozitif bir kesişim oldu — "
                "bu momentum değişiminin taze bir sinyali."
            )
    elif macd_event == "bearish_cross":
        if has_ext:
            p1_parts.append(
                "MACD'de negatif kesişim oluştu. Bu, müdahalenin zayıfladığına veya "
                "piyasanın baskıya rağmen kendi yolunu bulmaya çalıştığına işaret edebilir."
            )
        else:
            p1_parts.append(
                "Buna ek olarak MACD'de negatif kesişim yaşandı, "
                "bu da aşağı yönlü baskının taze olduğunu gösteriyor."
            )

    paragraph_1 = " ".join(p1_parts)

    # --- Paragraph 2: What stands out + what to watch ---
    p2_parts = []

    if bb_squeeze:
        if has_ext:
            p2_parts.append(
                "Bollinger bantları sıkışmış — volatilite yapay olarak baskılanıyor olabilir. "
                "Baskı kalktığında hareket normelden sert olabilir."
            )
        else:
            p2_parts.append(
                "Beni en çok heyecanlandıran şey Bollinger bantlarının ciddi şekilde sıkışmış olması. "
                "Bu genelde büyük bir hareketin habercisi — yön belli değil ama volatilite patlaması kapıda."
            )
    elif signals_mixed:
        if has_ext:
            p2_parts.append(
                f"Göstergeler çelişiyor: {summary['buy']} alım, {summary['sell']} satım. "
                f"Piyasa doğal dengesini bulamıyor, asıl risk müdahale gevşediğinde hangi yöne kırılacağı."
            )
        else:
            p2_parts.append(
                f"Açıkçası burada göstergeler birbiriyle çelişiyor: "
                f"{summary['buy']} tanesi alım, {summary['sell']} tanesi satım diyor. "
                f"Bu tür kararsız dönemlerde piyasa genelde bir süre yatay gidip sonra sert kırar."
            )
    elif momentum == "overbought" and trend == "strong_up":
        if has_ext:
            p2_parts.append(
                "Trend güçlü, göstergeler aşırı alımda. Fiyat dışarıdan destekleniyorsa "
                "düzeltme mekanizması çalışmayabilir — bu da riski artırır."
            )
        else:
            p2_parts.append(
                "İlginç olan şu: trend çok güçlü ama aynı zamanda göstergeler aşırı alım diyor. "
                "Bu tür durumlarda ya sert bir düzeltme gelir ya da fiyat yüksek seviyelerde konsolide olur. "
                "İkisine de hazırlıklı olmak lazım."
            )
    elif momentum == "oversold" and trend == "strong_down":
        if has_ext:
            p2_parts.append(
                "Hem trend aşağı hem göstergeler aşırı satımda. Dışsal baskı devredeyken "
                "göstergeler uzun süre bu bölgede kalabilir, dikkatli olmak lazım."
            )
        else:
            p2_parts.append(
                "Dikkat çekici bir nokta: hem trend aşağı hem de göstergeler aşırı satımda. "
                "Tarihsel olarak bu seviyelerden teknik tepkiler gelir ama trend bu kadar güçlüyken "
                "düşüşün devam etmesi de olasılıklar dahilinde."
            )
    elif macd_event in ("bullish_cross", "bearish_cross"):
        direction = "yukarı" if macd_event == "bullish_cross" else "aşağı"
        if has_ext:
            p2_parts.append(
                "MACD'deki taze kesişimin güvenilirliği bu koşullarda sorgulanmalı, "
                "ama baskının zayıfladığının ilk işareti de olabilir."
            )
        else:
            p2_parts.append(
                f"MACD'deki taze kesişim bence buradaki en önemli sinyal. "
                f"{direction.capitalize()} yönlü bir momentum başlangıcı olabilir, "
                f"ama bunu hacimle teyit etmek şart."
            )

    # Support/resistance
    if near_support and support_val is not None:
        if has_ext:
            p2_parts.append(
                f"Fiyat {support_val:,.2f} destek seviyesine yakın. Bu destek gerçek alıcılardan mı "
                f"yoksa müdahaleden mi oluşuyor — yapay destek kırıldığında tepki çok daha sert olur."
            )
        else:
            p2_parts.append(
                f"Bir de şuna dikkat etmekte fayda var: fiyat {support_val:,.2f} destek seviyesine çok yakın. "
                f"Buradan tepki gelebilir ama kırılırsa hızlı bir düşüş yaşanabilir — kritik bir eşikteyiz."
            )
    elif near_resistance and resistance_val is not None:
        if has_ext:
            p2_parts.append(
                f"{resistance_val:,.2f} direnç seviyesi yukarıda. Kırılması da yapay olabilir — "
                f"gerçek kırılım mı müdahale kaynaklı mı, hacim ve sonraki seanslar gösterecek."
            )
        else:
            p2_parts.append(
                f"Gözden kaçmaması gereken bir detay: {resistance_val:,.2f} direnç seviyesi hemen yukarıda. "
                f"Bu seviyeyi aşarsa momentum kazanabilir, aşamazsa geri çekilme görülür."
            )

    # General closing
    if not p2_parts:
        total = summary['buy'] + summary['sell'] + summary['neutral']
        if has_ext:
            p2_parts.append(
                f"Göstergelerin genel tablosu ({summary['buy']} alım, {summary['sell']} satım, "
                f"{summary['neutral']} nötr) normal koşullarda anlamlı olurdu ama bu sinyalleri "
                f"indirimli okumak gerekiyor. Yine de tamamen göz ardı etmemek lazım — baskının "
                f"ne zaman gevşeyeceğinin ipuçlarını yine bu göstergeler verecek."
            )
        else:
            if summary["overall"] == "buy":
                p2_parts.append(
                    f"Genel olarak göstergelerin çoğunluğu ({summary['buy']}/{total}) "
                    f"pozitif yönde. Teknik tablo iyimser görünüyor ama her zaman beklenmedik gelişmelere açık olmak lazım."
                )
            elif summary["overall"] == "sell":
                p2_parts.append(
                    f"Göstergelerin çoğunluğu ({summary['sell']}/{total}) "
                    f"negatif sinyal veriyor. Teknik tablo temkinli olmayı gerektiriyor."
                )
            else:
                p2_parts.append(
                    "Genel olarak karışık bir tablo var, net bir yön belirlemek zor. "
                    "Böyle dönemlerde sabırlı olmak ve sert hareketlere hazırlıklı olmak en mantıklısı."
                )

    paragraph_2 = " ".join(p2_parts)

    return [paragraph_1, paragraph_2]


def _build_narrative_en(
    name, close, rsi, trend, momentum, macd_event,
    bb_squeeze, near_support, near_resistance, support_val,
    resistance_val, signals_mixed, summary, sma_50, sma_200, stoch_k,
    ext="",
) -> list[str]:
    """Build English conversational narrative with optional external context."""
    has_ext = bool(ext)

    p1_parts = []

    if has_ext:
        p1_parts.append("Considering your additional context:")

    if trend == "strong_up":
        if has_ext:
            p1_parts.append(
                f"Price is at {close:,.2f}, technically in a strong uptrend above both major averages. "
                f"But given your context, the real question is whether this trend is organic or artificial. "
                f"If price is being shaped by external intervention rather than supply-demand, "
                f"the technical 'strength' could be misleading."
            )
        else:
            p1_parts.append(
                f"Looking at {name} right now, I see a solid uptrend. "
                f"Price is at {close:,.2f}, trading above both the 50 and 200-day moving averages — "
                f"that's a technically strong structure."
            )
    elif trend == "strong_down":
        if has_ext:
            p1_parts.append(
                f"Price is at {close:,.2f}, below both major averages — technically in a deep downtrend. "
                f"Given your context, distinguishing whether this decline is market-driven or externally "
                f"forced is critical. If external pressure is the cause, technical recovery signals may be unreliable."
            )
        else:
            p1_parts.append(
                f"Honestly, {name} doesn't look great right now. "
                f"Price is at {close:,.2f}, below both the 50 and 200-day averages — "
                f"we're in classic 'death cross' territory. Bears are in control."
            )
    elif trend == "up":
        if has_ext:
            p1_parts.append(
                f"{name} is at {close:,.2f} with a positive bias technically. "
                f"But how much of this rise is genuine market movement vs. your mentioned factor? "
                f"That distinction matters a lot."
            )
        else:
            p1_parts.append(
                f"{name} is at {close:,.2f} with a generally positive bias. "
                f"The 50-day average is above the 200-day, which is an encouraging sign."
            )
    elif trend == "down":
        if has_ext:
            p1_parts.append(
                f"{name} is at {close:,.2f} with short-term averages below long-term. "
                f"Combined with your context, this weakness may run deeper than charts suggest."
            )
        else:
            p1_parts.append(
                f"{name} is at {close:,.2f} and short-term averages have dipped below long-term ones. "
                f"The overall picture is somewhat concerning."
            )
    else:
        p1_parts.append(f"{name} is currently trading at {close:,.2f}.")

    if momentum == "overbought":
        if has_ext:
            p1_parts.append(
                f"RSI at {rsi:.0f} screams overbought. But in a controlled market, "
                f"overbought readings can be deceptive — the natural pullback mechanism may be suppressed, "
                f"meaning when the correction finally comes, it hits much harder."
            )
        else:
            p1_parts.append(
                f"RSI is at {rsi:.0f}, deep in overbought territory — "
                f"the market may be running a bit hot and could need a breather."
            )
    elif momentum == "oversold":
        if has_ext:
            p1_parts.append(
                f"RSI at {rsi:.0f} is in oversold territory. Normally a bounce signal, "
                f"but given your context, 'oversold' becomes relative — "
                f"under sustained pressure, indicators can stay extreme for extended periods."
            )
        else:
            p1_parts.append(
                f"RSI has dropped to {rsi:.0f}, well into oversold territory. "
                f"Sellers may be exhausted; a bounce from here wouldn't be surprising."
            )
    elif momentum == "bullish":
        if has_ext:
            p1_parts.append(
                f"RSI at {rsi:.0f} suggests positive momentum, but is it organic buying "
                f"or artificial support? That's the key question here."
            )
        else:
            p1_parts.append(f"RSI at {rsi:.0f} shows positive momentum continuing.")
    elif momentum == "bearish":
        if has_ext:
            p1_parts.append(
                f"RSI at {rsi:.0f} shows fading momentum. Given your context, "
                f"the question is whether this weakness is structural or temporary."
            )
        else:
            p1_parts.append(f"RSI at {rsi:.0f} suggests momentum is fading.")

    if macd_event == "bullish_cross":
        if has_ext:
            p1_parts.append(
                "MACD shows a bullish crossover, but in an intervened market, "
                "MACD crossovers frequently produce false signals — approach with caution."
            )
        else:
            p1_parts.append(
                "On top of that, MACD just had a bullish crossover — "
                "a fresh signal of potential momentum shift."
            )
    elif macd_event == "bearish_cross":
        if has_ext:
            p1_parts.append(
                "MACD crossed bearish. In this context, it could signal that intervention "
                "is weakening or the market is trying to find its own path despite external pressure."
            )
        else:
            p1_parts.append(
                "Adding to that, MACD just crossed bearish, "
                "confirming fresh downward pressure."
            )

    paragraph_1 = " ".join(p1_parts)

    p2_parts = []

    if bb_squeeze:
        if has_ext:
            p2_parts.append(
                "Bollinger Bands are extremely compressed. Normally a harbinger of a big move, "
                "but here volatility may be artificially suppressed. "
                "When the pressure lifts, the move could be far more violent."
            )
        else:
            p2_parts.append(
                "What really catches my eye is the Bollinger Band squeeze. "
                "Bands are tightly compressed, which historically precedes a major move. "
                "Direction is uncertain, but a volatility explosion is coming."
            )
    elif signals_mixed:
        if has_ext:
            p2_parts.append(
                f"Indicators are conflicting: {summary['buy']} buy, {summary['sell']} sell. "
                f"This confusion may be a direct reflection of the external factor you mentioned — "
                f"the market can't find its natural balance because an outside force is interfering. "
                f"The real risk is which direction it breaks when intervention loosens."
            )
        else:
            p2_parts.append(
                f"Here's the thing — indicators are conflicting: "
                f"{summary['buy']} say buy, {summary['sell']} say sell. "
                f"In these indecisive periods, markets tend to chop sideways before breaking sharply."
            )
    elif momentum == "overbought" and trend == "strong_up":
        if has_ext:
            p2_parts.append(
                "Trend is strong and indicators are overbought — normally a correction is due. "
                "But the correction mechanism may not function if price is externally supported. "
                "The real danger isn't when the correction comes — it's the delayed correction, "
                "which tends to be far more destructive."
            )
        else:
            p2_parts.append(
                "The interesting tension here is that the trend is strong but indicators scream overbought. "
                "Either a sharp correction comes, or price consolidates at these elevated levels. "
                "Worth being prepared for both scenarios."
            )

    if near_support and support_val is not None:
        if has_ext:
            p2_parts.append(
                f"Price is near the {support_val:,.2f} support. But is this support formed by "
                f"real buyers or by intervention? Artificial support, when broken, tends to "
                f"result in much sharper drops."
            )
        else:
            p2_parts.append(
                f"Also worth noting: price is very close to the {support_val:,.2f} support level. "
                f"A bounce is possible, but a break below could trigger a swift decline — we're at a critical threshold."
            )
    elif near_resistance and resistance_val is not None:
        if has_ext:
            p2_parts.append(
                f"The {resistance_val:,.2f} resistance is overhead. In this context, "
                f"a breakout could also be artificial — volume and follow-through "
                f"over the next few sessions will tell the real story."
            )
        else:
            p2_parts.append(
                f"Don't miss this: the {resistance_val:,.2f} resistance is right overhead. "
                f"A breakout could fuel more upside, but rejection here means a pullback."
            )

    if not p2_parts:
        total = summary['buy'] + summary['sell'] + summary['neutral']
        if has_ext:
            p2_parts.append(
                f"The indicator scorecard ({summary['buy']} buy, {summary['sell']} sell, "
                f"{summary['neutral']} neutral) would normally be meaningful, but these signals should be "
                f"read with a discount. Still, don't discard them entirely — when pressure eventually eases, "
                f"these same indicators will give the first clues."
            )
        else:
            if summary["overall"] == "buy":
                p2_parts.append(
                    f"Overall, the majority of indicators ({summary['buy']}/{total}) "
                    f"lean positive. The technical picture looks constructive, but always stay open to surprises."
                )
            elif summary["overall"] == "sell":
                p2_parts.append(
                    f"The majority of indicators ({summary['sell']}/{total}) "
                    f"are signaling negative. The technical picture warrants caution."
                )
            else:
                p2_parts.append(
                    "Overall it's a mixed bag — hard to call a clear direction. "
                    "In times like these, patience is key, and being ready for sharp moves in either direction is the smart play."
                )

    paragraph_2 = " ".join(p2_parts)

    return [paragraph_1, paragraph_2]
