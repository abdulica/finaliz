"""LLM Engine — OpenRouter üzerinden StepFun step-3.5-flash entegrasyonu.

Finaliz projesine yapay zeka destekli analiz ve sohbet kapasitesi kazandırır.
Mevcut kural tabanlı teknik analiz motorunu silmez; onun üzerine LLM katmanı ekler.

Kullanım:
    from analysis.llm_engine import build_context, ask_llm, stream_llm
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# OpenRouter istemcisi
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "stepfun/step-3.5-flash:free"


def _get_client() -> OpenAI | None:
    """OpenRouter istemcisini oluşturur. API key yoksa None döner."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_openrouter_api_key_here":
        return None
    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/finaliz",
            "X-Title": "Finaliz Financial Analysis",
        },
    )

# ---------------------------------------------------------------------------
# Bağlam oluşturucu — LLM'e gönderilecek piyasa özetini hazırlar
# ---------------------------------------------------------------------------

def build_context(
    df: pd.DataFrame,
    asset_name: str,
    asset_key: str,
    lang: str = "tr",
    extra_context: str = "",
) -> str:
    """Teknik indikatörlerden kısa özet üretir — token tasarrufu için kompakt format."""
    if df is None or df.empty or len(df) < 10:
        return ""

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    close = last.get("Close", 0)
    prev_close = prev.get("Close", close)
    daily_pct = (close - prev_close) / prev_close * 100 if prev_close else 0

    rsi = _v(last, "RSI")
    macd_hist = _v(last, "MACD_Hist")
    bb_upper = _v(last, "BB_Upper")
    bb_lower = _v(last, "BB_Lower")
    bb_width = _v(last, "BB_Width")
    sma_50 = _v(last, "SMA_50")
    sma_200 = _v(last, "SMA_200")
    atr = _v(last, "ATR")

    perf_5d = _perf(df, 5)
    perf_20d = _perf(df, 20)
    trend = _trend(close, sma_50, sma_200)
    today = datetime.now().strftime("%Y-%m-%d")

    lines = [f"[{asset_name} | {today}]"]
    lines.append(f"Fiyat: {close:,.4f} ({daily_pct:+.2f}%)" if lang == "tr" else f"Price: {close:,.4f} ({daily_pct:+.2f}%)")
    if perf_5d is not None and perf_20d is not None:
        lines.append(f"5g: {perf_5d:+.1f}% | 20g: {perf_20d:+.1f}%" if lang == "tr" else f"5d: {perf_5d:+.1f}% | 20d: {perf_20d:+.1f}%")
    if rsi:
        rsi_note = "aşırı alım" if rsi > 70 else ("aşırı satım" if rsi < 30 else "nötr")
        rsi_note_en = "overbought" if rsi > 70 else ("oversold" if rsi < 30 else "neutral")
        lines.append(f"RSI: {rsi:.0f} ({rsi_note if lang == 'tr' else rsi_note_en})")
    if macd_hist:
        lines.append(f"MACD hist: {macd_hist:+.4f} ({'↑' if macd_hist > 0 else '↓'})")
    if bb_upper and bb_lower:
        pos = "üst bant üstünde" if close > bb_upper else ("alt bant altında" if close < bb_lower else "bantlar içinde")
        pos_en = "above upper band" if close > bb_upper else ("below lower band" if close < bb_lower else "inside bands")
        lines.append(f"BB: {pos if lang == 'tr' else pos_en}" + (f" | sıkışma" if bb_width and bb_width < 0.02 else ""))
    if sma_50 and sma_200:
        lines.append(f"Trend: {trend}")
    if atr:
        lines.append(f"ATR: {atr:,.4f}")

    context = "\n".join(lines)
    if extra_context:
        context += f"\n{'Soru' if lang == 'tr' else 'Question'}: {extra_context}"
    return context


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_TR = """Sen Finaliz finansal analiz platformunun yapay zeka asistanısın.

Görevin:
- Sağlanan gerçek zamanlı teknik indikatör verilerini analiz etmek
- Piyasa dinamiklerini, senaryoları ve risk faktörlerini netçe açıklamak
- "Düşünce arkadaşı" gibi davranmak: ufuk açıcı, nüanslı, dürüst

Kurallar:
- KESİNLİKLE yatırım tavsiyesi verme. "Al", "sat", "yatır" gibi direktifler kullanma.
- Bunun yerine gözlemsel dil kullan: "teknik tablo şunu gösteriyor", "bu senaryo gerçekleşirse..."
- Verilen teknik veriyle çelişen iddialarda bulunma.
- Türkçe yaz. RSI, MACD, Bollinger gibi teknik terimleri orijinal haliyle kullan.
- Yapılandır: kısa özet → detaylı analiz → senaryo listesi (gerekirse).
- Her yanıtın sonuna ekle: "Bu analiz bilgilendirme amaçlıdır, yatırım tavsiyesi değildir." """

_SYSTEM_EN = """You are the AI assistant of Finaliz financial analysis platform.

Your role:
- Analyze real-time technical indicator data provided to you
- Clearly explain market dynamics, scenarios, and risk factors
- Act as a "thinking partner": insightful, nuanced, honest

Rules:
- NEVER give investment advice. No "buy", "sell", "invest" directives.
- Use observational language: "the technical picture shows...", "if this scenario plays out..."
- Do not contradict the provided technical data.
- Write in English. Keep technical terms (RSI, MACD, Bollinger, etc.) as-is.
- Structure: brief summary → detailed analysis → scenarios (if relevant).
- End every response with: "This analysis is for informational purposes only, not investment advice." """


def _system_prompt(lang: str) -> str:
    return _SYSTEM_TR if lang == "tr" else _SYSTEM_EN


# ---------------------------------------------------------------------------
# Streaming LLM çağrısı
# ---------------------------------------------------------------------------

def stream_llm(
    user_input: str,
    df: pd.DataFrame,
    asset_name: str,
    asset_key: str,
    lang: str = "tr",
    model: str = DEFAULT_MODEL,
    conversation_history: list | None = None,
):
    """LLM'e streaming isteği gönderir. st.write_stream() ile kullanılır.

    Yields:
        str — token parçacıkları.
    """
    client = _get_client()
    if client is None:
        msg = (
            "⚠️ OpenRouter API anahtarı bulunamadı. `.env` dosyasına `OPENROUTER_API_KEY` ekleyin."
            if lang == "tr"
            else "⚠️ OpenRouter API key not found. Add `OPENROUTER_API_KEY` to your `.env` file."
        )
        yield msg
        return

    context = build_context(df, asset_name, asset_key, lang, extra_context=user_input)

    messages = [{"role": "system", "content": _system_prompt(lang)}]

    if conversation_history:
        messages.extend(conversation_history[-12:])

    messages.append({"role": "user", "content": context})

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=1500,
            temperature=0.3,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    except Exception as e:
        err = str(e)
        if "401" in err or "authentication" in err.lower():
            yield "⚠️ API anahtarı geçersiz. OpenRouter API key'inizi kontrol edin." if lang == "tr" else "⚠️ Invalid API key."
        elif "429" in err:
            yield "⚠️ Rate limit aşıldı. Birkaç saniye bekleyip tekrar deneyin." if lang == "tr" else "⚠️ Rate limit exceeded. Please wait."
        else:
            yield f"⚠️ LLM hatası: {err}" if lang == "tr" else f"⚠️ LLM error: {err}"


def ask_llm(
    user_input: str,
    df: pd.DataFrame,
    asset_name: str,
    asset_key: str,
    lang: str = "tr",
    model: str = DEFAULT_MODEL,
) -> str:
    """Streaming olmayan tek seferlik LLM çağrısı (test için)."""
    return "".join(stream_llm(user_input, df, asset_name, asset_key, lang, model))


def is_llm_available() -> bool:
    """API key varlığını kontrol eder, istek atmaz."""
    key = os.getenv("OPENROUTER_API_KEY", "")
    return bool(key and key != "your_openrouter_api_key_here")


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def _v(row, col: str):
    """DataFrame satırından güvenli değer okur, NaN'ı None'a çevirir."""
    val = row.get(col)
    if val is None:
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _perf(df: pd.DataFrame, days: int) -> float | None:
    """N günlük performans yüzdesi hesaplar."""
    if len(df) <= days:
        return None
    now = df["Close"].iloc[-1]
    past = df["Close"].iloc[-(days + 1)]
    return None if past == 0 else (now - past) / past * 100


def _trend(close: float, sma_50, sma_200) -> str:
    """Trend durumunu metin olarak döner."""
    if sma_50 is None or sma_200 is None:
        return "belirsiz"
    if sma_50 > sma_200 and close > sma_50:
        return "güçlü yükseliş (SMA50 > SMA200, fiyat üstünde)"
    elif sma_50 > sma_200:
        return "yükseliş eğilimi (SMA50 > SMA200)"
    elif sma_50 < sma_200 and close < sma_50:
        return "güçlü düşüş (SMA50 < SMA200, fiyat altında)"
    else:
        return "düşüş eğilimi (SMA50 < SMA200)"
