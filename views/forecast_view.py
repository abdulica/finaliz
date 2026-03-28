"""Forecast page — Prophet tahminleri + AI destekli finansal chat asistanı."""

import streamlit as st
import pandas as pd
import numpy as np

from config import ASSETS
from analysis.technical import compute_all_indicators, get_signal_summary
from analysis.forecast import run_forecast, generate_forecast_commentary
from analysis.llm_engine import stream_llm, is_llm_available, build_context
from components.charts import create_forecast_chart, create_relationship_chart
from data.windowing import window_for_forecast, window_for_technical
from components.analysis_card import render_forecast_table, render_disclaimer
from utils.i18n import t, get_asset_name


# ---------------------------------------------------------------------------
# Ana sayfa
# ---------------------------------------------------------------------------

def render_forecast(
    data: dict[str, pd.DataFrame],
    asset_key: str,
    lang: str = "tr",
):
    st.header(f"🔮 {t('fc_title', lang)}")

    asset_names = {get_asset_name(k, v, lang): k for k, v in ASSETS.items() if k in data}
    asset_labels = list(asset_names.keys())

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        label_1 = "Birinci Varlık" if lang == "tr" else "First Asset"
        default_idx_1 = next((i for i, l in enumerate(asset_labels) if asset_names[l] == asset_key), 0)
        selected_1 = st.selectbox(label_1, asset_labels, index=default_idx_1, key="fc_asset_1")
        key_1 = asset_names[selected_1]

    with col_sel2:
        label_2 = "İkinci Varlık (İlişki Analizi)" if lang == "tr" else "Second Asset (Relationship)"
        default_idx_2 = min(1 if default_idx_1 != 1 else 2, len(asset_labels) - 1)
        selected_2 = st.selectbox(label_2, asset_labels, index=default_idx_2, key="fc_asset_2")
        key_2 = asset_names[selected_2]

    st.markdown("---")

    # --- Tahmin grafikleri ---
    _render_single_forecast(data, key_1, lang)
    st.markdown("---")
    if key_2 != key_1:
        _render_single_forecast(data, key_2, lang)
        st.markdown("---")

    # --- İlişki analizi ---
    df_1 = data.get(key_1)
    df_2 = data.get(key_2)
    if key_1 != key_2 and df_1 is not None and df_2 is not None:
        name_1 = get_asset_name(key_1, ASSETS[key_1], lang)
        name_2 = get_asset_name(key_2, ASSETS[key_2], lang)
        st.subheader(f"🔗 {name_1} ↔ {name_2} " + ("İlişki Analizi" if lang == "tr" else "Relationship"))
        fig = create_relationship_chart(df_1, df_2, key_1, key_2, lang)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")

    # --- AI Chat ---
    _render_ai_chat(data, key_1, key_2, lang)

    render_disclaimer(lang)


# ---------------------------------------------------------------------------
# Tahmin bölümü
# ---------------------------------------------------------------------------

def _render_single_forecast(data, asset_key, lang):
    asset_config = ASSETS.get(asset_key)
    df_full = data.get(asset_key)
    if df_full is None or df_full.empty:
        return

    name = get_asset_name(asset_key, asset_config, lang)
    st.subheader(f"🔮 {name}")

    df_forecast = window_for_forecast(df_full)
    with st.spinner(f"{name} tahmin hesaplanıyor..." if lang == "tr" else f"Calculating forecast for {name}..."):
        forecast_results = run_forecast(df_forecast)

    if forecast_results is None:
        st.warning(f"{name}: " + ("Tahmin modeli çalıştırılamadı." if lang == "tr" else "Could not run forecast model."))
        return

    render_forecast_table(forecast_results, lang)

    full_forecast = forecast_results.get("_full_forecast")
    actual_data = forecast_results.get("_actual_data")
    if full_forecast is not None and actual_data is not None:
        fig = create_forecast_chart(actual_data, full_forecast, asset_key, lang)
        st.plotly_chart(fig, use_container_width=True)

    df_ta = window_for_technical(df_full)
    df_with_ta = compute_all_indicators(df_ta.copy())
    ta_summary = get_signal_summary(df_with_ta)

    fc_comments = generate_forecast_commentary(forecast_results, name, ta_summary["overall"], lang)
    if fc_comments:
        for c in fc_comments:
            st.caption(c)


# ---------------------------------------------------------------------------
# AI Chat asistanı
# ---------------------------------------------------------------------------

# Sadece finans/ekonomi ile ilgili konular
_FINANCE_SYSTEM_TR = """Sen kıdemli bir piyasa analistisin. Görevin: koşullara bağlı, katmanlı, eğitici ve dürüst finansal analiz yapmak.

KONU KISITLAMASI:
- Yalnızca ekonomi, finans, piyasalar, teknik analiz, makroekonomik göstergeler, varlık sınıfları hakkında konuş.
- Konu dışı sorulara: "Bu konuda yardımcı olamam, finans/ekonomi dışında. Piyasalar hakkında soru sormak ister misin?" de.

NASIL DÜŞÜN VE YAZAR:
1. Önce "Bu varlık için hangi faktör baskın?" sorusunu sor kendine.
   - Teknik mi? Makro mu? Kurumsal müdahale mi? Jeopolitik mi? Mevsimsel mi?
   - Baskın faktörü belirle, diğerlerini ikincil yaz.

2. Eğitici ama teknik ol:
   - "RSI 89 aşırı alım gösteriyor" yerine: "RSI 89 — tarihsel olarak bu seviyedeki varlıklar %60-70 ihtimalle 1-2 hafta içinde düzeltmeye giriyor. Ama USDTRY bu istatistiğin geçerli olmadığı nadir varlıklardan biri, çünkü..."
   - Nedeni açıkla. Sadece ne olduğunu değil, neden olduğunu yaz.

3. Senaryo tabanlı tahmin yap:
   - "Tahmin edilemez" deme. Bunun yerine: "Şu an 3 senaryo var:"
   - Her senaryoya olasılık ver (kesin değil, yaklaşık: "daha olası / daha az olası").
   - Hangi gösterge veya haber senaryoyu değiştirir, onu söyle.

4. USDTRY, TCMB, SWAP gibi kurumsal müdahale konularında:
   - Teknik sinyallerin neden devre dışı kaldığını açıkla (mekanizma düzeyinde).
   - TCMB'nin geçmiş hamleleriyle kıyasla.
   - Makro bağlamı ver: rezerv durumu, faiz-enflasyon makası, sermaye akışı.

5. Yanıt uzunluğu — ÇOK ÖNEMLİ:
   - Maksimum 400-500 kelime yaz. Uzun listeler yapma.
   - Bullet point yerine kısa paragraflar tercih et.
   - Senaryo sayısını 3 ile sınırla, her senaryo 2-3 cümle.
   - "İzlenecek göstergeler" bölümünü en fazla 3 maddeyle bitir.
   - Kullanıcı "devam et" veya "daha fazla detay" derse genişlet.

6. Dil:
   - Türkçe yaz. RSI, MACD, SWAP, TCMB gibi terimleri orijinal haliyle kullan.
   - "Yatırım tavsiyesi değildir" notu ekle ama bunu mazeret olarak kullanma — analizden kaçınmak için değil, sorumluluk reddi için.
   - "Tahmin edilemez", "belirsiz", "dikkatli olun" gibi boş dolgu cümleleri yazma. Bunların yerine somut koşul yaz: "Eğer X olursa Y, eğer Z olursa W."

Sana canlı piyasa verisi ve teknik indikatörler verilecek. Bu verilerle konuş — veriye dayanmayan genel laflar etme."""

_FINANCE_SYSTEM_EN = """You are a senior market analyst. Your job: layered, educational, condition-based financial analysis.

TOPIC RESTRICTION:
- Only discuss economics, finance, markets, technical analysis, macroeconomic indicators, asset classes.
- Off-topic: "I can only help with finance/economics. Would you like to ask about markets?"

HOW TO THINK AND WRITE:
1. First ask yourself: "Which factor dominates this asset right now?"
   - Technical? Macro? Institutional intervention? Geopolitical? Seasonal?
   - Name the dominant factor, list others as secondary.

2. Be educational but technical:
   - Not: "RSI 89 shows overbought." Instead: "RSI 89 — historically, assets at this level correct within 1-2 weeks ~60-70% of the time. But USDTRY is one of the rare assets where this statistic breaks down, because..."
   - Explain the why, not just the what.

3. Give scenario-based forecasts:
   - Never say "unpredictable." Instead: "There are 3 scenarios right now:"
   - Assign rough probabilities (not precise — "more likely / less likely").
   - State which indicator or news event would change the scenario.

4. On institutional intervention (TCMB, SWAP, FED, etc.):
   - Explain why technical signals are overridden (at the mechanism level).
   - Compare to past interventions.
   - Give macro context: reserve levels, rate-inflation spread, capital flows.

5. Response length — IMPORTANT:
   - Maximum 400-500 words. No long bullet lists.
   - Prefer short paragraphs over bullet points.
   - Limit scenarios to 3, each 2-3 sentences.
   - "What to watch" section: max 3 items.
   - Expand only if user asks "continue" or "more detail".

6. Language:
   - Write in English. Keep terms like RSI, MACD, SWAP, TCMB as-is.
   - Add "not investment advice" note but don't use it as an excuse to avoid analysis.
   - Never write empty filler phrases like "it's uncertain," "be careful," "unpredictable." Replace with concrete conditions: "If X then Y, if Z then W."

You will receive live market data and technical indicators. Base your analysis on this data — no generic statements."""


def _render_ai_chat(data: dict, key_1: str, key_2: str, lang: str):
    """Sabit yükseklikli chat container + token tasarruflu bağlam yönetimi."""

    st.subheader("🤖 " + ("AI Finans Asistanı" if lang == "tr" else "AI Finance Assistant"))

    if not is_llm_available():
        st.info(
            "OpenRouter API anahtarı bulunamadı. `.env` dosyasına `OPENROUTER_API_KEY` ekleyin."
            if lang == "tr"
            else "OpenRouter API key not found. Add `OPENROUTER_API_KEY` to your `.env` file."
        )
        return

    st.caption(
        "💡 Piyasa verilerini görüyorum. Tahminler, teknik analiz, makro faktörler hakkında sor."
        if lang == "tr" else
        "💡 I have live market data. Ask about forecasts, technical analysis, macro factors."
    )

    # Chat kutusunu belirgin kıl
    st.markdown("""
        <style>
        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid rgba(128,128,128,0.3) !important;
            border-radius: 8px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    hist_key = "fc_chat_history"
    ctx_key = "fc_chat_context"

    if hist_key not in st.session_state:
        st.session_state[hist_key] = []

    # Bağlamı sadece bir kez hazırla ve sakla (yeniden render'da yeniden hesaplama)
    asset_sig = f"{key_1}_{key_2}"
    if st.session_state.get("fc_ctx_sig") != asset_sig:
        ctx_parts = []
        for ak in ([key_1] + ([key_2] if key_2 != key_1 else [])):
            df_full = data.get(ak)
            if df_full is not None and not df_full.empty:
                df_ta = window_for_technical(df_full)
                df_ta = compute_all_indicators(df_ta.copy())
                name = get_asset_name(ak, ASSETS[ak], lang)
                ctx_parts.append(build_context(df_ta, name, ak, lang))
        st.session_state[ctx_key] = "\n\n".join(ctx_parts)
        st.session_state["fc_ctx_sig"] = asset_sig

    # Temizle butonu — sağ üst köşe
    col_title, col_clear = st.columns([6, 1])
    with col_clear:
        if st.button("🗑️", key="fc_clear", help="Sohbeti temizle" if lang == "tr" else "Clear chat"):
            st.session_state[hist_key] = []
            st.rerun()

    # Sabit yükseklikli mesaj alanı — sayfa uzamaz
    chat_container = st.container(height=420)
    with chat_container:
        if not st.session_state[hist_key]:
            st.markdown(
                "<div style='color:gray;text-align:center;padding:20px;'>"
                + ("Henüz mesaj yok. Aşağıdan soru sor." if lang == "tr" else "No messages yet. Ask below.")
                + "</div>",
                unsafe_allow_html=True,
            )
        for msg in st.session_state[hist_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Chat input — container dışında kalır, her zaman görünür
    placeholder = (
        "Soru yaz... (ör: 'Altın neden düşüyor?', 'BTC için RSI ne söylüyor?')"
        if lang == "tr"
        else "Ask... (e.g. 'Why is gold falling?', 'What does RSI say for BTC?')"
    )
    user_msg = st.chat_input(placeholder, key="fc_chat_input")

    if user_msg:
        st.session_state[hist_key].append({"role": "user", "content": user_msg})

        # Sistem prompt: piyasa verisini her seferinde DEĞİL,
        # sadece bağlam değiştiğinde (yeni varlık / ilk mesaj) ekle
        context_data = st.session_state.get(ctx_key, "")
        system_content = (
            (_FINANCE_SYSTEM_TR if lang == "tr" else _FINANCE_SYSTEM_EN)
            + "\n\n=== CANLI PİYASA VERİSİ ===\n"
            + context_data
        )

        # Mesaj listesi: system + son 6 tur (12 mesaj) — daha az token
        messages = [{"role": "system", "content": system_content}]
        messages.extend(st.session_state[hist_key][-12:])

        # Yanıtı container içinde göster
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_msg)
            with st.chat_message("assistant"):
                response_chunks = []
                response_placeholder = st.empty()
                full_response = ""
                for chunk in _stream_chat(messages, lang):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                response = full_response

        st.session_state[hist_key].append({"role": "assistant", "content": response})

        # Yanıt token limitine takıldıysa devam et butonu göster
        # (son karakter noktalama değilse muhtemelen kesilmiştir)
        last_char = response.strip()[-1] if response.strip() else ""
        if last_char not in ".?!…\"'":
            st.session_state["fc_truncated"] = True
        else:
            st.session_state["fc_truncated"] = False

    # Devam et butonu — yanıt kesilmişse göster
    if st.session_state.get("fc_truncated"):
        st.caption("⚠️ Yanıt kısaltıldı." if lang == "tr" else "⚠️ Response was cut off.")
        if st.button("▶ Devam et" if lang == "tr" else "▶ Continue", key="fc_continue"):
            st.session_state[hist_key].append({"role": "user", "content": "devam et" if lang == "tr" else "continue"})
            st.session_state["fc_truncated"] = False
            st.rerun()


def _stream_chat(messages: list, lang: str):
    """Hazırlanmış mesaj listesiyle direkt OpenRouter'a istek atar."""
    from openai import OpenAI
    import os
    from dotenv import load_dotenv
    load_dotenv()

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/finaliz",
            "X-Title": "Finaliz Financial Analysis",
        },
    )

    try:
        stream = client.chat.completions.create(
            model="stepfun/step-3.5-flash:free",
            messages=messages,
            stream=True,
            max_tokens=3000,
            temperature=0.4,
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
            except (IndexError, AttributeError):
                continue
    except Exception as e:
        err = str(e)
        if "401" in err:
            yield "⚠️ API anahtarı geçersiz." if lang == "tr" else "⚠️ Invalid API key."
        elif "429" in err:
            yield "⚠️ Çok sık istek. 10-15 saniye bekleyip tekrar dene." if lang == "tr" else "⚠️ Too many requests. Wait 10-15 seconds and try again."
        elif "context" in err.lower() or "token" in err.lower():
            yield "⚠️ Sohbet çok uzadı. 🗑️ Temizle butonuna basıp yeni sohbet başlat." if lang == "tr" else "⚠️ Conversation too long. Press 🗑️ Clear to start fresh."
        else:
            yield f"⚠️ Hata: {err[:200]}"


# ---------------------------------------------------------------------------
# İlişki analizi (eski koddan korundu)
# ---------------------------------------------------------------------------

def _generate_relationship_commentary(df_a, df_b, key_a, key_b, lang):
    comments = []
    name_a = get_asset_name(key_a, ASSETS[key_a], lang)
    name_b = get_asset_name(key_b, ASSETS[key_b], lang)
    common_idx = df_a.index.intersection(df_b.index)
    if len(common_idx) < 30:
        return comments
    ret_a = df_a.loc[common_idx, "Close"].pct_change().dropna()
    ret_b = df_b.loc[common_idx, "Close"].pct_change().dropna()
    overall_corr = ret_a.corr(ret_b)
    recent_corr = ret_a.tail(30).corr(ret_b.tail(30))
    vol_a = ret_a.std() * np.sqrt(252) * 100
    vol_b = ret_b.std() * np.sqrt(252) * 100
    if lang == "tr":
        if abs(overall_corr) > 0.7:
            d = "pozitif" if overall_corr > 0 else "negatif"
            comments.append(f"Güçlü {d} korelasyon: {overall_corr:.2f}")
        elif abs(overall_corr) > 0.3:
            d = "pozitif" if overall_corr > 0 else "negatif"
            comments.append(f"Orta {d} korelasyon: {overall_corr:.2f}")
        else:
            comments.append(f"Anlamlı korelasyon yok: {overall_corr:.2f}")
        if abs(recent_corr - overall_corr) > 0.25:
            comments.append(f"Son 30g korelasyon ({recent_corr:.2f}) tarihsel ortalamayla ({overall_corr:.2f}) ayrışıyor.")
        comments.append(f"Yıllık volatilite: {name_a} %{vol_a:.1f} | {name_b} %{vol_b:.1f}")
    else:
        if abs(overall_corr) > 0.7:
            d = "positive" if overall_corr > 0 else "negative"
            comments.append(f"Strong {d} correlation: {overall_corr:.2f}")
        elif abs(overall_corr) > 0.3:
            d = "positive" if overall_corr > 0 else "negative"
            comments.append(f"Moderate {d} correlation: {overall_corr:.2f}")
        else:
            comments.append(f"No significant correlation: {overall_corr:.2f}")
        if abs(recent_corr - overall_corr) > 0.25:
            comments.append(f"Recent 30d correlation ({recent_corr:.2f}) diverging from historical ({overall_corr:.2f}).")
        comments.append(f"Annualized volatility: {name_a} {vol_a:.1f}% | {name_b} {vol_b:.1f}%")
    return comments
