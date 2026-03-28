"""Finaliz - Financial Analysis and Forecasting Platform."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
import re
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

from components.sidebar import render_sidebar
from data.cache import get_cached, set_cached, get_cache_timestamp
from data.fetcher import fetch_all_assets, fetch_macro_data
from utils.i18n import t
from views.dashboard import render_dashboard
from views.asset_detail import render_asset_detail
from views.forecast_view import render_forecast
from views.comparison import render_comparison
from views.macro_overview import render_macro_overview
from components.tradingview import tradingview_ticker_tape

load_dotenv()

st.set_page_config(
    page_title="Finaliz",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.stApp { background-color: #0e1117; }
section[data-testid="stSidebar"] { width: 200px !important; min-width: 200px !important; background: #13131f; }
section[data-testid="stSidebar"] > div { padding: 0.5rem 0.75rem; }
.main .block-container { padding-bottom: 240px !important; padding-top: 1rem !important; max-width: 100% !important; }
.stMetric { background-color: #1e1e2e; padding: 10px; border-radius: 8px; }
div[data-testid="stMetricValue"] { font-size: 1.2rem; }
iframe { border: none !important; }
@media (max-width: 768px) {
    section[data-testid="stSidebar"] { width: 0px !important; min-width: 0px !important; }
    .main .block-container { padding-bottom: 280px !important; }
}
</style>
""", unsafe_allow_html=True)


def load_data(force_refresh=False):
    cached_asset = get_cached("assets_5y")
    cached_macro = get_cached("macro")
    if cached_asset is not None and cached_macro is not None and not force_refresh:
        return cached_asset, cached_macro
    with st.spinner(t("loading", st.session_state.get("lang", "tr"))):
        asset_data = fetch_all_assets("5y")
        macro_data = fetch_macro_data(period_years=5)
        set_cached("assets_5y", asset_data)
        set_cached("macro", macro_data)
    return asset_data, macro_data


def _get_client():
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key or key == "your_openrouter_api_key_here":
        return None
    return OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "finaliz", "X-Title": "Finaliz"},
    )

def _page_name(page, lang):
    names = {
        "dashboard": ("Genel Bakış", "Dashboard"),
        "asset_detail": ("Varlık Detay", "Asset Detail"),
        "forecast": ("Tahmin", "Forecast"),
        "comparison": ("Karşılaştırma", "Comparison"),
        "macro": ("Makro Göstergeler", "Macro"),
    }
    pair = names.get(page, (page, page))
    return pair[0] if lang == "tr" else pair[1]


def _build_page_context(asset_data, page, lang):
    from analysis.llm_engine import build_context
    from analysis.technical import compute_all_indicators
    from data.windowing import window_for_technical
    from utils.i18n import get_asset_name
    from config import ASSETS
    page_assets = {
        "dashboard": list(asset_data.keys())[:4],
        "asset_detail": ["DXY", "USDTRY", "GOLD"],
        "forecast": ["DXY", "BTC"],
        "comparison": list(asset_data.keys())[:3],
        "macro": ["DXY", "GOLD", "OIL"],
    }
    parts = []
    for ak in page_assets.get(page, list(asset_data.keys())[:3]):
        df_full = asset_data.get(ak)
        if df_full is None or df_full.empty:
            continue
        df_ta = window_for_technical(df_full)
        df_ta = compute_all_indicators(df_ta.copy())
        name = get_asset_name(ak, ASSETS[ak], lang)
        parts.append(build_context(df_ta, name, ak, lang))
    return "\n\n".join(parts)


def _build_system(lang, ctx=""):
    base = (
        "Sen Finaliz finansal analiz asistanısın. Yalnızca ekonomi ve finans hakkında "
        "kısa, özlü, veri odaklı yanıtlar ver. Yatırım tavsiyesi verme. "
        "Güncel haber sorusunda eğer haber verisi sağlandıysa onu kullan; "
        "sağlanmadıysa 'Güncel haberler için Reuters/Bloomberg takip edin' de ve "
        "elindeki teknik veriden ne söylenebileceğini anlat."
        if lang == "tr" else
        "You are Finaliz financial analysis assistant. Give short, concise, data-driven "
        "answers only on economics and finance. No investment advice. "
        "For news questions, use provided news data if available; "
        "otherwise say 'Follow Reuters/Bloomberg for live news' and analyze with available technical data."
    )
    if ctx:
        return f"{base}\n\n=== CANLI VERİ ===\n{ctx}" if lang == "tr" else f"{base}\n\n=== LIVE DATA ===\n{ctx}"
    return base


def _is_news_query(text):
    keywords = ["haber", "gündem", "gelişme", "son dakika", "bu hafta", "bugün",
                 "duyuru", "açıklama", "karar", "toplantı", "veri", "rapor",
                 "news", "today", "this week", "latest", "recent", "update",
                 "announcement", "decision", "meeting", "report"]
    return any(kw in text.lower() for kw in keywords)


def _fetch_news_context(query, lang):
    """DuckDuckGo üzerinden finansal haber özeti çeker."""
    import urllib.request, urllib.parse, json
    try:
        q = urllib.parse.quote(query + " financial markets")
        url = f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode())
        results = []
        if data.get("Abstract"):
            results.append(data["Abstract"])
        for topic in data.get("RelatedTopics", [])[:4]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(topic["Text"])
        if results:
            sep = "=== GÜNCEL HABERLER ===" if lang == "tr" else "=== RECENT NEWS ==="
            return sep + "\n" + "\n".join(f"• {r}" for r in results)
    except Exception:
        pass
    return ""

def render_fixed_chat(asset_data, page, lang):
    client = _get_client()
    if client is None:
        return

    hist_key = "global_chat_history"
    if hist_key not in st.session_state:
        st.session_state[hist_key] = []

    # Sayfa değişince: temizle + özet bayrağı koy + rerun
    prev_page = st.session_state.get("prev_page", "__init__")
    if prev_page != page:
        st.session_state["prev_page"] = page
        st.session_state[hist_key] = []
        st.session_state["need_summary"] = True
        st.rerun()

    # Özet üret (rerun sonrası bu blok çalışır, need_summary=True, sayfa aynı)
    if st.session_state.pop("need_summary", False):
        ctx = _build_page_context(asset_data, page, lang)
        prompt = (
            f"'{_page_name(page, lang)}' sayfası yüklendi. Canlı verideki en dikkat çekici "
            f"2-3 noktayı söyle, sonra 'Ne üzerine konuşalım?' diye sor."
            if lang == "tr" else
            f"'{_page_name(page, lang)}' page loaded. Mention 2-3 most notable points "
            f"from live data, then ask 'What would you like to explore?'"
        )
        try:
            resp = client.chat.completions.create(
                model="stepfun/step-3.5-flash:free",
                messages=[
                    {"role": "system", "content": _build_system(lang, ctx)},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=350, temperature=0.3,
            )
            txt = resp.choices[0].message.content
            txt = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', txt)
            txt = re.sub(r'^#{1,3}\s+', '', txt, flags=re.MULTILINE)
            st.session_state[hist_key].append({"role": "assistant", "content": f"📄 {_page_name(page, lang)} — {txt}", "auto": True})
        except Exception as e:
            st.session_state[hist_key].append({"role": "assistant", "content": f"📄 {_page_name(page, lang)} — ⚠️ {str(e)[:60]}", "auto": True})

    # Mesajları göster
    recent = st.session_state[hist_key][-8:]
    if recent:
        with st.container(height=280):
            for msg in recent:
                icon = "🤖" if msg["role"] == "assistant" else "👤"
                color = "#ccc" if msg["role"] == "assistant" else "#7eb8f7"
                opacity = "0.8" if msg.get("auto") else "1"
                st.markdown(
                    f'<div style="color:{color};font-size:0.82em;opacity:{opacity};'
                    f'margin-bottom:6px;line-height:1.5;">'
                    f'{icon} {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

    col_in, col_clr = st.columns([9, 1])
    with col_in:
        user_msg = st.chat_input(
            "Sorun veya yorum..." if lang == "tr" else "Ask or comment...",
            key="global_chat_input",
        )
    with col_clr:
        if st.button("🗑️", key="global_clear"):
            st.session_state[hist_key] = []
            st.rerun()

    if user_msg:
        st.session_state[hist_key].append({"role": "user", "content": user_msg})
        ctx = _build_page_context(asset_data, page, lang)
        news_ctx = ""
        if _is_news_query(user_msg):
            with st.spinner("🔍 Haberler aranıyor..." if lang == "tr" else "🔍 Fetching news..."):
                news_ctx = _fetch_news_context(user_msg, lang)
        full_ctx = "\n\n".join(filter(None, [ctx, news_ctx]))
        messages = [{"role": "system", "content": _build_system(lang, full_ctx)}]
        messages.extend(st.session_state[hist_key][-12:])

        # Streaming yanıt — kelime kelime akış
        def _stream_response():
            try:
                stream = client.chat.completions.create(
                    model="stepfun/step-3.5-flash:free",
                    messages=messages,
                    max_tokens=1200,
                    temperature=0.4,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
            except Exception as e:
                yield f"⚠️ {str(e)[:100]}"

        # Streaming placeholder — mesaj kutusunun altında göster
        stream_placeholder = st.empty()
        full_answer = ""
        for token in _stream_response():
            full_answer += token
            clean = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', full_answer)
            stream_placeholder.markdown(
                f'<div style="color:#ccc;font-size:0.82em;line-height:1.5;">'
                f'🤖 {clean}▌</div>',
                unsafe_allow_html=True,
            )
        stream_placeholder.empty()  # placeholder'ı temizle

        answer = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', full_answer)
        st.session_state[hist_key].append({"role": "assistant", "content": answer})
        st.rerun()


def main():
    if "lang" not in st.session_state:
        st.session_state["lang"] = "tr"

    selections = render_sidebar(st.session_state["lang"])
    st.session_state["lang"] = selections["lang"]
    lang = selections["lang"]
    page = selections["page"]

    col1, col2 = st.columns([1, 6])
    with col1:
        refresh = st.button(t("sidebar_refresh", lang), key="top_refresh")
    with col2:
        ts = get_cache_timestamp()
        if ts:
            st.caption(f"{t('sidebar_last_update', lang)}: {ts.strftime('%Y-%m-%d %H:%M')}")

    tradingview_ticker_tape()

    asset_data, macro_data = load_data(force_refresh=refresh)

    if page == "dashboard":
        render_dashboard(asset_data, lang)
    elif page == "asset_detail":
        render_asset_detail(asset_data, "DXY", lang)
    elif page == "forecast":
        render_forecast(asset_data, "DXY", lang)
    elif page == "comparison":
        render_comparison(asset_data, list(asset_data.keys()), lang)
    elif page == "macro":
        render_macro_overview(asset_data, macro_data, lang)

    render_fixed_chat(asset_data, page, lang)


if __name__ == "__main__":
    main()
