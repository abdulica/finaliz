"""Asset detail page - full technical analysis with in-page asset selector."""

from datetime import datetime

import streamlit as st
import pandas as pd

from config import ASSETS
from analysis.technical import (
    compute_all_indicators,
    compute_support_resistance,
    get_signal_summary,
    generate_commentary,
)
from analysis.seasonal import compute_seasonal_pattern, get_seasonal_commentary
from components.charts import create_candlestick_chart, create_seasonal_chart
from data.windowing import window_for_technical, window_for_seasonal
from components.analysis_card import (
    render_commentary,
    render_signal_summary,
    render_disclaimer,
)
from utils.i18n import t, get_asset_name


def render_asset_detail(
    data: dict[str, pd.DataFrame],
    asset_key: str,
    lang: str = "tr",
):
    """Render the detailed analysis page for a single asset."""
    st.header(f"📊 {t('nav_asset_detail', lang)}")

    # In-page asset selector
    asset_names = {get_asset_name(k, v, lang): k for k, v in ASSETS.items() if k in data}
    asset_labels = list(asset_names.keys())

    # Default to sidebar selection
    default_idx = 0
    for i, lbl in enumerate(asset_labels):
        if asset_names[lbl] == asset_key:
            default_idx = i
            break

    selected_label = st.selectbox(
        "Varlık Seçin" if lang == "tr" else "Select Asset",
        asset_labels,
        index=default_idx,
        key="detail_asset_selector",
    )
    selected_key = asset_names[selected_label]

    st.markdown("---")

    _render_single_asset(data, selected_key, lang)


def _render_single_asset(
    data: dict[str, pd.DataFrame],
    asset_key: str,
    lang: str,
):
    """Render full technical analysis for a single asset."""
    asset_config = ASSETS.get(asset_key)
    if asset_config is None:
        st.error(t("no_data", lang))
        return

    df_full = data.get(asset_key)
    if df_full is None or df_full.empty:
        st.warning(f"{get_asset_name(asset_key, asset_config, lang)}: {t('error_data', lang)}")
        return

    # Smart windowing: 1y for technical, full for seasonal
    df = window_for_technical(df_full)

    name = get_asset_name(asset_key, asset_config, lang)
    st.subheader(f"📊 {name}")

    # Current price header
    close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2] if len(df) > 1 else close
    change_pct = (close - prev_close) / prev_close * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(t("dash_price", lang), f"${close:,.2f}", f"{change_pct:+.2f}%")
    with col2:
        st.metric(t("dash_high", lang), f"${df['High'].iloc[-1]:,.2f}")
    with col3:
        st.metric(t("dash_low", lang), f"${df['Low'].iloc[-1]:,.2f}")

    st.markdown("---")

    # Compute indicators
    df = compute_all_indicators(df)
    summary = get_signal_summary(df)

    # Signal summary
    st.subheader(t("ta_summary", lang))
    render_signal_summary(summary, lang)

    st.markdown("---")

    # Main chart
    st.subheader(t("ta_title", lang))
    fig = create_candlestick_chart(df, asset_key, lang=lang)
    st.plotly_chart(fig, use_container_width=True)

    # Support / Resistance
    sr = compute_support_resistance(df)
    if sr["support"] and sr["resistance"]:
        st.subheader(t("ta_support_resistance", lang))
        sr_col1, sr_col2, sr_col3 = st.columns(3)
        with sr_col1:
            st.metric("Pivot", f"${sr['pivot']:,.2f}")
        with sr_col2:
            label = "Destek" if lang == "tr" else "Support"
            for i, s in enumerate(sr["support"]):
                st.metric(f"{label} {i+1}", f"${s:,.2f}")
        with sr_col3:
            label = "Direnç" if lang == "tr" else "Resistance"
            for i, r in enumerate(sr["resistance"]):
                st.metric(f"{label} {i+1}", f"${r:,.2f}")

    st.markdown("---")

    # External context input
    ext_label = (
        "📝 Harici veri veya bağlam (jeopolitik, politika müdahalesi, vb.) — yoksa boş bırakın"
        if lang == "tr"
        else "📝 External context (geopolitical, policy intervention, etc.) — leave empty if none"
    )
    external_context = st.text_area(
        ext_label, value="", height=80,
        key=f"ext_context_detail_{asset_key}",
    )

    # Technical commentary
    title = "💬 " + ("Teknik Yorum" if lang == "tr" else "Technical Commentary")
    comments = generate_commentary(df, name, lang, external_context=external_context)
    render_commentary(comments, title, unified=True)

    # Seasonal analysis (uses full 5y data for pattern depth, with recency weighting)
    df_seasonal = window_for_seasonal(df_full)
    seasonal = compute_seasonal_pattern(df_seasonal)
    if seasonal is not None:
        st.markdown("---")
        st.subheader(t("seasonal_title", lang))

        fig_seasonal = create_seasonal_chart(
            seasonal, asset_key, datetime.now().month, lang
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)

        seasonal_comments = get_seasonal_commentary(
            seasonal, datetime.now().month, name, lang
        )
        render_commentary(seasonal_comments)

    render_disclaimer(lang)
