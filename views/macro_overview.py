"""Macro indicators overview page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config import MACRO_SERIES
from analysis.macro import analyze_macro_environment, compute_macro_correlations
from components.charts import create_correlation_heatmap
from components.analysis_card import render_commentary, render_disclaimer
from data.windowing import window_for_macro_correlation
from utils.i18n import t


def render_macro_overview(
    asset_data: dict[str, pd.DataFrame],
    macro_data: dict[str, pd.DataFrame],
    lang: str = "tr",
):
    """Render macroeconomic indicators overview page."""
    st.header(f"🌍 {t('macro_title', lang)}")

    if not macro_data:
        st.warning(
            "Makro veriler yüklenemedi. FRED API key'ini kontrol edin (.env dosyası)."
            if lang == "tr"
            else "Could not load macro data. Check your FRED API key (.env file)."
        )
        st.info(
            "Ücretsiz FRED API key almak için: https://fred.stlouisfed.org/docs/api/api_key.html"
            if lang == "tr"
            else "Get a free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        return

    # Display current macro values
    st.subheader("📊 " + ("Güncel Değerler" if lang == "tr" else "Current Values"))

    cols = st.columns(min(len(macro_data), 3))
    for i, (key, df) in enumerate(macro_data.items()):
        if df is None or df.empty:
            continue
        col = cols[i % len(cols)]
        info = MACRO_SERIES.get(key, {})
        name = info.get(f"name_{lang}", info.get("name_en", key))
        value = df["value"].iloc[-1]

        with col:
            # Calculate change
            if len(df) > 1:
                prev = df["value"].iloc[-2]
                delta = value - prev
                st.metric(name, f"{value:.2f}", f"{delta:+.2f}")
            else:
                st.metric(name, f"{value:.2f}")

    st.markdown("---")

    # Macro time series charts
    st.subheader("📈 " + ("Tarihsel Seyir" if lang == "tr" else "Historical Trend"))

    for key, df in macro_data.items():
        if df is None or df.empty:
            continue
        info = MACRO_SERIES.get(key, {})
        name = info.get(f"name_{lang}", info.get("name_en", key))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["value"],
                name=name,
                line=dict(width=2, color="#2196F3"),
                fill="tozeroy",
                fillcolor="rgba(33, 150, 243, 0.1)",
            )
        )
        fig.update_layout(
            title=name,
            height=300,
            template="plotly_dark",
            margin=dict(l=50, r=20, t=40, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Correlation between assets and macro (1y window - correlations are regime-dependent)
    st.subheader(t("macro_correlation", lang))
    windowed_assets = {k: window_for_macro_correlation(v) for k, v in asset_data.items()}
    windowed_macro = {k: window_for_macro_correlation(v) for k, v in macro_data.items()}
    corr = compute_macro_correlations(windowed_assets, windowed_macro)
    if corr is not None:
        fig = create_correlation_heatmap(corr, lang)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Korelasyon hesaplamak için yeterli veri yok."
            if lang == "tr"
            else "Not enough data to compute correlations."
        )

    st.markdown("---")

    # Macro commentary
    title = "💬 " + (t("macro_impact", lang))
    comments = analyze_macro_environment(macro_data, lang)
    render_commentary(comments, title)

    render_disclaimer(lang)
