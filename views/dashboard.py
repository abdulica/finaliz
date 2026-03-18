"""Dashboard page - market overview with summary table and mini charts."""

import streamlit as st
import pandas as pd

from config import ASSETS
from data.fetcher import get_latest_prices
from components.charts import create_mini_sparkline
from components.analysis_card import render_disclaimer
from utils.i18n import t, get_asset_name


def render_dashboard(data: dict[str, pd.DataFrame], lang: str = "tr"):
    """Render the main dashboard overview page."""
    st.header(t("dash_title", lang))

    prices = get_latest_prices(data)

    if not prices:
        st.warning(t("no_data", lang))
        return

    # Summary metrics row
    cols = st.columns(min(len(prices), 4))
    for i, (key, info) in enumerate(prices.items()):
        col = cols[i % len(cols)]
        name = get_asset_name(key, ASSETS[key], lang)
        delta_str = f"{info['change_pct']:+.2f}%"
        with col:
            st.metric(
                label=name,
                value=f"${info['close']:,.2f}",
                delta=delta_str,
            )

    st.markdown("---")

    # Detailed table
    table_data = []
    for key, info in prices.items():
        name = get_asset_name(key, ASSETS[key], lang)
        table_data.append({
            ("Varlık" if lang == "tr" else "Asset"): name,
            (t("dash_price", lang)): f"${info['close']:,.2f}",
            (t("dash_change", lang)): f"${info['change']:+,.2f}",
            (t("dash_change_pct", lang)): f"{info['change_pct']:+.2f}%",
            (t("dash_high", lang)): f"${info['high']:,.2f}",
            (t("dash_low", lang)): f"${info['low']:,.2f}",
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Mini charts grid
    st.subheader("📈 " + ("Son 1 Hafta" if lang == "tr" else "Last 1 Week"))
    chart_cols = st.columns(min(len(data), 4))
    for i, (key, df) in enumerate(data.items()):
        if df is None or df.empty:
            continue
        col = chart_cols[i % len(chart_cols)]
        name = get_asset_name(key, ASSETS[key], lang)
        color = ASSETS[key]["color"]
        with col:
            st.caption(name)
            last_30 = df["Close"].tail(7)
            fig = create_mini_sparkline(last_30, color)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Market sentiment summary
    st.markdown("---")
    st.subheader(t("dash_sentiment", lang))

    bullish = sum(1 for info in prices.values() if info["change_pct"] > 0)
    bearish = len(prices) - bullish

    if bullish > bearish:
        if lang == "tr":
            st.success(
                f"📈 Genel piyasa duyarlılığı pozitif: {bullish}/{len(prices)} varlık yükselişte. "
                f"Risk iştahı artmış görünüyor."
            )
        else:
            st.success(
                f"📈 Overall market sentiment is positive: {bullish}/{len(prices)} assets are up. "
                f"Risk appetite appears elevated."
            )
    elif bearish > bullish:
        if lang == "tr":
            st.error(
                f"📉 Genel piyasa duyarlılığı negatif: {bearish}/{len(prices)} varlık düşüşte. "
                f"Piyasada temkinli bir hava hakim."
            )
        else:
            st.error(
                f"📉 Overall market sentiment is negative: {bearish}/{len(prices)} assets are down. "
                f"A cautious mood prevails in the market."
            )
    else:
        if lang == "tr":
            st.info("➡️ Piyasa kararsız: Eşit sayıda yükselen ve düşen varlık var.")
        else:
            st.info("➡️ Market is mixed: Equal number of rising and falling assets.")

    render_disclaimer(lang)
