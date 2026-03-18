"""Dashboard page - market overview with summary table and mini charts."""

import streamlit as st
import pandas as pd

from config import ASSETS
from data.fetcher import get_latest_prices
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

    # Weekly high/low summary
    st.subheader("📊 " + ("Son 1 Hafta: En Yüksek / En Düşük" if lang == "tr" else "Last 1 Week: High / Low"))
    items = [(k, df) for k, df in data.items() if df is not None and not df.empty]
    for row_start in range(0, len(items), 3):
        row_items = items[row_start:row_start + 3]
        cols = st.columns(len(row_items))
        for col, (key, df) in zip(cols, row_items):
            name = get_asset_name(key, ASSETS[key], lang)
            last_7 = df.tail(7)
            week_high = last_7["High"].max()
            week_low = last_7["Low"].min()
            current = last_7["Close"].iloc[-1]
            # Position within range (0-100%)
            rng = week_high - week_low
            pos_pct = ((current - week_low) / rng * 100) if rng > 0 else 50
            with col:
                high_lbl = "En Yüksek" if lang == "tr" else "High"
                low_lbl = "En Düşük" if lang == "tr" else "Low"
                now_lbl = "Şu An" if lang == "tr" else "Current"
                st.markdown(
                    f'<div style="background:#1e1e2e; padding:12px; border-radius:8px; margin-bottom:8px;">'
                    f'<div style="font-weight:bold; margin-bottom:8px; font-size:0.95em;">{name}</div>'
                    f'<div style="display:flex; justify-content:space-between; font-size:0.8em; color:#888;">'
                    f'<span>{low_lbl}</span><span>{high_lbl}</span></div>'
                    f'<div style="background:#333; border-radius:4px; height:8px; margin:4px 0; position:relative;">'
                    f'<div style="background:#FFD700; width:{pos_pct:.0f}%; height:100%; border-radius:4px;"></div></div>'
                    f'<div style="display:flex; justify-content:space-between; font-size:0.85em;">'
                    f'<span style="color:#ef5350;">${week_low:,.2f}</span>'
                    f'<span style="color:#aaa;">{now_lbl}: ${current:,.2f}</span>'
                    f'<span style="color:#26a69a;">${week_high:,.2f}</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

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
