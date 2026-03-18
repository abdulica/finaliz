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

    # Daily change cards
    daily_title = "📅 Günlük Değişim" if lang == "tr" else "📅 Daily Change"
    st.subheader(daily_title)

    high_lbl = "En Yüksek" if lang == "tr" else "High"
    low_lbl = "En Düşük" if lang == "tr" else "Low"

    price_items = list(prices.items())
    for row_start in range(0, len(price_items), 3):
        row = price_items[row_start:row_start + 3]
        cols = st.columns(len(row))
        for col, (key, info) in zip(cols, row):
            name = get_asset_name(key, ASSETS[key], lang)
            pct = info["change_pct"]
            change = info["change"]
            close = info["close"]
            high = info["high"]
            low = info["low"]
            is_up = pct >= 0
            arrow = "▲" if is_up else "▼"
            pct_color = "#26a69a" if is_up else "#ef5350"
            border_color = "#26a69a" if is_up else "#ef5350"
            # Vertical bar position (bottom = low, top = high)
            rng = high - low
            pos_pct = ((close - low) / rng * 100) if rng > 0 else 50
            bar_color = "#26a69a" if is_up else "#ef5350"
            with col:
                st.markdown(
                    f'<div style="background:#1e1e2e; padding:14px; border-radius:10px; '
                    f'border-left:4px solid {border_color}; margin-bottom:8px;">'
                    # Header: name + price
                    f'<div style="display:flex; justify-content:space-between; align-items:flex-start;">'
                    f'<div>'
                    f'<div style="color:#fff; font-size:1.05em; font-weight:bold; margin-bottom:4px;">{name}</div>'
                    f'<div style="font-size:1.3em; font-weight:bold; color:#fff;">${close:,.2f}</div>'
                    f'<div style="color:{pct_color}; font-size:1.05em; font-weight:bold; margin-top:4px;">'
                    f'{arrow} {pct:+.2f}%'
                    f'<span style="color:#888; font-size:0.75em; font-weight:normal; margin-left:6px;">'
                    f'${change:+,.2f}</span></div>'
                    f'</div>'
                    # Vertical bar
                    f'<div style="display:flex; flex-direction:column; align-items:center; min-width:40px;">'
                    f'<span style="color:#26a69a; font-size:0.65em; margin-bottom:2px;">${high:,.2f}</span>'
                    f'<div style="background:#333; border-radius:4px; width:8px; height:60px; position:relative; '
                    f'display:flex; flex-direction:column-reverse;">'
                    f'<div style="background:{bar_color}; width:100%; height:{pos_pct:.0f}%; '
                    f'border-radius:4px; min-height:2px;"></div></div>'
                    f'<span style="color:#ef5350; font-size:0.65em; margin-top:2px;">${low:,.2f}</span>'
                    f'</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

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
