"""Dashboard page - compact market overview."""

import streamlit as st
import pandas as pd

from config import ASSETS
from data.fetcher import get_latest_prices
from components.analysis_card import render_disclaimer
from utils.i18n import t, get_asset_name


def render_dashboard(data: dict[str, pd.DataFrame], lang: str = "tr"):
    st.header(t("dash_title", lang))

    prices = get_latest_prices(data)
    if not prices:
        st.warning(t("no_data", lang))
        return

    st.subheader("📅 " + ("Günlük Değişim" if lang == "tr" else "Daily Change"))
    price_items = list(prices.items())
    cols = st.columns(len(price_items))
    for col, (key, info) in zip(cols, price_items):
        name = get_asset_name(key, ASSETS[key], lang)
        pct = info["change_pct"]
        close = info["close"]
        is_up = pct >= 0
        arrow = "▲" if is_up else "▼"
        pct_color = "#26a69a" if is_up else "#ef5350"
        border_color = "#26a69a" if is_up else "#ef5350"
        with col:
            st.markdown(
                f'<div style="background:#1e1e2e;padding:6px 8px;border-radius:6px;'
                f'border-left:2px solid {border_color};text-align:center;">'
                f'<div style="color:#aaa;font-size:0.68em;white-space:nowrap;overflow:hidden;'
                f'text-overflow:ellipsis;">{name}</div>'
                f'<div style="color:#fff;font-size:0.88em;font-weight:bold;">${close:,.2f}</div>'
                f'<div style="color:{pct_color};font-size:0.76em;">{arrow}{pct:+.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    st.subheader("📊 " + ("Haftalık En Düşük / En Yüksek" if lang == "tr" else "Weekly Low / High"))
    items = [(k, df) for k, df in data.items() if df is not None and not df.empty]
    cols2 = st.columns(len(items))
    for col, (key, df) in zip(cols2, items):
        name = get_asset_name(key, ASSETS[key], lang)
        last_7 = df.tail(7)
        week_high = last_7["High"].max()
        week_low = last_7["Low"].min()
        current = last_7["Close"].iloc[-1]
        rng = week_high - week_low
        pos_pct = ((current - week_low) / rng * 100) if rng > 0 else 50
        with col:
            st.markdown(
                f'<div style="background:#1e1e2e;padding:5px 6px;border-radius:6px;">'
                f'<div style="color:#aaa;font-size:0.62em;text-align:center;white-space:nowrap;'
                f'overflow:hidden;text-overflow:ellipsis;">{name}</div>'
                f'<div style="background:#333;border-radius:2px;height:4px;margin:4px 0;">'
                f'<div style="background:#FFD700;width:{pos_pct:.0f}%;height:100%;border-radius:2px;"></div></div>'
                f'<div style="display:flex;justify-content:space-between;font-size:0.6em;">'
                f'<span style="color:#ef5350;">${week_low:,.1f}</span>'
                f'<span style="color:#26a69a;">${week_high:,.1f}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    render_disclaimer(lang)
