"""Comparison page - compare multiple assets side by side."""

import streamlit as st
import pandas as pd
import numpy as np

from config import ASSETS
from data.fetcher import get_latest_prices
from components.charts import create_comparison_chart, create_correlation_heatmap
from components.analysis_card import render_commentary, render_disclaimer
from utils.i18n import t, get_asset_name


def render_comparison(
    data: dict[str, pd.DataFrame],
    selected_assets: list[str],
    lang: str = "tr",
):
    """Render comparison page for multiple assets."""
    st.header(f"📊 {t('comp_title', lang)}")

    # In-page multi-asset selector
    asset_names = {get_asset_name(k, v, lang): k for k, v in ASSETS.items() if k in data}
    asset_labels = list(asset_names.keys())
    default_labels = asset_labels[:3]

    selected_labels = st.multiselect(
        "Karşılaştırılacak Varlıklar" if lang == "tr" else "Assets to Compare",
        asset_labels,
        default=default_labels,
        key="comp_asset_selector",
    )
    selected_assets = [asset_names[lbl] for lbl in selected_labels]

    st.markdown("---")

    if len(selected_assets) < 2:
        st.info(
            "Karşılaştırma için en az 2 varlık seçin."
            if lang == "tr"
            else "Select at least 2 assets for comparison."
        )
        return

    # Normalized performance chart
    st.subheader(t("comp_normalized", lang))
    fig = create_comparison_chart(data, selected_assets, lang)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Price summary comparison
    prices = get_latest_prices({k: data[k] for k in selected_assets if k in data})

    if prices:
        cols = st.columns(min(len(prices), 4))
        for i, (key, info) in enumerate(prices.items()):
            col = cols[i % len(cols)]
            name = get_asset_name(key, ASSETS[key], lang)
            with col:
                st.metric(name, f"${info['close']:,.2f}", f"{info['change_pct']:+.2f}%")

    st.markdown("---")

    # Correlation matrix
    st.subheader(t("comp_correlation", lang))
    filtered_data = {k: data[k] for k in selected_assets if k in data}

    # Build correlation from daily returns
    returns = {}
    for key, df in filtered_data.items():
        if df is not None and not df.empty and len(df) > 10:
            name = get_asset_name(key, ASSETS[key], lang)
            returns[name] = df["Close"].pct_change().dropna()

    if len(returns) >= 2:
        combined = pd.DataFrame(returns).dropna()
        if len(combined) > 5:
            corr = combined.corr()
            fig = create_correlation_heatmap(corr, lang)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Comparative commentary
    comments = _generate_comparison_commentary(data, selected_assets, lang)
    if comments:
        title = "💬 " + ("Karşılaştırma Yorumu" if lang == "tr" else "Comparison Commentary")
        render_commentary(comments, title)

    render_disclaimer(lang)


def _generate_comparison_commentary(
    data: dict[str, pd.DataFrame],
    assets: list[str],
    lang: str,
) -> list[str]:
    """Generate commentary comparing selected assets."""
    comments = []
    prices = get_latest_prices({k: data[k] for k in assets if k in data})

    if not prices:
        return comments

    # Best and worst performer
    sorted_by_change = sorted(prices.items(), key=lambda x: x[1]["change_pct"], reverse=True)
    best_key, best_info = sorted_by_change[0]
    worst_key, worst_info = sorted_by_change[-1]

    best_name = get_asset_name(best_key, ASSETS[best_key], lang)
    worst_name = get_asset_name(worst_key, ASSETS[worst_key], lang)

    if lang == "tr":
        comments.append(
            f"📊 Günün en iyi performans gösteren varlığı {best_name} "
            f"(%{best_info['change_pct']:+.2f}), en zayıf ise {worst_name} "
            f"(%{worst_info['change_pct']:+.2f})."
        )
    else:
        comments.append(
            f"📊 Today's best performer is {best_name} "
            f"({best_info['change_pct']:+.2f}%), and the weakest is {worst_name} "
            f"({worst_info['change_pct']:+.2f}%)."
        )

    # Check for divergences (e.g., DXY up while gold up = unusual)
    dxy_info = prices.get("DXY")
    gold_info = prices.get("GOLD")
    if dxy_info and gold_info:
        if dxy_info["change_pct"] > 0.3 and gold_info["change_pct"] > 0.3:
            if lang == "tr":
                comments.append(
                    "🤔 İlginç: DXY ve altın aynı anda yükseliyor. Normalde ters korelasyon "
                    "beklenir. Bu tür durumlar genelde piyasada güçlü bir belirsizlik veya "
                    "güvenli liman arayışına işaret eder."
                )
            else:
                comments.append(
                    "🤔 Interesting: DXY and gold are rising simultaneously. Normally, an inverse "
                    "correlation is expected. Such situations typically indicate strong market "
                    "uncertainty or a flight to safety."
                )
        elif dxy_info["change_pct"] > 0.3 and gold_info["change_pct"] < -0.3:
            if lang == "tr":
                comments.append(
                    "📈 Klasik korelasyon çalışıyor: DXY güçlenirken altın baskı altında. "
                    "Dolar gücü emtia fiyatlarını aşağı çekiyor."
                )
            else:
                comments.append(
                    "📈 Classic correlation at work: DXY strengthening while gold is under pressure. "
                    "Dollar strength is pulling commodity prices down."
                )

    # BTC vs traditional assets
    btc_info = prices.get("BTC")
    if btc_info:
        btc_name = get_asset_name("BTC", ASSETS["BTC"], lang)
        if abs(btc_info["change_pct"]) > 3:
            if lang == "tr":
                comments.append(
                    f"⚡ {btc_name} bugün %{btc_info['change_pct']:+.1f} ile oldukça volatil. "
                    f"Kripto piyasasındaki bu sert hareketin geleneksel piyasalara etkisini "
                    f"takip etmekte fayda var."
                )
            else:
                comments.append(
                    f"⚡ {btc_name} is quite volatile today at {btc_info['change_pct']:+.1f}%. "
                    f"Worth monitoring how this sharp crypto move affects traditional markets."
                )

    return comments
