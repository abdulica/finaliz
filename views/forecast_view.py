"""Forecast page - multi-asset predictions with relationship analysis."""

import streamlit as st
import pandas as pd
import numpy as np

from config import ASSETS
from analysis.technical import compute_all_indicators, get_signal_summary, generate_commentary
from analysis.forecast import run_forecast, generate_forecast_commentary
from components.charts import create_forecast_chart, create_relationship_chart
from data.windowing import window_for_forecast, window_for_technical
from components.analysis_card import (
    render_commentary,
    render_forecast_table,
    render_disclaimer,
)
from utils.i18n import t, get_asset_name


def render_forecast(
    data: dict[str, pd.DataFrame],
    asset_key: str,
    lang: str = "tr",
):
    """Render forecast page with asset selectors and relationship analysis."""
    st.header(f"🔮 {t('fc_title', lang)}")

    # Asset selectors - user picks which assets to forecast and compare
    asset_names = {get_asset_name(k, v, lang): k for k, v in ASSETS.items() if k in data}

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        label_1 = "Birinci Varlık" if lang == "tr" else "First Asset"
        asset_labels = list(asset_names.keys())
        # Default to sidebar selection
        default_idx_1 = 0
        for i, lbl in enumerate(asset_labels):
            if asset_names[lbl] == asset_key:
                default_idx_1 = i
                break
        selected_1 = st.selectbox(label_1, asset_labels, index=default_idx_1, key="fc_asset_1")
        key_1 = asset_names[selected_1]

    with col_sel2:
        label_2 = "İkinci Varlık (İlişki Analizi)" if lang == "tr" else "Second Asset (Relationship)"
        # Default to something different from first
        default_idx_2 = 1 if default_idx_1 != 1 else 2
        default_idx_2 = min(default_idx_2, len(asset_labels) - 1)
        selected_2 = st.selectbox(label_2, asset_labels, index=default_idx_2, key="fc_asset_2")
        key_2 = asset_names[selected_2]

    st.markdown("---")

    # === Forecast for Asset 1 ===
    _render_single_forecast(data, key_1, lang)

    st.markdown("---")

    # === Forecast for Asset 2 ===
    if key_2 != key_1:
        _render_single_forecast(data, key_2, lang)
        st.markdown("---")

    # === Relationship Analysis ===
    df_1 = data.get(key_1)
    df_2 = data.get(key_2)

    if key_1 == key_2:
        st.info(
            "İlişki analizi için iki farklı varlık seçin."
            if lang == "tr"
            else "Select two different assets for relationship analysis."
        )
    elif df_1 is not None and df_2 is not None and not df_1.empty and not df_2.empty:
        name_1 = get_asset_name(key_1, ASSETS[key_1], lang)
        name_2 = get_asset_name(key_2, ASSETS[key_2], lang)

        title = f"🔗 {name_1} ↔ {name_2} " + ("İlişki Analizi" if lang == "tr" else "Relationship Analysis")
        st.header(title)

        # Relationship chart (4-panel)
        fig = create_relationship_chart(df_1, df_2, key_1, key_2, lang)
        st.plotly_chart(fig, use_container_width=True)

        # Relationship commentary
        comments = _generate_relationship_commentary(df_1, df_2, key_1, key_2, lang)
        if comments:
            render_commentary(comments)

    render_disclaimer(lang)


def _render_single_forecast(
    data: dict[str, pd.DataFrame],
    asset_key: str,
    lang: str,
):
    """Render forecast section for a single asset."""
    asset_config = ASSETS.get(asset_key)
    df_full = data.get(asset_key)

    if df_full is None or df_full.empty:
        name = get_asset_name(asset_key, asset_config, lang)
        st.warning(f"{name}: {t('error_data', lang)}")
        return

    name = get_asset_name(asset_key, asset_config, lang)
    st.subheader(f"🔮 {name}")

    # Smart windowing: 2y for forecast training, 1y for technical
    df_forecast = window_for_forecast(df_full)

    with st.spinner(f"{name} - {t('loading', lang)}"):
        forecast_results = run_forecast(df_forecast)

    if forecast_results is None:
        st.warning(
            f"{name}: Tahmin modeli çalıştırılamadı."
            if lang == "tr"
            else f"{name}: Could not run forecast model."
        )
        return

    # Forecast table
    render_forecast_table(forecast_results, lang)

    # Forecast chart
    full_forecast = forecast_results.get("_full_forecast")
    actual_data = forecast_results.get("_actual_data")

    if full_forecast is not None and actual_data is not None:
        fig = create_forecast_chart(actual_data, full_forecast, asset_key, lang)
        st.plotly_chart(fig, use_container_width=True)

    # Technical analysis (1y window)
    df_ta = window_for_technical(df_full)
    df_with_ta = compute_all_indicators(df_ta.copy())
    ta_summary = get_signal_summary(df_with_ta)

    # External context input
    ext_label = (
        "📝 Harici veri veya bağlam (jeopolitik, politika müdahalesi, vb.) — yoksa boş bırakın"
        if lang == "tr"
        else "📝 External context (geopolitical, policy intervention, etc.) — leave empty if none"
    )
    external_context = st.text_area(
        ext_label, value="", height=80,
        key=f"ext_context_forecast_{asset_key}",
    )

    # Technical commentary
    tech_title = "💬 " + ("Teknik Yorum" if lang == "tr" else "Technical Commentary")
    tech_comments = generate_commentary(df_with_ta, name, lang, external_context=external_context)
    if tech_comments:
        render_commentary(tech_comments, tech_title, unified=True)

    # Forecast commentary
    fc_title = "💬 " + ("Tahmin Yorumu" if lang == "tr" else "Forecast Commentary")
    fc_comments = generate_forecast_commentary(
        forecast_results, name, ta_summary["overall"], lang
    )
    if fc_comments:
        render_commentary(fc_comments, fc_title)


def _generate_relationship_commentary(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    key_a: str,
    key_b: str,
    lang: str,
) -> list[str]:
    """Generate commentary about the relationship between two assets."""
    comments = []
    name_a = get_asset_name(key_a, ASSETS[key_a], lang)
    name_b = get_asset_name(key_b, ASSETS[key_b], lang)

    common_idx = df_a.index.intersection(df_b.index)
    if len(common_idx) < 30:
        return comments

    close_a = df_a.loc[common_idx, "Close"]
    close_b = df_b.loc[common_idx, "Close"]
    ret_a = close_a.pct_change().dropna()
    ret_b = close_b.pct_change().dropna()

    # Overall correlation
    overall_corr = ret_a.corr(ret_b)

    # Recent vs historical correlation
    recent_ret_a = ret_a.tail(30)
    recent_ret_b = ret_b.tail(30)
    recent_corr = recent_ret_a.corr(recent_ret_b)

    if lang == "tr":
        # Overall correlation interpretation
        if abs(overall_corr) > 0.7:
            direction = "pozitif" if overall_corr > 0 else "negatif"
            comments.append(
                f"📊 {name_a} ve {name_b} arasında güçlü bir {direction} korelasyon var "
                f"({overall_corr:.2f}). Bu iki varlık genelde {'aynı yöne' if overall_corr > 0 else 'ters yönlere'} hareket ediyor."
            )
        elif abs(overall_corr) > 0.3:
            direction = "pozitif" if overall_corr > 0 else "negatif"
            comments.append(
                f"📊 {name_a} ve {name_b} arasında orta düzeyde {direction} korelasyon var "
                f"({overall_corr:.2f}). İlişki var ama her zaman tutarlı değil."
            )
        else:
            comments.append(
                f"📊 {name_a} ve {name_b} arasında anlamlı bir korelasyon yok "
                f"({overall_corr:.2f}). Bu varlıklar büyük ölçüde bağımsız hareket ediyor."
            )

        # Correlation shift
        if abs(recent_corr - overall_corr) > 0.25:
            if recent_corr > overall_corr:
                comments.append(
                    f"🔄 Dikkat çekici: Son 30 günde korelasyon ({recent_corr:.2f}) tarihsel "
                    f"ortalamanın ({overall_corr:.2f}) belirgin üzerinde. Bu iki varlık son dönemde "
                    f"normalden daha fazla birlikte hareket ediyor. Bu genelde ortak bir makro "
                    f"faktörün (faiz kararı, jeopolitik gelişme) her ikisini de etkilediğine işaret eder."
                )
            else:
                comments.append(
                    f"🔄 Dikkat çekici: Son 30 günde korelasyon ({recent_corr:.2f}) tarihsel "
                    f"ortalamanın ({overall_corr:.2f}) altına düştü. Ayrışma yaşanıyor. "
                    f"Bu tür ayrışmalar genelde ya kısa ömürlüdür ya da yeni bir rejime geçişe işaret eder."
                )

        # Volatility comparison
        vol_a = ret_a.std() * np.sqrt(252) * 100
        vol_b = ret_b.std() * np.sqrt(252) * 100
        comments.append(
            f"📈 Yıllık volatilite: {name_a} %{vol_a:.1f}, {name_b} %{vol_b:.1f}. "
            + (f"{name_a} daha volatil." if vol_a > vol_b else f"{name_b} daha volatil.")
        )

        # Lead-lag hint
        lag_corr_1 = ret_a.shift(1).corr(ret_b)  # A leads B
        lag_corr_2 = ret_b.shift(1).corr(ret_a)  # B leads A
        if abs(lag_corr_1) > abs(overall_corr) + 0.1:
            comments.append(
                f"🕐 İlginç bir gözlem: {name_a}'daki hareketler, {name_b}'yi bir gün gecikmeyle "
                f"etkiliyor olabilir (gecikmeli korelasyon: {lag_corr_1:.2f}). "
                f"Bu bir öncü-takipçi ilişkisine işaret edebilir."
            )
        elif abs(lag_corr_2) > abs(overall_corr) + 0.1:
            comments.append(
                f"🕐 İlginç bir gözlem: {name_b}'deki hareketler, {name_a}'yı bir gün gecikmeyle "
                f"etkiliyor olabilir (gecikmeli korelasyon: {lag_corr_2:.2f}). "
                f"Bu bir öncü-takipçi ilişkisine işaret edebilir."
            )

    else:
        if abs(overall_corr) > 0.7:
            direction = "positive" if overall_corr > 0 else "negative"
            comments.append(
                f"📊 Strong {direction} correlation between {name_a} and {name_b} "
                f"({overall_corr:.2f}). These assets generally move {'together' if overall_corr > 0 else 'in opposite directions'}."
            )
        elif abs(overall_corr) > 0.3:
            direction = "positive" if overall_corr > 0 else "negative"
            comments.append(
                f"📊 Moderate {direction} correlation between {name_a} and {name_b} "
                f"({overall_corr:.2f}). There's a relationship but it's not always consistent."
            )
        else:
            comments.append(
                f"📊 No significant correlation between {name_a} and {name_b} "
                f"({overall_corr:.2f}). These assets move largely independently."
            )

        if abs(recent_corr - overall_corr) > 0.25:
            if recent_corr > overall_corr:
                comments.append(
                    f"🔄 Notable: Recent 30-day correlation ({recent_corr:.2f}) is significantly above "
                    f"the historical average ({overall_corr:.2f}). These assets are moving together more "
                    f"than usual, often indicating a common macro factor driving both."
                )
            else:
                comments.append(
                    f"🔄 Notable: Recent 30-day correlation ({recent_corr:.2f}) has dropped below "
                    f"the historical average ({overall_corr:.2f}). A divergence is occurring."
                )

        vol_a = ret_a.std() * np.sqrt(252) * 100
        vol_b = ret_b.std() * np.sqrt(252) * 100
        comments.append(
            f"📈 Annualized volatility: {name_a} {vol_a:.1f}%, {name_b} {vol_b:.1f}%. "
            + (f"{name_a} is more volatile." if vol_a > vol_b else f"{name_b} is more volatile.")
        )

        lag_corr_1 = ret_a.shift(1).corr(ret_b)
        lag_corr_2 = ret_b.shift(1).corr(ret_a)
        if abs(lag_corr_1) > abs(overall_corr) + 0.1:
            comments.append(
                f"🕐 Interesting: {name_a} movements may lead {name_b} by one day "
                f"(lagged correlation: {lag_corr_1:.2f}). This could indicate a lead-lag relationship."
            )
        elif abs(lag_corr_2) > abs(overall_corr) + 0.1:
            comments.append(
                f"🕐 Interesting: {name_b} movements may lead {name_a} by one day "
                f"(lagged correlation: {lag_corr_2:.2f}). This could indicate a lead-lag relationship."
            )

    return comments
