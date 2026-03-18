"""Analysis card component for displaying commentary in a friendly tone."""

import streamlit as st

from config import DISCLAIMER_TR, DISCLAIMER_EN


def render_commentary(comments: list[str], title: str = "", unified: bool = False):
    """Render commentary paragraphs.

    Args:
        comments: List of paragraphs.
        title: Optional section title.
        unified: If True, merge all paragraphs into a single styled box.
    """
    if not comments:
        return

    if title:
        st.subheader(title)

    if unified:
        merged = "<br><br>".join(comments)
        st.markdown(
            f'<div style="background-color: #1e1e2e; padding: 20px; '
            f'border-radius: 10px; '
            f'border-left: 4px solid #FFD700; font-size: 0.95em; line-height: 1.7;">'
            f"{merged}</div>",
            unsafe_allow_html=True,
        )
    else:
        for comment in comments:
            st.markdown(
                f'<div style="background-color: #1e1e2e; padding: 15px; '
                f'border-radius: 10px; margin-bottom: 10px; '
                f'border-left: 4px solid #FFD700; font-size: 0.95em;">'
                f"{comment}</div>",
                unsafe_allow_html=True,
            )


def render_signal_badge(signal: str, lang: str = "tr"):
    """Render a colored signal badge."""
    colors = {"buy": "#26a69a", "sell": "#ef5350", "neutral": "#FFA726"}
    labels = {
        "buy": {"tr": "ALIM", "en": "BUY"},
        "sell": {"tr": "SATIM", "en": "SELL"},
        "neutral": {"tr": "NÖTR", "en": "NEUTRAL"},
    }
    color = colors.get(signal, "#FFA726")
    label = labels.get(signal, {}).get(lang, signal.upper())

    st.markdown(
        f'<span style="background-color: {color}; color: white; '
        f'padding: 5px 15px; border-radius: 20px; font-weight: bold; '
        f'font-size: 1.1em;">{label}</span>',
        unsafe_allow_html=True,
    )


def render_signal_summary(summary: dict, lang: str = "tr"):
    """Render technical signal summary with gauge-like display."""
    total = summary["buy"] + summary["sell"] + summary["neutral"]
    if total == 0:
        return

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        label = "Alım" if lang == "tr" else "Buy"
        st.metric(label, summary["buy"], delta=None)
    with col2:
        label = "Nötr" if lang == "tr" else "Neutral"
        st.metric(label, summary["neutral"], delta=None)
    with col3:
        label = "Satım" if lang == "tr" else "Sell"
        st.metric(label, summary["sell"], delta=None)
    with col4:
        label = "Genel Sinyal" if lang == "tr" else "Overall Signal"
        st.markdown(f"**{label}:**")
        render_signal_badge(summary["overall"], lang)


def render_forecast_table(forecast_results: dict, lang: str = "tr"):
    """Render forecast results as a clean table."""
    from config import FORECAST_HORIZONS

    rows = []
    for h_key in ["1d", "3d", "1w", "2w", "4w"]:
        result = forecast_results.get(h_key)
        if result is None:
            continue

        horizon_info = FORECAST_HORIZONS[h_key]
        label = horizon_info[f"label_{lang}"]

        direction_map = {
            "up": {"tr": "↑ Yükseliş", "en": "↑ Bullish"},
            "down": {"tr": "↓ Düşüş", "en": "↓ Bearish"},
            "sideways": {"tr": "→ Yatay", "en": "→ Sideways"},
        }
        direction = direction_map.get(result["direction"], {}).get(lang, result["direction"])

        rows.append({
            ("Periyot" if lang == "tr" else "Period"): label,
            ("Tahmini Fiyat" if lang == "tr" else "Predicted"): f"${result['predicted']:.2f}",
            ("Değişim %" if lang == "tr" else "Change %"): f"{result['change_pct']:+.1f}%",
            ("Alt Sınır" if lang == "tr" else "Lower"): f"${result['lower']:.2f}",
            ("Üst Sınır" if lang == "tr" else "Upper"): f"${result['upper']:.2f}",
            ("Yön" if lang == "tr" else "Direction"): direction,
        })

    if rows:
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_disclaimer(lang: str = "tr"):
    """Render the disclaimer notice."""
    text = DISCLAIMER_TR if lang == "tr" else DISCLAIMER_EN
    st.markdown(
        f'<div style="background-color: #2d1f1f; padding: 12px; '
        f'border-radius: 8px; border: 1px solid #ef5350; '
        f'font-size: 0.85em; margin-top: 20px;">{text}</div>',
        unsafe_allow_html=True,
    )
