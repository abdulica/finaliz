"""Sidebar navigation and controls - simplified."""

import streamlit as st

from utils.i18n import t


def render_sidebar(lang: str = "tr") -> dict:
    """Render sidebar with only navigation and language toggle.

    Returns dict with keys: page, lang.
    """
    st.sidebar.title("📊 Finaliz")

    # Language toggle
    lang_options = {"Türkçe": "tr", "English": "en"}
    selected_lang_label = st.sidebar.radio(
        t("sidebar_language", lang),
        list(lang_options.keys()),
        index=0 if lang == "tr" else 1,
        horizontal=True,
    )
    lang = lang_options[selected_lang_label]

    st.sidebar.markdown("---")

    # Navigation
    pages = {
        t("nav_dashboard", lang): "dashboard",
        t("nav_asset_detail", lang): "asset_detail",
        t("nav_forecast", lang): "forecast",
        t("nav_comparison", lang): "comparison",
        t("nav_macro", lang): "macro",
    }
    selected_page_label = st.sidebar.radio(
        "📌 " + ("Sayfa" if lang == "tr" else "Page"),
        list(pages.keys()),
    )
    page = pages[selected_page_label]

    return {
        "page": page,
        "lang": lang,
    }
