"""Kompakt sidebar — ikon + kısa etiket."""

import streamlit as st
from utils.i18n import t


PAGE_ICONS = {
    "dashboard":    "🏠",
    "asset_detail": "📊",
    "forecast":     "🔮",
    "comparison":   "⚖️",
    "macro":        "🌍",
}


def render_sidebar(lang: str = "tr") -> dict:
    st.sidebar.markdown(
        '<div style="font-size:1.2em;font-weight:bold;padding:4px 0 12px;">📈 Finaliz</div>',
        unsafe_allow_html=True,
    )

    # Dil geçişi — kompakt
    lang_options = {"TR": "tr", "EN": "en"}
    selected_lang_label = st.sidebar.radio(
        "", list(lang_options.keys()), index=0 if lang == "tr" else 1, horizontal=True
    )
    lang = lang_options[selected_lang_label]

    st.sidebar.markdown("---")

    pages = {
        "dashboard":    t("nav_dashboard", lang),
        "asset_detail": t("nav_asset_detail", lang),
        "forecast":     t("nav_forecast", lang),
        "comparison":   t("nav_comparison", lang),
        "macro":        t("nav_macro", lang),
    }

    # Mevcut aktif sayfayı session'dan al
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "dashboard"

    for key, label in pages.items():
        icon = PAGE_ICONS.get(key, "•")
        is_active = st.session_state["active_page"] == key
        style = "background:rgba(255,215,0,0.15);border-left:3px solid #FFD700;" if is_active else "border-left:3px solid transparent;"
        st.sidebar.markdown(
            f'<div style="{style}padding:6px 8px;border-radius:6px;margin-bottom:2px;">'
            f'<span style="font-size:0.9em;">{icon} {label}</span></div>',
            unsafe_allow_html=True,
        )
        if st.sidebar.button(f"{icon} {label}", key=f"nav_{key}", use_container_width=True):
            st.session_state["active_page"] = key

    page = st.session_state["active_page"]

    st.sidebar.markdown("---")
    st.sidebar.caption("v1.0 • Finaliz")

    return {"page": page, "lang": lang}
