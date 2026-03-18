"""Finaliz - Financial Analysis and Forecasting Platform.

A technical analysis and forecasting tool for currencies, commodities, and crypto.
Run with: streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from components.sidebar import render_sidebar
from data.cache import get_cached, set_cached, get_cache_timestamp
from data.fetcher import fetch_all_assets, fetch_macro_data
from utils.i18n import t
from views.dashboard import render_dashboard
from views.asset_detail import render_asset_detail
from views.forecast_view import render_forecast
from views.comparison import render_comparison
from views.macro_overview import render_macro_overview

# Page config
st.set_page_config(
    page_title="Finaliz - Financial Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Viewport meta for mobile
st.markdown(
    '<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">',
    unsafe_allow_html=True,
)

# Custom CSS
st.markdown(
    """
    <style>
    /* Base */
    .stApp { background-color: #0e1117; }
    .stMetric { background-color: #1e1e2e; padding: 10px; border-radius: 8px; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; }

    /* Mobile responsive */
    @media (max-width: 768px) {
        /* Sidebar auto-collapse */
        section[data-testid="stSidebar"] { width: 0px; min-width: 0px; }
        section[data-testid="stSidebar"][aria-expanded="true"] { width: 280px; min-width: 280px; }

        /* Stack columns vertically */
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }

        /* Smaller fonts */
        div[data-testid="stMetricValue"] { font-size: 1rem; }
        div[data-testid="stMetricLabel"] { font-size: 0.75rem; }
        div[data-testid="stMetricDelta"] { font-size: 0.7rem; }

        /* Charts full width */
        .js-plotly-plot { width: 100% !important; }

        /* Reduce padding */
        .block-container { padding: 0.5rem 1rem !important; }

        /* Tables scroll horizontal */
        div[data-testid="stDataFrame"] { overflow-x: auto; }

        /* Subheader smaller */
        h2, h3 { font-size: 1.1rem !important; }
    }

    /* Tablet tweaks */
    @media (min-width: 769px) and (max-width: 1024px) {
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            min-width: 45% !important;
        }
        .block-container { padding: 1rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_top_bar(lang: str) -> bool:
    """Render the top bar with refresh button and last update info.

    Returns refresh_clicked.
    """
    col1, col2 = st.columns([1, 3])

    with col1:
        refresh = st.button(t("sidebar_refresh", lang), use_container_width=True, key="top_refresh")

    with col2:
        ts = get_cache_timestamp()
        if ts:
            st.caption(f"{t('sidebar_last_update', lang)}: {ts.strftime('%Y-%m-%d %H:%M')}")

    return refresh


def load_data(force_refresh: bool = False):
    """Load or refresh all data. Fetches 5y for max coverage; each module trims its own window."""
    cached_asset = get_cached("assets_5y")
    cached_macro = get_cached("macro")

    if cached_asset is not None and cached_macro is not None and not force_refresh:
        return cached_asset, cached_macro

    with st.spinner(t("loading", st.session_state.get("lang", "tr"))):
        asset_data = fetch_all_assets("5y")
        macro_data = fetch_macro_data(period_years=5)

        set_cached("assets_5y", asset_data)
        set_cached("macro", macro_data)

    return asset_data, macro_data


def main():
    # Initialize language in session state
    if "lang" not in st.session_state:
        st.session_state["lang"] = "tr"

    # Render sidebar (only nav + language)
    selections = render_sidebar(st.session_state["lang"])
    st.session_state["lang"] = selections["lang"]
    lang = selections["lang"]

    # Top bar: refresh on every page
    refresh = render_top_bar(lang)

    # Load data (always 5y, modules trim their own windows)
    asset_data, macro_data = load_data(force_refresh=refresh)

    # Route to selected page
    page = selections["page"]

    if page == "dashboard":
        render_dashboard(asset_data, lang)
    elif page == "asset_detail":
        render_asset_detail(asset_data, "DXY", lang)
    elif page == "forecast":
        render_forecast(asset_data, "DXY", lang)
    elif page == "comparison":
        render_comparison(asset_data, list(asset_data.keys()), lang)
    elif page == "macro":
        render_macro_overview(asset_data, macro_data, lang)


if __name__ == "__main__":
    main()
