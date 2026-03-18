"""Simple session-based caching using Streamlit session_state."""

from datetime import datetime

import streamlit as st


def get_cached(key: str):
    """Get cached data from session state."""
    return st.session_state.get(f"cache_{key}")


def set_cached(key: str, data):
    """Store data in session state cache."""
    st.session_state[f"cache_{key}"] = data
    st.session_state["cache_timestamp"] = datetime.now()


def get_cache_timestamp() -> datetime | None:
    """Get last cache update timestamp."""
    return st.session_state.get("cache_timestamp")


def clear_cache():
    """Clear all cached data."""
    keys_to_remove = [k for k in st.session_state if k.startswith("cache_")]
    for k in keys_to_remove:
        del st.session_state[k]
