"""Simple in-memory cache — Streamlit bağımsız."""

from datetime import datetime

_cache: dict = {}
_timestamp: datetime | None = None


def get_cached(key: str):
    return _cache.get(key)


def set_cached(key: str, data):
    global _timestamp
    _cache[key] = data
    _timestamp = datetime.now()


def get_cache_timestamp() -> datetime | None:
    return _timestamp


def clear_cache():
    global _timestamp
    _cache.clear()
    _timestamp = None
