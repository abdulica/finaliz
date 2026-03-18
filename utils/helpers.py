"""General helper utilities."""

import pandas as pd


def format_currency(value: float, decimals: int = 2) -> str:
    """Format a number as USD currency."""
    return f"${value:,.{decimals}f}"


def format_pct(value: float, decimals: int = 2) -> str:
    """Format a number as percentage with sign."""
    return f"{value:+.{decimals}f}%"


def safe_pct_change(current: float, previous: float) -> float:
    """Calculate percentage change safely."""
    if previous == 0:
        return 0.0
    return (current - previous) / previous * 100
