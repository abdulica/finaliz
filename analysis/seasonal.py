"""Seasonal analysis - monthly patterns with recency weighting.

Recent years (last 2) get 2x weight compared to older data.
This reflects the reality that recent market regimes are more
predictive than distant history.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def compute_seasonal_pattern(df: pd.DataFrame) -> pd.DataFrame | None:
    """Compute weighted average monthly returns from historical data.

    Recent 2 years get double weight vs older data.
    Returns DataFrame with month, avg_return, median_return, positive_pct.
    """
    if df is None or df.empty or len(df) < 252:
        return None

    monthly = df["Close"].resample("M").last().pct_change().dropna()
    if len(monthly) < 12:
        return None

    # Assign weights: last 2 years = 2x, older = 1x
    cutoff = monthly.index[-1] - timedelta(days=730)
    weights = pd.Series(
        np.where(monthly.index >= cutoff, 2.0, 1.0),
        index=monthly.index,
    )

    monthly_df = pd.DataFrame({
        "month": monthly.index.month,
        "return": monthly.values,
        "weight": weights.values,
    })

    def weighted_mean(group):
        return np.average(group["return"], weights=group["weight"])

    def weighted_positive_pct(group):
        pos = group["return"] > 0
        return np.average(pos.astype(float), weights=group["weight"]) * 100

    result = monthly_df.groupby("month").apply(
        lambda g: pd.Series({
            "avg_return": weighted_mean(g),
            "median_return": g["return"].median(),
            "std_return": g["return"].std(),
            "positive_pct": weighted_positive_pct(g),
            "count": len(g),
        }),
        include_groups=False,
    ).reset_index()

    return result


def get_seasonal_commentary(
    seasonal: pd.DataFrame | None,
    current_month: int,
    asset_name: str,
    lang: str = "tr",
) -> list[str]:
    """Generate seasonal analysis commentary."""
    if seasonal is None:
        return []

    comments = []
    month_names_tr = [
        "", "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
        "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
    ]
    month_names_en = [
        "", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    current = seasonal[seasonal["month"] == current_month]
    if current.empty:
        return []

    row = current.iloc[0]
    avg_ret = row["avg_return"] * 100
    pos_pct = row["positive_pct"]
    month_name = month_names_tr[current_month] if lang == "tr" else month_names_en[current_month]

    if lang == "tr":
        comments.append(
            f"📅 Mevsimsel analiz: {asset_name} tarihsel olarak {month_name} ayında "
            f"ortalama %{avg_ret:+.1f} getiri sağlamış (son 2 yıla ağırlık verildi). "
            f"Pozitif kapanış oranı: %{pos_pct:.0f}."
        )

        best = seasonal.loc[seasonal["avg_return"].idxmax()]
        worst = seasonal.loc[seasonal["avg_return"].idxmin()]
        best_month = month_names_tr[int(best["month"])]
        worst_month = month_names_tr[int(worst["month"])]
        comments.append(
            f"📅 Tarihsel olarak en iyi ay: {best_month} (%{best['avg_return']*100:+.1f}), "
            f"en kötü ay: {worst_month} (%{worst['avg_return']*100:+.1f})."
        )
    else:
        comments.append(
            f"📅 Seasonal analysis: {asset_name} has historically returned "
            f"an average of {avg_ret:+.1f}% in {month_name} (recent 2 years weighted higher). "
            f"Positive close rate: {pos_pct:.0f}%."
        )

        best = seasonal.loc[seasonal["avg_return"].idxmax()]
        worst = seasonal.loc[seasonal["avg_return"].idxmin()]
        best_month = month_names_en[int(best["month"])]
        worst_month = month_names_en[int(worst["month"])]
        comments.append(
            f"📅 Historically best month: {best_month} ({best['avg_return']*100:+.1f}%), "
            f"worst month: {worst_month} ({worst['avg_return']*100:+.1f}%)."
        )

    # Warning if current month is historically volatile
    if row["std_return"] > 0.05:
        if lang == "tr":
            comments.append(
                f"⚠️ {month_name} ayı tarihsel olarak yüksek volatiliteye sahip. "
                f"Ani hareketlere hazırlıklı olmakta fayda var."
            )
        else:
            comments.append(
                f"⚠️ {month_name} has historically high volatility. "
                f"Be prepared for sudden movements."
            )

    return comments
