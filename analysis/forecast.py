"""Forecast engine using Prophet with technical and macro signal filtering."""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from config import FORECAST_HORIZONS

# Suppress Prophet's verbose logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def run_forecast(
    df: pd.DataFrame,
    horizons: list[str] | None = None,
) -> dict | None:
    """Run Prophet forecast for specified horizons.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        horizons: List of horizon keys from FORECAST_HORIZONS. Defaults to all.

    Returns:
        Dict with forecast results per horizon, or None on failure.
    """
    if df is None or df.empty or len(df) < 60:
        return None

    if horizons is None:
        horizons = list(FORECAST_HORIZONS.keys())

    max_days = max(FORECAST_HORIZONS[h]["days"] for h in horizons)

    try:
        from prophet import Prophet

        # Prepare data for Prophet
        prophet_df = df[["Close"]].reset_index()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            interval_width=0.80,
        )
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=max_days)
        forecast = model.predict(future)

        results = {}
        last_actual = df["Close"].iloc[-1]
        last_date = df.index[-1]

        for h_key in horizons:
            days = FORECAST_HORIZONS[h_key]["days"]
            target_idx = len(prophet_df) - 1 + days

            if target_idx >= len(forecast):
                continue

            row = forecast.iloc[target_idx]
            predicted = row["yhat"]
            upper = row["yhat_upper"]
            lower = row["yhat_lower"]
            change_pct = (predicted - last_actual) / last_actual * 100

            if change_pct > 1:
                direction = "up"
            elif change_pct < -1:
                direction = "down"
            else:
                direction = "sideways"

            results[h_key] = {
                "predicted": predicted,
                "upper": upper,
                "lower": lower,
                "last_actual": last_actual,
                "change": predicted - last_actual,
                "change_pct": change_pct,
                "direction": direction,
                "target_date": row["ds"],
            }

        # Store full forecast for charting
        results["_full_forecast"] = forecast
        results["_actual_data"] = prophet_df

        return results

    except Exception as e:
        logging.error(f"Forecast failed: {e}")
        return None


def generate_forecast_commentary(
    forecast_results: dict,
    asset_name: str,
    technical_signal: str,
    lang: str = "tr",
) -> list[str]:
    """Generate human-readable forecast commentary.

    Args:
        forecast_results: Output from run_forecast().
        asset_name: Display name of the asset.
        technical_signal: Overall technical signal ('buy', 'sell', 'neutral').
        lang: Language code.
    """
    if not forecast_results:
        return []

    comments = []
    horizon_labels_tr = {
        "1d": "yarın", "3d": "3 gün içinde", "1w": "bir hafta içinde",
        "2w": "iki hafta içinde", "4w": "bir ay içinde",
    }
    horizon_labels_en = {
        "1d": "tomorrow", "3d": "in 3 days", "1w": "in one week",
        "2w": "in two weeks", "4w": "in one month",
    }

    direction_tr = {"up": "yükseliş", "down": "düşüş", "sideways": "yatay"}
    direction_en = {"up": "bullish", "down": "bearish", "sideways": "sideways"}

    # Short-term vs long-term direction comparison
    short_dir = forecast_results.get("1d", {}).get("direction")
    long_dir = forecast_results.get("4w", {}).get("direction")

    for h_key in ["1d", "3d", "1w", "2w", "4w"]:
        result = forecast_results.get(h_key)
        if result is None:
            continue

        h_label = horizon_labels_tr.get(h_key) if lang == "tr" else horizon_labels_en.get(h_key)
        dir_label = direction_tr[result["direction"]] if lang == "tr" else direction_en[result["direction"]]

        if lang == "tr":
            comments.append(
                f"🔮 **{h_label.capitalize()}** ({FORECAST_HORIZONS[h_key]['label_tr']}): "
                f"Tahmini fiyat **{result['predicted']:.2f}** "
                f"(%{result['change_pct']:+.1f}, {dir_label}). "
                f"Güven aralığı: {result['lower']:.2f} - {result['upper']:.2f}."
            )
        else:
            comments.append(
                f"🔮 **{h_label.capitalize()}** ({FORECAST_HORIZONS[h_key]['label_en']}): "
                f"Predicted price **{result['predicted']:.2f}** "
                f"({result['change_pct']:+.1f}%, {dir_label}). "
                f"Confidence interval: {result['lower']:.2f} - {result['upper']:.2f}."
            )

    # Forecast vs technical analysis alignment
    if short_dir and technical_signal:
        ta_agrees = (
            (short_dir == "up" and technical_signal == "buy")
            or (short_dir == "down" and technical_signal == "sell")
        )
        if ta_agrees:
            if lang == "tr":
                comments.append(
                    f"✅ Tahmin modeli ve teknik analiz aynı yönü gösteriyor. "
                    f"Bu uyum genelde sinyalin güvenilirliğini artırır."
                )
            else:
                comments.append(
                    f"✅ The forecast model and technical analysis agree on direction. "
                    f"This alignment typically increases signal reliability."
                )
        elif short_dir != "sideways" and technical_signal != "neutral":
            if lang == "tr":
                comments.append(
                    f"⚠️ Dikkat: Tahmin modeli ve teknik analiz farklı yönlere işaret ediyor. "
                    f"Model {direction_tr.get(short_dir, short_dir)} derken, "
                    f"teknik göstergeler {'alım' if technical_signal == 'buy' else 'satım'} sinyali veriyor. "
                    f"Bu çelişki, piyasanın kararsız olduğuna işaret edebilir."
                )
            else:
                comments.append(
                    f"⚠️ Note: The forecast model and technical analysis point in different directions. "
                    f"The model suggests {direction_en.get(short_dir, short_dir)}, "
                    f"while technical indicators signal {'buy' if technical_signal == 'buy' else 'sell'}. "
                    f"This divergence may indicate market indecision."
                )

    # Short vs long term divergence
    if short_dir and long_dir and short_dir != long_dir and short_dir != "sideways" and long_dir != "sideways":
        if lang == "tr":
            comments.append(
                f"🤔 İlginç bir ayrışma var: Kısa vadede {direction_tr[short_dir]} beklentisi, "
                f"uzun vadede ise {direction_tr[long_dir]}. Bu tür ayrışmalar genelde "
                f"bir düzeltme veya trend değişimi öncesine denk gelir."
            )
        else:
            comments.append(
                f"🤔 Interesting divergence: Short-term outlook is {direction_en[short_dir]}, "
                f"while long-term is {direction_en[long_dir]}. Such divergences often "
                f"precede a correction or trend change."
            )

    return comments
