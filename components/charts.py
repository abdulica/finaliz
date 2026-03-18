"""Plotly chart components for the Finaliz dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import ASSETS, TA_PARAMS


def create_candlestick_chart(
    df: pd.DataFrame,
    asset_key: str,
    show_ma: bool = True,
    show_bb: bool = True,
    show_volume: bool = True,
    lang: str = "tr",
) -> go.Figure:
    """Create a candlestick chart with optional overlays and sub-indicators."""
    color = ASSETS[asset_key]["color"]
    name = ASSETS[asset_key].get(f"name_{lang}", asset_key)

    rows = 3 if show_volume else 2
    heights = [0.5, 0.25, 0.25] if show_volume else [0.6, 0.4]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=heights,
        subplot_titles=[name, "RSI / Stoch RSI"] + (["MACD"] if show_volume else []),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name=name,
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # Moving averages
    if show_ma:
        ma_colors = {"SMA_20": "#FFA726", "SMA_50": "#42A5F5", "SMA_200": "#AB47BC"}
        for col_name, ma_color in ma_colors.items():
            if col_name in df:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df[col_name], name=col_name,
                        line=dict(width=1, color=ma_color),
                    ),
                    row=1, col=1,
                )

    # Bollinger Bands
    if show_bb and "BB_Upper" in df:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BB_Upper"], name="BB Upper",
                line=dict(width=1, color="rgba(128,128,128,0.5)", dash="dash"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BB_Lower"], name="BB Lower",
                line=dict(width=1, color="rgba(128,128,128,0.5)", dash="dash"),
                fill="tonexty", fillcolor="rgba(128,128,128,0.1)",
            ),
            row=1, col=1,
        )

    # RSI on row 2
    if "RSI" in df:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["RSI"], name="RSI",
                line=dict(width=1.5, color="#FF6F00"),
            ),
            row=2, col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.05, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.05, row=2, col=1)

    # MACD on row 3
    if show_volume and "MACD" in df:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["MACD"], name="MACD",
                line=dict(width=1.5, color="#2196F3"),
            ),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["MACD_Signal"], name="Signal",
                line=dict(width=1, color="#FF9800"),
            ),
            row=3, col=1,
        )
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_Hist"].fillna(0)]
        fig.add_trace(
            go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram", marker_color=colors),
            row=3, col=1,
        )

    fig.update_layout(
        height=700,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=30),
    )

    return fig


def create_forecast_chart(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    asset_key: str,
    lang: str = "tr",
) -> go.Figure:
    """Create forecast chart with actual prices, prediction, and confidence interval."""
    name = ASSETS[asset_key].get(f"name_{lang}", asset_key)

    fig = go.Figure()

    # Actual prices (last 90 days)
    recent = actual_df.tail(90)
    fig.add_trace(
        go.Scatter(
            x=recent["ds"], y=recent["y"],
            name="Gerçek" if lang == "tr" else "Actual",
            line=dict(color="#2196F3", width=2),
        )
    )

    # Forecast line
    last_actual_date = actual_df["ds"].iloc[-1]
    forecast_future = forecast_df[forecast_df["ds"] >= last_actual_date]

    fig.add_trace(
        go.Scatter(
            x=forecast_future["ds"], y=forecast_future["yhat"],
            name="Tahmin" if lang == "tr" else "Forecast",
            line=dict(color="#FF9800", width=2, dash="dash"),
        )
    )

    # Confidence interval
    fig.add_trace(
        go.Scatter(
            x=forecast_future["ds"], y=forecast_future["yhat_upper"],
            name="Üst Sınır" if lang == "tr" else "Upper Bound",
            line=dict(width=0), showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_future["ds"], y=forecast_future["yhat_lower"],
            name="Alt Sınır" if lang == "tr" else "Lower Bound",
            line=dict(width=0), showlegend=False,
            fill="tonexty", fillcolor="rgba(255, 152, 0, 0.15)",
        )
    )

    # Vertical line at forecast start
    annotation_label = "Tahmin Başlangıcı" if lang == "tr" else "Forecast Start"
    x_str = str(last_actual_date)
    fig.add_shape(
        type="line", x0=x_str, x1=x_str, y0=0, y1=1,
        yref="paper", line=dict(color="white", width=1, dash="dot"), opacity=0.5,
    )
    fig.add_annotation(
        x=x_str, y=1, yref="paper", text=annotation_label,
        showarrow=False, font=dict(color="white", size=10), yshift=10,
    )

    title = f"{name} - {'Tahmin' if lang == 'tr' else 'Forecast'}"
    fig.update_layout(
        title=title,
        height=500,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=30),
        xaxis_title="",
        yaxis_title="USD",
    )

    return fig


def create_correlation_heatmap(corr_matrix: pd.DataFrame, lang: str = "tr") -> go.Figure:
    """Create correlation heatmap."""
    title = "Korelasyon Matrisi" if lang == "tr" else "Correlation Matrix"

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )

    fig.update_layout(
        title=title,
        height=500,
        template="plotly_dark",
        margin=dict(l=100, r=20, t=60, b=100),
    )

    return fig


def create_seasonal_chart(
    seasonal_df: pd.DataFrame,
    asset_key: str,
    current_month: int,
    lang: str = "tr",
) -> go.Figure:
    """Create seasonal pattern bar chart."""
    name = ASSETS[asset_key].get(f"name_{lang}", asset_key)
    month_labels_tr = [
        "Oca", "Şub", "Mar", "Nis", "May", "Haz",
        "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara",
    ]
    month_labels_en = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    labels = month_labels_tr if lang == "tr" else month_labels_en

    colors = []
    for _, row in seasonal_df.iterrows():
        if int(row["month"]) == current_month:
            colors.append("#FFD700")  # Highlight current month
        elif row["avg_return"] >= 0:
            colors.append("#26a69a")
        else:
            colors.append("#ef5350")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[labels[int(m) - 1] for m in seasonal_df["month"]],
            y=seasonal_df["avg_return"] * 100,
            marker_color=colors,
            name="Ort. Getiri %" if lang == "tr" else "Avg Return %",
            text=[f"{v*100:.1f}%" for v in seasonal_df["avg_return"]],
            textposition="outside",
        )
    )

    title = f"{name} - {'Mevsimsel Patern' if lang == 'tr' else 'Seasonal Pattern'}"
    fig.update_layout(
        title=title,
        height=400,
        template="plotly_dark",
        yaxis_title="%" if lang == "tr" else "%",
        showlegend=False,
        margin=dict(l=50, r=20, t=60, b=30),
    )

    return fig


def create_comparison_chart(
    data: dict[str, pd.DataFrame],
    selected_assets: list[str],
    lang: str = "tr",
) -> go.Figure:
    """Create normalized comparison chart for multiple assets."""
    fig = go.Figure()

    for key in selected_assets:
        df = data.get(key)
        if df is None or df.empty:
            continue

        name = ASSETS[key].get(f"name_{lang}", key)
        color = ASSETS[key]["color"]

        # Normalize to percentage change from start
        normalized = (df["Close"] / df["Close"].iloc[0] - 1) * 100

        fig.add_trace(
            go.Scatter(
                x=df.index, y=normalized,
                name=name, line=dict(color=color, width=2),
            )
        )

    title = "Normalize Edilmiş Performans Karşılaştırması" if lang == "tr" else "Normalized Performance Comparison"
    fig.update_layout(
        title=title,
        height=500,
        template="plotly_dark",
        yaxis_title="% Değişim" if lang == "tr" else "% Change",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=30),
    )

    return fig


def create_relationship_chart(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    key_a: str,
    key_b: str,
    lang: str = "tr",
) -> go.Figure:
    """Create a dual-axis time series + scatter plot showing relationship between two assets."""
    name_a = ASSETS[key_a].get(f"name_{lang}", key_a)
    name_b = ASSETS[key_b].get(f"name_{lang}", key_b)
    color_a = ASSETS[key_a]["color"]
    color_b = ASSETS[key_b]["color"]

    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.55, 0.45],
        column_widths=[0.6, 0.4],
        subplot_titles=[
            f"{name_a} vs {name_b}" if lang == "en" else f"{name_a} ve {name_b}",
            "Scatter" if lang == "en" else "Saçılım",
            "Rolling Correlation (30d)" if lang == "en" else "Hareketli Korelasyon (30g)",
            "Return Distribution" if lang == "en" else "Getiri Dağılımı",
        ],
        specs=[
            [{"secondary_y": True}, {}],
            [{}, {}],
        ],
    )

    # --- Row 1, Col 1: Dual axis time series ---
    # Align dates
    common_idx = df_a.index.intersection(df_b.index)
    if len(common_idx) < 10:
        # Fallback: use all data separately
        fig.add_trace(
            go.Scatter(x=df_a.index, y=df_a["Close"], name=name_a,
                       line=dict(color=color_a, width=2)),
            row=1, col=1, secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=df_b.index, y=df_b["Close"], name=name_b,
                       line=dict(color=color_b, width=2)),
            row=1, col=1, secondary_y=True,
        )
        close_a = df_a["Close"]
        close_b = df_b["Close"]
    else:
        close_a = df_a.loc[common_idx, "Close"]
        close_b = df_b.loc[common_idx, "Close"]

        fig.add_trace(
            go.Scatter(x=common_idx, y=close_a, name=name_a,
                       line=dict(color=color_a, width=2)),
            row=1, col=1, secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=common_idx, y=close_b, name=name_b,
                       line=dict(color=color_b, width=2)),
            row=1, col=1, secondary_y=True,
        )

    fig.update_yaxes(title_text=name_a, row=1, col=1, secondary_y=False, color=color_a)
    fig.update_yaxes(title_text=name_b, row=1, col=1, secondary_y=True, color=color_b)

    # --- Row 1, Col 2: Scatter plot ---
    if len(common_idx) >= 10:
        ret_a = close_a.pct_change().dropna()
        ret_b = close_b.pct_change().dropna()
        common_ret = pd.DataFrame({"a": ret_a, "b": ret_b}).dropna()

        fig.add_trace(
            go.Scatter(
                x=common_ret["a"] * 100, y=common_ret["b"] * 100,
                mode="markers",
                marker=dict(color="#FFD700", size=4, opacity=0.5),
                name="Daily Returns" if lang == "en" else "Günlük Getiriler",
                showlegend=False,
            ),
            row=1, col=2,
        )
        # Trend line
        if len(common_ret) > 5:
            z = np.polyfit(common_ret["a"], common_ret["b"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(common_ret["a"].min(), common_ret["a"].max(), 50)
            fig.add_trace(
                go.Scatter(
                    x=x_line * 100, y=p(x_line) * 100,
                    mode="lines", line=dict(color="red", width=2, dash="dash"),
                    name="Trend", showlegend=False,
                ),
                row=1, col=2,
            )
        fig.update_xaxes(title_text=f"{name_a} %", row=1, col=2)
        fig.update_yaxes(title_text=f"{name_b} %", row=1, col=2)

    # --- Row 2, Col 1: Rolling correlation ---
    if len(common_idx) >= 35:
        ret_a = close_a.pct_change().dropna()
        ret_b = close_b.pct_change().dropna()
        rolling_corr = ret_a.rolling(30).corr(ret_b).dropna()

        fig.add_trace(
            go.Scatter(
                x=rolling_corr.index, y=rolling_corr,
                line=dict(color="#AB47BC", width=2),
                fill="tozeroy", fillcolor="rgba(171, 71, 188, 0.15)",
                name="Korelasyon" if lang == "tr" else "Correlation",
                showlegend=False,
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3, row=2, col=1)
        fig.add_hline(y=0.5, line_dash="dot", line_color="green", opacity=0.3, row=2, col=1)
        fig.add_hline(y=-0.5, line_dash="dot", line_color="red", opacity=0.3, row=2, col=1)

    # --- Row 2, Col 2: Return distribution overlay ---
    if len(common_idx) >= 10:
        ret_a = close_a.pct_change().dropna() * 100
        ret_b = close_b.pct_change().dropna() * 100

        fig.add_trace(
            go.Histogram(x=ret_a, name=name_a, marker_color=color_a,
                         opacity=0.6, nbinsx=40, showlegend=False),
            row=2, col=2,
        )
        fig.add_trace(
            go.Histogram(x=ret_b, name=name_b, marker_color=color_b,
                         opacity=0.6, nbinsx=40, showlegend=False),
            row=2, col=2,
        )
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title_text="%", row=2, col=2)

    fig.update_layout(
        height=800,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=30),
    )

    return fig


def create_mini_sparkline(series: pd.Series, color: str = "#2196F3") -> go.Figure:
    """Create a tiny sparkline chart for dashboard overview."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(series))), y=series.values,
            mode="lines", line=dict(color=color, width=1.5),
            fill="tozeroy", fillcolor=f"rgba(33, 150, 243, 0.1)",
        )
    )
    fig.update_layout(
        height=60, width=150,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
