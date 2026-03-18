"""Dashboard page - market overview with summary table and mini charts."""

import streamlit as st
import pandas as pd

from config import ASSETS
from data.fetcher import get_latest_prices
from components.analysis_card import render_disclaimer
from utils.i18n import t, get_asset_name


def render_dashboard(data: dict[str, pd.DataFrame], lang: str = "tr"):
    """Render the main dashboard overview page."""
    st.header(t("dash_title", lang))

    prices = get_latest_prices(data)

    if not prices:
        st.warning(t("no_data", lang))
        return

    # Daily change cards
    daily_title = "📅 Günlük Değişim" if lang == "tr" else "📅 Daily Change"
    st.subheader(daily_title)

    high_lbl = "En Yüksek" if lang == "tr" else "High"
    low_lbl = "En Düşük" if lang == "tr" else "Low"

    price_items = list(prices.items())
    for row_start in range(0, len(price_items), 3):
        row = price_items[row_start:row_start + 3]
        cols = st.columns(len(row))
        for col, (key, info) in zip(cols, row):
            name = get_asset_name(key, ASSETS[key], lang)
            pct = info["change_pct"]
            change = info["change"]
            close = info["close"]
            high = info["high"]
            low = info["low"]
            is_up = pct >= 0
            arrow = "▲" if is_up else "▼"
            pct_color = "#26a69a" if is_up else "#ef5350"
            border_color = "#26a69a" if is_up else "#ef5350"
            # Vertical bar position (bottom = low, top = high)
            rng = high - low
            pos_pct = ((close - low) / rng * 100) if rng > 0 else 50
            bar_color = "#26a69a" if is_up else "#ef5350"
            with col:
                st.markdown(
                    f'<div style="background:#1e1e2e; padding:14px; border-radius:10px; '
                    f'border-left:4px solid {border_color}; margin-bottom:8px;">'
                    # Header: name + price
                    f'<div style="display:flex; justify-content:space-between; align-items:flex-start;">'
                    f'<div>'
                    f'<div style="color:#fff; font-size:1.05em; font-weight:bold; margin-bottom:4px;">{name}</div>'
                    f'<div style="font-size:1.3em; font-weight:bold; color:#fff;">${close:,.2f}</div>'
                    f'<div style="color:{pct_color}; font-size:1.05em; font-weight:bold; margin-top:4px;">'
                    f'{arrow} {pct:+.2f}%'
                    f'<span style="color:#888; font-size:0.75em; font-weight:normal; margin-left:6px;">'
                    f'${change:+,.2f}</span></div>'
                    f'</div>'
                    # Vertical bar
                    f'<div style="display:flex; flex-direction:column; align-items:center; min-width:40px;">'
                    f'<span style="color:#26a69a; font-size:0.65em; margin-bottom:2px;">${high:,.2f}</span>'
                    f'<div style="background:#333; border-radius:4px; width:8px; height:60px; position:relative; '
                    f'display:flex; flex-direction:column-reverse;">'
                    f'<div style="background:{bar_color}; width:100%; height:{pos_pct:.0f}%; '
                    f'border-radius:4px; min-height:2px;"></div></div>'
                    f'<span style="color:#ef5350; font-size:0.65em; margin-top:2px;">${low:,.2f}</span>'
                    f'</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # Weekly high/low summary
    st.subheader("📊 " + ("Son 1 Hafta: En Yüksek / En Düşük" if lang == "tr" else "Last 1 Week: High / Low"))
    items = [(k, df) for k, df in data.items() if df is not None and not df.empty]
    for row_start in range(0, len(items), 3):
        row_items = items[row_start:row_start + 3]
        cols = st.columns(len(row_items))
        for col, (key, df) in zip(cols, row_items):
            name = get_asset_name(key, ASSETS[key], lang)
            last_7 = df.tail(7)
            week_high = last_7["High"].max()
            week_low = last_7["Low"].min()
            current = last_7["Close"].iloc[-1]
            # Position within range (0-100%)
            rng = week_high - week_low
            pos_pct = ((current - week_low) / rng * 100) if rng > 0 else 50
            with col:
                high_lbl = "En Yüksek" if lang == "tr" else "High"
                low_lbl = "En Düşük" if lang == "tr" else "Low"
                now_lbl = "Şu An" if lang == "tr" else "Current"
                st.markdown(
                    f'<div style="background:#1e1e2e; padding:12px; border-radius:8px; margin-bottom:8px;">'
                    f'<div style="font-weight:bold; margin-bottom:8px; font-size:0.95em;">{name}</div>'
                    f'<div style="display:flex; justify-content:space-between; font-size:0.8em; color:#888;">'
                    f'<span>{low_lbl}</span><span>{high_lbl}</span></div>'
                    f'<div style="background:#333; border-radius:4px; height:8px; margin:4px 0; position:relative;">'
                    f'<div style="background:#FFD700; width:{pos_pct:.0f}%; height:100%; border-radius:4px;"></div></div>'
                    f'<div style="display:flex; justify-content:space-between; font-size:0.85em;">'
                    f'<span style="color:#ef5350;">${week_low:,.2f}</span>'
                    f'<span style="color:#aaa;">{now_lbl}: ${current:,.2f}</span>'
                    f'<span style="color:#26a69a;">${week_high:,.2f}</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Market sentiment summary
    st.markdown("---")
    st.subheader(t("dash_sentiment", lang))

    bullish = sum(1 for info in prices.values() if info["change_pct"] > 0)
    bearish = len(prices) - bullish
    total = len(prices)

    # Identify which assets are up/down for context
    up_assets = [get_asset_name(k, ASSETS[k], lang) for k, info in prices.items() if info["change_pct"] > 0]
    down_assets = [get_asset_name(k, ASSETS[k], lang) for k, info in prices.items() if info["change_pct"] < 0]
    flat_assets = [get_asset_name(k, ASSETS[k], lang) for k, info in prices.items() if info["change_pct"] == 0]

    # Biggest movers
    biggest_gainer = max(prices.items(), key=lambda x: x[1]["change_pct"])
    biggest_loser = min(prices.items(), key=lambda x: x[1]["change_pct"])
    gainer_name = get_asset_name(biggest_gainer[0], ASSETS[biggest_gainer[0]], lang)
    loser_name = get_asset_name(biggest_loser[0], ASSETS[biggest_loser[0]], lang)
    gainer_pct = biggest_gainer[1]["change_pct"]
    loser_pct = biggest_loser[1]["change_pct"]

    # Check DXY direction for context
    dxy_info = prices.get("DXY")
    dxy_up = dxy_info["change_pct"] > 0 if dxy_info else None

    # Check gold direction
    gold_info = prices.get("GOLD")
    gold_up = gold_info["change_pct"] > 0 if gold_info else None

    # Check BTC direction
    btc_info = prices.get("BTC")
    btc_up = btc_info["change_pct"] > 0 if btc_info else None

    # Check JPY direction (safe haven)
    jpy_info = prices.get("USDJPY")
    jpy_strengthening = jpy_info["change_pct"] < 0 if jpy_info else None  # USDJPY down = JPY strong

    # Check CHF direction (safe haven)
    chf_info = prices.get("USDCHF")
    chf_strengthening = chf_info["change_pct"] < 0 if chf_info else None  # USDCHF down = CHF strong

    if lang == "tr":
        sentiment_parts = []

        # Opening — overall tone
        if bullish > bearish:
            sentiment_parts.append(
                f"Bugün takip ettiğimiz {total} varlıktan {bullish} tanesi yükselişte, "
                f"{bearish} tanesi düşüşte. Genel tablo risk iştahının arttığına işaret ediyor."
            )
        elif bearish > bullish:
            sentiment_parts.append(
                f"Bugün {total} varlıktan {bearish} tanesi düşüşte, "
                f"sadece {bullish} tanesi artıda. Piyasada temkinli bir hava hakim."
            )
        else:
            sentiment_parts.append(
                f"Bugün piyasa kararsız: {bullish} varlık yükselişte, {bearish} varlık düşüşte. "
                f"Net bir yön belirlenemiyor."
            )

        # Biggest movers context
        sentiment_parts.append(
            f"Günün öne çıkanı **{gainer_name}** ({gainer_pct:+.2f}%) olurken, "
            f"en çok gerileyen **{loser_name}** ({loser_pct:+.2f}%). "
            f"Bu iki hareketin birlikte okunması önemli çünkü aralarındaki ilişki "
            f"piyasanın nereye baktığını gösteriyor."
        )

        # DXY + Gold inverse relationship
        if dxy_up is not None and gold_up is not None:
            if dxy_up and not gold_up:
                sentiment_parts.append(
                    "Doların güçlenirken altının gerilemesi klasik ters korelasyonun çalıştığını gösteriyor — "
                    "yatırımcılar güvenli liman olarak altın yerine doları tercih ediyor. Bu genelde "
                    "faiz beklentilerinin yukarı yönlü revize edildiği veya küresel risk algısının "
                    "değiştiği dönemlerde görülür."
                )
            elif not dxy_up and gold_up:
                sentiment_parts.append(
                    "Doların zayıflaması altını destekliyor — ters korelasyon beklendiği gibi çalışıyor. "
                    "Zayıflayan dolar, dolar dışı alıcılar için altını ucuzlatır ve talebi artırır. "
                    "Aynı zamanda dolar zayıflığı FED'in gevşeyeceği beklentisinin güçlendiğine işaret edebilir."
                )
            elif dxy_up and gold_up:
                sentiment_parts.append(
                    "İlginç bir durum: hem dolar hem altın aynı anda yükseliyor. Bu nadir görülür ve "
                    "genelde ciddi bir küresel belirsizlik veya jeopolitik risk olduğunda ortaya çıkar — "
                    "yatırımcılar her iki güvenli limana da aynı anda sığınıyor. Bu sinyal ciddiye alınmalı."
                )
            elif not dxy_up and not gold_up:
                sentiment_parts.append(
                    "Hem dolar hem altın düşüyor — bu da sıra dışı. Genelde likidite sıkışması veya "
                    "risk varlıklarına (hisse, kripto) güçlü bir rotasyon olduğunda görülür. "
                    "Yatırımcılar güvenli limanlardan çıkıp daha agresif pozisyon alıyor olabilir."
                )

        # BTC as risk appetite barometer
        if btc_up is not None:
            if btc_up and bullish > bearish:
                sentiment_parts.append(
                    "BTC'nin yükselişi risk iştahı tezini destekliyor — kripto piyasası küresel "
                    "likidite ve risk algısının en hassas barometrelerinden biri."
                )
            elif not btc_up and bearish > bullish:
                sentiment_parts.append(
                    "BTC'nin düşüşü genel risk-off tablosunu teyit ediyor. Kripto, yüksek beta "
                    "yapısıyla piyasa stresini erken ve abartılı yansıtma eğiliminde."
                )
            elif btc_up and bearish > bullish:
                sentiment_parts.append(
                    "Dikkat çekici: genel piyasa negatifken BTC pozitif. Bu ayrışma, kripto piyasasına "
                    "özgü bir katalizör olabileceğini veya akıllı paranın risk-on'a erken döndüğünü gösterebilir."
                )

        # JPY & CHF — safe haven barometers
        if jpy_strengthening is not None or chf_strengthening is not None:
            safe_haven_strong = []
            safe_haven_weak = []
            if jpy_strengthening is True:
                safe_haven_strong.append("Yen")
            elif jpy_strengthening is False:
                safe_haven_weak.append("Yen")
            if chf_strengthening is True:
                safe_haven_strong.append("İsviçre Frangı")
            elif chf_strengthening is False:
                safe_haven_weak.append("İsviçre Frangı")

            if len(safe_haven_strong) == 2:
                sentiment_parts.append(
                    "Hem Yen hem İsviçre Frangı dolara karşı güçleniyor — bu klasik güvenli liman "
                    "talebi sinyali. Yatırımcılar riskten kaçıyor ve geleneksel sığınaklara yöneliyor. "
                    "Bu hareket genelde küresel belirsizliğin arttığı dönemlerde belirginleşir."
                )
            elif len(safe_haven_weak) == 2:
                sentiment_parts.append(
                    "Hem Yen hem İsviçre Frangı dolara karşı zayıflıyor — güvenli liman talebi düşük. "
                    "Bu genelde risk iştahının güçlü olduğu ve sermayenin daha yüksek getirili "
                    "varlıklara aktığı dönemlerde görülür."
                )
            elif safe_haven_strong:
                sentiment_parts.append(
                    f"{safe_haven_strong[0]} güçlenirken {safe_haven_weak[0] if safe_haven_weak else 'diğeri'} "
                    f"zayıflıyor — güvenli liman sinyalleri karışık. Bu, bölgesel farklılıkları "
                    f"yansıtıyor olabilir."
                )

        commentary = " ".join(sentiment_parts)

        if bullish > bearish:
            st.markdown(
                f'<div style="background:rgba(38,166,154,0.1); border-left:4px solid #26a69a; '
                f'padding:16px; border-radius:8px;">'
                f'<div style="font-size:1.1em; margin-bottom:8px;">📈 <b>Piyasa Duyarlılığı: Pozitif</b></div>'
                f'<div style="color:#ccc; line-height:1.7;">{commentary}</div></div>',
                unsafe_allow_html=True,
            )
        elif bearish > bullish:
            st.markdown(
                f'<div style="background:rgba(239,83,80,0.1); border-left:4px solid #ef5350; '
                f'padding:16px; border-radius:8px;">'
                f'<div style="font-size:1.1em; margin-bottom:8px;">📉 <b>Piyasa Duyarlılığı: Negatif</b></div>'
                f'<div style="color:#ccc; line-height:1.7;">{commentary}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background:rgba(255,215,0,0.1); border-left:4px solid #FFD700; '
                f'padding:16px; border-radius:8px;">'
                f'<div style="font-size:1.1em; margin-bottom:8px;">➡️ <b>Piyasa Duyarlılığı: Kararsız</b></div>'
                f'<div style="color:#ccc; line-height:1.7;">{commentary}</div></div>',
                unsafe_allow_html=True,
            )

    else:  # English
        sentiment_parts = []

        if bullish > bearish:
            sentiment_parts.append(
                f"Today, {bullish} out of {total} tracked assets are in the green, "
                f"{bearish} in the red. The overall picture suggests risk appetite is elevated."
            )
        elif bearish > bullish:
            sentiment_parts.append(
                f"Today, {bearish} out of {total} assets are declining, "
                f"only {bullish} in the green. A cautious mood dominates."
            )
        else:
            sentiment_parts.append(
                f"Markets are undecided: {bullish} assets up, {bearish} down. No clear direction."
            )

        sentiment_parts.append(
            f"The standout mover is **{gainer_name}** ({gainer_pct:+.2f}%), while "
            f"**{loser_name}** ({loser_pct:+.2f}%) leads the decline. "
            f"Reading these two moves together is important — their relationship reveals "
            f"where the market's attention is focused."
        )

        if dxy_up is not None and gold_up is not None:
            if dxy_up and not gold_up:
                sentiment_parts.append(
                    "Dollar strength paired with gold weakness shows the classic inverse correlation at work — "
                    "investors are choosing the dollar over gold as their safe haven. This typically occurs when "
                    "rate expectations are being revised upward or global risk perception is shifting."
                )
            elif not dxy_up and gold_up:
                sentiment_parts.append(
                    "A weakening dollar is supporting gold — the inverse correlation is functioning as expected. "
                    "A softer dollar makes gold cheaper for non-dollar buyers, boosting demand. "
                    "It may also signal growing expectations of Fed easing."
                )
            elif dxy_up and gold_up:
                sentiment_parts.append(
                    "Both dollar and gold rising simultaneously is unusual and typically emerges during "
                    "serious global uncertainty or geopolitical risk — investors are seeking shelter "
                    "in both safe havens at once. This signal should be taken seriously."
                )
            elif not dxy_up and not gold_up:
                sentiment_parts.append(
                    "Both dollar and gold falling is also unusual — typically seen during liquidity squeezes "
                    "or strong rotation into risk assets (equities, crypto). Investors may be leaving "
                    "safe havens for more aggressive positions."
                )

        if btc_up is not None:
            if btc_up and bullish > bearish:
                sentiment_parts.append(
                    "BTC's rise supports the risk-on thesis — crypto is one of the most sensitive "
                    "barometers of global liquidity and risk appetite."
                )
            elif not btc_up and bearish > bullish:
                sentiment_parts.append(
                    "BTC's decline confirms the risk-off picture. Crypto, with its high-beta nature, "
                    "tends to reflect market stress early and amplified."
                )
            elif btc_up and bearish > bullish:
                sentiment_parts.append(
                    "Notable: BTC is positive while the broader market is negative. This divergence "
                    "could signal a crypto-specific catalyst or smart money rotating back to risk early."
                )

        # JPY & CHF — safe haven barometers
        if jpy_strengthening is not None or chf_strengthening is not None:
            safe_haven_strong = []
            safe_haven_weak = []
            if jpy_strengthening is True:
                safe_haven_strong.append("Yen")
            elif jpy_strengthening is False:
                safe_haven_weak.append("Yen")
            if chf_strengthening is True:
                safe_haven_strong.append("Swiss Franc")
            elif chf_strengthening is False:
                safe_haven_weak.append("Swiss Franc")

            if len(safe_haven_strong) == 2:
                sentiment_parts.append(
                    "Both Yen and Swiss Franc are strengthening against the dollar — a classic safe-haven "
                    "demand signal. Investors are de-risking and moving to traditional shelters. "
                    "This pattern typically intensifies during periods of rising global uncertainty."
                )
            elif len(safe_haven_weak) == 2:
                sentiment_parts.append(
                    "Both Yen and Swiss Franc are weakening against the dollar — safe-haven demand is low. "
                    "This typically occurs when risk appetite is strong and capital is flowing toward "
                    "higher-yielding assets."
                )
            elif safe_haven_strong:
                sentiment_parts.append(
                    f"{safe_haven_strong[0]} is strengthening while {safe_haven_weak[0] if safe_haven_weak else 'the other'} "
                    f"is weakening — mixed safe-haven signals, possibly reflecting regional divergences."
                )

        commentary = " ".join(sentiment_parts)

        if bullish > bearish:
            st.markdown(
                f'<div style="background:rgba(38,166,154,0.1); border-left:4px solid #26a69a; '
                f'padding:16px; border-radius:8px;">'
                f'<div style="font-size:1.1em; margin-bottom:8px;">📈 <b>Market Sentiment: Positive</b></div>'
                f'<div style="color:#ccc; line-height:1.7;">{commentary}</div></div>',
                unsafe_allow_html=True,
            )
        elif bearish > bullish:
            st.markdown(
                f'<div style="background:rgba(239,83,80,0.1); border-left:4px solid #ef5350; '
                f'padding:16px; border-radius:8px;">'
                f'<div style="font-size:1.1em; margin-bottom:8px;">📉 <b>Market Sentiment: Negative</b></div>'
                f'<div style="color:#ccc; line-height:1.7;">{commentary}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background:rgba(255,215,0,0.1); border-left:4px solid #FFD700; '
                f'padding:16px; border-radius:8px;">'
                f'<div style="font-size:1.1em; margin-bottom:8px;">➡️ <b>Market Sentiment: Mixed</b></div>'
                f'<div style="color:#ccc; line-height:1.7;">{commentary}</div></div>',
                unsafe_allow_html=True,
            )

    render_disclaimer(lang)
