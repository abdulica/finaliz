"""Macroeconomic analysis and commentary generation."""

import numpy as np
import pandas as pd

from config import MACRO_SERIES


def analyze_macro_environment(macro_data: dict[str, pd.DataFrame], lang: str = "tr") -> list[str]:
    """Analyze current macro environment and generate commentary.

    Returns list of commentary paragraphs.
    """
    comments = []

    # FED Rate analysis
    fed = macro_data.get("FED_RATE")
    if fed is not None and not fed.empty:
        current = fed["value"].iloc[-1]
        prev_6m = fed["value"].iloc[-min(6, len(fed))]
        trend = "rising" if current > prev_6m else "falling" if current < prev_6m else "stable"

        if lang == "tr":
            trend_tr = {"rising": "yükseliyor", "falling": "düşüyor", "stable": "sabit"}[trend]
            comments.append(
                f"🏦 FED faiz oranı şu an %{current:.2f} seviyesinde ve {trend_tr}. "
                + _fed_impact_commentary(trend, lang)
            )
        else:
            comments.append(
                f"🏦 Fed funds rate is currently at {current:.2f}% and {trend}. "
                + _fed_impact_commentary(trend, lang)
            )

    # CPI / Inflation
    cpi = macro_data.get("CPI")
    if cpi is not None and len(cpi) > 12:
        current_cpi = cpi["value"].iloc[-1]
        year_ago_cpi = cpi["value"].iloc[-12]
        yoy_inflation = (current_cpi - year_ago_cpi) / year_ago_cpi * 100

        if lang == "tr":
            comments.append(
                f"📊 Yıllık enflasyon (CPI bazlı) %{yoy_inflation:.1f} seviyesinde. "
                + _inflation_impact(yoy_inflation, lang)
            )
        else:
            comments.append(
                f"📊 Year-over-year inflation (CPI-based) is at {yoy_inflation:.1f}%. "
                + _inflation_impact(yoy_inflation, lang)
            )

    # VIX / Fear index
    vix = macro_data.get("VIX")
    if vix is not None and not vix.empty:
        current_vix = vix["value"].iloc[-1]
        if lang == "tr":
            comments.append(
                f"😰 VIX (Korku Endeksi) {current_vix:.1f} seviyesinde. "
                + _vix_commentary(current_vix, lang)
            )
        else:
            comments.append(
                f"😰 VIX (Fear Index) is at {current_vix:.1f}. "
                + _vix_commentary(current_vix, lang)
            )

    # US 10Y
    us10y = macro_data.get("US10Y")
    if us10y is not None and not us10y.empty:
        current_yield = us10y["value"].iloc[-1]
        if lang == "tr":
            comments.append(
                f"📈 ABD 10 yıllık tahvil faizi %{current_yield:.2f} seviyesinde. "
                + _yield_commentary(current_yield, lang)
            )
        else:
            comments.append(
                f"📈 US 10-year Treasury yield is at {current_yield:.2f}%. "
                + _yield_commentary(current_yield, lang)
            )

    # M2 Money Supply
    m2 = macro_data.get("M2")
    if m2 is not None and len(m2) > 12:
        current_m2 = m2["value"].iloc[-1]
        year_ago_m2 = m2["value"].iloc[-12]
        m2_growth = (current_m2 - year_ago_m2) / year_ago_m2 * 100

        if lang == "tr":
            comments.append(
                f"💰 M2 para arzı yıllık %{m2_growth:.1f} değişim gösteriyor. "
                + _m2_commentary(m2_growth, lang)
            )
        else:
            comments.append(
                f"💰 M2 money supply has changed {m2_growth:.1f}% year-over-year. "
                + _m2_commentary(m2_growth, lang)
            )

    # --- Overall market sentiment summary ---
    if len(comments) >= 3:
        summary = _build_sentiment_summary(macro_data, lang)
        if summary:
            comments.append(summary)

    return comments


def compute_macro_correlations(
    asset_data: dict[str, pd.DataFrame],
    macro_data: dict[str, pd.DataFrame],
) -> pd.DataFrame | None:
    """Compute correlation matrix between assets and macro indicators."""
    series_dict = {}

    for key, df in asset_data.items():
        if df is not None and not df.empty:
            monthly = df["Close"].resample("M").last()
            pct = monthly.pct_change().dropna()
            if len(pct) > 3:
                series_dict[key] = pct

    for key, df in macro_data.items():
        if df is not None and not df.empty:
            monthly = df["value"].resample("M").last()
            pct = monthly.pct_change().dropna()
            if len(pct) > 3:
                label = MACRO_SERIES.get(key, {}).get("name_en", key)
                series_dict[label] = pct

    if len(series_dict) < 2:
        return None

    combined = pd.DataFrame(series_dict).dropna()
    if len(combined) < 3:
        return None

    return combined.corr()


def _build_sentiment_summary(macro_data: dict, lang: str) -> str | None:
    """Build an overall market sentiment paragraph combining all macro signals."""
    signals = []  # (factor, direction) — direction: "risk_on", "risk_off", "neutral"

    fed = macro_data.get("FED_RATE")
    if fed is not None and not fed.empty:
        current = fed["value"].iloc[-1]
        prev = fed["value"].iloc[-min(6, len(fed))]
        if current > prev:
            signals.append(("fed", "risk_off"))
        elif current < prev:
            signals.append(("fed", "risk_on"))
        else:
            signals.append(("fed", "neutral"))

    vix = macro_data.get("VIX")
    if vix is not None and not vix.empty:
        v = vix["value"].iloc[-1]
        if v > 30:
            signals.append(("vix", "risk_off"))
        elif v > 20:
            signals.append(("vix", "neutral"))
        else:
            signals.append(("vix", "risk_on"))

    us10y = macro_data.get("US10Y")
    if us10y is not None and not us10y.empty:
        y = us10y["value"].iloc[-1]
        if y > 4.5:
            signals.append(("yield", "risk_off"))
        elif y > 3:
            signals.append(("yield", "neutral"))
        else:
            signals.append(("yield", "risk_on"))

    m2 = macro_data.get("M2")
    if m2 is not None and len(m2) > 12:
        growth = (m2["value"].iloc[-1] - m2["value"].iloc[-12]) / m2["value"].iloc[-12] * 100
        if growth > 5:
            signals.append(("m2", "risk_on"))
        elif growth < 0:
            signals.append(("m2", "risk_off"))
        else:
            signals.append(("m2", "neutral"))

    if not signals:
        return None

    risk_on = sum(1 for _, d in signals if d == "risk_on")
    risk_off = sum(1 for _, d in signals if d == "risk_off")
    neutral = sum(1 for _, d in signals if d == "neutral")
    total = len(signals)

    if lang == "tr":
        header = "🧭 **Genel Piyasa Duyarlılığı:** "
        if risk_off > risk_on and risk_off > neutral:
            return header + (
                f"Makro göstergelerin çoğunluğu ({risk_off}/{total}) risk-off sinyali veriyor. "
                f"Yükselen faizler, sıkışan likidite veya artan korku — bu kombinasyon genelde "
                f"güvenli liman arayışını artırır. Dolar ve ABD tahvilleri bu ortamda güçlenir, "
                f"altın karışık sinyaller alır (güvenli liman talebi vs yüksek faiz baskısı), "
                f"gelişen piyasa paraları ve riskli varlıklar baskı altında kalır. "
                f"Bu tablo kısa vadede 'savunmacı' bir duruş gerektiriyor."
            )
        elif risk_on > risk_off and risk_on > neutral:
            return header + (
                f"Makro göstergelerin çoğunluğu ({risk_on}/{total}) risk-on ortamına işaret ediyor. "
                f"Gevşeyen para politikası, artan likidite veya düşük korku — bu kombinasyon "
                f"genelde hisse senetleri, emtialar ve kripto gibi risk varlıklarını destekler. "
                f"Dolar bu ortamda zayıflama eğiliminde olur çünkü sermaye daha yüksek getiri "
                f"arayışıyla riskli varlıklara yönelir. Altın da genelde fayda görür çünkü "
                f"fırsat maliyeti düşer."
            )
        else:
            return header + (
                f"Makro göstergeler karışık sinyaller veriyor — net bir risk-on veya risk-off "
                f"tablosu yok. Bu tür geçiş dönemlerinde piyasa genelde yönsüz hareket eder "
                f"ve tek bir güçlü veri veya merkez bankası açıklaması dengeleri aniden değiştirebilir. "
                f"Pozisyon büyüklüğünü küçük tutmak ve esnekliği korumak bu dönemlerde daha mantıklı."
            )
    else:
        header = "🧭 **Overall Market Sentiment:** "
        if risk_off > risk_on and risk_off > neutral:
            return header + (
                f"The majority of macro signals ({risk_off}/{total}) point to a risk-off environment. "
                f"Rising rates, tightening liquidity, or elevated fear — this combination typically "
                f"drives safe-haven demand. The dollar and US Treasuries strengthen, gold gets mixed "
                f"signals (safe-haven demand vs high-rate pressure), and emerging market currencies "
                f"and risky assets come under pressure. This backdrop calls for a defensive posture."
            )
        elif risk_on > risk_off and risk_on > neutral:
            return header + (
                f"The majority of macro signals ({risk_on}/{total}) suggest a risk-on environment. "
                f"Easing monetary policy, expanding liquidity, or low fear — this combination typically "
                f"supports equities, commodities, and crypto. The dollar tends to weaken as capital "
                f"seeks higher returns in riskier assets. Gold also tends to benefit as opportunity "
                f"cost decreases."
            )
        else:
            return header + (
                f"Macro signals are mixed — no clear risk-on or risk-off picture. In these transition "
                f"periods, markets tend to drift sideways, and a single strong data point or central bank "
                f"statement can suddenly shift the balance. Keeping position sizes small and maintaining "
                f"flexibility makes more sense in these periods."
            )


def _fed_impact_commentary(trend: str, lang: str) -> str:
    if lang == "tr":
        if trend == "rising":
            return (
                "Yükselen faizler küresel sermaye akışlarını doğrudan etkiliyor — borçlanma maliyeti arttıkça "
                "şirketler ve devletler daha temkinli davranmak zorunda kalıyor. Dolar cephesinde bu genelde "
                "güçlenme anlamına gelir çünkü yüksek getiri küresel yatırımcıları ABD tahvillerine çeker. "
                "Altın ve BTC gibi getirisi olmayan varlıklar ise baskı altına girer — neden risksiz %5 varken "
                "volatil bir varlıkta durasın? Ama burada kritik bir ayrım var: piyasa faiz artışını zaten "
                "fiyatlamışsa, asıl hareket FED'in beklentileri aştığı veya altında kaldığı anlarda gelir."
            )
        elif trend == "falling":
            return (
                "Düşen faizler piyasada ciddi bir domino etkisi yaratır. Önce borçlanma ucuzlar, "
                "sonra şirketler ve tüketiciler daha rahat harcamaya başlar, bu da ekonomik aktiviteyi canlandırır. "
                "Dolar zayıflar çünkü ABD tahvillerinin getirisi azalır ve sermaye daha riskli ama getirisi yüksek "
                "varlıklara yönelir — altın, gelişen piyasalar, kripto. Ama dikkat: FED faiz indiriyorsa, "
                "bunun sebebi ekonominin yavaşlaması olabilir. Yani 'iyi haber' gibi görünen faiz indirimi, "
                "aslında ekonomik zayıflığın bir itirafı da olabilir."
            )
        return (
            "Sabit faiz ortamı görünüşte sakin bir tablo çizer ama aldatıcı olabilir. "
            "Piyasalar belirsizlikten nefret eder ve 'ne yapacak?' sorusu havada kaldıkça "
            "pozisyon alma zorlaşır. Bu tür dönemlerde genelde volatilite düşer ama biriken enerji, "
            "FED'in bir sonraki hamlesinde sert şekilde boşalabilir."
        )
    else:
        if trend == "rising":
            return (
                "Rising rates reshape global capital flows — as borrowing costs increase, corporations and "
                "governments become more cautious. For the dollar, this typically means strength since higher "
                "yields attract global investors to US Treasuries. Non-yielding assets like gold and BTC face "
                "pressure — why hold volatile assets when you can get risk-free 5%? But here's the key nuance: "
                "if markets have already priced in the hikes, the real moves come when the Fed exceeds or "
                "falls short of expectations."
            )
        elif trend == "falling":
            return (
                "Falling rates trigger a significant domino effect. Borrowing gets cheaper, then companies "
                "and consumers spend more freely, boosting economic activity. The dollar weakens as US Treasury "
                "yields drop and capital flows toward riskier, higher-yielding assets — gold, emerging markets, "
                "crypto. But be careful: if the Fed is cutting, it might be because the economy is slowing. "
                "What looks like 'good news' could actually be an admission of economic weakness."
            )
        return (
            "A stable rate environment looks calm on the surface but can be deceptive. Markets hate "
            "uncertainty, and when the 'what will they do?' question lingers, positioning becomes difficult. "
            "Volatility tends to drop in these periods, but the pent-up energy can release violently "
            "when the Fed finally makes its next move."
        )


def _inflation_impact(inflation: float, lang: str) -> str:
    if lang == "tr":
        if inflation > 5:
            return (
                "Bu seviyede enflasyon artık 'geçici' denilemeyecek noktada. Tüketicinin alım gücü eriyor, "
                "şirket marjları baskı altında ve merkez bankası üzerindeki sıkılaştırma baskısı artıyor. "
                "Altın tarihsel olarak yüksek enflasyon dönemlerinde parlar çünkü 'değer saklama aracı' "
                "rolünü üstlenir. Ancak FED şahinleşirse ve agresif faiz artırırsa, altını bile baskılayabilir. "
                "Emtialarda ise durum karışık: enflasyon emtia fiyatlarını yükseltir ama talep yavaşlarsa "
                "bu döngü kırılabilir. BTC'nin enflasyon hedge'i olup olmadığı hâlâ tartışmalı — "
                "kısa vadede daha çok risk iştahıyla hareket ediyor."
            )
        elif inflation > 3:
            return (
                "Orta seviye enflasyon, piyasalar için en belirsiz alan. Ne yeterince düşük ki FED rahat etsin, "
                "ne yeterince yüksek ki herkes aynı tarafa koşsun. Bu belirsizlik 'FED ne yapacak?' sorusunu "
                "her toplantı öncesi ana gündem maddesi yapar. Dolar bu bant aralığında genelde güçlü kalır "
                "çünkü FED'in sıkılaştırma eğilimi devam eder, ama risk varlıkları da tamamen çökmez "
                "çünkü ekonomi hâlâ büyüyor."
            )
        return (
            "Düşük enflasyon ortamı merkez bankalarının elini rahatlatır — faiz indirimi veya likidite "
            "genişlemesi için zemin hazırlar. Bu genelde hisse senetleri ve risk varlıkları için olumludur. "
            "Altın için ise karışık: enflasyon koruması argümanı zayıflar ama düşük faiz altını destekler. "
            "Asıl soru şu: düşük enflasyon ekonomik sağlığın mı yoksa talep zayıflığının mı göstergesi?"
        )
    else:
        if inflation > 5:
            return (
                "At this level, inflation can no longer be called 'transitory.' Consumer purchasing power "
                "is eroding, corporate margins are under pressure, and the central bank faces mounting pressure "
                "to tighten. Gold historically shines during high inflation as a store of value. But if the Fed "
                "turns hawkish and hikes aggressively, even gold can get crushed. Commodities are mixed: "
                "inflation pushes prices up but if demand slows, the cycle can break. Whether BTC is an "
                "inflation hedge remains debatable — short-term, it moves more with risk appetite."
            )
        elif inflation > 3:
            return (
                "Moderate inflation is the most uncertain zone. Not low enough for the Fed to relax, "
                "not high enough for everyone to run in the same direction. This ambiguity makes 'what will "
                "the Fed do?' the main agenda item before every meeting. The dollar tends to stay strong "
                "in this range as the Fed's tightening bias continues, but risk assets don't fully collapse "
                "because the economy is still growing."
            )
        return (
            "Low inflation gives central banks room to maneuver — setting the stage for rate cuts or "
            "liquidity expansion. This is generally positive for equities and risk assets. For gold, "
            "it's mixed: the inflation-hedge argument weakens but low rates support it. The real question: "
            "is low inflation a sign of economic health or demand weakness?"
        )


def _vix_commentary(vix: float, lang: str) -> str:
    if lang == "tr":
        if vix > 30:
            return (
                "Bu seviye ciddi piyasa stresine işaret ediyor. VIX 30 üzerine çıktığında tarihsel olarak "
                "piyasalarda panik satışları yoğunlaşır, likidite kurur ve spreadler açılır. Paradoks şu ki "
                "en iyi alım fırsatları genelde tam da bu korkunun zirve yaptığı dönemlerde ortaya çıkar — "
                "ama tabii zamanlamak son derece zor. Altın bu dönemlerde güvenli liman olarak parlar, "
                "dolar da güçlenir çünkü küresel sermaye ABD'ye sığınır. BTC ise 2020 sonrasında "
                "geleneksel risk varlığı gibi davranmaya başladı — yüksek VIX'te genelde düşer."
            )
        elif vix > 20:
            return (
                "Piyasa tedirgin ama henüz panik yok. VIX 20-30 bandı 'dikkatli ol' diyen bir alan — "
                "yatırımcılar pozisyon küçültüyor ama henüz toplu çıkış yok. Bu aralıkta genelde "
                "haber akışına duyarlılık artar: tek bir olumsuz veri veya jeopolitik gelişme VIX'i "
                "hızla 30'un üzerine taşıyabilir. Öte yandan, piyasa bu tedirginliğe rağmen toparlanırsa "
                "bu güçlü bir sinyal olur — korkunun içinden yükselen boğa rallileri en sert olanlarıdır."
            )
        return (
            "Düşük VIX 'herkes rahat' demek — ama bu tam da dikkatli olunması gereken an. "
            "Tarihsel olarak VIX'in uzun süre düşük kaldığı dönemlerin ardından ani ve sert yükselişler "
            "gelmiştir. Piyasa katılımcıları rehavete kapılır, kaldıraç artar, risk primleri sıkışır. "
            "Sonra tek bir tetikleyici — beklenmedik bir veri, jeopolitik kriz, merkez bankası sürprizi — "
            "ve volatilite patlar. Düşük VIX fırtına öncesi sessizlik olabilir."
        )
    else:
        if vix > 30:
            return (
                "This level signals serious market stress. When VIX crosses 30, historically panic selling "
                "intensifies, liquidity dries up, and spreads widen. The paradox: the best buying opportunities "
                "often emerge exactly when fear peaks — but timing it is extremely difficult. Gold shines "
                "as a safe haven in these periods, and the dollar strengthens as global capital seeks "
                "shelter in the US. BTC, since 2020, has behaved more like a traditional risk asset — "
                "it generally falls during high VIX episodes."
            )
        elif vix > 20:
            return (
                "Markets are nervous but not panicking. The 20-30 VIX band is the 'stay alert' zone — "
                "investors are trimming positions but there's no mass exodus yet. Sensitivity to news flow "
                "increases: a single negative data point or geopolitical event can quickly push VIX above 30. "
                "On the flip side, if markets recover despite this unease, it's a strong signal — "
                "bull rallies born from fear tend to be the most powerful."
            )
        return (
            "Low VIX means 'everyone is comfortable' — but that's precisely when you should be careful. "
            "Historically, prolonged low VIX periods have been followed by sudden, sharp spikes. Market "
            "participants become complacent, leverage increases, risk premiums compress. Then a single "
            "trigger — unexpected data, geopolitical crisis, central bank surprise — and volatility "
            "explodes. Low VIX can be the calm before the storm."
        )


def _yield_commentary(yield_val: float, lang: str) -> str:
    if lang == "tr":
        if yield_val > 4.5:
            return (
                "Bu seviyede tahviller ciddi bir rakip haline geliyor. Düşün: neredeyse risksiz %4.5+ getiri "
                "varken, neden hisse senedi riskini veya altının volatilitesini taşıyasın? Yüksek tahvil faizi "
                "aynı zamanda mortgage, kredi kartı ve kurumsal borçlanma maliyetlerini artırır — bu da "
                "ekonomik yavaşlama riskini beraberinde getirir. Gelişen piyasalar için durum daha da zor: "
                "ABD tahvillerinin getirisi yükseldikçe sermaye bu ülkelerden çıkar, paraları değer kaybeder "
                "ve dolarla borçları ağırlaşır. DXY'nin yükselmesinin arkasındaki en güçlü itici güçlerden "
                "biri budur."
            )
        elif yield_val > 3:
            return (
                "Orta seviye tahvil faizi piyasalar için 'ne iyi ne kötü' alanı. Tahviller henüz "
                "hisse senetleriyle tam rekabet edemiyor ama artık göz ardı da edilemiyor. Bu bant aralığında "
                "yatırımcılar genelde dengeli portföy tercih eder. Altın için kritik olan nominal faizden "
                "çok reel faizdir (nominal - enflasyon). Eğer enflasyon faizin üzerindeyse, negatif reel faiz "
                "altını destekler. Pozitif reel faiz ise altının cazibesini azaltır."
            )
        return (
            "Düşük tahvil faizi, piyasalara 'para ucuz, risk al' mesajı verir. Bu ortamda hisse senetleri, "
            "gayrimenkul ve kripto gibi risk varlıkları parlama eğilimindedir. Altın için de destekleyici "
            "çünkü fırsat maliyeti düşük — tahvilde kazanacağın çok az olduğunda altın tutmanın bedeli "
            "de azalır. Ama düşük faizin sebebi önemli: ekonomi güçlüyken düşük faiz muhteşem, "
            "ekonomi çökerken düşük faiz çaresizliğin işareti."
        )
    else:
        if yield_val > 4.5:
            return (
                "At this level, bonds become serious competition. Think about it: why take equity risk or "
                "gold's volatility when you can get nearly risk-free 4.5%+? High yields also raise mortgage, "
                "credit card, and corporate borrowing costs — bringing economic slowdown risk. For emerging "
                "markets it's even tougher: as US Treasury yields rise, capital flows out, currencies weaken, "
                "and dollar-denominated debt gets heavier. This is one of the strongest drivers behind "
                "DXY strength."
            )
        elif yield_val > 3:
            return (
                "Moderate bond yields create a 'neither good nor bad' zone. Bonds aren't fully competing "
                "with equities yet but can't be ignored either. Investors tend to prefer balanced portfolios "
                "in this range. For gold, what matters isn't nominal yields but real yields "
                "(nominal - inflation). If inflation exceeds the yield, negative real rates support gold. "
                "Positive real rates reduce gold's appeal."
            )
        return (
            "Low bond yields send a 'money is cheap, take risk' message. Equities, real estate, and crypto "
            "tend to shine in this environment. It's also supportive for gold since the opportunity cost "
            "is low — when bonds pay almost nothing, holding gold costs less. But the reason behind low "
            "rates matters: low rates in a strong economy are great; low rates in a collapsing economy "
            "are a sign of desperation."
        )


def _m2_commentary(growth: float, lang: str) -> str:
    if lang == "tr":
        if growth > 5:
            return (
                "Para arzındaki bu genişleme sisteme ciddi likidite enjekte ediyor. Tarihsel olarak "
                "M2 artışı ile varlık fiyatları arasında güçlü bir korelasyon var — 2020-2021'deki "
                "her şeyin fırladığı dönem tam da M2'nin patlama yaptığı dönemdi. Daha fazla para "
                "aynı miktarda varlığı kovalıyor demek, bu da fiyatları mekanik olarak yukarı iter. "
                "Ama madalyonun diğer yüzü enflasyon: sistem fazla parayla dolduğunda tüketici fiyatları "
                "da kaçınılmaz olarak yükselir. Bu da FED'i tekrar sıkılaştırmaya zorlayabilir. "
                "Altın ve BTC bu dönemlerde genelde güçlüdür çünkü paranın değer kaybına karşı "
                "korunma aracı olarak görülürler."
            )
        elif growth < 0:
            return (
                "Para arzı daralması piyasalar için ciddi bir uyarı sinyali. Sistemdeki likidite azalıyor — "
                "bu kredi sıkışması, tüketim yavaşlaması ve varlık fiyatlarında düşüş anlamına gelebilir. "
                "2022'deki kripto kışı ve teknoloji hisselerindeki çöküş tam da M2 daralmaya geçtiği "
                "dönemde yaşandı. Bu tesadüf değil. Daha az para aynı varlıkları kovalıyor demek, "
                "bu da değerlemeleri mekanik olarak aşağı çeker. Dolar bu dönemlerde güçlenir çünkü "
                "nakit kıt hale gelir ve herkes dolara sığınır."
            )
        return (
            "Ilımlı para arzı büyümesi, ekonomi için en sağlıklı senaryo. Ne aşırı likidite var ki "
            "balonlar oluşsun, ne de daralma var ki sistem çöksün. Bu ortamda varlık fiyatları "
            "genelde temel değerlerine yakın seyreder ve piyasa hareketleri makro veriler ile "
            "şirket performansına daha duyarlı hale gelir. Yani 'sıkıcı' ama aslında en sağlıklı dönem."
        )
    else:
        if growth > 5:
            return (
                "This expansion is injecting serious liquidity into the system. Historically, M2 growth "
                "strongly correlates with asset prices — the 2020-2021 'everything rally' coincided exactly "
                "with the M2 explosion. More money chasing the same amount of assets mechanically pushes "
                "prices up. But the flip side is inflation: when the system floods with money, consumer prices "
                "inevitably rise, potentially forcing the Fed to tighten again. Gold and BTC tend to be "
                "strong in these periods as they're seen as hedges against currency debasement."
            )
        elif growth < 0:
            return (
                "Money supply contraction is a serious warning signal. Liquidity is draining from the system — "
                "meaning potential credit crunches, slower consumption, and falling asset prices. The 2022 "
                "crypto winter and tech stock crash happened exactly when M2 turned negative. Not a coincidence. "
                "Less money chasing the same assets mechanically pulls valuations down. The dollar strengthens "
                "in these periods because cash becomes scarce and everyone seeks refuge in dollars."
            )
        return (
            "Moderate money supply growth is the healthiest scenario for the economy. Not enough excess "
            "liquidity for bubbles, not enough contraction for systemic stress. Asset prices tend to "
            "track fundamentals more closely, and market moves become more sensitive to macro data "
            "and corporate performance. It's 'boring' but actually the healthiest environment."
        )
