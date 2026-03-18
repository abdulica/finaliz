"""Technical analysis engine with auto-generated commentary."""

import re
import numpy as np
import pandas as pd
import ta

from config import TA_PARAMS


# ---------------------------------------------------------------------------
# External context classifier — detects the *type* of factor the user described
# so that commentary can be tailored instead of generic.
# ---------------------------------------------------------------------------

_EXT_CATEGORIES = {
    "monetary_intervention": {
        "keywords": [
            "swap", "faiz müdahale", "faiz mudahale", "faiz kontrol",
            "merkez bankası", "merkez bankasi", "tcmb", "cbrt", "fed",
            "para politikası", "para politikasi", "sıkılaştır", "sikilas",
            "gevşe", "gevse", "likidite", "repo", "açık piyasa", "acik piyasa",
            "zorunlu karşılık", "zorunlu karsilik", "faiz artır", "faiz artir",
            "faiz indir", "faiz düşür", "faiz dusur",
            "interest rate", "central bank", "monetary", "liquidity",
            "quantitative", "tightening", "easing", "rate hike", "rate cut",
        ],
        "tr_label": "para politikası müdahalesi",
        "en_label": "monetary policy intervention",
    },
    "currency_control": {
        "keywords": [
            "swap kanal", "sermaye kontrol", "döviz yasağ", "kambiyo",
            "tl kontrol", "kur kontrol", "kur manipül", "yapay kur",
            "peg", "managed float", "capital control", "forex ban",
            "currency peg", "fx intervention", "currency manipulation",
        ],
        "tr_label": "kur/sermaye kontrolü",
        "en_label": "currency/capital control",
    },
    "geopolitical": {
        "keywords": [
            "savaş", "savas", "yaptırım", "yaptirim", "ambargo", "ticaret savaş",
            "gümrük", "gumruk", "tarife", "darbe", "seçim", "secim",
            "protesto", "gerginlik", "çatışma", "catisma", "nato", "ab üyelik",
            "diplomatik", "askeri", "ukrayna", "rusya", "çin", "cin", "iran",
            "israil", "ortadoğu", "ortadogu", "suriye", "taiwan",
            "war", "sanction", "embargo", "trade war", "tariff",
            "coup", "election", "protest", "tension", "conflict",
            "nato", "diplomatic", "military", "invasion", "annex",
            "russia", "ukraine", "china", "iran", "israel", "middle east",
        ],
        "tr_label": "jeopolitik gelişme",
        "en_label": "geopolitical development",
    },
    "supply_disruption": {
        "keywords": [
            "opec", "üretim kes", "arz daral", "tedarik zincir", "kıtlık",
            "maden", "üretim düş", "stok", "depo", "ihracat yasağ",
            "supply chain", "shortage", "production cut", "stockpile",
            "mining", "refinery", "output cut", "export ban",
        ],
        "tr_label": "arz/tedarik bozulması",
        "en_label": "supply disruption",
    },
    "demand_shock": {
        "keywords": [
            "pandemi", "resesyon", "durgunluk", "talep düş", "tüketim",
            "büyüme yavaş", "gdp daral", "kriz", "iflas",
            "pandemic", "recession", "demand drop", "consumption",
            "slowdown", "gdp contract", "crisis", "bankruptcy", "default",
        ],
        "tr_label": "talep şoku / ekonomik yavaşlama",
        "en_label": "demand shock / economic slowdown",
    },
    "regulatory": {
        "keywords": [
            "regülasyon", "regulasyon", "düzenleme", "duzenleme", "yasakla",
            "kripto yasağ", "kripto yasag", "vergi", "sec", "mevzuat", "kanun",
            "yasa değişik", "yasa degisik", "bddk", "spk", "kgk",
            "regulation", "ban", "crypto ban", "tax", "sec",
            "legislation", "compliance", "legal", "ruling", "court",
        ],
        "tr_label": "düzenleyici/yasal değişiklik",
        "en_label": "regulatory/legal change",
    },
    "inflation_structural": {
        "keywords": [
            "enflasyon", "hiperenflasyon", "fiyat artış", "maliyet artış",
            "gıda fiyat", "enerji fiyat", "ücret artış", "stagflasyon",
            "inflation", "hyperinflation", "price surge", "cost push",
            "food price", "energy price", "wage", "stagflation",
        ],
        "tr_label": "yapısal enflasyon baskısı",
        "en_label": "structural inflation pressure",
    },
    "technical_observation": {
        "keywords": [
            "bollinger", "rsi", "macd", "sma", "ema", "fibonacci", "fib",
            "destek", "direnç", "direnc", "support", "resistance",
            "trend", "kanal", "channel", "formasyonlar", "formasyon",
            "pattern", "head and shoulder", "omuz baş", "omuz bas",
            "double top", "double bottom", "çift tepe", "cift tepe", "çift dip",
            "golden cross", "death cross", "breakout", "kırılım", "kirilim",
            "divergence", "diverjans", "uyumsuzluk", "ayrışma", "ayrisma",
            "hacim", "volume", "mum", "candle", "doji", "hammer", "çekiç",
            "genişleme", "genisleme", "sıkışma", "sikisma", "daralma",
            "squeeze", "expansion", "band", "bant", "ortalama", "average",
            "stochastic", "stokastik", "aşırı alım", "asiri alim",
            "aşırı satım", "asiri satim", "overbought", "oversold",
            "pivot", "ichimoku", "grafik", "chart", "mum grafik",
        ],
        "tr_label": "teknik gözlem",
        "en_label": "technical observation",
    },
}


def _classify_external_context(text: str) -> list[str]:
    """Return list of matching category keys, ordered by match strength."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for cat, info in _EXT_CATEGORIES.items():
        count = sum(1 for kw in info["keywords"] if kw in text_lower)
        if count > 0:
            scores[cat] = count
    # Sort by match count descending
    return [k for k, _ in sorted(scores.items(), key=lambda x: -x[1])]


_IMPACT_MATRIX_TR = {
    # (category, asset_key) → (impact_type, scenario_text)
    # impact_type: "direct", "inverse", "indirect", "tangential"
    # DXY
    ("currency_control", "DXY"): (
        "indirect",
        "Kur kontrolleri doğrudan DXY'yi değil, kontrol uygulayan ülkenin parasını etkiler. "
        "Ama geniş çapta kur müdahaleleri küresel döviz piyasasındaki dengeleri bozarak "
        "dolar endeksinin bileşenlerini dolaylı olarak etkiler."
    ),
    ("currency_control", "USDTRY"): (
        "direct",
        "Kur kontrolleri bu paritenin fiyatlamasını doğrudan şekillendiriyor. "
        "Swap kanallarının kapatılması TL'nin açık piyasada fiyatlanmasını engelliyor. "
        "Üç senaryo: (1) Kontrol sürdürülebilir — lineer yükseliş devam eder ama "
        "offshore-onshore makası açılır. (2) Kontrol maliyeti artır — CDS yükselir, "
        "yabancı yatırımcı çıkışı hızlanır. (3) Kontrol kırılır — birikmiş sapma "
        "sert düzelir, %10-20 üzeri ani hareket olası."
    ),
    ("currency_control", "EURUSD"): (
        "tangential",
        "Kur kontrolleri EUR/USD'yi doğrudan etkilemez ama kontrol uygulayan ülke "
        "önemli bir ticaret ortağıysa, sermaye akışları dolaylı etki yaratabilir."
    ),
    ("currency_control", "GOLD"): (
        "indirect",
        "Kur kontrolleri altını dolaylı etkiler. Kontrol uygulayan ülkede vatandaşlar "
        "parasını korumak için altına sığınır — bu fiziksel altın talebini artırır. "
        "Ayrıca kur kontrolleri küresel güvensizlik sinyali verir, bu da altın talebini destekler."
    ),
    ("currency_control", "BTC"): (
        "indirect",
        "Kur kontrolleri kripto için ilginç bir dinamik yaratır. Sermaye kontrollü ülkelerde "
        "BTC, sermayeyi sınır dışına çıkarmanın alternatif bir yolu haline gelebilir — "
        "bu yerel primi artırır. Aynı zamanda regülatörler bunu fark edip kripto kısıtlamaları "
        "getirebilir."
    ),
    ("currency_control", "OIL"): (
        "tangential",
        "Kur kontrolleri petrolü doğrudan etkilemez — petrol küresel dolar bazlı fiyatlanır. "
        "Ancak kontrol uygulayan ülke büyük bir petrol ithalatçısıysa, yerel para birimindeki "
        "baskı ithalat maliyetini artırır ve dolaylı talep etkisi yaratabilir."
    ),
    # Monetary intervention
    ("monetary_intervention", "DXY"): (
        "direct",
        "Para politikası DXY'nin ana belirleyicisi. FED şahinleşirse dolar güçlenir, "
        "güvercinleşirse zayıflar. Ama asıl hareket 'sürpriz' anlarında gelir — piyasa "
        "beklentisi ile gerçekleşen arasındaki fark. Senaryo 1: beklenen politika devam eder → "
        "mevcut trend korunur. Senaryo 2: beklenmedik hamle → mevcut yapı çözülür."
    ),
    ("monetary_intervention", "GOLD"): (
        "inverse",
        "Altın ve para politikası ters ilişkili. Faiz yükselirse altının fırsat maliyeti artar "
        "ve baskılanır. Faiz düşerse altın parlar. Ama burada önemli bir nüans var: "
        "negatif reel faiz (faiz < enflasyon) ortamında altın faiz yükselse bile güçlü kalabilir "
        "çünkü gerçek getiri hâlâ negatif."
    ),
    ("monetary_intervention", "BTC"): (
        "indirect",
        "BTC ve para politikası arasındaki ilişki karmaşık. Gevşek politika → artan likidite → "
        "risk iştahı → BTC yükselir. Sıkı politika → azalan likidite → risk kaçışı → BTC düşer. "
        "Ama BTC aynı zamanda 'merkez bankalarına karşı hedge' narratifi taşıyor — bu yüzden "
        "aşırı müdahale dönemlerinde paradoksal olarak güçlenebilir."
    ),
    ("monetary_intervention", "USDTRY"): (
        "direct",
        "TCMB'nin para politikası USD/TRY'nin birincil belirleyicisi. Faiz artışı TL'yi güçlendirir "
        "ama sürdürülebilirliği kritik. Heterodoks politika (düşük faiz + yüksek enflasyon) "
        "TL'yi yapısal zayıflığa iter. İzlenmesi gereken: reel faiz (faiz - enflasyon), "
        "döviz rezervleri ve net hata-noksan kalemi."
    ),
    ("monetary_intervention", "OIL"): (
        "indirect",
        "Para politikası petrolü dolaylı etkiler. Sıkı politika → güçlü dolar → petrol ucuzlar "
        "(dolar bazında). Gevşek politika → zayıf dolar → petrol pahalılaşır. "
        "Ayrıca sıkı politika ekonomik yavaşlama riski taşır → talep düşer → petrol baskılanır."
    ),
    # Geopolitical
    ("geopolitical", "GOLD"): (
        "direct",
        "Altın jeopolitik riskin birincil barometresi. Gerginlik tırmanırsa altın güvenli liman "
        "talebiyle sert yükselir — 2022 Rusya-Ukrayna başlangıcında %10+ ralliye tanık olduk. "
        "Diplomasi sinyalleri gelirse altın hızla düşer çünkü risk primi erir. "
        "Belirsizlik uzarsa altın yüksek seviyelerde konsolide olur, yeni bir taban oluşturur."
    ),
    ("geopolitical", "OIL"): (
        "direct",
        "Petrol jeopolitik olaylara en duyarlı emtia. Özellikle Ortadoğu'da gerginlik doğrudan "
        "arz riski yaratır — Hürmüz Boğazı'ndan küresel petrol arzının %20'si geçiyor. "
        "Tırmanma senaryosunda petrol $10-30 bandında sert yükselebilir. "
        "Çözüm senaryosunda risk primi hızla erir."
    ),
    ("geopolitical", "DXY"): (
        "direct",
        "Dolar küresel kriz dönemlerinde güvenli liman. Gerginlik arttıkça sermaye ABD'ye "
        "sığınır → DXY yükselir. Ama ABD'nin kendisi taraf ise (ticaret savaşı gibi) "
        "durum karmaşıklaşır — dolar güvenli liman olur ama aynı zamanda riskin kaynağı."
    ),
    ("geopolitical", "BTC"): (
        "tangential",
        "BTC'nin jeopolitik risklere tepkisi tutarsız. Bazen güvenli liman gibi davranır "
        "(yaptırım altındaki ülkelerde talep artar), bazen risk varlığı gibi düşer "
        "(küresel panik anlarında likidite ihtiyacıyla satılır). Spesifik olarak "
        "yaptırım/sermaye kontrolü boyutu varsa BTC talebi artma eğiliminde."
    ),
    ("geopolitical", "USDTRY"): (
        "indirect",
        "Jeopolitik gerginlik TL gibi gelişen piyasa paralarını baskılar. Küresel risk "
        "arttığında sermaye gelişen piyasalardan çıkar → TL zayıflar → USD/TRY yükselir. "
        "Türkiye'nin doğrudan taraf olduğu gerginliklerde etki çok daha sert."
    ),
    # Supply disruption
    ("supply_disruption", "OIL"): (
        "direct",
        "Arz bozulması petrolün fiyatını doğrudan ve birincil olarak etkiler. "
        "OPEC kesintisi veya boru hattı sorunu varsa etkisi anlık. Üç senaryo: "
        "(1) Kesinti geçici → fiyat spike atar ve geri döner. "
        "(2) Kesinti uzar ama stratejik stoklar devreye girer → fiyat yüksek ama kontrollü. "
        "(3) Kesinti yapısal → fiyat kalıcı yeni platoya oturur."
    ),
    ("supply_disruption", "GOLD"): (
        "indirect",
        "Arz bozulması altını dolaylı etkiler. Enerji/emtia fiyatları yükselirse → "
        "enflasyon beklentisi artar → altın enflasyon hedge'i olarak güçlenir. "
        "Ayrıca maden üretimi etkileniyorsa (enerji maliyeti, tedarik zinciri) "
        "altın arzı da kısılabilir — bu doğrudan etkidir."
    ),
    ("supply_disruption", "BTC"): (
        "tangential",
        "Arz bozulması BTC'yi mining tarafından etkiler — enerji maliyeti artarsa mining "
        "kârlılığı düşer, küçük madenciler kapanır, hash rate düşer. Ama BTC arzı "
        "algoritmik olarak sabit olduğu için fiyat etkisi daha çok sentiment kanalıyla gelir."
    ),
    ("supply_disruption", "DXY"): (
        "indirect",
        "Arz bozulması doları dolaylı etkiler. Emtia fiyatları yükselir → küresel enflasyon → "
        "FED sıkılaştırır → dolar güçlenir. Ama aynı zamanda yüksek enerji maliyeti "
        "ABD ekonomisini de yavaşlatabilir → karışık etki."
    ),
    # Demand shock
    ("demand_shock", "OIL"): (
        "direct",
        "Talep şoku petrolü doğrudan vurur — petrol talebi %80+ ulaşımdan gelir. "
        "Resesyon veya pandemi → uçuşlar azalır, fabrikalar yavaşlar → talep çöker → "
        "fiyat sert düşer. 2020'de petrol negatife bile gitti. "
        "Toparlanma ise aşamalı gelir — önce karayolu, sonra havayolu, en son sanayi."
    ),
    ("demand_shock", "GOLD"): (
        "inverse",
        "Talep şoku altın için paradoksal: ekonomi yavaşlarsa → merkez bankaları gevşer → "
        "faizler düşer → altın güçlenir. Ama şok çok sert gelirse (2020 Mart gibi) "
        "likidite krizi yaşanır ve altın bile satılır — margin call etkisi. "
        "İlk şok atlatıldıktan sonra altın genelde güçlü toparlanır."
    ),
    ("demand_shock", "BTC"): (
        "direct",
        "Talep şoku BTC'yi yüksek beta yapısıyla sert etkiler. Risk iştahı çöker → "
        "BTC %30-50 düşebilir. Ama merkez bankası tepkisi (para basma) gelirse "
        "BTC 'enflasyon hedge' narratifiyle toparlanır. 2020'de BTC şoktan sonra "
        "12 ayda %400+ yükseldi — şok sonrası likidite enjeksiyonu sayesinde."
    ),
    # Inflation
    ("inflation_structural", "GOLD"): (
        "direct",
        "Altın enflasyonun birincil hedge aracı. Yapısal enflasyonda altın nominal olarak "
        "sürekli yükselir ve reel değerini korur. Ama önemli bir ayrım: eğer merkez bankası "
        "enflasyonu agresif şekilde sıkıyorsa (Volcker tarzı) reel faiz pozitife döner "
        "ve altın baskılanır. Enflasyona rağmen altının düştüğü nadir dönemler bunlardır."
    ),
    ("inflation_structural", "BTC"): (
        "indirect",
        "BTC'nin enflasyon hedge'i olup olmadığı tartışmalı. Arzı sabit (21M) olduğu için "
        "teoride enflasyona dayanıklı ama pratikte kısa vadede risk iştahıyla hareket ediyor. "
        "Uzun vadeli enflasyonist ortamlarda BTC güçlenme eğiliminde — ama yolda %50 "
        "düşüşler görmek mümkün."
    ),
    ("inflation_structural", "DXY"): (
        "indirect",
        "Enflasyon ve dolar karmaşık bir ilişki. ABD enflasyonu yüksekse → FED sıkılaştırır → "
        "dolar güçlenir. Ama enflasyon küresel ise ve ABD diğerlerinden daha kötüyse → "
        "dolar zayıflayabilir. Burada göreli enflasyon (ABD vs diğer ülkeler) kritik."
    ),
    # Regulatory
    ("regulatory", "BTC"): (
        "direct",
        "Düzenleme BTC'yi doğrudan etkiler — kripto piyasası hâlâ regülasyona en duyarlı "
        "varlık sınıfı. ETF onayı gibi pozitif düzenleme → kurumsal giriş → sert yükseliş. "
        "Yasaklama veya kısıtlama → likidite çekilir → sert düşüş. "
        "Ülke bazlı yasaklar ise genelde fiyatı geçici baskılar, küresel yapı devam eder."
    ),
    ("regulatory", "GOLD"): (
        "tangential",
        "Düzenleme altını nadiren doğrudan etkiler — altın binlerce yıllık yerleşik bir varlık. "
        "Ama dolaylı etkiler olabilir: kripto düzenlemesi → sermaye kriptodan altına kayar. "
        "Veya vergi düzenlemesi → altın yatırımının vergi avantajı değişir."
    ),
}

# Generic fallbacks for categories not in the matrix
_IMPACT_GENERIC_TR = {
    "currency_control": "Kur kontrolleri bu varlığı dolaylı olarak etkiliyor — küresel risk algısı ve sermaye akışı kanalıyla.",
    "monetary_intervention": "Para politikası değişiklikleri tüm varlık sınıflarını farklı derecelerde etkiler.",
    "geopolitical": "Jeopolitik gerginlik risk algısını değiştirir ve bu varlık da bu değişimden payını alır.",
    "supply_disruption": "Arz bozulması bu varlığı emtia-enflasyon-merkez bankası tepki zinciri üzerinden etkiler.",
    "demand_shock": "Talep şoku küresel büyüme beklentilerini etkiler, bu da bu varlığa yansır.",
    "regulatory": "Düzenleyici değişiklikler piyasa yapısını etkiler ve bu varlık da bu değişimden etkilenir.",
    "inflation_structural": "Yapısal enflasyon tüm varlıkların reel getirisini etkiler.",
}


def _ext_trend_commentary_tr(categories: list[str], ext: str, trend: str, close: float, name: str, asset_key: str = "") -> str:
    """Generate scenario-based analytical commentary using factor × asset relationship matrix (Turkish)."""
    cat = categories[0] if categories else "generic"

    td = {
        "strong_up": ("güçlü yükseliş", "yukarı"),
        "strong_down": ("sert düşüş", "aşağı"),
        "up": ("yukarı eğilim", "yukarı"),
        "down": ("aşağı baskı", "aşağı"),
    }
    trend_label, trend_dir = td.get(trend, ("yatay seyir", "belirsiz"))

    # Look up specific factor × asset relationship
    matrix_key = (cat, asset_key)
    entry = _IMPACT_MATRIX_TR.get(matrix_key)

    if entry:
        impact_type, scenario_text = entry
        impact_label = {
            "direct": "doğrudan",
            "inverse": "ters yönlü",
            "indirect": "dolaylı",
            "tangential": "zayıf/dolaylı",
        }.get(impact_type, "dolaylı")

        return (
            f"{name} {close:,.2f} seviyesinde, teknik tablo {trend_label} gösteriyor. "
            f"Bu faktörün {name} üzerindeki etkisi **{impact_label}**. {scenario_text}"
        )

    # Fallback: generic category commentary
    generic = _IMPACT_GENERIC_TR.get(cat, "")
    return (
        f"{name} {close:,.2f} seviyesinde, {trend_label}. {generic} "
        f"İki olasılık var: ya bu dış faktör geçici ve teknik seviyeler sonunda yeniden geçerli "
        f"olacak, ya da kalıcı bir yapısal değişim söz konusu ve fiyat yeni bir dengeye oturacak."
    )


_IMPACT_MATRIX_EN = {
    ("currency_control", "USDTRY"): ("direct", "Currency controls are directly shaping this pair's pricing. Swap channel closures prevent TL from being freely priced. Three scenarios: (1) Controls hold — linear climb continues but offshore-onshore gap widens. (2) Control costs mount — CDS rises, foreign investor outflows accelerate. (3) Controls break — accumulated deviation corrects violently, 10-20%+ moves possible."),
    ("currency_control", "DXY"): ("indirect", "Currency controls don't directly affect DXY, but widespread FX interventions globally can distort dollar index components through capital flow disruptions."),
    ("currency_control", "GOLD"): ("indirect", "Currency controls push citizens toward gold as wealth preservation — boosting physical demand. Controls also signal instability, supporting global gold demand."),
    ("currency_control", "BTC"): ("indirect", "In capital-controlled countries, BTC becomes an alternative channel for moving money across borders — lifting local premiums. Regulators may respond with crypto restrictions."),
    ("currency_control", "OIL"): ("tangential", "Currency controls don't directly impact oil pricing. But if the controlled country is a major importer, local currency weakness raises import costs, creating indirect demand effects."),
    ("monetary_intervention", "DXY"): ("direct", "Monetary policy is DXY's primary driver. Fed hawkishness = dollar strength, dovishness = weakness. But the real move comes at 'surprise' moments — the gap between expectations and reality. If policy continues as expected, current trend holds. If it reverses, the technical structure unwinds within minutes."),
    ("monetary_intervention", "GOLD"): ("inverse", "Gold and monetary policy are inversely related. Rate hikes raise gold's opportunity cost, pressuring it. Rate cuts let gold shine. Key nuance: in negative real rate environments (rates < inflation), gold stays strong even as nominal rates rise."),
    ("monetary_intervention", "BTC"): ("indirect", "BTC's relationship with monetary policy is complex. Easy policy → more liquidity → risk appetite → BTC rises. Tight policy → less liquidity → risk-off → BTC falls. But BTC also carries the 'hedge against central banks' narrative — in extreme intervention periods, it can paradoxically strengthen."),
    ("monetary_intervention", "USDTRY"): ("direct", "CBRT's monetary policy is the primary determinant of USD/TRY. Rate hikes strengthen TL but sustainability is critical. Heterodox policy (low rates + high inflation) structurally weakens TL. Key metrics: real rates (rates - inflation), FX reserves, and net errors & omissions."),
    ("monetary_intervention", "OIL"): ("indirect", "Monetary policy affects oil indirectly. Tight policy → strong dollar → oil cheaper in dollar terms. Easy policy → weak dollar → oil more expensive. Also: tight policy risks economic slowdown → lower demand → oil pressured."),
    ("geopolitical", "GOLD"): ("direct", "Gold is the primary geopolitical risk barometer. Escalation → sharp safe-haven rally (10%+ like early 2022). Diplomacy → gold drops as risk premium evaporates. Prolonged uncertainty → gold consolidates at elevated levels, forming a new floor."),
    ("geopolitical", "OIL"): ("direct", "Oil is the most geopolitically sensitive commodity. Middle East tensions directly threaten supply — 20% of global oil transits the Strait of Hormuz. Escalation could spike oil $10-30. Resolution → risk premium evaporates quickly."),
    ("geopolitical", "DXY"): ("direct", "Dollar is the global crisis safe haven. Tensions rise → capital flows to US → DXY strengthens. But if the US is directly involved (trade wars), it gets complex — dollar is both safe haven and risk source."),
    ("geopolitical", "BTC"): ("tangential", "BTC's geopolitical response is inconsistent. Sometimes acts as safe haven (demand rises in sanctioned countries), sometimes drops as a risk asset (panic selling for liquidity). Sanctions/capital control angle specifically tends to boost BTC demand."),
    ("geopolitical", "USDTRY"): ("indirect", "Geopolitical tension pressures EM currencies like TRY. Global risk-off → capital exits emerging markets → TL weakens → USD/TRY rises. If Turkey is directly involved, the impact is much more severe."),
    ("supply_disruption", "OIL"): ("direct", "Supply disruption directly and primarily impacts oil pricing. OPEC cuts or pipeline issues have immediate effect. (1) Temporary disruption → price spikes then reverts. (2) Prolonged but strategic reserves deployed → elevated but controlled. (3) Structural → permanent new price plateau."),
    ("supply_disruption", "GOLD"): ("indirect", "Supply disruption affects gold indirectly. Energy/commodity prices rise → inflation expectations increase → gold strengthens as inflation hedge. If mining production is affected (energy costs, supply chain), gold supply itself tightens — a direct channel."),
    ("supply_disruption", "BTC"): ("tangential", "Supply disruption affects BTC through mining — higher energy costs reduce mining profitability, smaller miners shut down, hash rate drops. But BTC supply is algorithmically fixed, so price impact comes more through sentiment."),
    ("demand_shock", "OIL"): ("direct", "Demand shock hits oil directly — 80%+ of oil demand is transportation. Recession or pandemic → flights drop, factories slow → demand collapses → sharp price decline. 2020 saw oil go negative. Recovery is gradual — road first, then air, then industrial."),
    ("demand_shock", "GOLD"): ("inverse", "Demand shock creates a paradox for gold: economy slows → central banks ease → rates fall → gold strengthens. But if the shock is severe (March 2020), liquidity crisis hits and even gold gets sold for margin calls. After the initial shock, gold typically recovers strongly."),
    ("demand_shock", "BTC"): ("direct", "Demand shock hits BTC hard via its high-beta nature. Risk appetite collapses → BTC can drop 30-50%. But if central bank response follows (money printing), BTC recovers on the 'inflation hedge' narrative. Post-2020 shock, BTC rallied 400%+ in 12 months — fueled by liquidity injection."),
    ("inflation_structural", "GOLD"): ("direct", "Gold is the primary inflation hedge. In structural inflation, gold rises nominally and preserves real value. Key distinction: if the central bank fights inflation aggressively (Volcker-style), real rates go positive and gold gets pressured — the rare exception."),
    ("inflation_structural", "BTC"): ("indirect", "Whether BTC is an inflation hedge is debatable. Its fixed supply (21M) theoretically makes it resistant, but in practice it moves with risk appetite short-term. In prolonged inflationary environments, BTC tends to strengthen — but expect 50% drawdowns along the way."),
    ("inflation_structural", "DXY"): ("indirect", "Inflation and the dollar have a complex relationship. High US inflation → Fed tightens → dollar strengthens. But if inflation is global and the US is worse off relatively → dollar can weaken. Relative inflation (US vs others) is the key metric."),
    ("regulatory", "BTC"): ("direct", "Regulation directly impacts BTC — crypto is still the most regulation-sensitive asset class. Positive regulation (ETF approval) → institutional inflow → sharp rally. Bans/restrictions → liquidity pulls out → sharp decline. Country-level bans typically cause temporary pressure, global structure persists."),
    ("regulatory", "GOLD"): ("tangential", "Regulation rarely directly affects gold — it's a millennia-old established asset. But indirect effects exist: crypto regulation → capital flows from crypto to gold. Tax policy changes → gold investment tax advantages shift."),
}

_IMPACT_GENERIC_EN = {
    "currency_control": "Currency controls affect this asset indirectly through global risk perception and capital flow channels.",
    "monetary_intervention": "Monetary policy changes affect all asset classes to varying degrees.",
    "geopolitical": "Geopolitical tension shifts risk perception, and this asset absorbs its share of that shift.",
    "supply_disruption": "Supply disruption affects this asset through the commodity-inflation-central bank response chain.",
    "demand_shock": "Demand shock impacts global growth expectations, which feeds through to this asset.",
    "regulatory": "Regulatory changes affect market structure, which in turn impacts this asset.",
    "inflation_structural": "Structural inflation affects the real returns of all assets.",
}


def _ext_trend_commentary_en(categories: list[str], ext: str, trend: str, close: float, name: str, asset_key: str = "") -> str:
    """Generate scenario-based analytical commentary using factor × asset relationship matrix (English)."""
    cat = categories[0] if categories else "generic"

    td = {
        "strong_up": "strong uptrend",
        "strong_down": "sharp downtrend",
        "up": "upward bias",
        "down": "downward pressure",
    }
    trend_label = td.get(trend, "sideways")

    # Look up specific factor × asset relationship
    matrix_key = (cat, asset_key)
    entry = _IMPACT_MATRIX_EN.get(matrix_key)

    if entry:
        impact_type, scenario_text = entry
        impact_label = {
            "direct": "direct",
            "inverse": "inverse",
            "indirect": "indirect",
            "tangential": "weak/indirect",
        }.get(impact_type, "indirect")

        return (
            f"{name} is at {close:,.2f}, technically showing {trend_label}. "
            f"This factor's impact on {name} is **{impact_label}**. {scenario_text}"
        )

    # Fallback
    generic = _IMPACT_GENERIC_EN.get(cat, "")
    return (
        f"{name} is at {close:,.2f}, showing {trend_label}. {generic} "
        f"Two possibilities: either this factor is temporary and technical levels will eventually "
        f"reassert, or it represents a structural shift toward a new equilibrium."
    )


def _ext_momentum_commentary_tr(categories: list[str], rsi: float, momentum: str) -> str:
    """RSI commentary tailored to external context type (Turkish)."""
    cat = categories[0] if categories else "generic"

    if cat == "currency_control":
        if momentum == "overbought":
            return (
                f"RSI {rsi:.0f} ile aşırı alım bölgesinde ama kontrollü piyasada bu okumanın "
                f"anlamı farklı. Kontrol mekanizması doğal geri çekilmeyi engelliyor olabilir — "
                f"bu da RSI'ın uzun süre aşırı bölgede kalmasına neden olur."
            )
        elif momentum == "oversold":
            return (
                f"RSI {rsi:.0f} ile aşırı satımda. Kontrol altındaki piyasalarda aşırı satım "
                f"tepki garantisi değil — baskı sürdükçe gösterge uzun süre bu bölgede kalabilir."
            )
        else:
            m_desc = "pozitif" if momentum == "bullish" else "negatif" if momentum == "bearish" else "nötr"
            return f"RSI {rsi:.0f} ile momentum {m_desc}, ama kontrollü ortamda bu sinyalin güvenilirliği sınırlı."

    elif cat == "geopolitical":
        if momentum in ("overbought", "oversold"):
            zone = "aşırı alım" if momentum == "overbought" else "aşırı satım"
            return (
                f"RSI {rsi:.0f} ile {zone} bölgesinde. Jeopolitik gerginlik dönemlerinde RSI "
                f"uç seviyelerde uzun süre kalabilir — çünkü piyasayı haber akışı yönlendiriyor, "
                f"teknik seviyeler değil."
            )
        else:
            m_desc = "pozitif" if momentum == "bullish" else "negatif"
            return f"RSI {rsi:.0f} ile {m_desc} momentum var, ama jeopolitik gelişmeler bu yönü dakikalar içinde değiştirebilir."

    elif cat == "supply_disruption":
        if momentum == "overbought":
            return (
                f"RSI {rsi:.0f} ile aşırı alım diyor ama arz sıkıntısı devam ediyorsa "
                f"fiyat 'aşırı' seviyelerden daha da yukarı gidebilir. Arz normalleşmeden "
                f"RSI bazlı satış sinyallerine güvenmek riskli."
            )
        elif momentum == "oversold":
            return (
                f"RSI {rsi:.0f} ile aşırı satımda. Arz tarafında iyileşme yoksa bu düşük seviye "
                f"geçici bir zemin olabilir ama kalıcı dip garantisi değil."
            )
        else:
            m_desc = "pozitif" if momentum == "bullish" else "negatif"
            return f"RSI {rsi:.0f} — arz koşulları değişmedikçe momentum okumaları ikincil önemde."

    elif cat == "demand_shock":
        if momentum == "oversold":
            return (
                f"RSI {rsi:.0f} ile aşırı satım bölgesinde. Talep şoku dönemlerinde 'aşırı satım = alım fırsatı' "
                f"mantığı tehlikeli olabilir — talep yapısal olarak düştüyse dip sürekli yenilenir."
            )
        elif momentum == "overbought":
            return (
                f"RSI {rsi:.0f} ile aşırı alımda. Talep zayıfken bu yükseliş kısa ömürlü olabilir — "
                f"teknik toparlanma ile gerçek toparlanma arasındaki farkı ayırt etmek kritik."
            )
        else:
            m_desc = "pozitif" if momentum == "bullish" else "negatif"
            return f"RSI {rsi:.0f} ile {m_desc} momentum — ama talep tarafı desteklemezse sürdürülebilir değil."

    elif cat == "inflation_structural":
        if momentum == "overbought":
            return (
                f"RSI {rsi:.0f} ile aşırı alımda. Enflasyonist ortamda nominal fiyat yükselişi "
                f"RSI'ı sürekli aşırı alıma iter — ama bu reel bir güç değil, paranın erimesi."
            )
        else:
            m_desc = "pozitif" if momentum == "bullish" else "negatif" if momentum == "bearish" else "nötr"
            return f"RSI {rsi:.0f} — enflasyon ortamında nominal momentum okumaları yanıltıcı olabilir, reel getiriye bakmak daha sağlıklı."

    else:
        # monetary_intervention, regulatory, or unmatched
        if momentum in ("overbought", "oversold"):
            zone = "aşırı alım" if momentum == "overbought" else "aşırı satım"
            return (
                f"RSI {rsi:.0f} ile {zone} bölgesinde. Bu dış faktör devredeyken standart "
                f"RSI yorumları tam güvenilir olmayabilir."
            )
        m_desc = "pozitif" if momentum == "bullish" else "negatif" if momentum == "bearish" else "nötr"
        return f"RSI {rsi:.0f} ile {m_desc} momentum — dış faktör hesaba katılarak okunmalı."


def _ext_momentum_commentary_en(categories: list[str], rsi: float, momentum: str) -> str:
    """RSI commentary tailored to external context type (English)."""
    cat = categories[0] if categories else "generic"

    if cat == "currency_control":
        if momentum == "overbought":
            return (
                f"RSI at {rsi:.0f} shows overbought, but in a controlled market this reading means "
                f"something different. Controls may prevent natural pullbacks — causing RSI to stay "
                f"in extreme territory for extended periods."
            )
        elif momentum == "oversold":
            return (
                f"RSI at {rsi:.0f} in oversold territory. In controlled markets, oversold doesn't "
                f"guarantee a bounce — under sustained pressure, the indicator can stay extreme indefinitely."
            )
        else:
            m_desc = "positive" if momentum == "bullish" else "negative" if momentum == "bearish" else "neutral"
            return f"RSI at {rsi:.0f} shows {m_desc} momentum, but reliability is limited in a controlled environment."

    elif cat == "geopolitical":
        if momentum in ("overbought", "oversold"):
            zone = "overbought" if momentum == "overbought" else "oversold"
            return (
                f"RSI at {rsi:.0f} in {zone} territory. During geopolitical tensions, RSI can stay "
                f"at extremes for extended periods — headlines drive prices, not technical levels."
            )
        else:
            m_desc = "positive" if momentum == "bullish" else "negative"
            return f"RSI at {rsi:.0f} shows {m_desc} momentum, but geopolitical events can reverse direction within minutes."

    elif cat == "supply_disruption":
        if momentum == "overbought":
            return (
                f"RSI at {rsi:.0f} screams overbought, but if supply disruption persists, prices "
                f"can push well beyond 'extreme' levels. RSI-based sell signals are risky until supply normalizes."
            )
        elif momentum == "oversold":
            return (
                f"RSI at {rsi:.0f} in oversold territory. Without supply improvement, this low level "
                f"may be a temporary floor, not a guaranteed bottom."
            )
        else:
            m_desc = "positive" if momentum == "bullish" else "negative"
            return f"RSI at {rsi:.0f} — momentum readings are secondary until supply conditions change."

    elif cat == "demand_shock":
        if momentum == "oversold":
            return (
                f"RSI at {rsi:.0f} in oversold territory. During demand shocks, 'oversold = buying opportunity' "
                f"logic can be dangerous — if demand has structurally collapsed, bottoms keep getting redefined."
            )
        elif momentum == "overbought":
            return (
                f"RSI at {rsi:.0f} overbought. With weak demand, this rally may be short-lived — "
                f"distinguishing technical recovery from real recovery is critical."
            )
        else:
            m_desc = "positive" if momentum == "bullish" else "negative"
            return f"RSI at {rsi:.0f} shows {m_desc} momentum — unsustainable without demand-side support."

    elif cat == "inflation_structural":
        if momentum == "overbought":
            return (
                f"RSI at {rsi:.0f} overbought. In inflationary environments, nominal price rises "
                f"keep pushing RSI into overbought — but this isn't real strength, it's currency erosion."
            )
        else:
            m_desc = "positive" if momentum == "bullish" else "negative" if momentum == "bearish" else "neutral"
            return f"RSI at {rsi:.0f} — in an inflationary environment, nominal momentum readings can be misleading; real returns matter more."

    else:
        if momentum in ("overbought", "oversold"):
            zone = "overbought" if momentum == "overbought" else "oversold"
            return f"RSI at {rsi:.0f} in {zone} territory. Standard interpretation may not fully apply given this external factor."
        m_desc = "positive" if momentum == "bullish" else "negative" if momentum == "bearish" else "neutral"
        return f"RSI at {rsi:.0f} shows {m_desc} momentum — should be read with the external factor in mind."


def _ext_bb_commentary_tr(categories: list[str]) -> str:
    """Bollinger squeeze commentary by category (Turkish)."""
    cat = categories[0] if categories else "generic"
    if cat == "currency_control":
        return (
            "Bollinger bantları sıkışmış — kontrollü piyasalarda bu sıkışma volatilitenin "
            "yapay olarak baskılandığını gösterir. Kontrol zayıfladığı an hareket çok sert olabilir."
        )
    elif cat == "geopolitical":
        return (
            "Bollinger bantları sıkışmış — jeopolitik gerginlik ortamında bu sessizlik fırtına öncesi olabilir. "
            "Bir gelişme anında bantlar patlayabilir ve yön tamamen haber akışına bağlı."
        )
    elif cat == "supply_disruption":
        return (
            "Bollinger bantları sıkışmış. Arz bozulması devam ederken bu sıkışma "
            "fiyatın yukarı patlamasına zemin hazırlıyor olabilir."
        )
    elif cat == "demand_shock":
        return (
            "Bollinger bantları sıkışmış. Talep belirsizliği ortamında bu genelde "
            "piyasanın yön aradığını gösterir — bir veri veya gelişme ile sert kırılım gelebilir."
        )
    elif cat == "inflation_structural":
        return (
            "Bollinger bantları sıkışmış. Enflasyonist ortamda bu sıkışma genelde "
            "yukarı yönlü çözülür çünkü nominal fiyat baskısı sürekli."
        )
    return (
        "Bollinger bantları ciddi şekilde sıkışmış — büyük bir hareket kapıda. "
        "Dış faktör düşünüldüğünde hareketin yönü daha da öngörülemez."
    )


def _ext_bb_commentary_en(categories: list[str]) -> str:
    """Bollinger squeeze commentary by category (English)."""
    cat = categories[0] if categories else "generic"
    if cat == "currency_control":
        return (
            "Bollinger Bands are compressed — in controlled markets this indicates volatility "
            "is being artificially suppressed. When controls weaken, the move can be extreme."
        )
    elif cat == "geopolitical":
        return (
            "Bollinger Bands are squeezed — during geopolitical tension this quiet may be the calm "
            "before the storm. Bands can explode on a single development, direction entirely news-dependent."
        )
    elif cat == "supply_disruption":
        return (
            "Bollinger Bands are compressed. With ongoing supply disruption, this squeeze may be "
            "setting the stage for an upside breakout."
        )
    elif cat == "demand_shock":
        return (
            "Bollinger Bands are squeezed. In a demand-uncertain environment, this typically "
            "means the market is searching for direction — a sharp break could follow any data release."
        )
    elif cat == "inflation_structural":
        return (
            "Bollinger Bands are compressed. In inflationary environments, squeezes tend to "
            "resolve upward as nominal price pressure is persistent."
        )
    return (
        "Bollinger Bands are significantly compressed — a major move is coming. "
        "Given the external factor, the direction is even less predictable."
    )


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators on OHLCV DataFrame."""
    if df is None or df.empty or len(df) < 30:
        return df

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df.get("Volume", pd.Series(0, index=df.index))

    # RSI
    df["RSI"] = ta.momentum.rsi(close, window=TA_PARAMS["rsi_period"])

    # MACD
    macd = ta.trend.MACD(
        close,
        window_slow=TA_PARAMS["macd_slow"],
        window_fast=TA_PARAMS["macd_fast"],
        window_sign=TA_PARAMS["macd_signal"],
    )
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(
        close, window=TA_PARAMS["bb_period"], window_dev=TA_PARAMS["bb_std"]
    )
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]

    # Moving Averages
    for p in TA_PARAMS["sma_periods"]:
        df[f"SMA_{p}"] = ta.trend.sma_indicator(close, window=p)
    for p in TA_PARAMS["ema_periods"]:
        df[f"EMA_{p}"] = ta.trend.ema_indicator(close, window=p)

    # Stochastic RSI
    stoch = ta.momentum.StochRSIIndicator(close, window=TA_PARAMS["stoch_rsi_period"])
    df["StochRSI_K"] = stoch.stochrsi_k()
    df["StochRSI_D"] = stoch.stochrsi_d()

    # ATR
    df["ATR"] = ta.volatility.average_true_range(
        high, low, close, window=TA_PARAMS["atr_period"]
    )

    # Volume SMA
    if volume.sum() > 0:
        df["Volume_SMA_20"] = ta.trend.sma_indicator(volume.astype(float), window=20)

    return df


def compute_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """Compute support and resistance levels using pivot points and recent extremes."""
    if df is None or df.empty or len(df) < window:
        return {"support": [], "resistance": []}

    recent = df.tail(window)
    high = recent["High"].max()
    low = recent["Low"].min()
    close = recent["Close"].iloc[-1]

    # Classic pivot point
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)

    return {
        "pivot": pivot,
        "support": [round(s1, 4), round(s2, 4)],
        "resistance": [round(r1, 4), round(r2, 4)],
    }


def get_signal_summary(df: pd.DataFrame) -> dict:
    """Generate a summary of all technical signals.

    Returns dict with counts and overall signal.
    """
    if df is None or df.empty or len(df) < 30:
        return {"buy": 0, "sell": 0, "neutral": 0, "overall": "neutral"}

    last = df.iloc[-1]
    prev = df.iloc[-2]
    buy, sell, neutral = 0, 0, 0
    signals = []

    # RSI signal
    rsi = last.get("RSI")
    if rsi is not None and not np.isnan(rsi):
        if rsi < 30:
            buy += 1
            signals.append(("RSI", "buy", rsi))
        elif rsi > 70:
            sell += 1
            signals.append(("RSI", "sell", rsi))
        else:
            neutral += 1
            signals.append(("RSI", "neutral", rsi))

    # MACD signal
    macd_h = last.get("MACD_Hist")
    prev_macd_h = prev.get("MACD_Hist")
    if macd_h is not None and not np.isnan(macd_h):
        if macd_h > 0 and (prev_macd_h is not None and prev_macd_h <= 0):
            buy += 1
            signals.append(("MACD", "buy", macd_h))
        elif macd_h < 0 and (prev_macd_h is not None and prev_macd_h >= 0):
            sell += 1
            signals.append(("MACD", "sell", macd_h))
        elif macd_h > 0:
            buy += 1
            signals.append(("MACD", "buy", macd_h))
        else:
            sell += 1
            signals.append(("MACD", "sell", macd_h))

    # Bollinger Band signal
    close = last["Close"]
    bb_upper = last.get("BB_Upper")
    bb_lower = last.get("BB_Lower")
    if bb_upper is not None and not np.isnan(bb_upper):
        if close <= bb_lower:
            buy += 1
            signals.append(("BB", "buy", close))
        elif close >= bb_upper:
            sell += 1
            signals.append(("BB", "sell", close))
        else:
            neutral += 1
            signals.append(("BB", "neutral", close))

    # SMA signals
    for p in TA_PARAMS["sma_periods"]:
        sma = last.get(f"SMA_{p}")
        if sma is not None and not np.isnan(sma):
            if close > sma:
                buy += 1
                signals.append((f"SMA_{p}", "buy", sma))
            else:
                sell += 1
                signals.append((f"SMA_{p}", "sell", sma))

    # Stochastic RSI
    stoch_k = last.get("StochRSI_K")
    if stoch_k is not None and not np.isnan(stoch_k):
        if stoch_k < 0.2:
            buy += 1
            signals.append(("StochRSI", "buy", stoch_k))
        elif stoch_k > 0.8:
            sell += 1
            signals.append(("StochRSI", "sell", stoch_k))
        else:
            neutral += 1
            signals.append(("StochRSI", "neutral", stoch_k))

    total = buy + sell + neutral
    if total == 0:
        overall = "neutral"
    elif buy > sell and buy > neutral:
        overall = "buy"
    elif sell > buy and sell > neutral:
        overall = "sell"
    else:
        overall = "neutral"

    return {
        "buy": buy,
        "sell": sell,
        "neutral": neutral,
        "overall": overall,
        "signals": signals,
    }


def generate_commentary(
    df: pd.DataFrame,
    asset_name: str,
    lang: str = "tr",
    external_context: str = "",
    asset_key: str = "",
) -> list[str]:
    """Generate a conversational 1-2 paragraph overview of the technical picture.

    Instead of listing each indicator separately, this weaves all signals
    into a cohesive narrative — like a friend explaining what they see on the chart.

    If external_context is provided (user-supplied qualitative info like geopolitical
    factors, policy interventions, etc.), the commentary acknowledges that the pure
    technical picture may not tell the full story and integrates the external context.
    """
    if df is None or df.empty or len(df) < 30:
        return []

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    close = last["Close"]
    summary = get_signal_summary(df)
    sr = compute_support_resistance(df)

    # --- Gather all the raw observations ---
    rsi = _safe(last, "RSI")
    macd_h = _safe(last, "MACD_Hist")
    prev_macd_h = _safe(prev, "MACD_Hist")
    bb_width = _safe(last, "BB_Width")
    avg_bb_width = df["BB_Width"].tail(50).mean() if "BB_Width" in df else None
    sma_50 = _safe(last, "SMA_50")
    sma_200 = _safe(last, "SMA_200")
    stoch_k = _safe(last, "StochRSI_K")

    # Trend direction
    if sma_50 is not None and sma_200 is not None:
        if sma_50 > sma_200 and close > sma_200:
            trend = "strong_up"
        elif sma_50 > sma_200:
            trend = "up"
        elif sma_50 < sma_200 and close < sma_200:
            trend = "strong_down"
        elif sma_50 < sma_200:
            trend = "down"
        else:
            trend = "neutral"
    else:
        trend = "unknown"

    # Momentum state
    if rsi is not None:
        if rsi > 70:
            momentum = "overbought"
        elif rsi < 30:
            momentum = "oversold"
        elif rsi > 55:
            momentum = "bullish"
        elif rsi < 45:
            momentum = "bearish"
        else:
            momentum = "neutral"
    else:
        momentum = "unknown"

    # MACD event
    macd_event = None
    if macd_h is not None and prev_macd_h is not None:
        if macd_h > 0 and prev_macd_h <= 0:
            macd_event = "bullish_cross"
        elif macd_h < 0 and prev_macd_h >= 0:
            macd_event = "bearish_cross"
        elif macd_h > 0:
            macd_event = "positive"
        else:
            macd_event = "negative"

    # Volatility state
    bb_squeeze = False
    if bb_width is not None and avg_bb_width is not None:
        bb_squeeze = bb_width < avg_bb_width * 0.6

    # Support/resistance proximity
    near_support = near_resistance = False
    support_val = resistance_val = None
    if sr["support"] and sr["resistance"]:
        support_val = sr["support"][0]
        resistance_val = sr["resistance"][0]
        near_support = abs(close - support_val) / close * 100 < 1.5
        near_resistance = abs(close - resistance_val) / close * 100 < 1.5

    # Signal agreement
    total = summary["buy"] + summary["sell"] + summary["neutral"]
    if total > 0:
        agreement = max(summary["buy"], summary["sell"], summary["neutral"]) / total
    else:
        agreement = 0
    signals_mixed = agreement < 0.5

    # --- Build the narrative ---
    ext = external_context.strip() if external_context else ""

    if lang == "tr":
        paragraphs = _build_narrative_tr(
            asset_name, close, rsi, trend, momentum, macd_event,
            bb_squeeze, near_support, near_resistance, support_val,
            resistance_val, signals_mixed, summary, sma_50, sma_200, stoch_k,
            ext, asset_key,
        )
    else:
        paragraphs = _build_narrative_en(
            asset_name, close, rsi, trend, momentum, macd_event,
            bb_squeeze, near_support, near_resistance, support_val,
            resistance_val, signals_mixed, summary, sma_50, sma_200, stoch_k,
            ext, asset_key,
        )

    return paragraphs


def _build_tech_observation_response_tr(
    obs, name, close, rsi, trend, momentum, macd_event,
    bb_squeeze, near_support, near_resistance, support_val,
    resistance_val, signals_mixed, summary, sma_50, sma_200, stoch_k,
) -> list[str]:
    """Respond to a technical observation by validating against actual data."""
    obs_lower = obs.lower()
    parts = []

    # Detect what the user is observing
    mentions_bb = any(k in obs_lower for k in ["bollinger", "bant", "band", "bb", "genişleme", "genisleme", "sıkışma", "sikisma", "squeeze"])
    mentions_rsi = any(k in obs_lower for k in ["rsi", "aşırı alım", "asiri alim", "aşırı satım", "asiri satim", "overbought", "oversold"])
    mentions_macd = any(k in obs_lower for k in ["macd", "kesişim", "kesisim", "crossover", "diverjans", "divergence"])
    mentions_trend = any(k in obs_lower for k in ["trend", "sma", "ema", "ortalama", "average", "golden", "death"])
    mentions_support = any(k in obs_lower for k in ["destek", "direnç", "direnc", "support", "resistance", "kırılım", "kirilim", "breakout"])
    mentions_volume = any(k in obs_lower for k in ["hacim", "volume"])
    mentions_down = any(k in obs_lower for k in ["aşağı", "asagi", "düşüş", "dusus", "down", "bearish", "negatif"])
    mentions_up = any(k in obs_lower for k in ["yukarı", "yukari", "yükseliş", "yukselis", "up", "bullish", "pozitif"])

    # Bollinger observation
    if mentions_bb:
        bb_status = "sıkışmış" if bb_squeeze else "normal genişlikte"
        parts.append(
            f"Bollinger bantlarıyla ilgili gözlemini kontrol ediyorum. "
            f"Veriye göre bantlar şu an **{bb_status}**. "
        )
        if bb_squeeze:
            parts.append(
                f"Haklısın, bantlar daralmış durumda. Bu genelde büyük bir hareketin habercisi. "
            )
        if mentions_down:
            if trend in ("down", "strong_down"):
                parts.append(
                    f"Aşağı genişleme gözlemin tutarlı — fiyat {close:,.2f} seviyesinde ve trend zaten aşağı yönlü. "
                    f"Alt bant desteği kırılırsa satış hızlanabilir."
                )
            else:
                parts.append(
                    f"Aşağı genişleme diyorsun ama genel trend henüz tamamen aşağı dönmemiş. "
                    f"Kısa vadeli bir baskı olabilir, uzun vadeli yapı {'hâlâ yukarı' if trend in ('up', 'strong_up') else 'nötr'}."
                )
        elif mentions_up:
            if trend in ("up", "strong_up"):
                parts.append(
                    f"Yukarı genişleme gözlemin verilerle uyumlu — trend yukarı ve bantlar genişliyor. "
                    f"Momentum devam ederse üst bant hedef olabilir."
                )
            else:
                parts.append(
                    f"Yukarı genişleme diyorsun ama genel trend henüz bunu tam desteklemiyor. "
                    f"Kısa vadeli bir tepki olabilir."
                )

    # RSI observation
    if mentions_rsi:
        parts.append(
            f"RSI şu an **{rsi:.0f}** seviyesinde. "
        )
        if rsi > 70:
            parts.append(f"Evet, aşırı alım bölgesinde. MACD {'pozitif' if macd_event in ('bullish_cross', 'positive') else 'negatife dönmüş'} — {'momentum hâlâ arkada ama dikkatli olmak lazım' if macd_event in ('bullish_cross', 'positive') else 'momentum zayıflıyor, düzeltme olasılığı artıyor'}.")
        elif rsi < 30:
            parts.append(f"Evet, aşırı satım bölgesinde. {'Trend de aşağı, dipten alım riskli.' if trend in ('down', 'strong_down') else 'Ama genel yapı henüz kırılmamış, teknik tepki gelebilir.'}")
        else:
            parts.append(f"Ne aşırı alımda ne aşırı satımda — {'pozitif bölgede' if rsi > 50 else 'negatif bölgede'} ama uç değil.")

    # MACD observation
    if mentions_macd:
        if macd_event == "bullish_cross":
            parts.append(f"MACD'de taze bir **pozitif kesişim** var. {'Trend de yukarı, bu kesişim güçlü.' if trend in ('up', 'strong_up') else 'Ama genel trend henüz bunu desteklemiyor, sahte sinyal olabilir.'}")
        elif macd_event == "bearish_cross":
            parts.append(f"MACD'de taze bir **negatif kesişim** var. {'Trend de aşağı, sinyal güçlü görünüyor.' if trend in ('down', 'strong_down') else 'Trend henüz tam aşağı dönmemiş, geçici bir baskı olabilir.'}")
        elif macd_event == "positive":
            parts.append("MACD pozitif bölgede devam ediyor, momentum hâlâ yukarı yönlü.")
        elif macd_event == "negative":
            parts.append("MACD negatif bölgede, momentum aşağı yönlü.")
        else:
            parts.append("MACD şu an belirgin bir sinyal vermiyor.")

    # Trend/MA observation
    if mentions_trend and not mentions_bb:
        if sma_50 is not None and sma_200 is not None:
            if sma_50 > sma_200:
                parts.append(f"SMA 50 ({sma_50:,.2f}) SMA 200'ün ({sma_200:,.2f}) üzerinde — uzun vadeli yapı yukarı yönlü.")
            else:
                parts.append(f"SMA 50 ({sma_50:,.2f}) SMA 200'ün ({sma_200:,.2f}) altında — uzun vadeli yapı zayıf.")
            dist_50 = ((close - sma_50) / sma_50) * 100
            parts.append(f"Fiyat SMA 50'den %{dist_50:+.1f} uzaklıkta.")

    # Support/resistance observation
    if mentions_support:
        if support_val and resistance_val:
            parts.append(
                f"Pivot analizi: destek {support_val:,.2f}, direnç {resistance_val:,.2f}. "
                f"Fiyat {close:,.2f}. "
                f"{'Desteğe çok yakın — kırılma veya tepki kritik.' if near_support else 'Dirence yakın — aşarsa ivme kazanır.' if near_resistance else 'İki seviye arasında, net bir tetik yok.'}"
            )

    # If nothing specific matched, give general validation
    if not parts:
        parts.append(
            f"Gözlemini değerlendirelim: {name} {close:,.2f} seviyesinde. "
            f"RSI {rsi:.0f}, trend {'yukarı' if trend in ('up', 'strong_up') else 'aşağı' if trend in ('down', 'strong_down') else 'yatay'}. "
            f"Sinyaller: {summary['buy']} alım, {summary['sell']} satım, {summary['neutral']} nötr."
        )

    # Add what other indicators say as cross-check
    cross = []
    if not mentions_rsi:
        cross.append(f"RSI {rsi:.0f}")
    if not mentions_macd:
        macd_str = {"bullish_cross": "pozitif kesişim", "bearish_cross": "negatif kesişim", "positive": "pozitif", "negative": "negatif"}.get(macd_event, "nötr")
        cross.append(f"MACD {macd_str}")
    if not mentions_bb and bb_squeeze:
        cross.append("Bollinger sıkışmış")
    if cross:
        parts.append(f"Diğer göstergeler: {', '.join(cross)}.")

    return [" ".join(parts)]


def _build_tech_observation_response_en(
    obs, name, close, rsi, trend, momentum, macd_event,
    bb_squeeze, near_support, near_resistance, support_val,
    resistance_val, signals_mixed, summary, sma_50, sma_200, stoch_k,
) -> list[str]:
    """Respond to a technical observation by validating against actual data (English)."""
    obs_lower = obs.lower()
    parts = []

    mentions_bb = any(k in obs_lower for k in ["bollinger", "band", "bb", "expansion", "squeeze"])
    mentions_rsi = any(k in obs_lower for k in ["rsi", "overbought", "oversold"])
    mentions_macd = any(k in obs_lower for k in ["macd", "crossover", "divergence"])
    mentions_trend = any(k in obs_lower for k in ["trend", "sma", "ema", "average", "golden", "death"])
    mentions_support = any(k in obs_lower for k in ["support", "resistance", "breakout"])
    mentions_down = any(k in obs_lower for k in ["down", "bearish", "negative", "drop", "fall"])
    mentions_up = any(k in obs_lower for k in ["up", "bullish", "positive", "rise", "rally"])

    if mentions_bb:
        bb_status = "compressed" if bb_squeeze else "normal width"
        parts.append(f"Checking your Bollinger observation. Bands are currently **{bb_status}**. ")
        if mentions_down:
            if trend in ("down", "strong_down"):
                parts.append(f"Your downward expansion observation is consistent — price at {close:,.2f}, trend is down. If lower band breaks, selling could accelerate.")
            else:
                parts.append(f"You mention downward expansion but the broader trend hasn't fully turned down. Could be short-term pressure, long-term structure is {'still up' if trend in ('up', 'strong_up') else 'neutral'}.")
        elif mentions_up:
            if trend in ("up", "strong_up"):
                parts.append(f"Upward expansion checks out — trend is up and bands are widening. Upper band could be the target if momentum continues.")
            else:
                parts.append(f"You mention upward expansion but the broader trend doesn't fully support it yet. Could be a short-term bounce.")

    if mentions_rsi:
        parts.append(f"RSI is at **{rsi:.0f}**. ")
        if rsi > 70:
            macd_rsi = "positive, momentum still behind it but watch for exhaustion" if macd_event in ('bullish_cross', 'positive') else "turning negative, correction probability increasing"
            parts.append(f"Confirmed overbought. MACD is {macd_rsi}.")
        elif rsi < 30:
            if trend in ('down', 'strong_down'):
                parts.append("Confirmed oversold. Trend is also down, bottom-fishing is risky.")
            else:
                parts.append("Confirmed oversold. But structure has not broken, technical bounce possible.")
        else:
            zone = "positive zone" if rsi > 50 else "negative zone"
            parts.append(f"Neither overbought nor oversold, in {zone} but not extreme.")

    if mentions_macd:
        if macd_event == "bullish_cross":
            macd_conf = "Trend confirms, strong signal." if trend in ('up', 'strong_up') else "But trend does not confirm yet, could be false signal."
            parts.append(f"Fresh **bullish crossover** on MACD. {macd_conf}")
        elif macd_event == "bearish_cross":
            macd_conf = "Trend confirms, signal looks solid." if trend in ('down', 'strong_down') else "Trend has not fully turned, may be temporary."
            parts.append(f"Fresh **bearish crossover** on MACD. {macd_conf}")
        elif macd_event == "positive":
            parts.append("MACD remains positive, momentum still upward.")
        elif macd_event == "negative":
            parts.append("MACD in negative territory, momentum is down.")

    if mentions_trend and not mentions_bb:
        if sma_50 is not None and sma_200 is not None:
            if sma_50 > sma_200:
                parts.append(f"SMA 50 ({sma_50:,.2f}) above SMA 200 ({sma_200:,.2f}), long-term structure is bullish.")
            else:
                parts.append(f"SMA 50 ({sma_50:,.2f}) below SMA 200 ({sma_200:,.2f}), long-term structure is weak.")
            dist_50 = ((close - sma_50) / sma_50) * 100
            parts.append(f"Price is {dist_50:+.1f}% from SMA 50.")

    if mentions_support:
        if support_val and resistance_val:
            parts.append(
                f"Pivot analysis: support {support_val:,.2f}, resistance {resistance_val:,.2f}. "
                f"Price at {close:,.2f}. "
                f"{'Very close to support, break or bounce is critical.' if near_support else 'Near resistance, breakout means momentum.' if near_resistance else 'Between levels, no clear trigger.'}"
            )

    if not parts:
        parts.append(
            f"Evaluating your observation: {name} at {close:,.2f}. "
            f"RSI {rsi:.0f}, trend {'up' if trend in ('up', 'strong_up') else 'down' if trend in ('down', 'strong_down') else 'sideways'}. "
            f"Signals: {summary['buy']} buy, {summary['sell']} sell, {summary['neutral']} neutral."
        )

    cross = []
    if not mentions_rsi: cross.append(f"RSI {rsi:.0f}")
    if not mentions_macd:
        macd_str = {"bullish_cross": "bullish crossover", "bearish_cross": "bearish crossover", "positive": "positive", "negative": "negative"}.get(macd_event, "neutral")
        cross.append(f"MACD {macd_str}")
    if not mentions_bb and bb_squeeze: cross.append("Bollinger squeeze active")
    if cross:
        parts.append(f"Cross-check — {', '.join(cross)}.")

    return [" ".join(parts)]


def _safe(row, col):
    """Safely get a value, returning None if NaN."""
    val = row.get(col)
    if val is not None and not np.isnan(val):
        return val
    return None


def _build_narrative_tr(
    name, close, rsi, trend, momentum, macd_event,
    bb_squeeze, near_support, near_resistance, support_val,
    resistance_val, signals_mixed, summary, sma_50, sma_200, stoch_k,
    ext="", asset_key="",
) -> list[str]:
    """Build Turkish conversational narrative. If ext (external context) is provided,
    every observation is reframed through that lens using context classification."""
    has_ext = bool(ext)
    ext_cats = _classify_external_context(ext) if has_ext else []
    is_tech_obs = has_ext and ext_cats and ext_cats[0] == "technical_observation"

    # If user shared a technical observation, validate it against data and discuss
    if is_tech_obs:
        return _build_tech_observation_response_tr(
            ext, name, close, rsi, trend, momentum, macd_event,
            bb_squeeze, near_support, near_resistance, support_val,
            resistance_val, signals_mixed, summary, sma_50, sma_200, stoch_k,
        )

    # --- Paragraph 1: The big picture ---
    p1_parts = []

    # Opening with general state — no prefix repetition, dive straight into analysis
    if has_ext:
        p1_parts.append(_ext_trend_commentary_tr(ext_cats, ext, trend, close, name, asset_key))
    elif trend == "strong_up":
        p1_parts.append(
            f"Şu an {name}'a baktığımda genel olarak güçlü bir yükseliş trendi görüyorum. "
            f"Fiyat {close:,.2f} seviyesinde ve hem 50 hem 200 günlük ortalamaların üzerinde seyrediyor, "
            f"bu teknik açıdan sağlam bir yapıya işaret ediyor."
        )
    elif trend == "strong_down":
        p1_parts.append(
            f"{name}'da işler pek parlak görünmüyor açıkçası. "
            f"Fiyat {close:,.2f} seviyesinde, hem 50 hem 200 günlük ortalamaların altında — "
            f"yani klasik tabiriyle 'death cross' bölgesindeyiz. Aşağı yönlü baskı hakim."
        )
    elif trend == "up":
        p1_parts.append(
            f"{name} şu an {close:,.2f} seviyesinde ve genel eğilim yukarı yönlü. "
            f"50 günlük ortalama 200 günlüğün üzerinde, bu olumlu bir işaret."
        )
    elif trend == "down":
        p1_parts.append(
            f"{name} {close:,.2f} seviyesinde ve kısa vadeli ortalamalar uzun vadelilerin altına sarkmış durumda. "
            f"Genel tablo biraz tedirgin edici."
        )
    else:
        p1_parts.append(f"{name} şu an {close:,.2f} seviyesinde işlem görüyor.")

    # Momentum
    if has_ext:
        p1_parts.append(_ext_momentum_commentary_tr(ext_cats, rsi, momentum))
    elif momentum == "overbought":
        p1_parts.append(
            f"RSI {rsi:.0f} ile aşırı alım bölgesinde — "
            f"yani piyasa biraz fazla ısınmış durumda, bir nefes alma ihtimali var."
        )
    elif momentum == "oversold":
        p1_parts.append(
            f"RSI {rsi:.0f} seviyesiyle aşırı satım bölgesine girmiş. "
            f"Satıcılar yorulmuş olabilir, buradan bir tepki gelmesi şaşırtıcı olmaz."
        )
    elif momentum == "bullish":
        p1_parts.append(f"RSI {rsi:.0f} ile pozitif momentum devam ediyor.")
    elif momentum == "bearish":
        p1_parts.append(f"RSI {rsi:.0f} ile momentum zayıflıyor.")

    # MACD
    if macd_event == "bullish_cross":
        p1_parts.append(
            "Üstelik MACD'de tam da şimdi pozitif bir kesişim oldu — "
            "bu momentum değişiminin taze bir sinyali."
        )
    elif macd_event == "bearish_cross":
        p1_parts.append(
            "Buna ek olarak MACD'de negatif kesişim yaşandı, "
            "bu da aşağı yönlü baskının taze olduğunu gösteriyor."
        )

    paragraph_1 = " ".join(p1_parts)

    # --- Paragraph 2: What stands out + what to watch ---
    p2_parts = []

    if bb_squeeze:
        if has_ext:
            p2_parts.append(_ext_bb_commentary_tr(ext_cats))
        else:
            p2_parts.append(
                "Beni en çok heyecanlandıran şey Bollinger bantlarının ciddi şekilde sıkışmış olması. "
                "Bu genelde büyük bir hareketin habercisi — yön belli değil ama volatilite patlaması kapıda."
            )
    elif signals_mixed:
        p2_parts.append(
            f"Açıkçası burada göstergeler birbiriyle çelişiyor: "
            f"{summary['buy']} tanesi alım, {summary['sell']} tanesi satım diyor. "
            f"Bu tür kararsız dönemlerde piyasa genelde bir süre yatay gidip sonra sert kırar."
        )
    elif momentum == "overbought" and trend == "strong_up":
        p2_parts.append(
            "İlginç olan şu: trend çok güçlü ama aynı zamanda göstergeler aşırı alım diyor. "
            "Bu tür durumlarda ya sert bir düzeltme gelir ya da fiyat yüksek seviyelerde konsolide olur. "
            "İkisine de hazırlıklı olmak lazım."
        )
    elif momentum == "oversold" and trend == "strong_down":
        p2_parts.append(
            "Dikkat çekici bir nokta: hem trend aşağı hem de göstergeler aşırı satımda. "
            "Tarihsel olarak bu seviyelerden teknik tepkiler gelir ama trend bu kadar güçlüyken "
            "düşüşün devam etmesi de olasılıklar dahilinde."
        )
    elif macd_event in ("bullish_cross", "bearish_cross"):
        direction = "yukarı" if macd_event == "bullish_cross" else "aşağı"
        p2_parts.append(
            f"MACD'deki taze kesişim bence buradaki en önemli sinyal. "
            f"{direction.capitalize()} yönlü bir momentum başlangıcı olabilir, "
            f"ama bunu hacimle teyit etmek şart."
        )

    # Support/resistance
    if near_support and support_val is not None:
        p2_parts.append(
            f"Bir de şuna dikkat etmekte fayda var: fiyat {support_val:,.2f} destek seviyesine çok yakın. "
            f"Buradan tepki gelebilir ama kırılırsa hızlı bir düşüş yaşanabilir — kritik bir eşikteyiz."
        )
    elif near_resistance and resistance_val is not None:
        p2_parts.append(
            f"Gözden kaçmaması gereken bir detay: {resistance_val:,.2f} direnç seviyesi hemen yukarıda. "
            f"Bu seviyeyi aşarsa momentum kazanabilir, aşamazsa geri çekilme görülür."
        )

    # General closing
    if not p2_parts:
        total = summary['buy'] + summary['sell'] + summary['neutral']
        if summary["overall"] == "buy":
            p2_parts.append(
                f"Genel olarak göstergelerin çoğunluğu ({summary['buy']}/{total}) "
                f"pozitif yönde. Teknik tablo iyimser görünüyor."
            )
        elif summary["overall"] == "sell":
            p2_parts.append(
                f"Göstergelerin çoğunluğu ({summary['sell']}/{total}) "
                f"negatif sinyal veriyor. Teknik tablo temkinli olmayı gerektiriyor."
            )
        else:
            p2_parts.append(
                "Genel olarak karışık bir tablo var, net bir yön belirlemek zor. "
                "Net bir yön belirlemek zor, kararsız bir dönemdeyiz."
            )

    paragraph_2 = " ".join(p2_parts)

    return [paragraph_1, paragraph_2]


def _build_narrative_en(
    name, close, rsi, trend, momentum, macd_event,
    bb_squeeze, near_support, near_resistance, support_val,
    resistance_val, signals_mixed, summary, sma_50, sma_200, stoch_k,
    ext="", asset_key="",
) -> list[str]:
    """Build English conversational narrative with optional external context."""
    has_ext = bool(ext)
    ext_cats = _classify_external_context(ext) if has_ext else []
    is_tech_obs = has_ext and ext_cats and ext_cats[0] == "technical_observation"

    if is_tech_obs:
        return _build_tech_observation_response_en(
            ext, name, close, rsi, trend, momentum, macd_event,
            bb_squeeze, near_support, near_resistance, support_val,
            resistance_val, signals_mixed, summary, sma_50, sma_200, stoch_k,
        )

    p1_parts = []

    # Trend — no prefix, jump straight to analysis
    if has_ext:
        p1_parts.append(_ext_trend_commentary_en(ext_cats, ext, trend, close, name, asset_key))
    elif trend == "strong_up":
        p1_parts.append(
            f"Looking at {name} right now, I see a solid uptrend. "
            f"Price is at {close:,.2f}, trading above both the 50 and 200-day moving averages — "
            f"that's a technically strong structure."
        )
    elif trend == "strong_down":
        p1_parts.append(
            f"Honestly, {name} doesn't look great right now. "
            f"Price is at {close:,.2f}, below both the 50 and 200-day averages — "
            f"we're in classic 'death cross' territory. Bears are in control."
        )
    elif trend == "up":
        p1_parts.append(
            f"{name} is at {close:,.2f} with a generally positive bias. "
            f"The 50-day average is above the 200-day, which is an encouraging sign."
        )
    elif trend == "down":
        p1_parts.append(
            f"{name} is at {close:,.2f} and short-term averages have dipped below long-term ones. "
            f"The overall picture is somewhat concerning."
        )
    else:
        p1_parts.append(f"{name} is currently trading at {close:,.2f}.")

    # Momentum
    if has_ext:
        p1_parts.append(_ext_momentum_commentary_en(ext_cats, rsi, momentum))
    elif momentum == "overbought":
        p1_parts.append(
            f"RSI is at {rsi:.0f}, deep in overbought territory — "
            f"the market may be running a bit hot and could need a breather."
        )
    elif momentum == "oversold":
        p1_parts.append(
            f"RSI has dropped to {rsi:.0f}, well into oversold territory. "
            f"Sellers may be exhausted; a bounce from here wouldn't be surprising."
        )
    elif momentum == "bullish":
        p1_parts.append(f"RSI at {rsi:.0f} shows positive momentum continuing.")
    elif momentum == "bearish":
        p1_parts.append(f"RSI at {rsi:.0f} suggests momentum is fading.")

    # MACD
    if macd_event == "bullish_cross":
        p1_parts.append(
            "On top of that, MACD just had a bullish crossover — "
            "a fresh signal of potential momentum shift."
        )
    elif macd_event == "bearish_cross":
        p1_parts.append(
            "Adding to that, MACD just crossed bearish, "
            "confirming fresh downward pressure."
        )

    paragraph_1 = " ".join(p1_parts)

    p2_parts = []

    if bb_squeeze:
        if has_ext:
            p2_parts.append(_ext_bb_commentary_en(ext_cats))
        else:
            p2_parts.append(
                "What really catches my eye is the Bollinger Band squeeze. "
                "Bands are tightly compressed, which historically precedes a major move. "
                "Direction is uncertain, but a volatility explosion is coming."
            )
    elif signals_mixed:
        p2_parts.append(
            f"Here's the thing — indicators are conflicting: "
            f"{summary['buy']} say buy, {summary['sell']} say sell. "
            f"In these indecisive periods, markets tend to chop sideways before breaking sharply."
        )
    elif momentum == "overbought" and trend == "strong_up":
        p2_parts.append(
            "The interesting tension here is that the trend is strong but indicators scream overbought. "
            "Either a sharp correction comes, or price consolidates at these elevated levels. "
            "Worth being prepared for both scenarios."
        )
    elif momentum == "oversold" and trend == "strong_down":
        p2_parts.append(
            "What's notable is that both the trend is down and indicators are oversold. "
            "Historically, these levels trigger technical bounces, but with a trend this strong, "
            "further downside is also on the table."
        )
    elif macd_event in ("bullish_cross", "bearish_cross"):
        direction = "upward" if macd_event == "bullish_cross" else "downward"
        p2_parts.append(
            f"The fresh MACD crossover is arguably the most important signal here. "
            f"It could mark the beginning of {direction} momentum, "
            f"but volume confirmation is essential."
        )

    # Support/resistance
    if near_support and support_val is not None:
        p2_parts.append(
            f"Also worth noting: price is very close to the {support_val:,.2f} support level. "
            f"A bounce is possible, but a break below could trigger a swift decline — we're at a critical threshold."
        )
    elif near_resistance and resistance_val is not None:
        p2_parts.append(
            f"Don't miss this: the {resistance_val:,.2f} resistance is right overhead. "
            f"A breakout could fuel more upside, but rejection here means a pullback."
        )

    if not p2_parts:
        total = summary['buy'] + summary['sell'] + summary['neutral']
        if summary["overall"] == "buy":
            p2_parts.append(
                f"Overall, the majority of indicators ({summary['buy']}/{total}) "
                f"lean positive. The technical picture looks constructive."
            )
        elif summary["overall"] == "sell":
            p2_parts.append(
                f"The majority of indicators ({summary['sell']}/{total}) "
                f"are signaling negative. The technical picture warrants caution."
            )
        else:
            p2_parts.append(
                "Overall it's a mixed bag — hard to call a clear direction. "
                "Hard to call a clear direction — signals are indecisive."
            )

    paragraph_2 = " ".join(p2_parts)

    return [paragraph_1, paragraph_2]
