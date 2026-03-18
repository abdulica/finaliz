# Finaliz - Finansal Analiz ve Tahmin Platformu

## Context
Kullanıcı; Dolar, Euro, Altın, Gümüş, BTC, Petrol, Demir ve DXY için teknik analiz + makro filtreli forecast yapan, grafik destekli, tartışma arkadaşı gibi çalışan bir araç istiyor. Chatbot değil, kendi analizini otomatik üreten bir sistem. Yatırım tavsiyesi vermeyecek, sadece ufuk açıcı gözlemler sunacak.

## Tech Stack
- **Python + Streamlit** (ana framework)
- **yfinance** (fiyat verileri)
- **fredapi** (makro veriler - FRED ücretsiz API key gerekli, kullanıcı kendi alacak)
- **ta** (teknik analiz indikatörleri)
- **Prophet** (forecast modeli)
- **Plotly** (interaktif grafikler)
- **pandas/numpy** (veri işleme)

## Proje Dizini: `C:/Users/abdulica/projects/finaliz/`

## Mimari

```
finaliz/
├── app.py                    # Streamlit ana uygulama
├── requirements.txt
├── config.py                 # Ayarlar, asset tanımları, dil dosyaları
├── data/
│   ├── fetcher.py            # Veri çekme modülü (yfinance, FRED)
│   └── cache.py              # Basit dosya tabanlı cache
├── analysis/
│   ├── technical.py          # Teknik analiz (RSI, MACD, Bollinger, MA, vs.)
│   ├── macro.py              # Makro veri analizi ve filtreleme
│   ├── seasonal.py           # Mevsimsel analiz
│   └── forecast.py           # Prophet tabanlı forecast
├── views/
│   ├── dashboard.py          # Ana dashboard sayfası
│   ├── asset_detail.py       # Tek varlık detay sayfası
│   ├── forecast_view.py      # Tahmin sayfası (1g, 3g, 1h, 2h, 4h)
│   ├── comparison.py         # Varlık karşılaştırma
│   └── macro_overview.py     # Makro gösterge paneli
├── components/
│   ├── charts.py             # Plotly grafik bileşenleri
│   ├── analysis_card.py      # Analiz metin kartları
│   └── sidebar.py            # Sidebar navigasyon
└── utils/
    ├── helpers.py             # Yardımcı fonksiyonlar
    └── i18n.py               # Dil desteği (TR/EN)
```

## Varlıklar (USD bazında)

| Varlık | yfinance Ticker | Açıklama |
|--------|----------------|----------|
| Dolar (DXY) | DX-Y.NYB | Dolar endeksi |
| EUR/USD | EURUSD=X | Euro-Dolar paritesi |
| Altın | GC=F | Gold Futures |
| Gümüş | SI=F | Silver Futures |
| BTC | BTC-USD | Bitcoin |
| Petrol | CL=F | Crude Oil Futures |
| Demir | - | FRED: Iron ore price (PIORECRUSDM) veya yfinance alternatif |

## Uygulama Adımları

### Adım 1: Proje altyapısı
- Dizin yapısını oluştur
- `requirements.txt` yaz
- `config.py` - asset tanımları, renk paleti, dil sözlükleri
- `utils/i18n.py` - TR/EN dil desteği

### Adım 2: Veri katmanı
- `data/fetcher.py` - yfinance ile fiyat verileri çekme (1y, 2y, 5y seçenekleri)
- `data/fetcher.py` - FRED ile makro veriler (faiz, enflasyon, işsizlik, M2 para arzı)
- `data/cache.py` - Session-based cache (Streamlit session_state)

### Adım 3: Teknik analiz motoru
- `analysis/technical.py`:
  - RSI (14 gün)
  - MACD (12, 26, 9)
  - Bollinger Bantları (20, 2)
  - Hareketli Ortalamalar (SMA 20, 50, 200 + EMA 12, 26)
  - Stochastic RSI
  - ATR (volatilite)
  - Destek/Direnç seviyeleri (pivot noktaları)
  - Hacim analizi
  - Her indikatör için otomatik yorum üretimi (TR/EN)

### Adım 4: Makro ve mevsimsel analiz
- `analysis/macro.py`:
  - FED faiz oranı, CPI, işsizlik, M2 para arzı
  - Makro verilerin asset'ler üzerindeki tarihsel korelasyonu
  - Otomatik makro yorum: "FED faiz artışı dönemlerinde altın genellikle..."
- `analysis/seasonal.py`:
  - Aylık/çeyreklik mevsimsel pattern analizi
  - Tarihsel mevsimsel ortalamalar vs mevcut durum

### Adım 5: Forecast motoru
- `analysis/forecast.py`:
  - Prophet modeli ile 1, 3, 7, 14, 28 günlük tahmin
  - Güven aralıkları (confidence intervals) gösterimi
  - Teknik analiz sinyalleriyle filtreleme (trend uyumu kontrolü)
  - Makro verilerle ağırlıklandırma
  - Forecast sonuçlarını metin olarak yorumlama

### Adım 6: Grafik bileşenleri
- `components/charts.py`:
  - Candlestick + overlay indikatörler (MA, Bollinger)
  - RSI/MACD alt grafikleri
  - Forecast grafikleri (gerçek + tahmin + güven aralığı)
  - Korelasyon heatmap
  - Mevsimsel pattern grafikleri
  - Karşılaştırma grafikleri (normalize edilmiş)

### Adım 7: Sayfa görünümleri
- **Dashboard**: Tüm varlıkların özet tablosu, günlük değişim, mini grafikler, genel piyasa sentiment
- **Varlık Detay**: Seçilen varlık için tam teknik analiz + grafikler + metin yorumlar
- **Tahmin**: Forecast grafikleri + güven aralıkları + metin açıklamalar
- **Karşılaştırma**: İki veya daha fazla varlığı yan yana analiz
- **Makro Göstergeler**: Makro veri paneli + korelasyonlar

### Adım 8: Analiz metin üreteci
- Her varlık için otomatik analiz raporu üretimi
- "Arkadaş gibi" konuşma tonu:
  - "Altında ilginç bir durum var: RSI 30'un altına düştü ama MACD pozitif kesişim yapmak üzere. Bu tür çelişkili sinyaller genelde..."
  - "DXY güçlenirken altının bu kadar dirençli kalması dikkat çekici. Geçmişte benzer dönemlerde..."
- Yatırım tavsiyesi disclaimer'ı her sayfada

## Dil Desteği
- Sidebar'da TR/EN geçiş butonu
- Tüm metinler, etiketler, analizler iki dilde
- `utils/i18n.py` ile merkezi yönetim

## Veri Güncelleme
- Uygulama açılışında otomatik veri çekme
- Sidebar'da "Verileri Güncelle / Refresh Data" butonu
- Son güncelleme zamanı gösterimi

## Önemli Notlar
- FRED API key gerekli (ücretsiz, fred.stlouisfed.org'dan alınır) - `.env` dosyasında saklanacak
- Demir fiyatı için FRED verisi kullanılacak (aylık), alternatif olarak yfinance'den VALE veya proxy kullanılabilir
- Yatırım tavsiyesi disclaimer'ı her sayfada görünür olacak
- Tüm analizler "bilgilendirme amaçlıdır" notu ile sunulacak

## Doğrulama / Test
1. `pip install -r requirements.txt` ile bağımlılıkları kur
2. `.env` dosyasına FRED API key ekle
3. `streamlit run app.py` ile çalıştır
4. Her varlık için veri çekildiğini kontrol et
5. Teknik analiz grafiklerinin doğru çizildiğini kontrol et
6. Forecast grafiklerinin güven aralıklarıyla birlikte göründüğünü doğrula
7. TR/EN dil geçişinin çalıştığını test et
8. Manuel veri yenileme butonunu test et
