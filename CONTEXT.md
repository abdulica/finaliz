# Finaliz — Proje Bağlamı

## Çalışan Uygulama
- **Local:** `python gradio_app.py` → http://localhost:7860
- **Canlı:** https://finaliz.onrender.com
- **GitHub:** https://github.com/abdulica/finaliz
- **Deploy:** `git push origin master` → Render otomatik deploy eder

## Teknoloji Yığını
- **UI:** Gradio 6.x (Streamlit'ten geçildi)
- **Veri:** yfinance (1y), teknik analiz: `ta` kütüphanesi
- **AI:** OpenRouter → `stepfun/step-3.5-flash:free`
- **Deploy:** Render (ücretsiz, 15dk uyku)

## Dosya Yapısı (aktif)
```
gradio_app.py          ← Ana uygulama (tek dosya)
config.py              ← Varlık tanımları (ASSETS)
requirements.txt       ← gradio, yfinance, ta, openai...
data/
  cache.py             ← In-memory cache (Streamlit bağımsız)
  fetcher.py           ← yfinance veri çekme
  windowing.py         ← Teknik analiz penceresi
analysis/
  technical.py         ← RSI, MACD, BB, SMA (import ta)
  llm_engine.py        ← build_context() fonksiyonu
  forecast.py          ← Prophet (deploy'da devre dışı)
components/
  tradingview.py       ← TRADINGVIEW_SYMBOLS dict
utils/
  i18n.py              ← get_asset_name()
```

## Varlıklar (ASSET_TICKERS)
DXY, USD/TRY, EUR/USD, USD/JPY, USD/CHF, XAU/USD, XAG/USD, BTC/USD, WTI, BRENT

## gradio_app.py Yapısı
1. `get_data()` — yfinance 1y cache
2. `stream_chat()` — OpenRouter streaming
3. `price_cards(lang)` — Günlük + Haftalık HTML kartlar
4. `tv_embed(key)` — TradingView iframe
5. `ta_card(key, lang)` — Teknik analiz özet kartı
6. **UI Blokları:**
   - Header HTML (başlık + TR|EN|Verileri Güncelle linkleri)
   - CheckboxGroup (varlık seçimi, yatay)
   - Tabs: Genel Bakış / Varlık Detay / Tahmin / Karşılaştırma
   - Chatbot (streaming, dict format)

## Bekleyen Sorunlar
1. **TR/EN toggle çalışmıyor** — HTML link → JS → Gradio button click zinciri
   - Çözüm yolu: `gr.Button(elem_id=...)` + JS `document.getElementById().click()`
   - Alternatif: Gradio `js=` parametresi ile direkt JS tetikleme
2. **Verileri Güncelle linki pasif** — aynı JS sorunu

## Notlar
- `views/` klasörü artık kullanılmıyor (Streamlit kalıntısı)
- `app.py` artık kullanılmıyor (eski Streamlit)
- `data/cache.py` Streamlit'ten tamamen koparıldı (in-memory dict)
- Render'da FRED verisi atlanıyor (`md = {}`)
