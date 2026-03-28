"""Finaliz — Gradio arayüzü. Çalıştır: python gradio_app.py"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os, re
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from data.fetcher import fetch_all_assets, fetch_macro_data, get_latest_prices
from data.cache import get_cached, set_cached
from data.windowing import window_for_technical, window_for_forecast
from analysis.technical import compute_all_indicators, get_signal_summary
try:
    from analysis.forecast import run_forecast
    FORECAST_AVAILABLE = True
except Exception:
    FORECAST_AVAILABLE = False
    def run_forecast(*a, **k): return None
from analysis.llm_engine import build_context
from components.tradingview import TRADINGVIEW_SYMBOLS
from config import ASSETS
from utils.i18n import get_asset_name

def get_data(force=False):
    cached = get_cached("assets_5y")
    if cached and not force:
        return cached, get_cached("macro") or {}
    print("Veriler çekiliyor...")
    ad = fetch_all_assets("1y")  # Deploy'da 1y yeterli, daha hızlı
    md = {}  # FRED verisi deploy'da atla, yavaşlatıyor
    set_cached("assets_5y", ad)
    set_cached("macro", md)
    return ad, md

ASSET_TICKERS = {
    "DXY":"DXY","USDTRY":"USD/TRY","EURUSD":"EUR/USD","USDJPY":"USD/JPY",
    "USDCHF":"USD/CHF","GOLD":"XAU/USD","SILVER":"XAG/USD",
    "BTC":"BTC/USD","OIL":"WTI","BRENT":"BRENT",
}
ASSET_NAMES = {k: get_asset_name(k, v, "tr") for k, v in ASSETS.items()}
ALL_KEYS = list(ASSETS.keys())

def name_to_key(name):
    for k, t in ASSET_TICKERS.items():
        if t == name: return k
    for k, v in ASSET_NAMES.items():
        if v == name: return k
    return "GOLD"

SYSTEM = """Sen kıdemli bir piyasa analistisin. Görevin: spesifik, koşula bağlı, cesur analiz yapmak.
YANIT FORMATI: 1-Tek cümle özet 2-Baskın faktör analizi 3-Konsensüs vs veri karşılaştırması 4-2 senaryo + tetikleyici 5-İzlenecek gösterge.
YASAK: Belirsiz laflar, genel ders, bullet list, 300 kelime aşımı, yarım cümle."""

def _client():
    k = os.getenv("OPENROUTER_API_KEY","")
    if not k or "your_" in k: return None
    return OpenAI(api_key=k, base_url="https://openrouter.ai/api/v1",
                  default_headers={"HTTP-Referer":"finaliz","X-Title":"Finaliz"})

def _ctx(keys):
    ad,_ = get_data()
    parts = []
    for ak in keys:
        df = ad.get(ak)
        if df is None or df.empty: continue
        dft = compute_all_indicators(window_for_technical(df).copy())
        parts.append(build_context(dft, get_asset_name(ak,ASSETS[ak],"tr"), ak, "tr"))
    return "\n\n".join(parts)

def _news(q):
    import urllib.request, urllib.parse, json
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(q+' financial')}&format=json&no_html=1&skip_disambig=1"
        with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"}),timeout=5) as r:
            d = json.loads(r.read().decode())
        res = ([d["Abstract"]] if d.get("Abstract") else [])+[t["Text"] for t in d.get("RelatedTopics",[])[:4] if isinstance(t,dict) and t.get("Text")]
        if res: return "=== GÜNCEL ===\n"+"\n".join(f"• {x}" for x in res)
    except: pass
    return ""

NEWS_KW = ["haber","gündem","gelişme","son dakika","bu hafta","bugün","duyuru","karar","toplantı","veri","rapor","news","today","latest","recent"]

def stream_chat(msg, history, asset_keys, lang="tr"):
    cl = _client()
    if not cl: yield "⚠️ API anahtarı bulunamadı."; return
    keys = [name_to_key(n) for n in (asset_keys or [])]
    ctx = _ctx(keys)
    nc = _news(msg) if any(k in msg.lower() for k in NEWS_KW) else ""
    full_ctx = "\n\n".join(filter(None,[ctx,nc]))
    sys_msg = SYSTEM+(f"\n\n=== CANLI VERİ ===\n{full_ctx}" if full_ctx else "")
    msgs = [{"role":"system","content":sys_msg}]
    for item in (history or []):
        if isinstance(item,dict) and "role" in item:
            msgs.append({"role":item["role"],"content":item["content"]})
    msgs.append({"role":"user","content":msg})
    try:
        stream = cl.chat.completions.create(model="stepfun/step-3.5-flash:free",
            messages=msgs, stream=True, max_tokens=2500, temperature=0.4)
        partial = ""
        for chunk in stream:
            d = chunk.choices[0].delta
            if d and d.content:
                partial += d.content
                yield partial
    except Exception as e:
        yield f"⚠️ Hata: {str(e)[:150]}"

def price_cards(lang="tr"):
    ad,_ = get_data()
    prices = get_latest_prices(ad)
    keys = list(prices.keys())
    n = len(keys)
    GRID = f"display:grid;grid-template-columns:repeat({n},1fr);gap:5px;width:100%;box-sizing:border-box;"
    CARD = "background:#1e1e2e;border-radius:7px;padding:6px 4px;text-align:center;box-sizing:border-box;min-width:0;overflow:hidden;"
    daily_lbl = "📅 Günlük Değişim" if lang=="tr" else "📅 Daily Change"
    weekly_lbl = "📊 Haftalık En Düşük / En Yüksek" if lang=="tr" else "📊 Weekly Low / High"
    daily = []
    for key in keys:
        info = prices[key]; ticker = ASSET_TICKERS.get(key,key)
        pct,close = info["change_pct"],info["close"]
        c = "#26a69a" if pct>=0 else "#ef5350"; a = "▲" if pct>=0 else "▼"
        daily.append(f'<div style="{CARD}border-left:3px solid {c};"><div style="color:#aaa;font-size:0.75em;font-weight:600;">{ticker}</div><div style="color:#fff;font-size:0.92em;font-weight:bold;margin:3px 0;">${close:,.2f}</div><div style="color:{c};font-size:0.8em;">{a}{pct:+.1f}%</div></div>')
    weekly = []
    for key in keys:
        df = ad.get(key); ticker = ASSET_TICKERS.get(key,key)
        if df is None or df.empty: weekly.append(f'<div style="{CARD}"></div>'); continue
        l7=df.tail(7); wh=l7["High"].max(); wl=l7["Low"].min(); cur=l7["Close"].iloc[-1]
        pos=((cur-wl)/(wh-wl)*100) if (wh-wl)>0 else 50
        weekly.append(f'<div style="{CARD}"><div style="color:#aaa;font-size:0.75em;font-weight:600;">{ticker}</div><div style="background:#2a2a3e;border-radius:2px;height:5px;margin:5px 0;"><div style="background:#FFD700;width:{pos:.0f}%;height:100%;border-radius:2px;"></div></div><div style="display:flex;justify-content:space-between;font-size:0.72em;"><span style="color:#ef5350;">${wl:,.1f}</span><span style="color:#26a69a;">${wh:,.1f}</span></div></div>')
    return (f'<div style="color:#aaa;font-size:0.82em;margin-bottom:4px;font-weight:500;">{daily_lbl}</div>'
            f'<div style="{GRID}">{"".join(daily)}</div>'
            f'<div style="color:#aaa;font-size:0.82em;margin:8px 0 4px;font-weight:500;">{weekly_lbl}</div>'
            f'<div style="{GRID}">{"".join(weekly)}</div>')

def tv_embed(asset_key, height=460):
    sym = TRADINGVIEW_SYMBOLS.get(asset_key,"TVC:GOLD")
    enc = sym.replace("/","%2F").replace(":","%3A")
    url = (f"https://www.tradingview.com/widgetembed/?frameElementId=tv_{asset_key}"
           f"&symbol={enc}&interval=D&hidesidetoolbar=0&hidetoptoolbar=0"
           f"&saveimage=0&toolbarbg=1e1e2e&theme=dark&style=1&locale=tr"
           f"&studies=RSI%40tv-basicstudies%1FMACD%40tv-basicstudies&withdateranges=1")
    return (f'<div style="height:{height}px;width:100%;border-radius:8px;overflow:hidden;">'
            f'<iframe src="{url}" style="width:100%;height:100%;border:none;" allowtransparency="true" scrolling="no" frameborder="0"></iframe></div>')

def ta_card(asset_key, lang="tr"):
    ad,_ = get_data(); df = ad.get(asset_key)
    if df is None: return "<p>Veri yok</p>"
    df = compute_all_indicators(window_for_technical(df).copy())
    s = get_signal_summary(df)
    name = get_asset_name(asset_key,ASSETS[asset_key],lang)
    oc = {"buy":"#26a69a","sell":"#ef5350","neutral":"#FFA726"}.get(s["overall"],"#FFA726")
    ol = ({"buy":"ALIM","sell":"SATIM","neutral":"NÖTR"} if lang=="tr" else {"buy":"BUY","sell":"SELL","neutral":"NEUTRAL"}).get(s["overall"],"—")
    labels = ([("Alım","buy"),("Nötr","neutral"),("Satım","sell")] if lang=="tr" else [("Buy","buy"),("Neutral","neutral"),("Sell","sell")])
    ind = ([("RSI","RSI"),("MACD","MACD"),("BB Üst","BB_Upper"),("BB Alt","BB_Lower"),("SMA50","SMA_50"),("SMA200","SMA_200")] if lang=="tr"
           else [("RSI","RSI"),("MACD","MACD"),("BB Up","BB_Upper"),("BB Lo","BB_Lower"),("SMA50","SMA_50"),("SMA200","SMA_200")])
    last = df.iloc[-1]
    def v(c):
        import numpy as np
        try: f=float(last.get(c,float("nan"))); return "—" if (np.isnan(f) or np.isinf(f)) else f"{f:.2f}"
        except: return "—"
    cm = {"buy":"#26a69a","sell":"#ef5350","neutral":"#FFA726"}
    return (f'<div style="background:#1e1e2e;padding:12px;border-radius:10px;margin-top:6px;">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;"><b style="color:#fff;">{name}</b>'
            f'<span style="background:{oc};color:#fff;padding:2px 10px;border-radius:14px;font-size:0.85em;">{ol}</span></div>'
            f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:5px;font-size:0.8em;">'
            +"".join(f'<div style="background:#13131f;padding:5px;border-radius:5px;"><div style="color:#888;">{l}</div><b style="color:{cm.get(k,"#aaa")}">{s[k]}</b></div>' for l,k in labels)
            +f'</div><div style="display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-top:5px;font-size:0.78em;">'
            +"".join(f'<div style="background:#13131f;padding:5px;border-radius:5px;"><span style="color:#888;">{l}: </span><span style="color:#fff;">{v(c)}</span></div>' for l,c in ind)
            +"</div></div>")

CSS = """
body,.gradio-container{background:#0e1117!important;color:#e0e0e0!important}
footer{display:none!important}
.asset-cb fieldset>div,.asset-cb .wrap{display:flex!important;flex-wrap:wrap!important;flex-direction:row!important;gap:4px!important}
"""

with gr.Blocks(title="Finaliz", css=CSS) as demo:

    gr.HTML("""<style>
div[role="tablist"]{background:#0a0a14!important;border-bottom:2px solid #FFD700!important;padding:0 8px!important;margin-bottom:8px!important}
div[role="tablist"] button{color:#888!important;font-size:1.05em!important;font-weight:600!important;padding:12px 28px!important;border:none!important;background:transparent!important;cursor:pointer!important}
div[role="tablist"] button:hover{color:#ddd!important}
div[role="tablist"] button[aria-selected="true"]{color:#FFD700!important;border-bottom:3px solid #FFD700!important;background:rgba(255,215,0,0.07)!important;border-radius:4px 4px 0 0!important}
</style>""")

    gr.Markdown("# 📊 Finaliz — Finansal Analiz Platformu")

    with gr.Row():
        with gr.Column(scale=8):
            asset_cb = gr.CheckboxGroup(
                choices=list(ASSET_TICKERS.values()),
                value=list(ASSET_TICKERS.values()),
                label="Varlıklar", interactive=True,
                elem_classes=["asset-cb"])
        with gr.Column(scale=0, min_width=110):
            lang_btn_tr = gr.Button("TR", variant="primary", size="sm")
            lang_btn_en = gr.Button("EN", variant="secondary", size="sm")
            refresh_btn = gr.Button("🔄 Yenile", size="sm")

    with gr.Tabs():
        with gr.Tab("🏠 Genel Bakış"):
            dash_html = gr.HTML()
        with gr.Tab("📊 Varlık Detay"):
            detail_dd = gr.Dropdown(choices=list(ASSET_TICKERS.values()), value=ASSET_TICKERS["USDTRY"], label="Varlık Seç")
            detail_tv = gr.HTML()
            detail_ta = gr.HTML()
        with gr.Tab("🔮 Tahmin"):
            fc_dd = gr.Dropdown(choices=list(ASSET_TICKERS.values()), value=ASSET_TICKERS["GOLD"], label="Varlık")
            fc_btn = gr.Button("▶ Tahmini Çalıştır", variant="primary")
            fc_status = gr.Markdown("")
            fc_table = gr.Dataframe(headers=["Periyot","Tahmin","Δ%","Alt","Üst","Yön"], interactive=False, visible=False)
            fc_chart = gr.Plot(visible=False)
        with gr.Tab("⚖️ Karşılaştırma"):
            with gr.Row():
                cmp_a = gr.Dropdown(choices=list(ASSET_TICKERS.values()), value=ASSET_TICKERS["GOLD"], label="Varlık A")
                cmp_b = gr.Dropdown(choices=list(ASSET_TICKERS.values()), value=ASSET_TICKERS["BTC"], label="Varlık B")
            cmp_btn = gr.Button("Karşılaştır", variant="primary")
            cmp_chart = gr.Plot(visible=False)

    gr.Markdown("---\n### 🤖 AI Finans Asistanı")
    chatbot = gr.Chatbot(height=360, show_label=False)
    with gr.Row():
        chat_in = gr.Textbox(placeholder="Sorunuzu yazın...", show_label=False, scale=5, container=False)
        send_btn = gr.Button("▶ Gönder", variant="primary", scale=1)
        clr_btn  = gr.Button("🗑️", scale=0)
    gr.Examples(examples=["Haftanın finansal gelişmeleri neler?","USD/TRY için TCMB etkisini analiz et","Altın ve dolar korelasyonu nasıl?","BTC teknik görünümü?","FED faiz kararı piyasalara nasıl yansır?"], inputs=chat_in)

    # Handlers
    lang_state = gr.State("tr")

    def set_tr(): return "tr", gr.update(variant="primary"), gr.update(variant="secondary")
    def set_en(): return "en", gr.update(variant="secondary"), gr.update(variant="primary")

    def load_dash(lang): return price_cards(lang)
    def on_detail(name, lang): k=name_to_key(name); return tv_embed(k), ta_card(k,lang)
    def on_refresh(lang): get_data(force=True); return price_cards(lang)

    def on_forecast(name):
        import pandas as pd
        k=name_to_key(name); ad,_=get_data(); df=ad.get(k)
        if df is None: return "⚠️ Veri yok", gr.update(visible=False), gr.update(visible=False)
        res=run_forecast(window_for_forecast(df))
        if res is None: return "⚠️ Başarısız", gr.update(visible=False), gr.update(visible=False)
        rows=[]
        for h in ["1d","3d","1w","2w","4w"]:
            r=res.get(h)
            if r: rows.append([h,f"${r['predicted']:.2f}",f"{r['change_pct']:+.1f}%",f"${r['lower']:.2f}",f"${r['upper']:.2f}",{"up":"↑","down":"↓","sideways":"→"}.get(r["direction"],"?")])
        ffc,act=res.get("_full_forecast"),res.get("_actual_data"); fig=None
        if ffc is not None and act is not None:
            from components.charts import create_forecast_chart
            fig=create_forecast_chart(act,ffc,k,"tr")
        return "✅",gr.update(value=pd.DataFrame(rows,columns=["Periyot","Tahmin","Δ%","Alt","Üst","Yön"]),visible=True),gr.update(value=fig,visible=fig is not None)

    def on_compare(na,nb):
        ka,kb=name_to_key(na),name_to_key(nb); ad,_=get_data(); da,db=ad.get(ka),ad.get(kb)
        if da is None or db is None: return gr.update(visible=False)
        from components.charts import create_relationship_chart
        return gr.update(value=create_relationship_chart(da,db,ka,kb,"tr"),visible=True)

    def submit(msg, hist, names, lang):
        if not msg or not msg.strip(): return
        hist=hist or []; keys=[name_to_key(n) for n in (names or [])]
        partial=""
        for token in stream_chat(msg,hist,keys,lang):
            partial=token
            yield hist+[{"role":"user","content":msg},{"role":"assistant","content":partial}],""
        if partial:
            new_hist=hist+[{"role":"user","content":msg},{"role":"assistant","content":partial}]
            last_char=partial.strip()[-1] if partial.strip() else ""
            if last_char not in ".?!…\"'":
                note=" _(devam için 'devam et' yaz)_" if lang=="tr" else " _(type 'continue')_"
                new_hist[-1]["content"]=partial+note
            yield new_hist,""

    demo.load(load_dash, inputs=[lang_state], outputs=[dash_html])
    demo.load(on_detail, inputs=[detail_dd,lang_state], outputs=[detail_tv,detail_ta])

    lang_btn_tr.click(set_tr, outputs=[lang_state, lang_btn_tr, lang_btn_en]).then(load_dash,[lang_state],[dash_html]).then(on_detail,[detail_dd,lang_state],[detail_tv,detail_ta])
    lang_btn_en.click(set_en, outputs=[lang_state, lang_btn_tr, lang_btn_en]).then(load_dash,[lang_state],[dash_html]).then(on_detail,[detail_dd,lang_state],[detail_tv,detail_ta])
    refresh_btn.click(on_refresh,[lang_state],[dash_html])
    detail_dd.change(on_detail,[detail_dd,lang_state],[detail_tv,detail_ta])
    fc_btn.click(on_forecast,[fc_dd],[fc_status,fc_table,fc_chart])
    cmp_btn.click(on_compare,[cmp_a,cmp_b],[cmp_chart])
    chat_in.submit(submit,[chat_in,chatbot,asset_cb,lang_state],[chatbot,chat_in])
    send_btn.click(submit,[chat_in,chatbot,asset_cb,lang_state],[chatbot,chat_in])
    clr_btn.click(lambda:[],outputs=[chatbot])

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 http://localhost:{port}")
    demo.launch(server_name="0.0.0.0", server_port=port, inbrowser=False)
