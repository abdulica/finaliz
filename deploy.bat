@echo off
echo Degisiklikler commit ediliyor...
git add gradio_app.py requirements.txt README.md config.py
git add data/cache.py data/fetcher.py data/windowing.py data/__init__.py
git add analysis/technical.py analysis/forecast.py analysis/llm_engine.py
git add analysis/seasonal.py analysis/macro.py analysis/qa_engine.py analysis/__init__.py
git add components/tradingview.py components/__init__.py
git add utils/i18n.py utils/helpers.py utils/__init__.py

set /p msg="Commit mesaji: "
git commit -m "%msg%"

echo GitHub a push ediliyor...
git push origin master

echo.
echo Tamamlandi! Render otomatik deploy basladi.
echo https://finaliz.onrender.com
pause
