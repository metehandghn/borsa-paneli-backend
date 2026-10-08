# Borsa Analiz Paneli
<img width="1912" height="1097" alt="image" src="https://github.com/user-attachments/assets/8915e950-5673-44a3-a3f2-9feb9c6041dd" />

**Canlı site:** https://dynamic-meringue-fc4904.netlify.app

BIST ve ABD (NASDAQ/NYSE) hisseleri için dokuz teknik göstergeyi (RSI, MACD, Bollinger, Stochastic, ADX, MA Cross, Fibonacci, OBV, hacim sıçraması) tek ekranda gösteren teknik analiz paneli. Backtest ve eğitim/test ayrımlı optimizasyon ile kendi güvenilirliğini ölçer.

## Önemli bulgu

Bu göstergelerle, iki piyasada ve çok sayıda hissede yaptığım eğitim/test ayrımlı testlerde isabet oranı yazı-turadan (%50) anlamlı şekilde farklı çıkmadı. Bu yüzden panel bir "tahmin aracı" değil, göstergeleri düzenli okunur hale getiren bir araçtır. Yatırım tavsiyesi değildir.

## Teknoloji

- Backend: Python, FastAPI (Railway)
- Frontend: tek dosya HTML/CSS/JS (Netlify)
- Veri: ABD için Twelve Data, BIST için borsapy (TradingView), yedek olarak yfinance
- Backtest, eğitim/test ayrımı, önbellek, hız sınırlama ve girdi doğrulama içerir

## Geliştirici

Metehan Dağhan, https://github.com/metehandghn

---

# Stock Analysis Dashboard (English)

**Live site:** https://dynamic-meringue-fc4904.netlify.app

A technical analysis dashboard for Turkish (BIST) and US (NASDAQ/NYSE) stocks that combines nine indicators into one view and measures its own reliability with backtesting and train/test validation.

**Key finding:** in train/test validated experiments across both markets and many tickers, the hit rate was statistically indistinguishable from a coin flip (about 50%). The dashboard is therefore presented as an indicator reader, not a prediction tool. Not investment advice.

**Stack:** Python and FastAPI backend, single-file HTML/CSS/JS frontend. Data via Twelve Data (US) and borsapy/TradingView (BIST).
