"""
Çoklu piyasa (BIST / ABD) için OHLCV verisi çeker.

Bu proje sandbox ortamında finans API'lerine ağ erişimi kapalı olduğundan
test edilemedi -- kendi sunucunda/bilgisayarında normal internet erişimiyle
çalıştırdığında yfinance gerçek veriyi çekecektir. Ağ çağrısı başarısız
olursa (limit, kesinti vb.) ve ALLOW_SYNTHETIC_FALLBACK açıksa, geliştirme
amaçlı rastgele-yürüyüş tabanlı sahte veri üretilir; bunu üstte açıkça
işaretliyoruz ki gerçek veriyle karıştırılmasın.

ÖNBELLEK: Gerçek trafikte aynı hisseyi kısa sürede tekrar tekrar sorgulamak
hem yavaştır hem de yfinance'in seni geçici olarak engellemesine yol açabilir
(rate limit). Bu yüzden sonuçlar CACHE_TTL_SECONDS boyunca bellekte tutulur.
Bu basit bir bellek-içi önbellektir: sunucu yeniden başlayınca sıfırlanır,
birden fazla sunucu süreci arasında paylaşılmaz. Ciddi trafik için Redis gibi
paylaşılan bir önbelleğe geçmek daha doğru olur.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yfinance as yf

from app.config import MARKETS, DEFAULT_MARKET, ALLOW_SYNTHETIC_FALLBACK, CACHE_TTL_SECONDS


class DataFetchError(Exception):
    pass


_CACHE: dict[tuple, tuple[float, pd.DataFrame, bool]] = {}


def normalize_ticker(ticker: str, market: str = DEFAULT_MARKET) -> str:
    suffix = MARKETS.get(market, MARKETS[DEFAULT_MARKET])["suffix"]
    ticker = ticker.strip().upper()
    if suffix and not ticker.endswith(suffix):
        ticker += suffix
    return ticker


def fetch_ohlcv(
    ticker: str,
    market: str = DEFAULT_MARKET,
    period: str = "6mo",
    interval: str = "1d",
) -> tuple[pd.DataFrame, bool]:
    """Returns (df, is_synthetic). df columns: Open, High, Low, Close, Volume; index: Date"""
    symbol = normalize_ticker(ticker, market)
    cache_key = (market, symbol, period, interval)
    now = time.time()

    cached = _CACHE.get(cache_key)
    if cached and (now - cached[0]) < CACHE_TTL_SECONDS:
        return cached[1], cached[2]

    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            raise DataFetchError(f"'{symbol}' için veri bulunamadı.")

        # Yeni yfinance sürümleri tek hisse için bile çok katmanlı (MultiIndex)
        # sütun döndürebiliyor: ('Close','AAPL') gibi. Bu durumda df["Close"]
        # bir Series değil DataFrame olur ve ileride float() çağrıları patlar.
        # Tek seviyeye indiriyoruz.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()
        if len(df) < 30:
            raise DataFetchError(f"'{symbol}' için yeterli veri geçmişi yok (min 30 mum gerekli).")

        _CACHE[cache_key] = (now, df, False)
        return df, False
    except Exception as exc:
        if not ALLOW_SYNTHETIC_FALLBACK:
            raise DataFetchError(str(exc)) from exc
        synth = _synthetic_ohlcv(symbol)
        _CACHE[cache_key] = (now, synth, True)
        return synth, True


def _synthetic_ohlcv(symbol: str, n: int = 180) -> pd.DataFrame:
    """Sadece demo/geliştirme amaçlı: gerçek veri değildir."""
    # NOT: pandas sürümüne göre date_range(periods=n, freq="B") tam olarak n
    # tarih döndürmeyebilir (bazı sürümlerde iş günü hizalamasından ötürü
    # n-1 gelebiliyor). Bu yüzden önce tarihleri üretip GERÇEK uzunluğu
    # kullanıyoruz -- sabit n'e güvenmek sürümler arası kırılgan.
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")
    actual_n = len(dates)

    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    returns = rng.normal(loc=0.0004, scale=0.018, size=actual_n)
    price = 100 * np.exp(np.cumsum(returns))
    high = price * (1 + np.abs(rng.normal(0, 0.006, actual_n)))
    low = price * (1 - np.abs(rng.normal(0, 0.006, actual_n)))
    open_ = price * (1 + rng.normal(0, 0.003, actual_n))
    volume = rng.integers(500_000, 5_000_000, actual_n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": price, "Volume": volume},
        index=dates,
    )
