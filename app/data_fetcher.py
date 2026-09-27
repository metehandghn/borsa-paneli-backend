"""
Çoklu piyasa (BIST / ABD) için OHLCV verisi çeker.

VERİ KAYNAĞI STRATEJİSİ:
- ABD (NASDAQ/NYSE): Twelve Data resmi API'si kullanılır. Yahoo Finance
  (yfinance), Railway/Render gibi paylaşımlı bulut sunucularından gelen
  istekleri sıklıkla "bot trafiği" sayıp engelliyor -- bu, denenmiş ve
  doğrulanmış bir sorun (curl_cffi ile tarayıcı taklidi bile çözmedi).
  Twelve Data resmi, API-anahtarlı bir servis olduğu için bu riski taşımıyor.
- BIST: Twelve Data'nın ücretsiz planı BIST'i desteklemiyor (ücretli plan
  gerektiriyor). Bunun yerine, birincil kaynak olarak "borsapy" kütüphanesi
  kullanılır -- bu, Yahoo yerine TradingView WebSocket API'sini kaynak
  aldığı için Yahoo'nun bulut-sunucu engellemesinden BAĞIMSIZDIR. Kişisel/
  eğitim amaçlı kullanım için ücretsizdir (projemizin niteliğiyle uyumlu).
  borsapy başarısız olursa yfinance'e, o da başarısız olursa dürüst bir
  DEMO uyarısıyla sahte veriye düşülür. Üç katmanlı bu yedekleme, tek bir
  kaynağın kırılganlığına bağımlı kalmamak içindir.

ÖNBELLEK: Gerçek trafikte aynı hisseyi kısa sürede tekrar tekrar sorgulamak
hem yavaştır hem de dış API'lerin seni geçici engellemesine yol açabilir.
Sonuçlar CACHE_TTL_SECONDS boyunca bellekte tutulur. Bu basit bir bellek-içi
önbellektir: sunucu yeniden başlayınca sıfırlanır, birden fazla sunucu
süreci arasında paylaşılmaz.
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from app.config import (
    MARKETS, DEFAULT_MARKET, ALLOW_SYNTHETIC_FALLBACK,
    CACHE_TTL_SECONDS, TWELVE_DATA_API_KEY,
)

TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"


class DataFetchError(Exception):
    pass


_CACHE: dict[tuple, tuple[float, pd.DataFrame, bool]] = {}


def normalize_ticker(ticker: str, market: str = DEFAULT_MARKET) -> str:
    suffix = MARKETS.get(market, MARKETS[DEFAULT_MARKET])["suffix"]
    ticker = ticker.strip().upper()
    if suffix and not ticker.endswith(suffix):
        ticker += suffix
    return ticker


def _period_to_outputsize(period: str) -> int:
    """'6mo', '2y', '5y' gibi bir dönemi Twelve Data'nın beklediği bar sayısına çevirir."""
    m = re.match(r"(\d+)(mo|y)", period.strip().lower())
    if not m:
        return 300
    n, unit = int(m.group(1)), m.group(2)
    trading_days = n * 21 if unit == "mo" else n * 252
    return min(max(trading_days + 15, 60), 5000)


def _fetch_from_twelvedata(symbol: str, period: str, interval: str) -> pd.DataFrame:
    if not TWELVE_DATA_API_KEY:
        raise DataFetchError("TWELVE_DATA_API_KEY tanımlı değil (Railway > Variables).")

    td_interval = {"1d": "1day", "1wk": "1week", "1mo": "1month"}.get(interval, "1day")
    params = {
        "symbol": symbol,
        "interval": td_interval,
        "outputsize": _period_to_outputsize(period),
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON",
    }
    resp = requests.get(TWELVE_DATA_URL, params=params, timeout=15)
    data = resp.json()
    if not isinstance(data, dict) or data.get("status") == "error" or "values" not in data:
        msg = data.get("message", "bilinmeyen hata") if isinstance(data, dict) else "geçersiz yanıt"
        raise DataFetchError(f"Twelve Data: {msg}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _fetch_from_borsapy(raw_ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    BIST için birincil kaynak. TradingView WebSocket'ini kullanır -- Yahoo
    Finance'e hiç dokunmaz, bu yüzden Yahoo'nun bulut-sunucu engellemesinden
    etkilenmez. Ticker'da ".IS" eki OLMAMALI (borsapy düz kod bekler).
    """
    import borsapy as bp
    df = bp.Ticker(raw_ticker).history(period=period, interval=interval)
    if df is None or df.empty:
        raise DataFetchError(f"'{raw_ticker}' için borsapy'den veri alınamadı.")
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise DataFetchError(f"borsapy beklenmeyen bir format döndürdü (eksik sütun: {missing}).")
    return df[needed]


def _make_browser_session():
    """yfinance (yedek BIST kaynağı) için tarayıcı taklidi -- garanti değil ama şansı artırır."""
    try:
        from curl_cffi import requests as curl_requests
        return curl_requests.Session(impersonate="chrome")
    except Exception:
        return None


def _fetch_from_yfinance(symbol: str, period: str, interval: str) -> pd.DataFrame:
    session = _make_browser_session()
    kwargs = {"period": period, "interval": interval, "progress": False, "auto_adjust": True}
    if session is not None:
        kwargs["session"] = session
    df = yf.download(symbol, **kwargs)
    if df is None or df.empty:
        raise DataFetchError(f"'{symbol}' için veri bulunamadı.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


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
        if market == "us" and TWELVE_DATA_API_KEY:
            df = _fetch_from_twelvedata(symbol, period, interval)
        elif market == "bist":
            raw_ticker = ticker.strip().upper()
            try:
                df = _fetch_from_borsapy(raw_ticker, period, interval)
            except Exception:
                df = _fetch_from_yfinance(symbol, period, interval)  # yedek kaynak
        else:
            df = _fetch_from_yfinance(symbol, period, interval)

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

