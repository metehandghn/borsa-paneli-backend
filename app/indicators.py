"""
Teknik indikatörlerin saf pandas/numpy implementasyonları.
Harici bir TA kütüphanesine bağımlı olmamak için formüller elle yazıldı --
bu hem bağımlılığı azaltır hem de hesaplama mantığını tam kontrol altında tutar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3):
    low_min = df["Low"].rolling(k_window).min()
    high_max = df["High"].rolling(k_window).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_window).mean()
    return k.fillna(50), d.fillna(50)


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[(plus_dm < 0) | (plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm < 0) | (minus_dm < plus_dm)] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out = dx.ewm(alpha=1 / window, adjust=False).mean()

    # WARM-UP MASKESİ: ilk barlarda DI'lardan biri sıfır olabilir, DX=100
    # patlar ve bu değer EWMA'ya sızıp ADX'i kalıcı şişirir. Wilder'ın
    # orijinal yönteminde ADX ancak ~2*window bar sonra anlamlıdır.
    warmup = min(2 * window, len(out))
    out.iloc[:warmup] = np.nan
    return out


def fibonacci_levels(df: pd.DataFrame, lookback: int = 90) -> dict:
    window = df.tail(lookback)
    high = float(window["High"].max())
    low = float(window["Low"].min())
    diff = high - low
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    return {f"{r:.3f}": round(high - diff * r, 4) for r in ratios}


def trend_slope(series: pd.Series, window: int = 20) -> float:
    """Son `window` kapanışa doğrusal regresyon uygulayıp normalize eğim döndürür."""
    recent = series.tail(window).values
    if len(recent) < window:
        return 0.0
    x = np.arange(len(recent))
    slope, _ = np.polyfit(x, recent, 1)
    return float(slope / (recent.mean() or 1) * 100)  # yüzde eğim


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume: kapanış yukarıysa hacim eklenir, aşağıysa çıkarılır."""
    direction = np.sign(df["Close"].diff().fillna(0))
    return (direction * df["Volume"]).cumsum()


def volume_ratio(df: pd.DataFrame, window: int = 20) -> float:
    """Bugünkü hacmin son `window` günlük ortalamaya oranı."""
    avg = df["Volume"].tail(window).mean()
    if not avg:
        return 1.0
    return float(df["Volume"].iloc[-1] / avg)
