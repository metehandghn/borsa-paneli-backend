"""
Skorlama mantığı: her indikatör -1 (sat) ile +1 (al) arasında bir "oy" verir,
bu oylar AĞIRLIKLANDIRILIP -100..+100 aralığında tek skora indirgenir.

KRİTİK TASARIM KARARI -- REJİM UYARLAMALI AĞIRLIK:
Osilatörler (RSI, Stokastik, Bollinger) ortalamaya dönüş araçlarıdır ve
TRENDLİ piyasada sistematik olarak yanılırlar: güçlü bir yükselişte RSI
haftalarca 70+ kalır, bu bir satış sinyali değil GÜÇ göstergesidir. Sabit
ağırlıkla toplarsan sistem her güçlü trendde yanlış yöne sinyal üretir
(ilk sürümde tam olarak bu hata çıktı, test yakaladı).

Bu yüzden önce ADX ile rejim ölçülür:
  - ADX >= 25  -> TRENDLİ: trend göstergeleri (MA, ADX, MACD) ağır basar,
                  osilatörlerin ağırlığı düşer ve trend yönündeki "aşırılık"
                  ters sinyale çevrilmez, nötrlenir.
  - ADX <  20  -> YATAY: osilatörler ağır basar (doğru oldukları rejim budur).
  - Arası      -> geçiş, ağırlıklar doğrusal harmanlanır.

Bu KESİN bir gelecek tahmini değildir. Göstergeler çelişirse skor sıfıra
yakın kalır ve NÖTR döner; çelişkiyi gizleyip yapay yön uydurmuyoruz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app import indicators as ind

BASE_WEIGHTS = {
    "RSI": 16,
    "MACD": 17,
    "Bollinger": 12,
    "Stochastic": 11,
    "ADX_Trend": 12,
    "MA_Cross": 12,
    "Fibonacci": 7,
    "OBV": 8,
    "Volume_Spike": 5,
}

OSCILLATORS = {"RSI", "Bollinger", "Stochastic"}
TREND_FOLLOWERS = {"MACD", "ADX_Trend", "MA_Cross", "OBV"}

# YATAY REJİM STRATEJİSİ -- KALİBRE EDİLMESİ GEREKEN PARAMETRE
# -----------------------------------------------------------------
# "Düşük ADX'te ortalamaya dönüş çalışır" yaygın bir TA varsayımıdır ama
# EVRENSEL DEĞİLDİR. Backtest'te momentumlu bir seride bu varsayım isabeti
# %35'e düşürdü (rastgeleden kötü). Bazı hisse/dönemlerde düşük ADX'te bile
# momentum devam eder ve dip almak zarar ettirir.
#
# Bu yüzden varsayım sabit kodlanmadı: /api/calibrate endpoint'i gerçek BIST
# verisiyle iki modu da test edip hangisinin çalıştığını SANA söyler.
#   "mean_reversion" -> aşırı satımda AL  (klasik varsayım)
#   "momentum"       -> düşüşte SAT       (trend devam eder varsayımı)
RANGE_STRATEGY = "mean_reversion"


def set_range_strategy(mode: str) -> None:
    """Kalibrasyon sonucuna göre yatay rejim stratejisini değiştirir."""
    global RANGE_STRATEGY
    if mode not in ("mean_reversion", "momentum"):
        raise ValueError("mode 'mean_reversion' veya 'momentum' olmalı")
    RANGE_STRATEGY = mode


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def detect_regime(df: pd.DataFrame) -> tuple[str, float, float, float]:
    """Returns (regime_name, trendiness 0..1, slope %/gün, adx)"""
    adx_series = ind.adx(df)
    adx_raw = float(adx_series.iloc[-1])
    # Warm-up maskesi nedeniyle kısa veride ADX NaN olabilir; o durumda
    # rejimi "bilinmiyor" kabul edip nötr bir trendiness kullanıyoruz.
    adx_val = 20.0 if np.isnan(adx_raw) else adx_raw
    slope = ind.trend_slope(df["Close"])

    # Wilder/standart TA yorumu: ADX<20 trendsiz, 20-25 zayıf, 25-40 güçlü,
    # 40+ çok güçlü trend. Bu yüzden 18->0, 35->1 doğrusal haritalama
    # kullanıyoruz. (İlk sürümde 20->0, 25->1 kullanmıştım; ADX=25'te anında
    # saturate oluyordu ve zayıf trendler "tam trendli" sayılıyordu.)
    trendiness = _clip((adx_val - 18) / 17, 0.0, 1.0)
    if trendiness >= 0.7:
        regime = "TRENDLİ"
    elif trendiness <= 0.25:
        regime = "YATAY"
    else:
        regime = "GEÇİŞ"
    return regime, trendiness, slope, adx_val


def regime_weights(trendiness: float) -> dict:
    weights = {}
    for name, base in BASE_WEIGHTS.items():
        if name in OSCILLATORS:
            mult = 1.4 - 1.05 * trendiness      # yatayda güçlü, trendde zayıf
        elif name in TREND_FOLLOWERS:
            mult = 0.6 + 0.9 * trendiness       # trendde güçlü, yatayda zayıf
        else:
            mult = 1.0
        weights[name] = round(base * mult, 2)
    return weights


def compute_signals(df: pd.DataFrame) -> tuple[list[dict], dict]:
    close = df["Close"]
    last_close = float(close.iloc[-1])
    regime, trendiness, slope, adx_val = detect_regime(df)
    W = regime_weights(trendiness)
    trend_up = slope > 0
    signals = []

    # --- RSI (rejim duyarlı) ---
    rsi_val = float(ind.rsi(close).iloc[-1])
    if rsi_val >= 70:
        if trendiness > 0.5 and trend_up:
            vote = 0.25 * trendiness
            note = f"RSI {rsi_val:.1f} — yükselen trendde yüksek RSI güç göstergesidir, satış sinyali sayılmadı"
        else:
            vote = -1.0
            note = f"Aşırı alım ({rsi_val:.1f}) — trend zayıf, düzeltme riski"
    elif rsi_val <= 30:
        if trendiness > 0.5 and not trend_up:
            vote = -0.25 * trendiness
            note = f"RSI {rsi_val:.1f} — düşen trendde düşük RSI zayıflık teyididir, alım sinyali sayılmadı"
        else:
            vote = 1.0
            note = f"Aşırı satım ({rsi_val:.1f}) — trend zayıf, tepki alımı ihtimali"
    else:
        vote = _clip((50 - rsi_val) / 20)
        note = f"Nötr bölgede ({rsi_val:.1f})"
    signals.append({"name": "RSI(14)", "value": round(rsi_val, 2), "vote": vote,
                    "weight": W["RSI"], "note": note})

    # --- MACD ---
    macd_line, signal_line, hist = ind.macd(close)
    hist_val = float(hist.iloc[-1])
    hist_prev = float(hist.iloc[-2]) if len(hist) > 1 else 0.0
    macd_val = float(macd_line.iloc[-1])
    scale = abs(last_close) * 0.01 + 1e-9
    if hist_val > 0 and hist_val > hist_prev:
        vote, note = 0.9, "MACD histogramı pozitif ve genişliyor — momentum yukarı"
    elif hist_val < 0 and hist_val < hist_prev:
        vote, note = -0.9, "MACD histogramı negatif ve derinleşiyor — momentum aşağı"
    else:
        # Sabit eğimli trendde histogram sönümlenir ama MACD çizgisi yön taşır
        vote = _clip(_clip(hist_val / scale) * 0.4 + _clip(macd_val / scale) * 0.6)
        yon = "sıfır ekseni üstünde" if macd_val > 0 else "sıfır ekseni altında"
        note = f"Histogram sönümlenmiş, MACD çizgisi {yon}"
    signals.append({"name": "MACD(12,26,9)", "value": round(hist_val, 4), "vote": vote,
                    "weight": W["MACD"], "note": note})

    # --- Bollinger (rejim duyarlı) ---
    upper, mid, lower = ind.bollinger_bands(close)
    u, l = float(upper.iloc[-1]), float(lower.iloc[-1])
    band_width = (u - l) or 1e-9
    position = (last_close - l) / band_width
    if position >= 0.9:
        if trendiness > 0.5 and trend_up:
            vote = 0.2 * trendiness
            note = "Üst bantta yürüyüş — trendli piyasada güç işareti (band walking)"
        else:
            vote, note = -1.0, "Üst banda yapışık, trend zayıf — düzeltme potansiyeli"
    elif position <= 0.1:
        if trendiness > 0.5 and not trend_up:
            vote = -0.2 * trendiness
            note = "Alt bantta yürüyüş — düşen trendde zayıflık teyidi"
        else:
            vote, note = 1.0, "Alt banda yapışık, trend zayıf — tepki potansiyeli"
    else:
        vote = _clip((0.5 - position) * 2)
        note = f"Bantların %{position*100:.0f} noktasında"
    signals.append({"name": "Bollinger(20,2)", "value": round(last_close, 2), "vote": vote,
                    "weight": W["Bollinger"], "note": note})

    # --- Stochastic (rejim duyarlı) ---
    k, d = ind.stochastic(df)
    k_val, d_val = float(k.iloc[-1]), float(d.iloc[-1])
    if k_val > 80:
        if trendiness > 0.5 and trend_up:
            vote = 0.2 * trendiness
            note = f"%K={k_val:.1f} — trendli yükselişte aşırı alım normaldir, satış sayılmadı"
        else:
            vote, note = -0.9, f"Aşırı alım (%K={k_val:.1f}), trend zayıf"
    elif k_val < 20:
        if trendiness > 0.5 and not trend_up:
            vote = -0.2 * trendiness
            note = f"%K={k_val:.1f} — trendli düşüşte aşırı satım normaldir"
        else:
            vote, note = 0.9, f"Aşırı satım (%K={k_val:.1f}), trend zayıf"
    else:
        vote = _clip((50 - k_val) / 30)
        note = f"%K={k_val:.1f}, %D={d_val:.1f}"
    signals.append({"name": "Stochastic(14,3)", "value": round(k_val, 2), "vote": vote,
                    "weight": W["Stochastic"], "note": note})

    # --- ADX + trend yönü ---
    vote = _clip(slope / 1.5) * max(trendiness, 0.15)
    if adx_val < 20:
        note = f"ADX={adx_val:.1f} — belirgin trend yok, yatay seyir"
    else:
        yon = "yukarı" if slope > 0 else "aşağı"
        note = f"ADX={adx_val:.1f} — {yon} yönlü trend hakim (eğim %{slope:.2f}/gün)"
    signals.append({"name": "ADX / Trend", "value": round(adx_val, 2), "vote": vote,
                    "weight": W["ADX_Trend"], "note": note})

    # --- MA Cross (20/50) ---
    sma20, sma50 = ind.sma(close, 20), ind.sma(close, 50)
    if len(close) >= 51:
        s20, s50 = float(sma20.iloc[-1]), float(sma50.iloc[-1])
        s20_prev, s50_prev = float(sma20.iloc[-2]), float(sma50.iloc[-2])
        if s20_prev <= s50_prev and s20 > s50:
            vote, note = 1.0, "SMA20, SMA50'yi yukarı kesti (golden cross)"
        elif s20_prev >= s50_prev and s20 < s50:
            vote, note = -1.0, "SMA20, SMA50'yi aşağı kesti (death cross)"
        else:
            vote = _clip((s20 - s50) / s50 * 15)
            note = "SMA20 > SMA50 — yukarı eğilim" if s20 > s50 else "SMA20 < SMA50 — aşağı eğilim"
        val = round(s20, 2)
    else:
        vote, note, val = 0.0, "Yeterli veri yok (min 50 mum gerekli)", None
    signals.append({"name": "MA Cross (20/50)", "value": val, "vote": vote,
                    "weight": W["MA_Cross"], "note": note})

    # --- Fibonacci konumu ---
    fibs = ind.fibonacci_levels(df)
    levels = sorted(fibs.values())
    lo_lvl, hi_lvl = levels[0], levels[-1]
    pos = (last_close - lo_lvl) / ((hi_lvl - lo_lvl) or 1e-9)
    closest = min(fibs.items(), key=lambda kv: abs(kv[1] - last_close))
    if pos > 0.95:
        vote = -0.3 if trendiness > 0.5 else -0.8
        note = "Fiyat 90 günlük tepe bölgesinde — direnç"
    elif pos < 0.05:
        vote = 0.3 if trendiness > 0.5 else 0.8
        note = "Fiyat 90 günlük dip bölgesinde — destek"
    else:
        vote = _clip((0.5 - pos) * 1.2)
        note = f"En yakın Fibonacci seviyesi %{closest[0]} ({closest[1]:.2f})"
    signals.append({"name": "Fibonacci(90g)", "value": round(last_close, 2), "vote": vote,
                    "weight": W["Fibonacci"], "note": note})

    # --- OBV: hacim fiyat yönüyle uyumlu mu? ---
    # Fiyat ve OBV aynı yönde hareket ediyorsa trend hacimle destekleniyor
    # demektir (teyit). Ters yönde hareket ediyorsa (diverjans), fiyat
    # hareketinin arkasında gerçek katılım olmayabilir -- bu zayıflık işareti.
    has_volume = "Volume" in df.columns and df["Volume"].tail(20).sum() > 0
    if has_volume and len(close) >= 21:
        obv_series = ind.obv(df)
        obv_change = float(obv_series.iloc[-1] - obv_series.iloc[-20])
        price_change = float(close.iloc[-1] - close.iloc[-20])
        vol_scale = float(df["Volume"].tail(20).mean() * 20) or 1.0
        obv_norm = obv_change / vol_scale
        price_dir, obv_dir = np.sign(price_change), np.sign(obv_change)
        if price_dir != 0 and obv_dir == price_dir:
            vote = _clip(obv_norm * 3)
            note = "Hacim fiyat yönüyle uyumlu — trend hacimle destekleniyor"
        elif price_dir != 0 and obv_dir != 0 and obv_dir != price_dir:
            vote = _clip(-price_dir * 0.45)
            note = "Hacim fiyatla ters yönde (diverjans) — trendin arkasında zayıf katılım olabilir"
        else:
            vote, note = 0.0, "Belirgin bir hacim eğilimi yok"
        val = round(obv_change, 0)
    else:
        vote, note, val = 0.0, "Hacim verisi yok veya yetersiz", None
    signals.append({"name": "OBV(20g)", "value": val, "vote": vote,
                    "weight": W["OBV"], "note": note})

    # --- Hacim sıçraması: bugünkü hareketin arkasında güçlü katılım var mı? ---
    if has_volume and len(close) >= 2:
        vr = ind.volume_ratio(df)
        price_chg_pct = (float(close.iloc[-1]) / float(close.iloc[-2]) - 1) * 100
        if vr > 1.8:
            vote = _clip(price_chg_pct / 2)
            yon = "yükseliş" if price_chg_pct > 0 else "düşüş"
            note = f"Hacim ortalamanın {vr:.1f} katı — bugünkü {yon} güçlü katılımla destekleniyor"
        elif vr < 0.5:
            vote, note = 0.0, f"Hacim ortalamanın altında ({vr:.1f}x) — zayıf katılım, diğer sinyaller güvenilmeyebilir"
        else:
            vote, note = 0.0, f"Hacim normal seviyede ({vr:.1f}x ortalama)"
        val = round(vr, 2)
    else:
        vote, note, val = 0.0, "Hacim verisi yok veya yetersiz", None
    signals.append({"name": "Hacim Sıçraması", "value": val, "vote": vote,
                    "weight": W["Volume_Spike"], "note": note})

    # --- YATAY REJİM STRATEJİ UYGULAMASI ---
    # Osilatörler ortalamaya dönüş mantığıyla oy verir. Eğer kalibrasyon bu
    # hisse için momentum modunun doğru olduğunu söylediyse, yatay rejimdeki
    # osilatör oylarının işaretini çeviririz (düşüşte AL değil SAT deriz).
    if RANGE_STRATEGY == "momentum":
        range_weight = 1.0 - trendiness   # yatayda 1, trendli piyasada 0
        for s in signals:
            if s["name"].split("(")[0].strip() in ("RSI", "Bollinger", "Stochastic"):
                s["vote"] = s["vote"] * (1 - 2 * range_weight)
                s["note"] += " [momentum modu: işaret çevrildi]"

    meta = {"regime": regime, "trendiness": round(trendiness, 2),
            "slope_pct_per_day": round(slope, 3), "adx": round(adx_val, 2),
            "range_strategy": RANGE_STRATEGY,
            "fibonacci_levels": fibs}
    return signals, meta


def aggregate(signals: list[dict]) -> tuple[float, str, float]:
    total_weight = sum(s["weight"] for s in signals) or 1e-9
    weighted_sum = sum(s["vote"] * s["weight"] for s in signals)
    score = (weighted_sum / total_weight) * 100

    votes = np.array([s["vote"] for s in signals])
    w = np.array([s["weight"] for s in signals])
    pos = w[votes > 0.1].sum()
    neg = w[votes < -0.1].sum()
    agreement = max(pos, neg) / (pos + neg) if (pos + neg) > 0 else 0.0
    confidence = max(0.0, min(100.0, agreement * 55 + abs(score) * 0.45))

    if score >= 40:
        verdict = "GÜÇLÜ AL"
    elif score >= 12:
        verdict = "AL"
    elif score <= -40:
        verdict = "GÜÇLÜ SAT"
    elif score <= -12:
        verdict = "SAT"
    else:
        verdict = "NÖTR"
    return round(score, 2), verdict, round(confidence, 2)
