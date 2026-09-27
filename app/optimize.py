"""
Gösterge ağırlıklarını GERÇEK veriyle optimize eder -- ama bunu yaparken
kendini kandırmamak için EĞİTİM/TEST ayrımı zorunlu kılınmıştır.

NEDEN BU AYRIM ZORUNLU:
Aynı veride hem ağırlıkları arayıp hem de "başarılı" demek, bir sınavın
cevap anahtarını görüp sınavdan yüksek not almak gibidir -- kendini
kandırmaktır. Bu modül:
  1. Her hissenin geçmişini zaman sırasına göre EĞİTİM (ilk %70) ve
     TEST (son %30) olarak ikiye böler.
  2. En iyi ağırlık kombinasyonunu SADECE eğitim verisinde arar.
  3. Bulunan ağırlığı test verisinde (daha önce hiç görmediği veride) dener.
Eğitimde yüksek, testte düşük çıkan bir sonuç = ezberleme (overfit),
gerçek bir kenar değil. İkisi de yakınsa gerçek bir örüntü olabilir.

VERİMLİLİK NOTU: Gösterge oylarını (vote) her ağırlık denemesinde yeniden
hesaplamak çok yavaş olurdu. Bunun yerine her gün için ham oyları VE rejime
göre ağırlık çarpanını bir kere hesaplayıp saklıyoruz (precompute_raw).
Sonra binlerce ağırlık kombinasyonunu bu tablo üzerinde saniyeler içinde
deneyebiliyoruz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.scoring import compute_signals, BASE_WEIGHTS

# compute_signals'ın sinyalleri EKLEME SIRASI (scoring.py içinde sabit).
# BASE_WEIGHTS anahtarlarıyla eşlemek için bu sırayı kullanıyoruz.
SIGNAL_ORDER = ["RSI", "MACD", "Bollinger", "Stochastic", "ADX_Trend", "MA_Cross", "Fibonacci", "OBV", "Volume_Spike"]


def precompute_raw(df: pd.DataFrame, horizon: int = 10, min_history: int = 80) -> pd.DataFrame:
    """Her gün için ham oy + rejim ağırlık çarpanı + ileri getiriyi hesaplar."""
    rows = []
    n = len(df)
    for t in range(min_history, n - horizon):
        window = df.iloc[: t + 1]
        try:
            signals, meta = compute_signals(window)
        except Exception:
            continue
        if len(signals) != len(SIGNAL_ORDER):
            continue

        price_now = float(df["Close"].iloc[t])
        price_future = float(df["Close"].iloc[t + horizon])
        row = {"fwd_return": (price_future / price_now - 1) * 100, "date": df.index[t]}
        for key, s in zip(SIGNAL_ORDER, signals):
            base = BASE_WEIGHTS[key]
            row[f"vote_{key}"] = s["vote"]
            row[f"mult_{key}"] = (s["weight"] / base) if base else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def split_train_test(raw: pd.DataFrame, train_frac: float = 0.7, purge: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zaman sırasına göre böler; iki set arasında sızıntıyı önlemek için küçük bir boşluk (purge) bırakır."""
    n = len(raw)
    split = int(n * train_frac)
    train = raw.iloc[: split - purge] if split - purge > 0 else raw.iloc[:0]
    test = raw.iloc[split:]
    return train, test


def evaluate_weights(raw: pd.DataFrame, weights: dict, threshold: float = 12.0) -> dict:
    if raw.empty:
        return {"n": 0, "hit_rate": None, "avg_return": None}
    keys = SIGNAL_ORDER
    W = np.array([weights[k] for k in keys])
    votes = raw[[f"vote_{k}" for k in keys]].to_numpy()
    mults = raw[[f"mult_{k}" for k in keys]].to_numpy()
    eff_w = mults * W
    denom = eff_w.sum(axis=1)
    denom = np.where(denom == 0, 1e-9, denom)
    score = (votes * eff_w).sum(axis=1) / denom * 100

    mask = np.abs(score) >= threshold
    if mask.sum() == 0:
        return {"n": 0, "hit_rate": None, "avg_return": None}

    fwd = raw["fwd_return"].to_numpy()
    direction = np.sign(score[mask])
    correct = np.sign(fwd[mask]) == direction
    strat_ret = direction * fwd[mask]
    return {
        "n": int(mask.sum()),
        "hit_rate": round(float(correct.mean() * 100), 2),
        "avg_return": round(float(strat_ret.mean()), 3),
    }


def random_search(
    raw_train: pd.DataFrame,
    n_iter: int = 400,
    min_signals: int = 80,
    threshold: float = 12.0,
    seed: int = 42,
) -> dict | None:
    """
    Ağırlık uzayında rastgele arama yapar, EĞİTİM setinde en yüksek isabet
    oranını veren kombinasyonu döndürür. min_signals altındaki adaylar
    elenir -- yoksa 5 sinyalle %100 isabet gibi anlamsız sonuçlar "kazanır".
    """
    rng = np.random.default_rng(seed)
    keys = SIGNAL_ORDER
    candidates = [dict(BASE_WEIGHTS)]  # aday 0: mevcut (temel) ağırlıklar
    for _ in range(n_iter):
        candidates.append({k: float(rng.uniform(0.15, 3.0)) * BASE_WEIGHTS[k] for k in keys})

    results = []
    for w in candidates:
        r = evaluate_weights(raw_train, w, threshold)
        if r["hit_rate"] is None or r["n"] < min_signals:
            continue
        r["weights"] = w
        results.append(r)

    if not results:
        return None
    baseline = next((r for r in results if r["weights"] == BASE_WEIGHTS), results[0])
    best = max(results, key=lambda r: r["hit_rate"])
    return {"best": best, "baseline": baseline, "n_candidates_tried": len(candidates), "n_valid": len(results)}
