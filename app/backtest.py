"""
BACKTEST -- bu modül projenin en önemli parçasıdır.

Neden: Bir tahmin sisteminin geçmiş fiyatla "uyumlu" görünmesi hiçbir şey
ifade etmez. Tek anlamlı soru şudur: sinyal verildikten SONRA fiyat ne yaptı?
Bu modül geçmişe yürüyerek (walk-forward) her gün sinyal üretir ve o günden
sonraki N günlük getiriyi ölçer.

KRİTİK: Look-ahead bias'tan kaçınmak için t günündeki sinyal SADECE t gününe
kadar olan veriyle hesaplanır (df.iloc[:t+1]). Tüm seriyi indikatöre verip
sonra geçmişe bakmak, sistemi olduğundan çok daha başarılı gösterir --
amatör backtestlerin en sık yaptığı hata budur.

Ölçülen metrikler:
  - hit_rate: sinyal yönü ile gerçekleşen yönün uyuşma oranı
  - avg_forward_return: sinyal başına ortalama ileri getiri
  - edge_vs_random: sistemin "her zaman al-tut" stratejisine kıyasla farkı
Bir sistem hit_rate ~%50 ve edge ~0 veriyorsa, o sistem tahmin ETMİYOR demektir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.scoring import compute_signals, aggregate


def run_backtest(
    df: pd.DataFrame,
    horizon: int = 10,
    min_history: int = 80,
    step: int = 1,
    score_threshold: float = 12.0,
) -> dict:
    """
    df: tam OHLCV geçmişi
    horizon: sinyalden kaç gün sonrasına bakılacak
    min_history: indikatörlerin ısınması için gereken minimum bar
    score_threshold: |skor| bu değerin altındaysa "işlem yok" sayılır
    """
    rows = []
    n = len(df)
    for t in range(min_history, n - horizon, step):
        window = df.iloc[: t + 1]          # <-- look-ahead yok: sadece t'ye kadar
        try:
            signals, meta = compute_signals(window)
            score, verdict, confidence = aggregate(signals)
        except Exception:
            continue

        price_now = float(df["Close"].iloc[t])
        price_future = float(df["Close"].iloc[t + horizon])
        fwd_return = (price_future / price_now - 1) * 100

        rows.append({
            "date": df.index[t],
            "score": score,
            "verdict": verdict,
            "confidence": confidence,
            "regime": meta["regime"],
            "fwd_return": fwd_return,
        })

    if not rows:
        return {"error": "Backtest için yeterli veri yok."}

    bt = pd.DataFrame(rows)
    traded = bt[bt["score"].abs() >= score_threshold].copy()

    if traded.empty:
        return {"error": "Eşiği aşan sinyal üretilmedi.", "total_days": len(bt)}

    traded["direction"] = np.sign(traded["score"])
    traded["correct"] = np.sign(traded["fwd_return"]) == traded["direction"]
    # Sinyal yönünde pozisyon alındığında elde edilen getiri
    traded["strategy_return"] = traded["direction"] * traded["fwd_return"]

    buy_hold = bt["fwd_return"].mean()   # referans: her gün al-tut

    def regime_stats(g):
        return pd.Series({
            "sinyal_sayisi": len(g),
            "isabet_orani": round(g["correct"].mean() * 100, 1),
            "ort_getiri": round(g["strategy_return"].mean(), 2),
        })

    by_regime = traded.groupby("regime").apply(regime_stats, include_groups=False).to_dict("index")
    by_verdict = traded.groupby("verdict").apply(regime_stats, include_groups=False).to_dict("index")

    return {
        "horizon_gun": horizon,
        "toplam_gun": len(bt),
        "islem_sinyali": len(traded),
        "isabet_orani": round(traded["correct"].mean() * 100, 2),
        "ort_strateji_getirisi": round(traded["strategy_return"].mean(), 3),
        "al_tut_referansi": round(buy_hold, 3),
        "edge": round(traded["strategy_return"].mean() - abs(buy_hold), 3),
        "rejime_gore": by_regime,
        "karara_gore": by_verdict,
        "yorum": _interpret(traded["correct"].mean() * 100,
                            traded["strategy_return"].mean()),
    }


def _interpret(hit_rate: float, avg_ret: float) -> str:
    if hit_rate < 52:
        return ("İsabet oranı yazı-tura seviyesinde. Bu ayarlarla sistem anlamlı "
                "bir tahmin gücü göstermiyor — parametreleri veya ağırlıkları "
                "gözden geçir, ya da bu hisse/zaman diliminde kullanma.")
    if hit_rate < 56:
        return ("Zayıf ama sıfırdan farklı bir eğilim var. İşlem maliyeti ve "
                "slipaj düşüldüğünde bu fark eriyebilir.")
    return ("Anlamlı bir eğilim görünüyor. Yine de farklı hisse ve dönemlerde "
            "tekrarlanmadan güvenilmemeli (aşırı uyum riski).")
