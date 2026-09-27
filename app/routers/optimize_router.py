from fastapi import APIRouter, HTTPException, Query, Depends

from app.data_fetcher import fetch_ohlcv, DataFetchError
from app.optimize import precompute_raw, split_train_test, evaluate_weights, random_search
from app.scoring import BASE_WEIGHTS
from app.config import MARKETS, DEFAULT_MARKET
from app.security import validate_ticker, RateLimiter

router = APIRouter(prefix="/api", tags=["optimize"])

# Bu endpoint ağır bir işlem (birden fazla hissenin tüm geçmişini tarayıp
# yüzlerce ağırlık kombinasyonu deniyor). Herkese açık bir sitede kötüye
# kullanımı önlemek için diğer endpoint'lerden çok daha sıkı bir limit var.
_limiter = RateLimiter(limit=5, window=3600, name="çoklu hisse optimizasyonu")


@router.get("/optimize")
def optimize(
    tickers: str = Query("", description="Virgülle ayrılmış hisse kodları. Boş bırakılırsa piyasanın varsayılan listesi kullanılır."),
    market: str = Query(DEFAULT_MARKET, description="'bist' veya 'us'"),
    period: str = Query("5y"),
    horizon: int = Query(10, ge=1, le=60),
    n_iter: int = Query(400, ge=50, le=2000),
    train_frac: float = Query(0.7, gt=0.3, lt=0.95),
    _rl=Depends(_limiter),
):
    """
    Birden fazla hissede EĞİTİM setinde en iyi ağırlıkları arar, sonra bu
    ağırlıkları hiç görmediği TEST setinde dener. Asıl cevap TEST sonucudur;
    eğitim sonucu sadece referans içindir ve tek başına hiçbir şey kanıtlamaz.
    """
    market_cfg = MARKETS.get(market, MARKETS[DEFAULT_MARKET])
    symbol_list = [validate_ticker(t).upper() for t in tickers.split(",") if t.strip()] or market_cfg["ornekler"]

    train_frames, test_frames = [], []
    per_stock_info = {}
    any_real = False

    for sym in symbol_list:
        try:
            df, is_synth = fetch_ohlcv(sym, market=market, period=period)
        except DataFetchError as exc:
            per_stock_info[sym] = {"hata": str(exc)}
            continue
        any_real = any_real or not is_synth

        raw = precompute_raw(df, horizon=horizon)
        if raw.empty:
            per_stock_info[sym] = {"hata": "Yeterli veri yok."}
            continue

        train, test = split_train_test(raw, train_frac=train_frac)
        per_stock_info[sym] = {
            "kaynak": "demo_verisi" if is_synth else "gercek_veri",
            "egitim_gun": len(train),
            "test_gun": len(test),
        }
        if len(train) >= 30:
            train_frames.append(train)
        if len(test) >= 15:
            test_frames.append(test)

    if not train_frames or not test_frames:
        raise HTTPException(status_code=422, detail="Eğitim/test için yeterli veri toplanamadı.")

    import pandas as pd
    pooled_train = pd.concat(train_frames, ignore_index=True)
    pooled_test = pd.concat(test_frames, ignore_index=True)

    search = random_search(pooled_train, n_iter=n_iter)
    if search is None:
        raise HTTPException(status_code=422, detail="Eğitim setinde eşiği aşan hiçbir ağırlık kombinasyonu bulunamadı.")

    best_weights = search["best"]["weights"]
    baseline_test = evaluate_weights(pooled_test, BASE_WEIGHTS)
    best_test = evaluate_weights(pooled_test, best_weights)

    train_hit = search["best"]["hit_rate"]
    test_hit = best_test["hit_rate"] or 0

    degradation = round(train_hit - test_hit, 1)
    if test_hit is None or test_hit < 51:
        yorum = ("Test setinde isabet oranı yazı-turaya yakın veya altında. Bulunan ağırlıklar "
                 "EĞİTİM verisine ezberlenmiş (overfit) -- gerçek bir kenar değil. Bu göstergelerle "
                 "bu hisse grubunda güvenilir tahmin YOK.")
    elif degradation > 8:
        yorum = (f"Test seti, eğitim setinden {degradation:.1f} puan daha düşük çıktı -- ciddi bir "
                 "ezberleme işareti var. Sonuca temkinli yaklaş, gerçek işlemde performans muhtemelen "
                 f"eğitim değil test rakamına ({test_hit}%) daha yakın olur.")
    else:
        yorum = (f"Test seti eğitime yakın çıktı (fark {degradation:.1f} puan) -- bu, ezberden çok "
                 "gerçek bir örüntüye işaret ediyor olabilir. Yine de farklı bir hisse grubunda ve "
                 "farklı bir dönemde tekrar doğrulamadan güvenme.")

    return {
        "hisseler": per_stock_info,
        "egitim_toplam_gun": len(pooled_train),
        "test_toplam_gun": len(pooled_test),
        "denenen_kombinasyon": search["n_candidates_tried"],
        "gecerli_kombinasyon": search["n_valid"],
        "egitim_sonucu": {"isabet_orani": train_hit, "ort_getiri": search["best"]["avg_return"], "sinyal_sayisi": search["best"]["n"]},
        "test_sonucu_optimize_agirlik": best_test,
        "test_sonucu_mevcut_agirlik": baseline_test,
        "bulunan_agirliklar": {k: round(v, 2) for k, v in best_weights.items()},
        "mevcut_agirliklar": BASE_WEIGHTS,
        "yorum": yorum,
        "veri_kaynagi": "gercek_veri" if any_real else "demo_verisi",
    }
