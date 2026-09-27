from fastapi import APIRouter, HTTPException, Query, Depends

from app.data_fetcher import fetch_ohlcv, DataFetchError
from app.backtest import run_backtest
from app import scoring
from app.config import DEFAULT_MARKET
from app.security import validate_ticker, RateLimiter

_limiter = RateLimiter(limit=20, window=3600, name="backtest/kalibrasyon")

router = APIRouter(prefix="/api", tags=["backtest"])


@router.get("/backtest/{ticker}")
def backtest(
    ticker: str,
    market: str = Query(DEFAULT_MARKET, description="'bist' veya 'us'"),
    period: str = Query("2y", description="Backtest için uzun geçmiş öner: 2y, 5y"),
    horizon: int = Query(10, ge=1, le=60, description="Sinyalden kaç gün sonrasına bakılsın"),
    _rl=Depends(_limiter),
):
    """
    Sistemin bu hissedeki GERÇEK isabet oranını ölçer.
    Siteyi kullanmadan önce mutlaka çalıştır: isabet oranı %52'nin altındaysa
    üretilen sinyallere güvenme.
    """
    ticker = validate_ticker(ticker)
    try:
        df, is_synth = fetch_ohlcv(ticker, market=market, period=period)
    except DataFetchError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    result = run_backtest(df, horizon=horizon)
    result["ticker"] = ticker.upper()
    result["veri_kaynagi"] = "demo_verisi" if is_synth else "gercek_veri"
    if is_synth:
        result["uyari"] = ("DEMO veri kullanıldı, bu sonuçlar gerçek piyasa performansını "
                           "yansıtmaz. Gerçek veri erişimi olan bir ortamda çalıştır.")
    return result


@router.get("/calibrate/{ticker}")
def calibrate(
    ticker: str,
    market: str = Query(DEFAULT_MARKET, description="'bist' veya 'us'"),
    period: str = Query("5y"),
    horizon: int = Query(10, ge=1, le=60),
    _rl=Depends(_limiter),
):
    """
    Yatay rejimde hangi stratejinin (ortalamaya dönüş / momentum) bu hissede
    çalıştığını gerçek veriyle ölçer ve kazananı önerir.

    Bu endpoint'in varlık sebebi: "düşük ADX'te ortalamaya dönüş çalışır"
    yaygın bir TA varsayımıdır ama her hissede doğru değildir. Varsayımı
    kabul etmek yerine ölçüyoruz.
    """
    ticker = validate_ticker(ticker)
    try:
        df, is_synth = fetch_ohlcv(ticker, market=market, period=period)
    except DataFetchError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    original = scoring.RANGE_STRATEGY
    results = {}
    try:
        for mode in ("mean_reversion", "momentum"):
            scoring.set_range_strategy(mode)
            r = run_backtest(df, horizon=horizon)
            results[mode] = {
                "isabet_orani": r.get("isabet_orani"),
                "ort_getiri": r.get("ort_strateji_getirisi"),
                "sinyal_sayisi": r.get("islem_sinyali"),
            }
    finally:
        scoring.set_range_strategy(original)

    valid = {k: v for k, v in results.items() if v["isabet_orani"] is not None}
    if not valid:
        raise HTTPException(status_code=422, detail="Kalibrasyon için yeterli veri yok.")

    winner = max(valid, key=lambda k: valid[k]["isabet_orani"])
    win_rate = valid[winner]["isabet_orani"]

    if win_rate < 52:
        oneri = ("Hiçbir mod %52 isabeti aşamadı. Bu hissede sistem anlamlı tahmin "
                 "gücü göstermiyor — sinyallere göre işlem yapma.")
    else:
        oneri = f"'{winner}' modu daha iyi sonuç verdi (%{win_rate} isabet)."

    return {
        "ticker": ticker.upper(),
        "sonuclar": results,
        "onerilen_mod": winner if win_rate >= 52 else None,
        "oneri": oneri,
        "veri_kaynagi": "demo_verisi" if is_synth else "gercek_veri",
    }
