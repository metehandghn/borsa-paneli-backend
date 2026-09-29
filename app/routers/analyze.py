from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.data_fetcher import fetch_ohlcv, DataFetchError
from app.config import DEFAULT_MARKET
from app.security import validate_ticker, RateLimiter

_limiter = RateLimiter(limit=60, window=3600, name="hisse analizi")
from app.scoring import compute_signals, aggregate
from app.schemas import AnalysisResponse, SignalDetail
from app.models import AnalysisRecord

router = APIRouter(prefix="/api", tags=["analyze"])


@router.get("/analyze/{ticker}", response_model=AnalysisResponse)
def analyze_ticker(
    ticker: str,
    market: str = Query(DEFAULT_MARKET, description="'bist' veya 'us'"),
    period: str = Query("6mo", description="1mo, 3mo, 6mo, 1y, 2y ..."),
    db: Session = Depends(get_db),
    _rl=Depends(_limiter),
):
    ticker = validate_ticker(ticker)
    try:
        df, is_synthetic = fetch_ohlcv(ticker, market=market, period=period)
    except DataFetchError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    signals, meta = compute_signals(df)
    score, verdict, confidence = aggregate(signals)

    record = AnalysisRecord(
        ticker=ticker.upper(),
        source="ticker" + ("_demo" if is_synthetic else ""),
        score=score,
        verdict=verdict,
        confidence=confidence,
        signals=signals,
        last_price=float(df["Close"].iloc[-1]),
    )
    db.add(record)
    db.commit()

    response = AnalysisResponse(
        ticker=ticker.upper(),
        source="demo_verisi" if is_synthetic else "gercek_veri",
        score=score,
        verdict=verdict,
        confidence=confidence,
        last_price=float(df["Close"].iloc[-1]),
        signals=[SignalDetail(name=s["name"], value=s["value"], weight=s["weight"],
                               signal="AL" if s["vote"] > 0.15 else ("SAT" if s["vote"] < -0.15 else "NÖTR"),
                               note=s["note"]) for s in signals],
    )
    if is_synthetic:
        response.disclaimer += (
            " UYARI: Hiçbir kaynaktan gerçek veri alınamadığı için DEMO (rastgele "
            "üretilmiş) veri kullanıldı — bu sonuç gerçek fiyatı yansıtmaz. Bunun "
            "iki olası sebebi var: (1) hisse kodunu yanlış yazmış olabilirsin -- "
            "şirket adı değil BORSA KODUNU kullan (örn. Aselsan değil ASELS), "
            "ya da (2) veri kaynakları o an geçici olarak erişilemez olabilir. "
            "Kodu kontrol edip tekrar denemeni öneririz."
        )
    return response
