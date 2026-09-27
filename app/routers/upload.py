from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.database import get_db
from app.chart_vision import extract_price_series_from_image, ChartVisionError
from app.scoring import compute_signals, aggregate
from app.schemas import AnalysisResponse, SignalDetail
from app.models import AnalysisRecord
from app.security import RateLimiter

router = APIRouter(prefix="/api", tags=["upload"])

_limiter = RateLimiter(limit=20, window=3600, name="görsel analizi")
MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB -- büyük dosyalarla sunucuyu yormayı önler


@router.post("/analyze-image", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...), db: Session = Depends(get_db), _rl=Depends(_limiter)):
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg", "image/webp"):
        raise HTTPException(status_code=400, detail="Sadece PNG/JPG/WEBP görsel kabul edilir.")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Görsel çok büyük (max {MAX_UPLOAD_BYTES // (1024*1024)} MB).")

    try:
        df = extract_price_series_from_image(image_bytes)
    except ChartVisionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    signals, meta = compute_signals(df)
    score, verdict, confidence = aggregate(signals)

    record = AnalysisRecord(
        ticker=None,
        source="image",
        score=score,
        verdict=verdict,
        confidence=confidence,
        signals=signals,
        last_price=float(df["Close"].iloc[-1]),
    )
    db.add(record)
    db.commit()

    response = AnalysisResponse(
        ticker=None,
        source="gorsel",
        score=score,
        verdict=verdict,
        confidence=confidence * 0.7,  # görsel çıkarımı doğası gereği daha az güvenilir
        last_price=None,  # piksel-normalize değer gerçek fiyat değil, göstermiyoruz
        signals=[SignalDetail(name=s["name"], value=s["value"], weight=s["weight"],
                               signal="AL" if s["vote"] > 0.15 else ("SAT" if s["vote"] < -0.15 else "NÖTR"),
                               note=s["note"]) for s in signals],
    )
    response.disclaimer += (
        " Bu analiz yüklediğin görseldeki çizginin piksel takibiyle yapıldı; eksen "
        "ölçeği ve gerçek fiyat bilinmiyor. Kesinlik için ticker koduyla analiz etmeni öneririz."
    )
    return response
