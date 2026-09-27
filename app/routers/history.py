from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import AnalysisRecord

router = APIRouter(prefix="/api", tags=["history"])


@router.get("/history")
def list_history(
    ticker: str | None = Query(None, description="Belirli bir hisseyi filtrele"),
    limit: int = Query(50, le=500),
    db: Session = Depends(get_db),
):
    """
    Geçmiş analizleri döner. İleride bu kayıtları gerçek fiyat hareketiyle
    karşılaştırıp sistemin isabet oranını ölçebilirsin (bkz. README'deki
    backtest notu) -- bir tahmin sisteminin en kritik parçası budur.
    """
    q = db.query(AnalysisRecord).order_by(AnalysisRecord.created_at.desc())
    if ticker:
        q = q.filter(AnalysisRecord.ticker == ticker.upper())
    rows = q.limit(limit).all()
    return [
        {
            "id": r.id,
            "ticker": r.ticker,
            "source": r.source,
            "created_at": r.created_at,
            "score": r.score,
            "verdict": r.verdict,
            "confidence": r.confidence,
            "last_price": r.last_price,
        }
        for r in rows
    ]
