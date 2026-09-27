from typing import Optional
from pydantic import BaseModel


class SignalDetail(BaseModel):
    name: str          # indikatör adı, örn "RSI(14)"
    value: Optional[float] = None
    signal: str         # "AL" | "SAT" | "NÖTR"
    weight: float        # skora katkı ağırlığı
    note: str             # kısa açıklama, örn "Aşırı satım bölgesinde (28.4)"


class AnalysisResponse(BaseModel):
    ticker: Optional[str] = None
    source: str
    score: float
    verdict: str
    confidence: float
    last_price: Optional[float] = None
    signals: list[SignalDetail]
    disclaimer: str = (
        "Bu analiz otomatik teknik göstergelere dayanır, yatırım tavsiyesi "
        "değildir. Geçmiş fiyat hareketleri gelecekteki performansın garantisi değildir."
    )
