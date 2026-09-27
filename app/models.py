from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func

from app.database import Base


class AnalysisRecord(Base):
    """
    Her analiz çağrısının sonucunu saklar: ne zaman, hangi hisse, hangi
    kaynaktan (ticker/görsel), hangi skor ve hangi sinyaller ile.
    Bu tablo zamanla kendi backtest / doğruluk istatistiklerini
    çıkarmana da imkan tanır (üretilen sinyal gerçekten tuttu mu?).
    """
    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=True)  # görselden gelen analizde boş olabilir
    source = Column(String, nullable=False)  # "ticker" | "image"
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    score = Column(Float, nullable=False)          # -100 (çok satış) ... +100 (çok alış)
    verdict = Column(String, nullable=False)        # "GÜÇLÜ AL" | "AL" | "NÖTR" | "SAT" | "GÜÇLÜ SAT"
    confidence = Column(Float, nullable=False)       # 0-100

    signals = Column(JSON, nullable=False)           # her indikatörün ayrı sinyali
    last_price = Column(Float, nullable=True)

    class Config:
        orm_mode = True
