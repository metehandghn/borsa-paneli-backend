from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import ALLOWED_ORIGINS, MARKETS, DEFAULT_MARKET
from app.database import Base, engine
from app import models  # noqa: F401  (tabloların kaydolması için import şart)
from app.routers import analyze, upload, history, backtest_router, optimize_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Borsa Analiz Paneli API",
    description=(
        "BIST ve ABD (NASDAQ/NYSE) hisseleri için çoklu teknik gösterge tabanlı "
        "sinyal üretir. Yatırım tavsiyesi değildir."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)
app.include_router(upload.router)
app.include_router(history.router)
app.include_router(backtest_router.router)
app.include_router(optimize_router.router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/markets")
def markets():
    """Frontend'in piyasa seçim düğmesini ve örnek hisse listelerini doldurmak için."""
    return {"default": DEFAULT_MARKET, "markets": MARKETS}
