"""
Herkese açık bir sitede kötüye kullanıma karşı paylaşılan koruma katmanı.

Bu, "hacklenmeye karşı" gibi genel bir korkuya karşı somut bir cevaptır:
sitede şifre/hesap olmadığı için asıl risk kimlik hırsızlığı değil, birinin
sunucuyu bilerek (ya da kazayla, bir script döngüsüyle) aşırı yüklemesidir.
Burada iki şey var:
  1. validate_ticker: saçma/aşırı uzun girdileri backend'e hiç sokmadan eler.
  2. RateLimiter: IP başına, belirli bir pencerede kaç istek yapılabileceğini
     sınırlar. Bellek-içi basit bir sayaçtır -- sunucu yeniden başlayınca
     sıfırlanır, birden fazla sunucu süreci arasında paylaşılmaz. Ciddi
     trafik/çoklu sunucu için Redis gibi paylaşılan bir çözüm gerekir.
"""
from __future__ import annotations

import re
import time

from fastapi import HTTPException, Request

TICKER_RE = re.compile(r"^[A-Za-z0-9.\-]{1,12}$")


def validate_ticker(ticker: str) -> str:
    """Hisse kodunu doğrular; geçersizse 400 döner. Geçerliyse aynen döndürür."""
    candidate = ticker.strip()
    if not TICKER_RE.match(candidate):
        raise HTTPException(
            status_code=400,
            detail="Geçersiz hisse kodu. Sadece harf, rakam, nokta ve tire içerebilir (en fazla 12 karakter).",
        )
    return candidate


class RateLimiter:
    """IP başına, `window` saniyelik pencerede en fazla `limit` istek."""

    def __init__(self, limit: int, window: int = 3600, name: str = "genel"):
        self.limit = limit
        self.window = window
        self.name = name
        self._state: dict[str, list[float]] = {}

    def __call__(self, request: Request):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        history = self._state.setdefault(ip, [])
        history[:] = [t for t in history if now - t < self.window]
        if len(history) >= self.limit:
            raise HTTPException(
                status_code=429,
                detail=(f"Bu işlem sunucu kaynaklarını korumak için saatte en fazla "
                         f"{self.limit} kez çalıştırılabilir ({self.name}). Lütfen sonra tekrar deneyin."),
            )
        history.append(now)
