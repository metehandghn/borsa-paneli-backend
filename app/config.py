"""
Merkezi konfigürasyon. Ortam değişkenleriyle production'da override edilebilir.
"""
import os

# Veritabanı: varsayılan SQLite, istersen Postgres'e geçebilirsin
# örn: postgresql://kullanici:sifre@localhost:5432/borsa_paneli
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./borsa_paneli.db")

# Desteklenen piyasalar. yfinance BIST hisselerine ".IS" ekiyle erişir,
# ABD hisselerine (NASDAQ/NYSE) hiç ek gerekmez. Yeni bir piyasa eklemek
# için buraya bir satır eklemen yeterli -- kodun geri kalanı otomatik uyar.
MARKETS = {
    "bist": {"suffix": ".IS", "label": "BIST (Türkiye)",
              "ornekler": ["THYAO", "GARAN", "ASELS", "SISE", "EREGL", "SASA", "KCHOL", "AKBNK"]},
    "us":   {"suffix": "",     "label": "ABD (NASDAQ / NYSE)",
              "ornekler": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "JPM", "XOM"]},
}
DEFAULT_MARKET = "us"

# Gerçek veri çekilemezse (ağ kapalıysa / API limitine takılırsa) demo veri
# üretilsin mi? Geliştirme / test ortamında True, production'da False öner.
ALLOW_SYNTHETIC_FALLBACK = os.getenv("ALLOW_SYNTHETIC_FALLBACK", "true").lower() == "true"

# Aynı hisse/dönem için tekrar tekrar Yahoo Finance'e gitmemek için basit
# bir bellek-içi önbellek. Gerçek trafikte hem hızı artırır hem de dış API'nin
# seni geçici olarak engellemesini (rate limit) önler.
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 15 * 60))

# CORS - frontend'ini farklı bir origin'den servis ediyorsan burayı düzenle
ALLOWED_ORIGINS = ["https://dynamic-meringue-fc4904.netlify.app"]
