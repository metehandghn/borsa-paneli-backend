"""
Yüklenen bir grafik EKRAN GÖRÜNTÜSÜNDEN yaklaşık bir fiyat serisi çıkarır.

DÜRÜST SINIR: Bu, gerçek OHLCV verisi kadar güvenilir DEĞİLDİR. Yapılan şey,
görseldeki en belirgin renkli/koyu çizgiyi piksel piksel takip edip bunu bir
zaman serisine çevirmek. Mum grafiklerinde fitilleri ayırt etmez, eksen
etiketlerini okumaz (OCR yok), ölçek bilmez (0-100 aralığına normalize eder).
Bu yüzden çıkan sinyal "yaklaşık trend" niteliğindedir; ciddi kullanım için
1. seçenek olan gerçek veri (ticker) yolunu öner.

Geliştirme fikirleri (bu modülü ileride güçlendirmek için):
- Eksen etiketlerini okumak için Tesseract OCR entegrasyonu
- Mum gövdesi/fitil ayrımı için renk kümeleme (kırmızı/yeşil mum tespiti)
- Bir görüntü-anlama modeline (vision LLM) grafiği yorumlatıp
  bu modülün çıktısıyla çapraz doğrulama
"""
from __future__ import annotations

import io

import cv2
import numpy as np
import pandas as pd


class ChartVisionError(Exception):
    pass


def extract_price_series_from_image(image_bytes: bytes, target_points: int = 120) -> pd.DataFrame:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ChartVisionError("Görsel okunamadı. PNG/JPG formatında bir grafik yükleyin.")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Arkaplanı (genelde açık renk) elemek için Otsu eşikleme + kenar tespiti
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 120)

    # Her sütunda en "yoğun" (çizgiye ait olma ihtimali en yüksek) y-pikselini bul
    col_ys = []
    for x in range(w):
        col = edges[:, x]
        ys = np.nonzero(col)[0]
        if len(ys) == 0:
            col_ys.append(np.nan)
        else:
            # birden fazla kenar varsa (mumlu grafik gibi) ortanca noktayı al
            col_ys.append(float(np.median(ys)))

    series = pd.Series(col_ys).interpolate(limit_direction="both")
    if series.isna().all():
        raise ChartVisionError("Görselde takip edilebilir bir fiyat çizgisi bulunamadı.")

    # Piksel y-koordinatını fiyat gibi davranan bir seriye çevir (y ters eksen)
    inverted = (h - series)

    # Örnekleme: sütun sayısı çok fazlaysa hedef nokta sayısına indir
    idx = np.linspace(0, len(inverted) - 1, target_points).astype(int)
    sampled = inverted.iloc[idx].reset_index(drop=True)

    # 0-100 aralığına normalize et (gerçek fiyat ölçeği bilinmediği için)
    lo, hi = sampled.min(), sampled.max()
    norm = (sampled - lo) / ((hi - lo) or 1) * 100

    dates = pd.date_range(end=pd.Timestamp.today(), periods=len(norm), freq="B")
    df = pd.DataFrame({"Close": norm.values}, index=dates)
    # indikatör fonksiyonları High/Low/Open/Volume da bekliyor; yakın değerler üretelim
    df["Open"] = df["Close"].shift(1).fillna(df["Close"])
    df["High"] = df[["Open", "Close"]].max(axis=1) * 1.002
    df["Low"] = df[["Open", "Close"]].min(axis=1) * 0.998
    df["Volume"] = 0
    return df
