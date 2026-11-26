"""Arayüz için gerekli 'arayuz_varsayilanlar.json' dosyasını oluşturur.

Çalıştırmak için (train.csv ve konut_model.py aynı klasörde olmalı):
    python arayuz_varsayilanlar_olustur.py

Dosyada şunlar bulunur:
- "tipik_ev": Formda sorulmayan özellikler bu değerlerle doldurulur
  (sayısal sütunlarda medyan, kategorik sütunlarda en sık görülen değer).
- "mahalleler": Formdaki mahalle listesi.
- Nihai modelin (nihai_model.pkl -> Ridge, alpha=20) adı ve test sonuçları
  (arayüzde bilgi olarak gösterilir). Bu sonuçlar, 7. haftadaki model
  karşılaştırma tablosuyla aynı ayrım ve yöntemle yeniden hesaplanır
  (nihai_model.pkl tüm veriyle eğitildiği için doğrudan değerlendirilemez;
  dürüst bir test skoru için burada aynı ayarlarla sadece eğitim kümesinde
  yeniden eğitilir ve test kümesinde ölçülür).
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from konut_model import YOK_ANLAMINDAKI_KATEGORIK, YOK_ANLAMINDAKI_SAYISAL, hat_olustur

RANDOM_STATE = 42
EN_IYI_ALPHA = 20  # 6. ve 7. haftada çapraz doğrulamayla seçilen değer

df = pd.read_csv("train.csv")
df = df.drop(df[df["GrLivArea"] > 4000].index).reset_index(drop=True)  # 7. haftadaki 4 aykırı ev
X = df.drop(columns="SalePrice")
y = np.log1p(df["SalePrice"])

# Tipik ev: önce "yok" anlamındaki NA'lar doldurulur
# (yoksa örneğin havuz kalitesi yanlışlıkla dolu çıkar).
X_dolu = X.copy()
X_dolu[YOK_ANLAMINDAKI_KATEGORIK] = X_dolu[YOK_ANLAMINDAKI_KATEGORIK].fillna("None")
X_dolu[YOK_ANLAMINDAKI_SAYISAL] = X_dolu[YOK_ANLAMINDAKI_SAYISAL].fillna(0)
tipik_ev = {}
for sutun in X_dolu.columns:
    if pd.api.types.is_numeric_dtype(X_dolu[sutun]):
        tipik_ev[sutun] = float(X_dolu[sutun].median())
    else:
        tipik_ev[sutun] = str(X_dolu[sutun].mode().iloc[0])

# Nihai modelin test sonuçları (7. haftadaki ayrım ve yöntemin aynısı)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)
model = hat_olustur(Ridge(alpha=EN_IYI_ALPHA), X_train).fit(X_train, y_train)
gercek, tahmin = np.expm1(y_test), np.expm1(model.predict(X_test))
mutlak_hata = np.abs(gercek - tahmin)
p90_hata = round(float(np.percentile(mutlak_hata, 90)))

with open("arayuz_varsayilanlar.json", "w", encoding="utf-8") as f:
    json.dump({
        "tipik_ev": tipik_ev,
        "mahalleler": sorted(X["Neighborhood"].unique().tolist()),
        "model_adi": f"Ridge Regresyon (alpha={EN_IYI_ALPHA})",
        "test_mae": round(float(mean_absolute_error(gercek, tahmin))),
        "test_r2": round(float(r2_score(gercek, tahmin)), 4),
        # Not: gerçek fiyat ±test_mae aralığına test evlerinin yalnızca ~%66'sında
        # düşüyor (MAE ortalama hatadır, aralık değildir). %90 kapsama için bu
        # eşik (mutlak hatanın 90. yüzdelik dilimi) kullanılmalıdır.
        "test_p90": p90_hata,
    }, f, ensure_ascii=False, indent=2)

print("Oluşturuldu: arayuz_varsayilanlar.json")
print(f"Test MAE: {mean_absolute_error(gercek, tahmin):,.0f} $  |  Test R²: {r2_score(gercek, tahmin):.4f}  |  "
      f"%90 kapsama için hata payı: ±{p90_hata:,.0f} $")
