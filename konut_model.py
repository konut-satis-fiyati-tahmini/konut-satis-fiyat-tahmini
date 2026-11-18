"""Konut Fiyatlandırıcı - ortak model kodu.

Bu dosya 7. hafta notebook'u (hafta7_nihai_model.ipynb) tarafından kullanılır.
İçindeki adımlar önceki haftalarda kurulan ön işleme hattıdır (veri temizliği,
özellik türetme, kategorik dönüşüm ve ölçekleme); sadece tek bir yerde toplanmıştır.

Neden ayrı bir dosya?
Kaydedilen model (nihai_model.pkl), içinde kullandığı fonksiyon ve sınıfların
adını saklar. Model dosyadan yüklenirken bu fonksiyonlar aynı dosyadan bulunur.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

# "NA" değeri bu sütunlarda "özellik yok" anlamına gelir (havuz yok, garaj yok...)
YOK_ANLAMINDAKI_KATEGORIK = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "MasVnrType",
]
YOK_ANLAMINDAKI_SAYISAL = [
    "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath",
    "MasVnrArea",
]
# 4. haftada çıkarılan sütunlar
CIKARILAN_SUTUNLAR = ["Id", "GarageArea", "GarageYrBlt", "YrSold",
                      "YearBuilt", "YearRemodAdd"]

KALITE = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
SIRALI_KODLAMA = {
    "ExterQual": KALITE, "ExterCond": KALITE, "BsmtQual": KALITE,
    "BsmtCond": KALITE, "HeatingQC": KALITE, "KitchenQual": KALITE,
    "FireplaceQu": KALITE, "GarageQual": KALITE, "GarageCond": KALITE,
    "PoolQC": KALITE,
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
}
SIRALI_SUTUNLAR = list(SIRALI_KODLAMA.keys())


def temizle_ve_turet(df):
    """Ham konut verisini 4. haftadaki gibi temizler ve yeni özellikleri türetir.

    Bu adım veriden hiçbir istatistik öğrenmez (sadece sabit kurallar uygular),
    bu yüzden veri sızıntısına yol açmaz.
    """
    df = df.copy()
    df[YOK_ANLAMINDAKI_KATEGORIK] = df[YOK_ANLAMINDAKI_KATEGORIK].fillna("None")
    df[YOK_ANLAMINDAKI_SAYISAL] = df[YOK_ANLAMINDAKI_SAYISAL].fillna(0)

    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                       + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["MSSubClass"] = df["MSSubClass"].astype(str)

    return df.drop(columns=[c for c in CIKARILAN_SUTUNLAR if c in df.columns])


class NeighborhoodMedianImputer(BaseEstimator, TransformerMixin):
    """LotFrontage (sokak cephesi) eksiklerini mahalle medyanıyla doldurur.

    Medyanlar yalnızca eğitim kümesinden öğrenilir (fit).
    """

    def fit(self, X, y=None):
        self.medians_ = X.groupby("Neighborhood")["LotFrontage"].median()
        self.global_median_ = X["LotFrontage"].median()
        return self

    def transform(self, X):
        filled = X["LotFrontage"].fillna(X["Neighborhood"].map(self.medians_))
        return filled.fillna(self.global_median_).to_frame()

    def get_feature_names_out(self, input_features=None):
        return np.array(["LotFrontage"])


def on_isleyici_olustur(X_hazir):
    """4. haftanın 252 sütunluk ColumnTransformer'ını kurar.

    X_hazir: temizle_ve_turet() sonrasındaki veri (sütun tiplerini belirlemek için).
    """
    nominal = [c for c in X_hazir.select_dtypes(exclude=np.number).columns
               if c not in SIRALI_SUTUNLAR]
    sayisal = [c for c in X_hazir.select_dtypes(include=np.number).columns
               if c != "LotFrontage"]

    sirali_hat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="None")),
        ("encode", OrdinalEncoder(
            categories=[SIRALI_KODLAMA[c] for c in SIRALI_SUTUNLAR],
            handle_unknown="use_encoded_value", unknown_value=-1)),
        ("scale", StandardScaler()),
    ])
    nominal_hat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    sayisal_hat = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cephe_hat = Pipeline([
        ("impute", NeighborhoodMedianImputer()),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("ordinal", sirali_hat, SIRALI_SUTUNLAR),
        ("nominal", nominal_hat, nominal),
        ("numeric", sayisal_hat, sayisal),
        ("lotfrontage", cephe_hat, ["LotFrontage", "Neighborhood"]),
    ])


def hat_olustur(model, X_ham):
    """Ham veriyi alıp tahmin üreten tam hattı kurar:
    temizlik ve özellik türetme -> 252 sütunluk ön işleme -> model.
    """
    return Pipeline([
        ("hazirlik", FunctionTransformer(temizle_ve_turet)),
        ("on_isleme", on_isleyici_olustur(temizle_ve_turet(X_ham))),
        ("model", model),
    ])
