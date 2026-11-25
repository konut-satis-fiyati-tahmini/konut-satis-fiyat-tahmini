"""Konut Fiyatlandırıcı - Streamlit arayüzü (İP-6).

Çalıştırmak için (bu dosya, konut_model.py, nihai_model.pkl ve train.csv
aynı klasörde olmalı):

    pip install -r requirements.txt
    python arayuz_varsayilanlar_olustur.py   # sadece ilk seferde
    streamlit run app.py

Formda fiyatı en çok etkileyen özellikler sorulur. Sorulmayan diğer
özellikler veri setindeki "tipik ev"in değerleriyle doldurulur.
"""

import json

import joblib
import pandas as pd
import streamlit as st

import konut_model  # noqa: F401  (kaydedilen model bu dosyadaki fonksiyonları kullanır)

FT2_M2 = 0.092903  # 1 square foot = 0.0929 metrekare

MAHALLELER = {
    "Blmngtn": "Bloomington Heights", "Blueste": "Bluestem", "BrDale": "Briardale",
    "BrkSide": "Brookside", "ClearCr": "Clear Creek", "CollgCr": "College Creek",
    "Crawfor": "Crawford", "Edwards": "Edwards", "Gilbert": "Gilbert",
    "IDOTRR": "Iowa DOT and Rail Road", "MeadowV": "Meadow Village", "Mitchel": "Mitchell",
    "NAmes": "North Ames", "NoRidge": "Northridge", "NPkVill": "Northpark Villa",
    "NridgHt": "Northridge Heights", "NWAmes": "Northwest Ames", "OldTown": "Old Town",
    "SWISU": "South & West of Iowa State University", "Sawyer": "Sawyer",
    "SawyerW": "Sawyer West", "Somerst": "Somerset", "StoneBr": "Stone Brook",
    "Timber": "Timberland", "Veenker": "Veenker",
}
KALITE_SECENEKLERI = {"Ex": "Mükemmel", "Gd": "İyi", "TA": "Ortalama", "Fa": "Vasat", "Po": "Kötü"}


@st.cache_resource
def model_yukle():
    try:
        return joblib.load("nihai_model.pkl")
    except Exception as hata:  # çoğunlukla scikit-learn sürüm farkı
        st.error(
            "Model dosyası açılamadı. Büyük olasılıkla model başka bir scikit-learn sürümüyle kaydedildi. "
            "Çözüm: hafta7_nihai_model.ipynb dosyasını bu bilgisayarda baştan sona çalıştırın; "
            "nihai_model.pkl yeniden oluşur. Ardından bu sayfayı yenileyin.\n\n"
            f"Teknik ayrıntı: {hata}")
        st.stop()


@st.cache_data
def varsayilanlar_yukle():
    with open("arayuz_varsayilanlar.json", encoding="utf-8") as f:
        return json.load(f)


def alan_girisi(etiket, varsayilan, ust_sinir, anahtar):
    """Square foot cinsinden alan girişi; altında metrekare karşılığını gösterir."""
    deger = st.number_input(f"{etiket} (sq ft)", min_value=0, max_value=ust_sinir,
                            value=int(varsayilan), step=10, key=anahtar)
    st.caption(f"≈ {deger * FT2_M2:,.0f} m²")
    return deger


def ev_satiri_olustur(tipik, girdi):
    """Tipik evi alır, formdaki değerleri üzerine yazar ve birbirine bağlı
    sütunları tutarlı hale getirir (örneğin garaj yoksa garaj kalitesi de 'yok')."""
    ev = dict(tipik)
    ev.update({
        "Neighborhood": girdi["mahalle"],
        "OverallQual": girdi["kalite"],
        "OverallCond": girdi["durum"],
        "ExterQual": girdi["dis_kalite"],
        "KitchenQual": girdi["mutfak"],
        "YearBuilt": girdi["yapim"],
        "YearRemodAdd": max(girdi["yenileme"], girdi["yapim"]),
        "YrSold": girdi["satis_yili"],
        "1stFlrSF": girdi["kat1"],
        "2ndFlrSF": girdi["kat2"],
        "GrLivArea": girdi["kat1"] + girdi["kat2"] + ev.get("LowQualFinSF", 0),
        "FullBath": girdi["tam_banyo"],
        "HalfBath": girdi["yarim_banyo"],
        "GarageCars": girdi["garaj"],
        "Fireplaces": girdi["somine"],
    })

    # Bodrum: toplam alan değişince bitmiş/bitmemiş kısımlar aynı oranla ölçeklenir
    bodrum = girdi["bodrum"]
    if bodrum == 0:
        ev.update({"TotalBsmtSF": 0, "BsmtFinSF1": 0, "BsmtFinSF2": 0, "BsmtUnfSF": 0,
                   "BsmtFullBath": 0, "BsmtHalfBath": 0})
        for s in ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]:
            ev[s] = "None"
    else:
        oran = bodrum / tipik["TotalBsmtSF"]
        ev["TotalBsmtSF"] = bodrum
        for s in ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF"]:
            ev[s] = tipik[s] * oran

    # Garaj yoksa garaj özellikleri "yok"
    if girdi["garaj"] == 0:
        ev["GarageArea"] = 0
        for s in ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]:
            ev[s] = "None"

    # Şömine varsa kalitesi "yok" kalamaz
    if girdi["somine"] == 0:
        ev["FireplaceQu"] = "None"
    elif ev["FireplaceQu"] == "None":
        ev["FireplaceQu"] = "TA"

    return pd.DataFrame([ev])


# ---------------------------------------------------------------- Sayfa
st.set_page_config(page_title="Konut Fiyatlandırıcı", page_icon="🏠", layout="wide")
model = model_yukle()
ayar = varsayilanlar_yukle()
tipik = ayar["tipik_ev"]

st.title("🏠 Konut Fiyatlandırıcı")
st.write(
    "Evin özelliklerini girin, makine öğrenmesi modeli tahmini satış fiyatını hesaplasın. "
    "Model, Ames (Iowa, ABD) şehrinde 2006–2010 yılları arasında satılan 1.456 evden öğrenmiştir. "
    "Formda sorulmayan özellikler veri setindeki tipik bir evin değerleriyle doldurulur."
)

sol, sag = st.columns([3, 2], gap="large")

with sol:
    st.subheader("Konum ve kalite")
    k1, k2 = st.columns(2)
    mahalle_kodlari = ayar["mahalleler"]
    mahalle = k1.selectbox(
        "Mahalle", mahalle_kodlari, index=mahalle_kodlari.index(tipik["Neighborhood"]),
        format_func=lambda kod: MAHALLELER.get(kod, kod))
    kalite = k2.slider("Genel kalite (1 = çok kötü, 10 = mükemmel)", 1, 10, int(tipik["OverallQual"]))
    durum = k1.slider("Genel durum (1 = çok kötü, 10 = mükemmel)", 1, 10, int(tipik["OverallCond"]))
    kalite_kodlari = list(KALITE_SECENEKLERI)
    dis_kalite = k2.selectbox("Dış cephe malzeme kalitesi", kalite_kodlari,
                              index=kalite_kodlari.index(tipik["ExterQual"]),
                              format_func=KALITE_SECENEKLERI.get)
    mutfak = k1.selectbox("Mutfak kalitesi", kalite_kodlari,
                          index=kalite_kodlari.index(tipik["KitchenQual"]),
                          format_func=KALITE_SECENEKLERI.get)

    st.subheader("Alan")
    a1, a2, a3 = st.columns(3)
    with a1:
        kat1 = alan_girisi("Zemin kat alanı", tipik["1stFlrSF"], 5000, "kat1")
    with a2:
        kat2 = alan_girisi("Üst kat alanı (yoksa 0)", tipik["2ndFlrSF"], 3000, "kat2")
    with a3:
        bodrum = alan_girisi("Bodrum alanı (yoksa 0)", tipik["TotalBsmtSF"], 4000, "bodrum")

    st.subheader("Yaş ve donanım")
    y1, y2, y3 = st.columns(3)
    yapim = y1.number_input("Yapım yılı", 1872, 2010, int(tipik["YearBuilt"]))
    yenileme = y2.number_input("Son yenileme yılı", 1872, 2010, int(tipik["YearRemodAdd"]),
                               help="Ev hiç yenilenmediyse yapım yılını yazın.")
    satis_yili = y3.selectbox("Satış yılı", [2006, 2007, 2008, 2009, 2010], index=4,
                              help="Model yalnızca 2006–2010 satışlarını gördüğü için bu aralıkla sınırlıdır.")
    tam_banyo = y1.number_input("Tam banyo sayısı", 0, 4, int(tipik["FullBath"]))
    yarim_banyo = y2.number_input("Yarım banyo sayısı", 0, 3, int(tipik["HalfBath"]),
                                  help="Duşu veya küveti olmayan, yalnızca lavabo ve tuvaletten oluşan banyo.")
    garaj = y3.number_input("Garaj kapasitesi (araç, yoksa 0)", 0, 4, int(tipik["GarageCars"]))
    somine = y1.number_input("Şömine sayısı", 0, 3, int(tipik["Fireplaces"]))

girdi = dict(mahalle=mahalle, kalite=kalite, durum=durum, dis_kalite=dis_kalite, mutfak=mutfak,
             kat1=kat1, kat2=kat2, bodrum=bodrum, yapim=yapim, yenileme=yenileme,
             satis_yili=satis_yili, tam_banyo=tam_banyo, yarim_banyo=yarim_banyo,
             garaj=garaj, somine=somine)

with sag:
    st.subheader("Tahmin")
    if kat1 == 0:
        st.warning("Zemin kat alanı 0 olamaz.")
    else:
        fiyat = float(model.predict(ev_satiri_olustur(tipik, girdi))[0])
        hata = ayar["test_mae"]
        st.metric("Tahmini satış fiyatı", f"{fiyat:,.0f} $")
        st.write(f"Olası aralık: **{max(fiyat - hata, 0):,.0f} $ – {fiyat + hata:,.0f} $**")
        yasam_m2 = (kat1 + kat2) * FT2_M2
        st.caption(f"Yaşam alanının metrekare fiyatı: yaklaşık {fiyat / yasam_m2:,.0f} $/m²")

        st.divider()
        yuzde = f"{ayar['test_r2'] * 100:.1f}".replace(".", ",")
        st.markdown(
            f"**Model:** {ayar['model_adi']}  \n"
            f"**Test başarısı:** R² = {ayar['test_r2']:.3f} — model fiyatlardaki değişkenliğin "
            f"yaklaşık %{yuzde}'ini açıklıyor.  \n"
            f"**Ortalama hata:** yaklaşık {hata:,.0f} $ (görülmemiş 292 test evinde)."
        )
        st.info(
            "Bu bir ders projesidir. Tahminler 2006–2010 Ames, Iowa piyasasına dayanır; "
            "bugünkü fiyatları veya Türkiye piyasasını yansıtmaz ve resmi ekspertiz yerine geçmez."
        )
