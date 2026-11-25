"""Konut Fiyatlandırıcı - Lüks Gayrimenkul Değerleme Arayüzü (İP-6).

Çalıştırmak için (bu dosya, konut_model.py, nihai_model.pkl ve train.csv
aynı klasörde olmalı):

    pip install -r requirements.txt
    python arayuz_varsayilanlar_olustur.py   # sadece ilk seferde
    streamlit run app.py
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
    st.markdown(
        f"<div class='m2-pill'>≈ <b>{deger * FT2_M2:,.0f} m²</b></div>",
        unsafe_allow_html=True,
    )
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


def segment_belirle(fiyat):
    """Tahmini fiyata göre piyasa segmenti ve yüzde konumu döndürür."""
    if fiyat < 130000:
        return "Ekonomik Segment", 28, "#60a5fa"
    if fiyat < 200000:
        return "Orta Segment", 54, "#d4af37"
    if fiyat < 300000:
        return "Üst Segment", 78, "#f59e0b"
    return "Lüks Segment", 94, "#fde047"


# ---------------------------------------------------------------- Sayfa Ayarları
st.set_page_config(
    page_title="Konut Fiyatlandırıcı | AI Değerleme",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Lüks Koyu & Altın Tema CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #0a0e17;
        --bg-card: rgba(18, 25, 39, 0.88);
        --bg-elevated: rgba(26, 35, 53, 0.92);
        --gold: #d4af37;
        --gold-light: #f3dc8a;
        --border-subtle: rgba(212, 175, 55, 0.22);
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(212, 175, 55, 0.11), transparent 36%),
            radial-gradient(circle at 88% 14%, rgba(56, 189, 248, 0.08), transparent 32%),
            linear-gradient(180deg, #090d16 0%, #0d1320 55%, #090d16 100%);
        color: var(--text-main);
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }
    #MainMenu, footer {
        visibility: hidden;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.2rem;
        max-width: 1320px;
    }

    /* Hero Banner */
    .luxury-hero {
        background: linear-gradient(135deg, rgba(20, 27, 43, 0.95) 0%, rgba(13, 18, 30, 0.98) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 18px;
        padding: 1.45rem 1.85rem;
        margin-bottom: 1.35rem;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.45);
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1.5rem;
        flex-wrap: wrap;
    }
    .hero-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--gold);
        margin-bottom: 0.35rem;
    }
    .hero-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.05rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        line-height: 1.15;
    }
    .hero-sub {
        color: var(--text-muted);
        font-size: 0.91rem;
        margin-top: 0.45rem;
        max-width: 720px;
        line-height: 1.45;
    }
    .hero-badges {
        display: flex;
        gap: 0.55rem;
        flex-wrap: wrap;
    }
    .hero-badge {
        background: rgba(212, 175, 55, 0.10);
        border: 1px solid rgba(212, 175, 55, 0.30);
        color: var(--gold-light);
        padding: 0.42rem 0.85rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        white-space: nowrap;
    }

    /* Section Headings */
    .section-card-header {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        border-bottom: 1px solid rgba(212, 175, 55, 0.18);
        padding-bottom: 0.45rem;
        margin-top: 0.4rem;
        margin-bottom: 0.85rem;
    }
    .section-card-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.18rem;
        font-weight: 600;
        color: #f8fafc;
        letter-spacing: 0.01em;
    }
    .section-tag {
        margin-left: auto;
        font-size: 0.70rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--gold);
        font-weight: 600;
    }

    /* Input Widget Styling */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--bg-card);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 0.25rem;
    }
    label[data-testid="stWidgetLabel"] p {
        color: #cbd5e1 !important;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: rgba(11, 16, 27, 0.92) !important;
        border-color: rgba(148, 163, 184, 0.22) !important;
        border-radius: 10px !important;
        color: #f8fafc !important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="input"] input {
        color: #f8fafc !important;
    }
    .m2-pill {
        display: inline-block;
        margin-top: -0.3rem;
        margin-bottom: 0.35rem;
        padding: 0.16rem 0.55rem;
        border-radius: 6px;
        background: rgba(212, 175, 55, 0.10);
        border: 1px solid rgba(212, 175, 55, 0.22);
        color: var(--gold-light);
        font-size: 0.74rem;
    }

    /* Right Panel: Valuation Hero Card */
    .valuation-card {
        background: linear-gradient(155deg, rgba(24, 32, 51, 0.96) 0%, rgba(12, 17, 29, 0.98) 100%);
        border: 1px solid rgba(212, 175, 55, 0.45);
        border-radius: 18px;
        padding: 1.55rem 1.65rem;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.06);
        margin-bottom: 0.95rem;
    }
    .val-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--gold);
    }
    .val-price {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.85rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ffffff 0%, #f3dc8a 60%, #d4af37 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.12;
        margin: 0.35rem 0 0.5rem 0;
    }
    .val-range {
        display: inline-block;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 10px;
        padding: 0.45rem 0.78rem;
        font-size: 0.86rem;
        color: #e2e8f0;
    }
    .val-range b {
        color: var(--gold-light);
    }

    /* Mini KPI Grid */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.65rem;
        margin-bottom: 0.95rem;
    }
    .kpi-box {
        background: var(--bg-elevated);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 13px;
        padding: 0.8rem 0.85rem;
    }
    .kpi-label {
        font-size: 0.71rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.06rem;
        font-weight: 700;
        color: #ffffff;
    }
    .kpi-sub {
        font-size: 0.72rem;
        color: var(--gold-light);
        margin-top: 0.12rem;
    }

    /* Segment Bar Card */
    .info-card {
        background: var(--bg-card);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 1.05rem 1.2rem;
        margin-bottom: 0.9rem;
    }
    .segment-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.84rem;
        margin-bottom: 0.5rem;
    }
    .segment-track {
        height: 8px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 999px;
        overflow: hidden;
        margin-bottom: 0.45rem;
    }
    .segment-fill {
        height: 100%;
        border-radius: 999px;
    }
    .model-meta-row {
        display: flex;
        justify-content: space-between;
        padding: 0.34rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        font-size: 0.83rem;
    }
    .model-meta-row:last-child {
        border-bottom: none;
    }
    .meta-k { color: var(--text-muted); }
    .meta-v { color: #f8fafc; font-weight: 600; }

    .disclaimer-box {
        background: rgba(212, 175, 55, 0.07);
        border-left: 3px solid var(--gold);
        border-radius: 0 12px 12px 0;
        padding: 0.78rem 0.95rem;
        font-size: 0.78rem;
        color: #cbd5e1;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

model = model_yukle()
ayar = varsayilanlar_yukle()
tipik = ayar["tipik_ev"]
yuzde = f"{ayar['test_r2'] * 100:.1f}".replace(".", ",")

# Hero Banner
st.markdown(
    f"""
    <div class="luxury-hero">
        <div>
            <div class="hero-eyebrow">AI Gayrimenkul Değerleme Platformu • Ames, Iowa</div>
            <h1 class="hero-title">🏛️ Konut Fiyatlandırıcı</h1>
            <div class="hero-sub">
                2006–2010 yılları arasında satılan <b>1.456 konut</b> verisiyle eğitilmiş makine öğrenmesi modeli
                üzerinden anlık piyasa değerlemesi. Sorulmayan teknik detaylar tipik konut profiliyle otomatik tamamlanır.
            </div>
        </div>
        <div class="hero-badges">
            <span class="hero-badge">Model: {ayar['model_adi']}</span>
            <span class="hero-badge">Başarı: R² %{yuzde}</span>
            <span class="hero-badge">MAE: ±{ayar['test_mae']:,} $</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

sol, sag = st.columns([1.45, 1.0], gap="large")

with sol:
    with st.container(border=True):
        st.markdown(
            """
            <div class="section-card-header">
                <span>📍</span>
                <span class="section-card-title">Konum ve Mimari Kalite</span>
                <span class="section-tag">Bölüm 01</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
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

    with st.container(border=True):
        st.markdown(
            """
            <div class="section-card-header">
                <span>📐</span>
                <span class="section-card-title">Yaşam Alanı ve Metrekare</span>
                <span class="section-tag">Bölüm 02</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        a1, a2, a3 = st.columns(3)
        with a1:
            kat1 = alan_girisi("Zemin kat alanı", tipik["1stFlrSF"], 5000, "kat1")
        with a2:
            kat2 = alan_girisi("Üst kat alanı (yoksa 0)", tipik["2ndFlrSF"], 3000, "kat2")
        with a3:
            bodrum = alan_girisi("Bodrum alanı (yoksa 0)", tipik["TotalBsmtSF"], 4000, "bodrum")

    with st.container(border=True):
        st.markdown(
            """
            <div class="section-card-header">
                <span>🛋️</span>
                <span class="section-card-title">Yaş, Banyo ve Donanım</span>
                <span class="section-tag">Bölüm 03</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
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
    if kat1 == 0:
        st.warning("Zemin kat alanı 0 olamaz.")
    else:
        fiyat = float(model.predict(ev_satiri_olustur(tipik, girdi))[0])
        hata = ayar["test_mae"]
        alt_sinir = max(fiyat - hata, 0)
        ust_sinir = fiyat + hata
        yasam_m2 = (kat1 + kat2) * FT2_M2
        toplam_m2 = (kat1 + kat2 + bodrum) * FT2_M2
        birim_fiyat = fiyat / yasam_m2
        segment_adi, segment_oran, segment_renk = segment_belirle(fiyat)

        st.markdown(
            f"""
            <div class="valuation-card">
                <div class="val-eyebrow">Tahmini Piyasa Değeri</div>
                <div class="val-price">{fiyat:,.0f} $</div>
                <div class="val-range">
                    Olası Değer Aralığı: <b>{alt_sinir:,.0f} $ – {ust_sinir:,.0f} $</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-box">
                    <div class="kpi-label">Birim Fiyat</div>
                    <div class="kpi-value">{birim_fiyat:,.0f} $/m²</div>
                    <div class="kpi-sub">Yaşam alanı</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Toplam Alan</div>
                    <div class="kpi-value">{toplam_m2:,.0f} m²</div>
                    <div class="kpi-sub">{kat1 + kat2 + bodrum:,} sq ft</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Model Güveni</div>
                    <div class="kpi-value">%{yuzde}</div>
                    <div class="kpi-sub">R² = {ayar['test_r2']:.3f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="info-card">
                <div class="segment-head">
                    <span style="color:#94a3b8;">Piyasa Konumlandırması</span>
                    <span style="color:{segment_renk}; font-weight:700;">{segment_adi}</span>
                </div>
                <div class="segment-track">
                    <div class="segment-fill" style="width:{segment_oran}%; background:{segment_renk};"></div>
                </div>
                <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:0.75rem;">
                    Seçili mahalle: <b style="color:#f8fafc;">{MAHALLELER.get(mahalle, mahalle)}</b> •
                    Kalite puanı: <b style="color:#f8fafc;">{kalite}/10</b>
                </div>
                <div class="model-meta-row">
                    <span class="meta-k">Aktif Algoritma</span>
                    <span class="meta-v">{ayar['model_adi']}</span>
                </div>
                <div class="model-meta-row">
                    <span class="meta-k">Test Kümesi Başarısı</span>
                    <span class="meta-v">R² = {ayar['test_r2']:.3f} (292 konut)</span>
                </div>
                <div class="model-meta-row">
                    <span class="meta-k">Ortalama Mutlak Hata (MAE)</span>
                    <span class="meta-v">±{hata:,.0f} $</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="disclaimer-box">
                <b>Bilgilendirme:</b> Bu uygulama bir bilgisayar mühendisliği tasarım projesidir.
                Tahminler 2006–2010 Ames (Iowa, ABD) konut piyasasına dayanır; güncel fiyatları
                veya Türkiye piyasasını yansıtmaz ve resmi ekspertiz yerine geçmez.
            </div>
            """,
            unsafe_allow_html=True,
        )
