"""Öğrenci erken uyarı aracı (komut satırı).

Kullanım:
    python tahmin_araci.py --failures 2 --studytime 1 --absences 14 --higher no
    python tahmin_araci.py --interactive

Model, `ogrenci_basari_tahmini.ipynb` çalıştırılınca `models/model.joblib` olarak kaydedilir.
Verilmeyen alanlar eğitim verisinin tipik değeriyle (medyan/mod) doldurulur.
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

MODEL_YOLU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model.joblib")
VERI_YOLU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "student-mat.csv")

# alan -> (tip, geçerli aralık/seçenekler)
ALANLAR = {
    "failures": (int, (0, 3)), "studytime": (int, (1, 4)), "absences": (int, (0, 93)), "age": (int, (15, 22)),
    "Medu": (int, (0, 4)), "Fedu": (int, (0, 4)), "goout": (int, (1, 5)), "freetime": (int, (1, 5)),
    "Dalc": (int, (1, 5)), "Walc": (int, (1, 5)),
    "higher": (str, ("yes", "no")), "sex": (str, ("F", "M")), "schoolsup": (str, ("yes", "no")),
}


def ozellik_ekle(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["alkol_ort"] = (d.Dalc + d.Walc) / 2
    d["ebeveyn_egitim"] = d.Medu + d.Fedu
    d["sosyal_yuk"] = d.goout + d.freetime
    d["devamsiz_log"] = np.log1p(d.absences)
    return d


def dogrula(ogrenci: dict) -> dict:
    """Geçersiz giriş için ValueError fırlatır; tip dönüşümü yapar."""
    temiz = {}
    for k, v in ogrenci.items():
        if v is None:
            continue
        if k not in ALANLAR:
            raise ValueError(f"Bilinmeyen alan: {k}")
        tip, aralik = ALANLAR[k]
        if tip is int:
            v = int(v)
            if not aralik[0] <= v <= aralik[1]:
                raise ValueError(f"{k} {aralik[0]}–{aralik[1]} arasında olmalı (verilen: {v})")
        elif v not in aralik:
            raise ValueError(f"{k} şunlardan biri olmalı: {aralik} (verilen: {v})")
        temiz[k] = v
    return temiz


def yukle(model_yolu: str = MODEL_YOLU, veri_yolu: str = VERI_YOLU):
    if not os.path.exists(model_yolu):
        raise FileNotFoundError("models/model.joblib yok. Önce not defterini çalıştırın.")
    paket = joblib.load(model_yolu)
    df = ozellik_ekle(pd.read_csv(veri_yolu, sep=";"))
    X = df[paket["kolonlar"]]
    varsayilan = {c: (X[c].median() if pd.api.types.is_numeric_dtype(X[c]) else X[c].mode().iloc[0]) for c in paket["kolonlar"]}
    return paket, varsayilan


def risk_hesapla(ogrenci: dict, paket=None, varsayilan=None) -> float:
    """Kalma olasılığını (0–1) döndürür."""
    if paket is None:
        paket, varsayilan = yukle()
    satir = dict(varsayilan)
    satir.update(dogrula(ogrenci))
    gerekli = pd.DataFrame([satir]).assign(G1=0, G2=0)
    gerekli = ozellik_ekle(gerekli)[paket["kolonlar"]]
    return float(paket["model"].predict_proba(gerekli)[0, 1])


def yorumla(risk: float, esik: float = 0.4) -> str:
    seviye = "YÜKSEK risk" if risk >= 0.6 else "ORTA risk" if risk >= esik else "düşük risk"
    oneri = "Rehberlik görüşmesi önerilir." if risk >= esik else "Rutin takip yeterli."
    return f"Kalma olasılığı %{risk * 100:.0f} → {seviye}. {oneri}"


def etkilesimli() -> dict:
    print("Boş bırakılan alan, tipik değerle doldurulur. Çıkmak için Ctrl+C.")
    girdi = {}
    for k, (tip, aralik) in ALANLAR.items():
        ham = input(f"{k} {aralik}: ").strip()
        if ham:
            girdi[k] = ham
    return girdi


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Öğrenci kalma riski tahmini")
    for k, (tip, _) in ALANLAR.items():
        ap.add_argument(f"--{k}")
    ap.add_argument("--interactive", action="store_true", help="Alanları tek tek sor")
    a = ap.parse_args(argv)
    ogrenci = etkilesimli() if a.interactive else {k: getattr(a, k) for k in ALANLAR if getattr(a, k) is not None}
    try:
        print(yorumla(risk_hesapla(ogrenci)))
    except (ValueError, FileNotFoundError) as e:
        print("Hata:", e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
