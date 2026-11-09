# Öğrenci Başarısı Erken Uyarı Sistemi

Uçtan Uca Yapay Zeka Projesi: Veriden Tahmine. Gerçek bir veri seti (UCI Student Performance, 395 öğrenci) üzerinde "bu öğrenci dersi geçer mi?" sorusunu yanıtlayan, çoklu model karşılaştırmalı bir sınıflandırma çalışması ve komut satırı tahmin aracı.

## İçerik
| Dosya | Açıklama |
|---|---|
| `ogrenci_basari_tahmini.ipynb` | Çalıştırılmış not defteri: problem → veri → EDA → özellik mühendisliği → 6 model, tekrarlı CV → GridSearch → test → iki senaryo → permütasyon önemi, öğrenme eğrisi → eşik analizi → adalet kontrolü → model kaydı |
| `tahmin_araci.py` | Komut satırı aracı (argparse, girdi doğrulama, etkileşimli mod) |
| `tests/` | pytest ile 9 test |
| `Proje_Raporu.docx`, `Sunum.pptx` | 1 sayfalık rapor, 5 slayt |
| `data/student-mat.csv` | Cortez & Silva (2008), CC BY 4.0 |
| `metrics.json`, `figures/` | Not defterinin ürettiği sayısal sonuçlar ve grafikler |

## Kurulum ve kullanım
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace ogrenci_basari_tahmini.ipynb   # models/model.joblib üretir
python tahmin_araci.py --failures 2 --studytime 1 --absences 14 --higher no
python tahmin_araci.py --interactive
pytest
```
Colab'de: `!git clone https://github.com/b4tukk/python-ogrenci-basari` ardından `%cd python-ogrenci-basari`.

## Ana bulgular (dürüst özet)
- **Senaryo A (ara not yok):** Test doğruluğu ≈ %66, "hep geçer" taban çizgisi ≈ %67. Model doğrulukta taban çizgisini aşmıyor; AUC ≈ 0,62. Dönem başı bilgilerle tahmin zor.
- **Senaryo B (G1, G2 ile):** Doğruluk ≈ %87, kalan sınıfı recall ≈ %96, AUC ≈ 0,95. Ama uyarı geç gelir.
- Eşik 0,3'te recall ≈ %81 olur, ancak öğrencilerin ~%73'ü işaretlenir.
- Kalan sınıfı recall'ı kızlarda ≈ %29, erkeklerde ≈ %67 (küçük örnek; adalet riski).
- Test seti 79 öğrenci: sonuçlar oynaktır.
