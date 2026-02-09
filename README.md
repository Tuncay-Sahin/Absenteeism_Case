# End-to-End Employee Absenteeism Prediction Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Pandas%20|%20Sklearn-orange)
![Model](https://img.shields.io/badge/Model-Logistic%20Regression-brightgreen)
![Release](https://img.shields.io/badge/Release-v1.1-success)

Bu proje, çalışanların demografik ve sağlık verilerini kullanarak **devamsızlık olasılığını** tahmin eden uçtan uca bir veri bilimi uygulamasıdır.  

Çalışma yalnızca model eğitimi değil; ham veriden başlayarak **preprocessing → model training → model export → inference module → entegrasyon → sürümleme (release)** adımlarını kapsayan tam bir ML pipeline örneğidir.

Proje aynı zamanda eğitim ve portfolyo amaçlı, adım adım izlenebilir bir uygulama rehberi olarak tasarlanmıştır.

---

# Current Stable Version

**Latest release:** `v1.1 — Inference pipeline stabilized`

Bu sürümde:

- inference modülü stabilize edildi
- preprocessing akışı daha güvenli hale getirildi
- opsiyonel kolonlar için varsayılan değer desteği eklendi
- entegrasyon notebook finalize edildi
- git yapılandırması ve repo hijyeni düzenlendi

---

# Proje Klasör Yapısı

Absenteeism_Case/
│
├── Data_0/
│ └── Absenteeism_data.csv
│
├── Notebooks_1/
│ ├── EDA
│ ├── preprocessing
│ └── model training notebookları
│
├── Model_Integration/
│ ├── absenteeism_module.py
│ ├── Absenteeism_case_Integration.ipynb
│ ├── model
│ ├── scaler
│ └── Absenteeism_new_data.csv
│
├── requirements.txt
├── .gitignore
└── README.md


---

# Teknik Yaklaşım

## Veri Ön İşleme (Preprocessing)

- 28 farklı devamsızlık nedeni → **4 ana kategoriye** indirgenmiştir
- Tarih sütunundan:
  - Month Value
  - Day of Week
  türetilmiştir
- Eğitim seviyesi binary yapıya dönüştürülmüştür
- Sayısal değişkenler standartlaştırılmıştır
- Dummy değişkenler özel **CustomScaler** ile korunmuştur

---

## Makine Öğrenmesi Modeli

- Algoritma: **Logistic Regression**
- Problem: Binary classification
- Çıktılar:
  - Devamsızlık olasılığı (Probability)
  - Sınıf tahmini (0/1)

### Sigmoid fonksiyonu:

P(Y=1) = 1 / (1 + e^-z)


Model ve scaler tekrar kullanılabilirlik için pickle formatında kaydedilmiştir.

---

## Inference Modülü (OOP)

Notebook kodları üretim kullanımına uygun olacak şekilde:

absenteeism_module.py


modülüne dönüştürülmüştür.

Modül şunları sağlar:

- OOP tabanlı kullanım
- Model + scaler yükleme
- Otomatik preprocessing
- Kolon doğrulama
- Eksik opsiyonel kolonları otomatik tamamlama (örn: Pet → 0)
- Tahmin + olasılık çıktısı

---

## Quick Start — Tahmin Üretme

`Model_Integration` klasörüne gidin:

```python
from absenteeism_module import absenteeism_model

model = absenteeism_model("model", "scaler")

model.load_and_clean_data("Absenteeism_new_data.csv")

results = model.predicted_outputs()

print(results.head())

csv.çıktısı:
results.to_csv("Absenteeism_predictions.csv", index=False)

Opsiyonel — tarihli çıktı:
results.to_csv(
    f"Absenteeism_predictions_{pd.Timestamp.now().date()}.csv",
    index=False
)
```

## Robust Input Handling

Inference pipeline:

kritik kolonlar eksikse hata verir

bazı opsiyonel kolonları otomatik ekler

kolon sırası değişse bile isim bazlı çalışır

index hizasını korur.

## Kurulum

git clone https://github.com/Tuncay-Sahin/Absenteeism_Case.git
cd Absenteeism_Case
pip install -r requirements.txt

## Proje Amacı

Bu çalışma:

uçtan uca ML pipeline pratiği

inference mimarisi kurma

notebook → modül dönüşümü

model entegrasyonu

sürümleme ve repo yönetimi

konularında uygulamalı öğrenme amacı taşır.

## Not

Bu repo bir “black-box model demo” değil; 
izlenebilir, modüler ve sürümlenmiş bir ML inference pipeline örneğidir.


