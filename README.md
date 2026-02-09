# End-to-End Employee Absenteeism Prediction Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Pandas%20|%20Sklearn-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green)


Bu proje, bir kurumdaki çalışanların demografik özelliklerini ve sağlık verilerini analizerek, **Lojistik Regresyon** modeli yardımıyla gelecekteki devamsızlık olasılıklarını tahmin eden uçtan uca bir veri bilimi çözümüdür. Aynı zamanda ham veriden canlı sisteme (deployment) giden süreci adım adım gösteren **eğitici bir rehber** niteliğindedir.

---

## 📂 Proje Klasör Yapısı

Proje, yönetilebilirliği artırmak ve süreci şeffaf hale getirmek için modüler bir yapıda kurgulanmıştır:

- **`Data_0/`**: 
  - Projenin yakıtı olan ham veri (`Absenteeism_data.csv`) ve işlemden geçmiş ara dosyaları barındırır.
  
- **`Notebooks_1/`**: 
  - Veri keşfi (EDA), görselleştirme ve özellik mühendisliği (feature engineering) adımlarını içerir.
  - Model eğitim süreçlerinin adım adım dokümante edildiği Jupyter Notebook'lar buradadır.

- **`Model_Integration/`**: 
  - Sistemin "canlıya" alınmaya hazır versiyonudur. 
  - `absenteeism_module.py`: Özel olarak yazılmış Python modülü.
  - `model` & `scaler`: Eğitilmiş ve serileştirilmiş (pickled) makine öğrenmesi dosyaları.

---

## ⚙️ Teknik İş Akışı ve Metodoloji

Bu projede "Black Box" (Kara Kutu) modeller yerine, neden-sonuç ilişkisinin net olarak görülebildiği istatistiksel yöntemler tercih edilmiştir.

### 1. Veri Ön İşleme (Preprocessing)
Veri seti, makine öğrenmesi algoritmasına girmeden önce titiz bir temizlik sürecinden geçirilmiştir:
- **Kategorizasyon**: 28 farklı devamsızlık nedeni, istatistiksel anlamlılığı artırmak için **4 ana karakteristik grup** altında toplanmıştır.
- **Feature Engineering**: 'Date' (Tarih) değişkeninden ay, gün ve yıl bilgileri türetilmiş; kategorik değişkenler (Eğitim vb.) ikili (binary) yapıya dönüştürülmüştür.
- **Standardizasyon**: Sayısal değişkenler (Maaş, İş Yükü vb.), modelin katsayılarını (weights) dengeli hesaplayabilmesi için `StandardScaler` ile ölçeklendirilmiştir.

### 2. Makine Öğrenmesi Modeli
- **Algoritma**: Sınıflandırma problemi için endüstri standardı olan **Lojistik Regresyon** kullanılmıştır.
- **Matematiksel Temel**: Her çalışan için devamsızlık olasılığı aşağıdaki Sigmoid fonksiyonu ile hesaplanır:
  
  $$P(Y=1) = \frac{1}{1 + e^{-z}}$$
  
  *(Burada $z$, girdilerin ağırlıklı toplamıdır.)*

- **Performans & Kayıt**: Model %70+ doğruluk oranıyla eğitilmiş; tekrar kullanılabilirlik için `pickle` formatında `model` ve `scaler` dosyaları olarak kaydedilmiştir.

### 3. Modülerizasyon ve Entegrasyon (Deployment)
Notebook üzerindeki kodlar, dış dünyada (web sitesi, uygulama vb.) kullanılabilmesi için **Nesne Yönelimli Programlama (OOP)** prensipleriyle `absenteeism_module.py` dosyasına dönüştürülmüştür. Bu sayede tek bir satır kod ile yeni veriler üzerinde tahmin yapılabilir.

---

## 🚀 Kullanım (Quick Start)

Bu projeyi kendi bilgisayarınızda çalıştırmak veya yeni bir veri seti üzerinde tahmin yapmak için `Model_Integration` klasörünü kullanabilirsiniz:

```python
# Modülü projeye dahil et
from absenteeism_module import absenteeism_model

# Modeli ve Scaler'ı yükle (Dosyaların aynı dizinde olduğundan emin olun)
model = absenteeism_model('model', 'scaler')

# Yeni veriyi sisteme yükle ve temizle
model.load_and_clean_data('Absenteeism_new_data.csv')

# Tahmin sonuçlarını al
predictions = model.predicted_outputs()

# Sonuçları görüntüle
print(predictions.head())



## 📦 Kurulum

Projeyi kendi ortamınızda çalıştırmak için:

```bash
# 1. Repository'i klonlayın
git clone https://github.com/Tuncay-Sahin/Absenteeism_Case.git
cd Absenteeism_Case

# 2. Bağımlılıkları yükleyin
pip install -r requirements.txt
```

---
