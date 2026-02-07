

# -*- coding: utf-8 -*-



# ======================================================================================
# ABSENTEEISM MODULE (DEVAMSIZLIK MODÜLÜ)
# Veri Bilimi Projesi
# Amaç: Eğitilmiş Lojistik Regresyon modelini kullanarak yeni veriler üzerinde tahmin yapmak.
# =======================================================================================

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------------------
# ÖZEL ÖLÇEKLEYİCİ (CUSTOM SCALER) SINIFI
# ---------------------------------------------------------------------------------------
# Açıklama: Standart Scaler normalde tüm sütunları ölçeklendirir.
# Ancak biz dummy (0-1) değişkenlerin bozulmasını istemiyoruz.
# Bu sınıf, sadece bizim seçtiğimiz sayısal sütunları ölçeklendirir.
# ---------------------------------------------------------------------------------------
class CustomScaler(BaseEstimator, TransformerMixin): 
    
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy, with_mean, with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.array(np.mean(X[self.columns]))
        self.var_ = np.array(np.var(X[self.columns]))
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# ---------------------------------------------------------------------------------------
# ANA MODEL SINIFI (ABSENTEEISM MODEL)
# ---------------------------------------------------------------------------------------
# Açıklama: Bu sınıf, eğitilmiş model dosyalarını okur ve ham veriyi işleyip tahmin üretir.
# ---------------------------------------------------------------------------------------
class absenteeism_model():
      
    def __init__(self, model_file, scaler_file):
        # 1. ADIM: Model ve Scaler dosyalarını yükle
        # 'pickle' kütüphanesi ile daha önce kaydettiğimiz eğitimli dosyaları okuyoruz.
        with open('model', 'rb') as model_file, open('scaler', 'rb') as scaler_file:
            self.reg = pickle.load(model_file)     # Lojistik Regresyon Katsayıları
            self.scaler = pickle.load(scaler_file) # Ölçeklendirme İstatistikleri
            self.data = None
    
    # 2. ADIM: Veriyi Yükle ve Temizle (Preprocessing)
    # Burası "önişleme" aşamasında yapılan tüm işlemlerin otomatize edilmiş halidir.
    def load_and_clean_data(self, data_file):
        
        # CSV dosyasını oku
        df = pd.read_csv(data_file, delimiter=',')
        
        # Orijinal veriyi sakla (daha sonra tahminleri yanına eklemek için)
        self.df_with_predictions = df.copy()
        
        # Gereksiz 'ID' sütununu at
        df = df.drop(['ID'], axis = 1)
        
        # Kod uyumluluğu için boş bir hedef değişken sütunu oluştur
        df['Absenteeism Time in Hours'] = 'NaN'

        # NEDENLERİN GRUPLANMASI (REASON GROUPING)
        # "Reason for Absence" sütununu dummy değişkenlere çevir ve 4 ana gruba ayır.
        reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)
        
        # Grup 1: Ciddi Hastalıklar (1-14)
        reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
        # Grup 2: Hamilelik ve Doğum (15-17)
        reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
        # Grup 3: Zehirlenme vb. (18-21)
        reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
        # Grup 4: Hafif Sebepler (22-28)
        reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)
        
        # Orijinal sütunu sil ve grupları ekle
        df = df.drop(['Reason for Absence'], axis = 1)
        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis = 1)
        
        # Sütun isimlerini düzenle (Türkçe karakter kullanmıyoruz)
        column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                       'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                       'Pet', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
        df.columns = column_names

        # Sütun sıralamasını modelin beklediği hale getir
        column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense', 
                                  'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education', 
                                  'Children', 'Pet', 'Absenteeism Time in Hours']
        df = df[column_names_reordered]
      
        # TARİH İŞLEMLERİ (DATE PROCESSING)
        # Tarih formatını düzelt
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

        # Ay değerini (Month Value) çıkar
        list_months = []
        for i in range(df.shape[0]):
            list_months.append(df['Date'][i].month)
        df['Month Value'] = list_months

        # Haftanın gününü (Day of the Week) çıkar
        df['Day of the Week'] = df['Date'].apply(lambda x: x.weekday())

        # Artık işimiz biten 'Date' sütununu at
        df = df.drop(['Date'], axis = 1)

        # Sütunları tekrar sırala
        column_names_upd = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value', 'Day of the Week',
                            'Transportation Expense', 'Distance to Work', 'Age',
                            'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                            'Pet', 'Absenteeism Time in Hours']
        df = df[column_names_upd]

        # EĞİTİM (EDUCATION) 
        # Lise (1) -> 0, Diğerleri (2,3,4) -> 1 olarak haritala
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})

        # Eksik veri varsa (NaN) 0 ile doldur
        df = df.fillna(value=0)

        # GEREKSİZ SÜTUNLARI TEMİZLENİR 
        # Analiz sonucunda etkisiz bulduğumuz sütunları atıyoruz
        df = df.drop(['Absenteeism Time in Hours'], axis=1)
        df = df.drop(['Day of the Week', 'Daily Work Load Average', 'Distance to Work'], axis=1)
        
        # İşlenmiş veriyi sakla
        self.preprocessed_data = df.copy()
        
        #  ÖLÇEKLENDİRME (SCALING) 
        # CustomScaler kullanarak veriyi standartlaştır
        self.data = self.scaler.transform(df)
    
    # 3. ADIM: Olasılık Hesaplama (Probability)
    def predicted_probability(self):
        if (self.data is not None):  
            pred = self.reg.predict_proba(self.data)[:,1]
            return pred
    
    # 4. ADIM: Tahmin Sınıfı (0 veya 1)
    def predicted_output_category(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs
    
    # 5. ADIM: Final Çıktı Tablosunu Oluşturma
    def predicted_outputs(self):
        if (self.data is not None):
            # Olasılık sütununu ekle
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
            # Tahmin (0/1) sütununu ekle
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data