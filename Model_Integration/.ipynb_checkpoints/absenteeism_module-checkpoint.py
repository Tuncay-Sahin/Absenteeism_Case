# -*- coding: utf-8 -*-

"""
Absenteeism inference module.

Amaç:
- Pickle edilmiş Logistic Regression model + scaler ile yeni veride preprocessing yapıp tahmin üretmek.

Notlar:
- Yeni veri bazen 'Pet' kolonunu içermeyebilir. Bu durumda Pet=0 olarak tamamlanır.
- Kolon isimleri zorla df.columns = [...] ile set edilmez; isim bazlı ve güvenli dönüşüm yapılır.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Sabitler
# -----------------------------
REASON_GROUPS = {
    "Reason_1": (1, 14),
    "Reason_2": (15, 17),
    "Reason_3": (18, 21),
    # Reason_4: 22 ve üzeri (veri setinde genelde 28'e kadar)
    "Reason_4": (22, None),
}

# Yeni veride beklenen minimum kolonlar
REQUIRED_COLUMNS = {
    "ID",
    "Date",
    "Reason for Absence",
    "Transportation Expense",
    "Distance to Work",
    "Age",
    "Daily Work Load Average",
    "Body Mass Index",
    "Education",
    "Children",
    "Pet",  # -> yeni veride yoksa OPTIONAL_DEFAULTS ile tamamlanır
}

# Eksik gelirse otomatik tamamlanacak kolonlar
OPTIONAL_DEFAULTS = {
    "Pet": 0,
}


# -----------------------------
# CustomScaler
# -----------------------------
class CustomScaler(BaseEstimator, TransformerMixin):
    """
    Seçili kolonları StandardScaler ile ölçeklendirir; dummy vb. diğer kolonları korur.
    """

    def __init__(self, columns: Iterable[str], copy: bool = True, with_mean: bool = True, with_std: bool = True):
        self.columns = list(columns)
        self.scaler = StandardScaler(copy, with_mean, with_std)
        self.mean_ = None
        self.var_ = None

    def fit(self, X: pd.DataFrame, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.array(np.mean(X[self.columns]))
        self.var_ = np.array(np.var(X[self.columns]))
        return self

    def transform(self, X: pd.DataFrame, y=None, copy=None) -> pd.DataFrame:
        init_col_order = X.columns

        # index korunur (satır hizası bozulmaz)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X[self.columns]),
            columns=self.columns,
            index=X.index,
        )
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# -----------------------------
# Main model
# -----------------------------
class absenteeism_model:
    """
    Pickle edilmiş model + scaler ile inference yapan sınıf.

    Parameters
    ----------
    model_file : str | Path
        Pickle edilmiş model dosyası (örn: 'model')
    scaler_file : str | Path
        Pickle edilmiş scaler dosyası (örn: 'scaler')
    """

    def __init__(self, model_file: Union[str, Path], scaler_file: Union[str, Path]):
        model_path = Path(model_file)
        scaler_path = Path(scaler_file)

        try:
            with model_path.open("rb") as f_model, scaler_path.open("rb") as f_scaler:
                self.reg = pickle.load(f_model)
                self.scaler = pickle.load(f_scaler)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model/Scaler dosyası bulunamadı. model_file='{model_path}', scaler_file='{scaler_path}'. "
                f"Çalışma dizinini ve dosya adlarını kontrol edin."
            ) from e
        except Exception as e:
            raise RuntimeError(
                "Model/Scaler yüklenirken hata oluştu. Dosyalar bozuk olabilir veya pickle uyumsuz olabilir."
            ) from e

        self.data: Optional[pd.DataFrame] = None
        self.preprocessed_data: Optional[pd.DataFrame] = None
        self.df_with_predictions: Optional[pd.DataFrame] = None

    # --------------- helpers ---------------
    @staticmethod
    def _validate_and_fill_columns(df: pd.DataFrame) -> None:
        """
        Girdi kolonlarını kontrol eder.
        - Eksik opsiyonel kolonları default ile ekler.
        - Diğer eksik kolonlar varsa hata verir.
        """
        missing = REQUIRED_COLUMNS - set(df.columns)

        # Opsiyonelleri ekle
        for col, default in OPTIONAL_DEFAULTS.items():
            if col in missing:
                df[col] = default
                missing.remove(col)

        if missing:
            raise ValueError(f"Girdi verisinde eksik kolon(lar) var: {sorted(missing)}")

    @staticmethod
    def _coerce_reason_dummy_columns(reason_columns: pd.DataFrame) -> pd.DataFrame:
        """
        Dummy kolon isimleri bazen int, bazen string gelir.
        Mümkünse int'e çevirerek 1:14 gibi aralık seçimlerini sağlamlaştırır.
        """
        try:
            reason_columns.columns = reason_columns.columns.astype(int)
        except Exception:
            pass
        return reason_columns

    @staticmethod
    def _group_reason(reason_columns: pd.DataFrame, start: int, end: Optional[int]) -> pd.Series:
        """
        reason_columns içinden start-end aralığındaki dummy kolonlardan satır bazında max alır.
        end=None -> start ve sonrası.
        """
        if reason_columns.shape[1] == 0:
            return pd.Series(0, index=reason_columns.index)

        cols = reason_columns.columns

        # İnt kolonlar için hızlı yol
        if pd.api.types.is_integer_dtype(cols):
            sub = reason_columns.loc[:, start:] if end is None else reason_columns.loc[:, start:end]
            if sub.shape[1] == 0:
                return pd.Series(0, index=reason_columns.index)
            return sub.max(axis=1)

        # String/karma kolonlar için: sayıya çevrilebilenleri seç
        selected = []
        for c in cols:
            try:
                v = int(c)
            except Exception:
                continue

            if end is None:
                if v >= start:
                    selected.append(c)
            else:
                if start <= v <= end:
                    selected.append(c)

        if not selected:
            return pd.Series(0, index=reason_columns.index)

        return reason_columns[selected].max(axis=1)

    def _require_data_loaded(self) -> None:
        if self.data is None or self.preprocessed_data is None:
            raise ValueError("Veri yüklü değil. Önce load_and_clean_data(data_file) çağırın.")

    # --------------- public API ---------------
    def load_and_clean_data(self, data_file: Union[str, Path]) -> None:
        """
        Yeni veri dosyasını okur ve eğitimdeki preprocessing adımlarının inference versiyonunu uygular.
        """
        data_path = Path(data_file)
        try:
            df = pd.read_csv(data_path, delimiter=",")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Veri dosyası bulunamadı: '{data_path}'. Çalışma dizini ve dosya adını kontrol edin."
            ) from e

        # Kolon kontrolü + opsiyonel kolonları ekle
        self._validate_and_fill_columns(df)

        # Ham veriyi sakla
        self.df_with_predictions = df.copy()

        # ID düş
        df = df.drop(["ID"], axis=1)

        # Eğitim notebook uyumu için placeholder (string değil)
        df["Absenteeism Time in Hours"] = np.nan

        # Reason dummy -> 4 grup
        reason_columns = pd.get_dummies(df["Reason for Absence"], drop_first=True)
        reason_columns = self._coerce_reason_dummy_columns(reason_columns)

        r1 = self._group_reason(reason_columns, *REASON_GROUPS["Reason_1"]).rename("Reason_1")
        r2 = self._group_reason(reason_columns, *REASON_GROUPS["Reason_2"]).rename("Reason_2")
        r3 = self._group_reason(reason_columns, *REASON_GROUPS["Reason_3"]).rename("Reason_3")
        r4 = self._group_reason(reason_columns, *REASON_GROUPS["Reason_4"]).rename("Reason_4")

        # Reason for Absence düş + grup kolonlarını ekle (isimler net)
        df = df.drop(["Reason for Absence"], axis=1)
        df = pd.concat([df, r1, r2, r3, r4], axis=1)

        # ---- Kolon sırası (isimle; df.columns = [...] yok!) ----
        expected_after_reason = [
            "Date",
            "Transportation Expense",
            "Distance to Work",
            "Age",
            "Daily Work Load Average",
            "Body Mass Index",
            "Education",
            "Children",
            "Pet",
            "Absenteeism Time in Hours",
            "Reason_1",
            "Reason_2",
            "Reason_3",
            "Reason_4",
        ]

        # Eksik bir şey olursa burada net patlasın (model uyumu için)
        missing_after_reason = [c for c in expected_after_reason if c not in df.columns]
        if missing_after_reason:
            raise ValueError(
                "Preprocessing sırasında beklenen kolonlar oluşmadı. Eksikler: "
                f"{missing_after_reason}. (CSV kolonlarını kontrol edin.)"
            )

        df = df[expected_after_reason]

        # Modelin beklediği sıralama (Date sonra işlenip düşecek)
        df = df[
            [
                "Reason_1",
                "Reason_2",
                "Reason_3",
                "Reason_4",
                "Date",
                "Transportation Expense",
                "Distance to Work",
                "Age",
                "Daily Work Load Average",
                "Body Mass Index",
                "Education",
                "Children",
                "Pet",
                "Absenteeism Time in Hours",
            ]
        ]

        # Date processing
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
        df["Month Value"] = df["Date"].dt.month
        df["Day of the Week"] = df["Date"].dt.weekday
        df = df.drop(["Date"], axis=1)

        # Sütunları tekrar sırala (eğitim notebook uyumu)
        df = df[
            [
                "Reason_1",
                "Reason_2",
                "Reason_3",
                "Reason_4",
                "Month Value",
                "Day of the Week",
                "Transportation Expense",
                "Distance to Work",
                "Age",
                "Daily Work Load Average",
                "Body Mass Index",
                "Education",
                "Children",
                "Pet",
                "Absenteeism Time in Hours",
            ]
        ]

        # Education map
        df["Education"] = df["Education"].map({1: 0, 2: 1, 3: 1, 4: 1})

        # Eksikler
        df = df.fillna(value=0)

        # Eğitimde drop edilen kolonlar
        df = df.drop(["Absenteeism Time in Hours"], axis=1)
        df = df.drop(["Day of the Week", "Daily Work Load Average", "Distance to Work"], axis=1)

        # İşlenmiş veriyi sakla
        self.preprocessed_data = df.copy()

        # Ölçeklendir
        self.data = self.scaler.transform(df)

    def predicted_probability(self):
        self._require_data_loaded()
        return self.reg.predict_proba(self.data)[:, 1]

    def predicted_output_category(self):
        self._require_data_loaded()
        return self.reg.predict(self.data)

    def predicted_outputs(self):
        self._require_data_loaded()
        out = self.preprocessed_data.copy()
        out["Probability"] = self.reg.predict_proba(self.data)[:, 1]
        out["Prediction"] = self.reg.predict(self.data)
        return out
