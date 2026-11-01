"""Klasik (makine öğrenmesi) modeller: F0 eşiği ve Random Forest."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from src.config import SEED

# Random Forest'ın deneyeceği özellik setleri
F0_COLS = ["f0_mean", "f0_median", "f0_std", "f0_p5", "f0_p95"]
FORMANT_COLS = ["F1", "F2", "F3", "F4", "formant_dispersion"]
FEATURE_SETS = {
    "F0": F0_COLS,
    "Formant": FORMANT_COLS,
    "F0 + Formant": F0_COLS + FORMANT_COLS + ["voiced_ratio"],
}


class F0ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Tek bir F0 değerine bakan kural:  F0 >= eşik  ->  kadın (1), değilse erkek (0).
    Eşik, eğitim verisinde doğruluğu en yüksek yapan değer seçilerek 'öğrenilir'.

    X: tek sütunlu (F0, Hz) dizi / DataFrame.
    """

    def __init__(self, low=80.0, high=260.0, step=0.5, scale=10.0):
        self.low, self.high, self.step = low, high, step
        self.scale = scale  # predict_proba için yumuşatma (Hz)

    def _f0(self, X):
        return np.asarray(X, dtype=float).reshape(len(X), -1)[:, 0]

    def fit(self, X, y):
        f0, y = self._f0(X), np.asarray(y)
        self.classes_ = np.array([0, 1])
        grid = np.arange(self.low, self.high + self.step, self.step)
        acc = np.array([((f0 >= t).astype(int) == y).mean() for t in grid])
        best = np.flatnonzero(acc == acc.max())
        self.threshold_ = float(grid[best].mean())   # eşitlik varsa ortadaki eşik
        self.train_accuracy_ = float(acc.max())
        self.scan_ = (grid, acc)                      # grafik için
        return self

    def predict_proba(self, X):
        # Eşiğe uzaklığa göre kadın olasılığı (ROC/AUC için). Eşikte = 0.5
        p = 1.0 / (1.0 + np.exp(-(self._f0(X) - self.threshold_) / self.scale))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self._f0(X) >= self.threshold_).astype(int)


def make_random_forest():
    """Eksik değer (nadiren formant bulunamazsa) medyanla doldurulur."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                               n_jobs=-1, random_state=SEED),
    )