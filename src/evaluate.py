"""Tüm modeller için ortak değerlendirme: metrikler, CV, confusion matrix, ROC.
Pozitif sınıf = female (LABEL2ID['female'] = 1)."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

from src.config import (ID2LABEL, LABEL2ID, LABELS, OUT_FEATURES, OUT_MODELS,
                        OUT_SPLITS, SPLIT_MODE)
from src.plots import save_fig, save_table, set_style

METRICS = ["accuracy", "precision", "recall", "f1", "auc"]


# ---------------------------------------------------------------- veri
def load_feature_splits():
    """Akustik özellikleri 5-fold (dev) ve test olarak döndürür.
    dev: train+val birleşik, 'fold' sütunu var. test: hiç dokunulmayan set.
    Hangi bölmenin kullanılacağını SPLIT_MODE belirler (random / speaker)."""
    print(f"📂 Bölme türü: {SPLIT_MODE}  ({OUT_SPLITS})")
    feats = pd.read_csv(OUT_FEATURES / "acoustic_features.csv")
    folds = pd.read_csv(OUT_SPLITS / "cv_folds.csv")[["path", "fold"]]
    test_paths = set(pd.read_csv(OUT_SPLITS / "test.csv")["path"])

    dev = feats.merge(folds, on="path", how="inner").reset_index(drop=True)
    test = feats[feats["path"].isin(test_paths)].reset_index(drop=True)
    for d in (dev, test):
        d["y"] = d["label"].map(LABEL2ID)
    return dev, test


# ---------------------------------------------------------------- metrikler
def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    m["auc"] = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    return m


def cross_validate(model, X: pd.DataFrame, y: pd.Series, folds: pd.Series):
    """sklearn uyumlu model ile k-fold. Her fold için metrik döndürür."""
    rows = []
    for k in sorted(folds.unique()):
        tr, va = folds != k, folds == k
        m = clone(model).fit(X[tr], y[tr])
        prob = m.predict_proba(X[va])[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append({"fold": int(k), **compute_metrics(y[va], pred, prob)})
    return pd.DataFrame(rows)


def summarize_cv(cv: pd.DataFrame) -> dict:
    """Fold sonuçlarından ortalama ± std."""
    out = {}
    for m in METRICS:
        out[f"cv_{m}_mean"] = cv[m].mean()
        out[f"cv_{m}_std"] = cv[m].std()
    return out


def fmt_pm(mean, std, pct=True):
    return f"{mean * 100:.2f} ± {std * 100:.2f}" if pct else f"{mean:.3f} ± {std:.3f}"


# ---------------------------------------------------------------- kayıt
def model_dirs(name: str):
    base = OUT_MODELS / name
    (base / "tables").mkdir(parents=True, exist_ok=True)
    (base / "figures").mkdir(parents=True, exist_ok=True)
    return base, base / "tables", base / "figures"


def save_predictions(df: pd.DataFrame, y_prob, y_pred, path, variant: str):
    out = pd.DataFrame({
        "path": df["path"].values, "label": df["label"].values,
        "y_true": df["y"].values, "y_prob_female": np.round(y_prob, 5),
        "y_pred": y_pred, "pred_label": [ID2LABEL[int(p)] for p in y_pred],
        "variant": variant,
    })
    out["correct"] = out["y_true"] == out["y_pred"]
    save_table(out, path)
    return out


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"💾 Kaydedildi: {path}")


# ---------------------------------------------------------------- grafikler
def plot_confusion(y_true, y_pred, path, title):
    set_style()
    cm = confusion_matrix(y_true, y_pred, labels=[LABEL2ID[l] for l in LABELS])
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    annot = np.array([[f"{c}\n(%{p:.1f})" for c, p in zip(r1, r2)]
                      for r1, r2 in zip(cm, pct)])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(pct, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=100,
                xticklabels=LABELS, yticklabels=LABELS, cbar_kws={"label": "%"}, ax=ax)
    ax.set(xlabel="Tahmin", ylabel="Gerçek", title=title)
    save_fig(fig, path)


def plot_roc(curves: dict, path, title="ROC eğrisi (test)"):
    """curves = {'isim': (y_true, y_prob), ...}"""
    set_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (yt, yp) in curves.items():
        fpr, tpr, _ = roc_curve(yt, yp)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={roc_auc_score(yt, yp):.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set(xlabel="Yanlış pozitif oranı", ylabel="Doğru pozitif oranı",
           title=title, xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right")
    save_fig(fig, path)


def plot_cv_folds(cv: pd.DataFrame, path, title):
    """Fold başına doğruluk ve F1 (dağılımı görmek için)."""
    set_style()
    long = cv.melt(id_vars="fold", value_vars=["accuracy", "f1"],
                   var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=long, x="fold", y="score", hue="metric", ax=ax)
    lo = max(0, long["score"].min() - 0.05)
    ax.set(ylim=(lo, 1.0), title=title, xlabel="Fold", ylabel="Skor")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8, rotation=90, padding=2)
    save_fig(fig, path)