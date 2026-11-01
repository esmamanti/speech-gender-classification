"""
Model 1 - F0 eşiği (kural tabanlı baseline)
  "F0 >= eşik ise kadın, değilse erkek"
  - Eşik, eğitim verisinde en yüksek doğruluğu veren değer olarak seçilir
  - 5-fold CV: her fold için eşik yeniden öğrenilir -> eşik ne kadar kararlı?
  - Son model: train+val'in tamamıyla eşik bulunur, test setinde BİR KEZ ölçülür
  - f0_median (ana) ve f0_mean ile ayrı ayrı denenir

Çalıştırma (repo ana klasöründen):
    python -m models.f0_threshold
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import COLORS, LABELS
from src.evaluate import (compute_metrics, cross_validate, fmt_pm,
                          load_feature_splits, model_dirs, plot_confusion,
                          plot_roc, save_json, save_predictions, summarize_cv)
from src.ml_models import F0ThresholdClassifier
from src.plots import save_fig, save_table, set_style

NAME = "f0_threshold"
FEATURES = ["f0_median", "f0_mean"]   # ilki ana model
MAIN = FEATURES[0]


def fold_thresholds(dev, feat):
    """Her fold'da öğrenilen eşik (eşiğin kararlılığını görmek için)."""
    out = []
    for k in sorted(dev["fold"].unique()):
        tr = dev["fold"] != k
        m = F0ThresholdClassifier().fit(dev.loc[tr, [feat]], dev.loc[tr, "y"])
        out.append(m.threshold_)
    return out


def main():
    base, TABLES, FIGURES = model_dirs(NAME)
    dev, test = load_feature_splits()
    print(f"Dev (5-fold): {len(dev)}  |  Test: {len(test)}")

    summary_rows, curves, preds_all = [], {}, []
    for feat in FEATURES:
        X_dev, y_dev = dev[[feat]], dev["y"]

        # ---------- 5-fold CV
        cv = cross_validate(F0ThresholdClassifier(), X_dev, y_dev, dev["fold"])
        cv["threshold_hz"] = fold_thresholds(dev, feat)
        cv.insert(0, "feature", feat)
        save_table(cv.round(4), TABLES / f"cv_folds_{feat}.csv")

        # ---------- Son model: tüm dev -> test
        model = F0ThresholdClassifier().fit(X_dev, y_dev)
        prob = model.predict_proba(test[[feat]])[:, 1]
        pred = model.predict(test[[feat]])
        test_m = compute_metrics(test["y"], pred, prob)

        summary_rows.append({
            "model": "F0 eşiği", "variant": feat,
            "threshold_hz": round(model.threshold_, 1),
            **summarize_cv(cv),
            **{f"test_{k}": v for k, v in test_m.items()},
        })
        curves[f"F0 eşiği ({feat})"] = (test["y"].values, prob)
        preds_all.append(save_predictions(test, prob, pred,
                                          TABLES / f"predictions_test_{feat}.csv", feat))
        plot_confusion(test["y"], pred, FIGURES / f"confusion_test_{feat}.png",
                       f"F0 eşiği ({feat} ≥ {model.threshold_:.1f} Hz) - test")

        if feat == MAIN:
            main_model, main_cv = model, cv
            save_json({"feature": feat, "threshold_hz": model.threshold_,
                       "rule": f"{feat} >= threshold -> female, else male"},
                      base / "model.json")

    summary = pd.DataFrame(summary_rows)
    save_table(summary.round(4), TABLES / "summary.csv")

    # ---------- Hata analizi (ana model)
    p = preds_all[0].merge(test[["path", MAIN]], on="path")
    errors = p[~p["correct"]].copy()
    errors["distance_to_threshold_hz"] = (errors[MAIN] - main_model.threshold_).round(1)
    errors = errors.sort_values("distance_to_threshold_hz", key=abs, ascending=False)
    save_table(errors[["path", "label", "pred_label", MAIN, "distance_to_threshold_hz"]],
               TABLES / "errors_test.csv")

    # Eşiğe uzaklığa göre doğruluk: hatalar eşiğin yakınında mı toplanıyor?
    p["dist"] = (p[MAIN] - main_model.threshold_).abs()
    bins = [0, 10, 20, 30, 50, 1000]
    p["dist_bin"] = pd.cut(p["dist"], bins, labels=["0-10", "10-20", "20-30", "30-50", "50+"])
    by_dist = p.groupby("dist_bin", observed=False).agg(
        files=("correct", "size"), accuracy=("correct", "mean")).round(4)
    save_table(by_dist, TABLES / "accuracy_by_distance_to_threshold.csv", index=True)

    # ---------- Grafikler
    set_style()
    thr = main_model.threshold_

    # 1) Eşik taraması: eşik -> doğruluk
    grid, acc = main_model.scan_
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid, acc * 100, color="#333333", linewidth=2)
    ax.axvline(thr, color="#E84855", linestyle="--", label=f"Seçilen eşik = {thr:.1f} Hz")
    for i, t in enumerate(main_cv["threshold_hz"]):
        ax.axvline(t, color="gray", alpha=0.5, linewidth=1,
                   label="Fold eşikleri" if i == 0 else None)
    ax.set(xlabel=f"Eşik ({MAIN}, Hz)", ylabel="Doğruluk (%) - train+val",
           title="F0 eşik taraması", ylim=(40, 100))
    ax.legend()
    save_fig(fig, FIGURES / "threshold_scan.png")

    # 2) F0 dağılımı + eşik
    both = pd.concat([dev, test])
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data=both, x=MAIN, hue="label", hue_order=LABELS, palette=COLORS,
                 bins=70, element="step", ax=ax)
    ax.axvline(thr, color="black", linestyle="--", linewidth=2)
    ax.text(thr, ax.get_ylim()[1] * 0.92, f"  eşik = {thr:.1f} Hz", fontsize=11)
    ax.set(title="F0 dağılımı ve karar eşiği", xlabel=f"{MAIN} (Hz)", ylabel="Kayıt sayısı")
    save_fig(fig, FIGURES / "f0_hist_threshold.png")

    # 3) Eşiğe uzaklığa göre doğruluk
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=by_dist.index.astype(str), y=by_dist["accuracy"] * 100,
                color="#2E86AB", ax=ax)
    ax.bar_label(ax.containers[0], fmt="%.1f", padding=2)
    ax.set(xlabel="Eşiğe uzaklık (Hz)", ylabel="Doğruluk (%)", ylim=(0, 105),
           title="Hatalar eşiğin yakınında toplanıyor mu? (test)")
    save_fig(fig, FIGURES / "accuracy_by_distance.png")

    # 4) ROC
    plot_roc(curves, FIGURES / "roc_test.png", "F0 eşiği - ROC (test)")

    # ---------- Ekrana
    print("\n========== F0 EŞİĞİ ==========")
    for r in summary_rows:
        print(f"[{r['variant']}] eşik = {r['threshold_hz']} Hz")
        print(f"   5-fold CV doğruluk : %{fmt_pm(r['cv_accuracy_mean'], r['cv_accuracy_std'])}")
        print(f"   5-fold CV F1       : %{fmt_pm(r['cv_f1_mean'], r['cv_f1_std'])}")
        print(f"   TEST doğruluk      : %{r['test_accuracy'] * 100:.2f}   "
              f"F1: %{r['test_f1'] * 100:.2f}   AUC: {r['test_auc']:.4f}")
    print(f"\nFold eşikleri ({MAIN}): "
          + ", ".join(f"{t:.1f}" for t in main_cv["threshold_hz"]) + " Hz")
    print(f"Test hata sayısı ({MAIN}): {len(errors)} / {len(test)}")
    print("\nEşiğe uzaklığa göre doğruluk (test):")
    print(by_dist.to_string())
    print("\n✅ F0 eşiği modeli tamamlandı.")


if __name__ == "__main__":
    main()