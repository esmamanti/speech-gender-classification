"""
Model 2 - Random Forest (F0 + formant özellikleri)
  Üç özellik seti karşılaştırılır:
    F0            : f0_mean, f0_median, f0_std, f0_p5, f0_p95
    Formant       : F1, F2, F3, F4, formant_dispersion
    F0 + Formant  : hepsi + voiced_ratio   (ana model)
  - 5-fold CV + test setinde son değerlendirme
  - Özellik önemi (Gini + permütasyon)
  - F0 eşiğinin kararsız kaldığı bölgede formant yardımcı oluyor mu?

Çalıştırma (repo ana klasöründen):
    python -m models.random_forest
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from src.config import OUT_MODELS, SEED
from src.evaluate import (compute_metrics, cross_validate, fmt_pm,
                          load_feature_splits, model_dirs, plot_confusion,
                          plot_cv_folds, plot_roc, save_predictions,
                          summarize_cv)
from src.ml_models import FEATURE_SETS, make_random_forest
from src.plots import save_fig, save_table, set_style

NAME = "random_forest"
MAIN = "F0 + Formant"
F0_PRED_FILE = OUT_MODELS / "f0_threshold" / "tables" / "predictions_test_f0_median.csv"


def main():
    base, TABLES, FIGURES = model_dirs(NAME)
    dev, test = load_feature_splits()
    print(f"Dev (5-fold): {len(dev)}  |  Test: {len(test)}")

    summary_rows, cv_all, curves = [], [], {}
    for set_name, cols in FEATURE_SETS.items():
        print(f"  -> {set_name} ({len(cols)} özellik)")
        X_dev, y_dev = dev[cols], dev["y"]

        cv = cross_validate(make_random_forest(), X_dev, y_dev, dev["fold"])
        cv.insert(0, "feature_set", set_name)
        cv_all.append(cv)

        model = make_random_forest().fit(X_dev, y_dev)
        prob = model.predict_proba(test[cols])[:, 1]
        pred = (prob >= 0.5).astype(int)
        test_m = compute_metrics(test["y"], pred, prob)

        summary_rows.append({
            "model": "Random Forest", "variant": set_name, "n_features": len(cols),
            **summarize_cv(cv), **{f"test_{k}": v for k, v in test_m.items()},
        })
        curves[f"RF - {set_name}"] = (test["y"].values, prob)
        slug = set_name.lower().replace(" + ", "_").replace(" ", "_")
        p = save_predictions(test, prob, pred, TABLES / f"predictions_test_{slug}.csv", set_name)
        plot_confusion(test["y"], pred, FIGURES / f"confusion_test_{slug}.png",
                       f"Random Forest ({set_name}) - test")
        if set_name == MAIN:
            main_model, main_cols, main_pred, main_cv = model, cols, p, cv

    summary = pd.DataFrame(summary_rows)
    cv_all = pd.concat(cv_all, ignore_index=True)
    save_table(summary.round(4), TABLES / "summary.csv")
    save_table(cv_all.round(4), TABLES / "cv_folds.csv")
    joblib.dump({"model": main_model, "features": main_cols}, base / "model.pkl")
    print(f"💾 Kaydedildi: {base / 'model.pkl'}")

    # ---------- Özellik önemi (ana model)
    rf = main_model[-1]
    gini = pd.Series(rf.feature_importances_, index=main_cols)
    perm = permutation_importance(main_model, test[main_cols], test["y"],
                                  n_repeats=10, random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({
        "feature": main_cols,
        "gini_importance": gini.values,
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_std": perm.importances_std,
    }).sort_values("permutation_importance_mean", ascending=False).round(4)
    save_table(imp, TABLES / "feature_importance.csv")

    # ---------- F0 eşiği ile karşılaştırma: kararsız bölgede formant işe yarıyor mu?
    zone = None
    if F0_PRED_FILE.exists():
        f0p = pd.read_csv(F0_PRED_FILE)[["path", "correct"]].rename(columns={"correct": "f0_correct"})
        z = main_pred.merge(f0p, on="path").merge(test[["path", "f0_median"]], on="path")
        thr = pd.read_json(OUT_MODELS / "f0_threshold" / "model.json", typ="series")["threshold_hz"]
        z["dist"] = (z["f0_median"] - thr).abs()
        z["dist_bin"] = pd.cut(z["dist"], [0, 10, 20, 30, 50, 1000],
                               labels=["0-10", "10-20", "20-30", "30-50", "50+"])
        zone = z.groupby("dist_bin", observed=False).agg(
            files=("correct", "size"),
            f0_threshold_acc=("f0_correct", "mean"),
            random_forest_acc=("correct", "mean")).round(4)
        save_table(zone, TABLES / "f0_vs_rf_by_distance.csv", index=True)

        both = pd.crosstab(z["f0_correct"].map({True: "F0 doğru", False: "F0 yanlış"}),
                           z["correct"].map({True: "RF doğru", False: "RF yanlış"}))
        save_table(both, TABLES / "f0_vs_rf_agreement.csv", index=True)

    # ---------- Grafikler
    set_style()

    # 1) Özellik setlerine göre CV doğruluğu (± std)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(summary))
    ax.bar(x, summary["cv_accuracy_mean"] * 100, yerr=summary["cv_accuracy_std"] * 100,
           capsize=6, color=["#2E86AB", "#FFB000", "#4CAF50"])
    ax.scatter(x, summary["test_accuracy"] * 100, color="black", zorder=3, label="Test")
    for i, r in summary.iterrows():
        ax.text(i, r["cv_accuracy_mean"] * 100 + r["cv_accuracy_std"] * 100 + 0.3,
                f"{r['cv_accuracy_mean'] * 100:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x, summary["variant"])
    lo = max(0, summary["cv_accuracy_mean"].min() * 100 - 8)
    ax.set(ylim=(lo, 100.5), ylabel="Doğruluk (%)",
           title="Random Forest: özellik setlerine göre (5-fold CV ± std)")
    ax.legend(loc="lower right")
    save_fig(fig, FIGURES / "feature_set_comparison.png")

    # 2) Özellik önemi
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    s = imp.sort_values("gini_importance")
    axes[0].barh(s["feature"], s["gini_importance"], color="#2E86AB")
    axes[0].set(title="Gini önemi (eğitim)", xlabel="Önem")
    s = imp.sort_values("permutation_importance_mean")
    axes[1].barh(s["feature"], s["permutation_importance_mean"],
                 xerr=s["permutation_importance_std"], color="#E84855", capsize=3)
    axes[1].set(title="Permütasyon önemi (test)", xlabel="Doğruluktaki düşüş")
    save_fig(fig, FIGURES / "feature_importance.png")

    # 3) Fold başına sonuç (ana model)
    plot_cv_folds(main_cv, FIGURES / "cv_folds.png", f"Random Forest ({MAIN}) - fold sonuçları")

    # 4) ROC
    plot_roc(curves, FIGURES / "roc_test.png", "Random Forest - ROC (test)")

    # 5) F0 eşiği vs RF, eşiğe uzaklığa göre
    if zone is not None:
        long = zone.reset_index().melt(id_vars="dist_bin",
                                       value_vars=["f0_threshold_acc", "random_forest_acc"],
                                       var_name="model", value_name="acc")
        long["model"] = long["model"].map({"f0_threshold_acc": "F0 eşiği",
                                           "random_forest_acc": f"RF ({MAIN})"})
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=long, x="dist_bin", y="acc", hue="model",
                    palette=["#999999", "#4CAF50"], ax=ax)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", fontsize=9, padding=2)
        ax.set(xlabel="F0'ın eşiğe uzaklığı (Hz)", ylabel="Doğruluk", ylim=(0, 1.08),
               title="Kararsız bölgede (eşiğe yakın) formant yardımcı oluyor mu?")
        save_fig(fig, FIGURES / "f0_vs_rf_by_distance.png")

    # ---------- Ekrana
    print("\n========== RANDOM FOREST ==========")
    for r in summary_rows:
        print(f"[{r['variant']}]")
        print(f"   5-fold CV doğruluk : %{fmt_pm(r['cv_accuracy_mean'], r['cv_accuracy_std'])}")
        print(f"   5-fold CV F1       : %{fmt_pm(r['cv_f1_mean'], r['cv_f1_std'])}")
        print(f"   TEST doğruluk      : %{r['test_accuracy'] * 100:.2f}   "
              f"F1: %{r['test_f1'] * 100:.2f}   AUC: {r['test_auc']:.4f}")
    print("\nÖzellik önemi (permütasyon, test):")
    print(imp[["feature", "permutation_importance_mean", "gini_importance"]].to_string(index=False))
    if zone is not None:
        print("\nF0 eşiği vs RF - eşiğe uzaklığa göre doğruluk (test):")
        print(zone.to_string())
    else:
        print("\n(Not: Önce 'python -m models.f0_threshold' çalıştırılırsa "
              "F0 eşiği ile karşılaştırma tablosu da üretilir.)")
    print("\n✅ Random Forest tamamlandı.")


if __name__ == "__main__":
    main()