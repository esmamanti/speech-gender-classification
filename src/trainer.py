"""Derin öğrenme modelleri için ortak eğitim + 5-fold CV + test değerlendirmesi.

Protokol (MLP, CNN, CRNN için aynı):
  - 5 fold'un her birinde: 4 fold ile eğit, kalan fold ile doğrula (early stopping)
  - CV sonucu: 5 doğrulama fold'unun ortalaması ± std
  - TEST: 5 fold modelinin olasılıklarının ortalaması (ensemble) ile test setinde ölçülür.
          Ayrıca her fold modelinin test sonucu ayrı ayrı da raporlanır.
"""
import copy
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import SEED
from src.dl_models import count_params
from src.evaluate import (compute_metrics, fmt_pm, model_dirs, plot_confusion,
                          plot_cv_folds, plot_roc, save_json, save_predictions,
                          summarize_cv)
from src.plots import save_fig, save_table, set_style

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CFG = {
    "epochs": 30,          # en fazla epoch
    "patience": 6,         # val loss bu kadar epoch iyileşmezse dur (early stopping)
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(ds, cfg, shuffle):
    # num_workers=0: Windows'ta en sorunsuz seçenek (veri zaten bellekte)
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=0,
                      pin_memory=DEVICE.type == "cuda")


@torch.no_grad()
def predict_proba(model, loader) -> np.ndarray:
    model.eval()
    out = []
    for xb, _ in loader:
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            logits = model(xb.to(DEVICE, non_blocking=True))
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def train_one(model, train_ds, val_ds, cfg, desc=""):
    """Tek bir modeli eğitir. En iyi (en düşük val loss) ağırlıkları döndürür."""
    model.to(DEVICE)
    tl, vl = _loader(train_ds, cfg, True), _loader(val_ds, cfg, False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    loss_fn = nn.BCEWithLogitsLoss()
    use_amp = DEVICE.type == "cuda"               # GPU'da karışık hassasiyet: daha hızlı
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_loss, best_state, best_epoch, wait, hist = np.inf, None, 0, 0, []
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        tot, n, correct = 0.0, 0, 0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                logits = model(xb)
            loss = loss_fn(logits.float(), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(yb)
            n += len(yb)
            correct += ((logits.detach() > 0).float() == yb).sum().item()

        vprob = predict_proba(model, vl)
        vy = val_ds.y.numpy()
        vloss = float(nn.functional.binary_cross_entropy(
            torch.tensor(vprob).clamp(1e-7, 1 - 1e-7), torch.tensor(vy)))
        vacc = float(((vprob >= 0.5) == vy).mean())
        hist.append({"epoch": ep, "train_loss": tot / n, "train_acc": correct / n,
                     "val_loss": vloss, "val_acc": vacc, "lr": opt.param_groups[0]["lr"]})
        print(f"   {desc} epoch {ep:2d} | train loss {tot / n:.4f} acc {correct / n:.4f} | "
              f"val loss {vloss:.4f} acc {vacc:.4f}")

        sched.step(vloss)
        if vloss < best_loss - 1e-4:
            best_loss, best_epoch, wait = vloss, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print(f"   {desc} early stopping (en iyi epoch: {best_epoch})")
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist), best_epoch


def plot_history(hist: pd.DataFrame, path, title):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, h in hist.groupby("fold"):
        c = plt.cm.tab10(k)
        axes[0].plot(h["epoch"], h["train_loss"], "--", color=c, alpha=0.7)
        axes[0].plot(h["epoch"], h["val_loss"], "-", color=c, label=f"fold {k}")
        axes[1].plot(h["epoch"], h["train_acc"] * 100, "--", color=c, alpha=0.7)
        axes[1].plot(h["epoch"], h["val_acc"] * 100, "-", color=c, label=f"fold {k}")
    axes[0].set(title="Loss (kesikli = train, düz = val)", xlabel="Epoch", ylabel="BCE loss")
    axes[1].set(title="Doğruluk (kesikli = train, düz = val)", xlabel="Epoch", ylabel="%")
    axes[1].legend(fontsize=9)
    fig.suptitle(title)
    save_fig(fig, path)


def run_cv(name, display_name, build_model, make_ds, dev, test, cfg=None,
           on_fold_start=None, variant=""):
    """
    name          : çıktı klasörü adı (outputs/models/<name>)
    display_name  : tablolarda görünen ad
    build_model() : yeni (eğitilmemiş) model döndürür
    make_ds(df, train: bool) : Dataset döndürür
    on_fold_start(model, train_df) : (opsiyonel) ör. MLP normalizasyonu
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    base, TABLES, FIGURES = model_dirs(name)
    n_params = count_params(build_model())
    print(f"\n{display_name} | cihaz: {DEVICE} | parametre: {n_params:,} | "
          f"dev: {len(dev)} | test: {len(test)}")
    test_ds = make_ds(test, False)
    test_loader = _loader(test_ds, cfg, False)

    cv_rows, hists, fold_test_rows, test_probs = [], [], [], []
    t_start = time.time()
    for k in sorted(dev["fold"].unique()):
        set_seed(SEED + int(k))
        tr, va = dev[dev["fold"] != k], dev[dev["fold"] == k]
        model = build_model()
        if on_fold_start:
            on_fold_start(model, tr)
        t0 = time.time()
        model, hist, best_ep = train_one(model, make_ds(tr, True), make_ds(va, False),
                                         cfg, desc=f"[fold {k}]")
        vprob = predict_proba(model, _loader(make_ds(va, False), cfg, False))
        cv_rows.append({"fold": int(k),
                        **compute_metrics(va["y"].values, (vprob >= 0.5).astype(int), vprob),
                        "best_epoch": best_ep, "epochs_run": len(hist),
                        "train_time_sec": round(time.time() - t0, 1)})
        hist["fold"] = int(k)
        hists.append(hist)

        tp = predict_proba(model, test_loader)
        test_probs.append(tp)
        fold_test_rows.append({"fold": int(k), **compute_metrics(
            test["y"].values, (tp >= 0.5).astype(int), tp)})
        torch.save(model.state_dict(), base / f"fold{k}.pt")
        print(f"   [fold {k}] val doğruluk: %{cv_rows[-1]['accuracy'] * 100:.2f}")

    total_min = (time.time() - t_start) / 60
    cv = pd.DataFrame(cv_rows)
    hist_all = pd.concat(hists, ignore_index=True)
    fold_test = pd.DataFrame(fold_test_rows)

    # Test: 5 fold modelinin ortalaması (ensemble)
    ens_prob = np.mean(test_probs, axis=0)
    ens_pred = (ens_prob >= 0.5).astype(int)
    test_m = compute_metrics(test["y"].values, ens_pred, ens_prob)

    summary = pd.DataFrame([{
        "model": display_name, "variant": variant, "n_params": n_params,
        "train_time_min": round(total_min, 2),
        **summarize_cv(cv),
        **{f"test_{m}": v for m, v in test_m.items()},
        "test_accuracy_fold_models_mean": fold_test["accuracy"].mean(),
        "test_accuracy_fold_models_std": fold_test["accuracy"].std(),
    }])

    save_table(summary.round(4), TABLES / "summary.csv")
    save_table(cv.round(4), TABLES / "cv_folds.csv")
    save_table(hist_all.round(5), TABLES / "history.csv")
    save_table(fold_test.round(4), TABLES / "test_fold_models.csv")
    save_predictions(test, ens_prob, ens_pred, TABLES / "predictions_test.csv", "ensemble")
    save_json({**cfg, "n_params": n_params, "device": str(DEVICE),
               "test_protocol": "5 fold modelinin olasılık ortalaması"}, base / "config.json")

    plot_history(hist_all, FIGURES / "training_curves.png", f"{display_name} - eğitim eğrileri")
    plot_confusion(test["y"], ens_pred, FIGURES / "confusion_test.png",
                   f"{display_name} - test (5 model ortalaması)")
    plot_roc({f"{display_name} (ensemble)": (test["y"].values, ens_prob)},
             FIGURES / "roc_test.png", f"{display_name} - ROC (test)")
    plot_cv_folds(cv, FIGURES / "cv_folds.png", f"{display_name} - fold sonuçları")

    r = summary.iloc[0]
    print(f"\n========== {display_name.upper()} ==========")
    print(f"Parametre sayısı   : {n_params:,}")
    print(f"Toplam eğitim      : {total_min:.1f} dk")
    print(f"5-fold CV doğruluk : %{fmt_pm(r['cv_accuracy_mean'], r['cv_accuracy_std'])}")
    print(f"5-fold CV F1       : %{fmt_pm(r['cv_f1_mean'], r['cv_f1_std'])}")
    print(f"TEST (ensemble)    : doğruluk %{r['test_accuracy'] * 100:.2f}   "
          f"F1 %{r['test_f1'] * 100:.2f}   AUC {r['test_auc']:.4f}")
    print(f"TEST (tek modeller): %{fmt_pm(r['test_accuracy_fold_models_mean'], r['test_accuracy_fold_models_std'])}")
    print(f"\n✅ {display_name} tamamlandı.")
    return summary