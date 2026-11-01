"""
Aşama 4 - Özellik çıkarımı (VAD sonrası temiz seslerden)
  1) F0 + formant (Praat / Parselmouth)  -> acoustic_features.csv
  2) Log-mel spektrogram                -> mels/<split>/<label>/<dosya>.npy
  3) Tablo + grafik

Çalıştırma (repo ana klasöründen):
    python -m scripts.extract_features
"""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.acoustic import ACOUSTIC_COLS, extract_acoustic, get_tracks
from src.config import (COLORS, HOP_LENGTH, LABEL2ID, LABELS, OUT_FEATURES,
                        OUT_VAD, ROOT, SAMPLE_RATE, SEED)
from src.features import load_wav, log_mel
from src.plots import save_fig, save_table, set_style

TABLES = OUT_FEATURES / "tables"
FIGURES = OUT_FEATURES / "figures"
MELS = OUT_FEATURES / "mels"
# Paralel işlem sayısı. Çok yüksek olursa Windows'ta bellek (sayfa dosyası)
# yetmeyebilir; sorun olursa 2'ye veya 1'e düşürün.
N_JOBS = min(4, max(1, (os.cpu_count() or 2) - 1))

# Karşılaştırma için: Kaggle "voicegender" (primaryobjects) veri setindeki
# meanfun (ortalama F0) değerlerinin cinsiyete göre ortalaması (Hz)
VOICEGENDER_F0 = {"male": 115.9, "female": 169.7}


# ---------------------------------------------------------------- tek dosya
def process_one(args):
    """Paralel çalışır: bir dosyanın akustik özellikleri + log-mel'i."""
    clean_path, split, label = args
    wav = load_wav(ROOT / clean_path)
    feats = extract_acoustic(wav)

    mel = log_mel(wav)
    out = MELS / split / label / (Path(clean_path).stem + ".npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, mel.astype(np.float16))   # float16: diskte yarı yer
    feats["mel_path"] = out.relative_to(ROOT).as_posix()
    feats["mel_frames"] = mel.shape[1]
    return feats


def extract_all(vad: pd.DataFrame) -> pd.DataFrame:
    jobs = list(zip(vad["clean_path"], vad["split"], vad["label"]))
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        results = list(tqdm(ex.map(process_one, jobs, chunksize=16),
                            total=len(jobs), desc=f"Özellik çıkarımı ({N_JOBS} işlem)"))
    feats = pd.DataFrame(results)
    return pd.concat([vad[["path", "clean_path", "label", "split"]].reset_index(drop=True),
                      feats], axis=1)


# ---------------------------------------------------------------- tablolar
def cohen_d(a, b):
    a, b = a.dropna(), b.dropna()
    pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled


def make_tables(df: pd.DataFrame):
    # Cinsiyete göre istatistikler
    stats = (df.groupby("label")[ACOUSTIC_COLS]
               .agg(["mean", "std", "median"]).T.unstack())
    stats = stats.reindex(columns=pd.MultiIndex.from_product([LABELS, ["mean", "std", "median"]]))
    stats.columns = [f"{l}_{s}" for l, s in stats.columns]
    save_table(stats.round(2), TABLES / "acoustic_stats_by_label.csv", index=True)

    # Her özellik tek başına erkek/kadını ne kadar ayırıyor?
    y = (df["label"] == "female").astype(int)
    rows = []
    for c in ACOUSTIC_COLS:
        ok = df[c].notna()
        auc = roc_auc_score(y[ok], df.loc[ok, c])
        rows.append({
            "feature": c,
            "male_mean": df.loc[df.label == "male", c].mean(),
            "female_mean": df.loc[df.label == "female", c].mean(),
            "cohen_d": cohen_d(df.loc[df.label == "female", c], df.loc[df.label == "male", c]),
            # AUC 0.5 = ayırt edemiyor, 1.0 = mükemmel ayırıyor (yön önemsiz)
            "single_feature_auc": max(auc, 1 - auc),
        })
    sep = pd.DataFrame(rows).sort_values("single_feature_auc", ascending=False).round(3)
    save_table(sep, TABLES / "feature_separation.csv")

    # Eksik değerler (F0 bulunamayan kayıtlar)
    miss = df[ACOUSTIC_COLS].isna().groupby(df["label"]).sum().T[LABELS]
    save_table(miss, TABLES / "missing_values.csv", index=True)

    # Voicegender CSV ile karşılaştırma
    cmp = pd.DataFrame([{
        "label": l,
        "our_f0_mean_hz": df.loc[df.label == l, "f0_mean"].mean(),
        "our_f0_median_hz": df.loc[df.label == l, "f0_median"].median(),
        "voicegender_meanfun_hz": VOICEGENDER_F0[l],
    } for l in LABELS]).round(1)
    save_table(cmp, TABLES / "f0_vs_voicegender.csv")

    save_table(df.groupby("split")["label"].value_counts().unstack()[LABELS],
               TABLES / "feature_counts.csv", index=True)
    return sep, cmp, miss


# ---------------------------------------------------------------- grafikler
def plot_example_tracks(row, idx):
    """Spektrogram + F0 eğrisi + formant noktaları."""
    wav = load_wav(ROOT / row.clean_path)
    times, f0, fm = get_tracks(wav)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.specgram(wav, NFFT=512, Fs=SAMPLE_RATE, noverlap=384, cmap="Greys",
                vmin=-130, scale="dB")
    for i, c in enumerate(["#FFB000", "#FE6100", "#DC267F", "#785EF0"]):
        ax.scatter(times, fm[i], s=4, color=c, label=f"F{i + 1}")
    ax.set(ylim=(0, 5000), xlabel="Zaman (sn)", ylabel="Frekans (Hz)",
           title=f"{row.label}: spektrogram + formantlar + F0")
    ax2 = ax.twinx()
    ax2.plot(times, f0, color=COLORS[row.label], linewidth=2.5, label="F0")
    ax2.set(ylim=(0, 500), ylabel="F0 (Hz)")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", markerscale=3, ncol=5)
    save_fig(fig, FIGURES / f"example_tracks_{row.label}_{idx}.png")


def make_figures(df: pd.DataFrame, sep: pd.DataFrame):
    set_style()
    kw = dict(hue="label", hue_order=LABELS, palette=COLORS)

    # 1) F0 dağılımı
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data=df, x="f0_mean", bins=60, kde=True, element="step",
                 stat="density", common_norm=False, ax=ax, **kw)
    for l in LABELS:
        m = df.loc[df.label == l, "f0_mean"].mean()
        ax.axvline(m, color=COLORS[l], linestyle="--")
        ax.text(m, ax.get_ylim()[1] * 0.95, f" {m:.0f} Hz", color=COLORS[l])
    ax.set(title="Ortalama F0 dağılımı", xlabel="F0 (Hz)", ylabel="Yoğunluk")
    save_fig(fig, FIGURES / "f0_hist.png")

    # 2) F0 boxplot (mean, median, p5, p95)
    f0cols = ["f0_p5", "f0_median", "f0_mean", "f0_p95"]
    long = df.melt(id_vars="label", value_vars=f0cols, var_name="feature", value_name="Hz")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=long, x="feature", y="Hz", showfliers=False, ax=ax, **kw)
    ax.set(title="F0 istatistikleri", xlabel="", ylabel="Hz")
    save_fig(fig, FIGURES / "f0_boxplot.png")

    # 3) Formant boxplot
    fcols = [c for c in ACOUSTIC_COLS if c.startswith("F") and c[1:].isdigit()]
    long = df.melt(id_vars="label", value_vars=fcols, var_name="formant", value_name="Hz")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=long, x="formant", y="Hz", showfliers=False, ax=ax, **kw)
    ax.set(title="Formant frekansları (F1-F4)", xlabel="", ylabel="Hz")
    save_fig(fig, FIGURES / "formants_boxplot.png")

    # 4) F0 - F1 ve F0 - formant dispersion saçılım
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pts = df.sample(min(3000, len(df)), random_state=SEED)
    sns.scatterplot(data=pts, x="f0_mean", y="F1", s=10, alpha=0.5, ax=axes[0], **kw)
    axes[0].set(title="F0 - F1", xlabel="F0 (Hz)", ylabel="F1 (Hz)")
    sns.scatterplot(data=pts, x="f0_mean", y="formant_dispersion", s=10, alpha=0.5,
                    ax=axes[1], **kw)
    axes[1].set(title="F0 - formant dağılımı", xlabel="F0 (Hz)",
                ylabel="Formant dağılımı (Hz)")
    save_fig(fig, FIGURES / "f0_formant_scatter.png")

    # 5) Tek özellik AUC
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=sep, y="feature", x="single_feature_auc", color="#2E86AB", ax=ax)
    ax.bar_label(ax.containers[0], fmt="%.3f", padding=3, fontsize=9)
    ax.axvline(0.5, color="gray", linestyle="--")
    ax.set(xlim=(0.4, 1.05), title="Her özelliğin tek başına ayırt ediciliği (AUC)",
           xlabel="AUC (0.5 = şans, 1.0 = mükemmel)", ylabel="")
    save_fig(fig, FIGURES / "feature_auc.png")

    # 6) Korelasyon
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[ACOUSTIC_COLS].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, square=True, annot_kws={"size": 8}, ax=ax)
    ax.set(title="Akustik özellikler arası korelasyon")
    save_fig(fig, FIGURES / "acoustic_correlation.png")

    # 7) Örnek F0 + formant eğrileri ve log-mel örnekleri
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, l in zip(axes, LABELS):
        g = df[df.label == l]
        row = g.iloc[(g["f0_mean"] - g["f0_mean"].median()).abs().argsort().iloc[0]]
        plot_example_tracks(row, 1)
        mel = np.load(ROOT / row.mel_path).astype(np.float32)
        t = mel.shape[1] * HOP_LENGTH / SAMPLE_RATE
        im = ax.imshow(mel, origin="lower", aspect="auto", cmap="magma",
                       extent=[0, t, 0, mel.shape[0]])
        ax.set(title=f"{l} (F0 ≈ {row.f0_mean:.0f} Hz)", xlabel="Zaman (sn)")
        ax.grid(False)
    axes[0].set_ylabel("Mel bandı")
    fig.colorbar(im, ax=axes, label="dB")
    save_fig(fig, FIGURES / "logmel_examples.png")


# ---------------------------------------------------------------- main
def main():
    vad = pd.read_csv(OUT_VAD / "vad_results.csv")
    vad = vad[vad["has_speech"]].reset_index(drop=True)
    print(f"{len(vad)} temiz kayıt işlenecek...")

    df = extract_all(vad)
    df["label_id"] = df["label"].map(LABEL2ID)
    save_table(df, OUT_FEATURES / "acoustic_features.csv")

    sep, cmp, miss = make_tables(df)
    make_figures(df, sep)

    print("\n========== ÖZELLİK ÖZETİ ==========")
    print("Tek özellik ayırt ediciliği (AUC):")
    print(sep.to_string(index=False))
    print("\nF0 karşılaştırma (bizim veri vs voicegender CSV):")
    print(cmp.to_string(index=False))
    print(f"\nF0 bulunamayan kayıt: {int(miss.loc['f0_mean'].sum())}")
    print("\n✅ Özellik çıkarımı tamamlandı.")


if __name__ == "__main__":
    main()