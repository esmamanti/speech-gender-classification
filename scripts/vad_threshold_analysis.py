"""
VAD eşik duyarlılık analizi
Soru: Silero VAD eşiğini (threshold) değiştirirsek sonuç ne kadar değişir?

Yöntem:
  - Dengeli rastgele bir alt küme seçilir (varsayılan 500 erkek + 500 kadın)
  - Her eşik için (0.2 ... 0.8) VAD tekrar çalıştırılır
  - Konuşma oranı, konuşma süresi, bölge sayısı ve
    0.5 eşiğiyle örtüşme (IoU) hesaplanır
  - Ayrıca Silero'nun 32 ms'lik olasılıklarının dağılımı çizilir

Çalıştırma (repo ana klasöründen):
    python -m scripts.vad_threshold_analysis
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from silero_vad import get_speech_timestamps
from tqdm import tqdm

from src.config import (COLORS, LABELS, OUT_VAD, ROOT, SAMPLE_RATE, SEED,
                        VAD_MIN_SILENCE_MS, VAD_MIN_SPEECH_MS, VAD_PAD_MS,
                        VAD_THRESHOLD)
from src.plots import save_fig, save_table, set_style
from src.vad import load_model, read_audio, speech_probs

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
N_PER_CLASS = 500          # her cinsiyetten kaç dosya (toplam 1000)
TABLES = OUT_VAD / "tables"
FIGURES = OUT_VAD / "figures"


def segments(wav, threshold):
    return get_speech_timestamps(
        wav, load_model(), threshold=threshold, sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_PAD_MS,
    )


def to_mask(segs, n):
    m = np.zeros(n, dtype=bool)
    for s in segs:
        m[s["start"]:s["end"]] = True
    return m


def iou(a, b):
    union = (a | b).sum()
    return 1.0 if union == 0 else (a & b).sum() / union


def main():
    res = pd.read_csv(OUT_VAD / "vad_results.csv")
    sample = pd.concat([g.sample(min(N_PER_CLASS, len(g)), random_state=SEED)
                        for _, g in res.groupby("label")], ignore_index=True)
    print(f"{len(sample)} dosya x {len(THRESHOLDS)} eşik analiz edilecek...")
    load_model()

    rows, all_probs = [], {lab: [] for lab in LABELS}
    for r in tqdm(sample.itertuples(), total=len(sample), desc="Eşik analizi"):
        wav = read_audio(ROOT / r.path)
        n = len(wav)
        all_probs[r.label].append(speech_probs(wav))

        masks = {th: to_mask(segments(wav, th), n) for th in THRESHOLDS}
        ref = masks[VAD_THRESHOLD]
        for th in THRESHOLDS:
            m = masks[th]
            rows.append({
                "path": r.path, "label": r.label, "threshold": th,
                "speech_sec": m.sum() / SAMPLE_RATE,
                "speech_ratio": m.mean(),
                "has_speech": bool(m.any()),
                "iou_vs_default": iou(m, ref),
            })
    df = pd.DataFrame(rows)

    # ---------------- Tablolar
    agg = dict(files=("path", "size"),
               no_speech=("has_speech", lambda x: int((~x).sum())),
               mean_speech_sec=("speech_sec", "mean"),
               mean_speech_ratio=("speech_ratio", "mean"),
               mean_iou_vs_default=("iou_vs_default", "mean"),
               min_iou_vs_default=("iou_vs_default", "min"))
    overall = df.groupby("threshold").agg(**agg).round(4)
    by_label = df.groupby(["threshold", "label"]).agg(**agg).round(4)
    save_table(overall, TABLES / "threshold_sensitivity.csv", index=True)
    save_table(by_label, TABLES / "threshold_sensitivity_by_label.csv", index=True)

    probs = {lab: np.concatenate(v) for lab, v in all_probs.items()}
    p_all = np.concatenate(list(probs.values()))
    prob_tbl = pd.DataFrame([{
        "frames": len(p_all),
        "share_below_0.1": (p_all < 0.1).mean(),
        "share_0.1_to_0.9": ((p_all >= 0.1) & (p_all <= 0.9)).mean(),
        "share_above_0.9": (p_all > 0.9).mean(),
        "share_0.3_to_0.7": ((p_all >= 0.3) & (p_all <= 0.7)).mean(),
    }]).round(4)
    save_table(prob_tbl, TABLES / "vad_prob_distribution.csv")

    # ---------------- Grafikler
    set_style()

    # 1) Eşik -> konuşma oranı ve IoU
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.lineplot(data=df, x="threshold", y="speech_ratio", hue="label",
                 hue_order=LABELS, palette=COLORS, marker="o", errorbar=("ci", 95), ax=axes[0])
    axes[0].set(title="Eşik değerine göre konuşma oranı",
                xlabel="VAD eşiği", ylabel="Ortalama konuşma oranı")
    sns.lineplot(data=df, x="threshold", y="iou_vs_default", hue="label",
                 hue_order=LABELS, palette=COLORS, marker="o", errorbar=("ci", 95), ax=axes[1])
    axes[1].set(title=f"Eşik {VAD_THRESHOLD} ile örtüşme (IoU)",
                xlabel="VAD eşiği", ylabel="Ortalama IoU", ylim=(0, 1.02))
    for ax in axes:
        ax.axvline(VAD_THRESHOLD, color="gray", linestyle="--", linewidth=1)
    save_fig(fig, FIGURES / "threshold_sensitivity.png")

    # 2) Silero olasılık dağılımı
    pdf = pd.DataFrame({"prob": np.concatenate([probs[l] for l in LABELS]),
                        "label": np.concatenate([[l] * len(probs[l]) for l in LABELS])})
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data=pdf, x="prob", hue="label", hue_order=LABELS, palette=COLORS, bins=50,
                 stat="percent", common_norm=False, element="step", ax=ax)
    ax.axvspan(min(THRESHOLDS), max(THRESHOLDS), color="gray", alpha=0.12)
    ax.axvline(VAD_THRESHOLD, color="black", linestyle="--")
    ax.set(title="Silero VAD konuşma olasılıklarının dağılımı (32 ms pencereler)",
           xlabel="P(konuşma)", ylabel="Pencere yüzdesi (%)")
    handles = [Patch(color=COLORS[l], alpha=0.6, label=l) for l in LABELS] + [
        Patch(color="gray", alpha=0.25, label="Denenen eşik aralığı"),
        Line2D([0], [0], color="black", linestyle="--",
               label=f"Kullanılan eşik = {VAD_THRESHOLD}")]
    ax.legend(handles=handles, loc="upper center")
    save_fig(fig, FIGURES / "vad_prob_distribution.png")

    # ---------------- Ekrana
    print("\n========== EŞİK DUYARLILIK ==========")
    print(overall.to_string())
    print("\nOlasılık dağılımı (tüm pencereler):")
    print(prob_tbl.to_string(index=False))
    print("\n✅ Eşik analizi tamamlandı.")


if __name__ == "__main__":
    main()
