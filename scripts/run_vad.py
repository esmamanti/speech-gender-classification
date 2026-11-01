"""
Aşama 3 - Silero VAD
Her kayıt için:
  1) Konuşma bölgelerini bul
  2) Konuşma yoksa -> no_speech listesine yaz, atla
  3) Konuşma varsa -> sadece konuşma kısımlarını birleştir, clean_wav'a kaydet
Sonra tablo, grafik ve örnek sesler üret.

Çalıştırma (repo ana klasöründen):
    python -m scripts.run_vad
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
from tqdm import tqdm

from src.config import (COLORS, LABELS, OUT_DATA, OUT_VAD, ROOT, SAMPLE_RATE,
                        SEED, VAD_MIN_SILENCE_MS, VAD_MIN_SPEECH_MS,
                        VAD_PAD_MS, VAD_THRESHOLD)
from src.plots import save_fig, save_table, set_style
from src.vad import (get_segments, keep_speech, load_model, read_audio,
                     segments_to_seconds, speech_probs)

TABLES = OUT_VAD / "tables"
FIGURES = OUT_VAD / "figures"
EXAMPLES = OUT_VAD / "audio_examples"
CLEAN = OUT_VAD / "clean_wav"
N_EXAMPLES = 3  # her cinsiyetten kaç örnek çizilsin


def load_splits() -> pd.DataFrame:
    parts = []
    for s in ["train", "val", "test"]:
        df = pd.read_csv(OUT_DATA / f"{s}.csv")
        df["split"] = s
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- VAD
def process_all(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in tqdm(df.itertuples(), total=len(df), desc="Silero VAD"):
        wav = read_audio(ROOT / r.path)
        segs = get_segments(wav)
        speech = keep_speech(wav, segs)

        total_sec = len(wav) / SAMPLE_RATE
        speech_sec = len(speech) / SAMPLE_RATE
        clean_path = ""
        if segs:
            out = CLEAN / r.split / r.label / (ROOT / r.path).name
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, speech.numpy(), SAMPLE_RATE)
            clean_path = out.relative_to(ROOT).as_posix()

        rows.append({
            "path": r.path, "label": r.label, "split": r.split,
            "duration_sec": round(total_sec, 3),
            "speech_sec": round(speech_sec, 3),
            "silence_removed_sec": round(total_sec - speech_sec, 3),
            "speech_ratio": round(speech_sec / total_sec, 4) if total_sec else 0,
            "n_segments": len(segs),
            "has_speech": bool(segs),
            "segments_sec": json.dumps(segments_to_seconds(segs)),
            "clean_path": clean_path,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- tablolar
def make_tables(res: pd.DataFrame):
    save_table(res, OUT_VAD / "vad_results.csv")
    save_table(res[~res["has_speech"]][["path", "label", "split", "duration_sec"]],
               TABLES / "no_speech.csv")

    def summarize(g):
        return pd.Series({
            "files": len(g),
            "has_speech": int(g["has_speech"].sum()),
            "no_speech": int((~g["has_speech"]).sum()),
            "mean_duration_sec": g["duration_sec"].mean(),
            "mean_speech_sec": g["speech_sec"].mean(),
            "mean_silence_removed_sec": g["silence_removed_sec"].mean(),
            "mean_speech_ratio": g["speech_ratio"].mean(),
            "mean_n_segments": g["n_segments"].mean(),
            "total_hours_before": g["duration_sec"].sum() / 3600,
            "total_hours_after": g["speech_sec"].sum() / 3600,
        })

    by_label = res.groupby("label").apply(summarize, include_groups=False).reindex(LABELS)
    by_label.loc["all"] = summarize(res)
    by_label = by_label.round(3)
    save_table(by_label, TABLES / "vad_summary.csv", index=True)

    by_split = res.groupby(["split", "label"]).agg(
        files=("path", "size"),
        no_speech=("has_speech", lambda x: int((~x).sum())),
        mean_speech_ratio=("speech_ratio", "mean"),
    ).round(3)
    save_table(by_split, TABLES / "vad_by_split.csv", index=True)

    seg_tbl = (res.groupby(["n_segments", "label"]).size()
               .unstack(fill_value=0).reindex(columns=LABELS, fill_value=0))
    save_table(seg_tbl, TABLES / "n_segments_counts.csv", index=True)

    params = pd.DataFrame([{
        "threshold": VAD_THRESHOLD, "min_speech_ms": VAD_MIN_SPEECH_MS,
        "min_silence_ms": VAD_MIN_SILENCE_MS, "speech_pad_ms": VAD_PAD_MS,
        "sample_rate": SAMPLE_RATE,
    }])
    save_table(params, TABLES / "vad_params.csv")
    return by_label


# ---------------------------------------------------------------- grafikler
def plot_example(row, idx: int):
    """Üstte dalga formu + konuşma bölgeleri, altta konuşma olasılığı."""
    wav = read_audio(ROOT / row.path)
    segs = json.loads(row.segments_sec)
    probs = speech_probs(wav)
    t = np.arange(len(wav)) / SAMPLE_RATE
    tp = (np.arange(len(probs)) * 512 + 256) / SAMPLE_RATE

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    color = COLORS[row.label]
    ax1.plot(t, wav.numpy(), color=color, linewidth=0.5)
    for i, (s, e) in enumerate(segs):
        ax1.axvspan(s, e, color="#4CAF50", alpha=0.25,
                    label="Konuşma (VAD)" if i == 0 else None)
    ax1.set(ylabel="Genlik",
            title=f"{row.label} | {row.duration_sec:.2f} sn → konuşma "
                  f"{row.speech_sec:.2f} sn (%{row.speech_ratio * 100:.0f})")
    if segs:
        ax1.legend(loc="upper right")

    ax2.plot(tp, probs, color="#333333", linewidth=1)
    ax2.axhline(VAD_THRESHOLD, color="#E84855", linestyle="--",
                label=f"Eşik = {VAD_THRESHOLD}")
    ax2.fill_between(tp, 0, probs, where=probs >= VAD_THRESHOLD,
                     color="#4CAF50", alpha=0.3)
    ax2.set(ylim=(0, 1.05), xlabel="Zaman (sn)", ylabel="P(konuşma)")
    ax2.legend(loc="upper right")
    save_fig(fig, FIGURES / f"example_{row.label}_{idx}.png")

    # Dinlemek için önce / sonra
    name = f"{row.label}_{idx}"
    sf.write(EXAMPLES / f"{name}_original.wav", wav.numpy(), SAMPLE_RATE)
    if row.clean_path:
        sf.write(EXAMPLES / f"{name}_vad.wav",
                 read_audio(ROOT / row.clean_path).numpy(), SAMPLE_RATE)


def make_figures(res: pd.DataFrame):
    set_style()
    EXAMPLES.mkdir(parents=True, exist_ok=True)

    # 1) Örnek kayıtlar: her cinsiyetten en çok sessizlik atılanlardan biri + rastgele
    for label in LABELS:
        g = res[(res["label"] == label) & res["has_speech"]]
        picks = pd.concat([g.nsmallest(1, "speech_ratio"),
                           g.sample(min(N_EXAMPLES - 1, len(g)), random_state=SEED)])
        for i, row in enumerate(picks.drop_duplicates("path").itertuples(), 1):
            plot_example(row, i)

    # 2) Konuşma oranı dağılımı
    fig, ax = plt.subplots()
    sns.histplot(data=res, x="speech_ratio", hue="label", palette=COLORS,
                 bins=40, element="step", ax=ax)
    ax.set(title="Konuşma oranı dağılımı (konuşma süresi / toplam süre)",
           xlabel="Konuşma oranı", ylabel="Dosya sayısı")
    save_fig(fig, FIGURES / "speech_ratio_hist.png")

    # 3) VAD öncesi / sonrası süre
    long = res.melt(id_vars="label", value_vars=["duration_sec", "speech_sec"],
                    var_name="stage", value_name="sec")
    long["stage"] = long["stage"].map({"duration_sec": "VAD öncesi",
                                       "speech_sec": "VAD sonrası"})
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=long, x="stage", y="sec", hue="label", palette=COLORS, ax=ax)
    ax.set(title="Kayıt süresi: VAD öncesi ve sonrası", xlabel="", ylabel="Süre (sn)")
    save_fig(fig, FIGURES / "duration_before_after.png")

    # 4) Atılan sessizlik
    fig, ax = plt.subplots()
    sns.histplot(data=res, x="silence_removed_sec", hue="label", palette=COLORS,
                 bins=40, element="step", ax=ax)
    ax.set(title="Kayıt başına atılan sessizlik", xlabel="Süre (sn)",
           ylabel="Dosya sayısı")
    save_fig(fig, FIGURES / "silence_removed_hist.png")

    # 5) Konuşma bölgesi sayısı
    fig, ax = plt.subplots()
    sns.countplot(data=res, x="n_segments", hue="label", palette=COLORS, ax=ax)
    ax.set(title="Kayıt başına konuşma bölgesi sayısı",
           xlabel="Bölge sayısı", ylabel="Dosya sayısı")
    save_fig(fig, FIGURES / "n_segments.png")


# ---------------------------------------------------------------- main
def main():
    df = load_splits()
    print(f"{len(df)} dosya işlenecek. Model yükleniyor...")
    load_model()

    res = process_all(df)
    summary = make_tables(res)
    make_figures(res)

    print("\n========== VAD ÖZETİ ==========")
    cols = ["files", "has_speech", "no_speech", "mean_duration_sec",
            "mean_speech_sec", "mean_speech_ratio", "mean_n_segments"]
    print(summary[cols].to_string())
    print(f"\nToplam ses: {summary.loc['all', 'total_hours_before']:.2f} saat → "
          f"{summary.loc['all', 'total_hours_after']:.2f} saat (VAD sonrası)")
    print("\n✅ VAD tamamlandı.")


if __name__ == "__main__":
    main()