"""
Konuşmacı kontrolü (CMU ARCTIC)
Dosya adı:  arctic_a0001(3).wav  ->  cümle = arctic_a0001,  kopya no = 3
Tahmin: aynı klasördeki (1), (2)... ekleri FARKLI KONUŞMACILARI gösteriyor.
Doğrulama: her grupta ~1132 kayıt olmalı ve grubun F0'ı tutarlı olmalı.

Çalıştırma:
    python -m scripts.speaker_check
"""
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import COLORS, LABELS, OUT_DATA, OUT_FEATURES
from src.plots import save_fig, save_table, set_style

PATTERN = re.compile(r"^(arctic_[ab]\d{4})(?:\((\d+)\))?\.wav$", re.IGNORECASE)
TABLES = OUT_DATA / "tables"
FIGURES = OUT_DATA / "figures"


def parse(path: str):
    name = path.split("/")[-1]
    m = PATTERN.match(name)
    if not m:
        return pd.Series({"prompt": None, "copy": None})
    return pd.Series({"prompt": m.group(1), "copy": int(m.group(2) or 0)})


def main():
    allf = pd.read_csv(OUT_DATA / "all_files.csv")
    allf[["prompt", "copy"]] = allf["path"].apply(parse)
    print(f"Ad kalıbına uymayan dosya: {allf['prompt'].isna().sum()} / {len(allf)}")
    allf["speaker"] = allf["label"] + "_" + allf["copy"].astype("Int64").astype(str)

    # Grup başına kayıt sayısı (tüm ham veri)
    counts = (allf.groupby(["label", "speaker"]).agg(
        files=("path", "size"), unique_prompts=("prompt", "nunique"))
        .reset_index().sort_values(["label", "speaker"]))

    # Grup başına F0 (özellik çıkarılan, dengelenmiş veri)
    feats = pd.read_csv(OUT_FEATURES / "acoustic_features.csv")
    feats[["prompt", "copy"]] = feats["path"].apply(parse)
    feats["speaker"] = feats["label"] + "_" + feats["copy"].astype("Int64").astype(str)
    f0 = feats.groupby("speaker").agg(
        used_files=("path", "size"),
        f0_median_mean=("f0_median", "mean"),
        f0_median_std=("f0_median", "std"),
        train=("split", lambda s: (s == "train").sum()),
        val=("split", lambda s: (s == "val").sum()),
        test=("split", lambda s: (s == "test").sum()),
    ).round(1).reset_index()
    tbl = counts.merge(f0, on="speaker", how="left")
    save_table(tbl, TABLES / "speaker_check.csv")

    # Aynı cümle kaç farklı grupta var?
    per_prompt = allf.groupby("prompt")["speaker"].nunique().describe().round(1)

    set_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    order = tbl.sort_values("f0_median_mean")["speaker"]
    sns.boxplot(data=feats, x="speaker", y="f0_median", hue="label", order=order,
                hue_order=LABELS, palette=COLORS, showfliers=False, dodge=False, ax=ax)
    ax.set(title="Gruplara (olası konuşmacılara) göre F0", xlabel="Grup (etiket_kopyaNo)",
           ylabel="F0 medyan (Hz)")
    ax.tick_params(axis="x", rotation=45)
    save_fig(fig, FIGURES / "speaker_f0.png")

    print("\n========== KONUŞMACI KONTROLÜ ==========")
    print(tbl.to_string(index=False))
    print("\nBir cümle kaç farklı grupta geçiyor:")
    print(per_prompt.to_string())


if __name__ == "__main__":
    main()