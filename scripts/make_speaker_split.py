"""
Konuşmacıya göre bölme (speaker-independent split)
  - Her cinsiyetten N_TEST_SPEAKERS kişi TEST konuşmacısı olarak ayrılır
  - Kalan konuşmacılar 5 fold'a dağıtılır (her fold'da her cinsiyetten en az 1 kişi)
  - Böylece doğrulama/test konuşmacıları eğitimde HİÇ görülmez
  - Her parça (test ve her fold) kendi içinde erkek/kadın sayısı eşitlenir

Not: Özellik dosyası (acoustic_features.csv) zaten var; VAD ve özellik
çıkarımı tekrar yapılmaz, sadece hangi kaydın nerede kullanılacağı değişir.

Çalıştırma (repo ana klasöründen):
    python -m scripts.make_speaker_split
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (COLORS, LABELS, N_FOLDS, N_TEST_SPEAKERS, OUT_DATA,
                        OUT_FEATURES, SEED)
from src.plots import save_fig, save_table, set_style
from src.speakers import add_speaker

OUT = OUT_DATA / "speaker_split"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"


def assign_roles(spk: pd.DataFrame) -> pd.DataFrame:
    """Her cinsiyet için konuşmacıları karıştır: ilk N'i test, kalanlar fold 0..4."""
    rng = np.random.default_rng(SEED)
    rows = []
    for label in LABELS:
        names = spk.loc[spk["label"] == label, "speaker"].tolist()
        rng.shuffle(names)
        n_test = N_TEST_SPEAKERS[label]
        dev = names[n_test:]
        if len(dev) < N_FOLDS:
            raise SystemExit(f"❌ {label}: {len(dev)} dev konuşmacısı var, {N_FOLDS} fold için az.")
        for s in names[:n_test]:
            rows.append({"speaker": s, "role": "test", "fold": -1})
        for i, s in enumerate(dev):
            rows.append({"speaker": s, "role": "dev", "fold": i % N_FOLDS})
    return spk.merge(pd.DataFrame(rows), on="speaker")


def balance(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Parça içinde erkek/kadın sayısını eşitler (fazla olan sınıftan rastgele seçer)."""
    n = df["label"].value_counts().min()
    return pd.concat([g.sample(n=n, random_state=seed) for _, g in df.groupby("label")])


def main():
    feats = add_speaker(pd.read_csv(OUT_FEATURES / "acoustic_features.csv"))
    unknown = feats["speaker"].isna().sum()
    feats = feats[feats["speaker"].notna()].copy()
    print(f"Konuşmacısı bilinmeyen (ad kalıbına uymayan) kayıt çıkarıldı: {unknown}")

    spk = (feats.groupby(["speaker", "label"])
                .agg(files=("path", "size"), f0_median=("f0_median", "median"))
                .reset_index())
    spk = assign_roles(spk)
    feats = feats.merge(spk[["speaker", "role", "fold"]], on="speaker")

    # ---------- Test: kendi içinde dengele
    test = balance(feats[feats["role"] == "test"], SEED)
    # ---------- Dev: her fold kendi içinde dengelenir
    dev = pd.concat([balance(g, SEED + int(k))
                     for k, g in feats[feats["role"] == "dev"].groupby("fold")])

    OUT.mkdir(parents=True, exist_ok=True)
    save_table(test[["path", "label", "speaker"]], OUT / "test.csv")
    save_table(dev[["path", "label", "speaker", "fold"]], OUT / "cv_folds.csv")

    # ---------- Tablolar
    used = pd.concat([test, dev]).groupby("speaker").size().rename("files_used")
    spk = spk.merge(used, on="speaker", how="left").fillna({"files_used": 0})
    spk["files_used"] = spk["files_used"].astype(int)
    spk = spk.sort_values(["role", "fold", "label", "speaker"])
    save_table(spk.round(1), TABLES / "speaker_assignment.csv")

    part = pd.concat([test.assign(part="test"),
                      dev.assign(part=lambda d: "fold " + d["fold"].astype(str))])
    part_tbl = (part.groupby(["part", "label"]).agg(files=("path", "size"),
                                                    speakers=("speaker", "nunique"))
                .unstack(fill_value=0))
    part_tbl.columns = [f"{a}_{b}" for a, b in part_tbl.columns]
    save_table(part_tbl, TABLES / "split_counts.csv", index=True)

    # Kontrol: test konuşmacısı dev'de var mı? fold'lar arası çakışma var mı?
    leak_test = set(test["speaker"]) & set(dev["speaker"])
    leak_fold = dev.groupby("speaker")["fold"].nunique().max()
    assert not leak_test and leak_fold == 1, "Konuşmacı sızıntısı var!"

    # ---------- Grafik: konuşmacılar, F0'ları ve rolleri
    set_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    s = spk.sort_values("f0_median").reset_index(drop=True)
    bars = ax.bar(s["speaker"], s["f0_median"], color=[COLORS[l] for l in s["label"]],
                  edgecolor=["black" if r == "test" else "none" for r in s["role"]],
                  linewidth=2.5)
    for b, (_, r) in zip(bars, s.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 3,
                "TEST" if r["role"] == "test" else f"f{r['fold']}",
                ha="center", fontsize=9, fontweight="bold" if r["role"] == "test" else None)
    ax.set(title="Konuşmacı bölmesi (siyah çerçeve = test konuşmacısı, fN = fold)",
           xlabel="Konuşmacı", ylabel="F0 medyan (Hz)")
    ax.tick_params(axis="x", rotation=45)
    save_fig(fig, FIGURES / "speaker_split.png")

    print("\n========== KONUŞMACI BÖLMESİ ==========")
    print(spk[["speaker", "label", "f0_median", "role", "fold", "files_used"]]
          .round(1).to_string(index=False))
    print("\nParça başına kayıt / konuşmacı:")
    print(part_tbl.to_string())
    print("\n✔ Kontrol: test konuşmacıları eğitimde yok, her konuşmacı tek bir fold'da.")
    print("✅ Konuşmacı bölmesi hazır. Modelleri bu bölmeyle çalıştırmak için:")
    print('   $env:SPLIT_MODE="speaker"')


if __name__ == "__main__":
    main()