"""
Aşama 2 - Veri hazırlığı
1) Ses dosyalarını listele (etiket = klasör adı)
2) Bozuk / okunamayan dosyaları ayıkla
3) Tekrarlı dosyaları sil (ses içeriğine göre)
4) Sınıfları dengele (erkekleri kadın sayısına indir)
5) Train / val / test olarak ayır (stratified)
5b) Test hariç veriyi 5-fold'a böl (cross validation)
6) Tablo + grafik üret

Çalıştırma (repo ana klasöründen):
    python -m scripts.prepare_data
"""
import hashlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import soundfile as sf
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from src.config import (COLORS, DATA_RAW, LABELS, N_FOLDS, OUT_DATA, ROOT,
                        SEED, TEST_RATIO, TRAIN_RATIO, VAL_RATIO)
from src.plots import save_fig, save_table, set_style

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
TABLES = OUT_DATA / "tables"
FIGURES = OUT_DATA / "figures"


# ---------------------------------------------------------------- 1) listele
def list_files() -> pd.DataFrame:
    rows = []
    for label in LABELS:
        # data/raw/**/male/*.wav  (alt klasör adı ne olursa olsun bulur)
        for folder in DATA_RAW.rglob(label):
            if not folder.is_dir():
                continue
            for f in sorted(folder.iterdir()):
                if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                    rows.append({"abs_path": f, "label": label})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"❌ {DATA_RAW} içinde male/female klasörü bulunamadı.")
    # CSV'ye repo köküne göre göreli yol yazıyoruz (taşınabilir olsun)
    df["path"] = df["abs_path"].apply(lambda p: p.relative_to(ROOT).as_posix())
    return df


# ---------------------------------------------------------------- 2) bilgi + hash
def file_md5(path, chunk=1 << 20) -> str:
    """Dosyanın byte'larının hash'i: birebir kopyaları yakalar."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def audio_md5(path) -> str:
    """Ses örneklerinin hash'i: dosya başlığı/metadata farklı olsa bile
    içindeki ses aynıysa yakalar."""
    data, _ = sf.read(str(path), dtype="int16", always_2d=True)
    return hashlib.md5(data.tobytes()).hexdigest()


def add_info(df: pd.DataFrame) -> pd.DataFrame:
    durations, srs, channels, hashes, ahashes, ok = [], [], [], [], [], []
    for p in tqdm(df["abs_path"], desc="Dosyalar okunuyor"):
        try:
            info = sf.info(str(p))
            durations.append(info.duration)
            srs.append(info.samplerate)
            channels.append(info.channels)
            ahashes.append(audio_md5(p))
            ok.append(info.frames > 0)
        except Exception:
            durations.append(None); srs.append(None); channels.append(None)
            ahashes.append(None); ok.append(False)
        hashes.append(file_md5(p))
    df = df.copy()
    df["duration_sec"] = durations
    df["sample_rate"] = srs
    df["channels"] = channels
    df["md5"] = hashes
    df["audio_md5"] = ahashes
    df["readable"] = ok
    return df


# ---------------------------------------------------------------- 3) tekrarlar
def remove_duplicates(df: pd.DataFrame, key: str = "audio_md5"):
    """key='audio_md5': ses içeriği aynı olan dosyaları tekrar sayar
    (birebir aynı dosyalar da bunun içinde)."""
    # Aynı ses hem male hem female klasöründe varsa: etiket çelişkili -> hepsini at
    labels_per_hash = df.groupby(key)["label"].nunique()
    conflict_hashes = set(labels_per_hash[labels_per_hash > 1].index)
    conflict = df[df[key].isin(conflict_hashes)].copy()
    conflict["reason"] = "label_conflict"

    rest = df[~df[key].isin(conflict_hashes)]
    dup_mask = rest.duplicated(subset=key, keep="first")
    dups = rest[dup_mask].copy()
    dups["reason"] = "duplicate"

    removed = pd.concat([conflict, dups], ignore_index=True)
    clean = rest[~dup_mask].reset_index(drop=True)
    return clean, removed


# ---------------------------------------------------------------- 4) dengeleme
def balance(df: pd.DataFrame) -> pd.DataFrame:
    n_min = df["label"].value_counts().min()
    parts = [g.sample(n=n_min, random_state=SEED) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


# ---------------------------------------------------------------- 5) ayırma
def split(df: pd.DataFrame) -> pd.DataFrame:
    train, temp = train_test_split(df, train_size=TRAIN_RATIO,
                                   stratify=df["label"], random_state=SEED)
    val_share = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val, test = train_test_split(temp, train_size=val_share,
                                 stratify=temp["label"], random_state=SEED)
    out = pd.concat([train.assign(split="train"),
                     val.assign(split="val"),
                     test.assign(split="test")])
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- 5b) k-fold
def make_folds(final: pd.DataFrame) -> pd.DataFrame:
    """Test seti hariç (train + val) veriyi N_FOLDS parçaya böler.
    Tüm modeller aynı fold'ları kullanır -> adil karşılaştırma."""
    dev = final[final["split"].isin(["train", "val"])].reset_index(drop=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    dev["fold"] = -1
    for k, (_, idx) in enumerate(skf.split(dev, dev["label"])):
        dev.loc[idx, "fold"] = k
    return dev


# ---------------------------------------------------------------- 6) tablolar
def counts(df, name):
    c = df["label"].value_counts().reindex(LABELS, fill_value=0)
    return {"step": name, "male": int(c["male"]), "female": int(c["female"]),
            "total": int(c.sum())}


def make_figures(steps: pd.DataFrame, final: pd.DataFrame, clean: pd.DataFrame):
    set_style()

    # Sınıf dağılımı: her adımda erkek / kadın sayısı
    long = steps.melt(id_vars="step", value_vars=LABELS,
                      var_name="label", value_name="count")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=long, x="step", y="count", hue="label",
                palette=COLORS, ax=ax)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%d", fontsize=9)
    ax.set(title="Sınıf dağılımı (veri hazırlık adımları)",
           xlabel="", ylabel="Dosya sayısı")
    save_fig(fig, FIGURES / "class_distribution.png")

    # Train / val / test dağılımı
    sc = final.groupby(["split", "label"]).size().reset_index(name="count")
    sc["split"] = pd.Categorical(sc["split"], ["train", "val", "test"])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=sc, x="split", y="count", hue="label",
                palette=COLORS, ax=ax)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%d", fontsize=9)
    ax.set(title="Train / Val / Test dağılımı", xlabel="", ylabel="Dosya sayısı")
    save_fig(fig, FIGURES / "split_distribution.png")

    # Süre dağılımı (temizlenmiş, dengelenmemiş veri)
    fig, ax = plt.subplots()
    sns.histplot(data=clean, x="duration_sec", hue="label", palette=COLORS,
                 bins=50, element="step", stat="count", common_norm=False, ax=ax)
    ax.set(title="Kayıt süresi dağılımı", xlabel="Süre (sn)", ylabel="Dosya sayısı")
    save_fig(fig, FIGURES / "duration_hist.png")

    # Süre boxplot (son veri)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=final, x="label", y="duration_sec", hue="label",
                palette=COLORS, legend=False, ax=ax)
    ax.set(title="Kayıt süresi (dengelenmiş veri)", xlabel="", ylabel="Süre (sn)")
    save_fig(fig, FIGURES / "duration_boxplot.png")


# ---------------------------------------------------------------- main
def main():
    print("1) Dosyalar listeleniyor...")
    df = list_files()
    print(f"   {len(df)} dosya bulundu.")

    print("2) Süre / sample rate / hash hesaplanıyor...")
    df = add_info(df)
    unreadable = df[~df["readable"]].copy()
    df = df[df["readable"]].reset_index(drop=True)

    print("3) Tekrarlı dosyalar siliniyor...")
    clean, removed = remove_duplicates(df)

    print("4) Sınıflar dengeleniyor...")
    balanced = balance(clean)

    print("5) Train / val / test ayrılıyor...")
    final = split(balanced)

    # ---------- CSV'ler (sonraki adımlar bunları kullanır)
    keep = ["path", "label", "duration_sec", "sample_rate", "channels", "md5", "audio_md5"]
    save_table(df[keep], OUT_DATA / "all_files.csv")
    for s in ["train", "val", "test"]:
        save_table(final[final["split"] == s][keep], OUT_DATA / f"{s}.csv")

    # k-fold için: test hariç veri + fold numarası
    folds = make_folds(final)
    save_table(folds[["path", "label", "fold"]], OUT_DATA / "cv_folds.csv")

    # ---------- Tablolar
    steps = pd.DataFrame([
        counts(pd.concat([df, unreadable]), "Ham veri"),
        counts(df, "Okunabilir"),
        counts(clean, "Tekrarsız"),
        counts(balanced, "Dengelenmiş"),
    ])
    save_table(steps, TABLES / "summary.csv")

    split_tbl = (final.groupby(["split", "label"]).size().unstack(fill_value=0)
                 .reindex(["train", "val", "test"])[LABELS])
    split_tbl["total"] = split_tbl.sum(axis=1)
    split_tbl.loc["total"] = split_tbl.sum()
    save_table(split_tbl, TABLES / "split_counts.csv", index=True)

    dur_tbl = (final.groupby("label")["duration_sec"]
               .describe().round(2).reindex(LABELS))
    save_table(dur_tbl, TABLES / "duration_stats.csv", index=True)

    fold_tbl = (folds.groupby(["fold", "label"]).size().unstack(fill_value=0)[LABELS])
    fold_tbl["total"] = fold_tbl.sum(axis=1)
    save_table(fold_tbl, TABLES / "fold_counts.csv", index=True)

    sr_tbl = (df.groupby(["sample_rate", "channels"]).size()
              .reset_index(name="count"))
    save_table(sr_tbl, TABLES / "sample_rates.csv")

    save_table(removed[["path", "label", "audio_md5", "reason"]], TABLES / "removed_duplicates.csv")

    # Tekrar analizi: byte bazlı vs ses içeriği bazlı
    dup_tbl = pd.DataFrame([
        {"method": "byte (md5)",
         "duplicate_files": int(df.duplicated("md5").sum())},
        {"method": "audio content (audio_md5)",
         "duplicate_files": int(df.duplicated("audio_md5").sum())},
    ])
    save_table(dup_tbl, TABLES / "duplicate_check.csv")
    save_table(unreadable[["path", "label"]], TABLES / "unreadable_files.csv")

    # ---------- Grafikler
    make_figures(steps, final, clean)

    # ---------- Ekrana özet
    print("\n========== ÖZET ==========")
    print(steps.to_string(index=False))
    print("\nTekrar kontrolü:")
    print(dup_tbl.to_string(index=False))
    print(f"\nSilinen tekrar: {(removed.reason == 'duplicate').sum()}  |  "
          f"Etiket çelişkisi: {(removed.reason == 'label_conflict').sum()}  |  "
          f"Okunamayan: {len(unreadable)}")
    print("\nSplit:")
    print(split_tbl.to_string())
    print(f"\n{N_FOLDS}-fold (test hariç):")
    print(fold_tbl.to_string())
    print("\nSample rate / kanal:")
    print(sr_tbl.to_string(index=False))
    print("\n✅ Veri hazırlığı tamamlandı.")


if __name__ == "__main__":
    main()