"""
Model 4 - CNN (log-mel spektrogram)
  Model, özellikleri (perde, formant...) spektrogramdan KENDİSİ öğrenir.

Çalıştırma (repo ana klasöründen):
    python -m models.cnn
"""
import pandas as pd

from src.dataset import MelDataset, load_mels
from src.dl_models import CNN
from src.evaluate import load_feature_splits
from src.trainer import run_cv

CFG = {"epochs": 30, "patience": 6, "batch_size": 64, "lr": 1e-3}


def main():
    dev, test = load_feature_splits()
    mels = load_mels(pd.concat([dev, test])["mel_path"].unique())

    run_cv(
        name="cnn", display_name="CNN", variant="log-mel",
        build_model=lambda: CNN(),
        make_ds=lambda df, train: MelDataset(df, mels, train=train),
        dev=dev, test=test, cfg=CFG,
    )


if __name__ == "__main__":
    main()