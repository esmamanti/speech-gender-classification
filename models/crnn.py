"""
Model 5 - CRNN (log-mel -> CNN + çift yönlü GRU)
  Soru: zaman bilgisini (tonlama, ritim) eklemek CNN'e göre fayda sağlıyor mu?

Çalıştırma (repo ana klasöründen):
    python -m models.crnn
"""
import pandas as pd

from src.config import N_MELS
from src.dataset import MelDataset, load_mels
from src.dl_models import CRNN
from src.evaluate import load_feature_splits
from src.trainer import run_cv

CFG = {"epochs": 30, "patience": 6, "batch_size": 64, "lr": 1e-3}


def main():
    dev, test = load_feature_splits()
    mels = load_mels(pd.concat([dev, test])["mel_path"].unique())

    run_cv(
        name="crnn", display_name="CRNN", variant="log-mel",
        build_model=lambda: CRNN(n_mels=N_MELS),
        make_ds=lambda df, train: MelDataset(df, mels, train=train),
        dev=dev, test=test, cfg=CFG,
    )


if __name__ == "__main__":
    main()