"""
Model 3 - MLP (akustik özellikler -> sinir ağı)
  Random Forest ile AYNI özellikler (F0 + formant). Soru: aynı bilgiyle
  sinir ağı, klasik ML'den daha iyi mi?

Çalıştırma (repo ana klasöründen):
    python -m models.mlp
"""
from src.dataset import TabularDataset
from src.dl_models import MLP
from src.evaluate import load_feature_splits
from src.ml_models import FEATURE_SETS
from src.trainer import run_cv

COLS = FEATURE_SETS["F0 + Formant"]
CFG = {"epochs": 100, "patience": 12, "batch_size": 128, "lr": 1e-3}


def main():
    dev, test = load_feature_splits()
    # Nadiren eksik değer olursa train medyanıyla doldur
    med = dev[COLS].median()
    dev[COLS], test[COLS] = dev[COLS].fillna(med), test[COLS].fillna(med)

    run_cv(
        name="mlp", display_name="MLP", variant="F0 + Formant",
        build_model=lambda: MLP(n_in=len(COLS)),
        make_ds=lambda df, train: TabularDataset(df, COLS),
        # Normalizasyon SADECE o fold'un eğitim verisinden hesaplanır (sızıntı yok)
        on_fold_start=lambda model, tr: model.set_normalization(tr[COLS].values),
        dev=dev, test=test, cfg=CFG,
    )


if __name__ == "__main__":
    main()