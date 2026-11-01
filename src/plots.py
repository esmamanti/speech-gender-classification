"""Tüm grafik ve tablolar için ortak stil ve kaydetme fonksiyonları."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")      # script'lerde pencere açmadan dosyaya kaydet
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import COLORS, FIG_DPI


def set_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "savefig.bbox": "tight",
    })


def palette():
    """Erkek/kadın için sabit renkler (seaborn'a verilir)."""
    return COLORS


def save_fig(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"📊 Grafik kaydedildi: {path}")


def save_table(df: pd.DataFrame, path: Path, index: bool = False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: Excel'de açınca Türkçe karakterler bozulmasın
    df.to_csv(path, index=index, encoding="utf-8-sig")
    print(f"📋 Tablo kaydedildi: {path}")