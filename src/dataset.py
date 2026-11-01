"""PyTorch Dataset'leri: akustik özellik tablosu (MLP) ve log-mel (CNN/CRNN)."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.config import CLIP_SECONDS, HOP_LENGTH, ROOT, SAMPLE_RATE

N_FRAMES = int(CLIP_SECONDS * SAMPLE_RATE / HOP_LENGTH)   # 3 sn -> 300 çerçeve
DB_MIN = -80.0                                            # log-mel'deki en düşük değer (sessizlik)


class TabularDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cols):
        self.X = torch.tensor(df[cols].values, dtype=torch.float32)
        self.y = torch.tensor(df["y"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def load_mels(paths) -> dict:
    """Tüm log-mel'leri bir kez belleğe yükler (float16, yaklaşık 400 MB)."""
    return {p: np.load(ROOT / p) for p in tqdm(paths, desc="Log-mel yükleniyor")}


def fix_length(mel: np.ndarray, n: int, random_crop: bool) -> np.ndarray:
    """Spektrogramı n çerçeveye sabitler: uzunsa kes, kısaysa sessizlikle doldur."""
    t = mel.shape[1]
    if t > n:
        start = np.random.randint(0, t - n + 1) if random_crop else (t - n) // 2
        return mel[:, start:start + n]
    if t < n:
        pad = n - t
        left = np.random.randint(0, pad + 1) if random_crop else pad // 2
        return np.pad(mel, ((0, 0), (left, pad - left)), constant_values=DB_MIN)
    return mel


class MelDataset(Dataset):
    """Log-mel -> (1, n_mels, N_FRAMES) tensör.
    train=True iken: rastgele kesit + SpecAugment (frekans/zaman maskeleme).
    Böylece model ezberlemek yerine genel desenleri öğrenir."""

    def __init__(self, df: pd.DataFrame, mels: dict, train: bool,
                 n_frames=N_FRAMES, freq_mask=8, time_mask=30):
        self.paths = df["mel_path"].tolist()
        self.y = torch.tensor(df["y"].values, dtype=torch.float32)
        self.mels, self.train, self.n = mels, train, n_frames
        self.freq_mask, self.time_mask = freq_mask, time_mask

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        m = fix_length(self.mels[self.paths[i]].astype(np.float32), self.n, self.train)
        m = (m - DB_MIN / 2) / (-DB_MIN / 2)          # [-80, 0] dB -> [-1, 1]
        if self.train:
            m = m.copy()
            f = np.random.randint(0, self.freq_mask + 1)
            f0 = np.random.randint(0, m.shape[0] - f + 1)
            m[f0:f0 + f, :] = -1.0
            t = np.random.randint(0, self.time_mask + 1)
            t0 = np.random.randint(0, m.shape[1] - t + 1)
            m[:, t0:t0 + t] = -1.0
        return torch.from_numpy(m).unsqueeze(0), self.y[i]