"""Ses okuma + log-mel spektrogram (CNN / CRNN girdisi).
Not: Bu dosya bilerek torch import ETMEZ. Paralel işlemlerde her işlem
torch'u (GPU DLL'leriyle birlikte) yüklerse Windows'ta bellek yetmiyor."""
import librosa
import numpy as np
import soundfile as sf

from src.config import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE


def load_wav(path, sr=SAMPLE_RATE) -> np.ndarray:
    """Sesi 16 kHz mono float32 numpy dizisi olarak okur (torch'suz)."""
    wav, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if file_sr != sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    return wav


def log_mel(wav, sr=SAMPLE_RATE) -> np.ndarray:
    """(N_MELS, T) boyutunda log-mel spektrogram (dB).
    ref=np.max: her kaydı kendi en yüksek değerine göre ölçekler ->
    ses yüksekliği (mikrofona yakınlık) farkları modeli yanıltmaz."""
    mel = librosa.feature.melspectrogram(
        y=np.asarray(wav, dtype=np.float32), sr=sr, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=0, fmax=sr // 2)
    return librosa.power_to_db(mel, ref=np.max, top_db=80.0).astype(np.float32)