"""Silero VAD yardımcı fonksiyonları (tüm proje bunları kullanır)."""
import io
from functools import lru_cache
from importlib import resources

import numpy as np
import soundfile as sf
import torch
from silero_vad import collect_chunks, get_speech_timestamps

from src.config import (SAMPLE_RATE, VAD_MIN_SILENCE_MS, VAD_MIN_SPEECH_MS,
                        VAD_PAD_MS, VAD_THRESHOLD)

WINDOW = 512  # Silero VAD 16 kHz'de 512 örneklik (32 ms) pencerelerle çalışır


@lru_cache(maxsize=1)
def load_model():
    """Modeli bir kez yükler, sonra hep aynısını döndürür."""
    torch.set_num_threads(1)  # Silero VAD tek çekirdekte en verimli
    # Not: torch.jit.load'a dosya YOLU verince Windows'ta yolda Türkçe karakter
    # (ör. "Mantı") varsa dosyayı açamıyor. Bu yüzden dosyayı Python ile okuyup
    # byte olarak veriyoruz -> yoldaki karakterlerden etkilenmez.
    model_file = resources.files("silero_vad.data").joinpath("silero_vad.jit")
    model = torch.jit.load(io.BytesIO(model_file.read_bytes()), map_location="cpu")
    model.eval()
    return model


def read_audio(path) -> torch.Tensor:
    """Sesi 16 kHz mono float32 tensör olarak okur."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)  # stereo ise mono'ya çevir
    if sr != SAMPLE_RATE:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    return torch.from_numpy(np.ascontiguousarray(wav))


def get_segments(wav: torch.Tensor) -> list[dict]:
    """Konuşma bölgelerini örnek (sample) cinsinden döndürür:
    [{'start': 5120, 'end': 30208}, ...]"""
    return get_speech_timestamps(
        wav, load_model(),
        threshold=VAD_THRESHOLD,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_PAD_MS,
    )


def keep_speech(wav: torch.Tensor, segments: list[dict]) -> torch.Tensor:
    """Sadece konuşma bölgelerini birleştirip döndürür."""
    if not segments:
        return torch.zeros(0)
    return collect_chunks(segments, wav)


def speech_probs(wav: torch.Tensor) -> np.ndarray:
    """Her 32 ms'lik pencere için konuşma olasılığı (grafik çizmek için)."""
    model = load_model()
    model.reset_states()
    probs = []
    with torch.no_grad():
        for i in range(0, len(wav), WINDOW):
            chunk = wav[i:i + WINDOW]
            if len(chunk) < WINDOW:
                chunk = torch.nn.functional.pad(chunk, (0, WINDOW - len(chunk)))
            probs.append(model(chunk, SAMPLE_RATE).item())
    model.reset_states()
    return np.array(probs)


def segments_to_seconds(segments: list[dict]) -> list[tuple]:
    return [(round(s["start"] / SAMPLE_RATE, 3), round(s["end"] / SAMPLE_RATE, 3))
            for s in segments]