"""F0 (temel frekans) ve formant (F1-F4) çıkarımı - Praat (Parselmouth) ile."""
import numpy as np
import parselmouth

from src.config import F0_MAX, F0_MIN, FORMANT_MAX_HZ, N_FORMANTS, SAMPLE_RATE

TIME_STEP = 0.01  # 10 ms


def _sound(wav, sr=SAMPLE_RATE):
    return parselmouth.Sound(np.asarray(wav, dtype=np.float64), sampling_frequency=sr)


def get_tracks(wav, sr=SAMPLE_RATE):
    """Zaman içinde F0 ve formant eğrileri (grafik ve özet için).
    Dönüş: times, f0 (sessiz/sessiz-harf yerlerinde NaN), formants (N_FORMANTS x T)"""
    snd = _sound(wav, sr)
    pitch = snd.to_pitch_ac(time_step=TIME_STEP, pitch_floor=F0_MIN,
                            pitch_ceiling=F0_MAX)
    times = pitch.xs()
    f0 = pitch.selected_array["frequency"].astype(float)
    f0[f0 == 0] = np.nan  # Praat sessiz yerlerde 0 döndürür

    formant = snd.to_formant_burg(time_step=TIME_STEP, max_number_of_formants=5,
                                  maximum_formant=FORMANT_MAX_HZ)
    fm = np.array([[formant.get_value_at_time(i, t) for t in times]
                   for i in range(1, N_FORMANTS + 1)], dtype=float)
    return times, f0, fm


def extract_acoustic(wav, sr=SAMPLE_RATE) -> dict:
    """Bir kayıt için özet özellikler.
    Formantlar sadece SESLİ (F0 bulunan) çerçevelerden alınır: ünlülerde
    formantlar net, sessiz harflerde/gürültüde güvenilmez."""
    times, f0, fm = get_tracks(wav, sr)
    voiced = ~np.isnan(f0)
    out = {"voiced_ratio": float(voiced.mean()) if len(f0) else 0.0}

    v = f0[voiced]
    if len(v):
        out.update({
            "f0_mean": v.mean(),
            "f0_median": np.median(v),
            "f0_std": v.std(),
            "f0_p5": np.percentile(v, 5),    # min yerine: oktav hatalarına dayanıklı
            "f0_p95": np.percentile(v, 95),  # max yerine
        })
    else:
        out.update({k: np.nan for k in ["f0_mean", "f0_median", "f0_std", "f0_p5", "f0_p95"]})

    for i in range(N_FORMANTS):
        vals = fm[i][voiced]
        vals = vals[~np.isnan(vals)]
        out[f"F{i + 1}"] = np.median(vals) if len(vals) else np.nan

    # Formant dağılımı (dispersion): ardışık formantlar arası ortalama mesafe.
    # Ses yolu uzunluğuyla ters orantılı -> erkeklerde daha küçük.
    out["formant_dispersion"] = (out[f"F{N_FORMANTS}"] - out["F1"]) / (N_FORMANTS - 1)
    return out


ACOUSTIC_COLS = ["f0_mean", "f0_median", "f0_std", "f0_p5", "f0_p95",
                 "voiced_ratio"] + [f"F{i}" for i in range(1, N_FORMANTS + 1)] + \
                ["formant_dispersion"]