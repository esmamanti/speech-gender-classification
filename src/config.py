"""Projedeki tüm yollar ve ayarlar tek yerde."""
from pathlib import Path

# ---------- Yollar ----------
ROOT = Path(__file__).resolve().parents[1]      # repo ana klasörü
DATA_RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"

OUT_DATA = OUTPUTS / "data"
OUT_VAD = OUTPUTS / "vad"
OUT_FEATURES = OUTPUTS / "features"
OUT_MODELS = OUTPUTS / "models"
OUT_RESULTS = OUTPUTS / "results"

# ---------- Genel ----------
SEED = 42
LABELS = ["male", "female"]
LABEL2ID = {"male": 0, "female": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ---------- Veri ayırma ----------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
N_FOLDS = 5                # k-fold cross validation (train+val üzerinde)

# ---------- Ses ----------
SAMPLE_RATE = 16000        # Silero VAD 16 kHz istiyor
CLIP_SECONDS = 3.0         # CNN/CRNN için sabit uzunluk

# ---------- Silero VAD ----------
VAD_THRESHOLD = 0.5        # konuşma olasılık eşiği
VAD_MIN_SPEECH_MS = 250    # bundan kısa konuşma parçalarını yok say
VAD_MIN_SILENCE_MS = 100   # bundan kısa sessizlikleri bölme
VAD_PAD_MS = 30            # konuşma bölgelerinin kenarına eklenen pay

# ---------- F0 / Formant (Praat) ----------
F0_MIN = 75                # Hz
F0_MAX = 500               # Hz
FORMANT_MAX_HZ = 5500      # cinsiyet bilinmediği için herkese aynı tavan
N_FORMANTS = 4             # F1-F4

# ---------- Log-mel ----------
N_MELS = 64
N_FFT = 400                # 25 ms pencere
HOP_LENGTH = 160           # 10 ms adım

# ---------- Grafik ----------
COLORS = {"male": "#2E86AB", "female": "#E84855"}
FIG_DPI = 300


def out_dir(*parts) -> Path:
    """outputs altında klasör döndürür, yoksa oluşturur."""
    p = OUTPUTS.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p