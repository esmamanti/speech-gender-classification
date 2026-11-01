"""Dosya adından konuşmacı çıkarma (CMU ARCTIC).
  arctic_a0001(3).wav  ->  cümle: arctic_a0001,  kopya no: 3
Aynı klasördeki (1), (2)... ekleri farklı konuşmacılara karşılık geliyor
(scripts/speaker_check.py ile doğrulandı: her grup ~1132 / 593 kayıt, tutarlı F0)."""
import re

import pandas as pd

PATTERN = re.compile(r"^(arctic_[ab]\d{4})(?:\((\d+)\))?\.wav$", re.IGNORECASE)


def parse_name(path: str):
    m = PATTERN.match(str(path).split("/")[-1])
    if not m:
        return None, None
    return m.group(1), int(m.group(2) or 0)


def add_speaker(df: pd.DataFrame) -> pd.DataFrame:
    """'prompt' ve 'speaker' sütunlarını ekler. Ad kalıbına uymayanların speaker'ı None."""
    df = df.copy()
    parsed = df["path"].apply(parse_name)
    df["prompt"] = [p for p, _ in parsed]
    df["speaker"] = [f"{lab}_{c}" if c is not None else None
                     for lab, (_, c) in zip(df["label"], parsed)]
    return df