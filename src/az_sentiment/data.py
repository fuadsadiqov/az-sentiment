"""Data: yükləmə, normallaşdırma, label xəritəsi, dublikat təmizləmə, split."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED = DATA_DIR / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

HF_DATASET = "hajili/azerbaijani_review_sentiment_classification"
LABELS = ["negative", "neutral", "positive"]
SEED = 42

# Azərbaycan əlifbasına xas hərflər: mətnin "diakritikalı" yazılıb-yazılmadığını göstərir
AZ_CHARS = set("əığöüşçƏIĞÖÜŞÇ")


def score_to_label(score: int) -> int:
    """1-2 ulduz -> negative(0), 3 -> neutral(1), 4-5 -> positive(2)."""
    if score <= 2:
        return 0
    if score == 3:
        return 1
    return 2


def normalize(text: str) -> str:
    """Dublikat axtarışı üçün: kiçik hərf, artıq boşluqlar silinir."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    ds = load_dataset(HF_DATASET)
    tr = ds["train"].to_pandas()
    te = ds["test"].to_pandas()
    # pandas 3-də astype(str) NaN-ı saxlayır, ona görə boş rəyləri əvvəlcədən atırıq
    tr = tr.dropna(subset=["content"]).copy()
    te = te.dropna(subset=["content"]).copy()
    for df in (tr, te):
        df["text"] = df["content"].astype(str)
        df["label"] = df["score"].map(score_to_label)
    return tr[["text", "score", "label"]], te[["text", "score", "label"]]


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Eyni (normallaşdırılmış) mətni bir dəfə saxla.
    Fərqli label-lı dublikatlarda çoxluq label-ı (mode) götürülür."""
    df = df.copy()
    df["norm"] = df["text"].map(normalize)
    df = df[df["norm"].str.len() > 0]
    agg = df.groupby("norm").agg(
        text=("text", "first"),
        label=("label", lambda s: s.mode().iloc[0]),
        score=("score", lambda s: s.mode().iloc[0]),
        n_copies=("text", "size"),
        n_labels=("label", "nunique"),
    ).reset_index()
    return agg


def build_clean_splits() -> dict[str, pd.DataFrame]:
    """Orijinal train+test birləşir, dublikat silinir, stratified 80/10/10 bölünür.
    Beləliklə eyni mətn həm train, həm test-də ola bilməz (leakage yoxdur)."""
    tr, te = load_raw()
    full = dedup(pd.concat([tr, te], ignore_index=True))
    full = full.sample(frac=1, random_state=SEED).reset_index(drop=True)
    full.insert(0, "id", range(len(full)))
    train, rest = train_test_split(full, test_size=0.2, stratify=full["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=SEED)
    return {"train": train, "val": val, "test": test}


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} yoxdur. Əvvəlcə: uv run scripts/01_prepare_data.py")
    return pd.read_parquet(path)


def has_az_chars(text: str) -> bool:
    return any(c in AZ_CHARS for c in text)
