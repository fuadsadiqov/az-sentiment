"""Addım 1: datanı yüklə, təhlil et (EDA), təmiz split-ləri data/processed/ altına yaz.

uv run scripts/01_prepare_data.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from az_sentiment.data import LABELS, PROCESSED, RESULTS, build_clean_splits, has_az_chars, load_raw, normalize

PROCESSED.mkdir(parents=True, exist_ok=True)
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)

# --- Xam data statistikası (leakage sübutu) ---
tr, te = load_raw()
train_norm = set(tr["text"].map(normalize))
leak = te["text"].map(normalize).isin(train_norm).mean()
all_raw = pd.concat([tr, te])
stats = {
    "raw_train": len(tr),
    "raw_test": len(te),
    "raw_test_in_train_pct": round(float(leak) * 100, 1),
    "raw_unique_texts": int(all_raw["text"].map(normalize).nunique()),
    "raw_score_dist": all_raw["score"].value_counts(normalize=True).sort_index().round(4).to_dict(),
    "top_texts": all_raw["text"].map(normalize).value_counts().head(15).to_dict(),
}

# --- Təmiz split ---
splits = build_clean_splits()
for name, df in splits.items():
    df.to_parquet(PROCESSED / f"{name}.parquet", index=False)

# Orijinal (naiv) split-i də saxla: leakage təcrübəsi üçün
tr.assign(id=range(len(tr))).to_parquet(PROCESSED / "naive_train.parquet", index=False)
te.assign(id=range(len(te))).to_parquet(PROCESSED / "naive_test.parquet", index=False)

full = pd.concat(splits.values())
full["n_words"] = full["text"].str.split().str.len()
full["az_chars"] = full["text"].map(has_az_chars)
stats.update({
    "clean_total": len(full),
    "clean_sizes": {k: len(v) for k, v in splits.items()},
    "clean_label_dist": {LABELS[k]: round(v, 4) for k, v in
                         full["label"].value_counts(normalize=True).sort_index().items()},
    "conflicting_label_texts": int((full["n_labels"] > 1).sum()),
    "words_median": float(full["n_words"].median()),
    "words_p95": float(full["n_words"].quantile(0.95)),
    "az_chars_pct": round(float(full["az_chars"].mean()) * 100, 1),
})
with open(RESULTS / "data_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

# --- Qrafiklər ---
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
all_raw["score"].value_counts().sort_index().plot.bar(ax=axes[0], color="#4C72B0")
axes[0].set_title("Xam data: ulduz paylanması")
axes[0].set_xlabel("score")
full["label"].map(dict(enumerate(LABELS))).value_counts().reindex(LABELS).plot.bar(ax=axes[1], color="#55A868")
axes[1].set_title("Təmiz data: sinif paylanması")
axes[1].tick_params(axis="x", rotation=0)
full["n_words"].clip(upper=40).plot.hist(bins=40, ax=axes[2], color="#C44E52")
axes[2].set_title("Söz sayı (40-da kəsilib)")
fig.tight_layout()
fig.savefig(RESULTS / "figures" / "eda.png", dpi=150)

print(json.dumps({k: v for k, v in stats.items() if k != "top_texts"}, ensure_ascii=False, indent=2))
