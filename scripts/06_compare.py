"""Addım 6: bütün modellərin müqayisəsi -> results/comparison.md + qrafik.

İki cədvəl: (1) tam test (LLM-lər istisna), (2) LLM alt çoxluğu (bütün modellər eyni nümunələrdə).

uv run scripts/06_compare.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from az_sentiment.data import LABELS, RESULTS
from az_sentiment.metrics import compute


def row(name: str, m: dict) -> dict:
    pc = m["per_class"]
    return {"model": name, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            **{f"F1_{c[:3]}": pc[c]["f1-score"] for c in LABELS},
            **{f"R_{c[:3]}": pc[c]["recall"] for c in LABELS}}


def fmt(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


metrics = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (RESULTS / "metrics").glob("*.json")}
full = [row(n, m) for n, m in metrics.items() if not m.get("subset") and "NAIVE" not in n]
full_df = pd.DataFrame(full).sort_values("macro_f1", ascending=False)

parts = ["# Model müqayisəsi\n", "## Tam test (təmiz split)\n", fmt(full_df), ""]

# Leakage: eyni model, naiv vs təmiz split
naive = [n for n in metrics if "NAIVE" in n]
if naive and "tfidf_logreg" in metrics:
    nv, cl = metrics[naive[0]], metrics["tfidf_logreg"]
    parts += ["## Leakage təsiri (TF-IDF + LogReg)\n",
              fmt(pd.DataFrame([row("orijinal split (leakage)", nv), row("təmiz split", cl)])), ""]

# LLM alt çoxluğu
ids_path = RESULTS / "llm_subset_ids.csv"
sub_df = None
if ids_path.exists():
    ids = set(pd.read_csv(ids_path)["id"])
    sub_rows = []
    for n in metrics:
        if "NAIVE" in n:
            continue
        p = pd.read_csv(RESULTS / "preds" / f"{n}.csv")
        p = p[p["id"].isin(ids)]
        if len(p) == len(ids):
            sub_rows.append(row(n, compute(p["y_true"], p["y_pred"])))
    sub_df = pd.DataFrame(sub_rows).sort_values("macro_f1", ascending=False)
    parts += [f"## LLM alt çoxluğu (n={len(ids)}, bütün modellər eyni nümunələrdə)\n", fmt(sub_df), ""]
    llm_info = [{"model": n, "valid_json_%": m.get("valid_json_pct"), "sec/item": m.get("sec_per_item")}
                for n, m in metrics.items() if m.get("subset")]
    if llm_info:
        parts += ["### LLM xərci\n", fmt(pd.DataFrame(llm_info)), ""]

(RESULTS / "comparison.md").write_text("\n".join(parts), encoding="utf-8")
print("\n".join(parts))

# Qrafik: macro-F1 + sinif üzrə F1
plot_df = sub_df if sub_df is not None and len(sub_df) > len(full_df) else full_df
fig, ax = plt.subplots(figsize=(10, 0.5 * len(plot_df) + 1.5))
y = range(len(plot_df))
h = 0.2
for i, (col, color) in enumerate([("macro_f1", "black"), ("F1_neg", "#C44E52"),
                                  ("F1_neu", "#8C8C8C"), ("F1_pos", "#55A868")]):
    ax.barh([v + (i - 1.5) * h for v in y], plot_df[col], height=h, label=col, color=color)
ax.set_yticks(list(y))
ax.set_yticklabels(plot_df["model"])
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("F1")
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.3)
ax.set_title("Model müqayisəsi" + (" (LLM alt çoxluğu)" if plot_df is sub_df else ""))
fig.tight_layout()
fig.savefig(RESULTS / "figures" / "comparison.png", dpi=150)
