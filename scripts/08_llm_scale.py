"""Addım 8: LLM miqyas təcrübəsi. Model ölçüsü artdıqca zero-shot macro-F1 necə dəyişir?

Eyni ailə (Qwen2.5: 1.5B, 3B, 7B, ops. 14B) eyni prompt, eyni 500 test nümunəsi.
Müqayisə xətləri: öyrədilmiş XLM-R və TF-IDF, təsadüfi baseline (eyni 500 nümunədə).

uv run scripts/08_llm_scale.py
"""
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from az_sentiment.data import RESULTS
from az_sentiment.metrics import compute

ids = set(pd.read_csv(RESULTS / "llm_subset_ids.csv")["id"])


def subset_f1(name: str) -> float:
    p = pd.read_csv(RESULTS / "preds" / f"{name}.csv")
    p = p[p["id"].isin(ids)]
    return compute(p["y_true"], p["y_pred"])["macro_f1"]


rows = []
for path in sorted((RESULTS / "metrics").glob("llm_*.json")):
    m = json.loads(path.read_text(encoding="utf-8"))
    size = re.search(r"(\d+(?:\.\d+)?)b", m["llm"])
    rows.append({
        "model": m["llm"], "family": m["llm"].split(":")[0], "shots": m["shots_per_class"],
        "params_B": float(size.group(1)) if size else None,
        "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
        "F1_neu": m["per_class"]["neutral"]["f1-score"],
        "valid_json_%": m["valid_json_pct"], "sec/item": m["sec_per_item"],
    })
df = pd.DataFrame(rows).sort_values(["family", "shots", "params_B"])
refs = {n: subset_f1(n) for n in ("xlm-roberta-base_weighted", "tfidf_logreg", "baseline_random")}

md = ["# LLM miqyas təcrübəsi (n=500)\n", df.to_markdown(index=False, floatfmt=".3f"), "",
      "Müqayisə (eyni 500): " + ", ".join(f"{k} = {v:.3f}" for k, v in refs.items())]
(RESULTS / "llm_scale.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))

fig, ax = plt.subplots(figsize=(7, 4.5))
for (fam, shots), g in df.groupby(["family", "shots"]):
    g = g.dropna(subset=["params_B"])
    ax.plot(g["params_B"], g["macro_f1"], marker="o", label=f"{fam} {shots}-shot")
    for _, r in g.iterrows():
        ax.annotate(f"{r['macro_f1']:.2f}", (r["params_B"], r["macro_f1"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
styles = {"xlm-roberta-base_weighted": ("XLM-R fine-tune (0.28B)", "green"),
          "tfidf_logreg": ("TF-IDF + LogReg", "gray"),
          "baseline_random": ("təsadüfi baseline", "red")}
for k, (label, color) in styles.items():
    ax.axhline(refs[k], linestyle="--", color=color, alpha=0.7, label=f"{label}: {refs[k]:.2f}")
ax.set_xscale("log")
ax.set_xticks([1.5, 3, 7, 14])
ax.set_xticklabels(["1.5B", "3B", "7B", "14B"])
ax.set_xlabel("LLM parametr sayı (log)")
ax.set_ylabel("macro-F1 (500 test nümunəsi)")
ax.set_ylim(0, 0.75)
ax.set_title("LLM ölçüsü vs öyrədilmiş kiçik model")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS / "figures" / "llm_scale.png", dpi=150)
