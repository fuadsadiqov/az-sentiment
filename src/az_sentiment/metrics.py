"""Bütün modellər üçün eyni qiymətləndirmə: metrikalar JSON, confusion matrix PNG, proqnozlar CSV."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .data import LABELS, RESULTS


def compute(y_true, y_pred) -> dict:
    rep = classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=LABELS, output_dict=True, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=[0, 1, 2], zero_division=0),
        "per_class": {
            c: {k: rep[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in LABELS
        },
    }


def plot_cm(y_true, y_pred, title: str, path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    annot = np.array([[f"{cm_norm[i, j]:.2f}\n({cm[i, j]})" for j in range(3)] for i in range(3)])
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1,
                xticklabels=LABELS, yticklabels=LABELS, cbar=False, ax=ax)
    ax.set_xlabel("Proqnoz")
    ax.set_ylabel("Həqiqi")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(name: str, df: pd.DataFrame, y_pred, extra: dict | None = None, verbose: bool = True) -> dict:
    """df: id, text, label sütunları olan test DataFrame-i."""
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    (RESULTS / "preds").mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics").mkdir(parents=True, exist_ok=True)

    y_true = df["label"].to_numpy()
    y_pred = np.asarray(y_pred)
    res = {"model": name, "n_test": int(len(df)), **compute(y_true, y_pred), **(extra or {})}

    with open(RESULTS / "metrics" / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    plot_cm(y_true, y_pred, name, RESULTS / "figures" / f"cm_{name}.png")
    pd.DataFrame({"id": df["id"].to_numpy(), "text": df["text"].to_numpy(),
                  "y_true": y_true, "y_pred": y_pred}).to_csv(
        RESULTS / "preds" / f"{name}.csv", index=False, encoding="utf-8")

    if verbose:
        pc = res["per_class"]
        print(f"[{name}] acc={res['accuracy']:.3f}  macro_f1={res['macro_f1']:.3f}  "
              + "  ".join(f"{c[:3]}: P={pc[c]['precision']:.2f} R={pc[c]['recall']:.2f}" for c in LABELS))
    return res
