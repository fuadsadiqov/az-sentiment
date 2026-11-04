"""Addım 3: klassik ML. TF-IDF (söz + simvol n-qram) + Logistic Regression.

- Hiperparametrlər (C, class_weight) validation-da macro-F1 ilə seçilir, test-ə toxunulmur.
- Simvol n-qramları "ela/əla/eladi/əladır" kimi yazılış fərqlərinə davamlıdır.
- --naive: orijinal (leakage-li) split-də eyni model -> nəticə nə qədər şişir?
- Learning curve: train datasının 1%..100%-i ilə test macro-F1.

uv run scripts/03_tfidf.py
"""
import argparse
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from az_sentiment.data import MODELS, PROCESSED, RESULTS, SEED, load_split
from az_sentiment.metrics import evaluate


def make_pipeline(C: float = 1.0, class_weight=None) -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                 sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 sublinear_tf=True, lowercase=True, max_features=300_000)),
    ])
    clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=2000, random_state=SEED)
    return Pipeline([("tfidf", feats), ("clf", clf)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-curve", action="store_true")
    args = ap.parse_args()

    train, val, test = load_split("train"), load_split("val"), load_split("test")

    # 1) Validation-da grid search
    grid = []
    for cw in (None, "balanced"):
        for C in (0.5, 2.0, 8.0):
            t = time.time()
            pipe = make_pipeline(C, cw).fit(train["text"], train["label"])
            f1 = f1_score(val["label"], pipe.predict(val["text"]), average="macro")
            grid.append({"C": C, "class_weight": str(cw), "val_macro_f1": round(f1, 4),
                         "sec": round(time.time() - t, 1)})
            print(grid[-1])
    best = max(grid, key=lambda g: g["val_macro_f1"])
    best_cw = None if best["class_weight"] == "None" else "balanced"
    print("Ən yaxşı:", best)

    # 2) Ən yaxşı konfiqurasiya ilə train-də öyrət, test-də bir dəfə qiymətləndir
    pipe = make_pipeline(best["C"], best_cw).fit(train["text"], train["label"])
    MODELS.mkdir(exist_ok=True)
    joblib.dump(pipe, MODELS / "tfidf_logreg.joblib")
    evaluate("tfidf_logreg", test, pipe.predict(test["text"]), extra={"grid": grid, "best": best})

    # class_weight-in təsirini ayrıca göstərmək üçün digər variant da test-də
    other_cw = None if best_cw == "balanced" else "balanced"
    other = make_pipeline(best["C"], other_cw).fit(train["text"], train["label"])
    evaluate(f"tfidf_logreg_cw-{other_cw}", test, other.predict(test["text"]))

    # 3) Leakage təcrübəsi: orijinal split, dublikatlar silinmədən
    ntr = pd.read_parquet(PROCESSED / "naive_train.parquet")
    nte = pd.read_parquet(PROCESSED / "naive_test.parquet")
    naive = make_pipeline(best["C"], best_cw).fit(ntr["text"], ntr["label"])
    evaluate("tfidf_logreg_NAIVE_split", nte, naive.predict(nte["text"]),
             extra={"note": "orijinal split, dublikat və leakage var"})

    # 4) Learning curve
    if args.skip_curve:
        return
    rows = []
    for frac in (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
        sub = train.groupby("label", group_keys=False).sample(frac=frac, random_state=SEED)
        p = make_pipeline(best["C"], best_cw).fit(sub["text"], sub["label"])
        f1 = f1_score(test["label"], p.predict(test["text"]), average="macro")
        rows.append({"frac": frac, "n_train": len(sub), "test_macro_f1": round(f1, 4)})
        print(rows[-1])
    curve = pd.DataFrame(rows)
    curve.to_csv(RESULTS / "learning_curve_tfidf.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve["n_train"], curve["test_macro_f1"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Train nümunə sayı (log)")
    ax.set_ylabel("Test macro-F1")
    ax.set_title("TF-IDF + LogReg: learning curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "learning_curve_tfidf.png", dpi=150)


if __name__ == "__main__":
    main()
