"""Addım 7: səhv təhlili. Model harada və niyə yanılır?

- Dilimlər (slices): mətn uzunluğu, Azərbaycan hərfləri var/yox, ziddiyyətli label-lı mətnlər.
- Ən çox qarışdırılan sinif cütləri.
- Hər iki güclü modelin eyni yanıldığı nümunələr: çox vaxt bu, model yox, label səhvidir.

uv run scripts/07_error_analysis.py --model xlm-roberta-base_weighted --other tfidf_logreg
"""
import argparse
import json

import pandas as pd

from az_sentiment.data import LABELS, RESULTS, SEED, has_az_chars, load_split
from az_sentiment.metrics import compute


def slice_table(df: pd.DataFrame) -> pd.DataFrame:
    n_words = df["text"].str.split().str.len()
    slices = {
        "hamısı": df.index == df.index,
        "1-2 söz": n_words <= 2,
        "3-5 söz": n_words.between(3, 5),
        "6-15 söz": n_words.between(6, 15),
        "16+ söz": n_words >= 16,
        "Az. hərfləri var": df["text"].map(has_az_chars),
        "Az. hərfləri yox": ~df["text"].map(has_az_chars),
        "ziddiyyətli label (dublikatlarda)": df["n_labels"] > 1,
    }
    rows = []
    for name, mask in slices.items():
        d = df[mask]
        if len(d) < 20:
            continue
        m = compute(d["y_true"], d["y_pred"])
        rows.append({"slice": name, "n": len(d), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "F1_neg": m["per_class"]["negative"]["f1-score"],
                     "F1_neu": m["per_class"]["neutral"]["f1-score"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="results/preds/<model>.csv")
    ap.add_argument("--other", default=None, help="müqayisə üçün ikinci model")
    ap.add_argument("--k", type=int, default=12, help="hər qrupdan nümunə sayı")
    args = ap.parse_args()

    test = load_split("test")[["id", "n_labels", "n_copies", "score"]]
    df = pd.read_csv(RESULTS / "preds" / f"{args.model}.csv").merge(test, on="id")
    df["true"] = df["y_true"].map(dict(enumerate(LABELS)))
    df["pred"] = df["y_pred"].map(dict(enumerate(LABELS)))
    err = df[df["y_true"] != df["y_pred"]]

    out = [f"# Səhv təhlili: {args.model}\n",
           f"Test: {len(df)}, səhv: {len(err)} ({100 * len(err) / len(df):.1f}%)\n",
           "## Dilimlər üzrə performans\n", slice_table(df).to_markdown(index=False, floatfmt=".3f"), "",
           "## Ən çox qarışdırılan cütlər (həqiqi → proqnoz)\n",
           err.groupby(["true", "pred"]).size().sort_values(ascending=False)
              .rename("say").reset_index().to_markdown(index=False), ""]

    for (t, p), g in err.groupby(["true", "pred"]):
        out.append(f"### {t} → {p} (ulduz göstərilib)\n")
        for _, r in g.sample(n=min(args.k, len(g)), random_state=SEED).iterrows():
            out.append(f"- [{r['score']}★] {r['text'][:200]}")
        out.append("")

    if args.other:
        o = pd.read_csv(RESULTS / "preds" / f"{args.other}.csv")[["id", "y_pred"]].rename(
            columns={"y_pred": "y_other"})
        both = df.merge(o, on="id")
        both_wrong = both[(both["y_pred"] != both["y_true"]) & (both["y_other"] != both["y_true"])]
        agree_wrong = both_wrong[both_wrong["y_pred"] == both_wrong["y_other"]]
        out += [f"## Hər iki model ({args.model}, {args.other}) eyni cür yanılır: {len(agree_wrong)} nümunə\n",
                "Bunların bir hissəsi label səhvidir (rəy mətni ilə ulduz uyğun gəlmir). "
                "Əl ilə yoxlayıb faizini hesabla: bu, modelin 'tavanı' haqqında məlumat verir.\n"]
        for _, r in agree_wrong.sample(n=min(30, len(agree_wrong)), random_state=SEED).iterrows():
            out.append(f"- label={r['true']} [{r['score']}★], hər ikisi={r['pred']}: {r['text'][:200]}")
        agree_wrong[["id", "text", "score", "true", "pred"]].to_csv(
            RESULTS / f"label_noise_candidates_{args.model}.csv", index=False, encoding="utf-8")

    path = RESULTS / f"errors_{args.model}.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[:12]))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
