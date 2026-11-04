"""Addım 5: LLM ilə zero-shot / few-shot təsnifat (Ollama, lokal). Heç bir öyrətmə yoxdur.

Sual: heç bir label görmədən böyük dil modeli az resurslu dildə nə qədər işləyir?
Few-shot: prompt-a hər sinifdən k nümunə (train-dən) qoyulur.

LLM yavaş olduğu üçün test-dən təsadüfi n nümunə götürülür (default 1000). compare skripti
bütün modelləri həm də bu eyni alt çoxluqda müqayisə edir ki, müqayisə ədalətli olsun.

uv run scripts/05_llm.py --shots 0          # default: deepseek-r1:1.5b (lokal)
uv run scripts/05_llm.py --shots 3
deepseek-r1 "reasoning" modelidir: cavabdan əvvəl həmişə düşünür (think=False verilsə də).
Ona görə token limiti böyükdür; ilk 30 reasoning izi results/llm_traces_*.jsonl-a yazılır.
"""
import argparse
import json
import time

import ollama
from tqdm import tqdm

from az_sentiment.data import LABELS, RESULTS, SEED, load_split
from az_sentiment.metrics import evaluate

SYSTEM = (
    "You are a sentiment classifier for Azerbaijani mobile-app reviews (Google Play). "
    "Reviews may be written without Azerbaijani letters (e.g. 'ela' = 'əla'), mix in Russian/Turkish "
    "words, or contain only emojis.\n"
    "Labels:\n"
    "- negative: complaint, problem, bug, anger, disappointment\n"
    "- neutral: question, request, mixed opinion, or no clear sentiment\n"
    "- positive: praise, satisfaction, thanks\n"
    'Answer only with JSON: {"label": "negative" | "neutral" | "positive"}'
)
SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": LABELS}},
    "required": ["label"],
}


def build_shots(k: int) -> list[dict]:
    if k == 0:
        return []
    train = load_split("train")
    n_words = train["text"].str.split().str.len()
    pool = train[(n_words >= 3) & (n_words <= 25)]
    shots = pool.groupby("label", group_keys=False).sample(n=k, random_state=SEED)
    shots = shots.sample(frac=1, random_state=SEED)  # sinifləri qarışdır
    msgs = []
    for _, r in shots.iterrows():
        msgs.append({"role": "user", "content": f"Review: {r['text']}"})
        msgs.append({"role": "assistant", "content": json.dumps({"label": LABELS[r['label']]})})
    return msgs


def classify(model: str, text: str, shots: list[dict]) -> tuple[int, bool, str]:
    msgs = [{"role": "system", "content": SYSTEM}, *shots,
            {"role": "user", "content": f"Review: {text}"}]
    resp = ollama.chat(model=model, messages=msgs, format=SCHEMA,
                       options={"temperature": 0, "num_predict": 1024})
    thinking = resp["message"].get("thinking") or ""
    try:
        label = json.loads(resp["message"]["content"])["label"]
        return LABELS.index(label), True, thinking
    except (json.JSONDecodeError, KeyError, ValueError):
        return 1, False, thinking  # parse olunmadı -> neutral (sayı ayrıca qeyd olunur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-r1:1.5b")
    ap.add_argument("--shots", type=int, default=0, help="hər sinifdən neçə nümunə")
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    test = load_split("test")
    sub = test.sample(n=min(args.n, len(test)), random_state=SEED).reset_index(drop=True)
    sub[["id"]].to_csv(RESULTS / "llm_subset_ids.csv", index=False)

    shots = build_shots(args.shots)
    name = f"llm_{args.model.replace(':', '-').replace('/', '-')}_{args.shots}shot"
    preds, ok, traces, t0 = [], 0, [], time.time()
    for _, r in tqdm(sub.iterrows(), total=len(sub), desc=f"{args.model} {args.shots}-shot"):
        p, valid, thinking = classify(args.model, r["text"], shots)
        preds.append(p)
        ok += valid
        if len(traces) < 30:
            traces.append({"id": int(r["id"]), "text": r["text"], "true": LABELS[r["label"]],
                           "pred": LABELS[p], "valid": valid, "thinking": thinking})
    sec = time.time() - t0
    with open(RESULTS / f"llm_traces_{name}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(t, ensure_ascii=False) + "\n" for t in traces)

    evaluate(name, sub, preds, extra={
        "llm": args.model, "shots_per_class": args.shots, "valid_json_pct": round(100 * ok / len(sub), 1),
        "sec_per_item": round(sec / len(sub), 3), "subset": True,
    })


if __name__ == "__main__":
    main()
