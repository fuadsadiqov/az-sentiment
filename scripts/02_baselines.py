"""Addım 2: baseline-lar. Hər modelin "keçməli olduğu" alt hədd.

majority: hamıya çoxluq sinfini (positive) deyir -> accuracy yüksək, macro-F1 aşağı.
stratified: sinif paylanmasına görə təsadüfi proqnoz.

uv run scripts/02_baselines.py
"""
from sklearn.dummy import DummyClassifier

from az_sentiment.data import SEED, load_split
from az_sentiment.metrics import evaluate

train, test = load_split("train"), load_split("test")
for strategy in ("most_frequent", "stratified"):
    clf = DummyClassifier(strategy=strategy, random_state=SEED).fit(train[["text"]], train["label"])
    name = "baseline_majority" if strategy == "most_frequent" else "baseline_random"
    evaluate(name, test, clf.predict(test[["text"]]))
