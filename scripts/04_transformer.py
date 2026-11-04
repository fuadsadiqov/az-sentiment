"""Addım 4: Transformer fine-tune (default: xlm-roberta-base, 100 dil, Azərbaycan dili daxil).

Nə baş verir:
  tokenizer mətni subword token-lərə bölür -> pretrained encoder hər token üçün kontekstli vektor
  verir -> [CLS] vektorunun üstünə kiçik təsnifat qatı (3 çıxış) -> bütün şəbəkə bizim data ilə
  bir neçə epoch öyrədilir (transfer learning).

Balanssızlıq üçün: --weighted ilə loss sinif çəkiləri ilə hesablanır (az sinifdəki səhv daha bahalıdır).
Overfitting üçün: hər epoch validation-da yoxlanır, ən yaxşı macro-F1 olan checkpoint saxlanır,
2 epoch irəliləyiş olmasa dayanır (early stopping).

uv run scripts/04_transformer.py --weighted
uv run scripts/04_transformer.py --model bert-base-multilingual-cased --weighted
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from az_sentiment.data import LABELS, MODELS, RESULTS, SEED, load_split
from az_sentiment.metrics import evaluate


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(outputs.logits.float(), labels, weight=w)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro"),
            "accuracy": float((preds == labels).mean())}


def plot_history(history: list[dict], path, title: str) -> None:
    tr = [(h["epoch"], h["loss"]) for h in history if "loss" in h and "eval_loss" not in h]
    ev = [(h["epoch"], h["eval_loss"], h["eval_macro_f1"]) for h in history if "eval_loss" in h]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    if tr:
        ax1.plot(*zip(*tr), label="train_loss", alpha=0.6)
    if ev:
        ax1.plot([e[0] for e in ev], [e[1] for e in ev], marker="o", label="val_loss")
        ax2 = ax1.twinx()
        ax2.plot([e[0] for e in ev], [e[2] for e in ev], marker="s", color="green", label="val_macro_f1")
        ax2.set_ylabel("val macro-F1")
        ax2.legend(loc="lower right")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper right")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--weighted", action="store_true", help="sinif çəkili loss")
    ap.add_argument("--train-frac", type=float, default=1.0, help="learning curve üçün train payı")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    short = args.model.split("/")[-1]
    name = args.name or f"{short}{'_weighted' if args.weighted else ''}" + (
        f"_frac{args.train_frac}" if args.train_frac < 1 else "")
    out_dir = MODELS / name

    train, val, test = load_split("train"), load_split("val"), load_split("test")
    if args.train_frac < 1:
        train = train.groupby("label", group_keys=False).sample(frac=args.train_frac, random_state=SEED)

    tok = AutoTokenizer.from_pretrained(args.model)

    def to_ds(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(df[["text", "label"]].rename(columns={"label": "labels"}),
                                 preserve_index=False)
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
                      batched=True, remove_columns=["text"])

    ds_train, ds_val, ds_test = to_ds(train), to_ds(val), to_ds(test)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=3,
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})

    weights = None
    if args.weighted:
        w = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=train["label"])
        weights = torch.tensor(np.sqrt(w), dtype=torch.float)  # sqrt: tam "balanced" neutral-ı həddən artıq itələyir
        print("class weights:", weights.tolist())

    steps_per_epoch = int(np.ceil(len(ds_train) / args.batch))
    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        weight_decay=0.01,
        warmup_steps=int(0.06 * steps_per_epoch * args.epochs),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=max(steps_per_epoch // 10, 10),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # RTX 4060 / A100
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),  # Colab T4
        report_to="none",
        seed=SEED,
        dataloader_num_workers=0,
    )
    trainer = WeightedTrainer(
        model=model, args=targs,
        train_dataset=ds_train, eval_dataset=ds_val,
        processing_class=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=weights,
    )
    trainer.train()

    trainer.save_model(str(out_dir / "best"))
    history = trainer.state.log_history
    with open(RESULTS / f"history_{name}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=1)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    plot_history(history, RESULTS / "figures" / f"history_{name}.png", name)

    logits = trainer.predict(ds_test).predictions
    best_epoch = next((h["epoch"] for h in history if h.get("eval_macro_f1") == trainer.state.best_metric), None)
    evaluate(name, test, logits.argmax(-1), extra={
        "base_model": args.model, "epochs_run": history[-1].get("epoch"), "best_epoch": best_epoch,
        "best_val_macro_f1": trainer.state.best_metric, "weighted": args.weighted,
        "n_train": len(train), "lr": args.lr, "batch": args.batch, "max_len": args.max_len,
    })


if __name__ == "__main__":
    main()
