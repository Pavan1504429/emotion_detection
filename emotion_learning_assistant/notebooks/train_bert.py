"""
Phase 2: Fine-tune a BERT emotion classifier.
Run this on a Kaggle Notebook with GPU enabled.

Expected input: data/sample_emotions.csv (or your own dataset) with columns: text,emotion
Outputs saved to models/: bert_model/ (HF format), bert_label_encoder.pkl
"""
import os
import sys
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing import load_dataset, EMOTION_CLASSES  # noqa: E402

DATA_PATH = "data/sample_emotions.csv"
MODEL_DIR = "models"
BERT_OUT = os.path.join(MODEL_DIR, "bert_model")
BASE_MODEL = "bert-base-uncased"
MAX_LEN = 64
os.makedirs(BERT_OUT, exist_ok=True)


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LEN)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}


def main():
    df = load_dataset(DATA_PATH)
    # BERT uses its own tokenizer/wordpieces, so keep light cleaning without stopword removal
    texts = df["text"].tolist()
    labels_raw = df["emotion"].tolist()

    le = LabelEncoder()
    le.fit(EMOTION_CLASSES)
    labels = le.transform(labels_raw).tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42,
        stratify=labels if len(set(labels)) > 1 else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(EMOTION_CLASSES)
    )

    train_ds = EmotionDataset(X_train, y_train, tokenizer)
    val_ds = EmotionDataset(X_val, y_val, tokenizer)

    args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "bert_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=6,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"Validation accuracy: {metrics['eval_accuracy']:.4f}")

    model.save_pretrained(BERT_OUT)
    tokenizer.save_pretrained(BERT_OUT)
    with open(os.path.join(MODEL_DIR, "bert_label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print(f"Saved fine-tuned BERT to {BERT_OUT}/ and bert_label_encoder.pkl to models/")


if __name__ == "__main__":
    main()
