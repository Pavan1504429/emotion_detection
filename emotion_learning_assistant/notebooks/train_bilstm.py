"""
Phase 2: Train a BiLSTM emotion classifier.
Run this on a Kaggle Notebook with GPU enabled.

Expected input: data/sample_emotions.csv (or your own dataset) with columns: text,emotion
Outputs saved to models/: bilstm_model.h5, tokenizer.pkl, label_encoder.pkl
"""
import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing import load_dataset, EMOTION_CLASSES  # noqa: E402

DATA_PATH = "data/sample_emotions.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

VOCAB_SIZE = 10000
MAX_LEN = 40
EMBED_DIM = 128
BATCH_SIZE = 16
EPOCHS = 30


def build_model(vocab_size, num_classes):
    model = Sequential([
        Embedding(vocab_size, EMBED_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    df = load_dataset(DATA_PATH)
    texts = df["clean_no_stop"].tolist()
    labels = df["emotion"].tolist()

    le = LabelEncoder()
    le.fit(EMOTION_CLASSES)
    y = le.transform(labels)

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    model = build_model(min(VOCAB_SIZE, len(tokenizer.word_index) + 1), len(EMOTION_CLASSES))
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=2,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")

    model.save(os.path.join(MODEL_DIR, "bilstm_model.h5"))
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print("Saved bilstm_model.h5, tokenizer.pkl, label_encoder.pkl to models/")


if __name__ == "__main__":
    main()
