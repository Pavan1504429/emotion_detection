"""
Phase 3 (part 1): Text preprocessing utilities shared by BiLSTM training,
BERT training, and the live inference pipeline.
"""
import re
import pickle

import nltk
import pandas as pd
from nltk.corpus import stopwords

EMOTION_CLASSES = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]

for pkg in ("punkt", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))
# keep negations - they flip sentiment/emotion meaning
NEGATIONS = {"no", "not", "nor", "don't", "didn't", "isn't", "wasn't", "can't", "won't", "never"}
STOPWORDS -= NEGATIONS


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/punct/extra whitespace, keep letters/apostrophes only."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return " ".join(tokens) if tokens else text  # never return an empty string


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text", label_col: str = "emotion") -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].apply(clean_text)
    df["clean_no_stop"] = df[text_col].apply(remove_stopwords)
    df = df[df[label_col].isin(EMOTION_CLASSES)].reset_index(drop=True)
    return df


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return preprocess_dataframe(df)


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    df = load_dataset("data/sample_emotions.csv")
    print(f"Loaded {len(df)} rows across classes: {df['emotion'].value_counts().to_dict()}")
    print(df.head(3).to_string())
