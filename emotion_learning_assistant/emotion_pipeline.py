"""
Phase 3: Emotion Detection Pipeline
- Text preprocessing (via data_preprocessing.py)
- BiLSTM + BERT inference (loaded lazily, in case models aren't trained yet)
- Rule-based keyword enhancement (boosts/penalizes class scores using lexicons)
- Confidence scoring (softmax max prob)
- Mixed-emotion detection (any class within MIXED_THRESHOLD of the top class is reported too)

If no trained model files exist under models/, predict() falls back to a pure rule-based
classifier so the rest of the app (Streamlit UI, Gemini guidance, logging, analytics) is
runnable end-to-end from day one.
"""
import os
import re
from typing import Optional

import numpy as np

from data_preprocessing import clean_text, remove_stopwords, EMOTION_CLASSES

MODEL_DIR = "models"
MAX_LEN = 40
MIXED_THRESHOLD = 0.18  # if a 2nd/3rd class's prob is within this margin of the top prob, report it

# --- Rule-based keyword lexicon: used both as a standalone fallback AND to nudge model scores ---
KEYWORDS = {
    "Bored": [
        "boring", "bored", "tedious", "monotonous", "same old", "already know",
        "putting me to sleep", "checking the clock", "nothing new", "dull",
        "waste of time", "not interested", "don't care", "dont care", "boring class",
        "so boring", "too boring", "boring lecture", "sleepy", "tired of this",
        "repeating", "repeated", "already learned", "nothing new"
    ],
    "Confident": [
        "confident", "i can do this", "i understand", "i got this", "easy",
        "i solved", "feel great", "unstoppable", "mastered", "i know this",
        "cracked", "feel amazing", "nailed it", "i finally got", "i finally solved",
        "happy", "happiest", "happiness", "pleased", "delighted", "wonderful",
        "awesome", "fantastic", "excellent", "excellent", "proud", "accomplished",
        "achieved", "succeeded", "success", "positive", "optimistic", "joyful",
        "cheerful", "elated", "thrilled", "satisfied", "well done", "did well",
        "passed", "aced", "great job", "good job", "feel good", "feeling good",
        "happy with", "happy about", "glad", "relieved", "grateful", "thankful",
        "comfortable", "capable", "prepared", "ready", "strong", "clear",
        "makes sense", "perfect", "flawless", "brilliant", "outstanding"
    ],
    "Confused": [
        "confused", "don't understand", "dont understand", "lost", "no idea",
        "doesn't make sense", "not sure", "unclear", "what is", "can't figure",
        "cant figure", "makes no sense", "confusing", "puzzled", "perplexed",
        "bewildered", "don't get it", "dont get it", "missing something",
        "missing pieces", "all jumbled", "mixed up", "unclear about",
        "not following", "can't follow", "cant follow", "what does this mean"
    ],
    "Curious": [
        "curious", "wonder", "interested", "want to learn", "want to understand",
        "how does", "why does", "explain", "dig deeper", "want to know",
        "intrigued", "fascinated", "captivated", "tell me more", "learn about",
        "explore", "discover", "find out", "how do", "how can", "what if",
        "what happens when", "relationship between", "connection between",
        "how might", "how would", "i wonder", "i'm curious", "im curious"
    ],
    "Frustrated": [
        "frustrated", "frustrating", "annoyed", "stuck", "keep failing",
        "throw my laptop", "same error", "gave up", "so hard", "hate this",
        "angry", "fed up", "disappointed", "discouraged", "overwhelmed",
        "stressed", "anxious", "worried", "can't do this", "cant do this",
        "not working", "doesn't work", "doesnt work", "broken", "useless",
        "wasting my time", "no progress", "nowhere", "nowhere fast",
        "this is too hard", "impossible", "unfair", "ugh", "argh"
    ],
}


def rule_based_scores(text: str) -> dict:
    """Return a normalized score dict over EMOTION_CLASSES from keyword hits."""
    t = " " + text.lower() + " "
    scores = {c: 0.0 for c in EMOTION_CLASSES}
    for cls, kws in KEYWORDS.items():
        for kw in kws:
            if kw in t:
                scores[cls] += 1.0
    total = sum(scores.values())
    if total == 0:
        # neutral fallback: uniform distribution
        return {c: 1.0 / len(EMOTION_CLASSES) for c in EMOTION_CLASSES}
    return {c: v / total for c, v in scores.items()}


def _blend(model_scores: dict, rule_scores: dict, rule_weight: float = 0.25) -> dict:
    blended = {c: (1 - rule_weight) * model_scores.get(c, 0) + rule_weight * rule_scores.get(c, 0)
               for c in EMOTION_CLASSES}
    total = sum(blended.values()) or 1.0
    return {c: v / total for c, v in blended.items()}


def _scores_to_result(scores: dict, model_name: str) -> dict:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_label, top_prob = ranked[0]
    mixed = [{"emotion": lbl, "confidence": round(p, 4)}
             for lbl, p in ranked if (top_prob - p) <= MIXED_THRESHOLD]
    return {
        "model": model_name,
        "emotion": top_label,
        "confidence": round(top_prob, 4),
        "mixed_emotions": mixed,
        "all_scores": {k: round(v, 4) for k, v in scores.items()},
    }


# ----------------------- Lazy model loaders -----------------------
_bilstm_cache = {}
_bert_cache = {}


def _load_bilstm():
    if _bilstm_cache:
        return _bilstm_cache
    model_path = os.path.join(MODEL_DIR, "bilstm_model.h5")
    tok_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
    le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    if not (os.path.exists(model_path) and os.path.exists(tok_path) and os.path.exists(le_path)):
        return None
    from tensorflow.keras.models import load_model
    from data_preprocessing import load_pickle
    _bilstm_cache["model"] = load_model(model_path)
    _bilstm_cache["tokenizer"] = load_pickle(tok_path)
    _bilstm_cache["label_encoder"] = load_pickle(le_path)
    return _bilstm_cache


def _load_bert():
    if _bert_cache:
        return _bert_cache
    bert_dir = os.path.join(MODEL_DIR, "bert_model")
    le_path = os.path.join(MODEL_DIR, "bert_label_encoder.pkl")
    if not (os.path.isdir(bert_dir) and os.path.exists(le_path)):
        return None
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from data_preprocessing import load_pickle
    _bert_cache["tokenizer"] = AutoTokenizer.from_pretrained(bert_dir)
    _bert_cache["model"] = AutoModelForSequenceClassification.from_pretrained(bert_dir)
    _bert_cache["model"].eval()
    _bert_cache["label_encoder"] = load_pickle(le_path)
    _bert_cache["torch"] = torch
    return _bert_cache


# ----------------------- Predictors -----------------------
def predict_bilstm(text: str) -> Optional[dict]:
    bundle = _load_bilstm()
    cleaned = remove_stopwords(clean_text(text))
    rule_scores = rule_based_scores(text)
    if bundle is None:
        return _scores_to_result(rule_scores, "BiLSTM (fallback: rule-based, no trained model found)")
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = bundle["tokenizer"].texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = bundle["model"].predict(padded, verbose=0)[0]
    classes = bundle["label_encoder"].classes_
    if len(probs) != len(classes):
        raise ValueError(
            f"BiLSTM model outputs {len(probs)} classes but label encoder has {len(classes)} classes. "
            "Retrain or replace model files so they match."
        )
    model_scores = {cls: float(p) for cls, p in zip(classes, probs)}
    blended = _blend(model_scores, rule_scores)
    return _scores_to_result(blended, "BiLSTM")


def predict_bert(text: str) -> Optional[dict]:
    bundle = _load_bert()
    rule_scores = rule_based_scores(text)
    if bundle is None:
        return _scores_to_result(rule_scores, "BERT (fallback: rule-based, no trained model found)")
    torch = bundle["torch"]
    cleaned = clean_text(text)
    inputs = bundle["tokenizer"](
        cleaned, return_tensors="pt", truncation=True, padding="max_length", max_length=64
    )
    with torch.no_grad():
        logits = bundle["model"](**inputs).logits
        probs = torch.softmax(logits, dim=-1).numpy()[0]
    classes = bundle["label_encoder"].classes_
    if len(probs) != len(classes):
        raise ValueError(
            f"BERT model outputs {len(probs)} classes but label encoder has {len(classes)} classes. "
            "Retrain or replace model files so they match."
        )
    model_scores = {cls: float(p) for cls, p in zip(classes, probs)}
    blended = _blend(model_scores, rule_scores)
    return _scores_to_result(blended, "BERT")


def predict(text: str, model_choice: str = "both") -> dict:
    """
    model_choice: 'bilstm' | 'bert' | 'both'
    Returns dict with per-model results and a combined "final" verdict
    (averaged when both models are used) for the app to display.
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty input text")

    results = {}
    if model_choice in ("bilstm", "both"):
        results["bilstm"] = predict_bilstm(text)
    if model_choice in ("bert", "both"):
        results["bert"] = predict_bert(text)

    if len(results) == 2:
        avg_scores = {
            c: (results["bilstm"]["all_scores"][c] + results["bert"]["all_scores"][c]) / 2
            for c in EMOTION_CLASSES
        }
        results["final"] = _scores_to_result(avg_scores, "Ensemble (BiLSTM + BERT avg)")
    else:
        only_key = next(iter(results))
        results["final"] = results[only_key]

    return results


if __name__ == "__main__":
    tests = [
        "I'm so lost on recursion I don't even know where to start",
        "I finally cracked the DP problem and I feel amazing",
        "This lecture is going so slow I am checking the clock every minute",
        "Why does backpropagation actually work I want to dig deeper",
    ]
    for t in tests:
        r = predict(t, "both")
        print(f"\nInput: {t}")
        print(f"  Final -> {r['final']['emotion']} (conf {r['final']['confidence']})")
        print(f"  Mixed -> {r['final']['mixed_emotions']}")
