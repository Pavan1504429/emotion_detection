"""
Phase 7: CSV Logging.
Stores every user query, per-model predictions, final emotion, confidence,
generated guidance, and a timestamp so the analytics dashboard has data to plot.
"""
import os
import csv
from datetime import datetime

LOG_PATH = os.path.join("logs", "interactions.csv")
FIELDS = ["timestamp", "input_text", "bilstm_emotion", "bilstm_confidence",
          "bert_emotion", "bert_confidence", "final_emotion", "final_confidence",
          "mixed_emotions", "guidance"]


def _ensure_log_file():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def log_interaction(input_text: str, results: dict, guidance: str):
    _ensure_log_file()
    bilstm = results.get("bilstm", {})
    bert = results.get("bert", {})
    final = results.get("final", {})
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_text": input_text,
        "bilstm_emotion": bilstm.get("emotion", ""),
        "bilstm_confidence": bilstm.get("confidence", ""),
        "bert_emotion": bert.get("emotion", ""),
        "bert_confidence": bert.get("confidence", ""),
        "final_emotion": final.get("emotion", ""),
        "final_confidence": final.get("confidence", ""),
        "mixed_emotions": "; ".join(m["emotion"] for m in final.get("mixed_emotions", [])),
        "guidance": guidance.replace("\n", " "),
    }
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)


def read_logs():
    import pandas as pd
    _ensure_log_file()
    return pd.read_csv(LOG_PATH)
