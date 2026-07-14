# 🎓 AI-Driven Emotion Detection & Personalized Learning Support Platform

An emotion-aware learning assistant that detects a student's emotional state — **Bored, Confident, Confused, Curious, Frustrated** — from free-text input and delivers personalized, AI-generated guidance in real time.

Built with a hybrid **BiLSTM + BERT** classification pipeline, rule-based keyword boosting for edge cases, and **Gemini AI** for generating context-aware learning tips, all wrapped in an interactive **Streamlit** dashboard.

---

## ✨ Features

- **Dual-model emotion classification** — BiLSTM for fast inference, BERT for contextual accuracy, combined with confidence scoring
- **Mixed-emotion detection** — surfaces secondary emotions when a student's input reflects more than one state
- **Rule-based keyword boosting** — catches edge cases the models might miss
- **AI-generated personalized guidance** via Gemini API, tailored to the detected emotion
- **Interactive analytics dashboard** with Plotly visualizations of emotional trends
- **Automatic interaction logging** for tracking usage and model performance over time
- **Graceful fallback** — runs end-to-end on rule-based prediction even before models are trained

---

## 🧱 Project Structure

```
emotion_learning_assistant/
├── data/
│   └── sample_emotions.csv        # bootstrap dataset (replace with your Kaggle dataset)
├── models/                        # trained model artifacts saved here
├── logs/
│   └── interactions.csv           # auto-created, logs every user query
├── notebooks/
│   ├── train_bilstm.py            # run on Kaggle (GPU) to train BiLSTM
│   └── train_bert.py              # run on Kaggle (GPU) to fine-tune BERT
├── data_preprocessing.py          # cleaning, tokenization, label encoding
├── emotion_pipeline.py            # core inference: BiLSTM + BERT + rules + confidence + mixed emotion
├── gemini_guidance.py             # calls Gemini API to generate personalized tips
├── analytics.py                   # Plotly charts for the dashboard
├── logger.py                      # CSV interaction logging
├── app.py                         # Streamlit UI (entry point)
└── requirements.txt
```

---

## ⚙️ Tech Stack

| Layer               | Technology                          |
|----------------------|--------------------------------------|
| UI / Dashboard        | Streamlit, Plotly                   |
| Emotion Classification | TensorFlow (BiLSTM), HuggingFace Transformers (BERT) |
| Guidance Generation    | Google Gemini API                   |
| Data Handling          | Pandas, NLTK                        |
| Logging                | CSV-based interaction logs          |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Pavan1504429/emotion_learning_assistant.git
cd emotion_learning_assistant
```

### 2. Set up the environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download required NLTK resources (one-time):
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 3. Prepare the dataset
Replace `data/sample_emotions.csv` with your full dataset. Required format:

| text | emotion |
|------|---------|
| "I don't understand this at all" | Confused |

Valid `emotion` values: `Bored`, `Confident`, `Confused`, `Curious`, `Frustrated`.

**Dataset sources:** search Kaggle for "student emotion text classification," or repurpose a general emotion dataset (e.g. `dair-ai/emotion`, ISEAR) and remap labels to the 5 target classes.

### 4. Train the models (Kaggle GPU recommended)
Upload `notebooks/train_bilstm.py` and `notebooks/train_bert.py` along with your dataset to a Kaggle Notebook with GPU enabled (**Settings → Accelerator → GPU T4 x2**). Run each script cell-by-cell (or `%run train_bilstm.py`).

This produces the following artifacts:
- `models/bilstm_model.h5`
- `models/tokenizer.pkl`
- `models/label_encoder.pkl`
- `models/bert_model/` (HuggingFace format)
- `models/bert_label_encoder.pkl`

Download these and place them in your local `models/` directory.

### 5. Configure your Gemini API key
```bash
export GEMINI_API_KEY="your-key-here"     # Windows: set GEMINI_API_KEY=your-key-here
```
Get a free API key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

### 6. Run the app
```bash
streamlit run app.py
```

---

## 🗺️ How It Works

| Stage | Description | File(s) |
|-------|-------------|---------|
| 1. Environment Setup | Dependencies, NLTK resources | `requirements.txt` |
| 2. Model Training | BiLSTM + BERT training on Kaggle GPU | `notebooks/train_bilstm.py`, `notebooks/train_bert.py` |
| 3. Emotion Detection | Preprocessing + hybrid inference pipeline | `data_preprocessing.py`, `emotion_pipeline.py` |
| 4. AI Guidance | Personalized tips via Gemini | `gemini_guidance.py` |
| 5. User Interface | Streamlit dashboard | `app.py` |
| 6. Analytics | Visual trends and stats | `analytics.py` |
| 7. Logging | Interaction history | `logger.py` |
| 8. Testing | Standalone sanity checks | see below |

---

## 🧪 Testing & Optimization

- Run `python data_preprocessing.py` to sanity-check text cleaning on `sample_emotions.csv`.
- Run `python emotion_pipeline.py` to test inference on hardcoded sample sentences — it prints predicted emotion, confidence, and a mixed-emotion breakdown from both models. If no trained model files are found, it falls back to a rule-based predictor so the app remains runnable end-to-end.
- If BERT inference is slow on CPU, reduce `MAX_LEN` in `emotion_pipeline.py` or switch to a smaller distilled BERT model.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Rongali Pavan Kumar**
[GitHub](https://github.com/Pavan1504429)
