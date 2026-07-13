# AI-Driven Emotion Detection & Personalized Learning Support Platform

Detects a student's emotion (Bored, Confident, Confused, Curious, Frustrated) from free-text
and returns emotion-aware guidance using BiLSTM + BERT + rule-based keyword boosting + Gemini AI.

## Folder Structure
```
emotion_learning_assistant/
├── data/
│   └── sample_emotions.csv        # bootstrap dataset (replace with your Kaggle dataset)
├── models/                        # trained models get saved here
├── logs/
│   └── interactions.csv           # auto-created, logs every user query
├── notebooks/
│   ├── train_bilstm.py            # run on Kaggle (GPU) to train BiLSTM
│   └── train_bert.py              # run on Kaggle (GPU) to fine-tune BERT
├── data_preprocessing.py          # cleaning, tokenization, label encoding
├── emotion_pipeline.py            # core inference: BiLSTM+BERT+rules+confidence+mixed emotion
├── gemini_guidance.py             # calls Gemini API to generate personalized tips
├── analytics.py                   # Plotly charts for the dashboard
├── logger.py                      # CSV interaction logging
├── app.py                         # Streamlit UI (entry point)
└── requirements.txt
```

## 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You also need an NLTK download once:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 2. Get a Dataset
Replace `data/sample_emotions.csv` with your real Kaggle dataset. It must have two columns:
`text,emotion` where emotion ∈ {Bored, Confident, Confused, Curious, Frustrated}.

Good starting points on Kaggle: search "student emotion text classification" or repurpose a
general emotion dataset (dair-ai/emotion, ISEAR) and remap labels to the 5 classes above.

## 3. Train Models on Kaggle (GPU)
Upload `notebooks/train_bilstm.py` and `notebooks/train_bert.py` plus your dataset to a Kaggle
Notebook with GPU enabled (Settings → Accelerator → GPU T4 x2). Run each script cell-by-cell
(or `%run train_bilstm.py`). They save:
- `models/bilstm_model.h5`, `models/tokenizer.pkl`, `models/label_encoder.pkl`
- `models/bert_model/` (HuggingFace format), `models/bert_label_encoder.pkl`

Download those artifacts and place them in your local `models/` folder.

## 4. Set your Gemini API Key
```bash
export GEMINI_API_KEY="your-key-here"     # Windows: set GEMINI_API_KEY=your-key-here
```
Get a free key at https://aistudio.google.com/app/apikey

## 5. Run the App
```bash
streamlit run app.py
```

## What Each Phase Maps To
| Instruction Phase                     | File(s)                                    |
|----------------------------------------|---------------------------------------------|
| 1. Environment Setup                    | requirements.txt, this README                |
| 2. Kaggle Model Training                | notebooks/train_bilstm.py, notebooks/train_bert.py |
| 3. Emotion Detection Pipeline           | data_preprocessing.py, emotion_pipeline.py    |
| 4. AI-Powered Guidance                  | gemini_guidance.py                            |
| 5. Streamlit UI                         | app.py                                        |
| 6. Analytics & Visualization            | analytics.py                                  |
| 7. CSV Logging                          | logger.py                                     |
| 8. Testing & Optimization               | see "Testing" section below                   |

## Testing & Optimization
- Run `python data_preprocessing.py` standalone to sanity-check cleaning on sample_emotions.csv.
- Run `python emotion_pipeline.py` standalone — it runs a few hardcoded test sentences and
  prints predicted emotion + confidence + mixed-emotion breakdown from both models (falls back
  gracefully to a rule-based predictor if no trained model files are found yet, so the app is
  runnable end-to-end even before you finish Kaggle training).
- If BERT inference is too slow on CPU, reduce `MAX_LEN` in `emotion_pipeline.py` or use
  `bert-base` → a smaller distil model.
