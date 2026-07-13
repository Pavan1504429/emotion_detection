"""
Phase 6: Analytics & Visualization using Plotly.
Builds the charts shown on the Streamlit dashboard tab.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

DEFAULT_PLOTLY_TEMPLATE = "plotly_white"


def apply_plotly_theme(fig: go.Figure):
    fig.update_layout(
        template=DEFAULT_PLOTLY_TEMPLATE,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#111111"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def emotion_distribution_chart(df: pd.DataFrame):
    if df.empty:
        return go.Figure().update_layout(title="No interactions logged yet")
    counts = df["final_emotion"].value_counts().reset_index()
    counts.columns = ["emotion", "count"]
    fig = px.bar(counts, x="emotion", y="count", color="emotion",
                 title="Emotion Distribution Across All Interactions")
    fig.update_layout(showlegend=False)
    return apply_plotly_theme(fig)


def confidence_trend_chart(df: pd.DataFrame):
    if df.empty:
        return go.Figure().update_layout(title="No interactions logged yet")
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.sort_values("timestamp")
    fig = px.line(d, x="timestamp", y="final_confidence", color="final_emotion",
                  markers=True, title="Confidence Over Time")
    return apply_plotly_theme(fig)


def model_agreement_chart(df: pd.DataFrame):
    if df.empty or "bilstm_emotion" not in df or "bert_emotion" not in df:
        return go.Figure().update_layout(title="No interactions logged yet")
    d = df.dropna(subset=["bilstm_emotion", "bert_emotion"])
    if d.empty:
        return go.Figure().update_layout(title="No dual-model comparisons logged yet")
    agree = (d["bilstm_emotion"] == d["bert_emotion"]).sum()
    disagree = len(d) - agree
    fig = go.Figure(data=[go.Pie(labels=["Models Agree", "Models Disagree"],
                                  values=[agree, disagree], hole=0.5)])
    fig.update_layout(title="BiLSTM vs BERT Agreement Rate")
    return apply_plotly_theme(fig)


def mixed_emotion_rate_chart(df: pd.DataFrame):
    if df.empty:
        return go.Figure().update_layout(title="No interactions logged yet")
    d = df.copy()
    d["is_mixed"] = d["mixed_emotions"].fillna("").apply(lambda x: len(x.split(";")) > 1 if x else False)
    counts = d["is_mixed"].value_counts().rename({True: "Mixed Emotion", False: "Single Emotion"})
    fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values, hole=0.5)])
    fig.update_layout(title="Single vs Mixed Emotion Detections")
    return apply_plotly_theme(fig)
