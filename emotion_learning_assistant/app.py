"""
Phase 5: Streamlit UI.
Entry point for the AI-Driven Emotion Detection & Personalized Learning Support Platform.
Run with: streamlit run app.py
"""
import streamlit as st

from emotion_pipeline import predict, EMOTION_CLASSES
from gemini_guidance import generate_guidance
from logger import log_interaction, read_logs
from analytics import (
    emotion_distribution_chart, confidence_trend_chart,
    model_agreement_chart, mixed_emotion_rate_chart,
)


def configure_streamlit_plotly_theme():
    st.set_page_config(
        page_title="AI Learning Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        .stApp {
            background: #0f1117;
            font-family: 'Inter', sans-serif;
        }
        
        section[data-testid="stSidebar"] {
            background-color: #1a1d23;
        }
        
        .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
            max-width: 900px;
        }
        
        h1 {
            color: #f3f4f6;
            font-weight: 700;
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
        }
        
        .stCaption {
            color: #9ca3af !important;
            font-size: 1.05rem;
        }
        
        .stTextArea textarea {
            background-color: #1a1d23;
            color: #e0e0e0;
            border: 1px solid #2d3139;
            border-radius: 12px;
            padding: 16px;
            font-size: 16px;
            line-height: 1.6;
        }
        
        .stTextArea textarea:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
        }
        
        .stTextArea textarea::placeholder {
            color: #6b7280;
            opacity: 1;
        }
        
        .stRadio [role="radiogroup"] > label {
            color: #d1d5db;
            background-color: #1a1d23;
            border: 1px solid #2d3139;
            border-radius: 8px;
            padding: 10px 16px;
            margin: 4px;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .stRadio [role="radiogroup"] > label:hover {
            border-color: #4b5563;
            background-color: #22262e;
        }
        
        .stRadio [role="radiogroup"] > label[aria-checked="true"] {
            border-color: #3b82f6;
            background-color: rgba(59, 130, 246, 0.1);
        }
        
        .stToggle {
            color: #d1d5db;
        }
        
        .stToggle label {
            color: #e0e0e0;
            font-weight: 500;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            color: #ffffff;
            border: none;
            border-radius: 10px;
            padding: 14px 28px;
            font-weight: 600;
            font-size: 16px;
            width: 100%;
            transition: all 0.2s;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
        
        .stButton > button:hover:not(:disabled) {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            box-shadow: 0 6px 8px -1px rgba(0, 0, 0, 0.4);
            transform: translateY(-1px);
        }
        
        .stButton > button:active:not(:disabled) {
            transform: translateY(0);
        }
        
        .stButton > button:disabled {
            background: #374151;
            color: #6b7280;
            cursor: not-allowed;
            box-shadow: none;
        }
        
        .stMetric {
            background-color: #1a1d23;
            border: 1px solid #2d3139;
            border-radius: 10px;
            padding: 16px;
        }
        
        .stMetric [data-testid="stMetricLabel"] {
            color: #9ca3af;
            font-size: 14px;
            font-weight: 500;
        }
        
        .stMetric [data-testid="stMetricValue"] {
            color: #f3f4f6;
            font-size: 28px;
            font-weight: 700;
        }
        
        hr {
            border-color: #2d3139;
            margin: 1.5rem 0;
        }
        
        .stInfo {
            background-color: rgba(59, 130, 246, 0.1);
            color: #93c5fd;
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 8px;
        }
        
        .stSuccess {
            background-color: rgba(16, 185, 129, 0.1);
            color: #6ee7b7;
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 8px;
        }
        
        .stSubheader {
            color: #f3f4f6;
            font-weight: 600;
        }
        
        .results-card {
            background: #1a1d23;
            border: 1px solid #2d3139;
            border-radius: 12px;
            padding: 24px;
            margin-top: 1rem;
        }
        
        .emotion-badge {
            display: inline-block;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            color: white;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 8px;
        }
        
        .confidence-bar {
            background-color: #2d3139;
            border-radius: 4px;
            height: 8px;
            margin-top: 8px;
            overflow: hidden;
        }
        
        .confidence-fill {
            background: linear-gradient(90deg, #3b82f6, #10b981);
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    import plotly.io as pio
    pio.templates.default = "plotly_dark"


configure_streamlit_plotly_theme()

st.title("🎓 Tell me how you're feeling about what you're studying")
st.caption("I'm here to listen and help you work through it")

tab_assistant, tab_dashboard = st.tabs(["💬 Learning Assistant", "📊 Emotion Insights"])

# ----------------------------- Tab 1: Assistant -----------------------------
with tab_assistant:
    st.divider()
    
    # Input mode selection
    input_mode = st.radio(
        "How would you like to share?",
        ["Text Input", "Voice Input", "Quick Check-in"],
        horizontal=True,
        help="Choose how you want to tell us what's on your mind"
    )
    
    st.divider()
    
    # Text input area
    problem_text = st.text_area(
        "What's on your mind?",
        placeholder="e.g. I'm so lost on recursion, I don't even know where to start...",
        height=150,
        label_visibility="collapsed",
    )
    
    st.divider()
    
    # Controls row
    col1, col2 = st.columns([3, 2])
    with col1:
        model_choice_label = st.radio(
            "Model to use",
            ["Both (Ensemble)", "BiLSTM only", "BERT only"],
            horizontal=True,
        )
    with col2:
        show_ai_response = st.toggle(
            "Enable AI Guidance",
            value=True,
            help="Turn on for a deeper BERT-based emotion breakdown and personalized suggestions"
        )
        st.caption("Turn on for a deeper emotion breakdown and personalized suggestions")
    
    st.divider()
    
    # Analyze button
    choice_map = {"Both (Ensemble)": "both", "BiLSTM only": "bilstm", "BERT only": "bert"}
    
    analyze_disabled = not problem_text.strip()
    submitted = st.button(
        "🔍 Analyze Emotion",
        disabled=analyze_disabled,
        help="Please describe how you're feeling to continue" if analyze_disabled else None,
        width="stretch",
    )
    
    if not analyze_disabled and not submitted:
        st.caption("Click the button above when you're ready")
    
    if submitted:
        if not problem_text.strip():
            st.warning("Please describe how you're feeling to continue.")
        else:
            with st.spinner("Analyzing your emotions..."):
                results = predict(problem_text, choice_map[model_choice_label])
                final = results["final"]

            st.divider()
            st.subheader("📊 Your Emotion Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Detected Emotion", final["emotion"])
            with col2:
                st.metric("Confidence", f"{final['confidence']:.0%}")
            with col3:
                mixed_count = len(final["mixed_emotions"])
                st.metric("Mixed Emotions", str(mixed_count) if mixed_count > 1 else "None")

            if len(final["mixed_emotions"]) > 1:
                mixed_str = ", ".join(f"{m['emotion']} ({m['confidence']:.0%})" for m in final["mixed_emotions"])
                st.info(f"**Mixed emotions detected:** {mixed_str}")

            # Model comparison view
            if choice_map[model_choice_label] == "both":
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**BiLSTM Prediction**")
                    b = results["bilstm"]
                    st.write(f"{b['emotion']} — {b['confidence']:.0%}")
                    st.bar_chart(b["all_scores"], width="stretch")
                with c2:
                    st.markdown("**BERT Prediction**")
                    be = results["bert"]
                    st.write(f"{be['emotion']} — {be['confidence']:.0%}")
                    st.bar_chart(be["all_scores"], width="stretch")
            else:
                only = results.get("bilstm") or results.get("bert")
                st.bar_chart(only["all_scores"], width="stretch")

            guidance = ""
            if show_ai_response:
                with st.spinner("Generating personalized guidance..."):
                    guidance = generate_guidance(problem_text, final["emotion"], final["confidence"])
                st.divider()
                st.subheader("💡 Personalized Guidance")
                st.write(guidance)

            log_interaction(problem_text, results, guidance)
            st.success("✅ This interaction has been logged for the analytics dashboard.")

# ----------------------------- Tab 2: Dashboard -----------------------------
with tab_dashboard:
    st.divider()
    st.subheader("📈 Interaction Analytics")
    df = read_logs()
    st.write(f"Total interactions logged: **{len(df)}**")

    if df.empty:
        st.info("No interactions yet — analyze a problem in the first tab to populate this dashboard.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(emotion_distribution_chart(df), width="stretch")
        with c2:
            st.plotly_chart(model_agreement_chart(df), width="stretch")

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(confidence_trend_chart(df), width="stretch")
        with c4:
            st.plotly_chart(mixed_emotion_rate_chart(df), width="stretch")

        with st.expander("Raw interaction log"):
            st.dataframe(df, width="stretch")
