"""
EE6405 Sentiment Analysis — v1.0 (Binary Classification Only)
=============================================================
This is the ORIGINAL version of our project demo.
It supports only binary sentiment classification (Positive / Negative)
using two models:
  • Baseline: TF-IDF + SVM
  • Modern:   DistilBERT fine-tuned on IMDb

Run with:
    streamlit run demo/demo1.py --server.port 8502
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline_svm import BaselineSVM
from src.models.distilbert import load_finetuned_distilbert, predict_sentiment
from src.explainability.token_highlight import top_token_contributions

# ── Constants ────────────────────────────────────────────────
SAMPLE_TEXTS = {
    "Custom": "",
    "Clearly positive": "This movie is surprisingly good, emotional, and well acted.",
    "Clearly negative": "This film is dull, badly written, and a complete waste of time.",
    "Mixed review": "The visuals are excellent, but the story feels slow and predictable.",
}


def _label_text(label: int) -> str:
    return "positive" if label == 1 else "negative"


# ── Model loading ────────────────────────────────────────────
@st.cache_resource
def _load_baseline():
    vec_path = PROJECT_ROOT / "checkpoints" / "baseline" / "imdb" / "tfidf_vectorizer.joblib"
    mdl_path = PROJECT_ROOT / "checkpoints" / "baseline" / "imdb" / "svm_model.joblib"
    if not vec_path.exists() or not mdl_path.exists():
        return None, None
    return joblib.load(vec_path), joblib.load(mdl_path)


@st.cache_resource
def _load_distilbert():
    model_dir = PROJECT_ROOT / "checkpoints" / "distilbert" / "imdb" / "best"
    if not model_dir.exists():
        return None
    return load_finetuned_distilbert(model_dir)


def _predict_baseline(text: str):
    vectorizer, model = _load_baseline()
    if vectorizer is None:
        return None
    features = vectorizer.transform([text])
    pred = int(model.predict(features)[0])
    conf = float(model.predict_confidence(features)[0])
    return pred, conf


def _predict_distilbert(text: str):
    artifacts = _load_distilbert()
    if artifacts is None:
        return None
    return predict_sentiment(artifacts, text)


# ── Main app ─────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="EE6405 Sentiment v1.0",
        layout="centered",   # 刻意用 centered（窄版），对比 v3 的 wide
    )

    # ── Simple header (no fancy CSS, plain Streamlit) ────────
    st.title("🎬 Sentiment Analysis v1.0")
    st.caption("Binary classification: Positive / Negative")
    st.markdown("---")

    # ── Tab layout: only 2 simple tabs ───────────────────────
    tab_predict, tab_benchmark = st.tabs(["Predict", "Benchmark"])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Tab 1: Single text prediction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_predict:
        st.subheader("Enter text for sentiment prediction")

        example = st.selectbox("Quick example", list(SAMPLE_TEXTS), index=1)
        default_text = SAMPLE_TEXTS[example] if example != "Custom" else ""
        text = st.text_area("Input text", value=default_text, height=100)

        if st.button("Predict"):
            if not text.strip():
                st.warning("Please enter some text.")
            else:
                st.markdown("---")

                # ── SVM result ───────────────────────────────
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("TF-IDF + SVM")
                    result = _predict_baseline(text)
                    if result is None:
                        st.error("Checkpoint not found.")
                    else:
                        label, conf = result
                        st.metric("Sentiment", _label_text(label).upper())
                        st.write(f"Confidence: `{conf:.4f}`")

                # ── DistilBERT result ────────────────────────
                with col2:
                    st.subheader("DistilBERT")
                    result = _predict_distilbert(text)
                    if result is None:
                        st.error("Checkpoint not found.")
                    else:
                        label, conf = result
                        st.metric("Sentiment", _label_text(label).upper())
                        st.write(f"Confidence: `{conf:.4f}`")

                # ── Simple token contribution table ──────────
                st.markdown("---")
                st.subheader("Token Contributions (SVM)")
                vectorizer, model = _load_baseline()
                if vectorizer is not None:
                    contributions = top_token_contributions(text, vectorizer, model, top_k=10)
                    if contributions:
                        df = pd.DataFrame(contributions, columns=["Token", "Contribution"])
                        st.table(df)
                    else:
                        st.info("No contributions available.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Tab 2: Simple benchmark table
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_benchmark:
        st.subheader("Model Performance (IMDb Test Set)")

        # Baseline metrics
        baseline_path = PROJECT_ROOT / "results" / "tables" / "baseline_imdb_metrics.csv"
        if baseline_path.exists():
            st.write("**Baseline: TF-IDF + SVM**")
            st.dataframe(pd.read_csv(baseline_path), use_container_width=True)
        else:
            st.warning("Baseline metrics not found.")

        # DistilBERT metrics
        distilbert_path = PROJECT_ROOT / "results" / "tables" / "distilbert_imdb_test_metrics.csv"
        if distilbert_path.exists():
            st.write("**DistilBERT**")
            st.dataframe(pd.read_csv(distilbert_path), use_container_width=True)
        else:
            st.warning("DistilBERT metrics not found.")

        # Static figures
        st.markdown("---")
        st.subheader("Figures")
        cm_path = PROJECT_ROOT / "results" / "figures" / "baseline_imdb_confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Baseline Confusion Matrix (IMDb)")

        tc_path = PROJECT_ROOT / "results" / "figures" / "distilbert_imdb_training_curves.png"
        if tc_path.exists():
            st.image(str(tc_path), caption="DistilBERT Training Curves (IMDb)")

    # ── Footer ───────────────────────────────────────────────
    st.markdown("---")
    st.caption("EE6405 NLP Group Project — Version 1.0 (Binary Sentiment Only)")


if __name__ == "__main__":
    main()
