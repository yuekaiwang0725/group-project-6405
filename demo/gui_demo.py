import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.token_highlight import top_token_contributions
from src.models.baseline_svm import BaselineSVM
from src.models.distilbert import load_finetuned_distilbert, predict_sentiment
from src.robustness.perturbation import available_perturbations

MODEL_LABELS = {
    "baseline_svm": "Baseline SVM",
    "distilbert": "DistilBERT",
    "bert_base_uncased": "BERT-base-uncased",
}
EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}
EMOTION_ICONS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}
SAMPLE_TEXTS = {
    "Custom": "",
    "Clearly positive": "This movie is surprisingly good, emotional, and well acted.",
    "Clearly negative": "This film is dull, badly written, and a complete waste of time.",
    "Mixed review": "The visuals are excellent, but the story feels slow and predictable.",
    "Sarcastic": "What an amazing movie, I almost fell asleep from all the excitement.",
}
EMOTION_SAMPLE_TEXTS = {
    "Custom": "",
    "Joyful": "I am so excited and happy today, everything is going absolutely great!",
    "Sad": "I feel completely lost and alone, nothing seems to go right for me.",
    "Angry": "I can't believe how unfair and frustrating this whole situation is.",
    "Fearful": "I'm terrified about what might happen next, I can't stop worrying.",
    "Loving": "I cherish every moment with you, you mean the whole world to me.",
    "Surprised": "I had absolutely no idea this was going to happen, I'm completely shocked!",
}


def _label_text(label: int) -> str:
    return "positive" if label == 1 else "negative"


def _emotion_label_text(label: int) -> str:
    return EMOTION_LABELS.get(label, str(label))


def _resolve_checkpoint_dir(*parts: str) -> Path | None:
    base_dir = PROJECT_ROOT.joinpath("checkpoints", *parts)
    best_dir = base_dir / "best"
    if best_dir.exists():
        return best_dir
    if base_dir.exists():
        return base_dir
    return None


@st.cache_resource
def _load_baseline():
    vectorizer_path = (
        PROJECT_ROOT / "checkpoints" / "baseline" / "imdb" / "tfidf_vectorizer.joblib"
    )
    model_path = PROJECT_ROOT / "checkpoints" / "baseline" / "imdb" / "svm_model.joblib"
    if not vectorizer_path.exists() or not model_path.exists():
        return None, None
    vectorizer = joblib.load(vectorizer_path)
    model: BaselineSVM = joblib.load(model_path)
    return vectorizer, model


@st.cache_resource
def _load_emotion_baseline():
    vectorizer_path = (
        PROJECT_ROOT / "checkpoints" / "baseline" / "emotion" / "tfidf_vectorizer.joblib"
    )
    model_path = PROJECT_ROOT / "checkpoints" / "baseline" / "emotion" / "svm_model.joblib"
    if not vectorizer_path.exists() or not model_path.exists():
        return None, None
    vectorizer = joblib.load(vectorizer_path)
    model: BaselineSVM = joblib.load(model_path)
    return vectorizer, model


@st.cache_resource
def _load_distilbert():
    model_dir = _resolve_checkpoint_dir("distilbert", "imdb")
    if model_dir is None:
        return None
    return load_finetuned_distilbert(model_dir)


@st.cache_resource
def _load_emotion_distilbert():
    model_dir = _resolve_checkpoint_dir("distilbert", "emotion")
    if model_dir is None:
        return None
    return load_finetuned_distilbert(model_dir)


@st.cache_resource
def _load_bert():
    model_dir = _resolve_checkpoint_dir("bert_base_uncased", "imdb")
    if model_dir is None:
        return None
    return load_finetuned_distilbert(model_dir)


def _predict_baseline(text: str) -> tuple[int, float] | None:
    vectorizer, model = _load_baseline()
    if vectorizer is None or model is None:
        return None
    features = vectorizer.transform([text])
    pred = int(model.predict(features)[0])
    confidence = float(model.predict_confidence(features)[0])
    return pred, confidence


def _predict_emotion_baseline(text: str) -> tuple[int, float] | None:
    vectorizer, model = _load_emotion_baseline()
    if vectorizer is None or model is None:
        return None
    features = vectorizer.transform([text])
    pred = int(model.predict(features)[0])
    conf_scores = model.predict_confidence(features)[0]
    if hasattr(conf_scores, "__len__"):
        confidence = float(conf_scores[pred])
    else:
        confidence = float(conf_scores)
    return pred, confidence


def _predict_distilbert(text: str) -> tuple[int, float] | None:
    artifacts = _load_distilbert()
    if artifacts is None:
        return None
    return predict_sentiment(artifacts, text)


def _predict_emotion_distilbert(text: str) -> tuple[int, float] | None:
    artifacts = _load_emotion_distilbert()
    if artifacts is None:
        return None
    return predict_sentiment(artifacts, text)


def _predict_bert(text: str) -> tuple[int, float] | None:
    artifacts = _load_bert()
    if artifacts is None:
        return None
    return predict_sentiment(artifacts, text)


def _predict_with_model(model_name: str, text: str) -> tuple[int, float] | None:
    if model_name == "baseline_svm":
        return _predict_baseline(text)
    if model_name == "distilbert":
        return _predict_distilbert(text)
    return _predict_bert(text)


def _predict_all_models(text: str) -> dict[str, tuple[int, float] | None]:
    return {model_name: _predict_with_model(model_name, text) for model_name in MODEL_LABELS}


def _render_prediction_result(model_name: str, result: tuple[int, float] | None) -> None:
    st.subheader(MODEL_LABELS[model_name])
    if result is None:
        st.warning("Checkpoint missing.")
        return
    label, confidence = result
    st.metric("Sentiment", _label_text(label).title())
    st.caption(f"Confidence: {confidence:.4f}")


def _render_emotion_result(model_display: str, result: tuple[int, float] | None) -> None:
    st.subheader(model_display)
    if result is None:
        st.warning("Checkpoint missing.")
        return
    label, confidence = result
    emotion_name = _emotion_label_text(label)
    icon = EMOTION_ICONS.get(emotion_name, "")
    st.metric("Emotion", f"{icon} {emotion_name.title()}")
    st.caption(f"Confidence: {confidence:.4f}")


def _render_agreement_summary(results: dict[str, tuple[int, float] | None]) -> None:
    available = {
        model_name: prediction for model_name, prediction in results.items() if prediction is not None
    }
    if not available:
        st.error("No model checkpoints are available.")
        return
    labels = {_label_text(prediction[0]) for prediction in available.values()}
    if len(labels) == 1:
        agreed_label = next(iter(labels))
        st.success(f"All available models agree: {agreed_label}.")
    else:
        st.warning("Models disagree on this text. Compare confidence scores below.")


def _show_metric_table(path: Path, title: str) -> None:
    if not path.exists():
        st.warning(f"{title}: file not found at {path}")
        return
    df = pd.read_csv(path)
    st.subheader(title)
    st.dataframe(df, use_container_width=True)


def _show_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="EE6405 Sentiment GUI", layout="wide")
    st.title("Trustworthy English Sentiment Analysis")
    st.write("GUI demo for sentiment and emotion classification using SVM, DistilBERT, and BERT.")

    tab_predict, tab_emotion, tab_explain, tab_robustness, tab_benchmark = st.tabs(
        ["Sentiment", "Emotion (6-class)", "Explain", "Robustness", "Benchmark"]
    )

    with tab_predict:
        st.subheader("Sentiment Prediction (Positive / Negative)")
        if "predict_text" not in st.session_state:
            st.session_state["predict_text"] = SAMPLE_TEXTS["Clearly positive"]
        example_name = st.selectbox("Quick example", list(SAMPLE_TEXTS), index=1)
        if example_name != "Custom":
            st.session_state["predict_text"] = SAMPLE_TEXTS[example_name]
        text = st.text_area("Input text", key="predict_text")
        if st.button("Run prediction"):
            results = _predict_all_models(text)
            _render_agreement_summary(results)
            columns = st.columns(len(MODEL_LABELS))
            for column, model_name in zip(columns, MODEL_LABELS):
                with column:
                    _render_prediction_result(model_name, results[model_name])

    with tab_emotion:
        st.subheader("Emotion Classification (6 classes)")
        st.caption("Classes: 😢 sadness · 😄 joy · ❤️ love · 😠 anger · 😨 fear · 😲 surprise")
        if "emotion_text" not in st.session_state:
            st.session_state["emotion_text"] = EMOTION_SAMPLE_TEXTS["Joyful"]
        emotion_example = st.selectbox(
            "Quick example", list(EMOTION_SAMPLE_TEXTS), index=1, key="emotion_example"
        )
        if emotion_example != "Custom":
            st.session_state["emotion_text"] = EMOTION_SAMPLE_TEXTS[emotion_example]
        emotion_text = st.text_area("Input text", key="emotion_text")
        if st.button("Detect emotion"):
            col_svm, col_distilbert = st.columns(2)
            with col_svm:
                _render_emotion_result("TF-IDF + SVM", _predict_emotion_baseline(emotion_text))
            with col_distilbert:
                _render_emotion_result("DistilBERT", _predict_emotion_distilbert(emotion_text))

    with tab_explain:
        st.subheader("Explainability")
        text = st.text_area(
            "Text for explanation",
            "The story is engaging but some scenes are too long.",
            key="explain_text",
        )
        vectorizer, model = _load_baseline()
        if vectorizer is None or model is None:
            st.warning("Baseline checkpoint missing. Run baseline experiment first.")
        else:
            contributions = top_token_contributions(text, vectorizer, model, top_k=12)
            if not contributions:
                st.info("No token contribution available for this text.")
            else:
                explain_df = pd.DataFrame(
                    contributions, columns=["token/ngram", "contribution"]
                )
                st.dataframe(explain_df, use_container_width=True)

    with tab_robustness:
        st.subheader("Stress Test")
        source_text = st.text_area(
            "Original text",
            "I expected this film to be boring, but it was excellent.",
            key="robust_text",
        )
        model_name = st.selectbox("Model for stress test", list(MODEL_LABELS), key="robust_model")
        perturbation_name = st.selectbox(
            "Perturbation", list(available_perturbations().keys()), key="perturb_type"
        )
        if st.button("Run stress test"):
            perturb_fn = available_perturbations()[perturbation_name]
            perturbed_text = perturb_fn(source_text)
            original_pred = _predict_with_model(model_name, source_text)
            perturbed_pred = _predict_with_model(model_name, perturbed_text)
            if original_pred is None or perturbed_pred is None:
                st.error("Model checkpoint missing.")
            else:
                st.write(f"Original: `{source_text}`")
                st.write(f"Perturbed ({perturbation_name}): `{perturbed_text}`")
                st.info(
                    f"Original => {_label_text(original_pred[0])} ({original_pred[1]:.4f})"
                )
                st.info(
                    f"Perturbed => {_label_text(perturbed_pred[0])} ({perturbed_pred[1]:.4f})"
                )

    with tab_benchmark:
        st.subheader("Sentiment Analysis — Model Comparison")
        _show_metric_table(
            PROJECT_ROOT / "results" / "tables" / "baseline_imdb_metrics.csv",
            "Baseline IMDb Metrics",
        )
        _show_metric_table(
            PROJECT_ROOT / "results" / "tables" / "distilbert_imdb_test_metrics.csv",
            "DistilBERT IMDb Metrics",
        )
        _show_metric_table(
            PROJECT_ROOT / "results" / "tables" / "bert_base_uncased_imdb_test_metrics.csv",
            "BERT IMDb Metrics",
        )
        _show_metric_table(
            PROJECT_ROOT / "results" / "tables" / "metrics_cross_domain.csv",
            "Cross-domain Metrics",
        )
        _show_metric_table(
            PROJECT_ROOT / "results" / "tables" / "robustness_scores.csv",
            "Robustness Scores",
        )

        st.subheader("Emotion Classification — Model Comparison")
        _show_metric_table(
            PROJECT_ROOT / "results" / "tables" / "baseline_emotion_metrics.csv",
            "Baseline SVM — Emotion (macro)",
        )
        _show_metric_table(
            PROJECT_ROOT / "results" / "tables" / "distilbert_emotion_test_metrics.csv",
            "DistilBERT — Emotion (macro)",
        )

        st.subheader("Figures")
        col1, col2 = st.columns(2)
        with col1:
            _show_image(
                PROJECT_ROOT / "results" / "figures" / "baseline_imdb_confusion_matrix.png",
                "Baseline confusion matrix (IMDb)",
            )
            _show_image(
                PROJECT_ROOT / "results" / "figures" / "distilbert_imdb_training_curves.png",
                "DistilBERT training curves (IMDb)",
            )
            _show_image(
                PROJECT_ROOT / "results" / "figures" / "bert_base_uncased_imdb_training_curves.png",
                "BERT training curves (IMDb)",
            )
        with col2:
            _show_image(
                PROJECT_ROOT / "results" / "figures" / "baseline_emotion_confusion_matrix.png",
                "Baseline confusion matrix (Emotion)",
            )
            _show_image(
                PROJECT_ROOT / "results" / "figures" / "distilbert_emotion_training_curves.png",
                "DistilBERT training curves (Emotion)",
            )
        _show_image(
            PROJECT_ROOT / "results" / "figures" / "model_comparison_in_domain_f1.png",
            "In-domain model comparison",
        )
        _show_image(
            PROJECT_ROOT / "results" / "figures" / "robustness_drop_curve.png",
            "Robustness F1 drop",
        )


if __name__ == "__main__":
    main()
