from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
from pathlib import Path


# PATH SETUP
# ========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "checkpoints" / "logistic_regression" / "imdb"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "model.joblib"
SHAP_BG_PATH = MODEL_DIR / "shap_background.joblib"

SHAP_PLOT_PATH = PROJECT_ROOT / "results" / "figures" / "logistic_regression_imdb_shap.png"



# LABEL MAP
# ========================
def label_text(label: int) -> str:
    return "positive" if label == 1 else "negative"



# LOAD MODEL
# ========================
@st.cache_resource
def load_model():
    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        return None, None

    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model



# LOAD SHAP
# ========================
@st.cache_resource
def get_explainer():
    vectorizer, model = load_model()

    if vectorizer is None or model is None:
        return None, None, None

    if not SHAP_BG_PATH.exists():
        return None, None, None

    background = joblib.load(SHAP_BG_PATH)
    explainer = shap.LinearExplainer(model, background)

    return vectorizer, model, explainer



# PREDICTION
# ========================
def predict(text: str):
    vectorizer, model = load_model()
    if vectorizer is None:
        return None

    X = vectorizer.transform([text])
    pred = int(model.predict(X)[0])

    try:
        proba = model.predict_proba(X)[0][pred]
    except Exception:
        score = model.decision_function(X)[0]
        proba = 1 / (1 + np.exp(-score))

    return pred, float(proba)



# LOGISTIC REGRESSION COEFFICIENTS
# ========================
def explain_lr(text: str):
    vectorizer, model = load_model()
    if vectorizer is None:
        return None

    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]

    X = vectorizer.transform([text])
    indices = X.nonzero()[1]

    if len(indices) == 0:
        return [("No important words found", 0.0)]

    result = [
        (feature_names[i], float(coefs[i]))
        for i in indices
    ]

    return sorted(result, key=lambda x: abs(x[1]), reverse=True)



# SHAP (LOCAL)
# ========================
def explain_shap(text: str):
    vectorizer, model, explainer = get_explainer()

    if explainer is None:
        return None, None

    X = vectorizer.transform([text])

    pred = int(model.predict(X)[0])

    shap_values = explainer(X)

    values = shap_values.values[0]
    feature_names = vectorizer.get_feature_names_out()
    indices = X.nonzero()[1]

    if len(indices) == 0:
        return pred, [("No important words found", 0.0)]

    result = [
        (feature_names[i], float(values[i]))
        for i in indices
    ]

    return pred, sorted(result, key=lambda x: abs(x[1]), reverse=True)


# ========================
def main():
    st.set_page_config(page_title="Sentiment Analysis", layout="wide")

    st.title("Sentiment Analysis (Logistic Regression + SHAP)")


    # SINGLE SHARED INPUT
    # ========================
    if "input_text" not in st.session_state:
        st.session_state.input_text = "This movie is amazing but a bit slow"

    st.text_area(
        "Input text",
        key="input_text",
        height=120
    )

    text = st.session_state.input_text

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Predict", "LR Coeff Values", "SHAP Values" , "SHAP Plot"]
    )


    # TAB 1 - PREDICT
    # ========================
    with tab1:
        if st.button("Run Prediction"):
            result = predict(text)

            if result is None:
                st.error("Model not found.")
            else:
                label, conf = result
                st.metric("Prediction", label_text(label).title())
                st.write(f"Confidence: {conf:.4f}")
                st.write(f"Confidence = how sure the model is about its prediction, hence it is {conf*100:.2f}% sure")


    # TAB 2 - LR COEFFICIENTS
    # ========================
    with tab2:
        lr_result = explain_lr(text)

        if lr_result:
            st.write("+ve pushes towards Positive; -ve pushes towards Negetive; ~0 is Neutral Word")
            st.write("Global importance based on trained data (fixed value / word)")

            df = pd.DataFrame(lr_result, columns=["word", "lr_weight"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No explanation available.")


    # TAB 3 - SHAP LOCAL EXPLANATION
    # ========================
    with tab3:
        pred, shap_result = explain_shap(text)

        if shap_result:
            st.write("+ve pushes towards Positive; -ve pushes towards Negetive; ~0 is Neutral Word")
            st.write("Local Explanation (depends on sentence)")

            df = pd.DataFrame(shap_result, columns=["word", "shap_score"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No SHAP output.")


    # TAB 4 - GLOBAL SHAP PLOT
    # ========================
    with tab4:
        st.subheader("Global SHAP Summary")

        if SHAP_PLOT_PATH.exists():
            st.image(str(SHAP_PLOT_PATH), use_container_width=True)
        else:
            st.warning("Run training to generate SHAP plot.")


if __name__ == "__main__":
    main()