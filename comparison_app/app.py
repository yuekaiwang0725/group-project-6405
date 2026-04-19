"""
EE6405 Team Sentiment Classification : Comparison Dashboard
===========================================================

Main Streamlit entry point. Seven tabs:
  1. Overall comparison across all models
  2. SVM + DistilBERT (Wang Yuekai)
  3. LoRA fine-tuning (Shivaangii Jaiswal)
  4. BERTweet full fine-tune (WaiYarAung)
  5. Bi-GRU + Attention (YU JUNCHENG)
  6. BiLSTM + Attention (joannasj)
  7. TF-IDF + Logistic Regression (Kevin Wong)

Run with:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from tabs import bertweet_full, bilstm, gru, logreg, lora, overall, svm_distilbert

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
ASSETS_DIR = APP_DIR / "assets"

st.set_page_config(
    page_title="EE6405 · Sentiment Model Comparison",
    page_icon="EE",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _header() -> None:
    st.title("EE6405 · Team Sentiment Model Comparison")
    st.caption(
        "Interactive comparison of six sentiment-classification approaches "
        "across **IMDb**, **SST-2**, and **Emotion** datasets : "
        "classical (TF-IDF + SVM, TF-IDF + LogReg), recurrent (Bi-GRU, BiLSTM), and "
        "transformer (DistilBERT, BERTweet full-FT, LoRA on RoBERTa-large / BERTweet / DistilBERT)."
    )


def main() -> None:
    _header()

    tabs = st.tabs(
        [
            "Overall",
            "SVM + DistilBERT (Wang Yuekai)",
            "LoRA (Shivaangii Jaiswal)",
            "BERTweet Full-FT (WaiYarAung)",
            "Bi-GRU + Attn (YU JUNCHENG)",
            "BiLSTM + Attn (joannasj)",
            "TF-IDF + LogReg (Kevin Wong)",
        ]
    )

    with tabs[0]:
        overall.render(DATA_DIR, ASSETS_DIR)
    with tabs[1]:
        svm_distilbert.render(DATA_DIR, ASSETS_DIR)
    with tabs[2]:
        lora.render(DATA_DIR, ASSETS_DIR)
    with tabs[3]:
        bertweet_full.render(DATA_DIR, ASSETS_DIR)
    with tabs[4]:
        gru.render(DATA_DIR, ASSETS_DIR)
    with tabs[5]:
        bilstm.render(DATA_DIR, ASSETS_DIR)
    with tabs[6]:
        logreg.render(DATA_DIR, ASSETS_DIR)


if __name__ == "__main__":
    main()
