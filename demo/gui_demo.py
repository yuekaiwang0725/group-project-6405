import streamlit as st


def main() -> None:
    st.set_page_config(page_title="EE6405 Sentiment GUI", layout="wide")
    st.title("Trustworthy English Sentiment Analysis")
    st.write("GUI demo for baseline and DistilBERT sentiment inference.")

    tab_predict, tab_explain, tab_robustness, tab_benchmark = st.tabs(
        ["Predict", "Explain", "Robustness", "Benchmark"]
    )

    with tab_predict:
        st.subheader("Prediction")
        text = st.text_area("Input text", "This movie is surprisingly good.")
        st.button("Run prediction")
        st.info(f"Input preview: {text[:120]}")

    with tab_explain:
        st.subheader("Explainability")
        st.write("TODO: show token/word-level attribution.")

    with tab_robustness:
        st.subheader("Stress Test")
        st.write("TODO: generate perturbations and compare outputs.")

    with tab_benchmark:
        st.subheader("Model Comparison")
        st.write("TODO: load and visualize final metrics/figures.")


if __name__ == "__main__":
    main()
