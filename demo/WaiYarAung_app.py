"""
Combined Sentiment & Emotion Classifier  —  Stakeholder Demo
Page 1 : Text input  -> 3-model predictions + LIME explanations
Page 2 : YouTube URL -> batch comment analysis
Page 3 : Evaluation results for all three models
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)

MODELS = {
    "IMDB Sentiment": {
        "dir":     os.path.join(PARENT, "bertweet_sentiment", "results", "best_model"),
        "results": os.path.join(PARENT, "bertweet_sentiment", "results"),
        "labels":  ["Negative", "Positive"],
        "colors":  ["#ef4444", "#22c55e"],
        "icons":   ["👎", "👍"],
        "max_len": 128,
        "desc":    "Trained on IMDB movie reviews",
        "acc":     "86.5%",
    },
    "SST-2 Sentiment": {
        "dir":     os.path.join(PARENT, "sst2_sentiment", "results", "best_model"),
        "results": os.path.join(PARENT, "sst2_sentiment", "results"),
        "labels":  ["Negative", "Positive"],
        "colors":  ["#ef4444", "#22c55e"],
        "icons":   ["👎", "👍"],
        "max_len": 64,
        "desc":    "Trained on Stanford Sentiment Treebank",
        "acc":     "85.3%",
    },
    "Emotion Classifier": {
        "dir":     os.path.join(PARENT, "emotion_sentiment", "results", "best_model"),
        "results": os.path.join(PARENT, "emotion_sentiment", "results"),
        "labels":  ["sadness", "joy", "love", "anger", "fear", "surprise"],
        "colors":  ["#60a5fa", "#fbbf24", "#f472b6", "#ef4444", "#a78bfa", "#34d399"],
        "icons":   ["😢", "😊", "❤️", "😠", "😨", "😲"],
        "max_len": 64,
        "desc":    "6-class emotion on dair-ai/emotion",
        "acc":     "86.6%",
    },
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Sentiment & Emotion Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts & base ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Top hero banner ──────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #1d4ed8 100%);
    border-radius: 14px;
    padding: 2.2rem 2.5rem 1.8rem;
    margin-bottom: 1.6rem;
    color: white;
}
.hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 .4rem; color: white; }
.hero p  { font-size: 1rem; opacity: .85; margin: 0; color: #cbd5e1; }
.hero-badges { margin-top: .9rem; display: flex; gap: .5rem; flex-wrap: wrap; }
.badge {
    background: rgba(255,255,255,.15);
    border: 1px solid rgba(255,255,255,.25);
    color: white;
    font-size: .75rem;
    font-weight: 600;
    padding: .25rem .7rem;
    border-radius: 999px;
}

/* ── Section header ───────────────────────────────────────────── */
.sec-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f172a;
    border-left: 4px solid #1d4ed8;
    padding-left: .75rem;
    margin: 1.4rem 0 .7rem;
}

/* ── Prediction card ──────────────────────────────────────────── */
.pred-card {
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: .8rem;
    box-shadow: 0 1px 6px rgba(0,0,0,.08);
    border: 1px solid #e2e8f0;
    background: white;
}
.pred-card .model-name {
    font-size: .78rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .05em;
    margin-bottom: .5rem;
}
.pred-card .label-badge {
    display: inline-block;
    padding: .35rem 1rem;
    border-radius: 8px;
    font-size: 1.1rem;
    font-weight: 700;
    color: white;
    margin-bottom: .6rem;
}
.pred-card .conf-row {
    display: flex;
    align-items: center;
    gap: .6rem;
    font-size: .82rem;
    color: #475569;
    margin-top: .4rem;
}
.conf-bar-bg {
    flex: 1;
    background: #e2e8f0;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 8px;
    border-radius: 999px;
}
.pred-card .desc { font-size: .75rem; color: #94a3b8; margin-bottom: .3rem; }

/* ── KPI metric cards ─────────────────────────────────────────── */
.kpi-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.kpi-card .kpi-val {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1d4ed8;
    line-height: 1.1;
}
.kpi-card .kpi-lbl {
    font-size: .78rem;
    color: #64748b;
    font-weight: 500;
    margin-top: .3rem;
    text-transform: uppercase;
    letter-spacing: .04em;
}

/* ── Info box ─────────────────────────────────────────────────── */
.info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: .88rem;
    color: #1e40af;
    margin-bottom: 1rem;
}

/* ── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; }

/* ── Table tweaks ─────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Step card ────────────────────────────────────────────────── */
.step-card {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: .9rem 1.1rem;
    margin-bottom: .5rem;
}
.step-num {
    background: #1d4ed8;
    color: white;
    font-weight: 700;
    font-size: .85rem;
    min-width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading BERTweet models ...")
def load_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = {}
    for name, cfg in MODELS.items():
        tok = AutoTokenizer.from_pretrained(cfg["dir"], use_fast=False)
        mdl = AutoModelForSequenceClassification.from_pretrained(cfg["dir"]).to(device)
        mdl.eval()
        loaded[name] = (tok, mdl)
    return loaded, device


@st.cache_data(show_spinner=False)
def load_eval(results_dir):
    path = os.path.join(results_dir, "eval_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ── Prediction & LIME helpers ──────────────────────────────────────────────────
def predict_proba(texts, tokenizer, model, device, max_len):
    probs = []
    for text in texts:
        enc = tokenizer(text, truncation=True, max_length=max_len,
                        padding="max_length", return_tensors="pt")
        with torch.no_grad():
            out  = model(input_ids=enc["input_ids"].to(device),
                         attention_mask=enc["attention_mask"].to(device))
            prob = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
        probs.append(prob)
    return np.array(probs)


def run_lime(text, tokenizer, model, device, max_len, labels, pred_idx,
             num_features=10, num_samples=150):
    explainer  = LimeTextExplainer(class_names=labels)
    fn         = lambda t: predict_proba(t, tokenizer, model, device, max_len)
    return explainer.explain_instance(
        text, fn, num_features=num_features,
        num_samples=num_samples, labels=[pred_idx]
    )


def pred_card_html(model_name, desc, icon, label, color, confidence):
    bar_w = int(confidence * 100)
    return f"""
    <div class="pred-card">
      <div class="model-name">{model_name}</div>
      <div class="desc">{desc}</div>
      <div class="label-badge" style="background:{color}">{icon} {label.upper()}</div>
      <div class="conf-row">
        <span>Confidence</span>
        <div class="conf-bar-bg">
          <div class="conf-bar-fill" style="width:{bar_w}%;background:{color}"></div>
        </div>
        <span><b>{confidence:.1%}</b></span>
      </div>
    </div>"""


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 .5rem'>
      <div style='font-size:2.2rem'>🧠</div>
      <div style='font-size:1.05rem;font-weight:700;color:#f1f5f9'>NLP Sentiment Suite</div>
      <div style='font-size:.75rem;color:#94a3b8;margin-top:.2rem'>Powered by BERTweet</div>
    </div>
    <hr style='border-color:#1e293b;margin:.5rem 0 1rem'/>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🔍  Text Prediction", "▶️  YouTube Comments", "📊  Evaluation Results"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <hr style='border-color:#1e293b;margin:1rem 0'/>
    <div style='font-size:.78rem;color:#64748b;padding:.2rem 0'>
      <div style='color:#94a3b8;font-weight:600;margin-bottom:.5rem'>MODELS</div>
    """, unsafe_allow_html=True)

    for name, cfg in MODELS.items():
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;padding:.25rem 0;"
            f"font-size:.78rem'>"
            f"<span style='color:#cbd5e1'>{name}</span>"
            f"<span style='color:#34d399;font-weight:600'>{cfg['acc']}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("""
    </div>
    <hr style='border-color:#1e293b;margin:1rem 0'/>
    <div style='font-size:.72rem;color:#475569;text-align:center'>
      vinai/bertweet-base · AdamW<br>Stratified K-Fold · LIME XAI
    </div>
    """, unsafe_allow_html=True)

all_models, device = load_all_models()
page_key = page.split("  ", 1)[-1].strip()


# =============================================================================
#  PAGE 1 — Text Prediction
# =============================================================================
if page_key == "Text Prediction":
    st.markdown("""
    <div class="hero">
      <h1>🔍 Sentiment &amp; Emotion Analysis</h1>
      <p>Enter any text to receive instant predictions from three independent BERTweet
         models, each trained on a different dataset — with word-level LIME explanations.</p>
      <div class="hero-badges">
        <span class="badge">BERTweet</span>
        <span class="badge">IMDB · SST-2 · dair-ai/emotion</span>
        <span class="badge">LIME Explainability</span>
        <span class="badge">86%+ Accuracy</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    text_input = st.text_area(
        "Input Text",
        height=110,
        placeholder="e.g.  I absolutely loved this film — the performances were outstanding!",
    )

    adv_col1, adv_col2, _ = st.columns([2, 2, 4])
    with adv_col1:
        num_features = st.slider("LIME features", 5, 20, 10)
    with adv_col2:
        num_samples = st.slider("LIME samples", 50, 300, 150, step=50)

    run_btn = st.button("Analyze Text", type="primary", use_container_width=False)

    if run_btn:
        if not text_input.strip():
            st.warning("Please enter some text before analyzing.")
            st.stop()

        st.markdown('<div class="sec-header">Model Predictions</div>', unsafe_allow_html=True)
        cols = st.columns(3)

        results_cache = {}
        for col, (model_name, cfg) in zip(cols, MODELS.items()):
            tok, mdl  = all_models[model_name]
            probs     = predict_proba([text_input], tok, mdl, device, cfg["max_len"])[0]
            pred_idx  = int(np.argmax(probs))
            pred_label = cfg["labels"][pred_idx]
            confidence = float(probs[pred_idx])
            results_cache[model_name] = (probs, pred_idx, pred_label, confidence)

            with col:
                st.markdown(
                    pred_card_html(
                        model_name, cfg["desc"],
                        cfg["icons"][pred_idx], pred_label,
                        cfg["colors"][pred_idx], confidence,
                    ),
                    unsafe_allow_html=True,
                )

                # Probability distribution chart
                fig, ax = plt.subplots(figsize=(4, max(1.8, len(cfg["labels"]) * 0.42)))
                fig.patch.set_facecolor("#f8fafc")
                ax.set_facecolor("#f8fafc")
                bars = ax.barh(cfg["labels"][::-1], probs[::-1],
                               color=cfg["colors"][::-1], edgecolor="white", height=0.6)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability", fontsize=8)
                ax.tick_params(labelsize=8)
                ax.spines[["top","right","bottom"]].set_visible(False)
                for bar, p in zip(bars, probs[::-1]):
                    ax.text(min(p + 0.02, 0.88), bar.get_y() + bar.get_height() / 2,
                            f"{p:.0%}", va="center", fontsize=8, color="#374151")
                plt.tight_layout(pad=0.5)
                st.pyplot(fig, use_container_width=True)
                plt.close()

        # LIME explanations
        st.markdown('<div class="sec-header">LIME Word Explanations</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">🔍 <b>How to read LIME:</b> '
            'Green words push the model toward the predicted label. '
            'Red words push it away. Longer bars = stronger influence.</div>',
            unsafe_allow_html=True,
        )

        lime_cols = st.columns(3)
        for col, (model_name, cfg) in zip(lime_cols, MODELS.items()):
            probs, pred_idx, pred_label, _ = results_cache[model_name]
            tok, mdl = all_models[model_name]
            with col:
                st.markdown(f"**{model_name}** → *{cfg['icons'][pred_idx]} {pred_label}*")
                with st.spinner("Computing LIME ..."):
                    exp = run_lime(text_input, tok, mdl, device, cfg["max_len"],
                                   cfg["labels"], pred_idx, num_features, num_samples)

                with st.expander("Interactive explanation", expanded=True):
                    st.components.v1.html(exp.as_html(), height=360, scrolling=True)

                exp_list = exp.as_list(label=pred_idx)
                if exp_list:
                    words   = [e[0] for e in exp_list]
                    weights = [e[1] for e in exp_list]
                    clrs    = ["#22c55e" if w > 0 else "#ef4444" for w in weights]
                    fig2, ax2 = plt.subplots(figsize=(4, max(2, len(words) * 0.4)))
                    fig2.patch.set_facecolor("#f8fafc")
                    ax2.set_facecolor("#f8fafc")
                    ax2.barh(words[::-1], weights[::-1], color=clrs[::-1],
                             edgecolor="white", height=0.6)
                    ax2.axvline(0, color="#94a3b8", linewidth=0.8, linestyle="--")
                    ax2.set_xlabel("LIME Weight", fontsize=8)
                    ax2.tick_params(labelsize=8)
                    ax2.spines[["top","right","bottom"]].set_visible(False)
                    ax2.set_title(f"← against   for {pred_label} →",
                                  fontsize=8, color="#64748b")
                    plt.tight_layout(pad=0.5)
                    st.pyplot(fig2, use_container_width=True)
                    plt.close()


# =============================================================================
#  PAGE 2 — YouTube Comments
# =============================================================================
elif page_key == "YouTube Comments":
    st.markdown("""
    <div class="hero">
      <h1>▶️ YouTube Comment Analyzer</h1>
      <p>Paste any public YouTube video URL to automatically fetch comments and run
         all three NLP models — no API key required.</p>
      <div class="hero-badges">
        <span class="badge">No API Key Required</span>
        <span class="badge">Up to 80 Comments</span>
        <span class="badge">3 Models · CSV Export</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown('<div class="sec-header">How It Works</div>', unsafe_allow_html=True)
    hw_cols = st.columns(4)
    steps = [
        ("1", "Paste URL", "Copy any public YouTube video URL into the box below"),
        ("2", "Fetch Comments", "Up to 80 recent comments are scraped automatically"),
        ("3", "Run Models", "All 3 BERTweet models classify every comment"),
        ("4", "Explore", "Browse the table, charts, and download the CSV"),
    ]
    for col, (num, title, desc) in zip(hw_cols, steps):
        col.markdown(
            f'<div class="step-card"><div class="step-num">{num}</div>'
            f'<div><b style="font-size:.88rem">{title}</b>'
            f'<div style="font-size:.78rem;color:#64748b;margin-top:.2rem">{desc}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sec-header">Analyze a Video</div>', unsafe_allow_html=True)
    url_col, btn_col = st.columns([5, 1])
    url_input    = url_col.text_input("YouTube URL", label_visibility="collapsed",
                                       placeholder="https://www.youtube.com/watch?v=...")
    max_comments = st.slider("Max comments to fetch", 10, 80, 80, step=10)
    fetch_btn    = btn_col.button("Analyze", type="primary", use_container_width=True)

    if fetch_btn:
        if not url_input.strip():
            st.warning("Please paste a YouTube URL.")
            st.stop()

        try:
            from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
        except ImportError:
            st.error("Run `pip install youtube-comment-downloader` first.")
            st.stop()

        prog = st.progress(0, text="Fetching comments ...")
        try:
            dl, comments = YoutubeCommentDownloader(), []
            for c in dl.get_comments_from_url(url_input.strip(), sort_by=SORT_BY_RECENT):
                txt = c.get("text", "").strip()
                if txt:
                    comments.append(txt)
                prog.progress(min(len(comments) / max_comments, 0.5),
                              text=f"Fetching ... {len(comments)} comments")
                if len(comments) >= max_comments:
                    break
        except Exception as e:
            st.error(f"Could not fetch comments: {e}")
            st.stop()

        if not comments:
            st.warning("No comments found for this video.")
            st.stop()

        prog.progress(0.55, text="Running predictions ...")
        rows = []
        for i, text in enumerate(comments):
            row = {"#": i + 1, "Comment": text[:130] + ("…" if len(text) > 130 else "")}
            for model_name, cfg in MODELS.items():
                tok, mdl = all_models[model_name]
                probs    = predict_proba([text], tok, mdl, device, cfg["max_len"])[0]
                pi       = int(np.argmax(probs))
                row[model_name]           = f"{cfg['icons'][pi]} {cfg['labels'][pi]}"
                row[f"{model_name} conf"] = f"{probs[pi]:.0%}"
            rows.append(row)
            prog.progress(0.55 + 0.45 * (i + 1) / len(comments),
                          text=f"Classifying comment {i+1}/{len(comments)} ...")

        prog.empty()
        df = pd.DataFrame(rows)

        # Summary KPIs
        st.markdown('<div class="sec-header">Summary</div>', unsafe_allow_html=True)
        kpi_cols = st.columns(4)
        kpi_cols[0].markdown(
            f'<div class="kpi-card"><div class="kpi-val">{len(comments)}</div>'
            f'<div class="kpi-lbl">Comments Analyzed</div></div>',
            unsafe_allow_html=True,
        )
        for kpi_col, (model_name, cfg) in zip(kpi_cols[1:], MODELS.items()):
            counts  = df[model_name].value_counts()
            top_raw = counts.index[0] if len(counts) else "—"
            top_lbl = top_raw.split(" ", 1)[-1] if " " in top_raw else top_raw
            pct     = counts.iloc[0] / len(comments) if len(counts) else 0
            kpi_col.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-val" style="font-size:1.3rem">{top_raw}</div>'
                f'<div class="kpi-lbl">{model_name} top ({pct:.0%})</div></div>',
                unsafe_allow_html=True,
            )

        # Full results table
        st.markdown('<div class="sec-header">All Predictions</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=420, hide_index=True)

        # Distribution charts
        st.markdown('<div class="sec-header">Prediction Distribution</div>', unsafe_allow_html=True)
        chart_cols = st.columns(3)
        for col, (model_name, cfg) in zip(chart_cols, MODELS.items()):
            counts = df[model_name].value_counts()
            labels_raw = [f"{ic} {lb}" for ic, lb in zip(cfg["icons"], cfg["labels"])]
            vals   = [counts.get(f"{ic} {lb}", 0)
                      for ic, lb in zip(cfg["icons"], cfg["labels"])]

            fig, ax = plt.subplots(figsize=(4, 3.2))
            fig.patch.set_facecolor("#f8fafc")
            ax.set_facecolor("#f8fafc")
            bars = ax.barh(labels_raw[::-1], vals[::-1],
                           color=cfg["colors"][::-1], edgecolor="white", height=0.55)
            ax.set_xlabel("Comments", fontsize=8)
            ax.spines[["top","right","bottom"]].set_visible(False)
            ax.tick_params(labelsize=8)
            for bar, v in zip(bars, vals[::-1]):
                if v > 0:
                    ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                            str(v), va="center", fontsize=8)
            ax.set_title(model_name, fontsize=9, fontweight="bold", color="#0f172a")
            plt.tight_layout(pad=0.5)
            col.pyplot(fig, use_container_width=True)
            plt.close()

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download CSV", csv,
                           "youtube_predictions.csv", "text/csv",
                           use_container_width=False)


# =============================================================================
#  PAGE 3 — Evaluation Results
# =============================================================================
elif page_key == "Evaluation Results":
    st.markdown("""
    <div class="hero">
      <h1>📊 Model Evaluation Results</h1>
      <p>Detailed performance metrics for all three BERTweet models evaluated on
         their respective held-out validation sets.</p>
      <div class="hero-badges">
        <span class="badge">Stratified K-Fold CV</span>
        <span class="badge">Classification Report</span>
        <span class="badge">Confusion Matrix</span>
        <span class="badge">LIME Validated</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Cross-model summary
    st.markdown('<div class="sec-header">Performance Overview</div>', unsafe_allow_html=True)
    ov_cols = st.columns(3)
    for col, (model_name, cfg) in zip(ov_cols, MODELS.items()):
        res = load_eval(cfg["results"])
        if res:
            col.markdown(
                f'<div class="kpi-card">'
                f'<div style="font-size:.78rem;font-weight:600;color:#64748b;'
                f'text-transform:uppercase;letter-spacing:.05em;margin-bottom:.4rem">'
                f'{model_name}</div>'
                f'<div class="kpi-val">{res["test_accuracy"]:.1%}</div>'
                f'<div class="kpi-lbl">Validation Accuracy</div>'
                f'<div style="margin-top:.6rem;font-size:.8rem;color:#94a3b8">'
                f'CV: {res["best_cv_acc"]:.1%} &nbsp;|&nbsp; LR: {float(res["best_lr"]):.0e}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    tabs = st.tabs([f"{MODELS[m]['icons'][1]}  {m}" for m in MODELS])

    for tab, (model_name, cfg) in zip(tabs, MODELS.items()):
        res = load_eval(cfg["results"])
        with tab:
            if res is None:
                st.warning(f"No results found for {model_name}. Run train.py first.")
                continue

            report      = res.get("classification_report", {})
            label_names = cfg["labels"]

            # KPI row
            k1, k2, k3, k4 = st.columns(4)
            macro_f1 = report.get("macro avg", {}).get("f1-score", 0)
            weighted_f1 = report.get("weighted avg", {}).get("f1-score", 0)
            k1.markdown(
                f'<div class="kpi-card"><div class="kpi-val">{res["test_accuracy"]:.1%}</div>'
                f'<div class="kpi-lbl">Val Accuracy</div></div>', unsafe_allow_html=True)
            k2.markdown(
                f'<div class="kpi-card"><div class="kpi-val">{macro_f1:.3f}</div>'
                f'<div class="kpi-lbl">Macro F1</div></div>', unsafe_allow_html=True)
            k3.markdown(
                f'<div class="kpi-card"><div class="kpi-val">{res["best_cv_acc"]:.1%}</div>'
                f'<div class="kpi-lbl">Best CV Acc</div></div>', unsafe_allow_html=True)
            k4.markdown(
                f'<div class="kpi-card"><div class="kpi-val">{float(res["best_lr"]):.0e}</div>'
                f'<div class="kpi-lbl">Best LR</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            left_col, right_col = st.columns([1, 1])

            with left_col:
                # Classification report table
                st.markdown('<div class="sec-header">Classification Report</div>',
                            unsafe_allow_html=True)
                rpt_rows = []
                for cls in label_names + ["macro avg", "weighted avg"]:
                    if cls in report:
                        r = report[cls]
                        rpt_rows.append({
                            "Class":     cls,
                            "Precision": round(r["precision"], 3),
                            "Recall":    round(r["recall"],    3),
                            "F1":        round(r["f1-score"],  3),
                            "Support":   int(r["support"]),
                        })
                rpt_df = pd.DataFrame(rpt_rows)
                st.dataframe(
                    rpt_df.style.background_gradient(
                        subset=["Precision","Recall","F1"], cmap="Blues", vmin=0, vmax=1
                    ).format({"Precision":"{:.3f}","Recall":"{:.3f}","F1":"{:.3f}"}),
                    use_container_width=True, hide_index=True,
                )

                # CV table
                st.markdown('<div class="sec-header">Cross-Validation</div>',
                            unsafe_allow_html=True)
                cv_rows = []
                for lr_str, cv in res.get("cv_results", {}).items():
                    for i, acc in enumerate(cv["fold_accs"]):
                        cv_rows.append({"LR": lr_str, "Fold": f"Fold {i+1}",
                                        "Accuracy": round(acc, 4)})
                    cv_rows.append({"LR": lr_str, "Fold": "Mean",
                                    "Accuracy": round(cv["mean_acc"], 4)})
                st.dataframe(pd.DataFrame(cv_rows), use_container_width=True,
                             hide_index=True)

            with right_col:
                # Confusion matrix
                st.markdown('<div class="sec-header">Confusion Matrix</div>',
                            unsafe_allow_html=True)
                cm_png = os.path.join(cfg["results"], "confusion_matrix.png")
                cm_npy = os.path.join(cfg["results"], "confusion_matrix.npy")
                if os.path.exists(cm_png):
                    st.image(cm_png, use_container_width=True)
                elif os.path.exists(cm_npy):
                    cm  = np.load(cm_npy)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=label_names, yticklabels=label_names,
                                ax=ax, linewidths=0.5)
                    ax.set_ylabel("True Label", fontsize=9)
                    ax.set_xlabel("Predicted Label", fontsize=9)
                    plt.xticks(rotation=40, ha="right", fontsize=8)
                    plt.yticks(fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

            # Per-class F1 chart
            st.markdown('<div class="sec-header">Per-Class F1 Score</div>',
                        unsafe_allow_html=True)
            f1s  = [report[c]["f1-score"] for c in label_names if c in report]
            clrs = cfg["colors"][:len(f1s)]
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor("#f8fafc")
            ax.set_facecolor("#f8fafc")
            bars = ax.bar(
                [f"{ic} {lb}" for ic, lb in zip(cfg["icons"], label_names[:len(f1s)])],
                f1s, color=clrs, edgecolor="white", width=0.55,
            )
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("F1 Score", fontsize=9)
            ax.axhline(0.85, color="#94a3b8", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.spines[["top","right"]].set_visible(False)
            ax.tick_params(labelsize=9)
            for bar, f1 in zip(bars, f1s):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{f1:.2f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color="#0f172a")
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Sample predictions
            st.markdown('<div class="sec-header">Sample Validation Predictions</div>',
                        unsafe_allow_html=True)
            if "val_texts" in res:
                n = min(25, len(res["val_texts"]))
                srows = []
                for i in range(n):
                    pred = label_names[res["val_preds"][i]]
                    true = label_names[res["val_labels"][i]]
                    srows.append({
                        "Text":      res["val_texts"][i][:110] + "…",
                        "True":      f"{cfg['icons'][res['val_labels'][i]]} {true}",
                        "Predicted": f"{cfg['icons'][res['val_preds'][i]]} {pred}",
                        "Result":    "✅" if pred == true else "❌",
                    })
                df_s = pd.DataFrame(srows)
                st.dataframe(df_s, use_container_width=True,
                             height=380, hide_index=True)
