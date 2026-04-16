import os
import json
# Force allow duplicate OpenMP libs to prevent library conflict crashes
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from config import Config
from model import GRUAttention
from dataset import get_dataloader

# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. Resource Loading & Caching
# ==========================================
@st.cache_resource 
def load_all_assets(task_name):
    config = Config(task_name)
    # Reconstruct vocabulary from training data
    _, _, vocab = get_dataloader(config.train_csv, config.val_csv, config)
    
    # Initialize Model
    model = GRUAttention(config).to(DEVICE)
    if os.path.exists(config.model_save_path):
        model.load_state_dict(torch.load(config.model_save_path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model, vocab, config

# ==========================================
# 2. UI Configuration & Sidebar
# ==========================================
st.set_page_config(page_title="NLP Analysis Platform", layout="wide")

st.sidebar.title("🛠️ Control Panel")
task_choice = st.sidebar.selectbox(
    "Select Task", 
    ['emotion', 'sst2', 'imdb'],
    format_func=lambda x: {
        'emotion': 'Emotion Recognition (6-class)', 
        'imdb': 'Movie Reviews (Binary)', 
        'sst2': 'Sentiment Analysis (Binary)'
    }[x]
)

st.sidebar.markdown("---")
with st.sidebar.expander("📚 Edit References", expanded=False):
    ref_text = st.text_area(
        "Modify references here:", 
        "1. Vaswani, et al. 'Attention is All You Need'.\n2. Bahdanau, et al. 'Neural Machine Translation by Jointly Learning to Align and Translate'.",
        height=150
    )

model, vocab, config = load_all_assets(task_choice)

label_configs = {
    'emotion': {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'},
    'imdb': {0: 'Negative', 1: 'Positive'},
    'sst2': {0: 'Negative', 1: 'Positive'}
}
current_labels = label_configs[task_choice]

# ==========================================
# 3. Main Interface
# ==========================================
st.title("🚀 Bi-GRU + Attention Sentiment Analysis Dashboard")

tab_intro, tab_demo, tab_code, tab_ref = st.tabs([
    "📄 Project Report", "🔮 Interactive Demo", "💻 Source Code", "📖 References"
])

# --- Tab 1: Project Report (Updated with Charts and Tables) ---
with tab_intro:
    st.header("Technical Overview & Performance")
    
    # row 1: Metrics from JSON report
    st.subheader("📊 Model Evaluation Benchmarks")
    try:
        # Path to your report folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        report_path = os.path.join(current_dir, 'report', f"{task_choice}_report.json")
        
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test Accuracy", f"{report_data['test_accuracy']}%")
        m2.metric("Samples Tested", report_data['total_samples'])
        m3.metric("Vocab Size", report_data['vocab_size'])
        m4.metric("Status", "✅ Verified")
    except:
        st.warning(f"⚠️ No evaluation report found for {task_choice}. Please run 'test.py' first.")

    st.markdown("---")

    # row 2: Text Length Distribution
    st.subheader(f"📈 {task_choice.upper()} Dataset Token Distribution")
    col_dist, col_info = st.columns([2, 1])
    
    with col_dist:
        # Logic to generate the distribution chart
        df_train = pd.read_csv(config.train_csv)
        # Use your vocab's clean_text to stay consistent
        lengths = df_train['text'].apply(lambda x: len(vocab.clean_text(x).split()))
        
        fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
        sns.histplot(lengths, kde=True, ax=ax_dist, color='#1f77b4', bins=30)
        ax_dist.set_title(f"{task_choice.upper()} Text Length Distribution", fontsize=14)
        ax_dist.set_xlabel("Token Count")
        ax_dist.set_ylabel("Frequency")
        st.pyplot(fig_dist)

    with col_info:
        st.markdown(f"""
        **Distribution Statistics:**
        - **Mean Length:** {lengths.mean():.2f} tokens
        - **Median Length:** {lengths.median():.2f} tokens
        - **Max Length (Cap):** {config.seq_len} tokens
        
        *Note: Sentences longer than {config.seq_len} tokens are truncated, while shorter ones are padded.*
        """)

    st.markdown("---")
    
    # row 3: Architecture Info
    col_arch, col_params = st.columns(2)
    with col_arch:
        st.markdown("""
        ### 3. Model Architecture
        This project implements a **Bi-GRU + Bahdanau Attention** framework:
        - **Bi-directional GRU**: Captures contextual dependencies from both past and future directions.
        - **Attention Weights**: Quantifies token importance, solving the "bottleneck" problem in long sequences.
        """)
    with col_params:
        st.write("### 4. Training Hyperparameters")
        params_df = pd.DataFrame({
            "Hyperparameter": ["Embedding Dim", "Hidden Dim", "Learning Rate", "Batch Size", "Max Seq Len"],
            "Value": [config.embedding_dim, config.hidden_dim, config.learning_rate, config.batch_size, config.seq_len]
        })
        st.table(params_df)

# --- Tab 2: Interactive Demo ---
with tab_demo:
    st.header("Real-time Inference & Attention Visualization")
    user_text = st.text_area("Enter text for analysis (English):", "this project is surprisingly amazing and i really love the attention part")
    
    if st.button("Run Comprehensive Analysis"):
        text_ids, num_used = vocab.text_to_ids(user_text, config.seq_len)
        tokens = vocab.ids_to_tokens(text_ids, num_used)
        tensor_x = torch.tensor([text_ids], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            outputs, alpha = model(tensor_x)
            probs = torch.softmax(outputs, dim=1)[0]
            weights = alpha[0, :num_used, 0].cpu().numpy()
            pred_id = outputs.argmax().item()
            
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Predicted Label", current_labels[pred_id])
            st.write("Probability Distribution:")
            score_data = pd.DataFrame({
                'Category': [current_labels[i] for i in range(len(current_labels))],
                'Confidence': probs.cpu().numpy()
            })
            st.bar_chart(score_data, x='Category', y='Confidence')
            
        with c2:
            st.write("🔥 **Attention Heatmap**")
            fig, ax = plt.subplots(figsize=(max(num_used * 0.8, 6), 3)) 
            sns.heatmap(np.expand_dims(weights, axis=0), xticklabels=tokens, 
                        yticklabels=False, cmap='Reds', annot=True, fmt=".3f", cbar=False, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            fig.tight_layout() 
            st.pyplot(fig)
            
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
            st.download_button("📥 Download Analysis (PNG)", buf.getvalue(), "analysis.png", "image/png")

# --- Tab 3 & 4 (Keep your original code here) ---
with tab_code:
    st.header("Implementation Details")
    try:
        with open("model.py", "r", encoding="utf-8") as f: code = f.read()
        st.code(code, language="python")
    except Exception as e: st.warning(f"Error loading code: {e}")

with tab_ref:
    st.header("Project Bibliography (IEEE)")
    st.markdown(ref_text)
    st.caption("© 2026 NLP Research Project Portfolio")