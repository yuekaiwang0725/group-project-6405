# BERTweet Text Classification — Group Project

Fine-tuning **BERTweet** (`vinai/bertweet-base`) on three datasets for text classification,
with evaluation metrics, confusion matrices, and LIME explainability.

---

## Project Structure

```
project/
│
├── bertweet_imdb.ipynb        ← Notebook: IMDB binary sentiment
├── bertweet_sst2.ipynb        ← Notebook: SST-2 binary sentiment
├── bertweet_emotion.ipynb     ← Notebook: Emotion 6-class classification
│
├── saved_outputs/
│   ├── imdb/
│   │   ├── model/                  ← Fine-tuned model weights + tokenizer
│   │   ├── class_names.json        ← ["neg", "pos"]
│   │   ├── classification_report.json
│   │   ├── confusion_matrix.png
│   │   └── lime_explanations.json
│   ├── sst2/
│   │   └── ...                     ← Same structure as above
│   └── emotion/
│       └── ...                     ← Same structure as above
│
├── requirements.txt
└── README.md
```

---

## Datasets

All datasets are loaded automatically from the **Hugging Face Hub** — no manual download needed.

| Notebook | Dataset | HF Identifier | Task | Classes |
|---|---|---|---|---|
| `bertweet_imdb.ipynb` | IMDB | `imdb` | Binary sentiment | neg, pos |
| `bertweet_sst2.ipynb` | SST-2 | `glue` / `sst2` | Binary sentiment | negative, positive |
| `bertweet_emotion.ipynb` | Emotion | `dair-ai/emotion` | Multiclass emotion | sadness, joy, love, anger, fear, surprise |

---

## Model

**BERTweet** — `vinai/bertweet-base`

- RoBERTa-based architecture pre-trained on 850 million English tweets
- Tweet normalisation applied: user mentions → `@USER`, URLs → `HTTPURL`
- Max sequence length: 128 tokens
- Fine-tuned with AdamW optimizer, linear LR schedule, 3 epochs

> **Domain note**: BERTweet is designed for tweet-style text. IMDB and SST-2 are formal
> movie-review datasets — a domain mismatch exists and is discussed in each notebook.
> The Emotion dataset is the best natural fit for BERTweet.

---

## Option A — View Results Without Running Anything

The `saved_outputs/` folder contains all pre-computed results:

- **`classification_report.json`** — precision, recall, F1, accuracy per class
- **`confusion_matrix.png`** — visual confusion matrix
- **`lime_explanations.json`** — 5 pre-computed LIME word-importance explanations

Open these files directly, or launch the Streamlit app (see Option C below).

---

## Option B — Re-run the Notebooks

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch Jupyter

```bash
jupyter notebook
```

### 3. Open and run

Open any of the three `.ipynb` files and run **Kernel → Restart & Run All**.

> **Note**: Training takes approximately:
> - ~30–60 min per notebook on CPU
> - ~5–10 min per notebook on GPU
>
> An internet connection is required on first run to download the dataset and model from Hugging Face Hub.

### Known API fixes (already applied in notebooks)

| Issue | Fix |
|---|---|
| `evaluation_strategy` removed in transformers ≥ 4.41 | Replaced with `eval_strategy` |
| `warmup_ratio` deprecated in transformers ≥ 5.2 | Replaced with `warmup_steps` |
| `tokenizer` arg removed from `Trainer` in transformers 5.x | Replaced with `processing_class` |
| `trainer.evaluate()` fails without prior `train()` | Replaced with `trainer.predict()` |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Model | `vinai/bertweet-base` |
| Max sequence length | 128 |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Optimizer | AdamW |
| LR schedule | Linear with warm-up |
| Validation split | Stratified 80/20 hold-out |

---

## Requirements

See `requirements.txt` for the full list. Key libraries:

- `transformers >= 5.0`
- `datasets`
- `torch`
- `scikit-learn`
- `lime`

---

## Explainability — LIME

Each notebook includes a LIME (Local Interpretable Model-agnostic Explanations) section
that explains individual predictions by highlighting which words most influenced the model's decision.

- **Green bars** — words that support the predicted class
- **Red bars** — words that oppose the predicted class

Pre-computed explanations for 5 validation samples per dataset are saved in
`saved_outputs/<dataset>/lime_explanations.json` and visualised in the Streamlit app.

---

## Authors

Group project — NLP / Machine Learning course
