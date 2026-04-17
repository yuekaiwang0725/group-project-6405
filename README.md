# EE6405 Sentiment Project

Group project for EE6405 NLP assignment.

## Project Goal

Build a trustworthy English sentiment classification system with:

- A traditional baseline: TF-IDF + SVM
- A modern model: DistilBERT
- Cross-domain evaluation: IMDb <-> SST-2
- Robustness checks: simple text perturbations
- Explainability: token/word-level attribution
- Lightweight GUI demo for visualization

## Suggested Directory Structure

```text
ee6405-group/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── artifacts/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── robustness/
│   ├── explainability/
│   ├── visualization/
│   └── utils/
├── experiments/
├── notebooks/
├── results/
│   ├── figures/
│   ├── tables/
│   └── logs/
├── checkpoints/
├── demo/
├── docs/
├── report/
├── slides/
└── video/
```

## Quick Start

> **Prerequisites:** Python 3.10+ is required.

**One-command setup** (creates venv, installs deps, runs experiments, launches demo):

```bash
bash setup.sh
```

Or follow the steps below manually:

### 1) Create environment

```bash
python3 -m venv .venv
```

Activate the virtual environment:

- **macOS / Linux:** `source .venv/bin/activate`
- **Windows (cmd):** `.venv\Scripts\activate.bat`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`

Then install dependencies:

```bash
pip install -r requirements.txt
```

### 2) Prepare data

```bash
python3 -m src.main prepare-data
```

### 3) Run experiments

```bash
python3 -m experiments.run_baseline --dataset imdb
python3 -m experiments.run_baseline --dataset sst2
python3 -m experiments.run_distilbert --dataset imdb --epochs 3 --batch-size 16 --max-length 128
python3 -m experiments.run_distilbert --dataset sst2 --epochs 3 --batch-size 16 --max-length 128
python3 -m experiments.run_cross_domain
python3 -m experiments.run_robustness
```
Logistic Regression + SHAP
```bash
python3 -m experiments.run_Logistic_Regression --dataset imdb --max-features 10000
python3 -m experiments.run_Logistic_Regression --dataset sst2 --max-features 10000
```
Generated outputs:

- Processed datasets: `data/processed/*.csv`
- Statistics and labels: `data/artifacts/*.json`
- Model checkpoints: `checkpoints/baseline/*` and `checkpoints/distilbert/*`
- Metrics tables: `results/tables/*.csv` and `results/tables/*.json`
- Figures: `results/figures/*.png`

Logistic Regression + SHAP
- Processed datasets (Same)
- Statistics and labels (Same)
- Model checkpoints: `checkpoints/logistic_regression/*`
- Metrics tables (Same)
- Figures (Same)

### 4) Launch GUI demo

```bash
streamlit run demo/gui_demo.py
```

The dashboard will open at `http://localhost:8501`.

Logistic Regression + SHAP
```bash
python3 -m streamlit run demo/gui_3.py
```



## Deliverables Checklist

- [ ] Reproducible code (`src/` + `experiments/`)
- [ ] Metrics table (Accuracy / Precision / Recall / F1)
- [ ] Core figures (distribution, confusion matrix, training curves, comparison)
- [ ] GUI demo
- [ ] Report PDF
- [ ] Presentation slides
- [ ] <=10 min video
- [ ] Group reflection + GAI declaration

## Submission Checklist

1. Run data + experiments once from scratch (new terminal session).
2. Confirm key files exist in `results/figures` and `results/tables`.
3. Open GUI and verify prediction/explain/robustness/benchmark tabs.
4. Finalize report, slides, and video from templates.
5. Push all code and docs to GitHub with a clean `git status`.

---



# YU JUNCHEG — GRU with Attention Model

This module implements a Bi-directional GRU with Bahdanau Attention for text classification (emotion, IMDB, SST-2). 
More detail you can find in `gru_attention/README.md`
To run:

**Navigate to the gru_attention directory:**
```bash
cd gru_attention
```

**Install dependencies** (if not already installed):
```bash
pip install -r ../requirements.txt
```

## No train the models:
** Prepare data ** 
The data is from the .\GROUP-PROJECT-6045\data\processed
If you not find or miss any '.csv' data file, you can try this way to rebuild them:
```bash
cd ..
python -m src.main prepare-data
cd gru_attention
```

**Train the models:** 
```bash
python train.py
```

**Evaluate the models and generate reports:**
```bash
python test.py
```

## Have already train the models/ Just want ti use my trained models:
**Launch the interactive demo:**
```bash
streamlit run gru_demo.py
```

**Clean up generated files && data** (optional):
```bash
python cleanup.py
```


# WaiYarAung — BERTweet Text Classification

Fine-tuning **BERTweet** (`vinai/bertweet-base`) on three datasets for text classification,
with evaluation metrics, confusion matrices, and LIME explainability.

## Contribution Structure

```
notebooks/
├── WaiYarAung_bertweet_imdb.ipynb        ← Notebook: IMDB binary sentiment
├── WaiYarAung_bertweet_sst2.ipynb        ← Notebook: SST-2 binary sentiment
└── WaiYarAung_bertweet_emotion.ipynb     ← Notebook: Emotion 6-class classification

checkpoints/
├── bertweet_imdb/      ← Fine-tuned model weights + tokenizer
├── bertweet_sst2/
└── bertweet_emotion/

results/
├── figures/            ← Confusion matrices (WaiYarAung_*_confusion_matrix.png)
└── tables/             ← Classification reports, class names, LIME explanations
```

## Datasets

All datasets are loaded automatically from the **Hugging Face Hub** — no manual download needed.

| Notebook | Dataset | HF Identifier | Task | Classes |
|---|---|---|---|---|
| `WaiYarAung_bertweet_imdb.ipynb` | IMDB | `imdb` | Binary sentiment | neg, pos |
| `WaiYarAung_bertweet_sst2.ipynb` | SST-2 | `glue` / `sst2` | Binary sentiment | negative, positive |
| `WaiYarAung_bertweet_emotion.ipynb` | Emotion | `dair-ai/emotion` | Multiclass emotion | sadness, joy, love, anger, fear, surprise |

## Model

**BERTweet** — `vinai/bertweet-base`

- RoBERTa-based architecture pre-trained on 850 million English tweets
- Tweet normalisation applied: user mentions → `@USER`, URLs → `HTTPURL`
- Max sequence length: 128 tokens
- Fine-tuned with AdamW optimizer, linear LR schedule, 3 epochs

> **Domain note**: BERTweet is designed for tweet-style text. IMDB and SST-2 are formal
> movie-review datasets — a domain mismatch exists and is discussed in each notebook.
> The Emotion dataset is the best natural fit for BERTweet.

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

## Running the Notebooks

### 1. Install dependencies

```bash
pip install -r WaiYarAung_requirements.txt
```

### 2. Launch Jupyter

```bash
jupyter notebook
```

### 3. Open and run

Open any of the three `WaiYarAung_bertweet_*.ipynb` files and run **Kernel → Restart & Run All**.

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

## Pre-computed Results

The `results/` folder contains all pre-computed outputs (no re-training needed):

- **`WaiYarAung_*_classification_report.json`** — precision, recall, F1, accuracy per class
- **`WaiYarAung_*_confusion_matrix.png`** — visual confusion matrix
- **`WaiYarAung_*_lime_explanations.json`** — 5 pre-computed LIME word-importance explanations

## Explainability — LIME

Each notebook includes a LIME (Local Interpretable Model-agnostic Explanations) section
that explains individual predictions by highlighting which words most influenced the model's decision.

- **Green bars** — words that support the predicted class
- **Red bars** — words that oppose the predicted class


