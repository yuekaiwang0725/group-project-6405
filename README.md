# EE6405 Sentiment Project

One-person, one-week project template for EE6405 NLP group assignment.

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

### 1) Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare data

```bash
python -m src.main prepare-data
```

### 3) Run experiments

```bash
python -m experiments.run_baseline --dataset imdb
python -m experiments.run_baseline --dataset sst2
python -m experiments.run_distilbert --dataset imdb --epochs 3 --batch-size 16 --max-length 128
python -m experiments.run_distilbert --dataset sst2 --epochs 3 --batch-size 16 --max-length 128
python -m experiments.run_cross_domain
python -m experiments.run_robustness
```

Generated outputs:

- Processed datasets: `data/processed/*.csv`
- Statistics and labels: `data/artifacts/*.json`
- Model checkpoints: `checkpoints/baseline/*` and `checkpoints/distilbert/*`
- Metrics tables: `results/tables/*.csv` and `results/tables/*.json`
- Figures: `results/figures/*.png`

### 4) Launch GUI demo

```bash
streamlit run demo/gui_demo.py
```

## One-Week Execution Plan

- Day 1: data ingestion, cleaning, EDA plots
- Day 2: baseline TF-IDF + SVM
- Day 3: DistilBERT fine-tuning
- Day 4: cross-domain + robustness experiments
- Day 5: explainability + Streamlit integration
- Day 6: report and slides drafting
- Day 7: video recording and final packaging

## Minimal Deliverables Checklist

- [ ] Reproducible code (`src/` + `experiments/`)
- [ ] Metrics table (Accuracy / Precision / Recall / F1)
- [ ] Core figures (distribution, confusion matrix, training curves, comparison)
- [ ] GUI demo
- [ ] Report PDF
- [ ] Presentation slides
- [ ] <=10 min video script
- [ ] Group reflection + GAI declaration

## Submission Checklist (One-person workflow)

1. Run data + experiments once from scratch (new terminal session).
2. Confirm key files exist in `results/figures` and `results/tables`.
3. Open GUI and verify prediction/explain/robustness/benchmark tabs.
4. Finalize report, slides, and video script from templates.
5. Push all code and docs to GitHub with a clean `git status`.

