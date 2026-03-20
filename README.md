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
python -m experiments.run_baseline
python -m experiments.run_distilbert
python -m experiments.run_cross_domain
python -m experiments.run_robustness
```

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

