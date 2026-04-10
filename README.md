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

The dashboard will open at `http://localhost:8501`.

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

# group-project-6405
