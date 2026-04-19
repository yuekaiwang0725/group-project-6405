# Project Plan

## Objective

Deliver a one-week sentiment analysis project with strong visualization and clear model comparison.

## Scope

- English sentiment classification
- Baseline model + DistilBERT
- Cross-domain and robustness analysis
- Explainability and Streamlit demo

## Execution Order

1. `python -m src.main prepare-data`
2. `python -m experiments.run_baseline --dataset imdb`
3. `python -m experiments.run_baseline --dataset sst2`
4. `python -m experiments.run_distilbert --dataset imdb`
5. `python -m experiments.run_distilbert --dataset sst2`
6. `python -m experiments.run_cross_domain`
7. `python -m experiments.run_robustness`
8. `streamlit run demo/gui_yuekai.py`

## Must-have Outputs

- `results/tables/baseline_imdb_metrics.csv`
- `results/tables/distilbert_imdb_test_metrics.csv`
- `results/tables/metrics_cross_domain.csv`
- `results/tables/robustness_scores.csv`
- `results/figures/*_confusion_matrix.png`
- `results/figures/*_training_curves.png`
- `results/figures/*_cross_domain_heatmap.png`
- `results/figures/robustness_drop_curve.png`
