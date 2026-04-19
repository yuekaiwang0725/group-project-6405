# EE6405 : Team Sentiment Model Comparison App

An interactive Streamlit dashboard comparing all five teammate contributions to the
EE6405 sentiment classification project across three datasets (IMDb, SST-2, Emotion).

## Tabs

1. **Overall Comparison** : side-by-side F1 / accuracy across every model × dataset,
   best-model-per-dataset breakdowns, and an efficiency frontier (F1 vs params).
2. **LoRA (Shivaangii Jaiswal)** : LoRA fine-tuning on RoBERTa-large / BERTweet /
   DistilBERT at rank 8 and 16. Includes rank comparison, per-epoch learning curves,
   parameter efficiency, bf16→fp16 NaN case study, and per-class F1/precision/recall.
3. **BERTweet Full FT (WaiYarAung)** : full fine-tuning of BERTweet at LR 2e-5.
4. **Bi-GRU + Attention (YU JUNCHENG)** : from-scratch recurrent baseline.
5. **BiLSTM + Attention (joannasj)** : from-scratch BiLSTM with attention.
6. **TF-IDF + Logistic Regression** : classical baseline.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

App opens at http://localhost:8501

## Folder structure

```
comparison_app/
├── app.py                          # main Streamlit entry point
├── requirements.txt
├── README.md
├── tabs/                           # one module per tab
│   ├── overall.py
│   ├── lora.py
│   ├── bertweet_full.py
│   ├── gru.py
│   ├── bilstm.py
│   └── logreg.py
├── data/                           # all CSVs and JSON metrics
│   ├── overall_metrics.csv
│   ├── lora_efficiency_frontier.csv
│   ├── lora_*_bundle.json          # 12 runs
│   ├── lora_*_train_log.csv        # per-epoch training logs
│   ├── WaiYarAung_*_classification_report.json
│   ├── WaiYarAung_*_lime_explanations.json
│   ├── *_report.json               # GRU reports
│   ├── bilstm_classification_reports.json  # transcribed from notebook
│   └── logistic_regression_*.json
└── assets/                         # figures (PNG/JPG) bundled with each model
    ├── lora/
    ├── bertweet/
    ├── gru/
    ├── bilstm/
    └── logreg/
```

## Notes on test-set sizes

Different teammates used different test splits. The **Overall Comparison** tab lists
the test-set size under each cell so the numbers can be read in context. For
apples-to-apples comparison on the standard HuggingFace splits, use Emotion (2000)
and SST-2 dev (872) : both are used consistently across runs.
