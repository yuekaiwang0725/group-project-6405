#!/usr/bin/env bash
set -e

echo "===== 1/4 Creating virtual environment ====="
python3 -m venv .venv
source .venv/bin/activate

echo "===== 2/4 Installing dependencies ====="
pip install -r requirements.txt

echo "===== 3/4 Preparing data & running experiments ====="
python3 -m src.main prepare-data
python3 -m experiments.run_baseline --dataset imdb
python3 -m experiments.run_baseline --dataset sst2
python3 -m experiments.run_baseline --dataset emotion
python3 -m experiments.run_distilbert --dataset imdb --epochs 3 --batch-size 16 --max-length 128
python3 -m experiments.run_distilbert --dataset sst2 --epochs 3 --batch-size 16 --max-length 128
python3 -m experiments.run_distilbert --dataset emotion --epochs 3 --batch-size 16 --max-length 128
python3 -m experiments.run_cross_domain
python3 -m experiments.run_robustness

echo "===== 3b/4 Running Logistic Regression + SHAP (thegreatkevin) ====="
python3 -m experiments.run_Logistic_Regression --dataset imdb --max-features 10000
python3 -m experiments.run_Logistic_Regression --dataset sst2 --max-features 10000

echo "===== 4/4 Launching GUI demos ====="
echo "Main dashboard (SVM + DistilBERT) will open at http://localhost:8501"
echo "To launch Logistic Regression + SHAP demo, run: streamlit run demo/gui_3.py"
echo "To run BERTweet notebooks (WaiYarAung), run: pip install -r WaiYarAung_requirements.txt && jupyter notebook"
streamlit run demo/gui_demo.py
