#!/usr/bin/env bash
# ============================================================================
# Sweep v3 — partial rerun
# ============================================================================
# After v2 completed, 4 runs produced NaN (RoBERTa-large on SST-2 + Emotion)
# and 2 runs early-stopped at epoch 2 due to patience=1 being too aggressive
# (BERTweet + DistilBERT on IMDb).
#
# This script only re-runs those 6. The 6 healthy v2 runs are kept as-is and
# will be picked up by aggregate_efficiency.py at the end.
#
# Changes vs v2:
#   1. RoBERTa-large LR: 2e-4 → 1e-4 on SST-2 and Emotion only
#      (IMDb stays at 2e-4, which was stable and hit F1=0.931).
#   2. early_stopping_patience: 1 → 2 (all runs).
#
# Cost on H100 NVL: ~30-40 min, ~$1.50-2.00 total.
#
# Usage:
#   cd /workspace/option_b_lora_standalone
#   bash experiments/run_partial_rerun.sh 2>&1 | tee results/logs/sweep_v3_partial.log
# ============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p results/logs results/tables results/figures

# ---- Shared hyperparameters (identical to v2 except patience) --------------
EPOCHS="${EPOCHS:-5}"
PATIENCE="${PATIENCE:-2}"           # v3: was 1
WARMUP="${WARMUP:-0.06}"
GRADCLIP="${GRADCLIP:-1.0}"
ROBERTA_BS="${ROBERTA_BS:-32}"
BERTWEET_BS="${BERTWEET_BS:-32}"
DISTIL_BS="${DISTIL_BS:-32}"
EXTRA="${EXTRA:---bf16}"

# RoBERTa-large learning rates (v3)
ROBERTA_LR_IMDB="${ROBERTA_LR_IMDB:-2e-4}"       # unused in this partial script,
                                                 # shown for reference
ROBERTA_LR_SMALL_DS="${ROBERTA_LR_SMALL_DS:-1e-4}"
BERTWEET_LR="${BERTWEET_LR:-3e-4}"
DISTIL_LR="${DISTIL_LR:-3e-4}"

# ---- Stale-bundle cleanup --------------------------------------------------
# Remove the 6 targeted bundles so run_lora.py re-runs them. We DO NOT delete:
#   - lora_roberta_large_r8_imdb_bundle.json  (F1 0.931, keep)
#   - lora_roberta_large_r16_imdb_bundle.json (F1 0.929, keep)
#   - lora_bertweet_r8_sst2_bundle.json       (F1 0.942, keep)
#   - lora_distilbert_r8_sst2_bundle.json     (F1 0.886, keep)
#   - lora_bertweet_r8_emotion_bundle.json    (F1 0.870, keep)
#   - lora_distilbert_r8_emotion_bundle.json  (F1 0.880, keep)

STALE_BUNDLES=(
  "results/tables/lora_roberta_large_r8_sst2_bundle.json"      # NaN in v2
  "results/tables/lora_roberta_large_r16_sst2_bundle.json"     # NaN in v2
  "results/tables/lora_roberta_large_r8_emotion_bundle.json"   # NaN in v2
  "results/tables/lora_roberta_large_r16_emotion_bundle.json"  # NaN in v2
  "results/tables/lora_bertweet_r8_imdb_bundle.json"           # early-stopped at ep2 in v2
  "results/tables/lora_distilbert_r8_imdb_bundle.json"         # early-stopped at ep2 in v2
)

echo "============================================================"
echo "[v3 partial rerun] cleaning stale bundles"
echo "============================================================"
for b in "${STALE_BUNDLES[@]}"; do
  if [[ -f "$b" ]]; then
    echo "  rm $b"
    rm -f "$b"
  else
    echo "  (not found, skipping) $b"
  fi
done

# ---- Run the 6 targeted runs ----------------------------------------------

echo
echo "============================================================"
echo "[v3] 1/6 lora_roberta_large_r8 / sst2  (LR 1e-4, patience 2)"
echo "============================================================"
python -m experiments.run_lora \
  --run-name lora_roberta_large_r8 --dataset sst2 \
  --model-name FacebookAI/roberta-large \
  --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 \
  --target-modules "query,value" \
  --batch-size "$ROBERTA_BS" --epochs "$EPOCHS" --early-stopping-patience "$PATIENCE" --max-length 128 \
  --warmup-ratio "$WARMUP" --max-grad-norm "$GRADCLIP" \
  --learning-rate "$ROBERTA_LR_SMALL_DS" $EXTRA

echo
echo "============================================================"
echo "[v3] 2/6 lora_roberta_large_r16 / sst2  (LR 1e-4, patience 2)"
echo "============================================================"
python -m experiments.run_lora \
  --run-name lora_roberta_large_r16 --dataset sst2 \
  --model-name FacebookAI/roberta-large \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.1 \
  --target-modules "query,value" \
  --batch-size "$ROBERTA_BS" --epochs "$EPOCHS" --early-stopping-patience "$PATIENCE" --max-length 128 \
  --warmup-ratio "$WARMUP" --max-grad-norm "$GRADCLIP" \
  --learning-rate "$ROBERTA_LR_SMALL_DS" $EXTRA

echo
echo "============================================================"
echo "[v3] 3/6 lora_roberta_large_r8 / emotion  (LR 1e-4, patience 2)"
echo "============================================================"
python -m experiments.run_lora \
  --run-name lora_roberta_large_r8 --dataset emotion \
  --model-name FacebookAI/roberta-large \
  --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 \
  --target-modules "query,value" \
  --batch-size "$ROBERTA_BS" --epochs "$EPOCHS" --early-stopping-patience "$PATIENCE" --max-length 128 \
  --warmup-ratio "$WARMUP" --max-grad-norm "$GRADCLIP" \
  --learning-rate "$ROBERTA_LR_SMALL_DS" $EXTRA

echo
echo "============================================================"
echo "[v3] 4/6 lora_roberta_large_r16 / emotion  (LR 1e-4, patience 2)"
echo "============================================================"
python -m experiments.run_lora \
  --run-name lora_roberta_large_r16 --dataset emotion \
  --model-name FacebookAI/roberta-large \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.1 \
  --target-modules "query,value" \
  --batch-size "$ROBERTA_BS" --epochs "$EPOCHS" --early-stopping-patience "$PATIENCE" --max-length 128 \
  --warmup-ratio "$WARMUP" --max-grad-norm "$GRADCLIP" \
  --learning-rate "$ROBERTA_LR_SMALL_DS" $EXTRA

echo
echo "============================================================"
echo "[v3] 5/6 lora_bertweet_r8 / imdb  (LR 3e-4, patience 2)"
echo "============================================================"
python -m experiments.run_lora \
  --run-name lora_bertweet_r8 --dataset imdb \
  --model-name vinai/bertweet-base \
  --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 \
  --target-modules "query,value" \
  --batch-size "$BERTWEET_BS" --epochs "$EPOCHS" --early-stopping-patience "$PATIENCE" --max-length 128 \
  --warmup-ratio "$WARMUP" --max-grad-norm "$GRADCLIP" \
  --learning-rate "$BERTWEET_LR" $EXTRA

echo
echo "============================================================"
echo "[v3] 6/6 lora_distilbert_r8 / imdb  (LR 3e-4, patience 2)"
echo "============================================================"
python -m experiments.run_lora \
  --run-name lora_distilbert_r8 --dataset imdb \
  --model-name distilbert-base-uncased \
  --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 \
  --target-modules "q_lin,v_lin" \
  --batch-size "$DISTIL_BS" --epochs "$EPOCHS" --early-stopping-patience "$PATIENCE" --max-length 128 \
  --warmup-ratio "$WARMUP" --max-grad-norm "$GRADCLIP" \
  --learning-rate "$DISTIL_LR" $EXTRA

# ---- Re-aggregate all 12 bundles (6 kept from v2 + 6 fresh) ----------------
echo
echo "============================================================"
echo "[v3] Re-aggregating all 12 bundles into efficiency_frontier.csv"
echo "============================================================"
python -m experiments.aggregate_efficiency

echo
echo "Partial rerun complete."
echo "  CSV    : results/tables/efficiency_frontier.csv"
echo "  Figures: results/figures/efficiency_frontier_*.png"
echo "  Log    : results/logs/sweep_v3_partial.log (if you tee'd the output)"
