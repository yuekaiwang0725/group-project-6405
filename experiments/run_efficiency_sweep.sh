#!/usr/bin/env bash
# Orchestrates the full Option-B efficiency-frontier sweep on a single GPU node.
#
# Intended targets:
#   - H100 NVL (94 GB) : all runs fit comfortably, no gradient_checkpointing needed.
#   - H100 SXM (80 GB) : same, uses NVLink for slightly faster RoBERTa-large.
#   - A100 80GB        : same.
#   - A100 40GB        : add --gradient-checkpointing for RoBERTa-large.
#   - RTX 4090 (24GB)  : set ROBERTA_BS=16 --gradient-checkpointing; expect ~2x wall-clock.
#
# Behaviour: runs each experiment sequentially and is idempotent — if a bundle
# JSON already exists for a given (run_name, dataset) pair the run is skipped.
# Use FORCE=1 to re-run everything.

set -euo pipefail

cd "$(dirname "$0")/.."

DATASETS=("${DATASETS:-imdb sst2 emotion}")
ROBERTA_BS="${ROBERTA_BS:-32}"
BERTWEET_BS="${BERTWEET_BS:-32}"
DISTIL_BS="${DISTIL_BS:-32}"
EPOCHS="${EPOCHS:-5}"
PATIENCE="${PATIENCE:-2}"
WARMUP="${WARMUP:-0.06}"
GRADCLIP="${GRADCLIP:-1.0}"
# Per-model LR (sweep v3):
#   RoBERTa-large is gradient-sensitive in bf16. Use 2e-4 on large datasets
#   (IMDb, 22.5k train) but drop to 1e-4 on smaller datasets (SST-2 67k short
#   sentences, Emotion 16k) where the bf16 mantissa limit + LoRA noise
#   triggered NaN gradients mid-warmup in v2. See report/option_b_runbook.md.
#   BERTweet / DistilBERT are smaller & stable at 3e-4 across all datasets.
ROBERTA_LR_LARGE_DS="${ROBERTA_LR_LARGE_DS:-2e-4}"   # IMDb
ROBERTA_LR_SMALL_DS="${ROBERTA_LR_SMALL_DS:-1e-4}"   # SST-2, Emotion
BERTWEET_LR="${BERTWEET_LR:-3e-4}"
DISTIL_LR="${DISTIL_LR:-3e-4}"

# Pick the RoBERTa-large LR for a given dataset.
roberta_lr_for() {
  local ds="$1"
  case "$ds" in
    imdb)    echo "$ROBERTA_LR_LARGE_DS" ;;
    sst2|emotion) echo "$ROBERTA_LR_SMALL_DS" ;;
    *)       echo "$ROBERTA_LR_SMALL_DS" ;;  # conservative default
  esac
}
EXTRA="${EXTRA:---bf16}"
FORCE="${FORCE:-0}"

has_bundle() {
  local run_name="$1" ds="$2"
  [[ -f "results/tables/${run_name}_${ds}_bundle.json" ]]
}

launch() {
  local run_name="$1"; shift
  local ds="$1"; shift
  if [[ "$FORCE" != "1" ]] && has_bundle "$run_name" "$ds"; then
    echo "[skip] $run_name / $ds (bundle exists, set FORCE=1 to re-run)"
    return
  fi
  echo "============================================================"
  echo "[launch] $run_name / $ds"
  echo "============================================================"
  python -m experiments.run_lora --run-name "$run_name" --dataset "$ds" "$@"
}

mkdir -p results/logs

for DS in ${DATASETS[@]}; do
  RLR="$(roberta_lr_for "$DS")"

  # ── RoBERTa-large LoRA (r=8 and r=16) ──────────────────────
  launch "lora_roberta_large_r8" "$DS" \
    --model-name FacebookAI/roberta-large \
    --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 \
    --target-modules "query,value" \
    --batch-size "$ROBERTA_BS" --epochs "$EPOCHS" --early-stopping-patience "$PATIENCE" --max-length 128 \
    --warmup-ratio "$WARMUP" --max-grad-norm "$GRADCLIP" \
    --learning-rate "$RLR" $EXTRA

  launch "lora_roberta_large_r16" "$DS" \
    --model-name FacebookAI/roberta-large \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.1 \
    --target-modules "query,value" \
    --batch-size "$ROBERTA_BS" --epochs "$EPOCHS" --early-stopping-patience "$PATIENCE" --max-length 128 \
    --warmup-ratio "$WARMUP" --max-grad-norm "$GRADCLIP" \
    --learning-rate "$RLR" $EXTRA

  # ── BERTweet LoRA (complements WaiYarAung's full fine-tune) ─
  launch "lora_bertweet_r8" "$DS" \
    --model-name vinai/bertweet-base \
    --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 \
    --target-modules "query,value" \
    --batch-size "$BERTWEET_BS" --epochs "$EPOCHS" --early-stopping-patience "$PATIENCE" --max-length 128 \
    --warmup-ratio "$WARMUP" --max-grad-norm "$GRADCLIP" \
    --learning-rate "$BERTWEET_LR" $EXTRA

  # ── DistilBERT LoRA (apples-to-apples vs. full-FT) ─────────
  launch "lora_distilbert_r8" "$DS" \
    --model-name distilbert-base-uncased \
    --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 \
    --target-modules "q_lin,v_lin" \
    --batch-size "$DISTIL_BS" --epochs "$EPOCHS" --early-stopping-patience "$PATIENCE" --max-length 128 \
    --warmup-ratio "$WARMUP" --max-grad-norm "$GRADCLIP" \
    --learning-rate "$DISTIL_LR" $EXTRA
done

# Aggregate all bundle.json files into one CSV + produce figures.
python -m experiments.aggregate_efficiency
echo "Sweep complete. See results/tables/efficiency_frontier.csv + results/figures/efficiency_frontier_*.png"
