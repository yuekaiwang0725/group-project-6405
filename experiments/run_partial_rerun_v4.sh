#!/usr/bin/env bash
# ============================================================================
# Sweep v4 — partial rerun (fp16 fix for RoBERTa-large on small datasets)
# ============================================================================
# v3 kept the same NaN pattern as v2: RoBERTa-large + bf16 + LoRA hits a
# bf16 precision wall around step ~140 on SST-2 and Emotion, regardless of
# whether LR is 2e-4, 1e-4, or lower. The NaN trigger is *bf16 mantissa
# saturation in LayerNorm gradients*, not the learning rate.
#
# v4 fix: switch RoBERTa-large to **fp16** (10-bit mantissa vs bf16's 7-bit),
# drop LR to 5e-5, tighten grad clip to 0.3. HuggingFace Trainer handles
# fp16 loss scaling automatically.
#
# BERTweet + DistilBERT keep bf16 at LR 3e-4 (they work fine there).
#
# If v3 NaN runs are still in results/tables/, this script cleans them
# before rerunning. The 6 "known good" v2 bundles (IMDb RoBERTa-large,
# SST-2 BERTweet+DistilBERT, Emotion BERTweet+DistilBERT) are preserved.
# The 2 "under-trained" v2 runs (BERTweet+DistilBERT IMDb) are also rerun
# at patience=2.
#
# Cost on H100 NVL: ~30-40 min, ~$1.50-2.00 total.
#
# Usage:
#   cd /workspace/option_b_lora_standalone
#   bash experiments/run_partial_rerun_v4.sh 2>&1 | tee results/logs/sweep_v4_partial.log
# ============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p results/logs results/tables results/figures

# ---- Shared hyperparameters ------------------------------------------------
EPOCHS="${EPOCHS:-5}"
PATIENCE="${PATIENCE:-2}"
WARMUP="${WARMUP:-0.06}"
ROBERTA_BS="${ROBERTA_BS:-32}"
BERTWEET_BS="${BERTWEET_BS:-32}"
DISTIL_BS="${DISTIL_BS:-32}"

# ---- RoBERTa-large (fp16 + low LR + tight grad clip) -----------------------
# fp16 has a 10-bit mantissa (vs bf16's 7-bit), better for preserving small
# gradient magnitudes in LayerNorm. LR 5e-5 is still 2.5× full-FT LR (2e-5),
# well within the LoRA range. Grad clip 0.3 is aggressive but necessary.
ROBERTA_LR="${ROBERTA_LR:-5e-5}"
ROBERTA_CLIP="${ROBERTA_CLIP:-0.3}"
ROBERTA_PRECISION="${ROBERTA_PRECISION:---fp16}"

# ---- BERTweet / DistilBERT (unchanged from v3, bf16 works fine) ------------
BERTWEET_LR="${BERTWEET_LR:-3e-4}"
DISTIL_LR="${DISTIL_LR:-3e-4}"
SMALL_CLIP="${SMALL_CLIP:-1.0}"
SMALL_PRECISION="${SMALL_PRECISION:---bf16}"

# ---- Stale-bundle cleanup --------------------------------------------------
STALE_BUNDLES=(
  "results/tables/lora_roberta_large_r8_sst2_bundle.json"
  "results/tables/lora_roberta_large_r16_sst2_bundle.json"
  "results/tables/lora_roberta_large_r8_emotion_bundle.json"
  "results/tables/lora_roberta_large_r16_emotion_bundle.json"
  "results/tables/lora_bertweet_r8_imdb_bundle.json"
  "results/tables/lora_distilbert_r8_imdb_bundle.json"
)

echo "============================================================"
echo "[v4 partial rerun] cleaning stale bundles"
echo "============================================================"
for b in "${STALE_BUNDLES[@]}"; do
  if [[ -f "$b" ]]; then
    echo "  rm $b"
    rm -f "$b"
  else
    echo "  (not found, skipping) $b"
  fi
done

# ---- RoBERTa-large runs: fp16 + LR 5e-5 + clip 0.3 ------------------------
echo
echo "============================================================"
echo "[v4] 1/6 lora_roberta_large_r8 / sst2  (fp16, LR 5e-5, clip 0.3)"
echo "============================================================"
python -m experiments.run_lora \
  --run-name lora_roberta_large_r8 --dataset sst2 \
  --model-name FacebookAI/roberta-large \
  --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 \
  --target-modules "query,value" \
  --batch-size "$ROBERTA_BS" --epochs "$EPOCHS" --early-stopping-patience "$PATIENCE" --max-length 128 \
  --warmup-ratio "$WARMUP" --max-grad-norm "$ROBERTA_CLIP" \
  --learning-rate "$ROBERTA_LR" $ROBERTA_PRECISION

echo
echo "============================================================"
echo "[v4] 2/6 lora_roberta_large_r16 / sst2  (fp16, LR 5e-5, clip 0.3)"
echo "============================================================"
python -m experiments.run_lora \
  --run-name lora_roberta_large_r16 --dataset sst2 \
  --model-name FacebookAI/roberta-large \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.1 \
  --target-modules "query,value" \
  --batch-size "$ROBERTA_BS" --epochs "$EPOCHS" --early-stopping-patience "$PATIENCE" --max-length 128 \
  --warmup-ratio "$WARMUP" --max-grad-norm "$ROBERTA_CLIP" \
  --learning-rate "$ROBERTA_LR" $ROBERTA_PRECISION

echo
echo "============================================================"
echo "[v4] 3/6 lora_roberta_large_r8 / emotion  (fp16, LR 5e-5, clip 0.3)"
echo "============================================================"
python -m experiments.run_lora \
  --run-name lora_roberta_large_r8 --dataset emotion \
  --model-name FacebookAI/roberta-large \
  --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 \
  --target-modules "query,value" \
  --batch-size "$ROBERTA_BS" --epochs "$EPOCHS" --early-stopping-patience "$PATIENCE" --max-length 128 \
  --warmup-ratio "$WARMUP" --max-grad-norm "$ROBERTA_CLIP" \
  --learning-rate "$ROBERTA_LR" $ROBERTA_PRECISION

echo
echo "============================================================"
echo "[v4] 4/6 lora_roberta_large_r16 / emotion  (fp16, LR 5e-5, clip 0.3)"
echo "============================================================"
python -m experiments.run_lora \
  --run-name lora_roberta_large_r16 --dataset emotion \
  --model-name FacebookAI/roberta-large \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.1 \
  --target-modules "query,value" \
  --batch-size "$ROBERTA_BS" --epochs "$EPOCHS" --early-stopping-patience "$PATIENCE" --max-length 128 \
  --warmup-ratio "$WARMUP" --max-grad-norm "$ROBERTA_CLIP" \
  --learning-rate "$ROBERTA_LR" $ROBERTA_PRECISION

# ---- BERTweet + DistilBERT IMDb: keep bf16, just patience=2 ----------------
echo
echo "============================================================"
echo "[v4] 5/6 lora_bertweet_r8 / imdb  (bf16, LR 3e-4, patience 2)"
echo "============================================================"
python -m experiments.run_lora \
  --run-name lora_bertweet_r8 --dataset imdb \
  --model-name vinai/bertweet-base \
  --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 \
  --target-modules "query,value" \
  --batch-size "$BERTWEET_BS" --epochs "$EPOCHS" --early-stopping-patience "$PATIENCE" --max-length 128 \
  --warmup-ratio "$WARMUP" --max-grad-norm "$SMALL_CLIP" \
  --learning-rate "$BERTWEET_LR" $SMALL_PRECISION

echo
echo "============================================================"
echo "[v4] 6/6 lora_distilbert_r8 / imdb  (bf16, LR 3e-4, patience 2)"
echo "============================================================"
python -m experiments.run_lora \
  --run-name lora_distilbert_r8 --dataset imdb \
  --model-name distilbert-base-uncased \
  --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 \
  --target-modules "q_lin,v_lin" \
  --batch-size "$DISTIL_BS" --epochs "$EPOCHS" --early-stopping-patience "$PATIENCE" --max-length 128 \
  --warmup-ratio "$WARMUP" --max-grad-norm "$SMALL_CLIP" \
  --learning-rate "$DISTIL_LR" $SMALL_PRECISION

# ---- Re-aggregate ----------------------------------------------------------
echo
echo "============================================================"
echo "[v4] Re-aggregating all 12 bundles into efficiency_frontier.csv"
echo "============================================================"
python -m experiments.aggregate_efficiency

echo
echo "Partial rerun complete."
echo "  CSV    : results/tables/efficiency_frontier.csv"
echo "  Figures: results/figures/efficiency_frontier_*.png"
