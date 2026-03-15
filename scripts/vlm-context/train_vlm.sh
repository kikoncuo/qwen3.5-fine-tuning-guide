#!/usr/bin/env bash
# LoRA fine-tune Qwen3.5-0.8B-8bit for screen context extraction
# Usage: bash scripts/train_vlm.sh

set -euo pipefail

cd "$(dirname "$0")/.."

python3 scripts/train_vlm.py \
  --model-path mlx-community/Qwen3.5-0.8B-8bit \
  --dataset Enriqueag26/keysay-vlm-context-training \
  --iters 200 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --max-seq-length 4096 \
  --grad-checkpoint \
  --steps-per-report 10 \
  --steps-per-eval 50 \
  --steps-per-save 50 \
  --output-path vlm_training/vlm-context-lora-v3
