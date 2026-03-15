#!/usr/bin/env bash
# LoRA fine-tune Qwen3.5-0.8B for transcription cleaning
# Prerequisites: pip install mlx_lm
# Data: Run generate_training_data.py then format_dataset.py first

set -euo pipefail

python3 -m mlx_lm.lora \
    --model mlx-community/Qwen3.5-0.8B-8bit \
    --data dataset_v2 \
    --train \
    --iters 1000 \
    --batch-size 2 \
    --learning-rate 3e-5 \
    --lora-rank 8 \
    --steps-per-report 50 \
    --steps-per-eval 200 \
    --val-batches 25 \
    --adapter-path transcription-cleaner-lora

echo ""
echo "Training complete! Next steps:"
echo "  1. Fuse:   python3 -m mlx_lm.fuse --model mlx-community/Qwen3.5-0.8B-8bit --adapter-path transcription-cleaner-lora --save-path keysay-transcription-cleaner-0.8B-8bit"
echo "  2. Upload: huggingface-cli upload Enriqueag26/keysay-transcription-cleaner-0.8B-8bit keysay-transcription-cleaner-0.8B-8bit"
