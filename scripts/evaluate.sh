#!/bin/bash
# Evaluate a trained model checkpoint
# Usage: bash scripts/evaluate.sh [checkpoint_path] [mode_name]

PYTHON=/c/Users/22084/miniconda3/python
CKPT=${1:-"checkpoints/full/final_model.pt"}
MODE=${2:-"full"}

echo "Evaluating checkpoint: $CKPT"
echo "Mode: $MODE"

$PYTHON evaluate.py \
    --checkpoint "$CKPT" \
    --data_dir data/processed \
    --batch_size 8 \
    --output "results/${MODE}_metrics.json"
