#!/bin/bash
# Train full model with all three sample engineering methods
# Usage: bash scripts/train_full.sh

PYTHON=/c/Users/22084/miniconda3/python

echo "Training full model (RecCL + SANS + RecAug)..."
$PYTHON trainer.py \
    --config config/config.yaml \
    --mode full \
    --data_dir data/processed \
    --output_dir checkpoints/full \
    --device cuda
