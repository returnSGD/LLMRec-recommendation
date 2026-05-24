#!/bin/bash
# Train baseline model (no sample engineering)
# Usage: bash scripts/train_base.sh

PYTHON=/c/Users/22084/miniconda3/python

echo "Training baseline model..."
$PYTHON trainer.py \
    --config config/config.yaml \
    --mode base \
    --data_dir data/processed \
    --output_dir checkpoints/base \
    --device cuda
