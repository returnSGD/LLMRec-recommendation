#!/bin/bash
# ============================================================
# Component Ablation + Hyperparameter Sensitivity
# Goodreads Cross-Dataset Validation on Tesla V100 16GB
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  LLM-Rec Component Ablation on Goodreads"
echo "  GPU: Tesla V100 16GB | Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# Ensure preprocessed data exists
if [ ! -f "data/goodreads_processed/train.json" ]; then
    echo "Preprocessed data not found. Running preprocessing first..."
    python preprocess_goodreads.py \
        --data_dir data \
        --output_dir data/goodreads_processed \
        --k_core 5 --min_seq_len 5 --max_seq_len 50
fi

# Full ablation on Goodreads
python scripts/run_component_ablation.py \
    --dataset goodreads \
    --epochs 10

echo ""
echo "Ablation complete! Results in results/ablation/goodreads/"
