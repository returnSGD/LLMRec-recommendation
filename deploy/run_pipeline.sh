#!/bin/bash
# ============================================================
# Full Pipeline: Preprocess → Train → Evaluate
# Goodreads Cross-Dataset Validation on Tesla V100 16GB
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  LLM-Rec Goodreads Cross-Dataset Validation"
echo "  GPU: Tesla V100 16GB | Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# Check GPU
echo ""
echo "[Check] GPU status..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "WARNING: nvidia-smi not found"

# Step 1: Preprocessing
echo ""
echo "============================================"
echo "  STEP 1/4: Data Preprocessing"
echo "============================================"
python preprocess_goodreads.py \
    --data_dir data \
    --output_dir data/goodreads_processed \
    --k_core 5 \
    --min_seq_len 5 \
    --max_seq_len 50

# Step 2: Train baseline (flan-t5-base, no sample engineering)
echo ""
echo "============================================"
echo "  STEP 2/4: Train Baseline (base mode)"
echo "============================================"
python trainer.py \
    --config config/config_5090.yaml \
    --mode base \
    --data_dir data/goodreads_processed \
    --output_dir checkpoints/baseline \
    --device cuda

# Step 3: Train full (baseline + RecCL + SANS + RecAug)
echo ""
echo "============================================"
echo "  STEP 3/4: Train Full (RecCL + SANS + RecAug)"
echo "============================================"
python trainer.py \
    --config config/config_5090.yaml \
    --mode full \
    --data_dir data/goodreads_processed \
    --output_dir checkpoints/full \
    --device cuda

# Step 4: Evaluate both models
echo ""
echo "============================================"
echo "  STEP 4/4: Evaluation"
echo "============================================"

mkdir -p results

echo ""
echo "--- Baseline ---"
python evaluate.py \
    --checkpoint checkpoints/baseline/final_model.pt \
    --data_dir data/goodreads_processed \
    --config config/config_5090.yaml \
    --batch_size 16 \
    --top_k 5 10 20 \
    --device cuda \
    --output results/baseline_metrics.json \
    --save_per_sample results/baseline_per_sample.json

echo ""
echo "--- Full Model ---"
python evaluate.py \
    --checkpoint checkpoints/full/final_model.pt \
    --data_dir data/goodreads_processed \
    --config config/config_5090.yaml \
    --batch_size 16 \
    --top_k 5 10 20 \
    --device cuda \
    --output results/full_metrics.json \
    --save_per_sample results/full_per_sample.json

echo ""
echo "============================================"
echo "  PIPELINE COMPLETE!"
echo "  Results: results/"
echo "============================================"
