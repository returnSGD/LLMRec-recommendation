#!/bin/bash
# ============================================================
# LLM-Rec Goodreads — RTX PRO 6000 96GB Server Runner
# ============================================================
# Usage:
#   bash run_server.sh              # Full pipeline (preprocess + base + full + eval)
#   bash run_server.sh --skip-preprocess   # Skip preprocessing (data already processed)
#   bash run_server.sh --mode base  # Train baseline only
#   bash run_server.sh --mode full  # Train full model only (RecCL+SANS+RecAug)
#   bash run_server.sh --eval-only  # Evaluation only
#   bash run_server.sh --quick      # Quick test: 5000 samples, 1 epoch
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── HuggingFace Mirror (for China access) ──
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=1

# ── CUDA memory: reduce fragmentation (see OOM error recommendation) ──
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Default settings ──
CONFIG="config/config_rtx6000.yaml"
DATA_RAW="data"
DATA_PROCESSED="data/goodreads_processed"
MODE="pipeline"       # pipeline | base | full | eval
SKIP_PREPROCESS=false
QUICK_MODE=false
MAX_TRAIN=""
EPOCHS=""
DEEPSEEK_KEY="${DEEPSEEK_API_KEY:-}"

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-preprocess) SKIP_PREPROCESS=true ;;
        --mode) MODE="$2"; shift ;;
        --eval-only) MODE="eval" ;;
        --quick) QUICK_MODE=true ;;
        --epochs) EPOCHS="--epochs $2"; shift ;;
        *) echo "Unknown arg: $1" ;;
    esac
    shift
done

# ── Quick mode overrides ──
if $QUICK_MODE; then
    MAX_TRAIN="--max_train 5000"
    EPOCHS="--epochs 1"
    echo "[Quick mode] 5000 samples, 1 epoch"
fi

echo "============================================"
echo "  LLM-Rec Goodreads — RTX PRO 6000 96GB"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""
echo "  HF_ENDPOINT:  $HF_ENDPOINT"
echo "  Config:       $CONFIG"
echo "  Data dir:     $DATA_PROCESSED"
echo "  DeepSeek key: $(if [ -n "$DEEPSEEK_KEY" ]; then echo '***set***'; else echo 'NOT SET (LLM features will fallback)'; fi)"
echo ""

# ── Step 0: GPU check ──
echo "[GPU] Checking..."
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader 2>/dev/null || {
    echo "ERROR: nvidia-smi not found. Is CUDA installed?"
    exit 1
}
echo ""

# ── Step 1: Install Python dependencies (only if needed) ──
echo "[Install] Checking Python packages..."
pip install -q torch transformers accelerate peft \
    pandas numpy scipy scikit-learn tqdm pyyaml omegaconf matplotlib seaborn joblib \
    sentence-transformers anthropic || echo "[Install] WARNING: some packages may have failed"
echo "[Install] Dependencies OK"
echo ""

# ── Step 2: Preprocess Goodreads data ──
if $SKIP_PREPROCESS; then
    echo "[Preprocess] SKIPPED (data already processed)"
elif [ -f "$DATA_PROCESSED/train.json" ] && [ -f "$DATA_PROCESSED/item_catalog.json" ]; then
    echo "[Preprocess] SKIPPED (processed data already exists)"
    echo "  To force re-run: rm -rf $DATA_PROCESSED"
else
    echo "============================================"
    echo "  STEP: Preprocessing Goodreads Data"
    echo "============================================"
    python preprocess_goodreads.py \
        --data_dir "$DATA_RAW" \
        --output_dir "$DATA_PROCESSED" \
        --k_core 5 \
        --min_seq_len 5 \
        --max_seq_len 50

    # Free disk space: delete raw data (no longer needed)
    echo ""
    echo "[Cleanup] Removing raw data to free disk space..."
    rm -f "$DATA_RAW/goodreads_books.json.gz"
    rm -f "$DATA_RAW/goodreads_book_genres_initial.json.gz"
    rm -f "$DATA_RAW/goodreads_interactions.csv"
    echo "[Cleanup] Freed ~6.1GB"
    echo ""
fi

# ── Count train samples ──
if [ -f "$DATA_PROCESSED/train.json" ]; then
    N_TRAIN=$(python -c "import json; print(len(json.load(open('$DATA_PROCESSED/train.json'))))")
    echo "[Data] Train samples: $N_TRAIN"
fi

# ── Step 3: Train baseline ──
if [ "$MODE" = "pipeline" ] || [ "$MODE" = "base" ]; then
    echo ""
    echo "============================================"
    echo "  STEP: Train Baseline (base mode)"
    echo "============================================"
    python trainer.py \
        --config "$CONFIG" \
        --mode base \
        --data_dir "$DATA_PROCESSED" \
        --output_dir checkpoints/baseline \
        --device cuda \
        $MAX_TRAIN \
        $EPOCHS

    echo "[Baseline] Training complete."
fi

# ── Step 4: Train full model (RecCL + SANS + RecAug) ──
if [ "$MODE" = "pipeline" ] || [ "$MODE" = "full" ]; then
    echo ""
    echo "============================================"
    echo "  STEP: Train Full Model (RecCL+SANS+RecAug)"
    echo "============================================"
    python trainer.py \
        --config "$CONFIG" \
        --mode full \
        --data_dir "$DATA_PROCESSED" \
        --output_dir checkpoints/full \
        --device cuda \
        $MAX_TRAIN \
        $EPOCHS

    echo "[Full] Training complete."
fi

# ── Step 5: Evaluate ──
if [ "$MODE" = "pipeline" ] || [ "$MODE" = "eval" ] || [ "$MODE" = "base" ] || [ "$MODE" = "full" ]; then
    echo ""
    echo "============================================"
    echo "  STEP: Evaluation"
    echo "============================================"
    mkdir -p results

    if [ -f "checkpoints/baseline/final_model.pt" ]; then
        echo "[Eval] Baseline model..."
        python evaluate.py \
            --checkpoint checkpoints/baseline/final_model.pt \
            --data_dir "$DATA_PROCESSED" \
            --config "$CONFIG" \
            --batch_size 64 \
            --top_k 5 10 20 \
            --device cuda \
            --prompt_type book \
            --output results/baseline_metrics.json \
            --save_per_sample results/baseline_per_sample.json
    fi

    if [ -f "checkpoints/full/final_model.pt" ]; then
        echo "[Eval] Full model..."
        python evaluate.py \
            --checkpoint checkpoints/full/final_model.pt \
            --data_dir "$DATA_PROCESSED" \
            --config "$CONFIG" \
            --batch_size 64 \
            --top_k 5 10 20 \
            --device cuda \
            --prompt_type book \
            --output results/full_metrics.json \
            --save_per_sample results/full_per_sample.json
    fi

    echo "[Eval] Complete. Results in results/"
fi

# ── Step 6: Generate experiment report ──
if [ "$MODE" = "pipeline" ] || [ "$MODE" = "eval" ] || [ "$MODE" = "base" ] || [ "$MODE" = "full" ]; then
    echo ""
    echo "============================================"
    echo "  STEP: Generate Experiment Report"
    echo "============================================"
    python scripts/generate_report.py \
        --data_dir "$DATA_PROCESSED" \
        --baseline_metrics results/baseline_metrics.json \
        --full_metrics results/full_metrics.json \
        --config "$CONFIG" \
        --output results/experiment_report.md
fi

echo ""
echo "============================================"
echo "  ALL DONE! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
