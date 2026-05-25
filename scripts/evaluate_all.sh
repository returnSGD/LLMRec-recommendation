#!/bin/bash
# Batch evaluation script for all LLM-Rec experiments
set -e

DATA_DIR="data/processed"
BASE_MODEL="google/flan-t5-small"
DEVICE="cuda"
MAX_EVAL=2000
TOP_K="5 10 20"
RESULTS_DIR="results"

mkdir -p "$RESULTS_DIR"

evaluate_one() {
    local name=$1
    local checkpoint=$2
    local output="$RESULTS_DIR/${name}_metrics.json"

    if [ -f "$checkpoint" ]; then
        echo "============================================"
        echo "Evaluating: $name"
        echo "============================================"
        PYTHONIOENCODING=utf-8 python evaluate.py \
            --checkpoint "$checkpoint" \
            --data_dir "$DATA_DIR" \
            --base_model "$BASE_MODEL" \
            --device "$DEVICE" \
            --top_k $TOP_K \
            --max_eval $MAX_EVAL \
            --output "$output"
        echo "Done: $output"
    else
        echo "SKIP: checkpoint not found: $checkpoint"
    fi
}

echo "Starting batch evaluation..."
echo ""

# Baseline (always available)
evaluate_one "baseline" "checkpoints/base/final_model.pt"

# Ablation experiments
evaluate_one "reccl" "checkpoints/ablation_reccl/final_model.pt"
evaluate_one "sans" "checkpoints/ablation_sans/final_model.pt"
evaluate_one "recaug" "checkpoints/ablation_recaug/final_model.pt"
evaluate_one "full" "checkpoints/full/final_model.pt"

echo ""
echo "Batch evaluation complete!"
echo ""
echo "Generating comparison report..."
PYTHONIOENCODING=utf-8 python results/generate_report.py

echo "Report: results/comprehensive_experiment_report.md"
