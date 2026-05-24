#!/bin/bash
# Preprocess Steam data for LLM-Rec
# Usage: bash scripts/preprocess.sh [--quick]

PYTHON=/c/Users/22084/miniconda3/python
DATA_DIR="data"

if [ "$1" = "--quick" ]; then
    echo "Quick preprocessing (first 100K reviews only)..."
    $PYTHON preprocess.py --data_dir $DATA_DIR --k_core 5 --max_records 100000
else
    echo "Full preprocessing..."
    $PYTHON preprocess.py --data_dir $DATA_DIR --k_core 5
fi
