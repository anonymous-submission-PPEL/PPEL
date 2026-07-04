#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${1:?Usage: bash scripts/test.sh <results_dir> [gpu_id]}"
GPU="${2:-0}"

python main.py \
    --only_test \
    --results_dir "${RESULTS_DIR}" \
    --gpu "${GPU}"
