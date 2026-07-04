#!/usr/bin/env bash
set -euo pipefail

STUDY="${1:-tcga_stad}"
TYPE_OF_PATH="${2:-combine}"
MODE="${3:-swin}"
GPU="${4:-0}"

case "$MODE" in
    swin)     ENCODING_DIM=768 ;;
    resnet50) ENCODING_DIM=1024 ;;
    cluster)  ENCODING_DIM=1024 ;;
    *)        echo "Unknown mode: $MODE"; exit 1 ;;
esac

LABEL_FILE="./datasets_csv/metadata/${STUDY}.csv"
OMICS_DIR="./datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY##tcga_}/"

python main.py \
    --study "${STUDY}" \
    --type_of_path "${TYPE_OF_PATH}" \
    --omics_format pathways \
    --mode "${MODE}" \
    --encoding_dim "${ENCODING_DIM}" \
    --label_file "${LABEL_FILE}" \
    --omics_dir "${OMICS_DIR}" \
    --gpu "${GPU}"
