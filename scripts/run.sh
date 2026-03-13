#!/usr/bin/env bash
set -euo pipefail

# source setup (paths, GPU rename)
source /nethome/rhakim/projects/deepconf/scripts/setup.sh

# activate conda environment
source /nethome/rhakim/miniconda3/etc/profile.d/conda.sh
conda activate deepconf-env

# diagnostics
echo "=== Diagnostics ==="
echo "HOSTNAME: $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi
which python
python --version
echo "==================="

# build command from environment variables
CMD="python -m deepconf.main --model-path ${MODEL_PATH}"

[ -n "${MODE:-}" ]            && CMD="$CMD --mode ${MODE}"
[ -n "${DATASET_PATH:-}" ]    && CMD="$CMD --dataset-path ${DATASET_PATH}"
[ -n "${BUDGET:-}" ]          && CMD="$CMD --budget ${BUDGET}"
[ -n "${QUESTION_INDEX:-}" ]  && CMD="$CMD --question-index ${QUESTION_INDEX}"
[ -n "${WARMUP_TRACES:-}" ]   && CMD="$CMD --warmup-traces ${WARMUP_TRACES}"
[ -n "${TOTAL_BUDGET:-}" ]    && CMD="$CMD --total-budget ${TOTAL_BUDGET}"
[ -n "${MAX_TOKENS:-}" ]      && CMD="$CMD --max-tokens ${MAX_TOKENS}"
[ -n "${OUTPUT_PATH:-}" ]     && CMD="$CMD --output-path ${OUTPUT_PATH}"

echo "Running: $CMD"
eval $CMD
