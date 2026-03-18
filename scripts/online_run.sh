#!/usr/bin/env bash
set -euo pipefail

# source setup (paths, GPU rename)
source /nethome/rhakim/projects/deepconfTesting/scripts/setup.sh

# Activate conda environment
cd /nethome/rhakim/miniconda3/bin
source activate deepConfEnv
cd $PROJECT_DIR

# diagnostics
echo "=== Diagnostics ==="
echo "HOSTNAME: $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi
which python
python --version
echo "==================="

# build command from environment variables
CMD="python ${PROJECT_DIR}/examples/example_online.py --model ${MODEL_PATH} --qid ${QUESTION_INDEX:-0}"

[ -n "${DATASET_PATH:-}" ]              && CMD="$CMD --dataset ${DATASET_PATH}"
[ -n "${WARMUP_TRACES:-}" ]             && CMD="$CMD --warmup_traces ${WARMUP_TRACES}"
[ -n "${TOTAL_BUDGET:-}" ]              && CMD="$CMD --total_budget ${TOTAL_BUDGET}"
[ -n "${CONFIDENCE_PERCENTILE:-}" ]     && CMD="$CMD --confidence_percentile ${CONFIDENCE_PERCENTILE}"
[ -n "${MAX_TOKENS:-}" ]                && CMD="$CMD --max_tokens ${MAX_TOKENS}"
[ -n "${OUTPUT_DIR:-}" ]                && CMD="$CMD --output_dir ${OUTPUT_DIR}"
[ -n "${TENSOR_PARALLEL:-}" ]           && CMD="$CMD --tensor_parallel_size ${TENSOR_PARALLEL}"
[ -n "${MODEL_TYPE:-}" ]                && CMD="$CMD --model_type ${MODEL_TYPE}"
[ -n "${TEMPERATURE:-}" ]               && CMD="$CMD --temperature ${TEMPERATURE}"
[ -n "${WINDOW_SIZE:-}" ]               && CMD="$CMD --window_size ${WINDOW_SIZE}"
[ -n "${RUN_ID:-}" ]                    && CMD="$CMD --rid ${RUN_ID}"
[ -n "${REASONING_EFFORT:-}" ]          && CMD="$CMD --reasoning_effort ${REASONING_EFFORT}"
[ -n "${TOP_P:-}" ]                     && CMD="$CMD --top_p ${TOP_P}"
[ -n "${TOP_K:-}" ]                     && CMD="$CMD --top_k ${TOP_K}"
[ -n "${NO_MULTIPLE_VOTING:-}" ]        && CMD="$CMD --no_multiple_voting"

echo "Running: $CMD"
eval $CMD
