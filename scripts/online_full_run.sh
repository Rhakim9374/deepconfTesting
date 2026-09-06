#!/usr/bin/env bash
set -euo pipefail

# source setup (paths, GPU rename)
source /nethome/rhakim/projects/deepconfTesting/scripts/setup.sh

# Activate the conda env. An env that ships its own bin/activate (the
# conda-pack transplant on LST) is self-contained and activates directly;
# anything else activates through miniconda (LSV). Same dispatch as
# ExploreExploitThink/scripts/run.sh -- see the two conda-pack gotchas noted
# there: the activate script reads CONDA_PREFIX unguarded (so it cannot run
# under `set -u`) and prepends to PATH without exporting (so children would
# see no PATH at all in HTCondor's empty environment).
if [ -n "${CONDA_ENV:-}" ] && [ -f "${CONDA_ENV}/bin/activate" ]; then
    set +u
    source "${CONDA_ENV}/bin/activate"
    set -u
    export PATH
else
    cd /nethome/rhakim/miniconda3/bin
    source activate "${CONDA_ENV:-deepConfEnv}"
    cd "$PROJECT_DIR"
fi

# diagnostics
echo "=== Diagnostics ==="
echo "HOSTNAME: $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi
which python
python --version
echo "==================="

# build command from environment variables
CMD="python -m deepconfTesting.main --model ${MODEL_PATH} --dataset ${DATASET_PATH}"

[ -n "${RUN_ID:-}" ]                    && CMD="$CMD --run_id ${RUN_ID}"
[ -n "${WARMUP_TRACES:-}" ]             && CMD="$CMD --warmup_traces ${WARMUP_TRACES}"
[ -n "${TOTAL_BUDGET:-}" ]              && CMD="$CMD --total_budget ${TOTAL_BUDGET}"
[ -n "${CONFIDENCE_PERCENTILE:-}" ]     && CMD="$CMD --confidence_percentile ${CONFIDENCE_PERCENTILE}"
[ -n "${MAX_TOKENS:-}" ]                && CMD="$CMD --max_tokens ${MAX_TOKENS}"
[ -n "${OUTPUT_DIR:-}" ]                && CMD="$CMD --output_dir ${OUTPUT_DIR}"
[ -n "${TENSOR_PARALLEL:-}" ]           && CMD="$CMD --tensor_parallel_size ${TENSOR_PARALLEL}"
[ -n "${MODEL_TYPE:-}" ]                && CMD="$CMD --model_type ${MODEL_TYPE}"
[ -n "${DATASET_TYPE:-}" ]              && CMD="$CMD --dataset_type ${DATASET_TYPE}"
[ -n "${TEMPERATURE:-}" ]               && CMD="$CMD --temperature ${TEMPERATURE}"
[ -n "${WINDOW_SIZE:-}" ]               && CMD="$CMD --window_size ${WINDOW_SIZE}"
[ -n "${TOP_P:-}" ]                     && CMD="$CMD --top_p ${TOP_P}"
[ -n "${TOP_K:-}" ]                     && CMD="$CMD --top_k ${TOP_K}"
[ -n "${QID_START:-}" ]                 && CMD="$CMD --qid_start ${QID_START}"
[ -n "${QID_END:-}" ]                   && CMD="$CMD --qid_end ${QID_END}"
[ -n "${NO_MULTIPLE_VOTING:-}" ]        && CMD="$CMD --no_multiple_voting"

echo "Running: $CMD"
eval $CMD
