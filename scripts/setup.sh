#!/usr/bin/env bash

# setup basic paths
export PROJECT_DIR=/nethome/rhakim/projects/deepconfTesting

# Per-process, node-local compile/JIT caches. Every proc of a `queue N`
# submission would otherwise share one NFS cache (~/.cache/vllm) under a
# deterministic key and race to populate it, so a losing proc dies at vLLM
# engine init on a failed atomic rename. Cleaned up when the proc exits.
# (Same fix as ExploreExploitThink/scripts/setup.sh.)
export VLLM_CACHE_ROOT="/tmp/deepconf_cache_${USER:-u}_${CONDOR_PROCESS:-0}_$$"
export TRITON_CACHE_DIR="${VLLM_CACHE_ROOT}/triton"
trap 'rm -rf "${VLLM_CACHE_ROOT}"' EXIT

# rename gpus
source "${PROJECT_DIR}/scripts/rename_gpus.sh"

# cd to project dir
cd "$PROJECT_DIR"
