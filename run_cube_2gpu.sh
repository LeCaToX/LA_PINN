#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Physical GPUs 0 and 1 become torchrun ranks 0 and 1.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

exec torchrun \
    --standalone \
    --nproc_per_node=2 \
    compare_cube_2gpu.py \
    "$@"
