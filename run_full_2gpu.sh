#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/comparison_full_results}"
if [[ -n "${CUBE_DIR:-}" ]]; then
    :
elif [[ -d "$ROOT_DIR/comparison_cube_2gpu" ]]; then
    CUBE_DIR="$ROOT_DIR/comparison_cube_2gpu"
else
    CUBE_DIR="$OUTPUT_DIR/cube_2gpu"
fi
mkdir -p "$OUTPUT_DIR" "$CUBE_DIR"
cd "$ROOT_DIR"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "=== Plate comparison on GPU 0: Gauss 2, 3, 5 ==="
CUDA_VISIBLE_DEVICES=0 python -u compare_kan_mlp_full.py \
    --problem plate \
    --output-dir "$OUTPUT_DIR" \
    "$@" > "$OUTPUT_DIR/plate_run.log" 2>&1

echo "=== Sharded cube comparison on GPUs 0 and 1 ==="
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --standalone \
    --nproc_per_node=2 \
    compare_cube_2gpu.py \
    --output-dir "$CUBE_DIR" \
    --legacy-dir "$OUTPUT_DIR" \
    "$@" > "$CUBE_DIR/run.log" 2>&1

python merge_full_2gpu_reports.py \
    --plate-dir "$OUTPUT_DIR" \
    --cube-dir "$CUBE_DIR" \
    --output-dir "$OUTPUT_DIR"
echo "Full MLP/KAN comparison completed."
