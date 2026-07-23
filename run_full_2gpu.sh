#!/usr/bin/env bash
set -euo pipefail

# Complete restartable two-GPU pipeline for every translated MATLAB-KAN case,
# the previous plate comparison, and the sharded two-GPU cube comparison.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/all_kan_results}"
if [[ -n "${PAIR_DIR:-}" ]]; then
    :
elif [[ -f "$ROOT_DIR/comparison_full_results/comparison_report.json" ]]; then
    PAIR_DIR="$ROOT_DIR/comparison_full_results"
else
    PAIR_DIR="$OUTPUT_DIR/plate_pair_comparison"
fi
if [[ -n "${CUBE_DIR:-}" ]]; then
    :
elif [[ -d "$ROOT_DIR/comparison_cube_2gpu" ]]; then
    CUBE_DIR="$ROOT_DIR/comparison_cube_2gpu"
else
    CUBE_DIR="$OUTPUT_DIR/cube_2gpu"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUTPUT_DIR" "$PAIR_DIR" "$CUBE_DIR"
cd "$ROOT_DIR"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

CASE_ARGS=()
if [[ -n "${CASE_ADAM:-}" ]]; then CASE_ARGS+=(--adam "$CASE_ADAM"); fi
if [[ -n "${CASE_LBFGS:-}" ]]; then CASE_ARGS+=(--lbfgs "$CASE_LBFGS"); fi

run_case() {
    local gpu="$1"
    local case_name="$2"
    local script_name="$3"
    local case_dir="$OUTPUT_DIR/$case_name"
    local log_file="$case_dir/run.log"
    local done_file="$case_dir/.done"
    mkdir -p "$case_dir"

    if [[ -f "$done_file" ]]; then
        echo "[$case_name] already complete; skipping."
        return 0
    fi

    echo "[$case_name] starting on GPU $gpu"
    if env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$ROOT_DIR/$script_name" \
        --output-dir "$case_dir" "${CASE_ARGS[@]}" >"$log_file" 2>&1; then
        date -u +%Y-%m-%dT%H:%M:%SZ >"$done_file"
        echo "[$case_name] complete"
    else
        local exit_code=$?
        echo "[$case_name] FAILED with exit code $exit_code; see $log_file" >&2
        return "$exit_code"
    fi
}

run_gpu0_lane() {
    run_case 0 fast_plate Main_UB_PINN_hole_fast_KAN.py
    run_case 0 standard_plate Main_UB_PINN_Hole_adam_LBFGS_KAN.py
    run_case 0 high_gauss_plate Main_UB_PINN_Hole_adam_LBFGS_high_Gauss_KAN.py
    run_case 0 adaptive_plate LA_PINN_plate_hole_run_adaptive_Gauss_KAN.py
}

run_gpu1_lane() {
    run_case 1 activation_plate LA_PINN_plate_hole_run_hist_Activate_KAN.py
    run_case 1 history_gauss_plate LA_PINN_plate_hole_run_histGauss_KAN.py
    run_case 1 thin_cylinder LA_PINN_hollow_cylinder_thin_KAN.py
    run_case 1 thick_cylinder LA_PINN_hollow_cylinder_hist_KAN.py
}

echo "=== Running all translated MATLAB-KAN cases on two GPU lanes ==="
run_gpu0_lane & GPU0_PID=$!
run_gpu1_lane & GPU1_PID=$!
GPU0_STATUS=0; GPU1_STATUS=0
wait "$GPU0_PID" || GPU0_STATUS=$?
wait "$GPU1_PID" || GPU1_STATUS=$?
if [[ "$GPU0_STATUS" -ne 0 || "$GPU1_STATUS" -ne 0 ]]; then
    echo "A KAN lane failed. Rerun the same command; completed cases will be skipped." >&2
    exit 1
fi

if [[ "${RUN_PAIR_COMPARISON:-1}" == "1" ]]; then
    echo "=== Running plate MLP/KAN comparison on GPU 0 ==="
    env CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -u compare_kan_mlp_full.py \
        --problem plate --output-dir "$PAIR_DIR" >"$PAIR_DIR/run.log" 2>&1
fi

if [[ "${RUN_CUBE_COMPARISON:-1}" == "1" ]]; then
    echo "=== Running sharded cube comparison on GPUs 0 and 1 ==="
    env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
        "$ROOT_DIR/compare_cube_2gpu.py" --output-dir "$CUBE_DIR" \
        --legacy-dir "$PAIR_DIR" >"$CUBE_DIR/run.log" 2>&1
fi

if [[ "${RUN_PAIR_COMPARISON:-1}" == "1" && "${RUN_CUBE_COMPARISON:-1}" == "1" ]]; then
    "$PYTHON_BIN" "$ROOT_DIR/merge_full_2gpu_reports.py" \
        --plate-dir "$PAIR_DIR" --cube-dir "$CUBE_DIR" --output-dir "$OUTPUT_DIR"
fi

echo "=== All KAN cases and comparisons completed ==="
echo "Results: $OUTPUT_DIR"
