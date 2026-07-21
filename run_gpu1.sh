#!/usr/bin/env bash
set -euo pipefail

# Expose physical GPU 1 as CUDA device 0 inside this process.
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "$#" -eq 0 ]]; then
    set -- compare_kan_mlp_full.py --problem both
fi

python -c 'import torch; print(f"CUDA available: {torch.cuda.is_available()}"); print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}")'
exec python "$@"
