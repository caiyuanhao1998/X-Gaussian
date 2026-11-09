#!/usr/bin/env bash
# Recommended recipe: original product head, TV center isolation, 30k + 1k steps.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export GPU_ID="${GPU_ID:-0}" CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" MPLBACKEND=Agg
CT_PYTHON="${CT_PYTHON:-python}"
if [[ "$CT_PYTHON" == python && -x .venv_ct/bin/python ]]; then CT_PYTHON=.venv_ct/bin/python; fi
CONFIG_PATH="${CONFIG_PATH:-config/ct_teapot.yaml}"
BASE_PATH="${BASE_PATH:-output_ct/teapot_ct_base}"
OUTPUT_PATH="${OUTPUT_PATH:-output_ct/teapot_ct_recommended}"
if [[ -e "$BASE_PATH/config.json" || -e "$OUTPUT_PATH/config.json" ]]; then
  echo 'Choose unused BASE_PATH and OUTPUT_PATH to preserve existing experiment artifacts.' >&2
  exit 1
fi
extra=()
if [[ -n "${DATA_PATH:-}" ]]; then extra+=(--source_path "$DATA_PATH"); fi
bash run_ct.sh --config "$CONFIG_PATH" --iterations 30000 --eval_every 30000 --checkpoint_every 10000 --final_eval_only --model_path "$BASE_PATH" "${extra[@]}" "$@"
"$CT_PYTHON" -u finetune_ct_rays.py --model_path "$BASE_PATH" --output "$OUTPUT_PATH" --checkpoint 30000 --iterations 1000 --ray_weight 2 --ray_count 8192 --ray_method siddon --protocol test --skip_evaluation
"$CT_PYTHON" -u evaluate_ct_model.py --model_path "$OUTPUT_PATH" --iteration 1000
"$CT_PYTHON" -u verify_ct_export.py --model_path "$OUTPUT_PATH" --iteration 1000 --checks "$OUTPUT_PATH/export_checks.json"
# R2 is an optional comparison dependency, not required for training/export.
if [[ -n "${R2_BASELINE:-}" ]]; then
  r2_args=()
  if [[ -n "${R2_REPO:-}" ]]; then r2_args+=(--r2_repo "$R2_REPO"); fi
  "$CT_PYTHON" -u compare_ct.py --model_path "$OUTPUT_PATH" --iteration 1000 --baseline "$R2_BASELINE" "${r2_args[@]}"
fi
