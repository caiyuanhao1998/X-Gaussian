#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
CT_PYTHON="${CT_PYTHON:-python}"
if [[ "$CT_PYTHON" == python && -x .venv_ct/bin/python ]]; then CT_PYTHON=.venv_ct/bin/python; fi
exec "$CT_PYTHON" -u train_ct.py "$@"
