#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
CT_PYTHON="${CT_PYTHON:-python}"
if [[ "$CT_PYTHON" == python && -x .venv_ct/bin/python ]]; then CT_PYTHON=.venv_ct/bin/python; fi
export MAX_JOBS="${MAX_JOBS:-2}"
# Use the activated environment's PyTorch and the caller's CUDA_HOME / CC / CXX.
exec "$CT_PYTHON" -m pip install --no-build-isolation --no-deps submodules/simple-knn submodules/x-gaussian-ct-rasterization
