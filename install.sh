#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# The existing rasterizer submodule supplies GLM headers for both CUDA paths.
git submodule update --init --recursive submodules/simple-knn submodules/diff-gaussian-rasterization
export MAX_JOBS="${MAX_JOBS:-2}"
python -m pip install --no-build-isolation --no-deps submodules/simple-knn submodules/x-gaussian-ct-rasterization
