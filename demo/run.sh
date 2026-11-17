#!/usr/bin/env bash
set -euo pipefail

demo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cuda_python="$(conda run -n x_gaussian python -c 'import sys; print(sys.executable)')"
exec conda run --no-capture-output -n x_gaussian_demo \
  python "$demo_dir/app.py" --cuda-python "$cuda_python" "$@"
