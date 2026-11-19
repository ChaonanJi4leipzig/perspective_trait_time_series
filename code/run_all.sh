#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
OUT=${1:-"$ROOT/figures"}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
for fig in fig3 fig4 si1 si2 si3 si4 si5; do
  nice -n 19 "$PYTHON" -B "$ROOT/code/plot_${fig}.py" --output-dir "$OUT"
done
