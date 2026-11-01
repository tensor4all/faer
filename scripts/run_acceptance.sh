#!/usr/bin/env bash
# faer #176: invoke only from an observer shell pinned outside the measured L3.
set -euo pipefail
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
exec env -u MKLROOT -u MKL_LIB_DIR -u MKL_INCLUDE_DIR -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS -u CARGO_BUILD_TARGET -u OPENBLAS_CORETYPE \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GOTO_NUM_THREADS=1 \
  PINNED_RETRIES=1 PINNED_WAIT=0 PINNED_IDLE_SECONDS=3 PINNED_MAX_BUSY=0.05 \
  python3 "$root/scripts/pfaffian-acceptance/run.py" "$@"
