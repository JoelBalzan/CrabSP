#!/usr/bin/env bash
set -euo pipefail

VENV_PYTHON="${VENV_PYTHON:-/home/joel/Documents/GitHub/CrabSP/.venv/bin/python}"
TSEARCH=0.00000025
TD=1
ROOT="crab_0.25us_test"

mapfile -t ALL_FILES < <(find "$PWD" -maxdepth 1 -name "*.fil" | sort)
FILES=("${ALL_FILES[@]: -10:1}")

echo "Running on last 10 files:"
printf '  %s\n' "${FILES[@]##*/}"

transientx_fil -v \
  -t "$(nproc)" \
  --rootname "${ROOT}" \
  --td "${TD}" \
  --dms 56.65 \
  --ddm 0.005 \
  --ndm 30 \
  --overlap 0.1 \
  --cont \
  --thre 10 \
  --minw "$(awk "BEGIN{printf \"%.10f\", ${TSEARCH} * 2}")" \
  --maxw 0.000025 \
  --snrloss 0.1 \
  -l 1 \
  --drop \
  --baseline 0 0.1 \
  --iqr \
  --fill rand \
  --fillPatch rand \
  -z kadaneT 8 1 zdot \
  --threKadaneT 6 \
  --bandlimitKT 16 \
  --widthlimit 0.000025 \
  --threMask 6 \
  --zapthre 2.0 \
  -r 3 \
  -k 3 \
  --minpts 3 \
  --maxncand 100 \
  -f "${FILES[@]}"
