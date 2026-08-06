#!/usr/bin/env bash
# Usage:
#   ./tx.sh 1us
#   ./tx.sh 4us
#   ./tx.sh 10us
#   ./tx.sh 40us
#   ./tx.sh all

set -euo pipefail

MODE="${1:-}"

if [[ "$MODE" == "all" ]]; then
  MODES=(1us 5us 10us 20us 40us)
elif [[ -n "$MODE" ]]; then
  MODES=("$MODE")
else
  echo "Usage: $0 {1us|4us|5us|10us|20us|40us|all}"
  exit 1
fi

# Find all filterbanks once, in chronological order
mapfile -t FILES < <(find "$PWD" -maxdepth 1 -name "*.fil" | sort)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No .fil files found."
  exit 1
fi

echo "Found ${#FILES[@]} filterbanks."

# Create a top-level output directory
mkdir -p cands

for MODE in "${MODES[@]}"; do

  case "$MODE" in
  1us)
    ROOT="crab_1us"
    TD=1
    MINW=0.000001
    MAXW=0.0001
    LENGTH=1
    THRE=7
    ;;
  4us)
    ROOT="crab_4us"
    TD=4
    MINW=0.000004
    MAXW=0.0005
    LENGTH=2
    THRE=7
    ;;
  5us)
    ROOT="crab_5us"
    TD=5
    MINW=0.000005
    MAXW=0.0005
    LENGTH=2
    THRE=7
    ;;
  10us)
    ROOT="crab_10us"
    TD=10
    MINW=0.00001
    MAXW=0.003
    LENGTH=5
    THRE=7
    ;;
  20us)
    ROOT="crab_20us"
    TD=20
    MINW=0.00002
    MAXW=0.006
    LENGTH=9.65
    THRE=7
    ;;
  40us)
    ROOT="crab_40us"
    TD=40
    MINW=0.00004
    MAXW=0.02
    LENGTH=9.65
    THRE=7
    ;;
  esac

  OUTDIR="cands/${MODE}"

  echo
  echo "==========================================="
  echo "Running TransientX search"
  echo "  Mode        : $MODE"
  echo "  output dir  : ${OUTDIR}/"
  echo "==========================================="

  OUTDIR="cands/$MODE"

  mkdir -p "$OUTDIR"
  pushd "$OUTDIR" >/dev/null

  transientx_fil -v \
    --rootname "${ROOT}" \
    --td "${TD}" \
    --zapthre 3.0 \
    --dms 56.65 \
    --ddm 0.005 \
    --ndm 30 \
    --overlap 0.1 \
    --cont \
    --thre "${THRE}" \
    --minw "${MINW}" \
    --maxw "${MAXW}" \
    --snrloss 0.1 \
    -l "${LENGTH}" \
    --drop \
    --baseline 0 5 \
    --iqr \
    -r 1 \
    -k 3 \
    --minpts 3 \
    --maxncand 5 \
    -f "${FILES[@]}"

  popd >/dev/null

done

echo
echo "All requested searches completed."

