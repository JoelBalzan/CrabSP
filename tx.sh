#!/usr/bin/env bash
# Usage:
#   ./tx.sh 0.25us
#   ./tx.sh 0.5us
#   ./tx.sh 1us
#   ./tx.sh 4us
#   ./tx.sh 10us
#   ./tx.sh 40us
#   ./tx.sh all

set -euo pipefail

VENV_PYTHON="${VENV_PYTHON:-/home/joel/Documents/GitHub/CrabSP/.venv/bin/python}"

MODE="${1:-}"

if [[ "$MODE" == "all" ]]; then
  MODES=(0.5us 1us 5us 10us)
elif [[ -n "$MODE" ]]; then
  MODES=("$MODE")
else
  echo "Usage: $0 {0.25us|0.5us|1us|4us|5us|10us|20us|40us|all}"
  exit 1
fi

# Find all filterbanks once, in chronological order
mapfile -t FILES < <(find "$PWD" -maxdepth 1 -name "*.fil" | sort)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No .fil files found."
  exit 1
fi

echo "Found ${#FILES[@]} filterbanks."

# Read the input filterbank's native sampling from the first file so --td
# can be computed per mode for any input time resolution (not just 1 us).
IFS=' ' read -r input_tsamp input_nchans < <(
  "$VENV_PYTHON" -c "
import sys
from sigpyproc.readers import FilReader
h = FilReader(sys.argv[1]).header
print('%g %d' % (h.tsamp, h.nchans))
" "${FILES[0]}" 2>/dev/null || echo "0.000001 0"
)
if [[ -z "$input_tsamp" || -z "$input_nchans" || "$input_nchans" == "0" ]]; then
  echo "WARNING: could not read ${FILES[0]##*/} header; assuming 1 us input" >&2
  input_tsamp=0.000001
  input_nchans=0
fi
echo "  input sampling : ${input_tsamp} s (${input_nchans} chans, ${FILES[0]##*/})"

# Create a top-level output directory
mkdir -p cands

for MODE in "${MODES[@]}"; do

  case "$MODE" in
  0.25us)
    ROOT="crab_0.25us"
    TSEARCH=0.00000025
    MINW=0.00000025
    MAXW=0.000025
    LENGTH=1
    THRE=10
    ;;
  0.5us)
    ROOT="crab_0.5us"
    TSEARCH=0.0000005
    MINW=0.0000005
    MAXW=0.00005
    LENGTH=1
    THRE=7
    ;;
  1us)
    ROOT="crab_1us"
    TSEARCH=0.000001
    MINW=0.000001
    MAXW=0.0001
    LENGTH=1
    THRE=7
    ;;
  4us)
    ROOT="crab_4us"
    TSEARCH=0.000004
    MINW=0.000004
    MAXW=0.0005
    LENGTH=2
    THRE=7
    ;;
  5us)
    ROOT="crab_5us"
    TSEARCH=0.000005
    MINW=0.000005
    MAXW=0.0005
    LENGTH=2
    THRE=7
    ;;
  10us)
    ROOT="crab_10us"
    TSEARCH=0.00001
    MINW=0.00001
    MAXW=0.003
    LENGTH=5
    THRE=7
    ;;
  20us)
    ROOT="crab_20us"
    TSEARCH=0.00002
    MINW=0.00002
    MAXW=0.006
    LENGTH=9.65
    THRE=7
    ;;
  40us)
    ROOT="crab_40us"
    TSEARCH=0.00004
    MINW=0.00004
    MAXW=0.02
    LENGTH=9.65
    THRE=7
    ;;
  esac

  TD="$("$VENV_PYTHON" -c "import sys; print(max(1, round(float(sys.argv[1]) / float(sys.argv[2]))))" "$TSEARCH" "$input_tsamp")"

  if "$VENV_PYTHON" -c "import sys; sys.exit(int(float(sys.argv[1]) > float(sys.argv[2])))" "$input_tsamp" "$TSEARCH"; then
    echo "  WARNING: input tsamp ${input_tsamp} s is coarser than the ${TSEARCH} s requested by ${MODE}; will search at input resolution (td=1)" >&2
  fi

  OUTDIR="cands/$MODE"
  echo
  echo "==========================================="
  echo "Running TransientX search"
  echo "  Mode        : $MODE"
  echo "  search tsamp: ${TSEARCH} s (td=${TD} @ ${input_tsamp} s input)"
  echo "  output dir  : ${OUTDIR}/"
  echo "  files       : ${#FILES[@]}"
  echo "==========================================="

  mkdir -p "$OUTDIR"
  pushd "$OUTDIR" >/dev/null

  root_mjd="$("$VENV_PYTHON" -c "import sys; from sigpyproc.readers import FilReader; print('%.10f' % FilReader(sys.argv[1]).header.tstart)" "${FILES[0]}")"
  cands_file="${ROOT}_${root_mjd}_cfbf00000.cands"
  if [[ -s "$cands_file" ]]; then
    # transientX writes candidates incrementally, so an OOM-killed run can
    # leave a partial .cands. Only trust it if its LAST candidate's fragment
    # is this run's final file.
    last_fil=$(tail -1 "$cands_file" | awk -F'\t' '{gsub(/\r/, "", $NF); print $NF}')
    if [[ "$last_fil" == "${FILES[-1]}" ]]; then
      echo "  $cands_file covers ${FILES[-1]##*/}, skipping"
      popd >/dev/null
      continue
    fi
    echo "  $cands_file incomplete (last cand in ${last_fil##*/}, want ${FILES[-1]##*/}), re-searching"
    rm -f "$cands_file"
  fi
  echo "  searching ${FILES[0]##*/} .. ${FILES[-1]##*/} (${#FILES[@]} filterbanks, root MJD $root_mjd)"
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
    -z kadaneF 8 4 zdot \
    --widthlimit 2 \
    -r 4 \
    -k 3 \
    --minpts 3 \
    --maxncand 100 \
    -f "${FILES[@]}"

  popd >/dev/null

done

echo
echo "All requested searches completed."
