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

# --- auto-log to file + terminal (tee) so crashes are traceable ---
# Use LOGFILE env to override; default is timestamped in ./cands/
# transientx -v spams `finish X.XX seconds (Y%)` per 0.01s (10× per 0.01s) → 600 MB log;
# replot spams `time not contiguous` per gap (354 files → 100+ warnings) → also filtered.
LOGFILE="${LOGFILE:-cands/tx_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$LOGFILE")"
# shellcheck disable=SC2069
exec > >(stdbuf -oL sed -u 's/\r/\n/g' | grep -v "^finish" | grep -v "^Maximum width" | grep -v "YMW16_DIR" | grep -v "time not contiguous" | tee -a "$LOGFILE") 2>&1
echo "=== tx.sh started $(date -Iseconds) ==="
echo "cmd : $0 $*"
echo "pwd : $PWD"
echo "log : $LOGFILE (also on terminal via tee, filtered \\r -> \\n)"
echo
trap 'echo "=== tx.sh CRASHED at $(date -Iseconds) (mode=${MODE:-?} line=${LINENO}) ==="; echo "last file tried: ${FILES[-1]:-?}"; echo "see $LOGFILE"' ERR

VENV_PYTHON="${VENV_PYTHON:-/home/joel/Documents/GitHub/CrabSP/.venv/bin/python}"

MODE="${1:-}"

if [[ "$MODE" == "all" ]]; then
  MODES=(0.5us 1us 5us)
elif [[ -n "$MODE" ]]; then
  MODES=("$MODE")
else
  echo "Usage: $0 {0.25us|0.5us|1us|4us|5us|10us|20us|40us|all}"
  exit 1
fi

DO_REPLOT=${DO_REPLOT:-1}
REPLOT_DM_CUTOFF=${REPLOT_DM_CUTOFF:-15}
REPLOT_DDM_CUTOFF=${REPLOT_DDM_CUTOFF:-0}
REPLOT_SNRCUTOFF=${REPLOT_SNRCUTOFF:-0}
REPLOT_WIDTHCUTOFF=${REPLOT_WIDTHCUTOFF:-0}

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
    MINW=$(awk "BEGIN{printf \"%.10f\", ${TSEARCH} * 2}")
    MAXW=0.000025
    LENGTH=1
    THRE=10
    ;;
  0.5us)
    ROOT="crab_0.5us"
    TSEARCH=0.0000005
    MINW=$(awk "BEGIN{printf \"%.10f\", ${TSEARCH} * 2}")
    MAXW=0.00005
    LENGTH=1
    THRE=10
    ;;
  1us)
    ROOT="crab_1us"
    TSEARCH=0.000001
    MINW=$(awk "BEGIN{printf \"%.10f\", ${TSEARCH} * 2}")
    MAXW=0.0001
    LENGTH=1
    THRE=10
    ;;
  4us)
    ROOT="crab_4us"
    TSEARCH=0.000004
    MINW=$(awk "BEGIN{printf \"%.10f\", ${TSEARCH} * 2}")
    MAXW=0.0005
    LENGTH=2
    THRE=10
    ;;
  5us)
    ROOT="crab_5us"
    TSEARCH=0.000005
    MINW=$(awk "BEGIN{printf \"%.10f\", ${TSEARCH} * 2}")
    MAXW=0.0005
    LENGTH=2
    THRE=10
    ;;
  10us)
    ROOT="crab_10us"
    TSEARCH=0.00001
    MINW=$(awk "BEGIN{printf \"%.10f\", ${TSEARCH} * 2}")
    MAXW=0.003
    LENGTH=5
    THRE=10
    ;;
  20us)
    ROOT="crab_20us"
    TSEARCH=0.00002
    MINW=$(awk "BEGIN{printf \"%.10f\", ${TSEARCH} * 2}")
    MAXW=0.006
    LENGTH=9.65
    THRE=10
    ;;
  40us)
    ROOT="crab_40us"
    TSEARCH=0.00004
    MINW=$(awk "BEGIN{printf \"%.10f\", ${TSEARCH} * 2}")
    MAXW=0.02
    LENGTH=9.65
    THRE=10
    ;;
  esac

  # td = TSEARCH / input_tsamp must be >=1 — can't search finer than native
  if "$VENV_PYTHON" -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" "$input_tsamp" "$TSEARCH"; then
    echo "  SKIP $MODE: requested ${TSEARCH}s < native ${input_tsamp}s (td<1, would need upsampling) — skipping" >&2
    continue
  fi
  TD="$("$VENV_PYTHON" -c "import sys; print(max(1, round(float(sys.argv[1]) / float(sys.argv[2]))))" "$TSEARCH" "$input_tsamp")"

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
  # --- resume support: if a previous run crashed, cands_file is partial ---
  RESUME=0
  RESUME_FILES=()
  if [[ -s "$cands_file" ]]; then
    last_fil=$(tail -1 "$cands_file" | awk -F'\t' '{gsub(/\r/, "", $NF); print $NF}')
    if [[ "$last_fil" == "${FILES[-1]}" ]]; then
      echo "  $cands_file covers ${FILES[-1]##*/}, skipping"
      popd >/dev/null
      continue
    fi
    # find index of last_fil in FILES (last successfully written file)
    last_idx=-1
    for i in "${!FILES[@]}"; do
      if [[ "${FILES[$i]}" == "$last_fil" ]]; then last_idx=$i; break; fi
    done
    if (( last_idx >= 0 )); then
      if [[ "${FORCE:-0}" == "1" ]]; then
        echo "  FORCE=1: discarding partial $cands_file (last cand in ${last_fil##*/}) and restarting"
        rm -f "$cands_file"
      elif (( last_idx == 0 )); then
        echo "  $cands_file incomplete but only first file done (${last_fil##*/}) — restarting from scratch"
        rm -f "$cands_file"
      else
        RESUME=1
        # include last file for overlap; transientx --cont needs the boundary
        RESUME_FILES=("${FILES[@]:$last_idx}")
        # what the resume run will write (based on its first file's MJD)
        resume_root_mjd="$("$VENV_PYTHON" -c "import sys; from sigpyproc.readers import FilReader; print('%.10f' % FilReader(sys.argv[1]).header.tstart)" "${RESUME_FILES[0]}")"
        resume_cands="${ROOT}_${resume_root_mjd}_cfbf00000.cands"
        echo "  $cands_file incomplete (last cand in ${last_fil##*/}, want ${FILES[-1]##*/})"
        echo "  RESUME: ${#RESUME_FILES[@]} files from ${RESUME_FILES[0]##*/} (idx $last_idx) -> $resume_cands"
        echo "  (fix the problem file — e.g. rfi/blank_fil.py it — then re-run; this will resume)"
      fi
    else
      echo "  $cands_file incomplete but last cand file ${last_fil##*/} not in current file list — restarting"
      rm -f "$cands_file"
    fi
  fi

  if (( RESUME )); then
    echo "  resuming search ${RESUME_FILES[0]##*/} .. ${RESUME_FILES[-1]##*/} (${#RESUME_FILES[@]} filterbanks, root MJD $resume_root_mjd) -> $resume_cands"
    transientx_fil -v \
      -t "$(nproc)" \
      --rootname "${ROOT}" \
      --td "${TD}" \
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
      --baseline 0 0.1 \
      --iqr \
      --fill rand \
      --fillPatch rand \
      -z kadaneT 8 1 zdot \
      --threKadaneT 6 \
      --bandlimitKT 16 \
      --threMask 6 \
      --zapthre 2.0 \
      -r 3 \
      -k 3 \
      --minpts 3 \
      --maxncand 100 \
      -f "${RESUME_FILES[@]}"
    # merge resume output into the original cands_file
    if [[ -s "$resume_cands" ]]; then
      grep -v "^#" "$resume_cands" > "${resume_cands}.nohead" 2>/dev/null || true
      if head -1 "$cands_file" | grep -q "^#"; then
        head -1 "$cands_file" > "${cands_file}.merged"
        grep -v "^#" "$cands_file" >> "${cands_file}.merged"
      else
        cat "$cands_file" > "${cands_file}.merged"
      fi
      cat "${resume_cands}.nohead" >> "${cands_file}.merged"
      # de-dupe exact duplicate lines from the overlap file, preserve order
      awk '!seen[$0]++' "${cands_file}.merged" > "${cands_file}.tmp" && mv "${cands_file}.tmp" "$cands_file"
      rm -f "${resume_cands}" "${resume_cands}.nohead" "${cands_file}.merged"
      echo "  merged resume -> $cands_file ($(wc -l < "$cands_file") total cands)"
    else
      echo "  WARNING: resume produced no $resume_cands — keeping original partial" >&2
    fi
  else
    echo "  searching ${FILES[0]##*/} .. ${FILES[-1]##*/} (${#FILES[@]} filterbanks, root MJD $root_mjd)"
    transientx_fil -v \
      -t "$(nproc)" \
      --rootname "${ROOT}" \
      --td "${TD}" \
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
      --baseline 0 0.1 \
      --iqr \
      --fill rand \
      --fillPatch rand \
      -z kadaneT 8 1 zdot \
      --threKadaneT 6 \
      --bandlimitKT 16 \
      --threMask 6 \
      --zapthre 2.0 \
      -r 3 \
      -k 3 \
      --minpts 3 \
      --maxncand 100 \
      -f "${FILES[@]}"
  fi

  if [[ -s "$cands_file" ]] && (( DO_REPLOT )); then
    # replot_fil divides by the cand width; widths < 0.005 ms get written as
    # 0.00 in the .cands file (2dp) -> SIGFPE.  Replace 0.00 with the search
    # resolution on a copy so the original is preserved.
    MIN_WIDTH_MS=$(awk "BEGIN{printf \"%.4f\", ${TSEARCH} * 1000}")
    cp "$cands_file" "${cands_file}.orig"
    awk -F'\t' -v mw="$MIN_WIDTH_MS" 'BEGIN{OFS="\t"} $5=="0.00" {$5=mw} {print}' \
      "$cands_file" > "${cands_file}.tmp" && mv "${cands_file}.tmp" "$cands_file"

    echo "  running replot_fil -c (dmcutoff=$REPLOT_DM_CUTOFF ddcutoff=$REPLOT_DDM_CUTOFF snrcutoff=$REPLOT_SNRCUTOFF widthcutoff=$REPLOT_WIDTHCUTOFF --zdot)"
    BEFORE=$(wc -l < "$cands_file")
    replot_fil -f "${FILES[@]}" \
      --candfile "$cands_file" \
      -c \
      --cont \
      --dmcutoff "$REPLOT_DM_CUTOFF" \
      --ddmcutoff "$REPLOT_DDM_CUTOFF" \
      --snrcutoff "$REPLOT_SNRCUTOFF" \
      --widthcutoff "$REPLOT_WIDTHCUTOFF" \
      --zdot \
      -t "$(nproc)"
    # replot -c writes to *_replot.cands, not in-place — move it back
    replot_out="${cands_file%.cands}_replot.cands"
    if [[ -f "$replot_out" ]]; then
      mv "$replot_out" "$cands_file"
      # also move the json sidecar if present
      replot_json="${cands_file%.cands}_replot.json"
      [[ -f "$replot_json" ]] && mv "$replot_json" "${cands_file%.cands}.json"
    fi
    AFTER=$(wc -l < "$cands_file")
    echo "  cands: $BEFORE -> $AFTER after replot_fil"
  elif [[ -s "$cands_file" ]]; then
    echo "  replot_fil skipped (DO_REPLOT=0)"
  else
    echo "  WARNING: no cands file written by search" >&2
  fi

  popd >/dev/null

done

echo
echo "All requested searches completed."
