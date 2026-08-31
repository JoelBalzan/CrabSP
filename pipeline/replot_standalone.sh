#!/usr/bin/env bash
# replot_standalone.sh — rerun replot_fil exactly as tx.sh does, on all cands
# Usage (from data dir containing *.fil and cands/):
#   bash pipeline/replot_standalone.sh                # all modes in cands/
#   bash pipeline/replot_standalone.sh 0.5us           # one mode
#   REPLOT_DM_CUTOFF=15 REPLOT_DDM_CUTOFF=0 bash pipeline/replot_standalone.sh 5us
#   DO_REPLOT=0 bash pipeline/replot_standalone.sh   # dry-run, just show what would run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# same defaults as tx.sh
REPLOT_DM_CUTOFF=${REPLOT_DM_CUTOFF:-20}
#REPLOT_DDM_CUTOFF=${REPLOT_DDM_CUTOFF:-0}
#REPLOT_SNRCUTOFF=${REPLOT_SNRCUTOFF:-0}
#REPLOT_WIDTHCUTOFF=${REPLOT_WIDTHCUTOFF:-0}

# discover modes
if [[ $# -gt 0 ]]; then
  if [[ "$1" == "all" ]]; then
    mapfile -t MODES < <(find cands -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)
  else
    MODES=("$@")
  fi
else
  mapfile -t MODES < <(find cands -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)
fi
if [[ ${#MODES[@]} -eq 0 ]]; then
  echo "No cands/* subdirs found. Usage: $0 [0.25us 0.5us ... | all]  (run from data dir with cands/)" >&2
  exit 1
fi

# filterbank list (for replot -f, must be the search fil files, not cands)
mapfile -t FILES < <(find "$PWD" -maxdepth 1 -name "*.fil" | sort)
if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No *.fil in $PWD — run from the data dir (where tx.sh was run)" >&2
  exit 1
fi
echo "Found ${#FILES[@]} filterbanks for replot -f"
echo "Modes: ${MODES[*]}"
echo "Cutoffs: dm=$REPLOT_DM_CUTOFF --zdot"
echo

for MODE in "${MODES[@]}"; do
  # map MODE -> TSEARCH exactly as tx.sh
  case "$MODE" in
    0.25us) TSEARCH=0.00000025 ;;
    0.5us)  TSEARCH=0.0000005 ;;
    1us)    TSEARCH=0.000001 ;;
    4us)    TSEARCH=0.000004 ;;
    5us)    TSEARCH=0.000005 ;;
    10us)   TSEARCH=0.00001 ;;
    20us)   TSEARCH=0.00002 ;;
    40us)   TSEARCH=0.00004 ;;
    *) echo "Unknown mode $MODE, trying TSEARCH from dir name"; TSEARCH=0.000001 ;;
  esac

  OUTDIR="cands/$MODE"
  if [[ ! -d "$OUTDIR" ]]; then
    echo "skip $MODE: no $OUTDIR"
    continue
  fi
  # find cands files in this mode (tx.sh writes one per root MJD, e.g. crab_5us_...cands)
  mapfile -t CANDS < <(find "$OUTDIR" -maxdepth 1 -name "*.cands" ! -name "*.orig" | sort)
  if [[ ${#CANDS[@]} -eq 0 ]]; then
    echo "skip $MODE: no *.cands in $OUTDIR"
    continue
  fi

  for cands_file in "${CANDS[@]}"; do
    echo "=== $MODE : $cands_file ==="
    if [[ ! -s "$cands_file" ]]; then
      echo "  empty, skipping"
      continue
    fi
    # cd to cands/mode so any relative PNGs (replot archives, diagnostic plots) land there, not in run dir
    pushd "$OUTDIR" >/dev/null
    cands_base="$(basename "$cands_file")"
    # width fix exactly as tx.sh: 0.00 -> TSEARCH*1000 ms
    MIN_WIDTH_MS=$(awk "BEGIN{printf \"%.4f\", ${TSEARCH} * 1000}")
    if [[ ! -f "${cands_base}.orig" ]]; then
      cp "$cands_base" "${cands_base}.orig"
      echo "  backup -> $OUTDIR/${cands_base}.orig"
    else
      echo "  .orig already exists, not overwriting"
    fi
    BEFORE_FIX=$(wc -l < "$cands_base")
    awk -F'\t' -v mw="$MIN_WIDTH_MS" 'BEGIN{OFS="\t"} $5=="0.00" {$5=mw} {print}' \
      "$cands_base" > "${cands_base}.tmp" && mv "${cands_base}.tmp" "$cands_base"
    AFTER_FIX=$(wc -l < "$cands_base")
    if [[ "$BEFORE_FIX" != "$AFTER_FIX" ]]; then
      echo "  width fix 0.00 -> $MIN_WIDTH_MS ms"
    fi

    BEFORE=$(wc -l < "$cands_base")
    echo "  running (from $OUTDIR): replot_fil -f ${#FILES[@]} fils --candfile $cands_base -c --cont --dmcutoff $REPLOT_DM_CUTOFF --zdot -t $(nproc)"
    replot_fil -f "${FILES[@]}" \
      --candfile "$cands_base" \
      -c \
      --cont \
      --dm 56.7 \
      --dmcutoff "$REPLOT_DM_CUTOFF" \
      --zdot \
      -t "$(nproc)"
    # replot writes to *_replot.cands — move back to original name
    replot_out="${cands_base%.cands}_replot.cands"
    if [[ -f "$replot_out" ]]; then
      mv "$replot_out" "$cands_base"
      replot_json="${cands_base%.cands}_replot.json"
      [[ -f "$replot_json" ]] && mv "$replot_json" "${cands_base%.cands}.json"
    fi
    AFTER=$(wc -l < "$cands_base")
    echo "  cands: $BEFORE -> $AFTER after replot_fil"
    echo
    popd >/dev/null
  done
done

echo "All replots done."
