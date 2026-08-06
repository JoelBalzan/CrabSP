#!/usr/bin/env bash
# run_pipeline.sh — full CrabSP single-pulse pipeline for all 3 Crab nights.
#
# For every observation directory this:
#   1. forms a search filterbank  (<fragment>.dada.fil — 32x1 MHz, 8-bit, 1 us,
#      total intensity)  for every raw .dada fragment that lacks one
#      (digifil -F 32 -d 1 -b 8 -I 0)
#   2. runs the transientX multi-resolution search over all contiguous search
#      filterbanks (tx.sh all: 1us 5us 10us 20us 40us)  ->  cands/<res>/
#   3. extracts full-Stokes cutouts for every candidate (extract_cands.py,
#      finest-resolution-first + per-resolution MJD dedup)  ->  cutouts/<res>/
#
# Stage toggles:   DO_FIL=1      form missing search filterbanks
#                  DO_SEARCH=1   run transientX
#                  DO_EXTRACT=1  run extract_cands.py
# Parallelism:     FIL_JOBS=N    concurrent digifil jobs while forming filterbanks
#
# Usage:  ./run_pipeline.sh > pipeline.log 2>&1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${VENV_PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
DIGIFIL="${DIGIFIL:-/home/joel/Pulsar/bin/digifil}"

export PATH="/home/joel/Pulsar/bin:/opt/TransientX/bin:$PATH"
export DIGIFIL

DO_FIL=${DO_FIL:-1}
DO_SEARCH=${DO_SEARCH:-1}
DO_EXTRACT=${DO_EXTRACT:-1}
FIL_JOBS=${FIL_JOBS:-4}

# The three Crab observations (directories holding the raw *.dada fragments)
OBS=(
  "/mnt/exhdd/crab_pks/pks_crab_06-07-26/2026-07-05-23-59-24/J0534+2200/3968"
  "/mnt/exhdd/crab_pks/pks_crab_20-07-26/2026-07-19-23-14-19/J0534+2200/3968"
  "/mnt/exhdd/crab_pks/pks_crab_02-08-26/2026-08-02-23-09-56/J0534+2200/3968"
)

make_search_fil() {
  "$DIGIFIL" -F 32 -d 1 -b 8 -I 0 -o "$1.fil" "$1"
}
export -f make_search_fil

for dir in "${OBS[@]}"; do
  echo
  echo "############################################################"
  echo "## $dir"
  echo "############################################################"
  cd "$dir"

  if (( DO_FIL )); then
    mapfile -t DADA < <(find . -maxdepth 1 -name '*.dada' ! -name '*.dada.fil' | sort)
    mapfile -t NEED < <(for f in "${DADA[@]}"; do [[ -f "$f.fil" ]] || printf '%s\n' "$f"; done)

    echo "== [1/3] forming search filterbanks (${#NEED[@]} missing / ${#DADA[@]} fragments) =="
    if (( ${#NEED[@]} )); then
      set +e
      printf '%s\n' "${NEED[@]}" | xargs -P "$FIL_JOBS" -I{} bash -c 'make_search_fil "$@"' _ {}
      rc=$?
      set -e
      if (( rc )); then
        echo "  WARNING: digifil failed for at least one fragment (rc=$rc) — check for zero-byte .fil"
      fi
      n_fil=$(find . -maxdepth 1 -name '*.dada.fil' | wc -l)
      echo "  done: $n_fil / ${#DADA[@]} search filterbanks present"
    else
      echo "  nothing to do."
    fi
  fi

  if (( DO_SEARCH )); then
    echo "== [2/3] transientX search (1us 5us 10us 20us 40us) =="
    bash "$SCRIPT_DIR/tx.sh" all
  fi

  if (( DO_EXTRACT )); then
    echo "== [3/3] candidate extraction =="
    "$VENV_PYTHON" "$SCRIPT_DIR/extract_cands.py" \
      --cand-dir cands \
      --workdir "$PWD" \
      --outdir cutouts \
      --min-snr 5 \
      --digifil-min-block 2.0 \
      --plot
  fi
done

echo
echo "All observations processed."
