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
#                               (default 1 — the fragments are on a single
#                               external HDD; parallel reads thrash the disk)
#
# Usage:  ./run_pipeline.sh > pipeline.log 2>&1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${VENV_PYTHON:-/home/joel/Documents/GitHub/CrabSP/.venv/bin/python}"
DIGIFIL="${DIGIFIL:-/home/joel/Pulsar/bin/digifil}"

export PATH="/home/joel/Pulsar/bin:/opt/TransientX/bin:$PATH"
export DIGIFIL
# The venv lives with the scripts in this repo (NOT in the data directories),
# so export it explicitly for tx.sh / extract_cands.py.
export VENV_PYTHON

DO_FIL=${DO_FIL:-1}
DO_SEARCH=${DO_SEARCH:-1}
DO_EXTRACT=${DO_EXTRACT:-1}
# The raw fragments live on a single external HDD (~0.3x real-time sequential
# read). Running multiple digifil in parallel makes each job ~3x slower than
# sequential (head thrashing), so this defaults to 1 — do not raise it.
FIL_JOBS=${FIL_JOBS:-1}

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

# A .dada fragment is 4096-byte header + N * float32 samples. digifil emits a
# 351-byte .fil header + N * 32 * 1-byte (8-bit) samples. So the .fil must be
#   expected = (dada_size - 4096) / 4 + 351
# bytes. digifil intermittently stops early on USB I/O contention, producing a
# shorter .fil; deleting it here makes the loop below regenerate it.
verify_fil_size() {
  local dada=$1 fil="$1.fil"
  [[ -f "$fil" ]] || return 0
  local dsz fsz exp
  dsz=$(stat -c%s "$dada")
  fsz=$(stat -c%s "$fil")
  exp=$(( (dsz - 4096) / 4 + 351 ))
  if (( fsz < exp - 1024 )); then
    echo "  TRUNCATED: ${fil##*/} ($fsz B < expected $exp B) — removing to regenerate"
    rm -f "$fil"
  fi
}

for dir in "${OBS[@]}"; do
  echo
  echo "############################################################"
  echo "## $dir"
  echo "############################################################"
  cd "$dir"

  if (( DO_FIL )); then
    mapfile -t DADA < <(find . -maxdepth 1 -name '*.dada' ! -name '*.dada.fil' | sort)

    echo "== [1/3] forming search filterbanks (${#DADA[@]} fragments) =="
    for f in "${DADA[@]}"; do
      verify_fil_size "$f"
    done
    mapfile -t NEED < <(for f in "${DADA[@]}"; do [[ -f "$f.fil" ]] || printf '%s\n' "$f"; done)

    echo "  ${#NEED[@]} to form / ${#DADA[@]} fragments"
    if (( ${#NEED[@]} )); then
      set +e
      printf '%s\n' "${NEED[@]}" | xargs -P "$FIL_JOBS" -I{} bash -c 'make_search_fil "$@"' _ {}
      rc=$?
      set -e
      if (( rc )); then
        echo "  WARNING: digifil failed for at least one fragment (rc=$rc) — check for zero-byte .fil"
      fi
      mapfile -t STILL_BAD < <(for f in "${DADA[@]}"; do
        if [[ ! -f "$f.fil" ]]; then
          printf '%s\n' "${f##*/} (missing)"
        else
          dsz=$(stat -c%s "$f"); fsz=$(stat -c%s "$f.fil")
          exp=$(( (dsz - 4096) / 4 + 351 ))
          (( fsz < exp - 1024 )) && printf '%s\n' "${f##*/} (still truncated)"
        fi
      done)
      if (( ${#STILL_BAD[@]} )); then
        echo "  WARNING: still bad after regeneration — rerun the pipeline to retry:"
        printf '    %s\n' "${STILL_BAD[@]}"
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
