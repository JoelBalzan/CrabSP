#!/usr/bin/env python3
"""Compare candidate lists from two time resolutions.

For each candidate found at the finer resolution (x), check whether a
coinciding candidate exists at the coarser resolution (y) within some MJD
tolerance.

Usage:
    python compare_res.py cands/1us/*.cands cands/5us/*.cands
    python compare_res.py --gap 0.005 cands/1us cands/10us
"""
import argparse
import sys
from pathlib import Path


def parse_cands(path):
    """Return list of {mjd, dm, snr, width_ms, fil} dicts from a .cands file."""
    cands = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split()
            if len(p) < 6:
                continue
            try:
                cands.append({
                    'cand_id': p[1],
                    'mjd': float(p[2]),
                    'dm': float(p[3]),
                    'width_ms': float(p[4]),
                    'snr': float(p[5]),
                    'fil': p[-1],
                })
            except (ValueError, IndexError):
                continue
    return cands


def cluster(cands, gap_s):
    """Group candidates into events by MJD proximity."""
    cands = sorted(cands, key=lambda c: c['mjd'])
    events = []
    for c in cands:
        if events and (c['mjd'] - events[-1][-1]['mjd']) * 86400.0 <= gap_s:
            events[-1].append(c)
        else:
            events.append([c])
    return events


def best_per_event(events):
    """Return the highest-SNR candidate from each event."""
    return [max(ev, key=lambda c: c['snr']) for ev in events]


def load_cands(path):
    """Load all .cands files from a path (file or directory)."""
    p = Path(path)
    if p.is_file():
        return parse_cands(p)
    elif p.is_dir():
        all_cands = []
        for f in sorted(p.rglob('*.cands')):
            all_cands.extend(parse_cands(f))
        return all_cands
    else:
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('finer', help='finer resolution .cands file or directory')
    ap.add_argument('coarser', help='coarser resolution .cands file or directory')
    ap.add_argument('--gap', type=float, default=6.0,
                     help='clustering gap (ms) for merging same-pulse detections '
                          'within each resolution before comparing (default: 6)')
    ap.add_argument('--match-gap', type=float, default=None,
                     help='max MJD separation (ms) to consider a fine-res '
                          'candidate "found" in coarse-res (default: same as --gap)')
    ap.add_argument('--min-snr', type=float, default=0.0,
                     help='only compare candidates with S/N >= this threshold')
    args = ap.parse_args()

    gap_s = args.gap / 1000.0
    match_gap_s = (args.match_gap / 1000.0) if args.match_gap is not None else gap_s

    fine_raw = load_cands(args.finer)
    coarse_raw = load_cands(args.coarser)

    fine_ev = [c for c in best_per_event(cluster(fine_raw, gap_s))
               if c['snr'] >= args.min_snr]
    coarse_ev = [c for c in best_per_event(cluster(coarse_raw, gap_s))
                 if c['snr'] >= args.min_snr]

    # Sort coarse events by MJD for binary-ish search
    coarse_ev.sort(key=lambda c: c['mjd'])

    coarse_mjds = [c['mjd'] for c in coarse_ev]

    found = 0
    matched_cands = []
    missed = []
    for cand in fine_ev:
        mjd = cand['mjd']
        match = next(
            (coarse_ev[j] for j, cmjd in enumerate(coarse_mjds)
             if abs(mjd - cmjd) * 86400.0 <= match_gap_s),
            None,
        )
        if match is not None:
            found += 1
            matched_cands.append((cand, match))
        else:
            missed.append(cand)

    total = len(fine_ev)
    pct = 100.0 * found / total if total else 0

    print(f"Finer resolution : {args.finer}")
    print(f"Coarser resolution: {args.coarser}")
    print(f"Cluster gap       : {args.gap} ms")
    print(f"Match tolerance   : {args.match_gap or args.gap} ms")
    if args.min_snr > 0:
        print(f"Min S/N           : {args.min_snr}")
    print()
    print(f"Fine-res events   : {total}")
    print(f"Coarse-res events : {len(coarse_ev)}")
    print(f"Found in coarse   : {found} / {total} ({pct:.1f}%)")
    print(f"Missed            : {len(missed)}")
    print()

    if matched_cands:
        matched_cands.sort(key=lambda pair: pair[0]['snr'], reverse=True)
        print("Matched candidates (fine -> coarse):")
        print(f"  {'Cand':>10s}  {'Cand_co':>10s}  {'MJD':>20s}  {'DM':>8s}  {'S/N':>7s}  {'W (ms)':>7s}  |  {'S/N_co':>7s}  {'W_co (ms)':>9s}")
        for fc, cc in matched_cands:
            print(f"  {fc['cand_id']:>10s}  {cc['cand_id']:>10s}  {fc['mjd']:20.10f}  {fc['dm']:8.2f}  {fc['snr']:7.1f}  {fc['width_ms']:7.2f}"
                  f"  |  {cc['snr']:7.1f}  {cc['width_ms']:9.2f}")
        print()

    if missed:
        missed.sort(key=lambda c: c['snr'], reverse=True)
        print("Missed candidates (sorted by S/N):")
        print(f"  {'Cand':>10s}  {'MJD':>20s}  {'DM':>8s}  {'S/N':>7s}  {'Width (ms)':>10s}")
        for c in missed:
            print(f"  {c['cand_id']:>10s}  {c['mjd']:20.10f}  {c['dm']:8.2f}  {c['snr']:7.1f}  {c['width_ms']:10.2f}")
    else:
        print("All fine-res candidates found in coarse-res.")


if __name__ == '__main__':
    main()
