#!/usr/bin/env python3
"""Inventory unique bursts across all time resolutions.

Groups candidates from every tres subdir under a top-level cands/ dir into
global events (unique bursts) by MJD, then lists which tres found each.

Like compare_res.py but N-way: instead of finer vs coarser, it answers
"for each of the N unique bursts in the data set, which resolutions saw it?"

Usage:
    python burst_inventory.py cands
    python burst_inventory.py cands --gap 6 --min-snr 10
    python burst_inventory.py cands --gap 6 --match-gap 6 -v
    python burst_inventory.py cands --out bursts.txt
    python burst_inventory.py cands --out-cands unique.cands  # for extract_cands

    # feed unique bursts (one rep per global event) to the extractor:
    python burst_inventory.py cands --gap 6 --out-cands /tmp/unique.cands
    python extract_cands.py --cand-files /tmp/unique.cands --workdir . --outdir cutouts --cluster-gap-ms 0

Output (example):
    #  MJD                 N  DM     S/N  W(ms)  Found                                          Missing
    1  60345.1234567890    2  56.7   42.1  0.03   0.25us,0.5us                                     1us,5us,10us
    2  60345.1235678901    1  56.6   11.2  0.12   5us                                              0.25us,0.5us,1us,10us
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict


def parse_cands(path):
    """Return list of {beam,cand_id,mjd,dm,width_ms,snr,fil,raw} dicts from a .cands file."""
    cands = []
    with open(path) as f:
        for line in f:
            raw = line.rstrip("\n")
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith("#"):
                continue
            p = line_stripped.split()
            if len(p) < 6:
                continue
            try:
                # transientX: col0 beam col1 id col2 mjd col3 dm col4 width_ms col5 snr ... last fil
                cands.append({
                    'beam': p[0],
                    'cand_id': p[1],
                    'mjd': float(p[2]),
                    'dm': float(p[3]),
                    'width_ms': float(p[4]),
                    'snr': float(p[5]),
                    'fil': p[-1],
                    'raw': raw,
                })
            except (ValueError, IndexError):
                continue
    return cands


def cluster(cands, gap_s):
    """Group candidates into events by MJD proximity (sorted)."""
    cands = sorted(cands, key=lambda c: c['mjd'])
    events = []
    for c in cands:
        if events and (c['mjd'] - events[-1][-1]['mjd']) * 86400.0 <= gap_s:
            events[-1].append(c)
        else:
            events.append([c])
    return events


def best_per_event(events):
    """Return highest-SNR cand from each event."""
    return [max(ev, key=lambda c: c['snr']) for ev in events]


def load_res(cands_dir, res_name, gap_s, min_snr):
    """Load and reduce one resolution subdir to best-per-event list.

    Returns (best, n_raw, n_raw_before_snr) for stats. best is tagged with res.
    """
    p = Path(cands_dir) / res_name
    files = sorted(p.rglob("*.cands"))
    # avoid double-counting .cands.orig / .tmp? .orig ends with .orig, not .cands, so rglob *.cands excludes it;
    # but be explicit: only files ending exactly .cands
    files = [f for f in files if f.suffix == ".cands" or f.name.endswith(".cands")]
    # filter to .cands exactly (exclude .cands.orig)
    files = [f for f in files if f.name.endswith(".cands") and not f.name.endswith(".orig")]
    # simpler: keep only where suffixes == ['.cands']
    files = [f for f in sorted(p.glob("*.cands"))]  # top-level only, matches tx.sh
    if not files:
        # fallback rglob if nested
        files = [f for f in sorted(p.rglob("*.cands")) if f.suffix == ".cands"]
    raw = []
    for f in files:
        raw.extend(parse_cands(f))
    n_raw = len(raw)
    if not raw:
        return [], n_raw
    ev = cluster(raw, gap_s)
    best_all = best_per_event(ev)
    n_clustered = len(best_all)
    best = [c for c in best_all if c['snr'] >= min_snr]
    # tag with res
    for c in best:
        c['res'] = res_name
    return best, n_raw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cands_dir", help="top-level cands dir (contains 0.25us/ 0.5us/ ... subdirs)")
    ap.add_argument("--gap", type=float, default=6.0,
                    help="clustering gap (ms) within each res and globally (default 6)")
    ap.add_argument("--match-gap", type=float, default=None,
                    help="global cross-res match gap ms (default same as --gap)")
    ap.add_argument("--min-snr", type=float, default=0.0, help="only keep S/N >= this")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="also print per-res S/N/W for each burst")
    ap.add_argument("--out", type=str, default=None, help="write table to file as well")
    ap.add_argument("--out-cands", type=str, default=None,
                    help="write a merged deduped .cands file (one rep per global burst) for extract_cands --cand-files")
    args = ap.parse_args()

    cands_dir = Path(args.cands_dir)
    if not cands_dir.is_dir():
        print(f"ERROR: {cands_dir} not a directory", file=sys.stderr)
        sys.exit(1)

    gap_s = args.gap / 1000.0
    match_gap_s = (args.match_gap / 1000.0) if args.match_gap is not None else gap_s

    # discover res subdirs : immediate children containing *.cands
    res_names = []
    for child in sorted(cands_dir.iterdir()):
        if child.is_dir() and any(child.glob("*.cands")):
            res_names.append(child.name)
    if not res_names:
        # try one level deeper (in case user passed cands/0.25us)
        if any(cands_dir.glob("*.cands")):
            res_names = [cands_dir.name]
            cands_dir = cands_dir.parent
        else:
            print(f"no res subdirs with *.cands found in {cands_dir}", file=sys.stderr)
            sys.exit(1)

    # optional natural sort by numeric tres: 0.25us < 0.5us < 1us < 5us...
    def tres_key(s):
        import re
        m = re.match(r"([0-9.]+)us", s)
        if m:
            return float(m.group(1))
        return float("inf")
    res_names.sort(key=tres_key)

    print(f"cands_dir : {cands_dir}")
    print(f"resolutions: {', '.join(res_names)}")
    print(f"gap {args.gap} ms  match-gap {args.match_gap or args.gap} ms  min_snr {args.min_snr}")
    print()

    # load each res
    per_res = {}
    per_res_nraw = {}
    all_best = []
    for res in res_names:
        best, n_raw = load_res(cands_dir, res, gap_s, args.min_snr)
        per_res[res] = best
        per_res_nraw[res] = n_raw
        all_best.extend(best)

    if not all_best:
        print("\nno candidates after filtering")
        # still show raw counts
        for res in res_names:
            print(f"  {res:>8s}: {0:4d} events (0 raw cands)")
        return

    # global clustering to define N unique bursts
    all_best.sort(key=lambda c: c['mjd'])
    # reuse cluster logic but need to keep res tag; cluster expects list of cands
    global_events = cluster(all_best, match_gap_s)

    # build table rows
    rows = []
    for gi, ev in enumerate(sorted(global_events, key=lambda ev: min(c['mjd'] for c in ev)), 1):
        # representative = highest S/N in the global event
        rep = max(ev, key=lambda c: c['snr'])
        # which res contributed?
        seen = {}
        for c in ev:
            # keep best per res within this global event (if a res had multiple cands within gap, keep max S/N)
            if c['res'] not in seen or c['snr'] > seen[c['res']]['snr']:
                seen[c['res']] = c
        found = sorted(seen.keys(), key=tres_key)
        missing = [r for r in res_names if r not in seen]
        rows.append((rep, seen, found, missing, ev))

    # sort rows by rep MJD
    rows.sort(key=lambda r: r[0]['mjd'])

    # --- start summary: per-res counts + exclusive ---
    print(f"global unique bursts: {len(global_events)} (clustered with {args.match_gap or args.gap} ms)")
    print()
    # exclusive = bursts seen in exactly that one res
    exclusive = {res: sum(1 for _, _, found, _, _ in rows if found == [res]) for res in res_names}
    print("Per-res inventory (at start):")
    print(f"  {'res':>8s}  {'raw':>6s}  {'events':>6s}  {'exclusive':>9s}  {'shared':>6s}")
    print("  " + "-"*48)
    for res in res_names:
        n_raw = per_res_nraw[res]
        n_ev = len(per_res[res])
        n_exc = exclusive[res]
        n_shared = n_ev - n_exc
        print(f"  {res:>8s}  {n_raw:6d}  {n_ev:6d}  {n_exc:9d}  {n_shared:6d}")
    print()

    # print table
    header = f"{'#':>4s}  {'MJD':>20s}  {'N':>2s}  {'DM':>6s}  {'S/N':>6s}  {'W(ms)':>7s}  {'Found':<30s}  {'Missing'}"
    lines = [header, "-" * len(header)]
    for i, (rep, seen, found, missing, ev) in enumerate(rows, 1):
        lines.append(
            f"{i:4d}  {rep['mjd']:20.10f}  {len(found):2d}  {rep['dm']:6.2f}  {rep['snr']:6.1f}  {rep['width_ms']:7.3f}  "
            f"{','.join(found):<30s}  {','.join(missing)}"
        )
        if args.verbose:
            # per-res details for this burst
            for res in found:
                c = seen[res]
                lines.append(f"       - {res:>8s}: S/N {c['snr']:5.1f}  W {c['width_ms']:6.3f} ms  DM {c['dm']:5.2f}  MJD {c['mjd']:.10f}  {c['fil']}")
    out_text = "\n".join(lines)
    print(out_text)
    if args.out:
        Path(args.out).write_text(out_text + "\n")
        print(f"\nwrote {args.out}")

    if args.out_cands:
        out_cands = Path(args.out_cands)
        with open(out_cands, "w") as f:
            for rep, _, _, _, _ in rows:
                raw = rep.get("raw")
                if raw is not None:
                    f.write(raw + "\n")
                else:
                    # fallback minimal reconstruction (beam cand_id mjd dm width snr ... fil)
                    beam = rep.get("beam", "0")
                    f.write(f"{beam} {rep['cand_id']} {rep['mjd']:.10f} {rep['dm']:.2f} {rep['width_ms']:.4f} {rep['snr']:.2f} 0 0 0 {rep['fil']}\n")
        print(f"\nwrote merged {len(rows)} unique cands -> {out_cands} (one rep per global burst)")
        print(f"  use: python extract_cands.py --cand-files {out_cands} --workdir $PWD --outdir cutouts --cluster-gap-ms 0")
        print(f"       (already deduped with gap {args.match_gap or args.gap} ms; extractor should use --cluster-gap-ms 0)")

    # summary: how many bursts were seen in exactly k resolutions
    from collections import Counter
    cnt = Counter(len(found) for _, _, found, _, _ in rows)
    print("\nSummary (bursts by N_res):")
    for k in sorted(cnt):
        print(f"  {k} res : {cnt[k]} bursts")
    # per-res recall
    print("\nPer-res recall (fraction of global bursts seen):")
    for res in res_names:
        n = sum(1 for _, seen, _, _, _ in rows if res in seen)
        print(f"  {res:>8s}: {n}/{len(rows)} ({100*n/len(rows):.1f}%)")


if __name__ == "__main__":
    main()
