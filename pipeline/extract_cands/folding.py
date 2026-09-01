"""dspsr folding: plan fold windows, run dspsr to produce .ar archives."""
import subprocess
from pathlib import Path

import numpy as np


def plan_dspsr_fold(frags, cand_mjd, margin_s, period):
    """Covering fragments + seek MJD for a one-turn dspsr fold.

    dspsr seeks on absolute MJD, so fragments are selected by MJD overlap.
    The fold is anchored so phase 0 = the seek epoch, placing the burst near
    bin 0 of the folded profile.

    Returns (dada_paths, seek_mjd, note); (None, None, note) if uncovered.
    """
    start_mjd = cand_mjd - margin_s / 86400.0
    end_mjd = start_mjd + period / 86400.0
    cover = [f for f in frags
             if f['tstart_mjd'] < end_mjd and f['t_end_mjd'] > start_mjd]
    if not cover:
        return None, None, "candidate falls outside all fragments"
    first = cover[0]
    if start_mjd < first['tstart_mjd']:
        start_mjd = first['tstart_mjd']
    last = cover[-1]
    avail_s = (last['t_end_mjd'] - start_mjd) * 86400.0
    if avail_s < 0.5 * period:
        return (None, None,
                f"candidate is only {avail_s:.3f}s before the end of the "
                f"observation — cannot fold half a period; skipping")
    return [f['dada_path'] for f in cover], start_mjd, None


def fold_cutout(dada_paths, seek_mjd, dm, period, nbin, nchan, outname,
                outdir, rm=None):
    """Fold one spin period of coherency products around the pulse -> .ar file.

    -seek anchors the output at the candidate's MJD; -cepoch makes that the
    phase-0 reference so the burst lands in the first bins.  -D -K dedisperse;
    -d 4 requests PP,QQ,Re[PQ],Im[PQ] coherency products.  rm enables coherent
    Faraday derotation.

    Returns the output .ar Path, or None on failure.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    out_ar = outdir / f"{outname}.ar"
    if out_ar.exists():
        out_ar.unlink()
    cmd = [
        'dspsr',
        '-seek', f'{seek_mjd:.9f}',
        '-turns', '1',
        '-c', f'{period:.9f}',
        '-cepoch', f'{seek_mjd:.9f}',
        '-D', f'{dm:.4f}',
        '-K',
        '-F', str(nchan),
        '-d', '4',
        '-b', str(nbin),
        '-A',
        '-e', 'ar',
        '-O', str(outdir / outname),
    ]
    if nbin > 32768:
        nfft = 1
        while nfft < 2 * nbin:
            nfft <<= 1
        cmd += ['-x', str(nfft)]
        min_samples = 2 * nfft * nchan
        min_ram_mb = max(512, int(np.ceil(min_samples * 128 / 1e6)))
        cmd += ['-U', str(min_ram_mb)]
    if rm is not None:
        cmd += ['-derotate', '-rm', f'{rm:.6f}']
    if len(dada_paths) > 1:
        cmd.append('-cont')
    cmd.extend(str(p) for p in dada_paths)
    print("\nRunning dspsr fold")
    print(" ".join(cmd))
    timeout_s = max(120.0, 30.0 * period)
    try:
        r = subprocess.run(cmd, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"    dspsr fold TIMED OUT after {timeout_s:.0f}s")
        return None
    if r.returncode != 0:
        print(f"    dspsr fold FAILED (rc={r.returncode}); "
              f"re-run the printed command manually to see the error")
        return None
    if not out_ar.exists() or out_ar.stat().st_size == 0:
        print('    dspsr exited 0 but wrote no archive')
        return None
    return out_ar
