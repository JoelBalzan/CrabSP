"""dspsr folding: plan fold windows, run dspsr to produce .ar archives."""
import subprocess
from pathlib import Path

import numpy as np

from .headers import parse_dada_header


def plan_dspsr_fold(frags, cand_mjd, margin_s, period, turns):
    """Covering fragments + seek MJD for a dspsr fold.

    dspsr seeks on absolute MJD, so fragments are selected by MJD overlap.
    The fold is anchored so phase 0 = the seek epoch, placing the burst near
    bin 0 of the folded profile.

    Returns (dada_paths, seek_mjd, turns, note); (None, ...) if uncovered.
    """
    start_mjd = cand_mjd - margin_s / 86400.0
    end_mjd = start_mjd + (turns * period) / 86400.0
    cover = [f for f in frags
             if f['tstart_mjd'] < end_mjd and f['t_end_mjd'] > start_mjd]
    if not cover:
        return None, None, None, "candidate falls outside all fragments"
    first = cover[0]
    if start_mjd < first['tstart_mjd']:
        start_mjd = first['tstart_mjd']
    last = cover[-1]
    avail_s = (last['t_end_mjd'] - start_mjd) * 86400.0
    max_turns = avail_s / period
    if max_turns < 0.5:
        return (None, None, None,
                f"candidate is only {avail_s:.3f}s before the end of the "
                f"observation — cannot fold half a period; skipping")
    turns = max(1, min(int(turns), int(max_turns)))
    return [f['dada_path'] for f in cover], start_mjd, turns, None


def fold_cutout(dada_paths, seek_mjd, dm, period, nbin, nchan, turns, outname,
                outdir, dspsr_bin='dspsr', cf_offset_mhz=0.0, parfile=None,
                pac_dbase=None, rm=None):
    """Fold coherency products around one or more pulses with dspsr -> .ar file.

    -seek anchors the output at the candidate's MJD; -cepoch makes that the
    phase-0 reference so the burst lands in the first bins.  -D -K dedisperse;
    -d 4 requests PP,QQ,Re[PQ],Im[PQ] coherency products.

    If parfile is given, phase comes from -E (TEMPO2 ephemeris) instead of
    -c/-cepoch.  Required for turns > 1 to keep phase coherent.

    If cf_offset_mhz is non-zero, the DADA band centre is shifted for pac
    channel-grid alignment.  If pac_dbase is given, pac calibration is applied
    inline via -pac.  rm enables coherent Faraday derotation.

    Returns the output .ar Path, or None on failure.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    out_ar = outdir / f"{outname}.ar"
    if out_ar.exists():
        out_ar.unlink()
    cmd = [
        dspsr_bin,
        '-seek', f'{seek_mjd:.9f}',
        '-turns', str(turns),
    ]
    if parfile:
        cmd += ['-E', str(parfile)]
    else:
        cmd += ['-c', f'{period:.9f}', '-cepoch', f'{seek_mjd:.9f}']
    cmd += [
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
    if pac_dbase:
        cmd += ['-pac', str(pac_dbase)]
    if rm is not None:
        cmd += ['-derotate', '-rm', f'{rm:.6f}']
    if cf_offset_mhz:
        dada_hdr = parse_dada_header(dada_paths[0])
        band_mhz = float(dada_hdr.get('FREQ', 0.0))
        bw_mhz = float(dada_hdr.get('BW', 0.0))
        if band_mhz and bw_mhz:
            cmd[1:1] = ['-f', f'{band_mhz + cf_offset_mhz:.6f}',
                        '-B', f'{bw_mhz:.6f}']
            print(f"  fold centre shifted {band_mhz:g} -> "
                  f"{band_mhz + cf_offset_mhz:g} MHz (grid align for pac)")
    if len(dada_paths) > 1:
        cmd.append('-cont')
    cmd.extend(str(p) for p in dada_paths)
    print("\nRunning dspsr fold")
    print(" ".join(cmd))
    timeout_s = max(120.0, 30.0 * turns * period)
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
