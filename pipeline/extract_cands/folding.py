"""dspsr folding: plan fold windows, run dspsr to produce .ar archives."""
import subprocess
from pathlib import Path

import numpy as np

_OPTIMAL_NFFT_OVERSAMPLE = 2.0
_OPTIMAL_NFFT_MIN = 1024


def optimal_nfft(dm, bw_mhz, center_mhz, nchan):
    """Coherent-dedispersion FFT length for one output channel.

    The dspsr dedispersion filter must cover the dispersive sweep across a
    single channel of the fold band, and residual smearing per FFT bin kept
    below half an output sample (i.e. Nyquist-sampled output).  Both criteria
    reduce to ~2x the sweep, measured in samples at the per-channel rate.

    Returns the next power of two >= 2 * sweep_samples, or None if the
    inputs are insufficient to compute it.
    """
    if dm <= 0 or bw_mhz <= 0 or center_mhz <= 0 or nchan <= 0:
        return None
    chan_bw_mhz = bw_mhz / nchan
    f_lo_ghz = (center_mhz - chan_bw_mhz / 2.0) / 1e3
    f_hi_ghz = (center_mhz + chan_bw_mhz / 2.0) / 1e3
    if f_lo_ghz <= 0:
        return None
    sweep_s = (4.148808e-3 * dm * (1 / f_lo_ghz**2 - 1 / f_hi_ghz**2)
               / 1e3)
    sweep_samples = sweep_s * chan_bw_mhz * 1e6
    nfft = _OPTIMAL_NFFT_MIN
    while nfft < _OPTIMAL_NFFT_OVERSAMPLE * sweep_samples:
        nfft <<= 1
    return nfft


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
                outdir, rm=None, bw_mhz=0.0, center_mhz=0.0):
    """Fold one spin period of coherency products around the pulse -> .ar file.

    -seek anchors the output at the candidate's MJD; -cepoch makes that the
    phase-0 reference so the burst lands in the first bins.  -D -K dedisperse;
    -d 4 requests PP,QQ,Re[PQ],Im[PQ] coherency products.  rm enables coherent
    Faraday derotation.  -x sets the dedispersion FFT to the optimal length
    derived from DM / per-channel bandwidth / centre frequency, so the folded
    profile is Nyquist-sampled with minimum filter history.

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
    nfft = optimal_nfft(dm, bw_mhz, center_mhz, nchan)
    if nfft is not None:
        cmd += ['-x', str(nfft)]
        min_samples = 2 * nfft * nchan
        min_ram_mb = max(512, int(np.ceil(min_samples * 128 / 1e6)))
        cmd += ['-U', str(min_ram_mb)]
        print(f"\nOptimal dedispersion FFT length: -x {nfft} "
              f"({nfft / (bw_mhz * 1e6) * 1e3:.2f} ms at "
              f"{bw_mhz:.1f} MHz / {nchan} ch)")
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
