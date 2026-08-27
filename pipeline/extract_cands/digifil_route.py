"""digifil extraction route: plan, extract, read cutout filterbanks."""
import subprocess
from pathlib import Path

import numpy as np
from sigpyproc.readers import FilReader


def plan_extraction(frags, frag, offset_s, min_block_s=0.5, margin_s=0.05):
    """Plan the digifil extraction block.

    Starts the block a small margin before the candidate and lets it run
    min_block seconds.  If it crosses a fragment boundary, neighbouring .dada
    files are passed with -cont.

    Returns (dada_paths, seek_s, dur_s, first_frag, note); (None, ...) if the
    candidate cannot be covered.
    """
    SAFE_MIN_BLOCK_S = 0.5
    block_s = max(min_block_s, SAFE_MIN_BLOCK_S)

    seek_s = max(0.0, offset_s - margin_s)
    block_start_abs = frag['tstart_mjd'] + seek_s / 86400.0
    block_end_abs = block_start_abs + block_s / 86400.0

    cover = [
        f for f in frags
        if f['tstart_mjd'] < block_end_abs and f['t_end_mjd'] > block_start_abs
    ]
    if not cover:
        return None, None, None, None, "candidate falls outside all fragments"

    first = cover[0]
    last = cover[-1]
    if block_end_abs > last['t_end_mjd']:
        block_s = (last['t_end_mjd'] - block_start_abs) * 86400.0
        if block_s < SAFE_MIN_BLOCK_S:
            return (None, None, None, None,
                    f"candidate is only {block_s:.2f}s before the end of the "
                    f"observation — digifil cannot read past it; skipping")

    digifil_seek_s = (block_start_abs - first['tstart_mjd']) * 86400.0
    dada_paths = [f['dada_path'] for f in cover]
    return dada_paths, digifil_seek_s, block_s, first, None


def trim_to_window(stokes, block_tstart_mjd, tsamp_s, cand_mjd, window_s):
    """Trim a (4, nchan, nsamp) cutout to a centered window around cand_mjd.

    Returns (trimmed_stokes, trimmed_tstart_mjd).
    """
    nsamp = stokes.shape[-1]
    cand_offset_in_block_s = (cand_mjd - block_tstart_mjd) * 86400.0
    i0 = int(round((cand_offset_in_block_s - window_s / 2.0) / tsamp_s))
    i1 = int(round((cand_offset_in_block_s + window_s / 2.0) / tsamp_s))
    i0 = max(0, i0)
    i1 = min(nsamp, i1)
    if i1 <= i0:
        raise RuntimeError(
            f"candidate falls outside extracted block (i0={i0}, i1={i1}, nsamp={nsamp}) "
            f"— increase --digifil-min-block (or the candidate is within ~0.3 s "
            f"of the end of the observation and cannot be covered)"
        )
    trimmed_tstart_mjd = block_tstart_mjd + (i0 * tsamp_s) / 86400.0
    return stokes[..., i0:i1], trimmed_tstart_mjd


def extract_cutout(dada_paths, seek_s, dur_s, dm, outname, outdir,
                    digifil_bin='digifil', nbits=-32, fft=32):
    """Run digifil to form a full-Stokes filterbank cutout.

    Returns the output .fil Path, or None on failure.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    out_fil = outdir / f"{outname}.fil"
    if out_fil.exists():
        out_fil.unlink()
    cmd = [
        digifil_bin,
        '-S', f'{max(seek_s, 0):.6f}',
        '-T', f'{dur_s:.6f}',
        '-F', str(fft),
        '-d', '4',
        '-D', f'{dm:.4f}',
        '-K',
        '-b', str(nbits),
        '-I', '0',
        '-o', str(out_fil),
    ]
    if len(dada_paths) > 1:
        cmd.append('-cont')
    cmd.extend(str(p) for p in dada_paths)
    print("\nRunning digifil")
    print(" ".join(cmd))
    timeout_s = max(120.0, 15.0 * (seek_s + dur_s))
    try:
        r = subprocess.run(cmd, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"    digifil TIMED OUT after {timeout_s:.0f}s "
              f"(seek={seek_s:.2f}s + dur={dur_s:.2f}s) — this is the known "
              f"short--T hang; keep --digifil-min-block >= 2.0s")
        return None
    if r.returncode != 0:
        print(f'    digifil FAILED (rc={r.returncode}); re-run the printed command manually to see the error')
        return None
    if not out_fil.exists() or out_fil.stat().st_size == 0:
        print('    digifil exited 0 but wrote no data (blocksize/seek mismatch?)')
        return None
    return out_fil


def read_fil_cube(fil_path):
    """Read a digifil -d4 cutout into shape (4, nchan, nsamp).

    Pol order: PP, QQ, Re[PQ], Im[PQ].  Handles both 2D and 3D sigpyproc
    output layouts.
    """
    fil = FilReader(str(fil_path))
    h = fil.header
    nifs = getattr(h, 'nifs', 1)
    if nifs != 4:
        raise RuntimeError(
            f"{fil_path}: header reports nifs={nifs} (expected 4) — cutout "
            f"wasn't written with full coherency products, check digifil -d 4 applied."
        )

    block = fil.read_block(0, h.nsamples)
    arr = np.asarray(block.data)

    if arr.ndim == 3:
        if arr.shape[0] == nifs:
            cube = arr
        elif arr.shape[-1] == nifs:
            cube = np.transpose(arr, (2, 1, 0))
        else:
            raise RuntimeError(f"{fil_path}: couldn't infer pol axis from shape {arr.shape}, nifs={nifs}")
    elif arr.ndim == 2:
        nchan, flat = arr.shape
        if flat % nifs:
            raise RuntimeError(
                f"{fil_path}: flat sample axis {flat} not divisible by nifs={nifs} — "
                f"can't safely unpack pol/time, inspect arr.shape by hand."
            )
        nsamp_time = flat // nifs
        cube = arr.reshape(nchan, nsamp_time, nifs)
        cube = np.transpose(cube, (2, 0, 1))
    else:
        raise RuntimeError(f"{fil_path}: unexpected block ndim {arr.ndim}, shape {arr.shape}")

    return cube


def coherency_to_stokes(cube):
    """Convert coherency products (4, nchan, nsamp) PP,QQ,Re[PQ],Im[PQ] to IQUV.

    Matches dspsr's own Stokes formation: I=PP+QQ, Q=PP-QQ, U=2Re[PQ], V=2Im[PQ].
    """
    PP, QQ, RPQ, IPQ = cube[0], cube[1], cube[2], cube[3]
    I = PP + QQ
    Q = PP - QQ
    U = 2 * RPQ
    V = 2 * IPQ
    return np.stack([I, Q, U, V], axis=0)
