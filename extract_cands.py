#!/usr/bin/env python3
"""
extract_cands.py — match transientX MJD candidates to the correct raw
.dada fragment (using sigpyproc to read the .fil header for tstart/tsamp/
nsamples), pull full-Stokes cutouts with digifil, convert to IQUV, and
save as .npz cubes (array + header metadata).

Requires sigpyproc.

Usage:
    python3 extract_cands.py crab_4us_*.cands \
        --workdir /path/to/raw/dada/dir \
        --outdir cutouts \
        --min-snr 5 \
        --plot
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
from sigpyproc.readers import FilReader

# --------------------------------------------------------------------------
# Header / fragment indexing
# --------------------------------------------------------------------------

def parse_fil_header(fil_path):
    """Read sigproc header fields we need via sigpyproc."""
    fil = FilReader(str(fil_path))
    h = fil.header
    nifs = getattr(h, 'nifs', 1)  # number of pols/IFs
    return {
        'tstart_mjd': float(h.tstart),
        'tsamp_s': float(h.tsamp),
        'nsamp': int(h.nsamples),
        'obslen_s': float(h.tsamp) * int(h.nsamples),
        'f1_mhz': float(h.fch1),
        'bw_mhz': float(h.foff),
        'nchan': int(h.nchans),
        'nifs': int(nifs),
    }

def build_fragment_index(workdir):
    frags = []
    for fil in sorted(Path(workdir).glob('*.dada.fil')):
        dada_path = Path(str(fil)[:-4])
        if not dada_path.exists():
            print(f"  WARNING: no raw .dada for {fil.name}, skipping")
            continue
        h = parse_fil_header(fil)
        t_end = h['tstart_mjd'] + h['obslen_s'] / 86400.0
        frags.append({'dada_path': dada_path, 'fil_path': fil, **h, 't_end_mjd': t_end})
    frags.sort(key=lambda f: f['tstart_mjd'])
    if not frags:
        return frags, None
    # transientX timestamps every candidate in a continuous-search reference
    # frame rooted at the FIRST searched file's tstart (that tstart is literally
    # in the .cands filename). So a burst in the N-th searched fragment is
    # reported ~sum(durations of files 0..N-1) too early in absolute MJD, and
    # naive absolute-MJD matching fails for any fragment after the first.
    # Precompute where each fragment starts within that stream.
    cum = 0.0
    for f in frags:
        f['stream_start_s'] = cum
        cum += f['obslen_s']
    return frags, frags[0]['tstart_mjd']


def find_fragment(frags, stream_root, mjd, tol_s=0.01):
    """Locate the searched fragment containing a candidate.

    Primary: continuous-stream matching.  global_s = (mjd - stream_root)*86400
    is walked through the fragments cumulatively; the local offset within the
    matching fragment is returned.  Fallback: absolute-MJD matching for cands
    files (e.g. hand-made) that already carry absolute timestamps.

    Returns (frag, offset_within_frag_s) or (None, None).
    """
    if stream_root is not None:
        global_s = (mjd - stream_root) * 86400.0
        if global_s >= 0:
            for f in frags:
                if f['stream_start_s'] <= global_s < f['stream_start_s'] + f['obslen_s']:
                    return f, global_s - f['stream_start_s']
    tol_days = tol_s / 86400.0
    for f in frags:
        if f['tstart_mjd'] - tol_days <= mjd < f['t_end_mjd'] + tol_days:
            return f, (mjd - f['tstart_mjd']) * 86400.0
    return None, None


def is_burst_duplicate(mjd, seen_mjds, tol_s=0.001):
    """True if a candidate with an MJD within tol_s has already been extracted.

    transientX reports the same burst at several neighbouring trial DMs, each
    with a slightly different peak MJD (the dispersive delay moves the peak by
    ~0.2 ms per 0.05 DM here). Those are one physical event, so dedup on the
    burst time rather than the exact MJD / offset.
    """
    for prev in seen_mjds:
        if abs(mjd - prev) * 86400.0 <= tol_s:
            return True
    return False


# --------------------------------------------------------------------------
# Candidate parsing
# --------------------------------------------------------------------------

def parse_cand_line(line):
    p = line.split()
    return {
        'beam': p[0],
        'cand_id': p[1],
        'mjd': float(p[2]),
        'dm': float(p[3]),
        'width_ms': float(p[4]) if len(p) > 4 else None,
        'snr': float(p[5]) if len(p) > 5 else None,
        'fil_path_in_cand': p[-1],
    }


def dm_smear_s(dm, f1_mhz, bw_mhz, nchan):
    """Dispersive smear across the observing band at this DM (s)."""
    f_hi = f1_mhz
    f_lo = f1_mhz + bw_mhz * nchan
    f_lo, f_hi = sorted([f_lo, f_hi])
    return 4.1488e3 * dm * (f_lo**-2 - f_hi**-2)


def window_for_cand(c, frag, min_window_s=0.02, width_factor=10, snr_factor_floor=0.005,
                     max_window_s=None):
    """Window to request from digifil, in seconds."""
    width_s = (c['width_ms'] or 0) / 1000.0
    smear_s = dm_smear_s(c['dm'], frag['f1_mhz'], frag['bw_mhz'], frag['nchan'])
    half_span = max(width_factor * width_s, snr_factor_floor) + smear_s
    window_s = max(2 * half_span, min_window_s)
    if max_window_s is not None:
        window_s = min(window_s, max_window_s)
    return window_s


# --------------------------------------------------------------------------
# digifil extraction
# --------------------------------------------------------------------------

def digifil_extraction_span(offset_s, frag, min_block_s=6.0):
    """
    Forming a filterbank on-the-fly from voltages (-F) requires an FFT with a
    settle/edge region discarded on each side to avoid convolution artifacts.
    For short -T requests the whole window can fall inside that discarded edge,
    producing zero valid output samples — and, worse, requests below ~1.5 s
    HANG digifil entirely (confirmed: -T 2.0 works for every -F from 16 to 128,
    -T <= 1.2 never finishes). Work around this by always requesting a block
    >= SAFE_MIN_BLOCK_S (clipped to the fragment's duration), centred on the
    candidate, and trimming to the desired small window client-side after
    reading the data back.

    Returns (seek_s, duration_s) to pass to digifil.
    """
    SAFE_MIN_BLOCK_S = 0.5
    frag_dur_s = frag['nsamp'] * frag['tsamp_s']

    block_s = max(min_block_s, SAFE_MIN_BLOCK_S)
    if block_s > frag_dur_s:
        if min_block_s < SAFE_MIN_BLOCK_S:
            print(f"    [warning] fragment is only {frag_dur_s:.2f}s long — shorter than the "
                  f"safe digifil block {SAFE_MIN_BLOCK_S}s; this run may hang")
        block_s = frag_dur_s
    elif min_block_s < SAFE_MIN_BLOCK_S:
        print(f"    [warning] requested digifil block {min_block_s}s is below the safe "
              f"minimum {SAFE_MIN_BLOCK_S}s (shorter -T hangs digifil); using {block_s:.2f}s")

    seek_s = offset_s - block_s / 2.0
    seek_s = max(0.0, min(seek_s, frag_dur_s - block_s))
    return seek_s, block_s


def trim_to_window(stokes, block_tstart_mjd, tsamp_s, cand_mjd, window_s):
    """
    stokes: (4, nchan, nsamp) covering the larger digifil-safe block.
    block_tstart_mjd: MJD of sample 0 of that block (from the cutout .fil header).
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
            f"— increase --digifil-min-block"
        )
    trimmed_tstart_mjd = block_tstart_mjd + (i0 * tsamp_s) / 86400.0
    return stokes[..., i0:i1], trimmed_tstart_mjd


def extract_cutout(dada_path, offset_s, window_s, dm, outname, outdir,
                    digifil_bin='digifil', nbits=-32, fft=32):
    outdir.mkdir(parents=True, exist_ok=True)
    out_fil = outdir / f"{outname}.fil"
    if out_fil.exists():
        out_fil.unlink()  # digifil refuses to overwrite; clear stale/partial files first
    cmd = [
        digifil_bin,
        '-S', f'{max(offset_s, 0):.6f}',
        '-T', f'{window_s:.6f}',
        '-F', str(fft),  # FFT factor -> number of channels, sets the time
                          # resolution: with a 32 MHz band, -F 32 gives 32 x 1 MHz
                          # channels and 1 us raw dt; larger -F -> finer dt
        '-d', '4',         # npol=4 -> PP,QQ,PQ,QP coherency products
        '-D', f'{dm:.4f}',  # DM value to use for dedispersion
        '-K',                # actually remove inter-channel dispersion delays
                              # using that DM — -D alone only sets the value,
                              # it does not trigger dedispersion on its own
        '-b', str(nbits),  # negative = float output (SigProcDigitizer only
                            # accepts 1/2/4/8/16 unsigned int, or -32 float)
        '-I', '0',          # disable digifil's default rescale/mean-subtract —
                             # without this, AA/BB (physically non-negative
                             # powers) come back ~50% negative
        '-o', str(out_fil),
        str(dada_path),
    ]
    print("\nRunning digifil")
    print(" ".join(cmd))
    # Don't capture output: digifil writes a \r progress bar to stderr and
    # capturing it hides all progress (looks like a 10-minute freeze). The
    # timeout is a safety net for the known short--T hang.
    timeout_s = max(120.0, 15.0 * (offset_s + window_s))
    try:
        r = subprocess.run(cmd, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"    digifil TIMED OUT after {timeout_s:.0f}s "
              f"(seek={offset_s:.2f}s + dur={window_s:.2f}s) — this is the known "
              f"short--T hang; keep --digifil-min-block >= 2.0s")
        return None
    if r.returncode != 0:
        print(f'    digifil FAILED (rc={r.returncode}); re-run the printed command manually to see the error')
        return None
    if not out_fil.exists() or out_fil.stat().st_size == 0:
        print('    digifil exited 0 but wrote no data (blocksize/seek mismatch?)')
        return None
    return out_fil


# --------------------------------------------------------------------------
# Reading the cutout filterbank -> (npol, nchan, nsamp) array, via sigpyproc
# --------------------------------------------------------------------------

def read_fil_cube(fil_path):
    """
    Read a digifil -d4 filterbank cutout into shape (4, nchan, nsamp),
    pol order AA, BB, CR, CI, using sigpyproc.
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
        # Some sigpyproc versions already separate the pol/IF axis.
        if arr.shape[0] == nifs:
            cube = arr  # (npol, nchan, nsamp)
        elif arr.shape[-1] == nifs:
            cube = np.transpose(arr, (2, 1, 0))  # (nsamp, nchan, npol) -> (npol, nchan, nsamp)
        else:
            raise RuntimeError(f"{fil_path}: couldn't infer pol axis from shape {arr.shape}, nifs={nifs}")

    #elif arr.ndim == 2:
    #    # Observed on this sigpyproc/data combo: (nchan, nsamp_time * nifs) — the
    #    # IF/pol axis isn't split out, it's folded into the "samples" axis.
    #    # Empirically confirmed via the neg-fraction sanity check below: the
    #    # flat axis unpacks as (nifs, nsamp_time), i.e. IF varies slower than
    #    # time for a fixed channel (all of IF0's samples, then all of IF1's, ...).
    #    nchan, flat = arr.shape
    #    if flat % nifs != 0:
    #        raise RuntimeError(
    #            f"{fil_path}: flat sample axis {flat} not divisible by nifs={nifs} — "
    #            f"can't safely unpack pol/time, inspect arr.shape by hand."
    #        )
    #    nsamp_time = flat // nifs
    #    cube = arr.reshape(nchan, nifs, nsamp_time)  # confirmed order via neg-fraction check
    #    cube = np.transpose(cube, (1, 0, 2))  # -> (nifs, nchan, nsamp_time)
    #else:
    #    raise RuntimeError(f"{fil_path}: unexpected block ndim {arr.ndim}, shape {arr.shape}")

    elif arr.ndim == 2:
    
        #print("\n========== RAW ARRAY ==========")
        #print("arr.shape =", arr.shape)
        #print("\nFirst channel, first 32 values")
        #print(arr[0,:32])
        #print("\nSecond channel, first 32 values")
        #print(arr[1,:32])
        #print("\nColumn 0")
        #print(arr[:,0])
        #print("\nColumn 1")
        #print(arr[:,1])
        #print("\nColumn 2")
        #print(arr[:,2])
    
        nchan, flat = arr.shape
        nsamp_time = flat // nifs
    
        cube1 = arr.reshape(nchan, nifs, nsamp_time)
        cube1 = np.transpose(cube1,(1,0,2))
        
        cube2 = arr.reshape(nchan, nsamp_time, nifs)
        cube2 = np.transpose(cube2,(2,0,1))
        
        for name,cube in [("cube1",cube1),("cube2",cube2)]:
            print(name)
            for i in range(4):
                print(i,cube[i].mean(),cube[i].std())

    return cube  # (4, nchan, nsamp), order AA, BB, CR, CI


def coherency_to_stokes(cube):
    """cube: (4, nchan, nsamp) in AA, BB, CR, CI -> IQUV."""
    AA, BB, CR, CI = cube[0], cube[1], cube[2], cube[3]
    I = AA + BB
    Q = AA - BB
    U = 2 * CR
    V = 2 * CI
    return np.stack([I, Q, U, V], axis=0)  # (4, nchan, nsamp), order I,Q,U,V


# --------------------------------------------------------------------------
# Diagnostic plotting
# --------------------------------------------------------------------------

def debiased_L(Q_ts, U_ts, sigma):
    """Ricean-debiased linear polarisation profile (Wardle & Kronberg 1974)."""
    return np.sqrt(np.maximum(Q_ts ** 2 + U_ts ** 2 - sigma ** 2, 0.0))


def plot_diagnostic(stokes, tsamp_s, f1_mhz, bw_mhz, nchan, outpath, title='', tscrunch_us=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    I, Q, U, V = stokes

    if tscrunch_us is not None:
        target_tsamp = tscrunch_us * 1e-6

        factor = int(round(target_tsamp / tsamp_s))

        if factor > 1:
            nsamp = I.shape[-1]
            ntrim = nsamp - (nsamp // factor) * factor

            stokes = stokes[..., :-ntrim] if ntrim else stokes
            stokes = stokes.reshape(4, stokes.shape[1], -1, factor).mean(axis=-1)
            I, Q, U, V = stokes

            tsamp_s *= factor

    nsamp = I.shape[-1]
    t_ms = np.arange(nsamp) * tsamp_s * 1e3
    freqs = f1_mhz + bw_mhz * np.arange(nchan)

    # Per-channel baseline (bandpass) subtraction — without this, static
    # per-channel gain dominates both the image and the summed profile and
    # buries a weak (SNR~5) pulse entirely. Matches the approach in plot.py.
    def bp(a):
        return a - np.median(a, axis=1, keepdims=True)

    I_bp, Q_bp, U_bp, V_bp = bp(I), bp(Q), bp(U), bp(V)

    # Frequency-summed time series of every Stokes parameter.
    I_ts = np.nansum(I_bp, axis=0)
    Q_ts = np.nansum(Q_bp, axis=0)
    U_ts = np.nansum(U_bp, axis=0)
    V_ts = np.nansum(V_bp, axis=0)

    # Off-pulse noise estimate (samples below the median I level) used for the
    # Ricean debias of the linear-polarisation profile.
    off_mask = I_ts < np.median(I_ts)
    if off_mask.sum() > 10:
        sigma_L = np.hypot(Q_ts[off_mask].std(), U_ts[off_mask].std()) / np.sqrt(2)
    else:
        sigma_L = 0.0
    L_ts = debiased_L(Q_ts, U_ts, sigma_L)

    fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True,
                             gridspec_kw={'height_ratios': [1.6, 2.4, 1, 1]})

    colors = {'I': 'k', 'L': 'crimson', 'Q': 'darkorange', 'U': 'seagreen', 'V': 'royalblue'}

    # --- Top: total-intensity profile with debiased linear polarisation ---
    ax = axes[0]
    ax.plot(t_ms, I_ts, lw=0.9, c=colors['I'], label='I')
    ax.plot(t_ms, L_ts, lw=0.9, c=colors['L'], label=r'$L_{\rm deb}$')
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax.set_ylabel('I, L')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax.set_title(title or 'Stokes I profile (bandpass-subtracted)')

    # --- Dynamic spectrum of Stokes I ---
    ax = axes[1]
    vmin, vmax = np.percentile(I_bp, [5, 99.5])
    im = ax.imshow(I_bp, aspect='auto', origin='upper',
                   extent=[t_ms[0], t_ms[-1], freqs[-1], freqs[0]],
                   cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_ylabel('Freq (MHz)')

    # --- Q and U in distinct colours ---
    ax = axes[2]
    ax.plot(t_ms, Q_ts, lw=0.9, c=colors['Q'], label='Q')
    ax.plot(t_ms, U_ts, lw=0.9, c=colors['U'], label='U')
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax.set_ylabel('Q, U')
    ax.legend(frameon=False, fontsize=8, loc='upper right')

    # --- V in blue ---
    ax = axes[3]
    ax.plot(t_ms, V_ts, lw=0.9, c=colors['V'], label='V')
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax.set_ylabel('V')
    ax.set_xlabel('Time (ms)')
    ax.legend(frameon=False, fontsize=8, loc='upper right')

    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"    diagnostic plot -> {outpath}")

def get_tx_resolution(cand_file):
    """
    Extract TX search resolution from parent folder.
    e.g. 4us/foo.cands -> 4us
    """
    return Path(cand_file).parent.name


def tx_res_us(cand_file):
    """Numeric resolution (us) of a .cands file's parent folder, for ordering."""
    name = get_tx_resolution(cand_file)
    try:
        return float(name.replace('us', ''))
    except ValueError:
        return float('inf')
# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand-dir', default=None,
                help='recursively search this directory for *.cands files')
    ap.add_argument('--cand_files', nargs='+')
    ap.add_argument('--workdir', default='.')
    ap.add_argument('--outdir', default='cutouts')
    ap.add_argument('--min-window', type=float, default=0.005,
                     help='floor on cutout window length, seconds (default 5ms — '
                          'previous 20ms default swamped sub-ms Crab GPs in the plots)')
    ap.add_argument('--width-factor', type=float, default=10)
    ap.add_argument('--max-window', type=float, default=None,
                     help='cap on cutout window length, seconds — --min-window only sets '
                          'a floor, so wider-boxcar candidates can still produce a large '
                          'window via width_factor scaling; set this to cap it')
    ap.add_argument('--min-snr', type=float, default=0.0)
    ap.add_argument('--dedup-tol-s', type=float, default=0.001,
                     help='skip candidates whose MJD falls within this many seconds of an '
                          'already-extracted candidate from the SAME search resolution '
                          '(the same burst detected at neighbouring trial DMs). Dedup is '
                          'per-resolution only: every resolution gets its own cutout. '
                          'default 0.001 s (1 ms)')
    ap.add_argument('--digifil-bin', default='digifil')
    ap.add_argument('--digifil-fft', '--df-fft', type=int, default=32,
                     help='digifil FFT factor (-F), i.e. number of channels. Sets the '
                          'time resolution of the extracted cutouts: for a 32 MHz band, '
                          '-F 32 gives 32 x 1 MHz channels at 1 us raw dt; increase it '
                          '(64, 128, ...) for finer time resolution. Independent of the '
                          'transientX search resolution used in tx.sh.')
    ap.add_argument('--plot-ts-us', type=float, default=None,
                     help='override the diagnostic-png time scrunch (us). Default: use '
                          'the transientX search resolution from the .cands folder name.')
    ap.add_argument('--plot-no-ts', action='store_true',
                     help='plot the diagnostic png at the native extracted time '
                          'resolution instead of scrunching to the search resolution '
                          '(useful with a finer --digifil-fft)')
    ap.add_argument('--digifil-min-block', '--df-min-block', type=float, default=0.5,
                     help='minimum duration (s) to request from digifil per call. The '
                          'desired small window is trimmed out client-side afterward. '
                          'Values below 0.5s are clamped up: digifil HANGS on -T shorter')
    ap.add_argument('--keep-fil', action='store_true',
                     help='keep the intermediate .fil cutout (deleted by default once .npz is saved)')
    ap.add_argument('--plot', action='store_true',
                     help='save a diagnostic PNG (dynspec + IQUV time series) per candidate')

    args = ap.parse_args()

    print(f"Indexing fragments in {args.workdir} ...")
    frags, stream_root = build_fragment_index(args.workdir)
    print(f"  found {len(frags)} fragments, "
          f"MJD {frags[0]['tstart_mjd']:.9f} -> {frags[-1]['t_end_mjd']:.9f} "
          f"(continuous-search root MJD {stream_root:.9f})")

    base_outdir = Path(args.outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)

    if args.cand_dir:
        cand_files = sorted(Path(args.cand_dir).rglob('*.cands'), key=tx_res_us)
    else:
        cand_files = [Path(x) for x in args.cand_files]

    for cand_file in cand_files:

        tx_res = get_tx_resolution(cand_file)
        
        outdir = base_outdir / tx_res
        outdir.mkdir(parents=True, exist_ok=True)
        
        print(f"TX resolution: {tx_res}")

        # Dedup state is per-resolution: the same physical burst appears once
        # per search resolution, and every resolution should be extracted.
        # Within one .cands file, transientX lists the same burst at several
        # neighbouring trial DMs (~0.2 ms MJD spread), so skip only those.
        seen = set()
        seen_bursts = []

        cand_file = Path(cand_file)
        print(f'\n=== {cand_file} ===')
        cands = []
        for line in open(cand_file):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            c = parse_cand_line(line)
            if c['snr'] is not None and c['snr'] < args.min_snr:
                continue
            cands.append(c)
        # Process in MJD order: candidates in the same fragment are then read
        # from page cache (digifil re-reads each fragment from the start on
        # every call, so one cold read per fragment is all we want).
        cands.sort(key=lambda c: c['mjd'])
        for c in cands:

            frag, offset_s = find_fragment(frags, stream_root, c['mjd'])

            if frag is None:
                print(f"  cand {c['cand_id']} mjd={c['mjd']:.9f}: NO fragment contains this MJD")
                continue

            window_s = window_for_cand(c, frag, args.min_window, args.width_factor,
                                        max_window_s=args.max_window)
            digifil_seek_s, digifil_dur_s = digifil_extraction_span(
                offset_s, frag, min_block_s=args.digifil_min_block)

            print("\n==========================")
            print(f"Candidate {c['cand_id']}")
            print(f"Candidate MJD        : {c['mjd']:.12f}")
            print(f"Candidate DM         : {c['dm']}")
            print(f"Candidate Width (ms) : {c['width_ms']}")
            print(f"Candidate SNR        : {c['snr']}")
            
            print(f"\nFragment")
            print(f"  file              : {frag['dada_path'].name}")
            print(f"  start MJD         : {frag['tstart_mjd']:.12f}")
            print(f"  end MJD           : {frag['t_end_mjd']:.12f}")
            print(f"  duration (s)      : {frag['obslen_s']:.6f}")
            
            print(f"\nComputed")
            print(f"  offset_s (local)  : {offset_s:.6f}")
            print(f"  window_s          : {window_s:.6f}")
            print(f"  digifil_seek_s    : {digifil_seek_s:.6f}")
            print(f"  digifil_dur_s     : {digifil_dur_s:.6f}")

            key = (cand_file.name, frag['dada_path'], round(offset_s, 6))
            if key in seen:
                print(f"  cand {c['cand_id']}: skipping, duplicate of already-extracted cand")
                continue
            seen.add(key)

            if is_burst_duplicate(c['mjd'], seen_bursts, args.dedup_tol_s):
                print(f"  cand {c['cand_id']} mjd={c['mjd']:.9f}: skipping, same burst "
                      f"(MJD within {args.dedup_tol_s*1e3:.0f} ms of one already extracted)")
                continue
            seen_bursts.append(c['mjd'])

            outname = f"cand{c['cand_id']}_{c['mjd']:.9f}_dm{c['dm']:.2f}"
            print(f"  cand {c['cand_id']}: mjd={c['mjd']:.9f} width={c['width_ms']}ms "
                  f"dm={c['dm']:.2f} -> {frag['dada_path'].name} "
                  f"final_window={window_s*1000:.2f}ms "
                  f"digifil_seek={digifil_seek_s:.4f}s digifil_dur={digifil_dur_s:.4f}s snr={c['snr']}")

            fil_cutout = extract_cutout(frag['dada_path'], digifil_seek_s, digifil_dur_s,
                                         c['dm'], outname, outdir,
                                         digifil_bin=args.digifil_bin,
                                         fft=args.digifil_fft)
            if fil_cutout is None:
                continue

            try:
                cutout_hdr = parse_fil_header(fil_cutout)
                expected_start = frag['tstart_mjd'] + digifil_seek_s/86400.0

                print("\nCutout header")
                print(f"Header start MJD     : {cutout_hdr['tstart_mjd']:.12f}")
                print(f"Expected start MJD   : {expected_start:.12f}")
                print(f"Difference (ms)      : {(cutout_hdr['tstart_mjd']-expected_start)*86400*1000:.6f}")
                print(f"Header tsamp         : {cutout_hdr['tsamp_s']}")
                print(f"Header nsamp         : {cutout_hdr['nsamp']}")
                print(f"Header nifs          : {cutout_hdr['nifs']}")
                print(f"Header nchan         : {cutout_hdr['nchan']}")

                cube = read_fil_cube(fil_cutout)          # (4, nchan, nsamp) AA,BB,CR,CI 
                print("\nCube")
                print("cube shape:", cube.shape)
                
                #for i in range(4):
                #    arr = cube[i]
                #    print(f"IF {i}")
                #    print(f"   mean = {arr.mean():.6f}")
                #    print(f"   std  = {arr.std():.6f}")
                #    print(f"   min  = {arr.min():.6f}")
                #    print(f"   max  = {arr.max():.6f}")
                stokes_block = coherency_to_stokes(cube)  # (4, nchan, nsamp) I,Q,U,V

                I = stokes_block[0]
                
                profile = I.mean(axis=0)
                
                peak = np.argmax(profile)
                
                #print("\nUntrimmed profile")
                #print(f"Peak sample      : {peak}")
                #print(f"Peak time (s)    : {peak*cutout_hdr['tsamp_s']:.6f}")
                #print(f"Peak value       : {profile[peak]:.6f}")
                #print(f"Mean             : {profile.mean():.6f}")
                #print(f"Std              : {profile.std():.6f}")

                labels = ["I","Q","U","V"]
                
                print("\nStokes block")
                
                for i,label in enumerate(labels):
                    arr = stokes_block[i]
                    print(f"{label}")
                    print(f"   mean = {arr.mean():.6f}")
                    print(f"   std  = {arr.std():.6f}")
                    print(f"   min  = {arr.min():.6f}")
                    print(f"   max  = {arr.max():.6f}")
                stokes, trimmed_tstart_mjd = trim_to_window(
                    stokes_block, cutout_hdr['tstart_mjd'], cutout_hdr['tsamp_s'],
                    c['mjd'], window_s)

                print("\nTrimmed cube")
                print("shape:", stokes.shape)

                I = stokes[0]
                
                profile = I.mean(axis=0)
                
                peak = np.argmax(profile)
                
                print("\nTrimmed profile")
                print(f"Peak sample      : {peak}")
                print(f"Peak time (ms)   : {peak*cutout_hdr['tsamp_s']*1000:.3f}")
                print(f"Peak value       : {profile[peak]:.6f}")

                print(f"Trimmed start MJD : {trimmed_tstart_mjd:.12f}")
                
                new_offset = (c['mjd'] - trimmed_tstart_mjd)*86400.0
                
                print(f"Candidate offset after trimming : {new_offset*1000:.3f} ms")
                print(f"Trimmed duration               : {stokes.shape[-1]*cutout_hdr['tsamp_s']*1000:.3f} ms")
            except Exception as e:
                print(f"    FAILED to read/convert {fil_cutout}: {e}")
                continue

            # Sanity check on the TRIMMED window (avoids the untrimmed block's
            # FFT settle/edge artifacts): I = AA+BB should be ~non-negative.
            neg_I_frac = float((stokes[0] < 0).mean())
            print(f"    [debug] fraction of trimmed Stokes-I samples < 0: "
                  f"{neg_I_frac:.3f} (expect near 0.0; if not, pol-axis reshape "
                  f"likely still wrong)")

            npz_path = outdir / f"{outname}_iquv.npz"
            np.savez(
                npz_path,
                stokes=stokes.astype(np.float32),           # (4, nchan, nsamp), order I,Q,U,V
                pol_order=np.array(['I', 'Q', 'U', 'V']),
                # candidate info
                cand_id=c['cand_id'],
                cand_mjd=c['mjd'],
                cand_dm=c['dm'],
                cand_width_ms=c['width_ms'],
                cand_snr=c['snr'],
                source_dada=str(frag['dada_path']),
                # timing / frequency axes for the TRIMMED cube actually saved
                tstart_mjd=trimmed_tstart_mjd,
                tsamp_s=cutout_hdr['tsamp_s'],
                nsamp=stokes.shape[-1],
                fch1_mhz=cutout_hdr['f1_mhz'],
                foff_mhz=cutout_hdr['bw_mhz'],
                nchan=cutout_hdr['nchan'],
                window_s=window_s,
                digifil_seek_s=digifil_seek_s,
                digifil_dur_s=digifil_dur_s,
            )
            print(f"    saved -> {npz_path}  shape={stokes.shape} (pol,chan,samp)")

            if args.plot:
                if args.plot_no_ts:
                    tscrunch_us = None
                else:
                    tscrunch_us = (args.plot_ts_us
                                   if args.plot_ts_us is not None
                                   else float(tx_res.replace("us", "")))
                plot_dt_us = (cutout_hdr['tsamp_s'] * 1e6
                              if tscrunch_us is None else tscrunch_us)
                plot_dt_label = f"{plot_dt_us:g}us"
                plot_outdir = base_outdir / plot_dt_label
                plot_outdir.mkdir(parents=True, exist_ok=True)
                png_path = plot_outdir / f"{outname}_diag.png"
                try:
                    plot_diagnostic(stokes, cutout_hdr['tsamp_s'], cutout_hdr['f1_mhz'],
                                     cutout_hdr['bw_mhz'], cutout_hdr['nchan'], png_path,
                                     title=f"dt={plot_dt_label}  cand {c['cand_id']}  mjd={c['mjd']:.6f}  "
                                           f"dm={c['dm']:.2f}  snr={c['snr']}",
                                     tscrunch_us=tscrunch_us)
                except Exception as e:
                    print(f"    plot FAILED: {e}")

            if not args.keep_fil:
                fil_cutout.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
