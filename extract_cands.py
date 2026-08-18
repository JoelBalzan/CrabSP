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
import re
import subprocess
from pathlib import Path

import numpy as np
from sigpyproc.readers import FilReader

from plot_iquv_profile import generate_profile_plot

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

def parse_dada_header(dada_path):
    """Read the 4096-byte DADA key/value header into a dict.

    dspsr folds the baseband centred on FREQ with bandwidth BW.  With
    auto-scrunch, the calibrator sub-band is extracted to match the fold's
    integer-MHz channel grid directly, so --fold-cf-offset defaults to 0.
    """
    out = {}
    try:
        with open(dada_path, 'rb') as f:
            header = f.read(4096).decode('latin-1', errors='replace')
    except OSError as e:
        return out
    for line in header.splitlines():
        parts = line.split()
        if len(parts) >= 2 and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', parts[0]):
            out[parts[0]] = parts[1]
    return out


def crop_dada_file(dada_path, offset_s, dur_s, out_path, hdr_size=4096,
                   true_obslen_s=None):
    """Write a small standalone .dada file covering [offset_s, offset_s+dur_s)
    of dada_path's raw voltage data, with a correctly updated OBS_OFFSET.

    Why this exists: dspsr's own duration-limiting flags don't compose
    safely with -seek here (-S double-seeks/zero-pads, -T overflows an
    internal buffer -- see fold_cutout's docstring), and even -c <short
    period> with -turns 1 (the "fast fold" trick) does NOT stop dspsr
    reading to EOF -- -c/-turns only set subintegration granularity, not
    total input read. The only reliable way found to bound dspsr's read
    time is to shrink the actual input FILE, so EOF is naturally close by.

    Per the psrdada/DADA convention, absolute time = UTC_START +
    OBS_OFFSET/BYTES_PER_SECOND, so cropping correctly means: keep
    UTC_START unchanged, and set the new file's OBS_OFFSET to
    (original OBS_OFFSET) + (byte offset into the data corresponding to
    offset_s). dspsr/psrchive then compute the same absolute timestamps
    for the cropped file's samples as for the original, so -seek/-cepoch
    on the caller side need no changes.

    IMPORTANT: the header's own BYTES_PER_SECOND field was tried first and
    is NOT trustworthy here -- using it directly produced crops with far
    less real data than requested (confirmed: a 12ms crop request yielded
    only ~0.9ms of real signal before the rest was zero-padded by dspsr,
    i.e. off by roughly the same ~13x factor as the earlier -T buffer-
    overshoot bug; dspsr's own log lines like "corrected Analytic
    (complex-valued) sampling rate=..." suggest the header's nominal rate
    needs correction that this crop wasn't applying). Instead, if
    true_obslen_s is given (the fragment's real duration, as already
    computed elsewhere in this pipeline via sigpyproc reading the matching
    .fil header -- see parse_fil_header/build_fragment_index, which the
    rest of the pipeline already trusts for all its MJD/coverage math),
    the byte rate is derived from (file size - header size) / true_obslen_s
    instead. This is self-consistent with every other timing calculation
    in the pipeline, unlike the raw header field. Falls back to the header
    field only if true_obslen_s isn't provided.

    Returns out_path, or raises RuntimeError if no byte rate can be
    determined, or if the requested range runs past the end of dada_path.
    """
    hdr = parse_dada_header(dada_path)
    hdr_size = int(hdr.get('HDR_SIZE', hdr_size))

    if true_obslen_s:
        file_size = Path(dada_path).stat().st_size
        data_size = file_size - hdr_size
        if data_size <= 0 or true_obslen_s <= 0:
            raise RuntimeError(f"{dada_path}: can't derive byte rate from "
                               f"file_size={file_size}, "
                               f"true_obslen_s={true_obslen_s}")
        bytes_per_second = data_size / true_obslen_s
    else:
        bytes_per_second = hdr.get('BYTES_PER_SECOND')
        if not bytes_per_second:
            raise RuntimeError(f"{dada_path}: no true_obslen_s given and no "
                               f"BYTES_PER_SECOND in DADA header, can't crop "
                               f"safely")
        bytes_per_second = float(bytes_per_second)

    resolution = int(hdr.get('RESOLUTION', 1) or 1)

    byte_start = int(offset_s * bytes_per_second)
    byte_count = int(np.ceil(dur_s * bytes_per_second))
    if resolution > 1:
        byte_start -= byte_start % resolution
        rem = byte_count % resolution
        if rem:
            byte_count += resolution - rem

    orig_offset = int(hdr.get('OBS_OFFSET', 0) or 0)
    new_offset = orig_offset + byte_start

    with open(dada_path, 'rb') as f:
        header_bytes = f.read(hdr_size)
        f.seek(hdr_size + byte_start)
        data = f.read(byte_count)
    if len(data) < byte_count:
        raise RuntimeError(f"{dada_path}: requested {byte_count} bytes at "
                           f"offset {byte_start}, only {len(data)} available "
                           f"-- crop window runs past this file's end")

    header_text = header_bytes.decode('latin-1', errors='replace')
    lines = header_text.split('\n')
    out_lines = []
    replaced = False
    for line in lines:
        if line.strip().split()[:1] == ['OBS_OFFSET']:
            out_lines.append(f'OBS_OFFSET  {new_offset}')
            replaced = True
        else:
            out_lines.append(line)
    if not replaced:
        out_lines.insert(1, f'OBS_OFFSET  {new_offset}')
    new_header = '\n'.join(out_lines).encode('latin-1', errors='replace')
    new_header = new_header[:hdr_size].ljust(hdr_size, b'\x00')

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(new_header)
        f.write(data)
    return out_path


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


def cluster_candidates(cands, gap_s):
    """Group candidates into events by MJD.

    Equivalent to DBSCAN(radius=gap_s, minPts=1) on the time axis: a new event
    starts when consecutive (MJD-sorted) candidates are more than gap_s apart.
    One physical Crab pulse produces one .cands row per trial DM within a few
    tenths of a ms (the dispersive delay shifts the peak by ~0.2 ms per 0.05 DM),
    while different rotations sit a 33 ms period apart — so a gap of ~3 ms (the
    main-pulse window) merges same-pulse detections without merging rotations
    or an MP+IP pair (~half a period apart).
    """
    cands = sorted(cands, key=lambda c: c['mjd'])
    events = []
    for c in cands:
        if events and (c['mjd'] - events[-1][-1]['mjd']) * 86400.0 <= gap_s:
            events[-1].append(c)
        else:
            events.append([c])
    return events


def pick_representative(event):
    """The highest-SNR candidate of an event (best DM estimate + peak time)."""
    return max(event, key=lambda c: c['snr'] or 0.0)


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


# --------------------------------------------------------------------------
# digifil extraction
# --------------------------------------------------------------------------

def plan_extraction(frags, frag, offset_s, min_block_s=0.5, margin_s=0.05):
    """
    digifil returns LESS than the requested -T seconds (with this DADA header
    dspsr drops a fixed ~0.24 s settle) and it REFUSES any block whose end
    exceeds the last .dada file passed. The old code clamped the block to end
    at the candidate's fragment boundary, so pulses in the last ~0.25 s of a
    fragment fell past the data digifil actually wrote and the trim failed.

    Instead: start the block a small margin BEFORE the candidate (never clamped
    to the fragment end) and let it run min_block seconds. If that crosses a
    fragment boundary, pass the neighbouring .dada files too with -cont (the
    fragments are contiguous, so digifil reads straight across). The cutout
    header tstart then still lands on the candidate's true MJD, so the client-
    side trim is unchanged.

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
        # Candidate so close to the end of the observation that even the final
        # fragment can't contain the block: clamp and let the trim report it.
        block_s = (last['t_end_mjd'] - block_start_abs) * 86400.0
        if block_s < SAFE_MIN_BLOCK_S:
            return (None, None, None, None,
                    f"candidate is only {block_s:.2f}s before the end of the "
                    f"observation — digifil cannot read past it; skipping")

    digifil_seek_s = (block_start_abs - first['tstart_mjd']) * 86400.0
    dada_paths = [f['dada_path'] for f in cover]
    return dada_paths, digifil_seek_s, block_s, first, None


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
            f"— increase --digifil-min-block (or the candidate is within ~0.3 s "
            f"of the end of the observation and cannot be covered)"
        )
    trimmed_tstart_mjd = block_tstart_mjd + (i0 * tsamp_s) / 86400.0
    return stokes[..., i0:i1], trimmed_tstart_mjd


def extract_cutout(dada_paths, seek_s, dur_s, dm, outname, outdir,
                    digifil_bin='digifil', nbits=-32, fft=32):
    outdir.mkdir(parents=True, exist_ok=True)
    out_fil = outdir / f"{outname}.fil"
    if out_fil.exists():
        out_fil.unlink()  # digifil refuses to overwrite; clear stale/partial files first
    cmd = [
        digifil_bin,
        '-S', f'{max(seek_s, 0):.6f}',
        '-T', f'{dur_s:.6f}',
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
                             # without this, PP/QQ (physically non-negative
                             # powers) come back ~50% negative
        '-o', str(out_fil),
    ]
    if len(dada_paths) > 1:
        cmd.append('-cont')  # treat the fragments as one continuous stream so the
                             # block can span a fragment boundary (see plan_extraction)
    cmd.extend(str(p) for p in dada_paths)
    print("\nRunning digifil")
    print(" ".join(cmd))
    # Don't capture output: digifil writes a \r progress bar to stderr and
    # capturing it hides all progress (looks like a 10-minute freeze). The
    # timeout is a safety net for the known short--T hang.
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


# --------------------------------------------------------------------------
# Reading the cutout filterbank -> (npol, nchan, nsamp) array, via sigpyproc
# --------------------------------------------------------------------------

def read_fil_cube(fil_path):
    """
    Read a digifil -d4 cutout filterbank into shape (4, nchan, nsamp), pol
    order PP, QQ, Re[PQ], Im[PQ], using sigpyproc.

    digifil -d 4 emits the four coherency products PP, QQ, Re[PQ], Im[PQ]
    (dspsr Signal::Coherence).  sigpyproc's read_block folds the pol/IF axis
    into the samples axis, IF FASTEST-varying within each time sample: flat
    per channel = [t0: IF0..IF3, t1: IF0..IF3, ...].  (Verified: this unpack
    gives clean per-IF means, the transposed one smears them.)
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
    elif arr.ndim == 2:
        nchan, flat = arr.shape
        if flat % nifs:
            raise RuntimeError(
                f"{fil_path}: flat sample axis {flat} not divisible by nifs={nifs} — "
                f"can't safely unpack pol/time, inspect arr.shape by hand."
            )
        nsamp_time = flat // nifs
        cube = arr.reshape(nchan, nsamp_time, nifs)  # (chan, time, IF), IF-fast
        cube = np.transpose(cube, (2, 0, 1))  # -> (nifs, nchan, nsamp_time)
    else:
        raise RuntimeError(f"{fil_path}: unexpected block ndim {arr.ndim}, shape {arr.shape}")

    return cube  # (4, nchan, nsamp), order PP, QQ, Re[PQ], Im[PQ]


def coherency_to_stokes(cube):
    """cube: (4, nchan, nsamp) in PP, QQ, Re[PQ], Im[PQ] -> IQUV.

    Matches dspsr's own Stokes formation (stokes_detect.ic):
        S0 = PP + QQ,  S1 = PP - QQ,
        S2 = 2 Re[PQ], S3 = 2 Im[PQ].
    (Cutouts are written with -b -32 float, so the cross terms are stored
    signed directly — no 8-bit +128 offset to remove.)
    """
    PP, QQ, RPQ, IPQ = cube[0], cube[1], cube[2], cube[3]
    I = PP + QQ
    Q = PP - QQ
    U = 2 * RPQ
    V = 2 * IPQ
    return np.stack([I, Q, U, V], axis=0)  # (4, nchan, nsamp), order I,Q,U,V


# --------------------------------------------------------------------------
# dspsr folding + pac calibration route
# --------------------------------------------------------------------------

def plan_dspsr_fold(frags, cand_mjd, margin_s, period, turns):
    """Covering fragments + seek MJD for a dspsr fold of `turns` spin periods
    beginning `margin_s` before the candidate.

    Unlike digifil (whose seek is relative to the first .dada file), dspsr
    seeks on the ABSOLUTE MJD, so the fragments are selected purely by MJD
    overlap.  The fold is anchored so phase 0 = the seek epoch (see
    fold_cutout), placing the burst near bin 0 of the folded profile even if
    the period is only approximate — the phase bins are then only a matter of
    time resolution, not of knowing the pulse phase.

    Note: `period` here is used only to size the fold window / coverage
    check (max_turns), not necessarily to set the actual fold phase — when
    fold_cutout is called with a parfile (-E), phase comes from the
    ephemeris instead. See --fold-parfile.

    NOTE: a previous version of this function also computed a file-relative
    pre-seek offset for dspsr's -S, to avoid dspsr reading each covering
    .dada from its start. That combination (-S together with -seek) is
    unsafe -- see the note in fold_cutout -- and has been removed. -seek
    alone is the only seek mechanism used here.

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
        # dspsr cannot seek before the start of the data: clamp the fold start
        # to the fragment edge (the burst is then found by peak detection).
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
                pac_dbase=None, derotate=False, rm=None):
    """Fold the coherency products around one pulse with dspsr -> .ar file.

    -seek anchors the output at the candidate's MJD and -cepoch makes that the
    phase-0 reference, so the burst lands in the first bins of the profile.
    -D <dm> -K dedisperse exactly as the digifil route does; -d 4 requests the
    four PP, QQ, Re[PQ], Im[PQ] products (matching digifil -d 4).

    NOTE: an earlier version of this function also passed dspsr's -S (seek
    in seconds from file start) alongside -seek, to save dspsr from reading
    each covering .dada from its very beginning. That combination is NOT
    safe: -seek's internal MJD->sample accounting assumes it owns the seek
    from file start, so pre-seeking with -S on top of it caused dspsr to
    double-seek / misjudge how much real input remained for -turns, and it
    then zero-padded the remainder of the fold once input ran out -- output
    looked like a real pulse in the first ~0.2 ms followed by an exact-zero
    flatline for the rest of the window (verified against TransientX's own
    candidate plot showing the burst mid-fragment, nowhere near a real data
    edge). -S has been removed; -seek alone is the only safe seek mechanism
    here, even though it costs reading from each covering file's start.

    IMPORTANT: -turns N is a time-DIVISION option (like -L/-nsub), setting
    how many periods go into EACH subintegration -- it does NOT cap how much
    total input dspsr reads/folds. Without an explicit stop, dspsr folds
    everything from the seek point to the end of the input file(s), chopped
    into N-period subintegrations, and "archive has 79 integrations" (or
    even more) is normal, not a bug on its own.

    Two things were tried and REJECTED to bound this and both broke dspsr in
    different ways: (1) -S (seek in seconds) alongside -seek caused dspsr to
    double-seek / zero-pad once real input ran out; (2) -T (total seconds to
    process) alongside -seek/-F/coherent dedispersion caused an internal
    buffer overshoot (`read_sample=16777216 > ndat=...`, i.e. dspsr tried to
    read a buffer far larger than the file, rc=255). Both are real dspsr
    flag-interaction bugs in this build, not just tuning issues -- don't
    reintroduce -S or -T here without confirming a fix upstream.

    Instead the problem is handled entirely downstream, in
    read_ar_stokes/trim_folded_to_window: rather than picking whichever
    subintegration happens to have the highest mean Stokes I (which can pick
    an unrelated, coincidentally brighter pulse elsewhere in the fragment
    once there are many subints), subint 0 is used directly -- since -seek
    anchors the fold so the candidate lands at the very start of the folded
    stream, subint 0 is always the temporally correct one regardless of how
    far past it dspsr kept folding.

    If `parfile` is given, phase comes from `-E <parfile>` (a proper
    TEMPO2-style ephemeris, e.g. `psrcat -e2 J0534+2200 > J0534+2200.par`)
    instead of the fixed `-c <period> -cepoch <seek_mjd>`. This matters once
    turns > 1: folding multiple rotations together with only an approximate
    constant period lets true and assumed phase drift apart across the fold
    window, superposing pulses (and their surrounding baseline/gain level) at
    a time offset — this shows up as a sharp step partway through the
    trimmed cutout. A real ephemeris keeps phase coherent across turns, so
    it's required for --fold-turns > 1 to be safe.

    If cf_offset_mhz is non-zero, the DADA band centre (from the header) is
    shifted by that amount with dspsr -f/-B.  The full-band UWL cal is on the
    X.5-MHz channel grid while dspsr folds land on integer centres, so
    cf_offset_mhz = -0.5 makes pac's per-channel matching succeed (the 0.5-MHz
    relabel is negligible for the broadband Crab spectrum).

    If pac_dbase is given, dspsr applies the pac calibration matrix
    convolution INLINE via -pac <dbase> during its own dedispersion stage,
    instead of (or as well as) a separate post-hoc `pac` CLI call. dbase
    must be a database built by `pac -w -k` (the same format used elsewhere
    in this pipeline via resolve_calibration/build_cal_database) -- not a
    single -A calibrator model file.

    CAVEAT: apply_pac()'s receiver-header prep (psredit setting
    rcvr:hand=-1/rcvr:sa=0.0/rcvr:rph=0.0 and be:name to match the
    calibrator) normally happens on the folded .ar BEFORE pac is invoked.
    With inline -pac there is no intermediate .ar to edit first -- dspsr's
    default fold receiver params (hand=+1/sa=45) apply at calibration time
    unless dspsr itself was told otherwise, which it currently isn't. This
    can silently produce a wrong feed-orientation correction. Treat inline
    -pac output as unverified until cross-checked against the existing
    fold_cutout + apply_pac two-step path on a known-good candidate.

    derotate/rm enable dspsr's own coherent (pre-detection) Faraday rotation
    correction (-derotate -rm <rm>), applied during dedispersion alongside
    -pac if both are given.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    out_ar = outdir / f"{outname}.ar"
    if out_ar.exists():
        out_ar.unlink()  # dspsr won't overwrite; clear stale files first
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
        '-K',                 # actually remove inter-channel dispersion delays
        '-F', str(nchan),     # fold into N x (band/N MHz) channels
        '-d', '4',            # npol=4 -> PP,QQ,PQ,QP coherency products
        '-b', str(nbin),      # phase bins per period
        '-A',                 # single archive with multiple integrations:
                              # without this, -turns triggers dspsr's concurrent
                              # (one-file-per-pulse) mode and -O is rejected
                              # ("cannot set archive filename in single pulse
                              # mode"); -A restores the FilenameEpoch convention
        '-e', 'ar',
        '-O', str(outdir / outname),
    ]
    # Override dspsr's default FFT length when nbin exceeds what the default
    # (2^16) supports.  Overlap-save convolution needs nfft >= 2*nbin.
    # Also bump -U (RAM limit in MB) since the larger FFT needs more workspace;
    # dspsr's minimum block is 2*nfft*nchan samples; at ~100 bytes/sample
    # overhead for FFTW buffers, that gives the RAM floor.
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
    if derotate:
        cmd += ['-derotate']
    if rm is not None:
        cmd += ['-rm', f'{rm:.6f}']
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
        cmd.append('-cont')  # cross fragment boundaries in a contiguous stream
    cmd.extend(str(p) for p in dada_paths)
    print("\nRunning dspsr fold")
    print(" ".join(cmd))
    # Don't capture output (progress bars / diagnostics go to the terminal).
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


def fold_cutout_fast(dada_paths, seek_mjd, dm, window_s, nbin, nchan, outname,
                     outdir, dspsr_bin='dspsr', cf_offset_mhz=0.0,
                     pac_dbase=None, derotate=False, rm=None):
    """Fast short-window dspsr extraction: fold exactly ONE degenerate
    "period" equal to window_s, instead of the real Crab spin period.

    Motivation: the normal fold_cutout() path folds one real spin period
    (or several, with --fold-parfile) and then reads to EOF regardless (see
    fold_cutout's docstring -- -S/-T can't safely bound that here), which is
    slow and can pick up unrelated extra data. Since we only ever want a few
    ms around the candidate, treating that short window itself as "the
    period" makes dspsr fold exactly one span of duration window_s, aligned
    to start at seek_mjd -- no real period/ephemeris needed (-E/-c are
    irrelevant when there's no real periodicity to track), and no long read.
    This is functionally closer to what digifil's -S/-T does, but still
    produces a foldable .ar via dspsr's own machinery so -pac/-derotate/-rm
    can be applied.

    pac_dbase/derotate/rm: see fold_cutout -- would apply the pac
    calibration matrix convolution and/or coherent Faraday derotation
    INLINE during dspsr's processing if pac_dbase is given.

    KNOWN BROKEN, DON'T PASS pac_dbase HERE: dspsr's inline -pac does
    STRICT frequency/band matching against the database, with no
    equivalent of the standalone `pac` CLI's relaxed -F -b -T -S -a
    matching flags. Confirmed failure: "no match found ... frequency
    want=3967.5 have=2368 ... no match" against a database whose
    calibrator the standalone `pac -F -b -T -S -a -d database.txt` call
    calibrates against successfully in the normal fold_cutout+apply_pac
    path. So in practice: fold_cutout_fast is called with pac_dbase=None
    (uncalibrated fast fold), and calibration is applied afterward via the
    normal apply_pac()/pac CLI step, which does support the relaxed flags.
    derotate/rm are unaffected by this and can still be applied inline.

    Returns the output .ar Path, or None on failure.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    out_ar = outdir / f"{outname}.ar"
    if out_ar.exists():
        out_ar.unlink()
    cmd = [
        dspsr_bin,
        '-seek', f'{seek_mjd:.9f}',
        '-c', f'{window_s:.9f}',
        '-cepoch', f'{seek_mjd:.9f}',
        '-turns', '1',
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
    if derotate:
        cmd += ['-derotate']
    if rm is not None:
        cmd += ['-rm', f'{rm:.6f}']
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
    print("\nRunning dspsr fast fold (window-as-period, inline -pac)")
    print(" ".join(cmd))
    timeout_s = max(60.0, 30.0 * window_s)
    try:
        r = subprocess.run(cmd, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"    dspsr fast fold TIMED OUT after {timeout_s:.0f}s")
        return None
    if r.returncode != 0:
        print(f"    dspsr fast fold FAILED (rc={r.returncode}); "
              f"re-run the printed command manually to see the error")
        return None
    if not out_ar.exists() or out_ar.stat().st_size == 0:
        print('    dspsr fast fold exited 0 but wrote no archive')
        return None
    return out_ar


def cal_beam_name(calib, calib_db, psredit_bin='psredit'):
    """Beam/instrument name (be:name) of the calibrator pac will match against.

    pac matches the target's `instrument` (be:name) to the calibrator's, so a
    fold whose be:name differs (dspsr folds default to the first antenna's
    name) is rejected even when everything else matches.  Read it from the
    -A calibrator model archive, or from the PolnCal entry of a -w database
    (database line: name type RA DEC MJD nchan cfreq bw instrument receiver).
    Returns None if it cannot be determined (caller then leaves be:name alone).
    """
    if calib_db and Path(calib_db).exists():
        for line in open(calib_db):
            parts = line.split()
            if len(parts) >= 10 and parts[1] == 'PolnCal':
                return parts[8]
    if calib and Path(calib).exists():
        r = subprocess.run([psredit_bin, '-c', 'be:name', '-q', str(calib)],
                           capture_output=True, text=True)
        m = re.search(r'be:name=(\S+)', r.stdout or '')
        if m:
            return m.group(1)
    return None


def apply_pac(ar_path, calib=None, calib_db=None, pac_bin='pac',
              pac_flags='', out_ext='calib',
              rcvr_params='type=Pulsar,rcvr:basis=lin,rcvr:hand=-1,'
                          'rcvr:sa=0.0,rcvr:rph=0.0',
              reverse_freqs=False, psredit_bin='psredit', pam_bin='pam'):
    """Calibrate a folded .ar with pac, writing `<input>.<ext>` (default .calib).

    Either --calib (pac -A <pcm/pacv calibrator model>) or --calib-db
    (pac -d <database from pac -w>) must be set; anything else needed by the
    user's setup (e.g. -x for fluxcal Stokes) goes through pac_flags.

    Header prep before pac (dspsr search-mode folds are missing it; mirrors
    the manual `psredit -c ... -m *.ar` + `pam --reverse_freqs -m *.ar`
    workflow):
      * rcvr_params (a comma-separated psredit attribute string, default the
        UWL receiver set type=Pulsar, basis=lin, hand=-1, sa=0, rph=0) is
        written onto the archive: pac rejects a target whose receiver
        hand/orientation differs from the calibrator's, and dspsr folds default
        to hand=+1/sa=45 while the UWL cal carries hand=-1/sa=0.
      * the calibrator's be:name (instrument) is copied onto the target — pac
        matches on it, so a target/cal beam-name mismatch is rejected.
      * if reverse_freqs, the fold's channel order is reversed with pam
        (only needed when the cal archives are recorded descending).
    """
    beam = cal_beam_name(calib, calib_db)
    full = rcvr_params + (f',be:name={beam}' if beam else '')
    rcmd = [psredit_bin, '-c', full, '-m', str(ar_path)]
    print("\nSetting archive header (psredit)")
    print(" ".join(rcmd))
    r = subprocess.run(rcmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    psredit FAILED (rc={r.returncode}): {r.stderr.strip()}")
    if reverse_freqs:
        pamcmd = [pam_bin, '--reverse_freqs', '-m', str(ar_path)]
        print("Reversing fold channel order (pam --reverse_freqs)")
        print(" ".join(pamcmd))
        subprocess.run(pamcmd, capture_output=True, text=True)
    cmd = [pac_bin]
    if pac_flags:
        cmd += pac_flags.split()
    if calib_db:
        cmd += ['-d', str(calib_db)]
    elif calib:
        cmd += ['-A', str(calib)]
    else:
        raise ValueError('apply_pac requires --calib or --calib-db')
    cmd += ['-e', out_ext, str(ar_path)]
    print("\nRunning pac")
    print(" ".join(cmd))
    try:
        r = subprocess.run(cmd, timeout=600.0)
    except subprocess.TimeoutExpired:
        print("    pac TIMED OUT")
        return None
    if r.returncode != 0:
        print(f"    pac FAILED (rc={r.returncode}); "
              f"re-run the printed command manually to see the error")
        return None
    out = ar_path.with_suffix(f'.{out_ext}')
    if not out.exists() or out.stat().st_size == 0:
        print(f"    pac ran but wrote nothing at {out}")
        return None
    return out


def find_cal_extensions(cal_dir):
    """Extensions of the calibration material present in cal_dir, in the order
    pac should search them (mirrors the manual `pac -w -u pcm -u avfluxcal
    -u dzT` step).  .dzT (tscrunched, zapped cal obs from process_cal.py) is
    preferred over the raw .cf; the .pcm model and flux calibrator are
    included when present."""
    cal_dir = Path(cal_dir)
    exts = []
    if any(cal_dir.glob('*.dzT')):
        exts.append('dzT')
    elif any(cal_dir.glob('*.cf')):
        exts.append('cf')
    if any(cal_dir.glob('*avfluxcal*')):
        exts.append('avfluxcal')
    if any(cal_dir.glob('*.pcm')):
        exts.append('pcm')
    return exts


def build_cal_database(cal_dir, db_path, pac_bin='pac', pac_flags=''):
    """Generate a pac calibration database with `pac -w -k` from whatever cal
    material is found in cal_dir, then return (db_path, None) on success or
    (None, note).  pac searches the CURRENT directory for the cal files, so it
    is run with cwd=cal_dir."""
    cal_dir = Path(cal_dir).resolve()
    db_path = Path(db_path).resolve()
    exts = find_cal_extensions(cal_dir)
    if not exts:
        return None, (f"no calibration material in {cal_dir} "
                      f"(looked for .dzT/.cf, *avfluxcal*, *.pcm)")
    cmd = [pac_bin]
    if pac_flags:
        cmd += pac_flags.split()
    cmd += ['-w', '-k', str(db_path)]
    for e in exts:
        cmd += ['-u', e]
    print("\nGenerating calibration database (pac -w)")
    print(f"  cal files dir : {cal_dir}")
    print("  " + " ".join(cmd))
    try:
        r = subprocess.run(cmd, cwd=str(cal_dir), timeout=600.0)
    except subprocess.TimeoutExpired:
        return None, "pac -w (database build) TIMED OUT after 600s"
    if r.returncode != 0:
        return None, ("pac -w FAILED (rc={r.returncode}); re-run the printed "
                      "command manually to see the error")
    if not db_path.exists() or db_path.stat().st_size == 0:
        return None, f"pac -w exited 0 but wrote no database at {db_path}"
    return db_path, None





def cal_files_from_database(db_path):
    """Parse a pac database (from `pac -w -k`) for the raw calibrator
    archive filenames it references.

    Database line format (per cal_beam_name's existing convention):
    name type RA DEC MJD nchan cfreq bw instrument receiver -- name (col 0)
    is the file path, type (col 1) is e.g. 'PolnCal'.

    Bare filenames (no leading /) are resolved relative to the database's
    parent directory.

    Returns a list of (filepath, type) tuples.
    """
    db_dir = Path(db_path).resolve().parent
    out = []
    for line in open(db_path):
        parts = line.split()
        if len(parts) >= 2 and not line.startswith('#') and \
           not parts[0].startswith('Pulsar::'):
            fpath = Path(parts[0])
            if not fpath.is_absolute():
                fpath = db_dir / fpath
            out.append((fpath, parts[1]))
    return out


def auto_scrunch_cal(native_db_path, fold_nchan, cal_dir,
                     fold_bw_mhz=0.0, fold_center_mhz=0.0,
                     pam_bin='pam', pac_bin='pac', pac_flags=''):
    """Build a frequency-scrunched copy of the calibration material so pac's
    channel matching (-a) succeeds against a fold done at fold_nchan channels.

    The native PolnCal calibrator (e.g. uwl_260720_000057.cf.dzT.pazi) may
    span the full UWL band (3328 x 1 MHz).  This function extracts the
    sub-band that matches the fold's centre frequency and bandwidth using
    psrchive's remove_chan(), then frequency-scrunches to fold_nchan with
    fscrunch_to_nchan().  This produces an archive whose channel bandwidth
    matches the fold exactly, satisfying pac's strict -a matching.

    If fold_bw_mhz/fold_center_mhz are not provided (legacy callers), the
    function falls back to the old pam -f factor-based scrunch assuming the
    native archive already covers only the fold's band.

    Returns (new_db_path, note); (None, note) if the scrunch can't be done.
    """
    try:
        import psrchive
    except ImportError as e:
        return None, (f"psrchive python bindings not available ({e}); "
                      f"can't auto-scrunch calibrator")

    entries = cal_files_from_database(native_db_path)
    polncal_files = [f for f, t in entries if t == 'PolnCal']
    if not polncal_files:
        return None, (f"no PolnCal entries found in {native_db_path}, "
                      f"can't auto-scrunch")

    target_dir = Path(cal_dir) / f"scrunched_{fold_nchan}ch"
    target_dir.mkdir(parents=True, exist_ok=True)

    use_subband = fold_bw_mhz > 0 and fold_center_mhz > 0

    for f, ftype in entries:
        f = Path(f)
        if not f.exists():
            continue
        if ftype == 'PolnCal':
            out_name = f.name.split('.')[0] + '.dzT'
            out_f = target_dir / out_name
            if out_f.exists():
                continue
            if use_subband:
                # Load native archive, extract sub-band, scrunch to fold_nchan.
                a = psrchive.Archive.load(str(f))
                native_nchan = a.get_nchan()
                native_freqs = a.get_frequencies()
                lo_mhz = fold_center_mhz - fold_bw_mhz / 2.0
                hi_mhz = fold_center_mhz + fold_bw_mhz / 2.0
                # Find channel indices within the fold sub-band.
                in_band = [i for i, freq in enumerate(native_freqs)
                           if lo_mhz <= freq <= hi_mhz]
                if not in_band:
                    return None, (f"no channels of {f.name} "
                                 f"(nchan={native_nchan}, "
                                 f"{native_freqs[0]:.1f}-"
                                 f"{native_freqs[-1]:.1f} MHz) fall in "
                                 f"fold sub-band [{lo_mhz:.1f}, "
                                 f"{hi_mhz:.1f}] MHz")
                sub_nchan = len(in_band)
                lo_idx, hi_idx = in_band[0], in_band[-1]
                # Remove channels outside the sub-band (high end first
                # to avoid index shifting).
                if hi_idx < native_nchan - 1:
                    a.remove_chan(hi_idx + 1, native_nchan - 1)
                if lo_idx > 0:
                    a.remove_chan(0, lo_idx - 1)
                sub_bw = native_freqs[hi_idx] - native_freqs[lo_idx] \
                    + (native_freqs[1] - native_freqs[0])
                # Scrunch to target nchan if needed.
                if sub_nchan != fold_nchan:
                    a.fscrunch_to_nchan(fold_nchan)
                # out_name already computed above (matches *dzT glob).
                out_path = str(target_dir / out_name)
                a.unload(out_path)
                print(f"\nSub-band extracted+scrunch {f.name}: "
                      f"{native_nchan} -> {sub_nchan} -> {fold_nchan} "
                      f"ch, centre={a.get_centre_frequency():.1f} MHz, "
                      f"bw={a.get_bandwidth():.1f} MHz -> {out_path}")
            else:
                # Legacy path: assume native archive covers the fold band.
                # Use pam -f to frequency-scrunch by an integer factor.
                a = psrchive.Archive.load(str(f))
                native_nchan = a.get_nchan()
                if native_nchan % fold_nchan != 0:
                    return None, (f"{f.name} has {native_nchan} channels, "
                                  f"can't scrunch to {fold_nchan}")
                factor = native_nchan // fold_nchan
                cmd = [pam_bin, '-f', str(factor), '-e', 'dzT',
                       '-u', str(target_dir), str(f)]
                print(f"\nScrunching calibrator {f.name} by {factor}x "
                      f"({native_nchan} -> {fold_nchan} channels)")
                print(" ".join(cmd))
                subprocess.run(cmd, capture_output=True, text=True)
        else:
            # avfluxcal/pcm etc: left at native resolution, just made
            # available alongside the scrunched PolnCal material so
            # `pac -w` (run with cwd=target_dir) can find everything pac
            # -w -u pcm -u avfluxcal -u dzT expects in one directory.
            link = target_dir / f.name
            # Remove stale/broken/self-referencing symlinks before creating
            # new ones (a leftover from older code that used bare relative
            # paths, producing circular symlinks that confuse pac).
            if link.is_symlink():
                link.unlink()
            if not link.exists():
                try:
                    link.symlink_to(f.resolve())
                except OSError:
                    pass

    new_db_path = target_dir / 'database.txt'
    if not new_db_path.exists():
        new_db_path, note = build_cal_database(target_dir, new_db_path,
                                               pac_bin, pac_flags)
        if new_db_path is None:
            return None, f"scrunched files written but database build failed: {note}"
    return new_db_path, f"auto-scrunched calibration to {fold_nchan} channels -> {new_db_path}"


def resolve_calibration(args, cal_dir, fold_bw_mhz=0.0,
                        fold_center_mhz=0.0):
    """Decide which pac calibration to apply, building the database
    automatically if needed.

    Order of precedence:
      1. --calib        : explicit calibrator model, applied with pac -A
      2. --calib-db     : explicit database, applied with pac -d
      3. <cal-dir>/<cal-db-name> (default database.txt) if it exists
      4. auto-generate the database from the .cf/.dzT + avfluxcal + pcm

    In cases 3-4 the PolnCal archive may span a wider band than the fold's
    sub-band.  auto_scrunch_cal extracts the matching sub-band and scrunches
    to args.fold_nchan so pac's -a channel matching succeeds.

    This does NOT apply to an explicit --calib/--calib-db (step 1/2) -- if
    you hand-pick a calibrator, you're assumed to have matched its
    channelization yourself.

    Returns (calib_file, cal_db, note); both None means no calibration.
    """
    if args.calib:
        return args.calib, None, f"using explicit calibrator model {args.calib}"
    db_path = (Path(args.calib_db) if args.calib_db
               else Path(cal_dir) / args.cal_db_name)
    if db_path.exists():
        note = f"using existing calibration database {db_path}"
        scrunched_db, scrunch_note = auto_scrunch_cal(
            db_path, args.fold_nchan, cal_dir,
            fold_bw_mhz=fold_bw_mhz, fold_center_mhz=fold_center_mhz,
            pam_bin=args.pam_bin, pac_bin=args.pac_bin,
            pac_flags=args.pac_flags)
        if scrunched_db is None:
            return (None, None,
                    f"WARNING: {scrunch_note} — saving UNCALIBRATED folds")
        return None, scrunched_db, scrunch_note
    if args.calib_db:
        return (None, None,
                f"WARNING: --calib-db {db_path} not found — saving UNCALIBRATED folds")
    db_path, note = build_cal_database(cal_dir, db_path, args.pac_bin, args.pac_flags)
    if db_path is None:
        return None, None, f"WARNING: {note} — saving UNCALIBRATED folds"
    scrunched_db, scrunch_note = auto_scrunch_cal(
        db_path, args.fold_nchan, cal_dir,
        fold_bw_mhz=fold_bw_mhz, fold_center_mhz=fold_center_mhz,
        pam_bin=args.pam_bin, pac_bin=args.pac_bin,
        pac_flags=args.pac_flags)
    if scrunched_db is None:
        return (None, None,
                f"WARNING: {scrunch_note} — saving UNCALIBRATED folds")
    return None, scrunched_db, scrunch_note


def read_ar_stokes(ar_path):
    """Read a folded/calibrated psrchive archive into (4, nchan, nbin) IQUV.

    psrchive's get_data() on a Coherence-state archive returns the four
    products PP, QQ, Re[PQ], Im[PQ] as float32 pols (verified against the raw
    PSRFITS DAT_SCL/DAT_OFFS unpack); a Stokes-state archive (as written by
    pac) returns I, Q, U, V directly.

    Returns (stokes, meta) with meta = {state, nchan, nbin, freqs_mhz,
    fch1_mhz, foff_mhz, epoch_mjd}.
    """
    try:
        import psrchive
    except ImportError as e:
        raise RuntimeError(f"psrchive python bindings not available ({e}); "
                           f"the dspsr route needs them to read the .ar") from e
    a = psrchive.Archive.load(str(ar_path))
    state = str(a.get_state())
    d = a.get_data()  # (nsub, npol, nchan, nbin) float32
    if d.ndim != 4:
        raise RuntimeError(f"{ar_path}: get_data shape {d.shape} — "
                           f"expected (nsub, npol, nchan, nbin)")
    nsub, npol, nchan, nbin = d.shape
    if npol != 4:
        raise RuntimeError(f"{ar_path}: got {npol} pols (state={state}); "
                           f"need the full 4 coherency products")
    if nsub > 1:
        # -seek anchors the fold so the candidate lands at the very start of
        # the folded stream (subint 0), regardless of how many subints dspsr
        # went on to produce reading to EOF (see fold_cutout docstring for
        # why -turns doesn't cap that, and why -S/-T were tried and rejected
        # as a fix). Picking the highest-mean-I subint here previously let
        # this silently latch onto an unrelated, coincidentally brighter
        # pulse elsewhere in the fragment -- always use subint 0 instead.
        print(f"    archive has {nsub} integrations (dspsr folded past the "
              f"candidate to EOF); using subint 0, the temporally correct "
              f"one given the -seek anchor")
        d = d[0:1]
    if 'Stokes' in state:
        stokes = d[0]  # I, Q, U, V
    elif 'Coherence' in state:
        PP, QQ, RPQ, IPQ = d[0, 0], d[0, 1], d[0, 2], d[0, 3]
        stokes = np.stack([PP + QQ, PP - QQ, 2 * RPQ, 2 * IPQ], axis=0)
    else:
        raise RuntimeError(f"{ar_path}: unsupported archive state '{state}'")
    freqs = np.asarray(a.get_frequencies(), dtype=float)
    if freqs.shape[0] != nchan:
        raise RuntimeError(f"{ar_path}: get_frequencies {freqs.shape} != "
                           f"nchan {nchan} — archive state inconsistent")
    foff_mhz = float(freqs[1] - freqs[0]) if nchan > 1 else 0.0
    try:
        m = a.get_epoch()
        epoch_mjd = float(getattr(m, 'val', m.intday + m.fracday))
    except Exception:
        epoch_mjd = None
    return stokes, {'state': state, 'nchan': nchan, 'nbin': nbin,
                    'freqs_mhz': freqs, 'fch1_mhz': float(freqs[0]),
                    'foff_mhz': foff_mhz, 'epoch_mjd': epoch_mjd}


def trim_folded_to_window(stokes, epoch_mjd, tsamp_s, window_s):
    """Centre the extracted window on the detected peak of the folded profile.

    Robust to absolute-MJD offsets (and to a fold start clamped to a fragment
    edge): the burst is located by peak-finding in the folded Stokes-I profile
    rather than assumed to be at bin 0.

    Returns (trimmed_stokes, trimmed_tstart_mjd).
    """
    nbin = stokes.shape[-1]
    profile = stokes[0].mean(axis=0)
    peak = int(np.argmax(profile))
    win_bins = int(round(window_s / tsamp_s))
    if win_bins >= nbin:
        return stokes, epoch_mjd
    half = win_bins // 2
    i0 = min(max(peak - half, 0), nbin - win_bins)
    i1 = i0 + win_bins
    tstart_mjd = epoch_mjd + (i0 * tsamp_s) / 86400.0
    return stokes[..., i0:i1], tstart_mjd


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
    ap.add_argument('--window-s', type=float, default=0.006,
                     help='fixed cutout window length, seconds (default 6 ms), '
                          'centred on the burst. Crab GPs are sub-ms, so 6 ms is '
                          'ample and keeps the cutout focused on the pulse.')
    ap.add_argument('--min-snr', type=float, default=0.0)
    ap.add_argument('--cluster-gap-ms', type=float, default=3.0,
                     help='merge candidates into one event when consecutive MJDs are '
                          'less than this far apart (ms). The Crab main-pulse window is '
                          '~3 ms and rotations are 33.4 ms apart, so 3 ms groups the '
                          'same pulse (many trial-DM detections) without merging '
                          'different rotations or an MP+IP pair. default 3 ms')
    ap.add_argument('--digifil-bin', default='digifil')
    ap.add_argument('--digifil-fft', '--df-fft', '-F', type=int, default=32,
                     help='digifil FFT factor (-F), i.e. number of channels. Sets the '
                          'time resolution of the extracted cutouts: for a 32 MHz band, '
                          '-F 32 gives 32 x 1 MHz channels at 1 us raw dt; increase it '
                          '(64, 128, ...) for finer time resolution. Independent of the '
                          'transientX search resolution used in tx.sh.')
    ap.add_argument('--digifil-min-block', '--df-min-block', type=float, default=0.5,
                     help='minimum duration (s) to request from digifil per call. The '
                          'desired small window is trimmed out client-side afterward. '
                          'Values below 0.5s are clamped up: digifil HANGS on -T shorter')
    ap.add_argument('--keep-fil', action='store_true',
                     help='keep the intermediate .fil cutout (deleted by default once .npz is saved)')
    ap.add_argument('--plot', action='store_true',
                     help='save a polarimetric diagnostic PNG (PA/I/L/dynspec '
                          'overview + multi-scrunch zoom panels, via '
                          'plot_iquv_profile.generate_profile_plot) per candidate')

    # --- dspsr folding + pac calibration route ------------------------------
    ap.add_argument('--method', choices=['digifil', 'dspsr'], default='dspsr',
                     help='extraction backend: dspsr (default: fold the '
                          'coherency products around each pulse with dspsr and '
                          'calibrate with pac) or digifil (from the raw .dada '
                          'baseband via digifil filterbanks, uncalibrated)')
    ap.add_argument('--dspsr-bin', default='dspsr')
    ap.add_argument('--pac-bin', default='pac')
    ap.add_argument('--pam-bin', default='pam',
                     help='used by auto_scrunch_cal to frequency-scrunch '
                          'the calibration material when --fold-nchan '
                          'differs from the native channelization')
    ap.add_argument('--calib', default=None, metavar='FILE',
                     help='pac calibration model (output of pcm/pacv) to apply '
                          'with pac -A. Setting this (or --calib-db) switches '
                          'to the dspsr route: digifil filterbanks cannot be '
                          'calibrated by pac.')
    ap.add_argument('--calib-db', default=None, metavar='FILE',
                     help='pac calibration database summary (from pac -w -k) '
                          'to apply with pac -d. If not given, extract_cands '
                          'uses <cal-dir>/<cal-db-name> (default database.txt) '
                          'or automatically generates it from the cal files.')
    ap.add_argument('--cal-dir', default=None, metavar='DIR',
                     help='directory holding the calibration material (.cf/.dzT '
                          'cal observation, *avfluxcal*, *.pcm). Default: '
                          '--workdir.')
    ap.add_argument('--cal-db-name', default='database.txt',
                     help='name of the auto-detected/generated calibration '
                          'database (default database.txt, in --cal-dir)')
    ap.add_argument('--no-cal', action='store_true',
                     help='disable automatic pac calibration (saves uncalibrated '
                          'folds) unless an explicit --calib/--calib-db is given')
    ap.add_argument('--pac-flags', default='-F -b -T -S -a',
                     help='extra flags for the pac steps. The default "-F -b '
                          '-T -S -a" matches the full-band UWL cal (.dzT) to '
                          '32-MHz-subband folds: -F/-b relax frequency/band '
                          'matching, -T ignores the time gap to the cal '
                          'observation, -S calibrates the D-terms, -a matches '
                          'channels per-frequency. Overridable (quoted string).')
    ap.add_argument('--fold-cf-offset', type=float, default=0.0,
                     help='shift (MHz) of the fold centre frequency applied so '
                          'the fold\'s channel grid aligns with the pac cal '
                          'database. Default 0.0: the auto-scrunched calibrator '
                          'sub-band already matches dspsr\'s integer-MHz grid. '
                          'Set -0.5 only when matching an un-scrunched full-band '
                          'UWL cal on the X.5-MHz grid. Only applied when '
                          'calibrating; set 0.0 to fold on the true DADA centre.')
    ap.add_argument('--reverse-fold-freqs', action='store_true',
                     help='reverse the fold\'s channel order with '
                          'pam --reverse_freqs before pac (only needed when '
                          'the cal archives are recorded descending, i.e. the '
                          'manual `pam --reverse_freqs -m *.ar` step)')
    ap.add_argument('--fold-nbin', type=int, default=None,
                     help='phase bins per period for the dspsr fold. '
                          'Default: auto-computed from --fold-nchan and the '
                          'DADA bandwidth to achieve the channel-bandwidth-'
                          'limited maximum time resolution '
                          '(nbin = period * chan_bw_Hz).')
    ap.add_argument('--fold-nchan', type=int, default=8,
                     help='fold channels (-F). auto_scrunch_cal extracts the '
                          'matching sub-band from the native PolnCal archive '
                          'and scrunches to this channel count so pac\'s -a '
                          'channel matching succeeds.')
    ap.add_argument('--fold-period', type=float, default=0.0334,
                     help='fold period (s), used when --fold-parfile is not '
                          'given, and always used to size the fold window '
                          '(coverage / max-turns check). The fold is anchored '
                          'at the candidate MJD, so an approximate Crab period '
                          'is fine for locating the burst with --fold-turns 1; '
                          'this only sets the time resolution in that case.')
    ap.add_argument('--fold-parfile', default=None, metavar='FILE',
                     help='TEMPO2-style ephemeris (e.g. `psrcat -e2 J0534+2200 '
                          '> J0534+2200.par`) used for phase-coherent folding '
                          'via dspsr -E, instead of the fixed --fold-period. '
                          'Required for --fold-turns > 1 to be safe: folding '
                          'multiple rotations together on an approximate '
                          'constant period lets true and assumed phase drift '
                          'apart across the fold window, superposing pulses '
                          '(and their surrounding baseline/gain level) at a '
                          'time offset — this shows up as a sharp step '
                          'partway through the trimmed cutout. Check the '
                          'parfile\'s PEPOCH/START/FINISH cover your data\'s '
                          'MJD range, and be aware Crab glitches — for '
                          'epoch-critical work prefer the Jodrell Bank '
                          'monthly Crab ephemeris over a static catalogue par.')
    ap.add_argument('--fold-margin-s', type=float, default=None,
                     help='seek this many seconds before the candidate MJD before '
                          'folding (default: window_s/2)')
    ap.add_argument('--fold-turns', type=int, default=1,
                     help='number of spin periods to fold per candidate '
                          '(default 1). Only raise this if --fold-parfile is '
                          'given — without a real ephemeris, multi-turn folds '
                          'on the fixed --fold-period drift out of phase '
                          'across the fold window (see --fold-parfile).')
    ap.add_argument('--keep-ar', action='store_true',
                     help='keep the intermediate .ar/.calib archives (deleted by '
                          'default once the .npz is saved)')
    ap.add_argument('--fast-pac', action='store_true',
                     help='use fold_cutout_fast instead of the normal fold: '
                          'folds a single window_s-long "period" starting at '
                          'the candidate (no real ephemeris/period needed), '
                          'so dspsr reads much less data per candidate than '
                          'the normal fold (which reads to EOF regardless of '
                          '--fold-turns). Calibration, if a database or '
                          'model is available, is then applied the normal '
                          'way via apply_pac()/pac CLI on the resulting '
                          '.ar -- NOT inline via dspsr -pac, which was '
                          'tried and does not support the relaxed '
                          '-F -b -T -S -a matching flags apply_pac() relies '
                          'on (fails outright when the calibrator band '
                          'differs from the fold band, which it usually '
                          'does here).')
    ap.add_argument('--fast-min-crop-s', type=float, default=0.35,
                     help='minimum duration (s) of the cropped .dada input '
                          'built by --fast-pac, regardless of the requested '
                          'fold window. Must comfortably exceed dspsr\'s own '
                          'internally-chosen coherent-dedispersion block '
                          'size (observed ~0.16s / "blocksize=5117272 '
                          'samples" for this DM/bandwidth) -- a crop smaller '
                          'than that block forces dspsr into partial/short-'
                          'block handling, which was confirmed to produce '
                          'inconsistent integration counts and, in one case, '
                          'corrupted/missing signal entirely. Still a large '
                          'reduction versus reading a full ~10s fragment. '
                          'Raise this if blocksize is larger for your DM/BW '
                          '(check the "blocksize=..." line in dspsr\'s log).')
    ap.add_argument('--rm', type=float, default=None,
                     help='rotation measure (rad/m^2) for dspsr coherent '
                          '(pre-detection) Faraday rotation correction, '
                          'passed as -rm. Only applied together with '
                          '--derotate.')
    ap.add_argument('--derotate', action='store_true',
                     help='enable dspsr coherent Faraday rotation correction '
                          '(-derotate), using --rm as the rotation measure. '
                          'Applied during dspsr processing in both the '
                          'normal and --fast-pac fold paths.')

    args = ap.parse_args()
    use_dspsr = args.method == 'dspsr' or bool(args.calib or args.calib_db)
    if use_dspsr:
        print("Using dspsr folding route")
    if args.fold_turns > 1 and not args.fold_parfile:
        print("WARNING: --fold-turns > 1 without --fold-parfile — phase will "
              "drift across turns using an approximate constant "
              "--fold-period, likely producing a step-like artifact in the "
              "trimmed cutout (pulses/baseline from different turns "
              "superposed at a time offset). Strongly recommend passing "
              "--fold-parfile (e.g. `psrcat -e2 J0534+2200 > J0534+2200.par`) "
              "or using --fold-turns 1.")
    cal_dir = Path(args.cal_dir) if args.cal_dir else Path(args.workdir)

    # Read a DADA header early to get BW/FREQ for sub-band extraction in
    # auto_scrunch_cal (the PolnCal calibrator may span the full UWL band,
    # so we need to know which sub-band the fold covers).
    fold_bw_mhz = 0.0
    fold_center_mhz = 0.0
    if use_dspsr:
        _dada_files = sorted(Path(args.workdir).glob('*.dada'))
        if _dada_files:
            _hdr = parse_dada_header(_dada_files[0])
            fold_bw_mhz = abs(float(_hdr.get('BW', 0)))
            fold_center_mhz = float(_hdr.get('FREQ', 0))

    # Resolve pac calibration once up front: explicit --calib/--calib-db, an
    # existing <cal-dir>/<cal-db-name>, or an automatically generated database.
    calib_file = None
    cal_db = None
    if use_dspsr and (not args.no_cal or args.calib or args.calib_db):
        calib_file, cal_db, cal_note = resolve_calibration(
            args, cal_dir, fold_bw_mhz=fold_bw_mhz,
            fold_center_mhz=fold_center_mhz)
        print(f"  calibration : {cal_note}")
    else:
        print("  calibration : disabled (--no-cal)" if args.no_cal
              else "  calibration : none requested")

    print(f"Indexing fragments in {args.workdir} ...")
    frags, stream_root = build_fragment_index(args.workdir)
    print(f"  found {len(frags)} fragments, "
          f"MJD {frags[0]['tstart_mjd']:.9f} -> {frags[-1]['t_end_mjd']:.9f} "
          f"(continuous-search root MJD {stream_root:.9f})")

    # Auto-compute --fold-nbin from the DADA bandwidth if not explicitly set.
    # nbin = period * chan_bw_Hz achieves the channel-BW-limited maximum time
    # resolution: tsamp = 1/chan_bw, so nbin = period/tsamp = period*chan_bw.
    # dspsr's FFT convolution defaults to an FFT length of 2^16 = 65536,
    # capping nbin at 32768.  We override this with -x so the full
    # channel-BW-limited nbin is used.
    if use_dspsr and args.fold_nbin is None:
        dada_hdr = parse_dada_header(frags[0]['dada_path'])
        bw_mhz = abs(float(dada_hdr.get('BW', 0.0)))
        if bw_mhz > 0:
            chan_bw_hz = bw_mhz * 1e6 / args.fold_nchan
            args.fold_nbin = int(np.ceil(args.fold_period * chan_bw_hz))
            print(f"  auto --fold-nbin: {bw_mhz:.0f} MHz / {args.fold_nchan} "
                  f"ch = {chan_bw_hz/1e6:.1f} MHz/ch -> "
                  f"nbin = {args.fold_nbin} "
                  f"({args.fold_period / args.fold_nbin * 1e6:.2f} us/bin)")
        else:
            args.fold_nbin = 4096
            print("  WARNING: could not read BW from DADA header, "
                  "falling back to --fold-nbin 4096")

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

        # Cluster into events, per resolution: one physical pulse appears once
        # per trial DM in the .cands file, so collapse them and extract a single
        # cutout per event, centred on the highest-SNR detection.
        events = cluster_candidates(cands, args.cluster_gap_ms / 1000.0)
        n_events = len(events)
        n_cands = len(cands)
        print(f"  {n_cands} candidates -> {n_events} events "
              f"(cluster gap {args.cluster_gap_ms:g} ms; "
              f"{n_cands / max(n_events, 1):.1f} detections/event)")
        for i_event, event in enumerate(events):
            c = pick_representative(event)

            frag, offset_s = find_fragment(frags, stream_root, c['mjd'])

            if frag is None:
                print(f"  cand {c['cand_id']} mjd={c['mjd']:.9f}: NO fragment contains this MJD")
                continue

            if use_dspsr:
                window_s = args.window_s
                period = args.fold_period
                margin_s = (args.fold_margin_s if args.fold_margin_s is not None
                            else window_s / 2.0)

                print("\n==========================")
                print(f"Event {i_event+1}/{n_events} "
                      f"({len(event)} detections, cand {c['cand_id']})")
                print(f"Candidate MJD        : {c['mjd']:.12f}")
                print(f"Candidate DM         : {c['dm']}")
                print(f"Candidate Width (ms) : {c['width_ms']}")
                print(f"Candidate SNR        : {c['snr']}")

                print(f"\nFragment")
                print(f"  file              : {frag['dada_path'].name}")
                print(f"  start MJD         : {frag['tstart_mjd']:.12f}")
                print(f"  end MJD           : {frag['t_end_mjd']:.12f}")

                dada_paths, seek_mjd, turns, plan_note = plan_dspsr_fold(
                    frags, c['mjd'], margin_s, period, args.fold_turns)
                if dada_paths is None:
                    print(f"    SKIP cand {c['cand_id']} mjd={c['mjd']:.9f}: {plan_note}")
                    continue

                print(f"\nComputed")
                print(f"  seek MJD (fold start) : {seek_mjd:.12f}")
                print(f"  margin_s              : {margin_s:.6f}")
                print(f"  fold period (s)       : {period:.6f}"
                      f"{'  (ignored: using --fold-parfile)' if args.fold_parfile else ''}")
                print(f"  fold parfile          : {args.fold_parfile or '(none, using -c/-cepoch)'}")
                print(f"  fold turns            : {turns}")
                print(f"  fold bins             : {args.fold_nbin}")
                print(f"  fold nchan            : {args.fold_nchan}")
                print(f"  fold files            : {[p.name for p in dada_paths]}")

                outname = f"cand{c['cand_id']}_{c['mjd']:.9f}_dm{c['dm']:.2f}"

                if args.fast_pac:
                    print(f"  mode                  : fast (cropped input, "
                          f"real period"
                          f"{', -derotate' if args.derotate else ''})")
                    # NOTE: dspsr's inline -pac does STRICT frequency/band
                    # matching with no equivalent of pac CLI's relaxed
                    # -F -b -T -S -a flags -- it fails ("no match found")
                    # whenever the calibrator's band differs from the fold's
                    # (confirmed: dbase cal at 2368 MHz, fold at 3967.5 MHz,
                    # which the standalone `pac -F -b ...` call tolerates
                    # fine). So the fast fold is done UNCALIBRATED here, and
                    # calibration (if available) is applied afterward via the
                    # normal apply_pac()/pac CLI step, which does support
                    # those relaxed-matching flags.
                    #
                    # NOTE 2: an earlier version of this branch used
                    # fold_cutout_fast's "window-as-period" trick (-c
                    # window_s -turns 1, i.e. an artificially short ~6ms
                    # period). That was ABANDONED: even at crop sizes far
                    # past dspsr's own convolution block size (confirmed up
                    # to 1s), output was still wrong/inconsistent for some
                    # candidates -- the short artificial period itself
                    # appears to misbehave with dspsr's dedispersion kernel,
                    # not just a data-availability problem. Using the REAL
                    # Crab period instead (same as the normal fold_cutout
                    # path, already validated correct) avoids that; the
                    # speed win now comes purely from feeding dspsr a small
                    # CROPPED .dada file via crop_dada_file instead of the
                    # full ~10s fragment, so it still reaches EOF quickly.
                    fold_dada_paths = dada_paths
                    tmp_cropped = None
                    if len(dada_paths) == 1:
                        crop_offset_s = max(0.0, (seek_mjd - frag['tstart_mjd']) * 86400.0)
                        # Needs to comfortably cover turns*period (the real
                        # fold span) plus dspsr's own block-size overhead
                        # (observed ~0.16s -- see fold_cutout_fast's old
                        # note) plus a margin, not just window_s.
                        crop_dur_s = max(3.0 * (turns * period + 2 * margin_s),
                                         args.fast_min_crop_s)
                        crop_dur_s = min(crop_dur_s,
                                         frag['obslen_s'] - crop_offset_s)
                        try:
                            tmp_cropped = crop_dada_file(
                                dada_paths[0], crop_offset_s, crop_dur_s,
                                outdir / f".{outname}_crop.dada",
                                true_obslen_s=frag['obslen_s'])
                            fold_dada_paths = [tmp_cropped]
                            print(f"  cropped input         : "
                                  f"{crop_dur_s*1000:.1f} ms from "
                                  f"{dada_paths[0].name} -> {tmp_cropped.name}")
                        except RuntimeError as e:
                            print(f"  WARNING: crop_dada_file failed ({e}); "
                                  f"falling back to the full fragment "
                                  f"(will be slow)")
                    else:
                        print("  WARNING: candidate spans multiple fragments; "
                              "skipping the input-crop optimisation for this "
                              "one (will be slow)")

                    ar_path = fold_cutout(
                        fold_dada_paths, seek_mjd, c['dm'], period,
                        args.fold_nbin, args.fold_nchan, turns,
                        outname, outdir, dspsr_bin=args.dspsr_bin,
                        cf_offset_mhz=(
                            args.fold_cf_offset
                            if (calib_file or cal_db) else 0.0),
                        parfile=args.fold_parfile,
                        derotate=args.derotate, rm=args.rm)
                    if tmp_cropped is not None:
                        tmp_cropped.unlink(missing_ok=True)
                    if ar_path is None:
                        continue

                    cal_path = ar_path
                    if calib_file or cal_db:
                        cal_path = apply_pac(ar_path, calib=calib_file,
                                             calib_db=cal_db,
                                             pac_bin=args.pac_bin,
                                             pac_flags=args.pac_flags,
                                             reverse_freqs=args.reverse_fold_freqs)
                        if cal_path is None:
                            if not args.keep_ar:
                                ar_path.unlink(missing_ok=True)
                            continue
                    else:
                        print("  (no calibration available: saving the "
                              "UNCALIBRATED fast fold)")
                else:
                    ar_path = fold_cutout(dada_paths, seek_mjd, c['dm'], period,
                                          args.fold_nbin, args.fold_nchan, turns,
                                          outname, outdir, dspsr_bin=args.dspsr_bin,
                                          cf_offset_mhz=(
                                              args.fold_cf_offset
                                              if (calib_file or cal_db) else 0.0),
                                          parfile=args.fold_parfile,
                                          derotate=args.derotate, rm=args.rm)
                    if ar_path is None:
                        continue

                    cal_path = ar_path
                    if calib_file or cal_db:
                        cal_path = apply_pac(ar_path, calib=calib_file,
                                             calib_db=cal_db,
                                             pac_bin=args.pac_bin,
                                             pac_flags=args.pac_flags,
                                             reverse_freqs=args.reverse_fold_freqs)
                        if cal_path is None:
                            if not args.keep_ar:
                                ar_path.unlink(missing_ok=True)
                            continue
                    else:
                        print("  (no calibration available: saving the UNCALIBRATED fold)")

                try:
                    stokes, meta = read_ar_stokes(cal_path)
                except Exception as e:
                    print(f"    FAILED to read {cal_path}: {e}")
                    if not args.keep_ar:
                        ar_path.unlink(missing_ok=True)
                        if cal_path != ar_path:
                            cal_path.unlink(missing_ok=True)
                    continue

                # In --fast-pac mode the folded "period" is window_s (the
                # cutout duration), not the real spin period -- use whichever
                # was actually folded to get tsamp right.
                # Both the fast (cropped input, real period) and normal
                # paths now fold the real Crab period.
                folded_period_s = period
                tsamp_s = folded_period_s / meta['nbin']
                print(f"\nFolded archive")
                print(f"  state             : {meta['state']}")
                print(f"  shape             : nsub=1, npol=4, "
                      f"nchan={meta['nchan']}, nbin={meta['nbin']}")
                print(f"  tsamp (bin dur)   : {tsamp_s*1e6:.3f} us")
                print(f"  freqs (MHz)       : {meta['fch1_mhz']:.3f} .. "
                      f"{meta['fch1_mhz'] + meta['foff_mhz']*(meta['nchan']-1):.3f} "
                      f"(df={meta['foff_mhz']:.3f})")
                epoch_str = (f"{meta['epoch_mjd']:.12f}" if meta['epoch_mjd']
                             else "n/a (using fold seek)")
                print(f"  archive epoch MJD : {epoch_str}")

                stokes, trimmed_tstart_mjd = trim_folded_to_window(
                    stokes, meta['epoch_mjd'] or seek_mjd, tsamp_s, window_s)

                labels = ["I", "Q", "U", "V"]
                print("\nTrimmed Stokes block")
                for i, label in enumerate(labels):
                    arr = stokes[i]
                    print(f"{label}")
                    print(f"   mean = {arr.mean():.6f}")
                    print(f"   std  = {arr.std():.6f}")
                    print(f"   min  = {arr.min():.6f}")
                    print(f"   max  = {arr.max():.6f}")
                profile = stokes[0].mean(axis=0)
                peak = int(np.argmax(profile))
                print(f"Trimmed start MJD  : {trimmed_tstart_mjd:.12f}")
                print(f"Trimmed duration   : {stokes.shape[-1]*tsamp_s*1000:.3f} ms")
                print(f"Peak sample        : {peak}  ({peak*tsamp_s*1000:.3f} ms)")

                neg_I_frac = float((stokes[0] < 0).mean())
                print(f"    [debug] fraction of trimmed Stokes-I samples < 0: "
                      f"{neg_I_frac:.3f}")

                npz_path = outdir / f"{outname}_iquv.npz"
                np.savez(
                    npz_path,
                    stokes=stokes.astype(np.float32),           # (4, nchan, nsamp) I,Q,U,V
                    pol_order=np.array(['I', 'Q', 'U', 'V']),
                    cand_id=c['cand_id'],
                    cand_mjd=c['mjd'],
                    cand_dm=c['dm'],
                    cand_width_ms=c['width_ms'],
                    cand_snr=c['snr'],
                    source_dada=str(frag['dada_path']),
                    tstart_mjd=trimmed_tstart_mjd,
                    tsamp_s=tsamp_s,
                    nsamp=stokes.shape[-1],
                    fch1_mhz=meta['fch1_mhz'],
                    foff_mhz=meta['foff_mhz'],
                    nchan=meta['nchan'],
                    window_s=window_s,
                    method=('dspsr-fast-cropped' if args.fast_pac else 'dspsr'),
                    calib_applied=bool(calib_file or cal_db),
                    calib_file=str(calib_file or cal_db or ''),
                    pac_flags=args.pac_flags,
                    fold_seek_mjd=seek_mjd,
                    fold_period=folded_period_s,
                    fold_parfile=str(args.fold_parfile or ''),
                    fold_turns=turns,
                    fold_nbin=args.fold_nbin,
                    fold_nchan=args.fold_nchan,
                    derotate=bool(args.derotate),
                    rm=(args.rm if args.rm is not None else np.nan),
                )
                print(f"    saved -> {npz_path}  shape={stokes.shape} (pol,chan,samp)")

                if args.plot:
                    plot_outdir = base_outdir / 'profiles'
                    plot_outdir.mkdir(parents=True, exist_ok=True)
                    png_path = plot_outdir / f"{outname}_profile.png"
                    cal_applied_now = bool(calib_file or cal_db)
                    cal_tag = 'calibrated' if cal_applied_now else 'UNCALIBRATED'
                    if args.fast_pac:
                        cal_tag += ', fast'
                    try:
                        generate_profile_plot(npz_path, out=str(png_path),
                                               title_suffix=f"[{cal_tag}]")
                        print(f"    profile plot -> {png_path}")
                    except Exception as e:
                        print(f"    plot FAILED: {e}")

                if not args.keep_ar:
                    ar_path.unlink(missing_ok=True)
                    if cal_path != ar_path:
                        cal_path.unlink(missing_ok=True)

                continue

            window_s = args.window_s
            dada_paths, digifil_seek_s, digifil_dur_s, first_frag, plan_note = plan_extraction(
                frags, frag, offset_s, min_block_s=args.digifil_min_block)
            if dada_paths is None:
                print(f"    SKIP cand {c['cand_id']} mjd={c['mjd']:.9f}: {plan_note}")
                continue

            print("\n==========================")
            print(f"Event {i_event+1}/{n_events} "
                  f"({len(event)} detections, cand {c['cand_id']})")
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
            print(f"  digifil files     : {[p.name for p in dada_paths]}")

            outname = f"cand{c['cand_id']}_{c['mjd']:.9f}_dm{c['dm']:.2f}"
            print(f"  cand {c['cand_id']}: mjd={c['mjd']:.9f} width={c['width_ms']}ms "
                  f"dm={c['dm']:.2f} -> {frag['dada_path'].name} "
                  f"final_window={window_s*1000:.2f}ms "
                  f"digifil_seek={digifil_seek_s:.4f}s digifil_dur={digifil_dur_s:.4f}s snr={c['snr']}")

            fil_cutout = extract_cutout(dada_paths, digifil_seek_s, digifil_dur_s,
                                         c['dm'], outname, outdir,
                                         digifil_bin=args.digifil_bin,
                                         fft=args.digifil_fft)
            if fil_cutout is None:
                continue

            try:
                cutout_hdr = parse_fil_header(fil_cutout)
                expected_start = first_frag['tstart_mjd'] + digifil_seek_s/86400.0

                print("\nCutout header")
                print(f"Header start MJD     : {cutout_hdr['tstart_mjd']:.12f}")
                print(f"Expected start MJD   : {expected_start:.12f}")
                print(f"Difference (ms)      : {(cutout_hdr['tstart_mjd']-expected_start)*86400*1000:.6f}")
                print(f"Header tsamp         : {cutout_hdr['tsamp_s']}")
                print(f"Header nsamp         : {cutout_hdr['nsamp']}")
                print(f"Header nifs          : {cutout_hdr['nifs']}")
                print(f"Header nchan         : {cutout_hdr['nchan']}")

                cube = read_fil_cube(fil_cutout)          # (4, nchan, nsamp) PP,QQ,Re[PQ],Im[PQ]
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
            # FFT settle/edge artifacts): I = PP+QQ should be ~non-negative.
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
                plot_outdir = base_outdir / 'profiles'
                plot_outdir.mkdir(parents=True, exist_ok=True)
                png_path = plot_outdir / f"{outname}_profile.png"
                try:
                    generate_profile_plot(npz_path, out=str(png_path),
                                           title_suffix="[digifil, uncalibrated]")
                    print(f"    profile plot -> {png_path}")
                except Exception as e:
                    print(f"    plot FAILED: {e}")

            if not args.keep_fil:
                fil_cutout.unlink(missing_ok=True)


if __name__ == '__main__':
    main()